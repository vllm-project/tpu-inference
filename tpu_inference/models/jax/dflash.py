# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""DFlash draft model for speculative decoding on JAX/TPU."""

from typing import Any, List, Tuple

import jax
import jax.numpy as jnp
from flax import nnx
from jax.sharding import Mesh
from transformers import Qwen3Config
from vllm.config import VllmConfig

from tpu_inference import utils
from tpu_inference.kernels.ragged_paged_attention.v3.kernel import \
    ragged_paged_attention
from tpu_inference.layers.common.sharding import ShardingAxisName
from tpu_inference.layers.jax.rope_interface import apply_rope
from tpu_inference.logger import init_logger
from tpu_inference.models.jax.utils.weight_utils import (BaseWeightLoader,
                                                         get_default_maps,
                                                         load_hf_weights)
from tpu_inference.utils import get_mesh_shape_product

logger = init_logger(__name__)

init_fn = nnx.initializers.uniform()

# vmem budget for the flash_attention Pallas kernel (128 MiB).
_FA_VMEM_LIMIT = 128 * 1024 * 1024


class DFlashAttention(nnx.Module):
    """DFlash cross+self attention with on-device KV cache.

    Each call:
      1. Projects Q from noise embeddings, K/V from [context, noise].
      2. Applies RoPE to Q and K.
      3. Expands K/V for GQA.
      4. Writes NEW K/V into the pre-allocated cache via dynamic_update_slice.
      5. Runs non-causal flash_attention over the full cache up to the valid
         length, using segment_ids to mask padding.
    """

    def __init__(
        self,
        config: Qwen3Config,
        dtype: jnp.dtype,
        rng: nnx.Rngs,
        mesh: Mesh,
    ):
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.rope_theta = config.rope_theta
        self.rope_scaling = getattr(config, "rope_scaling", None)
        self.rms_norm_eps = config.rms_norm_eps

        self.head_dim_original = getattr(config, "head_dim",
                                         self.hidden_size // self.num_heads)
        self.head_dim = utils.get_padded_head_dim(self.head_dim_original)

        sharding_size = get_mesh_shape_product(mesh,
                                               ShardingAxisName.MLP_TENSOR)
        self.num_heads = utils.get_padded_num_heads(self.num_heads,
                                                    sharding_size)
        self.num_kv_heads = utils.get_padded_num_heads(self.num_kv_heads,
                                                       sharding_size)
        self.num_kv_groups = self.num_heads // self.num_kv_heads

        self.mesh = mesh

        self.q_proj = nnx.Einsum(
            "TD,DNH->TNH",
            (self.hidden_size, self.num_heads, self.head_dim),
            param_dtype=dtype,
            kernel_init=nnx.with_partitioning(
                init_fn, (None, ShardingAxisName.ATTN_HEAD, None)),
            rngs=rng,
        )
        self.k_proj = nnx.Einsum(
            "TD,DKH->TKH",
            (self.hidden_size, self.num_kv_heads, self.head_dim),
            param_dtype=dtype,
            kernel_init=nnx.with_partitioning(
                init_fn, (None, ShardingAxisName.ATTN_HEAD, None)),
            rngs=rng,
        )
        self.v_proj = nnx.Einsum(
            "TD,DKH->TKH",
            (self.hidden_size, self.num_kv_heads, self.head_dim),
            param_dtype=dtype,
            kernel_init=nnx.with_partitioning(
                init_fn, (None, ShardingAxisName.ATTN_HEAD, None)),
            rngs=rng,
        )
        self.o_proj = nnx.Einsum(
            "TNH,NHD->TD",
            (self.num_heads, self.head_dim, self.hidden_size),
            param_dtype=dtype,
            kernel_init=nnx.with_partitioning(
                init_fn, (ShardingAxisName.ATTN_HEAD, None, None)),
            rngs=rng,
        )

        self.q_norm = nnx.RMSNorm(
            self.head_dim,
            epsilon=self.rms_norm_eps,
            param_dtype=dtype,
            scale_init=nnx.with_partitioning(init_fn, (None, )),
            rngs=rng,
        )
        self.k_norm = nnx.RMSNorm(
            self.head_dim,
            epsilon=self.rms_norm_eps,
            param_dtype=dtype,
            scale_init=nnx.with_partitioning(init_fn, (None, )),
            rngs=rng,
        )

    def __call__(
        self,
        x_noise: jax.Array,
        target_hidden: jax.Array,
        attention_metadata: Any,
        target_query_start_loc: jax.Array,
        target_positions: jax.Array,
        kv_cache: jax.Array,
    ) -> Tuple[jax.Array, jax.Array]:
        """Non-causal attention with on-device paged KV cache."""
        md = attention_metadata
        T_noise = x_noise.shape[0]
        T_target = target_hidden.shape[0]

        q_noise = self.q_norm(self.q_proj(x_noise))
        k_noise = self.k_norm(self.k_proj(x_noise))
        v_noise = self.v_proj(x_noise)

        # Apply RoPE to noise K
        k_noise = apply_rope(k_noise, md.input_positions,
                             self.head_dim_original, self.rope_theta,
                             self.rope_scaling)

        q_noise = q_noise.reshape(T_noise, self.num_heads, self.head_dim)
        k_noise = k_noise.reshape(T_noise, self.num_kv_heads, self.head_dim)
        v_noise = v_noise.reshape(T_noise, self.num_kv_heads, self.head_dim)

        # Process target hidden tokens (newly accepted tokens)
        k_target = self.k_norm(self.k_proj(target_hidden))
        v_target = self.v_proj(target_hidden)

        # Apply RoPE to target K
        k_target = apply_rope(k_target, target_positions,
                              self.head_dim_original, self.rope_theta,
                              self.rope_scaling)

        k_target = k_target.reshape(T_target, self.num_kv_heads, self.head_dim)
        v_target = v_target.reshape(T_target, self.num_kv_heads, self.head_dim)

        q_target = jnp.zeros((T_target, self.num_heads, self.head_dim),
                             dtype=q_noise.dtype)
        # Step 1: Write target tokens to KV cache
        # kv_lens is the length AFTER appending target tokens. This is EXACTLY md.seq_lens.
        _, kv_cache = ragged_paged_attention(
            queries=q_target,
            keys=k_target,
            values=v_target,
            kv_cache=kv_cache,
            kv_lens=md.seq_lens,
            page_indices=md.block_tables,
            cu_q_lens=target_query_start_loc,
            distribution=md.request_distribution,
            use_causal_mask=True,
            update_kv_cache=True,
            sm_scale=self.head_dim_original**-0.5,
        )

        # Step 2: Write noise tokens to KV cache and compute attention
        # kv_lens is the length AFTER appending noise tokens.
        num_reqs = md.seq_lens.shape[0]
        block_size = T_noise // num_reqs
        seq_lens_noise = md.seq_lens + block_size
        attn_out, kv_cache = ragged_paged_attention(
            queries=q_noise,
            keys=k_noise,
            values=v_noise,
            kv_cache=kv_cache,
            kv_lens=seq_lens_noise,
            page_indices=md.block_tables,
            cu_q_lens=md.query_start_loc,
            distribution=md.request_distribution,
            use_causal_mask=False,  # Noise tokens attend to all KV tokens
            update_kv_cache=True,
            sm_scale=self.head_dim_original**-0.5,
        )

        # attn_out is already (T_noise, num_heads, head_dim)
        out = self.o_proj(attn_out)

        return out, kv_cache


class DFlashMLP(nnx.Module):

    def __init__(self, config: Qwen3Config, dtype: jnp.dtype, rng: nnx.Rngs):
        hidden_size = config.hidden_size
        intermediate_size = config.intermediate_size
        self.gate_proj = nnx.Linear(
            hidden_size,
            intermediate_size,
            use_bias=False,
            param_dtype=dtype,
            kernel_init=nnx.with_partitioning(
                init_fn, (None, ShardingAxisName.MLP_TENSOR)),
            rngs=rng,
        )
        self.up_proj = nnx.Linear(
            hidden_size,
            intermediate_size,
            use_bias=False,
            param_dtype=dtype,
            kernel_init=nnx.with_partitioning(
                init_fn, (None, ShardingAxisName.MLP_TENSOR)),
            rngs=rng,
        )
        self.down_proj = nnx.Linear(
            intermediate_size,
            hidden_size,
            use_bias=False,
            param_dtype=dtype,
            kernel_init=nnx.with_partitioning(
                init_fn, (ShardingAxisName.MLP_TENSOR, None)),
            rngs=rng,
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        return self.down_proj(jax.nn.silu(self.gate_proj(x)) * self.up_proj(x))


class DFlashDecoderLayer(nnx.Module):

    def __init__(
        self,
        config: Qwen3Config,
        dtype: jnp.dtype,
        rng: nnx.Rngs,
        mesh: Mesh,
    ):
        hidden_size = config.hidden_size
        rms_norm_eps = config.rms_norm_eps

        self.input_layernorm = nnx.RMSNorm(
            hidden_size,
            epsilon=rms_norm_eps,
            param_dtype=dtype,
            scale_init=nnx.with_partitioning(init_fn, (None, )),
            rngs=rng,
        )
        self.self_attn = DFlashAttention(
            config=config,
            dtype=dtype,
            rng=rng,
            mesh=mesh,
        )
        self.post_attention_layernorm = nnx.RMSNorm(
            hidden_size,
            epsilon=rms_norm_eps,
            param_dtype=dtype,
            scale_init=nnx.with_partitioning(init_fn, (None, )),
            rngs=rng,
        )
        self.mlp = DFlashMLP(config=config, dtype=dtype, rng=rng)

    def __call__(
        self,
        x: jax.Array,
        target_hidden: jax.Array,
        attention_metadata: Any,
        target_query_start_loc: jax.Array,
        target_positions: jax.Array,
        kv_cache: jax.Array,
    ) -> Tuple[jax.Array, jax.Array]:
        """Returns (hidden_states, new_kv_cache)."""
        residual = x
        x = self.input_layernorm(x)
        x, kv_cache = self.self_attn(
            x_noise=x,
            target_hidden=target_hidden,
            attention_metadata=attention_metadata,
            target_query_start_loc=target_query_start_loc,
            target_positions=target_positions,
            kv_cache=kv_cache,
        )
        x = residual + x

        residual = x
        x = self.post_attention_layernorm(x)
        x = self.mlp(x)
        x = residual + x
        return x, kv_cache


class DFlashModel(nnx.Module):

    def __init__(
        self,
        vllm_config: VllmConfig,
        rng: nnx.Rngs,
        mesh: Mesh,
    ) -> None:
        spec_config = vllm_config.speculative_config
        assert spec_config is not None
        hf_config = spec_config.draft_model_config.hf_config
        dtype = jnp.bfloat16
        hidden_size = hf_config.hidden_size
        rms_norm_eps = hf_config.rms_norm_eps

        self.embed_tokens = nnx.Embed(
            num_embeddings=hf_config.vocab_size,
            features=hidden_size,
            param_dtype=dtype,
            embedding_init=nnx.with_partitioning(
                init_fn, (ShardingAxisName.VOCAB, None)),
            rngs=rng,
        )

        self.layers = nnx.List([
            DFlashDecoderLayer(
                config=hf_config,
                dtype=dtype,
                rng=rng,
                mesh=mesh,
            ) for _ in range(hf_config.num_hidden_layers)
        ])

        dflash_config = getattr(hf_config, "dflash_config", {})
        target_layer_ids = dflash_config.get("target_layer_ids", None)
        num_target_layers = getattr(hf_config, "num_target_layers", None)
        if target_layer_ids is not None:
            num_context_features = len(target_layer_ids)
        elif num_target_layers is not None:
            num_context_features = num_target_layers
        else:
            num_context_features = hf_config.num_hidden_layers

        target_hidden_size = getattr(hf_config, "target_hidden_size",
                                     hidden_size)
        fc_in_features = num_context_features * target_hidden_size

        self.fc = nnx.Linear(
            fc_in_features,
            hidden_size,
            use_bias=False,
            param_dtype=dtype,
            kernel_init=nnx.with_partitioning(init_fn, (None, None)),
            rngs=rng,
        )

        self.hidden_norm = nnx.RMSNorm(
            hidden_size,
            epsilon=rms_norm_eps,
            param_dtype=dtype,
            scale_init=nnx.with_partitioning(init_fn, (None, )),
            rngs=rng,
        )
        self.norm = nnx.RMSNorm(
            hidden_size,
            epsilon=rms_norm_eps,
            param_dtype=dtype,
            scale_init=nnx.with_partitioning(init_fn, (None, )),
            rngs=rng,
        )


class DFlashWeightLoader(BaseWeightLoader):

    def __init__(self, vllm_config: VllmConfig, mesh: Mesh):
        super().__init__(vllm_config, framework="pt")
        self.vllm_config = vllm_config
        self.mesh = mesh

    def load_weights(self, model: "DFlashForCausalLM", mappings: dict):
        metadata_map = get_default_maps(
            self.vllm_config.speculative_config.draft_model_config,
            self.mesh,
            mappings,
        )
        load_hf_weights(
            vllm_config=self.vllm_config,
            model=model,
            metadata_map=metadata_map,
            mesh=self.mesh,
            is_draft_model=True,
        )

        # If the embedding is not initialized, initialize it with a dummy
        # array here to pass jit compilation. The real weights will be shared
        # from the target model.
        if isinstance(model.model.embed_tokens.embedding.value,
                      jax.ShapeDtypeStruct):
            model.model.embed_tokens.embedding.value = jnp.zeros(
                model.model.embed_tokens.embedding.shape,
                dtype=model.model.embed_tokens.embedding.dtype,
            )


class DFlashForCausalLM(nnx.Module):
    """DFlash draft model for speculative decoding on TPU."""

    WeightLoader = DFlashWeightLoader

    def __init__(
        self,
        vllm_config: VllmConfig,
        rng_key: jax.Array,
        mesh: Mesh,
    ) -> None:
        nnx.Module.__init__(self)
        self.vllm_config = vllm_config
        self.rng = nnx.Rngs(rng_key)
        self.mesh = mesh

        spec_config = vllm_config.speculative_config
        assert spec_config is not None
        hf_config = spec_config.draft_model_config.hf_config
        self.hf_config = hf_config
        self.block_size = getattr(hf_config, "block_size", 8)
        dflash_config = getattr(hf_config, "dflash_config", {})
        self.mask_token_id = dflash_config.get("mask_token_id", 0)

        self._position_scheme = dflash_config.get("position_scheme",
                                                  "incremental")

        self.model = DFlashModel(
            vllm_config=vllm_config,
            rng=self.rng,
            mesh=mesh,
        )

    def __call__(
        self,
        kv_caches: List[jax.Array],
        input_ids: jax.Array,
        target_hidden_states: Any,
        attention_metadata,
    ) -> Tuple[List[jax.Array], jax.Array, List[jax.Array]]:
        """Forward pass for the DFlash draft model.

        ``target_hidden_states`` is a tuple (target_hidden, target_query_start_loc, target_positions).
        ``kv_caches`` is the framework's global paged kv caches.
        """
        target_hidden, target_query_start_loc, target_positions = target_hidden_states

        noise_emb = self.model.embed_tokens(input_ids)
        x = noise_emb

        for i, layer in enumerate(self.model.layers):
            kv_cache = kv_caches[i]
            x, kv_cache = layer(
                x,
                target_hidden=target_hidden,
                attention_metadata=attention_metadata,
                target_query_start_loc=target_query_start_loc,
                target_positions=target_positions,
                kv_cache=kv_cache,
            )
            kv_caches[i] = kv_cache

        x = self.model.norm(x)

        return kv_caches, x, []

    def compute_logits(self, hidden_states: jax.Array) -> jax.Array:
        """Compute logits using tied embedding weights."""
        return jnp.dot(hidden_states,
                       self.model.embed_tokens.embedding.value.T)

    def combine_hidden_states(self, hidden_states: jax.Array) -> jax.Array:
        """Project concatenated target auxiliary hidden states.

        Args:
            hidden_states: (T, num_target_layers * target_hidden_size)

        Returns:
            (T, hidden_size) projected + normalised context features.
        """
        return self.model.hidden_norm(self.model.fc(hidden_states))

    def load_weights(self, rng_key: jax.Array):
        self.rng = jax.random.key(self.vllm_config.model_config.seed)

        mappings = {
            "layers.*.input_layernorm": "model.layers.*.input_layernorm.scale",
            "layers.*.self_attn.q_proj":
            "model.layers.*.self_attn.q_proj.kernel",
            "layers.*.self_attn.k_proj":
            "model.layers.*.self_attn.k_proj.kernel",
            "layers.*.self_attn.v_proj":
            "model.layers.*.self_attn.v_proj.kernel",
            "layers.*.self_attn.o_proj":
            "model.layers.*.self_attn.o_proj.kernel",
            "layers.*.self_attn.q_norm":
            "model.layers.*.self_attn.q_norm.scale",
            "layers.*.self_attn.k_norm":
            "model.layers.*.self_attn.k_norm.scale",
            "layers.*.post_attention_layernorm":
            "model.layers.*.post_attention_layernorm.scale",
            "layers.*.mlp.gate_proj": "model.layers.*.mlp.gate_proj.kernel",
            "layers.*.mlp.up_proj": "model.layers.*.mlp.up_proj.kernel",
            "layers.*.mlp.down_proj": "model.layers.*.mlp.down_proj.kernel",
            "fc": "model.fc.kernel",
            "hidden_norm": "model.hidden_norm.scale",
            "norm": "model.norm.scale",
            "embed_tokens": "model.embed_tokens.embedding",
        }

        loader = self.WeightLoader(self.vllm_config, self.mesh)
        loader.load_weights(self, mappings)
