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

from typing import List, Tuple

import jax
import jax.numpy as jnp
from flax import nnx
from jax import lax
from jax.sharding import Mesh
from transformers import Qwen3Config
from vllm.config import VllmConfig

from tpu_inference import utils
from tpu_inference.kernels.flash_attention.kernel import (BlockSizes,
                                                          SegmentIds,
                                                          flash_attention)
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
        noise_positions: jax.Array,
        ctx_positions: jax.Array,
        kv_cache_k: jax.Array,
        kv_cache_v: jax.Array,
        cache_len: jax.Array,
        actual_ctx_count: jax.Array,
    ) -> Tuple[jax.Array, jax.Array, jax.Array]:
        """Non-causal attention with on-device KV cache.

        Uses a two-phase cache write to handle padded context correctly:
          Phase A: write context K/V (with padding zeroed) at ``cache_len``.
          Phase B: write noise K/V at ``cache_len + actual_ctx_count``,
                   overwriting any padding zeros from Phase A.

        Args:
            x_noise: (T_noise, D) noise hidden states.
            target_hidden: (T_padded, D) padded context features.
            noise_positions: (T_noise,) position ids for noise tokens.
            ctx_positions: (T_padded,) position ids for context tokens.
            kv_cache_k: (1, N_heads, max_kv_len, H) pre-allocated K cache.
            kv_cache_v: (1, N_heads, max_kv_len, H) pre-allocated V cache.
            cache_len: scalar int, valid entries already in cache.
            actual_ctx_count: scalar int, real (non-padding) context tokens.

        Returns:
            (output, new_kv_cache_k, new_kv_cache_v)
        """
        T_noise = x_noise.shape[0]
        T_padded = target_hidden.shape[0]

        q = self.q_proj(x_noise)
        q = self.q_norm(q)
        q = apply_rope(
            q,
            noise_positions,
            self.head_dim_original,
            self.rope_theta,
            self.rope_scaling,
        )

        x_new = jnp.concatenate([target_hidden, x_noise], axis=0)
        k_new = self.k_proj(x_new)
        v_new = self.v_proj(x_new)
        k_new = self.k_norm(k_new)

        new_positions = jnp.concatenate([ctx_positions, noise_positions],
                                        axis=0)
        k_new = apply_rope(
            k_new,
            new_positions,
            self.head_dim_original,
            self.rope_theta,
            self.rope_scaling,
        )

        if self.num_kv_groups > 1:
            k_new = jnp.repeat(k_new, self.num_kv_groups, axis=1)
            v_new = jnp.repeat(v_new, self.num_kv_groups, axis=1)

        k_ctx = k_new[:T_padded]
        v_ctx = v_new[:T_padded]
        k_noise = k_new[T_padded:]
        v_noise = v_new[T_padded:]

        ctx_mask = (jnp.arange(T_padded) < actual_ctx_count)  # (T_padded,)
        ctx_mask_kv = ctx_mask[:, jnp.newaxis, jnp.newaxis]  # (T_padded, 1, 1)
        k_ctx = jnp.where(ctx_mask_kv, k_ctx, 0.0)
        v_ctx = jnp.where(ctx_mask_kv, v_ctx, 0.0)

        k_ctx_4d = k_ctx.transpose(1, 0, 2)[jnp.newaxis, :, :, :]
        v_ctx_4d = v_ctx.transpose(1, 0, 2)[jnp.newaxis, :, :, :]
        kv_cache_k = lax.dynamic_update_slice(kv_cache_k, k_ctx_4d,
                                              (0, 0, cache_len, 0))
        kv_cache_v = lax.dynamic_update_slice(kv_cache_v, v_ctx_4d,
                                              (0, 0, cache_len, 0))

        noise_start = cache_len + actual_ctx_count
        k_noise_4d = k_noise.transpose(1, 0, 2)[jnp.newaxis, :, :, :]
        v_noise_4d = v_noise.transpose(1, 0, 2)[jnp.newaxis, :, :, :]
        kv_cache_k = lax.dynamic_update_slice(kv_cache_k, k_noise_4d,
                                              (0, 0, noise_start, 0))
        kv_cache_v = lax.dynamic_update_slice(kv_cache_v, v_noise_4d,
                                              (0, 0, noise_start, 0))

        new_cache_len = cache_len + actual_ctx_count + T_noise
        max_kv_len = kv_cache_k.shape[2]

        q_4d = q.transpose(1, 0, 2)[jnp.newaxis, :, :, :]
        kv_ids = (jnp.arange(max_kv_len) < new_cache_len).astype(jnp.int32)
        q_ids = jnp.ones(T_noise, dtype=jnp.int32)
        seg_ids = SegmentIds(
            q=q_ids[jnp.newaxis, :],
            kv=kv_ids[jnp.newaxis, :],
        )

        sm_scale = self.head_dim_original**-0.5
        block_sizes = BlockSizes(
            block_q=T_noise,
            block_k_major=max_kv_len,
            block_k=max_kv_len,
            block_b=1,
        )
        attn_out = flash_attention(
            q_4d,
            kv_cache_k,
            kv_cache_v,
            segment_ids=seg_ids,
            causal=False,
            sm_scale=sm_scale,
            block_sizes=block_sizes,
            vmem_limit_bytes=_FA_VMEM_LIMIT,
        )

        attn_out = attn_out[0].transpose(1, 0, 2)
        output = self.o_proj(attn_out)

        return output, kv_cache_k, kv_cache_v


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
        noise_positions: jax.Array,
        ctx_positions: jax.Array,
        kv_cache_k: jax.Array,
        kv_cache_v: jax.Array,
        cache_len: jax.Array,
        actual_ctx_count: jax.Array,
    ) -> Tuple[jax.Array, jax.Array, jax.Array]:
        """Returns (hidden_states, new_kv_cache_k, new_kv_cache_v)."""
        residual = x
        x = self.input_layernorm(x)
        x, kv_cache_k, kv_cache_v = self.self_attn(
            x,
            target_hidden,
            noise_positions,
            ctx_positions,
            kv_cache_k,
            kv_cache_v,
            cache_len,
            actual_ctx_count,
        )
        x = residual + x

        residual = x
        x = self.post_attention_layernorm(x)
        x = self.mlp(x)
        x = residual + x
        return x, kv_cache_k, kv_cache_v


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
        target_hidden_states: jax.Array,
        attention_metadata,
    ) -> Tuple[List[jax.Array], jax.Array, List[jax.Array]]:
        """Forward pass for the DFlash draft model.

        ``target_hidden_states`` is a 3-tuple:
            (ctx_hidden, cache_len_arr, actual_ctx_count_arr)
        where:
            ctx_hidden: (T_padded, D) — padded context features.
            cache_len_arr: (1,) int32 — valid entries already in KV cache.
            actual_ctx_count_arr: (1,) int32 — real (non-padding) context count.

        ``kv_caches`` is a flat list of length ``2 * num_layers``:
            [k_cache_0, v_cache_0, k_cache_1, v_cache_1, ...]
        Each cache has shape ``(1, num_heads, max_kv_len, head_dim)``.

        Returns:
            (kv_caches, hidden_states, [target_hidden_states])
        """
        ctx_hidden, cache_len_arr, actual_ctx_count_arr = target_hidden_states
        cache_len = cache_len_arr[0]  # scalar
        actual_ctx_count = actual_ctx_count_arr[0]  # scalar

        noise_emb = self.model.embed_tokens(input_ids)
        pos_offset = cache_len if self._position_scheme == "incremental" else 0
        T_padded = ctx_hidden.shape[0]
        T_noise = input_ids.shape[0]
        ctx_positions = jnp.arange(T_padded, dtype=jnp.int32) + pos_offset
        noise_positions = (jnp.arange(T_noise, dtype=jnp.int32) + pos_offset +
                           actual_ctx_count)

        x = noise_emb
        for i, layer in enumerate(self.model.layers):
            kv_k = kv_caches[2 * i]
            kv_v = kv_caches[2 * i + 1]
            x, kv_k, kv_v = layer(
                x,
                ctx_hidden,
                noise_positions,
                ctx_positions,
                kv_k,
                kv_v,
                cache_len,
                actual_ctx_count,
            )
            kv_caches[2 * i] = kv_k
            kv_caches[2 * i + 1] = kv_v

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
