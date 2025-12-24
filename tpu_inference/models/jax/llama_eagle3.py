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

from typing import List, Tuple

import jax
import jax.numpy as jnp
from flax import nnx
from jax.sharding import Mesh
from transformers import LlamaConfig
from vllm.config import VllmConfig

from tpu_inference.layers.common.attention_metadata import AttentionMetadata
from tpu_inference.logger import init_logger
from tpu_inference.models.jax.llama3 import LlamaDecoderLayer
from tpu_inference.models.jax.utils.weight_utils import (MetadataMap,
                                                         get_default_maps,
                                                         load_hf_weights)

logger = init_logger(__name__)

init_fn = nnx.initializers.uniform()


class Eagle3LlamaDecoderLayer(LlamaDecoderLayer):

    def __init__(self, config: LlamaConfig, dtype: jnp.dtype, rng: nnx.Rngs,
                 mesh: Mesh, kv_cache_dtype: str):
        super().__init__(config,
                         dtype=dtype,
                         rng=rng,
                         mesh=mesh,
                         kv_cache_dtype=kv_cache_dtype)
        self.config = config
        # Override qkv
        hidden_size = 2 * self.self_attn.hidden_size
        self.self_attn.q_proj = nnx.Einsum(
            "TD,DNH->TNH",
            (hidden_size, self.self_attn.num_heads, self.self_attn.head_dim),
            param_dtype=dtype,
            dtype=dtype,
            kernel_init=nnx.with_partitioning(init_fn, (None, "model", None)),
            rngs=rng,
        )
        self.self_attn.k_proj = nnx.Einsum(
            "TD,DKH->TKH",
            (hidden_size, self.self_attn.num_kv_heads,
             self.self_attn.head_dim),
            param_dtype=dtype,
            dtype=dtype,
            kernel_init=nnx.with_partitioning(init_fn, (None, "model", None)),
            rngs=rng,
        )
        self.self_attn.v_proj = nnx.Einsum(
            "TD,DKH->TKH",
            (hidden_size, self.self_attn.num_kv_heads,
             self.self_attn.head_dim),
            param_dtype=dtype,
            dtype=dtype,
            kernel_init=nnx.with_partitioning(init_fn, (None, "model", None)),
            rngs=rng,
        )
        # Override input layernorm and specify dtype to avoid unexpected upcasting.
        self.input_layernorm = nnx.RMSNorm(
            config.hidden_size,
            epsilon=config.rms_norm_eps,
            param_dtype=dtype,
            dtype=dtype,
            scale_init=nnx.with_partitioning(init_fn, (None, )),
            rngs=rng,
        )
        self.hidden_norm = nnx.RMSNorm(
            config.hidden_size,
            epsilon=config.rms_norm_eps,
            param_dtype=dtype,
            scale_init=nnx.with_partitioning(init_fn, (None, )),
            rngs=rng,
        )

    def _norm_before_residual(
            self, hidden_states: jax.Array) -> tuple[jax.Array, jax.Array]:
        hidden_states = self.hidden_norm(hidden_states)
        residual = hidden_states
        return hidden_states, residual

    def _norm_after_residual(
            self, hidden_states: jax.Array) -> tuple[jax.Array, jax.Array]:
        residual = hidden_states
        hidden_states = self.hidden_norm(hidden_states)
        return hidden_states, residual

    def __call__(
        self,
        kv_cache: jax.Array,
        embeds: jax.Array,
        hidden_states: jax.Array,
        attention_metadata: AttentionMetadata,
    ) -> Tuple[jax.Array, jax.Array, jax.Array]:
        embeds = self.input_layernorm(embeds)
        if getattr(self.config, "norm_before_residual", False):
            hidden_states, residual = self._norm_before_residual(
                hidden_states=hidden_states)
        else:
            hidden_states, residual = self._norm_after_residual(
                hidden_states=hidden_states)
        hidden_states = jnp.concatenate([embeds, hidden_states], axis=-1)

        kv_cache, attn_output = self.self_attn(
            kv_cache,
            hidden_states,
            attention_metadata,
        )

        # TODO(ranlihao): Check if this residual connection is correct.
        hidden_states = attn_output + residual
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        mlp_output = self.mlp(hidden_states)

        return kv_cache, mlp_output, residual


class Eagle3LlamaModel(nnx.Module):

    def __init__(self, vllm_config: VllmConfig, rng: nnx.Rngs, mesh: Mesh):
        super().__init__()
        hf_config = vllm_config.speculative_config.draft_model_config.hf_config
        dtype: jnp.dtype = jnp.bfloat16

        self.embed_tokens = nnx.Embed(
            num_embeddings=hf_config.vocab_size,
            features=hf_config.hidden_size,
            param_dtype=dtype,
            embedding_init=nnx.with_partitioning(init_fn, ("model", None)),
            rngs=rng,
        )

        self.layers = [
            Eagle3LlamaDecoderLayer(
                config=hf_config,
                dtype=dtype,
                rng=rng,
                mesh=mesh,
                # TODO (jacobplatin): we should refactor this to pass a dtype (or config) directly
                kv_cache_dtype=vllm_config.cache_config.cache_dtype)
        ]

        if hasattr(hf_config, "target_hidden_size"):
            input_size = hf_config.target_hidden_size * 3
        else:
            input_size = hf_config.hidden_size * 3

        self.fc = nnx.Linear(
            in_features=input_size,
            out_features=hf_config.hidden_size,
            use_bias=False,
            param_dtype=dtype,
            kernel_init=nnx.with_partitioning(init_fn, (None, "model")),
            rngs=rng,
        )

        self.norm = nnx.RMSNorm(
            hf_config.hidden_size,
            epsilon=hf_config.rms_norm_eps,
            param_dtype=dtype,
            scale_init=nnx.with_partitioning(init_fn, (None, )),
            rngs=rng,
        )

    def __call__(
        self,
        kv_caches: List[jax.Array],
        input_ids: jax.Array,
        hidden_states: jax.Array,
        attention_metadata: AttentionMetadata,
    ) -> Tuple[List[jax.Array], jax.Array, List[jax.Array]]:
        embeds = self.embed_tokens(input_ids)
        assert hidden_states.shape[-1] == embeds.shape[-1]

        assert len(self.layers) == 1
        # The first N - 1 KV caches are for the target model, and the last one is for the draft model.
        # N is the number of layers in the target model.
        # The draft model has only 1 layer.
        kv_caches[-1], hidden_states, residual = self.layers[0](
            kv_caches[-1],
            embeds,
            hidden_states,
            attention_metadata,
        )

        # TODO(ranlihao): Check if this residual connection is correct.
        hidden_states = hidden_states + residual
        residual = hidden_states
        hidden_states = self.norm(hidden_states)
        return kv_caches, hidden_states, [residual]


def update_reshape_map_for_eagle3(vllm_config: VllmConfig,
                                  metadata_map: MetadataMap):
    model_config = vllm_config.speculative_config.draft_model_config
    hf_config = model_config.hf_config

    num_heads = hf_config.num_attention_heads
    num_kv_heads = hf_config.num_key_value_heads
    hidden_size = hf_config.hidden_size
    head_dim_original = model_config.get_head_size()

    metadata_map.reshape_map.update({
        "q_proj": (num_heads, head_dim_original, 2 * hidden_size),
        "k_proj": (num_kv_heads, head_dim_original, 2 * hidden_size),
        "v_proj": (num_kv_heads, head_dim_original, 2 * hidden_size),\
    })


class EagleLlama3ForCausalLM(nnx.Module):

    def __init__(self, vllm_config: VllmConfig, rng_key: jax.Array,
                 mesh: Mesh):
        nnx.Module.__init__(self)
        self.vllm_config = vllm_config
        self.rng = nnx.Rngs(rng_key)
        self.mesh = mesh
        dtype: jnp.dtype = jnp.bfloat16

        spec_config = vllm_config.speculative_config
        assert spec_config is not None
        model_config = spec_config.draft_model_config
        assert model_config is not None
        hf_config = model_config.hf_config

        self.model = Eagle3LlamaModel(
            vllm_config=vllm_config,
            rng=self.rng,
            mesh=mesh,
        )

        self.lm_head = nnx.Linear(
            hf_config.hidden_size,
            hf_config.draft_vocab_size,
            use_bias=False,
            param_dtype=dtype,
            kernel_init=nnx.with_partitioning(init_fn, (None, "model")),
            rngs=self.rng,
        )

        self.draft_id_to_target_id = nnx.Param(jnp.zeros(
            hf_config.draft_vocab_size, dtype=jnp.int32),
                                               sharding=(None, ))

    def __call__(
        self,
        kv_caches: List[jax.Array],
        input_ids: jax.Array,
        hidden_states: jax.Array,
        attention_metadata: AttentionMetadata,
    ) -> Tuple[List[jax.Array], jax.Array, List[jax.Array]]:
        return self.model(
            kv_caches,
            input_ids,
            hidden_states,
            attention_metadata,
        )

    def compute_logits(self, hidden_states: jax.Array) -> jax.Array:
        logits = self.lm_head(hidden_states)

        target_vocab_size = self.vllm_config.model_config.get_vocab_size()
        draft_vocab_size = self.vllm_config.speculative_config.draft_model_config.hf_config.draft_vocab_size

        base = jnp.arange(draft_vocab_size, dtype=jnp.int32)
        targets = base + self.draft_id_to_target_id.value

        logits_new = jnp.full((logits.shape[0], target_vocab_size),
                              -jnp.inf,
                              dtype=logits.dtype)

        logits_new = logits_new.at[:, targets].set(logits)

        return logits_new

    def combine_hidden_states(self, hidden_states: jax.Array) -> jax.Array:
        return self.model.fc(hidden_states)

    def load_weights(self, rng_key: jax.Array):
        # Create a new Rngs object for the draft model to avoid sharing RNG state
        self.rng = jax.random.key(self.vllm_config.model_config.seed)
        spec_config = self.vllm_config.speculative_config
        assert spec_config is not None

        mappings = {
            "midlayer.input_layernorm": "model.layers.0.input_layernorm.scale",
            "midlayer.hidden_norm": "model.layers.0.hidden_norm.scale",
            "midlayer.mlp.down_proj": "model.layers.0.mlp.down_proj.kernel",
            "midlayer.mlp.gate_proj": "model.layers.0.mlp.gate_proj.kernel",
            "midlayer.mlp.up_proj": "model.layers.0.mlp.up_proj.kernel",
            "midlayer.post_attention_layernorm":
            "model.layers.0.post_attention_layernorm.scale",
            "midlayer.self_attn.k_proj":
            "model.layers.0.self_attn.k_proj.kernel",
            "midlayer.self_attn.o_proj":
            "model.layers.0.self_attn.o_proj.kernel",
            "midlayer.self_attn.q_proj":
            "model.layers.0.self_attn.q_proj.kernel",
            "midlayer.self_attn.v_proj":
            "model.layers.0.self_attn.v_proj.kernel",
            "norm": "model.norm.scale",
            "fc": "model.fc.kernel",
            "lm_head": "lm_head.kernel",
            "d2t": "draft_id_to_target_id",
            "embed_tokens":
            "model.embed_tokens.embedding",  # Some checkpoints need this
        }

        # Define keys to keep in original dtype (e.g., float32 for stability)
        keep_original_dtype_keys_regex = [
            r".*d2t.*",
        ]

        metadata_map = get_default_maps(
            self.vllm_config.speculative_config.draft_model_config, self.mesh,
            mappings)

        update_reshape_map_for_eagle3(self.vllm_config, metadata_map)

        load_hf_weights(
            vllm_config=self.vllm_config,
            model=self,
            metadata_map=metadata_map,
            mesh=self.mesh,
            is_draft_model=True,
            keep_original_dtype_keys_regex=keep_original_dtype_keys_regex)

        # If the embedding is not initialized, initialize it with a dummy array here to pass jit compilation. The real weights will be shared from the target model in eagle3 class.
        if isinstance(self.model.embed_tokens.embedding.value,
                      jax.ShapeDtypeStruct):
            self.model.embed_tokens.embedding.value = jnp.zeros(
                self.model.embed_tokens.embedding.shape,
                dtype=self.model.embed_tokens.embedding.dtype,
            )
