from typing import List, Optional, Tuple

import jax
import jax.numpy as jnp
from flax import nnx
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P
from vllm.config import VllmConfig
from vllm.model_executor.layers.sampler import SamplingMetadata

from tpu_commons.logger import init_logger
from tpu_commons.models.jax.common.attention.attention import (
    Attention, AttentionMetadata)
from tpu_commons.models.jax.common.constants import KVCacheType
from tpu_commons.models.jax.common.layers import (DenseFFW, Embedder, LMhead,
                                                  RMSNorm)
from tpu_commons.models.jax.common.transformer_block import TransformerBlock
from tpu_commons.models.jax.utils.weight_utils import (MetadataMap,
                                                       load_hf_weights)

logger = init_logger(__name__)


class LlamaForCausalLM(nnx.Module):

    def __init__(self,
                 vllm_config: VllmConfig,
                 rng_key: jax.Array,
                 mesh: Mesh,
                 force_random_weights: bool = False):
        assert mesh is not None

        self.vllm_config = vllm_config
        self.rng = nnx.Rngs(rng_key)
        self.mesh = mesh

        model_name = self.vllm_config.model_config.model.lower()
        if "70b" in model_name:
            logger.info("Initializing Llama3 70B model variant.")
            self.hidden_size = 8192
            num_layers = 80
            self.num_attention_heads = 64
            self.num_key_value_heads = 8
            intermediate_size = 28672
        elif "8b" in model_name:
            logger.info("Initializing Llama3 8B model variant.")
            self.hidden_size = 4096
            num_layers = 32
            self.num_attention_heads = 32
            self.num_key_value_heads = 8
            intermediate_size = 14336
        else:
            raise ValueError(
                f"Could not determine Llama3 variant (8B or 70B) from model name: '{model_name}'. "
                "Please ensure '8b' or '70b' is in the model path.")

        dtype = jnp.bfloat16
        self.head_dim = 128
        rope_theta = 500000.0
        vocab_size = 128256
        rms_norm_eps = 1e-5

        self.embedder = Embedder(vocab_size=vocab_size,
                                 hidden_size=self.hidden_size,
                                 dtype=dtype,
                                 rngs=self.rng,
                                 random_init=force_random_weights,
                                 vd_sharding=("model", None))

        self.layers = []
        for _ in range(num_layers):
            self.layers.append(
                TransformerBlock(
                    pre_attention_norm=RMSNorm(
                        dims=self.hidden_size,
                        random_init=force_random_weights,
                        epsilon=rms_norm_eps,
                        rngs=self.rng,
                        with_scale=True,
                        dtype=dtype,
                    ),
                    pre_mlp_norm=RMSNorm(
                        dims=self.hidden_size,
                        rngs=self.rng,
                        random_init=force_random_weights,
                        epsilon=rms_norm_eps,
                        with_scale=True,
                        dtype=dtype,
                    ),
                    attn=Attention(
                        hidden_size=self.hidden_size,
                        num_attention_heads=self.num_attention_heads,
                        num_key_value_heads=self.num_key_value_heads,
                        head_dim=self.head_dim,
                        rope_theta=rope_theta,
                        rope_scaling={},
                        rngs=self.rng,
                        dtype=dtype,
                        mesh=self.mesh,
                        random_init=force_random_weights,
                        dnh_sharding=(None, "model", None),
                        dkh_sharding=(None, "model", None),
                        nhd_sharding=("model", None, None),
                        query_tnh=P(None, "model", None),
                        keyvalue_skh=P(None, "model", None),
                        attn_o_tnh=P(None, "model", None),
                    ),
                    custom_module=DenseFFW(dtype=dtype,
                                           hidden_act="silu",
                                           hidden_size=self.hidden_size,
                                           intermediate_size=intermediate_size,
                                           rngs=self.rng,
                                           df_sharding=(None, "model"),
                                           fd_sharding=("model", None),
                                           random_init=force_random_weights),
                ))

        self.final_norm = RMSNorm(
            dims=self.hidden_size,
            rngs=self.rng,
            random_init=force_random_weights,
            epsilon=rms_norm_eps,
            with_scale=True,
            dtype=dtype,
        )

        self.lm_head = LMhead(vocab_size=vocab_size,
                              hidden_size=self.hidden_size,
                              dtype=dtype,
                              rngs=self.rng,
                              dv_sharding=(None, 'model'),
                              random_init=force_random_weights)

    def load_weights(self, rng_key: jax.Array):
        self.rng = nnx.Rngs(rng_key)

        transpose_map = {
            "lm_head": (1, 0),
            "gate_proj": (1, 0),
            "up_proj": (1, 0),
            "down_proj": (1, 0),
            "q_proj": (2, 0, 1),
            "k_proj": (2, 0, 1),
            "v_proj": (2, 0, 1),
            "o_proj": (1, 2, 0),
        }
        weight_shape_map = {
            "q_proj": (self.num_attention_heads, -1, self.hidden_size),
            "k_proj": (self.num_key_value_heads, -1, self.hidden_size),
            "v_proj": (self.num_key_value_heads, -1, self.hidden_size),
            "o_proj": (self.hidden_size, self.num_attention_heads, -1),
        }
        bias_shape_map = {
            "q_proj.bias": (self.num_attention_heads, self.head_dim),
            "k_proj.bias": (self.num_key_value_heads, self.head_dim),
            "v_proj.bias": (self.num_key_value_heads, self.head_dim),
        }
        loaded_to_standardized_keys = {
            "model.embed_tokens": "embedder.input_embedding_table_VD",
            "model.layers.*.input_layernorm":
            "layers.*.pre_attention_norm.scale",
            "model.layers.*.mlp.down_proj":
            "layers.*.custom_module.kernel_down_proj_FD",
            "model.layers.*.mlp.gate_proj":
            "layers.*.custom_module.kernel_gating_DF",
            "model.layers.*.mlp.up_proj":
            "layers.*.custom_module.kernel_up_proj_DF",
            "model.layers.*.post_attention_layernorm":
            "layers.*.pre_mlp_norm.scale",
            "model.layers.*.self_attn.k_proj":
            "layers.*.attn.kernel_k_proj_DKH",
            "model.layers.*.self_attn.o_proj":
            "layers.*.attn.kernel_o_proj_NHD",
            "model.layers.*.self_attn.q_proj":
            "layers.*.attn.kernel_q_proj_DNH",
            "model.layers.*.self_attn.v_proj":
            "layers.*.attn.kernel_v_proj_DKH",
            "model.norm": "final_norm.scale",
            "lm_head": "lm_head.input_embedding_table_DV",
        }

        metadata_map = MetadataMap(name_map=loaded_to_standardized_keys,
                                   reshape_map=weight_shape_map,
                                   bias_reshape_map=bias_shape_map,
                                   transpose_map=transpose_map)
        load_hf_weights(vllm_config=self.vllm_config,
                        model=self,
                        metadata_map=metadata_map,
                        mesh=self.mesh)

    def _build_jax_attention_metadata(
            self, positions: jax.Array,
            attn_metadata: dict) -> AttentionMetadata:
        """Translates vLLM-style metadata into the JAX AttentionMetadata required by layers."""
        return AttentionMetadata(
            input_positions=positions,
            block_tables=attn_metadata.block_tables,
            seq_lens=attn_metadata.seq_lens,
            query_start_loc=attn_metadata.query_start_loc,
            request_distribution=attn_metadata.request_distribution,
        )

    def forward(
        self,
        input_ids: jax.Array,
        positions: jax.Array,
        kv_caches: List[jax.Array],
        attn_metadata: AttentionMetadata,
    ) -> Tuple[List[KVCacheType], jax.Array]:
        """vLLM-compatible forward pass that remains functionally pure for JAX."""
        jax_attn_metadata = self._build_jax_attention_metadata(
            positions, attn_metadata)

        is_prefill = False
        with jax.named_scope("llama_embed_input"):
            x_TD = self.embedder.encode(input_ids)

        with jax.named_scope("llama_model_transformer_blocks"):
            new_kv_caches = []
            for i, layer in enumerate(self.layers):
                kv_cache = kv_caches[i]

                # The first layer is unscoped to avoid JAX tracing issues.
                # JAX's profiler may incorrectly apply the scope name from the first
                # layer's kernel compilation to all subsequent layers. Skipping the
                # first layer ensures distinct scope names for the remaining layers.
                if i == 0:
                    new_kv_cache, x_TD = layer(x_TD, is_prefill, kv_cache,
                                               jax_attn_metadata)
                else:
                    with jax.named_scope(f'layer_{i}'):
                        new_kv_cache, x_TD = layer(x_TD, is_prefill, kv_cache,
                                                   jax_attn_metadata)

                new_kv_caches.append(new_kv_cache)

        with jax.named_scope("llama_final_norm"):
            final_activation_TD = self.final_norm(x_TD)

        return new_kv_caches, final_activation_TD

    def __call__(
        self,
        kv_caches: List[jax.Array],
        input_ids: jax.Array,
        attn_metadata: AttentionMetadata,
        *args,
    ) -> Tuple[List[KVCacheType], jax.Array]:
        """Standard JAX/NNX entry point, delegating to the forward method."""
        return self.forward(
            input_ids=input_ids,
            positions=attn_metadata.input_positions,
            kv_caches=kv_caches,
            attn_metadata=attn_metadata,
        )

    def compute_logits(
        self,
        hidden_states: jax.Array,
        sampling_metadata: Optional[SamplingMetadata] = None,
    ) -> jax.Array:
        """Computes logits from hidden states."""
        with jax.named_scope("llama_lm_head_projection"):
            logits_TV = jnp.dot(hidden_states,
                                self.lm_head.input_embedding_table_DV.value)
        return logits_TV
