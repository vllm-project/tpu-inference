import jax
import jax.numpy as jnp
import numpy as np
from dataclasses import dataclass, fields
from typing import Dict, Tuple, Any, Callable
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from flax import nnx
from jaxtyping import Float, Array, Int

from tpu_commons.models.jax.layers import sharded_flash_attention, sharded_paged_attention
from tpu_commons.models.jax.attention_metadata import AttentionMetadata
from tpu_commons.models.jax.rope.generic_rope import apply_rope

from tpu_commons.models.jax.common.sharding import *
from tpu_commons.models.jax.common.constants import *
from tpu_commons.models.jax.common.layers import *
from tpu_commons.models.jax.common.kv_cache import *

# A lightweight block serves as a blue-print.
@dataclass
class AttentionConfig(Config):
    d_model: int
    num_q_heads: int
    num_kv_heads: int   
    head_dim: int
    rope_theta: float = 10000.0
    rope_scaling: Dict[str, Any] = None
    dtype: Any = jnp.float32

# A heavy weight module which serves as the stateful live blocks in serving
@dataclass
class Attention(nnx.Module):
    """Attention Block"""
    cfg: AttentionConfig
    mesh: Mesh
    param_factory: ParamFactory
    sharding_cfg: ShardingConfig
    quant: Any | None = None

    def __post_init__(self):
        self.create_sharding()
        self._generate_kernel()

    def _generate_kernel(self):

        shape_q_proj = (self.cfg.num_q_heads, self.cfg.d_model, self.cfg.head_dim)
        shape_kv_proj = (self.cfg.num_kv_heads, self.cfg.d_model, self.cfg.head_dim)
        shape_o_proj = (self.cfg.num_q_heads, self.cfg.head_dim, self.cfg.d_model)

        self.kernel_q_proj_QDH = self.param_factory.create_kernel_init(shape_q_proj, self.qdh_sharding, self.cfg.dtype)
        self.kernel_k_proj_NDH = self.param_factory.create_kernel_init(shape_kv_proj, self.ndh_sharding, self.cfg.dtype)
        self.kernel_v_proj_NDH = self.param_factory.create_kernel_init(shape_kv_proj, self.ndh_sharding, self.cfg.dtype)
        self.kernel_o_proj_QHD = self.param_factory.create_kernel_init(shape_o_proj, self.qhd_sharding, self.cfg.dtype)

    def create_sharding(self):

        mode_dependent_attrs = [
            "activation_attention_bsd",
            "activation_q_bsd",
            "query_bsqh",
            "keyvalue_bsnh",
            "activation_attention_out_bsd"
        ]
        for attr_name in mode_dependent_attrs:
            prefill_sharding_config = getattr(self.sharding_cfg.prefill_sharding_cfg, attr_name)
            generate_sharding_config = getattr(self.sharding_cfg.generate_sharding_cfg, attr_name)

            sharding_dict = {
                'prefill': NamedSharding(
                    self.mesh, P(prefill_sharding_config)),
                'generate': NamedSharding(
                    self.mesh, P(generate_sharding_config))
            }
            setattr(self, attr_name, sharding_dict)

        # static sharding for kernel/weights
        self.qdh_sharding = NamedSharding(
            self.mesh, P(self.sharding_cfg.generate_sharding_cfg.attn_q_weight_qdh))
        self.ndh_sharding = NamedSharding(
            self.mesh, P(self.sharding_cfg.generate_sharding_cfg.attn_k_weight_ndh))
        self.qhd_sharding = NamedSharding(
            self.mesh, P(self.sharding_cfg.generate_sharding_cfg.attn_o_weight_qhd))

    def __call__(
        self, 
        x,
        op_mode,
        kv_cache: KVCache,
        attention_metadata: AttentionMetadata,
    ):  
        md = attention_metadata
        is_prefill = True if op_mode == 'prefill' else False
        x = jnp.asarray(x, self.cfg.dtype)
        x_BSD = nnx.with_sharding_constraint(x, self.activation_attention_bsd[op_mode])
        x_q_BSD = nnx.with_sharding_constraint(x, self.activation_q_bsd[op_mode]) 


        with jax.named_scope("q_proj"):
            q_BSQH = jnp.einsum('BSD,QDH -> BSQH', x_q_BSD, self.kernel_q_proj_QDH.value)
            q_BSQH = apply_rope(
                q_BSQH, md.input_positions, self.cfg.head_dim, self.cfg.rope_theta,
                self.cfg.rope_scaling)
            q_BSQH = nnx.with_sharding_constraint(q_BSQH, self.query_bsqh[op_mode])
        with jax.named_scope("k_proj"):
            k_BSNH = jnp.einsum('BSD,NDH -> BSNH', x, self.kernel_k_proj_NDH.value)
            k_BSNH = apply_rope(
                k_BSNH, md.input_positions, self.cfg.head_dim, self.cfg.rope_theta,
                self.cfg.rope_scaling)
            k_BSNH = nnx.with_sharding_constraint(k_BSNH, self.keyvalue_bsnh[op_mode]) 
    
        with jax.named_scope("v_proj"):
            v_BSNH = jnp.einsum('BSD,NDH -> BSNH', x, self.kernel_v_proj_NDH.value)
            v_BSNH = nnx.with_sharding_constraint(v_BSNH, self.keyvalue_bsnh[op_mode]) 

        with jax.named_scope("attn_op"):
            new_kv_cache, outputs_BSQH = self.base_attention(
                is_prefill,
                kv_cache,
                q_BSQH,
                k_BSNH,
                v_BSNH,
                attention_metadata,
                self.mesh,
                self.cfg.num_q_heads,
                self.cfg.num_kv_heads,
            )

        with jax.named_scope("o_proj"):
            o_BSNH = jnp.einsum('BSQH,QHD -> BSD', outputs_BSQH, self.kernel_o_proj_QHD.value)
            o_BSNH = nnx.with_sharding_constraint(o_BSNH, self.activation_attention_out_bsd[op_mode])
        return new_kv_cache, o_BSNH


    def get_cfg(self) -> AttentionConfig:
        return self.cfg

    def base_attention(
            self,
            is_prefill: bool,
            kv_cache: KVCache,
            q_BSQH: jax.Array,
            k_BTNH: jax.Array,
            v_BTNH: jax.Array,
            attention_metadata: AttentionMetadata,
            mesh: Mesh,
            num_heads: int,
            num_kv_heads: int,
        ) -> Tuple[KVCache, jax.Array]:
            """
            Performs standard scaled dot-product attention using jnp.einsum,
            updates the KV cache, and returns the attention output.

            Args:
                is_prefill: Flag indicating if this is a prefill or generation step.
                kv_cache: The key-value cache object.
                q: Query tensors.
                k: Key tensors.
                v: Value tensors.
                attention_metadata: Metadata containing sequence lengths.
                mesh: JAX mesh (unused).
                num_heads: Number of query heads.
                num_kv_heads: Number of key/value heads.

            Returns:
                A tuple containing the updated KVCache and the attention output.
            """
            # Determine the operating mode for the cache
            op_mode = 'prefill' if is_prefill else 'generate'

            # For GQA/MQA, the ratio of query heads to key/value heads.
            head_repeats = num_heads // num_kv_heads

            if is_prefill:
                kv_cache.update(k_BTNH, v_BTNH, attention_metadata.seq_lens, op_mode="prefill")
                # Repeat K and V heads if necessary for MQA/GQA
                k_attn_BTQH = jnp.repeat(k_BTNH, head_repeats, axis=2) if head_repeats > 1 else k
                v_attn_BTQH = jnp.repeat(v_BTNH, head_repeats, axis=2) if head_repeats > 1 else v

                # Calculate attention scores using einsum
                # Here, query sequence length S is the same as key sequence length T.
                scores_BQST = jnp.einsum('BSQH,BTQH->BQST', q_BSQH, k_attn_BTQH) / jnp.sqrt(self.cfg.head_dim)

                # Apply causal mask to prevent attending to future tokens
                seq_len = q_BSQH.shape[1]
                causal_mask_ST = jnp.tril(jnp.ones((1, 1, seq_len, seq_len), dtype=jnp.bool_))
                scores_BQST = jnp.where(causal_mask_ST, scores_BQST, -1e9)

                # Compute attention weights
                attention_weights_BQST = jax.nn.softmax(scores_BQST, axis=-1)

                # Compute output using einsum
                output_BSQH = jnp.einsum('BQST,BTQH->BSQH', attention_weights_BQST, v_attn_BTQH)

            else:
                # Retrieve the full K and V sequences from the cache
                kv_cache.update(k_BTNH, v_BTNH, attention_metadata.seq_lens, op_mode="generate")
                k_cache_full = kv_cache.key_cache[op_mode]
                v_cache_full = kv_cache.value_cache[op_mode]

                # Repeat K and V heads if necessary for MQA/GQA
                if head_repeats > 1:
                    k_cache_full = jnp.repeat(k_cache_full, head_repeats, axis=2)
                    v_cache_full = jnp.repeat(v_cache_full, head_repeats, axis=2)

                # Calculate attention scores using einsum
                scores_BQST = jnp.einsum('BSQH,BTQH->BQST', q_BSQH, k_cache_full) / jnp.sqrt(self.cfg.head_dim)

                # Apply mask to ignore padding in the KV cache
                current_lengths_B = attention_metadata.seq_lens + 1
                max_cache_len = k_cache_full.shape[1]
                mask = jnp.arange(max_cache_len) < current_lengths_B[:, None]
                scores_BQST = jnp.where(mask[:, None, None, :], scores_BQST, -1e9)

                # Compute attention weights
                attention_weights_BQST = jax.nn.softmax(scores_BQST, axis=-1)

                # Compute output using einsum
                output_BSQH = jnp.einsum('BQST,BTQH->BSQH', attention_weights_BQST, v_cache_full)

            return kv_cache, output_BSQH

    # def attention(
    #     self,
    #     is_prefill: bool,
    #     kv_cache: KVCache,
    #     q: jax.Array,
    #     k: jax.Array,
    #     v: jax.Array,
    #     attention_metadata: AttentionMetadata,
    #     mesh: Mesh,
    #     num_heads: int,
    #     num_kv_heads: int,
    # ) -> KVCache:
    #     # B: batch_size
    #     # T: seq_len
    #     # N: num_heads
    #     # K: num_kv_heads
    #     # D: hidden_size
    #     # H: head_dim
    #     # L: num_blocks
    #     # S: block_size

    #     # (K, L, S, H)
    #     k_cache = kv_cache.key_cache
    #     v_cache = kv_cache.value_cache
    #     md = attention_metadata
    #     if md.chunked_prefill_enabled:
    #         k_cache = sharded_chunked_prefill_update_cache(mesh)(
    #             k_cache, md.kv_cache_write_indices, k, md.num_decode_seqs)
    #         v_cache = sharded_chunked_prefill_update_cache(mesh)(
    #             v_cache, md.kv_cache_write_indices, v, md.num_decode_seqs)
    #         outputs = sharded_chunked_prefill_attention(mesh)(
    #             q,
    #             k_cache,
    #             v_cache,
    #             attention_metadata.decode_lengths,
    #             attention_metadata.decode_page_indices,
    #             attention_metadata.num_decode_seqs,
    #             attention_metadata.prefill_lengths,
    #             attention_metadata.prefill_page_indices,
    #             attention_metadata.prefill_query_start_offsets,
    #             attention_metadata.num_prefill_seqs,
    #         )
    #     else:
            
    #         if is_prefill:
    #             # (B, N, T, H)
    #             # TODO(xiang): support MQA and GQA
    #             kv_cache.update(k, v, md.seq_lens, 'prefill')
    #             if num_kv_heads != num_heads:
    #                 k = jnp.repeat(k, num_heads // num_kv_heads, axis=1)
    #                 v = jnp.repeat(v, num_heads // num_kv_heads, axis=1)
    #             outputs = sharded_flash_attention(mesh)(q, k, v)
    #         else:
    #             # (B, N, H)
    #             kv_cache.update(k, v, md.seq_lens,'generate')
    #             q = jnp.squeeze(q, 2)
    #             outputs = sharded_paged_attention(mesh)(q, k_cache, v_cache,
    #                                                     md.seq_lens,
    #                                                     md.block_indices)
    #             # (B, N, 1, H)
    #             outputs = jnp.expand_dims(outputs, 2)


    #    return kv_cache, outputs