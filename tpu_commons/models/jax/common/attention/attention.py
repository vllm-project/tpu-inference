from dataclasses import dataclass, field, make_dataclass
from typing import Any, Dict, Tuple

import jax
import jax.numpy as jnp
from flax import nnx
from jax.experimental import shard_map
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from jax.typing import DTypeLike
from microbenchmark_utils import VllmConfig

from tpu_commons.kernels.ragged_paged_attention.kernel import \
    ragged_paged_attention
from tpu_commons.kernels.ragged_paged_attention.v3.kernel import (
    prepare_inputs, prepare_outputs)
from tpu_commons.kernels.ragged_paged_attention.v3.kernel import \
    ragged_paged_attention as ragged_paged_attention_v3
from tpu_commons.kernels.ragged_paged_attention.v3.util import \
    get_dtype_packing
from tpu_commons.models.jax.attention_interface import update_kv_cache
from tpu_commons.models.jax.attention_metadata import AttentionMetadata
from tpu_commons.models.jax.common.base import Config, ParamFactory
from tpu_commons.models.jax.common.constants import HuggingFaceArgNames
from tpu_commons.models.jax.common.sharding import ShardingConfig
from tpu_commons.models.jax.layers.rope import apply_rope

KVCache = Tuple[jax.Array, jax.Array]

AttentionConfig = make_dataclass(
    "AttentionConfig",
    [(HuggingFaceArgNames.HIDDEN_SIZE.value, int),
     (HuggingFaceArgNames.NUM_ATTENTION_HEADS.value, int),
     (HuggingFaceArgNames.NUM_KEY_VALUE_HEADS.value, int),
     (HuggingFaceArgNames.HEAD_DIM.value, int),
     (HuggingFaceArgNames.ROPE_SCALING.value, Dict[str, Any]),
     (HuggingFaceArgNames.ROPE_THETA.value, float), ("dtype", DTypeLike),
     ("rope_input_ordering", str, "split"),
     (HuggingFaceArgNames.ATTENTION_CHUNK_SIZE.value, int, None),
     ("vllm_config", VllmConfig, field(repr=False, default=None))],
    bases=(Config, ))
AttentionConfig.__doc__ = f"""Configuration for the Attention module.
         Attributes:
        {HuggingFaceArgNames.HIDDEN_SIZE.value}: The dimension of the model.
        {HuggingFaceArgNames.NUM_ATTENTION_HEADS.value}: The number of query heads.
        {HuggingFaceArgNames.NUM_KEY_VALUE_HEADS.value}: The number of key/value heads.
        {HuggingFaceArgNames.HEAD_DIM.value}: The dimension of each attention head.
        {HuggingFaceArgNames.ROPE_SCALING.value}: Optional dictionary of scaling factors for RoPE.
        {HuggingFaceArgNames.ROPE_THETA.value}: The base period for Rotary Position Embeddings.
        {HuggingFaceArgNames.ATTENTION_CHUNK_SIZE.value}: The chunk size for sliding window attention.
         dtype: The data type for computations (default: jnp.float32).
         "rope_input_ordering": Whether the inputs to rotate should be adjacent, interleaved dimensions or first and second half of the dimensions.
         vllm_config: The VLLM config containing any overrides to apply.
    """


@dataclass
class Attention(nnx.Module):
    """An implementation of attention.

    This module performs the attention mechanism for a transformer model,
    including query, key, and value projections, application of Rotary
    Position Embeddings (RoPE), and management of a KV cache for efficient
    autoregressive generation. It supports both prefill and generation
    (decode) modes and handles tensor sharding for distributed computation.

    Attributes:
        cfg: The configuration object of type `AttentionConfig`.
        mesh: The JAX device mesh for distributed computation.
        param_factory: A factory for creating and initializing model parameters.
        sharding_cfg: Configuration for tensor sharding strategies.
        quant: Optional configuration for quantization.
    """
    cfg: AttentionConfig
    mesh: Mesh
    param_factory: ParamFactory
    sharding_cfg: ShardingConfig
    quant: Any | None = None

    def __post_init__(self):
        self.create_sharding()

    def generate_kernel(self, rngs: nnx.Rngs):
        """Initializes the weight kernels for Q, K, V, and O projections."""
        N = getattr(self.cfg, HuggingFaceArgNames.NUM_ATTENTION_HEADS.value)
        K = getattr(self.cfg, HuggingFaceArgNames.NUM_KEY_VALUE_HEADS.value)
        D = getattr(self.cfg, HuggingFaceArgNames.HIDDEN_SIZE.value)
        H = getattr(self.cfg, HuggingFaceArgNames.HEAD_DIM.value)

        self.kernel_q_proj_DNH = self.param_factory.create_kernel_param(
            rngs, (D, N, H), self.dnh_sharding, self.cfg.dtype)
        self.kernel_k_proj_DKH = self.param_factory.create_kernel_param(
            rngs, (D, K, H), self.dkh_sharding, self.cfg.dtype)
        self.kernel_v_proj_DKH = self.param_factory.create_kernel_param(
            rngs, (D, K, H), self.dkh_sharding, self.cfg.dtype)
        self.kernel_o_proj_NHD = self.param_factory.create_kernel_param(
            rngs, (N, H, D), self.nhd_sharding, self.cfg.dtype)

    def create_sharding(self):
        """Creates sharding rules for activations and weights."""
        mode_dependent_attrs = [
            "activation_attention_td", "activation_q_td", "query_tnh",
            "keyvalue_skh", "activation_attention_out_td"
        ]
        for attr_name in mode_dependent_attrs:
            prefill_sharding_config = getattr(self.sharding_cfg.prefill_rules,
                                              attr_name)
            generate_sharding_config = getattr(
                self.sharding_cfg.generate_rules, attr_name)

            sharding_dict = {
                'prefill': NamedSharding(self.mesh,
                                         P(*prefill_sharding_config)),
                'generate': NamedSharding(self.mesh,
                                          P(*generate_sharding_config))
            }
            setattr(self, attr_name, sharding_dict)

        # static sharding for kernel/weights
        self.dnh_sharding = NamedSharding(
            self.mesh, P(*self.sharding_cfg.generate_rules.attn_q_weight_dnh))
        self.dkh_sharding = NamedSharding(
            self.mesh, P(*self.sharding_cfg.generate_rules.attn_k_weight_dkh))
        self.nhd_sharding = NamedSharding(
            self.mesh, P(*self.sharding_cfg.generate_rules.attn_o_weight_nhd))

        # TODO: the pallas kernels of flash_attention/paged_attention need to be called
        # via shard_map with sharding specs, However, the q/k/v have been sharded outside of attention()
        # So we replicate the sharding below but it should be better organized if we use pallas kernels
        self.pallas_q_spec = {
            'prefill': P(*self.sharding_cfg.prefill_rules.query_tnh),
            'generate': P(*self.sharding_cfg.generate_rules.query_tnh)
        }
        self.pallas_kv_spec = {
            'prefill': P(*self.sharding_cfg.prefill_rules.keyvalue_skh),
            'generate': P(*self.sharding_cfg.generate_rules.keyvalue_skh)
        }
        self.pallas_cache_page_spec = {
            'prefill': P(*self.sharding_cfg.prefill_rules.keyvalue_cache_lskh),
            'generate':
            P(*self.sharding_cfg.generate_rules.keyvalue_cache_lskh)
        }

    def __call__(self,
                 x,
                 is_prefill,
                 kv_cache: KVCache,
                 attention_metadata: AttentionMetadata,
                 use_attention_rope: bool = True):
        """Performs the forward pass of the attention module.

        This method computes the attention output by projecting the input `x`
        to queries, keys, and values, applying RoPE, performing scaled
        dot-product attention, and projecting the result back to the model
        dimension. It updates and utilizes a KV cache.

        Args:
            x: The input tensor of shape `(seq_len, d_model)`.
            is_prefill: Whether the operation mode is prefill (otherwise it is generate).
            kv_cache: The key-value cache for storing past attention states.
            attention_metadata: Metadata for attention, such as input positions.
            use_attention_rope: Whether to use RoPE.

        Returns:
            A tuple containing:
                - The updated KV cache.
                - The attention output tensor of shape
                  `(batch_size, seq_len, d_model)`.
        """
        op_mode = "prefill" if is_prefill else "generate"
        md = attention_metadata
        x_SD = jnp.asarray(x, self.cfg.dtype)
        x_q_TD = nnx.with_sharding_constraint(x, self.activation_q_td[op_mode])
        rope_scaling = getattr(self.cfg,
                               HuggingFaceArgNames.ROPE_SCALING.value)
        rope_theta = getattr(self.cfg, HuggingFaceArgNames.ROPE_THETA.value)
        rope_input_ordering = self.cfg.rope_input_ordering
        H = getattr(self.cfg, HuggingFaceArgNames.HEAD_DIM.value)
        with jax.named_scope("q_proj"):
            q_TNH = jnp.einsum('TD,DNH -> TNH', x_q_TD,
                               self.kernel_q_proj_DNH.value)
            if use_attention_rope:
                q_TNH = apply_rope(q_TNH, md.input_positions, H, rope_theta,
                                   rope_scaling, rope_input_ordering)
            q_TNH = nnx.with_sharding_constraint(q_TNH,
                                                 self.query_tnh[op_mode])
        with jax.named_scope("k_proj"):
            k_SKH = jnp.einsum('SD,DKH -> SKH', x_SD,
                               self.kernel_k_proj_DKH.value)
            if use_attention_rope:
                k_SKH = apply_rope(k_SKH, md.input_positions, H, rope_theta,
                                   rope_scaling, rope_input_ordering)
            k_SKH = nnx.with_sharding_constraint(k_SKH,
                                                 self.keyvalue_skh[op_mode])

        with jax.named_scope("v_proj"):
            v_SKH = jnp.einsum('SD,DKH -> SKH', x_SD,
                               self.kernel_v_proj_DKH.value)

        with jax.named_scope("attn_op"):
            new_kv_cache, outputs_TNH = self.attention(
                is_prefill,
                kv_cache,
                q_TNH,
                k_SKH,
                v_SKH,
                attention_metadata,
                self.mesh,
            )

        with jax.named_scope("o_proj"):
            o_TD = jnp.einsum('TNH,NHD -> TD', outputs_TNH,
                              self.kernel_o_proj_NHD.value)
        return new_kv_cache, o_TD

    def get_cfg(self) -> AttentionConfig:
        return self.cfg

    def attention(
        self,
        is_prefill: bool,
        kv_cache: KVCache,
        q_TNH: jax.Array,
        k_SKH: jax.Array,
        v_SKH: jax.Array,
        attention_metadata: AttentionMetadata,
        mesh: Mesh,
    ) -> Tuple[KVCache, jax.Array]:
        """Performs scaled dot-product attention and updates the KV cache.

        This function handles the core attention logic, which varies between
        prefill and generation modes. In prefill, it computes self-attention
        over the input sequence with a causal mask. In generation, it attends
        to the full history of keys and values stored in the cache.

        Args:
            is_prefill: A boolean indicating if the mode is 'prefill'.
            kv_cache: The key-value cache to be updated and used.
            q_TNH: Query tensor of shape `(query_seq, num_attention_heads, head_dim)`.
            k_SKH: Key tensor of shape `(kv_seq, num_key_value_heads, head_dim)`.
            v_SKH: Value tensor of shape `(kv_seq, num_key_value_heads, head_dim)`.
            attention_metadata: Metadata containing sequence lengths.
            mesh: The JAX device mesh (unused in this specific function but
                kept for potential future use or API consistency).
            attention_chunk_size: The chunk size during sliding window attention.

        Returns:
            A tuple containing:
                - The updated KV cache.
                - The attention output tensor of shape
                  `(seq, num_q_heads, head_dim)`.
        """
        md = attention_metadata
        kv_cache = update_kv_cache(k_SKH, v_SKH, kv_cache, md.slot_mapping,
                                   md.num_slices, mesh)

        H = q_TNH.shape[-1]
        #TODO: we use generate_rules as the default sharding for ragged_paged_attention,
        # but it could be configurable based on the op_mode.
        in_specs = (
            P(*self.sharding_cfg.generate_rules.query_tnh),  # q_TNH
            P(*self.sharding_cfg.generate_rules.keyvalue_cache_lskh
              ),  # kv_cache:
            P(),  # md.seq_lens: Replicated
            P(),  # md.block_tables: Replicated
            P(),  # md.query_start_loc: Replicated
            P(),  # md.num_seqs: Replicated
        )
        out_specs = P(*self.sharding_cfg.generate_rules.attn_o_tnh
                      )  # output_TNH: Shard the 'model' dimension

        def _ragged_paged_attention(*args):
            return ragged_paged_attention(
                *args,
                sm_scale=H**-0.5,
                soft_cap=None,
                mask_value=None,
                # NOTE(xiang): v6e chip has 128M VMEM capacity,
                # set this to 64M to avoid VMEM OOM,
                # otherwise the default value is 16M.
                sliding_window=getattr(self.cfg, "attention_chunk_size", None),
                vmem_limit_bytes=64 * 1024 * 1024,
            )

        output_TNH = jax.jit(
            shard_map.shard_map(
                _ragged_paged_attention,
                mesh=mesh,
                in_specs=in_specs,
                out_specs=out_specs,
                check_rep=False,
            ))(
                q_TNH,
                kv_cache,
                md.seq_lens,
                md.block_tables,
                md.query_start_loc,
                md.num_seqs,
            )

        return kv_cache, output_TNH

    def attention_v3(
        self,
        is_prefill: bool,
        kv_cache: KVCache,
        q_TNH: jax.Array,
        k_SKH: jax.Array,
        v_SKH: jax.Array,
        attention_metadata: AttentionMetadata,
        mesh: Mesh,
    ) -> Tuple[KVCache, jax.Array]:
        """Performs scaled dot-product attention and updates the KV cache.

        This function handles the core attention logic, which varies between
        prefill and generation modes. In prefill, it computes self-attention
        over the input sequence with a causal mask. In generation, it attends
        to the full history of keys and values stored in the cache.

        Args:
            is_prefill: A boolean indicating if the mode is 'prefill'.
            kv_cache: The key-value cache to be updated and used.
            q_TNH: Query tensor of shape `(query_seq, num_attention_heads, head_dim)`.
            k_SKH: Key tensor of shape `(kv_seq, num_key_value_heads, head_dim)`.
            v_SKH: Value tensor of shape `(kv_seq, num_key_value_heads, head_dim)`.
            attention_metadata: Metadata containing sequence lengths.
            mesh: The JAX device mesh (unused in this specific function but
                kept for potential future use or API consistency).

        Returns:
            A tuple containing:
                - The updated KV cache.
                - The attention output tensor of shape
                  `(seq, num_q_heads, head_dim)`.
        """
        md = attention_metadata
        kv_cache = update_kv_cache(k_SKH, v_SKH, kv_cache, md.slot_mapping,
                                   md.num_slices, mesh)
        q_transformed, actual_num_q_heads_per_kv_head, actual_head_dim = prepare_inputs(
            q_TNH, self.K)
        kv_packing = get_dtype_packing(
            kv_cache.dtype)  # 2 for bf16, 4 for fp8.
        L, S, K_2, H = kv_cache.shape
        kv_cache_transformed = kv_cache.reshape(L, S, K_2 // kv_packing,
                                                kv_packing, H)
        page_indices_flat = md.block_tables.flatten()

        num_decode, num_prefill, _ = md.request_distribution
        distribution = jnp.array(
            [num_decode, num_decode + num_prefill, md.num_seqs[0]])
        in_specs = (
            P(*self.sharding_cfg.generate_rules.query_ktnph),  # q_transformed
            P(*self.sharding_cfg.generate_rules.keyvalue_cache_nbkph
              ),  # kv_cache_transformed
            P(),  # md.seq_lens: Replicated
            P(),  # page_indices_flat: Replicated
            P(),  # query_start_loc: Replicated
            P(),  # distribution: Replicated
        )
        out_specs = P(*self.sharding_cfg.generate_rules.attn_o_ktnph
                      )  #output_transformed

        def _ragged_paged_attention(*args):
            return ragged_paged_attention_v3(
                *args,
                sm_scale=q_TNH.shape[-1]**-0.5,
                sliding_window=None,
                soft_cap=None,
                vmem_limit_bytes=64 * 1024 * 1024,
            )

        output_transformed = jax.jit(
            shard_map.shard_map(
                _ragged_paged_attention,
                mesh=mesh,
                in_specs=in_specs,
                out_specs=out_specs,
                check_rep=False,
            ))(
                q_transformed,
                kv_cache_transformed,
                md.seq_lens,
                page_indices_flat,
                md.query_start_loc,
                distribution,
            )
        output_TNH = prepare_outputs(output_transformed,
                                     actual_num_q_heads_per_kv_head,
                                     actual_head_dim)
        return kv_cache, output_TNH
