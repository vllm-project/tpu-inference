import math
from dataclasses import dataclass, field, make_dataclass
from typing import Any, Dict, Tuple

import jax
import jax.numpy as jnp
from flax import nnx
from jax.experimental import shard_map
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from jax.typing import DTypeLike
from vllm.config import VllmConfig

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
from tpu_commons.models.jax.common.layers import RMSNorm
from tpu_commons.models.jax.common.rope import DeepseekScalingRotaryEmbedding

KVCache = Tuple[jax.Array, jax.Array]

MLAConfig = make_dataclass(
    "MLAConfig",
    [
        (HuggingFaceArgNames.HIDDEN_SIZE.value, int),
        (HuggingFaceArgNames.NUM_ATTENTION_HEADS.value, int),
        (HuggingFaceArgNames.NUM_KEY_VALUE_HEADS.value, int),
        (HuggingFaceArgNames.ROPE_THETA.value, float),
        (HuggingFaceArgNames.ROPE_SCALING.value, Dict[str, Any]),
        (HuggingFaceArgNames.Q_LORA_RANK.value, int),
        (HuggingFaceArgNames.KV_LORA_RANK.value, int),
        (HuggingFaceArgNames.QK_NOPE_HEAD_DIM.value, int),
        (HuggingFaceArgNames.QK_ROPE_HEAD_DIM.value, int),
        (HuggingFaceArgNames.V_HEAD_DIM.value, int),
        (HuggingFaceArgNames.RMS_NORM_EPS.value, float),
        (
            "dtype",
            DTypeLike,
        ),
        ("vllm_config", VllmConfig, field(repr=False, default=None)),
    ],
    bases=(Config, ),
)

MLAConfig.__doc__ = f"""Configuration for the MLA module.
         Attributes:
        {HuggingFaceArgNames.HIDDEN_SIZE.value}: The dimension of the model.
        {HuggingFaceArgNames.NUM_ATTENTION_HEADS.value}: The number of query heads.
        {HuggingFaceArgNames.NUM_KEY_VALUE_HEADS.value}: The number of key/value heads.
        {HuggingFaceArgNames.ROPE_THETA.value}: The base period for Rotary Position Embeddings.
        {HuggingFaceArgNames.ROPE_SCALING.value}: Optional dictionary of scaling factors for RoPE.
        {HuggingFaceArgNames.Q_LORA_RANK.value}: The dimension for the latent query vector.
        {HuggingFaceArgNames.KV_LORA_RANK.value}: The dimension for the latent key/value vector.
        {HuggingFaceArgNames.QK_NOPE_HEAD_DIM.value}: The dimension of the no rope portion of the qv tensor.
        {HuggingFaceArgNames.QK_ROPE_HEAD_DIM.value}: The dimension of the rope portion of the qv tensor.
        {HuggingFaceArgNames.V_HEAD_DIM.value}: The dimension of the value vector.
        {HuggingFaceArgNames.RMS_NORM_EPS.value}: The epsilon for the RMS normalization.
    """


# TODO (wenxindongwork): Add MLA KV cache implementation. For now, cache complete KV vectors.
@dataclass(kw_only=True)
class MLA(nnx.Module):
    """An implementation of Multi-Head Latent Attention as
    described in the DeepSeek V3 paper.

    Attributes:
        mesh: The JAX device mesh for distributed computation.
        param_factory: A factory for creating and initializing model parameters.
        quant: Optional configuration for quantization.
    """
    hidden_size: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    rope_theta: float
    rope_scaling: dict[str, Any]
    dtype: jnp.dtype
    mesh: Mesh
    param_factory: ParamFactory

    q_lora_rank: int
    kv_lora_rank: int
    qk_nope_head_dim: int
    qk_rope_head_dim: int
    v_head_dim: int
    rms_norm_eps: float

    # Sharding attributes
    nhd_sharding: NamedSharding
    q_da_sharding: NamedSharding
    anh_sharding: NamedSharding
    kv_da_sharding: NamedSharding

    activation_attention_td: NamedSharding
    activation_q_td: NamedSharding
    query_tnh: NamedSharding
    query_ktnph: NamedSharding

    keyvalue_skh: NamedSharding
    keyvalue_cache_lskh: NamedSharding
    keyvalue_cache_nbkph: NamedSharding

    attn_o_tnh: NamedSharding
    attn_o_ktnph: NamedSharding
    activation_attention_out_td: NamedSharding

    attention_chunk_size: int | None = None
    rope_input_ordering: str = "split"
    quant: Any | None = None

    def __post_init__(self):
        self.N = self.num_attention_heads
        self.K = self.num_key_value_heads
        self.D = self.hidden_size

        self.qk_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim

        assert self.N == self.K, "N and K must be equal for MLA"

        self.rope = DeepseekScalingRotaryEmbedding(
            self.qk_rope_head_dim,
            self.rope_theta,
            self.rope_scaling["original_max_position_embeddings"],
            self.rope_scaling["factor"],
            self.dtype,
            beta_fast=self.rope_scaling["beta_fast"],
            beta_slow=self.rope_scaling["beta_slow"],
            mscale=self.rope_scaling["mscale"],
            mscale_all_dim=self.rope_scaling["mscale_all_dim"],
        )

    def generate_kernel(self, rngs: nnx.Rngs):
        """Initializes the weight kernels."""

        self.kernel_q_down_proj_DA = self.param_factory.create_kernel_param(
            rngs, (self.D, self.q_lora_rank), self.q_da_sharding, self.dtype)
        # FP8 Scale
        # TODO: update 128 to use config
        self.kernel_q_down_proj_scale_DA = self.param_factory.create_kernel_param(
            rngs, (self.D // 128, self.q_lora_rank // 128), self.q_da_sharding,
            self.dtype)
        self.kernel_q_up_proj_ANH = self.param_factory.create_kernel_param(
            rngs,
            (self.q_lora_rank, self.N, self.qk_head_dim),
            self.anh_sharding,
            self.dtype,
        )
        # FP8 Scale
        # TODO: update 192 to use config
        self.kernel_q_up_proj_scale_ANH = self.param_factory.create_kernel_param(
            rngs,
            (self.q_lora_rank // 128, 192, self.qk_head_dim // 128),
            self.anh_sharding,
            self.dtype,
        )
        self.kernel_kv_down_proj_DA = self.param_factory.create_kernel_param(
            rngs,
            (self.D, self.kv_lora_rank + self.qk_rope_head_dim),
            self.kv_da_sharding,
            self.dtype,
        )
        # FP8 Scale
        # TODO: update 128 to use config
        self.kernel_kv_down_proj_scale_DA = self.param_factory.create_kernel_param(
            rngs,
            (math.ceil(self.D // 128),
             math.ceil((self.kv_lora_rank + self.qk_rope_head_dim) / 128)),
            self.kv_da_sharding,
            self.dtype,
        )
        self.kernel_kv_up_proj_ANH = self.param_factory.create_kernel_param(
            rngs,
            (self.kv_lora_rank, self.N,
             self.qk_nope_head_dim + self.v_head_dim),
            self.anh_sharding,
            self.dtype,
        )
        # FP8 Scale
        # TODO: update 128 to use config
        self.kernel_kv_up_proj_scale_ANH = self.param_factory.create_kernel_param(
            rngs,
            (self.kv_lora_rank // 128, self.N,
             (self.qk_nope_head_dim + self.v_head_dim) // 128),
            self.anh_sharding,
            self.dtype,
        )
        self.kernel_o_proj_NHD = self.param_factory.create_kernel_param(
            rngs, (self.N, self.v_head_dim, self.D), self.nhd_sharding,
            self.dtype)
        # FP8 Scale
        # TODO: update 128 to use config
        self.kernel_o_proj_scale_NHD = self.param_factory.create_kernel_param(
            # TODO: 56
            rngs,
            (self.N, self.v_head_dim // 128, 56),
            self.nhd_sharding,
            self.dtype)
        self.q_rms_norm = RMSNorm(
            dims=self.q_lora_rank,
            mesh=self.mesh,
            param_factory=self.param_factory,
            activation_ffw_td=NamedSharding(self.mesh, P()),
            epsilon=self.rms_norm_eps,
            with_scale=True,
            dtype=jnp.bfloat16,
        )
        self.q_rms_norm.generate_kernel(rngs)

        self.kv_rms_norm = RMSNorm(
            dims=self.kv_lora_rank,
            mesh=self.mesh,
            param_factory=self.param_factory,
            activation_ffw_td=NamedSharding(self.mesh, P()),
            epsilon=self.rms_norm_eps,
            with_scale=True,
            dtype=jnp.bfloat16,
        )
        self.kv_rms_norm.generate_kernel(rngs)

    def __call__(self,
                 x,
                 is_prefill,
                 kv_cache: KVCache,
                 attention_metadata: AttentionMetadata,
                 use_attention_rope: bool = True):
        """Performs the forward pass of the attention module.

        Args:
            x: The input tensor of shape `(batch_size, seq_len, d_model)`.
            is_prefill: Whether the operation mode is prefill (otherwise it is generate).
            kv_cache: The key-value cache for storing past attention states.
            attention_metadata: Metadata for attention, such as input positions.

        Returns:
            A tuple containing:
                - The updated KV cache.
                - The attention output tensor of shape
                  `(batch_size, seq_len, d_model)`.
        """
        md = attention_metadata
        x = jnp.asarray(x, self.dtype)
        x_SD = nnx.with_sharding_constraint(x, self.activation_attention_td)
        x_q_TD = nnx.with_sharding_constraint(x, self.activation_q_td)

        with jax.named_scope("q_proj"):
            # Query down projection.
            q_TA = jnp.einsum("TD,DA -> TA", x_q_TD,
                              self.kernel_q_down_proj_DA.value)
            q_TA = self.q_rms_norm(q_TA)
            # Query up projection.
            q_TNH = jnp.einsum("TA,ANH -> TNH", q_TA,
                               self.kernel_q_up_proj_ANH.value)
            # Split the query into nope and rope.
            q_nope_TNH = q_TNH[..., :self.qk_nope_head_dim]
            q_rope_TNH = q_TNH[..., self.qk_nope_head_dim:]
            q_rope_TNH = self.rope.apply_rope(md.input_positions, q_rope_TNH)
            # Concatenate the nope and rope queries.
            q_TNH = jnp.concatenate([q_nope_TNH, q_rope_TNH], axis=-1)
            # Multiple the query by scaling factor
            q_TNH = q_TNH * self.qk_head_dim**-0.5
            q_TNH = nnx.with_sharding_constraint(q_TNH, self.query_tnh)

        with jax.named_scope("kv_proj"):
            # KV down projection.
            kv_SA = jnp.einsum("SD,DA -> SA", x_SD,
                               self.kernel_kv_down_proj_DA.value)
            # Split the key and value into latent kv vector and k rope vector.
            k_rope_SH = kv_SA[..., self.kv_lora_rank:]
            # Reshape k_rope_BSH to include head dimension for RoPE application
            k_rope_SNH = k_rope_SH[..., None, :]
            k_rope_SNH = self.rope.apply_rope(md.input_positions, k_rope_SNH)
            k_rope_SNH = jnp.broadcast_to(
                k_rope_SNH,
                (k_rope_SNH.shape[0], self.N, self.qk_rope_head_dim))
            kv_SA = kv_SA[..., :self.kv_lora_rank]
            kv_SA = self.kv_rms_norm(kv_SA)
            # KV up projection.
            kv_nope_SNH = jnp.einsum("SA,ANH -> SNH", kv_SA,
                                     self.kernel_kv_up_proj_ANH.value)
            # Split the latent kv vector into k nope vector and v vector.
            k_nope_SNH = kv_nope_SNH[..., :self.qk_nope_head_dim]
            v_SNH = kv_nope_SNH[..., self.qk_nope_head_dim:]
            # Concatenate the key vector.
            k_SNH = jnp.concatenate([k_nope_SNH, k_rope_SNH], axis=-1)
            k_SNH = nnx.with_sharding_constraint(k_SNH, self.keyvalue_skh)
            v_SNH = nnx.with_sharding_constraint(v_SNH, self.keyvalue_skh)

        with jax.named_scope("attn_op"):
            # TODO(wenxindongwork): K and V have different head dimension,
            # which is not supported by the current kv cache implementation.
            # For now we are padding the v dimension to match the k dimension.
            # Furthermore, deepseekv3 k head dimension is 192, which is
            # not supported by the current attention kernel, which expects
            # q, k, v head dimension to be multiple of 128. For now, we will
            # pad the q, k, v dimension to multiple of 128.
            # We should update the MLA kv cache implementation in the future.
            multiple_of_128 = ((self.qk_head_dim - 1) // 128 + 1) * 128
            q_TNH = jnp.pad(q_TNH, ((0, 0), (0, 0),
                                    (0, multiple_of_128 - self.qk_head_dim)))
            k_SNH = jnp.pad(k_SNH, ((0, 0), (0, 0),
                                    (0, multiple_of_128 - self.qk_head_dim)))
            v_SNH = jnp.pad(v_SNH, ((0, 0), (0, 0),
                                    (0, multiple_of_128 - self.v_head_dim)))
            new_kv_cache, outputs_TNH = self.attention_v3(
                is_prefill,
                kv_cache,
                q_TNH,
                k_SNH,
                v_SNH,
                attention_metadata,
                self.mesh,
            )
            # TODO(wenxindongwork): For now, unpad the outputs_TNH to match the v_head_dim.
            # We shall add the MLA kv cache implementation in the future.
            outputs_TNH = outputs_TNH[..., :self.v_head_dim]

        with jax.named_scope("o_proj"):
            o_TD = jnp.einsum("TNH,NHD -> TD", outputs_TNH,
                              self.kernel_o_proj_NHD.value)
            o_TD = nnx.with_sharding_constraint(
                o_TD, self.activation_attention_out_td)
        return new_kv_cache, o_TD

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
            self.query_tnh.spec,  # q_TNH
            self.keyvalue_cache_lskh.spec,  # kv_cache:
            P(),  # md.seq_lens: Replicated
            P(),  # md.block_tables: Replicated
            P(),  # md.query_start_loc: Replicated
            P(),  # md.num_seqs: Replicated
        )
        out_specs = self.attn_o_tnh.spec  # output_TNH: Shard the 'model' dimension

        def _ragged_paged_attention(*args):
            return ragged_paged_attention(
                *args,
                sm_scale=H**-0.5,
                soft_cap=None,
                mask_value=None,
                # NOTE(xiang): v6e chip has 128M VMEM capacity,
                # set this to 64M to avoid VMEM OOM,
                # otherwise the default value is 16M.
                sliding_window=self.attention_chunk_size,
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
            self.query_ktnph.spec,  # q_transformed
            self.keyvalue_cache_nbkph.spec,  # kv_cache_transformed
            P(),  # md.seq_lens: Replicated
            P(),  # page_indices_flat: Replicated
            P(),  # query_start_loc: Replicated
            P(),  # distribution: Replicated
        )
        out_specs = self.attn_o_ktnph.spec  #output_transformed

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
