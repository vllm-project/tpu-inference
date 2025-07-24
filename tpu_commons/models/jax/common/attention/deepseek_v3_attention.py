from dataclasses import dataclass, field, make_dataclass
from typing import Any, Dict, Tuple

import jax
import jax.numpy as jnp
from flax import nnx
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from jax.typing import DTypeLike
from vllm.config import VllmConfig
from jax.experimental import shard_map

from tpu_commons.models.jax.attention_metadata import AttentionMetadata
from tpu_commons.models.jax.common.attention.attention import (Attention,
                                                               KVCache)
from tpu_commons.models.jax.common.base import Config, ParamFactory
from tpu_commons.models.jax.common.constants import HuggingFaceArgNames
from tpu_commons.models.jax.common.rope import DeepseekScalingRotaryEmbedding
from tpu_commons.models.jax.common.sharding import ShardingConfig

from tpu_commons.models.jax.attention_interface import update_mla_kv_cache
from tpu_commons.kernels.ragged_paged_attention.mla_kernel import \
    ragged_mla_paged_attention


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
@dataclass
class MLA(Attention):
    """An implementation of Multi-Head Latent Attention as
    described in the DeepSeek V3 paper.

    Attributes:
        cfg: The configuration object of type `MLAConfig`.
        mesh: The JAX device mesh for distributed computation.
        param_factory: A factory for creating and initializing model parameters.
        sharding_cfg: Configuration for tensor sharding strategies.
        quant: Optional configuration for quantization.
    """

    cfg: MLAConfig
    mesh: Mesh
    param_factory: ParamFactory
    sharding_cfg: ShardingConfig
    quant: Any | None = None

    def __post_init__(self):
        self.N = getattr(self.cfg,
                         HuggingFaceArgNames.NUM_ATTENTION_HEADS.value)
        self.K = getattr(self.cfg,
                         HuggingFaceArgNames.NUM_KEY_VALUE_HEADS.value)
        self.D = getattr(self.cfg, HuggingFaceArgNames.HIDDEN_SIZE.value)
        self.query_lora_rank = getattr(self.cfg,
                                       HuggingFaceArgNames.Q_LORA_RANK.value)
        self.kv_lora_rank = getattr(self.cfg,
                                    HuggingFaceArgNames.KV_LORA_RANK.value)
        self.rope_scaling = getattr(self.cfg,
                                    HuggingFaceArgNames.ROPE_SCALING.value)
        self.rope_theta = getattr(self.cfg,
                                  HuggingFaceArgNames.ROPE_THETA.value)
        self.rms_norm_eps = getattr(self.cfg,
                                    HuggingFaceArgNames.RMS_NORM_EPS.value)
        self.qk_nope_head_dim = getattr(
            self.cfg, HuggingFaceArgNames.QK_NOPE_HEAD_DIM.value)
        self.qk_rope_head_dim = getattr(
            self.cfg, HuggingFaceArgNames.QK_ROPE_HEAD_DIM.value)
        self.qk_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        self.v_head_dim = getattr(self.cfg,
                                  HuggingFaceArgNames.V_HEAD_DIM.value)
        self.dtype = getattr(self.cfg, "dtype")

        assert self.N == self.K, "N and K must be equal for MLA"

        self.create_sharding()

    def generate_kernel(self, rngs: nnx.Rngs):
        """Initializes the weight kernels."""

        self.kernel_q_down_proj_DA = self.param_factory.create_kernel_param(
            rngs, (self.D, self.query_lora_rank), self.q_da_sharding,
            self.cfg.dtype)
        self.kernel_q_up_proj_ANH = self.param_factory.create_kernel_param(
            rngs,
            (self.query_lora_rank, self.N, self.qk_head_dim),
            self.anh_sharding,
            self.cfg.dtype,
        )
        self.kernel_kv_down_proj_DA = self.param_factory.create_kernel_param(
            rngs,
            (self.D, self.kv_lora_rank + self.qk_rope_head_dim),
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
        self.kernel_o_proj_NHD = self.param_factory.create_kernel_param(
            rngs, (self.N, self.v_head_dim, self.D), self.nhd_sharding,
            self.cfg.dtype)
        self.q_rms_norm = nnx.RMSNorm(
            self.query_lora_rank,
            epsilon=self.rms_norm_eps,
            param_dtype=self.dtype,
            rngs=rngs,
        )
        self.kv_rms_norm = nnx.RMSNorm(
            self.kv_lora_rank,
            epsilon=self.rms_norm_eps,
            param_dtype=self.dtype,
            rngs=rngs,
        )
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

    def __call__(
        self,
        x,
        is_prefill,
        kv_cache: KVCache,
        attention_metadata: AttentionMetadata,
    ):
        """Performs the forward pass of the attention module.

        Args:
            x: The input tensor of shape `(batch_size, seq_len, d_model)`.
            op_mode: The operational mode, either 'prefill' or 'generate'.
            kv_cache: The key-value cache for storing past attention states.
            attention_metadata: Metadata for attention, such as input positions.

        Returns:
            A tuple containing:
                - The updated KV cache.
                - The attention output tensor of shape
                  `(batch_size, seq_len, d_model)`.
        """
        op_mode = "prefill" if is_prefill else "generate"
        md = attention_metadata
        x = jnp.asarray(x, self.cfg.dtype)
        x_SD = nnx.with_sharding_constraint(
            x, self.activation_attention_td[op_mode])
        x_q_TD = nnx.with_sharding_constraint(x, self.activation_q_td[op_mode])

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
            q_TNH = nnx.with_sharding_constraint(q_TNH,
                                                 self.query_tnh[op_mode])

        with jax.named_scope("kv_proj"):
            # KV down projection.
            kv_SA = jnp.einsum("SD,DA -> SA", x_SD,
                               self.kernel_kv_down_proj_DA.value)
            
            # Split the output into latent kv vector and k rope vector.
            k_rope_SH = kv_SA[..., self.kv_lora_rank:]
            # Reshape k_rope_SH to include head dimension for RoPE application
            k_rope_SNH = k_rope_SH[..., None, :]
            k_rope_SNH = self.rope.apply_rope(md.input_positions, k_rope_SNH)
            # k_rope_SNH = jnp.broadcast_to(
            #     k_rope_SNH,
            #     (k_rope_SNH.shape[0], self.N, self.qk_rope_head_dim))
            kv_SA = kv_SA[..., :self.kv_lora_rank]
            kv_SA = self.kv_rms_norm(kv_SA) 

            # what is cached is [kv_SA, k_rope_SH]

            ##### This should be computed inside the kernel #######

            # # KV up projection.
            # kv_nope = jnp.einsum("SA,ANH -> SNH", kv_SA,
            #                      self.kernel_kv_up_proj_ANH.value)
            # # Split the latent kv vector into k nope vector and v vector.
            # k_nope_SNH = kv_nope[..., :self.qk_nope_head_dim]
            # v_SNH = kv_nope[..., self.qk_nope_head_dim:]
            # # Concatenate the key vector.
            # k_SNH = jnp.concatenate([k_nope_SNH, k_rope_SNH], axis=-1)
            # k_SNH = nnx.with_sharding_constraint(k_SNH,
            #                                      self.keyvalue_skh[op_mode])
            # v_SNH = nnx.with_sharding_constraint(v_SNH,
            #                                      self.keyvalue_skh[op_mode])

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
            # k_SNH = jnp.pad(k_SNH, ((0, 0), (0, 0),
            #                         (0, multiple_of_128 - self.qk_head_dim)))
            # v_SNH = jnp.pad(v_SNH, ((0, 0), (0, 0),
            #                         (0, multiple_of_128 - self.v_head_dim)))
            new_kv_cache, outputs_TNH = self.attention(
                is_prefill,
                kv_cache,
                q_TNH,
                k_rope_SNH,
                kv_SA,
                # k_SNH,
                # v_SNH,
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
                o_TD, self.activation_attention_out_td[op_mode])
        return new_kv_cache, o_TD

    def create_sharding(self):
        """Creates sharding rules for activations and weights."""
        mode_dependent_attrs = [
            "activation_attention_td",
            "activation_q_td",
            "query_tnh",
            "keyvalue_skh",
            "activation_attention_out_td",
        ]
        for attr_name in mode_dependent_attrs:
            prefill_sharding_config = getattr(self.sharding_cfg.prefill_rules,
                                              attr_name)
            generate_sharding_config = getattr(
                self.sharding_cfg.generate_rules, attr_name)

            sharding_dict = {
                "prefill": NamedSharding(self.mesh,
                                         P(*prefill_sharding_config)),
                "generate": NamedSharding(self.mesh,
                                          P(*generate_sharding_config)),
            }
            setattr(self, attr_name, sharding_dict)

        # static sharding for kernel/weights
        self.q_da_sharding = NamedSharding(
            self.mesh,
            P(*self.sharding_cfg.generate_rules.attn_mla_qa_weight_da))
        self.anh_sharding = NamedSharding(
            self.mesh,
            P(*self.sharding_cfg.generate_rules.attn_mla_qb_weight_anh))
        self.kv_da_sharding = NamedSharding(
            self.mesh,
            P(*self.sharding_cfg.generate_rules.attn_mla_kva_weight_da))
        self.anh_sharding = NamedSharding(
            self.mesh,
            P(*self.sharding_cfg.generate_rules.attn_mla_kvb_weight_anh))
        self.nhd_sharding = NamedSharding(
            self.mesh, P(*self.sharding_cfg.generate_rules.attn_o_weight_nhd))

        # TODO: the pallas kernels of flash_attention/paged_attention need to be called
        # via shard_map with sharding specs, However, the q/k/v have been sharded outside of attention()
        # So we replicate the sharding below but it should be better organized if we use pallas kernels
        self.pallas_q_spec = {
            "prefill": P(*self.sharding_cfg.prefill_rules.query_tnh),
            "generate": P(*self.sharding_cfg.generate_rules.query_tnh),
        }
        self.pallas_kv_spec = {
            "prefill": P(*self.sharding_cfg.prefill_rules.keyvalue_skh),
            "generate": P(*self.sharding_cfg.generate_rules.keyvalue_skh),
        }
        self.pallas_cache_page_spec = {
            "prefill": P(*self.sharding_cfg.prefill_rules.keyvalue_cache_lskh),
            "generate":
            P(*self.sharding_cfg.generate_rules.keyvalue_cache_lskh),
        }
    
    def attention(
        self,
        is_prefill: bool,
        kv_cache: KVCache,
        q_TNH: jax.Array,
        k_rope_SNH: jax.Array,
        k_SA: jax.Array,
        # k_SKH: jax.Array,
        # v_SKH: jax.Array,
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
        # remove head dimension from kv_cache
        kv_cache = kv_cache.reshape(kv_cache.shape[0], kv_cache.shape[1], 2,  -1)
        # k_rope = k_rope_SNH.reshape(k_rope_SNH.shape[0], -1)

        print("1. kv_cache.shape", kv_cache.shape)
        # print("k_rope.shape", k_rope.shape)
        kv_cache = update_mla_kv_cache(k_SA, k_rope_SNH, kv_cache, md.slot_mapping,
                                   md.num_slices, mesh)

        H = q_TNH.shape[-1]
        #TODO: we use generate_rules as the default sharding for ragged_paged_attention,
        # but it could be configurable based on the op_mode.
        in_specs = (
            P(*self.sharding_cfg.generate_rules.query_tnh),  # q_TNH
            P(*self.sharding_cfg.generate_rules.query_tnh),  # k_rope_SNH
            P(*self.sharding_cfg.generate_rules.keyvalue_cache_lskh),  # kv_cache
            P(),  # kv_up_proj_weights
            P(),  # md.seq_lens: Replicated
            P(),  # md.block_tables: Replicated
            P(),  # md.query_start_loc: Replicated
            P(),  # md.num_seqs: Replicated
        )
        out_specs = P(*self.sharding_cfg.generate_rules.attn_o_tnh
                      )  # output_TNH: Shard the 'model' dimension

        def _ragged_mla_paged_attention(*args):
            return ragged_mla_paged_attention(
                *args,
                sm_scale=H**-0.5,
                sliding_window=None,
                soft_cap=None,
                mask_value=None,
                # NOTE(xiang): v6e chip has 128M VMEM capacity,
                # set this to 64M to avoid VMEM OOM,
                # otherwise the default value is 16M.
                vmem_limit_bytes=64 * 1024 * 1024,
                qk_nope_head_dim=self.qk_nope_head_dim,
                qk_rope_head_dim=self.qk_rope_head_dim,
                v_head_dim=self.v_head_dim,
                num_heads=self.N,
            )
        print("in_specs", in_specs)
        print("mesh", mesh)
        # pad kv_cache


        output_TNH = jax.jit(
            shard_map.shard_map(
                _ragged_mla_paged_attention,
                mesh=mesh,
                in_specs=in_specs,
                out_specs=out_specs,
                check_rep=False,
            ))(
                q_TNH,
                k_rope_SNH,
                kv_cache,
                self.kernel_kv_up_proj_ANH.value.reshape(self.kv_lora_rank, -1),
                md.seq_lens,
                md.block_tables,
                md.query_start_loc,
                md.num_seqs,
            )


        return kv_cache, output_TNH
