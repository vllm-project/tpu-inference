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
     (HuggingFaceArgNames.ROPE_THETA.value, float),
     (HuggingFaceArgNames.ROPE_SCALING.value, Dict[str, Any]),
     (
         "dtype",
         DTypeLike,
     ), ("vllm_config", VllmConfig, field(repr=False, default=None))],
    bases=(Config, ))

AttentionConfig.__doc__ = f"""Configuration for the Attention module.
         Attributes:
        {HuggingFaceArgNames.HIDDEN_SIZE.value}: The dimension of the model.
        {HuggingFaceArgNames.NUM_ATTENTION_HEADS.value}: The number of query heads.
        {HuggingFaceArgNames.NUM_KEY_VALUE_HEADS.value}: The number of key/value heads.
        {HuggingFaceArgNames.HEAD_DIM.value}: The dimension of each attention head.
        {HuggingFaceArgNames.ROPE_THETA.value}: The base period for Rotary Position Embeddings.
        {HuggingFaceArgNames.ROPE_SCALING.value}: Optional dictionary of scaling factors for RoPE.
         dtype: The data type for computations (default: jnp.float32).
         vllm_config: The VLLM config containing any overrides to apply.
    """

MLAConfig = make_dataclass(
    "MLAConfig",
    [(HuggingFaceArgNames.HIDDEN_SIZE.value, int),
     (HuggingFaceArgNames.NUM_ATTENTION_HEADS.value, int),
     (HuggingFaceArgNames.NUM_KEY_VALUE_HEADS.value, int),
     (HuggingFaceArgNames.ROPE_THETA.value, float),
     (HuggingFaceArgNames.ROPE_SCALING.value, Dict[str, Any]),
     (HuggingFaceArgNames.Q_LORA_RANK.value, int),
     (HuggingFaceArgNames.KV_LORA_RANK.value, int),
     (HuggingFaceArgNames.QK_NOPE_HEAD_DIM.value, int),
     (HuggingFaceArgNames.QK_ROPE_HEAD_DIM.value, int),
     (HuggingFaceArgNames.V_HEAD_DIM.value, int),
     (HuggingFaceArgNames.RMS_NORM_EPS.value, float), (
         "dtype",
         DTypeLike,
     ), ("vllm_config", VllmConfig, field(repr=False, default=None))],
    bases=(Config, ),
)

MLAConfig.__doc__ = f"""Configuration for the MLA module.
         Attributes:
        {HuggingFaceArgNames.Q_LORA_RANK.value}: The dimension for the latent query vector.
        {HuggingFaceArgNames.KV_LORA_RANK.value}: The dimension for the latent key/value vector.
        {HuggingFaceArgNames.QK_NOPE_HEAD_DIM.value}: The dimension of the no rope portion of the qv tensor.
        {HuggingFaceArgNames.QK_ROPE_HEAD_DIM.value}: The dimension of the rope portion of the qv tensor.
        {HuggingFaceArgNames.V_HEAD_DIM.value}: The dimension of the value vector.
        {HuggingFaceArgNames.RMS_NORM_EPS.value}: The epsilon for the RMS normalization.
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
        #self._generate_kernel()

    def generate_kernel(self, rngs: nnx.Rngs):
        """Initializes the weight kernels for Q, K, V, and O projections."""
        N = getattr(self.cfg, HuggingFaceArgNames.NUM_ATTENTION_HEADS.value)
        K = getattr(self.cfg, HuggingFaceArgNames.NUM_KEY_VALUE_HEADS.value)
        D = getattr(self.cfg, HuggingFaceArgNames.HIDDEN_SIZE.value)
        H = getattr(self.cfg, HuggingFaceArgNames.HEAD_DIM.value)

        self.kernel_q_proj_NDH = self.param_factory.create_kernel_param(
            rngs, (N, D, H), self.ndh_sharding, self.cfg.dtype)
        self.kernel_k_proj_KDH = self.param_factory.create_kernel_param(
            rngs, (K, D, H), self.kdh_sharding, self.cfg.dtype)
        self.kernel_v_proj_KDH = self.param_factory.create_kernel_param(
            rngs, (K, D, H), self.kdh_sharding, self.cfg.dtype)
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
        self.ndh_sharding = NamedSharding(
            self.mesh, P(*self.sharding_cfg.generate_rules.attn_q_weight_ndh))
        self.kdh_sharding = NamedSharding(
            self.mesh, P(*self.sharding_cfg.generate_rules.attn_k_weight_kdh))
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

    def __call__(
        self,
        x,
        is_prefill,
        kv_cache: KVCache,
        attention_metadata: AttentionMetadata,
    ):
        """Performs the forward pass of the attention module.

        This method computes the attention output by projecting the input `x`
        to queries, keys, and values, applying RoPE, performing scaled
        dot-product attention, and projecting the result back to the model
        dimension. It updates and utilizes a KV cache.

        Args:
            x: The input tensor of shape `(seq_len, d_model)`.
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
        rope_scaling = getattr(self.cfg,
                               HuggingFaceArgNames.ROPE_SCALING.value)
        rope_theta = getattr(self.cfg, HuggingFaceArgNames.ROPE_THETA.value)
        H = getattr(self.cfg, HuggingFaceArgNames.HEAD_DIM.value)
        with jax.named_scope("q_proj"):
            q_TNH = jnp.einsum('TD,NDH -> TNH', x_q_TD,
                               self.kernel_q_proj_NDH.value)
            q_TNH = apply_rope(q_TNH, md.input_positions, H, rope_theta,
                               rope_scaling)
            q_TNH = nnx.with_sharding_constraint(q_TNH,
                                                 self.query_tnh[op_mode])
        with jax.named_scope("k_proj"):
            k_SKH = jnp.einsum('SD,KDH -> SKH', x_SD,
                               self.kernel_k_proj_KDH.value)
            k_SKH = apply_rope(k_SKH, md.input_positions, H, rope_theta,
                               rope_scaling)
            k_SKH = nnx.with_sharding_constraint(k_SKH,
                                                 self.keyvalue_skh[op_mode])

        with jax.named_scope("v_proj"):
            v_SKH = jnp.einsum('SD,KDH -> SKH', x_SD,
                               self.kernel_v_proj_KDH.value)
            v_SKH = nnx.with_sharding_constraint(v_SKH,
                                                 self.keyvalue_skh[op_mode])

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
            o_TD = nnx.with_sharding_constraint(
                o_TD, self.activation_attention_out_td[op_mode])
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
                sliding_window=None,
                soft_cap=None,
                mask_value=None,
                # NOTE(xiang): v6e chip has 128M VMEM capacity,
                # set this to 64M to avoid VMEM OOM,
                # otherwise the default value is 16M.
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


# TODO (wenxindongwork): Add MLA KV cache implementation. For now, cache complete KV vectors.
@dataclass
class MLA(Attention):
    """An implementation of Multi-Head Latent Attention.

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
            rngs, (self.D, self.N, self.qk_head_dim), self.anh_sharding,
            self.cfg.dtype)

        self.kernel_kv_down_proj_DA = self.param_factory.create_kernel_param(
            rngs,
            (self.D, self.kv_lora_rank + self.qk_rope_head_dim),
            self.kv_da_sharding,
            self.dtype,
        )
        self.kernel_kv_up_proj_ANK = self.param_factory.create_kernel_param(
            rngs,
            (self.kv_lora_rank, self.N,
             self.qk_nope_head_dim + self.v_head_dim),
            self.ank_sharding,
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
        x_BSD = nnx.with_sharding_constraint(
            x, self.activation_attention_btd[op_mode])
        x_q_BTD = nnx.with_sharding_constraint(x,
                                               self.activation_q_btd[op_mode])

        with jax.named_scope("q_proj"):
            # Query down projection. [B, T, query_lora_rank]
            q_BTA = jnp.einsum("BTD,DA -> BTA", x_q_BTD,
                               self.kernel_q_down_proj_DA.value)
            q_BTA = self.q_rms_norm(q_BTA)

            # Query up projection. [B, T, N, H]
            q_BTNH = jnp.einsum("BTA,ANH -> BTNH", q_BTA,
                                self.kernel_q_up_proj_ANH.value)
            # Split the query into nope and rope.
            q_nope_BTNH = q_BTNH[..., :self.qk_nope_head_dim]
            q_rope_BTNH = self.apply_rope(q_BTNH[..., self.qk_nope_head_dim:],
                                          md.input_positions,
                                          self.qk_rope_head_dim,
                                          self.rope_theta, self.rope_scaling)

            # Concatenate the nope and rope queries.
            q_BTNH = jnp.concatenate([q_nope_BTNH, q_rope_BTNH], axis=-1)
            # Multiple the query by scaling factor
            q_BTNH = q_BTNH * self.qk_head_dim**-0.5

            q_BTNH = nnx.with_sharding_constraint(q_BTNH,
                                                  self.query_btnh[op_mode])

        with jax.named_scope("kv_proj"):

            # KV down projection. [B, T, kv_lora_rank + qk_rope_head_dim]
            kv_BSA = jnp.einsum("BSD,DA -> BSA", x_BSD,
                                self.kernel_kv_down_proj_DA.value)

            # Split the key and value into latent kv vector and k rope vector.
            k_rope_BSH = kv_BSA[..., self.kv_lora_rank:]
            k_rope_BSH = self.apply_rope(k_rope_BSH, md.input_positions,
                                         self.qk_rope_head_dim,
                                         self.rope_theta, self.rope_scaling)
            k_rope_BSNH = jnp.repeat(k_rope_BSH, self.N, axis=2)

            kv_BSA = kv_BSA[..., :self.kv_lora_rank]
            kv_BSA = self.kv_rms_norm(kv_BSA)

            # KV up projection. [B, T, N, qk_nope_head_dim + v_head_dim]
            kv_nope = jnp.einsum("BSA,ANK -> BSA", kv_BSA,
                                 self.kernel_kv_up_proj_ANK.value)

            # Split the latent kv vector into k nope vector and v vector.
            k_nope_BSNH = kv_nope[..., :self.qk_nope_head_dim]
            v_BSNH = kv_nope[..., self.qk_nope_head_dim:]

            # Concatenate the key vector.
            k_BSNH = jnp.concatenate([k_nope_BSNH, k_rope_BSNH], axis=-1)

            k_BSKH = nnx.with_sharding_constraint(k_BSNH,
                                                  self.keyvalue_bskh[op_mode])
            v_BSNH = nnx.with_sharding_constraint(v_BSNH,
                                                  self.keyvalue_bskh[op_mode])

        with jax.named_scope("attn_op"):
            new_kv_cache, outputs_BTNH = self.attention(
                is_prefill,
                kv_cache,
                q_BTNH,
                k_BSKH,
                v_BSNH,
                attention_metadata,
                self.mesh,
                self.N,
                self.K,
            )

        with jax.named_scope("o_proj"):
            o_BTD = jnp.einsum("BTNH,NHD -> BTD", outputs_BTNH,
                               self.kernel_o_proj_NHD.value)
            o_BTD = nnx.with_sharding_constraint(
                o_BTD, self.activation_attention_out_btd[op_mode])
        return new_kv_cache, o_BTD

    def create_sharding(self):
        """Creates sharding rules for activations and weights."""
        mode_dependent_attrs = [
            "activation_attention_btd",
            "activation_q_btd",
            "query_btnh",
            "keyvalue_bskh",
            "activation_attention_out_btd",
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
        self.ank_sharding = NamedSharding(
            self.mesh,
            P(*self.sharding_cfg.generate_rules.attn_mla_qb_weight_ank))

        self.nhd_sharding = NamedSharding(
            self.mesh, P(*self.sharding_cfg.generate_rules.attn_o_weight_nhd))

        # TODO: the pallas kernels of flash_attention/paged_attention need to be called
        # via shard_map with sharding specs, However, the q/k/v have been sharded outside of attention()
        # So we replicate the sharding below but it should be better organized if we use pallas kernels
        self.pallas_q_spec = {
            "prefill": P(*self.sharding_cfg.prefill_rules.query_btnh),
            "generate": P(*self.sharding_cfg.generate_rules.query_btnh),
        }
        self.pallas_kv_spec = {
            "prefill": P(*self.sharding_cfg.prefill_rules.keyvalue_bskh),
            "generate": P(*self.sharding_cfg.generate_rules.keyvalue_bskh),
        }
        self.pallas_cache_page_spec = {
            "prefill": P(*self.sharding_cfg.prefill_rules.keyvalue_cache_kbsh),
            "generate":
            P(*self.sharding_cfg.generate_rules.keyvalue_cache_kbsh),
        }
