from dataclasses import dataclass, field, make_dataclass
from typing import Any, Dict, List, Tuple, Union

import jax
import jax.numpy as jnp
from flax import nnx
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from jax.typing import DTypeLike

from jax.experimental.pallas.ops.tpu.flash_attention import flash_attention
from jax.experimental.pallas.ops.tpu.paged_attention import paged_attention

from vllm.config import VllmConfig
from tpu_commons.models.jax.common.base import Config, ParamFactory
from tpu_commons.models.jax.common.sharding import ShardingConfig
from tpu_commons.models.jax.common.constants import HuggingFaceArgNames
from tpu_commons.models.jax.layers.rope import apply_rope_scaling
from tpu_commons.models.jax.layers.attention import update_cache
from tpu_commons.utils_jax import get_megacore

KVCache = Tuple[jax.Array, jax.Array]

@dataclass
class AttentionMetadata(object):
    input_positions: jax.Array
    # If mix attention, this is a list of len 2
    seq_lens: Union[jax.Array, List[jax.Array]]
    # If mix attention, this is a list of len 2
    block_indices: Union[jax.Array, List[jax.Array]]
    # If mix attention, this is a list of len 2
    kv_cache_write_indices: Union[jax.Array, List[jax.Array]]

    # The following fields are set only when chunked prefill is enabled
    chunked_prefill_enabled: bool = False
    decode_lengths: jax.Array = None  # [max_num_decode_seqs]
    decode_page_indices: jax.Array = None  # [max_num_decode_seqs, pages_per_sequence]
    num_decode_seqs: jax.Array = None  # [1]
    prefill_lengths: jax.Array = None  # [max_num_prefill_seqs]
    prefill_page_indices: jax.Array = None  # [max_num_prefill_seqs, pages_per_sequence]
    prefill_query_start_offsets: jax.Array = None  # [max_num_prefill_seqs + 1]
    num_prefill_seqs: jax.Array = None  # [1]


@dataclass
class AttentionMetadata(object):
    input_positions: jax.Array
    # If mix attention, this is a list of len 2
    seq_lens: Union[jax.Array, List[jax.Array]]
    # If mix attention, this is a list of len 2
    block_indices: Union[jax.Array, List[jax.Array]]
    # If mix attention, this is a list of len 2
    kv_cache_write_indices: Union[jax.Array, List[jax.Array]]

    # The following fields are set only when chunked prefill is enabled
    chunked_prefill_enabled: bool = False
    decode_lengths: jax.Array = None  # [max_num_decode_seqs]
    decode_page_indices: jax.Array = None  # [max_num_decode_seqs, pages_per_sequence]
    num_decode_seqs: jax.Array = None  # [1]
    prefill_lengths: jax.Array = None  # [max_num_prefill_seqs]
    prefill_page_indices: jax.Array = None  # [max_num_prefill_seqs, pages_per_sequence]
    prefill_query_start_offsets: jax.Array = None  # [max_num_prefill_seqs + 1]
    num_prefill_seqs: jax.Array = None  # [1]


@dataclass
class AttentionMetadata(object):
    input_positions: jax.Array
    # If mix attention, this is a list of len 2
    seq_lens: Union[jax.Array, List[jax.Array]]
    # If mix attention, this is a list of len 2
    block_indices: Union[jax.Array, List[jax.Array]]
    # If mix attention, this is a list of len 2
    kv_cache_write_indices: Union[jax.Array, List[jax.Array]]

    # The following fields are set only when chunked prefill is enabled
    chunked_prefill_enabled: bool = False
    decode_lengths: jax.Array = None  # [max_num_decode_seqs]
    decode_page_indices: jax.Array = None  # [max_num_decode_seqs, pages_per_sequence]
    num_decode_seqs: jax.Array = None  # [1]
    prefill_lengths: jax.Array = None  # [max_num_prefill_seqs]
    prefill_page_indices: jax.Array = None  # [max_num_prefill_seqs, pages_per_sequence]
    prefill_query_start_offsets: jax.Array = None  # [max_num_prefill_seqs + 1]
    num_prefill_seqs: jax.Array = None  # [1]

from tpu_commons.models.jax.common.base import Config, ParamFactory
from tpu_commons.models.jax.common.sharding import ShardingConfig
from tpu_commons.models.jax.layers.rope import apply_rope_scaling
from tpu_commons.models.jax.layers.attention import update_cache
from tpu_commons.utils_jax import get_megacore

KVCache = Tuple[jax.Array, jax.Array]

@dataclass
class AttentionMetadata(object):
    input_positions: jax.Array
    # If mix attention, this is a list of len 2
    seq_lens: Union[jax.Array, List[jax.Array]]
    # If mix attention, this is a list of len 2
    block_indices: Union[jax.Array, List[jax.Array]]
    # If mix attention, this is a list of len 2
    kv_cache_write_indices: Union[jax.Array, List[jax.Array]]

    # The following fields are set only when chunked prefill is enabled
    chunked_prefill_enabled: bool = False
    decode_lengths: jax.Array = None  # [max_num_decode_seqs]
    decode_page_indices: jax.Array = None  # [max_num_decode_seqs, pages_per_sequence]
    num_decode_seqs: jax.Array = None  # [1]
    prefill_lengths: jax.Array = None  # [max_num_prefill_seqs]
    prefill_page_indices: jax.Array = None  # [max_num_prefill_seqs, pages_per_sequence]
    prefill_query_start_offsets: jax.Array = None  # [max_num_prefill_seqs + 1]
    num_prefill_seqs: jax.Array = None  # [1]


AttentionConfig = make_dataclass("AttentionConfig", [
    (HuggingFaceArgNames.HIDDEN_SIZE.value, int),
    (HuggingFaceArgNames.NUM_ATTENTION_HEADS.value, int),
    (HuggingFaceArgNames.NUM_KEY_VALUE_HEADS.value, int),
    (HuggingFaceArgNames.HEAD_DIM.value, int),
    (HuggingFaceArgNames.ROPE_THETA.value, float),
    (HuggingFaceArgNames.ROPE_SCALING.value, Dict[str, Any]),
    ("dtype", DTypeLike),
    ("vllm_config", VllmConfig, field(repr=False, default=None))
    ],
    bases=(Config,)
)
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
            "activation_attention_btd", "activation_q_btd", "query_btnh",
            "keyvalue_bskh", "activation_attention_out_btd"
        ]
        for attr_name in mode_dependent_attrs:
            prefill_sharding_config = getattr(
                self.sharding_cfg.prefill_rules, attr_name)
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
            self.mesh,
            P(*self.sharding_cfg.generate_rules.attn_q_weight_ndh))
        self.kdh_sharding = NamedSharding(
            self.mesh,
            P(*self.sharding_cfg.generate_rules.attn_k_weight_kdh))
        self.nhd_sharding = NamedSharding(
            self.mesh,
            P(*self.sharding_cfg.generate_rules.attn_o_weight_nhd))
        
        # TODO: the pallas kernels of flash_attention/paged_attention need to be called 
        # via shard_map with sharding specs, However, the q/k/v have been sharded outside of attention()
        # So we replicate the sharding below but it should be better organized if we use pallas kernels
        self.pallas_q_spec = {
            'prefill': P(*self.sharding_cfg.prefill_rules.query_btnh),
            'generate': P(*self.sharding_cfg.generate_rules.query_btnh)
        }
        self.pallas_kv_spec = {
            'prefill': P(*self.sharding_cfg.prefill_rules.keyvalue_bskh),
            'generate': P(*self.sharding_cfg.generate_rules.keyvalue_bskh)
        }
        self.pallas_cache_page_spec = {
            'prefill': P(*self.sharding_cfg.prefill_rules.keyvalue_cache_kbsh),
            'generate': P(*self.sharding_cfg.generate_rules.keyvalue_cache_kbsh)
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
        N = getattr(self.cfg, HuggingFaceArgNames.NUM_ATTENTION_HEADS.value)
        K = getattr(self.cfg, HuggingFaceArgNames.NUM_KEY_VALUE_HEADS.value)
        rope_scaling = getattr(self.cfg, HuggingFaceArgNames.ROPE_SCALING.value)
        rope_theta = getattr(self.cfg, HuggingFaceArgNames.ROPE_THETA.value)
        H = getattr(self.cfg, HuggingFaceArgNames.HEAD_DIM.value)
        with jax.named_scope("q_proj"):
            q_BTNH = jnp.einsum('BTD,NDH -> BTNH', x_q_BTD,
                                self.kernel_q_proj_NDH.value)
            q_BTNH = self.apply_rope(q_BTNH, md.input_positions, H,
                                rope_theta, rope_scaling)
            q_BTNH = nnx.with_sharding_constraint(q_BTNH,
                                                  self.query_btnh[op_mode])
        with jax.named_scope("k_proj"):
            k_BSKH = jnp.einsum('BSD,KDH -> BSKH', x_BSD,
                                self.kernel_k_proj_KDH.value)
            k_BSKH = self.apply_rope(k_BSKH, md.input_positions, H,
                                rope_theta, rope_scaling)
            k_BSKH = nnx.with_sharding_constraint(k_BSKH,
                                                  self.keyvalue_bskh[op_mode])

        with jax.named_scope("v_proj"):
            v_BSKH = jnp.einsum('BSD,KDH -> BSKH', x_BSD,
                                self.kernel_v_proj_KDH.value)
            v_BSKH = nnx.with_sharding_constraint(v_BSKH,
                                                  self.keyvalue_bskh[op_mode])

        with jax.named_scope("attn_op"):
            new_kv_cache, outputs_BTNH = self.attention(
                is_prefill,
                kv_cache,
                q_BTNH,
                k_BSKH,
                v_BSKH,
                attention_metadata,
                self.mesh,
                N,
                K,
            )

        with jax.named_scope("o_proj"):
            o_BTD = jnp.einsum('BTNH,NHD -> BTD', outputs_BTNH,
                               self.kernel_o_proj_NHD.value)
            o_BTD = nnx.with_sharding_constraint(
                o_BTD, self.activation_attention_out_btd[op_mode])
        return new_kv_cache, o_BTD

    def get_cfg(self) -> AttentionConfig:
        return self.cfg

    # TODO: As there's a shape mismatch when using the rope lib,
    # for function verification purpose, we add a local apply_rope function, 
    # which is mainly inherited from the tpu_commons.models.jax.layers.rope,
    def apply_rope(
        self,
        inputs: jax.Array,
        positions: jax.Array,
        head_dim: int,
        rope_theta: float = 10000,
        rope_scaling: Dict[str, Any] = None,
    ) -> jax.Array:
        fraction = 2 * jnp.arange(0, head_dim // 2) / head_dim
        timescale = rope_theta**fraction
        timescale = 1.0 / timescale

        if rope_scaling:
            timescale = apply_rope_scaling(timescale, rope_scaling)

        sinusoid_inp = positions[..., jnp.newaxis] * timescale[jnp.newaxis,
                                                            jnp.newaxis, :]
        sinusoid_inp = sinusoid_inp[:, :, jnp.newaxis, :]
        sin = jnp.sin(sinusoid_inp)
        cos = jnp.cos(sinusoid_inp)

        # Some models pad the inputs head_dim with zeros,
        # so we need to split the inputs using the head_dim before padding.
        padded_head_dim = inputs.shape[-1]
        first_half = inputs[..., :head_dim // 2]
        second_half = inputs[..., head_dim // 2:head_dim]
        first_part = first_half * cos - second_half * sin
        second_part = second_half * cos + first_half * sin
        out = jnp.concatenate([first_part, second_part], axis=-1)
        if padded_head_dim > head_dim:
            out = jnp.pad(out, ((0, 0), (0, 0), (0, 0),
                                (0, padded_head_dim - head_dim)))
        return out.astype(inputs.dtype)

    def attention(
        self,
        is_prefill: bool,
        kv_cache: KVCache,
        q_BTNH: jax.Array,
        k_BSKH: jax.Array,
        v_BSKH: jax.Array,
        attention_metadata: AttentionMetadata,
        mesh: Mesh,
        num_heads: int,
        num_key_value_heads: int,
    ) -> Tuple[KVCache, jax.Array]:
        """Performs scaled dot-product attention and updates the KV cache.

        This function handles the core attention logic, which varies between
        prefill and generation modes. In prefill, it computes self-attention
        over the input sequence with a causal mask. In generation, it attends
        to the full history of keys and values stored in the cache.

        Args:
            is_prefill: A boolean indicating if the mode is 'prefill'.
            kv_cache: The key-value cache to be updated and used.
            q_BTNH: Query tensor of shape `(batch, query_seq, num_attention_heads, head_dim)`.
            k_BSKH: Key tensor of shape `(batch, kv_seq, num_key_value_heads, head_dim)`.
            v_BSKH: Value tensor of shape `(batch, kv_seq, num_key_value_heads, head_dim)`.
            attention_metadata: Metadata containing sequence lengths.
            mesh: The JAX device mesh (unused in this specific function but
                kept for potential future use or API consistency).
            num_heads: The number of query heads.
            num_key_value_heads: The number of key/value heads.

        Returns:
            A tuple containing:
                - The updated KV cache.
                - The attention output tensor of shape
                  `(batch, seq, num_q_heads, head_dim)`.
        """
        def _attention_kernel(q_BTNH, k_BSKH, v_BSKH, key_cache, value_cache, seq_lens, block_indices):
            head_repeats = num_heads // num_key_value_heads

            if is_prefill:
                # Transpose K/V for attention calculation
                k_BKSH = k_BSKH.swapaxes(1, 2)
                v_BKSH = v_BSKH.swapaxes(1, 2)

                k_attn_BNSH = jnp.repeat(k_BKSH, head_repeats, axis=1) if head_repeats > 1 else k_BKSH
                v_attn_BNSH = jnp.repeat(v_BKSH, head_repeats, axis=1) if head_repeats > 1 else v_BKSH

                # Transpose Q for flash_attention
                q_BNTH = q_BTNH.swapaxes(1, 2)

                attn_output_BNTH = flash_attention(
                    q_BNTH,
                    k_attn_BNSH,
                    v_attn_BNSH,
                    causal=True,
                )
                attn_output_BTNH = attn_output_BNTH.swapaxes(1, 2)
            else:
                q_BNH = q_BTNH.squeeze(axis=1)
                attn_output_BNH = paged_attention(
                    q=q_BNH,
                    k_pages=key_cache,
                    v_pages=value_cache,
                    lengths=seq_lens,
                    page_indices=block_indices,
                    pages_per_compute_block=min(
                        16, block_indices.shape[1]),  # 512 / page_size:32,
                    megacore_mode="kv_head" if get_megacore() else None,
                )
                attn_output_BTNH = jnp.expand_dims(attn_output_BNH, 1)

            return attn_output_BTNH
        
        op_mode = "prefill" if is_prefill else "generate"
        key_cache, value_cache = kv_cache

        with jax.named_scope("kv_cache_update"):
            k_BKSH = k_BSKH.swapaxes(1, 2)
            v_BKSH = v_BSKH.swapaxes(1, 2)
            key_cache = update_cache(is_prefill, key_cache, attention_metadata.kv_cache_write_indices, k_BKSH)
            value_cache = update_cache(is_prefill, value_cache, attention_metadata.kv_cache_write_indices, v_BKSH)
             
        seq_lens_spec = P()
        block_indices_spec = P()

        in_specs = (
            self.pallas_q_spec[op_mode],
            self.pallas_kv_spec[op_mode],
            self.pallas_kv_spec[op_mode],
            self.pallas_cache_page_spec[op_mode],
            self.pallas_cache_page_spec[op_mode],
            seq_lens_spec,
            block_indices_spec
        )

        out_specs = self.pallas_q_spec[op_mode]

        outputs_BTNH = jax.experimental.shard_map.shard_map(
            _attention_kernel,
            mesh=self.mesh,
            in_specs=in_specs,
            out_specs=out_specs,
            check_rep=False
        )(q_BTNH, k_BSKH, v_BSKH, key_cache, value_cache, attention_metadata.seq_lens, attention_metadata.block_indices)
        
        return (key_cache, value_cache), outputs_BTNH