from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Union

import jax
import jax.numpy as jnp
from flax import nnx
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from tpu_commons.models.jax.common.base import Config, ParamFactory
from tpu_commons.models.jax.common.kv_cache import KVCache
from tpu_commons.models.jax.common.sharding import ShardingConfig
from tpu_commons.models.jax.rope.generic_rope import apply_rope


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
class AttentionConfig(Config):
    """Configuration for the Attention module.

    Attributes:
        d_model: The dimension of the model.
        num_q_heads: The number of query heads.
        num_kv_heads: The number of key/value heads.
        head_dim: The dimension of each attention head.
        rope_theta: The base period for Rotary Position Embeddings.
        rope_scaling: Optional dictionary of scaling factors for RoPE.
        dtype: The data type for computations (default: jnp.float32).
    """
    d_model: int
    num_q_heads: int
    num_kv_heads: int
    head_dim: int
    rope_theta: float = 10000.0
    rope_scaling: Dict[str, Any] = None
    dtype: Any = jnp.float32


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
        self._generate_kernel()

    def _generate_kernel(self):
        """Initializes the weight kernels for Q, K, V, and O projections."""
        N = self.cfg.num_q_heads
        K = self.cfg.num_kv_heads
        D = self.cfg.d_model
        H = self.cfg.head_dim

        self.kernel_q_proj_NDH = self.param_factory.create_kernel_init(
            (N, D, H), self.ndh_sharding, self.cfg.dtype)
        self.kernel_k_proj_KDH = self.param_factory.create_kernel_init(
            (K, D, H), self.kdh_sharding, self.cfg.dtype)
        self.kernel_v_proj_KDH = self.param_factory.create_kernel_init(
            (K, D, H), self.kdh_sharding, self.cfg.dtype)
        self.kernel_o_proj_NHD = self.param_factory.create_kernel_init(
            (N, H, D), self.nhd_sharding, self.cfg.dtype)

    def create_sharding(self):
        """Creates sharding rules for activations and weights."""
        mode_dependent_attrs = [
            "activation_attention_btd", "activation_q_btd", "query_btnh",
            "keyvalue_bskh", "activation_attention_out_btd"
        ]
        for attr_name in mode_dependent_attrs:
            prefill_sharding_config = getattr(
                self.sharding_cfg.prefill_sharding_cfg, attr_name)
            generate_sharding_config = getattr(
                self.sharding_cfg.generate_sharding_cfg, attr_name)

            sharding_dict = {
                'prefill': NamedSharding(self.mesh,
                                         P(prefill_sharding_config)),
                'generate': NamedSharding(self.mesh,
                                          P(generate_sharding_config))
            }
            setattr(self, attr_name, sharding_dict)

        # static sharding for kernel/weights
        self.ndh_sharding = NamedSharding(
            self.mesh,
            P(self.sharding_cfg.generate_sharding_cfg.attn_q_weight_ndh))
        self.kdh_sharding = NamedSharding(
            self.mesh,
            P(self.sharding_cfg.generate_sharding_cfg.attn_k_weight_kdh))
        self.nhd_sharding = NamedSharding(
            self.mesh,
            P(self.sharding_cfg.generate_sharding_cfg.attn_o_weight_nhd))

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

        with jax.named_scope("q_proj"):
            q_BTNH = jnp.einsum('BTD,NDH -> BTNH', x_q_BTD,
                                self.kernel_q_proj_NDH.value)
            q_BTNH = apply_rope(q_BTNH, md.input_positions, self.cfg.head_dim,
                                self.cfg.rope_theta, self.cfg.rope_scaling)
            q_BTNH = nnx.with_sharding_constraint(q_BTNH,
                                                  self.query_btnh[op_mode])
        with jax.named_scope("k_proj"):
            k_BSKH = jnp.einsum('BSD,KDH -> BSKH', x_BSD,
                                self.kernel_k_proj_KDH.value)
            k_BSKH = apply_rope(k_BSKH, md.input_positions, self.cfg.head_dim,
                                self.cfg.rope_theta, self.cfg.rope_scaling)
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
                self.cfg.num_q_heads,
                self.cfg.num_kv_heads,
            )

        with jax.named_scope("o_proj"):
            o_BTD = jnp.einsum('BTNH,NHD -> BTD', outputs_BTNH,
                               self.kernel_o_proj_NHD.value)
            o_BTD = nnx.with_sharding_constraint(
                o_BTD, self.activation_attention_out_btd[op_mode])
        return new_kv_cache, o_BTD

    def get_cfg(self) -> AttentionConfig:
        return self.cfg

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
        num_kv_heads: int,
    ) -> Tuple[KVCache, jax.Array]:
        """Performs scaled dot-product attention and updates the KV cache.

        This function handles the core attention logic, which varies between
        prefill and generation modes. In prefill, it computes self-attention
        over the input sequence with a causal mask. In generation, it attends
        to the full history of keys and values stored in the cache.

        Args:
            is_prefill: A boolean indicating if the mode is 'prefill'.
            kv_cache: The key-value cache to be updated and used.
            q_BTNH: Query tensor of shape `(batch, query_seq, num_q_heads, head_dim)`.
            k_BSKH: Key tensor of shape `(batch, kv_seq, num_kv_heads, head_dim)`.
            v_BSKH: Value tensor of shape `(batch, kv_seq, num_kv_heads, head_dim)`.
            attention_metadata: Metadata containing sequence lengths.
            mesh: The JAX device mesh (unused in this specific function but
                kept for potential future use or API consistency).
            num_heads: The number of query heads.
            num_kv_heads: The number of key/value heads.

        Returns:
            A tuple containing:
                - The updated KV cache.
                - The attention output tensor of shape
                  `(batch, seq, num_q_heads, head_dim)`.
        """
        op_mode = 'prefill' if is_prefill else 'generate'
        head_repeats = num_heads // num_kv_heads

        if is_prefill:
            current_lengths_for_update = jnp.zeros_like(
                attention_metadata.seq_lens)
        else:
            current_lengths_for_update = attention_metadata.seq_lens

        # Update the cache with the new keys and values.
        with jax.named_scope("kv_cache_update"):
            kv_cache.update(k_BSKH,
                            v_BSKH,
                            current_lengths_for_update,
                            op_mode=op_mode)

        if is_prefill:
            # In prefill, attention is calculated on the just-added K/V.
            k_attn_BSNH = jnp.repeat(k_BSKH, head_repeats,
                                     axis=2) if head_repeats > 1 else k_BSKH
            v_attn_BSNH = jnp.repeat(v_BSKH, head_repeats,
                                     axis=2) if head_repeats > 1 else v_BSKH

            scores_BNTS = jnp.einsum('BTNH,BSNH->BNTS', q_BTNH,
                                     k_attn_BSNH) / jnp.sqrt(self.cfg.head_dim)

            T = q_BTNH.shape[1]
            S = k_BSKH.shape[1]
            causal_mask_TS = jnp.tril(jnp.ones((1, 1, T, S), dtype=jnp.bool_))
            scores_BNTS = jnp.where(causal_mask_TS, scores_BNTS, -jnp.inf)

            attention_weights_BNTS = jax.nn.softmax(scores_BNTS, axis=-1)

            output_BTNH = jnp.einsum('BNTS,BSNH->BTNH', attention_weights_BNTS,
                                     v_attn_BSNH)
        else:
            # In generate, attention is calculated over the entire cached history.
            with jax.named_scope("kv_cache_retrieve"):
                k_cache_full = kv_cache.key_cache[op_mode].value
                v_cache_full = kv_cache.value_cache[op_mode].value

            if head_repeats > 1:
                #[B, S, N, H]
                k_cache_full = jnp.repeat(k_cache_full, head_repeats, axis=2)
                v_cache_full = jnp.repeat(v_cache_full, head_repeats, axis=2)

            scores_BNTS = jnp.einsum('BTNH,BSNH->BNTS', q_BTNH,
                                     k_cache_full) / jnp.sqrt(
                                         self.cfg.head_dim)

            current_lengths_B = attention_metadata.seq_lens + 1
            max_cache_len = k_cache_full.shape[1]
            mask = jnp.arange(max_cache_len) < current_lengths_B[:, None]
            scores_BNTS = jnp.where(mask[:, None, None, :], scores_BNTS,
                                    -jnp.inf)

            attention_weights_BNTS = jax.nn.softmax(scores_BNTS, axis=-1)
            output_BTNH = jnp.einsum('BNTS,BSNH->BTNH', attention_weights_BNTS,
                                     v_cache_full)

        return kv_cache, output_BTNH
