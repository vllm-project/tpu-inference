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

from dataclasses import InitVar, dataclass
from typing import Any, Tuple

import jax
import jax.numpy as jnp
from flax import nnx
from flax.typing import Sharding
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P

from tpu_inference import utils
from tpu_inference.kernels.ragged_paged_attention.v3.kernel import \
    ragged_paged_attention
from tpu_inference.layers.common.attention_metadata import AttentionMetadata
from tpu_inference.layers.common.quantization import quantize_kv
from tpu_inference.layers.common.sharding import ShardingAxisName
from tpu_inference.layers.jax.base import create_param
from tpu_inference.layers.jax.rope_interface import apply_rope

KVCache = Tuple[jax.Array, jax.Array]


@dataclass(kw_only=True)
class Attention(nnx.Module):
    """An implementation of attention.

    This module performs the attention mechanism for a transformer model,
    including query, key, and value projections, application of Rotary
    Position Embeddings (RoPE), and management of a KV cache for efficient
    autoregressive generation. It supports both prefill and generation
    (decode) modes and handles tensor sharding for distributed computation.

    Attributes:
        mesh: The JAX device mesh for distributed computation.
    """
    hidden_size: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    rope_theta: float
    rope_scaling: dict[str, Any]
    dtype: jnp.dtype
    mesh: Mesh
    kv_cache_dtype: str

    dnh_sharding: Sharding = ()
    dkh_sharding: Sharding = ()
    nhd_sharding: Sharding = ()

    activation_q_td: Sharding = (ShardingAxisName.ATTN_DATA)
    query_tnh: P = P(ShardingAxisName.ATTN_DATA)
    keyvalue_skh: P = P(ShardingAxisName.ATTN_DATA)

    attn_o_tnh: P = P(ShardingAxisName.ATTN_DATA)
    rngs: InitVar[nnx.Rngs]

    random_init: bool = False
    attention_chunk_size: int | None = None
    rope_input_ordering: str = "split"

    _q_scale: float = 1.0
    _k_scale: float = 1.0
    _v_scale: float = 1.0

    kv_cache_quantized_dtype = None

    def __post_init__(self, rngs: nnx.Rngs):
        """Initializes the weight kernels for Q, K, V, and O projections."""
        N = self.num_attention_heads
        K = self.num_key_value_heads
        D = self.hidden_size
        H = self.head_dim

        self.kernel_q_proj_DNH = create_param(rngs, (D, N, H),
                                              self.dnh_sharding,
                                              self.dtype,
                                              random_init=self.random_init)
        self.kernel_k_proj_DKH = create_param(rngs, (D, K, H),
                                              self.dkh_sharding,
                                              self.dtype,
                                              random_init=self.random_init)
        self.kernel_v_proj_DKH = create_param(rngs, (D, K, H),
                                              self.dkh_sharding,
                                              self.dtype,
                                              random_init=self.random_init)
        self.kernel_o_proj_NHD = create_param(rngs, (N, H, D),
                                              self.nhd_sharding,
                                              self.dtype,
                                              random_init=self.random_init)

        if self.kv_cache_dtype != "auto":
            self.kv_cache_quantized_dtype = utils.get_jax_dtype_from_str_dtype(
                self.kv_cache_dtype)

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
        md = attention_metadata
        x_SD = jnp.asarray(x, self.dtype)
        x_q_TD = nnx.with_sharding_constraint(x, self.activation_q_td)
        H = self.head_dim
        with jax.named_scope("q_proj"):
            q_TNH = jnp.einsum('TD,DNH -> TNH', x_q_TD,
                               self.kernel_q_proj_DNH.value)
            if use_attention_rope:
                q_TNH = apply_rope(q_TNH, md.input_positions, H,
                                   self.rope_theta, self.rope_scaling,
                                   self.rope_input_ordering)
            q_TNH = nnx.with_sharding_constraint(q_TNH, self.query_tnh)
        with jax.named_scope("k_proj"):
            k_SKH = jnp.einsum('SD,DKH -> SKH', x_SD,
                               self.kernel_k_proj_DKH.value)
            if use_attention_rope:
                k_SKH = apply_rope(k_SKH, md.input_positions, H,
                                   self.rope_theta, self.rope_scaling,
                                   self.rope_input_ordering)
            k_SKH = nnx.with_sharding_constraint(k_SKH, self.keyvalue_skh)

        with jax.named_scope("v_proj"):
            v_SKH = jnp.einsum('SD,DKH -> SKH', x_SD,
                               self.kernel_v_proj_DKH.value)

        q_scale = k_scale = v_scale = None
        if self.kv_cache_quantized_dtype:
            # TODO(kyuyeunk/jacobplatin): Enable w8a8 when VREG spill issue is resolved.
            # q_scale = self._q_scale
            k_scale = self._k_scale
            v_scale = self._v_scale
            k_SKH, v_SKH = quantize_kv(self.kv_cache_quantized_dtype, k_SKH,
                                       v_SKH, k_scale, v_scale)

        with jax.named_scope("attn_op"):
            new_kv_cache, outputs_TNH = self.attention(
                is_prefill,
                kv_cache,
                q_TNH,
                k_SKH,
                v_SKH,
                attention_metadata,
                self.mesh,
                q_scale=q_scale,
                k_scale=k_scale,
                v_scale=v_scale,
            )

        with jax.named_scope("o_proj"):
            o_TD = jnp.einsum('TNH,NHD -> TD', outputs_TNH,
                              self.kernel_o_proj_NHD.value)
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
        q_scale: float | None = None,
        k_scale: float | None = None,
        v_scale: float | None = None,
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
            q_scale: Quantization scale for q.
            k_scale: Quantization scale for k.
            v_scale: Quantization scale for v.

        Returns:
            A tuple containing:
                - The updated KV cache.
                - The attention output tensor of shape
                  `(seq, num_q_heads, head_dim)`.
        """
        md = attention_metadata
        kv_cache_spec = P(ShardingAxisName.ATTN_DATA, None, "model")
        in_specs = (
            self.query_tnh,  # q
            self.keyvalue_skh,  # k
            self.keyvalue_skh,  # v
            kv_cache_spec,  # kv_cache
            P(ShardingAxisName.ATTN_DATA),  # md.seq_lens
            P(ShardingAxisName.ATTN_DATA),  # page_indices_flat
            P(ShardingAxisName.ATTN_DATA),  # query_start_loc
            P(ShardingAxisName.ATTN_DATA),  # distribution
        )

        out_specs = (self.attn_o_tnh, kv_cache_spec)

        def _ragged_paged_attention(*args):
            return ragged_paged_attention(
                *args,
                sm_scale=q_TNH.shape[-1]**-0.5,
                q_scale=q_scale,
                k_scale=k_scale,
                v_scale=v_scale,
            )

        output_TNH, kv_cache = jax.jit(
            jax.shard_map(
                _ragged_paged_attention,
                mesh=mesh,
                in_specs=in_specs,
                out_specs=out_specs,
                check_vma=False,
            ))(
                q_TNH,
                k_SKH,
                v_SKH,
                kv_cache,
                md.seq_lens,
                md.block_tables,
                md.query_start_loc,
                md.request_distribution,
            )
        return kv_cache, output_TNH
