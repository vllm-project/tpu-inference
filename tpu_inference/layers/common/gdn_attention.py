# Copyright 2026 Google LLC
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
"""
Bridge the torch gdn_attention_core op for gated deltanet attention TPU impl

"""
import dataclasses
import enum
import functools
from typing import Optional, Tuple

import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P

from tpu_inference.kernels.gdn.fused_gdn_kernel_wrapper import \
    ragged_gated_delta_rule as ragged_gated_delta_rule_fused
from tpu_inference.layers.common.ragged_conv1d_jax import \
    ragged_conv1d as ragged_conv1d_jax
from tpu_inference.layers.common.ragged_gated_delta_rule_chunked import \
    ragged_gated_delta_rule as ragged_gated_delta_rule_chunked
from tpu_inference.layers.common.ragged_gated_delta_rule_ref import \
    ragged_gated_delta_rule as ragged_gated_delta_rule_ref
from tpu_inference.layers.common.sharding import ShardingAxisName
from tpu_inference.utils import get_mesh_shape_product


class RaggedConv1dImpl(enum.Enum):
    JAX = "ragged_conv1d_jax"


class RaggedGatedDeltaRuleImpl(enum.Enum):
    REF = "ragged_gated_delta_rule_ref"
    CHUNKED = "ragged_gated_delta_rule_chunked"
    FUSED = "fused_gdn_kernel"


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class GdnAttentionConfig:
    ragged_conv1d_impl: RaggedConv1dImpl = RaggedConv1dImpl.JAX
    ragged_gated_delta_rule_impl: RaggedGatedDeltaRuleImpl = (
        RaggedGatedDeltaRuleImpl.REF)


def run_jax_gdn_attention_local(
    mixed_qkv: jnp.ndarray,
    b: jnp.ndarray,
    a: jnp.ndarray,
    conv_state: jnp.ndarray,
    recurrent_state: jnp.ndarray,
    conv_weight: jnp.ndarray,
    conv_bias: Optional[jnp.ndarray],
    A_log: jnp.ndarray,
    dt_bias: jnp.ndarray,
    query_start_loc: jnp.ndarray,
    state_indices: jnp.ndarray,
    distribution: jnp.ndarray,
    seq_lens: jnp.ndarray,
    n_kq: int,
    n_v: int,
    d_k: int,
    d_v: int,
    kernel_size: int,
    config: GdnAttentionConfig = GdnAttentionConfig(),
) -> Tuple[Tuple[jnp.ndarray, jnp.ndarray], jnp.ndarray]:
    """Runs the local JAX GDN attention mechanism with combined QKV tensors.

    Args:
        mixed_qkv: Combined QKV tensor of shape `(num_tokens, dim)`.
        b: B tensor of shape `(num_tokens, n_v)`.
        a: A tensor of shape `(num_tokens, n_v)`.
        conv_state: Combined convolutional state of shape `(num_blocks,
          kernel_size - 1, dim)`. `num_blocks` is always equal or larger than
          `max_seqs + 1`. The first block is a null_block and only used for
          padded / invalid tokens.
        recurrent_state: Recurrent state of shape `(num_blocks, n_v, d_k, d_v)`.
        conv_weight: Combined convolutional weight of shape `(dim, 1,
          kernel_size)`.
        conv_bias: Optional combined convolutional bias of shape `(dim,)`.
        A_log: Log of A parameter of shape `(n_v,)`.
        dt_bias: Delta T bias of shape `(n_v,)`.
        query_start_loc: Tensor of shape `(num_seqs + 1,)` with start locations of
          each sequence.
        state_indices: Tensor of shape `(max_reqs,)` mapping request index to
          state index.
        distribution: Tensor of shape `(3,)` int32 — `(decode_end, prefill_end,
          mixed_end)`.
        seq_lens: Tensor of shape `(max_reqs,)` with the total sequence length
          per request (computed + scheduled). Used to derive
          ``has_initial_state`` so brand-new prefills don't read stale state
          from a reused mamba slot, mirroring GPU's
          ``initial_state[~has_initial_state, ...] = 0`` in
          ``gdn_linear_attn._forward_core``.
        n_kq: Number of key/query heads.
        n_v: Number of value heads.
        d_k: Dimension of key.
        d_v: Dimension of value.
        kernel_size: Convolution kernel size.
        config: Configuration for implementation selection.

    Returns:
        A tuple containing:
        - A tuple of (new_conv_state, new_recurrent_state).
        - The output tensor of shape `(num_tokens, n_v * d_v)`.
    """
    # has_initial_state[i] = True iff request i already has computed
    # tokens in its mamba slot (chunked-prefill continuation, prefix-cache
    # hit, or running decode). False for brand-new prefills, in which
    # case the conv1d, the chunked / ref delta-rule impls, and the fused
    # Pallas recurrent kernel all zero the slot's prior state before
    # the update so a freshly-allocated mamba slot can't leak its
    # previous tenant's state. context_len = seq_len - query_len.
    max_reqs = seq_lens.shape[0]
    query_lens = query_start_loc[1:max_reqs + 1] - query_start_loc[:max_reqs]
    has_initial_state = (seq_lens - query_lens) > 0

    # TODO: Switch conv implementaion based on config once we have more than 1 impl
    conv_impl = ragged_conv1d_jax

    out_mixed_qkv, new_conv_state = conv_impl(
        mixed_qkv,
        conv_state,
        conv_weight,
        conv_bias,
        query_start_loc,
        state_indices,
        distribution,
        has_initial_state,
        kernel_size=kernel_size,
    )

    out_mixed_qkv = jax.nn.silu(out_mixed_qkv)

    if config.ragged_gated_delta_rule_impl == RaggedGatedDeltaRuleImpl.FUSED:
        ragged_gdn_impl = functools.partial(
            ragged_gated_delta_rule_fused,
            n_kq=n_kq,
            n_v=n_v,
            d_k=d_k,
            d_v=d_v,
        )
    elif config.ragged_gated_delta_rule_impl == RaggedGatedDeltaRuleImpl.REF:
        ragged_gdn_impl = functools.partial(
            ragged_gated_delta_rule_ref,
            n_kq=n_kq,
            n_v=n_v,
            d_k=d_k,
            d_v=d_v,
        )
    else:
        ragged_gdn_impl = functools.partial(
            ragged_gated_delta_rule_chunked,
            n_kq=n_kq,
            n_v=n_v,
            d_k=d_k,
            d_v=d_v,
            use_qk_norm_in_gdn=True,
        )

    new_recurrent_state, output = ragged_gdn_impl(
        out_mixed_qkv,
        b,
        a,
        recurrent_state,
        A_log,
        dt_bias,
        query_start_loc,
        state_indices,
        distribution,
        has_initial_state,
    )

    return (new_conv_state, new_recurrent_state), output


def run_jax_gdn_attention(
    j_mixed_qkv: jnp.ndarray,
    j_b: jnp.ndarray,
    j_a: jnp.ndarray,
    conv_state: jnp.ndarray,
    recurrent_state: jnp.ndarray,
    j_conv_weight: jnp.ndarray,
    j_conv_bias: Optional[jnp.ndarray],
    j_A_log: jnp.ndarray,
    j_dt_bias: jnp.ndarray,
    state_indices: jnp.ndarray,
    query_start_loc: jnp.ndarray,
    distribution: jnp.ndarray,
    seq_lens: jnp.ndarray,
    n_kq: int,
    n_v: int,
    d_k: int,
    d_v: int,
    kernel_size: int,
    mesh: jax.sharding.Mesh,
    config: GdnAttentionConfig = GdnAttentionConfig(),
) -> Tuple[Tuple[jnp.ndarray, jnp.ndarray], jnp.ndarray]:
    """Runs the Jax GDN attention mechanism.

    Args:
        j_mixed_qkv: Input tensor of shape `(num_tokens, dim)`.
        j_b: Input tensor of shape `(num_tokens, n_v)`.
        j_a: Input tensor of shape `(num_tokens, n_v)`.
        conv_state: Convolutional state tensor of shape `(num_blocks, kernel_size
          - 1, dim)`. `num_blocks` is always equal or larger than `max_seqs +
          1`. The first block is a null_block and only used for padded / invalid
          tokens.
        recurrent_state: Recurrent state tensor of shape `(num_blocks, n_v, d_k,
          d_v)`.
        j_conv_weight: Convolutional weight tensor of shape `(dim, 1,
          kernel_size)`.
        j_conv_bias: Optional convolutional bias tensor of shape `(dim,)`.
        j_A_log: Log of A parameter tensor of shape `(n_v,)`.
        j_dt_bias: Delta T bias tensor of shape `(n_v,)`.
        state_indices: Tensor of shape `(max_reqs,)` mapping request index to
          state index.
        query_start_loc: Tensor of shape `(num_seqs + 1,)` with start locations of
          each sequence.
        distribution: Tensor of shape `(3,)` int32 — `(decode_end, prefill_end,
          mixed_end)`.
        seq_lens: Tensor of shape `(max_reqs,)` with the total sequence length
          per request (computed + scheduled). Used inside the local function
          to derive ``has_initial_state``.
        n_kq: Number of key/query heads.
        n_v: Number of value heads.
        d_k: Dimension of key.
        d_v: Dimension of value.
        kernel_size: Convolution kernel size.
        mesh: The device mesh for distributed computation.
        config: Configuration for implementation selection.

    Returns:
        A tuple containing:
        - A tuple of (new_conv_state, new_recurrent_state).
          - new_conv_state: `(num_blocks, kernel_size - 1, dim)`
          - new_recurrent_state: `(num_blocks, n_v, d_k, d_v)`
        - The output tensor of shape `(num_tokens, n_v * d_v)`.
    """
    in_specs = (
        P(ShardingAxisName.ATTN_DATA,
          ShardingAxisName.ATTN_HEAD),  # j_mixed_qkv
        P(ShardingAxisName.ATTN_DATA, ShardingAxisName.ATTN_HEAD),  # j_b
        P(ShardingAxisName.ATTN_DATA, ShardingAxisName.ATTN_HEAD),  # j_a
        P(ShardingAxisName.ATTN_DATA, None,
          ShardingAxisName.ATTN_HEAD),  # conv_state
        P(ShardingAxisName.ATTN_DATA, ShardingAxisName.ATTN_HEAD, None,
          None),  # recurrent_state
        P(ShardingAxisName.ATTN_HEAD, None, None),  # j_conv_weight
        P(ShardingAxisName.ATTN_HEAD)
        if j_conv_bias is not None else None,  # j_conv_bias
        P(ShardingAxisName.ATTN_HEAD),  # j_A_log
        P(ShardingAxisName.ATTN_HEAD),  # j_dt_bias
        P(ShardingAxisName.ATTN_DATA),  # query_start_loc
        P(ShardingAxisName.ATTN_DATA),  # state_indices
        P(ShardingAxisName.ATTN_DATA),  # distribution
        P(ShardingAxisName.ATTN_DATA),  # seq_lens
    )

    out_specs = (
        (
            P(ShardingAxisName.ATTN_DATA, None,
              ShardingAxisName.ATTN_HEAD),  # new_conv_state
            P(ShardingAxisName.ATTN_DATA, ShardingAxisName.ATTN_HEAD, None,
              None),  # new_recurrent_state
        ),
        P(ShardingAxisName.ATTN_DATA, ShardingAxisName.ATTN_HEAD),  # output
    )

    tp_size = get_mesh_shape_product(mesh, ShardingAxisName.ATTN_HEAD)

    p_run_jax_gdn_attention_local = functools.partial(
        run_jax_gdn_attention_local,
        n_kq=n_kq // tp_size,
        n_v=n_v // tp_size,
        d_k=d_k,
        d_v=d_v,
        kernel_size=kernel_size,
        config=config,
    )

    mapped_fn = jax.shard_map(
        p_run_jax_gdn_attention_local,
        mesh=mesh,
        in_specs=in_specs,
        out_specs=out_specs,
        check_vma=False,
    )

    (new_conv_state, new_recurrent_state), output = mapped_fn(
        j_mixed_qkv,
        j_b,
        j_a,
        conv_state,
        recurrent_state,
        j_conv_weight,
        j_conv_bias,
        j_A_log,
        j_dt_bias,
        query_start_loc,
        state_indices,
        distribution,
        seq_lens,
    )

    return (new_conv_state, new_recurrent_state), output
