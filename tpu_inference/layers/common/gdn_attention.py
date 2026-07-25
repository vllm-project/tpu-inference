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
import functools
from typing import Optional, Tuple

import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P

from tpu_inference.kernels.gdn.v3 import wrapper
from tpu_inference.layers.common.sharding import ShardingAxisName
from tpu_inference.utils import get_mesh_shape_product


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
        P(None),  # query_start_loc
        P(None),  # state_indices
        P(None),  # distribution
        P(None),  # seq_lens
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

    def local_gdn_kernel(
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
    ):
        local_num_seqs = conv_state.shape[0]
        global_num_seqs = state_indices.shape[0]
        
        # When Data Parallelism is enabled, state_indices and query_start_loc
        # passed into run_jax_gdn_attention are global. Compute exact local metadata for this DP shard.
        if global_num_seqs != local_num_seqs and local_num_seqs > 0:
            local_num_tokens = j_mixed_qkv.shape[0]
            tokens_per_seq = local_num_tokens // local_num_seqs
            query_start_loc = jnp.arange(0, (local_num_seqs + 1) * tokens_per_seq, tokens_per_seq, dtype=jnp.int32)
            state_indices = jnp.arange(local_num_seqs, dtype=jnp.int32)
            distribution = jnp.array([0, local_num_tokens, 0], dtype=jnp.int32)
            seq_lens = jnp.full((local_num_seqs,), tokens_per_seq, dtype=jnp.int32)

        return wrapper.fused_conv1d_gdn(
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
            n_kq=n_kq // tp_size,
            n_v=n_v // tp_size,
            d_k=d_k,
            d_v=d_v,
            kernel_size=kernel_size,
        )

    mapped_fn = jax.shard_map(
        local_gdn_kernel,
        mesh=mesh,
        in_specs=in_specs,
        out_specs=out_specs,
        check_vma=False,
    )

    def apply_sharding(x, spec):
        if x is None or spec is None:
            return x
        return jax.lax.with_sharding_constraint(x, jax.sharding.NamedSharding(mesh, spec))

    j_mixed_qkv = apply_sharding(j_mixed_qkv, in_specs[0])
    j_b = apply_sharding(j_b, in_specs[1])
    j_a = apply_sharding(j_a, in_specs[2])
    conv_state = apply_sharding(conv_state, in_specs[3])
    recurrent_state = apply_sharding(recurrent_state, in_specs[4])
    j_conv_weight = apply_sharding(j_conv_weight, in_specs[5])
    j_conv_bias = apply_sharding(j_conv_bias, in_specs[6])
    j_A_log = apply_sharding(j_A_log, in_specs[7])
    j_dt_bias = apply_sharding(j_dt_bias, in_specs[8])

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
