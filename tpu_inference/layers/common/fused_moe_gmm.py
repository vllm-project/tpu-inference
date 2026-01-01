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

import functools

import jax
from jax import numpy as jnp
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from tpu_inference.kernels.megablox.gmm import gmm
from tpu_inference.layers.common.sharding import ShardingAxisName
from tpu_inference.layers.common.utils import \
    slice_sharded_tensor_for_concatenation
from tpu_inference.utils import get_mesh_shape_product


def activation_fn(activation: str, x1: jax.Array, x2: jax.Array) -> jax.Array:
    match activation:
        case "silu":
            return jax.nn.silu(x1) * x2
        case "swigluoai":
            return _swigluoai(x1, x2)
        case _:
            raise NotImplementedError(
                f"FusedMoE does not support {activation} activation")


def _swigluoai(x1: jax.Array,
               x2: jax.Array,
               alpha=1.702,
               limit=7.0) -> jax.Array:
    x1 = jnp.clip(x1, a_max=limit)
    x2 = jnp.clip(x2, a_min=-limit, a_max=limit)

    gated_activation = x1 * jax.nn.sigmoid(alpha * x1)

    return gated_activation * (x2 + 1)


def _round_up_to_multiple_of_128_within_limit(x: int, limit: int) -> int:
    """
    Rounds the given integer `x` up to the nearest multiple of 128, without
    exceeding the specified `limit`.

    If `x` is less than or equal to 128, returns 128.
    If `x` is less than `limit`, returns the smallest multiple of 128 greater
    than or equal to `x`.
    If `x` is greater than or equal to `limit`, searches for the largest
    multiple of 128 less than or equal to `limit` (down to 512) that divides `x`
    evenly, and returns it.
    If no such candidate is found, returns `limit`.

    Args:
        x (int): The integer to round up.
        limit (int): The upper bound (must be a multiple of 128).

    Returns:
        int: The rounded value according to the rules above.

    Raises:
        AssertionError: If `limit` is less than 128 or not a multiple of 128.
    """
    assert limit >= 128 and limit % 128 == 0
    if x <= 128:
        return 128
    if x < limit:
        return (x + 127) // 128 * 128
    for candidate in range(limit, 511, -128):
        if x % candidate == 0:
            return candidate
    return limit


def _get_tiling_size_for_gmm_kernel(m: int, k: int, n: int,
                                    g: int) -> tuple[int, int, int]:
    """
    Calculate optimal tiling sizes for a GMM kernel in a Mixture of Experts
    (MoE) setting.

    Args:
        m (int): The total number of tokens.
        n (int): The output feature dimension.
        k (int): The input feature dimension.
        g (int): The number of experts.

    Returns:
        tuple[int, int, int]: A tuple (tm, tk, tn)
    """

    # TODO(Chengji): increase the upper limit tiling size of m when we can set
    # the vmem size to be used for gmm kernel.
    # NOTE: In average each expert has m // g tokens, but as it might be
    # unbalanced, here we doubled the token size when choosing tiling size of m.
    # 2m//g can be either greater or less than 512. If there are 32 tokens and
    # topk=2, m=topk * num_tokens=64, in this case, 2*m//g will be less than
    # 512.
    tm = _round_up_to_multiple_of_128_within_limit(2 * m // g, 512)
    tm = min(tm, m)  # there's a requirement that m % tm == 0
    # k/n correspond to n_input_features/n_output_features in the matmul so they
    # are normally greater than 2048, unless the num shards is large.
    tk = _round_up_to_multiple_of_128_within_limit(k, 2048)
    tn = _round_up_to_multiple_of_128_within_limit(n, 2048)
    return tm, tk, tn


def tensor_sharded_gmm_merged_column_parallel(
    lhs: jax.Array,
    rhs: jax.Array,
    rhs_scale: jax.Array | None,
    rhs_bias: jax.Array | None,
    group_sizes: jax.Array,
    mesh: Mesh,
) -> list[jax.Array]:

    def _gmm(lhs, rhs, rhs_scale, rhs_bias, group_sizes):
        m, g, n, k = lhs.shape[0], *rhs.shape
        tm, tk, tn = _get_tiling_size_for_gmm_kernel(m, k, n, g)
        return gmm(
            lhs,
            rhs,
            group_sizes,
            rhs_scale=rhs_scale,
            rhs_bias=rhs_bias,
            preferred_element_type=lhs.dtype,
            tiling=(tm, tk, tn),
            transpose_rhs=True,
            group_offset=jnp.array(0),
        )

    rhs_scale_spec = None if rhs_scale is None else P(
        None, None, None, ShardingAxisName.MLP_TENSOR)
    rhs_bias_spec = None if rhs_bias is None else P(
        None, None, ShardingAxisName.MLP_TENSOR)

    gmm_result = jax.shard_map(
        _gmm,
        mesh=mesh,
        in_specs=(P(ShardingAxisName.MLP_DATA,
                    None), P(None, ShardingAxisName.MLP_TENSOR,
                             None), rhs_scale_spec, rhs_bias_spec,
                  P(ShardingAxisName.MLP_DATA)),
        out_specs=(P(ShardingAxisName.MLP_DATA, ShardingAxisName.MLP_TENSOR)),
        check_vma=False,
    )(lhs, rhs, rhs_scale, rhs_bias, group_sizes)

    tp_size = get_mesh_shape_product(mesh, ShardingAxisName.MLP_TENSOR)
    intermediate_size = gmm_result.shape[-1] // 2
    output_sizes = [intermediate_size, intermediate_size]
    return slice_sharded_tensor_for_concatenation(gmm_result, output_sizes,
                                                  tp_size)


def tensor_sharded_gmm_row_parallel(
    lhs: jax.Array,
    rhs: jax.Array,
    rhs_scale: jax.Array | None,
    rhs_bias: jax.Array | None,
    group_sizes: jax.Array,
    mesh: Mesh,
) -> jax.Array:

    def _gmm_all_reduce(lhs, rhs, rhs_scale, rhs_bias, group_sizes):
        m, g, n, k = lhs.shape[0], *rhs.shape
        tm, tk, tn = _get_tiling_size_for_gmm_kernel(m, k, n, g)
        if rhs_bias is not None:
            shard_id = jax.lax.axis_index(ShardingAxisName.MLP_TENSOR).sum()
            rhs_bias = jnp.where(shard_id == 0, rhs_bias, 0)
        out = gmm(
            lhs,
            rhs,
            group_sizes,
            rhs_scale=rhs_scale,
            rhs_bias=rhs_bias,
            preferred_element_type=lhs.dtype,
            tiling=(tm, tk, tn),
            transpose_rhs=True,
            group_offset=jnp.array(0),
        )
        return jax.lax.psum(out, axis_name=ShardingAxisName.MLP_TENSOR)

    num_blocks = 1 if rhs_scale is None else rhs_scale.shape[1]
    rhs_scale_spec = None if num_blocks == 1 else P(
        None, ShardingAxisName.MLP_TENSOR, None, None)
    rhs_bias_spec = None if rhs_bias is None else P(None, None, None)
    gmm_result = jax.shard_map(
        _gmm_all_reduce,
        mesh=mesh,
        in_specs=(P(ShardingAxisName.MLP_DATA, ShardingAxisName.MLP_TENSOR),
                  P(None, None, ShardingAxisName.MLP_TENSOR), rhs_scale_spec,
                  rhs_bias_spec, P(ShardingAxisName.MLP_DATA)),
        out_specs=(P(ShardingAxisName.MLP_DATA)),
        check_vma=False,
    )(lhs, rhs, rhs_scale, rhs_bias, group_sizes)

    return gmm_result.astype(lhs.dtype)


def expert_sharded_gmm(
    lhs: jax.Array,
    rhs: jax.Array,
    rhs_scale: jax.Array | None,
    rhs_bias: jax.Array | None,
    group_sizes: jax.Array,
    is_last_expert: bool,
    mesh: Mesh,
) -> jax.Array:
    ep_size = get_mesh_shape_product(mesh, ShardingAxisName.MLP_TENSOR)
    ep_p_spec = P(ShardingAxisName.EXPERT)
    num_experts = rhs.shape[0]
    num_experts_per_shard = num_experts // ep_size
    group_offset = jnp.arange(0, num_experts, num_experts_per_shard)

    def _gmm(lhs, rhs, rhs_scale, rhs_bias, group_sizes, group_offset):
        m, g, n, k = lhs.shape[0], *rhs.shape
        tm, tk, tn = _get_tiling_size_for_gmm_kernel(m, k, n, g)

        gmm_res = gmm(
            lhs=lhs,
            rhs=rhs,
            rhs_scale=rhs_scale,
            rhs_bias=rhs_bias,
            group_sizes=group_sizes,
            preferred_element_type=lhs.dtype,
            tiling=(tm, tk, tn),
            transpose_rhs=True,
            group_offset=group_offset[0],
        )
        return gmm_res

    # The result from gmm on each shard has the same shape, but only the rows
    # for this shard has non-zero values. Taking below as an working example:
    #       A, A, A, A     0, 0, 0, 0     0, 0, 0, 0     0, 0, 0, 0
    #       A, A, A, A     0, 0, 0, 0     0, 0, 0, 0     0, 0, 0, 0
    #       A, A, A, A     0, 0, 0, 0     0, 0, 0, 0     0, 0, 0, 0
    #       0, 0, 0, 0     B, B, B, B     0, 0, 0, 0     0, 0, 0, 0
    #       0, 0, 0, 0     B, B, B, B     0, 0, 0, 0     0, 0, 0, 0
    #       0, 0, 0, 0     0, 0, 0, 0     C, C, C, C     0, 0, 0, 0
    #       0, 0, 0, 0     0, 0, 0, 0     C, C, C, C     0, 0, 0, 0
    #       0, 0, 0, 0     0, 0, 0, 0     C, C, C, C     0, 0, 0, 0
    #       0, 0, 0, 0     0, 0, 0, 0     C, C, C, C     0, 0, 0, 0
    #       0, 0, 0, 0     0, 0, 0, 0     C, C, C, C     0, 0, 0, 0
    #       0, 0, 0, 0     0, 0, 0, 0     0, 0, 0, 0     D, D, D, D
    #       0, 0, 0, 0     0, 0, 0, 0     0, 0, 0, 0     D, D, D, D
    #       0, 0, 0, 0     0, 0, 0, 0     0, 0, 0, 0     D, D, D, D
    #       0, 0, 0, 0     0, 0, 0, 0     0, 0, 0, 0     D, D, D, D
    #        shard-0        shard-1        shard-2        shard-3
    # Each shards has 3 (row A), 2 (row B), 5 (row C) and 4 (row D).
    lhs_spec = ep_p_spec if is_last_expert else P()
    rhs_spec = ep_p_spec
    rhs_scale_spec = None if rhs_scale is None else ep_p_spec
    rhs_bias_spec = None if rhs_bias is None else ep_p_spec
    gmm_res = jax.shard_map(
        _gmm,
        mesh=mesh,
        in_specs=(
            lhs_spec,
            rhs_spec,
            rhs_scale_spec,
            rhs_bias_spec,
            P(),
            ep_p_spec,
        ),
        out_specs=ep_p_spec,
        check_vma=False,
    )(lhs, rhs, rhs_scale, rhs_bias, group_sizes, group_offset)

    if not is_last_expert:
        return gmm_res

    # For i-th shard, it is responsible groups (AKA experts) from
    # i*num_experts_per_shard to (i+1)*num_experts_per_shard We sum them up to
    # get total rows in that shard, and that is the size for shard to send to
    # its peers. This is also the number of non-zero rows from the gmm results.
    # In the working example, send_sizes would be [3, 2, 5, 4].

    # group_sizes has shape of [num_tokens_per_shard * num_experts_per_shard].
    # So reshaping to [num_tokens_per_shard, num_experts_per_shard] and applying
    # sum(axis=1) will get desired send_sizes shaped [num_tokens_per_shard].
    send_sizes = group_sizes.reshape(-1, num_experts_per_shard).sum(axis=1)
    # In the working example, input_offsets would be [0, 3, 5, 10]
    input_offsets = jnp.concatenate((jnp.array([0]), send_sizes.cumsum()[:-1]))
    output_offsets = input_offsets
    recv_sizes = send_sizes

    def _ragged_all_to_all(operand, input_offsets, send_sizes, output_offsets,
                           recv_sizes):
        output = jnp.zeros_like(operand)

        # input_offsets, send_sizes and output_offsets are sharded and there is
        # only 1 elemnt in each shard, we are taking the 0-th element from them
        # just so that jnp.repeat generates the arrays with correct shape.
        input_offsets_of_shard = jnp.repeat(input_offsets[0], ep_size)
        send_sizes_of_shard = jnp.repeat(send_sizes[0], ep_size)
        output_offsets_of_shard = jnp.repeat(output_offsets[0], ep_size)

        # recv_sizes is replicated across shards, because all the shards receive
        # the same data and write to the output in the same way (same
        # output_offsets and same recv_sizes) and thus generates replicated
        # output.
        recv_sizes_of_shard = recv_sizes

        # In the working example, for each shard, the values of the offsets and
        # sizes would be:
        #                                shard-0         shard-1         shard-2         shard-3
        # input_offsets_of_shard       [0, 0, 0, 0]    [3, 3, 3, 3]    [5, 5, 5, 5]    [10,10,10,10]
        # send_sizes_of_shard          [3, 3, 3, 3]    [2, 2, 2, 2]    [5, 5, 5, 5]    [4, 4, 4, 4 ]
        # output_offsets_of_shard      [0, 0, 0, 0]    [0, 0, 0, 0]    [0, 0, 0, 0]    [10,10,10,10]
        # recv_sizes_of_shard          [3, 2, 5, 4]    [3, 2, 5, 4]    [3, 2, 5, 4]    [3, 2, 5, 4]
        return jax.lax.ragged_all_to_all(operand,
                                         output,
                                         input_offsets_of_shard,
                                         send_sizes_of_shard,
                                         output_offsets_of_shard,
                                         recv_sizes_of_shard,
                                         axis_name=ShardingAxisName.EXPERT)

    # Use ragged_all_to_all to send the result from gmm for each expert to all
    # the shards.  In the working example, the result would be:
    #       A, A, A, A     A, A, A, A     A, A, A, A     A, A, A, A
    #       A, A, A, A     A, A, A, A     A, A, A, A     A, A, A, A
    #       A, A, A, A     A, A, A, A     A, A, A, A     A, A, A, A
    #       B, B, B, B     B, B, B, B     B, B, B, B     B, B, B, B
    #       B, B, B, B     B, B, B, B     B, B, B, B     B, B, B, B
    #       C, C, C, C     C, C, C, C     C, C, C, C     C, C, C, C
    #       C, C, C, C     C, C, C, C     C, C, C, C     C, C, C, C
    #       C, C, C, C     C, C, C, C     C, C, C, C     C, C, C, C
    #       C, C, C, C     C, C, C, C     C, C, C, C     C, C, C, C
    #       C, C, C, C     C, C, C, C     C, C, C, C     C, C, C, C
    #       D, D, D, D     D, D, D, D     D, D, D, D     D, D, D, D
    #       D, D, D, D     D, D, D, D     D, D, D, D     D, D, D, D
    #       D, D, D, D     D, D, D, D     D, D, D, D     D, D, D, D
    #       D, D, D, D     D, D, D, D     D, D, D, D     D, D, D, D
    #        shard-0        shard-1        shard-2        shard-3
    return jax.shard_map(
        _ragged_all_to_all,
        mesh=mesh,
        in_specs=(ep_p_spec, ep_p_spec, ep_p_spec, ep_p_spec, P()),
        out_specs=(P(ShardingAxisName.MLP_DATA)),
        check_vma=False,
    )(gmm_res, input_offsets, send_sizes, output_offsets, recv_sizes)


@functools.partial(
    jax.jit,
    static_argnames=(
        "topk",
        "renormalize",
        "mesh",
        "use_ep",
        "activation",
    ),
)
def fused_moe_func(
    hidden_states: jax.Array,
    w1: jax.Array,
    w2: jax.Array,
    w1_scale: jax.Array | None,
    w2_scale: jax.Array | None,
    w1_bias: jax.Array | None,
    w2_bias: jax.Array | None,
    gating_output: jax.Array,
    topk: int,
    renormalize: bool,
    mesh: Mesh,
    use_ep: bool,
    activation: str,
) -> jax.Array:
    """Route tokens in hidden_states into each experts based on routing.

    Args:
        hidden_states: [num_tokens, hidden_size]
        w1: first moe weights [num_experts, intermediate_size * 2, hidden_size]
        w2: second moe weights [num_experts, hidden_size, intermediate_size]
        w1_scale: w1 scale [num_experts, num_blocks, 1, intermediate_size * 2]
        w2_scale: w2 scale [num_experts, num_blocks, 1, hidden_size]
        w1_bias: optional bias of w1 [num_experts, 1, intermediate_size * 2]
        w2_bias: optional bias of w2 [num_experts, 1, hidden_size]
        gating_output: routing information of tokens [num_tokens, num_experts]
        topk: number of experts to choose per token.
        renormalize: normalize gating_output.
        mesh: mesh to perform moe.
        use_ep: use expert parallelism.
        activation: activation function to perform on the output of w1.

    Returns:
        Output of moe operation [num_tokens, hidden_size]
    """
    num_tokens, hidden_size = hidden_states.shape
    global_num_experts, _, padded_hidden_size = w1.shape
    dtype = hidden_states.dtype

    assert (num_tokens * topk) % 16 == 0, (
        "The kernel requires num_tokens * topk to be a multiple of "
        f"16 but got {num_tokens}*{topk}={num_tokens*topk}")

    assert gating_output.shape == (num_tokens, global_num_experts)

    topk_weights = jax.nn.softmax(gating_output.astype(jnp.float32), axis=-1)
    # All-gather topk weights for attention dp
    topk_weights = jax.lax.with_sharding_constraint(
        topk_weights, NamedSharding(mesh, P(ShardingAxisName.MLP_DATA, None)))
    topk_weights, topk_indices = jax.lax.top_k(topk_weights, k=topk)
    if renormalize:
        topk_weights = topk_weights / topk_weights.sum(axis=-1, keepdims=True)
    topk_weights = topk_weights.astype(dtype)

    def _process_tokens_locally(hidden_states_local, topk_indices_local):
        num_tokens_local = hidden_states_local.shape[0]
        topk_indices_flat = topk_indices_local.flatten()
        topk_argsort_indices = jnp.argsort(topk_indices_flat)
        topk_argsort_revert_indices = jnp.argsort(topk_argsort_indices)
        token_indices = jnp.arange(num_tokens_local,
                                   dtype=jnp.int32).repeat(topk)
        token_indices_sorted = token_indices[topk_argsort_indices]
        group_sizes_local = jnp.bincount(topk_indices_flat,
                                         length=global_num_experts)

        x = hidden_states_local[token_indices_sorted]
        return x, group_sizes_local, topk_argsort_revert_indices

    x, group_sizes, topk_argsort_revert_indices = jax.shard_map(
        _process_tokens_locally,
        mesh=mesh,
        in_specs=(P(ShardingAxisName.MLP_DATA,
                    None), P(ShardingAxisName.MLP_DATA, None)),
        out_specs=(P(ShardingAxisName.MLP_DATA, None),
                   P(ShardingAxisName.MLP_DATA), P(ShardingAxisName.MLP_DATA)),
    )(hidden_states, topk_indices)

    x = jnp.pad(x, ((0, 0), (0, padded_hidden_size - hidden_size)))

    if use_ep:
        x = expert_sharded_gmm(
            x,
            w1,
            w1_scale,
            w1_bias,
            group_sizes,
            is_last_expert=False,
            mesh=mesh,
        )
        x1, x2 = jnp.split(x, 2, -1)

        x = activation_fn(activation, x1, x2)

        x = expert_sharded_gmm(
            x,
            w2,
            w2_scale,
            w2_bias,
            group_sizes,
            is_last_expert=True,
            mesh=mesh,
        )
    else:
        x1, x2 = tensor_sharded_gmm_merged_column_parallel(
            x,
            w1,
            w1_scale,
            w1_bias,
            group_sizes,
            mesh=mesh,
        )

        x = activation_fn(activation, x1, x2)

        x = tensor_sharded_gmm_row_parallel(
            x,
            w2,
            w2_scale,
            w2_bias,
            group_sizes,
            mesh=mesh,
        )

    def _finalize_output(x_local, topk_argsort_revert_indices_local,
                         topk_weights_local):
        x_local = x_local[topk_argsort_revert_indices_local].reshape(
            -1, topk, padded_hidden_size)
        x_local = x_local * jnp.expand_dims(topk_weights_local, axis=-1)
        x_local = x_local.sum(axis=-2)
        return x_local

    x = jax.shard_map(
        _finalize_output,
        mesh=mesh,
        in_specs=(P(ShardingAxisName.MLP_DATA,
                    None), P(ShardingAxisName.MLP_DATA),
                  P(ShardingAxisName.MLP_DATA, None)),
        out_specs=(P(ShardingAxisName.ATTN_DATA, None)),
        check_vma=False,
    )(x, topk_argsort_revert_indices, topk_weights)

    return x[:num_tokens, :hidden_size]
