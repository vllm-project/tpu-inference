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
from typing import Literal

import jax
from jax import numpy as jnp
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from tpu_inference.kernels.megablox.gmm import gmm
from tpu_inference.kernels.megablox.gmm_v2 import (gmm_v2,
                                                   is_supported_by_gmm_v2)
from tpu_inference.layers.common.sharding import ShardingAxisName
from tpu_inference.utils import get_mesh_shape_product


def apply_scoring_fn(scoring_fn: str, x: jax.Array) -> jax.Array:
    match scoring_fn:
        case "softmax":
            return jax.nn.softmax(x, axis=-1)
        case "sigmoid":
            return jax.nn.sigmoid(x)
        case _:
            raise NotImplementedError(
                f"FusedMoE does not support {scoring_fn} scoring function")


def apply_act_fn(activation: str, x1: jax.Array, x2: jax.Array) -> jax.Array:
    match activation:
        case "silu":
            return jax.nn.silu(x1) * x2
        case "swigluoai":
            return _swigluoai(x1, x2)
        case _:
            raise NotImplementedError(
                f"FusedMoE does not support {activation} activation function")


def _swigluoai(x1: jax.Array,
               x2: jax.Array,
               alpha=1.702,
               limit=7.0) -> jax.Array:
    x1 = jnp.clip(x1, a_max=limit)
    x2 = jnp.clip(x2, a_min=-limit, a_max=limit)

    gated_activation = x1 * jax.nn.sigmoid(alpha * x1)

    return gated_activation * (x2 + 1)


# TODO (jacobplatin): make this more generic
def round_up_to_multiple_of_128_within_limit(x: int, limit: int) -> int:
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
    tm = round_up_to_multiple_of_128_within_limit(2 * m // g, 512)
    tm = min(tm, m)  # there's a requirement that m % tm == 0
    # k/n correspond to n_input_features/n_output_features in the matmul so they
    # are normally greater than 2048, unless the num shards is large.
    tk = round_up_to_multiple_of_128_within_limit(k, 2048)
    tn = round_up_to_multiple_of_128_within_limit(n, 2048)
    return tm, tk, tn


def gmm_wrapper(lhs, rhs, rhs_scale, rhs_bias, group_sizes, group_offset):
    if is_supported_by_gmm_v2(lhs, rhs, rhs_scale):
        gmm_res = gmm_v2(
            lhs=lhs,
            rhs=rhs,
            rhs_scale=rhs_scale,
            rhs_bias=rhs_bias,
            group_sizes=group_sizes,
            group_offset=group_offset[0],
        )
    else:
        m, g, k, n = lhs.shape[0], *rhs.shape
        tm, tk, tn = _get_tiling_size_for_gmm_kernel(m, k, n, g)

        gmm_res = gmm(
            lhs=lhs,
            rhs=rhs,
            rhs_scale=rhs_scale,
            rhs_bias=rhs_bias,
            group_sizes=group_sizes,
            preferred_element_type=lhs.dtype,
            tiling=(tm, tk, tn),
            group_offset=group_offset[0],
        )

    return gmm_res


def moe_gmm_local(
    x: jax.Array,
    w1: jax.Array,
    w1_scale: jax.Array | None,
    w1_bias: jax.Array | None,
    w2: jax.Array,
    w2_scale: jax.Array | None,
    w2_bias: jax.Array | None,
    group_sizes: jax.Array,
    group_offset: jax.Array,
    topk_argsort_revert_indices: jax.Array,
    topk_weights: jax.Array,
    *,
    activation: str,
    topk: int,
    parallelism: Literal["tp", "ep"],
) -> jax.Array:
    """ Main MoE logic on a local shard can run in TP or EP mode.
    
    Set parallelism for "tp" or "ep"
    """

    assert parallelism in ["tp", "ep"]

    # GMM1 computes x @ (W_up | W_gate) tegether and then split out to apply activation
    # to the gate result
    gmm1_res_gate_up = gmm_wrapper(x, w1, w1_scale, w1_bias, group_sizes,
                                   group_offset)
    gmm1_res_gate, gmm1_res_up = jnp.split(gmm1_res_gate_up, 2, -1)
    gmm1_res = apply_act_fn(activation, gmm1_res_gate, gmm1_res_up)

    # When the parallelism is TP since w2_bias is not sharded, we should only apply bias
    # once, not applying to every shard. So we set w2_bias to 0 to all shards other than
    # shard 0. For EP, it is not needed since bias is sharded on leading expert axis.
    if parallelism == "tp" and w2_bias is not None:
        shard_id = jax.lax.axis_index(ShardingAxisName.MLP_TENSOR).sum()
        w2_bias = jnp.where(shard_id == 0, w2_bias, 0)

    gmm2_res = gmm_wrapper(gmm1_res, w2, w2_scale, w2_bias, group_sizes,
                           group_offset)

    # First run local reduction on topk experts owned by the rank for all tokens
    token_topk_hidden = gmm2_res[topk_argsort_revert_indices].reshape(
        (-1, topk, gmm2_res.shape[-1]))
    token_topk_hidden = token_topk_hidden * jnp.expand_dims(topk_weights,
                                                            axis=-1)
    token_hidden = token_topk_hidden.sum(axis=-2)

    reduction_axis = ShardingAxisName.MLP_TENSOR if parallelism == "tp" else ShardingAxisName.EXPERT
    # Then global reduction on all ranks for all tokens and all experts
    return jax.lax.psum(token_hidden, axis_name=reduction_axis)


def tensor_parallel_gmm(
    x: jax.Array,
    w1: jax.Array,
    w1_scale: jax.Array | None,
    w1_bias: jax.Array | None,
    w2: jax.Array,
    w2_scale: jax.Array | None,
    w2_bias: jax.Array | None,
    group_sizes: jax.Array,
    topk_argsort_revert_indices: jax.Array,
    topk_weights: jax.Array,
    *,
    activation: str,
    topk: int,
    mesh: Mesh,
) -> jax.Array:
    data_p_spec = P(ShardingAxisName.MLP_DATA)
    group_offset = jnp.array([0])

    w1_spec = P(None, None, ShardingAxisName.MLP_TENSOR)
    w2_spec = P(None, ShardingAxisName.MLP_TENSOR, None)

    w1_scale_spec = None if w1_scale is None else P(
        None, None, None, ShardingAxisName.MLP_TENSOR)
    w1_bias_spec = None if w1_bias is None else P(None, None,
                                                  ShardingAxisName.MLP_TENSOR)

    num_blocks = 1 if w2_scale is None else w2_scale.shape[1]
    w2_scale_spec = None if num_blocks == 1 else P(
        None, ShardingAxisName.MLP_TENSOR, None, None)
    w2_bias_spec = None if w2_bias is None else P(None, None, None)

    return jax.shard_map(
        functools.partial(moe_gmm_local,
                          activation=activation,
                          topk=topk,
                          parallelism="tp"),
        mesh=mesh,
        in_specs=(data_p_spec, w1_spec, w1_scale_spec, w1_bias_spec, w2_spec,
                  w2_scale_spec, w2_bias_spec, data_p_spec, data_p_spec,
                  data_p_spec, data_p_spec),
        out_specs=(data_p_spec),
        check_vma=False,
    )(x, w1, w1_scale, w1_bias, w2, w2_scale, w2_bias, group_sizes,
      group_offset, topk_argsort_revert_indices, topk_weights)


def expert_parallel_gmm(
    x: jax.Array,
    w1: jax.Array,
    w1_scale: jax.Array | None,
    w1_bias: jax.Array | None,
    w2: jax.Array,
    w2_scale: jax.Array | None,
    w2_bias: jax.Array | None,
    group_sizes: jax.Array,
    topk_argsort_revert_indices: jax.Array,
    topk_weights: jax.Array,
    *,
    activation: str,
    topk: int,
    mesh: Mesh,
) -> jax.Array:
    ep_size = get_mesh_shape_product(mesh, ShardingAxisName.EXPERT)
    ep_p_spec = P(ShardingAxisName.EXPERT)
    data_p_spec = P(ShardingAxisName.MLP_DATA)
    num_experts = w1.shape[0]
    num_experts_per_shard = num_experts // ep_size
    group_offset = jnp.arange(0, num_experts, num_experts_per_shard)

    w1_scale_spec = None if w1_scale is None else ep_p_spec
    w1_bias_spec = None if w1_bias is None else ep_p_spec
    w2_scale_spec = None if w2_scale is None else ep_p_spec
    w2_bias_spec = None if w2_bias is None else ep_p_spec

    return jax.shard_map(
        functools.partial(moe_gmm_local,
                          activation=activation,
                          topk=topk,
                          parallelism="ep"),
        mesh=mesh,
        in_specs=(data_p_spec, ep_p_spec, w1_scale_spec, w1_bias_spec,
                  ep_p_spec, w2_scale_spec, w2_bias_spec, data_p_spec,
                  ep_p_spec, data_p_spec, data_p_spec),
        out_specs=(data_p_spec),
        check_vma=False,
    )(x, w1, w1_scale, w1_bias, w2, w2_scale, w2_bias, group_sizes,
      group_offset, topk_argsort_revert_indices, topk_weights)


@functools.partial(
    jax.jit,
    static_argnames=(
        "topk",
        "renormalize",
        "mesh",
        "use_ep",
        "activation",
        "scoring_fn",
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
    scoring_fn: str,
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
        scoring_fn: scoring function to apply on gating_output.

    Returns:
        Output of moe operation [num_tokens, hidden_size]
    """
    num_tokens, hidden_size = hidden_states.shape
    global_num_experts, padded_hidden_size, _ = w1.shape
    dtype = hidden_states.dtype

    assert (num_tokens * topk) % 16 == 0, (
        "The kernel requires num_tokens * topk to be a multiple of "
        f"16 but got {num_tokens}*{topk}={num_tokens*topk}")

    assert gating_output.shape == (num_tokens, global_num_experts)

    # hidden_states: [num_tokens, hidden_size]=[16, 6144]
    # scoring_fn='softmax'. gating_output: [num_tokens, global_num_experts]=[16, 160]
    # w1: [num_experts, intermediate_size * 2, hidden_size]=[160, 6144, 5120]
    # w2: [num_experts, hidden_size, intermediate_size]=[160, 2560, 6144]
    topk_weights = apply_scoring_fn(scoring_fn, gating_output)
    # topk_weights: [num_tokens, global_num_experts]=[16, 160]

    # All-gather topk weights for attention dp
    topk_weights = jax.lax.with_sharding_constraint(
        topk_weights, NamedSharding(mesh, P(ShardingAxisName.MLP_DATA, None)))
    # topk=8, topk_weights: [num_tokens, global_num_experts]=[16, 160]
    topk_weights, topk_indices = jax.lax.top_k(topk_weights, k=topk)
    # topk_weights: [num_tokens, topk]=[16, 8], topk_indices: [num_tokens, topk]=[16, 8]
    if renormalize:
        topk_weights = topk_weights / topk_weights.sum(axis=-1, keepdims=True)
    topk_weights = topk_weights.astype(dtype)

    def _process_tokens_locally(hidden_states_local, topk_indices_local):
        # GMM requires tokens grouped by expert. This _process_tokens_locally function does it.
        # hidden_states_local: [num_tokens_local, hidden_size]=[16, 6144]
        # topk_indices_local: [num_tokens_local, topk]=[16, 8]
        num_tokens_local = hidden_states_local.shape[0]

        # eg0, if num_tokens_local=2, topk=2, and topk_indices_local=[[2, 0], [1, 0]]
        # Token 0 --> experts [2, 0]
        # Token 1 --> experts [1, 0]
        # topk_indices_flat = [2, 0, 1, 0]  # flatten: 4 (token, expert) pairs
        topk_indices_flat = topk_indices_local.flatten()
        # topk_indices_flat: [num_tokens_local * topk]=[128]. eg0, topk_indices_flat=[2, 0, 1, 0]

        topk_argsort_indices = jnp.argsort(topk_indices_flat)
        # topk_argsort_indices: [num_tokens_local * topk]. eg0, topk_argsort_indices=[1, 3, 2, 0]
        # xw32: this is important. 
        # **topk_argsort_indices tells you which positions in the flattened token-expert list you need to pick,
        # in order, to group all tokens by expert.**
        # Walking through the example in the comments (2 tokens, topk=2):
        # 1. Start: topk_indices_local = [[2, 0], [1, 0]] — Token 0 goes to experts 2,0; Token 1 goes to
        # experts 1,0.
        # 2. Flatten: topk_indices_flat = [2, 0, 1, 0] — each position is a (token, expert) pair:
        # | Flat index | 0   | 1   | 2   | 3   |
        # |------------|-----|-----|-----|-----|
        # | Token      | 0   | 0   | 1   | 1   |
        # | Expert     | 2   | 0   | 1   | 0   |
        # 3. Argsort: topk_argsort_indices = [1, 3, 2, 0] — these are the flat indices that would sort
        # topk_indices_flat by expert ID in ascending order:
        #   - topk_indices_flat[1] = 0 (expert 0)
        #   - topk_indices_flat[3] = 0 (expert 0)
        #   - topk_indices_flat[2] = 1 (expert 1)
        #   - topk_indices_flat[0] = 2 (expert 2)
        # After this reordering, experts are grouped: [0, 0, 1, 2].
        # Why it matters: GMM (Group Matmul) requires tokens to be contiguous by expert — all tokens for expert
        #  0 together, then expert 1, etc. topk_argsort_indices is the permutation that achieves this grouping.
        #  It's used on line 467 "token_indices_sorted = token_indices[topk_argsort_indices]" to reorder token_indices and then on line 481 "x = hidden_states_local[token_indices_sorted]" to reorder the actual
        # hidden_states into expert-grouped order before feeding into GMM.

        topk_argsort_revert_indices = jnp.argsort(topk_argsort_indices)
        #  topk_argsort_revert_indices is the inverse permutation of topk_argsort_indices. It undoes the
        #  expert-grouping sort to restore the original token order.
        #
        #  Using the same example:
        #
        #  1. Before GMM, we sorted tokens by expert using topk_argsort_indices = [1, 3, 2, 0], producing:
        #
        #  | Sorted position | 0   | 1   | 2   | 3   |   ----> notice, it's **Sorted** position.
        #  |-----------------|-----|-----|-----|-----|
        #  | Expert          | 0   | 0   | 1   | 2   |
        #  | Token           | 0   | 1   | 1   | 0   |
        #
        #  2. GMM runs on this expert-grouped order and produces outputs in the same sorted order.
        #  3. After GMM, we need to put results back in original order (token 0's experts, then token 1's
        #  experts):
        #
        #  | Original position | 0   | 1   | 2   | 3   |
        #  |-------------------|-----|-----|-----|-----|
        #  | Expert            | 2   | 0   | 1   | 0   |
        #  | Token             | 0   | 0   | 1   | 1   |
        #
        #  4. topk_argsort_revert_indices = [3, 0, 2, 1] does exactly this. When you do
        #  out[topk_argsort_revert_indices]:
        #    - Position 0 gets out[3] (token 0, expert 2)
        #    - Position 1 gets out[0] (token 0, expert 0)
        #    - Position 2 gets out[2] (token 1, expert 1)
        #    - Position 3 gets out[1] (token 1, expert 0)
        #
        #  The key mathematical property: argsort(argsort(x)) gives the inverse permutation. So
        #  topk_argsort_revert_indices = argsort(topk_argsort_indices) is the permutation that reverses the
        #  sort.

        token_indices = jnp.arange(num_tokens_local,
                                   dtype=jnp.int32).repeat(topk)
        # token_indices maps each position in the flattened (token, expert) list back to its original token ID.
        #
        #  Using the same example (2 tokens, topk=2):
        #
        #  token_indices = jnp.arange(num_tokens_local).repeat(topk)
        #  # arange(2) = [0, 1]
        #  # repeat(2) = [0, 0, 1, 1]
        #
        #  Each token appears topk times because each token is routed to topk experts:
        #  ┌────────────┬─────┬─────┬─────┬─────┐
        #  │ Flat index │  0  │  1  │  2  │  3  │
        #  ├────────────┼─────┼─────┼─────┼─────┤
        #  │ Token ID   │ 0   │ 0   │ 1   │ 1   │
        #  ├────────────┼─────┼─────┼─────┼─────┤
        #  │ Expert     │ 2   │ 0   │ 1   │ 0   │
        #  └────────────┴─────┴─────┴─────┴─────┘

        # token_indices: [num_tokens_local * topk]. eg0, token_indices=[0, 0, 1, 1]
        token_indices_sorted = token_indices[topk_argsort_indices]
        # token_indices_sorted answers the question: after sorting by expert, which token is at each position?
        # It's computed on line 467:
        # token_indices_sorted = token_indices[topk_argsort_indices]
        # This takes token_indices (the token ID for each flat position) and reorders it using the
        # expert-sorting permutation.
        # Using the example:
        # - token_indices = [0, 0, 1, 1] — flat order, grouped by token
        # - topk_argsort_indices = [1, 3, 2, 0] — the permutation that sorts by expert
        # - token_indices_sorted = [0, 0, 1, 1][[1, 3, 2, 0]] = [0, 1, 1, 0]
        # ┌─────────────────┬─────┬─────┬─────┬─────┐
        # │ Sorted position │  0  │  1  │  2  │  3  │
        # ├─────────────────┼─────┼─────┼─────┼─────┤
        # │ Expert          │ 0   │ 0   │ 1   │ 2   │
        # ├─────────────────┼─────┼─────┼─────┼─────┤
        # │ Token ID        │ 0   │ 1   │ 1   │ 0   │
        # └─────────────────┴─────┴─────┴─────┴─────┘
        # Its sole purpose is line 481:
        # x = hidden_states_local[token_indices_sorted]

        # token_indices_sorted: [num_tokens_local * topk]. eg0, token_indices_sorted=[0, 1, 1, 0]
        group_sizes_local = jnp.bincount(topk_indices_flat,
                                         length=global_num_experts)
        #   group_sizes_local tells GMM how many tokens each expert needs to process.
        # 
        #   It's computed on line 478:
        # 
        #   group_sizes_local = jnp.bincount(topk_indices_flat, length=global_num_experts)
        # 
        #   bincount counts how many times each expert ID appears in the flattened token-expert assignments.
        # 
        #   Using the example:
        # 
        #   - topk_indices_flat = [2, 0, 1, 0]
        #   - global_num_experts = 3
        #   - group_sizes_local = [2, 1, 1] — expert 0 has 2 tokens, expert 1 has 1 token, expert 2 has 1 token

        # group_size_local: (global_num_experts,)
        # token_indices_sorted = [0, 1, 1, 0]
        x = hidden_states_local[token_indices_sorted]
        # x: [num_tokens_local * topk, hidden_size]
        # This uses token_indices_sorted as a gather index to build the expert-grouped input tensor for GMM. It
        #    fetches hidden_states[0], hidden_states[1], hidden_states[1], hidden_states[0] — placing each
        #   token's hidden state at the position where GMM expects it based on expert grouping.
        # IOW:
        # position 0 gets hidden_states_local[0]
        # position 1 gets hidden_states_local[1]
        # position 2 gets hidden_states_local[1]
        # position 3 gets hidden_states_local[0]

        return x, group_sizes_local, topk_argsort_revert_indices

    # hidden_states: [num_tokens, hidden_size]=[16, 6144], topk_indices: [num_tokens, topk]=[16, 8]
    x, group_sizes, topk_argsort_revert_indices = jax.shard_map(
        _process_tokens_locally,
        mesh=mesh,
        in_specs=(P(ShardingAxisName.MLP_DATA,
                    None), P(ShardingAxisName.MLP_DATA, None)),
        out_specs=(P(ShardingAxisName.MLP_DATA,
                     None), P(ShardingAxisName.MLP_DATA),
                   P(ShardingAxisName.MLP_DATA)))(hidden_states, topk_indices)

    x = jnp.pad(x, ((0, 0), (0, padded_hidden_size - hidden_size)))

    if use_ep:
        x = expert_parallel_gmm(x,
                                w1,
                                w1_scale,
                                w1_bias,
                                w2,
                                w2_scale,
                                w2_bias,
                                group_sizes,
                                topk_argsort_revert_indices,
                                topk_weights,
                                activation=activation,
                                topk=topk,
                                mesh=mesh)
    else:
        x = tensor_parallel_gmm(x,
                                w1,
                                w1_scale,
                                w1_bias,
                                w2,
                                w2_scale,
                                w2_bias,
                                group_sizes,
                                topk_argsort_revert_indices,
                                topk_weights,
                                activation=activation,
                                topk=topk,
                                mesh=mesh)

    return x[:num_tokens, :hidden_size]
