import functools

import jax
from jax import numpy as jnp
from jax.experimental.pallas.ops.tpu.megablox.gmm import gmm
from jax.experimental.shard_map import shard_map
from jax.sharding import Mesh, NamedSharding, PartitionSpec

from tpu_inference.layers.vllm.linear_common import \
    slice_sharded_tensor_for_concatenation

P = PartitionSpec


def activation_fn(activation: str, x1, x2):
    match activation:
        case "silu":
            return jax.nn.silu(x1) * x2
        case "swigluoai":
            return _swigluoai(x1, x2)
        case _:
            raise NotImplementedError(
                f"FusedMoE does not support {activation} activation")


def _swigluoai(x1, x2, alpha=1.702, limit=7.0):
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
    rhs_bias: jax.Array | None,
    group_sizes: jax.Array,
    transpose_rhs: bool,
    mesh: Mesh,
    intermediate_size: int,
) -> jax.Array:
    # adapted from https://github.com/pytorch/xla/blob/1d409399474197c484894be90b75d9855393dda5/torch_xla/experimental/custom_kernel.py#L1401
    m, k, g = lhs.shape[0], lhs.shape[1], rhs.shape[0]
    n = rhs.shape[1] if transpose_rhs else rhs.shape[2]
    tm, tk, tn = _get_tiling_size_for_gmm_kernel(m, k, n, g)

    _gmm = functools.partial(
        gmm,
        preferred_element_type=lhs.dtype,
        tiling=(tm, tk, tn),
        transpose_rhs=transpose_rhs,
        group_offset=jnp.array(0),
    )

    gmm_result = shard_map(
        _gmm,
        mesh=mesh,
        in_specs=(P(), P(None, "model", None), P()),
        out_specs=(P(None, "model")),
        check_rep=False,
    )(lhs, rhs, group_sizes)

    if rhs_bias is not None:
        rhs_bis = jnp.repeat(rhs_bias, group_sizes, 0, total_repeat_length=m)
        gmm_result = (gmm_result + rhs_bis).astype(gmm_result.dtype)

    n_shards = mesh.shape["model"]
    output_sizes = [intermediate_size, intermediate_size]

    return slice_sharded_tensor_for_concatenation(gmm_result, output_sizes,
                                                  n_shards)


def tensor_sharded_gmm_row_parallel(
    lhs: jax.Array,
    rhs: jax.Array,
    rhs_bias: jax.Array | None,
    group_sizes: jax.Array,
    transpose_rhs: bool,
    mesh: Mesh,
) -> jax.Array:
    # adapted from https://github.com/pytorch/xla/blob/1d409399474197c484894be90b75d9855393dda5/torch_xla/experimental/custom_kernel.py#L1401
    m, k, g = lhs.shape[0], lhs.shape[1], rhs.shape[0]
    n = rhs.shape[1] if transpose_rhs else rhs.shape[2]
    tm, tk, tn = _get_tiling_size_for_gmm_kernel(m, k, n, g)

    _gmm = functools.partial(
        gmm,
        preferred_element_type=lhs.dtype,
        tiling=(tm, tk, tn),
        transpose_rhs=transpose_rhs,
        group_offset=jnp.array(0),
    )

    def _gmm_all_reduce(lhs, rhs, group_sizes):
        r = _gmm(lhs, rhs, group_sizes)
        return jax.lax.psum(r, axis_name="model")

    gmm_result = shard_map(
        _gmm_all_reduce,
        mesh=mesh,
        in_specs=(P(None, "model"), P(None, None, "model"), P()),
        out_specs=(P()),
        check_rep=False,
    )(lhs, rhs, group_sizes)

    if rhs_bias is not None:
        rhs_bias = jnp.repeat(rhs_bias, group_sizes, 0, total_repeat_length=m)
        gmm_result = (gmm_result + rhs_bias).astype(gmm_result.dtype)

    return gmm_result


def expert_sharded_gmm(
    lhs: jax.Array,
    rhs: jax.Array,
    group_sizes: jax.Array,
    transpose_rhs: bool,
    mesh: Mesh,
    num_experts: int,
    ep_size: int,
) -> jax.Array:
    # adapted from https://github.com/pytorch/xla/blob/1d409399474197c484894be90b75d9855393dda5/torch_xla/experimental/custom_kernel.py#L1401
    m, k, g = lhs.shape[0], lhs.shape[1], rhs.shape[0]
    n = rhs.shape[1] if transpose_rhs else rhs.shape[2]
    tm, tk, tn = _get_tiling_size_for_gmm_kernel(m, k, n, g)

    num_experts_per_shard = num_experts // ep_size
    group_offset = jnp.arange(0, num_experts, num_experts_per_shard)
    group_offset = jax.lax.with_sharding_constraint(
        group_offset, NamedSharding(mesh, P("model")))

    def _gmm(lhs, rhs, group_sizes, group_offset):
        # Group offset for this shard. `group_offset` is sharded, and in this
        # sharded function, it has only 1 element and `group_offset.shape` is
        # (1,) but gmm kernel requires the group_offset to be a ()-shaped array,
        # so we group_offset[0].
        group_offset_of_shard = group_offset[0]
        gmm_res = gmm(
            lhs=lhs,
            rhs=rhs,
            group_sizes=group_sizes,
            preferred_element_type=lhs.dtype,
            tiling=(tm, tk, tn),
            transpose_rhs=transpose_rhs,
            group_offset=group_offset_of_shard,
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
    gmm_res = shard_map(
        _gmm,
        mesh=mesh,
        in_specs=(P(), P("model", None, None), P(), P("model")),
        out_specs=(P("model", None)),
        check_rep=False,
    )(lhs, rhs, group_sizes, group_offset)

    # For i-th shard, it is responsible groups (AKA experts) from
    # i*num_experts_per_shard to (i+1)*num_experts_per_shard We sum them up to
    # get total rows in that shard, and that is the size for shard to send to
    # its peers. This is also the number of non-zero rows from the gmm results.
    # In the working example, send_sizes would be [3, 2, 5, 4]
    send_sizes = jnp.array([
        group_sizes[i * num_experts_per_shard:(i + 1) *
                    num_experts_per_shard].sum() for i in range(ep_size)
    ])
    # In the working example, input_offsets would be [0, 3, 5, 10]
    input_offsets = jnp.concatenate((jnp.array([0]), send_sizes.cumsum()[:-1]))
    output_offsets = input_offsets
    recv_sizes = send_sizes

    input_offsets = jax.lax.with_sharding_constraint(
        input_offsets, NamedSharding(mesh, P("model")))
    send_sizes = jax.lax.with_sharding_constraint(
        send_sizes, NamedSharding(mesh, P("model")))
    output_offsets = jax.lax.with_sharding_constraint(
        output_offsets, NamedSharding(mesh, P("model")))

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
                                         axis_name="model")

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
    return shard_map(
        _ragged_all_to_all,
        mesh=mesh,
        in_specs=(P("model", None), P("model"), P("model"), P("model"), P()),
        out_specs=(P()),
        check_rep=False,
    )(gmm_res, input_offsets, send_sizes, output_offsets, recv_sizes)


def fused_moe_func(
    hidden_states: jax.Array,
    w1: jax.Array,
    w2: jax.Array,
    w1_bias: jax.Array | None,
    w2_bias: jax.Array | None,
    gating_output: jax.Array,
    topk: int,
    global_num_experts: int,
    renormalize: bool,
    reduce_results: bool,
    mesh: Mesh,
    use_ep: bool,
    activation: str,
):
    """
    Args:
        hidden_states: [*, hidden_size]
        w1: [num_experts, intermediate_size * 2, hidden_size]
        w2: [num_experts, hidden_size, intermediate_size]
        gating_output: [*, num_experts]
    """
    # adapted from https://github.com/vllm-project/vllm/blob/29fa5cac1cd731026f59084d93a822921507573c/vllm/model_executor/layers/fused_moe/moe_pallas.py#L26
    if use_ep and (w1_bias is not None or w2_bias is not None):
        raise NotImplementedError(
            "Bias is not supported when using expert parallelism.")
    orig_shape = hidden_states.shape
    hidden_size = hidden_states.shape[-1]
    num_tokens = hidden_states.size // hidden_size
    assert global_num_experts == w1.shape[0]
    ep_size = mesh.shape["model"]  # only used if use_ep is True.
    intermediate_size = w2.shape[-1]
    dtype = hidden_states.dtype
    assert (num_tokens * topk) % 16 == 0, (
        "The kernel requires num_tokens * topk to be a multiple of "
        f"16 but got {num_tokens}*{topk}={num_tokens*topk}")

    hidden_states = hidden_states.reshape(num_tokens, hidden_size)
    gating_output = gating_output.reshape(num_tokens, global_num_experts)

    topk_weights = jax.nn.softmax(gating_output.astype(jnp.float32), axis=-1)
    topk_weights, topk_indices = jax.lax.top_k(topk_weights, k=topk)
    if renormalize:
        topk_weights = topk_weights / topk_weights.sum(axis=-1, keepdims=True)
    topk_weights = topk_weights.astype(dtype)

    topk_indices_flat = topk_indices.flatten()
    topk_argsort_indices = jnp.argsort(topk_indices_flat)
    topk_argsort_revert_indices = jnp.argsort(topk_argsort_indices)
    token_indices = jnp.arange(num_tokens, dtype=jnp.int32).repeat(topk)
    token_indices_sorted = token_indices[topk_argsort_indices]
    group_sizes = jnp.bincount(topk_indices_flat, length=global_num_experts)

    x = hidden_states[token_indices_sorted]

    if use_ep:
        x = expert_sharded_gmm(
            x,
            w1,
            group_sizes,
            transpose_rhs=True,
            mesh=mesh,
            num_experts=global_num_experts,
            ep_size=ep_size,
        )
        x1, x2 = x[..., :intermediate_size], x[..., intermediate_size:]
    else:
        x1, x2 = tensor_sharded_gmm_merged_column_parallel(
            x,
            w1,
            w1_bias,
            group_sizes,
            transpose_rhs=True,
            mesh=mesh,
            intermediate_size=intermediate_size,
        )

    x = activation_fn(activation, x1, x2)

    if use_ep:
        x = expert_sharded_gmm(
            x,
            w2,
            group_sizes,
            transpose_rhs=True,
            mesh=mesh,
            num_experts=global_num_experts,
            ep_size=ep_size,
        )
    else:
        x = jax.lax.with_sharding_constraint(
            x, NamedSharding(mesh, P(None, "model")))
        x = tensor_sharded_gmm_row_parallel(
            x,
            w2,
            w2_bias,
            group_sizes,
            transpose_rhs=True,
            mesh=mesh,
        )

    x = x[topk_argsort_revert_indices].reshape(-1, topk, hidden_size)
    x = x * jnp.expand_dims(topk_weights, axis=-1)
    x = x.sum(axis=-2)
    x = x.reshape(orig_shape)

    if reduce_results:
        x = jax.lax.with_sharding_constraint(x, NamedSharding(mesh, P()))
    return x


@functools.partial(
    jax.jit,
    static_argnames=(
        "topk",
        "global_num_experts",
        "renormalize",
        "reduce_results",
        "mesh",
        "use_ep",
        "activation",
    ),
)
def fused_moe_func_padded(
    hidden_states: jax.Array,
    w1: jax.Array,
    w2: jax.Array,
    w1_bias: jax.Array | None,
    w2_bias: jax.Array | None,
    gating_output: jax.Array,
    topk: int,
    global_num_experts: int,
    renormalize: bool,
    reduce_results: bool,
    mesh: Mesh,
    use_ep: bool,
    activation: str,
):
    # TODO(fanhongmin@google.com): Once the jax runner pads the input, we no longer need this.
    hidden_size = hidden_states.shape[-1]
    num_tokens = hidden_states.size // hidden_size
    if num_tokens * topk < 16:
        assert 16 % (num_tokens *
                     topk) == 0, f"Cannot pad to 16: {num_tokens=}, {topk=}"
        n_repeats = 16 // (num_tokens * topk)

        reps = (n_repeats, ) + (1, ) * (hidden_states.ndim - 1)
        expanded_hidden_states = jnp.tile(hidden_states, reps)

        reps = (n_repeats, ) + (1, ) * (gating_output.ndim - 1)
        expanded_gating_output = jnp.tile(gating_output, reps)

        expanded_x = fused_moe_func(
            expanded_hidden_states,
            w1,
            w2,
            w1_bias,
            w2_bias,
            expanded_gating_output,
            topk,
            global_num_experts,
            renormalize,
            reduce_results,
            mesh,
            use_ep,
            activation,
        )
        x = expanded_x[:hidden_states.shape[0]]
        return x
    else:
        return fused_moe_func(
            hidden_states,
            w1,
            w2,
            w1_bias,
            w2_bias,
            gating_output,
            topk,
            global_num_experts,
            renormalize,
            reduce_results,
            mesh,
            use_ep,
            activation,
        )
