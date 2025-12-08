import functools

import jax
from jax import numpy as jnp
from jax import shard_map
from jax.sharding import Mesh, PartitionSpec

from tpu_inference.kernels.megablox.gmm import gmm
from tpu_inference.layers.vllm.linear_common import \
    slice_sharded_tensor_for_concatenation

P = PartitionSpec


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

    rhs_scale_spec = None if rhs_scale is None else P(None, None, None,
                                                      "model")
    rhs_bias_spec = None if rhs_bias is None else P(None, None, "model")

    gmm_result = shard_map(
        _gmm,
        mesh=mesh,
        in_specs=(
            P("data", None),
            P(None, "model", None),
            rhs_scale_spec,
            rhs_bias_spec,
            P("data"),
        ),
        out_specs=(P("data", "model")),
        check_vma=False,
    )(lhs, rhs, rhs_scale, rhs_bias, group_sizes)

    tp_size = mesh.shape["model"]
    output_sizes = [gmm_result.shape[-1] // 2] * 2
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
        m, g, n, k = lhs.shape[0], *rhs.shape # k = 2560/8 = 320
        jax.debug.print("GMM shapes: lhs {lhs.shape}, rhs {rhs.shape}", lhs=lhs, rhs=rhs)
        tm, tk, tn = _get_tiling_size_for_gmm_kernel(m, k, n, g)
        shard_id = jax.lax.axis_index("model")
        if rhs_bias:
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
        return jax.lax.psum(out, axis_name="model")
    num_blocks = 1 if rhs_scale is None else rhs_scale.shape[1]
    rhs_scale_spec = None if rhs_scale is None or num_blocks == 1 else P(None, "model", None,
                                                      None)
    rhs_bias_spec = None if rhs_bias is None else P(None, None, None)
    gmm_result = shard_map(
        _gmm_all_reduce,
        mesh=mesh,
        in_specs=(
            P("data", "model"),
            P(None, None, "model"),
            rhs_scale_spec,
            rhs_bias_spec,
            P("data"),
        ),
        out_specs=(P("data")),
        check_vma=False,
    )(lhs, rhs, rhs_scale, rhs_bias, group_sizes)

    return gmm_result.astype(lhs.dtype)


def expert_sharded_gmm(
    lhs: jax.Array,
    rhs: jax.Array,
    group_sizes: jax.Array,
    mesh: Mesh,
) -> jax.Array:
    ep_size = mesh.shape["model"]

    num_experts = rhs.shape[0]
    num_experts_per_shard = num_experts // ep_size
    group_offset = jnp.arange(0, num_experts, num_experts_per_shard)

    def _gmm(lhs, rhs, group_sizes, group_offset):
        m, g, n, k = lhs.shape[0], *rhs.shape
        tm, tk, tn = _get_tiling_size_for_gmm_kernel(m, k, n, g)

        gmm_res = gmm(
            lhs=lhs,
            rhs=rhs,
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
    gmm_res = shard_map(
        _gmm,
        mesh=mesh,
        in_specs=(P(), P("model", None, None), P(), P("model")),
        out_specs=(P("model", None)),
        check_vma=False,
    )(lhs, rhs, group_sizes, group_offset)

    # For i-th shard, it is responsible groups (AKA experts) from
    # i*num_experts_per_shard to (i+1)*num_experts_per_shard We sum them up to
    # get total rows in that shard, and that is the size for shard to send to
    # its peers. This is also the number of non-zero rows from the gmm results.
    # In the working example, send_sizes would be [3, 2, 5, 4]
    send_sizes = group_sizes.reshape(-1, num_experts_per_shard).sum()
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
        return jax.lax.ragged_all_to_all(
            operand,
            output,
            input_offsets_of_shard,
            send_sizes_of_shard,
            output_offsets_of_shard,
            recv_sizes_of_shard,
            axis_name="model",
        )

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
    w13_weight: jax.Array,
    w2_weight: jax.Array,
    w13_weight_scale: jax.Array | None,
    w2_weight_scale: jax.Array | None,
    w13_bias: jax.Array | None,
    w2_bias: jax.Array | None,
    gating_output: jax.Array,
    topk: int,
    renormalize: bool,
    mesh: Mesh,
    use_ep: bool,
    activation: str,
) -> jax.Array:
    """Args:

  hidden_states: [num_tokens, hidden_size]
  w13_weight: [num_experts, intermediate_size * 2, hidden_size]
  w2_weight: [num_experts, hidden_size, intermediate_size]
  w13_scale: [num_experts, intermediate_size * 2, num_blocks]
  w2_scale: [num_experts, hidden_size, num_blocks] 
  w13_bias: [num_experts, intermediate_size * 2]
  w2_bias: [num_experts, hidden_size]
  gating_output: [num_tokens, num_experts]
  topk: int
  renormalize: bool
  mesh: Mesh
  use_ep: bool
  activation: str
  """
    if use_ep and (w13_bias is not None or w2_bias is not None):
        raise NotImplementedError(
            "Bias is not supported when using expert parallelism.")
    num_tokens, hidden_size = hidden_states.shape
    global_num_experts, _, padded_hidden_size = w13_weight.shape
    dtype = hidden_states.dtype

    assert (num_tokens * topk) % 16 == 0, (
        "The kernel requires num_tokens * topk to be a multiple of "
        f"16 but got {num_tokens}*{topk}={num_tokens*topk}")

    topk_weights = jax.nn.softmax(gating_output.astype(jnp.float32), axis=-1)
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

    x, group_sizes, topk_argsort_revert_indices = shard_map(
        _process_tokens_locally,
        mesh=mesh,
        in_specs=(P("data", None), P("data", None)),
        out_specs=(P("data", None), P("data"), P("data")),
    )(hidden_states, topk_indices)

    x = jnp.pad(x, ((0, 0), (0, padded_hidden_size - hidden_size)))

    if use_ep:
        x = expert_sharded_gmm(
            x,
            w13_weight,
            group_sizes,
            mesh=mesh,
        )
        x1, x2 = jnp.split(x, 2, -1)

        x = activation_fn(activation, x1, x2)

        x = expert_sharded_gmm(
            x,
            w2_weight,
            group_sizes,
            mesh=mesh,
        )
    else:
        x1, x2 = tensor_sharded_gmm_merged_column_parallel(
            x,
            w13_weight,
            w13_weight_scale,
            w13_bias,
            group_sizes,
            mesh=mesh,
        )

        x = activation_fn(activation, x1, x2)

        x = tensor_sharded_gmm_row_parallel(
            x,
            w2_weight,
            w2_weight_scale,
            w2_bias,
            group_sizes,
            mesh=mesh,
        )

    x = x[:, :hidden_size]

    def _finalize_output(x_local, topk_argsort_revert_indices_local,
                         topk_weights_local):
        x_local = x_local[topk_argsort_revert_indices_local].reshape(
            -1, topk, hidden_size)
        x_local = x_local * jnp.expand_dims(topk_weights_local, axis=-1)
        x_local = x_local.sum(axis=-2)
        return x_local

    return shard_map(
        _finalize_output,
        mesh=mesh,
        in_specs=(P("data", None), P("data"), P("data", None)),
        out_specs=(P("data", None)),
    )(x, topk_argsort_revert_indices, topk_weights)
