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

import functools
from typing import Literal

import jax
from jax import numpy as jnp
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from tokamax._src.ops.experimental.gmm_v2.gmm_v2 import gmm_v2

import tpu_inference.envs as envs
from tpu_inference.kernels.collectives.hierrs_sc import wrapper as hier_rs_sc
from tpu_inference.kernels.experimental.moe_dense_expert import (
    can_use_dense_expert, dense_expert_moe)
from tpu_inference.kernels.experimental.moe_lowtok import (can_use_moe_lowtok,
                                                           moe_lowtok)
from tpu_inference.kernels.experimental.moe_lowtok import \
    routing_table as moe_lowtok_routing_table
from tpu_inference.kernels.experimental.small_collectives import \
    small_all_reduce
from tpu_inference.kernels.experimental.small_collectives.common import LANES
from tpu_inference.kernels.router_topk import router_topk
from tpu_inference.kernels.sparse_core.dense_gather_reduce import \
    dense_gather_reduce
from tpu_inference.kernels.sparse_core.ragged_gather_reduce_v2 import \
    ragged_gather_reduce as ragged_gather_reduce_v2
from tpu_inference.kernels.sparse_core.ragged_gather_reduce_v3 import \
    ragged_gather_reduce as ragged_gather_reduce_v3
from tpu_inference.kernels.sparse_core.ragged_gather_v2 import \
    ragged_gather_v2 as ragged_gather
from tpu_inference.layers.common.quantization import quantize_tensor
from tpu_inference.layers.common.sharding import ShardingAxisName
from tpu_inference.logger import init_logger
from tpu_inference.utils import get_mesh_shape_product

logger = init_logger(__name__)

# Above 2,048 routed rows, the inverse-permutation scatter costs more than
# the two sorts it replaces.
_SINGLE_SORT_MAX_ROWS = 2048


def _single_sort_permutation(
        sort_keys: jax.Array, topk: int,
        flat_bits: int) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Returns the stable argsort of `sort_keys`, its inverse, and token ids.

    Packing (key, flat position) into one int32 makes a single unstable sort
    order entries like a stable argsort. The inverse is a scatter, and flat
    position p belongs to token p // topk.
    """
    pos = jnp.arange(sort_keys.shape[0], dtype=jnp.int32)
    packed = jax.lax.sort(jnp.left_shift(sort_keys, flat_bits) | pos,
                          is_stable=False)
    sorted_indices = jnp.bitwise_and(packed, (1 << flat_bits) - 1)
    revert = jnp.zeros_like(pos).at[sorted_indices].set(pos,
                                                        unique_indices=True)
    return sorted_indices, revert, sorted_indices // topk


def _can_use_moe_lowtok_ep(hidden, w1, w2, s1, s2, b1, b2, *, activation,
                           rhs_quant_dtype, local_num_experts):
    if activation != "silu" or rhs_quant_dtype is not None:
        return False
    if b1 is not None or b2 is not None or s1 is None or s2 is None:
        return False
    if hidden.ndim != 2 or w1.ndim != 3 or w2.ndim != 3:
        return False
    h = hidden.shape[1]
    _, i, out = w2.shape
    e = local_num_experts
    if not 0 < e <= 128 or out != h or w1.shape[1] != h or w1.shape[2] != 2 * i:
        return False
    if (2 * i) % 512:
        return False
    if (hidden.dtype != jnp.bfloat16 or w1.dtype != jnp.float8_e4m3fn
            or w2.dtype != jnp.float8_e4m3fn):
        return False
    return s1.shape[-1] == 2 * i and s2.shape[-1] == h


def _can_use_dense_expert_ep(
    hidden,
    w1,
    w2,
    s1,
    s2,
    b1,
    b2,
    *,
    activation,
    rhs_quant_dtype,
    skip_padded_tokens=False,
    local_num_experts: int,
):
    if activation != "silu":
        return False
    if rhs_quant_dtype not in (None, jnp.float8_e4m3fn):
        return False
    if b1 is not None or b2 is not None or s1 is None or s2 is None:
        return False
    if hidden.ndim != 2 or w1.ndim != 3 or w2.ndim != 3:
        return False
    m, h = hidden.shape
    _, i, out = w2.shape
    e = local_num_experts
    if min(m, h, i, e) <= 0:
        return False
    if m % 8 or h % 512 or i % 512:
        return False
    if out != h or w1.shape[1] != h or w1.shape[2] != 2 * i:
        return False
    if (hidden.dtype != jnp.bfloat16 or w1.dtype != jnp.float8_e4m3fn
            or w2.dtype != jnp.float8_e4m3fn):
        return False
    return (s1.shape[-1] == 2 * i and s2.shape[-1] == h
            and s1.dtype == jnp.float32 and s2.dtype == jnp.float32)


def _ep_kernel_local(
    hidden_local,
    w1_local,
    w2_local,
    s1_local,
    s2_local,
    topk_weights_local,
    topk_ids_local,
    *,
    kernel_mode: str,
    local_num_experts: int,
    scatter_results: bool,
):
    shard_idx = jax.lax.axis_index(ShardingAxisName.EXPERT)
    experts_start = shard_idx * local_num_experts
    local_ids = topk_ids_local - experts_start
    valid = (local_ids >= 0) & (local_ids < local_num_experts)
    local_weights = jnp.where(valid, topk_weights_local, 0.0)
    local_ids = jnp.where(valid, local_ids, -1)

    if kernel_mode == "lowtok":
        route = moe_lowtok_routing_table(local_ids, local_weights,
                                         local_num_experts)
        local_out = moe_lowtok(hidden_local, route, w1_local, s1_local,
                               w2_local, s2_local)
    else:
        local_out = dense_expert_moe(
            hidden_local,
            w1_local,
            w2_local,
            s1_local,
            s2_local,
            local_weights,
            local_ids,
            num_primary_experts=local_num_experts,
        )

    if scatter_results:
        dp_axes = ShardingAxisName.ATTN_DATA
        if not isinstance(dp_axes, tuple):
            dp_axes = (dp_axes, )
        reduction_axis = ShardingAxisName.EXPERT
        if isinstance(reduction_axis, tuple):
            reduce_axes = tuple(a for a in reduction_axis if a not in dp_axes)
            scatter_axes = tuple(a for a in reduction_axis if a in dp_axes)
        else:
            reduce_axes = () if reduction_axis in dp_axes else (
                reduction_axis, )
            scatter_axes = (
                reduction_axis, ) if reduction_axis in dp_axes else ()
        if reduce_axes:
            local_out = jax.lax.psum(local_out, axis_name=reduce_axes)
        if scatter_axes:
            out = jax.lax.psum_scatter(
                local_out,
                axis_name=scatter_axes,
                scatter_dimension=0,
                tiled=True,
            ).astype(hidden_local.dtype)
        else:
            out = local_out
    else:
        out = jax.lax.psum(local_out,
                           axis_name=ShardingAxisName.EXPERT).astype(
                               hidden_local.dtype)
    return out


def _ep_dispatch_kernel(
    hidden_states,
    w1,
    w2,
    w1_scale,
    w2_scale,
    topk_weights,
    topk_indices,
    *,
    mesh: Mesh,
    kernel_mode: str,
    local_num_experts: int,
    scatter_results: bool,
):
    ep_p_spec = P(ShardingAxisName.EXPERT)
    data_p_spec = P(ShardingAxisName.MLP_DATA)
    attn_data_p_spec = P(ShardingAxisName.ATTN_DATA)
    w1_scale_spec = None if w1_scale is None else ep_p_spec
    w2_scale_spec = None if w2_scale is None else ep_p_spec

    final_out_specs = attn_data_p_spec if scatter_results else data_p_spec

    return jax.shard_map(
        functools.partial(
            _ep_kernel_local,
            kernel_mode=kernel_mode,
            local_num_experts=local_num_experts,
            scatter_results=scatter_results,
        ),
        mesh=mesh,
        in_specs=(
            data_p_spec,
            ep_p_spec,
            ep_p_spec,
            w1_scale_spec,
            w2_scale_spec,
            data_p_spec,
            data_p_spec,
        ),
        out_specs=final_out_specs,
        check_vma=False,
    )(
        hidden_states,
        w1,
        w2,
        w1_scale,
        w2_scale,
        topk_weights,
        topk_indices,
    )


def _select_ragged_gather_reduce(version: str | None = None):
    v = (version or envs.RAGGED_GATHER_REDUCE_VERSION or "v3").lower()
    if v == "v3":
        return ragged_gather_reduce_v3
    return ragged_gather_reduce_v2


# Target chunk size of 2048 slots was found empirically to be optimal
# for MoE workloads (e.g., Qwen) to hide ICI/DMA latency during AllReduce.
TARGET_SLOT_CHUNK_SIZE = 2048


def _override_token_indices_for_random_routing(
        topk_indices: jax.Array, global_num_experts: int) -> jax.Array:
    logger.warning(
        "Forcing random routing should be used for performance testing only.")
    original_topk_indices = topk_indices
    num_tokens, topk = original_topk_indices.shape
    # Forcing random routing is useful to get rid of the effect
    # of routing imbalance during performance debugging.
    # (original_topk_indices // global_num_experts) is just zero, but we keep it so that
    # the all-gather of topk_indices won't be skipped so that the performance comparison between
    # with and without random routing is fair.
    rng_key = jax.random.PRNGKey(42)
    topk_indices = jax.vmap(lambda key: jax.random.choice(
        key, global_num_experts, shape=(topk, ), replace=False))(
            jax.random.split(rng_key, num_tokens)) + (original_topk_indices //
                                                      global_num_experts)
    return topk_indices


def all_gather_topk_indices_and_weights(
        topk_indices: jax.Array, topk_weights: jax.Array, dtype: jnp.dtype,
        mesh: Mesh) -> tuple[jax.Array, jax.Array]:
    # `topk_indices` and `topk_weights` are relatively small (and last dimension is top-k),
    # directly all-gather them is inefficient. We use reshape, bitcast to convert the data into one array,
    #  all gather, then unpack.
    top_k = topk_indices.shape[-1]
    topk_indices = topk_indices.astype(jnp.int32).reshape(-1)
    topk_weights = topk_weights.astype(jnp.float32).reshape(-1)
    topk_weights = jax.lax.bitcast_convert_type(topk_weights,
                                                topk_indices.dtype)

    blob = jnp.stack([topk_indices, topk_weights])
    # The optimization barrier here is to prevent the compiler from reordering the all-gather the operations above.
    blob = jax.lax.optimization_barrier(blob)
    gathered_blob = jax.lax.with_sharding_constraint(
        blob, NamedSharding(mesh, P(None, ShardingAxisName.MLP_DATA)))

    topk_indices = gathered_blob[0]
    topk_weights = gathered_blob[1]
    topk_indices = topk_indices.reshape(-1, top_k)
    topk_weights = jax.lax.bitcast_convert_type(topk_weights, jnp.float32)
    topk_weights = topk_weights.reshape(-1, top_k).astype(jnp.float32)

    return topk_indices, topk_weights


def apply_scoring_fn(scoring_fn: str, x: jax.Array) -> jax.Array:
    match scoring_fn:
        case "softmax":
            return jax.nn.softmax(x, axis=-1)
        case "sigmoid":
            return jax.nn.sigmoid(x)
        case "sqrtsoftplus":
            return jnp.sqrt(jax.nn.softplus(x))
        case _:
            raise NotImplementedError(
                f"FusedMoE does not support {scoring_fn} scoring function")


def gmm_wrapper(lhs,
                rhs,
                rhs_scale,
                rhs_bias,
                group_sizes,
                group_offset,
                fuse_act=None,
                preferred_element_type=None):
    gmm_res = gmm_v2(
        lhs=lhs,
        rhs=rhs,
        rhs_scale=rhs_scale,
        rhs_bias=rhs_bias,
        group_sizes=group_sizes,
        group_offset=group_offset[0],
        zero_initialize=False,
        fuse_act=fuse_act,
        preferred_element_type=preferred_element_type,
    )
    return gmm_res


def valid_rows_mask(batch_size: int, group_sizes: jax.Array,
                    group_start: jax.Array, group_end: jax.Array) -> jax.Array:
    """Mask indicating rows processed by current shard."""

    group_sizes_sum = jnp.cumulative_sum(group_sizes, include_initial=True)

    token_start = group_sizes_sum[group_start]
    token_end = group_sizes_sum[group_end]

    index = jnp.arange(batch_size)
    return jnp.where(jnp.logical_and(token_start <= index, index < token_end),
                     True, False)


def _permute_tokens_for_chunked_rs(x: jax.Array, dp_size: int,
                                   num_chunks: int) -> jax.Array:
    """Permutes tokens to correct for chunked reduce-scatter.

    This reordering ensures that when we slice the tokens into chunks and run
    reduce-scatter on each chunk locally, the resulting scattered outputs
    end up in the correct non-interleaved global sequence order after concatenation.
    """
    num_tokens = x.shape[0]
    chunk_size_per_device = num_tokens // (num_chunks * dp_size)

    # Reshape to split the token dimension into (dp_size, num_chunks, chunk_size_per_device)
    new_shape = (dp_size, num_chunks, chunk_size_per_device) + x.shape[1:]
    x_reshaped = x.reshape(new_shape)

    # Transpose the dp_size and num_chunks axes (axes 0 and 1)
    transpose_axes = (1, 0, 2) + tuple(range(3, x_reshaped.ndim))
    x_transposed = jnp.transpose(x_reshaped, transpose_axes)

    # Flatten the first 3 dimensions back to num_tokens
    return x_transposed.reshape(x.shape)


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
        enable_rs_kernel: bool = False,
        onehot_moe_permute_threshold: int = 0,
        scatter_results: bool = False,
        moe_chunk_size: int = 0,
        defer_all_reduce: bool = False,
        ragged_gather_reduce_version: str | None = None) -> jax.Array:
    """Main MoE logic on a local shard can run in TP or EP mode.

    Set parallelism for "tp" or "ep"
    """

    assert parallelism in ["tp", "ep"]

    # GMM1 computes x @ (W_up | W_gate) together and activation, output is [tokens,padded_intermediate_size]
    gmm1_res = gmm_wrapper(
        x,
        w1,
        w1_scale,
        w1_bias,
        group_sizes,
        group_offset,
        fuse_act=activation,
        preferred_element_type=x.dtype,
    )

    # When the parallelism is TP since w2_bias is not sharded, we should only apply bias
    # once, not applying to every shard. So we set w2_bias to 0 to all shards other than
    # shard 0. For EP, it is not needed since bias is sharded on leading expert axis.
    if parallelism == "tp" and w2_bias is not None:
        shard_id = jax.lax.axis_index(ShardingAxisName.MLP_TENSOR).sum()
        w2_bias = jnp.where(shard_id == 0, w2_bias, 0)
    gmm1_res = gmm1_res[:, :w2.shape[1]]  # trim to hidden size if padded
    gmm2_res = gmm_wrapper(gmm1_res, w2, w2_scale, w2_bias, group_sizes,
                           group_offset)

    batch_size = gmm2_res.shape[0]
    local_group_size = w1.shape[0]

    reduction_axis = (ShardingAxisName.MLP_TENSOR
                      if parallelism == "tp" else ShardingAxisName.EXPERT)

    scatter_axis_size = 1
    reduce_axes = ()
    scatter_axes = ()

    if enable_rs_kernel:
        reduction_axes = reduction_axis if isinstance(
            reduction_axis, tuple) else (reduction_axis, )
        for axis in reduction_axes:
            scatter_axis_size *= jax.lax.axis_size(axis)
    elif scatter_results:
        dp_axes = ShardingAxisName.ATTN_DATA
        if not isinstance(dp_axes, tuple):
            dp_axes = (dp_axes, )
        if isinstance(reduction_axis, tuple):
            reduce_axes = tuple(a for a in reduction_axis if a not in dp_axes)
            scatter_axes = tuple(a for a in reduction_axis if a in dp_axes)
        else:
            reduce_axes = () if reduction_axis in dp_axes else (
                reduction_axis, )
            scatter_axes = (
                reduction_axis, ) if reduction_axis in dp_axes else ()
        for a in scatter_axes:
            scatter_axis_size *= jax.lax.axis_size(a)

    # Chunk Size Resolution
    num_tokens = batch_size // topk
    is_onehot = (local_group_size < group_sizes.size) and (
        onehot_moe_permute_threshold > 0
        and batch_size <= onehot_moe_permute_threshold)
    if moe_chunk_size > 0 and num_tokens > moe_chunk_size * scatter_axis_size and not is_onehot:
        actual_chunk_size = moe_chunk_size * scatter_axis_size
    else:
        actual_chunk_size = num_tokens

    is_pipelined = actual_chunk_size < num_tokens
    if is_pipelined and scatter_axis_size > 1:
        num_chunks = num_tokens // actual_chunk_size
        topk_weights = _permute_tokens_for_chunked_rs(topk_weights,
                                                      scatter_axis_size,
                                                      num_chunks)

        revert_indices_2d = topk_argsort_revert_indices.reshape(
            num_tokens, topk)
        revert_indices_2d = _permute_tokens_for_chunked_rs(
            revert_indices_2d, scatter_axis_size, num_chunks)
        topk_argsort_revert_indices = revert_indices_2d.flatten()

    if local_group_size < group_sizes.size:
        row_mask = valid_rows_mask(
            gmm1_res.shape[0],
            group_sizes,
            group_offset,
            group_offset + local_group_size,
        )
        if is_onehot:
            # gmm_v2 runs with zero_initialize=False, leaving rows outside this
            # EP shard's [token_start, token_end) range uninitialized in HBM.
            # Zero them out before `combine @ gmm2_res` so IEEE-754 `0.0 * NaN`
            # (or `0.0 * Inf`) from stale HBM bits cannot corrupt valid rows.
            gmm2_res = jnp.where(row_mask[:, None], gmm2_res, 0)
        mask = row_mask[topk_argsort_revert_indices].reshape(-1, topk, 1)
    else:
        mask = jnp.full((batch_size, ), True).reshape(-1, topk, 1)

    out_list = []

    # Pipelining Loop
    for start_tok in range(0, num_tokens, actual_chunk_size):
        end_tok = min(num_tokens, start_tok + actual_chunk_size)
        start_idx = start_tok * topk
        end_idx = end_tok * topk

        cur_indices = topk_argsort_revert_indices[start_idx:end_idx]
        cur_weights = topk_weights[start_tok:end_tok]
        cur_mask = mask[start_tok:end_tok]

        if local_group_size < group_sizes.size:
            if onehot_moe_permute_threshold > 0 and batch_size <= onehot_moe_permute_threshold:
                revert_indices = cur_indices.reshape(-1, topk)
                onehot = jax.nn.one_hot(revert_indices,
                                        batch_size,
                                        dtype=gmm2_res.dtype)
                combine = (onehot * cur_weights[..., None] *
                           cur_mask).sum(axis=1)
                chunk_hidden = combine @ gmm2_res
            else:
                rgr_fn = _select_ragged_gather_reduce(
                    ragged_gather_reduce_version)
                chunk_hidden = rgr_fn(gmm2_res, cur_indices,
                                      cur_weights.reshape(-1),
                                      cur_mask.reshape(-1), topk)
        else:
            chunk_hidden = dense_gather_reduce(
                gmm2_res,
                topk_argsort_revert_indices,
                topk_weights,
                topk,
            )
        if enable_rs_kernel:
            rs_out = hier_rs_sc.hierarchical_reduce_scatter_local(
                chunk_hidden,
                num_devices=scatter_axis_size,
                axis_name=reduction_axis)
            out = rs_out.astype(x.dtype)
        elif scatter_results:
            if reduce_axes:
                chunk_hidden = jax.lax.psum(chunk_hidden,
                                            axis_name=reduce_axes)
            if scatter_axes:
                out = jax.lax.psum_scatter(chunk_hidden,
                                           axis_name=scatter_axes,
                                           scatter_dimension=0,
                                           tiled=True).astype(x.dtype)
            else:
                out = chunk_hidden.astype(x.dtype)
        else:
            if not defer_all_reduce:
                can_use_small_ar = False
                if envs.TPU_SMALL_ALL_REDUCE and jax.devices(
                )[0].platform == "tpu":
                    try:
                        reduction_axes = (reduction_axis, ) if isinstance(
                            reduction_axis, str) else tuple(reduction_axis)
                        num_cores = 1
                        for a in reduction_axes:
                            num_cores *= jax.lax.axis_size(a)
                        if (chunk_hidden.ndim == 2
                                and chunk_hidden.dtype == jnp.bfloat16
                                and chunk_hidden.shape[0]
                                <= envs.TPU_SMALL_ALL_REDUCE_MAX_ROWS
                                and num_cores > 1 and chunk_hidden.shape[1] %
                            (num_cores * LANES) == 0):
                            can_use_small_ar = True
                    except Exception:
                        can_use_small_ar = False

                if can_use_small_ar:
                    out = small_all_reduce(
                        chunk_hidden,
                        num_cores=num_cores,
                        axis=reduction_axis,
                    ).astype(x.dtype)
                else:
                    out = jax.lax.psum(
                        chunk_hidden, axis_name=reduction_axis).astype(x.dtype)
            else:
                out = chunk_hidden.astype(x.dtype)
        out_list.append(out)

    return jnp.concatenate(out_list,
                           axis=0) if len(out_list) > 1 else out_list[0]


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
    enable_rs_kernel: bool = False,
    onehot_moe_permute_threshold: int = 0,
    scatter_results: bool = False,
    moe_chunk_size: int = 0,
    defer_all_reduce: bool = False,
    ragged_gather_reduce_version: str | None = None,
) -> jax.Array:
    data_p_spec = P(ShardingAxisName.MLP_DATA)
    attn_data_p_spec = P(ShardingAxisName.ATTN_DATA)
    group_offset = jnp.array([0])

    w1_spec = P(None, None, ShardingAxisName.MLP_TENSOR)
    w2_spec = P(None, ShardingAxisName.MLP_TENSOR, None)

    w1_scale_spec = (None if w1_scale is None else P(
        None, None, None, ShardingAxisName.MLP_TENSOR))
    w1_bias_spec = (None if w1_bias is None else P(
        None, None, ShardingAxisName.MLP_TENSOR))

    num_blocks = 1 if w2_scale is None else w2_scale.shape[1]
    w2_scale_spec = (None if num_blocks == 1 else P(
        None, ShardingAxisName.MLP_TENSOR, None, None))
    w2_bias_spec = None if w2_bias is None else P(None, None, None)

    if scatter_results:
        final_out_specs = attn_data_p_spec
    else:
        final_out_specs = data_p_spec

    return jax.shard_map(
        functools.partial(
            moe_gmm_local,
            activation=activation,
            topk=topk,
            parallelism="tp",
            enable_rs_kernel=False,
            onehot_moe_permute_threshold=onehot_moe_permute_threshold,
            scatter_results=scatter_results,
            moe_chunk_size=moe_chunk_size,
            defer_all_reduce=defer_all_reduce,
            ragged_gather_reduce_version=ragged_gather_reduce_version,
        ),
        mesh=mesh,
        in_specs=(
            data_p_spec,
            w1_spec,
            w1_scale_spec,
            w1_bias_spec,
            w2_spec,
            w2_scale_spec,
            w2_bias_spec,
            data_p_spec,
            P(),
            data_p_spec,
            data_p_spec,
        ),
        out_specs=(final_out_specs),
        check_vma=False,
    )(
        x,
        w1,
        w1_scale,
        w1_bias,
        w2,
        w2_scale,
        w2_bias,
        group_sizes,
        group_offset,
        topk_argsort_revert_indices,
        topk_weights,
    )


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
    enable_rs_kernel: bool = False,
    onehot_moe_permute_threshold: int = 0,
    moe_chunk_size: int = 0,
    scatter_results: bool = False,
    defer_all_reduce: bool = False,
    ragged_gather_reduce_version: str | None = None,
) -> jax.Array:
    ep_size = get_mesh_shape_product(mesh, ShardingAxisName.EXPERT)
    ep_p_spec = P(ShardingAxisName.EXPERT)
    data_p_spec = P(ShardingAxisName.MLP_DATA)
    ep_data_p_spec = P(ShardingAxisName.EXPERT_DATA)
    attn_data_p_spec = P(ShardingAxisName.ATTN_DATA)
    num_experts = w1.shape[0]
    num_experts_per_shard = num_experts // ep_size
    group_offset = jnp.arange(0, num_experts, num_experts_per_shard)

    w1_scale_spec = None if w1_scale is None else ep_p_spec
    w1_bias_spec = None if w1_bias is None else ep_p_spec
    w2_scale_spec = None if w2_scale is None else ep_p_spec
    w2_bias_spec = None if w2_bias is None else ep_p_spec

    if scatter_results:
        final_out_specs = attn_data_p_spec
    elif enable_rs_kernel:
        final_out_specs = ep_data_p_spec
    else:
        final_out_specs = data_p_spec

    return jax.shard_map(
        functools.partial(
            moe_gmm_local,
            activation=activation,
            topk=topk,
            parallelism="ep",
            onehot_moe_permute_threshold=onehot_moe_permute_threshold,
            enable_rs_kernel=enable_rs_kernel,
            scatter_results=scatter_results,
            moe_chunk_size=moe_chunk_size,
            defer_all_reduce=defer_all_reduce,
            ragged_gather_reduce_version=ragged_gather_reduce_version,
        ),
        mesh=mesh,
        in_specs=(
            data_p_spec,
            ep_p_spec,
            w1_scale_spec,
            w1_bias_spec,
            ep_p_spec,
            w2_scale_spec,
            w2_bias_spec,
            data_p_spec,
            ep_p_spec,
            data_p_spec,
            data_p_spec,
        ),
        out_specs=(final_out_specs),
        check_vma=False,
    )(
        x,
        w1,
        w1_scale,
        w1_bias,
        w2,
        w2_scale,
        w2_bias,
        group_sizes,
        group_offset,
        topk_argsort_revert_indices,
        topk_weights,
    )


def _apply_all_gather_fp8(hidden_states: jax.Array, mesh: Mesh,
                          dtype: jnp.dtype) -> jax.Array:
    logger.info("Apply FP8 all-gather on input of MOE")
    hidden_states_q, scale = quantize_tensor(
        jnp.float8_e4m3fn,
        hidden_states,
        axis=-1,
    )
    # quantize_tensor squeezes the scale if axis is int. We need to expand it back.
    scale = jnp.expand_dims(scale, -1)

    # Dequantize if needed
    return jax.shard_map(
        lambda x, s: (x.astype(jnp.float32) * s).astype(dtype),
        mesh=mesh,
        in_specs=(
            P(ShardingAxisName.MLP_DATA, None),
            P(ShardingAxisName.MLP_DATA, None),
        ),
        out_specs=P(ShardingAxisName.MLP_DATA, None),
    )(hidden_states_q, scale)


@jax.jit(static_argnames=(
    "topk",
    "renormalize",
    "mesh",
    "use_ep",
    "activation",
    "scoring_fn",
    "all_gather_fp8",
    "enable_rs_kernel",
    "use_gmm_fused_rs_kernel",
    "onehot_moe_permute_threshold",
    "scatter_results",
    "moe_chunk_size",
    "defer_all_reduce",
    "ragged_gather_reduce_version",
    "rhs_quant_dtype",
    "skip_padded_tokens",
    "num_primary_experts",
    "use_sparse_core",
))
def fused_moe_func(
    hidden_states: jax.Array,
    w1: jax.Array,
    w2: jax.Array,
    w1_scale: jax.Array | None,
    w2_scale: jax.Array | None,
    w1_bias: jax.Array | None,
    w2_bias: jax.Array | None,
    gating_output: jax.Array | None = None,
    topk_ids: jax.Array | None = None,
    experts_start: jax.Array | None = None,
    topk: int = 1,
    renormalize: bool = True,
    mesh: Mesh | None = None,
    use_ep: bool = False,
    activation: str = "silu",
    scoring_fn: str = "softmax",
    all_gather_fp8: bool = False,
    enable_rs_kernel: bool = False,
    use_gmm_fused_rs_kernel: bool = False,
    onehot_moe_permute_threshold: int = 0,
    scatter_results: bool = False,
    defer_all_reduce: bool = False,
    hash_based_topk_indices: jax.Array | None = None,
    expert_score_correction_bias: jax.Array | None = None,
    moe_chunk_size: int = 0,
    num_valid_tokens: jax.Array | None = None,
    ragged_gather_reduce_version: str | None = None,
    *,
    topk_weights: jax.Array | None = None,
    rhs_quant_dtype: jnp.dtype | None = None,
    skip_padded_tokens: bool = False,
    num_primary_experts: int | None = None,
    use_sparse_core: bool = True,
) -> jax.Array:
    """Route tokens in hidden_states into each experts based on routing.

    Args:
        hidden_states: [num_tokens, hidden_size]
        w1: first moe weights [num_experts, hidden_size, intermediate_size * 2]
        w2: second moe weights [num_experts, intermediate_size, hidden_size]
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
        enable_rs_kernel: enable custom Hierarchical Reduce-Scatter kernel.
        use_gmm_fused_rs_kernel: enable fused GMM reduce-scatter kernel.

    Returns:
        Output of moe operation [num_tokens, hidden_size]
    """

    if use_ep and use_gmm_fused_rs_kernel:
        from tpu_inference.kernels.experimental.fused_moe.fused_moe_rs import \
            fused_moe_func_rs
        logger.info("fused_moe_rs kernel in use")
        return fused_moe_func_rs(hidden_states=hidden_states,
                                 w1=w1,
                                 w2=w2,
                                 w1_scale=w1_scale,
                                 w2_scale=w2_scale,
                                 w1_bias=w1_bias,
                                 w2_bias=w2_bias,
                                 gating_output=gating_output,
                                 topk=topk,
                                 renormalize=renormalize,
                                 mesh=mesh,
                                 activation=activation,
                                 scoring_fn=scoring_fn,
                                 fp8_post_gather=all_gather_fp8)

    # Convert 3D scale [experts, blocks, N] to 4D [experts, blocks, 1, N]
    # expected by gmm_v2 and dense_expert_moe.
    if w1_scale is not None and w1_scale.ndim == 3:
        w1_scale = jnp.expand_dims(w1_scale, 2)
    if w2_scale is not None and w2_scale.ndim == 3:
        w2_scale = jnp.expand_dims(w2_scale, 2)

    num_tokens, hidden_size = hidden_states.shape
    global_num_experts, padded_hidden_size, _ = w1.shape
    dtype = hidden_states.dtype

    if topk_ids is not None:
        topk_indices = topk_ids
        if topk_weights is None:
            topk_weights = gating_output
        gating_output = None

    if experts_start is not None:
        owned = w1.shape[
            0] if num_primary_experts is None else num_primary_experts
        if not 0 < owned <= w1.shape[0]:
            raise ValueError(
                "num_primary_experts must fit the local weight slots")
        local_ids = topk_indices - experts_start
        valid = (local_ids >= 0) & (local_ids < owned)
        topk_weights = jnp.where(valid, topk_weights,
                                 jnp.zeros_like(topk_weights))
        topk_ids_local = jnp.where(valid, local_ids,
                                   jnp.full_like(local_ids, -1))

        if (envs.TPU_MOE_LOWTOK and use_ep
                and 0 < num_tokens <= envs.TPU_MOE_LOWTOK_MAX_TOKENS
                and can_use_moe_lowtok(
                    hidden_states,
                    w1,
                    w2,
                    w1_scale,
                    w2_scale,
                    w1_bias,
                    w2_bias,
                    activation=activation,
                    rhs_quant_dtype=rhs_quant_dtype,
                )):
            logger.info_once(
                "Selected experimental small-batch MoE (max tokens %d)",
                envs.TPU_MOE_LOWTOK_MAX_TOKENS,
            )
            route = moe_lowtok_routing_table(topk_ids_local, topk_weights,
                                             owned)
            return moe_lowtok(hidden_states, route, w1, w1_scale, w2, w2_scale)

        if 0 < num_tokens <= envs.TPU_MOE_DENSE_EXPERT_THRESHOLD and can_use_dense_expert(
                hidden_states,
                w1,
                w2,
                w1_scale,
                w2_scale,
                w1_bias,
                w2_bias,
                activation=activation,
                rhs_quant_dtype=rhs_quant_dtype,
                skip_padded_tokens=skip_padded_tokens,
        ):
            logger.info_once("Selected experimental dense-expert decode MoE")
            return dense_expert_moe(
                hidden_states,
                w1,
                w2,
                w1_scale,
                w2_scale,
                topk_weights,
                topk_ids_local,
                num_primary_experts=owned,
            )

    if gating_output is not None:
        assert gating_output.shape == (num_tokens, global_num_experts)

        topk_weights = apply_scoring_fn(scoring_fn, gating_output)
        if hash_based_topk_indices is not None:
            topk_indices = hash_based_topk_indices
            topk_weights = jnp.take_along_axis(topk_weights,
                                               topk_indices,
                                               axis=-1)
        elif envs.MOE_APPROX_TOPK:
            topk_weights, topk_indices = jax.lax.approx_max_k(
                topk_weights,
                k=topk,
                recall_target=envs.MOE_APPROX_TOPK_RECALL_TARGET)
        else:
            def _run_router_topk(
                    scores: jax.Array) -> tuple[jax.Array, jax.Array]:
                fn = functools.partial(router_topk, k=topk)
                if mesh is not None:
                    data_spec = P(
                        ShardingAxisName.ATTN_DATA
                        if get_mesh_shape_product(
                            mesh, ShardingAxisName.ATTN_DATA) > 1 else
                        ShardingAxisName.MLP_DATA,
                        None,
                    )
                    fn = jax.shard_map(
                        fn,
                        mesh=mesh,
                        in_specs=data_spec,
                        out_specs=(data_spec, data_spec),
                        check_vma=False,
                    )
                return fn(scores)

            if expert_score_correction_bias is not None:
                if envs.TPU_MOE_ROUTER_TOPK == "rowmax":
                    _, topk_indices = _run_router_topk(
                        topk_weights + expert_score_correction_bias[None, :])
                else:
                    _, topk_indices = jax.lax.top_k(
                        topk_weights + expert_score_correction_bias[None, :],
                        k=topk)
                topk_weights = jnp.take_along_axis(topk_weights,
                                                   topk_indices,
                                                   axis=-1)
            else:
                if envs.TPU_MOE_ROUTER_TOPK == "rowmax":
                    topk_weights, topk_indices = _run_router_topk(topk_weights)
                else:
                    topk_weights, topk_indices = jax.lax.top_k(topk_weights,
                                                               k=topk)
        if renormalize:
            topk_weights = topk_weights / topk_weights.sum(axis=-1,
                                                           keepdims=True)
        # Route padding tokens to expert 0 instead of picking a selected expert. This
        # is especially useful when we have a low number of tokens (e.g. low
        # concurrency), where padding tokens may activate unnecessary expert weights
        # and slow down the gmm kernel.
        if num_valid_tokens is not None:
            token_valid = (jnp.arange(num_tokens) < num_valid_tokens)[:, None]
            topk_indices = jnp.where(token_valid, topk_indices, 0)
            topk_weights = jnp.where(token_valid, topk_weights, 0.0)
        # All gathering topk_indices and topk_weights if attention dp is used.
        if mesh is not None and get_mesh_shape_product(
                mesh, ShardingAxisName.ATTN_DATA) > 1:
            topk_indices, topk_weights = all_gather_topk_indices_and_weights(
                topk_indices, topk_weights, dtype, mesh)
        topk_weights = topk_weights.astype(jnp.float32)
        if mesh is not None:
            topk_weights = jax.lax.with_sharding_constraint(
                topk_weights,
                NamedSharding(mesh, P(ShardingAxisName.MLP_DATA, None)))

    if use_ep and mesh is not None:
        ep_size = get_mesh_shape_product(mesh, ShardingAxisName.EXPERT)
        local_num_experts = global_num_experts // ep_size

        if (envs.TPU_MOE_LOWTOK
                and 0 < num_tokens <= envs.TPU_MOE_LOWTOK_MAX_TOKENS
                and _can_use_moe_lowtok_ep(
                    hidden_states,
                    w1,
                    w2,
                    w1_scale,
                    w2_scale,
                    w1_bias,
                    w2_bias,
                    activation=activation,
                    rhs_quant_dtype=rhs_quant_dtype,
                    local_num_experts=local_num_experts,
                )):
            logger.info_once(
                "Selected experimental small-batch MoE (max tokens %d)",
                envs.TPU_MOE_LOWTOK_MAX_TOKENS,
            )
            return _ep_dispatch_kernel(
                hidden_states,
                w1,
                w2,
                w1_scale,
                w2_scale,
                topk_weights,
                topk_indices,
                mesh=mesh,
                kernel_mode="lowtok",
                local_num_experts=local_num_experts,
                scatter_results=scatter_results,
            )

        if (0 < num_tokens <= envs.TPU_MOE_DENSE_EXPERT_THRESHOLD
                and _can_use_dense_expert_ep(
                    hidden_states,
                    w1,
                    w2,
                    w1_scale,
                    w2_scale,
                    w1_bias,
                    w2_bias,
                    activation=activation,
                    rhs_quant_dtype=rhs_quant_dtype,
                    skip_padded_tokens=skip_padded_tokens,
                    local_num_experts=local_num_experts,
                )):
            logger.info_once("Selected experimental dense-expert decode MoE")
            return _ep_dispatch_kernel(
                hidden_states,
                w1,
                w2,
                w1_scale,
                w2_scale,
                topk_weights,
                topk_indices,
                mesh=mesh,
                kernel_mode="dense_expert",
                local_num_experts=local_num_experts,
                scatter_results=scatter_results,
            )

    assert (num_tokens * topk) % 16 == 0, (
        "The kernel requires num_tokens * topk to be a multiple of "
        f"16 but got {num_tokens}*{topk}={num_tokens*topk}")

    # Only enable Reduce-Scatter if flag is on and Attention is pure DP
    if mesh is not None:
        total_num_devices = mesh.devices.size
        is_attn_dp = get_mesh_shape_product(
            mesh, ShardingAxisName.ATTN_DATA) == total_num_devices
        actual_enable_rs_kernel = enable_rs_kernel and is_attn_dp
    else:
        actual_enable_rs_kernel = False

    if envs.FORCE_MOE_RANDOM_ROUTING:
        logger.warning(
            "Forcing random routing should be used for performance testing only."
        )
        # Forcing random routing is useful to get rid of the effect
        # of routing imbalance during performance debugging.
        topk_indices = _override_token_indices_for_random_routing(
            topk_indices, global_num_experts)

    def _process_tokens_locally(hidden_states_local, topk_indices_local):
        num_tokens_local = hidden_states_local.shape[0]
        topk_indices_flat = topk_indices_local.flatten()

        flat_bits = max(1, (topk_indices_flat.shape[0] - 1).bit_length())
        single_sort = (envs.TPU_MOE_SINGLE_SORT
                       and topk_indices_flat.shape[0] <= _SINGLE_SORT_MAX_ROWS
                       and
                       (global_num_experts + 1).bit_length() + flat_bits < 31)
        if single_sort:
            logger.info_once("MoE permutation runs as one packed sort.")
            topk_argsort_indices, topk_argsort_revert_indices, token_indices_sorted = (
                _single_sort_permutation(topk_indices_flat, topk, flat_bits))
        else:
            topk_argsort_indices = jnp.argsort(topk_indices_flat)
            token_indices = jnp.arange(num_tokens_local,
                                       dtype=jnp.int32).repeat(topk)
            token_indices_sorted = token_indices[topk_argsort_indices]
            topk_argsort_revert_indices = jnp.argsort(topk_argsort_indices)

        # Below one_hot is equivalent to jnp.bincount(topk_indices_flat,
        # length=global_num_experts) but is more performant.
        group_sizes_local = jax.nn.one_hot(topk_indices_flat,
                                           global_num_experts,
                                           dtype=jnp.int32).sum(axis=0)

        if use_ep:
            if mesh is not None:
                num_ep_shard = get_mesh_shape_product(mesh,
                                                      ShardingAxisName.EXPERT)
                local_num_experts = global_num_experts // num_ep_shard
                shard_idx = jax.lax.axis_index(ShardingAxisName.EXPERT)
            else:
                num_ep_shard = 1
                local_num_experts = global_num_experts
                shard_idx = 0

            experts_start_idx = experts_start if experts_start is not None else shard_idx * local_num_experts
            experts_start_val = experts_start_idx
            experts_end_val = experts_start_val + local_num_experts
            group_offsets = jnp.cumulative_sum(group_sizes_local,
                                               include_initial=True)
            shard_output_start = group_offsets[experts_start_val]
            shard_output_end = group_offsets[experts_end_val]
            num_tokens = token_indices_sorted.shape[0]
            if num_tokens <= onehot_moe_permute_threshold:
                # Use one-hot matmul for permutation, which can be faster
                # for small batch size
                onehot = jax.nn.one_hot(token_indices_sorted,
                                        hidden_states_local.shape[0],
                                        dtype=hidden_states_local.dtype)
                x = onehot @ hidden_states_local
            else:
                x = ragged_gather(
                    hidden_states_local,
                    token_indices_sorted,
                    shard_output_start,
                    shard_output_end,
                )
        else:
            x = hidden_states_local[token_indices_sorted]

        return x, group_sizes_local, topk_argsort_revert_indices

    if all_gather_fp8 and mesh is not None:
        hidden_states = _apply_all_gather_fp8(hidden_states, mesh, dtype)

    if mesh is not None:
        x, group_sizes, topk_argsort_revert_indices = jax.shard_map(
            _process_tokens_locally,
            mesh=mesh,
            in_specs=(
                P(ShardingAxisName.MLP_DATA, None),
                P(ShardingAxisName.MLP_DATA, None),
            ),
            out_specs=(
                P(ShardingAxisName.MLP_DATA),
                P(ShardingAxisName.MLP_DATA),
                P(ShardingAxisName.MLP_DATA),
            ),
            check_vma=False,
        )(hidden_states, topk_indices)
    else:
        x, group_sizes, topk_argsort_revert_indices = _process_tokens_locally(
            hidden_states, topk_indices)

    try:
        x = jnp.pad(x, ((0, 0), (0, padded_hidden_size - hidden_size)))
    except Exception as e:
        raise ValueError(
            f"Error when padding input hidden states from {hidden_size} to {padded_hidden_size}."
        ) from e

    if use_ep:
        x = expert_parallel_gmm(
            x,
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
            mesh=mesh,
            enable_rs_kernel=actual_enable_rs_kernel,
            onehot_moe_permute_threshold=onehot_moe_permute_threshold,
            scatter_results=scatter_results,
            moe_chunk_size=moe_chunk_size,
            defer_all_reduce=defer_all_reduce,
            ragged_gather_reduce_version=ragged_gather_reduce_version,
        )
    else:
        x = tensor_parallel_gmm(
            x,
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
            mesh=mesh,
            enable_rs_kernel=actual_enable_rs_kernel,
            onehot_moe_permute_threshold=onehot_moe_permute_threshold,
            scatter_results=scatter_results,
            moe_chunk_size=moe_chunk_size,
            defer_all_reduce=defer_all_reduce,
            ragged_gather_reduce_version=ragged_gather_reduce_version,
        )

    return x[:num_tokens, :hidden_size]
