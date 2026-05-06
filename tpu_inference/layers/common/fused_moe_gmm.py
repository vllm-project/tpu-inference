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
import math
from typing import Literal

import jax
from jax import numpy as jnp
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

import tpu_inference.envs as envs
from tpu_inference.kernels.megablox.gmm_v2 import gmm_v2
from tpu_inference.kernels.sparse_core import gather_reduce as gather_reduce_sc
from tpu_inference.kernels.sparse_core.ragged_gather import ragged_gather
from tpu_inference.kernels.sparse_core.ragged_scatter import ragged_scatter
from tpu_inference.layers.common.quantization import quantize_tensor
from tpu_inference.layers.common.sharding import ShardingAxisName
from tpu_inference.logger import init_logger
from tpu_inference.utils import get_mesh_shape_product

logger = init_logger(__name__)

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
    topk_weights = topk_weights.reshape(-1, top_k).astype(dtype)

    return topk_indices, topk_weights


def apply_scoring_fn(scoring_fn: str, x: jax.Array) -> jax.Array:
    match scoring_fn:
        case "softmax":
            return jax.nn.softmax(x, axis=-1)
        case "sigmoid":
            return jax.nn.sigmoid(x)
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


def moe_gmm_local(x: jax.Array, w1: jax.Array, w1_scale: jax.Array | None,
                  w1_bias: jax.Array | None, w2: jax.Array,
                  w2_scale: jax.Array | None, w2_bias: jax.Array | None,
                  group_sizes: jax.Array, group_offset: jax.Array,
                  topk_argsort_revert_indices: jax.Array,
                  topk_weights: jax.Array, *, activation: str, topk: int,
                  parallelism: Literal["tp", "ep"]) -> jax.Array:
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

    if local_group_size < group_sizes.size:
        mask = valid_rows_mask(
            gmm1_res.shape[0],
            group_sizes,
            group_offset,
            group_offset + local_group_size,
        )[topk_argsort_revert_indices].reshape(-1, topk, 1)
    else:
        mask = jnp.full((batch_size, ), True).reshape(-1, topk, 1)

    reduction_axis = (ShardingAxisName.MLP_TENSOR
                      if parallelism == "tp" else ShardingAxisName.EXPERT)

    lcm = (128 * topk) // math.gcd(128, topk)
    chunk_size = max(lcm, (TARGET_SLOT_CHUNK_SIZE + lcm // 2) // lcm * lcm)

    use_sc = gather_reduce_sc.is_supported_by_sc_gather_reduce(
        gmm1_res.shape[0], chunk_size) and topk == 8

    if batch_size <= chunk_size:
        # Path 3: No pipeline at all no kernel
        if local_group_size < group_sizes.size:
            group_offsets = jnp.cumulative_sum(group_sizes,
                                               include_initial=True)
            experts_start = group_offset[0]
            experts_end = group_offset[0] + local_group_size
            shard_output_start = group_offsets[experts_start]
            shard_output_end = group_offsets[experts_end]
            token_hidden_full = ragged_scatter(gmm2_res,
                                               topk_argsort_revert_indices,
                                               shard_output_start,
                                               shard_output_end)
        else:
            token_hidden_full = gmm2_res[topk_argsort_revert_indices]

        cur_sorted = token_hidden_full.reshape((-1, topk, gmm2_res.shape[-1]))
        cur_topk_weights = jnp.expand_dims(topk_weights, axis=-1)
        cur_weighted = cur_sorted * cur_topk_weights
        cur_masked = jnp.where(mask, cur_weighted, 0.0)
        cur_reduced = cur_masked.sum(axis=-2)
        out = jax.lax.psum(cur_reduced, axis_name=reduction_axis)
        return out

    # Pipelined paths
    if use_sc:
        # Path 1: Kernel pipeline
        hidden_size = gmm2_res.shape[1]
        sc_kernel_col_chunk_size = gather_reduce_sc.get_valid_col_chunk_size(
            hidden_size)

        if local_group_size < group_sizes.size:
            mask_flat = mask.reshape(-1, topk)
            topk_weights_sc = jnp.where(mask_flat, topk_weights, 0)
            topk_wgt_zero_nan = True
        else:
            topk_weights_sc = topk_weights
            topk_wgt_zero_nan = False
        topk_weights_flat = topk_weights_sc.flatten()
    else:
        # Path 2: No kernel pipeline
        if local_group_size < group_sizes.size:
            group_offsets = jnp.cumulative_sum(group_sizes,
                                               include_initial=True)
            experts_start = group_offset[0]
            experts_end = group_offset[0] + local_group_size
            shard_output_start = group_offsets[experts_start]
            shard_output_end = group_offsets[experts_end]
            token_hidden_full = ragged_scatter(gmm2_res,
                                               topk_argsort_revert_indices,
                                               shard_output_start,
                                               shard_output_end)
        else:
            token_hidden_full = gmm2_res[topk_argsort_revert_indices]

    out_list = []
    for start in range(0, batch_size, chunk_size):
        end = min(batch_size, start + chunk_size)
        start_tok = start // topk
        end_tok = end // topk
        cur_indices = topk_argsort_revert_indices[start:end]

        if use_sc:
            cur_weights = topk_weights_flat[start:end].reshape(-1, 128)
            cur_reduced = gather_reduce_sc.sc_gather_reduce(
                op=gmm2_res,
                idx=cur_indices,
                reduce_group_size=topk,
                topk_weights=cur_weights,
                col_chunk_size=sc_kernel_col_chunk_size,
                topk_wgt_zero_nan=topk_wgt_zero_nan,
            )
        else:
            cur_topk_weights = topk_weights[start_tok:end_tok]
            cur_mask = mask[start_tok:end_tok]
            cur_sorted = token_hidden_full[start:end].reshape(
                (-1, topk, gmm2_res.shape[-1]))
            cur_topk_weights = jnp.expand_dims(cur_topk_weights, axis=-1)
            cur_weighted = cur_sorted * cur_topk_weights
            cur_masked = jnp.where(cur_mask, cur_weighted, 0.0)
            cur_reduced = cur_masked.sum(axis=-2)

        out = jax.lax.psum(cur_reduced, axis_name=reduction_axis)
        out_list.append(out)

    return jnp.concat(out_list, axis=0)


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

    w1_scale_spec = (None if w1_scale is None else P(
        None, None, None, ShardingAxisName.MLP_TENSOR))
    w1_bias_spec = (None if w1_bias is None else P(
        None, None, ShardingAxisName.MLP_TENSOR))

    num_blocks = 1 if w2_scale is None else w2_scale.shape[1]
    w2_scale_spec = (None if num_blocks == 1 else P(
        None, ShardingAxisName.MLP_TENSOR, None, None))
    w2_bias_spec = None if w2_bias is None else P(None, None, None)

    return jax.shard_map(
        functools.partial(
            moe_gmm_local,
            activation=activation,
            topk=topk,
            parallelism="tp",
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
        out_specs=(data_p_spec),
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
        functools.partial(
            moe_gmm_local,
            activation=activation,
            topk=topk,
            parallelism="ep",
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
        out_specs=(data_p_spec),
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
))
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
    all_gather_fp8: bool = False,
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

    topk_weights = apply_scoring_fn(scoring_fn, gating_output)
    topk_weights, topk_indices = jax.lax.top_k(topk_weights, k=topk)
    if renormalize:
        topk_weights = topk_weights / topk_weights.sum(axis=-1, keepdims=True)
    # All gathering topk_indices and topk_weights if attention dp is used.
    if get_mesh_shape_product(mesh, ShardingAxisName.ATTN_DATA) > 1:
        topk_indices, topk_weights = all_gather_topk_indices_and_weights(
            topk_indices, topk_weights, dtype, mesh)
    topk_weights = topk_weights.astype(dtype)
    topk_weights = jax.lax.with_sharding_constraint(
        topk_weights, NamedSharding(mesh, P(ShardingAxisName.MLP_DATA, None)))

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
        topk_argsort_indices = jnp.argsort(topk_indices_flat)
        token_indices = jnp.arange(num_tokens_local,
                                   dtype=jnp.int32).repeat(topk)
        token_indices_sorted = token_indices[topk_argsort_indices]
        # Below one_hot is equivalent to jnp.bincount(topk_indices_flat,
        # length=global_num_experts) but is more performant.
        group_sizes_local = jax.nn.one_hot(topk_indices_flat,
                                           global_num_experts,
                                           dtype=jnp.int32).sum(axis=0)
        topk_argsort_revert_indices = jnp.argsort(topk_argsort_indices)

        if use_ep:
            num_ep_shard = get_mesh_shape_product(mesh,
                                                  ShardingAxisName.EXPERT)
            local_num_experts = global_num_experts // num_ep_shard
            shard_idx = jax.lax.axis_index(ShardingAxisName.EXPERT)

            experts_start = shard_idx * local_num_experts
            experts_end = experts_start + local_num_experts
            group_offsets = jnp.cumulative_sum(group_sizes_local,
                                               include_initial=True)
            shard_output_start = group_offsets[experts_start]
            shard_output_end = group_offsets[experts_end]
            x = ragged_gather(
                hidden_states_local,
                token_indices_sorted,
                shard_output_start,
                shard_output_end,
            )
        else:
            x = hidden_states_local[token_indices_sorted]

        return x, group_sizes_local, topk_argsort_revert_indices

    if all_gather_fp8:
        hidden_states = _apply_all_gather_fp8(hidden_states, mesh, dtype)

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
        )

    return x[:num_tokens, :hidden_size]
