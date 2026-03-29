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
import os
from typing import Literal

import jax
from jax import numpy as jnp
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from tpu_inference.kernels.megablox.gmm import gmm
from tpu_inference.kernels.megablox.gmm_v2 import (gmm_v2,
                                                   is_supported_by_gmm_v2)

ENABLE_SPARSECORE_OFFLOADING_BASE_FLAGS = (
    " --xla_tpu_use_tc_device_shape_on_sc=true"
    " --xla_sc_enable_instruction_fusion=false"
    " --xla_sc_disjoint_spmem=false"
    " --xla_sc_disable_megacore_partitioning=true")

# Enable SparseCore All Gather (1D), Reduce Scatter (1D) and All Reduce (ND)
# On Ironwood, by default:
# xla_tpu_enable_sparse_core_collective_offload_all_gather as True
# xla_tpu_enable_sparse_core_collective_offload_reduce_scatter as True
# xla_tpu_enable_sparse_core_collective_offload_all_reduce as True
ENABLE_SPARSECORE_OFFLOADING_FOR_RS_AG_AR = (
    # Disable async collective fusion so collectives can be offloaded individually
    " --xla_tpu_enable_async_collective_fusion_fuse_all_gather=false"
    " --xla_tpu_enable_async_collective_fusion_fuse_all_reduce=false"
    " --xla_tpu_enable_async_collective_fusion_fuse_reduce_scatter=false"
    # Enable SparseCore offloading for each collective type
    " --xla_tpu_enable_sparse_core_collective_offload_all_gather=true"
    " --xla_tpu_enable_sparse_core_collective_offload_reduce_scatter=true"
    " --xla_tpu_enable_sparse_core_collective_offload_all_reduce=true"
    # Enable tracing for offloaded collectives (visible in profiler)
    " --xla_tpu_enable_all_gather_offload_tracing=true"
    " --xla_tpu_enable_reduce_scatter_offload_tracing=true"
    " --xla_tpu_enable_all_reduce_offload_tracing=true"
    # Disable all-reduce combiner to prevent merging multiple all-reduces into
    # a single tuple all-reduce, which is not eligible for SparseCore offloading.
    " --xla_jf_crs_combiner_threshold_in_bytes=0"
    " --xla_jf_crs_combiner_threshold_count=1"
) + ENABLE_SPARSECORE_OFFLOADING_BASE_FLAGS


def enable_sparsecore_offload():
    """Enable SparseCore offloading for collectives.

    Must be called before any JAX compilation (i.e., before any jitted
    function is first invoked). Requires Ironwood TPU with compatible
    libtpu version.

    TPU-specific flags (xla_tpu_*, xla_sc_*) must be set via
    LIBTPU_INIT_ARGS, not XLA_FLAGS.
    """
    existing = os.environ.get("LIBTPU_INIT_ARGS", "")
    os.environ["LIBTPU_INIT_ARGS"] = (
        existing + ENABLE_SPARSECORE_OFFLOADING_FOR_RS_AG_AR)


def print_xla_flags():
    """Print the current XLA_FLAGS and LIBTPU_INIT_ARGS for verification."""
    xla_flags = os.environ.get("XLA_FLAGS", "")
    libtpu_args = os.environ.get("LIBTPU_INIT_ARGS", "")

    if xla_flags:
        print("\nActive XLA_FLAGS:")
        for flag in xla_flags.strip().split("--"):
            flag = flag.strip()
            if flag:
                print(f"  --{flag}")

    if libtpu_args:
        print("\nActive LIBTPU_INIT_ARGS:")
        for flag in libtpu_args.strip().split("--"):
            flag = flag.strip()
            if flag:
                print(f"  --{flag}")

    if not xla_flags and not libtpu_args:
        print("\nNo XLA_FLAGS or LIBTPU_INIT_ARGS set.")


MLP_TENSOR = "model"
EXPERT = "model"
MLP_DATA = "data"


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
        gmm_res = gmm(
            lhs=lhs,
            rhs=rhs,
            rhs_scale=rhs_scale,
            rhs_bias=rhs_bias,
            group_sizes=group_sizes,
            preferred_element_type=lhs.dtype,
            tiling=None,
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
    """Main MoE logic on a local shard can run in TP or EP mode.

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
        shard_id = jax.lax.axis_index(MLP_TENSOR).sum()
        w2_bias = jnp.where(shard_id == 0, w2_bias, 0)

    gmm2_res = gmm_wrapper(gmm1_res, w2, w2_scale, w2_bias, group_sizes,
                           group_offset)

    # First run local reduction on topk experts owned by the rank for all tokens
    token_topk_hidden = gmm2_res[topk_argsort_revert_indices].reshape(
        (-1, topk, gmm2_res.shape[-1]))
    token_topk_hidden = token_topk_hidden * jnp.expand_dims(topk_weights,
                                                            axis=-1)
    token_hidden = token_topk_hidden.sum(axis=-2)

    reduction_axis = MLP_TENSOR if parallelism == "tp" else EXPERT
    # Then global reduction on all ranks for all tokens and all experts
    return jax.lax.psum(token_hidden, axis_name=reduction_axis)


def moe_gmm_local_RS(
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
    """Main MoE logic on a local shard can run in TP or EP mode.

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
        shard_id = jax.lax.axis_index(MLP_TENSOR).sum()
        w2_bias = jnp.where(shard_id == 0, w2_bias, 0)

    gmm2_res = gmm_wrapper(gmm1_res, w2, w2_scale, w2_bias, group_sizes,
                           group_offset)

    # First run local reduction on topk experts owned by the rank for all tokens
    token_topk_hidden = gmm2_res[topk_argsort_revert_indices].reshape(
        (-1, topk, gmm2_res.shape[-1]))
    token_topk_hidden = token_topk_hidden * jnp.expand_dims(topk_weights,
                                                            axis=-1)
    token_hidden = token_topk_hidden.sum(axis=-2)

    reduction_axis = MLP_TENSOR if parallelism == "tp" else EXPERT

    rs = jax.lax.psum_scatter(token_hidden,
                              axis_name=reduction_axis,
                              scatter_dimension=0,
                              tiled=True)
    # Then global reduction on all ranks for all tokens and all experts
    return jax.lax.all_gather(rs, axis_name=reduction_axis, axis=0, tiled=True)


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
    ep_size = 8
    ep_p_spec = P(EXPERT)
    data_p_spec = P(MLP_DATA)
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

    topk_weights = apply_scoring_fn(scoring_fn, gating_output)
    # All-gather topk weights for attention dp
    topk_weights = jax.lax.with_sharding_constraint(
        topk_weights, NamedSharding(mesh, P(MLP_DATA, None)))
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
        in_specs=(
            P(MLP_DATA, None),
            P(MLP_DATA, None),
        ),
        out_specs=(
            P(MLP_DATA, None),
            P(MLP_DATA),
            P(MLP_DATA),
        ),
    )(hidden_states, topk_indices)

    x = jnp.pad(x, ((0, 0), (0, padded_hidden_size - hidden_size)))

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

    return x[:num_tokens, :hidden_size]


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
def fused_moe_func2(
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

    # TODO consider apply this seprately for each half
    topk_weights = apply_scoring_fn(scoring_fn, gating_output)
    # All-gather topk weights for attention dp
    topk_weights = jax.lax.with_sharding_constraint(
        topk_weights, NamedSharding(mesh, P(MLP_DATA, None)))
    topk_weights, topk_indices = jax.lax.top_k(topk_weights, k=topk)
    if renormalize:
        topk_weights = topk_weights / topk_weights.sum(axis=-1, keepdims=True)
    topk_weights = topk_weights.astype(dtype)

    # Split hidden_states into two halves along the first dimension (num_tokens)
    half_tokens = num_tokens // 2
    hidden_states_first_half = hidden_states[:half_tokens]
    hidden_states_second_half = hidden_states[half_tokens:]

    topk_indices_first_half = topk_indices[:half_tokens]
    topk_indices_second_half = topk_indices[half_tokens:]

    topk_weights_first_half = topk_weights[:half_tokens]
    topk_weights_second_half = topk_weights[half_tokens:]

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

    x_1, group_sizes_1, topk_argsort_revert_indices_1 = jax.shard_map(
        _process_tokens_locally,
        mesh=mesh,
        in_specs=(
            P(MLP_DATA, None),
            P(MLP_DATA, None),
        ),
        out_specs=(
            P(MLP_DATA, None),
            P(MLP_DATA),
            P(MLP_DATA),
        ),
    )(hidden_states_first_half, topk_indices_first_half)

    x_1 = jnp.pad(x_1, ((0, 0), (0, padded_hidden_size - hidden_size)))

    x_2, group_sizes_2, topk_argsort_revert_indices_2 = jax.shard_map(
        _process_tokens_locally,
        mesh=mesh,
        in_specs=(
            P(MLP_DATA, None),
            P(MLP_DATA, None),
        ),
        out_specs=(
            P(MLP_DATA, None),
            P(MLP_DATA),
            P(MLP_DATA),
        ),
    )(hidden_states_second_half, topk_indices_second_half)

    x_2 = jnp.pad(x_2, ((0, 0), (0, padded_hidden_size - hidden_size)))

    x_1 = expert_parallel_gmm(
        x_1,
        w1,
        w1_scale,
        w1_bias,
        w2,
        w2_scale,
        w2_bias,
        group_sizes_1,
        topk_argsort_revert_indices_1,
        topk_weights_first_half,
        activation=activation,
        topk=topk,
        mesh=mesh,
    )

    x_2 = expert_parallel_gmm(
        x_2,
        w1,
        w1_scale,
        w1_bias,
        w2,
        w2_scale,
        w2_bias,
        group_sizes_2,
        topk_argsort_revert_indices_2,
        topk_weights_second_half,
        activation=activation,
        topk=topk,
        mesh=mesh,
    )
    x_n = jnp.concatenate([x_1, x_2], axis=0)
    return x_n[:num_tokens, :hidden_size]


@functools.partial(
    jax.jit,
    static_argnames=(
        "topk",
        "renormalize",
        "mesh",
        "use_ep",
        "activation",
        "scoring_fn",
        "num_stages",
    ),
)
def fused_moe_func3(
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
    num_stages: int = 2,
) -> jax.Array:
    """Route tokens in hidden_states into each experts based on routing.

    This version splits the tokens into num_stages chunks along the first
    dimension and processes each chunk sequentially to overlap computation
    with communication.

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
        num_stages: number of stages to split the tokens into for pipelining.

    Returns:
        Output of moe operation [num_tokens, hidden_size]
    """
    num_tokens, hidden_size = hidden_states.shape
    global_num_experts, padded_hidden_size, _ = w1.shape
    dtype = hidden_states.dtype

    assert num_stages >= 1, f"num_stages must be >= 1 but got {num_stages}"
    assert num_tokens % num_stages == 0, (
        f"num_tokens ({num_tokens}) must be divisible by num_stages ({num_stages})"
    )

    chunk_size = num_tokens // num_stages

    assert (chunk_size * topk) % 16 == 0, (
        "The kernel requires chunk_size * topk to be a multiple of "
        f"16 but got {chunk_size}*{topk}={chunk_size*topk}")

    assert gating_output.shape == (num_tokens, global_num_experts)

    topk_weights = apply_scoring_fn(scoring_fn, gating_output)
    topk_weights = jax.lax.with_sharding_constraint(
        topk_weights, NamedSharding(mesh, P(MLP_DATA, None)))
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

    # Reshape inputs to (num_stages, chunk_size, ...) for scan
    hidden_states_stages = hidden_states.reshape(num_stages, chunk_size,
                                                 hidden_size)
    topk_indices_stages = topk_indices.reshape(num_stages, chunk_size, topk)
    topk_weights_stages = topk_weights.reshape(num_stages, chunk_size, topk)

    def scan_body(carry, stage_inputs):
        hidden_states_chunk, topk_indices_chunk, topk_weights_chunk = stage_inputs

        x_chunk, group_sizes_chunk, revert_indices_chunk = jax.shard_map(
            _process_tokens_locally,
            mesh=mesh,
            in_specs=(
                P(MLP_DATA, None),
                P(MLP_DATA, None),
            ),
            out_specs=(
                P(MLP_DATA, None),
                P(MLP_DATA),
                P(MLP_DATA),
            ),
        )(hidden_states_chunk, topk_indices_chunk)

        x_chunk = jnp.pad(x_chunk,
                          ((0, 0), (0, padded_hidden_size - hidden_size)))

        x_chunk = expert_parallel_gmm(
            x_chunk,
            w1,
            w1_scale,
            w1_bias,
            w2,
            w2_scale,
            w2_bias,
            group_sizes_chunk,
            revert_indices_chunk,
            topk_weights_chunk,
            activation=activation,
            topk=topk,
            mesh=mesh,
        )

        return carry, x_chunk

    _, x_stages = jax.lax.scan(
        scan_body,
        init=None,
        xs=(hidden_states_stages, topk_indices_stages, topk_weights_stages),
        unroll=True,  # unroll=False for debugging
    )

    # x_stages shape: (num_stages, chunk_size, padded_hidden_size)
    x_out = x_stages.reshape(num_tokens, -1)
    return x_out[:num_tokens, :hidden_size]


@functools.partial(
    jax.jit,
    static_argnames=(
        "topk",
        "renormalize",
        "mesh",
        "use_ep",
        "activation",
        "scoring_fn",
        "num_stages",
    ),
)
def fused_moe_func4(
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
    num_stages: int = 2,
) -> jax.Array:
    """Route tokens in hidden_states into each experts based on routing.

    This version splits the tokens into num_stages chunks along the first
    dimension and calls fused_moe_func for each chunk via jax.lax.scan
    with unroll=True.

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
        num_stages: number of stages to split the tokens into for pipelining.

    Returns:
        Output of moe operation [num_tokens, hidden_size]
    """
    num_tokens, hidden_size = hidden_states.shape
    global_num_experts = gating_output.shape[1]

    assert num_stages >= 1, f"num_stages must be >= 1 but got {num_stages}"
    assert num_tokens % num_stages == 0, (
        f"num_tokens ({num_tokens}) must be divisible by num_stages ({num_stages})"
    )

    chunk_size = num_tokens // num_stages

    # Reshape inputs to (num_stages, chunk_size, ...) for scan
    hidden_states_stages = hidden_states.reshape(num_stages, chunk_size,
                                                 hidden_size)
    gating_output_stages = gating_output.reshape(num_stages, chunk_size,
                                                 global_num_experts)

    def scan_body(carry, stage_inputs):
        hidden_states_chunk, gating_output_chunk = stage_inputs

        x_chunk = fused_moe_func(
            hidden_states_chunk,
            w1,
            w2,
            w1_scale,
            w2_scale,
            w1_bias,
            w2_bias,
            gating_output_chunk,
            topk=topk,
            renormalize=renormalize,
            mesh=mesh,
            use_ep=use_ep,
            activation=activation,
            scoring_fn=scoring_fn,
        )

        return carry, x_chunk

    _, x_stages = jax.lax.scan(
        scan_body,
        init=None,
        xs=(hidden_states_stages, gating_output_stages),
        unroll=True,  # unroll=False for debugging
    )

    # x_stages shape: (num_stages, chunk_size, hidden_size)
    return x_stages.reshape(num_tokens, hidden_size)
