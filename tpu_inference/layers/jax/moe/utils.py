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

import enum
import math

import jax
import jax.numpy as jnp
from jax.experimental import xla_metadata
from jax.sharding import Mesh

from tpu_inference import envs
from tpu_inference.kernels.megablox.gmm import gmm as megablox_gmm
from tpu_inference.layers.common.fused_moe_gmm import \
    round_up_to_multiple_of_128_within_limit
from tpu_inference.layers.common.moe import MoEBackend
from tpu_inference.layers.jax.layers import FlaxUtils
from tpu_inference.logger import init_logger

logger = init_logger(__name__)
modeling_flax_utils = FlaxUtils()
set_xla_metadata = xla_metadata.set_xla_metadata


# --- Helper Functions/Class for Sparse MoE ---
class TransformStrategy(enum.Enum):
    INPUT_OFFSET = enum.auto()
    SEND_SIZE = enum.auto()
    OUTPUT_OFFSET = enum.auto()
    RECV_SIZE = enum.auto()


def sort_activations_fn(inputs: jax.Array,
                        sort_indices: jax.Array) -> jax.Array:
    """Stateless sort of activations."""
    return inputs[sort_indices, ...]


def global_permute_fn(inputs_TD: jax.Array, selected_experts_TX: jax.Array,
                      num_experts_per_tok: int, num_local_experts: int):
    """Stateless global permute: Sorts tokens by assigned expert."""
    total_tokens = inputs_TD.shape[0]
    flat_expert_indices = selected_experts_TX.flatten()
    sort_indices_t = jnp.argsort(flat_expert_indices)

    replicated_inputs_tD = jnp.repeat(inputs_TD, num_experts_per_tok, axis=0)
    sorted_inputs_tD = sort_activations_fn(replicated_inputs_tD,
                                           sort_indices_t)

    # number of tokens assigned to each expert
    group_sizes_E = jnp.bincount(flat_expert_indices, length=num_local_experts)

    expert_ids = jnp.arange(num_local_experts)
    total_assignments = total_tokens * num_experts_per_tok
    sorted_expert_assignments_t = jnp.repeat(
        expert_ids,
        repeats=group_sizes_E,
        total_repeat_length=total_assignments)
    return (
        sorted_inputs_tD,
        sort_indices_t,
        group_sizes_E,
        sorted_expert_assignments_t,
    )


def unpermute_fn(processed_tokens: jax.Array, sort_indices: jax.Array,
                 router_weights_TX: jax.Array, num_experts_per_tok: int,
                 output_dtype):
    """Stateless global unpermute logic."""
    with jax.named_scope("unpermute"):
        unsorted_tokens_tD = sort_activations_fn(processed_tokens,
                                                 jnp.argsort(sort_indices))
        local_D = unsorted_tokens_tD.shape[-1]
        reshaped_tokens_TXD = unsorted_tokens_tD.reshape(
            -1, num_experts_per_tok, local_D)

    with jax.named_scope("combine_weights"):
        tokens_f32 = reshaped_tokens_TXD.astype(jnp.float32)
        weights_f32 = router_weights_TX.astype(jnp.float32)
        weights_expanded = jnp.expand_dims(weights_f32, axis=-1)
        output_TD = jnp.sum(tokens_f32 * weights_expanded, axis=1)

    return output_TD.astype(output_dtype)


def local_permute_fn(inputs,
                     global_group_sizes,
                     local_expert_size,
                     shard_index,
                     is_offset,
                     global_sorted_experts=None):
    """Stateless local permutation logic."""
    # global_group_sizes: (tokens parallelism, num_total_experts)
    # all_shard_local_sizes: (tokens parallelism, num local experts in the shard)
    start_index = shard_index * local_expert_size
    all_shard_local_sizes = jax.lax.dynamic_slice_in_dim(global_group_sizes,
                                                         start_index,
                                                         local_expert_size,
                                                         axis=1)
    local_sizes = all_shard_local_sizes.reshape(-1)

    # local_group_size: (tokens parallelism, )
    local_group_size = jnp.sum(all_shard_local_sizes, axis=0)

    # When token replicated in devices
    if is_offset:
        global_sorted_shard_assignments = jnp.floor_divide(
            global_sorted_experts, local_expert_size)
        expert_indices = jnp.where(
            global_sorted_shard_assignments == shard_index,
            jnp.mod(global_sorted_experts, local_expert_size),
            local_expert_size)

    # When token sharded in devices
    else:
        base_indices = jnp.mod(jnp.arange(local_sizes.shape[0]),
                               local_expert_size)
        expert_indices = jnp.repeat(base_indices,
                                    local_sizes,
                                    total_repeat_length=inputs.shape[0])

    sorted_indices = jnp.argsort(expert_indices)
    # sort the inputs based on the local expert_indices
    sorted_inputs = sort_activations_fn(inputs, sorted_indices)
    # sorted local expert id from 0 to local expert size
    sorted_experts_ids = expert_indices[sorted_indices]
    return (
        sorted_inputs,
        sorted_indices,
        local_group_size,
        sorted_experts_ids,
    )


def get_all_to_all_params_fn(all_shards_group_sizes,
                             shard_id,
                             num_expert_parallelism,
                             is_batch_sharded=True):
    """Stateless parameter generation for ragged_all_to_all."""

    def transform_array(input_array, shard_id, strategy, is_batch_sharded):
        if is_batch_sharded:
            if strategy == TransformStrategy.INPUT_OFFSET:
                local_array = input_array[shard_id]
                return jnp.concatenate(
                    (jnp.array([0]), jnp.cumsum(local_array)[:-1]))
            elif strategy == TransformStrategy.SEND_SIZE:
                return input_array[shard_id]
            elif strategy == TransformStrategy.OUTPUT_OFFSET:
                zero_row = jnp.zeros((1, ) + input_array.shape[1:],
                                     dtype=input_array.dtype)
                array_with_zeros = jnp.concatenate((zero_row, input_array),
                                                   axis=0)
                cumulated_array = jnp.cumsum(array_with_zeros,
                                             axis=0,
                                             dtype=input_array.dtype)
                return cumulated_array[shard_id]
            elif strategy == TransformStrategy.RECV_SIZE:
                return input_array[:, shard_id]
            else:
                raise ValueError(
                    f"Unknown transform array strategy: {strategy}")
        else:
            if strategy == TransformStrategy.INPUT_OFFSET:
                return jnp.zeros(num_expert_parallelism,
                                 dtype=input_array.dtype)
            elif strategy == TransformStrategy.SEND_SIZE:
                return jnp.repeat(input_array[shard_id],
                                  num_expert_parallelism)
            elif strategy == TransformStrategy.OUTPUT_OFFSET:
                output_offset = jnp.concatenate(
                    (jnp.array([0]), jnp.cumsum(input_array[:-1])))[shard_id]
                return jnp.repeat(output_offset, num_expert_parallelism)
            elif strategy == TransformStrategy.RECV_SIZE:
                return input_array
            else:
                raise ValueError(
                    f"Unknown transform array strategy: {strategy}")

    input_offsets = transform_array(all_shards_group_sizes, shard_id,
                                    TransformStrategy.INPUT_OFFSET,
                                    is_batch_sharded)
    send_sizes = transform_array(all_shards_group_sizes, shard_id,
                                 TransformStrategy.SEND_SIZE, is_batch_sharded)
    output_offsets = transform_array(all_shards_group_sizes, shard_id,
                                     TransformStrategy.OUTPUT_OFFSET,
                                     is_batch_sharded)
    recv_sizes = transform_array(all_shards_group_sizes, shard_id,
                                 TransformStrategy.RECV_SIZE, is_batch_sharded)
    return input_offsets, send_sizes, output_offsets, recv_sizes


def gmm_fn(inputs, kernel, group_sizes, tile_size, moe_backend, dtype,
           quantized_dtype):
    """Stateless Grouped Matrix Multiply."""
    num_rows = inputs.shape[0]
    pad_amount = (tile_size[0] - num_rows % tile_size[0]) % tile_size[0]
    if pad_amount > 0:
        inputs = jnp.pad(inputs, ((0, pad_amount), (0, 0)))

    if moe_backend == MoEBackend.MEGABLX_GMM:
        if quantized_dtype:
            kernel_qvalue, kernel_scale = kernel
            kernel_scale = jnp.expand_dims(kernel_scale, 2)
        else:
            kernel_qvalue = kernel
            kernel_scale = None
        m = inputs.shape[0]
        _, k, n = kernel_qvalue.shape
        tm = round_up_to_multiple_of_128_within_limit(m, 512)
        tk = round_up_to_multiple_of_128_within_limit(k, 2048)
        tn = round_up_to_multiple_of_128_within_limit(n, 2048)

        # TODO (jacobplatin/bzgoogle): replace this with the DS-specific megablox
        output = megablox_gmm(lhs=inputs,
                              rhs=kernel_qvalue,
                              rhs_scale=kernel_scale,
                              group_sizes=group_sizes,
                              preferred_element_type=dtype,
                              tiling=(tm, tk, tn))

    if pad_amount > 0:
        output = output[:num_rows, :]
    return output


def get_expert_parallelism(expert_axis_name: str, mesh: Mesh) -> int:
    """
    Returns the expert parallelism number from the mesh.

    Args:
        expert_axis_name: The expert axis name.
        mesh: The mesh.

    Returns:
        The expert parallelism number.
    """
    if expert_axis_name is None:
        return 1
    else:
        if isinstance(expert_axis_name, str):
            return mesh.shape[expert_axis_name]
        else:
            return math.prod(mesh.shape[axis] for axis in expert_axis_name)


def select_moe_backend(use_ep: bool) -> MoEBackend:
    """
    Selects the MoE backend for the JAX path.

    Args:
        use_ep: Whether to use expert parallelism.

    Returns:
        The selected MoE backend.
    """
    if envs.USE_MOE_EP_KERNEL:
        if use_ep:
            logger.info_once("[MoE]: Using fused MoE EP kernel")
            return MoEBackend.FUSED_MOE

    if envs.USE_UNFUSED_MEGABLOCKS:
        logger.info_once(
            "[MoE]: Mega Blocks is enabled for GMM in Sparse Matmul")
        return MoEBackend.MEGABLX_GMM

    if use_ep:
        logger.warning_once(
            "USE_MOE_EP_KERNEL=1 but expert parallelism is not "
            "enabled. Falling back to gmm implementation.")
        logger.info_once("[MoE]: Using GMM EP kernel")
        return MoEBackend.GMM_EP

    if envs.USE_DENSE_MOE:
        logger.info_once("[MoE]: Using DENSE_MOE")
        logger.warning_once(
            "[MoE]: DENSE_MOE is naive and not intended for production.")
        return MoEBackend.DENSE_MAT

    logger.info_once("[MoE]: Using GMM TP kernel")
    return MoEBackend.GMM_TP
