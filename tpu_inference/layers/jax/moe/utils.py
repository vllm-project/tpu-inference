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

from dataclasses import InitVar, dataclass
from functools import partial
from typing import Optional, Tuple
import math
import enum

import jax
import jax.numpy as jnp
from jax.experimental import xla_metadata
#from jax.experimental.pallas.ops.tpu.megablox.gmm import gmm as megablox_gmm
from tpu_inference.kernels.megablox.gmm_ds import gmm as megablox_gmm

from flax import nnx
from flax.typing import Sharding
from jax.sharding import PartitionSpec
from jaxtyping import Float
from qwix._src.core.ragged_dot import ragged_dot as qwix_ragged_dot

from tpu_inference import envs
from tpu_inference.layers.jax.layers import FlaxUtils
from tpu_inference.layers.jax.misc import round_up_to_multiple_of_128_within_limit
from tpu_inference.logger import init_logger
from tpu_inference.models.jax.utils.qwix.qwix_utils import (
    manually_quantize_qwix_activation)

logger = init_logger(__name__)
modeling_flax_utils = FlaxUtils()
set_xla_metadata = xla_metadata.set_xla_metadata

class MoEBackend(enum.Enum):
    FUSED_MOE = "fused_moe"
    VLLM_MOE = "vllm_moe"
    DENSE_MAT = "dense_mat"
    MEGABLX_GMM = "megablox_gmm"
    RAGGED_DOT = "ragged_dot_gmm"

def select_moe_backend():

    assert sum([envs.USE_MOE_EP_KERNEL, envs.USE_VLLM_MOE_KERNEL]) <= 1, "You can enable at most one MoE kernels." 

    match (envs.USE_MOE_EP_KERNEL, envs.USE_VLLM_MOE_KERNEL, envs.USE_MEGABLOCKS, envs.USE_RAGGED_DOT):
        case (True, _, _, _):
            logger.info("[MoE]: Fused MoE kernel is enabled")
            return MoEBackend.FUSED_MOE
        case (_, True, _, _):
            logger.info("[MoE]: VLLM MoE kernel is enabled")
            return MoEBackend.VLLM_MOE
        case (_, _, True,_):
            logger.info("[MoE]: Mega Blocks is enabled for GMM in Sparse Matmul")
            return MoEBackend.MEGABLX_GMM
        case (_, _, _, True):
            logger.info("[MoE]: Ragged Dot is enabled for GMM in Sparse Matmul")
            return MoEBackend.RAGGED_DOT
        case _:
            logger.info("[MoE]: Dense Matmul is enabled")
            return MoEBackend.DENSE_MAT

# --- Helper Functions/Class for Sparse MoE ---
class TransformStrategy(enum.Enum):
    INPUT_OFFSET = enum.auto()
    SEND_SIZE = enum.auto()
    OUTPUT_OFFSET = enum.auto()
    RECV_SIZE = enum.auto()

def sort_activations_fn(inputs: jax.Array, sort_indices: jax.Array) -> jax.Array:
    """Stateless sort of activations."""
    return inputs[sort_indices, ...]

def global_permute_fn(inputs_TD: jax.Array, 
                       selected_experts_TX: jax.Array,
                       num_experts_per_tok: int, 
                       num_local_experts: int):
    """Stateless global permute: Sorts tokens by assigned expert."""
    total_tokens = inputs_TD.shape[0]
    flat_expert_indices = selected_experts_TX.flatten()
    sort_indices_t = jnp.argsort(flat_expert_indices)

    replicated_inputs_tD = jnp.repeat(inputs_TD,
                                      num_experts_per_tok,
                                      axis=0)
    sorted_inputs_tD = sort_activations_fn(replicated_inputs_tD,
                                            sort_indices_t)

    # number of tokens assigned to each expert
    group_sizes_E = jnp.bincount(flat_expert_indices,
                                 length=num_local_experts)

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

def unpermute_fn(processed_tokens: jax.Array, 
                  sort_indices: jax.Array,
                  router_weights_TX: jax.Array,
                  num_experts_per_tok: int,
                  hidden_size: int,
                  output_dtype):
    """Stateless global unpermute logic."""
    with jax.named_scope("unpermute"):
        unsorted_tokens_tD = sort_activations_fn(
            processed_tokens, jnp.argsort(sort_indices))
        reshaped_tokens_TXD = unsorted_tokens_tD.reshape(
            -1, num_experts_per_tok, hidden_size)
    
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
    all_shard_local_sizes = jax.lax.dynamic_slice_in_dim(
        global_group_sizes,
        shard_index * local_expert_size,
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
                    (jnp.array([0]),
                     jnp.cumsum(input_array[:-1])))[shard_id]
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
                                 TransformStrategy.SEND_SIZE,
                                 is_batch_sharded)
    output_offsets = transform_array(all_shards_group_sizes, shard_id,
                                     TransformStrategy.OUTPUT_OFFSET,
                                     is_batch_sharded)
    recv_sizes = transform_array(all_shards_group_sizes, shard_id,
                                 TransformStrategy.RECV_SIZE,
                                 is_batch_sharded)
    return input_offsets, send_sizes, output_offsets, recv_sizes

def gmm_fn(inputs, kernel, group_sizes, 
            tile_size, moe_backend, dtype, quantized_dtype):
    """Stateless Grouped Matrix Multiply."""
    num_rows = inputs.shape[0]
    pad_amount = (tile_size[0] - num_rows % tile_size[0]) % tile_size[0]
    if pad_amount > 0:
        inputs = jnp.pad(inputs, ((0, pad_amount), (0, 0)))

    if moe_backend == MoEBackend.MEGABLX_GMM:
        if quantized_dtype:
            kernel_qvalue, kernel_scale = kernel
            #kernel_qvalue = jnp.swapaxes(kernel_qvalue, 1, 2)
            kernel_scale = jnp.expand_dims(kernel_scale, 2)
        else:
            #kernel_qvalue = jnp.swapaxes(kernel, 1, 2)
            kernel_qvalue = kernel
            kernel_scale = None
        m, g, k, n = inputs.shape[0], *kernel_qvalue.shape
        tm = round_up_to_multiple_of_128_within_limit(m, 512)
        tk = round_up_to_multiple_of_128_within_limit(k, 2048)
        tn = round_up_to_multiple_of_128_within_limit(n, 2048)

        output = megablox_gmm(
            lhs=inputs,
            rhs=kernel_qvalue,
            rhs_scale=kernel_scale,
            group_sizes=group_sizes,
            preferred_element_type=dtype,
            tiling=(tm, tk, tn)
        )

    elif moe_backend == MoEBackend.RAGGED_DOT:
        inputs = manually_quantize_qwix_activation(
            inputs, "ragged_dot", jnp.float8_e4m3fn, [0], {},
            "absmax") if quantized_dtype else inputs
        ragged_dot_func = qwix_ragged_dot if quantized_dtype else jax.lax.ragged_dot
        with set_xla_metadata(
            ragged_dot_tiling=",".join([str(t) for t in tile_size]),
            mosaic_fusion_group="ragged-dot"):
            output = ragged_dot_func(
                lhs=inputs,
                rhs=kernel,
                group_sizes=group_sizes,
                preferred_element_type=dtype)

    if pad_amount > 0:
        output = output[:num_rows, :]
    return output
