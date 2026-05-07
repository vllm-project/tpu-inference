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
from typing import List, Tuple

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec
from vllm.config import get_current_vllm_config
from vllm.distributed.kv_transfer.kv_connector.factory import \
    KVConnectorFactory

from tpu_inference.distributed import kv_transfer
from tpu_inference.logger import init_logger

ReqId = str

CpuChunkId = int

# Corresponds to the initial hash value
NONE_HASH = 0

logger = init_logger(__name__)


def get_kv_connector_cache_layout():
    """
    Retrieve the required kv cache layout for the configured kv connector
    Return: None, when no kv_transfer_config is found; otherwise, the layout str
    """
    vllm_config = get_current_vllm_config()
    kv_config = vllm_config.kv_transfer_config
    if kv_config is not None:
        connector_cls = KVConnectorFactory.get_connector_class(kv_config)
        required_kvcache_layout = \
            connector_cls.get_required_kvcache_layout(vllm_config)
        if required_kvcache_layout is not None:
            return required_kvcache_layout
        logger.info_once(
            "Connectors do not specify a kv cache layout, defaulting to NHD.")
    return None


# TODO(jcgu): cleanup
@functools.partial(
    jax.jit,
    static_argnames=("block_size"),
    donate_argnames=(
        "kv_caches",
        "kv_cache_slices",
    ),
)
def jitted_insert_kv_cache_slices(
    block_size,
    kv_caches: List[jax.Array],
    kv_cache_slices: List[List[jax.Array]],
    block_numbers: jax.Array,
) -> List[jax.Array]:
    """
    JIT-compiled function to insert KV cache slices into the physical
    cache for all layers at once. This fuses reshape, and scatter
    operations into a single efficient kernel.
    """

    def _update_layer(cache, slices):
        """The function to apply to each layer's cache and slices."""
        # new_shape = (1, block_size, *slices[0].shape[1:])
        for (i, block_idx) in enumerate(block_numbers):
            # reshaped_block = slices[i].reshape(new_shape)
            reshaped_block = jax.lax.expand_dims(slices[i], dimensions=(0, ))
            cache = jax.lax.dynamic_update_slice_in_dim(cache,
                                                        reshaped_block,
                                                        block_idx,
                                                        axis=0)
        return cache

    return jax.tree.map(_update_layer, kv_caches, kv_cache_slices)


@functools.partial(
    jax.jit,
    static_argnames=['num_blocks'],
    donate_argnames=('kv_caches', ),
)
def stack_kv_cache_cross_layers(
        kv_caches: List[jax.Array], block_ids: jax.Array,
        num_blocks: int) -> Tuple[List[jax.Array], List[jax.Array]]:
    """
    This uses jax.tree.map to apply the operation across all layers.
    """

    def _gather_blocks(layer_kv_cache):
        return layer_kv_cache.at[block_ids].get()

    gathered_kv_layers = jax.tree.map(_gather_blocks, kv_caches)
    stacked_blocks = jnp.stack(gathered_kv_layers, axis=1)

    # Split the stacked_blocks along axis=0 into individual blocks
    # NOTE(jcgu): num_blocks == len(block_ids)
    split_blocks = jnp.split(stacked_blocks,
                             indices_or_sections=num_blocks,
                             axis=0)
    # split_blocks = jnp.array_split(stacked_blocks, num_blocks, axis=0)

    kv_caches = jax.lax.optimization_barrier(kv_caches)

    return kv_caches, split_blocks


# @functools.partial(
#     jax.jit,
#     static_argnames=("dim_nums", ),
#     donate_argnames=(
#         "kv_caches",
#         "stacked_blocks",
#     ),
# )
# def update_kv_caches(
#         kv_caches: List[jax.Array], stacked_blocks: List[jax.Array],
#         block_indices: jax.Array,
#         dim_nums: jax.lax.ScatterDimensionNumbers) -> List[jax.Array]:
#     """
#     Updates KV caches by unstacking gathered blocks and inserting slices using scatter.

#     Args:
#       kv_caches: List of original KV caches for each layer.
#       stacked_blocks: List of gathered blocks, each with shape (1, num_layers, ...).
#       block_indices: Array of block indices to update.

#     Returns:
#       List of updated KV caches for each layer.
#     """
#     concatenated_blocks = jnp.concatenate(stacked_blocks, axis=0)
#     layer_slices_tuple = jnp.unstack(concatenated_blocks, axis=1)
#     layer_slices_list = list(layer_slices_tuple)

#     def _update_layer(cache_layer, slices):
#         # NOTE(jcgu): make it as a static arg
#         # dnums = jax.lax.ScatterDimensionNumbers(
#         #     update_window_dims=tuple(range(1, len(slices.shape))),
#         #     inserted_window_dims=(0,),
#         #     scatter_dims_to_operand_dims=(0,)
#         # )
#         return jax.lax.scatter(cache_layer, block_indices[:, None], slices,
#                                dim_nums)

#     return jax.tree.map(_update_layer, kv_caches, layer_slices_list)


def update_kv_caches_one(
    kv_caches: List[jax.Array],
    stacked_blocks: List[jax.Array],
    block_indices: List[int],
    mesh: Mesh,
    cached_kv_sharding_spec: PartitionSpec | None = None,
    replicated_sharding: PartitionSpec | None = None,
) -> List[jax.Array]:
    """Update KV caches using cached sharding spec to avoid recompilation.
    
    Args:
        kv_caches: List of KV cache arrays
        stacked_blocks: List of stacked KV blocks
        block_indices: List of destination block indices
        mesh: JAX mesh for sharding
        cached_kv_sharding_spec: Pre-cached sharding spec from initialization.
                                 If None, derived from kv_caches[0].sharding.spec (fallback).
        replicated_sharding: PartitionSpec for replicated sharding
    """
    # Use cached spec if provided, otherwise fall back to extracting from input
    if cached_kv_sharding_spec is None:
        cached_kv_sharding_spec = kv_caches[0].sharding.spec

    src_offsets, dest_offsets, chunk_sizes, num_chunks = pre_update_kv_caches(
        block_indices, mesh, replicated_sharding)
    return update_kv_caches(
        kv_caches,
        stacked_blocks,
        src_offsets,
        dest_offsets,
        chunk_sizes,
        num_chunks,
        mesh,
        cached_kv_sharding_spec,
        cached_kv_sharding_spec,
        replicated_sharding.spec,
    )


def pre_update_kv_caches(
    block_indices: List[int],
    mesh: Mesh,
    replicated_sharding: PartitionSpec | None = None,
) -> Tuple[jax.Array, jax.Array, jax.Array, jax.Array]:

    num_blocks = len(block_indices)
    src_offsets = jnp.arange(num_blocks, dtype=jnp.int32)
    dest_offsets = jnp.array(block_indices, dtype=jnp.int32)
    chunk_sizes = jnp.ones(num_blocks, dtype=jnp.int32)
    num_chunks = jnp.array([num_blocks], dtype=jnp.int32)

    if replicated_sharding is None:
        replicated_sharding = jax.sharding.NamedSharding(
            mesh, jax.sharding.PartitionSpec(), memory_kind='device')
    src_offsets = jax.device_put(src_offsets, replicated_sharding)
    dest_offsets = jax.device_put(dest_offsets, replicated_sharding)
    chunk_sizes = jax.device_put(chunk_sizes, replicated_sharding)
    num_chunks = jax.device_put(num_chunks, replicated_sharding)

    return src_offsets, dest_offsets, chunk_sizes, num_chunks


@functools.partial(
    jax.jit,
    static_argnames=("mesh", "src_sharding_spec", "dest_sharding_spec",
                     "replicated_sharding_spec"),
    donate_argnames=("kv_caches", ),
)
def update_kv_caches(kv_caches: List[jax.Array],
                     stacked_blocks: List[jax.Array], src_offsets: jax.Array,
                     dest_offsets: jax.Array, chunk_sizes: jax.Array,
                     num_chunks: jax.Array, mesh, src_sharding_spec,
                     dest_sharding_spec,
                     replicated_sharding_spec) -> List[jax.Array]:
    """
    Updates KV caches by unstacking gathered blocks and copying slices.

    Args:
      kv_caches: List of original KV caches for each layer.
      stacked_blocks: List of gathered blocks, each with shape (1, num_layers, ...).
      src_offsets: Source offsets for copying.
      dest_offsets: Destination offsets for copying.
      chunk_sizes: Sizes of chunks to copy.
      num_chunks: Number of chunks.
      mesh: JAX mesh for sharding.
      src_sharding_spec: PartitionSpec for source arrays.
      dest_sharding_spec: PartitionSpec for destination arrays.
      replicated_sharding_spec: PartitionSpec for replicated sharding.

    Returns:
      List of updated KV caches for each layer.
    """
    concatenated_blocks = jnp.concatenate(stacked_blocks, axis=0)
    layer_slices_tuple = jnp.unstack(concatenated_blocks, axis=1)
    layer_slices_list = list(layer_slices_tuple)

    output = kv_transfer.multi_layer_copy(
        src_array=layer_slices_list,
        dest_array=kv_caches,
        src_offsets=src_offsets,
        dest_offsets=dest_offsets,
        chunk_sizes=chunk_sizes,
        num_chunks=num_chunks,
        mesh=mesh,
        src_sharding_spec=src_sharding_spec,
        dest_sharding_spec=dest_sharding_spec,
        replicated_sharding_spec=replicated_sharding_spec)
    return output
