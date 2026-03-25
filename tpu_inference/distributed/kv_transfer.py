# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Utilities for array copying on TPU."""

import functools
from typing import Any

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

Future = Any
shard_map = jax.shard_map
P = jax.sharding.PartitionSpec

_NUM_CHUNKS_CACHE = {}


def _start_async_copy_kernel(
    num_chunks,
    src_offsets_ref,
    dest_offsets_ref,
    chunk_sizes_ref,
    src_ref,
    src_dependency_ref,  # pylint: disable=unused-argument
    dest_ref_in,
    *args,
):
    """Internal kernel for starting async copy."""
    semaphore_ref = args[-1]

    @pl.loop(0, num_chunks[0])
    def start_copy(i):
        block_m = chunk_sizes_ref[i]
        pltpu.make_async_copy(
            src_ref.at[pl.ds(src_offsets_ref[i], block_m)],
            dest_ref_in.at[pl.ds(dest_offsets_ref[i], block_m)],
            semaphore_ref,
        ).start()


@jax.named_call
def _start_chunked_copy_kernel(
    src_offsets: jax.Array,
    dest_offsets: jax.Array,
    chunk_sizes: jax.Array,
    num_chunks: jax.Array,
    src_array: jax.Array,
    src_dependency: jax.Array,
    dest_array: jax.Array,
) -> tuple[jax.Array, Future]:
    """Starts an asynchronous chunked copy."""

    pallas_out_shape = (
        pltpu.HBM(dest_array.shape, dest_array.dtype),
        pltpu.SemaphoreType.DMA(()),
    )
    in_specs = (
        pl.BlockSpec(memory_space=pltpu.HBM),  # src_array
        pl.BlockSpec(memory_space=pltpu.HBM),  # src_dependency
        pl.BlockSpec(memory_space=pltpu.HBM),  # dest_array
    )
    input_output_aliases = {6: 0}

    out_specs = (
        pl.BlockSpec(memory_space=pltpu.HBM),
        pl.BlockSpec(memory_space=pltpu.SEMAPHORE),
    )

    dest_array, semaphore = pl.pallas_call(
        _start_async_copy_kernel,
        out_shape=pallas_out_shape,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            grid=(1, ),
            num_scalar_prefetch=4,
            in_specs=in_specs,
            out_specs=out_specs,
        ),
        input_output_aliases=input_output_aliases,
        compiler_params=pltpu.CompilerParams(has_side_effects=True),
    )(
        num_chunks,
        src_offsets,
        dest_offsets,
        chunk_sizes,
        src_array,
        src_dependency,
        dest_array,
    )

    return src_array, (dest_array, semaphore)


def _wait_async_copy_kernel(
    num_chunks,
    src_offsets_ref,
    dest_offsets_ref,
    chunk_sizes_ref,
    src_ref,
    src_dependency_ref,  # pylint: disable=unused-argument
    *args,
):
    """Internal kernel for waiting for async copy."""
    dest_ref_in = args[0]
    semaphore_ref = args[1]
    dest_ref_out = args[2]
    del dest_ref_out

    @pl.loop(0, num_chunks[0])
    def wait_copy(i):
        block_m = chunk_sizes_ref[i]
        pltpu.make_async_copy(
            src_ref.at[pl.ds(src_offsets_ref[i], block_m)],
            dest_ref_in.at[pl.ds(dest_offsets_ref[i], block_m)],
            semaphore_ref,
        ).wait()


@jax.named_call
def _wait_for_chunked_copy_kernel(
    src_offsets: jax.Array,
    dest_offsets: jax.Array,
    chunk_sizes: jax.Array,
    num_chunks: jax.Array,
    src_array: jax.Array,
    src_dependency: jax.Array,  # pylint: disable=unused-argument
    copy_future: Future,
) -> jax.Array:
    """Waits for an asynchronous chunked copy to complete."""
    dest_array, semaphore = copy_future

    dest_array = pl.pallas_call(
        _wait_async_copy_kernel,
        out_shape=pltpu.HBM(dest_array.shape, dest_array.dtype),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            grid=(1, ),
            num_scalar_prefetch=4,
            in_specs=[
                pl.BlockSpec(memory_space=pltpu.HBM),  # src_array
                pl.BlockSpec(memory_space=pltpu.HBM),  # src_dependency
                pl.BlockSpec(memory_space=pltpu.HBM),  # dest_array
                pl.BlockSpec(memory_space=pltpu.SEMAPHORE),  # semaphore
            ],
            out_specs=pl.BlockSpec(memory_space=pltpu.HBM),
        ),
        input_output_aliases={6: 0},
        compiler_params=pltpu.CompilerParams(has_side_effects=True),
    )(
        num_chunks,
        src_offsets,
        dest_offsets,
        chunk_sizes,
        src_array,
        src_dependency,
        dest_array,
        semaphore,
    )
    return dest_array


@functools.partial(
    jax.jit,
    static_argnames=(
        'mesh',
        'src_sharding_spec',
        'dest_sharding_spec',
        'src_offsets_sharding_spec',
        'dest_offsets_sharding_spec',
        'chunk_sizes_sharding_spec',
        'num_chunks_sharding_spec',
    ),
    donate_argnames=('dest_array', ),
)
def _async_copy_jit(
    *,
    src_array,
    dest_array,
    src_offsets,
    dest_offsets,
    chunk_sizes,
    num_chunks,
    mesh,
    src_sharding_spec,
    dest_sharding_spec,
    src_offsets_sharding_spec,
    dest_offsets_sharding_spec,
    chunk_sizes_sharding_spec,
    num_chunks_sharding_spec,
):
    """Performs asynchronous chunked copy on TPU with JIT compilation."""

    orig_shape = src_array[0].shape
    was_padded = orig_shape[-1] < 128

    if was_padded:
        padding = [(0, 0)] * (len(orig_shape) - 1) + [(0, 128 - orig_shape[-1])
                                                      ]
        src_array = [jnp.pad(x, padding) for x in src_array]
        dest_array = [jnp.pad(x, padding) for x in dest_array]

    src_aliased_list = []
    copy_future_list = []
    if len(src_array) > 1:
        last_alias = dest_array[-1]
        last_alias_sharding = dest_sharding_spec
    else:
        last_alias = src_array[0]
        last_alias_sharding = src_sharding_spec

    for kv_slices, kv_cache in zip(src_array, dest_array):
        src_aliased, copy_future = shard_map(
            _start_chunked_copy_kernel,
            mesh=mesh,
            in_specs=(
                src_offsets_sharding_spec,
                dest_offsets_sharding_spec,
                chunk_sizes_sharding_spec,
                num_chunks_sharding_spec,
                src_sharding_spec,
                last_alias_sharding,
                dest_sharding_spec,
            ),
            out_specs=(src_sharding_spec, (dest_sharding_spec, P())),
            check_vma=False,
        )(
            src_offsets,
            dest_offsets,
            chunk_sizes,
            num_chunks,
            kv_slices,
            last_alias,
            kv_cache,
        )

        src_aliased_list.append(src_aliased)
        copy_future_list.append(copy_future)

        last_alias = copy_future[0]
        last_alias_sharding = dest_sharding_spec

    last_dest = src_aliased_list[-1]
    last_dest_sharding = src_sharding_spec

    dest_out_list = []
    for _, (src_aliased,
            copy_future) in zip(src_array,
                                zip(src_aliased_list, copy_future_list)):
        waited_dest = shard_map(
            _wait_for_chunked_copy_kernel,
            mesh=mesh,
            in_specs=(
                src_offsets_sharding_spec,
                dest_offsets_sharding_spec,
                chunk_sizes_sharding_spec,
                num_chunks_sharding_spec,
                src_sharding_spec,
                last_dest_sharding,
                (dest_sharding_spec, P()),
            ),
            out_specs=dest_sharding_spec,
            check_vma=False,
        )(
            src_offsets,
            dest_offsets,
            chunk_sizes,
            num_chunks,
            src_aliased,
            last_dest,
            copy_future,
        )
        last_dest = waited_dest
        last_dest_sharding = dest_sharding_spec

        dest_out_list.append(waited_dest)

    if was_padded:
        dest_out_list = [x[..., :orig_shape[-1]] for x in dest_out_list]

    return dest_out_list


@jax.named_call
def multi_layer_copy(
    *,
    src_array: list[jax.Array],
    dest_array: list[jax.Array],
    src_offsets: jax.Array,
    dest_offsets: jax.Array,
    chunk_sizes: jax.Array,
    num_chunks: jax.Array | None = None,
):
    """Performs asynchronous chunked copy on TPU.

  This utility overlaps an asynchronous DMA transfer (HBM to HBM).
  It uses Pallas kernels and optimization barriers to ensure that the data
  transfer progresses overlap with each other as much as possible.

  The source and destination arrays are sliced into "chunks" or "blocks".
  The specific offsets of blocks to copy are provided dynamically via
  `src_offsets` and `dest_offsets`. The slicing logic (which dimension to
  chunk on, block size, etc.) is encapsulated in the `copy_f` parameter.

  Requirements:
    - `src_array` must be sharded across the TPU memory.
    - `dest_array` must be sharded across the TPU memory.
    - Both `src_offsets` and `dest_offsets` must be JAX arrays to allow
      dynamic indexing without recompilation.

  Args:
    src_array: The source JAX array on TPU (in Device HBM).
    dest_array: The destination JAX array (also in HBM).
    src_offsets: A 1D JAX array of block offsets to copy from the source.
    dest_offsets: A 1D JAX array of block offsets identifying where to copy in
      the destination. Must have the same length as `src_offsets`.
    chunk_sizes: A 1D JAX array of chunk sizes for each block. Must have the
      same length as `src_offsets`.
    num_chunks: Optional. A scalar JAX array indicating the number of chunks.
      If None, it's derived from `chunk_sizes.shape[0]`.

  Returns:
    Updated_dest_array: The destination array reflecting the copied chunks.
  """

    def get_spec(sharding):
        return getattr(sharding, 'spec', P())

    mesh = src_array[0].sharding.mesh
    src_sharding_spec = get_spec(src_array[0].sharding)
    dest_sharding_spec = get_spec(dest_array[0].sharding)
    if num_chunks is None:
        num_chunks_val = chunk_sizes.shape[0]
        num_chunks_sharding = chunk_sizes.sharding
        if num_chunks_sharding not in _NUM_CHUNKS_CACHE:
            _NUM_CHUNKS_CACHE[num_chunks_sharding] = (
                num_chunks_val,
                jax.device_put(jnp.array([num_chunks_val], dtype=jnp.int32),
                               num_chunks_sharding),
            )
        cached_val, cached_arr = _NUM_CHUNKS_CACHE[num_chunks_sharding]
        if cached_val != num_chunks_val:
            cached_arr = cached_arr.at[0].set(num_chunks_val)
            _NUM_CHUNKS_CACHE[num_chunks_sharding] = (num_chunks_val,
                                                      cached_arr)
        num_chunks = cached_arr

    return _async_copy_jit(
        src_array=src_array,
        dest_array=dest_array,
        src_offsets=src_offsets,
        dest_offsets=dest_offsets,
        chunk_sizes=chunk_sizes,
        mesh=mesh,
        src_sharding_spec=src_sharding_spec,
        dest_sharding_spec=dest_sharding_spec,
        src_offsets_sharding_spec=get_spec(src_offsets.sharding),
        dest_offsets_sharding_spec=get_spec(dest_offsets.sharding),
        chunk_sizes_sharding_spec=get_spec(chunk_sizes.sharding),
        num_chunks_sharding_spec=get_spec(num_chunks.sharding),
        num_chunks=num_chunks,
    )
