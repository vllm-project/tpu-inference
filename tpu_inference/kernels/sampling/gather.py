"""TPU-optimized gather operations using Pallas.

This module provides efficient implementations of array gathering operations
(take_along_axis) optimized for TPU hardware.
"""

import functools
import jax
import jax.numpy as jnp
from jax import jit, lax
from jax.experimental import pallas as pl

from tpu_inference.kernels.sampling.utils import (
  NUM_LANES,
  NUM_SUBLANES,
  pad,
  to_32bit_dtype,
  map_batch_dim_to_smaller_than_hardware_tile_size,
)


@functools.partial(map_batch_dim_to_smaller_than_hardware_tile_size, num_args=2)
def take_along_axis_arrays(val, idx, *, axis=1):
  """Take values from array using indices along specified axis.

  Array-based implementation that processes data in tiles for efficient
  TPU execution.

  Args:
    val: Source array to gather from.
    idx: Indices to gather.
    axis: Axis along which to gather (default: 1).

  Returns:
    Gathered values with same shape as idx.
  """
  shape = idx.shape
  out_dtype = val.dtype
  val = val.astype(to_32bit_dtype(val.dtype))
  tile_shape = (NUM_SUBLANES, NUM_LANES)
  val, idx = (pad(x, tile_shape, val=0) for x in (val, idx))
  # Initialize accumulators
  accumulators = [
    jnp.zeros(tile_shape, dtype=val.dtype)
    for _ in range(idx.shape[axis] // tile_shape[axis])
  ]
  batch_axis = 1 - axis
  assert val.shape[batch_axis] == idx.shape[batch_axis]
  for val_offset in range(0, val.shape[axis], tile_shape[axis]):
    # Load values for this block once
    val_tile = lax.slice_in_dim(
      val, val_offset, val_offset + tile_shape[axis], axis=axis
    )

    # Apply to all K blocks
    for idx_offset in range(0, idx.shape[axis], tile_shape[axis]):
      idx_tile = lax.slice_in_dim(
        idx, idx_offset, idx_offset + tile_shape[axis], axis=axis
      )
      mask = (idx_tile >= val_offset) & (
        idx_tile < val_offset + tile_shape[axis]
      )
      gather_tile = jnp.take_along_axis(
        val_tile, (idx_tile - val_offset) % tile_shape[axis], axis=axis
      )
      i = idx_offset // tile_shape[axis]
      accumulators[i] = jnp.where(mask, gather_tile, accumulators[i])
  return jnp.concatenate(accumulators, axis=axis)[
    : shape[0], : shape[1]
  ].astype(out_dtype)


def take_along_axis_refs(values_ref, indices_ref, output_ref, *, axis: int):
  """Gather values by indexing in to all of value with a mask.

  This kernel processes multiple tiles of output (NUM_SUBLANES x K).
  It scans across the entire values_ref (which contains full vocab for the
  corresponding tokens) once, updating all output tiles.
  """
  output_ref[...] = take_along_axis_arrays(
    values_ref[...], indices_ref[...], axis=axis
  )


@functools.partial(
  jit,
  static_argnames=(
    "axis",
    "interpret",
  ),
)
def take_along_axis(
  values,
  indices,
  axis,
  interpret: bool = False,
):
  """
  Gather values from `values` array using `indices`.

  Args:
      values: Input values [Batch, VocabSize].
      indices: Indices to gather [Batch, K].
      interpret: Run in interpreter mode (CPU compatible).

  Returns:
      Gathered values: [Batch, K].
  """
  return pl.pallas_call(
    functools.partial(
      take_along_axis_refs,
      axis=axis,
    ),
    out_shape=jax.ShapeDtypeStruct(indices.shape, values.dtype),
    interpret=interpret,
  )(values, indices)
