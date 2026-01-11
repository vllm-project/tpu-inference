"""Pallas TPU lowerable cumulative sum operation."""

import functools
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl

from tpu_inference.kernels.sampling.utils import (
  iota_tile,
  NUM_LANES,
  NUM_SUBLANES,
  log2,
  pad,
  map_batch_dim_to_smaller_than_hardware_tile_size,
)


def reverse_tiles(tiles, axis):
  """Reverse the order of tiles and elements within each tile along axis.

  Args:
    tiles: List of tile arrays to reverse.
    axis: Axis along which to reverse.

  Returns:
    List of reversed tiles with reversed elements.
  """
  tile_shape = tiles[0].shape
  reverse_perm = tile_shape[axis] - 1 - iota_tile(axis)
  return [
    jnp.take_along_axis(tile, reverse_perm, axis=axis) for tile in tiles[::-1]
  ]


def cumsum_tile(tile, axis):
  """Compute cumulative sum within a single tile using parallel scan.

  Args:
    tile: Input tile array.
    axis: Axis along which to compute cumsum.

  Returns:
    Tile with cumulative sum computed.
  """
  n = tile.shape[axis]
  idx = iota_tile(axis)
  for stage in range(log2(n)):
    permutation = idx - 2**stage
    tile += jnp.where(
      permutation >= 0, jnp.take_along_axis(tile, permutation % n, axis=axis), 0
    )
  return tile


@map_batch_dim_to_smaller_than_hardware_tile_size
def cumsum_arrays(arr, *, axis=1, reverse=False):
  """
  TPU Pallas lowerable array based implementation of jax.lax.cumsum

  Note: most TPU versions do not allow lane sums in bfloat16, so suggest  casting to jnp.float32 before passing in
  """
  assert arr.ndim == 2
  shape = arr.shape
  tile_shape = (NUM_SUBLANES, NUM_LANES)
  batch_axis = 1 - axis
  assert arr.shape[batch_axis] <= NUM_LANES, (
    "decorator split to chunks should have ensured this assert"
  )
  arr = pad(arr, tile_shape, val=0)

  n = arr.shape[axis] // tile_shape[axis]
  tiles = jnp.split(arr, n, axis=axis)
  if reverse:
    tiles = reverse_tiles(tiles, axis=axis)
  outs = [cumsum_tile(tile, axis) for tile in tiles]
  tile_sums = [tile.sum(axis, keepdims=True) for tile in tiles]
  for i in range(1, n):
    outs[i] += tile_sums[i - 1]
    tile_sums[i] += tile_sums[i - 1]
  if reverse:
    outs = reverse_tiles(outs, axis=axis)
  return jnp.concatenate(outs, axis=axis)[: shape[0], : shape[1]]


def cumsum_refs(input_ref, output_ref, *, axis: int, reverse: bool):
  """Cumulative sum kernel.

  Computes the cumulative sum of the input array along the specified axis.

  Args:
    input_ref: Input array reference.
    output_ref: Output array reference.
    axis: Axis along which to compute cumsum.
    reverse: If True, compute cumsum in reverse order.
  """
  output_ref[...] = cumsum_arrays(input_ref[...], axis=axis, reverse=reverse)


@functools.partial(jax.jit, static_argnames=("axis", "reverse", "interpret"))
def cumsum(
  arr,
  axis,
  reverse: bool = False,
  interpret: bool = False,
):
  """
  Cumulative sum using Pallas.

  Args:
      arr: Input array.
      axis: Axis along which to compute cumsum.
      reverse: If True, compute cumsum in reverse order.
      interpret: Run in interpreter mode (CPU compatible).

  Returns:
      Cumulative sum array.
  """
  return pl.pallas_call(
    functools.partial(
      cumsum_refs,
      axis=axis,
      reverse=reverse,
    ),
    out_shape=jax.ShapeDtypeStruct(arr.shape, arr.dtype),
    interpret=interpret,
  )(arr)
