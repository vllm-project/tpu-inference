"""
Bitonic Top-K using compressed transpose format.

This module contains the array-based top-k implementation using bitonic algorithms.
"""

import functools
from functools import lru_cache

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from tpu_inference.kernels.sampling.utils import (
  NUM_LANES,
  NUM_SUBLANES,
  log2,
  ceil_multiple,
  iota_tile,
  pad,
  transpose_list_of_lists,
  canonicalize_operand,
  to_compressed_transpose_format,
  from_compressed_transpose_format,
  to_32bit_dtype,
  create_bit_indicator,
  join_tiles_to_array,
  split_array_to_tiles,
  map_batch_dim_to_smaller_than_hardware_tile_size,
)


def _compute_padded_shape(
  unpadded_dim0: int, unpadded_dim1: int, k: int
) -> tuple[int, int]:
  """Compute padded shape compatible with compressed transpose format requirements.

  This function finds the minimal padded shape that satisfies the constraints:
  - dim0 is a power of 2 between NUM_SUBLANES and NUM_LANES (inclusive)
  - dim1 is a multiple of k which after compressed transpose becomes dim0 with a multiple of k.
  Post compressed transpose dim1 maps to dim1 // pl.cdiv(NUM_LANES, dim0)
  - num_elems must be divisible by NUM_LANES^2 so mosaic lowers the split and
    concat on full tiles, subtile concat not supported

  Args:
    unpadded_dim0: Original first dimension size
    unpadded_dim1: Original second dimension size
    k: Target top-k size (must be power of 2 for padding calculation purposes)

  Returns:
    Tuple of (padded_dim0, padded_dim1) compatible with compressed transpose format
  """
  if unpadded_dim0 >= NUM_LANES:
    # it won't be compressed so simpler rules
    dim0 = ceil_multiple(unpadded_dim0, NUM_LANES)
    dim1 = ceil_multiple(unpadded_dim1, max(k, NUM_SUBLANES))
    return (dim0, dim1)

  dim0s = [
    2**i
    for i in range(log2(NUM_SUBLANES), log2(NUM_LANES) + 1)
    if 2**i >= unpadded_dim0
  ]
  shapes = [
    (
      dim0,
      ceil_multiple(
        ceil_multiple(unpadded_dim1, NUM_LANES * NUM_LANES // dim0),
        # ensure dim1 after compression (but before transpose) is multiple of k
        max(k, NUM_SUBLANES) * NUM_LANES // dim0,
      ),
    )
    for dim0 in dim0s
  ]
  # take minimal num elements, larger dim0 on ties as cross tile ops are faster than cross lane
  return sorted(shapes, key=lambda x: (x[0] * x[1], -x[0]))[0]


@functools.partial(
  map_batch_dim_to_smaller_than_hardware_tile_size, max_batch_size=NUM_LANES
)
def bitonic_topk_arrays(
  operands: list[jax.Array],
  k: int,
  axis: int = 1,
):
  """
  Progressive bitonic merge for top-k selection.

  Args:
      operands: List of JAX arrays of shape (dim0, dim1)
      k: Number of top elements to return (default: NUM_LANES)
      axis: Axis along which to perform top-k (0 or 1)

  Returns:
      List of JAX arrays of shape (original_batch_size, k) with top-k elements
  """
  operands, shape = canonicalize_operand(operands)
  dtypes = [x.dtype for x in operands]
  sort_axis = axis
  batch_axis = 1 - sort_axis
  unpadded_k = k
  k = 2 ** log2(k)
  # Compute padded shape that satisfies alignment requirements
  unpadded_sort_dim = shape[sort_axis]
  if unpadded_k > unpadded_sort_dim:
    raise ValueError
  if sort_axis == 1:
    padded_shape = _compute_padded_shape(shape[0], shape[1], k=k)
  elif sort_axis == 0:
    padded_shape = (
      ceil_multiple(shape[0], max(NUM_SUBLANES, k)),
      ceil_multiple(shape[1], NUM_LANES),
    )
  else:
    raise ValueError
  # Pad both dimensions if needed
  arrs = [pad(op, block_shape=padded_shape, val="min") for op in operands]
  arrs = [x.astype(to_32bit_dtype(x.dtype)) for x in arrs]

  batch_size = arrs[0].shape[batch_axis]
  assert batch_size <= NUM_LANES
  _bitonic_sort_substage = functools.partial(
    bitonic_sort_substage, batch_size=batch_size
  )

  def max_reduce_stage(arrs_tiles, reduce_stage):
    for substage in range(log2(k))[::-1]:
      arrs_tiles = _bitonic_sort_substage(
        arrs_tiles, substage=substage, stage=reduce_stage
      )
    return _bitonic_sort_substage(
      arrs_tiles, substage=reduce_stage, max_reduce=True
    )

  # Convert to compressed transpose format
  if sort_axis == 1:
    arrs = jax.tree.map(to_compressed_transpose_format, arrs)
  arrs_tiles = jax.tree.map(split_array_to_tiles, arrs)
  num_tiles = len(arrs_tiles[0])
  unsplit_dim0 = num_tiles * arrs_tiles[0][0].shape[0]
  assert (unsplit_dim0 % k) == 0
  num_merges = log2(unpadded_sort_dim) - log2(k)
  num_sublane_merges = log2(pl.cdiv(NUM_SUBLANES, k))
  num_lane_merges = log2(pl.cdiv(unpadded_sort_dim, num_tiles * NUM_SUBLANES))
  num_tile_merges = num_merges - num_sublane_merges - num_lane_merges

  # Build bitonic sequences up to length k/2
  for stage in range(1, log2(k)):
    for substage in range(stage)[::-1]:
      arrs_tiles = _bitonic_sort_substage(
        arrs_tiles, substage=substage, stage=stage
      )

  # Progressive merge tiles together as far as possible first
  for _ in range(num_tile_merges):
    # special handling for cross tile as tile to compare to may not exist
    remainder_length = len(arrs_tiles[0]) % (2 * pl.cdiv(k, NUM_SUBLANES))
    if remainder_length:
      remainder_arrs_tiles = [x[-remainder_length:] for x in arrs_tiles]
      arrs_tiles = [x[:-remainder_length] for x in arrs_tiles]
    arrs_tiles = max_reduce_stage(
      arrs_tiles, reduce_stage=log2(ceil_multiple(k, NUM_SUBLANES))
    )
    if remainder_length:
      arrs_tiles = [
        x + rem for x, rem in zip(arrs_tiles, remainder_arrs_tiles, strict=True)
      ]

  for i in range(num_lane_merges)[::-1]:
    arrs_tiles = max_reduce_stage(
      arrs_tiles, reduce_stage=log2(ceil_multiple(k, NUM_SUBLANES)) + i
    )
  for i in range(num_sublane_merges)[::-1]:
    arrs_tiles = max_reduce_stage(arrs_tiles, reduce_stage=log2(k) + i)

  # Final sort: convert bitonic sequence to fully descending order
  # Use sort_dim_offset=k to ensure descending direction
  for substage in range(log2(k))[::-1]:
    arrs_tiles = _bitonic_sort_substage(
      arrs_tiles, substage=substage, stage=log2(k), sort_dim_offset=k
    )

  if sort_axis == 1:
    arrs = [
      from_compressed_transpose_format(tiles, dim0=batch_size)
      for tiles in arrs_tiles
    ]
    arrs = [arr[: shape[batch_axis], :unpadded_k] for arr in arrs]
  else:
    arrs = [
      join_tiles_to_array(tiles, dim0=ceil_multiple(k, NUM_SUBLANES))
      for tiles in arrs_tiles
    ]
    arrs = [arr[:unpadded_k, : shape[batch_axis]] for arr in arrs]
    
  return [arr.astype(dtype) for arr, dtype in zip(arrs, dtypes, strict=True)]


def max_arrays(operands, axis):
  arrs = bitonic_topk_arrays(operands, k=1, axis=axis)
  return [x.squeeze(axis) for x in arrs]


def compare_and_swap(
  lefts,
  rights,
  *,
  is_descending: jax.Array | bool | None,
  is_right_half=None,
):
  """Compare and conditionally swap array pairs.

  Args:
    lefts: Tuple of left arrays to compare
    rights: Tuple of right arrays to compare
    is_descending: Boolean mask for sort direction (None implies ascending)
    is_right_half: Mask for subtile comparisons. Needed for handling ties in values correctly.

  Returns:
    Tuple of (sorted_lefts, sorted_rights) or sorted values for subtile.
  """
  left, right = lefts[0], rights[0]
  if is_right_half is not None:
    # swap
    left, right = (
      jnp.where(is_right_half, right, left),
      jnp.where(is_right_half, left, right),
    )

  mask = (
    # if possible resolve is_descending at compile time
    left > right
    if type(is_descending) == bool and is_descending
    else right > left
  )

  if is_descending is not None and type(is_descending) != bool:
    # Dynamic descending mask
    mask = mask.astype(bool)
    is_descending = is_descending.astype(bool)
    mask = mask ^ is_descending

  return jax.tree.map(
    lambda left, right: (
      (jnp.where(mask, left, right), jnp.where(mask, right, left))
      if is_right_half is None
      else jnp.where(mask, left, right)
    ),
    lefts,
    rights,
  )


@lru_cache
def compute_pair_slice_start_index(i, separation, slice_length=1):
  """Compute start index for pair-wise array slicing."""
  if slice_length > separation:
    raise ValueError(
      f"Separation must be at least slice length, {separation=} {slice_length=}"
    )
  slices_per_pair = separation // slice_length
  pair_idx = i // slices_per_pair
  slice_idx = i % slices_per_pair
  return pair_idx * 2 * separation + slice_idx * slice_length


def _compute_is_descending(
  stage: int,
  tile_start_offset: int,
  tile_local_offset: jax.Array,
  sort_dim_offset: int,
  full_size: int,
):
  # Check if we can optimize based on stage comparisons
  if stage < log2(NUM_SUBLANES) or stage >= log2(full_size):
    # Same pattern for all tiles
    return create_bit_indicator(
      stage, tile_local_offset + sort_dim_offset
    )

  if stage >= log2(NUM_SUBLANES) and stage < log2(full_size):
    # Bit set by tile_offset, constant within tile, differs across tiles
    return create_bit_indicator(
      stage, tile_start_offset + sort_dim_offset
    )

  # Can't optimize - use full computation
  return create_bit_indicator(
    stage,
    tile_start_offset + tile_local_offset + sort_dim_offset,
  )


@functools.partial(
  jax.jit,
  static_argnames=(
    "substage",
    "batch_size",
    "stage",
    "sort_dim_offset",
    "max_reduce",
  ),
)
def bitonic_sort_substage(
  arrs_tiles,
  *,
  substage,
  batch_size: int,
  stage: int | None = None,
  sort_dim_offset: int = 0,
  max_reduce: bool = False,
):
  """Perform intra-tile bitonic comparison for sort.

  Args:
    arrs_tiles: Tuple of lists of tile arrays
    substage: Substage within current stage (determines separation = 2**substage)
    batch_size: Batch size for computing tile offsets
    stage: Current sorting stage
    sort_dim_offset: Offset for bitonic order calculation
    max_reduce: If True, discard lower half (for top-k)

  Returns:
    Tuple of lists of tiles with updated values
  """
  assert max_reduce or stage is not None
  separation = 2**substage
  # if still arrays, we make it into one big tile so its sanitized to list[list[jax.ndarray]]
  arrs_tiles = list(map(jax.tree.leaves, arrs_tiles))
  full_size = len(arrs_tiles[0]) * arrs_tiles[0][0].shape[0]
  if separation < NUM_SUBLANES or separation >= full_size:
    # we need to permute within tiles
    axis = int(separation >= full_size)
    intra_tile_separation = (
      separation if axis == 0 else ((separation * batch_size) // full_size)
    )

    # Compute is_descending for each tile based on bitonic pattern
    tile_local_offset = iota_tile(0) + (iota_tile(1) // batch_size) * full_size
    is_right_half = create_bit_indicator(
      log2(intra_tile_separation), iota_tile(axis)
    )
    permutation = jnp.bitwise_xor(iota_tile(axis), intra_tile_separation)
    # Apply permutation to all tiles
    arrs_tiles_permuted = jax.tree.map(
      lambda tile: jnp.take_along_axis(tile, permutation, axis=axis), arrs_tiles
    )

    # Compare and merge with permuted values
    outs_tiles = [[None for _ in t] for t in arrs_tiles]
    for idx, (lefts, rights) in enumerate(
      zip(
        *map(transpose_list_of_lists, (arrs_tiles, arrs_tiles_permuted)),
        strict=True,
      )
    ):
      for arr_idx, out in enumerate(
        compare_and_swap(
          lefts,
          rights,
          is_descending=_compute_is_descending(
            stage=stage,
            tile_start_offset=idx * NUM_SUBLANES,
            tile_local_offset=tile_local_offset,
            sort_dim_offset=sort_dim_offset,
            full_size=full_size,
          )
          if not max_reduce
          else True,
          is_right_half=is_right_half,
        )
      ):
        outs_tiles[arr_idx][idx] = out
  else:
    # Comparison between tiles
    tile_shape = arrs_tiles[0][0].shape
    num_tiles = len(arrs_tiles[0])
    tile_separation = separation // tile_shape[0]

    tile_local_offset = (
      iota_tile(0, tile_shape)
      + (iota_tile(1, tile_shape) // batch_size) * full_size
    )

    outs_tiles = [[None for _ in t] for t in arrs_tiles]
    for i in range(num_tiles // 2):
      idx = compute_pair_slice_start_index(i, separation=tile_separation)
      lefts, rights = (
        transpose_list_of_lists(arrs_tiles)[j]
        for j in (idx, idx + tile_separation)
      )
      for arr_idx, (out_left, out_right) in enumerate(
        compare_and_swap(
          lefts,
          rights,
          is_descending=_compute_is_descending(
            stage=stage,
            tile_start_offset=idx * tile_shape[0],
            tile_local_offset=tile_local_offset,
            sort_dim_offset=sort_dim_offset,
            full_size=full_size,
          )
          if not max_reduce
          else True,
        )
      ):
        outs_tiles[arr_idx][idx] = out_left
        if not max_reduce:
          outs_tiles[arr_idx][idx + tile_separation] = out_right
  if max_reduce:
    # remove the Nones, the lower half we discard for top-k usage
    outs_tiles = [
      [v for v in out_tiles if v is not None] for out_tiles in outs_tiles
    ]
  assert all(not any(v is None for v in out_tiles) for out_tiles in outs_tiles)
  return outs_tiles

def bitonic_topk_refs(input_refs, outputs_refs):
  """Top-k refs kernel using bitonic_topk_arrays."""
  k = outputs_refs[0].shape[1]
  outputs = bitonic_topk_arrays(
    [ref[...] for ref in input_refs], k=k, axis=1
  )
  for ref, output in zip(outputs_refs, outputs, strict=True):
    ref[...] = output

@functools.partial(jax.jit, static_argnames=("k", "interpret"))
def bitonic_topk(operands, k: int,      interpret=False):
  operands = jax.tree.leaves(operands)
  out_shape = (operands[0].shape[0], k)
  return pl.pallas_call(
    bitonic_topk_refs,
    out_shape=([
      jax.ShapeDtypeStruct(out_shape, x.dtype) for x in operands],),
    compiler_params=pltpu.CompilerParams(
      vmem_limit_bytes=int(0.9 * 2**27)
    ),
    interpret=interpret,
  )(operands)[0]

