"""Utility functions for TPU-optimized operations.

This module provides common utility functions, constants, and helpers used
throughout the Tallax library for TPU-specific operations including tiling,
padding, data format conversions, and loop constructs.
"""

import functools
import inspect
import math
import warnings
from itertools import chain
import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl


# TPU hardware constants
NUM_SUBLANES = 8
NUM_LANES = 128


def is_cpu_platform():
  """Check if code is running on CPU platform.

  Returns:
    True if running on CPU, False otherwise. Emits warning if on CPU.
  """
  is_cpu = jax.default_backend() == "cpu"
  if is_cpu:
    warnings.warn("Running on CPU, interpret=True will be used.")
  return is_cpu


def log2(x: int) -> int:
  """Returns ceiling of log2(x)."""
  if x == 0:
    return 0
  return math.ceil(math.log2(x))


def flatten(xs):
  """Flatten a nested list by one level."""
  return list(chain.from_iterable(xs))


def ceil_multiple(i, n):
  """Round up i to the nearest multiple of n."""
  return pl.cdiv(i, n) * n


def get_dtype_info(x):
  """Get finfo or iinfo for array dtype."""
  dtype = x.dtype
  if jnp.issubdtype(dtype, jnp.floating):
    return jnp.finfo(dtype)
  elif jnp.issubdtype(dtype, jnp.integer):
    return jnp.iinfo(dtype)
  else:
    raise ValueError("Only int and float supported")


def pad(
  arr: jax.Array,
  block_shape: tuple[int | str, ...] = None,
  prepend: bool | tuple[bool, ...] = False,
  val="max_nan",
) -> jax.Array:
  """Pad array to satisfy alignment requirements.

  Args:
    arr: Input array to pad.
    block_shape: Target block shape for each dimension. Can be:
      - int: Pad to be multiple of this value
      - 'power_of_2_lanes': Pad to next power of 2 (at least NUM_LANES)
      Defaults to (NUM_SUBLANES, NUM_LANES).
    prepend: Whether to prepend (True) or append (False) padding.
      Can be a single bool or tuple of bools for each dimension.
    val: Padding value. If None, uses max value (or nan) for sorting.

  Returns:
    Padded array.
  """
  # Handle default block_shape
  if block_shape is None:
    block_shape = (NUM_SUBLANES, NUM_LANES)

  if len(block_shape) != arr.ndim:
    raise ValueError(
      f"block_shape length {len(block_shape)} must match array ndim {arr.ndim}"
    )

  # Normalize prepend to tuple
  if isinstance(prepend, bool):
    prepend = (prepend,) * arr.ndim

  if len(prepend) != arr.ndim:
    raise ValueError(
      f"prepend length {len(prepend)} must match array ndim {arr.ndim}"
    )

  # Calculate padding for each dimension
  pad_widths = []
  for i, (dim_size, block_spec) in enumerate(zip(arr.shape, block_shape)):
    if block_spec == "power_of_2_lanes":
      target_size = max(2 ** log2(dim_size), NUM_LANES)
    elif isinstance(block_spec, int):
      target_size = pl.cdiv(dim_size, block_spec) * block_spec
    else:
      raise ValueError(f"Invalid block_shape element: {block_spec}")

    pad_size = target_size - dim_size
    if prepend[i]:
      pad_widths.append((pad_size, 0))
    else:
      pad_widths.append((0, pad_size))

  # Determine padding value
  if isinstance(val, str):
    info = get_dtype_info(arr)
    if val == "min":
      pad_val = info.min
    elif val == "max":
      pad_val = info.max
    elif val == "max_nan":
      pad_val = info.max
      if jnp.issubdtype(arr.dtype, jnp.floating):
        pad_val = jnp.nan
    else:
      raise ValueError
  else:
    pad_val = val

  # Return early if no padding needed
  if all(w == (0, 0) for w in pad_widths):
    return arr

  return jnp.pad(arr, pad_widths, mode="constant", constant_values=pad_val)


def standardize(x, nans=True, zeros=True):
  """Standardize float values for sorting.

  Converts NaNs to a specific value and normalizes +/-0.
  """
  if nans:
    nan_val = sortable_int_to_float(jnp.iinfo(jnp.int32).max - 1)
    x = jnp.where(jnp.isnan(x), nan_val, x)
  if zeros:
    x = jnp.where(x == 0, 0, x)
  return x


def to_32bit_dtype(operand_dtype):
  """Convert dtype to corresponding 32-bit dtype."""
  for dtype_class, dtype_32bit in {
    jnp.floating: jnp.float32,
    jnp.integer: jnp.int32,
    jnp.bool_: jnp.int32,
  }.items():
    if jnp.issubdtype(operand_dtype, dtype_class):
      return dtype_32bit
  raise ValueError("dtype not recognized")


def canonicalize_operand(operand):
  """Convert operand to list of arrays and validate shapes."""
  operands = jax.tree.leaves(operand)
  shapes = [x.shape for x in operands]
  if len(set(shapes)) != 1:
    raise ValueError(
      f"Inputs must all have the same shape, but found {shapes=}"
    )
  shape = shapes[0]
  if len(shape) != 2:
    raise ValueError("Only 2D inputs supported")
  return operands, shape


### Float-Int Conversion for Sortable Representation


def float_to_sortable_int(
  x: jnp.ndarray, standardize_nans=True, standardize_zeros=True
) -> jnp.ndarray:
  """Transform float32 bits into sortable int32 representation.

  Negative floats map to [INT_MIN, -1] with reversed order.
  Positive floats map to [0, INT_MAX].

  Standardization applied of NaNs to i32.max-1,
    unstandardized they are values near either i32.max or i32.min after conversion.
  Standardisation of the +0 and -0 of f32 into i32, critical for stable sorts with jax.lax.sort behavior of treating as equivalent
  """
  x = standardize(
    x.astype(jnp.float32), nans=standardize_nans, zeros=standardize_zeros
  )
  i = x.view(jnp.int32)
  return jnp.where(i < 0, i ^ 0x7FFFFFFF, i)


def sortable_int_to_float(i: jnp.ndarray) -> jnp.ndarray:
  """Inverse transformation from sortable int32 back to float32."""
  return jnp.where(i < 0, i ^ 0x7FFFFFFF, i).view(jnp.float32)


### BF16-U16 Packing for Optimization


def pack_bf16_u16_to_i32(val, index, stable=True):
  """Pack bfloat16 value and uint16 index into f32 then convert to sortable int32.

  BF16 in F32 has empty lower 16 bits where we pack the index.
  This allows sorting while preserving original indices.
  stable=True standardizes the bit patterns of NaNs and zeros, and packs the int array so on value ties the lower index element is larger in comparisons.
  """
  assert index.dtype == jnp.int32
  val = val.astype(jnp.float32)
  if stable:
    val = standardize(val)
    # ensure stable sort of indices
    index = jnp.where(val < 0, 2**16 - 1 - index, index)
  return float_to_sortable_int(
    ((val.view(jnp.int32) & ~0xFFFF) | index).view(jnp.float32),
    standardize_nans=False,
    standardize_zeros=False,
  )


def unpack_bf16_u16_from_i32(packed, stable=True):
  """Extract original bfloat16 value and uint16 index from packed int32."""
  assert packed.dtype == jnp.int32, f"found {packed.dtype}"
  packed = sortable_int_to_float(packed)
  val = (
    (packed.view(jnp.int32) & ~0xFFFF).view(jnp.float32).astype(jnp.bfloat16)
  )
  index = packed.view(jnp.int32) & 0xFFFF
  if stable:
    # reverse the int mapping required for stable sort
    index = jnp.where(val.astype(jnp.float32) < 0, 2**16 - 1 - index, index)
  return val, index


### Tile Operations


def split_array_to_tiles(arr, tile_shape=(NUM_SUBLANES, NUM_LANES)):
  """Split 2D array into flat list of tiles with specified shape.

  Args:
    arr: 2D array to split
    tile_shape: Shape of each tile (tile_dim0, tile_dim1), defaults to (NUM_SUBLANES, NUM_LANES)

  Returns:
    List of tiles in row-major order
  """
  tile_dim0, tile_dim1 = tile_shape
  tile_rows = arr.shape[0] // tile_dim0
  tile_cols = arr.shape[1] // tile_dim1
  assert arr.shape[0] % tile_dim0 == 0, (
    f"Array dim0 {arr.shape[0]} not divisible by tile_dim0 {tile_dim0}"
  )
  assert arr.shape[1] % tile_dim1 == 0, (
    f"Array dim1 {arr.shape[1]} not divisible by tile_dim1 {tile_dim1}"
  )

  # Use jnp.split for efficient tile extraction
  return flatten([
    jnp.split(row_strip, tile_cols, axis=1)
    for row_strip in jnp.split(arr, tile_rows, axis=0)
  ])


def join_tiles_to_array(tiles, dim0):
  """Reconstruct 2D array from flat list of tiles."""
  num_tiles = len(tiles)
  tile_rows, tile_cols = tiles[0].shape
  num_rows = dim0
  num_cols = (num_tiles * tile_rows * tile_cols) // dim0
  grid_cols = num_cols // tile_cols

  rows = []
  for i in range(len(tiles) // grid_cols):
    row_tiles = tiles[i * grid_cols : (i + 1) * grid_cols]
    rows.append(jnp.concatenate(row_tiles, axis=-1))

  return jnp.concatenate(rows, axis=-2)


def iota_tile(dim, tile_shape=(NUM_SUBLANES, NUM_LANES)):
  """Create iota array with tile shape."""
  return lax.broadcasted_iota(jnp.int32, tile_shape, dim)


def create_bit_indicator(bit_position: int, index):
  """Returns if the bit in bit_position is 1 or 0."""
  return (index & (1 << bit_position)) > 0


def to_compressed_transpose_format(arr):
  """Convert array to sublane-oriented format for faster permutes.

  For small arrays where dim0 <= NUM_LANES, pads dim1 to NUM_LANES before
  transformation, then unpads to the correct transposed size.
  """
  dim0, original_dim1 = arr.shape
  assert NUM_LANES % dim0 == 0 and dim0 <= NUM_LANES

  # Pad dim1 to NUM_LANES if needed (use 1 to avoid padding dim0)
  if original_dim1 < NUM_LANES:
    arr = pad(arr, block_shape=(1, NUM_LANES))

  arrs = jnp.split(arr, NUM_LANES // dim0, axis=1)
  arr = jnp.concatenate(arrs, axis=0).T

  # Unpad to the correct transposed size
  if original_dim1 < NUM_LANES:
    arr = arr[:original_dim1, :]

  return arr


def from_compressed_transpose_format(tiles, dim0):
  """Convert from compressed transpose format back to original layout.

  Inverse of to_compressed_transpose_format. Pads dim0 to NUM_LANES before
  transformation, then unpads dim1 to the correct final size.
  """
  assert NUM_LANES % dim0 == 0 and dim0 <= NUM_LANES
  arr = jnp.concatenate(tiles, axis=0)
  original_dim1 = arr.shape[0]

  # Pad dim0 to NUM_LANES if needed (use 1 to avoid padding dim1)
  if original_dim1 < NUM_LANES:
    arr = pad(arr, block_shape=(NUM_LANES, 1))

  arr = arr.T
  assert arr.shape[0] == NUM_LANES
  arrs = jnp.split(arr, arr.shape[0] // dim0, axis=0)
  arr = jnp.concatenate(arrs, axis=1)

  # Unpad dim1 to the correct final size
  if original_dim1 < NUM_LANES:
    arr = arr[:, :original_dim1]

  return arr


### Loop Utilities


def unrolled_fori_loop(length: int, body_fn, init_val, unroll: int):
  """Execute for loop with manual unrolling for better performance."""
  if length <= 0:
    return init_val
  unroll = min(length, unroll)

  def unrolled_body(i, carry):
    i *= unroll
    for j in range(unroll):
      carry = body_fn(i + j, carry)
    return carry

  carry = jax.lax.fori_loop(0, length // unroll, unrolled_body, init_val)
  for j in range(length % unroll):
    carry = body_fn((length // unroll) * unroll + j, carry)
  return carry


def transpose_list_of_lists(tree):
  """Transpose nested list structure."""
  outer = jax.tree.structure(type(tree)("*") * len(tree))
  inner = jax.tree.structure(type(tree[0])("*") * len(tree[0]))
  return jax.tree.transpose(outer, inner, tree)


def map_batch_dim_to_smaller_than_hardware_tile_size(
  unsplit_f, num_args=1, max_batch_size=None
):
  """Decorator to handle chunking in the batch dimension"""
  # Get the default value for 'axis' from the original function
  sig = inspect.signature(unsplit_f)
  assert "axis" in sig.parameters, (
    f"Function {unsplit_f.__name__} must have 'axis' parameter"
  )
  axis_default = sig.parameters["axis"].default

  @functools.wraps(unsplit_f)
  def split_f(*args, axis=axis_default, **kwargs):
    if len(args) < num_args:
      raise ValueError(
        f"Please pass the first {num_args} as args rather than kwargs to {unsplit_f.__name__}"
      )
    batch_axis = 1 - axis
    batch_size = jax.tree.leaves(args[0])[0].shape[batch_axis]
    # split so the batch axis matches hardware sizes
    # make inputs (NUM_SUBLANES, *) if f on axis 1,
    # or (*, NUM_LANES) if f axis 0
    max_chunk_size = (
      (NUM_SUBLANES, NUM_LANES)[batch_axis]
      if max_batch_size is None
      else max_batch_size
    )
    # Generate split indices, excluding any that equal batch_size to avoid empty arrays
    split_indices = tuple(
      (i + 1) * max_chunk_size
      for i in range(batch_size // max_chunk_size)
      if (i + 1) * max_chunk_size != batch_size
    )

    # If no splits needed, call directly
    if not split_indices:
      return unsplit_f(*args, axis=axis, **kwargs)

    flat_args_to_chunk, treedef = jax.tree.flatten(args[:num_args])

    flat_chunks = transpose_list_of_lists(
      jax.tree.map(
        lambda arr: jnp.split(arr, split_indices, axis=batch_axis),
        flat_args_to_chunk,
      )
    )
    chunks_outputs = [
      unsplit_f(
        *jax.tree.unflatten(treedef, flat_chunk),
        *args[num_args:],
        axis=axis,
        **kwargs,
      )
      for flat_chunk in flat_chunks
    ]
    treedef = jax.tree.structure(chunks_outputs[0])
    flat_chunks_outputs = list(map(jax.tree.leaves, chunks_outputs))
    return jax.tree.unflatten(
      treedef,
      [
        jnp.concatenate(output_chunks, axis=batch_axis)
        for output_chunks in transpose_list_of_lists(flat_chunks_outputs)
      ],
    )

  return split_f
