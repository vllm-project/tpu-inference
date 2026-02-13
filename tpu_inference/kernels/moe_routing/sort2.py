"""
Idea of sort2.py.

The problem of original Pallas kernel sort.py is that in the `inner_kernel`, we use a 2D grid (num_out_tiles, num_in_tiles):
for out_tile_id in range(num_out_tiles):
  for in_tile_id in range(num_in_tiles):
    # inside the kernel
    for output_row in range(all_output_rows):
      # find the source row

For each output_row, it checks all in_tile_id (num_in_tiles times). However, the source row of the output_row can only exist in one of the num_in_tiles, so we are wasting VPU ops.

IOW, for each (out_tile, in_tile) pair, we process all tile_out output rows, even though most rows' source data is NOT in the current input tile.

Idea:
Calculate a bucket metadata before the kernel starts. The bucket records which output rows map to which input tile. Then we pass the bucket metadata to the kernel so that each (out_tile, in_tile) only touches the rows that actually belong. The bucket records which output rows map to which input tile. Then we pass the bucket metadata to the kernel so that each (out_tile, in_tile) only touches the rows that actually belong.
"""
import functools
from typing import Optional, Tuple

import jax
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp


def debug_print(debug_mode, msg, *args):
  if debug_mode:
    pl.debug_print(msg, *args)


# Fast path: original kernel for num_in_tiles == 1
def inner_kernel(
    tiled_in_ref, tiled_out_ref, *, sorted_token_indices_ref, debug_mode
):
  tile_out = tiled_out_ref.shape[0]
  dtype = tiled_out_ref.dtype
  out_id = pl.program_id(0)

  num_sublanes = pltpu.get_tpu_info().num_sublanes
  num_sublanes *= 32 // jax.dtypes.itemsize_bits(dtype)

  out_offset = out_id * tile_out

  for start in range(0, tile_out, num_sublanes):
    end = start + num_sublanes
    vals_list = []
    for i in range(start, end):
      in_row = sorted_token_indices_ref[out_offset + i]
      val = tiled_in_ref[in_row, :]
      vals_list.append(val)
    vals_concat = jnp.concat(vals_list, axis=0).astype(dtype)
    tiled_out_ref[start:end, :] = vals_concat


def gather_kernel(
    # Prefetch
    sorted_token_indices_ref: jax.Array,
    # In
    x_ref: jax.Array,
    # Out
    x_sorted_ref: jax.Array,
    *,
    tile_out: int,
    tile_in: int,
    debug_mode: bool,
):
  hidden_size = x_ref.shape[-1]
  num_out_tokens = sorted_token_indices_ref.shape[0]
  num_out_tiles = num_out_tokens // tile_out

  pltpu.emit_pipeline(
      functools.partial(
          inner_kernel,
          sorted_token_indices_ref=sorted_token_indices_ref,
          debug_mode=debug_mode,
      ),
      grid=(num_out_tiles, 1),
      in_specs=[
          pl.BlockSpec(
              (tile_in, 1, hidden_size),
              lambda i, j: (j, 0, 0),
              pipeline_mode=pl.Buffered(buffer_count=2),
          ),
      ],
      out_specs=[
          pl.BlockSpec(
              (tile_out, hidden_size),
              lambda i, j: (i, 0),
              pipeline_mode=pl.Buffered(buffer_count=2),
          ),
      ],
  )(x_ref, x_sorted_ref)


# Bucketed path: pre-bucketed kernel for num_in_tiles > 1
#
# Instead of iterating all tile_out rows per (out_tile, in_tile) and
# masking irrelevant ones, we pre-classify each output row by its
# source input tile. The kernel then processes only the rows that
# belong to the current (out_tile, in_tile) bucket.
def inner_kernel_bucketed(
    tiled_in_ref, tiled_out_ref, *,
    bucketed_src_rows_ref,
    bucketed_dst_rows_ref,
    bucket_starts_ref,
    bucket_sizes_ref,
    num_in_tiles,
    debug_mode,
):
  tile_out = tiled_out_ref.shape[0]
  dtype = tiled_out_ref.dtype
  out_id = pl.program_id(0)
  in_id = pl.program_id(1)

  # Look up bucket info for this (out_tile, in_tile) pair.
  bucket_idx = out_id * num_in_tiles + in_id
  bucket_start = bucket_starts_ref[bucket_idx]
  bucket_size = bucket_sizes_ref[bucket_idx]

  # Base offset into bucketed arrays for this output tile.
  tile_base = out_id * tile_out

  debug_print(
      debug_mode,
      "=== bucketed === out_id={}, in_id={}, bucket_start={}, bucket_size={}",
      out_id, in_id, bucket_start, bucket_size,
  )

  def _main(first_in_tile: bool):
    if first_in_tile:
      tiled_out_ref[...] = jnp.zeros_like(tiled_out_ref)

    def body_fn(k, carry):
      src_row = bucketed_src_rows_ref[tile_base + bucket_start + k]
      dst_row = bucketed_dst_rows_ref[tile_base + bucket_start + k]
      debug_print(debug_mode, "  k={}, src_row={}, dst_row={}", k, src_row, dst_row)
      tiled_out_ref[dst_row, :] = tiled_in_ref[src_row, :].astype(dtype)
      return carry

    lax.fori_loop(0, bucket_size, body_fn, jnp.int32(0))

  is_first_in_tile = in_id == 0
  lax.cond(
      is_first_in_tile,
      functools.partial(_main, True),
      functools.partial(_main, False),
  )


def gather_kernel_bucketed(
    # Prefetch
    bucketed_src_rows_ref: jax.Array,
    bucketed_dst_rows_ref: jax.Array,
    bucket_starts_ref: jax.Array,
    bucket_sizes_ref: jax.Array,
    # In
    x_ref: jax.Array,
    # Out
    x_sorted_ref: jax.Array,
    *,
    tile_out: int,
    tile_in: int,
    debug_mode: bool,
):
  hidden_size = x_ref.shape[-1]
  num_tokens = x_ref.shape[0]
  num_out_tokens = bucketed_src_rows_ref.shape[0]
  num_out_tiles = num_out_tokens // tile_out
  num_in_tiles = num_tokens // tile_in

  pltpu.emit_pipeline(
      functools.partial(
          inner_kernel_bucketed,
          bucketed_src_rows_ref=bucketed_src_rows_ref,
          bucketed_dst_rows_ref=bucketed_dst_rows_ref,
          bucket_starts_ref=bucket_starts_ref,
          bucket_sizes_ref=bucket_sizes_ref,
          num_in_tiles=num_in_tiles,
          debug_mode=debug_mode,
      ),
      grid=(num_out_tiles, num_in_tiles),
      in_specs=[
          pl.BlockSpec(
              (tile_in, hidden_size),
              lambda i, j: (j, 0),
              pipeline_mode=pl.Buffered(buffer_count=2),
          ),
      ],
      out_specs=[
          pl.BlockSpec(
              (tile_out, hidden_size),
              lambda i, j: (i, 0),
              pipeline_mode=pl.Buffered(buffer_count=2),
          ),
      ],
  )(x_ref, x_sorted_ref)


@jax.jit(static_argnames=["num_experts", "tile_out", "tile_in", "debug_mode"])
def sort_tokens(
    x: jax.Array,
    topk_indices: jax.Array,
    num_experts: int,
    *,
    tile_out: Optional[int] = None,
    tile_in: Optional[int] = None,
    debug_mode: bool = False,
) -> Tuple[jax.Array, jax.Array, jax.Array]:
  vmem_limit_bytes = 100 * 1024 * 1024  # 100 MB
  num_tokens, hidden_size = x.shape
  topk = topk_indices.shape[1]
  num_out_tokens = num_tokens * topk
  out_shape = (num_out_tokens, hidden_size)

  orig_dtype = x.dtype
  x = x.astype(jnp.float32)

  # Compute sorting indices and group sizes in JAX.
  topk_indices_flat = topk_indices.flatten()
  topk_argsort_indices = jnp.argsort(topk_indices_flat)
  topk_argsort_revert_indices = jnp.argsort(topk_argsort_indices)
  token_indices = jnp.arange(num_tokens, dtype=jnp.int32).repeat(topk)
  sorted_token_indices = token_indices[topk_argsort_indices]
  group_sizes = jnp.bincount(topk_indices_flat, length=num_experts).astype(jnp.int32)

  default_tile_out = 512
  default_tile_in = 512
  if num_tokens == 64:
    default_tile_out = 256
    default_tile_in = 64
  elif num_tokens == 8:
    default_tile_out = 16
    default_tile_in = 8

  if tile_out is None:
    tile_out = default_tile_out
  if tile_in is None:
    tile_in = default_tile_in

  num_in_tiles = num_tokens // tile_in

  scope_name = f"sort_tokens-m_{num_tokens}-k_{hidden_size}-topk_{topk}"

  if num_in_tiles <= 1:
    # Fast path: single input tile, no bucketing needed.
    x_3d = x.reshape(num_tokens, 1, hidden_size)
    (x_sorted,) = pl.pallas_call(
        functools.partial(gather_kernel, tile_out=tile_out, tile_in=tile_in, debug_mode=debug_mode),
        out_shape=[
            jax.ShapeDtypeStruct(out_shape, orig_dtype),
        ],
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=1,
            in_specs=[pl.BlockSpec(memory_space=pltpu.HBM)],
            out_specs=[
                pl.BlockSpec(memory_space=pltpu.HBM),
            ],
        ),
        compiler_params=pltpu.CompilerParams(vmem_limit_bytes=vmem_limit_bytes),
        name=scope_name,
    )(sorted_token_indices, x_3d)
  else:
    #Pre-bucketing
    num_out_tiles = num_out_tokens // tile_out 
    source_tile_id = sorted_token_indices // tile_in # [num_tokens*topk] which input tile holds the source.
    source_in_row = (sorted_token_indices % tile_in).astype(jnp.int32) # [num_tokens*topk] row offset within that input tile.

    # Map each output row to its output tile
    out_tile_ids = jnp.arange(num_out_tokens, dtype=jnp.int32) // tile_out # [num_tokens*topk]  which output tile
    position_in_out_tile = (jnp.arange(num_out_tokens, dtype=jnp.int32) % tile_out).astype(jnp.int32) # [num_tokens*topk] row within output tile

    # This creates a composite sort key that groups rows first by output tile, then by source input tile within each output tile. After sorting, all rows belonging to the same (out_tile, in_tile) pair are contiguous
    sort_key = out_tile_ids * num_in_tiles + source_tile_id # [num_tokens*topk]
    order = jnp.argsort(sort_key, stable=True) # [num_tokens*topk]

    # the source_in_row values, reordered so within each output tile, all rows needing input tile 0 come first, then input tile 1, etc.
    bucketed_src_rows = source_in_row[order] # source row within input tile (for reads)
    # the original position within the output tile (needed because we reordered, so writes become scattered)
    bucketed_dst_rows = position_in_out_tile[order] # destination row within output tile (for writes)
    # These are what the kernel will use: for each work item, read from src_row in the current input tile and write to dst_row in the current output tile.

    # Compute bucket boundaries: for each (out_tile, in_tile) pair,
    # the start offset (relative to tile start) and count.
    combined_idx = (out_tile_ids[order] * num_in_tiles + source_tile_id[order]).astype(jnp.int32)
    bucket_sizes_flat = jnp.bincount(
        combined_idx, length=num_out_tiles * num_in_tiles
    ).astype(jnp.int32)
    # Exclusive prefix sum gives start offsets (global).
    # for (out_tile o, in_tile t), the relevant work items are at indices [bucket_boundaries[o, t], bucket_boundaries[o, t+1]) relative to the output tile's section.
    bucket_starts_flat = (jnp.cumsum(bucket_sizes_flat) - bucket_sizes_flat).astype(jnp.int32)
    # bucket_sizes_flat[bucket_idx] tells how many rows belong to that (out_tile, in_tile) pair. bucket_starts_flat gives the global start offset.

    # Make starts relative to each output tile's section.
    tile_offsets = jnp.repeat(
        jnp.arange(num_out_tiles, dtype=jnp.int32) * tile_out, num_in_tiles
    )
    bucket_starts_flat = bucket_starts_flat - tile_offsets

    # Input is 2D for bucketed path (no extra dim needed).
    (x_sorted,) = pl.pallas_call(
        functools.partial(gather_kernel_bucketed, tile_out=tile_out, tile_in=tile_in, debug_mode=debug_mode),
        out_shape=[
            jax.ShapeDtypeStruct(out_shape, jnp.float32),
        ],
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=4,
            in_specs=[pl.BlockSpec(memory_space=pltpu.HBM)],
            out_specs=[
                pl.BlockSpec(memory_space=pltpu.HBM),
            ],
        ),
        compiler_params=pltpu.CompilerParams(vmem_limit_bytes=vmem_limit_bytes),
        name=scope_name,
    )(bucketed_src_rows, bucketed_dst_rows, bucket_starts_flat, bucket_sizes_flat, x)
    x_sorted = x_sorted.astype(orig_dtype)

  return x_sorted, group_sizes, topk_argsort_revert_indices
