import dataclasses
import functools
from typing import Callable, Optional, Tuple
import jax
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp


def debug_print(debug_mode, msg, *args):
  if debug_mode:
    pl.debug_print(msg, *args)


def inner_kernel(
    tiled_in_ref, tiled_out_ref, *, sorted_token_indices_ref, debug_mode
):
  tile_in = tiled_in_ref.shape[0]
  tile_out = tiled_out_ref.shape[0]

  dtype = tiled_out_ref.dtype
  out_id = pl.program_id(0)
  in_id = pl.program_id(1)

  num_in_tiles = pl.num_programs(1)
  multiple_in_tiles = num_in_tiles != 1
  debug_print(debug_mode, "=== debug_print === out_id={}, num_out_tiles={}, in_id={}, num_in_tiles={}", out_id, pl.num_programs(0), in_id, num_in_tiles)

  def _main(first_in_tile: bool):
    if first_in_tile and multiple_in_tiles:
      tiled_out_ref[...] = jnp.zeros_like(tiled_out_ref)

    num_sublanes = pltpu.get_tpu_info().num_sublanes
    num_sublanes *= 32 // jax.dtypes.itemsize_bits(dtype)

    out_offset = out_id * tile_out
    in_offset = in_id * tile_in

    for start in range(0, tile_out, num_sublanes):
      end = start + num_sublanes

      vals_list = []
      for i in range(start, end):
        in_row = sorted_token_indices_ref[out_offset + i] - in_offset
        debug_print(debug_mode, "=== debug_print === in_row={}", in_row)

        if multiple_in_tiles:
          mask = jnp.logical_and(0 <= in_row, in_row < tile_in)
          in_row = jnp.clip(in_row, 0, tile_in - 1)
          val = jnp.where(mask, tiled_in_ref[in_row, :], 0)
        else:
          val = tiled_in_ref[in_row, :]
        vals_list.append(val)
      vals_concat = jnp.concat(vals_list, axis=0).astype(dtype)

      if multiple_in_tiles:
        # Notice that tile_out_ref is a Pallas Ref, so += means scatter-add.
        # This is not python list which += means concatenation.
        tiled_out_ref[start:end, :] += vals_concat
      else:
        tiled_out_ref[start:end, :] = vals_concat

  is_first_in_tile = in_id == 0
  lax.cond(
      is_first_in_tile,
      functools.partial(_main, True),
      functools.partial(_main, False),
  )


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
  num_tokens = x_ref.shape[0]
  num_out_tokens = sorted_token_indices_ref.shape[0]

  num_out_tiles = num_out_tokens // tile_out
  num_in_tiles = num_tokens // tile_in

  pltpu.emit_pipeline(
      functools.partial(
          inner_kernel,
          sorted_token_indices_ref=sorted_token_indices_ref,
          debug_mode=debug_mode,
      ),
      grid=(num_out_tiles, num_in_tiles),
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
  out_shape = (num_tokens * topk, hidden_size)

  orig_dtype = x.dtype
  x = x.astype(jnp.float32)
  x = x.reshape(num_tokens, 1, hidden_size)

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

  scope_name = f"sort_tokens-m_{num_tokens}-k_{hidden_size}-topk_{topk}"
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
  )(sorted_token_indices, x)

  return x_sorted, group_sizes, topk_argsort_revert_indices
