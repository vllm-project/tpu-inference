import dataclasses
import functools
from typing import Callable, Tuple
import jax
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp


def inner_kernel(
    tiled_in_ref, tiled_out_ref, *, topk_argsort_revert_indices_ref
):
  tile_in = tiled_in_ref.shape[0]
  tile_out = tiled_out_ref.shape[0]

  dtype = tiled_out_ref.dtype
  out_id = pl.program_id(0)
  in_id = pl.program_id(1)

  num_in_tiles = pl.num_programs(1)
  multiple_in_tiles = num_in_tiles != 1

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
        in_row = topk_argsort_revert_indices_ref[out_offset + i] - in_offset

        if multiple_in_tiles:
          mask = jnp.logical_and(0 <= in_row, in_row < tile_in)
          in_row = jnp.clip(in_row, 0, tile_in - 1)
          val = jnp.where(mask, tiled_in_ref[in_row, :], 0)
        else:
          val = tiled_in_ref[in_row, :]
        vals_list.append(val)
      vals_concat = jnp.concat(vals_list, axis=0).astype(dtype)

      if multiple_in_tiles:
        tiled_out_ref[start:end, :] += vals_concat
      else:
        tiled_out_ref[start:end, :] = vals_concat

  is_first_in_tile = in_id == 0
  lax.cond(
      is_first_in_tile,
      functools.partial(_main, True),
      functools.partial(_main, False),
  )


def kernel(
    # Prefetch
    topk_indices_ref: jax.Array,
    # In
    x_ref: jax.Array,
    # Out
    x_sorted_ref: jax.Array,
    group_sizes_ref: jax.Array,
    topk_argsort_revert_indices_ref: jax.Array,
    # Scratch
    group_offset_ref: jax.Array,
    *,
    tile_out: int,
    tile_in: int,
):
  hidden_size = x_ref.shape[-1]
  num_tokens, topk = topk_indices_ref.shape
  num_out_tokens = num_tokens * topk
  num_experts = group_sizes_ref.shape[0]

  for i in range(num_experts):
    group_sizes_ref[i] = 0

  for t in range(num_tokens):
    for k in range(topk):
      expert_idx = topk_indices_ref[t, k]
      group_size = group_sizes_ref[expert_idx]
      group_sizes_ref[expert_idx] = group_size + 1

  curr_offset = 0
  for e in range(num_experts):
    next_offset = curr_offset + group_sizes_ref[e]
    group_offset_ref[e] = next_offset
    curr_offset = next_offset

  for t in range(num_tokens):
    for k in range(topk):
      expert_idx = topk_indices_ref[t, k]
      group_offset = group_offset_ref[expert_idx]
      topk_argsort_revert_indices_ref[group_offset] = t
      group_offset_ref[expert_idx] = group_offset + 1

  num_out_tiles = num_out_tokens // tile_out
  num_in_tiles = num_tokens // tile_in

  pltpu.emit_pipeline(
      functools.partial(
          inner_kernel,
          topk_argsort_revert_indices_ref=topk_argsort_revert_indices_ref,
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


@jax.jit(static_argnames=["num_experts"])
def sort_tokens(
    x: jax.Array, topk_indices: jax.Array, num_experts: int
) -> Tuple[jax.Array, jax.Array, jax.Array]:
  vmem_limit_bytes = int(pltpu.get_tpu_info().vmem_capacity_bytes * 0.8)
  num_tokens, hidden_size = x.shape
  topk = topk_indices.shape[1]
  out_shape = (num_tokens * topk, hidden_size)

  orig_dtype = x.dtype
  x = x.astype(jnp.float32)
  x = x.reshape(num_tokens, 1, hidden_size)

  if num_tokens == 64:
    tile_out = 256
    tile_in = 64
  else:
    tile_out = tile_in = 2048

  scope_name = f"sort_tokens-m_{num_tokens}-k_{hidden_size}-topk_{topk}"
  return pl.pallas_call(
      functools.partial(kernel, tile_out=tile_out, tile_in=tile_in),
      out_shape=[
          jax.ShapeDtypeStruct(out_shape, orig_dtype),
          jax.ShapeDtypeStruct((num_experts,), jnp.int32), # group_sizes
          jax.ShapeDtypeStruct((topk_indices.size,), jnp.int32), # topk_argsort_revert_indices
      ],
      grid_spec=pltpu.PrefetchScalarGridSpec(
          num_scalar_prefetch=1,
          in_specs=[pl.BlockSpec(memory_space=pltpu.HBM)],
          out_specs=[
              pl.BlockSpec(memory_space=pltpu.HBM),
              pl.BlockSpec(memory_space=pltpu.SMEM),
              pl.BlockSpec(memory_space=pltpu.SMEM),
          ],
          scratch_shapes=[
              pltpu.SMEM((num_experts,), jnp.int32),
          ],
      ),
      compiler_params=pltpu.CompilerParams(vmem_limit_bytes=vmem_limit_bytes),
      name=scope_name,
  )(topk_indices, x)
