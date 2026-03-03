import functools
from typing import Sequence

from absl import flags
import jax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.experimental.pallas import tpu_sc as plsc
import jax.numpy as jnp
import jaxtyping as jt
import typeguard

@jax.jit
def gather(data, indices):
  num_indices = indices.shape[0]
  value_dim = data.shape[1]
  gather_window_size = 16
  vector_mesh = plsc.VectorSubcoreMesh(
      core_axis_name="core",
      subcore_axis_name="subcore",
  )

  indices = indices.reshape((1, num_indices))
  @pl.kernel(out_shape=jax.ShapeDtypeStruct((num_indices, value_dim), data.dtype),
             mesh=vector_mesh)
  def kernel(x_hbm, i_hbm, o_hbm):
    def body(i_vmem, o_vmem):
      pltpu.sync_copy(x_hbm.at[i_vmem.at[0]], o_vmem)  # The gather op

    pltpu.emit_pipeline(
        body,
        grid=(num_indices // gather_window_size,),
        in_specs=[pl.BlockSpec((1, gather_window_size),
                               index_map=lambda i: (0, i))],
        out_specs=[pl.BlockSpec((gather_window_size, value_dim),
                                index_map=lambda i: (i, 0))],
        core_axis_name='subcore',
        dimension_semantics=(pltpu.PARALLEL,),
    )(i_hbm, o_hbm)

  return kernel(data, indices)


@jax.jit
def ragged_gather(data, indices, ep_token_start, ep_token_end):
  """Gather with ragged range: only gathers for indices in [ep_token_start, ep_token_end).

  Skips DMA copies for tiles whose index range falls entirely outside
  [ep_token_start, ep_token_end), reducing HBM bandwidth by ~ep_size x.
  Out-of-range positions are left uninitialized (zeroed downstream by topk_weights).
  """
  num_indices = indices.shape[0]
  value_dim = data.shape[1]
  gather_window_size = 16
  num_tiles = num_indices // gather_window_size

  vector_mesh = plsc.VectorSubcoreMesh(
      core_axis_name="core",
      subcore_axis_name="subcore",
  )

  # Precompute per-tile mask: 1 if tile overlaps [start, end), 0 otherwise
  tile_starts = jnp.arange(num_tiles) * gather_window_size
  tile_mask = ((tile_starts + gather_window_size) > ep_token_start) & (
      tile_starts < ep_token_end)
  tile_mask = tile_mask.astype(jnp.int32).reshape(1, num_tiles)

  indices = indices.reshape((1, num_indices))

  @pl.kernel(out_shape=jax.ShapeDtypeStruct((num_indices, value_dim),
                                             data.dtype),
             mesh=vector_mesh)
  def kernel(x_hbm, i_hbm, mask_hbm, o_hbm):
    def body(i_vmem, mask_vmem, o_vmem):
      should_gather = mask_vmem[0, 0] > 0

      @pl.when(should_gather)
      def _():
        pltpu.sync_copy(x_hbm.at[i_vmem.at[0]], o_vmem)

    pltpu.emit_pipeline(
        body,
        grid=(num_tiles,),
        in_specs=[
            pl.BlockSpec((1, gather_window_size), index_map=lambda i: (0, i)),
            pl.BlockSpec((1, 1), index_map=lambda i: (0, i)),
        ],
        out_specs=[pl.BlockSpec((gather_window_size, value_dim),
                                index_map=lambda i: (i, 0))],
        core_axis_name='subcore',
        dimension_semantics=(pltpu.PARALLEL,),
    )(i_hbm, mask_hbm, o_hbm)

  return kernel(data, indices, tile_mask)
