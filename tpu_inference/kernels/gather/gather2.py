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
def ragged_gather(data, indices, ep_range):
  """Gather with ragged range: only gathers for indices in [ep_range[0], ep_range[1]).

  Args:
    data: [batch_size, value_dim] source data.
    indices: [num_indices] gather indices.
    ep_range: i32[2] array where ep_range[0] = ep_token_start, ep_range[1] = ep_token_end.

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

  ep_start = ep_range[0].reshape(1, 1)
  ep_end = ep_range[1].reshape(1, 1)
  indices = indices.reshape((1, num_indices))

  @pl.kernel(out_shape=jax.ShapeDtypeStruct((num_indices, value_dim),
                                             data.dtype),
             mesh=vector_mesh)
  def kernel(x_hbm, i_hbm, ep_start_hbm, ep_end_hbm, o_hbm):
    def body(i_vmem, ep_start_vmem, ep_end_vmem, o_vmem):
      tile_idx = pl.program_id(0)
      tile_start = tile_idx * gather_window_size
      token_start = ep_start_vmem[0, 0]
      token_end = ep_end_vmem[0, 0]
      should_gather = ((tile_start + gather_window_size) > token_start) & (
          tile_start < token_end)

      @pl.when(should_gather)
      def _():
        pltpu.sync_copy(x_hbm.at[i_vmem.at[0]], o_vmem)

    pltpu.emit_pipeline(
        body,
        grid=(num_tiles,),
        in_specs=[
            pl.BlockSpec((1, gather_window_size), index_map=lambda i: (0, i)),
            pl.BlockSpec((1, 1), index_map=lambda i: (0, 0)),
            pl.BlockSpec((1, 1), index_map=lambda i: (0, 0)),
        ],
        out_specs=[pl.BlockSpec((gather_window_size, value_dim),
                                index_map=lambda i: (i, 0))],
        core_axis_name='subcore',
        dimension_semantics=(pltpu.PARALLEL,),
    )(i_hbm, ep_start_hbm, ep_end_hbm, o_hbm)

  return kernel(data, indices, ep_start, ep_end)
