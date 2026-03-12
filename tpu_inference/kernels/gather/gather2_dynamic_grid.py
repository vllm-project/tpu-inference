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


@functools.partial(jax.jit, static_argnames=("gather_window_size",))
def ragged_gather(data, indices, ep_range, gather_window_size=16):
  """Gather with ragged range: only gathers for indices in [ep_range[0], ep_range[1]).

  Args:
    data: [batch_size, value_dim] source data.
    indices: [num_indices] gather indices.
    ep_range: i32[2] array where ep_range[0] = ep_token_start, ep_range[1] = ep_token_end.

  """
  # TODO(xiowei): should check num_indices%gather_window_size == 0
  num_indices = indices.shape[0]
  batch_size, value_dim = data.shape
  num_tiles = num_indices // gather_window_size
  sc_info = plsc.get_sparse_core_info()
  vector_mesh = plsc.VectorSubcoreMesh(
      num_cores=sc_info.num_cores,
      num_subcores=sc_info.num_subcores,
      core_axis_name="core",
      subcore_axis_name="subcore",
  )
  # xw32q: How should I pass in the "gather_window_size"? Does the closure work?
  # A: Yes, the closure works so you don't need to explicitly pass in the "gather_window_size".
  # xw32q: why do we need to do the reshape "indices.reshape((1, num_indices))"
  # A: actually it's not needed. I've tested it out.
  indices = indices.reshape((1, num_indices))
  # xw32: why do we need to separate the ep_range into ep_range[0] and ep_range[1]?
  # A: The SparseCore backend doesn't support arbitrary byte-offset element access within a VMEM buffer. ep_range_vmem[0, 0] (offset 0) would work, but ep_range_vmem[0, 1] (offset 4 bytes) generates a GEP that the backend can't handle.
  # xw32: why do we need to reshape them into (1, 1)?
  # A: 2 reasons:
  # - 1: DMA operations on vectors instead of scalars: pltpu.async_copy transfers data between HBM and VMEM using the DMA engine, which operates on multi-dimensional tiles — not scalars. A shape-() scalar can't be the source or destination of a DMA copy. The (1, 1) shape is the minimum 2D tile.
  # - 2: SC VMEM refs require vectors: SC VMEM refs require 2D indexing: plsc.MemoryRef allocates SC VMEM buffers and expects a 2D shape. Once allocated, you read from it with ep_start_vmem[0, 0]. A scalar ref (shape ()) wouldn't support this indexing pattern.
  # Also, I tested that 1d also works. So we can reshape them into (1,)
  ep_start = ep_range[0].reshape(1, 1)
  ep_end = ep_range[1].reshape(1, 1)

  @pl.kernel(
    out_shape=jax.ShapeDtypeStruct((num_indices, value_dim), data.dtype),
    mesh=vector_mesh,
  )
  def kernel(x_hbm, i_hbm, ep_start_hbm, ep_end_hbm, o_hbm):
    # xw32: how should I write the pl.core_map?
    # A: We don't necessarily need a pl.core_map. pl.run_scoped can be a standalone function call.
    # xw32q: should I pass x_hbm, i_hbm? Or the closure works?
    # A: the closure works.
    def scoped_func(ep_start_vmem, ep_end_vmem, sem):
      # xw32q: how do I copy the ep_start_hbm to vmem?
      # A: see the code below. You need to allocate vmem using the pl.run_scoped.
      pltpu.async_copy(ep_start_hbm, ep_start_vmem, sem).wait()
      pltpu.async_copy(ep_end_hbm, ep_end_vmem, sem).wait()
      start_tile = ep_start_vmem[0, 0]//gather_window_size
      end_tile = jnp.minimum((ep_end_vmem[0, 0]+gather_window_size-1)//gather_window_size, num_tiles)
      num_active_tiles = jnp.maximum(end_tile-start_tile, 1)

      def body(i_vmem, o_vmem):
        pltpu.sync_copy(x_hbm.at[i_vmem.at[0]], o_vmem)

      pltpu.emit_pipeline(
          body,
          grid=(num_active_tiles,),
          in_specs=[
            pl.BlockSpec((1, gather_window_size), index_map=lambda i: (0, start_tile+i)),
          ],
          out_specs=[
            pl.BlockSpec((gather_window_size, value_dim), index_map=lambda i: (start_tile + i, 0)),
          ],
          core_axis_name='subcore',
          dimension_semantics=(pltpu.PARALLEL,),
      )(i_hbm, o_hbm)

    # xw32: how to pass arguments to pl.run_scoped?
    # A: Example is https://docs.jax.dev/en/latest/pallas/tpu/core_map.html#a-simple-per-core-kernel. Use "partial" to pass in positional arguments and kwargs to pass keywords arguments.
    pl.run_scoped(
        scoped_func,
        # xw32: how do I create VMEM? 
        # A: plsc.MemoryRef((1, 1), jnp.int32, memory_space=pltpu.VMEM)
        # xw32: Is it VMEM or SMEM that I should create?
        # A: VMEM. Conceptually yes — you can use SMEM (scalar memory) and SMEM is the right home for scalar values. But practically there's a constraint: pltpu.async_copy uses the DMA engine, which transfers data between HBM and VMEM. It doesn't support SMEM as a destination.
        # xw32: Why should the shape be (1, 1) instead of a scalar ()?
        # A: Same reason as the above answer about DMA. DMA operations on vectors instead of scalars.
        ep_start_vmem=plsc.MemoryRef((1, 1), jnp.int32,
                                     memory_space=pltpu.VMEM),
        ep_end_vmem=plsc.MemoryRef((1, 1), jnp.int32,
                                   memory_space=pltpu.VMEM),
        sem=pltpu.SemaphoreType.DMA,
    )
  
  # xw32q: how should I pass the "gather_window_size"? Should it be before the output or after?
  # A: it seems the closure can capture it.
  return kernel(data, indices, ep_start, ep_end)
