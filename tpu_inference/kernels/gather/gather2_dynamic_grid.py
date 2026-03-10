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
  num_indices = indices.shape[0]
  value_dim = data.shape[1]
  batch_size = data.shape[0]
  num_tiles = num_indices // gather_window_size

  sc_info = plsc.get_sparse_core_info()
  vector_mesh = plsc.VectorSubcoreMesh(
      num_cores=sc_info.num_cores,
      num_subcores=sc_info.num_subcores,
      core_axis_name="core",
      subcore_axis_name="subcore",
  )

  ep_start = ep_range[0].reshape(1, 1)
  ep_end = ep_range[1].reshape(1, 1)
  indices = indices.reshape((1, num_indices))

  # Note to myself:
  # - we cannot calculate num_active_tiles outside of the kernel because otherwise it fails with an error "ValueError: default_mesh_discharge_rule only supports Ref inputs/outputs." (https://fusion2.corp.google.com/invocations/dad2b2ac-8e86-4530-8f22-be0bf51d54ec)
  # - we have to pass ep_start and ep_end as arguments to the kernel
  # - then we cannot just directly read ep_start and ep_end from HBM. We have to read them into VMEM first.
  # - To do that, we need to allocate a sem and vmem space with pl.run_scoped.

  @pl.kernel(out_shape=jax.ShapeDtypeStruct((num_indices, value_dim),
                                             data.dtype),
             mesh=vector_mesh)
  def kernel(x_hbm, i_hbm, ep_start_hbm, ep_end_hbm, o_hbm):
    def scoped_fn(ep_start_vmem, ep_end_vmem, sem):
      # Copy scalars from HBM to VMEM via async DMA.
      pltpu.async_copy(ep_start_hbm, ep_start_vmem, sem).wait()
      pltpu.async_copy(ep_end_hbm, ep_end_vmem, sem).wait()

      # Compute active tile range from VMEM values.
      start_tile = jnp.maximum(ep_start_vmem[0, 0] // gather_window_size, 0)
      end_tile = jnp.minimum(
          (ep_end_vmem[0, 0] + gather_window_size - 1) // gather_window_size,
          num_tiles,
      )
      num_active_tiles = jnp.maximum(end_tile - start_tile, 1)

      def body(i_vmem, o_vmem):
        pltpu.sync_copy(x_hbm.at[i_vmem.at[0]], o_vmem)

      pltpu.emit_pipeline(
          body,
          grid=(num_active_tiles,),
          in_specs=[
              pl.BlockSpec((1, gather_window_size),
                           index_map=lambda i: (0, start_tile + i)),
          ],
          out_specs=[pl.BlockSpec((gather_window_size, value_dim),
                                  index_map=lambda i: (start_tile + i, 0))],
          core_axis_name='subcore',
          dimension_semantics=(pltpu.PARALLEL,),
      )(i_hbm, o_hbm)

    pl.run_scoped(
        scoped_fn,
        ep_start_vmem=plsc.MemoryRef((1, 1), jnp.int32,
                                     memory_space=pltpu.VMEM),
        ep_end_vmem=plsc.MemoryRef((1, 1), jnp.int32,
                                   memory_space=pltpu.VMEM),
        sem=pltpu.SemaphoreType.DMA,
    )

  kernel_name = f"ragged_gather_b{batch_size}_v{value_dim}_n{num_indices}_w{gather_window_size}"
  return jax.named_call(kernel, name=kernel_name)(
      data, indices, ep_start, ep_end)
