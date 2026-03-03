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
