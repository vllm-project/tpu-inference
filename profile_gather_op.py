"""Explore the efficiency of gather via one-hot encoding (permutation matrix)."""

import jax
import jax.numpy as jnp
import time
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.experimental.pallas import tpu_sc as plsc

@jax.jit
def gather_xla(indices, x):
  """Gather using direct indexing."""
  return x[indices]

@jax.jit(static_argnames="vector_mesh")
def gather_sparse_pallas(indices, x, vector_mesh):
  num_indices = indices.shape[0]
  value_dim = x.shape[1]
  gather_window_size = num_indices//2
  indices = indices.reshape((1, num_indices))
  @pl.kernel(out_shape=jax.ShapeDtypeStruct((num_indices, value_dim), x.dtype),
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

  return kernel(x, indices)


@jax.jit
def gather2(indices, x):
  out = jax.lax.gather(
    x,
    indices[:, None],
    dimension_numbers=jax.lax.GatherDimensionNumbers(
      offset_dims=(1,),
      collapsed_slice_dims=(0,),
      start_index_map=(0,)
    ),
    slice_sizes=(1, 6144),
    indices_are_sorted=False, # Set to True if they are!
    unique_indices=False,
    mode=jax.lax.GatherScatterMode.PROMISE_IN_BOUNDS # This is the key
  )
  return out


def check_jax_lax_gather():
  num_tokens = 8192
  hidden_size = 6144
  num_indices = 65536

  key = jax.random.key(0)
  k1, k2 = jax.random.split(key)
  hidden_states = jax.random.normal(k1, (num_tokens, hidden_size), dtype=jnp.bfloat16)
  indices = jax.random.randint(k2, (num_indices,), 0, num_tokens)

  out_xla = gather_xla(indices, hidden_states)
  out_2 = gather2(indices, hidden_states)
  assert jnp.allclose(out_xla, out_2)

  profile_path = "/tmp/sort2_tokens_profile"
  jax.profiler.start_trace(profile_path)
  for _ in range(3):
    gather_xla(indices, hidden_states).block_until_ready()
    gather2(indices, hidden_states).block_until_ready()
  jax.profiler.stop_trace()

def check_sparse_pallas():
  num_tokens = 8192
  hidden_size = 6144
  num_indices = 65536
  vector_mesh = plsc.VectorSubcoreMesh(
    core_axis_name="core", subcore_axis_name="subcore"
  )

  key = jax.random.key(0)
  k1, k2 = jax.random.split(key)
  hidden_states = jax.random.normal(k1, (num_tokens, hidden_size), dtype=jnp.bfloat16)
  indices = jax.random.randint(k2, (num_indices,), 0, num_tokens)

  out_xla = gather_xla(indices, hidden_states)
  out_2 = gather_sparse_pallas(indices, hidden_states, vector_mesh)
  assert jnp.allclose(out_xla, out_2)

  # profile_path = "/tmp/sort2_tokens_profile"
  # jax.profiler.start_trace(profile_path)
  # for _ in range(3):
  #   gather_xla(indices, hidden_states).block_until_ready()
  #   gather2(indices, hidden_states).block_until_ready()
  # jax.profiler.stop_trace()


if __name__ == "__main__":
  # check_jax_lax_gather()
  check_sparse_pallas()
