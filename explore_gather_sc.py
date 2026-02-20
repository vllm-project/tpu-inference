"""Run this and get the HLO and LLO for the gather operation.
"""

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.experimental.pallas import tpu_sc as plsc

@jax.jit
def gather_direct(indices, x):
  """Gather using direct indexing."""
  return x[indices]

# SparseCore Pallas kernel doesn't work in OSS as of 20260217.
# The same code works in G3.
def test_sparse_core_pallas_gather():
  num_local_tokens = 8192
  hidden_size = 6144
  num_experts = 160
  topk = 8

  k1, k2 = jax.random.split(jax.random.key(0))
  indices = jax.random.randint(k1, (num_local_tokens * topk,), 0, num_local_tokens)
  hidden_states = jax.random.uniform(k2, (num_local_tokens, hidden_size))

  num_indices = indices.shape[0]
  vector_mesh = plsc.VectorSubcoreMesh(
      core_axis_name="core", subcore_axis_name="subcore"
  )
  value_dim = hidden_states.shape[1]
  gather_window_size = 8

  @jax.jit
  def gather(data, indices):
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

  out = gather(hidden_states, indices)
  expected = gather_direct(indices, hidden_states)
  jnp.allclose(out, expected)

def main():
  num_tokens = 8192
  hidden_size = 6144
  num_indices = 65536

  key = jax.random.PRNGKey(0)
  k1, k2 = jax.random.split(key)
  hidden_states = jax.random.normal(k1, (num_tokens, hidden_size), dtype=jnp.bfloat16)
  indices = jax.random.randint(k2, (num_indices,), 0, num_tokens)

  gather_direct(indices, hidden_states).block_until_ready()

  # profile_path = "/tmp/sort2_tokens_profile"
  # jax.profiler.start_trace(profile_path)
  for _ in range(3):
    gather_direct(indices, hidden_states).block_until_ready()
  # jax.profiler.stop_trace()

if __name__ == "__main__":
  test_sparse_core_pallas_gather()
