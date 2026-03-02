"""TODO: xiowei - DO NOT SUBMIT without either providing a detailed docstring or
removing it altogether.
"""

from functools import partial
from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
from jax._src import test_util as jtu
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.experimental.pallas import tpu_sc as plsc
from tpu_inference.kernels.gather.gather import gather

jax.config.parse_flags_with_absl()


TOPK = 8


@jax.jit(static_argnames="global_num_experts")
def ref_impl(topk_indices_local, hidden_states, global_num_experts):
  topk = TOPK
  topk_indices_flat = topk_indices_local.flatten()
  topk_argsort_indices = jnp.argsort(topk_indices_flat)
  topk_argsort_revert_indices = jnp.argsort(topk_argsort_indices)
  num_local_tokens = hidden_states.shape[0]
  token_indices = jnp.arange(num_local_tokens,
                             dtype=jnp.int32).repeat(topk)
  token_indices_sorted = token_indices[topk_argsort_indices]
  group_sizes_local = jnp.bincount(topk_indices_flat, length=global_num_experts)
  x = hidden_states[token_indices_sorted]
  return x, group_sizes_local, topk_argsort_revert_indices



@jtu.with_config(jax_numpy_dtype_promotion="standard")
class MoeRoutingTest(jtu.JaxTestCase):

  def test_sparse_core_pallas_gather2d_sc_impl(self):
    batch_size = 8192
    value_dim = 6144
    num_indices = 65536
    a = jnp.arange(batch_size * value_dim).reshape(batch_size, value_dim).astype(jnp.bfloat16)
    indices = jax.random.randint(jax.random.key(0), (num_indices,), 0, batch_size, jnp.int32)
    
    @jax.jit
    def gather_direct(indices, x):
      return x[indices]

    expected = gather_direct(indices, a)
    actual = gather(a, indices)
    self.assertAllClose(actual, expected)

if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
