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
from tpu_inference.kernels.gather.gather2 import ragged_gather

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


@jtu.with_config(jax_numpy_dtype_promotion="standard")
class RaggedGatherTest(jtu.JaxTestCase):

  def _run_ragged_gather_test(self, batch_size, value_dim, num_indices,
                              ep_token_start, ep_token_end):
    """Helper: runs ragged_gather and checks in-range output matches reference."""
    a = jnp.arange(batch_size * value_dim).reshape(
        batch_size, value_dim).astype(jnp.bfloat16)
    indices = jax.random.randint(
        jax.random.key(0), (num_indices,), 0, batch_size, jnp.int32)

    ep_range = jnp.array([ep_token_start, ep_token_end], dtype=jnp.int32)
    actual = ragged_gather(a, indices, ep_range)
    expected = a[indices]

    # Verify the in-range portion matches
    actual_slice = actual[ep_token_start:ep_token_end]
    expected_slice = expected[ep_token_start:ep_token_end]
    self.assertAllClose(actual_slice, expected_slice)

  @parameterized.named_parameters(
      dict(testcase_name="range_in_middle",
           ep_token_start=1024, ep_token_end=4096),
      dict(testcase_name="range_at_start",
           ep_token_start=0, ep_token_end=2048),
      dict(testcase_name="range_at_end",
           ep_token_start=63488, ep_token_end=65536),
      dict(testcase_name="full_range",
           ep_token_start=0, ep_token_end=65536),
      dict(testcase_name="non_aligned_boundaries",
           ep_token_start=20, ep_token_end=50),
  )
  def test_ragged_gather(self, ep_token_start, ep_token_end):
    self._run_ragged_gather_test(8192, 6144, 65536, ep_token_start,
                                 ep_token_end)

  @parameterized.named_parameters(
      dict(testcase_name="range_in_middle",
           ep_token_start=512, ep_token_end=1024),
      dict(testcase_name="range_at_start",
           ep_token_start=0, ep_token_end=512),
      dict(testcase_name="range_at_end",
           ep_token_start=1536, ep_token_end=2048),
      dict(testcase_name="full_range",
           ep_token_start=0, ep_token_end=2048),
      dict(testcase_name="non_aligned_boundaries",
           ep_token_start=20, ep_token_end=50),
  )
  def test_ragged_gather_small(self, ep_token_start, ep_token_end):
    self._run_ragged_gather_test(512, 128, 2048, ep_token_start,
                                 ep_token_end)


if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
