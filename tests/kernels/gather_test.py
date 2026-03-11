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
from tpu_inference.kernels.gather.gather2d_sc_impl import gather_3d_to_2d
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

  @parameterized.product(
      (
          {"a_shape": (1, 8, 128), "num_indices": 1},
          {"a_shape": (32, 8, 128), "num_indices": 32},
          {"a_shape": (32, 8, 384), "num_indices": 32},
      ),
      dtype=(jnp.uint16, jnp.bfloat16),
  )
  def test_output_matches_jnp_take(
      self,
      a_shape: tuple[int, int, int],
      num_indices: int,
      dtype: jnp.dtype,
  ):
    a, indices = self._make_test_data(a_shape, num_indices, dtype)
    gather_compiled = gather_3d_to_2d.lower(
        a, indices, window_bounds=(8,)
    ).compile()
    take_compiled = _take.lower(a, indices).compile()
    mosaic_result = gather_compiled(a, indices)
    jnp_result = take_compiled(a, indices)

    self.assertAllClose(jnp_result, mosaic_result)

@jax.jit
def _take(a, indices):
  return jnp.take(a, indices, axis=0).reshape(
      indices.shape[0], np.prod(a.shape[1:])
  )


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

  def _run_ragged_gather_perf_test(self, ep_token_start, ep_token_end):
    batch_size = 8192
    value_dim = 6144
    topk = 8
    num_indices = batch_size * topk
    k1, k2 = jax.random.split(jax.random.key(0))
    data = jax.random.uniform(k1, (batch_size, value_dim), dtype=jnp.bfloat16)
    indices = jax.random.randint(
        k2,
        (num_indices,),
        0,
        batch_size,
        jnp.int32,
    )
    ep_range = jnp.array([ep_token_start, ep_token_end], dtype=jnp.int32)
    gather.ragged_gather(
        data, indices, ep_range, gather_window_size=16
    ).block_until_ready()

  def test_ragged_gather_perf(self):
    batch_size = 8192
    value_dim = 6144
    topk = 8
    num_indices = batch_size * topk
    k1, k2 = jax.random.split(jax.random.key(0))
    data = jax.random.uniform(k1, (batch_size, value_dim), dtype=jnp.bfloat16)
    indices = jax.random.randint(
        k2,
        (num_indices,),
        0,
        batch_size,
        jnp.int32,
    )

    @jax.jit
    def gather_direct(indices, x):
      return x[indices]

    # Run once to warm up
    gather_direct(indices, data).block_until_ready()

    ep_token_start, ep_token_end = 1024, 1024+num_indices/16
    self._run_ragged_gather_perf_test(ep_token_start, ep_token_end)

    ep_token_start, ep_token_end = 1024, 1024+num_indices/8
    self._run_ragged_gather_perf_test(ep_token_start, ep_token_end)

    ep_token_start, ep_token_end = 1024, 1024+num_indices/4
    self._run_ragged_gather_perf_test(ep_token_start, ep_token_end)

    ep_token_start, ep_token_end = 1024, 1024+num_indices/2
    self._run_ragged_gather_perf_test(ep_token_start, ep_token_end)

    ep_token_start, ep_token_end = 0, num_indices
    self._run_ragged_gather_perf_test(ep_token_start, ep_token_end)

    # Run with xprof
    xprof_sess = xprof_session.XprofSession()
    xprof_sess.start_session()

    gather_direct(indices, data).block_until_ready()

    ep_token_start, ep_token_end = 1024, 1024+num_indices/16
    self._run_ragged_gather_perf_test(ep_token_start, ep_token_end)

    ep_token_start, ep_token_end = 1024, 1024+num_indices/8
    self._run_ragged_gather_perf_test(ep_token_start, ep_token_end)

    ep_token_start, ep_token_end = 1024, 1024+num_indices/4
    self._run_ragged_gather_perf_test(ep_token_start, ep_token_end)

    ep_token_start, ep_token_end = 1024, 1024+num_indices/2
    self._run_ragged_gather_perf_test(ep_token_start, ep_token_end)

    ep_token_start, ep_token_end = 0, num_indices
    self._run_ragged_gather_perf_test(ep_token_start, ep_token_end)

    session_id = xprof_sess.end_session_and_get_session_id()
    print(f"xprof: http://xprof/?session_id={session_id}")


  # $ blaze test -c opt //experimental/users/kyuyeunk/vllm/tests:gather_test_gf --test_filter=RaggedGatherTest.test_ragged_gather_single_perf --test_arg=--xla_xprof_register_llo_debug_info=true
  # $ blaze test -c opt //experimental/users/kyuyeunk/vllm/tests:gather_test_gf --test_filter=RaggedGatherTest.test_ragged_gather_single_perf --test_arg=--xla_xprof_register_llo_debug_info=true --test_arg=--xla_tpu_emit_tracing_vwaits=true
  # ref: https://source.corp.google.com/piper///depot/google3/experimental/users/kyuyeunk/debugging/benchmark/run_benchmark.sh;l=23-25;rcl=873173398
  # https://source.corp.google.com/piper///depot/google3/experimental/users/yuyanpeng/ullm/test_tmp.py;l=1
  def test_ragged_gather_single_perf(self):
    batch_size = 8192
    topk = 8
    num_indices = batch_size * topk

    @jax.jit
    def gather_direct(indices, x):
      return x[indices]

    # Run once to warm up
    ep_token_start, ep_token_end = 1024, 1024+num_indices/8
    self._run_ragged_gather_perf_test(ep_token_start, ep_token_end)

    kernel_name = "ragged_gather_b8192_v6144_n65536_w16"
    xprof_sess = xprof_session.XprofSession()
    xprof_sess.start_session(
        device_name="",
        enable_python_tracer=False,
        host_trace_level=2,
        host_cpu_profile=False,
        trace_mode="TRACE_COMPUTE_AND_SYNC",
        power_trace_level="POWER_TRACE_NORMAL",
        enable_fw_throttle_event=True,
        enable_fw_power_level_event=True,
        enable_fw_thermal_event=True,
    )
    ep_token_start, ep_token_end = 1024, 1024+num_indices/8
    self._run_ragged_gather_perf_test(ep_token_start, ep_token_end)
    session_id = xprof_sess.end_session_and_get_session_id()
    print(f"✅ [xprof] http://xprof/?session_id={session_id}")
    print(f"✅ http://percale.corp.google.com/{session_id}/{kernel_name}")

if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
