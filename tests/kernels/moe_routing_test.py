"""TODO: xiowei - DO NOT SUBMIT without either providing a detailed docstring or
removing it altogether.
"""

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
from jax._src import test_util as jtu

import tpu_inference.kernels.moe_routing.sort as sort


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

  # blaze test -c opt --test_output=errors  //experimental/users/xiowei/ullm:tests/moe_routing_test_gf --test_filter=test_sort_basic
  # blaze test -c opt --test_output=errors  //experimental/users/xiowei/ullm:tests/moe_routing_test_gf --test_filter=test_sort_basic --test_arg=--xla_tpu_enable_log_recorder
  def test_sort_basic(self):
    num_local_tokens = 8
    hidden_size = 128
    num_experts = 16
    topk = TOPK
    topk_indices_local = jnp.array(
        [
            [0, 1, 2, 3, 4, 5, 6, 7],
            [1, 2, 3, 4, 5, 6, 7, 8],
            [2, 3, 4, 5, 6, 7, 8, 9],
            [3, 4, 5, 6, 7, 8, 9, 10],
            [4, 5, 6, 7, 8, 9, 10, 11],
            [5, 6, 7, 8, 9, 10, 11, 12],
            [6, 7, 8, 9, 10, 11, 12, 13],
            [7, 8, 9, 10, 11, 12, 13, 14],
        ],
        dtype=jnp.int32,
    )
    hidden_states = jax.lax.broadcasted_iota(
        jnp.int32, (num_local_tokens, hidden_size), 0
    )
    expected_x, expected_group_sizes, expected_topk_argsort_revert_indices= ref_impl(topk_indices_local, hidden_states, num_experts)
    actual_x, actual_group_sizes, actual_topk_argsort_revert_indices= sort.sort_tokens(
        hidden_states, topk_indices_local, num_experts, debug_mode=True
    )

    self.assertAllClose(expected_group_sizes, actual_group_sizes)
    # self.assertAllClose(expected_topk_argsort_revert_indices, actual_topk_argsort_revert_indices)
    print(
        f"Output max diff {jnp.max(jnp.abs(expected_x - actual_x))}"
    )
    print(
        f"Output mean diff {jnp.mean(jnp.abs(expected_x - actual_x))}"
    )
    self.assertAllClose(expected_x, actual_x)

  def test_sort_random(self):
    num_local_tokens = 64
    num_local_tokens = 64
    hidden_size = 128
    num_experts = 16
    topk = TOPK

    k1, k2 = jax.random.split(jax.random.key(0))
    random_logits = jax.random.uniform(k1, (num_local_tokens, num_experts))
    _, topk_indices_local = jax.lax.top_k(random_logits, k=topk)
    hidden_states = jax.random.uniform(k2, (num_local_tokens, hidden_size))
    expected_x, expected_group_sizes, expected_topk_argsort_revert_indices= ref_impl(topk_indices_local, hidden_states, num_experts)
    actual_x, actual_group_sizes, actual_topk_argsort_revert_indices= sort.sort_tokens(
        hidden_states, topk_indices_local, num_experts
    )

    self.assertAllClose(expected_group_sizes, actual_group_sizes)
    self.assertAllClose(expected_topk_argsort_revert_indices, actual_topk_argsort_revert_indices)
    print(
        f"Output max diff {jnp.max(jnp.abs(expected_x - actual_x))}"
    )
    print(
        f"Output mean diff {jnp.mean(jnp.abs(expected_x - actual_x))}"
    )
    self.assertAllClose(expected_x, actual_x)

  def test_sort_real_workload(self):
    num_local_tokens = 8192
    hidden_size = 6144
    num_experts = 160
    topk = TOPK

    k1, k2 = jax.random.split(jax.random.key(0))
    random_logits = jax.random.uniform(k1, (num_local_tokens, num_experts))
    _, topk_indices_local = jax.lax.top_k(random_logits, k=topk)
    hidden_states = jax.random.uniform(k2, (num_local_tokens, hidden_size))
    actual_x, _, _= sort.sort_tokens(
        hidden_states, topk_indices_local, num_experts
    )
    actual_x.block_until_ready()
    

  def test_sort_perf(self):
    num_local_tokens = 8192
    hidden_size = 6144
    num_experts = 160
    topk = TOPK

    k1, k2 = jax.random.split(jax.random.key(0))
    random_logits = jax.random.uniform(k1, (num_local_tokens, num_experts))
    _, topk_indices_local = jax.lax.top_k(random_logits, k=topk)
    hidden_states = jax.random.uniform(k2, (num_local_tokens, hidden_size))

    # Warmup
    sort.sort_tokens(hidden_states, topk_indices_local, num_experts)[0].block_until_ready()

    profile_path = "/tmp/sort_tokens_profile"
    jax.profiler.start_trace(profile_path)
    for _ in range(2):
      sort.sort_tokens(hidden_states, topk_indices_local, num_experts)[0].block_until_ready()
    jax.profiler.stop_trace()
    print(f"Profile saved to {profile_path}")


if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
