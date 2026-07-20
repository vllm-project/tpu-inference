# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from absl.testing import parameterized

from tpu_inference.layers.common.fused_moe_gmm import (_resolve_topk_backend,
                                                       iterative_top_k_kernel)


class IterativeTopKKernelTest(parameterized.TestCase):

    @parameterized.named_parameters(
        dict(testcase_name="example_shape_1",
             num_tokens=16384,
             num_experts=128,
             k=8),
        dict(testcase_name="example_shape_2",
             num_tokens=4096,
             num_experts=256,
             k=10))
    def test_matches_lax_top_k(self, num_tokens, num_experts, k):
        rng = np.random.default_rng(0)
        x = jnp.asarray(rng.standard_normal(
            (num_tokens, num_experts)).astype(np.float32),
                        dtype=jnp.bfloat16)

        ref_vals, _ = jax.jit(functools.partial(jax.lax.top_k, k=k))(x)
        got_vals, got_idxs = jax.jit(
            functools.partial(iterative_top_k_kernel, k=k))(x)

        # Values must match jax.lax.top_k exactly (descending order). Index
        # tie-break may differ on an exact-value tie (assumed rare/
        # inconsequential for real router logits - see
        # iterative_top_k_kernel's docstring), so instead of comparing
        # indices to jax.lax.top_k's, just check got_idxs is self-consistent
        # with got_vals.
        np.testing.assert_allclose(
            np.asarray(got_vals).astype(np.float32),
            np.asarray(ref_vals).astype(np.float32))
        gathered = np.take_along_axis(np.asarray(x).astype(np.float32),
                                      np.asarray(got_idxs),
                                      axis=-1)
        np.testing.assert_allclose(gathered,
                                   np.asarray(got_vals).astype(np.float32))

    def test_output_shape_and_dtype(self):
        x = jnp.ones((5, 10), dtype=jnp.bfloat16)
        vals, idxs = iterative_top_k_kernel(x, k=4)
        self.assertEqual(vals.shape, (5, 4))
        self.assertEqual(idxs.shape, (5, 4))
        self.assertEqual(idxs.dtype, jnp.int32)


class ResolveTopkBackendTest(parameterized.TestCase):

    @parameterized.named_parameters(
        ("topk", "topk", "topk", 0.9),
        ("pallas_topk", "pallas_topk", "pallas_topk", 0.9),
        ("approx_topk_default", "approx_topk", "approx_topk", 0.9),
        ("approx_topk_explicit", "approx_topk:recall_target=0.95",
         "approx_topk", 0.95),
    )
    def test_parses(self, backend_str, expected_name, expected_recall):
        name, recall_target = _resolve_topk_backend(backend_str)
        self.assertEqual(name, expected_name)
        self.assertEqual(recall_target, expected_recall)


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__]))
