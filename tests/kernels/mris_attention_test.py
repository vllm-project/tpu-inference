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
"""Tests for MRIS (Multi-Request Interleaved Streaming) fused attention kernel.

These tests validate:
  1. Numerical equivalence between the MRIS kernel and the reference
     implementation (independent per-request attention).
  2. Request isolation (changes to one request's data do not affect other
     requests' outputs).
  3. Varying KV lengths across fused requests.
  4. Different head dimensions and number of query heads per KV head.

Reference tests (MrisReferenceTest, MrisRequestIsolationTest) run on CPU.
Pallas kernel tests (MrisPagedV3CorrectnessTest) require TPU.
"""

from absl.testing import absltest, parameterized
import jax
import jax.numpy as jnp
import numpy as np

# The kernel module lives in the tpu_inference package. When running via blaze
# the import path is resolved automatically. For standalone execution the
# caller must ensure google3 is on sys.path.
from tpu_inference.kernels.mris_attention.kernel import (
    N_FUSED,
    mris_fused_paged_attention_v3,
    ref_mris_fused_attention,
)
from tpu_inference.kernels.ragged_paged_attention.v3.util import cdiv


def _create_random_inputs(
    *,
    num_q_heads_per_kv_head: int,
    head_dim: int,
    kv_lens: list[int],
    seed: int = 0,
    dtype: jnp.dtype = jnp.bfloat16,
):
    """Creates random Q, K, V inputs for N_FUSED requests.

    Returns:
      queries: list of [1, num_q_heads_per_kv_head, head_dim] arrays.
      k_pages: list of [max_kv_len, head_dim] arrays (padded to max).
      v_pages: list of [max_kv_len, head_dim] arrays (padded to max).
    """
    rng = np.random.RandomState(seed)
    max_kv_len = max(kv_lens)

    queries = []
    k_pages = []
    v_pages = []

    for i in range(len(kv_lens)):
        q = rng.randn(1, num_q_heads_per_kv_head, head_dim).astype(np.float32)
        k = rng.randn(max_kv_len, head_dim).astype(np.float32)
        v = rng.randn(max_kv_len, head_dim).astype(np.float32)

        queries.append(jnp.array(q, dtype=dtype))
        k_pages.append(jnp.array(k, dtype=dtype))
        v_pages.append(jnp.array(v, dtype=dtype))

    return queries, k_pages, v_pages


# ===================================================================
# Reference implementation tests — run on CPU
# ===================================================================
class MrisReferenceTest(parameterized.TestCase):
    """Tests for the pure-JAX reference implementation."""

    @parameterized.parameters(
        dict(num_q_heads=1, head_dim=128, kv_len=64),
        dict(num_q_heads=1, head_dim=128, kv_len=256),
        dict(num_q_heads=4, head_dim=128, kv_len=128),
        dict(num_q_heads=1, head_dim=64, kv_len=128),
    )
    def test_ref_output_shape(self, num_q_heads, head_dim, kv_len):
        """Validates that the reference implementation returns correct shapes."""
        kv_lens = [kv_len] * N_FUSED
        queries, k_pages, v_pages = _create_random_inputs(
            num_q_heads_per_kv_head=num_q_heads,
            head_dim=head_dim,
            kv_lens=kv_lens,
        )
        sm_scale = head_dim**-0.5
        outputs = ref_mris_fused_attention(
            queries, k_pages, v_pages, kv_lens, sm_scale=sm_scale
        )

        self.assertLen(outputs, N_FUSED)
        for out in outputs:
            self.assertEqual(out.shape, (1, num_q_heads, head_dim))

    @parameterized.parameters(
        dict(num_q_heads=1, head_dim=128, kv_lens=[64, 128, 256, 32]),
        dict(num_q_heads=4, head_dim=128, kv_lens=[100, 200, 50, 150]),
    )
    def test_ref_mixed_kv_lens(self, num_q_heads, head_dim, kv_lens):
        """Tests reference with different KV lengths per request."""
        queries, k_pages, v_pages = _create_random_inputs(
            num_q_heads_per_kv_head=num_q_heads,
            head_dim=head_dim,
            kv_lens=kv_lens,
        )
        sm_scale = head_dim**-0.5
        outputs = ref_mris_fused_attention(
            queries, k_pages, v_pages, kv_lens, sm_scale=sm_scale
        )
        self.assertLen(outputs, N_FUSED)
        for out in outputs:
            self.assertEqual(out.shape, (1, num_q_heads, head_dim))

    def test_ref_matches_manual_attention(self):
        """Verifies ref implementation against manually computed attention."""
        head_dim = 128
        kv_len = 4
        kv_lens = [kv_len] * N_FUSED
        sm_scale = head_dim**-0.5

        queries, k_pages, v_pages = _create_random_inputs(
            num_q_heads_per_kv_head=1,
            head_dim=head_dim,
            kv_lens=kv_lens,
            seed=42,
        )

        ref_outputs = ref_mris_fused_attention(
            queries, k_pages, v_pages, kv_lens, sm_scale=sm_scale
        )

        # Manual computation for request 0
        q0 = queries[0]  # [1, 1, head_dim]
        k0 = k_pages[0][:kv_len]  # [kv_len, head_dim]
        v0 = v_pages[0][:kv_len]  # [kv_len, head_dim]

        # [1, kv_len]
        s = (
            jnp.matmul(
                q0.reshape(1, head_dim),
                k0.T,
                preferred_element_type=jnp.float32,
            )
            * sm_scale
        )
        p = jax.nn.softmax(s, axis=-1)
        expected = jnp.matmul(
            p.astype(v0.dtype), v0, preferred_element_type=jnp.float32
        )  # [1, head_dim]

        np.testing.assert_allclose(
            ref_outputs[0].reshape(1, head_dim),
            expected,
            atol=1e-2,
            rtol=1e-2,
        )

    def test_ref_single_kv_returns_value(self):
        """For kv_len=1, the attention output should be exactly V."""
        head_dim = 128
        kv_lens = [1, 1, 1, 1]
        sm_scale = head_dim**-0.5

        queries, k_pages, v_pages = _create_random_inputs(
            num_q_heads_per_kv_head=1,
            head_dim=head_dim,
            kv_lens=[1] * N_FUSED,
            seed=33,
        )

        ref_outputs = ref_mris_fused_attention(
            queries, k_pages, v_pages, kv_lens, sm_scale=sm_scale
        )

        # For kv_len=1, softmax of a single element is 1.0, so output = V
        for i in range(N_FUSED):
            v_expected = v_pages[i][:1]  # [1, head_dim]
            np.testing.assert_allclose(
                ref_outputs[i].reshape(1, head_dim),
                v_expected,
                atol=1e-2,
                rtol=1e-2,
                err_msg=f"Request {i}: kv_len=1 output should equal V",
            )


class MrisRequestIsolationTest(parameterized.TestCase):
    """Tests that MRIS reference maintains strict request isolation."""

    def test_ref_request_isolation(self):
        """Modifying one request's K/V should not affect other outputs."""
        head_dim = 128
        kv_lens = [128, 128, 128, 128]
        sm_scale = head_dim**-0.5

        queries, k_pages, v_pages = _create_random_inputs(
            num_q_heads_per_kv_head=1,
            head_dim=head_dim,
            kv_lens=kv_lens,
            seed=123,
        )

        # Run baseline
        baseline_outputs = ref_mris_fused_attention(
            queries, k_pages, v_pages, kv_lens, sm_scale=sm_scale
        )

        # Modify request 2's key data
        k_pages_modified = list(k_pages)
        k_pages_modified[2] = k_pages[2] * 2.0 + 1.0

        modified_outputs = ref_mris_fused_attention(
            queries, k_pages_modified, v_pages, kv_lens, sm_scale=sm_scale
        )

        # Request 0, 1, 3 should be unchanged
        for i in [0, 1, 3]:
            np.testing.assert_array_equal(
                baseline_outputs[i],
                modified_outputs[i],
                err_msg=(
                    f"Request {i} output changed when only request 2 was"
                    " modified"
                ),
            )

        # Request 2 should be different
        with self.assertRaises(AssertionError):
            np.testing.assert_array_equal(
                baseline_outputs[2], modified_outputs[2]
            )

    def test_ref_value_isolation_across_all_pairs(self):
        """Each request's V modification only affects its own output."""
        head_dim = 128
        kv_lens = [64, 128, 96, 200]
        sm_scale = head_dim**-0.5

        queries, k_pages, v_pages = _create_random_inputs(
            num_q_heads_per_kv_head=1,
            head_dim=head_dim,
            kv_lens=kv_lens,
            seed=77,
        )

        baseline_outputs = ref_mris_fused_attention(
            queries, k_pages, v_pages, kv_lens, sm_scale=sm_scale
        )

        for modified_idx in range(N_FUSED):
            v_pages_mod = list(v_pages)
            v_pages_mod[modified_idx] = v_pages[modified_idx] * -1.5

            mod_outputs = ref_mris_fused_attention(
                queries, k_pages, v_pages_mod, kv_lens, sm_scale=sm_scale
            )

            for i in range(N_FUSED):
                if i == modified_idx:
                    continue
                np.testing.assert_array_equal(
                    baseline_outputs[i],
                    mod_outputs[i],
                    err_msg=(
                        f"Request {i} changed when modifying request"
                        f" {modified_idx}'s V"
                    ),
                )

    @parameterized.parameters(
        dict(head_dim=64),
        dict(head_dim=128),
        dict(head_dim=256),
    )
    def test_ref_different_head_dims(self, head_dim):
        """Tests reference with different head dimensions."""
        kv_lens = [128] * N_FUSED
        sm_scale = head_dim**-0.5

        queries, k_pages, v_pages = _create_random_inputs(
            num_q_heads_per_kv_head=1,
            head_dim=head_dim,
            kv_lens=kv_lens,
            seed=11,
        )

        outputs = ref_mris_fused_attention(
            queries, k_pages, v_pages, kv_lens, sm_scale=sm_scale
        )

        self.assertLen(outputs, N_FUSED)
        for out in outputs:
            self.assertEqual(out.shape, (1, 1, head_dim))
            # Check no NaN/Inf
            self.assertTrue(jnp.all(jnp.isfinite(out)), f"Non-finite in output")


def _create_paged_kv_cache(
    *,
    num_reqs: int,
    kv_lens: list[int],
    head_dim: int,
    page_size: int = 16,
    seed: int = 0,
    dtype: jnp.dtype = jnp.bfloat16,
):
    """Creates a paged KV cache matching the production RPA V3 layout.

    Returns:
      kv_cache: [num_pages, page_size, 1, 2, head_dim] — physical page table.
      page_indices: [num_reqs, pages_per_seq] — logical-to-physical mapping.
      queries: list of [1, 1, head_dim] arrays (1 Q head per KV head for
      simplicity).
      k_pages_ref: list of [kv_len, head_dim] arrays for reference comparison.
      v_pages_ref: list of [kv_len, head_dim] arrays for reference comparison.
    """
    rng = np.random.RandomState(seed)
    max_kv_len = max(kv_lens)
    pages_per_seq = cdiv(max_kv_len, page_size)
    num_pages = pages_per_seq * num_reqs + 8  # extra padding pages

    kv_cache = np.zeros(
        (num_pages, page_size, 1, 2, head_dim), dtype=np.float32
    )
    page_indices = np.zeros((num_reqs, pages_per_seq), dtype=np.int32)
    queries = []
    k_pages_ref = []
    v_pages_ref = []

    page_counter = 0
    for i in range(num_reqs):
        kv_len = kv_lens[i]
        q = rng.randn(1, 1, head_dim).astype(np.float32)
        queries.append(jnp.array(q, dtype=dtype))

        k_data = rng.randn(kv_len, head_dim).astype(np.float32)
        v_data = rng.randn(kv_len, head_dim).astype(np.float32)
        k_pages_ref.append(jnp.array(k_data, dtype=dtype))
        v_pages_ref.append(jnp.array(v_data, dtype=dtype))

        num_pages_needed = cdiv(kv_len, page_size)
        for p in range(num_pages_needed):
            page_indices[i, p] = page_counter
            start = p * page_size
            end = min(start + page_size, kv_len)
            kv_cache[page_counter, : end - start, 0, 0, :] = k_data[start:end]
            kv_cache[page_counter, : end - start, 0, 1, :] = v_data[start:end]
            page_counter += 1

    return (
        jnp.array(kv_cache, dtype=dtype),
        jnp.array(page_indices, dtype=jnp.int32),
        queries,
        k_pages_ref,
        v_pages_ref,
    )


# ===================================================================
# Paged V3 kernel tests — require TPU
# ===================================================================
class MrisPagedV3CorrectnessTest(parameterized.TestCase):
    """Correctness tests for the Paged V3 kernel on TPU.

    This kernel fetches KV data directly from the production paged KV cache
    via in-kernel scattered HBM page lookups, matching the RPA V3 cache layout.
    """

    def setUp(self):
        super().setUp()

    @parameterized.parameters(
        dict(head_dim=128, kv_len=64, page_size=16),
        dict(head_dim=128, kv_len=128, page_size=16),
        dict(head_dim=128, kv_len=256, page_size=16),
        dict(head_dim=128, kv_len=512, page_size=16),
        dict(head_dim=64, kv_len=128, page_size=16),
    )
    def test_paged_v3_matches_ref_uniform_kv(self, head_dim, kv_len, page_size):
        """Paged V3 kernel output matches reference for uniform KV lengths."""
        kv_lens = [kv_len] * N_FUSED
        sm_scale = head_dim**-0.5

        kv_cache, page_indices, queries, k_ref, v_ref = _create_paged_kv_cache(
            num_reqs=N_FUSED,
            kv_lens=kv_lens,
            head_dim=head_dim,
            page_size=page_size,
            seed=60,
        )

        ref_outputs = ref_mris_fused_attention(
            queries, k_ref, v_ref, kv_lens, sm_scale=sm_scale
        )

        paged_outputs = mris_fused_paged_attention_v3(
            queries,
            kv_cache,
            jnp.array(kv_lens, dtype=jnp.int32),
            page_indices,
            sm_scale=sm_scale,
            bkv_sz=128,
            page_size=page_size,
        )

        for i in range(N_FUSED):
            np.testing.assert_allclose(
                paged_outputs[i],
                ref_outputs[i],
                atol=5e-2,
                rtol=5e-2,
                err_msg=f"Request {i} Paged V3 output diverges from reference",
            )

    def test_paged_v3_matches_ref_ragged_kv(self):
        """Paged V3 kernel output matches reference for ragged KV lengths."""
        head_dim = 128
        kv_lens = [64, 128, 256, 192]
        page_size = 16
        sm_scale = head_dim**-0.5

        kv_cache, page_indices, queries, k_ref, v_ref = _create_paged_kv_cache(
            num_reqs=N_FUSED,
            kv_lens=kv_lens,
            head_dim=head_dim,
            page_size=page_size,
            seed=61,
        )

        ref_outputs = ref_mris_fused_attention(
            queries, k_ref, v_ref, kv_lens, sm_scale=sm_scale
        )

        paged_outputs = mris_fused_paged_attention_v3(
            queries,
            kv_cache,
            jnp.array(kv_lens, dtype=jnp.int32),
            page_indices,
            sm_scale=sm_scale,
            bkv_sz=128,
            page_size=page_size,
        )

        for i in range(N_FUSED):
            np.testing.assert_allclose(
                paged_outputs[i],
                ref_outputs[i],
                atol=5e-2,
                rtol=5e-2,
                err_msg=f"Request {i} diverges with kv_len={kv_lens[i]}",
            )

    @parameterized.named_parameters(
        ("batch_1", 1),
        ("batch_3", 3),
        ("batch_4", 4),
        ("batch_5", 5),
        ("batch_7", 7),
        ("batch_8", 8),
        ("batch_15", 15),
        ("batch_16", 16),
    )
    def test_paged_v3_arbitrary_batch_sizes(self, batch_size):
        """Tests Paged V3 kernel with non-multiple-of-4 batch sizes."""
        head_dim = 128
        page_size = 16
        np.random.seed(200 + batch_size)
        kv_lens = list(np.random.randint(64, 512, size=batch_size))
        sm_scale = head_dim**-0.5

        kv_cache, page_indices, queries, k_ref, v_ref = _create_paged_kv_cache(
            num_reqs=batch_size,
            kv_lens=kv_lens,
            head_dim=head_dim,
            page_size=page_size,
            seed=200 + batch_size,
        )

        ref_outputs = ref_mris_fused_attention(
            queries, k_ref, v_ref, kv_lens, sm_scale=sm_scale
        )

        paged_outputs = mris_fused_paged_attention_v3(
            queries,
            kv_cache,
            jnp.array(kv_lens, dtype=jnp.int32),
            page_indices,
            sm_scale=sm_scale,
            bkv_sz=128,
            page_size=page_size,
        )

        self.assertEqual(len(paged_outputs), batch_size)
        for i in range(batch_size):
            np.testing.assert_allclose(
                paged_outputs[i],
                ref_outputs[i],
                atol=5e-2,
                rtol=5e-2,
                err_msg=f"Batch {batch_size}, Request {i} output diverges",
            )


if __name__ == "__main__":
    absltest.main()
