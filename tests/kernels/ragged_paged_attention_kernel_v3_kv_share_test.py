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
"""Tests for the KV-share path in v3 ragged_paged_attention.

When `update_kv_cache=False` (used by gemma-4 KV-shared layers, where the
cache slot has been redirected to a source layer that already wrote its
normed/roped K,V), the kernel must:

  1. Compute attention reading K,V *only* from the (pre-populated) cache.
  2. Ignore the input `keys` / `values` arrays entirely.
  3. Leave the cache unchanged.

The pre-fix kernel split each block into `(past from cache, current from
input k,v)`, producing a corrupt mix of source-normed-roped-K with
shared-raw-K. The fix (kernel.py `_fetch_bkv`) forces all of `kv_left`
to come from the cache when `update_kv_cache=False`. These tests guard
that behavior against regression.
"""

import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest
from jax._src import test_util as jtu

from tpu_inference.kernels.ragged_paged_attention.v3.kernel import (
    ragged_paged_attention, )
from tpu_inference.kernels.ragged_paged_attention.v3.util import (
    align_to, cdiv, get_dtype_packing)

jax.config.parse_flags_with_absl()


@jtu.with_config(jax_numpy_dtype_promotion="standard")
class KvShareKernelTest(jtu.JaxTestCase):

    def setUp(self):
        super().setUp()
        if not jtu.is_device_tpu_at_least(version=4):
            self.skipTest("Expect TPUv4+")

    def _build_inputs(
        self,
        *,
        q_len: int,
        kv_len: int,
        kv_input_seed: int,
        num_q_heads: int = 8,
        num_kv_heads: int = 1,
        head_dim: int = 128,
        page_size: int = 16,
        num_pages: int = 8,
        max_num_seqs: int = 8,
        cache_seed: int = 42,
        q_seed: int = 123,
        dtype=jnp.bfloat16,
    ):
        """Build a single-seq kernel input tuple with a pre-populated cache.

        Cache contents are determined by `cache_seed`; q is determined by
        `q_seed`. Input k,v are determined by `kv_input_seed` — varying
        this between calls lets us check that the kernel's output is
        invariant to input k,v when `update_kv_cache=False`.
        """
        rng_q = np.random.default_rng(q_seed)
        rng_cache = np.random.default_rng(cache_seed)
        rng_input = np.random.default_rng(kv_input_seed)

        pages_per_seq = cdiv(kv_len, page_size)
        max_num_batched_tokens = max(align_to(q_len, 128), 128)
        kv_packing = get_dtype_packing(dtype)
        num_kv_heads_x2 = align_to(num_kv_heads * 2, kv_packing)
        padded_hd = align_to(head_dim, 128)

        q = jnp.array(
            rng_q.random((max_num_batched_tokens, num_q_heads, head_dim),
                         dtype=np.float32)).astype(dtype)
        k = jnp.array(
            rng_input.random((max_num_batched_tokens, num_kv_heads, head_dim),
                             dtype=np.float32)).astype(dtype)
        v = jnp.array(
            rng_input.random((max_num_batched_tokens, num_kv_heads, head_dim),
                             dtype=np.float32)).astype(dtype)

        cache_data = jnp.array(
            rng_cache.random((pages_per_seq * page_size,
                              num_kv_heads_x2 // kv_packing, kv_packing,
                              padded_hd),
                             dtype=np.float32)).astype(dtype)
        cache_pages = cache_data.reshape(pages_per_seq, page_size,
                                         num_kv_heads_x2 // kv_packing,
                                         kv_packing, padded_hd)
        # Padding pages stay nan to surface any out-of-bounds reads.
        kv_cache = jnp.pad(
            cache_pages,
            ((0, num_pages - pages_per_seq), (0, 0), (0, 0), (0, 0), (0, 0)),
            constant_values=jnp.nan,
        )

        page_indices = jnp.zeros((max_num_seqs * pages_per_seq, ),
                                 dtype=jnp.int32)
        page_indices = page_indices.at[:pages_per_seq].set(
            jnp.arange(pages_per_seq, dtype=jnp.int32))

        kv_lens_arr = jnp.zeros((max_num_seqs, ),
                                dtype=jnp.int32).at[0].set(kv_len)
        cu_q_lens_arr = jnp.zeros((max_num_seqs + 1, ),
                                  dtype=jnp.int32).at[1].set(q_len)
        # distribution[3] = (decode_end, prefill_end, mixed_end). One seq:
        # decode if q_len==1 else prefill.
        if q_len == 1:
            distribution = jnp.array([1, 1, 1], dtype=jnp.int32)
        else:
            distribution = jnp.array([0, 1, 1], dtype=jnp.int32)

        return (q, k, v, kv_cache, kv_lens_arr, page_indices, cu_q_lens_arr,
                distribution)

    def _kwargs(self, head_dim: int = 128):
        return dict(
            sm_scale=1.0 / float(head_dim)**0.5,
            update_kv_cache=False,
            m_block_sizes=(64, 256, 32, 128),
        )

    def test_prefill_input_kv_is_ignored(self):
        """q_len == kv_len. Two calls with different input k,v but the same
        pre-populated cache and same q produce bit-identical outputs."""
        args1 = self._build_inputs(q_len=16, kv_len=16, kv_input_seed=11)
        args2 = self._build_inputs(q_len=16, kv_len=16, kv_input_seed=99)
        # Sanity (must happen BEFORE the kernel call — the kernel donates
        # queries/keys/values/kv_cache, so args1[3]/args2[3] are deleted
        # after the call).
        np.testing.assert_array_equal(args1[0], args2[0])  # q
        np.testing.assert_array_equal(args1[3], args2[3])  # kv_cache
        self.assertFalse(np.array_equal(args1[1], args2[1]))  # k differs
        self.assertFalse(np.array_equal(args1[2], args2[2]))  # v differs
        cache_before = np.asarray(args1[3])

        out1, cache_after_1 = ragged_paged_attention(*args1, **self._kwargs())
        out2, cache_after_2 = ragged_paged_attention(*args2, **self._kwargs())

        # Output invariant to input k,v.
        self.assertArraysEqual(out1, out2)
        # Cache unchanged in both runs (use the pre-donation snapshot).
        mask = ~np.isnan(cache_before)
        np.testing.assert_array_equal(
            np.asarray(cache_after_1)[mask], cache_before[mask])
        np.testing.assert_array_equal(
            np.asarray(cache_after_2)[mask], cache_before[mask])

    def test_decode_input_kv_is_ignored(self):
        """q_len == 1, kv_len > 1 (decode step). Same invariance."""
        args1 = self._build_inputs(q_len=1, kv_len=33, kv_input_seed=11)
        args2 = self._build_inputs(q_len=1, kv_len=33, kv_input_seed=99)
        cache_before = np.asarray(args1[3])

        out1, cache_after_1 = ragged_paged_attention(*args1, **self._kwargs())
        out2, cache_after_2 = ragged_paged_attention(*args2, **self._kwargs())

        # Decode emits q_len = 1 token. Compare just that token (the rest of
        # the max_num_batched_tokens buffer is junk padding).
        self.assertArraysEqual(out1[:1], out2[:1])
        mask = ~np.isnan(cache_before)
        np.testing.assert_array_equal(
            np.asarray(cache_after_1)[mask], cache_before[mask])

    def test_prefill_default_update_kv_cache_still_writes(self):
        """Sanity: the fix only changes the `update_kv_cache=False` branch.
        With the default `update_kv_cache=True`, input k,v *are* written to
        the cache slot (and used for attention), so two calls with
        different input k,v produce different outputs."""
        args1 = self._build_inputs(q_len=16, kv_len=16, kv_input_seed=11)
        args2 = self._build_inputs(q_len=16, kv_len=16, kv_input_seed=99)

        kw = dict(
            sm_scale=1.0 / float(128)**0.5,
            update_kv_cache=True,
            m_block_sizes=(64, 256, 32, 128),
        )
        out1, _ = ragged_paged_attention(*args1, **kw)
        out2, _ = ragged_paged_attention(*args2, **kw)
        # Outputs MUST differ — input k,v participates in attention.
        self.assertFalse(
            np.allclose(np.asarray(out1[:16]).astype(np.float32),
                        np.asarray(out2[:16]).astype(np.float32),
                        atol=0,
                        rtol=0))


if __name__ == "__main__":
    absltest.main()
