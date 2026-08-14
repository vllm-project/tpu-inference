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
"""Tests for the DSA top-k mask path of the MLA v2 kernel.

These avoid hand-rolling a masked-MLA reference by leaning on two exact
equivalences against behaviour the kernel is already trusted for:

  1. An all-ones top-k mask must reproduce dense attention bit-for-bit.
  2. A mask that encodes a sliding window must reproduce the kernel's own
     ``sliding_window`` result bit-for-bit.

(1) alone would pass even for a kernel that ignored the mask entirely, so (2)
is the one that actually proves the mask is read, indexed and applied
correctly -- it excludes real positions and the expected answer is independent
of the mask code path.
"""

import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest
from jax._src import test_util as jtu

import tpu_inference.kernels.mla.v2.kernel as kernel_v2
from tests.kernels.mla_v2_test import align_to, generate_mla_inputs

jax.config.parse_flags_with_absl()


class MlaV2SparseTest(jtu.JaxTestCase):
    kv_dtype = jnp.float8_e4m3fn

    def _run(self,
             *,
             seq_lens,
             num_heads,
             lkv_dim,
             r_dim,
             page_size,
             num_pages,
             topk_mask,
             sliding_window,
             q_dtype=jnp.bfloat16):
        rng = np.random.default_rng(1234)
        (
            ql_nope,
            q_pe,
            new_kv_c,
            new_k_pe,
            cache_kv,
            kv_lens,
            page_indices,
            cu_q_lens,
            distribution,
        ) = generate_mla_inputs(
            seq_lens,
            num_heads,
            lkv_dim,
            r_dim,
            page_size,
            q_dtype,
            self.kv_dtype,
            num_pages,
            rng=rng,
        )
        out, _ = kernel_v2.mla_ragged_paged_attention(
            jnp.transpose(ql_nope, (1, 0, 2)),
            q_pe,
            new_kv_c,
            new_k_pe,
            cache_kv.copy(),
            kv_lens,
            page_indices,
            cu_q_lens,
            distribution,
            sm_scale=1.0,
            sliding_window=sliding_window,
            topk_mask=topk_mask,
            s_dtype=jnp.float32,
            decode_batch_size=4,
            num_kv_pages_per_block=8,
            num_queries_per_block=8,
            vmem_limit_bytes=100 * 1024 * 1024,
        )
        return jnp.transpose(out, (1, 0, 2))

    @staticmethod
    def _mask_width(seq_lens):
        return align_to(max(kv for _, kv in seq_lens), 128)

    @staticmethod
    def _sliding_window_mask(seq_lens, width, window):
        """1 where a token may attend, matching the kernel's window semantics.

        The kernel masks a position out when ``q_pos - sliding_window >= k``,
        so a position is kept when ``k > q_pos - window``. Causality is left to
        the kernel's own causal mask.
        """
        total_q = sum(q for q, _ in seq_lens)
        mask = np.zeros((total_q, width), np.uint8)
        row = 0
        for q_len, kv_len in seq_lens:
            for i in range(q_len):
                q_pos = kv_len - q_len + i
                lo = max(q_pos - window + 1, 0)
                mask[row, lo:kv_len] = 1
                row += 1
        return jnp.asarray(mask)

    def setUp(self):
        super().setUp()
        if not jtu.is_device_tpu_at_least(version=4):
            self.skipTest("Expect TPUv4+")

    def test_all_ones_mask_matches_dense(self):
        """A fully permissive mask must not change the result at all."""
        seq_lens = [(192, 328), (128, 180), (64, 255)]
        cfg = dict(seq_lens=seq_lens,
                   num_heads=128,
                   lkv_dim=512,
                   r_dim=64,
                   page_size=128,
                   num_pages=1024)
        width = self._mask_width(seq_lens)
        total_q = sum(q for q, _ in seq_lens)

        dense = self._run(**cfg, topk_mask=None, sliding_window=None)
        sparse = self._run(
            **cfg,
            topk_mask=jnp.ones((total_q, width), jnp.uint8),
            sliding_window=None,
        )
        self.assertArraysEqual(dense, sparse)

    def test_mask_encoding_sliding_window_matches_sliding_window(self):
        """The discriminating case: the mask must actually exclude positions.

        Expected values come from the kernel's independent sliding-window path,
        so this cannot be satisfied by ignoring the mask.
        """
        seq_lens = [(192, 328), (128, 180), (64, 255)]
        cfg = dict(seq_lens=seq_lens,
                   num_heads=128,
                   lkv_dim=512,
                   r_dim=64,
                   page_size=128,
                   num_pages=1024)
        window = 64
        width = self._mask_width(seq_lens)

        expected = self._run(**cfg, topk_mask=None, sliding_window=window)
        got = self._run(
            **cfg,
            topk_mask=self._sliding_window_mask(seq_lens, width, window),
            sliding_window=None,
        )
        self.assertArraysEqual(expected, got)

    def test_decode_mask_encoding_sliding_window(self):
        """Same equivalence on the batched-decode path (q_len == 1)."""
        seq_lens = [(1, 328), (1, 180), (1, 255), (1, 97)]
        cfg = dict(seq_lens=seq_lens,
                   num_heads=64,
                   lkv_dim=512,
                   r_dim=64,
                   page_size=128,
                   num_pages=1024)
        window = 128
        width = self._mask_width(seq_lens)

        expected = self._run(**cfg, topk_mask=None, sliding_window=window)
        got = self._run(
            **cfg,
            topk_mask=self._sliding_window_mask(seq_lens, width, window),
            sliding_window=None,
        )
        self.assertArraysEqual(expected, got)

    def test_rejects_malformed_mask(self):
        seq_lens = [(64, 128)]
        cfg = dict(seq_lens=seq_lens,
                   num_heads=64,
                   lkv_dim=512,
                   r_dim=64,
                   page_size=128,
                   num_pages=64)
        total_q = sum(q for q, _ in seq_lens)

        with self.assertRaisesRegex(ValueError, "multiple of 128"):
            self._run(**cfg,
                      topk_mask=jnp.ones((total_q, 100), jnp.uint8),
                      sliding_window=None)

        with self.assertRaisesRegex(ValueError, "max_num_tokens"):
            self._run(**cfg,
                      topk_mask=jnp.ones((total_q + 8, 128), jnp.uint8),
                      sliding_window=None)

        with self.assertRaisesRegex(ValueError, "dtype"):
            self._run(**cfg,
                      topk_mask=jnp.ones((total_q, 128), jnp.int32),
                      sliding_window=None)


if __name__ == "__main__":
    absltest.main(testLoader=jtu.JaxTestLoader())
