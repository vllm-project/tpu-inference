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
"""Scatter-bounds tests for prepare_boundary_batch.

The combined-batch scatter routes every non-boundary / invalid token to a
sentinel destination one past the end of the output. These tests pin that
sentinel rows can never corrupt live output rows, at several output sizes:
on TPU, drop-mode scatter semantics for out-of-bounds indices have been
observed (jax 0.10.2) to hold at one row width and silently wrap at
another, so the packing must keep every index in bounds by construction
rather than rely on them being dropped.
"""

import jax.numpy as jnp
import numpy as np
from absl.testing import absltest, parameterized

from tpu_inference.kernels.experimental.deepseek_v4.compress_and_store.compressor_v1 import \
    prepare_boundary_batch


def _run(positions, req_indices, kv_slot_mapping, is_boundary, is_decode,
         compress_ratio, num_reqs):
    return prepare_boundary_batch(
        jnp.asarray(positions, dtype=jnp.int32),
        jnp.asarray(req_indices, dtype=jnp.int32),
        jnp.asarray(kv_slot_mapping, dtype=jnp.int32),
        jnp.asarray(is_boundary, dtype=bool),
        jnp.asarray(is_decode, dtype=bool),
        compress_ratio,
        num_reqs,
    )


class BoundaryBatchScatterTest(parameterized.TestCase):

    @parameterized.parameters(64, 128, 1024, 1030)
    def test_no_valid_tokens_leaves_outputs_at_defaults(self, num_tokens):
        """All-sentinel scatter: every output row must keep its default.

        This is the wrap canary: if the sentinel destinations are not
        discarded (e.g. wrap back into range), row 0 is the first row to
        be silently overwritten. NOTE: CPU honors drop-mode semantics for
        out-of-bounds indices, so on CPU this canary passes even against
        the old sentinel-scatter code — it only has teeth on the TPU
        kernels CI job, where the wrap miscompile lives. A green CPU run
        checks the packing contract, not the miscompile.
        """
        compress_ratio = 4
        num_reqs = 2
        positions = np.arange(num_tokens)
        req_indices = np.zeros(num_tokens, dtype=np.int32)
        # Boundary tokens exist, but none map to a KV slot (kv = -1), so
        # every packed row is invalid and every destination is the
        # sentinel.
        is_boundary = (positions + 1) % compress_ratio == 0
        kv_slot_mapping = np.full(num_tokens, -1, dtype=np.int32)
        is_decode = np.zeros(num_tokens, dtype=bool)

        pos_out, req_out, kv_out = _run(positions, req_indices,
                                        kv_slot_mapping, is_boundary,
                                        is_decode, compress_ratio, num_reqs)

        np.testing.assert_array_equal(np.asarray(pos_out), 0)
        np.testing.assert_array_equal(np.asarray(req_out), 0)
        np.testing.assert_array_equal(np.asarray(kv_out), -1)

    @parameterized.parameters(64, 1024)
    def test_valid_tokens_land_exactly_once_and_nothing_else(self, num_tokens):
        """Every valid boundary token appears exactly once; all other rows
        keep defaults."""
        compress_ratio = 4
        num_reqs = 2
        rng = np.random.default_rng(0)
        positions = np.arange(num_tokens)
        req_indices = (positions >= num_tokens // 2).astype(np.int32)
        is_boundary = (positions + 1) % compress_ratio == 0
        # Half the boundary tokens get real KV slots, half stay -1
        # (mid-block, not yet at a compression boundary in the cache).
        kv_slot_mapping = np.full(num_tokens, -1, dtype=np.int32)
        boundary_idx = np.nonzero(is_boundary)[0]
        chosen = rng.choice(boundary_idx,
                            size=len(boundary_idx) // 2,
                            replace=False)
        # Distinct positive slots so multiset comparison is exact.
        kv_slot_mapping[np.sort(chosen)] = 1000 + np.arange(len(chosen)) * 8
        is_decode = positions < 4  # a few decode rows

        pos_out, req_out, kv_out = _run(positions, req_indices,
                                        kv_slot_mapping, is_boundary,
                                        is_decode, compress_ratio, num_reqs)
        kv_out = np.asarray(kv_out)
        pos_out = np.asarray(pos_out)
        req_out = np.asarray(req_out)

        expected_kv = np.sort(kv_slot_mapping[(kv_slot_mapping >= 0)
                                              & is_boundary])
        actual_kv = np.sort(kv_out[kv_out >= 0])
        np.testing.assert_array_equal(actual_kv, expected_kv)

        # Rows that did not receive a valid token keep their defaults.
        invalid_rows = kv_out < 0
        np.testing.assert_array_equal(kv_out[invalid_rows], -1)
        np.testing.assert_array_equal(pos_out[invalid_rows], 0)
        np.testing.assert_array_equal(req_out[invalid_rows], 0)

        # And the valid rows carry the position/req of their token.
        pos_by_kv = {
            int(kv_slot_mapping[i]): (int(positions[i]), int(req_indices[i]))
            for i in np.nonzero(kv_slot_mapping >= 0)[0]
        }
        for row in np.nonzero(kv_out >= 0)[0]:
            expect_pos, expect_req = pos_by_kv[int(kv_out[row])]
            self.assertEqual(int(pos_out[row]), expect_pos)
            self.assertEqual(int(req_out[row]), expect_req)

    def test_decode_rows_stay_in_packed_prefix(self):
        """Decode boundary tokens land at their packed indices [0, K)."""
        compress_ratio = 4
        num_tokens = 32
        num_reqs = 4
        positions = np.arange(num_tokens)
        req_indices = positions // (num_tokens // num_reqs)
        is_boundary = (positions + 1) % compress_ratio == 0
        kv_slot_mapping = np.where(is_boundary, 100 + positions * 4,
                                   -1).astype(np.int32)
        is_decode = np.ones(num_tokens, dtype=bool)

        _, _, kv_out = _run(positions, req_indices, kv_slot_mapping,
                            is_boundary, is_decode, compress_ratio, num_reqs)
        kv_out = np.asarray(kv_out)

        num_decode = int(is_boundary.sum())
        expected = kv_slot_mapping[is_boundary]
        np.testing.assert_array_equal(kv_out[:num_decode], expected)
        np.testing.assert_array_equal(kv_out[num_decode:], -1)


if __name__ == "__main__":
    absltest.main()
