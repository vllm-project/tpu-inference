# Copyright 2025 Google LLC
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
"""Device-free tests for the RPA v3 tuned 4-tuple block-size table.

These run on any host (including CPU CI): the table and its lookup are pure
Python. Two things are guaranteed here:

  1. Every entry satisfies the SAME constraints the kernel's caller-side
     validator (`_validate_block_sizes`) enforces on operator overrides, so a
     malformed tuned entry is caught at test time, never on silicon.
  2. A miss returns None so the kernel falls back to the heuristic (an untuned
     shape is never regressed), and the one seeded shape resolves to its
     measured 4-tuple under the exact production lookup key.
"""

from unittest import mock

import jax.numpy as jnp
from absl.testing import absltest, parameterized
from jax._src import test_util as jtu

from tpu_inference.kernels.ragged_paged_attention.v3 import (
    tuned_block_sizes_v3 as tb3)

_VALID_CASES = ("decode", "prefill", "mixed")


def _iter_entries():
    """Yield (device, page_size, dtypes, heads, extra, case_name, value)."""
    for device, by_page in tb3.TUNED_BLOCK_SIZES_V3.items():
        for page_size, by_dtype in by_page.items():
            for dtypes, by_heads in by_dtype.items():
                for heads, by_len in by_heads.items():
                    for extra, by_case in by_len.items():
                        for case_name, value in by_case.items():
                            yield (device, page_size, dtypes, heads, extra,
                                   case_name, value)


class TunedBlockSizesV3TableTest(parameterized.TestCase):
    """Structural / self-consistency checks on the table itself."""

    def test_every_entry_is_a_valid_four_tuple(self):
        entries = list(_iter_entries())
        self.assertGreater(len(entries), 0, "table must not be empty")
        for (device, page_size, dtypes, heads, extra, case_name,
             value) in entries:
            ctx = f"{device}/{page_size}/{dtypes}/{heads}/{extra}/{case_name}"
            self.assertIn(case_name, _VALID_CASES, ctx)
            self.assertLen(value, 4, f"{ctx}: expected (bq,bkv,bqc,bkvc)")
            for v in value:
                self.assertIsInstance(v, int, ctx)
                self.assertGreater(v, 0, ctx)

    def test_every_entry_passes_caller_validator_constraints(self):
        # Mirror kernel._validate_block_sizes exactly. The page_size used by
        # the validator is the runtime page_size; the table key's page_size is
        # the same value (next_power_of_2), so use it as the divisor.
        for (device, page_size, dtypes, heads, extra, case_name,
             value) in _iter_entries():
            bq_sz, bkv_sz, bq_csz, bkv_csz = value
            ctx = f"{device}/{page_size}/{dtypes}/{heads}/{extra}/{case_name}"
            self.assertTrue(0 < bq_csz and bq_sz % bq_csz == 0,
                            f"{ctx}: bq_sz % bq_csz != 0")
            self.assertTrue(0 < bkv_csz and bkv_sz % bkv_csz == 0,
                            f"{ctx}: bkv_sz % bkv_csz != 0")
            self.assertEqual(bkv_sz % page_size, 0,
                             f"{ctx}: bkv_sz not page-aligned")
            self.assertEqual(bkv_csz % page_size, 0,
                             f"{ctx}: bkv_csz not page-aligned")

    def test_decode_entries_use_fetch_compute_split(self):
        # The whole reason this table exists: decode entries decouple fetch
        # (bkv_sz) from compute (bkv_csz). If they were equal the heuristic
        # would already produce them and the entry would be pointless.
        saw_decode = False
        for (_d, _p, _dt, _h, _e, case_name, value) in _iter_entries():
            if case_name == "decode":
                saw_decode = True
                _, bkv_sz, _, bkv_csz = value
                self.assertGreaterEqual(
                    bkv_sz, bkv_csz,
                    "decode fetch block should be >= compute block")
        self.assertTrue(saw_decode, "expected at least one decode entry")


class TunedBlockSizesV3LookupTest(parameterized.TestCase):
    """Lookup behaviour: hit resolves, miss returns None (heuristic fallback)."""

    def setUp(self):
        super().setUp()
        # get_device_name() reads jax.devices(); pin it so the lookup runs on
        # any host (CPU CI included).
        p = mock.patch(
            "tpu_inference.kernels.ragged_paged_attention.v3."
            "tuned_block_sizes.get_device_name",
            return_value="TPU v7")
        p.start()
        self.addCleanup(p.stop)

    def test_seeded_qwen3_decode_shape_resolves(self):
        # Qwen3-0.6B, DP16xTP4 per-shard: q_head=4, kv_head=2, head_dim=128,
        # page_size=128, max_model_len=16384 -> the measured +48% decode tuple.
        value = tb3.get_tuned_block_sizes_v3(
            q_dtype=jnp.bfloat16,
            kv_dtype=jnp.bfloat16,
            actual_num_q_heads=4,
            actual_num_kv_heads=2,
            head_dim=128,
            page_size=128,
            max_num_tokens=64,
            pages_per_seq=128,  # 128 * 128 = 16384 max_model_len
            case_name="decode",
        )
        self.assertEqual(value, (1, 16384, 1, 4096))

    def test_untuned_shape_returns_none(self):
        # A head config not in the table must miss -> None -> heuristic.
        value = tb3.get_tuned_block_sizes_v3(
            q_dtype=jnp.bfloat16,
            kv_dtype=jnp.bfloat16,
            actual_num_q_heads=32,
            actual_num_kv_heads=8,
            head_dim=128,
            page_size=128,
            max_num_tokens=64,
            pages_per_seq=64,
            case_name="decode",
        )
        self.assertIsNone(value)

    def test_untuned_case_returns_none(self):
        # The seeded shape has only a decode entry; prefill/mixed must miss.
        for case_name in ("prefill", "mixed"):
            value = tb3.get_tuned_block_sizes_v3(
                q_dtype=jnp.bfloat16,
                kv_dtype=jnp.bfloat16,
                actual_num_q_heads=4,
                actual_num_kv_heads=2,
                head_dim=128,
                page_size=128,
                max_num_tokens=64,
                pages_per_seq=64,
                case_name=case_name,
            )
            self.assertIsNone(value, f"{case_name} should miss")


if __name__ == "__main__":
    absltest.main(testLoader=jtu.JaxTestLoader())
