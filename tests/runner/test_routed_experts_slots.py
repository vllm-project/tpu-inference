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
"""Unit tests for routed-experts physical slot reconstruction.

``_reconstruct_slots_for_request`` produces the slot_mapping under which a
step's routed experts are stored into the scheduler-side slot buffer
(``RoutedExpertsManager``). The scheduler reads them back block-relative from
position 0, so the slots must be derived from the chunk's *start* position
(``num_computed_tokens - num_tokens``), since ``num_computed_tokens`` has
already been advanced past the step's tokens by reconstruction time.
"""
from types import SimpleNamespace

import numpy as np

from tpu_inference.runner.tpu_runner import _reconstruct_slots_for_request


def _req(num_computed_tokens, block_ids):
    # Only num_computed_tokens and block_ids are read; CachedRequestState
    # stores the per-attention-group block IDs as block_ids[0].
    return SimpleNamespace(num_computed_tokens=num_computed_tokens,
                           block_ids=[block_ids])


class TestReconstructSlotsForRequest:

    def test_prefill_slots_are_block_relative_from_zero(self):
        # 5-token prompt prefill; num_computed_tokens has advanced to 5.
        # Tokens occupy positions 0..4 in block 1 -> slots 16..20.
        # (Using num_computed_tokens directly would give the wrong 21..25.)
        slots = _reconstruct_slots_for_request(_req(5, [1]),
                                               num_tokens=5,
                                               block_size=16)
        np.testing.assert_array_equal(
            slots, np.array([16, 17, 18, 19, 20], dtype=np.int32))

    def test_decode_step_slot(self):
        # One decode token after a 5-token prompt; num_computed_tokens -> 6,
        # so the token sits at absolute position 5 in block 1 -> slot 21.
        slots = _reconstruct_slots_for_request(_req(6, [1]),
                                               num_tokens=1,
                                               block_size=16)
        np.testing.assert_array_equal(slots, np.array([21], dtype=np.int32))

    def test_prefill_spanning_multiple_blocks(self):
        # 20-token prefill across two blocks [1, 2], block_size 16:
        # positions 0..15 -> block 1, positions 16..19 -> block 2.
        slots = _reconstruct_slots_for_request(_req(20, [1, 2]),
                                               num_tokens=20,
                                               block_size=16)
        expected = np.concatenate(
            [1 * 16 + np.arange(16), 2 * 16 + np.arange(4)]).astype(np.int32)
        np.testing.assert_array_equal(slots, expected)

    def test_zero_tokens_returns_empty(self):
        slots = _reconstruct_slots_for_request(_req(5, [1]),
                                               num_tokens=0,
                                               block_size=16)
        assert slots.size == 0
