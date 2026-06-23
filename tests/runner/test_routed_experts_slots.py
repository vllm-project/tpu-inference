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

The routed experts of a step are stored into the scheduler-side slot buffer
(``RoutedExpertsManager``) keyed by physical KV-cache slot, and read back
block-relative from position 0. So the slot_mapping produced here must place a
request's tokens at their true absolute positions. ``_reconstruct_slots_for_request``
takes an explicit ``start_pos`` (the two call sites have different token
contracts); ``_reconstruct_routed_experts`` derives it for the routed-experts
path as ``num_computed_tokens - num_tokens`` (num_computed_tokens has already
been advanced past the step's tokens by reconstruction time).
"""
from types import SimpleNamespace

import numpy as np
import pytest

from tpu_inference.runner.tpu_runner import (_reconstruct_routed_experts,
                                             _reconstruct_slots_for_request)


def _req(num_computed_tokens, block_ids):
    # Only num_computed_tokens and block_ids are read; CachedRequestState
    # stores the per-attention-group block IDs as block_ids[0].
    return SimpleNamespace(num_computed_tokens=num_computed_tokens,
                           block_ids=[block_ids])


class TestReconstructSlotsForRequest:
    """Slot math given an explicit absolute start position."""

    def test_slots_are_block_relative_from_start_pos(self):
        # 5 tokens starting at absolute position 0 in block 1 -> slots 16..20.
        slots = _reconstruct_slots_for_request(_req(5, [1]),
                                               num_tokens=5,
                                               block_size=16,
                                               start_pos=0)
        np.testing.assert_array_equal(
            slots, np.array([16, 17, 18, 19, 20], dtype=np.int32))

    def test_single_token_at_offset(self):
        # One token at absolute position 5 in block 1 -> slot 21.
        slots = _reconstruct_slots_for_request(_req(6, [1]),
                                               num_tokens=1,
                                               block_size=16,
                                               start_pos=5)
        np.testing.assert_array_equal(slots, np.array([21], dtype=np.int32))

    def test_spanning_multiple_blocks(self):
        # 20 tokens from position 0 across blocks [1, 2], block_size 16.
        slots = _reconstruct_slots_for_request(_req(20, [1, 2]),
                                               num_tokens=20,
                                               block_size=16,
                                               start_pos=0)
        expected = np.concatenate(
            [1 * 16 + np.arange(16), 2 * 16 + np.arange(4)]).astype(np.int32)
        np.testing.assert_array_equal(slots, expected)

    def test_zero_tokens_returns_empty(self):
        slots = _reconstruct_slots_for_request(_req(5, [1]),
                                               num_tokens=0,
                                               block_size=16,
                                               start_pos=0)
        assert slots.size == 0

    def test_negative_start_pos_raises(self):
        # A negative start_pos would silently produce wrong slots via numpy
        # negative indexing; the helper must reject it loudly instead.
        with pytest.raises(AssertionError):
            _reconstruct_slots_for_request(_req(3, [1]),
                                           num_tokens=5,
                                           block_size=16,
                                           start_pos=-2)


class TestReconstructRoutedExperts:
    """End-to-end slot derivation for the routed-experts reconstruction path
    (the caller that had the off-by-num_tokens bug)."""

    def test_prefill_slots_block_relative_and_routing_preserved(self):
        req_id = "req0"
        n = 5  # 5-token prompt prefill
        block_size = 16
        num_layers, top_k = 2, 4
        # num_computed_tokens already advanced past the chunk (post-prefill = 5).
        req_state = _req(num_computed_tokens=n, block_ids=[1])
        runner = SimpleNamespace(block_size=block_size,
                                 dp_size=1,
                                 requests={req_id: req_state})
        scheduler_output = SimpleNamespace(
            num_scheduled_tokens={req_id: n},
            total_num_scheduled_tokens=n,
        )
        # Distinct per-(layer, token) values to detect any mis-mapping.
        expert_indices_cpu = np.arange(num_layers * n * top_k,
                                       dtype=np.int32).reshape(
                                           num_layers, n, top_k)

        result = _reconstruct_routed_experts(
            runner=runner,
            scheduler_output=scheduler_output,
            expert_indices_cpu=expert_indices_cpu,
            req_ids=[req_id],
            req_ids_dp={0: [req_id]},
            padded_num_scheduled_tokens_per_dp_rank=n,
        )

        # The fix: prompt tokens at positions 0..4 in block 1 -> slots 16..20.
        # (The pre-fix code produced 21..25, which the scheduler reads as zeros.)
        np.testing.assert_array_equal(
            result.slot_mapping, np.array([16, 17, 18, 19, 20],
                                          dtype=np.int32))
        # dp_size==1 -> identity reorder; routing_data is the model output
        # transposed to (tokens, layers, top_k).
        np.testing.assert_array_equal(result.routing_data,
                                      expert_indices_cpu.transpose(1, 0, 2))
