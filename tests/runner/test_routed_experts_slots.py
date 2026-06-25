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
contracts); ``_reconstruct_routed_experts`` sources it from ``scheduler_output``
(the pre-step computed-token count), which is correct on both the sync and
async output paths -- unlike ``req_state.num_computed_tokens``, whose
advancement timing differs between them.
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


def _new_req_sched(req_id, num_computed_tokens):
    return SimpleNamespace(req_id=req_id,
                           num_computed_tokens=num_computed_tokens)


def _empty_cached():
    return SimpleNamespace(req_ids=[], num_computed_tokens=[])


class TestReconstructRoutedExperts:
    """End-to-end slot derivation for the routed-experts reconstruction path
    (the caller that had the slot-start bug). The chunk start is sourced from
    scheduler_output (pre-step), NOT from req_state.num_computed_tokens, whose
    advancement timing differs between the sync and async output paths."""

    def test_new_request_prefill_slots_block_relative(self):
        req_id = "req0"
        n = 5  # 5-token prompt prefill
        block_size = 16
        num_layers, top_k = 2, 4
        # Set req_state.num_computed_tokens to a wrong sentinel: the result must
        # be derived from scheduler_output (chunk start 0), not this field.
        req_state = _req(num_computed_tokens=99999, block_ids=[1])
        runner = SimpleNamespace(block_size=block_size,
                                 dp_size=1,
                                 requests={req_id: req_state})
        scheduler_output = SimpleNamespace(
            num_scheduled_tokens={req_id: n},
            total_num_scheduled_tokens=n,
            scheduled_new_reqs=[_new_req_sched(req_id, 0)],
            scheduled_cached_reqs=_empty_cached(),
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

        # Prompt tokens at positions 0..4 in block 1 -> slots 16..20.
        # (The buggy num_computed_tokens-based code underflowed to -5 on the
        # sync path and produced 21..25 on the async path; sourcing the chunk
        # start from scheduler_output is correct for both.)
        np.testing.assert_array_equal(
            result.slot_mapping, np.array([16, 17, 18, 19, 20],
                                          dtype=np.int32))
        # dp_size==1 -> identity reorder; routing_data is the model output
        # transposed to (tokens, layers, top_k).
        np.testing.assert_array_equal(result.routing_data,
                                      expert_indices_cpu.transpose(1, 0, 2))

    def test_cached_request_chunk_slots_use_pre_step_start(self):
        # A cached/running request whose chunk starts at absolute position 20.
        req_id = "reqC"
        n = 1
        block_size = 16
        num_layers, top_k = 2, 4
        req_state = _req(num_computed_tokens=99999, block_ids=[1, 2])
        runner = SimpleNamespace(block_size=block_size,
                                 dp_size=1,
                                 requests={req_id: req_state})
        scheduler_output = SimpleNamespace(
            num_scheduled_tokens={req_id: n},
            total_num_scheduled_tokens=n,
            scheduled_new_reqs=[],
            scheduled_cached_reqs=SimpleNamespace(req_ids=[req_id],
                                                  num_computed_tokens=[20]),
        )
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

        # Position 20 -> block_ids[20 // 16 = 1] = block 2 -> 2*16 + 20%16 = 36.
        np.testing.assert_array_equal(result.slot_mapping,
                                      np.array([36], dtype=np.int32))
