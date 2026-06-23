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

import unittest
from unittest.mock import MagicMock, patch

import numpy as np
from vllm.sampling_params import SamplingParams

from tpu_inference.runner.input_batch import CachedRequestState, InputBatch
from tpu_inference.runner.persistent_batch_manager import \
    PersistentBatchManager


def _create_cached_request(req_id: str) -> CachedRequestState:
    return CachedRequestState(
        req_id=req_id,
        prompt_token_ids=[1, 2, 3],
        mm_features=[],
        sampling_params=SamplingParams(temperature=0.0),
        pooling_params=None,
        block_ids=[[1]],
        num_computed_tokens=0,
        lora_request=None,
        output_token_ids=[],
    )


def _make_scheduler_output(*,
                           scheduled_req_ids=(),
                           preempted_req_ids=None,
                           resumed_req_ids=(),
                           new_block_ids=None):
    scheduler_output = MagicMock()
    req_data = MagicMock()
    req_data.req_ids = list(scheduled_req_ids)
    req_data.resumed_req_ids = set(resumed_req_ids)
    req_data.num_computed_tokens = [0] * len(req_data.req_ids)
    req_data.new_token_ids = [[] for _ in req_data.req_ids]
    if new_block_ids is None:
        req_data.new_block_ids = [None for _ in req_data.req_ids]
    else:
        req_data.new_block_ids = new_block_ids
    req_data.num_output_tokens = [0] * len(req_data.req_ids)
    scheduler_output.scheduled_cached_reqs = req_data
    scheduler_output.scheduled_new_reqs = []
    scheduler_output.scheduled_spec_decode_tokens = {}
    scheduler_output.num_scheduled_tokens = {
        req_id: 1
        for req_id in scheduled_req_ids
    }
    scheduler_output.total_num_scheduled_tokens = len(scheduled_req_ids)
    scheduler_output.finished_req_ids = set()
    scheduler_output.free_encoder_mm_hashes = []
    scheduler_output.preempted_req_ids = set(preempted_req_ids or ())
    scheduler_output.assigned_dp_rank = {}
    return scheduler_output


class MockInputBatch:
    """Lightweight mock InputBatch that tracks the state needed by
    _reorder_batch: req_ids, request_distribution, and swap_states."""

    def __init__(self, req_ids: list[str]):
        self._req_ids = list(req_ids)
        self.req_id_to_index = {rid: i for i, rid in enumerate(req_ids)}
        self.request_distribution = [0, 0, 0]

    @property
    def num_reqs(self):
        return len(self._req_ids)

    @property
    def req_ids(self):
        return self._req_ids

    def swap_states(self, i: int, j: int):
        self._req_ids[i], self._req_ids[j] = (self._req_ids[j],
                                              self._req_ids[i])
        id_i, id_j = self._req_ids[i], self._req_ids[j]
        self.req_id_to_index[id_i] = i
        self.req_id_to_index[id_j] = j


def _create_manager(req_ids, num_scheduled_tokens_map, spec_req_ids=()):
    """Helper to create a PersistentBatchManager with a MockInputBatch
    and a mock scheduler_output.

    Args:
        req_ids: list of request id strings, in the order they appear
            in the batch.
        num_scheduled_tokens_map: dict mapping req_id -> num_scheduled_tokens.
        spec_req_ids: request ids that have scheduled spec-decode tokens. These
            are reflected in scheduler_output.scheduled_spec_decode_tokens so
            _reorder_batch classifies them as decode-region requests.

    Returns:
        (manager, scheduler_output) tuple.
    """
    input_batch = MockInputBatch(req_ids)

    manager = PersistentBatchManager(
        requests={},
        input_batch=input_batch,
        encoder_cache={},
        uses_mrope=False,
        model_config=MagicMock(),
        is_last_rank=True,
    )

    scheduler_output = MagicMock()
    scheduler_output.num_scheduled_tokens = num_scheduled_tokens_map
    scheduler_output.total_num_scheduled_tokens = sum(
        num_scheduled_tokens_map.values())
    scheduler_output.scheduled_spec_decode_tokens = {
        rid: [0]
        for rid in spec_req_ids
    }

    return manager, scheduler_output


class TestReorderBatch(unittest.TestCase):
    """Tests for the _reorder_batch method."""

    def test_empty_batch(self):
        """An empty batch should return 0 swaps and not modify anything."""
        manager, sched_out = _create_manager([], {})

        swap_cnt = manager._reorder_batch(sched_out)

        self.assertEqual(swap_cnt, 0)

    def test_all_decode_fast_path(self):
        """When all requests are decode (1 token each), the fast path should
        skip the two-pointer loop, return 0 swaps, and set distribution to
        all-decode."""
        req_ids = ["r0", "r1", "r2", "r3"]
        num_scheduled = {r: 1 for r in req_ids}
        manager, sched_out = _create_manager(req_ids, num_scheduled)

        with patch.object(manager.input_batch,
                          'swap_states',
                          wraps=manager.input_batch.swap_states) as mock_swap:
            swap_cnt = manager._reorder_batch(sched_out)

            self.assertEqual(swap_cnt, 0)
            self.assertEqual(manager.input_batch.request_distribution,
                             [4, 4, 4])
            mock_swap.assert_not_called()

    def test_mixed_batch_needs_swap(self):
        """Batch is [prefill, decode, decode, prefill] — needs reordering
        to move decodes to front."""
        req_ids = ["r0", "r1", "r2", "r3"]
        num_scheduled = {"r0": 10, "r1": 1, "r2": 1, "r3": 20}
        manager, sched_out = _create_manager(req_ids, num_scheduled)

        swap_cnt = manager._reorder_batch(sched_out)

        self.assertEqual(swap_cnt, 1)
        self.assertEqual(manager.input_batch.request_distribution, [2, 2, 4])
        result_ids = manager.input_batch.req_ids
        # First 2 should be decode requests
        for rid in result_ids[:2]:
            self.assertEqual(num_scheduled[rid], 1)
        # Last 2 should be prefill requests
        for rid in result_ids[2:]:
            self.assertGreater(num_scheduled[rid], 1)

    def test_spec_decode_classified_as_decode(self):
        """Spec-decode requests (q>1 but with scheduled spec tokens) go to the
        decode region."""
        req_ids = ["r0", "r1", "r2", "r3"]
        num_scheduled = {r: 4 for r in req_ids}  # 1 + 3 draft tokens
        manager, sched_out = _create_manager(req_ids,
                                             num_scheduled,
                                             spec_req_ids=req_ids)
        swap_cnt = manager._reorder_batch(sched_out)
        self.assertEqual(swap_cnt, 0)
        self.assertEqual(manager.input_batch.request_distribution, [4, 4, 4])

    def test_mixed_spec_and_prefill(self):
        """Spec decodes (in spec dict) move to front; chunked prefills (q>1, no
        spec tokens) stay at the back."""
        req_ids = ["r0", "r1", "r2", "r3"]
        # r0,r3 are chunked prefills (q>1, NOT spec); r1,r2 are spec decodes.
        num_scheduled = {"r0": 10, "r1": 4, "r2": 4, "r3": 20}
        manager, sched_out = _create_manager(req_ids,
                                             num_scheduled,
                                             spec_req_ids=["r1", "r2"])
        manager._reorder_batch(sched_out)
        self.assertEqual(manager.input_batch.request_distribution, [2, 2, 4])
        result_ids = manager.input_batch.req_ids
        for rid in result_ids[:2]:
            self.assertIn(rid, ("r1", "r2"))  # decode region = the spec reqs
        for rid in result_ids[2:]:
            self.assertIn(rid, ("r0", "r3"))  # mixed region = the prefills

    def test_spec_off_prefill_not_decode(self):
        """Spec OFF (empty spec dict): a q>1 request is a prefill, not decode."""
        req_ids = ["r0", "r1"]
        num_scheduled = {"r0": 1, "r1": 8}
        manager, sched_out = _create_manager(req_ids, num_scheduled)
        manager._reorder_batch(sched_out)
        self.assertEqual(manager.input_batch.request_distribution, [1, 1, 2])


class TestPersistentBatchManager(unittest.TestCase):

    def test_update_states_preserves_mamba_slot_for_unscheduled_request(self):
        req = _create_cached_request("req-0")
        requests = {req.req_id: req}
        input_batch = InputBatch(
            max_num_reqs=4,
            max_model_len=16,
            max_num_batched_tokens=16,
            pin_memory=False,
            vocab_size=128,
            block_sizes=[16],
        )
        input_batch.add_request(req)
        slot = int(input_batch.mamba_state_indices_cpu[0])

        manager = PersistentBatchManager(requests,
                                         input_batch,
                                         encoder_cache={},
                                         uses_mrope=False,
                                         model_config=MagicMock(),
                                         is_last_rank=True)

        manager.update_states(_make_scheduler_output(), None)

        self.assertEqual(req.mamba_state_slot, slot)
        self.assertFalse(
            any(slot in pool
                for pool in input_batch._free_mamba_slots_per_rank))

    def test_update_states_releases_mamba_slot_for_preempted_request(self):
        req = _create_cached_request("req-0")
        requests = {req.req_id: req}
        input_batch = InputBatch(
            max_num_reqs=4,
            max_model_len=16,
            max_num_batched_tokens=16,
            pin_memory=False,
            vocab_size=128,
            block_sizes=[16],
        )
        input_batch.add_request(req)
        slot = int(input_batch.mamba_state_indices_cpu[0])

        manager = PersistentBatchManager(requests,
                                         input_batch,
                                         encoder_cache={},
                                         uses_mrope=False,
                                         model_config=MagicMock(),
                                         is_last_rank=True)

        manager.update_states(
            _make_scheduler_output(preempted_req_ids={req.req_id}), None)

        self.assertIsNone(req.mamba_state_slot)
        self.assertTrue(
            any(slot in pool
                for pool in input_batch._free_mamba_slots_per_rank))

    def test_update_states_releases_preserved_mamba_slot_when_preempted(self):
        req = _create_cached_request("req-0")
        requests = {req.req_id: req}
        input_batch = InputBatch(
            max_num_reqs=4,
            max_model_len=16,
            max_num_batched_tokens=16,
            pin_memory=False,
            vocab_size=128,
            block_sizes=[16],
        )
        input_batch.add_request(req)
        slot = int(input_batch.mamba_state_indices_cpu[0])

        manager = PersistentBatchManager(requests,
                                         input_batch,
                                         encoder_cache={},
                                         uses_mrope=False,
                                         model_config=MagicMock(),
                                         is_last_rank=True)

        manager.update_states(_make_scheduler_output(), None)
        self.assertEqual(req.mamba_state_slot, slot)
        self.assertEqual(input_batch.num_reqs, 0)
        self.assertFalse(
            any(slot in pool
                for pool in input_batch._free_mamba_slots_per_rank))

        manager.update_states(
            _make_scheduler_output(preempted_req_ids={req.req_id}), None)

        self.assertIsNone(req.mamba_state_slot)
        self.assertTrue(
            any(slot in pool
                for pool in input_batch._free_mamba_slots_per_rank))

    def test_update_states_resets_preserved_mamba_slot_when_resumed(self):
        req = _create_cached_request("req-0")
        requests = {req.req_id: req}
        input_batch = InputBatch(
            max_num_reqs=4,
            max_model_len=16,
            max_num_batched_tokens=16,
            pin_memory=False,
            vocab_size=128,
            block_sizes=[16],
        )
        input_batch.add_request(req)
        slot = int(input_batch.mamba_state_indices_cpu[0])

        manager = PersistentBatchManager(requests,
                                         input_batch,
                                         encoder_cache={},
                                         uses_mrope=False,
                                         model_config=MagicMock(),
                                         is_last_rank=True)

        manager.update_states(_make_scheduler_output(), None)
        self.assertEqual(req.mamba_state_slot, slot)

        with patch.object(input_batch,
                          "release_mamba_slot",
                          wraps=input_batch.release_mamba_slot) as release:
            manager.update_states(
                _make_scheduler_output(
                    scheduled_req_ids=[req.req_id],
                    resumed_req_ids={req.req_id},
                    new_block_ids=[[[2]]],
                ), None)

        release.assert_called_with(slot)
        self.assertEqual(input_batch.num_reqs, 1)
        self.assertEqual(req.mamba_state_slot,
                         int(input_batch.mamba_state_indices_cpu[0]))

    def test_update_states_pp_non_last_rank(self):
        """
        the current rank is not the last rank.

        This test verifies that when new tokens are received from the scheduler,
        the internal state of the PersistentBatchManager (including request
        states and the input batch) is correctly updated.
        """

        req_id = 101
        initial_output_tokens = [10, 20]

        req_state = MagicMock()
        req_state.num_tokens = 2
        req_state.output_token_ids = list(initial_output_tokens)

        requests = {req_id: req_state}

        input_batch = MagicMock()
        input_batch.req_id_to_index = {req_id: 0}
        input_batch.num_prompt_tokens = np.array([2], dtype=np.int32)
        input_batch.token_ids_cpu = np.zeros((1, 10), dtype=np.int32)
        input_batch.num_tokens = np.array([2], dtype=np.int32)
        input_batch.num_tokens_no_spec = np.array([2], dtype=np.int32)
        input_batch.num_reqs = 1
        input_batch.req_ids = [req_id]
        input_batch.request_distribution = [0, 0, 0]

        encoder_cache = MagicMock()
        model_config = MagicMock()

        manager = PersistentBatchManager(requests,
                                         input_batch,
                                         encoder_cache,
                                         False,
                                         model_config,
                                         is_last_rank=False)

        scheduler_output = MagicMock()
        req_data = MagicMock()
        req_data.req_ids = [req_id]
        req_data.num_computed_tokens = [2]
        new_token_id = [30]
        req_data.new_token_ids = [new_token_id]
        req_data.new_block_ids = [None]
        req_data.num_output_tokens = [len(initial_output_tokens) + 1]
        scheduler_output.scheduled_cached_reqs = req_data
        scheduler_output.scheduled_spec_decode_tokens = {}
        scheduler_output.num_scheduled_tokens = {req_id: 1}
        scheduler_output.total_num_scheduled_tokens = 1

        manager.update_states(scheduler_output, None)

        expected_output_token_ids = initial_output_tokens + new_token_id
        self.assertEqual(req_state.output_token_ids, expected_output_token_ids)

        np.testing.assert_array_equal(
            manager.input_batch.token_ids_cpu[0, 2:3],
            np.array(new_token_id, dtype=np.int32))

        self.assertEqual(manager.input_batch.num_tokens[0], 3)
        self.assertEqual(manager.input_batch.num_tokens_no_spec[0], 3)
