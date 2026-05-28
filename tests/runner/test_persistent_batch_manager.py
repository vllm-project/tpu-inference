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

from tpu_inference.runner.persistent_batch_manager import \
    PersistentBatchManager


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


def _create_manager(req_ids, num_scheduled_tokens_map):
    """Helper to create a PersistentBatchManager with a MockInputBatch
    and a mock scheduler_output.

    Args:
        req_ids: list of request id strings, in the order they appear
            in the batch.
        num_scheduled_tokens_map: dict mapping req_id -> num_scheduled_tokens.

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


class TestPersistentBatchManager(unittest.TestCase):

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
