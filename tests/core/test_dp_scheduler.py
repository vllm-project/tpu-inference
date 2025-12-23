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

from unittest.mock import MagicMock, patch

import pytest
from vllm.config import VllmConfig
from vllm.v1.core.sched.output import CachedRequestData, SchedulerOutput
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.metrics.stats import PrefixCacheStats, SchedulerStats
from vllm.v1.request import Request

from tpu_inference.core.sched.dp_scheduler import (
    DPScheduler, DPSchedulerOutput, SchedulerCommand,
    update_vllm_config_for_dp_scheduler)


class TestDPScheduler:

    @pytest.fixture
    def mock_vllm_config(self):
        """Create a mock VllmConfig for testing."""
        config = MagicMock(spec=VllmConfig)
        config.sharding_config = MagicMock()
        config.sharding_config.total_dp_size = 2
        config.scheduler_config = MagicMock()
        config.scheduler_config._original_scheduler_cls = Scheduler
        config.scheduler_config.max_num_seqs = 8
        config.scheduler_config.max_num_batched_tokens = 1024
        config.scheduler_config.async_scheduling = False
        return config

    @pytest.fixture
    def mock_kv_cache_config(self):
        """Create a mock KVCacheConfig for testing."""
        config = MagicMock(spec=KVCacheConfig)
        config.num_blocks = 100
        return config

    @pytest.fixture
    def mock_structured_output_manager(self):
        """Create a mock StructuredOutputManager."""
        return MagicMock()

    def test_init_creates_worker_processes(
        self,
        mock_vllm_config,
        mock_kv_cache_config,
        mock_structured_output_manager,
    ):
        """Test initialization creates worker processes for each DP rank."""
        with patch(
                'tpu_inference.core.sched.dp_scheduler._scheduler_worker_process'
        ):
            with patch('multiprocessing.get_context') as mock_get_context:
                # Setup mock context
                mock_ctx = MagicMock()
                mock_process = MagicMock()
                mock_queue = MagicMock()

                mock_ctx.Queue = MagicMock(return_value=mock_queue)
                mock_ctx.Process = MagicMock(return_value=mock_process)
                mock_get_context.return_value = mock_ctx

                scheduler = DPScheduler(
                    vllm_config=mock_vllm_config,
                    kv_cache_config=mock_kv_cache_config,
                    structured_output_manager=mock_structured_output_manager,
                    block_size=16,
                    log_stats=True,
                )

                # Verify processes and queues were created
                assert scheduler.dp_size == 2
                assert len(scheduler.processes) == 2
                assert len(scheduler.input_queues) == 2
                # output_queues is a dict with (rank, command) tuple keys
                # 2 ranks × 14 commands (SchedulerCommand enum)
                assert len(scheduler.output_queues) == 28
                assert scheduler.log_stats is True
                assert len(scheduler.per_rank_kv_cache_configs) == 2

                # Verify each rank got the correct config
                for rank_config in scheduler.per_rank_kv_cache_configs:
                    assert rank_config.num_blocks == 50  # 100 / 2

                # Verify processes were started
                assert mock_process.start.call_count == 2

    def test_get_rank_token_counts(self, mock_vllm_config,
                                   mock_kv_cache_config,
                                   mock_structured_output_manager):
        """Test _get_rank_token_counts queries workers and aggregates tokens."""
        with patch(
                'tpu_inference.core.sched.dp_scheduler._scheduler_worker_process'
        ):
            with patch('multiprocessing.get_context'):
                scheduler = DPScheduler(
                    vllm_config=mock_vllm_config,
                    kv_cache_config=mock_kv_cache_config,
                    structured_output_manager=mock_structured_output_manager,
                    block_size=16,
                )

                # Mock the queues - need to mock the .get() method to return the value
                scheduler.input_queues = [MagicMock(), MagicMock()]

                mock_queue_0 = MagicMock()
                mock_queue_0.get.return_value = 30
                mock_queue_1 = MagicMock()
                mock_queue_1.get.return_value = 15

                scheduler.output_queues = {
                    (0, "get_token_count"): mock_queue_0,
                    (1, "get_token_count"): mock_queue_1,
                }

                rank_tokens = scheduler._get_rank_token_counts()

                # Verify correct commands were sent
                scheduler.input_queues[0].put.assert_called_with(
                    (SchedulerCommand.GET_TOKEN_COUNT, None))
                scheduler.input_queues[1].put.assert_called_with(
                    (SchedulerCommand.GET_TOKEN_COUNT, None))

                assert rank_tokens[0] == 30
                assert rank_tokens[1] == 15

    def test_find_best_rank_with_cache_hit(self, mock_vllm_config,
                                           mock_kv_cache_config,
                                           mock_structured_output_manager):
        """Test _find_best_rank_for_request prefers cache hits."""
        with patch(
                'tpu_inference.core.sched.dp_scheduler._scheduler_worker_process'
        ):
            with patch('multiprocessing.get_context'):
                scheduler = DPScheduler(
                    vllm_config=mock_vllm_config,
                    kv_cache_config=mock_kv_cache_config,
                    structured_output_manager=mock_structured_output_manager,
                    block_size=16,
                )

                mock_request = MagicMock(spec=Request)

                # Mock the queues with tuple keys (rank, command)
                scheduler.input_queues = [MagicMock(), MagicMock()]

                # Create proper mocks for queue.get() calls
                mock_queue_get_token_0 = MagicMock()
                mock_queue_get_token_0.get.return_value = 100
                mock_queue_get_token_1 = MagicMock()
                mock_queue_get_token_1.get.return_value = 50
                mock_queue_computed_0 = MagicMock()
                mock_queue_computed_0.get.return_value = ([], 10)
                mock_queue_computed_1 = MagicMock()
                mock_queue_computed_1.get.return_value = ([], 25)

                scheduler.output_queues = {
                    (0, "get_token_count"): mock_queue_get_token_0,
                    (1, "get_token_count"): mock_queue_get_token_1,
                    (0, "get_computed_blocks"): mock_queue_computed_0,
                    (1, "get_computed_blocks"): mock_queue_computed_1,
                }

                rank = scheduler._find_best_rank_for_request(mock_request)

                # Should prefer rank with better cache hit
                assert rank == 1

    def test_find_best_rank_without_cache_hit(self, mock_vllm_config,
                                              mock_kv_cache_config,
                                              mock_structured_output_manager):
        """Test _find_best_rank_for_request uses load balancing without cache hit."""
        with patch(
                'tpu_inference.core.sched.dp_scheduler._scheduler_worker_process'
        ):
            with patch('multiprocessing.get_context'):
                scheduler = DPScheduler(
                    vllm_config=mock_vllm_config,
                    kv_cache_config=mock_kv_cache_config,
                    structured_output_manager=mock_structured_output_manager,
                    block_size=16,
                )

                mock_request = MagicMock(spec=Request)

                # Mock the queues with tuple keys (rank, command)
                scheduler.input_queues = [MagicMock(), MagicMock()]

                # Create proper mocks for queue.get() calls
                mock_queue_get_token_0 = MagicMock()
                mock_queue_get_token_0.get.return_value = 100
                mock_queue_get_token_1 = MagicMock()
                mock_queue_get_token_1.get.return_value = 50
                mock_queue_computed_0 = MagicMock()
                mock_queue_computed_0.get.return_value = ([], 0)
                mock_queue_computed_1 = MagicMock()
                mock_queue_computed_1.get.return_value = ([], 0)

                scheduler.output_queues = {
                    (0, "get_token_count"): mock_queue_get_token_0,
                    (1, "get_token_count"): mock_queue_get_token_1,
                    (0, "get_computed_blocks"): mock_queue_computed_0,
                    (1, "get_computed_blocks"): mock_queue_computed_1,
                }

                rank = scheduler._find_best_rank_for_request(mock_request)

                # Should choose rank with fewer tokens (rank 1)
                assert rank == 1

    def test_add_request_assigns_to_best_rank(self, mock_vllm_config,
                                              mock_kv_cache_config,
                                              mock_structured_output_manager):
        """Test add_request assigns request to best rank."""
        with patch(
                'tpu_inference.core.sched.dp_scheduler._scheduler_worker_process'
        ):
            with patch('multiprocessing.get_context'):
                scheduler = DPScheduler(
                    vllm_config=mock_vllm_config,
                    kv_cache_config=mock_kv_cache_config,
                    structured_output_manager=mock_structured_output_manager,
                    block_size=16,
                )

                mock_request = MagicMock(spec=Request)
                mock_request.request_id = "req1"

                # Mock the queues with tuple keys
                scheduler.input_queues = [MagicMock(), MagicMock()]
                scheduler.output_queues = {
                    (0, "add_request"): MagicMock(),
                    (1, "add_request"): MagicMock(),
                }

                # Mock _find_best_rank_for_request to return rank 1
                scheduler._find_best_rank_for_request = MagicMock(
                    return_value=1)

                scheduler.add_request(mock_request)

                # Verify request was assigned to rank 1
                assert scheduler.assigned_dp_rank["req1"] == 1

                # Verify ADD_REQUEST command was sent to rank 1
                scheduler.input_queues[1].put.assert_called_with(
                    (SchedulerCommand.ADD_REQUEST, mock_request))

                # Verify we waited for completion
                scheduler.output_queues[(
                    1, "add_request")].get.assert_called_once()

    def test_schedule_sends_commands_and_combines_output(
            self, mock_vllm_config, mock_kv_cache_config,
            mock_structured_output_manager):
        """Test schedule sends SCHEDULE command to all workers and combines output."""
        with patch(
                'tpu_inference.core.sched.dp_scheduler._scheduler_worker_process'
        ):
            with patch('multiprocessing.get_context'):
                scheduler = DPScheduler(
                    vllm_config=mock_vllm_config,
                    kv_cache_config=mock_kv_cache_config,
                    structured_output_manager=mock_structured_output_manager,
                    block_size=16,
                )

                # Mock the queues with tuple keys
                scheduler.input_queues = [MagicMock(), MagicMock()]

                # Create mock scheduler outputs
                mock_output_0 = MagicMock(spec=SchedulerOutput)
                mock_output_0.scheduled_new_reqs = []
                mock_output_0.num_scheduled_tokens = {"req1": 10}
                mock_output_0.total_num_scheduled_tokens = 10
                mock_output_0.finished_req_ids = set()
                mock_output_0.scheduled_cached_reqs = CachedRequestData(
                    req_ids=[],
                    resumed_req_ids=[],
                    new_token_ids=[],
                    all_token_ids=[],
                    new_block_ids=[],
                    num_computed_tokens=[],
                    num_output_tokens=[],
                )
                mock_output_0.scheduled_spec_decode_tokens = {}
                mock_output_0.scheduled_encoder_inputs = {}
                mock_output_0.num_common_prefix_blocks = []

                mock_output_1 = MagicMock(spec=SchedulerOutput)
                mock_output_1.scheduled_new_reqs = []
                mock_output_1.num_scheduled_tokens = {"req2": 20}
                mock_output_1.total_num_scheduled_tokens = 20
                mock_output_1.finished_req_ids = set()
                mock_output_1.scheduled_cached_reqs = CachedRequestData(
                    req_ids=[],
                    resumed_req_ids=[],
                    new_token_ids=[],
                    all_token_ids=[],
                    new_block_ids=[],
                    num_computed_tokens=[],
                    num_output_tokens=[],
                )
                mock_output_1.scheduled_spec_decode_tokens = {}
                mock_output_1.scheduled_encoder_inputs = {}
                mock_output_1.num_common_prefix_blocks = []

                # Setup mock queue responses with tuple keys - need to mock .get()
                mock_queue_0 = MagicMock()
                mock_queue_0.get.return_value = mock_output_0
                mock_queue_1 = MagicMock()
                mock_queue_1.get.return_value = mock_output_1

                scheduler.output_queues = {
                    (0, "schedule"): mock_queue_0,
                    (1, "schedule"): mock_queue_1,
                }

                # Setup assigned ranks
                scheduler.assigned_dp_rank = {"req1": 0, "req2": 1}

                output = scheduler.schedule()

                # Verify SCHEDULE commands were sent
                scheduler.input_queues[0].put.assert_called_with(
                    (SchedulerCommand.SCHEDULE, None))
                scheduler.input_queues[1].put.assert_called_with(
                    (SchedulerCommand.SCHEDULE, None))

                # Verify combined output
                assert isinstance(output, DPSchedulerOutput)
                assert output.total_num_scheduled_tokens == 30
                assert "req1" in output.num_scheduled_tokens
                assert "req2" in output.num_scheduled_tokens
                assert output.assigned_dp_rank == {"req1": 0, "req2": 1}

    def test_combine_cached_request_data(self, mock_vllm_config,
                                         mock_kv_cache_config,
                                         mock_structured_output_manager):
        """Test _combine_cached_request_data combines data from all ranks."""
        with patch(
                'tpu_inference.core.sched.dp_scheduler._scheduler_worker_process'
        ):
            with patch('multiprocessing.get_context'):
                scheduler = DPScheduler(
                    vllm_config=mock_vllm_config,
                    kv_cache_config=mock_kv_cache_config,
                    structured_output_manager=mock_structured_output_manager,
                    block_size=16,
                )

                # Create mock rank outputs
                output_0 = MagicMock(spec=SchedulerOutput)
                output_0.scheduled_cached_reqs = CachedRequestData(
                    req_ids=["req1"],
                    resumed_req_ids=["req1"],
                    new_token_ids=[[1, 2, 3]],
                    all_token_ids=[[1, 2, 3, 4, 5]],
                    new_block_ids=[[10, 11]],
                    num_computed_tokens=[5],
                    num_output_tokens=[3],
                )

                output_1 = MagicMock(spec=SchedulerOutput)
                output_1.scheduled_cached_reqs = CachedRequestData(
                    req_ids=["req2"],
                    resumed_req_ids=[],
                    new_token_ids=[[6, 7]],
                    all_token_ids=[[6, 7, 8, 9]],
                    new_block_ids=[[20, 21]],
                    num_computed_tokens=[4],
                    num_output_tokens=[2],
                )

                combined = scheduler._combine_cached_request_data(
                    [output_0, output_1])

                # Verify combined data
                assert combined.req_ids == ["req1", "req2"]
                assert combined.resumed_req_ids == ["req1"]
                assert combined.new_token_ids == [[1, 2, 3], [6, 7]]
                assert combined.num_computed_tokens == [5, 4]
                assert combined.num_output_tokens == [3, 2]

    def test_finish_requests_routes_to_workers(self, mock_vllm_config,
                                               mock_kv_cache_config,
                                               mock_structured_output_manager):
        """Test finish_requests sends FINISH_REQUESTS command to appropriate workers."""
        with patch(
                'tpu_inference.core.sched.dp_scheduler._scheduler_worker_process'
        ):
            with patch('multiprocessing.get_context'):
                scheduler = DPScheduler(
                    vllm_config=mock_vllm_config,
                    kv_cache_config=mock_kv_cache_config,
                    structured_output_manager=mock_structured_output_manager,
                    block_size=16,
                )

                scheduler.input_queues = [MagicMock(), MagicMock()]
                scheduler.output_queues = {
                    (0, "finish_requests"): MagicMock(),
                    (1, "finish_requests"): MagicMock(),
                }

                scheduler.assigned_dp_rank = {"req1": 0, "req2": 1, "req3": 0}

                # Test with list of requests
                scheduler.finish_requests(["req1", "req2"],
                                          finished_status="completed")

                # Verify FINISH_REQUESTS commands were sent to correct ranks
                scheduler.input_queues[0].put.assert_called()
                scheduler.input_queues[1].put.assert_called()

    def test_get_num_unfinished_requests(self, mock_vllm_config,
                                         mock_kv_cache_config,
                                         mock_structured_output_manager):
        """Test get_num_unfinished_requests queries all workers."""
        with patch(
                'tpu_inference.core.sched.dp_scheduler._scheduler_worker_process'
        ):
            with patch('multiprocessing.get_context'):
                scheduler = DPScheduler(
                    vllm_config=mock_vllm_config,
                    kv_cache_config=mock_kv_cache_config,
                    structured_output_manager=mock_structured_output_manager,
                    block_size=16,
                )

                scheduler.input_queues = [MagicMock(), MagicMock()]

                # Create proper mocks for queue.get() calls
                mock_queue_0 = MagicMock()
                mock_queue_0.get.return_value = 5
                mock_queue_1 = MagicMock()
                mock_queue_1.get.return_value = 3

                scheduler.output_queues = {
                    (0, "get_num_unfinished_requests"): mock_queue_0,
                    (1, "get_num_unfinished_requests"): mock_queue_1,
                }

                total = scheduler.get_num_unfinished_requests()

                # Verify commands were sent
                scheduler.input_queues[0].put.assert_called_with(
                    (SchedulerCommand.GET_NUM_UNFINISHED_REQUESTS, None))
                scheduler.input_queues[1].put.assert_called_with(
                    (SchedulerCommand.GET_NUM_UNFINISHED_REQUESTS, None))

                assert total == 8

    def test_has_finished_requests(self, mock_vllm_config,
                                   mock_kv_cache_config,
                                   mock_structured_output_manager):
        """Test has_finished_requests checks all workers."""
        with patch(
                'tpu_inference.core.sched.dp_scheduler._scheduler_worker_process'
        ):
            with patch('multiprocessing.get_context'):
                scheduler = DPScheduler(
                    vllm_config=mock_vllm_config,
                    kv_cache_config=mock_kv_cache_config,
                    structured_output_manager=mock_structured_output_manager,
                    block_size=16,
                )

                scheduler.input_queues = [MagicMock(), MagicMock()]

                # Create proper mocks for queue.get() calls
                mock_queue_0 = MagicMock()
                mock_queue_0.get.return_value = False
                mock_queue_1 = MagicMock()
                mock_queue_1.get.return_value = True

                scheduler.output_queues = {
                    (0, "has_finished_requests"): mock_queue_0,
                    (1, "has_finished_requests"): mock_queue_1,
                }

                result = scheduler.has_finished_requests()

                assert result is True

                # Verify commands were sent
                scheduler.input_queues[0].put.assert_called_with(
                    (SchedulerCommand.HAS_FINISHED_REQUESTS, None))
                scheduler.input_queues[1].put.assert_called_with(
                    (SchedulerCommand.HAS_FINISHED_REQUESTS, None))

    def test_get_request_counts(self, mock_vllm_config, mock_kv_cache_config,
                                mock_structured_output_manager):
        """Test get_request_counts queries all workers."""
        with patch(
                'tpu_inference.core.sched.dp_scheduler._scheduler_worker_process'
        ):
            with patch('multiprocessing.get_context'):
                scheduler = DPScheduler(
                    vllm_config=mock_vllm_config,
                    kv_cache_config=mock_kv_cache_config,
                    structured_output_manager=mock_structured_output_manager,
                    block_size=16,
                )

                scheduler.input_queues = [MagicMock(), MagicMock()]

                # Create proper mocks for queue.get() calls
                mock_queue_0 = MagicMock()
                mock_queue_0.get.return_value = (2, 1)
                mock_queue_1 = MagicMock()
                mock_queue_1.get.return_value = (1, 3)

                scheduler.output_queues = {
                    (0, "get_request_counts"): mock_queue_0,
                    (1, "get_request_counts"): mock_queue_1,
                }

                running, waiting = scheduler.get_request_counts()

                # Verify commands were sent
                scheduler.input_queues[0].put.assert_called_with(
                    (SchedulerCommand.GET_REQUEST_COUNTS, None))
                scheduler.input_queues[1].put.assert_called_with(
                    (SchedulerCommand.GET_REQUEST_COUNTS, None))

                assert running == 3
                assert waiting == 4

    def test_reset_prefix_cache(self, mock_vllm_config, mock_kv_cache_config,
                                mock_structured_output_manager):
        """Test reset_prefix_cache sends command to all workers."""
        with patch(
                'tpu_inference.core.sched.dp_scheduler._scheduler_worker_process'
        ):
            with patch('multiprocessing.get_context'):
                scheduler = DPScheduler(
                    vllm_config=mock_vllm_config,
                    kv_cache_config=mock_kv_cache_config,
                    structured_output_manager=mock_structured_output_manager,
                    block_size=16,
                )

                scheduler.input_queues = [MagicMock(), MagicMock()]

                # Create proper mocks for queue.get() calls
                mock_queue_0 = MagicMock()
                mock_queue_0.get.return_value = True
                mock_queue_1 = MagicMock()
                mock_queue_1.get.return_value = True

                scheduler.output_queues = {
                    (0, "reset_prefix_cache"): mock_queue_0,
                    (1, "reset_prefix_cache"): mock_queue_1,
                }

                result = scheduler.reset_prefix_cache()

                # Verify commands were sent
                scheduler.input_queues[0].put.assert_called_with(
                    (SchedulerCommand.RESET_PREFIX_CACHE, None))
                scheduler.input_queues[1].put.assert_called_with(
                    (SchedulerCommand.RESET_PREFIX_CACHE, None))

                assert result is True

    def test_make_stats_aggregates_from_workers(
            self, mock_vllm_config, mock_kv_cache_config,
            mock_structured_output_manager):
        """Test make_stats aggregates statistics from all workers."""
        with patch(
                'tpu_inference.core.sched.dp_scheduler._scheduler_worker_process'
        ):
            with patch('multiprocessing.get_context'):
                scheduler = DPScheduler(
                    vllm_config=mock_vllm_config,
                    kv_cache_config=mock_kv_cache_config,
                    structured_output_manager=mock_structured_output_manager,
                    block_size=16,
                    log_stats=True,
                )

                scheduler.input_queues = [MagicMock(), MagicMock()]

                # Create mock stats
                stats_0 = SchedulerStats(
                    num_running_reqs=3,
                    num_waiting_reqs=2,
                    kv_cache_usage=0.5,
                    prefix_cache_stats=PrefixCacheStats(reset=False,
                                                        requests=10,
                                                        queries=8,
                                                        hits=5),
                    connector_prefix_cache_stats=PrefixCacheStats(reset=False,
                                                                  requests=5,
                                                                  queries=4,
                                                                  hits=2),
                    spec_decoding_stats=None,
                    kv_connector_stats=None,
                )

                stats_1 = SchedulerStats(
                    num_running_reqs=4,
                    num_waiting_reqs=1,
                    kv_cache_usage=0.7,
                    prefix_cache_stats=PrefixCacheStats(reset=False,
                                                        requests=15,
                                                        queries=12,
                                                        hits=8),
                    connector_prefix_cache_stats=PrefixCacheStats(reset=False,
                                                                  requests=6,
                                                                  queries=5,
                                                                  hits=3),
                    spec_decoding_stats=None,
                    kv_connector_stats=None,
                )

                # Create proper mocks for queue.get() calls
                mock_queue_0 = MagicMock()
                mock_queue_0.get.return_value = stats_0
                mock_queue_1 = MagicMock()
                mock_queue_1.get.return_value = stats_1

                scheduler.output_queues = {
                    (0, "make_stats"): mock_queue_0,
                    (1, "make_stats"): mock_queue_1,
                }

                combined_stats = scheduler.make_stats()

                # Verify commands were sent
                scheduler.input_queues[0].put.assert_called_with(
                    (SchedulerCommand.MAKE_STATS, (None, None)))
                scheduler.input_queues[1].put.assert_called_with(
                    (SchedulerCommand.MAKE_STATS, (None, None)))

                assert combined_stats.num_running_reqs == 7
                assert combined_stats.num_waiting_reqs == 3
                assert combined_stats.kv_cache_usage == 0.6

    def test_make_stats_returns_none_when_disabled(
            self, mock_vllm_config, mock_kv_cache_config,
            mock_structured_output_manager):
        """Test make_stats returns None when logging disabled."""
        with patch(
                'tpu_inference.core.sched.dp_scheduler._scheduler_worker_process'
        ):
            with patch('multiprocessing.get_context'):
                scheduler = DPScheduler(
                    vllm_config=mock_vllm_config,
                    kv_cache_config=mock_kv_cache_config,
                    structured_output_manager=mock_structured_output_manager,
                    block_size=16,
                    log_stats=False,
                )

                stats = scheduler.make_stats()
                assert stats is None

    def test_update_draft_token_ids(self, mock_vllm_config,
                                    mock_kv_cache_config,
                                    mock_structured_output_manager):
        """Test update_draft_token_ids routes to correct workers."""
        with patch(
                'tpu_inference.core.sched.dp_scheduler._scheduler_worker_process'
        ):
            with patch('multiprocessing.get_context'):
                scheduler = DPScheduler(
                    vllm_config=mock_vllm_config,
                    kv_cache_config=mock_kv_cache_config,
                    structured_output_manager=mock_structured_output_manager,
                    block_size=16,
                )

                scheduler.input_queues = [MagicMock(), MagicMock()]
                scheduler.output_queues = {
                    (0, "update_draft_token_ids"): MagicMock(),
                    (1, "update_draft_token_ids"): MagicMock(),
                }

                scheduler.assigned_dp_rank = {"req1": 0, "req2": 1, "req3": 0}

                draft_token_ids = MagicMock()
                draft_token_ids.req_ids = ["req1", "req2", "req3"]
                draft_token_ids.draft_token_ids = [
                    [101, 102, 103],
                    [201, 202],
                    [301, 302, 303, 304],
                ]

                scheduler.update_draft_token_ids(draft_token_ids)

                # Verify commands were sent to correct workers
                scheduler.input_queues[0].put.assert_called()
                scheduler.input_queues[1].put.assert_called()

    def test_shutdown(self, mock_vllm_config, mock_kv_cache_config,
                      mock_structured_output_manager):
        """Test shutdown sends SHUTDOWN command to all workers."""
        with patch(
                'tpu_inference.core.sched.dp_scheduler._scheduler_worker_process'
        ):
            with patch('multiprocessing.get_context'):
                scheduler = DPScheduler(
                    vllm_config=mock_vllm_config,
                    kv_cache_config=mock_kv_cache_config,
                    structured_output_manager=mock_structured_output_manager,
                    block_size=16,
                )

                scheduler.input_queues = [MagicMock(), MagicMock()]
                scheduler.output_queues = {
                    (0, "shutdown"): MagicMock(),
                    (1, "shutdown"): MagicMock(),
                }

                mock_process_0 = MagicMock()
                mock_process_1 = MagicMock()
                mock_process_0.is_alive = MagicMock(return_value=False)
                mock_process_1.is_alive = MagicMock(return_value=False)
                scheduler.processes = [mock_process_0, mock_process_1]

                scheduler.shutdown()

                # Verify SHUTDOWN commands were sent
                scheduler.input_queues[0].put.assert_called_with(
                    (SchedulerCommand.SHUTDOWN, None))
                scheduler.input_queues[1].put.assert_called_with(
                    (SchedulerCommand.SHUTDOWN, None))

                # Verify processes were joined
                mock_process_0.join.assert_called()
                mock_process_1.join.assert_called()


class TestUpdateVllmConfigForDPScheduler:
    """Test the update_vllm_config_for_dp_scheduler function."""

    def test_update_config_with_dp_size_greater_than_one(self):
        """Test Config is updated when DP size > 1."""
        mock_config = MagicMock()
        mock_config.sharding_config.total_dp_size = 2
        mock_config.scheduler_config._original_scheduler_cls = None
        mock_config.scheduler_config.scheduler_cls = "vllm.v1.core.sched.scheduler.Scheduler"
        mock_config.scheduler_config.async_scheduling = False

        update_vllm_config_for_dp_scheduler(mock_config)

        # Verify config was updated
        assert mock_config.scheduler_config._original_scheduler_cls == Scheduler
        assert mock_config.scheduler_config.scheduler_cls == DPScheduler

    def test_update_config_with_dp_size_one(self):
        """Test that config is NOT updated when DP size == 1."""
        mock_config = MagicMock()
        mock_config.sharding_config.total_dp_size = 1
        original_scheduler_cls = "vllm.v1.core.sched.scheduler.Scheduler"
        mock_config.scheduler_config.scheduler_cls = original_scheduler_cls

        update_vllm_config_for_dp_scheduler(mock_config)

        # Verify config was NOT changed
        assert mock_config.scheduler_config.scheduler_cls == original_scheduler_cls


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
