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
from vllm.v1.core.sched.interface import PauseState
from vllm.v1.core.sched.output import CachedRequestData, SchedulerOutput
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.metrics.stats import PrefixCacheStats, SchedulerStats
from vllm.v1.request import Request

from tpu_inference.core.sched.dp_scheduler import (
    DPScheduler, DPSchedulerOutput, SchedulerCommand,
    update_vllm_config_for_dp_scheduler)


def _make_mock_mp_context():
    """Create a mock multiprocessing context that properly handles Pipe()."""
    mock_ctx = MagicMock()
    mock_process = MagicMock()
    mock_ctx.Pipe = MagicMock(return_value=(MagicMock(), MagicMock()))
    mock_ctx.Process = MagicMock(return_value=mock_process)
    return mock_ctx


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
        config.cache_config = MagicMock()
        config.cache_config.enable_prefix_caching = False
        config.cache_config.prefix_caching_hash_algo = "sha256"
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

    def _create_scheduler(self,
                          mock_vllm_config,
                          mock_kv_cache_config,
                          mock_structured_output_manager,
                          log_stats=False):
        """Helper to create a DPScheduler with mocked multiprocessing."""
        mock_ctx = _make_mock_mp_context()
        with patch(
                'tpu_inference.core.sched.dp_scheduler._scheduler_worker_process'
        ):
            with patch('multiprocessing.get_context', return_value=mock_ctx):
                scheduler = DPScheduler(
                    vllm_config=mock_vllm_config,
                    kv_cache_config=mock_kv_cache_config,
                    structured_output_manager=mock_structured_output_manager,
                    block_size=16,
                    log_stats=log_stats,
                )
        return scheduler

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
                mock_ctx = _make_mock_mp_context()
                mock_get_context.return_value = mock_ctx

                scheduler = DPScheduler(
                    vllm_config=mock_vllm_config,
                    kv_cache_config=mock_kv_cache_config,
                    structured_output_manager=mock_structured_output_manager,
                    block_size=16,
                    log_stats=True,
                )

                # Verify processes and connections were created
                assert scheduler.dp_size == 2
                assert len(scheduler.processes) == 2
                # One input conn and one output conn per rank
                assert len(scheduler.input_conns) == 2
                assert len(scheduler.output_conns) == 2
                assert scheduler.log_stats is True
                assert len(scheduler.per_rank_kv_cache_configs) == 2

                # Verify each rank got the correct config
                for rank_config in scheduler.per_rank_kv_cache_configs:
                    assert rank_config.num_blocks == 50  # 100 / 2

                # Verify processes were started
                mock_process = mock_ctx.Process.return_value
                assert mock_process.start.call_count == 2

    def test_init_with_prefix_caching_enabled(
        self,
        mock_vllm_config,
        mock_kv_cache_config,
        mock_structured_output_manager,
    ):
        """Test initialization with prefix caching enabled initializes NONE_HASH."""
        mock_vllm_config.cache_config.enable_prefix_caching = True
        mock_vllm_config.cache_config.prefix_caching_hash_algo = "sha256"

        with patch(
                'tpu_inference.core.sched.dp_scheduler._scheduler_worker_process'
        ):
            with patch('multiprocessing.get_context') as mock_get_context:
                with patch('vllm.v1.core.kv_cache_utils.init_none_hash'
                           ) as mock_init_none_hash:
                    with patch('vllm.utils.hashing.get_hash_fn_by_name'
                               ) as mock_get_hash_fn:
                        mock_ctx = _make_mock_mp_context()
                        mock_get_context.return_value = mock_ctx

                        mock_hash_fn = MagicMock()
                        mock_get_hash_fn.return_value = mock_hash_fn

                        scheduler = DPScheduler(
                            vllm_config=mock_vllm_config,
                            kv_cache_config=mock_kv_cache_config,
                            structured_output_manager=
                            mock_structured_output_manager,
                            block_size=16,
                            log_stats=True,
                        )

                        # Verify init_none_hash was called with correct hash function
                        mock_get_hash_fn.assert_called_once_with("sha256")
                        mock_init_none_hash.assert_called_once_with(
                            mock_hash_fn)

                        assert scheduler.dp_size == 2

    def test_init_without_prefix_caching_skips_initialization(
        self,
        mock_vllm_config,
        mock_kv_cache_config,
        mock_structured_output_manager,
    ):
        """Test initialization without prefix caching skips NONE_HASH initialization."""
        mock_vllm_config.cache_config.enable_prefix_caching = False

        with patch(
                'tpu_inference.core.sched.dp_scheduler._scheduler_worker_process'
        ):
            with patch('multiprocessing.get_context') as mock_get_context:
                with patch('vllm.v1.core.kv_cache_utils.init_none_hash'
                           ) as mock_init_none_hash:
                    mock_ctx = _make_mock_mp_context()
                    mock_get_context.return_value = mock_ctx

                    scheduler = DPScheduler(
                        vllm_config=mock_vllm_config,
                        kv_cache_config=mock_kv_cache_config,
                        structured_output_manager=
                        mock_structured_output_manager,
                        block_size=16,
                        log_stats=True,
                    )

                    # Verify init_none_hash was NOT called
                    mock_init_none_hash.assert_not_called()

                    assert scheduler.dp_size == 2

    def test_get_rank_token_counts(self, mock_vllm_config,
                                   mock_kv_cache_config,
                                   mock_structured_output_manager):
        """Test _get_rank_token_counts queries workers and aggregates tokens."""
        scheduler = self._create_scheduler(mock_vllm_config,
                                           mock_kv_cache_config,
                                           mock_structured_output_manager)

        # Mock _send_command and _get_result (the pipe-based API)
        scheduler._send_command = MagicMock()
        scheduler._get_result = MagicMock(side_effect=[30, 15])

        rank_tokens = scheduler._get_rank_token_counts()

        # Verify correct commands were sent
        scheduler._send_command.assert_any_call(
            0, SchedulerCommand.GET_TOKEN_COUNT)
        scheduler._send_command.assert_any_call(
            1, SchedulerCommand.GET_TOKEN_COUNT)

        # Verify results were collected
        scheduler._get_result.assert_any_call(0,
                                              SchedulerCommand.GET_TOKEN_COUNT)
        scheduler._get_result.assert_any_call(1,
                                              SchedulerCommand.GET_TOKEN_COUNT)

        assert rank_tokens[0] == 30
        assert rank_tokens[1] == 15

    def test_find_best_rank_with_cache_hit(self, mock_vllm_config,
                                           mock_kv_cache_config,
                                           mock_structured_output_manager):
        """Test _find_best_rank_for_request prefers cache hits."""
        scheduler = self._create_scheduler(mock_vllm_config,
                                           mock_kv_cache_config,
                                           mock_structured_output_manager)

        mock_request = MagicMock(spec=Request)

        # Mock _send_command and _get_result
        # _get_rank_token_counts calls _get_result for GET_TOKEN_COUNT (2 ranks)
        # then _find_best_rank calls _get_result for PROBE_COMPUTED_BLOCKS (2 ranks)
        scheduler._send_command = MagicMock()
        scheduler._get_result = MagicMock(side_effect=[
            100,
            50,  # GET_TOKEN_COUNT: rank 0=100, rank 1=50
            10,
            25,  # PROBE_COMPUTED_BLOCKS: rank 0=10, rank 1=25
        ])

        rank = scheduler._find_best_rank_for_request(mock_request)

        # Should prefer rank with better cache hit (rank 1 has 25 cached tokens)
        assert rank == 1

    def test_find_best_rank_without_cache_hit(self, mock_vllm_config,
                                              mock_kv_cache_config,
                                              mock_structured_output_manager):
        """Test _find_best_rank_for_request uses load balancing without cache hit."""
        scheduler = self._create_scheduler(mock_vllm_config,
                                           mock_kv_cache_config,
                                           mock_structured_output_manager)

        mock_request = MagicMock(spec=Request)

        # Mock _send_command and _get_result
        scheduler._send_command = MagicMock()
        scheduler._get_result = MagicMock(side_effect=[
            100,
            50,  # GET_TOKEN_COUNT: rank 0=100, rank 1=50
            0,
            0,  # PROBE_COMPUTED_BLOCKS: no cache hits
        ])

        rank = scheduler._find_best_rank_for_request(mock_request)

        # Should choose rank with fewer tokens (rank 1)
        assert rank == 1

    def test_add_request_assigns_to_best_rank(self, mock_vllm_config,
                                              mock_kv_cache_config,
                                              mock_structured_output_manager):
        """Test add_request assigns request to best rank."""
        scheduler = self._create_scheduler(mock_vllm_config,
                                           mock_kv_cache_config,
                                           mock_structured_output_manager)

        mock_request = MagicMock(spec=Request)
        mock_request.request_id = "req1"

        # Mock _find_best_rank_for_request to return rank 1
        scheduler._find_best_rank_for_request = MagicMock(return_value=1)
        scheduler._send_command = MagicMock()
        scheduler._get_result = MagicMock(return_value=None)

        scheduler.add_request(mock_request)

        # Verify request was assigned to rank 1
        assert scheduler.assigned_dp_rank["req1"] == 1

        # Verify ADD_REQUEST command was sent to rank 1
        scheduler._send_command.assert_called_with(
            1, SchedulerCommand.ADD_REQUEST, mock_request)

        # Verify we waited for completion
        scheduler._get_result.assert_called_with(1,
                                                 SchedulerCommand.ADD_REQUEST)

    def test_schedule_sends_commands_and_combines_output(
            self, mock_vllm_config, mock_kv_cache_config,
            mock_structured_output_manager):
        """Test schedule sends SCHEDULE command to all workers and combines output."""
        scheduler = self._create_scheduler(mock_vllm_config,
                                           mock_kv_cache_config,
                                           mock_structured_output_manager)

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
            all_token_ids={},
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
            all_token_ids={},
            new_block_ids=[],
            num_computed_tokens=[],
            num_output_tokens=[],
        )
        mock_output_1.scheduled_spec_decode_tokens = {}
        mock_output_1.scheduled_encoder_inputs = {}
        mock_output_1.num_common_prefix_blocks = []

        # Mock _send_command and _get_result
        scheduler._send_command = MagicMock()
        scheduler._get_result = MagicMock(
            side_effect=[mock_output_0, mock_output_1])

        # Setup assigned ranks
        scheduler.assigned_dp_rank = {"req1": 0, "req2": 1}

        output = scheduler.schedule()

        # Verify SCHEDULE commands were sent
        scheduler._send_command.assert_any_call(0, SchedulerCommand.SCHEDULE)
        scheduler._send_command.assert_any_call(1, SchedulerCommand.SCHEDULE)

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
        scheduler = self._create_scheduler(mock_vllm_config,
                                           mock_kv_cache_config,
                                           mock_structured_output_manager)

        # Create mock rank outputs
        output_0 = MagicMock(spec=SchedulerOutput)
        output_0.scheduled_cached_reqs = CachedRequestData(
            req_ids=["req1"],
            resumed_req_ids=["req1"],
            new_token_ids=[[1, 2, 3]],
            all_token_ids={"req1": [1, 2, 3, 4, 5]},
            new_block_ids=[[10, 11]],
            num_computed_tokens=[5],
            num_output_tokens=[3],
        )

        output_1 = MagicMock(spec=SchedulerOutput)
        output_1.scheduled_cached_reqs = CachedRequestData(
            req_ids=["req2"],
            resumed_req_ids=[],
            new_token_ids=[[6, 7]],
            all_token_ids={"req2": [6, 7, 8, 9]},
            new_block_ids=[[20, 21]],
            num_computed_tokens=[4],
            num_output_tokens=[2],
        )

        combined = scheduler._combine_cached_request_data([output_0, output_1])

        # Verify combined data
        assert combined.req_ids == ["req1", "req2"]
        assert combined.resumed_req_ids == ["req1"]
        assert combined.new_token_ids == [[1, 2, 3], [6, 7]]
        assert combined.all_token_ids == {
            "req1": [1, 2, 3, 4, 5],
            "req2": [6, 7, 8, 9]
        }
        assert combined.num_computed_tokens == [5, 4]
        assert combined.num_output_tokens == [3, 2]

    def test_finish_requests_routes_to_workers(self, mock_vllm_config,
                                               mock_kv_cache_config,
                                               mock_structured_output_manager):
        """Test finish_requests sends FINISH_REQUESTS command to appropriate workers."""
        scheduler = self._create_scheduler(mock_vllm_config,
                                           mock_kv_cache_config,
                                           mock_structured_output_manager)

        scheduler._send_command = MagicMock()
        scheduler._get_result = MagicMock(return_value=None)

        scheduler.assigned_dp_rank = {"req1": 0, "req2": 1, "req3": 0}

        # Test with list of requests
        scheduler.finish_requests(["req1", "req2"],
                                  finished_status="completed")

        # Verify FINISH_REQUESTS commands were sent to correct ranks
        scheduler._send_command.assert_any_call(
            0, SchedulerCommand.FINISH_REQUESTS, (["req1"], "completed"))
        scheduler._send_command.assert_any_call(
            1, SchedulerCommand.FINISH_REQUESTS, (["req2"], "completed"))

    def test_get_num_unfinished_requests(self, mock_vllm_config,
                                         mock_kv_cache_config,
                                         mock_structured_output_manager):
        """Test get_num_unfinished_requests queries all workers."""
        scheduler = self._create_scheduler(mock_vllm_config,
                                           mock_kv_cache_config,
                                           mock_structured_output_manager)

        scheduler._send_command = MagicMock()
        scheduler._get_result = MagicMock(side_effect=[5, 3])

        total = scheduler.get_num_unfinished_requests()

        # Verify commands were sent
        scheduler._send_command.assert_any_call(
            0, SchedulerCommand.GET_NUM_UNFINISHED_REQUESTS)
        scheduler._send_command.assert_any_call(
            1, SchedulerCommand.GET_NUM_UNFINISHED_REQUESTS)

        assert total == 8

    def test_has_finished_requests(self, mock_vllm_config,
                                   mock_kv_cache_config,
                                   mock_structured_output_manager):
        """Test has_finished_requests checks all workers."""
        scheduler = self._create_scheduler(mock_vllm_config,
                                           mock_kv_cache_config,
                                           mock_structured_output_manager)

        scheduler._send_command = MagicMock()
        scheduler._get_result = MagicMock(side_effect=[False, True])

        result = scheduler.has_finished_requests()

        assert result is True

        # Verify commands were sent
        scheduler._send_command.assert_any_call(
            0, SchedulerCommand.HAS_FINISHED_REQUESTS)
        scheduler._send_command.assert_any_call(
            1, SchedulerCommand.HAS_FINISHED_REQUESTS)

    def test_get_request_counts(self, mock_vllm_config, mock_kv_cache_config,
                                mock_structured_output_manager):
        """Test get_request_counts queries all workers."""
        scheduler = self._create_scheduler(mock_vllm_config,
                                           mock_kv_cache_config,
                                           mock_structured_output_manager)

        scheduler._send_command = MagicMock()
        scheduler._get_result = MagicMock(side_effect=[(2, 1), (1, 3)])

        running, waiting = scheduler.get_request_counts()

        # Verify commands were sent
        scheduler._send_command.assert_any_call(
            0, SchedulerCommand.GET_REQUEST_COUNTS)
        scheduler._send_command.assert_any_call(
            1, SchedulerCommand.GET_REQUEST_COUNTS)

        assert running == 3
        assert waiting == 4

    def test_reset_prefix_cache(self, mock_vllm_config, mock_kv_cache_config,
                                mock_structured_output_manager):
        """Test reset_prefix_cache sends command to all workers."""
        scheduler = self._create_scheduler(mock_vllm_config,
                                           mock_kv_cache_config,
                                           mock_structured_output_manager)

        scheduler._send_command = MagicMock()
        scheduler._get_result = MagicMock(side_effect=[True, True])

        result = scheduler.reset_prefix_cache()

        # Verify commands were sent with default args
        scheduler._send_command.assert_any_call(
            0, SchedulerCommand.RESET_PREFIX_CACHE, (False, False))
        scheduler._send_command.assert_any_call(
            1, SchedulerCommand.RESET_PREFIX_CACHE, (False, False))

        assert result is True

    def test_reset_prefix_cache_with_args(self, mock_vllm_config,
                                          mock_kv_cache_config,
                                          mock_structured_output_manager):
        """Test reset_prefix_cache forwards reset_running_requests and reset_connector args."""
        scheduler = self._create_scheduler(mock_vllm_config,
                                           mock_kv_cache_config,
                                           mock_structured_output_manager)

        scheduler._send_command = MagicMock()
        scheduler._get_result = MagicMock(side_effect=[True, False])

        result = scheduler.reset_prefix_cache(reset_running_requests=True,
                                              reset_connector=True)

        # Verify commands were sent with provided args
        scheduler._send_command.assert_any_call(
            0, SchedulerCommand.RESET_PREFIX_CACHE, (True, True))
        scheduler._send_command.assert_any_call(
            1, SchedulerCommand.RESET_PREFIX_CACHE, (True, True))

        # One rank returned False, so overall result should be False
        assert result is False

    def test_reset_encoder_cache(self, mock_vllm_config, mock_kv_cache_config,
                                 mock_structured_output_manager):
        """Test reset_encoder_cache sends command to all workers."""
        scheduler = self._create_scheduler(mock_vllm_config,
                                           mock_kv_cache_config,
                                           mock_structured_output_manager)

        scheduler._send_command = MagicMock()
        scheduler._get_result = MagicMock(return_value=None)

        scheduler.reset_encoder_cache()

        # Verify commands were sent
        scheduler._send_command.assert_any_call(
            0, SchedulerCommand.RESET_ENCODER_CACHE)
        scheduler._send_command.assert_any_call(
            1, SchedulerCommand.RESET_ENCODER_CACHE)

    def test_pause_state_default(self, mock_vllm_config, mock_kv_cache_config,
                                 mock_structured_output_manager):
        """Test pause_state queries worker and defaults to UNPAUSED."""
        scheduler = self._create_scheduler(mock_vllm_config,
                                           mock_kv_cache_config,
                                           mock_structured_output_manager)

        scheduler._send_command = MagicMock()
        scheduler._get_result = MagicMock(return_value=PauseState.UNPAUSED)

        assert scheduler.pause_state == PauseState.UNPAUSED

        # Verify GET_PAUSE_STATE was sent only to rank 0
        scheduler._send_command.assert_called_once_with(
            0, SchedulerCommand.GET_PAUSE_STATE)

    def test_set_pause_state(self, mock_vllm_config, mock_kv_cache_config,
                             mock_structured_output_manager):
        """Test set_pause_state sends command to all workers."""
        scheduler = self._create_scheduler(mock_vllm_config,
                                           mock_kv_cache_config,
                                           mock_structured_output_manager)

        scheduler._send_command = MagicMock()
        scheduler._get_result = MagicMock(return_value=None)

        scheduler.set_pause_state(PauseState.PAUSED_NEW)

        # Verify SET_PAUSE_STATE was sent to all ranks
        scheduler._send_command.assert_any_call(
            0, SchedulerCommand.SET_PAUSE_STATE, PauseState.PAUSED_NEW)
        scheduler._send_command.assert_any_call(
            1, SchedulerCommand.SET_PAUSE_STATE, PauseState.PAUSED_NEW)

    def test_make_stats_aggregates_from_workers(
            self, mock_vllm_config, mock_kv_cache_config,
            mock_structured_output_manager):
        """Test make_stats aggregates statistics from all workers."""
        scheduler = self._create_scheduler(mock_vllm_config,
                                           mock_kv_cache_config,
                                           mock_structured_output_manager,
                                           log_stats=True)

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

        scheduler._send_command = MagicMock()
        scheduler._get_result = MagicMock(side_effect=[stats_0, stats_1])

        combined_stats = scheduler.make_stats()

        # Verify commands were sent
        scheduler._send_command.assert_any_call(0, SchedulerCommand.MAKE_STATS,
                                                (None, None))
        scheduler._send_command.assert_any_call(1, SchedulerCommand.MAKE_STATS,
                                                (None, None))

        assert combined_stats.num_running_reqs == 7
        assert combined_stats.num_waiting_reqs == 3
        assert combined_stats.kv_cache_usage == 0.6

    def test_make_stats_returns_none_when_disabled(
            self, mock_vllm_config, mock_kv_cache_config,
            mock_structured_output_manager):
        """Test make_stats returns None when logging disabled."""
        scheduler = self._create_scheduler(mock_vllm_config,
                                           mock_kv_cache_config,
                                           mock_structured_output_manager,
                                           log_stats=False)

        stats = scheduler.make_stats()
        assert stats is None

    def test_update_draft_token_ids(self, mock_vllm_config,
                                    mock_kv_cache_config,
                                    mock_structured_output_manager):
        """Test update_draft_token_ids routes to correct workers."""
        scheduler = self._create_scheduler(mock_vllm_config,
                                           mock_kv_cache_config,
                                           mock_structured_output_manager)

        scheduler._send_command = MagicMock()
        scheduler._get_result = MagicMock(return_value=None)

        scheduler.assigned_dp_rank = {"req1": 0, "req2": 1, "req3": 0}

        draft_token_ids = MagicMock()
        draft_token_ids.req_ids = ["req1", "req2", "req3"]
        draft_token_ids.draft_token_ids = [
            [101, 102, 103],
            [201, 202],
            [301, 302, 303, 304],
        ]

        scheduler.update_draft_token_ids(draft_token_ids)

        # Verify commands were sent to both workers (rank 0 and rank 1)
        assert scheduler._send_command.call_count == 2
        # Verify the command type for each call
        for c in scheduler._send_command.call_args_list:
            assert c[0][1] == SchedulerCommand.UPDATE_DRAFT_TOKEN_IDS

    def test_combine_scheduler_outputs_max_tokens(
            self, mock_vllm_config, mock_kv_cache_config,
            mock_structured_output_manager):
        """Test _combine_scheduler_outputs calculates max_num_scheduled_tokens_per_dp_rank."""
        scheduler = self._create_scheduler(mock_vllm_config,
                                           mock_kv_cache_config,
                                           mock_structured_output_manager)

        # Mock SchedulerOutput for two DP ranks
        # Rank 0 has 10 tokens, Rank 1 has 20 tokens
        output_0 = MagicMock(spec=SchedulerOutput)
        output_0.num_scheduled_tokens = {"req1": 10}
        output_0.total_num_scheduled_tokens = 10
        output_0.scheduled_new_reqs = []
        output_0.scheduled_spec_decode_tokens = {}
        output_0.scheduled_encoder_inputs = {}
        output_0.finished_req_ids = set()
        output_0.scheduled_cached_reqs = MagicMock(spec=CachedRequestData)
        output_0.scheduled_cached_reqs.req_ids = ["req1"]
        output_0.scheduled_cached_reqs.resumed_req_ids = set()
        output_0.scheduled_cached_reqs.new_token_ids = [[1, 2]]
        output_0.scheduled_cached_reqs.all_token_ids = {}
        output_0.scheduled_cached_reqs.new_block_ids = [None]
        output_0.scheduled_cached_reqs.num_computed_tokens = [10]
        output_0.scheduled_cached_reqs.num_output_tokens = [1]
        output_0.num_common_prefix_blocks = []

        output_1 = MagicMock(spec=SchedulerOutput)
        output_1.num_scheduled_tokens = {"req2": 20}
        output_1.total_num_scheduled_tokens = 20
        output_1.scheduled_new_reqs = []
        output_1.scheduled_spec_decode_tokens = {}
        output_1.scheduled_encoder_inputs = {}
        output_1.finished_req_ids = set()
        output_1.scheduled_cached_reqs = MagicMock(spec=CachedRequestData)
        output_1.scheduled_cached_reqs.req_ids = ["req2"]
        output_1.scheduled_cached_reqs.resumed_req_ids = set()
        output_1.scheduled_cached_reqs.new_token_ids = [[3, 4]]
        output_1.scheduled_cached_reqs.all_token_ids = {}
        output_1.scheduled_cached_reqs.new_block_ids = [None]
        output_1.scheduled_cached_reqs.num_computed_tokens = [20]
        output_1.scheduled_cached_reqs.num_output_tokens = [2]
        output_1.num_common_prefix_blocks = []

        scheduler.assigned_dp_rank = {"req1": 0, "req2": 1}

        combined = scheduler._combine_scheduler_outputs([output_0, output_1])

        assert combined.total_num_scheduled_tokens == 30

    def test_shutdown(self, mock_vllm_config, mock_kv_cache_config,
                      mock_structured_output_manager):
        """Test shutdown sends SHUTDOWN command to all workers."""
        scheduler = self._create_scheduler(mock_vllm_config,
                                           mock_kv_cache_config,
                                           mock_structured_output_manager)

        scheduler._send_command = MagicMock()
        scheduler._get_result = MagicMock(return_value=None)

        mock_process_0 = MagicMock()
        mock_process_1 = MagicMock()
        mock_process_0.is_alive = MagicMock(return_value=False)
        mock_process_1.is_alive = MagicMock(return_value=False)
        scheduler.processes = [mock_process_0, mock_process_1]

        scheduler.shutdown()

        # Verify SHUTDOWN commands were sent
        scheduler._send_command.assert_any_call(0, SchedulerCommand.SHUTDOWN)
        scheduler._send_command.assert_any_call(1, SchedulerCommand.SHUTDOWN)

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
