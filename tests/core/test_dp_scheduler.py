from unittest.mock import MagicMock, patch

import pytest
import torch
from vllm.config import VllmConfig
from vllm.v1.core.sched.output import (CachedRequestData, GrammarOutput,
                                       SchedulerOutput)
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.engine import EngineCoreOutputs
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.metrics.stats import PrefixCacheStats, SchedulerStats
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import Request

from tpu_inference.core.sched.dp_scheduler import (
    DPScheduler, DPSchedulerOutput, update_vllm_config_for_dp_scheduler)


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

    def _create_dp_scheduler_with_mocks(self, mock_vllm_config,
                                        mock_kv_cache_config,
                                        mock_structured_output_manager,
                                        **kwargs):
        """Helper to create a DPScheduler with properly mocked schedulers."""
        # Create individual mock scheduler instances
        mock_scheduler_0 = MagicMock()
        mock_scheduler_1 = MagicMock()

        # Patch the Scheduler class to return our mock instances
        with patch.object(
                mock_vllm_config.scheduler_config, '_original_scheduler_cls',
                MagicMock(side_effect=[mock_scheduler_0, mock_scheduler_1])):
            scheduler = DPScheduler(
                vllm_config=mock_vllm_config,
                kv_cache_config=mock_kv_cache_config,
                structured_output_manager=mock_structured_output_manager,
                block_size=16,
                **kwargs)

            return scheduler

    def test_init_creates_per_rank_schedulers(
        self,
        mock_vllm_config,
        mock_kv_cache_config,
        mock_structured_output_manager,
    ):
        """Test Initialization creates schedulers for each DP rank."""
        # Mock the scheduler class
        mock_scheduler_instance = MagicMock()
        mock_scheduler_cls = MagicMock(return_value=mock_scheduler_instance)

        with patch.object(mock_vllm_config.scheduler_config,
                          '_original_scheduler_cls', mock_scheduler_cls):
            scheduler = DPScheduler(
                vllm_config=mock_vllm_config,
                kv_cache_config=mock_kv_cache_config,
                structured_output_manager=mock_structured_output_manager,
                block_size=16,
                log_stats=True,
            )

            # Verify schedulers were created
            assert len(scheduler.schedulers) == 2
            assert scheduler.dp_size == 2
            assert scheduler.log_stats is True
            assert len(scheduler.per_rank_kv_cache_configs) == 2

            # Verify each rank got the correct config
            for rank_config in scheduler.per_rank_kv_cache_configs:
                assert rank_config.num_blocks == 50  # 100 / 2

    def test_get_rank_token_counts(self, mock_vllm_config,
                                   mock_kv_cache_config,
                                   mock_structured_output_manager):
        """Test _get_rank_token_counts calculates tokens per rank."""
        scheduler = self._create_dp_scheduler_with_mocks(
            mock_vllm_config, mock_kv_cache_config,
            mock_structured_output_manager)

        # Mock requests on different ranks
        req1 = MagicMock()
        req1.num_tokens = 10
        req2 = MagicMock()
        req2.num_tokens = 20
        req3 = MagicMock()
        req3.num_tokens = 15

        scheduler.schedulers[0].running = [req1]
        scheduler.schedulers[0].waiting = [req2]
        scheduler.schedulers[1].running = [req3]
        scheduler.schedulers[1].waiting = []

        rank_tokens = scheduler._get_rank_token_counts()

        assert rank_tokens[0] == 30  # 10 + 20
        assert rank_tokens[1] == 15

    def test_find_best_rank_with_cache_hit(self, mock_vllm_config,
                                           mock_kv_cache_config,
                                           mock_structured_output_manager):
        """Test _find_best_rank_for_request with cache hit."""
        scheduler = self._create_dp_scheduler_with_mocks(
            mock_vllm_config, mock_kv_cache_config,
            mock_structured_output_manager)

        # Mock request
        mock_request = MagicMock(spec=Request)

        # Mock KV cache managers with different cache hits
        scheduler.schedulers[0].kv_cache_manager = MagicMock()
        scheduler.schedulers[
            0].kv_cache_manager.get_computed_blocks.return_value = (
                [],
                10,
            )  # 10 cached tokens

        scheduler.schedulers[1].kv_cache_manager = MagicMock()
        scheduler.schedulers[
            1].kv_cache_manager.get_computed_blocks.return_value = (
                [],
                20,
            )  # 20 cached tokens (better)

        # Mock empty running/waiting queues
        scheduler.schedulers[0].running = []
        scheduler.schedulers[0].waiting = []
        scheduler.schedulers[1].running = []
        scheduler.schedulers[1].waiting = []

        rank = scheduler._find_best_rank_for_request(mock_request)

        # Should choose rank 1 with better cache hit
        assert rank == 1

    def test_find_best_rank_without_cache_hit(self, mock_vllm_config,
                                              mock_kv_cache_config,
                                              mock_structured_output_manager):
        """Test _find_best_rank_for_request without cache hit (load balancing)."""
        scheduler = self._create_dp_scheduler_with_mocks(
            mock_vllm_config, mock_kv_cache_config,
            mock_structured_output_manager)

        # Mock request
        mock_request = MagicMock(spec=Request)

        # Mock KV cache managers with no cache hits
        scheduler.schedulers[0].kv_cache_manager = MagicMock()
        scheduler.schedulers[
            0].kv_cache_manager.get_computed_blocks.return_value = ([], 0)

        scheduler.schedulers[1].kv_cache_manager = MagicMock()
        scheduler.schedulers[
            1].kv_cache_manager.get_computed_blocks.return_value = ([], 0)

        # Mock requests with different token counts
        req1 = MagicMock()
        req1.num_tokens = 50
        req2 = MagicMock()
        req2.num_tokens = 30

        scheduler.schedulers[0].running = [req1]
        scheduler.schedulers[0].waiting = []
        scheduler.schedulers[1].running = [req2]
        scheduler.schedulers[1].waiting = []

        rank = scheduler._find_best_rank_for_request(mock_request)

        # Should choose rank 1 with fewer tokens
        assert rank == 1

    def test_add_request_assigns_to_best_rank(self, mock_vllm_config,
                                              mock_kv_cache_config,
                                              mock_structured_output_manager):
        """Test add_request assigns and adds request to best rank."""
        scheduler = self._create_dp_scheduler_with_mocks(
            mock_vllm_config, mock_kv_cache_config,
            mock_structured_output_manager)

        # Mock the rank selection
        mock_request = MagicMock(spec=Request)
        mock_request.request_id = "req1"

        # Mock _find_best_rank_for_request to return rank 1
        scheduler._find_best_rank_for_request = MagicMock(return_value=1)

        # Mock schedulers
        scheduler.schedulers[0].add_request = MagicMock()
        scheduler.schedulers[1].add_request = MagicMock()

        scheduler.add_request(mock_request)

        # Verify request was assigned to rank 1
        assert scheduler.assigned_dp_rank["req1"] == 1
        scheduler.schedulers[1].add_request.assert_called_once_with(
            mock_request)
        scheduler.schedulers[0].add_request.assert_not_called()

    def test_schedule_runs_all_schedulers(self, mock_vllm_config,
                                          mock_kv_cache_config,
                                          mock_structured_output_manager):
        """Test schedule runs all schedulers and combines output."""
        scheduler = self._create_dp_scheduler_with_mocks(
            mock_vllm_config, mock_kv_cache_config,
            mock_structured_output_manager)

        # Mock scheduler outputs
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

        scheduler.schedulers[0].schedule = MagicMock(
            return_value=mock_output_0)
        scheduler.schedulers[1].schedule = MagicMock(
            return_value=mock_output_1)
        scheduler.schedulers[0].running = []
        scheduler.schedulers[0].waiting = []
        scheduler.schedulers[1].running = []
        scheduler.schedulers[1].waiting = []

        # Assign ranks for requests
        scheduler.assigned_dp_rank = {"req1": 0, "req2": 1}

        output = scheduler.schedule()

        # Verify combined output
        assert isinstance(output, DPSchedulerOutput)
        assert output.total_num_scheduled_tokens == 30  # 10 + 20
        assert "req1" in output.num_scheduled_tokens
        assert "req2" in output.num_scheduled_tokens
        assert output.assigned_dp_rank == {"req1": 0, "req2": 1}

    def test_combine_cached_request_data(self, mock_vllm_config,
                                         mock_kv_cache_config,
                                         mock_structured_output_manager):
        """Test _combine_cached_request_data combines data from all ranks."""
        mock_scheduler_cls = MagicMock(return_value=MagicMock())
        with patch.object(mock_vllm_config.scheduler_config,
                          '_original_scheduler_cls', mock_scheduler_cls):
            scheduler = DPScheduler(
                vllm_config=mock_vllm_config,
                kv_cache_config=mock_kv_cache_config,
                structured_output_manager=mock_structured_output_manager,
                block_size=16,
            )

            # Create mock rank outputs with different cached request data
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

            rank_outputs = [output_0, output_1]
            combined = scheduler._combine_cached_request_data(rank_outputs)

            # Verify combined data
            assert combined.req_ids == ["req1", "req2"]
            assert combined.resumed_req_ids == ["req1"]
            assert combined.new_token_ids == [[1, 2, 3], [6, 7]]
            assert combined.all_token_ids == [[1, 2, 3, 4, 5], [6, 7, 8, 9]]
            assert combined.new_block_ids == [[10, 11], [20, 21]]
            assert combined.num_computed_tokens == [5, 4]
            assert combined.num_output_tokens == [3, 2]

    def test_get_grammar_bitmask_with_structured_output(
            self, mock_vllm_config, mock_kv_cache_config,
            mock_structured_output_manager):
        """Test get_grammar_bitmask combines bitmasks from all ranks."""
        scheduler = self._create_dp_scheduler_with_mocks(
            mock_vllm_config, mock_kv_cache_config,
            mock_structured_output_manager)

        # Create mock scheduler outputs
        mock_output_0 = MagicMock()
        mock_output_1 = MagicMock()

        # Mock grammar outputs from each rank
        grammar_output_0 = GrammarOutput(
            structured_output_request_ids=["req1"],
            grammar_bitmask=torch.ones((1, 100), dtype=torch.bool),
        )
        grammar_output_1 = GrammarOutput(
            structured_output_request_ids=["req2"],
            grammar_bitmask=torch.ones((1, 100), dtype=torch.bool) * 0,
        )

        scheduler.schedulers[0].get_grammar_bitmask = MagicMock(
            return_value=grammar_output_0)
        scheduler.schedulers[1].get_grammar_bitmask = MagicMock(
            return_value=grammar_output_1)

        # Cache scheduler outputs
        scheduler.cached_schedulers_output.append(
            [mock_output_0, mock_output_1])

        # Create a DPSchedulerOutput
        dp_output = DPSchedulerOutput(
            scheduled_new_reqs=[],
            scheduled_cached_reqs=CachedRequestData(
                req_ids=[],
                resumed_req_ids=[],
                new_token_ids=[],
                all_token_ids=[],
                new_block_ids=[],
                num_computed_tokens=[],
                num_output_tokens=[],
            ),
            num_scheduled_tokens={},
            total_num_scheduled_tokens=0,
            scheduled_spec_decode_tokens={},
            scheduled_encoder_inputs={},
            num_common_prefix_blocks=[],
            finished_req_ids=set(),
            free_encoder_mm_hashes=set(),
        )

        result = scheduler.get_grammar_bitmask(dp_output)

        assert result is not None
        assert result.structured_output_request_ids == ["req1", "req2"]
        assert result.grammar_bitmask.shape == (2, 100)

    def test_get_grammar_bitmask_no_structured_output(
            self, mock_vllm_config, mock_kv_cache_config,
            mock_structured_output_manager):
        """Test get_grammar_bitmask returns None when no structured output."""
        mock_scheduler_cls = MagicMock(return_value=MagicMock())
        with patch.object(mock_vllm_config.scheduler_config,
                          '_original_scheduler_cls', mock_scheduler_cls):
            scheduler = DPScheduler(
                vllm_config=mock_vllm_config,
                kv_cache_config=mock_kv_cache_config,
                structured_output_manager=mock_structured_output_manager,
                block_size=16,
            )

            # Mock schedulers returning None
            scheduler.schedulers[0].get_grammar_bitmask = MagicMock(
                return_value=None)
            scheduler.schedulers[1].get_grammar_bitmask = MagicMock(
                return_value=None)

            # Cache scheduler outputs
            mock_output_0 = MagicMock()
            mock_output_1 = MagicMock()
            scheduler.cached_schedulers_output.append(
                [mock_output_0, mock_output_1])

            dp_output = DPSchedulerOutput(
                scheduled_new_reqs=[],
                scheduled_cached_reqs=CachedRequestData(
                    req_ids=[],
                    resumed_req_ids=[],
                    new_token_ids=[],
                    all_token_ids=[],
                    new_block_ids=[],
                    num_computed_tokens=[],
                    num_output_tokens=[],
                ),
                num_scheduled_tokens={},
                total_num_scheduled_tokens=0,
                scheduled_spec_decode_tokens={},
                scheduled_encoder_inputs={},
                num_common_prefix_blocks=[],
                finished_req_ids=set(),
                free_encoder_mm_hashes=set(),
            )

            result = scheduler.get_grammar_bitmask(dp_output)
            assert result is None

    def test_update_from_output_routes_to_schedulers(
            self, mock_vllm_config, mock_kv_cache_config,
            mock_structured_output_manager):
        """Test update_from_output splits output and updates each scheduler."""
        mock_scheduler_cls = MagicMock(return_value=MagicMock())
        with patch.object(mock_vllm_config.scheduler_config,
                          '_original_scheduler_cls', mock_scheduler_cls):
            scheduler = DPScheduler(
                vllm_config=mock_vllm_config,
                kv_cache_config=mock_kv_cache_config,
                structured_output_manager=mock_structured_output_manager,
                block_size=16,
            )

            # Setup assigned ranks
            scheduler.assigned_dp_rank = {"req1": 0, "req2": 1, "req3": 0}

            # Create DPSchedulerOutput
            dp_output = DPSchedulerOutput(
                scheduled_new_reqs=[],
                scheduled_cached_reqs=CachedRequestData(
                    req_ids=[],
                    resumed_req_ids=[],
                    new_token_ids=[],
                    all_token_ids=[],
                    new_block_ids=[],
                    num_computed_tokens=[],
                    num_output_tokens=[],
                ),
                num_scheduled_tokens={
                    "req1": 10,
                    "req2": 20,
                    "req3": 15
                },
                total_num_scheduled_tokens=45,
                scheduled_spec_decode_tokens={},
                scheduled_encoder_inputs={},
                num_common_prefix_blocks=[],
                finished_req_ids={"req3"},  # req3 finished
                free_encoder_mm_hashes=set(),
                assigned_dp_rank={
                    "req1": 0,
                    "req2": 1,
                    "req3": 0
                },
            )

            # Create mock model runner output
            model_output = ModelRunnerOutput(
                req_ids=["req1", "req2", "req3"],
                req_id_to_index={
                    "req1": 0,
                    "req2": 1,
                    "req3": 2
                },
                sampled_token_ids=torch.tensor([100, 200, 300]),
                logprobs=None,
                prompt_logprobs_dict={},
                pooler_output=None,
                num_nans_in_logits=0,
                kv_connector_output=None,
            )

            # Mock rank scheduler outputs (cached from schedule call)
            rank_output_0 = MagicMock()
            rank_output_1 = MagicMock()
            scheduler.cached_schedulers_output.append(
                [rank_output_0, rank_output_1])

            # Mock scheduler update_from_output
            engine_output_0 = EngineCoreOutputs()
            engine_output_0.engine_index = 0
            engine_output_0.outputs = []
            engine_output_0.finished_requests = {"req3"}

            engine_output_1 = EngineCoreOutputs()
            engine_output_1.engine_index = 0
            engine_output_1.outputs = []
            engine_output_1.finished_requests = set()

            scheduler.schedulers[0].update_from_output = MagicMock(
                return_value={0: engine_output_0})
            scheduler.schedulers[1].update_from_output = MagicMock(
                return_value={0: engine_output_1})

            # Mock make_stats
            scheduler.make_stats = MagicMock(return_value=None)

            _ = scheduler.update_from_output(dp_output, model_output)

            # Verify schedulers were updated
            assert scheduler.schedulers[0].update_from_output.called
            assert scheduler.schedulers[1].update_from_output.called

            # Verify finished request was cleaned up
            assert "req3" not in scheduler.assigned_dp_rank
            assert "req1" in scheduler.assigned_dp_rank
            assert "req2" in scheduler.assigned_dp_rank

    def test_split_model_output_by_rank(self, mock_vllm_config,
                                        mock_kv_cache_config,
                                        mock_structured_output_manager):
        """Test _split_model_output_by_rank distributes output correctly."""
        mock_scheduler_cls = MagicMock(return_value=MagicMock())
        with patch.object(mock_vllm_config.scheduler_config,
                          '_original_scheduler_cls', mock_scheduler_cls):
            scheduler = DPScheduler(
                vllm_config=mock_vllm_config,
                kv_cache_config=mock_kv_cache_config,
                structured_output_manager=mock_structured_output_manager,
                block_size=16,
            )

            # Setup assigned ranks
            scheduler.assigned_dp_rank = {
                "req1": 0,
                "req2": 1,
                "req3": 0,
                "req4": 1
            }

            # Create global model output
            global_output = ModelRunnerOutput(
                req_ids=["req1", "req2", "req3", "req4"],
                req_id_to_index={
                    "req1": 0,
                    "req2": 1,
                    "req3": 2,
                    "req4": 3
                },
                sampled_token_ids=torch.tensor([100, 200, 300, 400]),
                logprobs=None,
                prompt_logprobs_dict={},
                pooler_output=None,
                num_nans_in_logits=0,
                kv_connector_output=None,
            )

            rank_outputs = scheduler._split_model_output_by_rank(global_output)

            # Verify split outputs
            assert len(rank_outputs) == 2
            assert rank_outputs[0].req_ids == ["req1", "req3"]
            assert rank_outputs[1].req_ids == ["req2", "req4"]

    def test_cleanup_finished_requests(self, mock_vllm_config,
                                       mock_kv_cache_config,
                                       mock_structured_output_manager):
        """Test _cleanup_finished_requests removes finished requests."""
        mock_scheduler_cls = MagicMock(return_value=MagicMock())
        with patch.object(mock_vllm_config.scheduler_config,
                          '_original_scheduler_cls', mock_scheduler_cls):
            scheduler = DPScheduler(
                vllm_config=mock_vllm_config,
                kv_cache_config=mock_kv_cache_config,
                structured_output_manager=mock_structured_output_manager,
                block_size=16,
            )

            # Setup assigned ranks
            scheduler.assigned_dp_rank = {"req1": 0, "req2": 1, "req3": 0}

            # Clean up finished requests
            scheduler._cleanup_finished_requests({"req1", "req3"})

            # Verify cleanup
            assert "req1" not in scheduler.assigned_dp_rank
            assert "req3" not in scheduler.assigned_dp_rank
            assert "req2" in scheduler.assigned_dp_rank

    def test_finish_requests_single_and_multiple(
            self, mock_vllm_config, mock_kv_cache_config,
            mock_structured_output_manager):
        """Test finish_requests handles single string and list."""
        scheduler = self._create_dp_scheduler_with_mocks(
            mock_vllm_config, mock_kv_cache_config,
            mock_structured_output_manager)

        # Setup assigned ranks
        scheduler.assigned_dp_rank = {"req1": 0, "req2": 1, "req3": 0}

        # Mock scheduler finish_requests
        scheduler.schedulers[0].finish_requests = MagicMock()
        scheduler.schedulers[1].finish_requests = MagicMock()

        # Test with single string
        scheduler.finish_requests("req1", finished_status="completed")
        scheduler.schedulers[0].finish_requests.assert_called_with(["req1"],
                                                                   "completed")

        # Test with list
        scheduler.schedulers[0].finish_requests.reset_mock()
        scheduler.schedulers[1].finish_requests.reset_mock()

        scheduler.finish_requests(["req1", "req2"],
                                  finished_status="completed")
        scheduler.schedulers[0].finish_requests.assert_called_once_with(
            ["req1"], "completed")
        scheduler.schedulers[1].finish_requests.assert_called_once_with(
            ["req2"], "completed")

    def test_get_num_unfinished_requests(self, mock_vllm_config,
                                         mock_kv_cache_config,
                                         mock_structured_output_manager):
        """Test get_num_unfinished_requests aggregates across ranks."""
        scheduler = self._create_dp_scheduler_with_mocks(
            mock_vllm_config, mock_kv_cache_config,
            mock_structured_output_manager)

        scheduler.schedulers[0].get_num_unfinished_requests = MagicMock(
            return_value=5)
        scheduler.schedulers[1].get_num_unfinished_requests = MagicMock(
            return_value=3)

        total = scheduler.get_num_unfinished_requests()
        assert total == 8

    def test_has_finished_requests(self, mock_vllm_config,
                                   mock_kv_cache_config,
                                   mock_structured_output_manager):
        """Test has_finished_requests checks all ranks."""
        mock_scheduler_cls = MagicMock(return_value=MagicMock())
        with patch.object(mock_vllm_config.scheduler_config,
                          '_original_scheduler_cls', mock_scheduler_cls):
            scheduler = DPScheduler(
                vllm_config=mock_vllm_config,
                kv_cache_config=mock_kv_cache_config,
                structured_output_manager=mock_structured_output_manager,
                block_size=16,
            )

            # Test when one rank has finished requests
            scheduler.schedulers[0].has_finished_requests = MagicMock(
                return_value=False)
            scheduler.schedulers[1].has_finished_requests = MagicMock(
                return_value=True)

            assert scheduler.has_finished_requests() is True

            # Test when no rank has finished requests
            scheduler.schedulers[1].has_finished_requests = MagicMock(
                return_value=False)
            assert scheduler.has_finished_requests() is False

    def test_get_request_counts(self, mock_vllm_config, mock_kv_cache_config,
                                mock_structured_output_manager):
        """Test get_request_counts aggregates across ranks."""
        scheduler = self._create_dp_scheduler_with_mocks(
            mock_vllm_config, mock_kv_cache_config,
            mock_structured_output_manager)

        # Mock running and waiting queues
        scheduler.schedulers[0].running = [MagicMock(),
                                           MagicMock()]  # 2 running
        scheduler.schedulers[0].waiting = [MagicMock()]  # 1 waiting
        scheduler.schedulers[1].running = [MagicMock()]  # 1 running
        scheduler.schedulers[1].waiting = [
            MagicMock(), MagicMock(), MagicMock()
        ]  # 3 waiting

        running, waiting = scheduler.get_request_counts()

        assert running == 3  # 2 + 1
        assert waiting == 4  # 1 + 3

    def test_reset_prefix_cache(self, mock_vllm_config, mock_kv_cache_config,
                                mock_structured_output_manager):
        """Test reset_prefix_cache resets all ranks."""
        scheduler = self._create_dp_scheduler_with_mocks(
            mock_vllm_config, mock_kv_cache_config,
            mock_structured_output_manager)

        scheduler.schedulers[0].reset_prefix_cache = MagicMock(
            return_value=True)
        scheduler.schedulers[1].reset_prefix_cache = MagicMock(
            return_value=True)

        result = scheduler.reset_prefix_cache()

        assert result is True
        scheduler.schedulers[0].reset_prefix_cache.assert_called_once()
        scheduler.schedulers[1].reset_prefix_cache.assert_called_once()

    def test_make_stats_with_logging_enabled(self, mock_vllm_config,
                                             mock_kv_cache_config,
                                             mock_structured_output_manager):
        """Test make_stats aggregates stats from all ranks."""
        scheduler = self._create_dp_scheduler_with_mocks(
            mock_vllm_config,
            mock_kv_cache_config,
            mock_structured_output_manager,
            log_stats=True)

        # Create mock stats for each rank
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

        scheduler.schedulers[0].make_stats = MagicMock(return_value=stats_0)
        scheduler.schedulers[1].make_stats = MagicMock(return_value=stats_1)

        combined_stats = scheduler.make_stats()

        # Verify aggregated stats
        assert combined_stats.num_running_reqs == 7  # 3 + 4
        assert combined_stats.num_waiting_reqs == 3  # 2 + 1
        assert combined_stats.kv_cache_usage == 0.6  # (0.5 + 0.7) / 2

        # Verify prefix cache stats
        assert combined_stats.prefix_cache_stats.requests == 25  # 10 + 15
        assert combined_stats.prefix_cache_stats.queries == 20  # 8 + 12
        assert combined_stats.prefix_cache_stats.hits == 13  # 5 + 8

        # Verify connector prefix cache stats
        assert combined_stats.connector_prefix_cache_stats.requests == 11  # 5 + 6
        assert combined_stats.connector_prefix_cache_stats.queries == 9  # 4 + 5
        assert combined_stats.connector_prefix_cache_stats.hits == 5  # 2 + 3

    def test_make_stats_with_logging_disabled(self, mock_vllm_config,
                                              mock_kv_cache_config,
                                              mock_structured_output_manager):
        """Test make_stats returns None when logging is disabled."""
        mock_scheduler_cls = MagicMock(return_value=MagicMock())
        with patch.object(mock_vllm_config.scheduler_config,
                          '_original_scheduler_cls', mock_scheduler_cls):
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
        """Test update_draft_token_ids routes tokens to correct ranks."""
        scheduler = self._create_dp_scheduler_with_mocks(
            mock_vllm_config, mock_kv_cache_config,
            mock_structured_output_manager)

        # Setup assigned ranks
        scheduler.assigned_dp_rank = {"req1": 0, "req2": 1, "req3": 0}

        # Create mock draft token IDs
        draft_token_ids = MagicMock()
        draft_token_ids.req_ids = ["req1", "req2", "req3"]
        draft_token_ids.draft_token_ids = [
            [101, 102, 103],
            [201, 202],
            [301, 302, 303, 304],
        ]

        # Mock scheduler update_draft_token_ids
        scheduler.schedulers[0].update_draft_token_ids = MagicMock()
        scheduler.schedulers[1].update_draft_token_ids = MagicMock()

        scheduler.update_draft_token_ids(draft_token_ids)

        # Verify each scheduler received correct tokens
        assert scheduler.schedulers[0].update_draft_token_ids.called
        assert scheduler.schedulers[1].update_draft_token_ids.called

        # Check rank 0 got req1 and req3
        call_args_0 = scheduler.schedulers[0].update_draft_token_ids.call_args[
            0][0]
        assert "req1" in call_args_0.req_ids
        assert "req3" in call_args_0.req_ids

        # Check rank 1 got req2
        call_args_1 = scheduler.schedulers[1].update_draft_token_ids.call_args[
            0][0]
        assert "req2" in call_args_1.req_ids

    def test_shutdown(self, mock_vllm_config, mock_kv_cache_config,
                      mock_structured_output_manager):
        """Test shutdown calls shutdown on all schedulers."""
        scheduler = self._create_dp_scheduler_with_mocks(
            mock_vllm_config, mock_kv_cache_config,
            mock_structured_output_manager)

        scheduler.schedulers[0].shutdown = MagicMock()
        scheduler.schedulers[1].shutdown = MagicMock()

        scheduler.shutdown()

        scheduler.schedulers[0].shutdown.assert_called_once()
        scheduler.schedulers[1].shutdown.assert_called_once()


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
