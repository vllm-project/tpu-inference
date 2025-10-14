# SPDX-License-Identifier: Apache-2.0

import itertools
import unittest
from unittest.mock import MagicMock, Mock, patch
from dataclasses import dataclass

from vllm.config import VllmConfig
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.request import Request
from vllm.v1.metrics.stats import PrefixCacheStats, SchedulerStats
from vllm.v1.spec_decode.metrics import SpecDecodingStats

from tpu_inference.core.sched.dp_scheduler import create_dp_scheduler, DPSchedulerOutput


class TestDPScheduler(unittest.TestCase):

    def setUp(self):
        """Set up test fixtures for DP scheduler tests."""
        # Create mock vllm_config with DP configuration
        self.mock_vllm_config = MagicMock(spec=VllmConfig)
        self.mock_vllm_config.additional_config = {
            "sharding": {
                "sharding_strategy": {
                    "data_parallelism": 4
                }
            }
        }
        
        # Create mock configs
        self.mock_kv_cache_config = MagicMock()
        self.mock_kv_cache_config.num_blocks = 1000
        
        self.mock_cache_config = MagicMock()
        self.mock_cache_config.enable_prefix_caching = True
        
        # Create mock KV cache manager
        self.mock_kv_cache_manager = MagicMock(spec=KVCacheManager)
        
        # Create base scheduler args
        self.base_scheduler_args = {
            'vllm_config': self.mock_vllm_config,
            'kv_cache_config': self.mock_kv_cache_config,
            'cache_config': self.mock_cache_config,
            'max_model_len': 2048,
            'use_eagle': False,
            'log_stats': True,
            'enable_kv_cache_events': False,
            'dcp_world_size': 1,
        }
        
        # Mock the get_dp_size function
        self.get_dp_size_patcher = patch('tpu_inference.core.sched.dp_scheduler.get_dp_size')
        self.mock_get_dp_size = self.get_dp_size_patcher.start()
        self.mock_get_dp_size.return_value = (4, 1, 4)  # dp_size, attn_dp, total_dp
        
        # Mock KVCacheManager constructor
        self.kv_cache_manager_patcher = patch('tpu_inference.core.sched.dp_scheduler.KVCacheManager')
        self.mock_kv_cache_manager_class = self.kv_cache_manager_patcher.start()
        self.mock_kv_cache_manager_class.return_value = self.mock_kv_cache_manager

    def tearDown(self):
        """Clean up test fixtures."""
        self.get_dp_size_patcher.stop()
        self.kv_cache_manager_patcher.stop()

    def test_create_dp_scheduler_returns_class(self):
        """Test that create_dp_scheduler returns a class."""
        DPScheduler = create_dp_scheduler()
        self.assertTrue(issubclass(DPScheduler, Scheduler))

    def test_create_dp_scheduler_with_custom_base(self):
        """Test creating DP scheduler with custom base class."""
        class CustomScheduler(Scheduler):
            pass
        
        DPScheduler = create_dp_scheduler(CustomScheduler)
        self.assertTrue(issubclass(DPScheduler, CustomScheduler))

    @patch('tpu_inference.core.sched.dp_scheduler.Scheduler.__init__')
    def test_dp_scheduler_initialization(self, mock_super_init):
        """Test DP scheduler initialization."""
        mock_super_init.return_value = None
        
        DPScheduler = create_dp_scheduler()
        
        # Create instance with mocked attributes
        scheduler = DPScheduler.__new__(DPScheduler)
        scheduler.vllm_config = self.mock_vllm_config
        scheduler.kv_cache_config = self.mock_kv_cache_config
        scheduler.cache_config = self.mock_cache_config
        scheduler.max_model_len = 2048
        scheduler.use_eagle = False
        scheduler.log_stats = True
        scheduler.enable_kv_cache_events = False
        scheduler.dcp_world_size = 1
        scheduler.kv_cache_manager = self.mock_kv_cache_manager
        
        # Call __init__ manually
        scheduler.__init__()
        
        # Verify initialization
        self.assertEqual(scheduler.dp_size, 4)
        self.assertIsInstance(scheduler.assigned_dp_rank, dict)
        self.assertIsInstance(scheduler.round_robin_counter, itertools.cycle)
        self.assertEqual(len(scheduler.dp_kv_cache_managers), 4)

    def test_dp_scheduler_output_dataclass(self):
        """Test DPSchedulerOutput dataclass."""
        output = DPSchedulerOutput(
            scheduled_new_reqs=[],
            scheduled_cached_reqs=None,
            num_scheduled_tokens=0,
            total_num_scheduled_tokens=0,
            scheduled_spec_decode_tokens=0,
            scheduled_encoder_inputs=[],
            num_common_prefix_blocks=0,
            finished_req_ids=[],
            free_encoder_mm_hashes=[],
            structured_output_request_ids=[],
            grammar_bitmask=None,
            kv_connector_metadata=None,
            assigned_dp_rank={'req1': 0, 'req2': 1}
        )
        
        self.assertEqual(output.assigned_dp_rank, {'req1': 0, 'req2': 1})
        self.assertIsInstance(output, SchedulerOutput)

    def test_get_or_assign_dp_rank_new_request(self):
        """Test DP rank assignment for new requests."""
        DPScheduler = create_dp_scheduler()
        
        # Create scheduler instance with mocked setup
        scheduler = self._create_mock_scheduler_instance(DPScheduler)
        
        # Create mock request
        mock_request = MagicMock(spec=Request)
        mock_request.request_id = "test_req_1"
        
        # Test assignment for new request
        rank = scheduler._get_or_assign_dp_rank(mock_request)
        
        # Should assign rank 0 (first in round robin)
        self.assertEqual(rank, 0)
        self.assertEqual(scheduler.assigned_dp_rank["test_req_1"], 0)

    def test_get_or_assign_dp_rank_existing_request(self):
        """Test DP rank retrieval for existing requests."""
        DPScheduler = create_dp_scheduler()
        
        # Create scheduler instance with mocked setup
        scheduler = self._create_mock_scheduler_instance(DPScheduler)
        
        # Create mock request
        mock_request = MagicMock(spec=Request)
        mock_request.request_id = "test_req_1"
        
        # Pre-assign a rank
        scheduler.assigned_dp_rank["test_req_1"] = 2
        
        # Test retrieval
        rank = scheduler._get_or_assign_dp_rank(mock_request)
        
        # Should return existing rank
        self.assertEqual(rank, 2)

    def test_round_robin_assignment(self):
        """Test round-robin assignment across multiple requests."""
        DPScheduler = create_dp_scheduler()
        
        # Create scheduler instance with mocked setup
        scheduler = self._create_mock_scheduler_instance(DPScheduler)
        
        # Create multiple mock requests
        requests = [MagicMock(spec=Request) for _ in range(6)]
        for i, req in enumerate(requests):
            req.request_id = f"test_req_{i}"
        
        # Assign ranks
        ranks = [scheduler._get_or_assign_dp_rank(req) for req in requests]
        
        # Should cycle through [0, 1, 2, 3, 0, 1]
        expected_ranks = [0, 1, 2, 3, 0, 1]
        self.assertEqual(ranks, expected_ranks)

    def test_kv_cache_manager_proxy_allocate_slots(self):
        """Test KV cache manager proxy allocate_slots method."""
        DPScheduler = create_dp_scheduler()
        scheduler = self._create_mock_scheduler_instance(DPScheduler)
        
        # Create mock request
        mock_request = MagicMock(spec=Request)
        mock_request.request_id = "test_req"
        
        # Mock manager allocate_slots method
        mock_manager = MagicMock()
        scheduler.dp_kv_cache_managers = [mock_manager] * 4
        
        # Call allocate_slots through proxy
        scheduler.kv_cache_manager.allocate_slots(mock_request, "arg1", key="value")
        
        # Verify correct manager called
        mock_manager.allocate_slots.assert_called_once_with(mock_request, "arg1", key="value")

    def test_kv_cache_manager_proxy_get_blocks(self):
        """Test KV cache manager proxy get_blocks method."""
        DPScheduler = create_dp_scheduler()
        scheduler = self._create_mock_scheduler_instance(DPScheduler)
        
        # Pre-assign request to rank
        scheduler.assigned_dp_rank["test_req"] = 2
        
        # Mock managers
        mock_managers = [MagicMock() for _ in range(4)]
        scheduler.dp_kv_cache_managers = mock_managers
        
        # Call get_blocks
        scheduler.kv_cache_manager.get_blocks("test_req")
        
        # Verify correct manager called
        mock_managers[2].get_blocks.assert_called_once_with("test_req")

    def test_kv_cache_manager_proxy_free(self):
        """Test KV cache manager proxy free method."""
        DPScheduler = create_dp_scheduler()
        scheduler = self._create_mock_scheduler_instance(DPScheduler)
        
        # Create mock request
        mock_request = MagicMock(spec=Request)
        mock_request.request_id = "test_req"
        
        # Pre-assign request to rank
        scheduler.assigned_dp_rank["test_req"] = 1
        
        # Mock managers
        mock_managers = [MagicMock() for _ in range(4)]
        scheduler.dp_kv_cache_managers = mock_managers
        
        # Call free
        scheduler.kv_cache_manager.free(mock_request)
        
        # Verify correct manager called
        mock_managers[1].free.assert_called_once_with(mock_request)

    def test_get_computed_blocks_dp_cache_hit(self):
        """Test get_computed_blocks_dp with cache hit."""
        DPScheduler = create_dp_scheduler()
        scheduler = self._create_mock_scheduler_instance(DPScheduler)
        
        # Create mock request
        mock_request = MagicMock(spec=Request)
        mock_request.request_id = "test_req"
        
        # Mock managers with different cache results
        mock_managers = []
        for i in range(4):
            manager = MagicMock()
            if i == 2:
                # Manager 2 has best cache hit
                blocks = MagicMock()
                blocks.blocks = [[1, 2, 3, 4, 5]]  # 5 cached tokens
                manager.get_computed_blocks.return_value = (blocks, 5)
            else:
                blocks = MagicMock()
                blocks.blocks = [[]]  # No cached tokens
                manager.get_computed_blocks.return_value = (blocks, 0)
            mock_managers.append(manager)
        
        scheduler.dp_kv_cache_managers = mock_managers
        
        # Call get_computed_blocks_dp
        with patch('builtins.print'):  # Suppress print output
            result_blocks, result_tokens = scheduler._get_computed_blocks_dp(mock_request)
        
        # Should assign to manager 2 and return its blocks
        self.assertEqual(scheduler.assigned_dp_rank["test_req"], 2)
        self.assertEqual(result_tokens, 5)

    def test_get_computed_blocks_dp_no_cache_hit(self):
        """Test get_computed_blocks_dp with no cache hit."""
        DPScheduler = create_dp_scheduler()
        scheduler = self._create_mock_scheduler_instance(DPScheduler)
        
        # Create mock request
        mock_request = MagicMock(spec=Request)
        mock_request.request_id = "test_req"
        
        # Mock managers with no cache hits
        mock_managers = []
        for i in range(4):
            manager = MagicMock()
            blocks = MagicMock()
            blocks.blocks = [[]]  # No cached tokens
            manager.get_computed_blocks.return_value = (blocks, 0)
            mock_managers.append(manager)
        
        scheduler.dp_kv_cache_managers = mock_managers
        
        # Call get_computed_blocks_dp
        result_blocks, result_tokens = scheduler._get_computed_blocks_dp(mock_request)
        
        # Should assign using round robin (rank 0)
        self.assertEqual(scheduler.assigned_dp_rank["test_req"], 0)

    @patch('tpu_inference.core.sched.dp_scheduler.Scheduler.schedule')
    def test_schedule_returns_dp_output(self, mock_super_schedule):
        """Test that schedule returns DPSchedulerOutput."""
        DPScheduler = create_dp_scheduler()
        scheduler = self._create_mock_scheduler_instance(DPScheduler)
        
        # Mock base scheduler output
        base_output = MagicMock(spec=SchedulerOutput)
        base_output.scheduled_new_reqs = []
        base_output.scheduled_cached_reqs = MagicMock()
        base_output.scheduled_cached_reqs.req_ids = ["req1", "req2"]
        base_output.num_scheduled_tokens = 100
        base_output.total_num_scheduled_tokens = 200
        base_output.scheduled_spec_decode_tokens = 50
        base_output.scheduled_encoder_inputs = []
        base_output.num_common_prefix_blocks = 10
        base_output.finished_req_ids = []
        base_output.free_encoder_mm_hashes = []
        base_output.structured_output_request_ids = []
        base_output.grammar_bitmask = None
        base_output.kv_connector_metadata = None
        
        mock_super_schedule.return_value = base_output
        
        # Set up assigned ranks
        scheduler.assigned_dp_rank = {"req1": 0, "req2": 2}
        
        # Call schedule
        with patch('builtins.print'):  # Suppress print output
            result = scheduler.schedule()
        
        # Verify result type and contents
        self.assertIsInstance(result, DPSchedulerOutput)
        self.assertEqual(result.assigned_dp_rank, {"req1": 0, "req2": 2})
        self.assertEqual(result.num_scheduled_tokens, 100)

    def test_free_blocks_removes_request(self):
        """Test _free_blocks removes request from assigned ranks."""
        DPScheduler = create_dp_scheduler()
        scheduler = self._create_mock_scheduler_instance(DPScheduler)
        
        # Create mock request
        mock_request = MagicMock(spec=Request)
        mock_request.request_id = "test_req"
        mock_request.is_finished.return_value = True
        
        # Set up state
        scheduler.assigned_dp_rank["test_req"] = 1
        scheduler.requests = {"test_req": mock_request}
        
        # Mock managers
        mock_managers = [MagicMock() for _ in range(4)]
        scheduler.dp_kv_cache_managers = mock_managers
        
        # Call _free_blocks
        scheduler._free_blocks(mock_request)
        
        # Verify manager free called and request removed
        mock_managers[1].free.assert_called_once_with(mock_request)
        self.assertNotIn("test_req", scheduler.requests)

    def test_make_stats_combines_manager_stats(self):
        """Test make_stats combines statistics from all DP managers."""
        DPScheduler = create_dp_scheduler()
        scheduler = self._create_mock_scheduler_instance(DPScheduler)
        
        # Set up scheduler state
        scheduler.log_stats = True
        scheduler.running = [MagicMock(), MagicMock()]
        scheduler.waiting = [MagicMock()]
        
        # Mock managers with stats
        mock_managers = []
        for i in range(4):
            manager = MagicMock()
            manager.usage = 0.5 + i * 0.1  # Different usage per manager
            
            prefix_stats = MagicMock(spec=PrefixCacheStats)
            prefix_stats.reset = False
            prefix_stats.requests = 10 + i
            prefix_stats.queries = 20 + i
            prefix_stats.hits = 5 + i
            manager.make_prefix_cache_stats.return_value = prefix_stats
            
            mock_managers.append(manager)
        
        scheduler.dp_kv_cache_managers = mock_managers
        
        # Call make_stats
        stats = scheduler.make_stats()
        
        # Verify combined stats
        self.assertIsInstance(stats, SchedulerStats)
        self.assertEqual(stats.num_running_reqs, 2)
        self.assertEqual(stats.num_waiting_reqs, 1)
        
        # Average usage should be (0.5 + 0.6 + 0.7 + 0.8) / 4 = 0.65
        self.assertAlmostEqual(stats.kv_cache_usage, 0.65)
        
        # Combined prefix cache stats should sum up
        self.assertEqual(stats.prefix_cache_stats.requests, 10 + 11 + 12 + 13)
        self.assertEqual(stats.prefix_cache_stats.queries, 20 + 21 + 22 + 23)
        self.assertEqual(stats.prefix_cache_stats.hits, 5 + 6 + 7 + 8)

    def test_make_stats_returns_none_when_disabled(self):
        """Test make_stats returns None when log_stats is False."""
        DPScheduler = create_dp_scheduler()
        scheduler = self._create_mock_scheduler_instance(DPScheduler)
        
        scheduler.log_stats = False
        
        stats = scheduler.make_stats()
        self.assertIsNone(stats)

    def test_kv_cache_manager_proxy_usage_property(self):
        """Test KV cache manager proxy usage property."""
        DPScheduler = create_dp_scheduler()
        scheduler = self._create_mock_scheduler_instance(DPScheduler)
        
        # Mock managers with different usage
        mock_managers = []
        usages = [0.2, 0.4, 0.6, 0.8]
        for usage in usages:
            manager = MagicMock()
            manager.usage = usage
            mock_managers.append(manager)
        
        scheduler.dp_kv_cache_managers = mock_managers
        
        # Test usage property
        avg_usage = scheduler.kv_cache_manager.usage
        expected_avg = sum(usages) / len(usages)  # 0.5
        self.assertEqual(avg_usage, expected_avg)

    def _create_mock_scheduler_instance(self, DPScheduler):
        """Helper method to create a mock scheduler instance."""
        scheduler = DPScheduler.__new__(DPScheduler)
        
        # Mock required attributes
        scheduler.vllm_config = self.mock_vllm_config
        scheduler.kv_cache_config = self.mock_kv_cache_config
        scheduler.cache_config = self.mock_cache_config
        scheduler.max_model_len = 2048
        scheduler.use_eagle = False
        scheduler.log_stats = True
        scheduler.enable_kv_cache_events = False
        scheduler.dcp_world_size = 1
        scheduler.kv_cache_manager = self.mock_kv_cache_manager
        scheduler.requests = {}
        scheduler.running = []
        scheduler.waiting = []
        
        # Initialize DP-specific attributes
        scheduler._init_dp_config()
        scheduler._init_dp_state()
        scheduler._replace_kv_cache_manager()
        
        return scheduler


if __name__ == '__main__':
    unittest.main()