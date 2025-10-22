# SPDX-License-Identifier: Apache-2.0

import itertools
import unittest
from unittest.mock import MagicMock, patch

from vllm.config import VllmConfig
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.metrics.stats import PrefixCacheStats, SchedulerStats
from vllm.v1.request import Request

from tpu_inference.core.sched.dp_scheduler import DPScheduler


class TestDPScheduler(unittest.TestCase):

    def setUp(self):
        """Set up test fixtures for DP scheduler tests."""
        # Create mock vllm_config with DP configuration
        self.mock_vllm_config = MagicMock(spec=VllmConfig)
        self.mock_sharding_config = MagicMock()
        self.mock_sharding_config.total_dp_size = 4
        self.mock_vllm_config.sharding_config = self.mock_sharding_config

        # Create mock scheduler config
        self.mock_scheduler_config = MagicMock()
        self.mock_scheduler_config.max_num_seqs = 32
        self.mock_scheduler_config.max_num_batched_tokens = 1024

        # Create mock configs
        self.mock_kv_cache_config = MagicMock()
        self.mock_kv_cache_config.num_blocks = 1000

        self.mock_cache_config = MagicMock()
        self.mock_cache_config.enable_prefix_caching = True

        # Create mock KV cache manager
        self.mock_kv_cache_manager = MagicMock(spec=KVCacheManager)
        self.mock_kv_cache_manager.num_kv_cache_groups = 1

        # Create base scheduler args
        self.base_scheduler_args = {
            'vllm_config': self.mock_vllm_config,
            'scheduler_config': self.mock_scheduler_config,
            'kv_cache_config': self.mock_kv_cache_config,
            'cache_config': self.mock_cache_config,
            'max_model_len': 2048,
            'use_eagle': False,
            'log_stats': True,
            'enable_kv_cache_events': False,
            'dcp_world_size': 1,
        }

        # Mock KVCacheManager constructor
        self.kv_cache_manager_patcher = patch(
            'tpu_inference.core.sched.dp_scheduler.KVCacheManager')
        self.mock_kv_cache_manager_class = self.kv_cache_manager_patcher.start(
        )
        self.mock_kv_cache_manager_class.return_value = self.mock_kv_cache_manager

    def tearDown(self):
        """Clean up test fixtures."""
        self.kv_cache_manager_patcher.stop()

    def _create_mock_scheduler_instance(self):
        """Helper to create a properly mocked DPScheduler instance."""
        # Mock the parent Scheduler.__init__ to avoid complex setup
        with patch('tpu_inference.core.sched.dp_scheduler.Scheduler.__init__'
                   ) as mock_super_init:
            mock_super_init.return_value = None

            # Create instance with manually set attributes (similar to working test)
            scheduler = DPScheduler.__new__(DPScheduler)
            scheduler.vllm_config = self.mock_vllm_config
            scheduler.scheduler_config = self.mock_scheduler_config
            scheduler.kv_cache_config = self.mock_kv_cache_config
            scheduler.cache_config = self.mock_cache_config
            scheduler.max_model_len = 2048
            scheduler.use_eagle = False
            scheduler.log_stats = True
            scheduler.enable_kv_cache_events = False
            scheduler.dcp_world_size = 1
            scheduler.kv_cache_manager = self.mock_kv_cache_manager

            # Now call the real __init__ to set up DP state
            scheduler.__init__(**self.base_scheduler_args)

            # Mock running and waiting queues (these might not be set by real init)
            if not hasattr(scheduler, 'running'):
                scheduler.running = []
            if not hasattr(scheduler, 'waiting'):
                scheduler.waiting = []

            return scheduler

    def test_dp_scheduler_is_subclass(self):
        """Test that DPScheduler is a subclass of Scheduler."""
        self.assertTrue(issubclass(DPScheduler, Scheduler))

    @patch('tpu_inference.core.sched.dp_scheduler.Scheduler.__init__')
    def test_dp_scheduler_initialization(self, mock_super_init):
        """Test DP scheduler initialization."""
        mock_super_init.return_value = None

        # Create instance with mocked attributes
        scheduler = DPScheduler.__new__(DPScheduler)
        scheduler.vllm_config = self.mock_vllm_config
        scheduler.scheduler_config = self.mock_scheduler_config
        scheduler.kv_cache_config = self.mock_kv_cache_config
        scheduler.cache_config = self.mock_cache_config
        scheduler.max_model_len = 2048
        scheduler.use_eagle = False
        scheduler.log_stats = True
        scheduler.enable_kv_cache_events = False
        scheduler.dcp_world_size = 1
        scheduler.kv_cache_manager = self.mock_kv_cache_manager

        scheduler.__init__(**self.base_scheduler_args)

        # Verify initialization
        self.assertEqual(scheduler.dp_size, 4)
        self.assertIsInstance(scheduler.assigned_dp_rank, dict)
        self.assertIsInstance(scheduler.round_robin_counter, itertools.cycle)
        self.assertEqual(len(scheduler.dp_kv_cache_managers), 4)
        self.assertEqual(scheduler.max_reqs_per_dp_rank, 8)  # 32 // 4
        self.assertEqual(scheduler.max_tokens_per_dp_rank, 256)  # 1024 // 4

    def test_get_dp_rank_for_allocation_new_request(self):
        """Test DP rank allocation for new requests."""
        # Create scheduler instance with mocked setup
        scheduler = self._create_mock_scheduler_instance()

        # Create mock request
        mock_request = MagicMock(spec=Request)
        mock_request.request_id = "test_req_1"

        # Test allocation for new request
        rank = scheduler._get_or_find_dp_rank(mock_request,
                                              num_tokens=10)

        # Should assign rank 0 (first in round robin)
        self.assertEqual(rank, 0)

    def test_get_dp_rank_for_allocation_existing_request(self):
        """Test DP rank allocation for existing requests."""
        # Create scheduler instance with mocked setup
        scheduler = self._create_mock_scheduler_instance()

        # Create mock request
        mock_request = MagicMock(spec=Request)
        mock_request.request_id = "test_req_1"

        # Pre-assign a rank
        scheduler.assigned_dp_rank["test_req_1"] = 2

        # Test allocation with existing request
        rank = scheduler._get_or_find_dp_rank(mock_request,
                                              num_tokens=10)

        # Should return existing rank
        self.assertEqual(rank, 2)

    def test_round_robin_assignment(self):
        """Test round-robin assignment across multiple requests."""
        # Create scheduler instance with mocked setup
        scheduler = self._create_mock_scheduler_instance()

        # Create multiple mock requests
        requests = [MagicMock(spec=Request) for _ in range(6)]
        for i, req in enumerate(requests):
            req.request_id = f"test_req_{i}"

        # Allocate ranks
        ranks = [
            scheduler._get_or_find_dp_rank(req, num_tokens=10)
            for req in requests
        ]

        # Should cycle through [0, 1, 2, 3, 0, 1]
        expected_ranks = [0, 1, 2, 3, 0, 1]
        self.assertEqual(ranks, expected_ranks)

    def test_get_computed_blocks_dp_cache_hit(self):
        """Test get_computed_blocks_dp with cache hit."""
        scheduler = self._create_mock_scheduler_instance()

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
        result_blocks, result_tokens = scheduler._get_computed_blocks_dp(
            mock_request)

        # Should assign to manager 2 and return its blocks
        self.assertEqual(scheduler.assigned_dp_rank["test_req"], 2)
        self.assertEqual(result_tokens, 5)

    def test_get_computed_blocks_dp_no_cache_hit(self):
        """Test get_computed_blocks_dp with no cache hit."""
        scheduler = self._create_mock_scheduler_instance()

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
        result_blocks, result_tokens = scheduler._get_computed_blocks_dp(
            mock_request)

        # Should return empty blocks and 0 tokens since no cache hit
        self.assertEqual(result_tokens, 0)

    def test_kv_cache_manager_proxy_free_request(self):
        """Test KV cache manager proxy free method removes request from assigned ranks."""
        scheduler = self._create_mock_scheduler_instance()

        # Create mock request
        mock_request = MagicMock(spec=Request)
        mock_request.request_id = "test_req"

        # Set up state
        scheduler.assigned_dp_rank["test_req"] = 1
        scheduler.num_scheduled_tokens["test_req"] = 50

        # Mock managers
        mock_managers = [MagicMock() for _ in range(4)]
        scheduler.dp_kv_cache_managers = mock_managers

        # Call free through the proxy
        scheduler.kv_cache_manager.free(mock_request)

        # Verify manager free called and request removed from state
        mock_managers[1].free.assert_called_once_with(mock_request)
        self.assertNotIn("test_req", scheduler.assigned_dp_rank)
        self.assertNotIn("test_req", scheduler.num_scheduled_tokens)
        self.assertEqual(scheduler.request_budget[1],
                         9)  # Should be incremented
        self.assertEqual(scheduler.token_budget[1],
                         306)  # Should be incremented by 50

    def test_make_stats_combines_manager_stats(self):
        """Test make_stats combines statistics from all DP managers."""
        scheduler = self._create_mock_scheduler_instance()

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

    def test_dynamic_token_budget(self):
        """Test get_dynamic_token_budget method."""
        scheduler = self._create_mock_scheduler_instance()

        # Create mock request
        mock_request = MagicMock(spec=Request)
        mock_request.request_id = "test_req"

        # Test with unassigned request
        budget = scheduler.get_dynamic_token_budget(mock_request,
                                                    available_budget=1000)
        self.assertEqual(budget, 256)  # Should return max budget

        # Test with assigned request
        scheduler.assigned_dp_rank["test_req"] = 1
        scheduler.token_budget[1] = 100
        budget = scheduler.get_dynamic_token_budget(mock_request,
                                                    available_budget=1000)
        self.assertEqual(budget, 100)  # Should return specific DP rank budget

    def test_rank_capacity_checks(self):
        """Test DP rank capacity check methods."""
        scheduler = self._create_mock_scheduler_instance()

        # Test new request capacity
        self.assertTrue(scheduler._rank_has_capacity_for_new_requests(0, 50))
        self.assertFalse(scheduler._rank_has_capacity_for_new_requests(
            0, 500))  # Too many tokens

        # Exhaust request budget
        scheduler.request_budget[0] = 0
        self.assertFalse(scheduler._rank_has_capacity_for_new_requests(0, 50))

        # Test existing request capacity
        scheduler.request_budget[0] = 8  # Reset
        self.assertTrue(
            scheduler._rank_has_capacity_for_existing_requests(0, 50))
        self.assertFalse(
            scheduler._rank_has_capacity_for_existing_requests(0, 500))

    def test_update_dp_rank_budget(self):
        """Test DP rank budget update method."""
        scheduler = self._create_mock_scheduler_instance()

        # Create mock request
        mock_request = MagicMock(spec=Request)
        mock_request.request_id = "test_req"

        # Update budget for new request
        scheduler._assign_rank_and_update_budget(mock_request, 1, 50)

        # Verify state updates
        self.assertEqual(scheduler.assigned_dp_rank["test_req"], 1)
        self.assertEqual(scheduler.num_scheduled_tokens["test_req"], 50)
        self.assertEqual(scheduler.request_budget[1], 7)  # Decremented from 8
        self.assertEqual(scheduler.token_budget[1],
                         206)  # Decremented by 50 from 256


if __name__ == '__main__':
    unittest.main()
