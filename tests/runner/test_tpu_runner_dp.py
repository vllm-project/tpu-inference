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

from contextlib import nullcontext
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from tpu_inference.runner.tpu_runner import TPUModelRunner


class TestTPUJaxRunnerDPInputsLightweight:

    def setup_method(self):
        self.runner = MagicMock()

        # Basic DP configuration
        self.runner.dp_size = 2
        self.runner.max_num_tokens = 64
        self.runner.max_num_reqs = 8
        self.runner.max_num_blocks_per_req = 8
        self.runner.num_tokens_paddings = [16, 32, 64]

        # Mock input batch - adjust num_reqs to match test data
        self.runner.input_batch = MagicMock()
        self.runner.input_batch.num_reqs = 2
        self.runner.input_batch.req_ids = ["req1", "req2", "req3", "req4"]
        self.runner.input_batch.req_id_to_index = {
            "req1": 0,
            "req2": 1,
            "req3": 2,
            "req4": 3
        }
        self.runner.input_batch.num_computed_tokens_cpu = np.array(
            [10, 20, 5, 15])
        self.runner.input_batch.token_ids_cpu = np.random.randint(
            0, 1000, (8, 64), dtype=np.int32)

        # Mock block table
        mock_block_table = MagicMock()
        mock_block_table.get_cpu_tensor.return_value = np.arange(32).reshape(
            4, 8)
        self.runner.input_batch.block_table = [mock_block_table]

        # Initialize CPU arrays that the method modifies
        self.runner.input_ids_cpu = np.zeros(64, dtype=np.int32)
        self.runner.positions_cpu = np.zeros(64, dtype=np.int32)
        self.runner.query_start_loc_cpu = np.zeros(10, dtype=np.int32)
        self.runner.seq_lens_cpu = np.zeros(8, dtype=np.int32)
        self.runner.logits_indices_cpu = np.zeros(8, dtype=np.int32)
        self.runner.block_tables_cpu = [np.zeros((8, 8), dtype=np.int32)]
        self.runner.arange_cpu = np.arange(64, dtype=np.int64)

        # mock kv cache group
        mock_kv_cache_config = MagicMock()
        mock_kv_cache_group = MagicMock()
        mock_kv_cache_config.kv_cache_groups = [mock_kv_cache_group]
        self.runner.kv_cache_config = mock_kv_cache_config
        self.runner.use_hybrid_kvcache = False

        # Mock scheduler config for async scheduling
        self.runner.scheduler_config = MagicMock()
        self.runner.scheduler_config.async_scheduling = False  # Default to False for most tests
        self.runner._pre_async_results = None  # Default to None for most tests

        # Bind the actual methods to our mock
        self.runner._prepare_inputs_dp = TPUModelRunner._prepare_inputs_dp.__get__(
            self.runner)
        self.runner._prepare_dp_input_metadata = TPUModelRunner._prepare_dp_input_metadata.__get__(
            self.runner)
        self.runner._prepare_async_token_substitution_indices_dp = TPUModelRunner._prepare_async_token_substitution_indices_dp.__get__(
            self.runner)

    def _create_mock_scheduler_output(self,
                                      num_scheduled_tokens_dict,
                                      assigned_dp_ranks,
                                      scheduled_spec_decode_tokens=None):
        """Create a minimal mock scheduler output."""
        mock_output = MagicMock()
        mock_output.num_scheduled_tokens = num_scheduled_tokens_dict
        mock_output.assigned_dp_rank = assigned_dp_ranks
        mock_output.total_num_scheduled_tokens = sum(
            num_scheduled_tokens_dict.values())
        mock_output.scheduled_spec_decode_tokens = scheduled_spec_decode_tokens or {}
        mock_output.grammar_bitmask = None
        return mock_output

    def _create_mock_hybrid_kv_cache_config(self):
        mock_kv_cache_config = MagicMock()
        mock_kv_cache_group1 = MagicMock()
        mock_kv_cache_group1.layer_names = [f'layer.{i}' for i in range(10)]
        mock_kv_cache_group2 = MagicMock()
        mock_kv_cache_group2.layer_names = [
            f'layer.{i}' for i in range(10, 20)
        ]
        mock_kv_cache_config.kv_cache_groups = [
            mock_kv_cache_group1, mock_kv_cache_group2
        ]
        self.runner.kv_cache_config = mock_kv_cache_config
        self.runner.use_hybrid_kvcache = True

    @patch('tpu_inference.runner.tpu_runner.NamedSharding')
    @patch('tpu_inference.runner.tpu_runner.runner_utils')
    @patch('tpu_inference.runner.tpu_runner.device_array',
           side_effect=lambda mesh, tensors, **kwargs: tensors)
    @patch('tpu_inference.runner.tpu_runner.TPUSupportedSamplingMetadata')
    def test_prepare_inputs_dp_basic_functionality(self,
                                                   mock_sampling_metadata,
                                                   mock_device_array,
                                                   mock_runner_utils,
                                                   mock_named_sharding):
        """Test basic functionality of _prepare_inputs_dp."""
        # Mock utility functions
        mock_runner_utils.get_padded_token_len.return_value = 16
        mock_sampling_metadata.from_input_batch.return_value = MagicMock()
        mock_named_sharding.return_value = MagicMock()

        # Create test data - only use req1 and req2 to match num_reqs=2
        num_scheduled_tokens = {"req1": 5, "req2": 3}
        assigned_dp_ranks = {"req1": 0, "req2": 1}
        scheduler_output = self._create_mock_scheduler_output(
            num_scheduled_tokens, assigned_dp_ranks)

        # Execute the method
        result = self.runner._prepare_inputs_dp(scheduler_output)

        # Basic assertions
        assert len(result) == 8
        input_ids, positions, attention_metadata, sampling_metadata, logits_indices, spec_decode_metadata, logits_indices_selector, padded_num_reqs = result

        # Verify utility functions were called
        mock_runner_utils.get_padded_token_len.assert_called()

    def test_prepare_inputs_dp_error_conditions(self):
        """Test error handling in DP input preparation."""
        # Test with zero scheduled tokens - should fail assertion: total_num_scheduled_tokens > 0
        scheduler_output = self._create_mock_scheduler_output({}, {})
        scheduler_output.total_num_scheduled_tokens = 0

        with pytest.raises(AssertionError):
            self.runner._prepare_inputs_dp(scheduler_output)

        # Test with zero requests - should fail assertion: num_reqs > 0
        self.runner.input_batch.num_reqs = 0
        scheduler_output = self._create_mock_scheduler_output({"req1": 5},
                                                              {"req1": 0})

        with pytest.raises(AssertionError):
            self.runner._prepare_inputs_dp(scheduler_output)

    @patch('tpu_inference.runner.tpu_runner.NamedSharding')
    @patch('tpu_inference.runner.tpu_runner.runner_utils')
    @patch('tpu_inference.runner.tpu_runner.device_array',
           side_effect=lambda mesh, tensors, **kwargs: tensors)
    @patch('tpu_inference.runner.tpu_runner.TPUSupportedSamplingMetadata')
    def test_prepare_inputs_dp_hybrid_kvcache(self, mock_sampling_metadata,
                                              mock_device_array,
                                              mock_runner_utils,
                                              mock_named_sharding):
        """Test basic functionality of _prepare_inputs_dp."""
        # Mock utility functions
        mock_runner_utils.get_padded_token_len.return_value = 16
        mock_sampling_metadata.from_input_batch.return_value = MagicMock()
        mock_named_sharding.return_value = MagicMock()

        # Create test data - only use req1 and req2 to match num_reqs=2
        num_scheduled_tokens = {"req1": 5, "req2": 3}
        assigned_dp_ranks = {"req1": 0, "req2": 1}
        scheduler_output = self._create_mock_scheduler_output(
            num_scheduled_tokens, assigned_dp_ranks)

        # Create hybrid kv cache config with 10 full attn layers, 10 sw attn layers
        self._create_mock_hybrid_kv_cache_config()

        # update input_batch's block_table
        mock_block_table = MagicMock()
        mock_block_table.get_cpu_tensor.return_value = np.arange(32).reshape(
            4, 8)
        self.runner.input_batch.block_table = [
            mock_block_table, mock_block_table
        ]

        # update model runner's block_tables_cpu:
        self.runner.block_tables_cpu = [
            np.zeros((8, 8), dtype=np.int32),
            np.zeros((8, 8), dtype=np.int32)
        ]

        # Execute the method
        result = self.runner._prepare_inputs_dp(scheduler_output)

        # Basic assertions
        assert len(result) == 8
        input_ids, positions, attention_metadata, sampling_metadata, logits_indices, spec_decode_metadata, logits_indices_selector, padded_num_reqs = result

        # Verify utility functions were called
        mock_runner_utils.get_padded_token_len.assert_called()

        # Verify there's attention_metadata for each layer
        assert isinstance(attention_metadata, dict)
        assert len(attention_metadata) == 20

    def test_prepare_dp_input_metadata(self):
        num_scheduled_tokens = {"req1": 10, "req2": 5, "req3": 8, "req4": 3}
        assigned_dp_ranks = {"req1": 0, "req2": 0, "req3": 1, "req4": 1}

        self.runner.input_batch.num_reqs = 4
        self.runner.input_batch.req_ids = ["req1", "req2", "req3", "req4"]
        self.runner.max_num_reqs = 8

        scheduler_output = self._create_mock_scheduler_output(
            num_scheduled_tokens, assigned_dp_ranks)

        with patch('tpu_inference.runner.tpu_runner.runner_utils'
                   ) as mock_runner_utils:
            mock_runner_utils.get_padded_token_len.side_effect = lambda paddings_list, val: 16 if val <= 15 else 32  # Padded tokens per DP rank

            result = self.runner._prepare_dp_input_metadata(scheduler_output)

            (req_ids_dp, req_indices_dp, num_scheduled_tokens_per_dp_rank,
             scheduled_tokens_per_dp_rank, num_req_per_dp_rank,
             padded_num_scheduled_tokens_per_dp_rank, padded_num_reqs,
             padded_total_num_scheduled_tokens, padded_num_reqs_per_dp_rank,
             logits_indices_selector, max_num_reqs_per_dp_rank) = result

            # 1. req_ids_dp: Dictionary mapping DP rank to request IDs
            assert isinstance(req_ids_dp, dict)
            assert req_ids_dp[0] == ["req1", "req2"]
            assert req_ids_dp[1] == ["req3", "req4"]

            # 2. req_indices_dp: Dictionary mapping DP rank to request indices
            assert isinstance(req_indices_dp, dict)
            assert req_indices_dp[0] == [0, 1]  # indices of req1, req2
            assert req_indices_dp[1] == [2, 3]  # indices of req3, req4

            # 3. num_scheduled_tokens_per_dp_rank: Total tokens per DP rank
            assert isinstance(num_scheduled_tokens_per_dp_rank, dict)
            assert num_scheduled_tokens_per_dp_rank[0] == 15  # 10 + 5
            assert num_scheduled_tokens_per_dp_rank[1] == 11  # 8 + 3

            # 4. scheduled_tokens_per_dp_rank: List of token counts per request per DP rank
            assert isinstance(scheduled_tokens_per_dp_rank, dict)
            assert scheduled_tokens_per_dp_rank[0] == [10,
                                                       5]  # req1=10, req2=5
            assert scheduled_tokens_per_dp_rank[1] == [8, 3]  # req3=8, req4=3

            # 5. num_req_per_dp_rank: Number of requests per DP rank
            assert isinstance(num_req_per_dp_rank, dict)
            assert num_req_per_dp_rank[0] == 2
            assert num_req_per_dp_rank[1] == 2

            # 6. padded_num_scheduled_tokens_per_dp_rank: Padded token count per rank
            assert padded_num_scheduled_tokens_per_dp_rank == 16

            # 7. padded_num_reqs: Total padded requests across all ranks
            assert padded_num_reqs == 32  # 2 DP ranks * 16 padded reqs per rank

            # 8. padded_total_num_scheduled_tokens: Total padded tokens across all ranks
            assert padded_total_num_scheduled_tokens == 32  # 2 DP ranks * 16 padded tokens per rank

            # 9. padded_num_reqs_per_dp_rank: Padded requests per DP rank
            assert padded_num_reqs_per_dp_rank == 16

            # 10. logits_indices_selector: Array to map back to original request order
            assert isinstance(logits_indices_selector, np.ndarray)
            assert len(logits_indices_selector) == 4  # One for each request
            # Should map distributed positions back to original order
            expected_selector = np.array([0, 1, 16, 17])
            np.testing.assert_array_equal(logits_indices_selector,
                                          expected_selector)

            # 11. max_num_reqs_per_dp_rank: Maximum requests per DP rank
            assert max_num_reqs_per_dp_rank == 4  # max_num_reqs (8) // dp_size (2)

    def test_prepare_dp_input_metadata_empty_rank(self):
        """Test metadata preparation with one empty DP rank"""
        # Create test data where all requests go to rank 0, leaving rank 1 empty
        num_scheduled_tokens = {"req1": 10, "req2": 5}
        assigned_dp_ranks = {"req1": 0, "req2": 0}

        self.runner.input_batch.num_reqs = 2
        self.runner.input_batch.req_ids = ["req1", "req2"]
        self.runner.max_num_reqs = 8

        scheduler_output = self._create_mock_scheduler_output(
            num_scheduled_tokens, assigned_dp_ranks)

        with patch('tpu_inference.runner.tpu_runner.runner_utils'
                   ) as mock_runner_utils:
            mock_runner_utils.get_padded_token_len.side_effect = lambda paddings_list, val: 16 if val <= 15 else 32

            result = self.runner._prepare_dp_input_metadata(scheduler_output)

            (req_ids_dp, req_indices_dp, num_scheduled_tokens_per_dp_rank,
             scheduled_tokens_per_dp_rank, num_req_per_dp_rank,
             padded_num_scheduled_tokens_per_dp_rank, padded_num_reqs,
             padded_total_num_scheduled_tokens, padded_num_reqs_per_dp_rank,
             logits_indices_selector, max_num_reqs_per_dp_rank) = result

            # 1. req_ids_dp
            assert isinstance(req_ids_dp, dict)
            assert req_ids_dp[0] == ["req1", "req2"]
            assert req_ids_dp[1] == []  # Empty rank

            # 2. req_indices_dp
            assert isinstance(req_indices_dp, dict)
            assert req_indices_dp[0] == [0, 1]  # req1, req2 indices
            assert req_indices_dp[1] == []  # Empty rank

            # 3. num_scheduled_tokens_per_dp_rank
            assert isinstance(num_scheduled_tokens_per_dp_rank, dict)
            assert num_scheduled_tokens_per_dp_rank[0] == 15  # 10 + 5
            assert num_scheduled_tokens_per_dp_rank[1] == 0  # Empty rank

            # 4. scheduled_tokens_per_dp_rank
            assert isinstance(scheduled_tokens_per_dp_rank, dict)
            assert scheduled_tokens_per_dp_rank[0] == [10,
                                                       5]  # req1=10, req2=5
            assert scheduled_tokens_per_dp_rank[1] == []  # Empty rank

            # 5. num_req_per_dp_rank
            assert isinstance(num_req_per_dp_rank, dict)
            assert num_req_per_dp_rank[0] == 2  # Both requests on rank 0
            assert num_req_per_dp_rank[1] == 0  # No requests on rank 1

            # 6. padded_num_scheduled_tokens_per_dp_rank
            assert padded_num_scheduled_tokens_per_dp_rank == 16

            # 7. padded_num_reqs
            assert padded_num_reqs == 32  # 2 DP ranks * 16 padded reqs per rank

            # 8. padded_total_num_scheduled_tokens
            assert padded_total_num_scheduled_tokens == 32  # 2 DP ranks * 16 padded tokens per rank

            # 10. padded_num_reqs_per_dp_rank: Padded requests per DP rank
            assert padded_num_reqs_per_dp_rank == 16

            # 11. logits_indices_selector: Should preserve original order since no reordering needed
            assert isinstance(logits_indices_selector, np.ndarray)
            assert len(logits_indices_selector) == 2
            # Both requests on DP rank 0, positions 0 and 1
            expected_selector = np.array([0, 1])
            np.testing.assert_array_equal(logits_indices_selector,
                                          expected_selector)

            # 12. max_num_reqs_per_dp_rank: Maximum requests per DP rank
            assert max_num_reqs_per_dp_rank == 4  # max_num_reqs (8) // dp_size (2)

    def test_prepare_dp_input_metadata_logits_indices_selector_ordering(self):
        """Test logits_indices_selector with mixed DP rank assignment."""
        # Create requests with mixed assignment to test reordering
        num_scheduled_tokens = {"req1": 4, "req2": 6, "req3": 2}
        assigned_dp_ranks = {
            "req1": 1,
            "req2": 0,
            "req3": 1
        }  # req2 on rank 0, req1&req3 on rank 1

        self.runner.input_batch.num_reqs = 3
        self.runner.input_batch.req_ids = ["req1", "req2", "req3"]

        scheduler_output = self._create_mock_scheduler_output(
            num_scheduled_tokens, assigned_dp_ranks)

        with patch('tpu_inference.runner.tpu_runner.runner_utils'
                   ) as mock_runner_utils:
            mock_runner_utils.get_padded_token_len.side_effect = lambda paddings_list, val: 8 if val <= 6 else 16

            result = self.runner._prepare_dp_input_metadata(scheduler_output)

            (req_ids_dp, req_indices_dp, _, _, _, _, _, _, _,
             logits_indices_selector, _) = result

            # Verify request distribution
            assert req_ids_dp[0] == ["req2"]  # rank 0: req2 (index 1)
            assert req_ids_dp[1] == [
                "req1", "req3"
            ]  # rank 1: req1 (index 0), req3 (index 2)

            assert req_indices_dp[0] == [1]  # req2 has original index 1
            assert req_indices_dp[1] == [
                0, 2
            ]  # req1 has index 0, req3 has index 2

            # The logits_indices_selector should map the DP-distributed positions back to original order

            assert isinstance(logits_indices_selector, np.ndarray)
            assert len(logits_indices_selector) == 3

            expected_positions = np.array([8, 0, 9])
            np.testing.assert_array_equal(logits_indices_selector,
                                          expected_positions)

    @patch('tpu_inference.runner.tpu_runner.NamedSharding')
    @patch('tpu_inference.runner.tpu_runner.runner_utils')
    @patch('tpu_inference.runner.tpu_runner.device_array',
           side_effect=lambda mesh, tensors, **kwargs: tensors)
    @patch('tpu_inference.runner.tpu_runner.TPUSupportedSamplingMetadata')
    def test_prepare_inputs_dp_verify_content_balanced(self,
                                                       mock_sampling_metadata,
                                                       mock_device_array,
                                                       mock_runner_utils,
                                                       mock_named_sharding):
        """Test _prepare_inputs_dp with content verification for balanced distribution."""

        # Setup mocking with specific behavior for tokens vs requests
        def mock_get_padded_token_len(paddings_list, val):
            # For tokens: 8 if val <= 3 else 16
            # For requests: 4 if val <= 1 else 8
            if val <= 1:
                return 4  # For request padding
            elif val <= 3:
                return 8  # For token padding
            else:
                return 16

        mock_runner_utils.get_padded_token_len.side_effect = mock_get_padded_token_len
        mock_sampling_instance = MagicMock()
        mock_sampling_metadata.from_input_batch.return_value = mock_sampling_instance
        mock_named_sharding.return_value = MagicMock()

        # Setup deterministic test data
        num_scheduled_tokens = {"req1": 2, "req2": 3}
        assigned_dp_ranks = {"req1": 0, "req2": 1}

        self.runner.input_batch.num_reqs = 2
        self.runner.input_batch.req_ids = ["req1", "req2"]
        self.runner.input_batch.num_computed_tokens_cpu = np.array(
            [5, 6])  # Starting positions

        # Setup known token sequences for verification
        self.runner.input_batch.token_ids_cpu = np.zeros((8, 64),
                                                         dtype=np.int32)
        # req1: [1001, 1002, 1003, ...]
        # req2: [2001, 2002, 2003, ...]
        for i in range(2):
            start_val = (i + 1) * 1000 + 1
            for j in range(64):
                self.runner.input_batch.token_ids_cpu[i, j] = start_val + j

        scheduler_output = self._create_mock_scheduler_output(
            num_scheduled_tokens, assigned_dp_ranks)

        # Setup additional required attributes
        self.runner.uses_mrope = False
        self.runner.phase_based_profiler = None
        self.runner.lora_config = None
        self.runner.mesh = MagicMock()
        self.runner.data_parallel_sharding = MagicMock()
        self.runner.data_parallel_attn_sharding = MagicMock()
        self.runner.mm_manager = MagicMock()
        self.runner.speculative_decoding_manager = MagicMock()
        self.runner.lora_utils = MagicMock()
        # self.runner.mrope_positions_cpu = np.zeros((3, 64), dtype=np.int64)

        # Execute the method
        result = self.runner._prepare_inputs_dp(scheduler_output)
        input_ids, positions, attention_metadata, sampling_metadata, logits_indices, spec_decode_metadata, logits_indices_selector, padded_num_reqs = result
        # 1. Verify input_ids content
        expected_input_ids = np.zeros(16, dtype=np.int32)
        expected_input_ids[:2] = [1006, 1007]
        expected_input_ids[8:11] = [2007, 2008, 2009]
        assert np.array_equal(input_ids, expected_input_ids)

        # 2. Verify attention_metadata positions content
        expected_positions = np.zeros(16, dtype=np.int32)
        expected_positions[:2] = [5, 6]  # req1 positions
        expected_positions[8:11] = [6, 7, 8]
        assert np.array_equal(attention_metadata.input_positions,
                              expected_positions)

        # 3. Verify query_start_loc content
        query_start_loc = attention_metadata.query_start_loc_cpu
        max_num_reqs_per_dp = self.runner.max_num_reqs // 2
        expected_query_start = np.zeros(self.runner.max_num_reqs + 2,
                                        dtype=np.int32)
        # DP rank 0: cumsum([2]) = [2] at positions [1:2] → [0, 2, 1, 1, 1]
        expected_query_start[1] = 2  # req1 has 2 tokens
        expected_query_start[2:max_num_reqs_per_dp + 1] = 1
        # DP rank 1: cumsum([3]) = [3] at positions [6:7] → [0, 3, 1, 1, 1]
        expected_query_start[max_num_reqs_per_dp + 2] = 3  # req2 has 3 tokens
        expected_query_start[max_num_reqs_per_dp + 3:] = 1
        assert np.array_equal(query_start_loc, expected_query_start)

        # 4. Verify seq_lens content
        seq_lens = attention_metadata.seq_lens_cpu
        # Should be computed_tokens + scheduled_tokens for each request
        # DP rank 0: req1 at position 0, DP rank 1: req2 at position 4
        expected_seq_lens = np.array([7, 0, 0, 0, 9, 0, 0,
                                      0])  # req1: 5+2=7, req2: 6+3=9
        assert np.array_equal(seq_lens, expected_seq_lens)

        # 5. Verify request_distribution content
        expected_distribution = np.array([[0, 0, 1], [0, 0, 1]]).flatten()
        np.testing.assert_array_equal(attention_metadata.request_distribution,
                                      expected_distribution)

        # 6. Verify logits_indices content
        assert len(logits_indices) == 8  # padded_num_reqs
        expected_logits = np.full(8, -1, dtype=np.int32)
        expected_logits[0] = 1  # req1 last token position (2-1)
        expected_logits[
            4] = 2  # req2 last token position (3-1) at DP rank 1 offset (4*1)
        assert np.array_equal(logits_indices, expected_logits)

        # 7. Verify logits_indices_selector
        assert len(logits_indices_selector) == 2
        assert np.array_equal(logits_indices_selector, np.array([0, 4]))

    @patch('tpu_inference.runner.tpu_runner.NamedSharding')
    @patch('tpu_inference.runner.tpu_runner.runner_utils')
    @patch('tpu_inference.runner.tpu_runner.device_array',
           side_effect=lambda mesh, tensors, **kwargs: tensors)
    @patch('tpu_inference.runner.tpu_runner.TPUSupportedSamplingMetadata')
    def test_prepare_inputs_dp_verify_content_empty_rank(
            self, mock_sampling_metadata, mock_device_array, mock_runner_utils,
            mock_named_sharding):
        """Test _prepare_inputs_dp with detailed content verification for empty rank case."""

        # Setup mocking
        def mock_get_padded_token_len(paddings_list, val):
            if val <= 2:
                return 4  # For request padding (max 2 requests)
            elif val <= 5:
                return 8  # For token padding
            else:
                return 16

        mock_runner_utils.get_padded_token_len.side_effect = mock_get_padded_token_len
        mock_sampling_instance = MagicMock()
        mock_sampling_metadata.from_input_batch.return_value = mock_sampling_instance
        mock_named_sharding.return_value = MagicMock()

        # Setup test data with all requests on rank 0 (empty rank 1)
        num_scheduled_tokens = {"req1": 3, "req2": 2}
        assigned_dp_ranks = {
            "req1": 0,
            "req2": 0
        }  # Both on rank 0, rank 1 empty

        self.runner.input_batch.num_reqs = 2
        self.runner.input_batch.req_ids = ["req1", "req2"]
        self.runner.input_batch.num_computed_tokens_cpu = np.array(
            [4, 6])  # Starting positions

        # Setup deterministic token sequences for verification
        self.runner.input_batch.token_ids_cpu = np.zeros((8, 64),
                                                         dtype=np.int32)
        # req1: [5001, 5002, 5003, ...] starting at position 4
        # req2: [6001, 6002, 6003, ...] starting at position 6
        for i in range(2):
            start_val = (i + 5) * 1000 + 1  # 5001, 6001
            for j in range(64):
                self.runner.input_batch.token_ids_cpu[i, j] = start_val + j

        scheduler_output = self._create_mock_scheduler_output(
            num_scheduled_tokens, assigned_dp_ranks)

        # Setup required attributes
        self.runner.uses_mrope = False
        self.runner.phase_based_profiler = None
        self.runner.lora_config = None
        self.runner.mesh = MagicMock()
        self.runner.data_parallel_sharding = MagicMock()
        self.runner.data_parallel_attn_sharding = MagicMock()
        self.runner.mm_manager = MagicMock()
        self.runner.speculative_decoding_manager = MagicMock()
        self.runner.lora_utils = MagicMock()

        # Execute the method
        result = self.runner._prepare_inputs_dp(scheduler_output)
        input_ids, positions, attention_metadata, sampling_metadata, logits_indices, spec_decode_metadata, logits_indices_selector, padded_num_reqs = result

        # 1. Verify input_ids
        expected_input_ids = np.zeros(16, dtype=np.int32)
        # Rank 0
        expected_input_ids[:5] = [5005, 5006, 5007, 6007, 6008]
        # Rank 1 (positions 8-15) should remain zeros
        assert np.array_equal(input_ids, expected_input_ids)

        # 2. Verify attention_metadata
        expected_positions = np.zeros(16, dtype=np.int32)
        expected_positions[:3] = [4, 5, 6]  # req1 positions: 4 + [0, 1, 2]
        expected_positions[3:5] = [6, 7]  # req2 positions: 6 + [0, 1]
        # Rank 1 positions (8-15) remain zeros
        assert np.array_equal(attention_metadata.input_positions,
                              expected_positions)

        # 3. Verify query_start_loc
        query_start_loc = attention_metadata.query_start_loc_cpu
        max_num_reqs_per_dp = self.runner.max_num_reqs // 2  # 4
        expected_query_start = np.zeros(self.runner.max_num_reqs + 2,
                                        dtype=np.int32)
        # Rank 0: req1 (3 tokens), req2 (2 tokens)
        expected_query_start[1] = 3  # req1 has 3 tokens
        expected_query_start[2] = 5  # cumulative: 3 + 2 = 5
        expected_query_start[3:max_num_reqs_per_dp + 1] = 1  # padding
        # Rank 1: empty (all zeros)
        expected_query_start[max_num_reqs_per_dp +
                             1:] = 0  # Empty rank sets to 0
        assert np.array_equal(query_start_loc, expected_query_start)

        # 4. Verify seq_lens
        seq_lens = attention_metadata.seq_lens_cpu
        expected_seq_lens = np.zeros(8, dtype=np.int32)
        # Rank 0: req1 (4+3=7), req2 (6+2=8), then padding
        expected_seq_lens[
            0] = 7  # req1: computed_tokens(4) + scheduled_tokens(3)
        expected_seq_lens[
            1] = 8  # req2: computed_tokens(6) + scheduled_tokens(2)
        # Rank 1: all zeros
        assert np.array_equal(seq_lens, expected_seq_lens)

        # 5. Verify request_distribution
        expected_distribution = np.array([[0, 0, 2], [0, 0, 0]]).flatten()
        np.testing.assert_array_equal(attention_metadata.request_distribution,
                                      expected_distribution)

        # 6. Verify logits_indices
        assert len(
            logits_indices) == 8  # padded_num_reqs (8 in this case, not 16)
        # Rank 0: req1 ends at pos 2, req2 ends at pos 4
        # Rank 1: empty, so -1 padding
        expected_logits = np.full(8, -1, dtype=np.int32)
        expected_logits[0] = 2  # req1 ends at position 2 (3-1)
        expected_logits[1] = 4  # req2 ends at position 4 (5-1)
        assert np.array_equal(logits_indices, expected_logits)

        # 7. Verify logits_indices_selector
        assert len(logits_indices_selector) == 2
        expected_selector = np.array([0, 1])
        np.testing.assert_array_equal(logits_indices_selector,
                                      expected_selector)

    @patch('tpu_inference.runner.tpu_runner.NamedSharding')
    @patch('tpu_inference.runner.tpu_runner.runner_utils')
    @patch('tpu_inference.runner.tpu_runner.device_array',
           side_effect=lambda mesh, tensors, **kwargs: tensors)
    @patch('tpu_inference.runner.tpu_runner.TPUSupportedSamplingMetadata')
    def test_prepare_inputs_dp_with_decode_requests(self,
                                                    mock_sampling_metadata,
                                                    mock_device_array,
                                                    mock_runner_utils,
                                                    mock_named_sharding):
        """Test _prepare_inputs_dp with decode requests (1 token each) to verify request_distribution."""

        # Setup mocking
        def mock_get_padded_token_len(paddings_list, val):
            if val <= 2:
                return 4  # For request padding
            elif val <= 4:
                return 8  # For token padding
            else:
                return 16

        mock_runner_utils.get_padded_token_len.side_effect = mock_get_padded_token_len
        mock_sampling_instance = MagicMock()
        mock_sampling_metadata.from_input_batch.return_value = mock_sampling_instance
        mock_named_sharding.return_value = MagicMock()

        # Setup test data with decode requests (1 token) and prefill requests (>1 token)
        # req1: decode (1 token), req2: decode (1 token), req3: prefill (3 tokens), req4: decode (1 token)
        num_scheduled_tokens = {"req1": 1, "req2": 1, "req3": 3, "req4": 1}
        assigned_dp_ranks = {"req1": 0, "req2": 0, "req3": 1, "req4": 1}

        self.runner.input_batch.num_reqs = 4
        self.runner.input_batch.req_ids = ["req1", "req2", "req3", "req4"]
        self.runner.input_batch.num_computed_tokens_cpu = np.array(
            [5, 6, 7, 8])
        self.runner.input_batch.token_ids_cpu = np.zeros((8, 64),
                                                         dtype=np.int32)

        scheduler_output = self._create_mock_scheduler_output(
            num_scheduled_tokens, assigned_dp_ranks)

        # Setup required attributes
        self.runner.uses_mrope = False
        self.runner.phase_based_profiler = None
        self.runner.lora_config = None
        self.runner.mesh = MagicMock()
        self.runner.data_parallel_sharding = MagicMock()
        self.runner.data_parallel_attn_sharding = MagicMock()
        self.runner.mm_manager = MagicMock()
        self.runner.speculative_decoding_manager = MagicMock()
        self.runner.lora_utils = MagicMock()

        # Execute the method
        result = self.runner._prepare_inputs_dp(scheduler_output)
        input_ids, positions, attention_metadata, sampling_metadata, logits_indices, spec_decode_metadata, logits_indices_selector, padded_num_reqs = result

        # Verify request_distribution
        # DP rank 0: req1 (decode), req2 (decode) -> [2, 2, 2]
        # DP rank 1: req3 (prefill), req4 (decode) -> [1, 1, 2]
        expected_distribution = np.array([[2, 2, 2], [1, 1, 2]]).flatten()
        np.testing.assert_array_equal(attention_metadata.request_distribution,
                                      expected_distribution)

    @patch('tpu_inference.runner.tpu_runner.NamedSharding')
    @patch('tpu_inference.runner.tpu_runner.runner_utils')
    @patch('tpu_inference.runner.tpu_runner.device_array',
           side_effect=lambda mesh, tensors, **kwargs: tensors)
    @patch('tpu_inference.runner.tpu_runner.TPUSupportedSamplingMetadata')
    def test_prepare_inputs_dp_all_decode_requests(self,
                                                   mock_sampling_metadata,
                                                   mock_device_array,
                                                   mock_runner_utils,
                                                   mock_named_sharding):
        """Test _prepare_inputs_dp with all decode requests."""

        # Setup mocking
        def mock_get_padded_token_len(paddings_list, val):
            if val <= 2:
                return 4
            elif val <= 4:
                return 8
            else:
                return 16

        mock_runner_utils.get_padded_token_len.side_effect = mock_get_padded_token_len
        mock_sampling_instance = MagicMock()
        mock_sampling_metadata.from_input_batch.return_value = mock_sampling_instance
        mock_named_sharding.return_value = MagicMock()

        # All requests are decode (1 token each)
        num_scheduled_tokens = {"req1": 1, "req2": 1}
        assigned_dp_ranks = {"req1": 0, "req2": 1}

        self.runner.input_batch.num_reqs = 2
        self.runner.input_batch.req_ids = ["req1", "req2"]
        self.runner.input_batch.num_computed_tokens_cpu = np.array([5, 6])
        self.runner.input_batch.token_ids_cpu = np.zeros((8, 64),
                                                         dtype=np.int32)

        scheduler_output = self._create_mock_scheduler_output(
            num_scheduled_tokens, assigned_dp_ranks)

        # Setup required attributes
        self.runner.uses_mrope = False
        self.runner.phase_based_profiler = None
        self.runner.lora_config = None
        self.runner.mesh = MagicMock()
        self.runner.data_parallel_sharding = MagicMock()
        self.runner.data_parallel_attn_sharding = MagicMock()
        self.runner.mm_manager = MagicMock()
        self.runner.speculative_decoding_manager = MagicMock()
        self.runner.lora_utils = MagicMock()

        # Execute the method
        result = self.runner._prepare_inputs_dp(scheduler_output)
        input_ids, positions, attention_metadata, sampling_metadata, logits_indices, spec_decode_metadata, logits_indices_selector, padded_num_reqs = result

        # Verify request_distribution
        # Both ranks have only decode requests
        # DP rank 0: req1 (decode) -> [1, 1, 1]
        # DP rank 1: req2 (decode) -> [1, 1, 1]
        expected_distribution = np.array([[1, 1, 1], [1, 1, 1]]).flatten()
        np.testing.assert_array_equal(attention_metadata.request_distribution,
                                      expected_distribution)

    @patch('tpu_inference.runner.tpu_runner.NamedSharding')
    @patch('tpu_inference.runner.tpu_runner.runner_utils')
    @patch('tpu_inference.runner.tpu_runner.device_array',
           side_effect=lambda mesh, tensors, **kwargs: tensors)
    @patch('tpu_inference.runner.tpu_runner.TPUSupportedSamplingMetadata')
    def test_prepare_async_token_substitution_indices_dp(
            self, mock_sampling_metadata, mock_device_array, mock_runner_utils,
            mock_named_sharding):

        # Setup test data
        req_ids_dp = {0: ["req1", "req2"], 1: ["req3"]}
        scheduled_tokens_per_dp_rank = {0: [3, 2], 1: [4]}
        padded_num_scheduled_tokens_per_dp_rank = 8
        dp_size = 2

        # Setup _pre_async_results with placeholder mapping
        self.runner._pre_async_results = MagicMock()
        self.runner._pre_async_results.placeholder_req_id_to_index = {
            "req1": 0,
            "req3": 2
        }  # req2 is not a placeholder

        # Call the method
        result = self.runner._prepare_async_token_substitution_indices_dp(
            req_ids_dp, scheduled_tokens_per_dp_rank,
            padded_num_scheduled_tokens_per_dp_rank, dp_size)

        token_in_tpu_cur_input_indices_dp, token_in_tpu_pre_next_tokens_indices_dp = result

        # Verify DP rank 0
        # req1: token_offset=0, acc_cur_len starts at 0, after 3 tokens: 3, so last token at 2
        # req2: not a placeholder, should be skipped
        assert token_in_tpu_cur_input_indices_dp[0] == [2]
        assert token_in_tpu_pre_next_tokens_indices_dp[0] == [0]

        # Verify DP rank 1
        # req3: token_offset=8, acc_cur_len starts at 8, after 4 tokens: 12, so last token at 11
        assert token_in_tpu_cur_input_indices_dp[1] == [11]
        assert token_in_tpu_pre_next_tokens_indices_dp[1] == [2]

    @patch('tpu_inference.runner.tpu_runner.NamedSharding')
    @patch('tpu_inference.runner.tpu_runner.runner_utils')
    @patch('tpu_inference.runner.tpu_runner.device_array',
           side_effect=lambda mesh, tensors, **kwargs: tensors)
    @patch('tpu_inference.runner.tpu_runner.TPUSupportedSamplingMetadata')
    def test_prepare_async_token_substitution_indices_dp_no_placeholders(
            self, mock_sampling_metadata, mock_device_array, mock_runner_utils,
            mock_named_sharding):
        """Test when no requests are placeholders."""

        req_ids_dp = {0: ["req1", "req2"], 1: ["req3"]}
        scheduled_tokens_per_dp_rank = {0: [3, 2], 1: [4]}
        padded_num_scheduled_tokens_per_dp_rank = 8
        dp_size = 2

        # No placeholders
        self.runner._pre_async_results = MagicMock()
        self.runner._pre_async_results.placeholder_req_id_to_index = {}

        result = self.runner._prepare_async_token_substitution_indices_dp(
            req_ids_dp, scheduled_tokens_per_dp_rank,
            padded_num_scheduled_tokens_per_dp_rank, dp_size)

        token_in_tpu_cur_input_indices_dp, token_in_tpu_pre_next_tokens_indices_dp = result

        # All lists should be empty since no placeholders
        assert token_in_tpu_cur_input_indices_dp[0] == []
        assert token_in_tpu_pre_next_tokens_indices_dp[0] == []
        assert token_in_tpu_cur_input_indices_dp[1] == []
        assert token_in_tpu_pre_next_tokens_indices_dp[1] == []

    def test_apply_async_token_substitution_empty_indices(self):
        """Test _apply_async_token_substitution with empty indices (line 1025)."""

        # Bind the actual method
        self.runner._apply_async_token_substitution = TPUModelRunner._apply_async_token_substitution.__get__(
            self.runner)

        input_ids = np.array([1, 2, 3, 4, 5])
        token_in_tpu_cur_input_indices = np.array([])
        token_in_tpu_pre_next_tokens_indices = np.array([])

        # Setup _pre_async_results
        self.runner._pre_async_results = MagicMock()
        self.runner._pre_async_results.next_tokens = np.array([10, 20, 30])
        self.runner.mesh = MagicMock()

        result = self.runner._apply_async_token_substitution(
            input_ids, token_in_tpu_cur_input_indices,
            token_in_tpu_pre_next_tokens_indices)

        # Should return input_ids unchanged
        np.testing.assert_array_equal(result, input_ids)

    @patch('tpu_inference.runner.tpu_runner.device_array',
           side_effect=lambda mesh, tensors, **kwargs: tensors)
    def test_apply_async_token_substitution_with_padding(
            self, mock_device_array):
        """Test _apply_async_token_substitution with padding."""

        # Bind the actual method
        self.runner._apply_async_token_substitution = TPUModelRunner._apply_async_token_substitution.__get__(
            self.runner)

        input_ids = np.array([1, 2, 3, 4, 5, 6, 7, 8])
        # Substitute positions 2 and 5
        token_in_tpu_cur_input_indices = np.array([2, 5])
        token_in_tpu_pre_next_tokens_indices = np.array([0, 1])

        # Setup _pre_async_results
        self.runner._pre_async_results = MagicMock()
        self.runner._pre_async_results.next_tokens = np.array([100, 200, 300])
        self.runner.mesh = MagicMock()
        self.runner.maybe_forbid_compile = nullcontext()

        # Mock the substitute function to verify it's called correctly
        mock_substitute_fn = MagicMock(
            return_value=np.array([1, 2, 100, 4, 5, 200, 7, 8]))
        self.runner._substitute_placeholder_token_fn = mock_substitute_fn

        _ = self.runner._apply_async_token_substitution(
            input_ids, token_in_tpu_cur_input_indices,
            token_in_tpu_pre_next_tokens_indices)

        # Verify the substitute function was called
        mock_substitute_fn.assert_called_once()
        call_args = mock_substitute_fn.call_args[0]

        # Verify input_ids
        np.testing.assert_array_equal(call_args[0], input_ids)

        # Verify padded indices length matches input_ids length
        assert len(call_args[1]) == len(input_ids)
        assert len(call_args[2]) == len(input_ids)

        # Verify placeholder_num
        assert call_args[4] == 2  # Number of actual substitutions

    def test_prepare_inputs_routing_to_dp(self):
        """Test _prepare_inputs routes to _prepare_inputs_dp when dp_size > 1."""

        # Bind the actual _prepare_inputs method
        self.runner._prepare_inputs = TPUModelRunner._prepare_inputs.__get__(
            self.runner)

        self.runner.dp_size = 2
        self.runner._prepare_inputs_dp = MagicMock(return_value=(None, None,
                                                                 None, None,
                                                                 None, None))

        scheduler_output = MagicMock()
        self.runner._prepare_inputs(scheduler_output)

        # Verify _prepare_inputs_dp was called
        self.runner._prepare_inputs_dp.assert_called_once_with(
            scheduler_output)

    def test_prepare_inputs_routing_to_non_dp(self):
        """Test _prepare_inputs routes to _prepare_inputs_non_dp when dp_size == 1."""

        # Bind the actual _prepare_inputs method
        self.runner._prepare_inputs = TPUModelRunner._prepare_inputs.__get__(
            self.runner)

        self.runner.dp_size = 1
        self.runner._prepare_inputs_non_dp = MagicMock(
            return_value=(None, None, None, None, None, None, None))

        scheduler_output = MagicMock()
        self.runner._prepare_inputs(scheduler_output)

        # Verify _prepare_inputs_non_dp was called
        self.runner._prepare_inputs_non_dp.assert_called_once_with(
            scheduler_output)

    @patch('tpu_inference.runner.tpu_runner.NamedSharding')
    @patch('tpu_inference.runner.tpu_runner.runner_utils')
    @patch('tpu_inference.runner.tpu_runner.device_array',
           side_effect=lambda mesh, tensors, **kwargs: tensors)
    @patch('tpu_inference.runner.tpu_runner.TPUSupportedSamplingMetadata')
    def test_prepare_inputs_dp_with_async_scheduling(self,
                                                     mock_sampling_metadata,
                                                     mock_device_array,
                                                     mock_runner_utils,
                                                     mock_named_sharding):

        # Setup mocking
        def mock_get_padded_token_len(paddings_list, val):
            if val <= 2:
                return 4
            elif val <= 5:
                return 8
            else:
                return 16

        mock_runner_utils.get_padded_token_len.side_effect = mock_get_padded_token_len
        mock_sampling_instance = MagicMock()
        mock_sampling_metadata.from_input_batch.return_value = mock_sampling_instance
        mock_named_sharding.return_value = MagicMock()

        # Setup test data
        num_scheduled_tokens = {"req1": 3, "req2": 2}
        assigned_dp_ranks = {"req1": 0, "req2": 1}

        self.runner.input_batch.num_reqs = 2
        self.runner.input_batch.req_ids = ["req1", "req2"]
        self.runner.input_batch.num_computed_tokens_cpu = np.array([4, 6])
        self.runner.input_batch.token_ids_cpu = np.zeros((8, 64),
                                                         dtype=np.int32)

        scheduler_output = self._create_mock_scheduler_output(
            num_scheduled_tokens, assigned_dp_ranks)

        # Enable async scheduling
        self.runner.scheduler_config.async_scheduling = True
        self.runner._pre_async_results = MagicMock()
        self.runner._pre_async_results.placeholder_req_id_to_index = {
            "req1": 0
        }
        self.runner._pre_async_results.next_tokens = np.array([100])

        # Setup required attributes
        self.runner.uses_mrope = False
        self.runner.phase_based_profiler = None
        self.runner.lora_config = None
        self.runner.mesh = MagicMock()
        self.runner.data_parallel_sharding = MagicMock()
        self.runner.data_parallel_attn_sharding = MagicMock()
        self.runner.mm_manager = MagicMock()
        self.runner.speculative_decoding_manager = MagicMock()
        self.runner.lora_utils = MagicMock()

        # Mock the token substitution preparation
        mock_prepare_async = MagicMock(return_value=({
            0: [2],
            1: []
        }, {
            0: [0],
            1: []
        }))
        self.runner._prepare_async_token_substitution_indices_dp = mock_prepare_async

        # Execute the method
        _ = self.runner._prepare_inputs_dp(scheduler_output)

        # Verify async token substitution was called
        mock_prepare_async.assert_called_once()

    @patch('tpu_inference.runner.tpu_runner.NamedSharding')
    @patch('tpu_inference.runner.tpu_runner.runner_utils')
    @patch('tpu_inference.runner.tpu_runner.device_array',
           side_effect=lambda mesh, tensors, **kwargs: tensors)
    @patch('tpu_inference.runner.tpu_runner.TPUSupportedSamplingMetadata')
    def test_prepare_inputs_dp_async_token_substitution_application(
            self, mock_sampling_metadata, mock_device_array, mock_runner_utils,
            mock_named_sharding):
        """Test async token substitution application in DP mode."""

        # Setup mocking
        def mock_get_padded_token_len(paddings_list, val):
            if val <= 2:
                return 4
            elif val <= 5:
                return 8
            else:
                return 16

        mock_runner_utils.get_padded_token_len.side_effect = mock_get_padded_token_len
        mock_sampling_instance = MagicMock()
        mock_sampling_metadata.from_input_batch.return_value = mock_sampling_instance
        mock_named_sharding.return_value = MagicMock()

        # Setup test data
        num_scheduled_tokens = {"req1": 3, "req2": 2}
        assigned_dp_ranks = {"req1": 0, "req2": 1}

        self.runner.input_batch.num_reqs = 2
        self.runner.input_batch.req_ids = ["req1", "req2"]
        self.runner.input_batch.num_computed_tokens_cpu = np.array([4, 6])
        self.runner.input_batch.token_ids_cpu = np.zeros((8, 64),
                                                         dtype=np.int32)

        scheduler_output = self._create_mock_scheduler_output(
            num_scheduled_tokens, assigned_dp_ranks)

        # Enable async scheduling with placeholders
        self.runner.scheduler_config.async_scheduling = True
        self.runner._pre_async_results = MagicMock()
        self.runner._pre_async_results.placeholder_req_id_to_index = {
            "req1": 0,
            "req2": 1
        }
        self.runner._pre_async_results.next_tokens = np.array([100, 200])

        # Setup required attributes
        self.runner.uses_mrope = False
        self.runner.phase_based_profiler = None
        self.runner.lora_config = None
        self.runner.mesh = MagicMock()
        self.runner.data_parallel_sharding = MagicMock()
        self.runner.data_parallel_attn_sharding = MagicMock()
        self.runner.mm_manager = MagicMock()
        self.runner.speculative_decoding_manager = MagicMock()
        self.runner.lora_utils = MagicMock()

        # Mock the async token substitution application
        mock_apply_async = MagicMock(
            return_value=np.array([1, 2, 100, 4, 5, 200, 7, 8]))
        self.runner._apply_async_token_substitution = mock_apply_async

        # Execute the method
        _ = self.runner._prepare_inputs_dp(scheduler_output)

        # Verify _apply_async_token_substitution was called
        mock_apply_async.assert_called_once()
        call_args = mock_apply_async.call_args[0]

        # Verify indices were concatenated from both DP ranks
        token_in_tpu_cur_input_indices = call_args[1]
        token_in_tpu_pre_next_tokens_indices = call_args[2]

        # Should have indices from both ranks
        assert len(token_in_tpu_cur_input_indices) == 2
        assert len(token_in_tpu_pre_next_tokens_indices) == 2


if __name__ == "__main__":
    pytest.main([__file__])
