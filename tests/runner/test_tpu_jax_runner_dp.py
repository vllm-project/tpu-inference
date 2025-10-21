from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from tpu_inference.runner.tpu_jax_runner import TPUModelRunner


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
        self.runner.block_table_cpu = np.zeros((8, 8), dtype=np.int32)
        self.runner.arange_cpu = np.arange(64, dtype=np.int64)

        # Bind the actual methods to our mock
        self.runner._prepare_inputs_dp = TPUModelRunner._prepare_inputs_dp.__get__(
            self.runner)
        self.runner._prepare_dp_input_metadata = TPUModelRunner._prepare_dp_input_metadata.__get__(
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

    @patch('tpu_inference.runner.tpu_jax_runner.NamedSharding')
    @patch('tpu_inference.runner.tpu_jax_runner.runner_utils')
    @patch('tpu_inference.runner.tpu_jax_runner.device_array',
           side_effect=lambda mesh, tensors, **kwargs: tensors)
    @patch('tpu_inference.runner.tpu_jax_runner.TPUSupportedSamplingMetadata')
    def test_prepare_inputs_dp_basic_functionality(self,
                                                   mock_sampling_metadata,
                                                   mock_device_array,
                                                   mock_runner_utils,
                                                   mock_named_sharding):
        """Test basic functionality of _prepare_inputs_dp."""
        # Mock utility functions
        mock_runner_utils.get_padded_token_len.return_value = 16
        mock_runner_utils.get_padded_num_reqs_with_upper_limit.return_value = 2
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
        assert len(result) == 6
        input_ids, attention_metadata, sampling_metadata, logits_indices, spec_decode_metadata, logits_indices_selector = result

        # Verify utility functions were called
        mock_runner_utils.get_padded_token_len.assert_called()
        mock_runner_utils.get_padded_num_reqs_with_upper_limit.assert_called()

    def test_prepare_inputs_dp_error_conditions(self):
        """Test error handling in DP input preparation."""
        # Test with zero scheduled tokens
        scheduler_output = self._create_mock_scheduler_output({}, {})
        scheduler_output.total_num_scheduled_tokens = 0

        with pytest.raises(AssertionError):
            self.runner._prepare_inputs_dp(scheduler_output)

        # Test with zero requests
        self.runner.input_batch.num_reqs = 0
        scheduler_output = self._create_mock_scheduler_output({"req1": 5},
                                                              {"req1": 0})

        with pytest.raises(AssertionError):
            self.runner._prepare_inputs_dp(scheduler_output)

    def test_prepare_dp_input_metadata(self):
        num_scheduled_tokens = {"req1": 10, "req2": 5, "req3": 8, "req4": 3}
        assigned_dp_ranks = {"req1": 0, "req2": 0, "req3": 1, "req4": 1}

        self.runner.input_batch.num_reqs = 4
        self.runner.input_batch.req_ids = ["req1", "req2", "req3", "req4"]
        self.runner.max_num_reqs = 8

        scheduler_output = self._create_mock_scheduler_output(
            num_scheduled_tokens, assigned_dp_ranks)

        with patch('tpu_inference.runner.tpu_jax_runner.runner_utils'
                   ) as mock_runner_utils:
            mock_runner_utils.get_padded_token_len.return_value = 16  # Padded tokens per DP rank
            mock_runner_utils.get_padded_num_reqs_with_upper_limit.return_value = 2  # Padded reqs per DP rank

            result = self.runner._prepare_dp_input_metadata(scheduler_output)

            (req_ids_dp, req_indices_dp, num_scheduled_tokens_per_dp_rank,
             scheduled_tokens_per_dp_rank, num_req_per_dp_rank,
             padded_num_scheduled_tokens_per_dp_rank, padded_num_reqs,
             padded_total_num_scheduled_tokens, cum_num_req_per_dp_rank_list,
             padded_num_reqs_per_dp_rank, logits_indices_selector,
             max_num_reqs_per_dp_rank) = result

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
            assert padded_num_reqs == 4  # 2 DP ranks * 2 padded reqs per rank

            # 8. padded_total_num_scheduled_tokens: Total padded tokens across all ranks
            assert padded_total_num_scheduled_tokens == 32  # 2 DP ranks * 16 padded tokens per rank

            # 9. cum_num_req_per_dp_rank_list: Cumulative request counts
            assert isinstance(cum_num_req_per_dp_rank_list, np.ndarray)
            expected_cum = np.array(
                [0, 2, 4])  # [0, rank0_reqs, rank0_reqs + rank1_reqs]
            np.testing.assert_array_equal(cum_num_req_per_dp_rank_list,
                                          expected_cum)

            # 10. padded_num_reqs_per_dp_rank: Padded requests per DP rank
            assert padded_num_reqs_per_dp_rank == 2

            # 11. logits_indices_selector: Array to map back to original request order
            assert isinstance(logits_indices_selector, np.ndarray)
            assert len(logits_indices_selector) == 4  # One for each request
            # Should map distributed positions back to original order
            expected_selector = np.array([0, 1, 2,
                                          3])  # Original order preserved
            np.testing.assert_array_equal(logits_indices_selector,
                                          expected_selector)

            # 12. max_num_reqs_per_dp_rank: Maximum requests per DP rank
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

        with patch('tpu_inference.runner.tpu_jax_runner.runner_utils'
                   ) as mock_runner_utils:
            mock_runner_utils.get_padded_token_len.return_value = 16
            mock_runner_utils.get_padded_num_reqs_with_upper_limit.return_value = 2

            result = self.runner._prepare_dp_input_metadata(scheduler_output)

            (req_ids_dp, req_indices_dp, num_scheduled_tokens_per_dp_rank,
             scheduled_tokens_per_dp_rank, num_req_per_dp_rank,
             padded_num_scheduled_tokens_per_dp_rank, padded_num_reqs,
             padded_total_num_scheduled_tokens, cum_num_req_per_dp_rank_list,
             padded_num_reqs_per_dp_rank, logits_indices_selector,
             max_num_reqs_per_dp_rank) = result

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
            assert padded_num_reqs == 4  # 2 DP ranks * 2 padded reqs per rank

            # 8. padded_total_num_scheduled_tokens
            assert padded_total_num_scheduled_tokens == 32  # 2 DP ranks * 16 padded tokens per rank

            # 9. cum_num_req_per_dp_rank_list
            assert isinstance(cum_num_req_per_dp_rank_list, np.ndarray)
            expected_cum = np.array([0, 2, 2])
            np.testing.assert_array_equal(cum_num_req_per_dp_rank_list,
                                          expected_cum)

            # 10. padded_num_reqs_per_dp_rank: Padded requests per DP rank
            assert padded_num_reqs_per_dp_rank == 2

            # 11. logits_indices_selector: Should preserve original order since no reordering needed
            assert isinstance(logits_indices_selector, np.ndarray)
            assert len(logits_indices_selector) == 2
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

        with patch('tpu_inference.runner.tpu_jax_runner.runner_utils'
                   ) as mock_runner_utils:
            mock_runner_utils.get_padded_token_len.return_value = 8
            mock_runner_utils.get_padded_num_reqs_with_upper_limit.return_value = 2

            result = self.runner._prepare_dp_input_metadata(scheduler_output)

            (req_ids_dp, req_indices_dp, _, _, _, _, _, _, _, _,
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

            expected_positions = np.array(
                [2, 0, 3])  # req1 at pos 2, req2 at pos 0, req3 at pos 3
            np.testing.assert_array_equal(logits_indices_selector,
                                          expected_positions)

    @patch('tpu_inference.runner.tpu_jax_runner.NamedSharding')
    @patch('tpu_inference.runner.tpu_jax_runner.runner_utils')
    @patch('tpu_inference.runner.tpu_jax_runner.device_array',
           side_effect=lambda mesh, tensors, **kwargs: tensors)
    @patch('tpu_inference.runner.tpu_jax_runner.TPUSupportedSamplingMetadata')
    def test_prepare_inputs_dp_verify_content_balanced(self,
                                                       mock_sampling_metadata,
                                                       mock_device_array,
                                                       mock_runner_utils,
                                                       mock_named_sharding):
        """Test _prepare_inputs_dp with content verification for balanced distribution."""
        # Setup mocking
        mock_runner_utils.get_padded_token_len.return_value = 8
        mock_runner_utils.get_padded_num_reqs_with_upper_limit.return_value = 2
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
        input_ids, attention_metadata, sampling_metadata, logits_indices, spec_decode_metadata, logits_indices_selector = result

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
        # [0, 2, 1, 1, 1, 0, 3, 1, 1, 1]
        query_start_loc = attention_metadata.query_start_loc_cpu
        max_num_reqs_per_dp = self.runner.max_num_reqs // 2
        expected_query_start = np.zeros(self.runner.max_num_reqs + 2,
                                        dtype=np.int32)
        expected_query_start[1] = 2  # req1 has 2 tokens
        expected_query_start[2:max_num_reqs_per_dp + 1] = 1
        expected_query_start[max_num_reqs_per_dp + 2] = 3  # req2 has 3 tokens
        expected_query_start[max_num_reqs_per_dp + 3:] = 1
        assert np.array_equal(query_start_loc, expected_query_start)

        # 4. Verify seq_lens content
        seq_lens = attention_metadata.seq_lens_cpu
        # Should be computed_tokens + scheduled_tokens for each request
        expected_seq_lens = np.array([7, 0, 0, 0, 9, 0, 0, 0])
        assert np.array_equal(seq_lens, expected_seq_lens)

        # 5. Verify request_distribution content
        expected_distribution = np.array([[0, 0, 1], [0, 0, 1]]).flatten()
        np.testing.assert_array_equal(attention_metadata.request_distribution,
                                      expected_distribution)

        # 6. Verify logits_indices content
        assert len(logits_indices) == 4  # padded_num_reqs
        assert np.array_equal(logits_indices, np.array([1, -1, 2, -1]))

        # 7. Verify logits_indices_selector
        assert len(logits_indices_selector) == 2
        assert np.array_equal(logits_indices_selector, np.array([0, 2]))

    @patch('tpu_inference.runner.tpu_jax_runner.NamedSharding')
    @patch('tpu_inference.runner.tpu_jax_runner.runner_utils')
    @patch('tpu_inference.runner.tpu_jax_runner.device_array',
           side_effect=lambda mesh, tensors, **kwargs: tensors)
    @patch('tpu_inference.runner.tpu_jax_runner.TPUSupportedSamplingMetadata')
    def test_prepare_inputs_dp_verify_content_empty_rank(
            self, mock_sampling_metadata, mock_device_array, mock_runner_utils,
            mock_named_sharding):
        """Test _prepare_inputs_dp with detailed content verification for empty rank case."""
        # Setup mocking
        mock_runner_utils.get_padded_token_len.return_value = 8
        mock_runner_utils.get_padded_num_reqs_with_upper_limit.return_value = 2
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
        input_ids, attention_metadata, sampling_metadata, logits_indices, spec_decode_metadata, logits_indices_selector = result

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
        # Rank 1: empty
        expected_query_start[max_num_reqs_per_dp + 1:] = 1
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
        assert len(logits_indices) == 4  # padded_num_reqs
        # Rank 0: req1 ends at pos 2, req2 ends at pos 4
        # Rank 1: empty, so -1 padding
        expected_logits = np.array([2, 4, -1, -1])
        assert np.array_equal(logits_indices, expected_logits)

        # 7. Verify logits_indices_selector
        assert len(logits_indices_selector) == 2
        expected_selector = np.array([0, 1])
        np.testing.assert_array_equal(logits_indices_selector,
                                      expected_selector)


if __name__ == "__main__":
    pytest.main([__file__])
