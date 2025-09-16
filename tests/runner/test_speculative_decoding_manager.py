from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from vllm.config import (CacheConfig, ModelConfig, ParallelConfig,
                         SchedulerConfig, SpeculativeConfig, VllmConfig)
from vllm.sampling_params import SamplingType
from vllm.v1.outputs import DraftTokenIds
from vllm.v1.spec_decode.ngram_proposer import NgramProposer

from tpu_commons.runner.input_batch_jax import CachedRequestState, InputBatch
from tpu_commons.runner.speculative_decoding_manager import SpecDecodeMetadata
from tpu_commons.runner.tpu_jax_runner import TPUModelRunner


class TestSpeculativeDecodingManager:

    def setup_method(self):
        # Mock JAX dependencies
        self.mock_devices = [MagicMock()] * 4
        self.mock_mesh = MagicMock()
        self.mock_rng_key = MagicMock()

        with patch('jax.devices', return_value=self.mock_devices), \
             patch('jax.make_mesh', return_value=self.mock_mesh), \
             patch('jax.random.key', return_value=self.mock_rng_key), \
             patch('tpu_commons.runner.tpu_jax_runner.get_model', return_value=MagicMock()):

            model_config = ModelConfig(tokenizer_mode="auto",
                                       trust_remote_code=False,
                                       seed=0,
                                       dtype='bfloat16')
            cache_config = CacheConfig(
                block_size=16,
                gpu_memory_utilization=0.9,
                swap_space=4,
                cache_dtype="auto",
            )
            scheduler_config = SchedulerConfig(max_num_seqs=16, )
            parallel_config = ParallelConfig(
                pipeline_parallel_size=1,
                tensor_parallel_size=1,
                worker_use_ray=False,
            )
            speculative_config = SpeculativeConfig(
                model='ngram',
                num_speculative_tokens=5,
                prompt_lookup_max=4,
            )
            vllm_config = VllmConfig(
                model_config=model_config,
                cache_config=cache_config,
                scheduler_config=scheduler_config,
                parallel_config=parallel_config,
                speculative_config=speculative_config,
                observability_config=None,
                additional_config={},
            )

            self.runner = TPUModelRunner(vllm_config,
                                         devices=self.mock_devices)

    def test_propose_draft_token_ids_wrong_drafter_type(self):
        """Tests that an assertion is raised if the drafter is not an NgramProposer."""
        # The default drafter is NgramProposer, so we replace it with a generic mock
        self.runner.drafter = MagicMock()
        with pytest.raises(AssertionError):
            self.runner.speculative_decoding_manager.propose_draft_token_ids(
                [[1]])

    def test_propose_ngram_draft_token_ids(self):
        """Tests the logic for proposing N-gram draft tokens under various conditions."""
        # 1. ===== Setup =====
        # Mock the NgramProposer
        self.runner.drafter = MagicMock(spec=NgramProposer)

        # Re-initialize input_batch for a clean state for this specific test
        self.runner.input_batch = InputBatch(
            max_num_reqs=self.runner.max_num_reqs,
            max_model_len=self.runner.max_model_len,
            max_num_batched_tokens=self.runner.max_num_tokens,
            pin_memory=False,
            vocab_size=self.runner.vocab_size,
            block_sizes=[self.runner.block_size],
            is_spec_decode=True,
        )

        # Patch is_spec_decode_unsupported to control which requests are marked
        # as unsupported for speculative decoding.
        with patch(
                'tpu_commons.runner.input_batch_jax.is_spec_decode_unsupported'
        ) as mock_is_unsupported:
            # We want req-2 to be unsupported. Let's use a simple condition.
            mock_is_unsupported.return_value = False

            # Setup input_batch with 5 requests for different scenarios
            for i in range(5):
                mock_sampling_params = MagicMock()
                mock_sampling_params.sampling_type = SamplingType.GREEDY
                # This will trigger the mock for req-2
                mock_sampling_params.top_k = -1
                mock_sampling_params.top_p = 1.0
                mock_sampling_params.temperature = 0.0
                mock_sampling_params.min_tokens = 0
                mock_sampling_params.logprobs = None
                mock_sampling_params.logit_bias = None
                mock_sampling_params.allowed_token_ids = set()
                mock_sampling_params.bad_words_token_ids = None
                mock_sampling_params.all_stop_token_ids = set()
                req_state = CachedRequestState(
                    req_id=f"req-{i}",
                    prompt_token_ids=[i] * 10,  # Give some content to tokens
                    output_token_ids=[],
                    sampling_params=mock_sampling_params,
                    block_ids=([1], ),
                    num_computed_tokens=10,
                    lora_request=None,
                    mm_features=[],
                    pooling_params=None,
                    generator=None,
                )
                self.runner.input_batch.add_request(req_state)

        # Configure other individual requests for different test cases
        # req-0: Normal case, should propose tokens.
        # req-1: No sampled tokens provided, should propose nothing.
        # req-2: Unsupported for spec decode (handled by mock).
        self.runner.input_batch.spec_decode_unsupported_reqs.add("req-2")
        # req-3: Max length reached, should propose nothing.
        self.runner.input_batch.num_tokens_no_spec[
            3] = self.runner.max_model_len

        # req-4: Drafter returns None, should propose nothing.

        # Mock the drafter's propose method to handle different cases
        def propose_side_effect(tokens):
            # Identify request by its unique token id
            if tokens[0] == 0:  # req-0
                return np.array([10, 11, 12])
            if tokens[0] == 4:  # req-4
                return None
            # Should not be called for other requests
            return np.array([])

        self.runner.drafter.propose.side_effect = propose_side_effect

        # Input to the function being tested
        sampled_token_ids = [
            [100],  # req-0: has a new token
            [],  # req-1: has no new tokens
            [102],  # req-2: has a new token
            [103],  # req-3: has a new token
            [104],  # req-4: has a new token
        ]

        # 2. ===== Act =====
        result = self.runner.speculative_decoding_manager.propose_ngram_draft_token_ids(
            sampled_token_ids)

        # 3. ===== Assert =====
        expected_result = [
            [10, 11, 12],  # req-0: normal proposal
            [],  # req-1: no sampled tokens
            [],  # req-2: unsupported
            [],  # req-3: max length
            [],  # req-4: drafter returns None
        ]
        assert result == expected_result

        # Verify that drafter.propose was called for the correct requests (req-0 and req-4)
        assert self.runner.drafter.propose.call_count == 2

        # Get the tokens passed to the mock
        called_with_tokens = [
            call.args[0] for call in self.runner.drafter.propose.call_args_list
        ]

        # Check that one call was for req-0's tokens
        expected_tokens_req0 = self.runner.input_batch.token_ids_cpu[0, :10]
        assert any(
            np.array_equal(arg, expected_tokens_req0)
            for arg in called_with_tokens)

        # Check that one call was for req-4's tokens
        expected_tokens_req4 = self.runner.input_batch.token_ids_cpu[4, :10]
        assert any(
            np.array_equal(arg, expected_tokens_req4)
            for arg in called_with_tokens)

    def test_take_draft_token_ids(self):
        """Tests the take_draft_token_ids method for speculative decoding."""
        # Case 1: No draft tokens are available.
        self.runner.speculative_decoding_manager._draft_token_ids = None
        result = self.runner.take_draft_token_ids()
        assert result is None

        # Case 2: Draft tokens are available.
        mock_req_ids = ["req-1", "req-2"]
        mock_draft_ids = [[10, 11], [20, 21, 22]]

        # Re-initialize input_batch for a clean state for this specific test
        self.runner.input_batch = InputBatch(
            max_num_reqs=self.runner.max_num_reqs,
            max_model_len=self.runner.max_model_len,
            max_num_batched_tokens=self.runner.max_num_tokens,
            pin_memory=False,
            vocab_size=self.runner.vocab_size,
            block_sizes=[self.runner.block_size],
            is_spec_decode=True,
        )

        # Add some requests to populate `input_batch.req_ids`
        mock_sampling_params = MagicMock()
        mock_sampling_params.sampling_type = SamplingType.GREEDY
        mock_sampling_params.top_k = -1
        mock_sampling_params.top_p = 1.0
        mock_sampling_params.temperature = 0.0
        mock_sampling_params.min_tokens = 0
        mock_sampling_params.logprobs = None
        mock_sampling_params.logit_bias = None
        mock_sampling_params.allowed_token_ids = set()
        mock_sampling_params.bad_words_token_ids = None
        mock_sampling_params.all_stop_token_ids = set()

        req1 = CachedRequestState(req_id="req-1",
                                  prompt_token_ids=[1],
                                  output_token_ids=[],
                                  sampling_params=mock_sampling_params,
                                  block_ids=([1], ),
                                  num_computed_tokens=1,
                                  lora_request=None,
                                  mm_features=[],
                                  pooling_params=None,
                                  generator=None)
        req2 = CachedRequestState(req_id="req-2",
                                  prompt_token_ids=[2],
                                  output_token_ids=[],
                                  sampling_params=mock_sampling_params,
                                  block_ids=([2], ),
                                  num_computed_tokens=1,
                                  lora_request=None,
                                  mm_features=[],
                                  pooling_params=None,
                                  generator=None)
        self.runner.input_batch.add_request(req1)
        self.runner.input_batch.add_request(req2)

        # Set the draft tokens to be taken
        self.runner.speculative_decoding_manager._draft_token_ids = mock_draft_ids

        # Call the method to be tested
        result = self.runner.take_draft_token_ids()

        # Assertions for the returned object
        assert result is not None
        assert isinstance(result, DraftTokenIds)
        assert result.req_ids == mock_req_ids
        assert result.draft_token_ids == mock_draft_ids

        # Assert that the internal state is reset
        assert self.runner.speculative_decoding_manager._draft_token_ids is None

        # Case 3: Call again after taking, should return None
        result_after = self.runner.take_draft_token_ids()
        assert result_after is None

    def _setup_spec_decode_metadata_test(self):
        """Helper method to set up common test infrastructure for spec decode metadata tests."""
        # Mock runner attributes needed by the function
        self.runner.arange_cpu = np.arange(1024, dtype=np.int64)
        # Make input_ids_cpu a sequence of numbers for easy verification
        self.runner.input_ids_cpu = np.arange(1024, dtype=np.int32) * 10
        self.runner.num_tokens_paddings = [16, 32, 64, 128, 256, 512, 1024]

        # Mock the device_array function to just return the numpy arrays
        def mock_device_array(mesh, *args, **kwargs):
            # Skip mesh parameter and return the actual arrays
            if len(args) == 1 and isinstance(args[0], tuple):
                return args[0]
            return args

        self.mock_device_array = mock_device_array

    @pytest.mark.parametrize(
        "num_draft_tokens,cu_num_scheduled_tokens,padded_num_reqs,expected_logits_indices,expected_bonus_logits_indices,expected_target_logits_indices,expected_draft_token_ids",
        [
            (
                # Normal case
                [3, 0, 2, 0, 1],
                [4, 104, 107, 207, 209],
                8,
                [0, 1, 2, 3, 103, 104, 105, 106, 206, 207, 208],
                [3, 4, 7, 8, 10, 0, 0, 0],
                [0, 1, 2, 5, 6, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [10, 20, 30, 1050, 1060, 2080, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
            (
                # High speculative tokens case
                [5, 3, 4, 2, 1],
                [6, 10, 18, 22, 26],
                8,
                [
                    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 13, 14, 15, 16, 17, 19, 20,
                    21, 24, 25
                ],
                [5, 9, 14, 17, 19, 0, 0, 0],
                [
                    0, 1, 2, 3, 4, 6, 7, 8, 10, 11, 12, 13, 15, 16, 18, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
                ],
                [
                    10, 20, 30, 40, 50, 70, 80, 90, 140, 150, 160, 170, 200,
                    210, 250
                ]),
        ])
    def test_get_spec_decode_metadata_parametrized(
            self, num_draft_tokens, cu_num_scheduled_tokens, padded_num_reqs,
            expected_logits_indices, expected_bonus_logits_indices,
            expected_target_logits_indices, expected_draft_token_ids):
        """Comprehensive parametrized test for _get_spec_decode_metadata function."""
        # Setup
        self._setup_spec_decode_metadata_test()

        # Convert Python lists to numpy arrays for function input
        num_draft_tokens_np = np.array(num_draft_tokens, dtype=np.int32)
        cu_num_scheduled_tokens_np = np.array(cu_num_scheduled_tokens,
                                              dtype=np.int32)

        # Act
        with patch(
                "tpu_commons.runner.speculative_decoding_manager.device_array",
                side_effect=self.mock_device_array):
            metadata = self.runner.speculative_decoding_manager.get_spec_decode_metadata(
                num_draft_tokens_np,
                cu_num_scheduled_tokens_np,
                padded_num_reqs=padded_num_reqs)

        # Assert basic properties
        assert isinstance(metadata, SpecDecodeMetadata)

        # Determine padding length based on expected_logits_indices length
        if len(expected_logits_indices) <= 16:
            padded_len = 16
        else:
            padded_len = 32

        # final_logits_indices - pad to bucket size and compare as Python lists
        expected_padded_logits_indices = expected_logits_indices + [0] * (
            padded_len - len(expected_logits_indices))
        assert np.asarray(metadata.final_logits_indices).tolist(
        ) == expected_padded_logits_indices

        # bonus_logits_indices - compare as Python lists
        assert np.asarray(metadata.bonus_logits_indices).tolist(
        ) == expected_bonus_logits_indices

        # target_logits_indices - pad to same length as final_logits_indices and compare as Python lists
        expected_padded_target_logits_indices = expected_target_logits_indices + [
            0
        ] * (padded_len - len(expected_target_logits_indices))
        assert np.asarray(metadata.target_logits_indices).tolist(
        ) == expected_padded_target_logits_indices

        # draft_token_ids - pad the expected values to the correct length and compare as Python lists
        expected_padded_draft_token_ids = expected_draft_token_ids + [0] * (
            padded_len - len(expected_draft_token_ids))
        assert np.asarray(metadata.draft_token_ids).tolist(
        ) == expected_padded_draft_token_ids

        # draft_lengths - pad and compare as Python lists
        expected_padded_num_draft_tokens = num_draft_tokens + [0] * (
            padded_num_reqs - len(num_draft_tokens))
        assert np.asarray(metadata.draft_lengths).tolist(
        ) == expected_padded_num_draft_tokens
