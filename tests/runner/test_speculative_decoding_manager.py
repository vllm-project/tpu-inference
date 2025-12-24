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

import jax
import numpy as np
import pytest
from vllm.config import (CacheConfig, ModelConfig, ParallelConfig,
                         SchedulerConfig, SpeculativeConfig, VllmConfig)
from vllm.sampling_params import SamplingType
from vllm.v1.outputs import DraftTokenIds

from tpu_inference.runner.input_batch import CachedRequestState, InputBatch
from tpu_inference.runner.speculative_decoding_manager import \
    SpecDecodeMetadata
from tpu_inference.runner.tpu_runner import TPUModelRunner
from tpu_inference.spec_decode.jax.eagle3 import Eagle3Proposer


class TestSpeculativeDecodingManager:

    def setup_method(self):
        # Mock JAX dependencies
        self.mock_devices = [MagicMock(coords=i) for i in range(1)]
        device_array = np.array(jax.devices()[:1]).reshape(1, 1, 1, 1)
        self.mock_mesh = jax.make_mesh(device_array.shape,
                                       ('data', 'attn_dp', 'expert', 'model'))
        self.mock_rng_key = MagicMock()

        with patch('jax.devices', return_value=self.mock_devices), \
             patch('jax.make_mesh', return_value=self.mock_mesh), \
             patch('jax.random.key', return_value=self.mock_rng_key), \
             patch('tpu_inference.runner.tpu_runner.get_model', return_value=MagicMock()), \
             patch('tpu_inference.runner.tpu_runner.make_optimized_mesh', return_value=self.mock_mesh):

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
            scheduler_config = SchedulerConfig(max_num_seqs=16,
                                               max_model_len=1024,
                                               is_encoder_decoder=False)
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
                observability_config={},
                additional_config={},
            )

            self.runner = TPUModelRunner(vllm_config,
                                         devices=self.mock_devices)

    def test_propose_draft_token_ids_dispatches_to_eagle(self):
        """Tests that propose_draft_token_ids calls the correct eagle method."""
        # 1. ===== Setup =====
        # Set the drafter to be an Eagle3Proposer
        self.runner.drafter = MagicMock(spec=Eagle3Proposer)
        self.runner.speculative_config.method = "eagle3"

        # Mock the eagle-specific proposal method
        with patch.object(self.runner.speculative_decoding_manager,
                          'propose_eagle3_draft_token_ids',
                          return_value=[[10, 11]]) as mock_propose_eagle:

            # 2. ===== Act =====
            self.runner.speculative_decoding_manager.propose_draft_token_ids(
                sampled_token_ids=[[1]],
                aux_hidden_states=None,
                attn_metadata=MagicMock(),
                spec_decode_metadata=None,
            )

            # 3. ===== Assert =====
            mock_propose_eagle.assert_called_once()
            assert self.runner.speculative_decoding_manager._draft_token_ids == [
                [10, 11]
            ]

    def test_propose_draft_token_ids_wrong_drafter_type(self):
        """Tests that an assertion is raised if the drafter is not an NgramProposer."""
        # The default drafter is NgramProposer, so we replace it with a generic mock
        self.runner.drafter = MagicMock()
        self.runner.speculative_config.method = "ngram"
        with pytest.raises(AssertionError):
            self.runner.speculative_decoding_manager.propose_draft_token_ids(
                [[1]], None, MagicMock(), None)

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
                "tpu_inference.runner.speculative_decoding_manager.device_array",
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

    @pytest.mark.parametrize("spec_decode_metadata_is_none", [True, False])
    def test_propose_eagle3_draft_token_ids(self,
                                            spec_decode_metadata_is_none):
        """Tests the logic for proposing Eagle3 draft tokens."""
        # 1. ===== Setup =====
        self.runner.drafter = MagicMock(spec=Eagle3Proposer)
        self.runner.speculative_config.method = "eagle3"

        # Mock TPUModelRunner attributes
        self.runner.input_batch = MagicMock()
        self.runner.input_batch.req_ids = ["req-1", "req-2"]
        self.runner.requests = {
            "req-1": MagicMock(),
            "req-2": MagicMock(),
        }
        self.runner.mesh = self.mock_mesh
        self.runner.kv_caches = MagicMock()

        # Mock drafter methods
        mock_attn_metadata = MagicMock()
        mock_target_token_ids = MagicMock()
        mock_last_token_indices = MagicMock()
        mock_target_hidden_states = MagicMock()
        self.runner.drafter.prepare_inputs.return_value = (
            mock_target_hidden_states,
            mock_target_token_ids,
            mock_last_token_indices,
            mock_attn_metadata,
        )
        mock_draft_token_ids = [[10, 11], [20, 21]]
        self.runner.drafter.propose.return_value = (
            self.runner.kv_caches,
            mock_draft_token_ids,
        )

        # Inputs
        sampled_token_ids = [[1], [2]]
        aux_hidden_states = MagicMock()
        attn_metadata = MagicMock()
        attn_metadata.seq_lens.shape = [2]
        if spec_decode_metadata_is_none:
            spec_decode_metadata = None
        else:
            spec_decode_metadata = MagicMock(spec=SpecDecodeMetadata)
            spec_decode_metadata.draft_lengths_cpu = np.array([2, 3])
        scheduler_output = MagicMock()
        input_ids = MagicMock()

        # 2. ===== Act =====
        with patch(
                "tpu_inference.runner.speculative_decoding_manager.device_array",
                side_effect=lambda mesh, x: x):
            result = self.runner.speculative_decoding_manager.propose_eagle3_draft_token_ids(
                sampled_token_ids,
                aux_hidden_states,
                attn_metadata,
                spec_decode_metadata,
                scheduler_output,
                input_ids,
            )

        # 3. ===== Assert =====
        assert result == [[10, 11], [20, 21]]
        self.runner.drafter.prepare_inputs.assert_called_once()
        self.runner.drafter.propose.assert_called_once()
