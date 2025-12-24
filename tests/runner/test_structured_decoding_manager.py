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
import jax.numpy as jnp
import numpy as np
from vllm.config import (CacheConfig, ModelConfig, ParallelConfig,
                         SchedulerConfig, SpeculativeConfig, VllmConfig)
from vllm.sampling_params import SamplingType

from tpu_inference.runner.input_batch import CachedRequestState
from tpu_inference.runner.tpu_runner import TPUModelRunner


class TestStructuredDecodingManager:

    def setup_method(self):
        # Mock JAX dependencies
        self.mock_rng_key = MagicMock()
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

    def test_structured_decoding(self):
        # 1. ===== Setup =====
        # Configure runner for the test
        self.runner.model_config.get_vocab_size = MagicMock(return_value=64)
        self.runner._init_inputs()  # re-initialize with new vocab size

        # Mock device_array to avoid JAX sharding issues with MagicMock mesh
        def mock_device_array(mesh, *args, sharding=None, **kwargs):
            # Simply return the arguments without any sharding (skip mesh parameter)
            if len(args) == 1 and isinstance(args[0], tuple):
                return args[0]  # Return tuple as is
            elif len(args) == 1:
                return args[0]  # Return single array as is
            else:
                return args  # Return all arguments as tuple

        # Patch the centralized device_array function instead of runner's method
        with patch(
                'tpu_inference.runner.structured_decoding_manager.device_array',
                side_effect=mock_device_array):

            # Create a mock for sampling_params to avoid TypeErrors in add_request
            mock_sampling_params = MagicMock()
            mock_sampling_params.sampling_type = SamplingType.GREEDY
            mock_sampling_params.temperature = 0.0
            mock_sampling_params.top_p = 1.0
            mock_sampling_params.top_k = -1
            mock_sampling_params.min_tokens = 0
            mock_sampling_params.logprobs = None
            mock_sampling_params.logit_bias = None
            mock_sampling_params.allowed_token_ids = set()
            mock_sampling_params.bad_words_token_ids = None
            mock_sampling_params.all_stop_token_ids = set()

            # Add requests to the input batch
            req1 = CachedRequestState(
                req_id="req-1",
                prompt_token_ids=[1],
                output_token_ids=[],
                sampling_params=mock_sampling_params,
                block_ids=([1], ),
                num_computed_tokens=1,
                lora_request=None,
                mm_features=[],
                pooling_params=None,
                generator=None,
            )
            req2 = CachedRequestState(
                req_id="req-2",
                prompt_token_ids=[2],
                output_token_ids=[],
                sampling_params=mock_sampling_params,
                block_ids=([2], ),
                num_computed_tokens=1,
                lora_request=None,
                mm_features=[],
                pooling_params=None,
                generator=None,
            )
            req3 = CachedRequestState(
                req_id="req-3",
                prompt_token_ids=[3],
                output_token_ids=[],
                sampling_params=mock_sampling_params,
                block_ids=([3], ),
                num_computed_tokens=1,
                lora_request=None,
                mm_features=[],
                pooling_params=None,
                generator=None,
            )
            self.runner.input_batch.add_request(req1)  # index 0
            self.runner.input_batch.add_request(req2)  # index 1
            self.runner.input_batch.add_request(req3)  # index 2
            num_reqs = 3

            # Mock scheduler output for structured decoding
            # req-1 and req-3 require structured decoding
            mock_scheduler_output = MagicMock()
            mock_scheduler_output.structured_output_request_ids = {
                "req-1": 0,  # maps req_id to index in grammar_bitmask
                "req-3": 1,
            }
            # Bitmask: vocab_size=64, so 2 int32s per request
            # Mask for req-1: allow tokens 0-31
            mask1 = np.array([-1, 0], dtype=np.int32)
            # Mask for req-3: allow tokens 32-63
            mask2 = np.array([0, -1], dtype=np.int32)
            mock_scheduler_output.grammar_bitmask = np.array([mask1, mask2])

            # Mock logits
            logits_shape = (num_reqs, self.runner.vocab_size)
            mock_logits_device = jnp.ones(logits_shape, dtype=jnp.bfloat16)

            # 2. ===== Test prepare_structured_decoding_input =====
            (
                require_struct_decoding, grammar_bitmask, arange
            ) = self.runner.structured_decoding_manager.prepare_structured_decoding_input(
                mock_logits_device, mock_scheduler_output)

            # Assertions for prepare_structured_decoding_input
            # require_structured_out_cpu should be [True, False, True]
            # because req-1 is at batch index 0, req-2 at 1, req-3 at 2
            expected_require_struct = np.array([[True], [False], [True]],
                                               dtype=np.bool_)
            np.testing.assert_array_equal(np.array(require_struct_decoding),
                                          expected_require_struct)

            # grammar_bitmask_cpu should have mask1 at index 0, mask2 at index 2
            expected_grammar_bitmask = np.zeros_like(
                self.runner.grammar_bitmask_cpu[:num_reqs])
            expected_grammar_bitmask[0] = mask1
            expected_grammar_bitmask[2] = mask2
            np.testing.assert_array_equal(np.array(grammar_bitmask),
                                          expected_grammar_bitmask)

            np.testing.assert_array_equal(np.array(arange),
                                          np.arange(0, 32, dtype=np.int32))

            # 3. ===== Test structured_decode_fn =====
            # This function is jitted, so we call it with the device arrays
            modified_logits = self.runner.structured_decoding_manager.structured_decode_fn(
                require_struct_decoding, grammar_bitmask, mock_logits_device,
                arange)

            modified_logits_cpu = np.array(modified_logits)

            # Assertions for structured_decode_fn
            # Logits for req-1 (index 0) should be masked for tokens 32-63
            assert np.all(modified_logits_cpu[0, :32] == 1.0)
            assert np.all(modified_logits_cpu[0, 32:] == -np.inf)

            # Logits for req-2 (index 1) should be unchanged
            np.testing.assert_array_equal(modified_logits_cpu[1],
                                          np.ones(self.runner.vocab_size))

            # Logits for req-3 (index 2) should be masked for tokens 0-31
            assert np.all(modified_logits_cpu[2, :32] == -np.inf)
            assert np.all(modified_logits_cpu[2, 32:] == 1.0)
