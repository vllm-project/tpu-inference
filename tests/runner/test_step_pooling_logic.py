# Copyright 2026 Google LLC
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

import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import torch
from vllm.model_executor.layers.pooler.tokwise.methods import StepPool
from vllm.pooling_params import PoolingParams
from vllm.v1.pool.metadata import PoolingMetadata, PoolingStates


class TestStepPoolingLogic(unittest.TestCase):

    @patch(
        'vllm.model_executor.layers.pooler.tokwise.methods.get_current_vllm_config'
    )
    def test_step_pooling_chunked_prefill(self, mock_get_config):
        # Setup mock config to enable chunked prefill
        mock_config = MagicMock()
        mock_config.scheduler_config.enable_chunked_prefill = True
        mock_get_config.return_value = mock_config

        # Instantiate StepPool
        pooler = StepPool()

        # Setup test data

        # Request 1: 4 tokens total, processed in 2 chunks of 2 tokens.
        # Tag is at position 1 (0-indexed) and position 3.
        # prompt_token_ids: [10, 99, 20, 99]  (99 is the tag)
        prompt_token_ids = torch.tensor([[10, 99, 20, 99]], dtype=torch.int32)
        prompt_lens = torch.tensor([4], dtype=torch.int32)

        pooling_params = PoolingParams(task="embed",
                                       step_tag_id=99,
                                       returned_token_ids=None)
        pooling_states = PoolingStates()

        metadata = PoolingMetadata(
            prompt_lens=prompt_lens,
            prompt_token_ids=prompt_token_ids,
            prompt_token_ids_cpu=prompt_token_ids,
            pooling_params=[pooling_params],
            pooling_states=[pooling_states],
        )

        # SIMULATE PASS 1
        # Hidden states for first chunk (2 tokens)
        hidden_states_1 = torch.tensor(
            [
                [1.0, 1.0, 1.0, 1.0],
                [2.0, 2.0, 2.0, 2.0],  # This corresponds to tag 99
            ],
            dtype=torch.float32)

        # Build cursor for Pass 1
        # num_scheduled_tokens = 2
        metadata.build_pooling_cursor(
            num_scheduled_tokens_np=np.array([2]),
            seq_lens_cpu=torch.tensor([2]),  # Processed 2 tokens so far
            device=hidden_states_1.device)

        # Run forward
        output_1 = pooler(hidden_states_1, metadata)

        # Verify Pass 1
        self.assertEqual(len(output_1), 1)
        self.assertIsNone(output_1[0])
        self.assertEqual(len(pooling_states.hidden_states_cache), 1)
        self.assertTrue(
            torch.allclose(pooling_states.hidden_states_cache[0],
                           hidden_states_1))

        # SIMULATE PASS 2
        # Hidden states for second chunk (2 tokens)
        hidden_states_2 = torch.tensor(
            [
                [3.0, 3.0, 3.0, 3.0],
                [4.0, 4.0, 4.0, 4.0],  # This corresponds to tag 99
            ],
            dtype=torch.float32)

        # Build cursor for Pass 2
        metadata.build_pooling_cursor(num_scheduled_tokens_np=np.array([2]),
                                      seq_lens_cpu=torch.tensor([4]),
                                      device=hidden_states_2.device)

        # Run forward
        output_2 = pooler(hidden_states_2, metadata)

        # Verify Pass 2
        self.assertEqual(len(output_2), 1)
        self.assertIsNotNone(output_2[0])

        # Expected output: tokens matching tag 99 (row 1 from chunk 1 and row 1 from chunk 2)
        # Chunk 1 row 1: [2.0, 2.0, 2.0, 2.0]
        # Chunk 2 row 1: [4.0, 4.0, 4.0, 4.0]
        expected_output = torch.tensor(
            [[2.0, 2.0, 2.0, 2.0], [4.0, 4.0, 4.0, 4.0]], dtype=torch.float32)

        self.assertTrue(torch.allclose(output_2[0], expected_output))

        # Verify cleanup
        self.assertEqual(len(pooling_states.hidden_states_cache), 0)

    @patch(
        'vllm.model_executor.layers.pooler.tokwise.methods.get_current_vllm_config'
    )
    def test_step_pooling_multi_request(self, mock_get_config):
        # Setup mock config
        mock_config = MagicMock()
        mock_config.scheduler_config.enable_chunked_prefill = True
        mock_get_config.return_value = mock_config

        pooler = StepPool()

        prompt_token_ids_1 = torch.tensor([10, 99], dtype=torch.int32)
        prompt_token_ids_2 = torch.tensor([20, 30, 40, 99], dtype=torch.int32)

        prompt_token_ids = torch.zeros((2, 4), dtype=torch.int32)
        prompt_token_ids[0, :2] = prompt_token_ids_1
        prompt_token_ids[1, :4] = prompt_token_ids_2

        prompt_lens = torch.tensor([2, 4], dtype=torch.int32)

        params_1 = PoolingParams(task="embed", step_tag_id=99)
        params_2 = PoolingParams(task="embed", step_tag_id=99)

        states_1 = PoolingStates()
        states_2 = PoolingStates()

        metadata = PoolingMetadata(
            prompt_lens=prompt_lens,
            prompt_token_ids=prompt_token_ids,
            prompt_token_ids_cpu=prompt_token_ids,
            pooling_params=[params_1, params_2],
            pooling_states=[states_1, states_2],
        )

        hidden_states_1 = torch.tensor([
            [1.0, 1.0, 1.0, 1.0],
            [2.0, 2.0, 2.0, 2.0],
            [10.0, 10.0, 10.0, 10.0],
            [11.0, 11.0, 11.0, 11.0],
        ],
                                       dtype=torch.float32)

        metadata.build_pooling_cursor(num_scheduled_tokens_np=np.array([2, 2]),
                                      seq_lens_cpu=torch.tensor([2, 2]),
                                      device=hidden_states_1.device)

        output_1 = pooler(hidden_states_1, metadata)

        self.assertEqual(len(output_1), 2)
        self.assertIsNotNone(output_1[0])
        self.assertIsNone(output_1[1])

        expected_1 = torch.tensor([[2.0, 2.0, 2.0, 2.0]], dtype=torch.float32)
        self.assertTrue(torch.allclose(output_1[0], expected_1))

        self.assertEqual(len(states_2.hidden_states_cache), 1)

        hidden_states_2 = torch.tensor([
            [12.0, 12.0, 12.0, 12.0],
            [13.0, 13.0, 13.0, 13.0],
        ],
                                       dtype=torch.float32)

        metadata_2 = PoolingMetadata(
            prompt_lens=torch.tensor([4], dtype=torch.int32),
            prompt_token_ids=prompt_token_ids[1:2],
            prompt_token_ids_cpu=prompt_token_ids[1:2],
            pooling_params=[params_2],
            pooling_states=[states_2],
        )

        metadata_2.build_pooling_cursor(num_scheduled_tokens_np=np.array([2]),
                                        seq_lens_cpu=torch.tensor([4]),
                                        device=hidden_states_2.device)

        output_2 = pooler(hidden_states_2, metadata_2)

        self.assertEqual(len(output_2), 1)
        self.assertIsNotNone(output_2[0])

        expected_2 = torch.tensor([[13.0, 13.0, 13.0, 13.0]],
                                  dtype=torch.float32)
        self.assertTrue(torch.allclose(output_2[0], expected_2))

        self.assertEqual(len(states_2.hidden_states_cache), 0)


if __name__ == '__main__':
    unittest.main()
