import unittest
from unittest.mock import MagicMock, patch

import os
os.environ["TPU_BACKEND_TYPE"] = "jax"

import jax.numpy as jnp
import numpy as np
from vllm.config import (CacheConfig, ModelConfig, ParallelConfig,
                         SchedulerConfig, VllmConfig)
from vllm.v1.request import Request
from vllm.sampling_params import SamplingType
from tpu_commons.runner.jax.input_batch_jax import CachedRequestState

from tpu_commons.runner.jax.tpu_jax_runner import (
    _get_padded_num_kv_cache_update_slices,
    _get_padded_token_len,
)

from tpu_commons.runner.jax.tpu_jax_runner import TPUModelRunner


class TestTPUJaxRunner(unittest.TestCase):

    def setUp(self):
        # Mock JAX dependencies
        self.mock_devices = [MagicMock()] * 4
        self.mock_mesh = MagicMock()
        self.mock_rng_key = MagicMock()

        with patch('jax.devices', return_value=self.mock_devices), \
             patch('jax.make_mesh', return_value=self.mock_mesh), \
             patch('jax.random.key', return_value=self.mock_rng_key), \
             patch('tpu_commons.runner.jax.tpu_jax_runner.get_model', return_value=MagicMock()):

            model_config = ModelConfig(
                tokenizer_mode="auto",
                trust_remote_code=False,
                seed=0,
                dtype='bfloat16'
            )
            cache_config = CacheConfig(
                block_size=16,
                gpu_memory_utilization=0.9,
                swap_space=4,
                cache_dtype="auto",
            )
            scheduler_config = SchedulerConfig(
                max_num_seqs=16,
            )
            parallel_config = ParallelConfig(
                pipeline_parallel_size=1,
                tensor_parallel_size=1,
                worker_use_ray=False,
            )
            vllm_config = VllmConfig(
                model_config=model_config,
                cache_config=cache_config,
                scheduler_config=scheduler_config,
                parallel_config=parallel_config,
                speculative_config=None,
                prompt_adapter_config=None,
                observability_config=None,
                additional_config={},
            )

            self.runner = TPUModelRunner(vllm_config, devices=self.mock_devices)

    def test_get_slot_mapping_metadata_multiple_requests(self):
        # Setup test case with two requests
        num_reqs = 2
        num_scheduled_tokens_per_req = np.array([10, 30], dtype=np.int32)

        # Configure the runner's state
        self.runner.block_size = 16
        self.runner.max_num_blocks_per_req = 16  # 256 / 16

        # Mock input_batch state
        # Request 0 starts from 0, Request 1 starts from 10
        self.runner.input_batch.num_computed_tokens_cpu = np.array([0, 10], dtype=np.int32)

        block_table = np.zeros((self.runner.max_num_reqs, self.runner.max_num_blocks_per_req), dtype=np.int32)
        # Request 0 uses 1 block, id 100
        block_table[0, 0] = 100
        # Request 1 uses 3 blocks, ids 200, 201, 202
        block_table[1, 0:3] = [200, 201, 202]

        self.runner.input_batch.block_table[0].block_table_cpu = block_table

        # Expected output
        expected_metadata = np.array([
            [1600,  0, 10],  # Req 0, block 0 (id 100), 10 tokens
            [3210, 10,  6],  # Req 1, block 0 (id 200), 6 tokens (10..15)
            [3216, 16, 16],  # Req 1, block 1 (id 201), 16 tokens (16..31)
            [3232, 32,  8]   # Req 1, block 2 (id 202), 8 tokens (32..39)
        ], dtype=np.int64)
        print(expected_metadata)

        # Call the function
        result = self.runner._get_slot_mapping_metadata(num_reqs, num_scheduled_tokens_per_req)

        # Assertions
        self.assertIsInstance(result, np.ndarray)
        np.testing.assert_array_equal(result, expected_metadata)

    def test_get_slot_mapping_metadata_single_request_crossing_blocks(self):
        # Setup test case for a single request that crosses block boundaries
        num_reqs = 1
        num_scheduled_tokens_per_req = np.array([20], dtype=np.int32)

        # Configure the runner's state
        self.runner.block_size = 16
        self.runner.max_num_blocks_per_req = 16

        # Mock input_batch state
        # Request starts from token 5
        self.runner.input_batch.num_computed_tokens_cpu = np.array([5], dtype=np.int32)

        block_table = np.zeros((self.runner.max_num_reqs, self.runner.max_num_blocks_per_req), dtype=np.int32)
        # Request uses 2 blocks, ids 50, 51
        block_table[0, 0:2] = [50, 51]

        self.runner.input_batch.block_table[0].block_table_cpu = block_table

        # Expected output
        expected_metadata = np.array([
            [805,  0, 11],  # Req 0, block 0 (id 50), 11 tokens (5..15)
            [816, 11,  9]   # Req 0, block 1 (id 51), 9 tokens (16..24)
        ], dtype=np.int64)

        # Call the function
        result = self.runner._get_slot_mapping_metadata(num_reqs, num_scheduled_tokens_per_req)

        # Assertions
        np.testing.assert_array_equal(result, expected_metadata)

    def test_insert_request_with_kv_cache(self):
        # This test refines the insertion test by first extracting a KV cache
        # using get_kv_cache_for_block_ids, simulating a prefill->decode
        # transfer, and then inserting it. This ensures the extraction and
        # insertion logic are compatible.

        # 1. ===== Setup source runner for prefill simulation =====
        self.runner.block_size = 64
        num_layers = 2
        num_kv_heads = 16
        head_size = 128
        num_blocks = 50
        prompt_len = 63

        # Populate a source KV cache with data. This represents the state
        # of the prefill runner's KV cache.
        source_kv_cache_shape = (num_blocks, self.runner.block_size,
                                 2 * num_kv_heads, head_size)
        prod_val = int(np.prod(source_kv_cache_shape))
        source_kv_caches = [
            jnp.arange(prod_val,
                       dtype=jnp.bfloat16).reshape(source_kv_cache_shape),
            jnp.arange(prod_val,
                       2 * prod_val,
                       dtype=jnp.bfloat16).reshape(source_kv_cache_shape)
        ]
        self.runner.kv_caches = source_kv_caches

        # Create a mock for sampling_params to avoid TypeErrors in add_request
        mock_sampling_params = MagicMock()
        mock_sampling_params.sampling_type = SamplingType.GREEDY
        mock_sampling_params.temperature = 0.0
        mock_sampling_params.top_p = 1.0
        mock_sampling_params.top_k = -1  # Common value for greedy
        mock_sampling_params.min_tokens = 0
        mock_sampling_params.logprobs = None
        mock_sampling_params.logit_bias = None
        mock_sampling_params.allowed_token_ids = set()
        mock_sampling_params.bad_words_token_ids = None
        mock_sampling_params.all_stop_token_ids = set()

        # 2. ===== Simulate prefill execution state =====
        prefill_block_ids = [5]
        # Create a request state for prefill.
        prefill_request_state = CachedRequestState(
            req_id="test_req_1",
            prompt_token_ids=list(range(prompt_len)),
            output_token_ids=[],
            sampling_params=mock_sampling_params,
            block_ids=tuple([prefill_block_ids]),
            num_computed_tokens=0,
            lora_request=None,
            mm_inputs=[],
            mm_hashes=[],
            mm_positions=[],
            pooling_params=None,
            generator=None,
        )

        # Add the request to the input_batch to simulate it being scheduled.
        self.runner.input_batch.add_request(prefill_request_state)

        # 3. ===== Extract KV cache using get_kv_cache_for_block_ids =====
        # Extract the full KV cache for the allocated block.
        full_block_kv_cache = self.runner.get_kv_cache_for_block_ids(
            block_ids=prefill_block_ids)

        # Since get_kv_cache_for_block_ids returns the full block, but the
        # prompt only fills part of it, we need to slice it to the actual
        # prompt length for the insertion test to be accurate.
        extracted_kv_cache_slices = [
            layer_cache[:prompt_len] for layer_cache in full_block_kv_cache
        ]

        # 4. ===== Setup destination runner for decode simulation =====
        # Reset runner state to simulate a fresh decode runner.
        self.runner.requests = {}
        req_index = self.runner.input_batch.remove_request("test_req_1")
        if req_index is not None:
            self.runner.input_batch.condense([req_index])

        # Initialize destination KV caches with zeros.
        dest_kv_cache_shape = (num_blocks, self.runner.block_size,
                               2 * num_kv_heads, head_size)
        self.runner.kv_caches = [
            jnp.zeros(dest_kv_cache_shape, dtype=jnp.bfloat16)
            for _ in range(num_layers)
        ]

        # Create a mock request as it would be after prefill + 1 token.
        decode_request = MagicMock(spec=Request)
        decode_request.request_id = "test_req_1"
        decode_request.num_tokens = prompt_len + 1 # Total tokens
        decode_request.prompt_token_ids = list(range(prompt_len))
        decode_request.output_token_ids = [100]
        decode_request.sampling_params = mock_sampling_params
        decode_request.lora_request = None
        decode_request.mm_inputs, decode_request.mm_positions = [], []
        decode_request.pooling_params, decode_request.generator = None, None

        # Prepare the KV cache slices for insertion. They must be padded to the
        # full block size and have a leading dimension for the number of blocks.
        padded_kv_cache_slices = []
        padding_size = self.runner.block_size - prompt_len
        for slice_per_layer in extracted_kv_cache_slices:
            padded_slice = jnp.pad(
                slice_per_layer,
                ((0, padding_size), (0, 0), (0, 0)),
                mode='constant'
            )
            # Add a dimension for the number of blocks.
            padded_kv_cache_slices.append(padded_slice[jnp.newaxis, ...])

        # Allocate new block IDs for the decode runner.
        decode_block_ids = [[10]]

        # 5. ===== Call the method to be tested =====
        self.runner.insert_request_with_kv_cache(decode_request,
                                                 padded_kv_cache_slices,
                                                 decode_block_ids)

        # 6. ===== Assertions =====
        self.assertIn("test_req_1", self.runner.requests)
        self.assertIn("test_req_1", self.runner.input_batch.req_id_to_index)
        self.assertEqual(self.runner.requests["test_req_1"].num_computed_tokens,
                         prompt_len + 1)

        # Verify the content of the inserted KV cache.
        target_block_id = decode_block_ids[0][0]
        for i, layer_kv_cache in enumerate(self.runner.kv_caches):
            updated_block_content = layer_kv_cache[target_block_id]

            # The extracted slice should be padded to the block size.
            padding_size = self.runner.block_size - prompt_len
            expected_padded_slice = jnp.pad(
                extracted_kv_cache_slices[i],
                ((0, padding_size), (0, 0), (0, 0)),
                mode='constant')
            np.testing.assert_array_equal(updated_block_content,
                                          expected_padded_slice)

if __name__ == '__main__':
    unittest.main()