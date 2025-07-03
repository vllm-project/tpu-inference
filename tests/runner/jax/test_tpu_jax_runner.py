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

    def test_get_kv_cache_for_requests(self):
        # Setup mock data
        self.runner.block_size = 16
        self.runner.max_num_blocks_per_req = 16
        num_blocks = 50
        num_kv_heads = 2
        head_size = 4
        # The shape of kv_cache is (num_blocks, block_size, num_kv_heads * 2, head_size)
        kv_cache_shape = (num_blocks, self.runner.block_size, 2 * num_kv_heads,
                          head_size)
        kv_cache_layer1 = jnp.arange(np.prod(kv_cache_shape),
                                     dtype=jnp.float32).reshape(kv_cache_shape)
        kv_cache_layer2 = jnp.arange(
            np.prod(kv_cache_shape),
            2 * np.prod(kv_cache_shape),
            dtype=jnp.float32).reshape(kv_cache_shape)
        self.runner.kv_caches = [kv_cache_layer1, kv_cache_layer2]

        # Setup input batch state for 3 requests
        self.runner.input_batch.req_id_to_index = {
            'req1': 0,
            'req2': 1,
            'req3': 2
        }
        num_reqs = 3
        # req1: new request, 10 scheduled tokens
        # req2: running, 20 computed, 15 scheduled (crosses block boundary)
        # req3: running, 5 computed, 0 scheduled
        self.runner.input_batch.num_computed_tokens_cpu = np.array(
            [0, 20, 5], dtype=np.int32)
        num_scheduled_tokens_per_req = np.array([10, 15, 0], dtype=np.int32)

        block_table = np.zeros(
            (self.runner.max_num_reqs, self.runner.max_num_blocks_per_req),
            dtype=np.int32)
        # req1 uses block 10
        block_table[0, 0] = 10
        # req2 has 20 computed tokens, using blocks 20 and 21.
        # It's scheduled for 15 more, which will use part of block 21 and 22.
        block_table[1, 0:3] = [20, 21, 22]
        self.runner.input_batch.block_table[0].block_table_cpu = block_table

        # Generate slot mapping metadata
        slot_mapping_metadata_np = self.runner._get_slot_mapping_metadata(
            num_reqs, num_scheduled_tokens_per_req)

        # Simulate padding and transpose from _prepare_inputs
        padded_total_num_scheduled_tokens = _get_padded_token_len(
            self.runner.num_tokens_paddings, 25)
        padded_num_slices = _get_padded_num_kv_cache_update_slices(
            padded_total_num_scheduled_tokens, self.runner.max_num_reqs,
            self.runner.block_size)
        padded_slot_mapping_metadata_np = np.pad(
            slot_mapping_metadata_np,
            [[0, padded_num_slices - len(slot_mapping_metadata_np)], [0, 0]],
            constant_values=0)
        kv_cache_write_indices = jnp.array(
            np.transpose(padded_slot_mapping_metadata_np))

        print(kv_cache_write_indices)

        # Call the function with existing and non-existing requests
        request_ids = ['req1', 'req2', 'req3', 'non_existent_req']
        result = self.runner.get_kv_cache_for_requests(
            request_ids, kv_cache_write_indices, num_scheduled_tokens_per_req)

        # Assertions
        self.assertIn('req1', result)
        self.assertIn('req2', result)
        self.assertIn('req3', result)
        self.assertNotIn('non_existent_req', result)

        # Check req1's KV cache
        # 10 tokens from block 10, starting at offset 0.
        # Flat cache index: 10 * 16 + 0 = 160. Length is 10.
        flat_cache1 = kv_cache_layer1.reshape(-1, *kv_cache_layer1.shape[2:])
        flat_cache2 = kv_cache_layer2.reshape(-1, *kv_cache_layer2.shape[2:])
        expected_req1_l1 = flat_cache1[160:170]
        expected_req1_l2 = flat_cache2[160:170]
        self.assertEqual(len(result['req1']), 2)  # 2 layers
        np.testing.assert_array_equal(result['req1'][0], expected_req1_l1)
        np.testing.assert_array_equal(result['req1'][1], expected_req1_l2)

        # Check req2's KV cache
        # 15 tokens starting from computed_token 20.
        # Slice 1: 12 tokens from block 21, starting at offset 4.
        # Flat cache index: 21 * 16 + 4 = 340. Length 12.
        # Slice 2: 3 tokens from block 22, starting at offset 0.
        # Flat cache index: 22 * 16 + 0 = 352. Length 3.
        indices_req2 = np.concatenate(
            [np.arange(340, 352),
             np.arange(352, 355)])
        expected_req2_l1 = flat_cache1.take(jnp.array(indices_req2), axis=0)
        expected_req2_l2 = flat_cache2.take(jnp.array(indices_req2), axis=0)
        self.assertEqual(len(result['req2']), 2)  # 2 layers
        np.testing.assert_array_equal(result['req2'][0], expected_req2_l1)
        np.testing.assert_array_equal(result['req2'][1], expected_req2_l2)

        # Check req3's KV cache (should be empty)
        self.assertEqual(len(result['req3']), 0)

    def test_insert_request_with_kv_cache(self):
        # This test refines the insertion test by first extracting a KV cache
        # using get_kv_cache_for_requests, simulating a prefill->decode
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
        # Create a request state for prefill.
        prefill_request_state = CachedRequestState(
            req_id="test_req_1",
            prompt_token_ids=list(range(prompt_len)),
            output_token_ids=[],
            sampling_params=mock_sampling_params,
            block_ids=tuple([[5]]),  # Block 5 is allocated for prefill
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

        # 3. ===== Generate metadata and extract KV cache =====
        num_scheduled_tokens_per_req = np.array([prompt_len], dtype=np.int32)

        # This simulates the metadata that would be generated during prefill.
        slot_mapping_metadata = self.runner._get_slot_mapping_metadata(
            num_reqs=1,
            num_scheduled_tokens_per_req=num_scheduled_tokens_per_req)

        # The metadata is padded and transposed before being used.
        padded_num_slices = _get_padded_num_kv_cache_update_slices(
            prompt_len, self.runner.max_num_reqs, self.runner.block_size)
        padded_slot_mapping_metadata = np.pad(
            slot_mapping_metadata,
            [[0, padded_num_slices - len(slot_mapping_metadata)], [0, 0]],
            constant_values=0)
        kv_cache_write_indices = jnp.array(
            np.transpose(padded_slot_mapping_metadata))

        # Extract the KV cache slices for the prefilled request.
        extracted_kv_cache_dict = self.runner.get_kv_cache_for_requests(
            request_ids=['test_req_1'],
            kv_cache_write_indices=kv_cache_write_indices,
            num_scheduled_tokens_per_req=num_scheduled_tokens_per_req)
        extracted_kv_cache_slices = extracted_kv_cache_dict['test_req_1']

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

        # Allocate new block IDs for the decode runner.
        decode_block_ids = [[10]]

        # 5. ===== Call the method to be tested =====
        self.runner.insert_request_with_kv_cache(decode_request,
                                                 extracted_kv_cache_slices,
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