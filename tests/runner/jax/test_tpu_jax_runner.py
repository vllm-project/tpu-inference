import unittest
from unittest.mock import MagicMock, patch

import jax.numpy as jnp
import numpy as np
from vllm.config import (CacheConfig, ModelConfig, ParallelConfig,
                         SchedulerConfig, VllmConfig)
from vllm.model_executor.layers.rotary_embedding import MRotaryEmbedding
from vllm.multimodal.inputs import (MultiModalBatchedField,
                                    MultiModalFieldElem, MultiModalKwargsItem)
from vllm.sampling_params import SamplingType
from vllm.v1.core.sched.output import SchedulerOutput as VllmSchedulerOutput
from vllm.v1.request import PlaceholderRange, Request

from tpu_commons.runner.jax.input_batch_jax import (CachedRequestState,
                                                    InputBatch)
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
            vllm_config = VllmConfig(
                model_config=model_config,
                cache_config=cache_config,
                scheduler_config=scheduler_config,
                parallel_config=parallel_config,
                speculative_config=None,
                observability_config=None,
                additional_config={},
            )

            self.runner = TPUModelRunner(vllm_config,
                                         devices=self.mock_devices)

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
        # This is needed for the padding logic in insert_request_with_kv_cache
        self.runner.vllm_config.cache_config.num_gpu_blocks = num_blocks

        prompt_len = 63

        # Populate a source KV cache with data. This represents the state
        # of the prefill runner's KV cache.
        source_kv_cache_shape = (num_blocks, self.runner.block_size,
                                 2 * num_kv_heads, head_size)
        prod_val = int(np.prod(source_kv_cache_shape))
        source_kv_caches = [
            jnp.arange(prod_val,
                       dtype=jnp.bfloat16).reshape(source_kv_cache_shape),
            jnp.arange(prod_val, 2 * prod_val,
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
            mm_kwargs=[],
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
        decode_request.num_tokens = prompt_len + 1  # Total tokens
        decode_request.num_computed_tokens = prompt_len
        decode_request.prompt_token_ids = list(range(prompt_len))
        decode_request.all_token_ids = [123, 232, 908]
        decode_request.output_token_ids = [100]
        decode_request.sampling_params = mock_sampling_params

        decode_request.lora_request = None
        decode_request.mm_kwargs, decode_request.mm_positions = [], []
        decode_request.pooling_params, decode_request.generator = None, None

        # Prepare the KV cache slices for insertion. They must be padded to the
        # full block size and have a leading dimension for the number of blocks.

        # Allocate new block IDs for the decode runner.
        decode_block_ids = [[10]]
        # 5. ===== Call the method to be tested =====
        self.runner.insert_request_with_kv_cache(decode_request,
                                                 extracted_kv_cache_slices,
                                                 decode_block_ids)

        # 6. ===== Assertions =====
        self.assertIn("test_req_1", self.runner.requests)
        self.assertIn("test_req_1", self.runner.input_batch.req_id_to_index)
        self.assertEqual(
            self.runner.requests["test_req_1"].num_computed_tokens, prompt_len)
        self.assertEqual(self.runner.requests["test_req_1"].output_token_ids,
                         [908])

        # Verify the content of the inserted KV cache.
        target_block_id = decode_block_ids[0][0]
        for i, layer_kv_cache in enumerate(self.runner.kv_caches):
            updated_block_content = layer_kv_cache[target_block_id]

            # The extracted slice should be padded to the block size.
            padding_size = self.runner.block_size - prompt_len
            expected_padded_slice = jnp.pad(extracted_kv_cache_slices[i],
                                            ((0, padding_size), (0, 0),
                                             (0, 0)),
                                            mode='constant')
            np.testing.assert_array_equal(updated_block_content,
                                          expected_padded_slice)

    def test_get_supported_tasks_runner(self):
        """Test get_supported_tasks for generate runner type."""
        supported_tasks = self.runner.get_supported_tasks()
        assert supported_tasks == ("generate", )

    def test_structured_decoding(self):
        # 1. ===== Setup =====
        # Configure runner for the test
        self.runner.model_config.get_vocab_size = MagicMock(return_value=64)
        self.runner._init_inputs()  # re-initialize with new vocab size

        # Mock _device_array to avoid JAX sharding issues with MagicMock mesh
        def mock_device_array(*args, sharding=None, **kwargs):
            # Simply return the first argument (the array) without any sharding
            if len(args) == 1 and isinstance(args[0], tuple):
                return args[0]  # Return tuple as is
            elif len(args) == 1:
                return args[0]  # Return single array as is
            else:
                return args  # Return all arguments as tuple

        self.runner._device_array = mock_device_array

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
            mm_kwargs=[],
            mm_hashes=[],
            mm_positions=[],
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
            mm_kwargs=[],
            mm_hashes=[],
            mm_positions=[],
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
            mm_kwargs=[],
            mm_hashes=[],
            mm_positions=[],
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
        (require_struct_decoding, grammar_bitmask,
         arange) = self.runner.prepare_structured_decoding_input(
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
        modified_logits = self.runner.structured_decode_fn(
            require_struct_decoding, grammar_bitmask, mock_logits_device,
            arange)

        modified_logits_cpu = np.array(modified_logits)

        # Assertions for structured_decode_fn
        # Logits for req-1 (index 0) should be masked for tokens 32-63
        self.assertTrue(np.all(modified_logits_cpu[0, :32] == 1.0))
        self.assertTrue(np.all(modified_logits_cpu[0, 32:] == -np.inf))

        # Logits for req-2 (index 1) should be unchanged
        np.testing.assert_array_equal(modified_logits_cpu[1],
                                      np.ones(self.runner.vocab_size))

        # Logits for req-3 (index 2) should be masked for tokens 0-31
        self.assertTrue(np.all(modified_logits_cpu[2, :32] == -np.inf))
        self.assertTrue(np.all(modified_logits_cpu[2, 32:] == 1.0))

    def test_reorder_batch(self):
        # Helper to set up the runner's input_batch for a test case
        def setup_batch(req_token_map: dict[str, int]):
            # Re-initialize input_batch for a clean state
            self.runner.input_batch = InputBatch(
                max_num_reqs=self.runner.max_num_reqs,
                max_model_len=self.runner.max_model_len,
                max_num_batched_tokens=self.runner.max_num_tokens,
                pin_memory=False,
                vocab_size=self.runner.vocab_size,
                block_sizes=[self.runner.block_size],
            )
            self.runner.requests = {}

            # Create a mock for sampling_params
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

            # Add requests in the order they appear in the map
            for req_id in req_token_map.keys():
                req_state = CachedRequestState(
                    req_id=req_id,
                    prompt_token_ids=[0],
                    output_token_ids=[],
                    sampling_params=mock_sampling_params,
                    block_ids=([1], ),
                    num_computed_tokens=1,
                    lora_request=None,
                    mm_kwargs=[],
                    mm_hashes=[],
                    mm_positions=[],
                    pooling_params=None,
                    generator=None,
                )
                self.runner.input_batch.add_request(req_state)
                self.runner.requests[req_id] = req_state

            # Mock scheduler output
            mock_scheduler_output = MagicMock()
            mock_scheduler_output.num_scheduled_tokens = req_token_map
            return mock_scheduler_output

        # Test case 1: Requests already sorted.
        req_map = {"d1": 1, "d2": 1, "p1": 10, "p2": 10, "p3": 5}
        scheduler_output = setup_batch(req_map)
        modified = self.runner._reorder_batch(scheduler_output)
        self.assertFalse(modified)
        self.assertEqual(self.runner.input_batch.req_ids,
                         ["d1", "d2", "p1", "p2", "p3"])
        np.testing.assert_array_equal(
            self.runner.input_batch.request_distribution, [2, 2, 1])

        # Test case 2: Simple 2-way swap.
        req_map = {"p1": 10, "d1": 1, "p2": 10, "p3": 5}
        scheduler_output = setup_batch(req_map)
        modified = self.runner._reorder_batch(scheduler_output)
        self.assertTrue(modified)
        final_ids = self.runner.input_batch.req_ids
        self.assertEqual(self.runner.input_batch.req_ids,
                         ["d1", "p1", "p2", "p3"])
        np.testing.assert_array_equal(
            self.runner.input_batch.request_distribution, [1, 2, 1])

        # Test case 3: Complex reordering with multiple swaps
        req_map = {"p1": 5, "p2": 10, "d1": 1, "p3": 10, "d2": 1, "p4": 20}
        scheduler_output = setup_batch(req_map)
        modified = self.runner._reorder_batch(scheduler_output)
        self.assertTrue(modified)
        final_ids = self.runner.input_batch.req_ids
        self.assertEqual(set(final_ids[0:2]), {"d1", "d2"})
        self.assertEqual(set(final_ids[2:4]), {"p2", "p3"})
        self.assertEqual(set(final_ids[4:6]), {"p1", "p4"})
        np.testing.assert_array_equal(
            self.runner.input_batch.request_distribution, [2, 2, 2])

    def test_execute_mm_encoder_single_image(self):
        import torch
        """Tests _execute_mm_encoder with a single request and a single image."""
        # 1. ===== Setup =====
        self.runner.is_multimodal_model = True
        self.mock_get_mm_embed_fn = MagicMock()
        self.runner.get_multimodal_embeddings_fn = self.mock_get_mm_embed_fn

        self.runner.state = MagicMock()
        # Mock scheduler output
        mock_scheduler_output = MagicMock(spec=VllmSchedulerOutput)
        mock_scheduler_output.scheduled_encoder_inputs = {"req-1": [0]}

        # Mock request state
        dummy_pixel_values = torch.randn(3, 224, 224, dtype=torch.bfloat16)
        dummy_grid_thw = torch.tensor([[1, 1, 1]], dtype=torch.int64)
        mm_item = MultiModalKwargsItem.from_elems([
            MultiModalFieldElem("image", "pixel_values", dummy_pixel_values,
                                MultiModalBatchedField()),
            MultiModalFieldElem("image", "image_grid_thw", dummy_grid_thw,
                                MultiModalBatchedField())
        ])

        req_state = CachedRequestState(
            req_id="req-1",
            prompt_token_ids=[1, 2, 3],
            output_token_ids=[],
            sampling_params=MagicMock(),
            block_ids=(),
            num_computed_tokens=0,
            mm_kwargs=[mm_item],
            mm_positions=[PlaceholderRange(offset=0, length=1)],
            lora_request=None,
            mm_hashes=[],
            pooling_params=None,
            generator=None,
        )
        self.runner.requests = {"req-1": req_state}

        # Mock the return value of the multimodal encoder
        dummy_embedding = jnp.ones((10, 128), dtype=jnp.bfloat16)
        self.mock_get_mm_embed_fn.return_value = (dummy_embedding, )

        # 2. ===== Act =====
        self.runner._execute_mm_encoder(mock_scheduler_output)

        # 3. ===== Assert =====
        # Check if encoder_cache is populated correctly
        self.assertIn("req-1", self.runner.encoder_cache)
        self.assertIn(0, self.runner.encoder_cache["req-1"])
        cached_embedding = self.runner.encoder_cache["req-1"][0]
        np.testing.assert_array_equal(np.asarray(cached_embedding),
                                      np.asarray(dummy_embedding))

        # Check if get_multimodal_embeddings_fn was called with correct args
        self.mock_get_mm_embed_fn.assert_called_once()
        call_args = self.mock_get_mm_embed_fn.call_args

        # Positional args: (state, image_grid_thw)
        state_arg, grid_arg = call_args.args
        # Keyword args: **batched_mm_inputs
        kwargs_arg = call_args.kwargs

        self.assertEqual(state_arg, self.runner.state)
        self.assertEqual(grid_arg, ((1, 1, 1), ))
        self.assertIn("pixel_values", kwargs_arg)

        # Verify the pixel values tensor passed to the mock
        passed_pixel_values = kwargs_arg['pixel_values']
        self.assertIsInstance(passed_pixel_values, np.ndarray)
        self.assertEqual(passed_pixel_values.dtype, jnp.bfloat16)

        # Convert torch tensor for comparison
        expected_pixel_values = dummy_pixel_values.unsqueeze(0).unsqueeze(
            0).to(torch.float32).numpy().astype(jnp.bfloat16)
        np.testing.assert_array_equal(np.asarray(passed_pixel_values),
                                      expected_pixel_values)

    def test_execute_mm_encoder_multiple_images(self):
        import torch
        """Tests _execute_mm_encoder with multiple requests and images."""
        # 1. ===== Setup =====
        self.runner.is_multimodal_model = True
        self.mock_get_mm_embed_fn = MagicMock()
        self.runner.get_multimodal_embeddings_fn = self.mock_get_mm_embed_fn

        self.runner.state = MagicMock()
        # Mock scheduler output for two requests
        mock_scheduler_output = MagicMock(spec=VllmSchedulerOutput)
        mock_scheduler_output.scheduled_encoder_inputs = {
            "req-1": [0],
            "req-2": [0]
        }

        # Mock request states
        px_1 = torch.randn(3, 224, 224, dtype=torch.bfloat16)
        grid_1 = torch.tensor([[1, 1, 1]], dtype=torch.int64)

        mm_item_1 = MultiModalKwargsItem.from_elems([
            MultiModalFieldElem("image", "pixel_values", px_1,
                                MultiModalBatchedField()),
            MultiModalFieldElem("image", "image_grid_thw", grid_1,
                                MultiModalBatchedField())
        ])

        req_state_1 = CachedRequestState(
            req_id="req-1",
            prompt_token_ids=[],
            output_token_ids=[],
            sampling_params=MagicMock(),
            block_ids=(),
            num_computed_tokens=0,
            mm_kwargs=[mm_item_1],
            mm_positions=[PlaceholderRange(offset=0, length=1)],
            lora_request=None,
            mm_hashes=[],
            pooling_params=None,
            generator=None)

        px_2 = torch.randn(3, 224, 224, dtype=torch.bfloat16)
        grid_2 = torch.tensor([[1, 2, 2]], dtype=torch.int64)
        mm_item_2 = MultiModalKwargsItem.from_elems([
            MultiModalFieldElem("image", "pixel_values", px_2,
                                MultiModalBatchedField()),
            MultiModalFieldElem("image", "image_grid_thw", grid_2,
                                MultiModalBatchedField())
        ])

        req_state_2 = CachedRequestState(
            req_id="req-2",
            prompt_token_ids=[],
            output_token_ids=[],
            sampling_params=MagicMock(),
            block_ids=(),
            num_computed_tokens=0,
            mm_kwargs=[mm_item_2],
            mm_positions=[PlaceholderRange(offset=0, length=1)],
            lora_request=None,
            mm_hashes=[],
            pooling_params=None,
            generator=None)

        self.runner.requests = {"req-1": req_state_1, "req-2": req_state_2}

        emb_1 = jnp.ones((10, 128), dtype=jnp.bfloat16)
        emb_2 = jnp.ones((20, 128), dtype=jnp.bfloat16) * 2
        self.mock_get_mm_embed_fn.return_value = (emb_1, emb_2)

        # 2. ===== Act =====
        self.runner._execute_mm_encoder(mock_scheduler_output)

        # 3. ===== Assert =====
        self.assertIn("req-1", self.runner.encoder_cache)
        np.testing.assert_array_equal(
            np.asarray(self.runner.encoder_cache["req-1"][0]),
            np.asarray(emb_1))
        self.assertIn("req-2", self.runner.encoder_cache)
        np.testing.assert_array_equal(
            np.asarray(self.runner.encoder_cache["req-2"][0]),
            np.asarray(emb_2))

        self.mock_get_mm_embed_fn.assert_called_once()
        call_args = self.mock_get_mm_embed_fn.call_args

        state_arg, grid_arg = call_args.args
        kwargs_arg = call_args.kwargs

        self.assertEqual(state_arg, self.runner.state)
        self.assertEqual(grid_arg, ((1, 1, 1), (1, 2, 2)))
        self.assertIn("pixel_values", kwargs_arg)

        passed_pixel_values = kwargs_arg['pixel_values']
        self.assertEqual(passed_pixel_values.shape, (2, 1, 3, 224, 224))

        expected_pixel_values = torch.stack(
            [px_1, px_2],
            dim=0).unsqueeze(1).to(torch.float32).numpy().astype(jnp.bfloat16)
        np.testing.assert_array_equal(np.asarray(passed_pixel_values),
                                      expected_pixel_values)

    def test_gather_mm_embeddings_chunked_prefill(self):
        """Tests _gather_mm_embeddings with chunked prefill scenarios."""
        # 1. ===== Setup =====
        self.runner.is_multimodal_model = True
        req_id = "req-1"

        # Mock encoder output
        encoder_embedding = jnp.arange(56 * 128, dtype=jnp.bfloat16).reshape(
            (56, 128))
        self.runner.encoder_cache = {req_id: {0: encoder_embedding}}

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

        # Mock request state
        req_state = CachedRequestState(
            req_id=req_id,
            prompt_token_ids=list(range(100)),
            output_token_ids=[],
            sampling_params=mock_sampling_params,
            block_ids=([], ),
            num_computed_tokens=0,  # This will be updated per step
            mm_kwargs=[],
            mm_positions=[PlaceholderRange(offset=10, length=56)],
            lora_request=None,
            mm_hashes=[],
            pooling_params=None,
            generator=None,
        )
        self.runner.requests = {req_id: req_state}
        self.runner.input_batch.add_request(req_state)

        # 2. ===== Act & Assert =====

        # ----- Step 1: First chunk of prefill -----
        req_state.num_computed_tokens = 0
        mock_scheduler_output_1 = MagicMock(spec=VllmSchedulerOutput)
        mock_scheduler_output_1.num_scheduled_tokens = {req_id: 20}

        gathered_embeds_1 = self.runner._gather_mm_embeddings(
            mock_scheduler_output_1)

        self.assertEqual(len(gathered_embeds_1), 1)
        expected_embeds_1 = encoder_embedding[0:10]
        np.testing.assert_array_equal(np.asarray(gathered_embeds_1[0]),
                                      np.asarray(expected_embeds_1))

        # ----- Step 2: Middle chunk of prefill -----
        req_state.num_computed_tokens = 20
        mock_scheduler_output_2 = MagicMock(spec=VllmSchedulerOutput)
        mock_scheduler_output_2.num_scheduled_tokens = {req_id: 30}

        gathered_embeds_2 = self.runner._gather_mm_embeddings(
            mock_scheduler_output_2)

        self.assertEqual(len(gathered_embeds_2), 1)
        expected_embeds_2 = encoder_embedding[10:40]
        np.testing.assert_array_equal(np.asarray(gathered_embeds_2[0]),
                                      np.asarray(expected_embeds_2))

        # ----- Step 3: Last chunk of prefill -----
        req_state.num_computed_tokens = 50
        mock_scheduler_output_3 = MagicMock(spec=VllmSchedulerOutput)
        mock_scheduler_output_3.num_scheduled_tokens = {req_id: 30}

        gathered_embeds_3 = self.runner._gather_mm_embeddings(
            mock_scheduler_output_3)

        self.assertEqual(len(gathered_embeds_3), 1)
        expected_embeds_3 = encoder_embedding[40:56]
        np.testing.assert_array_equal(np.asarray(gathered_embeds_3[0]),
                                      np.asarray(expected_embeds_3))

    def test_get_input_ids_embeds(self):
        """Tests _get_input_ids_embeds for both multimodal and text-only models."""
        # 1. ===== Setup =====
        dummy_input_ids = jnp.array([1, 2, 3])
        dummy_mm_embeds = [jnp.ones((10, 128))]
        dummy_final_embeds = jnp.ones((3, 128))

        # Mock the embedding function
        self.mock_get_input_embed_fn = MagicMock()
        self.runner.get_input_embeddings_fn = self.mock_get_input_embed_fn
        self.mock_get_input_embed_fn.return_value = dummy_final_embeds
        self.runner.state = MagicMock()

        # 2. ===== Act & Assert (Multimodal) =====
        self.runner.is_multimodal_model = True

        input_ids_res, inputs_embeds_res = self.runner._get_input_ids_embeds(
            dummy_input_ids, dummy_mm_embeds)

        self.assertIsNone(input_ids_res)
        np.testing.assert_array_equal(np.asarray(inputs_embeds_res),
                                      np.asarray(dummy_final_embeds))
        self.mock_get_input_embed_fn.assert_called_once_with(
            self.runner.state,
            input_ids=dummy_input_ids,
            multimodal_embeddings=dummy_mm_embeds)

        # 3. ===== Act & Assert (Text-only) =====
        self.mock_get_input_embed_fn.reset_mock()
        self.runner.is_multimodal_model = False

        input_ids_res, inputs_embeds_res = self.runner._get_input_ids_embeds(
            dummy_input_ids, dummy_mm_embeds)

        self.assertIsNone(inputs_embeds_res)
        np.testing.assert_array_equal(np.asarray(input_ids_res),
                                      np.asarray(dummy_input_ids))
        self.mock_get_input_embed_fn.assert_not_called()

    def test_calc_mrope_positions(self):
        """Tests the calculation of M-RoPE positions for mixed prompt/completion."""
        # 1. ===== Setup =====
        self.runner.uses_mrope = True
        req_id = "req-1"
        prompt_len = 20
        num_computed = 15
        num_scheduled = 10
        mrope_delta = 100

        # Mock request state with pre-computed mrope positions for the prompt
        mock_mrope_positions = np.arange(3 * prompt_len,
                                         dtype=np.int64).reshape(
                                             3, prompt_len)
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

        req_state = CachedRequestState(
            req_id=req_id,
            prompt_token_ids=list(range(prompt_len)),
            output_token_ids=[],
            sampling_params=mock_sampling_params,
            block_ids=([], ),
            num_computed_tokens=num_computed,
            mm_kwargs=[],
            mm_positions=[],
            lora_request=None,
            mm_hashes=[],
            pooling_params=None,
            generator=None,
            mrope_positions=mock_mrope_positions,
            mrope_position_delta=mrope_delta,
        )
        self.runner.requests = {req_id: req_state}
        self.runner.input_batch.add_request(req_state)
        # Manually set num_computed_tokens in the batch as add_request sets it to 0
        self.runner.input_batch.num_computed_tokens_cpu[0] = num_computed

        # Mock scheduler output
        mock_scheduler_output = MagicMock(spec=VllmSchedulerOutput)
        mock_scheduler_output.num_scheduled_tokens = {req_id: num_scheduled}

        # Patch the static method that computes completion positions
        with patch.object(MRotaryEmbedding,
                          "get_next_input_positions_tensor") as mock_get_next:
            # 2. ===== Act =====
            self.runner._calc_mrope_positions(mock_scheduler_output)

            # 3. ===== Assert =====
            # The first 5 positions should be copied from the pre-computed prompt positions
            expected_prompt_part = mock_mrope_positions[:, 15:20]
            actual_prompt_part = self.runner.mrope_positions_cpu[:, 0:5]
            np.testing.assert_array_equal(actual_prompt_part,
                                          expected_prompt_part)

            # The next 5 positions should be computed on-the-fly
            mock_get_next.assert_called_once()
            call_kwargs = mock_get_next.call_args.kwargs
            np.testing.assert_array_equal(call_kwargs["out"],
                                          self.runner.mrope_positions_cpu)
            self.assertEqual(call_kwargs["out_offset"], 5)
            self.assertEqual(call_kwargs["mrope_position_delta"], mrope_delta)
            self.assertEqual(call_kwargs["context_len"], prompt_len)
            self.assertEqual(call_kwargs["num_new_tokens"], 5)


class TestTPUJaxRunnerMultimodalModelLoadedForTextOnly(unittest.TestCase):

    def setUp(self):
        # Mock JAX dependencies
        self.mock_devices = [MagicMock()] * 4
        self.mock_mesh = MagicMock()
        self.mock_rng_key = MagicMock()

        # Setup the runner with the model_config.is_multimodal_model set to True but get_model returning None for get_multimodal_embeddings_fn and get_input_embeddings_fn.
        with patch('jax.devices', return_value=self.mock_devices), \
             patch('jax.make_mesh', return_value=self.mock_mesh), \
             patch('jax.random.key', return_value=self.mock_rng_key), \
             patch('tpu_commons.runner.jax.tpu_jax_runner.nnx.Rngs', return_value=self.mock_rng_key), \
             patch('tpu_commons.runner.jax.tpu_jax_runner.get_model', return_value=self._model_get_model()):

            model_config = ModelConfig(tokenizer_mode="auto",
                                       trust_remote_code=False,
                                       seed=0,
                                       dtype='bfloat16')
            # Set multimodal_config to not None, such that the is_multimodal_model property of model_config is True.
            model_config.multimodal_config = MagicMock()

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
            vllm_config = VllmConfig(
                model_config=model_config,
                cache_config=cache_config,
                scheduler_config=scheduler_config,
                parallel_config=parallel_config,
                speculative_config=None,
                observability_config=None,
                additional_config={},
            )

            self.runner = TPUModelRunner(vllm_config,
                                         devices=self.mock_devices)
            self.runner.load_model()

    def _model_get_model(self):
        return (
            MagicMock(),  # TPUModelRunner.model_fn
            MagicMock(),  # TPUModelRunner.compute_logits_fn
            None,  # TPUModelRunner.get_multimodal_embeddings_fn
            None,  # TPUModelRunner.get_input_embeddings_fn
            MagicMock(),  # TPUModelRunner.state (model params)
        )

    def test_is_multimodal_model(self):
        # Precondition: make sure the model_config claims the model supports MM.
        assert self.runner.model_config.is_multimodal_model

        # Precondition: load the model and returns get_multimodal_embeddings_fn as None.
        assert self.runner.get_multimodal_embeddings_fn is None

        assert not self.runner.is_multimodal_model

        self.runner.get_input_embeddings_fn = MagicMock()
        dummy_input_ids = jnp.array([1, 2, 3])
        dummy_mm_embeds = [jnp.ones((10, 128))]
        _ = self.runner._get_input_ids_embeds(dummy_input_ids, dummy_mm_embeds)
        self.runner.get_input_embeddings_fn.assert_not_called()


if __name__ == '__main__':
    unittest.main()
