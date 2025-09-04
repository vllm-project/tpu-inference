from unittest.mock import MagicMock, patch

import jax
import jax.numpy as jnp
import numpy as np
import torch
from vllm.attention import Attention
from vllm.attention.backends.abstract import AttentionType
from vllm.config import (CacheConfig, ModelConfig, ParallelConfig,
                         SchedulerConfig, VllmConfig)
from vllm.sampling_params import SamplingType
from vllm.v1.kv_cache_interface import (FullAttentionSpec, KVCacheConfig,
                                        KVCacheGroupSpec, KVCacheTensor,
                                        SlidingWindowSpec)
from vllm.v1.request import Request

from tpu_commons import utils as common_utils
from tpu_commons.runner.input_batch_jax import CachedRequestState
from tpu_commons.runner.tpu_jax_runner import TPUModelRunner


class TestKVCacheManager:

    def setup_method(self):
        # Mock JAX dependencies
        self.mock_devices = [MagicMock()] * 4
        self.mock_rng_key = MagicMock()

        # create 1x1 mesh
        devices = np.asarray(jax.devices()[:1])
        axis_names = ('data', 'model')
        mesh_shape = (1, 1)
        self.mock_mesh = jax.sharding.Mesh(devices.reshape(mesh_shape),
                                           axis_names)

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
            vllm_config = VllmConfig(
                model_config=model_config,
                cache_config=cache_config,
                scheduler_config=scheduler_config,
                parallel_config=parallel_config,
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

        prompt_len = 64

        # Populate a source KV cache with data. This represents the state
        # of the prefill runner's KV cache.
        source_kv_cache_shape = (num_blocks, self.runner.block_size,
                                 2 * num_kv_heads // 2, 2, head_size)
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
                               2 * num_kv_heads // 2, 2, head_size)
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
        assert "test_req_1" in self.runner.requests
        assert "test_req_1" in self.runner.input_batch.req_id_to_index
        assert self.runner.requests[
            "test_req_1"].num_computed_tokens == prompt_len
        assert self.runner.requests["test_req_1"].output_token_ids == [908]

        # Verify the content of the inserted KV cache.
        target_block_id = decode_block_ids[0][0]
        for i, layer_kv_cache in enumerate(self.runner.kv_caches):
            updated_block_content = layer_kv_cache[target_block_id]

            # The extracted slice should be padded to the block size.
            padding_size = self.runner.block_size - prompt_len
            expected_padded_slice = jnp.pad(extracted_kv_cache_slices[i],
                                            ((0, padding_size), (0, 0), (0, 0),
                                             (0, 0)),
                                            mode='constant')
            np.testing.assert_array_equal(updated_block_content,
                                          expected_padded_slice)

    def test_get_kv_cache_spec_with_compilation_cfg(self):
        # tests we create kv cache spec from compilation config
        # create a static forward context with
        # 10 full attention layers +
        # 10 sliding window attention layers
        # 1 layer with shared kv cache.
        num_kv_heads = 16
        head_size = 128
        attn_type = AttentionType.DECODER
        sliding_window = 10
        static_forward_context = {}
        for i in range(10):
            static_forward_context[f'layer.{i}'] = MagicMock(
                spec=Attention,
                num_kv_heads=num_kv_heads,
                head_size=head_size,
                attn_type=attn_type,
                sliding_window=None,
                kv_sharing_target_layer_name=None,
            )
        for i in range(10, 20):
            static_forward_context[f'layer.{i}'] = MagicMock(
                spec=Attention,
                num_kv_heads=num_kv_heads,
                head_size=head_size,
                attn_type=attn_type,
                sliding_window=sliding_window,
                kv_sharing_target_layer_name=None,
            )
        static_forward_context['layer.20'] = MagicMock(
            spec=Attention,
            num_kv_heads=num_kv_heads,
            head_size=head_size,
            attn_type=attn_type,
            sliding_window=None,
            kv_sharing_target_layer_name='layer.0',
        )
        self.runner.vllm_config.compilation_config.static_forward_context = \
            static_forward_context

        kv_cache_spec = self.runner.get_kv_cache_spec()

        expected_full_attn_spec = FullAttentionSpec(
            block_size=self.runner.vllm_config.cache_config.block_size,
            num_kv_heads=common_utils.get_padded_num_heads(
                num_kv_heads, self.runner.mesh.shape["model"]),
            head_size=common_utils.get_padded_head_dim(head_size),
            dtype=torch.bfloat16,
            use_mla=self.runner.vllm_config.model_config.use_mla,
        )
        expected_sliding_window_spec = SlidingWindowSpec(
            block_size=self.runner.vllm_config.cache_config.block_size,
            num_kv_heads=common_utils.get_padded_num_heads(
                num_kv_heads, self.runner.mesh.shape["model"]),
            head_size=common_utils.get_padded_head_dim(head_size),
            dtype=torch.bfloat16,
            sliding_window=sliding_window,
            use_mla=self.runner.vllm_config.model_config.use_mla,
        )
        assert len(kv_cache_spec) == 20
        for i in range(10):
            assert kv_cache_spec[f'layer.{i}'] == expected_full_attn_spec
        for i in range(10, 20):
            assert kv_cache_spec[f'layer.{i}'] == expected_sliding_window_spec
        assert 'layer.20' not in kv_cache_spec
        assert self.runner.kv_cache_manager.shared_kv_cache_layers == {
            'layer.20': 'layer.0'
        }

    def test_get_kv_cache_spec_without_compilation_cfg(self):
        # tests if there's no compilation config, we use full attention kv
        # cache for each layer.
        model_config = self.runner.vllm_config.model_config
        parallel_config = self.runner.vllm_config.parallel_config
        head_size = model_config.get_head_size()
        num_kv_heads = model_config.get_total_num_kv_heads()
        num_layers = model_config.get_num_layers(parallel_config)

        self.runner.vllm_config.compilation_config.static_forward_context = {}
        kv_cache_spec = self.runner.get_kv_cache_spec()

        assert len(kv_cache_spec) == num_layers
        expected_full_attn_spec = FullAttentionSpec(
            block_size=self.runner.vllm_config.cache_config.block_size,
            num_kv_heads=common_utils.get_padded_num_heads(
                num_kv_heads, self.runner.mesh.shape["model"]),
            head_size=common_utils.get_padded_head_dim(head_size),
            dtype=torch.bfloat16,
            use_mla=self.runner.vllm_config.model_config.use_mla,
        )
        for i in range(num_layers):
            assert kv_cache_spec[f'layer.{i}'] == expected_full_attn_spec
        assert len(self.runner.kv_cache_manager.shared_kv_cache_layers) == 0

    def test_initialize_kv_cache(self):
        # create a kv cache config with 10 layers full attention and 10 layers
        # sliding window attention.
        block_size = self.runner.vllm_config.cache_config.block_size
        num_kv_heads = 8
        head_size = 128
        use_mla = False
        sliding_window = 100
        num_blocks = 100
        kv_packing = 2  #bf16
        sliding_window_spec = SlidingWindowSpec(
            block_size=block_size,
            num_kv_heads=num_kv_heads,
            head_size=head_size,
            dtype=torch.bfloat16,
            sliding_window=sliding_window,
            use_mla=use_mla,
        )
        full_attn_spec = FullAttentionSpec(
            block_size=block_size,
            num_kv_heads=num_kv_heads,
            head_size=head_size,
            dtype=torch.bfloat16,
            use_mla=use_mla,
        )
        kv_cache_groups = [
            KVCacheGroupSpec(layer_names=[f'layer.{i}' for i in range(10)],
                             kv_cache_spec=full_attn_spec),
            KVCacheGroupSpec(layer_names=[f'layer.{i}' for i in range(10, 20)],
                             kv_cache_spec=sliding_window_spec),
        ]
        kv_cache_tensors = []
        page_size_bytes = full_attn_spec.page_size_bytes
        for i in range(10):
            kv_cache_tensors.append(
                KVCacheTensor(
                    size=num_blocks * page_size_bytes,
                    shared_by=[f'layer.{i}', f'layer.{i+10}'],
                ))
        kv_cache_config = KVCacheConfig(
            num_blocks=num_blocks,
            kv_cache_tensors=kv_cache_tensors,
            kv_cache_groups=kv_cache_groups,
        )

        original_input_batch = self.runner.input_batch
        self.runner.initialize_kv_cache(kv_cache_config)

        # assert kv cache config with multiple kv cache groups will reinit
        # input batch.
        assert original_input_batch != self.runner.input_batch
        assert len(self.runner.kv_caches) == 10
        for i in range(10):
            assert self.runner.kv_caches[i].shape == (num_blocks, block_size,
                                                      num_kv_heads * 2 //
                                                      kv_packing, kv_packing,
                                                      head_size)
            assert self.runner.layer_name_to_kvcache_index[f'layer.{i}'] == i
            assert self.runner.layer_name_to_kvcache_index[
                f'layer.{i + 10}'] == i
