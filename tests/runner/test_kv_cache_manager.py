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
import pytest
import torch
from vllm.config import (CacheConfig, ModelConfig, ParallelConfig,
                         SchedulerConfig, VllmConfig, set_current_vllm_config)
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.mamba.abstract import MambaBase
from vllm.sampling_params import SamplingType
from vllm.v1.attention.backend import AttentionType
from vllm.v1.kv_cache_interface import (FullAttentionSpec, KVCacheConfig,
                                        KVCacheGroupSpec, KVCacheTensor,
                                        MambaSpec, MLAAttentionSpec,
                                        SlidingWindowSpec)
from vllm.v1.request import Request

from tpu_inference import utils as common_utils
from tpu_inference.runner.input_batch import CachedRequestState
from tpu_inference.runner.kv_cache import get_attention_page_size_bytes
from tpu_inference.runner.tpu_runner import TPUModelRunner


class TestKVCacheManager:

    def _setup_runner(self, use_mla: bool = False):
        # Mock JAX dependencies
        self.mock_rng_key = MagicMock()

        self.mock_devices = [MagicMock(coords=i) for i in range(4)]
        self.mock_rng_key = MagicMock()

        # create 1x1 mesh
        devices = np.asarray(jax.devices()[:1])
        axis_names = ('data', 'attn_dp', 'model', 'expert')
        mesh_shape = (1, 1, 1, 1)
        self.mock_mesh = jax.sharding.Mesh(devices.reshape(mesh_shape),
                                           axis_names)

        with patch('jax.devices', return_value=self.mock_devices), \
             patch('jax.make_mesh', return_value=self.mock_mesh), \
             patch('jax.experimental.mesh_utils.create_device_mesh', return_value=self.mock_mesh), \
             patch('tpu_inference.runner.tpu_runner.TPUModelRunner._create_new_model_mesh', return_value=self.mock_mesh), \
             patch('tpu_inference.runner.tpu_runner.TPUModelRunner._init_mesh', return_value=self.mock_mesh), \
             patch('jax.random.key', return_value=self.mock_rng_key), \
             patch('tpu_inference.runner.tpu_runner.get_model', return_value=MagicMock()):

            model_config = ModelConfig()
            cache_config = CacheConfig(
                block_size=16,
                gpu_memory_utilization=0.9,
                cache_dtype="auto",
            )
            scheduler_config = SchedulerConfig(max_num_seqs=16,
                                               max_model_len=1024,
                                               is_encoder_decoder=False)
            parallel_config = ParallelConfig(
                pipeline_parallel_size=1,
                tensor_parallel_size=1,
            )
            vllm_config = VllmConfig(
                model_config=model_config,
                cache_config=cache_config,
                scheduler_config=scheduler_config,
                parallel_config=parallel_config,
                observability_config={},
                additional_config={},
            )
            self.runner = TPUModelRunner(vllm_config,
                                         devices=self.mock_devices)
            self.runner.mesh = self.mock_mesh

    @pytest.fixture(autouse=True)
    def setup_runner_fixture(self):
        self._setup_runner(use_mla=False)
        with set_current_vllm_config(self.runner.vllm_config):
            yield

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
            mm_features=[],
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

    @pytest.mark.parametrize("num_kv_heads", [16, 32])
    @pytest.mark.parametrize("head_size", [64, 100, 200])
    def test_get_kv_cache_spec_with_compilation_cfg(self, num_kv_heads,
                                                    head_size):
        # tests we create kv cache spec from compilation config
        # create a static forward context with
        # 10 full attention layers +
        # 10 sliding window attention layers
        # 1 layer with shared kv cache.
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

        block_size = self.runner.vllm_config.cache_config.block_size
        num_kv_heads = common_utils.get_padded_num_heads(
            num_kv_heads, self.runner.mesh.shape["model"])
        head_size = common_utils.get_padded_head_dim(head_size)

        expected_full_attn_spec = FullAttentionSpec(
            block_size=block_size,
            num_kv_heads=num_kv_heads,
            head_size=head_size,
            dtype=torch.bfloat16,
            page_size_padded=get_attention_page_size_bytes(
                self.runner.mesh, block_size, num_kv_heads, head_size,
                self.runner.kv_cache_dtype, False))
        expected_sliding_window_spec = SlidingWindowSpec(
            block_size=block_size,
            num_kv_heads=num_kv_heads,
            head_size=head_size,
            dtype=torch.bfloat16,
            sliding_window=sliding_window,
            page_size_padded=get_attention_page_size_bytes(
                self.runner.mesh, block_size, num_kv_heads, head_size,
                self.runner.kv_cache_dtype, False))
        assert len(kv_cache_spec) == 20
        for i in range(10):
            assert kv_cache_spec[f'layer.{i}'] == expected_full_attn_spec
        for i in range(10, 20):
            assert kv_cache_spec[f'layer.{i}'] == expected_sliding_window_spec
        assert 'layer.20' not in kv_cache_spec
        assert self.runner.kv_cache_manager.shared_kv_cache_layers == {
            'layer.20': 'layer.0'
        }

    def test_get_kv_cache_spec_with_compilation_cfg_mla(self):
        # tests we create kv cache spec from compilation config with mla
        self.runner.kv_cache_manager.use_mla = True

        # Mock hf_text_config to have kv_lora_rank and qk_rope_head_dim
        mock_hf_text_config = MagicMock()
        mock_hf_text_config.kv_lora_rank = 400
        mock_hf_text_config.qk_rope_head_dim = 40
        self.runner.model_config.hf_text_config = mock_hf_text_config

        num_kv_heads = 16
        head_size = 512  # Aggregated padding amount may be passed to the model instead.
        expected_head_size = 640  # 640 = align(512, 128) + alignto(40, 128)
        attn_type = AttentionType.DECODER
        static_forward_context = {}
        # Mock one layer, as the logic is the same for all
        mock_attn_module = MagicMock(
            spec=Attention,
            num_kv_heads=num_kv_heads,
            head_size=head_size,
            attn_type=attn_type,
            sliding_window=None,
            kv_sharing_target_layer_name=None,
        )
        mock_attn_module.use_mla = True
        static_forward_context['layer.0'] = mock_attn_module
        self.runner.vllm_config.compilation_config.static_forward_context = \
            static_forward_context

        kv_cache_spec = self.runner.get_kv_cache_spec()

        assert len(kv_cache_spec) == 1
        spec = kv_cache_spec['layer.0']
        assert isinstance(spec, MLAAttentionSpec)
        assert spec.num_kv_heads == 1
        assert spec.head_size == expected_head_size

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
        block_size = self.runner.vllm_config.cache_config.block_size
        num_kv_heads = common_utils.get_padded_num_heads(
            num_kv_heads, self.runner.mesh.shape["model"])
        head_size = common_utils.get_padded_head_dim(head_size)
        expected_full_attn_spec = FullAttentionSpec(
            block_size=block_size,
            num_kv_heads=num_kv_heads,
            head_size=head_size,
            dtype=torch.bfloat16,
            page_size_padded=get_attention_page_size_bytes(
                self.runner.mesh, block_size, num_kv_heads, head_size,
                self.runner.kv_cache_dtype, False))
        for i in range(num_layers):
            assert kv_cache_spec[f'layer.{i}'] == expected_full_attn_spec
        assert len(self.runner.kv_cache_manager.shared_kv_cache_layers) == 0

    def test_get_kv_cache_spec_without_compilation_cfg_mla(self):
        self.runner.kv_cache_manager.use_mla = True
        model_config = self.runner.vllm_config.model_config
        parallel_config = self.runner.vllm_config.parallel_config
        num_layers = model_config.get_num_layers(parallel_config)

        mock_hf_text_config = MagicMock()
        mock_hf_text_config.kv_lora_rank = 400
        mock_hf_text_config.qk_rope_head_dim = 40
        self.runner.model_config.hf_text_config = mock_hf_text_config
        expected_head_size = 640  # 640 = align(512, 128) + alignto(40, 128)

        self.runner.vllm_config.compilation_config.static_forward_context = {}
        with patch('vllm.config.ModelConfig.get_num_layers',
                   return_value=num_layers):
            kv_cache_spec = self.runner.get_kv_cache_spec()

        assert len(kv_cache_spec) == num_layers
        for i in range(num_layers):
            spec = kv_cache_spec[f"layer.{i}"]
            assert isinstance(spec, MLAAttentionSpec)
            assert spec.num_kv_heads == 1
            assert spec.head_size == expected_head_size

    def test_initialize_kv_cache(self):
        # create a kv cache config with 10 layers full attention and 10 layers
        # sliding window attention.
        block_size = self.runner.vllm_config.cache_config.block_size
        num_kv_heads = 8
        head_size = 128
        sliding_window = 100
        num_blocks = 100
        kv_packing = 2  #bf16
        sliding_window_spec = SlidingWindowSpec(
            block_size=block_size,
            num_kv_heads=num_kv_heads,
            head_size=head_size,
            dtype=torch.bfloat16,
            sliding_window=sliding_window,
        )
        full_attn_spec = FullAttentionSpec(
            block_size=block_size,
            num_kv_heads=num_kv_heads,
            head_size=head_size,
            dtype=torch.bfloat16,
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

    def test_get_kv_cache_spec_with_eagle3(self):
        # tests we create kv cache spec for eagle3 draft model
        self.runner.vllm_config.compilation_config.static_forward_context = {}
        mock_speculative_config = MagicMock()
        mock_speculative_config.method = "eagle3"
        mock_draft_model_config = MagicMock()
        mock_hf_config = MagicMock()
        mock_hf_config.num_key_value_heads = 4
        mock_hf_config.hidden_size = 1024
        mock_hf_config.num_attention_heads = 8
        mock_draft_model_config.hf_config = mock_hf_config
        mock_speculative_config.draft_model_config = mock_draft_model_config
        self.runner.speculative_config = mock_speculative_config

        kv_cache_spec = self.runner.get_kv_cache_spec()

        assert "draft_layer.0" in kv_cache_spec
        draft_spec = kv_cache_spec["draft_layer.0"]
        assert isinstance(draft_spec, FullAttentionSpec)
        assert draft_spec.block_size == self.runner.vllm_config.cache_config.block_size
        assert draft_spec.num_kv_heads == common_utils.get_padded_num_heads(
            4, self.runner.mesh.shape["model"])
        assert draft_spec.head_size == common_utils.get_padded_head_dim(128)
        assert draft_spec.dtype == torch.bfloat16

    def test_get_kv_cache_spec_with_eagle3_mla(self):
        # tests we create kv cache spec for eagle3 draft model with mla
        self.runner.kv_cache_manager.use_mla = True

        self.runner.vllm_config.compilation_config.static_forward_context = {}
        mock_speculative_config = MagicMock()
        mock_speculative_config.method = "eagle3"
        mock_draft_model_config = MagicMock()
        mock_hf_config = MagicMock()
        mock_hf_config.num_key_value_heads = 4
        mock_hf_config.hidden_size = 1024
        mock_hf_config.num_attention_heads = 8
        mock_hf_config.num_layers = 16
        model_layers = 1
        mock_hf_text_config = MagicMock()
        mock_hf_text_config.kv_lora_rank = 400
        mock_hf_text_config.qk_rope_head_dim = 40
        self.runner.model_config.hf_text_config = mock_hf_text_config
        mock_draft_model_config.hf_config = mock_hf_config
        mock_speculative_config.draft_model_config = mock_draft_model_config
        self.runner.speculative_config = mock_speculative_config

        kv_cache_spec = self.runner.get_kv_cache_spec()

        # Without compilation context, it will create specs for the main model layers
        # as well as the draft model layer.
        assert len(kv_cache_spec) > model_layers

        assert "draft_layer.0" in kv_cache_spec
        draft_spec = kv_cache_spec["draft_layer.0"]
        assert isinstance(draft_spec, FullAttentionSpec)

        for i in range(model_layers):
            assert f"layer.{i}" in kv_cache_spec
            spec = kv_cache_spec[f"layer.{i}"]
            assert isinstance(spec, MLAAttentionSpec)
            assert spec.num_kv_heads == 1

    def test_delete_kv_cache(self):
        """Test that delete_kv_cache deletes JAX arrays and clears state."""
        # First, initialize KV cache using the same setup as
        # test_initialize_kv_cache.
        block_size = self.runner.vllm_config.cache_config.block_size
        num_kv_heads = 8
        head_size = 128
        num_blocks = 100
        full_attn_spec = FullAttentionSpec(
            block_size=block_size,
            num_kv_heads=num_kv_heads,
            head_size=head_size,
            dtype=torch.bfloat16,
        )
        kv_cache_groups = [
            KVCacheGroupSpec(layer_names=[f'layer.{i}' for i in range(10)],
                             kv_cache_spec=full_attn_spec),
        ]
        page_size_bytes = full_attn_spec.page_size_bytes
        kv_cache_tensors = [
            KVCacheTensor(
                size=num_blocks * page_size_bytes,
                shared_by=[f'layer.{i}'],
            ) for i in range(10)
        ]
        kv_cache_config = KVCacheConfig(
            num_blocks=num_blocks,
            kv_cache_tensors=kv_cache_tensors,
            kv_cache_groups=kv_cache_groups,
        )

        self.runner.initialize_kv_cache(kv_cache_config)
        assert len(self.runner.kv_caches) == 10
        assert len(self.runner.layer_name_to_kvcache_index) == 10

        # Now reset.
        self.runner.delete_kv_cache()

        assert len(self.runner.kv_caches) == 0
        assert len(self.runner.layer_name_to_kvcache_index) == 0

    def test_reinitialize_kv_cache(self):
        """Test that reinitialize_kv_cache reallocates fresh KV cache."""
        block_size = self.runner.vllm_config.cache_config.block_size
        num_kv_heads = 8
        head_size = 128
        num_blocks = 100
        kv_packing = 2  # bf16
        full_attn_spec = FullAttentionSpec(
            block_size=block_size,
            num_kv_heads=num_kv_heads,
            head_size=head_size,
            dtype=torch.bfloat16,
        )
        kv_cache_groups = [
            KVCacheGroupSpec(layer_names=[f'layer.{i}' for i in range(10)],
                             kv_cache_spec=full_attn_spec),
        ]
        page_size_bytes = full_attn_spec.page_size_bytes
        kv_cache_tensors = [
            KVCacheTensor(
                size=num_blocks * page_size_bytes,
                shared_by=[f'layer.{i}'],
            ) for i in range(10)
        ]
        kv_cache_config = KVCacheConfig(
            num_blocks=num_blocks,
            kv_cache_tensors=kv_cache_tensors,
            kv_cache_groups=kv_cache_groups,
        )

        self.runner.initialize_kv_cache(kv_cache_config)
        assert len(self.runner.kv_caches) == 10

        # Reset and then reinitialize.
        self.runner.delete_kv_cache()
        assert len(self.runner.kv_caches) == 0

        self.runner.reinitialize_kv_cache()
        assert len(self.runner.kv_caches) == 10
        for i in range(10):
            assert self.runner.kv_caches[i].shape == (num_blocks, block_size,
                                                      num_kv_heads * 2 //
                                                      kv_packing, kv_packing,
                                                      head_size)
            assert self.runner.layer_name_to_kvcache_index[f'layer.{i}'] == i

    def test_reinitialize_kv_cache_without_init_raises(self):
        """Test that reinitialize raises if initialize was never called."""
        # kv_cache_config is not set on a fresh runner.
        with pytest.raises(RuntimeError, match="Cannot reinitialize KV cache"):
            self.runner.reinitialize_kv_cache()

    def test_delete_kv_cache_no_op_when_empty(self):
        """Test that delete_kv_cache is safe to call when no KV cache exists."""
        assert len(self.runner.kv_caches) == 0
        # Should not raise.
        self.runner.delete_kv_cache()
        assert len(self.runner.kv_caches) == 0

    def test_get_kv_cache_spec_with_compilation_cfg_mamba(self):
        mock_attn_module = MagicMock(spec=MambaBase)

        mock_mamba_spec = MagicMock(spec=MambaSpec)
        mock_attn_module.get_kv_cache_spec.return_value = mock_mamba_spec

        def get_layers_side_effect(vllm_config, attn_cls):
            if MambaBase in attn_cls:
                return {'layer.0': mock_attn_module}
            return {}

        self.runner.vllm_config.compilation_config.static_forward_context = {
            'layer.0': mock_attn_module
        }

        with patch(
                'tpu_inference.runner.kv_cache_manager.get_layers_from_vllm_config',
                side_effect=get_layers_side_effect):
            kv_cache_spec = self.runner.get_kv_cache_spec()

        assert len(kv_cache_spec) == 1
        assert kv_cache_spec['layer.0'] == mock_mamba_spec

    def test_initialize_kv_cache_mamba(self):
        num_blocks = 100
        page_size_bytes = 16 * 1024

        mamba_spec = MagicMock(spec=MambaSpec)
        mamba_spec.block_size = self.runner.vllm_config.cache_config.block_size
        mamba_spec.page_size_bytes = page_size_bytes
        mamba_spec.shapes = [(4, 128), (8, 64, 32)]
        mamba_spec.dtypes = [torch.bfloat16, torch.float32]

        layer_names = ['layer.0', 'layer.1']
        kv_cache_groups = [
            KVCacheGroupSpec(layer_names=layer_names,
                             kv_cache_spec=mamba_spec),
        ]
        kv_cache_tensors = [
            KVCacheTensor(
                size=num_blocks * page_size_bytes,
                shared_by=layer_names,
            )
        ]
        kv_cache_config = KVCacheConfig(
            num_blocks=num_blocks,
            kv_cache_tensors=kv_cache_tensors,
            kv_cache_groups=kv_cache_groups,
        )

        if not hasattr(self.runner.vllm_config, 'sharding_config'
                       ) or self.runner.vllm_config.sharding_config is None:
            self.runner.vllm_config.sharding_config = MagicMock()
            self.runner.vllm_config.sharding_config.total_dp_size = 1

        with patch('dataclasses.replace') as mock_replace:
            mock_replaced_spec = MagicMock()
            mock_replaced_spec.page_size_bytes = page_size_bytes
            mock_replace.return_value = mock_replaced_spec
            self.runner.initialize_kv_cache(kv_cache_config)

        assert len(self.runner.kv_caches) == 2
        for i in range(2):
            mamba_states = self.runner.kv_caches[i]
            assert isinstance(mamba_states, tuple)
            assert len(mamba_states) == 2

            expected_num_blocks = num_blocks // len(layer_names)
            assert mamba_states[0].shape == (expected_num_blocks, 4, 128)
            assert mamba_states[1].shape == (expected_num_blocks, 8, 64, 32)

            assert self.runner.layer_name_to_kvcache_index[f'layer.{i}'] == i

    def test_initialize_kv_cache_no_duplicate_shared_layers(self):
        block_size = self.runner.vllm_config.cache_config.block_size
        num_kv_heads = 8
        head_size = 128
        num_blocks = 100
        kv_packing = 2  #bf16

        full_attn_spec = FullAttentionSpec(
            block_size=block_size,
            num_kv_heads=num_kv_heads,
            head_size=head_size,
            dtype=torch.bfloat16,
        )
        layer_names = [f'layer.{i}' for i in range(4)]
        kv_cache_groups = [
            KVCacheGroupSpec(layer_names=layer_names,
                             kv_cache_spec=full_attn_spec),
        ]

        page_size_bytes = full_attn_spec.page_size_bytes
        kv_cache_tensors = [
            KVCacheTensor(
                size=num_blocks * page_size_bytes,
                shared_by=layer_names,
            )
        ]

        kv_cache_config = KVCacheConfig(
            num_blocks=num_blocks,
            kv_cache_tensors=kv_cache_tensors,
            kv_cache_groups=kv_cache_groups,
        )

        if not hasattr(self.runner.vllm_config, 'sharding_config'
                       ) or self.runner.vllm_config.sharding_config is None:
            self.runner.vllm_config.sharding_config = MagicMock()
            self.runner.vllm_config.sharding_config.total_dp_size = 1

        self.runner.initialize_kv_cache(kv_cache_config)

        # it should initialize 1 KV cache shared by all 4 layers
        assert len(self.runner.kv_caches) == 1

        assert self.runner.kv_caches[0].shape == (num_blocks, block_size,
                                                  num_kv_heads * 2 //
                                                  kv_packing, kv_packing,
                                                  head_size)

        # Ensure all layer indices map to the same underlying KV cache (0)
        for i in range(4):
            assert self.runner.layer_name_to_kvcache_index[f'layer.{i}'] == 0

    def test_initialize_kv_cache_mamba_duplicate_fallback(self):
        num_blocks = 100
        page_size_bytes = 16 * 1024

        mamba_spec = MagicMock(spec=MambaSpec)
        mamba_spec.block_size = self.runner.vllm_config.cache_config.block_size
        mamba_spec.page_size_bytes = page_size_bytes
        mamba_spec.shapes = [(4, 128), (8, 64, 32)]
        mamba_spec.dtypes = [torch.bfloat16, torch.float32]

        layer_names = [f'layer.{i}' for i in range(4)]
        kv_cache_groups = [
            KVCacheGroupSpec(layer_names=layer_names,
                             kv_cache_spec=mamba_spec),
        ]
        kv_cache_tensors = [
            KVCacheTensor(
                size=num_blocks * page_size_bytes,
                shared_by=layer_names,
            )
        ]
        kv_cache_config = KVCacheConfig(
            num_blocks=num_blocks,
            kv_cache_tensors=kv_cache_tensors,
            kv_cache_groups=kv_cache_groups,
        )

        if not hasattr(self.runner.vllm_config, 'sharding_config'
                       ) or self.runner.vllm_config.sharding_config is None:
            self.runner.vllm_config.sharding_config = MagicMock()
            self.runner.vllm_config.sharding_config.total_dp_size = 1

        with patch('tpu_inference.runner.kv_cache_manager.logger.warning_once'
                   ) as mock_warning_once, patch(
                       'dataclasses.replace') as mock_replace:
            mock_replaced_spec = MagicMock()
            mock_replaced_spec.page_size_bytes = page_size_bytes
            mock_replace.return_value = mock_replaced_spec
            self.runner.initialize_kv_cache(kv_cache_config)

        # Even though DUPLICATE_SHARED_KV_CACHE_LAYERS=False, Mamba does not
        # support shared layers right now, so we force it to fallback to True.
        assert len(self.runner.kv_caches) == 4

        # Verify the warning is triggered exactly once for the fallback
        mock_warning_once.assert_called_once_with(
            "MambaSpec does not support shared layers for now, defaulting to single KV cache per layer..."
        )

        for i in range(4):
            mamba_states = self.runner.kv_caches[i]
            assert isinstance(mamba_states, tuple)
            assert len(mamba_states) == 2
            assert self.runner.layer_name_to_kvcache_index[f'layer.{i}'] == i

    def test_get_kv_cache_spec_hybrid_mamba_cache_config_updates(self):
        self.runner.cache_config.block_size = 1056
        self.runner.cache_config.mamba_block_size = 1056
        self.runner.kv_cache_dtype = torch.float8_e4m3fn
        self.runner.cache_config.mamba_page_size_padded = 533504

        class DummyMamba(MambaBase):

            def __init__(self):
                super().__init__()

            def get_state_shape(self):
                return ((3, 12288), (64, 128, 128))

            def get_state_dtype(self):
                return (torch.bfloat16, torch.float32)

            @property
            def mamba_type(self):
                return "dummy"

        mock_mamba = DummyMamba()
        mock_attn = MagicMock(spec=Attention)
        mock_attn.attn_type = AttentionType.DECODER
        mock_attn.num_kv_heads = 2
        mock_attn.head_size = 256
        mock_attn.sliding_window = None
        mock_attn.kv_sharing_target_layer_name = None

        layers = {
            'linear_attn': mock_mamba,
            'full_attn': mock_attn,
        }
        self.runner.vllm_config.compilation_config.static_forward_context = layers

        # Unpadded mamba page size for DummyMamba: (3*12288)*2 + (64*128*128)*4
        expected_mamba_unpadded = 3 * 12288 * 2 + 64 * 128 * 128 * 4
        expected_attn_page_size = 1081344  # what update_mamba_page_size_padded
        # previously set. For this 1:1 hybrid the new uniform page size is
        # attn_page + mamba_unpadded.
        expected_uniform = expected_attn_page_size + expected_mamba_unpadded

        with patch(
                'tpu_inference.runner.kv_cache_manager.get_layers_from_vllm_config',
                return_value=layers):
            kv_cache_spec = self.runner.get_kv_cache_spec()

            # Every layer spec (both mamba and attn) is padded to the
            # per-shared_by sum so vLLM sees a uniform page size and sizes
            # its block pool to match per-layer TPU allocation.
            assert self.runner.cache_config.mamba_page_size_padded == \
                expected_uniform

            mamba_spec = kv_cache_spec['linear_attn']
            assert isinstance(mamba_spec, MambaSpec)
            assert mamba_spec.page_size_padded == expected_uniform

            attn_spec = kv_cache_spec['full_attn']
            assert isinstance(attn_spec, FullAttentionSpec)
            assert attn_spec.page_size_padded == expected_uniform
            # Uniform check: vLLM would fail otherwise.
            assert attn_spec.page_size_bytes == mamba_spec.page_size_bytes

    def test_get_kv_cache_spec_hybrid_uniform_page_size_qwen35_ratio(self):
        """Verify update_mamba_page_size_padded for a 10:30 attn:mamba model.

        For Qwen3.5 (10 full-attn layers + 30 linear-attn layers), vLLM's
        hybrid grouping produces `group_size=10` and 4 kv-cache groups
        (1 attn + 3 mamba). The fix pads each layer spec to
        `attn_page + 3 * mamba_unpadded` so vLLM's num_blocks matches
        per-layer TPU allocation.
        """
        self.runner.cache_config.block_size = 1056
        self.runner.cache_config.mamba_block_size = 1056
        self.runner.kv_cache_dtype = torch.float8_e4m3fn
        self.runner.cache_config.mamba_page_size_padded = None

        class DummyMamba(MambaBase):

            def __init__(self):
                super().__init__()

            def get_state_shape(self):
                return ((3, 12288), (64, 128, 128))

            def get_state_dtype(self):
                return (torch.bfloat16, torch.float32)

            @property
            def mamba_type(self):
                return "dummy"

        layers: dict = {}
        for i in range(30):
            layers[f'linear_attn.{i}'] = DummyMamba()
        for i in range(10):
            mock_attn = MagicMock(spec=Attention)
            mock_attn.attn_type = AttentionType.DECODER
            mock_attn.num_kv_heads = 2
            mock_attn.head_size = 256
            mock_attn.sliding_window = None
            mock_attn.kv_sharing_target_layer_name = None
            layers[f'full_attn.{i}'] = mock_attn

        mamba_unpadded = 3 * 12288 * 2 + 64 * 128 * 128 * 4
        attn_page = 1081344
        expected_uniform = attn_page + 3 * mamba_unpadded

        self.runner.vllm_config.compilation_config.static_forward_context = \
            layers

        with patch(
                'tpu_inference.runner.kv_cache_manager.get_layers_from_vllm_config',
                return_value=layers):
            kv_cache_spec = self.runner.get_kv_cache_spec()

        assert self.runner.cache_config.mamba_page_size_padded == \
            expected_uniform, (
                f"expected {expected_uniform}, "
                f"got {self.runner.cache_config.mamba_page_size_padded}")
        for name, spec in kv_cache_spec.items():
            assert spec.page_size_bytes == expected_uniform, (
                f"layer {name} has page_size_bytes={spec.page_size_bytes} "
                f"but expected uniform={expected_uniform}")

    def test_get_kv_cache_spec_pure_attention_no_cache_config_updates(self):
        mock_attn = MagicMock(spec=MambaBase)
        layers = {'layer.0': mock_attn}
        self.runner.vllm_config.compilation_config.static_forward_context = layers

        with patch('tpu_inference.runner.kv_cache_manager.get_layers_from_vllm_config', return_value=layers), \
             patch.object(self.runner.kv_cache_manager, 'update_mamba_page_size_padded') as mock_update:
            mock_update.assert_not_called()

    def test_hybrid_mamba_num_blocks(self):
        num_blocks = 100
        # The duplicate-path num_blocks calc now uses the TPU-actual
        # attention per-block size (`get_attention_page_size_bytes`) on
        # both sides of the comparison — so the mock spec's `num_kv_heads`,
        # `head_size`, `block_size`, and `dtype` have to produce the
        # expected `attn_page_size` when fed through that function.
        attn_spec_block_size = self.runner.vllm_config.cache_config.block_size
        attn_spec_num_kv_heads = 8
        attn_spec_head_size = 128
        attn_spec_dtype = torch.bfloat16
        attn_page_size = get_attention_page_size_bytes(self.runner.mesh,
                                                       attn_spec_block_size,
                                                       attn_spec_num_kv_heads,
                                                       attn_spec_head_size,
                                                       attn_spec_dtype, False)

        mamba_shapes = ((4, 128), (8, 32, 32))
        mamba_dtypes = (torch.bfloat16, torch.float32)

        mamba_unpadded_page_size = sum(
            int(np.prod(shape)) * torch.tensor([], dtype=dtype).element_size()
            for shape, dtype in zip(mamba_shapes, mamba_dtypes))

        mamba_spec = MambaSpec(
            block_size=self.runner.vllm_config.cache_config.block_size,
            shapes=mamba_shapes,
            dtypes=mamba_dtypes,
            page_size_padded=attn_page_size)

        # Total tensor size derived from sum of the unpadded Mamba page size and the Attention page size
        tensor_size = num_blocks * (mamba_unpadded_page_size + attn_page_size)

        attn_spec = MagicMock(spec=FullAttentionSpec)
        attn_spec.block_size = attn_spec_block_size
        attn_spec.page_size_bytes = attn_page_size
        attn_spec.num_kv_heads = attn_spec_num_kv_heads
        attn_spec.head_size = attn_spec_head_size
        attn_spec.dtype = attn_spec_dtype

        layer_names = ['layer.0', 'layer.1']
        kv_cache_groups = [
            KVCacheGroupSpec(layer_names=['layer.0'],
                             kv_cache_spec=mamba_spec),
            KVCacheGroupSpec(layer_names=['layer.1'], kv_cache_spec=attn_spec),
        ]
        kv_cache_tensors = [
            KVCacheTensor(
                size=tensor_size,
                shared_by=layer_names,
            )
        ]
        kv_cache_config = KVCacheConfig(
            num_blocks=num_blocks,
            kv_cache_tensors=kv_cache_tensors,
            kv_cache_groups=kv_cache_groups,
        )

        self.runner.initialize_kv_cache(kv_cache_config)

        # Should duplicate the shared cache and allocate successfully with the appropriate num_blocks
        assert len(self.runner.kv_caches) == 2

        # Check that num_blocks are set using unpadded mamba page size.
        mamba_states = self.runner.kv_caches[0]
        assert len(mamba_states) == 2
        assert mamba_states[0].shape[0] == num_blocks
        assert mamba_states[1].shape[0] == num_blocks

        attn_cache = self.runner.kv_caches[1]
        assert attn_cache.shape[0] == num_blocks

    def test_hybrid_num_blocks_matches_vllm_pool_for_qwen35_topology(self):
        """Regression test for the OOB block-id bug observed on Qwen3.5.

        vLLM's hybrid allocator groups Qwen3.5's 40 layers into 4 kv-cache
        groups (1 full-attn + 3 linear-attn/mamba), then builds
        `group_size=10` `KVCacheTensor`s, each `shared_by` one layer per
        group (so shared_by has 4 layer names). Block ids are drawn from a
        single shared pool of size `kv_cache_config.num_blocks`. Before the
        fix, per-layer num_blocks on TPU was `tensor.size /
        total_group_page_size` (~vllm_num_blocks / 3.5 for Qwen3.5), so the
        scheduler could hand out a block id beyond a layer's cache range.

        After the fix, every layer spec is padded so
        `tensor.size / uniform_page_size == kv_cache_config.num_blocks`, and
        each duplicated per-layer cache has exactly that many slots.
        """
        num_blocks = 7  # matches gpu-memory-utilization=0.36 in the repro

        # Attention per-block size comes from `get_attention_page_size_bytes`
        # (same as what `update_mamba_page_size_padded` uses) — on fp8
        # models this differs from `spec.real_page_size_bytes` because of
        # TPU packing, so stay consistent.
        attn_spec_block_size = self.runner.vllm_config.cache_config.block_size
        attn_spec_num_kv_heads = 8
        attn_spec_head_size = 128
        attn_spec_dtype = torch.bfloat16
        attn_page_size = get_attention_page_size_bytes(self.runner.mesh,
                                                       attn_spec_block_size,
                                                       attn_spec_num_kv_heads,
                                                       attn_spec_head_size,
                                                       attn_spec_dtype, False)

        mamba_shapes = ((3, 12288), (64, 128, 128))
        mamba_dtypes = (torch.bfloat16, torch.float32)
        mamba_unpadded = sum(
            int(np.prod(shape)) * torch.tensor([], dtype=dtype).element_size()
            for shape, dtype in zip(mamba_shapes, mamba_dtypes))

        # 1 full-attn + 3 linear-attn per shared_by, so the spec padding
        # applied by the fix is (1*attn + 3*mamba_unpadded).
        uniform_page_size = attn_page_size + 3 * mamba_unpadded

        # vLLM computes this for hybrid: tensor.size = uniform * num_blocks.
        tensor_size = uniform_page_size * num_blocks

        # Every layer spec reports `page_size_bytes == uniform_page_size`
        # after the fix (MambaSpec via page_size_padded, AttentionSpec via
        # page_size_padded as well).
        mamba_spec = MambaSpec(
            block_size=self.runner.vllm_config.cache_config.block_size,
            shapes=mamba_shapes,
            dtypes=mamba_dtypes,
            page_size_padded=uniform_page_size,
        )

        attn_spec = MagicMock(spec=FullAttentionSpec)
        attn_spec.block_size = attn_spec_block_size
        attn_spec.page_size_bytes = uniform_page_size
        attn_spec.num_kv_heads = attn_spec_num_kv_heads
        attn_spec.head_size = attn_spec_head_size
        attn_spec.dtype = attn_spec_dtype

        # 10 shared_by tensors, each holding 1 attn + 3 mamba layers
        # (Qwen3.5: 10 full-attn, 30 linear-attn).
        layer_names_per_tensor = [(f'attn.{i}', f'mamba_a.{i}', f'mamba_b.{i}',
                                   f'mamba_c.{i}') for i in range(10)]
        attn_layer_names = [t[0] for t in layer_names_per_tensor]
        mamba_a_names = [t[1] for t in layer_names_per_tensor]
        mamba_b_names = [t[2] for t in layer_names_per_tensor]
        mamba_c_names = [t[3] for t in layer_names_per_tensor]

        kv_cache_groups = [
            KVCacheGroupSpec(layer_names=attn_layer_names,
                             kv_cache_spec=attn_spec),
            KVCacheGroupSpec(layer_names=mamba_a_names,
                             kv_cache_spec=mamba_spec),
            KVCacheGroupSpec(layer_names=mamba_b_names,
                             kv_cache_spec=mamba_spec),
            KVCacheGroupSpec(layer_names=mamba_c_names,
                             kv_cache_spec=mamba_spec),
        ]
        kv_cache_tensors = [
            KVCacheTensor(size=tensor_size, shared_by=list(names))
            for names in layer_names_per_tensor
        ]
        kv_cache_config = KVCacheConfig(
            num_blocks=num_blocks,
            kv_cache_tensors=kv_cache_tensors,
            kv_cache_groups=kv_cache_groups,
        )

        self.runner.initialize_kv_cache(kv_cache_config)

        # 10 tensors × 4 duplicated layers = 40 caches, one per layer.
        assert len(self.runner.kv_caches) == 40

        # The whole point of the fix: every layer's physical cache holds
        # exactly `kv_cache_config.num_blocks` slots so block ids from
        # vLLM's shared pool can never index out of range.
        for name in (attn_layer_names + mamba_a_names + mamba_b_names +
                     mamba_c_names):
            idx = self.runner.layer_name_to_kvcache_index[name]
            cache = self.runner.kv_caches[idx]
            if isinstance(cache, tuple):  # mamba: (conv_state, ssm_state)
                for state in cache:
                    assert state.shape[0] == num_blocks, (
                        f"layer {name} mamba state has "
                        f"{state.shape[0]} blocks but vLLM pool has "
                        f"{num_blocks}")
            else:
                assert cache.shape[0] == num_blocks, (
                    f"layer {name} attn cache has {cache.shape[0]} blocks "
                    f"but vLLM pool has {num_blocks}")
