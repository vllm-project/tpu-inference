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
from vllm.config import (CacheConfig, ModelConfig, ParallelConfig,
                         SchedulerConfig, SpeculativeConfig, VllmConfig)
from vllm.config.multimodal import BaseDummyOptions

from tpu_inference.runner.tpu_runner import TPUModelRunner


class TestTPUJaxRunner:

    def setup_method(self):
        # Mock JAX dependencies
        self.mock_devices = [MagicMock(coords=i) for i in range(1)]
        self.mock_rng_key = MagicMock()
        device_array = np.array(jax.devices()[:1]).reshape(1, 1, 1, -1)
        self.mock_mesh = jax.make_mesh(device_array.shape,
                                       ('data', 'attn_dp', 'expert', 'model'))
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
                cache_dtype="auto",
            )
            scheduler_config = SchedulerConfig(max_num_seqs=16,
                                               max_model_len=1024,
                                               is_encoder_decoder=False)
            parallel_config = ParallelConfig(
                pipeline_parallel_size=1,
                tensor_parallel_size=1,
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

    def test_get_supported_tasks_runner(self):
        """Test get_supported_tasks for generate runner type."""
        supported_tasks = self.runner.get_supported_tasks()
        assert supported_tasks == ("generate", )

    def test_get_input_ids_embeds(self):
        """Tests _get_input_ids_embeds for both multimodal and text-only models."""
        # 1. ===== Setup =====
        dummy_input_ids = jnp.array([1, 2, 3])
        dummy_mm_embeds = jnp.ones((10, 128))
        dummy_is_mm_embed = jnp.array([False, True, True], dtype=jnp.bool_)
        dummy_final_embeds = jnp.ones((3, 128))

        # Mock the embedding function
        self.mock_get_input_embed_fn = MagicMock()
        self.runner.embed_input_ids_fn = self.mock_get_input_embed_fn
        self.mock_get_input_embed_fn.return_value = dummy_final_embeds
        self.runner.state = MagicMock()

        # 2. ===== Act & Assert (Multimodal) =====
        self.runner.is_multimodal_model = True

        input_ids_res, inputs_embeds_res = self.runner._get_input_ids_embeds(
            dummy_input_ids, dummy_mm_embeds, dummy_is_mm_embed)

        assert input_ids_res is None
        np.testing.assert_array_equal(np.asarray(inputs_embeds_res),
                                      np.asarray(dummy_final_embeds))
        self.mock_get_input_embed_fn.assert_called_once_with(
            self.runner.state,
            dummy_input_ids,
            dummy_mm_embeds,
            is_multimodal=dummy_is_mm_embed)

        # 3. ===== Act & Assert (Multimodal w/o mm embeds) =====
        self.mock_get_input_embed_fn.reset_mock()
        self.runner.is_multimodal_model = True

        # Without mm_embeds in the current scheduled tokens
        input_ids_res, inputs_embeds_res = self.runner._get_input_ids_embeds(
            dummy_input_ids, None, None)

        assert inputs_embeds_res is None
        np.testing.assert_array_equal(np.asarray(input_ids_res),
                                      np.asarray(dummy_input_ids))
        self.mock_get_input_embed_fn.assert_not_called()

        # 4. ===== Act & Assert (Text-only) =====
        self.mock_get_input_embed_fn.reset_mock()
        self.runner.is_multimodal_model = False

        input_ids_res, inputs_embeds_res = self.runner._get_input_ids_embeds(
            dummy_input_ids, dummy_mm_embeds, dummy_is_mm_embed)

        assert inputs_embeds_res is None
        np.testing.assert_array_equal(np.asarray(input_ids_res),
                                      np.asarray(dummy_input_ids))
        self.mock_get_input_embed_fn.assert_not_called()

    @patch('tpu_inference.runner.tpu_runner.TPUSupportedSamplingMetadata')
    def test_prepare_inputs_hybrid_kvcache(self, mock_sampling_metadata):
        # create hybrid kv cache config
        # 20 layers, 10 full attn + 10 sw attn
        self._create_mock_hybrid_kv_cache_config()

        # Mock scheduler output.
        scheduler_output = MagicMock()
        scheduler_output.total_num_scheduled_tokens = 10
        scheduler_output.num_scheduled_tokens = {'req1': 10}
        scheduler_output.scheduled_spec_decode_tokens = {}
        scheduler_output.grammar_bitmask = None

        # Mock input_batch
        self.runner.input_batch = MagicMock()
        self.runner.input_batch.num_reqs = 1
        self.runner.input_batch.req_ids = ['req1']
        self.runner.input_batch.req_id_to_index = {'req1': 0}
        self.runner.input_batch.num_computed_tokens_cpu = np.array([10])
        self.runner.input_batch.token_ids_cpu = np.random.randint(
            0, 1000, (8, 64), dtype=np.int32)

        # Mock block tables
        # there will be 2 block tables since there are 2 kv cache groups
        mock_block_table = MagicMock()
        mock_block_table.max_num_blocks_per_req = 8
        mock_block_table.get_cpu_tensor.return_value = np.zeros((1, 8),
                                                                dtype=np.int32)
        self.runner.input_batch.block_table = [
            mock_block_table, mock_block_table
        ]

        mock_sampling_instance = MagicMock()
        mock_sampling_metadata.from_input_batch.return_value = mock_sampling_instance

        output = self.runner._prepare_inputs_non_dp(scheduler_output)
        assert len(output) == 8
        input_ids, positions, attention_metadata, sampling_metadata, logits_indices, spec_decode_metadata, logits_indices_selector, padded_num_reqs = output
        # assert it will create attention metadata for each layer.
        assert isinstance(attention_metadata, dict)
        assert len(attention_metadata) == 20

    def _create_mock_hybrid_kv_cache_config(self):
        mock_kv_cache_config = MagicMock()
        mock_kv_cache_group1 = MagicMock()
        mock_kv_cache_group1.layer_names = [f'layer.{i}' for i in range(10)]
        mock_kv_cache_group2 = MagicMock()
        mock_kv_cache_group2.layer_names = [
            f'layer.{i}' for i in range(10, 20)
        ]
        mock_kv_cache_config.kv_cache_groups = [
            mock_kv_cache_group1, mock_kv_cache_group2
        ]
        self.runner.kv_cache_config = mock_kv_cache_config
        self.runner.use_hybrid_kvcache = True

    @patch(
        'vllm.model_executor.layers.fused_moe.routed_experts_capturer.RoutedExpertsCapturer'
    )
    def test_execute_model_moe_experts(self, mock_capturer_class):
        """Test that _execute_model correctly stores experts in the shared buffer."""
        # 1. ===== Setup =====
        mock_capturer = MagicMock()
        mock_capturer_class.get_instance.return_value = mock_capturer
        self.runner.experts_capturer = mock_capturer

        max_tokens = 10
        num_layers = 2
        top_k = 1
        dummy_buffer = np.zeros((max_tokens, num_layers, top_k),
                                dtype=np.int32)
        mock_capturer._host_buffer_view = dummy_buffer
        mock_capturer._lock_file = "dummy.lock"

        dummy_experts_layer0 = jnp.array([[72], [73]], dtype=jnp.int32)
        dummy_experts_layer1 = jnp.array([[82], [83]], dtype=jnp.int32)

        self.runner.model_fn = MagicMock()
        self.runner.model_fn.return_value = (
            MagicMock(),  # kv_caches
            MagicMock(),  # hidden_states
            MagicMock(),  # aux_hidden_states
            [dummy_experts_layer0, dummy_experts_layer1]  # all_experts
        )

        self.runner.input_batch = MagicMock()
        self.runner.input_batch.num_reqs = 1
        self.runner.input_batch.req_ids = ['req1']
        self.runner.input_batch.num_computed_tokens_cpu = np.array([0])
        self.runner.input_batch.token_ids_cpu = np.zeros((8, 64),
                                                         dtype=np.int32)
        self.runner.input_batch.max_num_logprobs = 0
        self.runner.input_batch.request_distribution = []
        self.runner.get_mrope_input_positions_fn = None
        self.runner.state = MagicMock()

        mock_kv_cache_config = MagicMock()
        mock_kv_cache_config.kv_cache_groups = [MagicMock()]
        self.runner.kv_cache_config = mock_kv_cache_config

        self.runner._prepare_inputs_non_dp = MagicMock(return_value=(
            jnp.zeros((2, ), dtype=jnp.int32),  # input_ids
            jnp.zeros((2, ), dtype=jnp.int32),  # positions
            {},  # attention_metadata
            MagicMock(),  # sampling_metadata
            jnp.zeros((2, ), dtype=jnp.int32),  # logits_indices
            None,  # spec_decode_metadata
            None,  # logits_indices_selector
            1,  # padded_num_reqs
        ))

        self.runner._select_from_array_fn = MagicMock(
            side_effect=lambda x, *args, **kwargs: x)
        self.runner.compute_logits_fn = MagicMock(
            return_value=jnp.zeros((2, 1000), dtype=jnp.float32))
        self.runner.pooler_fn = MagicMock()
        self.runner.combine_hidden_states_fn = MagicMock()
        self.runner.lora_manager = None
        self.runner.model = None

        scheduler_output = MagicMock()
        scheduler_output.total_num_scheduled_tokens = 2
        scheduler_output.num_scheduled_tokens = {'req1': 2}

        dummy_slots = np.array([2, 5], dtype=np.int32)
        self.runner._get_slot_mapping = MagicMock(return_value=dummy_slots)

        # 2. ===== Act =====
        with patch(
                'vllm.model_executor.layers.fused_moe.routed_experts_capturer._file_lock',
                return_value=MagicMock()):
            self.runner._execute_model(scheduler_output)

        # 3. ===== Assert =====
        # The buffer dimensions are (Token_Slot, Layer, Expert_Rank)
        # We mocked _get_slot_mapping to return [2, 5] for the 2 tokens in the batch.

        # Token at slot 2, Layer 0, Top-1 expert is 72
        assert dummy_buffer[2, 0, 0] == 72
        # Token at slot 5, Layer 0, Top-1 expert is 73
        assert dummy_buffer[5, 0, 0] == 73
        # Token at slot 2, Layer 1, Top-1 expert is 82
        assert dummy_buffer[2, 1, 0] == 82
        # Token at slot 5, Layer 1, Top-1 expert is 83
        assert dummy_buffer[5, 1, 0] == 83


class TestTPUJaxRunnerMultimodalModelLoadedForTextOnly:

    def setup_method(self):
        # Mock JAX dependencies
        self.mock_devices = [MagicMock(coords=i) for i in range(4)]
        self.mock_rng_key = MagicMock()
        device_array = np.array(jax.devices()[:1]).reshape(1, 1, 1, -1)
        self.mock_mesh = jax.make_mesh(device_array.shape,
                                       ('data', 'attn_dp', 'expert', 'model'))
        # Setup the runner with the model_config.is_multimodal_model set to True but get_model returning None for embed_multimodal_fn and embed_input_ids_fn.
        with patch('jax.devices', return_value=self.mock_devices), \
             patch('jax.make_mesh', return_value=self.mock_mesh), \
             patch('jax.random.key', return_value=self.mock_rng_key), \
             patch('tpu_inference.runner.tpu_runner.nnx.Rngs', return_value=self.mock_rng_key), \
             patch('tpu_inference.runner.tpu_runner.get_model', return_value=self._model_get_model()), \
             patch('tpu_inference.runner.tpu_runner.make_optimized_mesh', return_value=self.mock_mesh), \
             patch('jax.device_put', side_effect=lambda x, *args, **kwargs: x):

            model_config = ModelConfig(tokenizer_mode="auto",
                                       trust_remote_code=False,
                                       seed=0,
                                       dtype='bfloat16')
            # Set multimodal_config to not None, such that the is_multimodal_model property of model_config is True.
            model_config.multimodal_config = MagicMock()

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
                speculative_config=None,
                observability_config={},
                additional_config={},
            )

            self.runner = TPUModelRunner(vllm_config,
                                         devices=self.mock_devices)
            self.runner.load_model()

    def _model_get_model(self):
        mock_multimodal_fns = {
            "precompile_vision_encoder_fn": None,
            "embed_multimodal_fn": None,
            "embed_input_ids_fn": None,
            "get_mrope_input_positions_fn": None
        }
        return (
            MagicMock(),  # TPUModelRunner.model_fn
            MagicMock(),  # TPUModelRunner.compute_logits_fn
            MagicMock(),  # TPUModelRunner.pooler_fn
            MagicMock(),  # TPUModelRunner.combine_hidden_states_fn
            mock_multimodal_fns,  # TPUModelRunner.multimodal_fns
            MagicMock(),  # TPUModelRunner.state (model params)
            None,  # TPUModelRunner.lora_manager
            None,  # TPUModelRunner.model
        )

    def test_is_multimodal_model(self):
        # Precondition: make sure the model_config claims the model supports MM.
        assert self.runner.model_config.is_multimodal_model

        # Precondition: load the model and returns embed_multimodal_fn as None.
        assert self.runner.embed_multimodal_fn is None

        assert not self.runner.is_multimodal_model

        self.runner.embed_input_ids_fn = MagicMock()
        dummy_input_ids = jnp.array([1, 2, 3])
        dummy_mm_embeds = jnp.ones((10, 128))
        dummy_is_mm_embed = jnp.array([False, True, True], dtype=jnp.bool_)
        _ = self.runner._get_input_ids_embeds(dummy_input_ids, dummy_mm_embeds,
                                              dummy_is_mm_embed)
        self.runner.embed_input_ids_fn.assert_not_called()


class TestTPUJaxRunnerDisableMM:

    def setup_method(self):
        # Mock JAX dependencies
        self.mock_devices = [MagicMock(coords=i) for i in range(4)]
        self.mock_rng_key = MagicMock()
        device_array = np.array(jax.devices()[:1]).reshape(1, 1, 1, -1)
        self.mock_mesh = jax.make_mesh(device_array.shape,
                                       ('data', 'attn_dp', 'expert', 'model'))

    def _model_get_model(self):
        mock_multimodal_fns = {
            "precompile_vision_encoder_fn": None,
            "embed_multimodal_fn": MagicMock(),
            "embed_input_ids_fn": MagicMock(),
            "get_mrope_input_positions_fn": None
        }
        return (
            MagicMock(),  # TPUModelRunner.model_fn
            MagicMock(),  # TPUModelRunner.compute_logits_fn
            MagicMock(),  # TPUModelRunner.pooler_fn
            MagicMock(),  # TPUModelRunner.combine_hidden_states_fn
            mock_multimodal_fns,  # TPUModelRunner.multimodal_fns
            MagicMock(),  # TPUModelRunner.state (model params)
            None,  # TPUModelRunner.lora_manager
            None,  # TPUModelRunner.model
        )

    @pytest.mark.parametrize(
        "limit_per_prompt, still_mm_after_loading_model",
        [
            ({
                "image": BaseDummyOptions(count=0),
                "video": BaseDummyOptions(count=0)
            }, False),
            ({
                "video": BaseDummyOptions(count=0)
            }, False),
            ({
                "image": BaseDummyOptions(count=0),
                "video": BaseDummyOptions(count=1)
            }, True),
            (
                {
                    # Empty limit means no limit, which should not disable MM.
                },
                True)
        ])
    def test_multimodal_model_loading_with_limits(
            self, limit_per_prompt, still_mm_after_loading_model):
        """Test that "--limit-mm-per-prompt" config can disable multi-modality for a multi-modal model.

        If an user *explicitly* sets the limit for all modalities to 0, then we can safely disable multi-modality even if the model itself claims to be multimodal.
        """
        with patch('jax.devices', return_value=self.mock_devices), \
             patch('jax.make_mesh', return_value=self.mock_mesh), \
             patch('jax.random.key', return_value=self.mock_rng_key), \
             patch('tpu_inference.runner.tpu_runner.nnx.Rngs', return_value=self.mock_rng_key), \
             patch('tpu_inference.runner.tpu_runner.get_model', return_value=self._model_get_model()), \
             patch('tpu_inference.runner.tpu_runner.make_optimized_mesh', return_value=self.mock_mesh), \
             patch('jax.device_put', side_effect=lambda x, *args, **kwargs: x):

            model_config = ModelConfig(tokenizer_mode="auto",
                                       trust_remote_code=False,
                                       seed=0,
                                       dtype='bfloat16')
            model_config.multimodal_config = MagicMock()
            model_config.multimodal_config.limit_per_prompt = limit_per_prompt

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
                speculative_config=None,
                observability_config={},
                additional_config={},
            )

            runner = TPUModelRunner(vllm_config, devices=self.mock_devices)
            # Precondition: make sure the model_config claims the model supports MM.
            assert runner.model_config.is_multimodal_model
            runner.load_model()

            assert runner.is_multimodal_model == still_mm_after_loading_model
