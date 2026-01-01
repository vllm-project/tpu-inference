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

    def test_get_supported_tasks_runner(self):
        """Test get_supported_tasks for generate runner type."""
        supported_tasks = self.runner.get_supported_tasks()
        assert supported_tasks == ("generate", )

    def test_get_input_ids_embeds(self):
        """Tests _get_input_ids_embeds for both multimodal and text-only models."""
        # 1. ===== Setup =====
        dummy_input_ids = jnp.array([1, 2, 3])
        dummy_mm_embeds = jnp.ones((10, 128))
        dummy_final_embeds = jnp.ones((3, 128))

        # Mock the embedding function
        self.mock_get_input_embed_fn = MagicMock()
        self.runner.embed_input_ids_fn = self.mock_get_input_embed_fn
        self.mock_get_input_embed_fn.return_value = dummy_final_embeds
        self.runner.state = MagicMock()

        # 2. ===== Act & Assert (Multimodal) =====
        self.runner.is_multimodal_model = True

        input_ids_res, inputs_embeds_res = self.runner._get_input_ids_embeds(
            dummy_input_ids, dummy_mm_embeds)

        assert input_ids_res is None
        np.testing.assert_array_equal(np.asarray(inputs_embeds_res),
                                      np.asarray(dummy_final_embeds))
        self.mock_get_input_embed_fn.assert_called_once_with(
            self.runner.state, dummy_input_ids, dummy_mm_embeds)

        # 3. ===== Act & Assert (Text-only) =====
        self.mock_get_input_embed_fn.reset_mock()
        self.runner.is_multimodal_model = False

        input_ids_res, inputs_embeds_res = self.runner._get_input_ids_embeds(
            dummy_input_ids, dummy_mm_embeds)

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
        mock_block_table.get_cpu_tensor.return_value = np.zeros(
            self.runner.block_tables_cpu[0].shape)
        self.runner.input_batch.block_table = [
            mock_block_table, mock_block_table
        ]
        self.runner.block_tables_cpu = [
            np.zeros(self.runner.block_tables_cpu[0].shape, dtype=np.int32),
            np.zeros(self.runner.block_tables_cpu[0].shape, dtype=np.int32)
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
             patch('tpu_inference.runner.tpu_runner.make_optimized_mesh', return_value=self.mock_mesh):

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
            scheduler_config = SchedulerConfig(max_num_seqs=16,
                                               max_model_len=1024,
                                               is_encoder_decoder=False)
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
        _ = self.runner._get_input_ids_embeds(dummy_input_ids, dummy_mm_embeds)
        self.runner.embed_input_ids_fn.assert_not_called()
