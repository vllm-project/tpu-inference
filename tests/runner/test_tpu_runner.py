from unittest.mock import MagicMock, patch

import jax
import jax.numpy as jnp
import numpy as np
from vllm.config import (CacheConfig, ModelConfig, ParallelConfig,
                         SchedulerConfig, SpeculativeConfig, VllmConfig)
from vllm.config.pooler import PoolerConfig

from tpu_inference.layers.jax.pool.pooler import Pooler
from tpu_inference.layers.jax.pool.pooling_metadata import (
    TPUSupportedPoolingMetadata, )
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
            scheduler_config = SchedulerConfig(max_num_seqs=16, )
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
        self.runner.get_input_embeddings_fn = self.mock_get_input_embed_fn
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


class TestTPUJaxRunnerMultimodalModelLoadedForTextOnly:

    def setup_method(self):
        # Mock JAX dependencies
        self.mock_devices = [MagicMock(coords=i) for i in range(4)]
        self.mock_rng_key = MagicMock()
        device_array = np.array(jax.devices()[:1]).reshape(1, 1, 1, -1)
        self.mock_mesh = jax.make_mesh(device_array.shape,
                                       ('data', 'attn_dp', 'expert', 'model'))
        # Setup the runner with the model_config.is_multimodal_model set to True but get_model returning None for get_multimodal_embeddings_fn and get_input_embeddings_fn.
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
                observability_config={},
                additional_config={},
            )

            self.runner = TPUModelRunner(vllm_config,
                                         devices=self.mock_devices)
            self.runner.load_model()

    def _model_get_model(self):
        mock_multimodal_fns = {
            "precompile_vision_encoder_fn": None,
            "get_multimodal_embeddings_fn": None,
            "get_input_embeddings_fn": None,
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

        # Precondition: load the model and returns get_multimodal_embeddings_fn as None.
        assert self.runner.get_multimodal_embeddings_fn is None

        assert not self.runner.is_multimodal_model

        self.runner.get_input_embeddings_fn = MagicMock()
        dummy_input_ids = jnp.array([1, 2, 3])
        dummy_mm_embeds = jnp.ones((10, 128))
        _ = self.runner._get_input_ids_embeds(dummy_input_ids, dummy_mm_embeds)
        self.runner.get_input_embeddings_fn.assert_not_called()


class TestTPUJaxRunnerEmbeddingModel:

    def setup_method(self):
        self.mock_devices = [MagicMock(coords=i) for i in range(1)]
        self.mock_rng_key = MagicMock()
        device_array = np.array(jax.devices()[:1]).reshape(1, 1, 1, -1)
        self.mock_mesh = jax.make_mesh(device_array.shape,
                                       ('data', 'attn_dp', 'expert', 'model'))

        pooler_config = PoolerConfig(pooling_type="LAST", normalize=False)
        pooler = Pooler.for_embed(pooler_config)



        with patch('jax.devices', return_value=self.mock_devices), \
             patch('jax.make_mesh', return_value=self.mock_mesh), \
             patch('jax.random.key', return_value=jax.random.PRNGKey(0)), \
             patch('tpu_inference.runner.tpu_runner.nnx.Rngs', return_value=self.mock_rng_key), \
             patch('tpu_inference.runner.tpu_runner.get_model',
                   return_value=self._model_get_model(pooler)), \
             patch('tpu_inference.runner.tpu_runner.make_optimized_mesh',
                   return_value=self.mock_mesh):

            model_config = ModelConfig(tokenizer_mode="auto",
                                       trust_remote_code=False,
                                       convert = "embed",
                                       runner = "pooling",
                                       seed=0,
                                       dtype='bfloat16')
            model_config.pooler_config = pooler_config
            cache_config = CacheConfig(block_size=16,
                                       gpu_memory_utilization=0.9,
                                       swap_space=4,
                                       cache_dtype="auto")
            scheduler_config = SchedulerConfig(max_num_seqs=4)
            parallel_config = ParallelConfig(pipeline_parallel_size=1,
                                             tensor_parallel_size=1,
                                             worker_use_ray=False)
            vllm_config = VllmConfig(model_config=model_config,
                                     cache_config=cache_config,
                                     scheduler_config=scheduler_config,
                                     parallel_config=parallel_config,
                                     speculative_config=None,
                                     observability_config={},
                                     additional_config={})
            self.runner = TPUModelRunner(vllm_config,
                                         devices=self.mock_devices)
            self.runner.load_model()

    def _model_get_model(self, pooler):
        class DummyEmbeddingModel:

            def __init__(self, pooler):
                self.is_pooling_model = True
                self.pooler = pooler

        mock_multimodal_fns = {
                "precompile_vision_encoder_fn": None,
                "get_multimodal_embeddings_fn": None,
                "get_input_embeddings_fn": None,
                "get_mrope_input_positions_fn": None
        }
        return (
            MagicMock(),  # TPUModelRunner.model_fn
            MagicMock(),  # TPUModelRunner.compute_logits_fn
            MagicMock(),  # TPUModelRunner.combine_hidden_states_fn
            mock_multimodal_fns,  # TPUModelRunner.multimodal_fns
            MagicMock(),  # TPUModelRunner.state (model params)
            None,  # TPUModelRunner.lora_manager
            DummyEmbeddingModel(pooler),  # TPUModelRunner.model
            )

    def test_get_supported_tasks_pooling(self):
        assert self.runner.is_pooling_model
        assert self.runner.get_supported_tasks() == ("embed", )

    def test_pooler_forward(self):
        hidden_states = jnp.arange(6, dtype=jnp.float32).reshape(3, 2)

        metadata = TPUSupportedPoolingMetadata(
            prompt_lens=jnp.array([3], dtype=jnp.int32),
            first_token_indices=jnp.array([0], dtype=jnp.int32),
            last_token_indices=jnp.array([2], dtype=jnp.int32),
            num_scheduled_tokens=jnp.array([3], dtype=jnp.int32),
        )
        mock_pooler = MagicMock(return_value=hidden_states[-1])
        self.runner.pooler = mock_pooler
        outputs = self.runner.pooler(hidden_states, metadata)
        np.testing.assert_array_equal(np.asarray(outputs),
                                      np.asarray(hidden_states[-1]))
