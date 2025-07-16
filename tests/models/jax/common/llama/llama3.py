import unittest
from unittest.mock import MagicMock, patch

import jax.numpy as jnp
import numpy as np
from flax import nnx

from tpu_commons.models.jax.common.transformer_block import \
    TransformerBlockConfig
from tpu_commons.models.jax.recipes.llama3 import (Llama3WeightLoader,
                                                   Llama8BConfig,
                                                   Llama8BModelConfig)

# --- Mock Dependencies ---


class MockModelConfig:
    """A mock for llama_model.ModelConfig."""

    def __init__(self):
        self.hf_overrides = {}
        self.override_generation_config = {}
        self.additional_config = {}


class MockVllmConfig:
    """A mock for vllm.config.VllmConfig."""

    def __init__(self, additional_config=None, model_name=""):
        self.additional_config = additional_config or {}
        self.model = model_name
        self.model_config = MockModelConfig()


class MockMesh:
    """A mock for jax.sharding.Mesh."""

    def __init__(self, devices=None, axis_names=None):
        self.devices = devices if devices is not None else np.array([[0]])
        self.axis_names = axis_names if axis_names is not None else ('dp',
                                                                     'mp')

    @property
    def size(self):
        return self.devices.size


# --- Test Cases ---


class TestLlamaConfig(unittest.TestCase):
    """Tests for the configuration data classes."""

    def test_model_config_defaults_and_post_init(self):
        """Tests Llama8BModelConfig defaults and its __post_init__ logic."""
        cfg = Llama8BModelConfig()

        self.assertEqual(cfg.hidden_size, 4096)
        self.assertEqual(cfg.num_layers, 32)
        self.assertIsNotNone(cfg.emb)
        self.assertIsNotNone(cfg.layers)
        self.assertIsInstance(cfg.layers, TransformerBlockConfig)

        self.assertEqual(cfg.emb.hidden_size, 4096)
        self.assertEqual(cfg.emb.vocab_size, 128256)
        self.assertEqual(cfg.layers.attention.hidden_size, 4096)
        self.assertEqual(cfg.layers.dense_ffw.intermediate_size, 14336)

    def test_model_config_override(self):
        """Tests overriding defaults in Llama8BModelConfig."""
        mock_vllm_config = MockVllmConfig()
        cfg = Llama8BModelConfig(hidden_size=1024,
                                 num_layers=4,
                                 vllm_config=mock_vllm_config)

        self.assertEqual(cfg.hidden_size, 1024)
        self.assertEqual(cfg.num_layers, 4)
        self.assertIsNotNone(cfg.emb)
        self.assertEqual(cfg.emb.hidden_size, 1024)
        self.assertIs(cfg.emb.vllm_config, mock_vllm_config)

    def test_top_level_config(self):
        """Tests the top-level Llama8BConfig."""
        cfg = Llama8BConfig()

        self.assertIsInstance(cfg.model, Llama8BModelConfig)
        self.assertIsNotNone(cfg.sharding)
        self.assertIsNotNone(cfg.serving)
        self.assertEqual(cfg.model.hidden_size, 4096)


class TestLlama3WeightLoader(unittest.TestCase):
    """Tests for the Llama3WeightLoader class."""

    def setUp(self):
        """Set up a Llama3WeightLoader instance for testing."""
        mock_vllm_config = MockVllmConfig()
        mock_model_config = Llama8BModelConfig(vllm_config=mock_vllm_config)

        self.loader = Llama3WeightLoader(
            vllm_config=mock_vllm_config,
            model_config=mock_model_config,
        )

        # Manually set attributes and call the setup method to test it
        self.loader.model_config = mock_model_config
        self.loader.loaded_to_standardized_keys = {}
        self.loader.transpose_param_map = {}
        # nnx.State uses a custom dict, so a real one is used for `reshape_param_map`
        self.loader.reshape_param_map = nnx.State({})
        self.loader.setup()

    def test_setup_populates_maps(self):
        """Tests that the setup method correctly populates the mapping dictionaries."""
        from tpu_commons.models.jax.utils.weight_utils import ParameterType

        self.assertIn("gate_proj", self.loader.transpose_param_map)
        self.assertIn("o_proj", self.loader.transpose_param_map)
        self.assertIn("q_proj",
                      self.loader.reshape_param_map[ParameterType.weight])
        self.assertIn("q_proj.bias",
                      self.loader.reshape_param_map[ParameterType.bias])
        self.assertIn("model.embed_tokens",
                      self.loader.loaded_to_standardized_keys)
        self.assertIn("lm_head", self.loader.loaded_to_standardized_keys)

    def test_map_loaded_to_standardized_name(self):
        """Tests the name mapping logic for various key types."""
        # Test a non-layer key
        self.assertEqual(
            self.loader.map_loaded_to_standardized_name("model.norm"),
            "final_norm.scale")
        # Test a layer-specific key
        self.assertEqual(
            self.loader.map_loaded_to_standardized_name(
                "model.layers.10.self_attn.q_proj"),
            "layers.10.attn.kernel_q_proj_NDH")
        # Test another layer-specific key
        self.assertEqual(
            self.loader.map_loaded_to_standardized_name(
                "model.layers.5.mlp.down_proj"),
            "layers.5.mlp.kernel_down_proj_FD")
        # Test a key that is not in the map should return itself
        self.assertEqual(
            self.loader.map_loaded_to_standardized_name("some.unmapped.key"),
            "some.unmapped.key")

    @patch('llama_model.shard_put', side_effect=lambda x, *args, **kwargs: x)
    @patch('llama_model.get_param')
    def test_load_weights_orchestration(self, mock_get_param, mock_shard_put):
        """Tests the overall weight loading process and its interactions."""
        mock_model = MagicMock()
        mock_model.mesh = MockMesh()

        # Mock the parameter object that get_param will return
        mock_param = MagicMock()
        mock_param.value = jnp.ones((4096, 4096))  # Dummy shape
        mock_param.sharding.spec = MagicMock()
        mock_get_param.return_value = mock_param

        # Mock the generator of weights that the loader iterates over
        dummy_weights_generator = [
            ("model.layers.0.self_attn.q_proj.weight", jnp.ones((4096, 4096))),
            ("lm_head.weight", jnp.ones((128256, 4096))),
        ]
        self.loader.names_and_weights_generator = dummy_weights_generator

        with patch('flax.nnx.state', return_value={}) as mock_state, \
             patch('flax.nnx.update') as mock_update:

            self.loader.load_weights(mock_model)

            mock_state.assert_called_once_with(mock_model)

            # Check that get_param was called for each weight with the correctly mapped name
            self.assertEqual(mock_get_param.call_count, 2)
            mock_get_param.assert_any_call({},
                                           "layers.0.attn.kernel_q_proj_NDH")
            mock_get_param.assert_any_call({},
                                           "lm_head.input_embedding_table_VD")

            # Check that parameters were reshaped/transposed (by checking the mock calls)
            self.assertEqual(mock_shard_put.call_count, 2)

            # Check that the model state is updated at the end
            mock_update.assert_called_once()


# Patch tpu_commons dependencies to run tests in a standard environment
# @patch('llama_model.sharding.Sharding')
# @patch('llama_model.Embedder', autospec=True)
# @patch('llama_model.TransformerBlock', autospec=True)
# @patch('llama_model.RMSNorm', autospec=True)
# class TestLlama3_8BModel(unittest.TestCase):
#     """Tests for the main Llama3_8B model class."""

#     def setUp(self):
#         """Set up a mock environment for the model tests."""
#         self.mesh = MockMesh()
#         self.rng = jax.random.PRNGKey(0)

#         sharding_dict = {"sharding_strategy": {"tensor_parallelism": 1}}
#         self.vllm_config = MockVllmConfig(additional_config=sharding_dict)

#     def test_model_initialization(self, MockRMSNorm, MockTransformerBlock, MockEmbedder, MockSharding):
#         """Tests that the Llama3_8B model initializes its components correctly."""
#         # Setup mocks
#         mock_sharding_inst = MockSharding.return_value
#         mock_sharding_inst.mesh = self.mesh

#         # Instantiate the model
#         model = Llama3_8B(self.vllm_config, self.rng, self.mesh)

#         self.assertIsInstance(model.cfg, Llama8BConfig)
#         self.assertEqual(model.cfg.model.num_layers, 32)

#         # Check that layers were instantiated
#         self.assertEqual(MockEmbedder.call_count, 2)  # embedder and lm_head
#         self.assertEqual(MockTransformerBlock.call_count, 32)
#         self.assertEqual(MockRMSNorm.call_count, 1)

#         # Check that kernels were generated for all layers
#         self.assertEqual(MockEmbedder.return_value.generate_kernel.call_count, 2)
#         self.assertEqual(MockTransformerBlock.return_value.generate_kernel.call_count, 32)
#         self.assertEqual(MockRMSNorm.return_value.generate_kernel.call_count, 1)

# @patch('llama_model.Llama3WeightLoader')
# def test_load_weights_normal_path(self, MockWeightLoader, *_):
#     """Tests the standard weight loading path."""
#     model = Llama3_8B(self.vllm_config, self.rng, self.mesh)
#     mock_loader_instance = MockWeightLoader.return_value

#     model.load_weights(rng=self.rng)

#     MockWeightLoader.assert_called_once()
#     mock_loader_instance.load_weights.assert_called_once_with(model)

# @patch('llama_model.Llama3WeightLoader')
# def test_load_weights_random_path(self, MockWeightLoader, *_):
#     """Tests the 'random_weights' path, which should skip loading."""
#     random_weights_config = {
#         "sharding_strategy": {"tensor_parallelism": 1},
#         "random_weights": True,
#     }
#     vllm_config_random = MockVllmConfig(additional_config=random_weights_config)

#     model = Llama3_8B(vllm_config_random, self.rng, self.mesh)
#     model.load_weights(rng=self.rng)

#     # The weight loader should NOT be instantiated or called
#     MockWeightLoader.assert_not_called()

# def test_forward_pass_data_flow(self, MockRMSNorm, MockTransformerBlock, MockEmbedder, MockSharding):
#     """Tests the data flow through __call__ and compute_logits."""
#     # --- Setup model with mock components ---
#     mock_sharding_inst = MockSharding.return_value
#     mock_sharding_inst.mesh = self.mesh
#     model = Llama3_8B(self.vllm_config, self.rng, self.mesh)

#     num_layers = model.cfg.model.num_layers
#     hidden_size = model.cfg.model.hidden_size

#     # --- Mock behavior of components ---
#     batch_size, seq_len = 2, 10
#     input_ids = jnp.ones((batch_size, seq_len), dtype=jnp.int32)
#     mock_kv_caches = [MagicMock(name=f"kv_{i}") for i in range(num_layers)]
#     mock_attn_meta = MagicMock(name="attn_meta")

#     embedding_output = jnp.ones((batch_size, seq_len, hidden_size))
#     model.embedder.encode.return_value = embedding_output

#     block_output_hidden = jnp.ones((batch_size, seq_len, hidden_size)) * 2
#     block_output_kv = MagicMock(name="new_kv")
#     for block in model.layers:
#         block.return_value = (block_output_kv, block_output_hidden)

#     final_norm_output = jnp.ones((batch_size, seq_len, hidden_size)) * 3
#     model.final_norm.return_value = final_norm_output

#     logits_output = jnp.ones((batch_size, seq_len, model.cfg.model.emb.vocab_size))
#     model.lm_head.decode.return_value = logits_output

#     # --- Execute __call__ ---
#     result_kv, result_hidden = model(
#         kv_caches=mock_kv_caches,
#         input_ids=input_ids,
#         attention_metadata=mock_attn_meta
#     )

#     # --- Assertions for __call__ ---
#     model.embedder.encode.assert_called_once_with(input_ids)

#     self.assertEqual(model.layers[0].call_count, 1)
#     # Check that the first block was called with the embeddings
#     np.testing.assert_array_equal(model.layers[0].call_args.args[0], embedding_output)

#     # Check that the final norm was called with the output of the last block
#     model.final_norm.assert_called_once()
#     np.testing.assert_array_equal(model.final_norm.call_args.args[0], block_output_hidden)

#     # Check final outputs of __call__
#     self.assertEqual(len(result_kv), num_layers)
#     self.assertIs(result_kv[0], block_output_kv)
#     np.testing.assert_array_equal(result_hidden, final_norm_output)

#     # --- Execute and assert compute_logits ---
#     result_logits = model.compute_logits(result_hidden)

#     model.lm_head.decode.assert_called_once()
#     np.testing.assert_array_equal(model.lm_head.decode.call_args.args[0], result_hidden)
#     np.testing.assert_array_equal(result_logits, logits_output)

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
