# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os
import sys
import unittest
from unittest.mock import MagicMock, mock_open, patch

import jax
import jax.numpy as jnp
import numpy as np
import qwix
import yaml
from flax import nnx
from jax.sharding import Mesh

# Target file for testing
from tpu_commons.models.jax.utils.quantization import quantization_utils

# Configure JAX to use CPU for test portability
jax.config.update("jax_platform_name", "cpu")

# Mock modules that are dependencies of the file under test *before* import.
# This allows testing the file's logic in isolation without heavy dependencies.
mock_utils_jax = MagicMock()
mock_logger_module = MagicMock()
mock_runner_utils = MagicMock()
mock_attention_metadata_module = MagicMock()

# Configure the mock logger to return a logger instance
mock_logger_instance = MagicMock()
mock_logger_module.init_logger.return_value = mock_logger_instance
mock_utils_jax.hbm_usage_gb.return_value = "mocked_hbm"

# Add mock modules to sys.modules
sys.modules['tpu_commons.utils_jax'] = mock_utils_jax
sys.modules['tpu_commons.logger'] = mock_logger_module
sys.modules['tpu_commons.runner.utils'] = mock_runner_utils
sys.modules[
    'tpu_commons.models.jax.attention_metadata'] = mock_attention_metadata_module

import tpu_commons.models.jax.utils.quantization.quantization_utils as quantize_qwix  # noqa: E402

# Un-patch create_kv_caches so we can mock it per-test class
quantize_qwix.create_kv_caches = mock_runner_utils.create_kv_caches


class TestQuantizeFunction(unittest.TestCase):
    """Tests for the per-tensor `quantize` helper function."""

    def test_quantize_int8(self):
        """Verify int8 quantization and scaling."""
        x = jnp.arange(-4, 5, dtype=jnp.float32)
        q_x, scale = quantization_utils.quantize(x, jnp.int8)

        self.assertEqual(q_x.dtype, jnp.int8)
        expected_scale = (4.0 / quantization_utils.MAX_INT8)
        np.testing.assert_allclose(scale.item(), expected_scale, rtol=1e-6)

        # Dequantized value should be close to original
        dequantized_x = q_x.astype(jnp.float32) * scale
        np.testing.assert_allclose(dequantized_x, x, atol=1.0)

    def test_quantize_int4(self):
        """Verify int4 quantization and scaling."""
        x = jnp.arange(-4, 5, dtype=jnp.float32) * 1.5  # Max abs value is 6.0
        q_x, scale = quantization_utils.quantize(x, jnp.int4)

        self.assertEqual(q_x.dtype, jnp.int4)
        expected_scale = 6.0 / quantization_utils.MAX_INT4
        np.testing.assert_allclose(scale.item(), expected_scale, rtol=1e-6)

        # Dequantized value should be close to original
        dequantized_x = q_x.astype(jnp.float32) * scale
        np.testing.assert_allclose(dequantized_x, x, atol=1, rtol=0.5)

    def test_quantize_float8(self):
        """Verify float8_e4m3fn quantization and scaling."""
        e4m3_max = jnp.finfo(jnp.float8_e4m3fn).max.astype(jnp.float32)
        x = jnp.array([-0.5, 0.1, 0.8, 1.0], dtype=jnp.float32) * e4m3_max
        q_x, scale = quantization_utils.quantize(x, jnp.float8_e4m3fn)

        self.assertEqual(q_x.dtype, jnp.float8_e4m3fn)
        np.testing.assert_allclose(scale.item(), 1.0, rtol=1e-6)

        # Dequantized value should be close to original
        dequantized_x = q_x.astype(jnp.float32) * scale
        np.testing.assert_allclose(dequantized_x, x, rtol=1e-1, atol=10)

    def test_quantize_zero_input(self):
        """Verify that a zero input produces a non-zero scale."""
        x = jnp.zeros((10, 10), dtype=jnp.float32)
        _, scale = quantization_utils.quantize(x, jnp.int8)
        # Scale should be clamped to 1e-6 to avoid division by zero
        self.assertAlmostEqual(scale.item(), 1e-6)

    def test_unsupported_dtype_raises_error(self):
        """Verify an unsupported dtype raises ValueError."""
        x = jnp.ones(10)
        with self.assertRaises(ValueError):
            quantization_utils.quantize(x, jnp.float32)

    def test_scale_shape_and_dtype(self):
        """Verify the scale output has shape (1,) and dtype float32."""
        x = jnp.ones((5, 5))
        _, scale = quantization_utils.quantize(x, jnp.int8)
        self.assertEqual(scale.shape, (1, ))
        self.assertEqual(scale.dtype, jnp.float32)


class TestParseQwixConfigToRules(unittest.TestCase):
    """Tests for the parse_qwix_config_to_rules function."""

    def test_empty_config(self):
        """Test parsing an empty list of rules."""
        qwix_config = []
        rules = quantize_qwix.parse_qwix_config_to_rules(qwix_config)
        self.assertEqual(rules, [])

    def test_single_rule(self):
        """Test parsing a single quantization rule."""
        qwix_config = [{
            "module_path": ".*attn.*",
            "weight_qtype": "int8",
        }]
        rules = quantize_qwix.parse_qwix_config_to_rules(qwix_config)
        self.assertEqual(len(rules), 1)
        self.assertIsInstance(rules[0], qwix.QuantizationRule)
        self.assertEqual(rules[0].module_path, ".*attn.*")
        self.assertEqual(rules[0].weight_qtype, "int8")
        self.assertIsNone(rules[0].act_qtype)

    def test_multiple_rules(self):
        """Test parsing multiple quantization rules."""
        qwix_config = [
            {
                "module_path": ".*attn.*",
                "weight_qtype": "int8",
            },
            {
                "module_path": ".*mlp.*",
                "weight_qtype": "int4",
                "act_qtype": "int8",
            },
        ]
        rules = quantize_qwix.parse_qwix_config_to_rules(qwix_config)
        self.assertEqual(len(rules), 2)
        self.assertIsInstance(rules[0], qwix.QuantizationRule)
        self.assertIsInstance(rules[1], qwix.QuantizationRule)
        self.assertEqual(rules[0].module_path, ".*attn.*")
        self.assertEqual(rules[1].module_path, ".*mlp.*")
        self.assertEqual(rules[1].weight_qtype, "int4")
        self.assertEqual(rules[1].act_qtype, "int8")

    def test_invalid_rule_key_raises_error(self):
        """Test that an invalid key in a rule raises a TypeError."""
        qwix_config = [{
            "module_path": ".*attn.*",
            "invalid_key": "some_value",
        }]
        with self.assertRaises(TypeError):
            # qwix.QuantizationRule constructor will raise this error
            quantize_qwix.parse_qwix_config_to_rules(qwix_config)


class TestConvertQuantizationConfigFile(unittest.TestCase):
    """Tests for the quantization_config_file_path_to_dict function."""

    @patch(
        'tpu_commons.models.jax.utils.quantization.quantize_qwix.QUANTIZATION_CONFIG_PATH',
        '/fake/path')
    @patch('os.listdir')
    @patch('builtins.open', new_callable=mock_open)
    def test_convert_valid_file(self, mock_file, mock_listdir):
        """Test converting a valid YAML config file to a dictionary."""
        config_filename = "my_config.yaml"
        config_content_str = """
        rules:
          - module_path: ".*"
            weight_qtype: "int8"
        """
        expected_dict = yaml.safe_load(config_content_str)

        mock_listdir.return_value = [config_filename, "another.yaml"]
        mock_file().read.return_value = config_content_str

        result = quantize_qwix.quantization_config_file_path_to_dict(
            config_filename)

        mock_listdir.assert_called_once_with('/fake/path')
        mock_file.assert_called_once_with(
            os.path.join('/fake/path', config_filename), 'r')
        self.assertEqual(result, expected_dict)

    @patch(
        'tpu_commons.models.jax.utils.quantization.quantize_qwix.QUANTIZATION_CONFIG_PATH',
        '/fake/path')
    @patch('os.listdir')
    def test_file_not_found_raises_error(self, mock_listdir):
        """Test that a ValueError is raised for a non-existent file."""
        mock_listdir.return_value = ['other_file.yaml']

        with self.assertRaisesRegex(
                ValueError,
                "Could not find quantization config file with name 'not_found.yaml'"
        ):
            quantize_qwix.quantization_config_file_path_to_dict(
                "not_found.yaml")

        mock_listdir.assert_called_once_with('/fake/path')


# A simple NNX module for testing quantization
class SimpleModel(nnx.Module):

    def __init__(self, *, rngs: nnx.Rngs):
        self.linear = nnx.Linear(10, 20, rngs=rngs)

    def __call__(self, **kwargs):
        # A simplified call signature for testing purposes
        return self.linear(kwargs['input_ids'])


@patch('qwix.quantize_model', autospec=True)
class TestQwixQuantizeNnxModel(unittest.TestCase):
    """Tests for the qwix_quantize_nnx_model function."""

    def setUp(self):
        """Set up a mock environment for testing."""
        if not jax.devices():
            self.skipTest(
                "JAX device not found, skipping JAX-dependent tests.")

        self.mesh = Mesh(jax.devices(), ('data', ))
        self.rng = jax.random.PRNGKey(0)
        self.model = SimpleModel(rngs=nnx.Rngs(0))

        self.qwix_config = [
            {
                "module_path": ".*linear.*",
                "weight_qtype": "int8",
            },
        ]

        self.num_hidden_layers = 1
        self.kv_cache_block_size = 16
        self.kv_cache_num_combined_kv_heads = 4
        self.kv_cache_head_size = 64

        self.mock_kv_caches = {"layer.0": "dummy_cache"}
        mock_runner_utils.create_kv_caches.return_value = self.mock_kv_caches

        self.mock_attn_meta = "dummy_attention_metadata"
        mock_attention_metadata_module.AttentionMetadata.return_value = self.mock_attn_meta

    def test_quantization_call_with_correct_args(self, mock_quantize_model):
        """Test that qwix.quantize_model is called with the correct arguments."""
        quantized_model_mock = MagicMock(spec=nnx.Module)
        mock_quantize_model.return_value = quantized_model_mock

        returned_model = quantize_qwix.qwix_quantize_nnx_model(
            model=self.model,
            qwix_config=self.qwix_config,
            rng=self.rng,
            mesh=self.mesh,
            num_hidden_layers=self.num_hidden_layers,
            kv_cache_block_size=self.kv_cache_block_size,
            kv_cache_num_combined_kv_heads=self.kv_cache_num_combined_kv_heads,
            kv_cache_head_size=self.kv_cache_head_size,
            kv_cache_quant_dtype=None,
        )

        self.assertIs(returned_model, quantized_model_mock)
        mock_quantize_model.assert_called_once()
        args, kwargs = mock_quantize_model.call_args

        # Assert positional arguments for qwix.quantize_model
        self.assertIs(args[0], self.model)
        self.assertIsInstance(args[1], qwix.PtqProvider)
        self.assertEqual(len(args[1].rules), 1)
        self.assertEqual(args[1].rules[0].module_path, ".*linear.*")

        # Assert keyword arguments (model inputs for tracing)
        self.assertIn("kv_caches", kwargs)
        self.assertEqual(kwargs["kv_caches"], self.mock_kv_caches)
        self.assertIn("input_ids", kwargs)
        self.assertEqual(kwargs["input_ids"].shape, (512, ))
        self.assertIn("attention_metadata", kwargs)
        self.assertEqual(kwargs["attention_metadata"], self.mock_attn_meta)

        # Assert that create_kv_caches was called correctly
        mock_runner_utils.create_kv_caches.assert_called_once_with(
            num_blocks=quantize_qwix.DEFAULT_NUM_BLOCKS_FOR_JIT_KV_CACHE,
            block_size=self.kv_cache_block_size,
            num_kv_heads=self.kv_cache_num_combined_kv_heads,
            head_size=self.kv_cache_head_size,
            mesh=self.mesh,
            layer_names=[f"layer.{i}" for i in range(self.num_hidden_layers)],
            devices=jax.local_devices(),
            kv_cache_quant_dtype=None,
        )

    def test_kv_cache_quantization_dtype_is_passed(self, mock_quantize_model):
        """Test that the kv_cache_quant_dtype is passed correctly."""
        quantize_qwix.qwix_quantize_nnx_model(
            model=self.model,
            qwix_config=self.qwix_config,
            rng=self.rng,
            mesh=self.mesh,
            num_hidden_layers=self.num_hidden_layers,
            kv_cache_block_size=self.kv_cache_block_size,
            kv_cache_num_combined_kv_heads=self.kv_cache_num_combined_kv_heads,
            kv_cache_head_size=self.kv_cache_head_size,
            kv_cache_quant_dtype='int8',
        )

        mock_runner_utils.create_kv_caches.assert_called_once()
        call_kwargs = mock_runner_utils.create_kv_caches.call_args.kwargs
        self.assertEqual(call_kwargs['kv_cache_quant_dtype'], 'int8')


if __name__ == '__main__':
    unittest.main()
