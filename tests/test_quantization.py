# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os
import tempfile
import unittest
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np
import qwix
from flax import nnx
from jax.sharding import Mesh

# Target file for testing
from tpu_commons.models.jax.utils.quantization import quantization_utils

# Configure JAX to use CPU for test portability
jax.config.update("jax_platform_name", "cpu")


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


class TestParseQuantizationYaml(unittest.TestCase):
    """Tests for `parse_quantization_yaml_file_to_rules`."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.valid_yaml_content = """
rules:
  - module_path: '.*attn.*'
    weight_qtype: 'int8'
  - module_path: '.*mlp.*'
    weight_qtype: 'int4'
    act_qtype: 'int4'
"""
        self.valid_yaml_path = os.path.join(self.temp_dir.name, "valid.yaml")
        with open(self.valid_yaml_path, "w") as f:
            f.write(self.valid_yaml_content)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_parse_valid_yaml(self):
        """Verify correct parsing of a valid YAML file."""
        rules = quantization_utils.parse_quantization_yaml_file_to_rules(
            self.valid_yaml_path)
        self.assertIsInstance(rules, list)
        self.assertEqual(len(rules), 2)
        self.assertIsInstance(rules[0], qwix.QuantizationRule)
        self.assertEqual(rules[0].module_path, '.*attn.*')
        self.assertEqual(rules[0].weight_qtype, 'int8')
        self.assertEqual(rules[1].module_path, '.*mlp.*')
        self.assertEqual(rules[1].weight_qtype, 'int4')
        self.assertEqual(rules[1].act_qtype, 'int4')

    def test_nonexistent_file_raises_error(self):
        """Verify FileNotFoundError for a non-existent file path."""
        with self.assertRaises(FileNotFoundError):
            quantization_utils.parse_quantization_yaml_file_to_rules(
                "nonexistent.yaml")

    def test_malformed_yaml_raises_error(self):
        """Verify KeyError for a YAML file missing the 'rules' key."""
        malformed_yaml_path = os.path.join(self.temp_dir.name,
                                           "malformed.yaml")
        with open(malformed_yaml_path, "w") as f:
            f.write("invalid_key: []")

        with self.assertRaises(KeyError):
            quantization_utils.parse_quantization_yaml_file_to_rules(
                malformed_yaml_path)


# --- Mocks for TestQwixQuantizeNnxModel ---


class SimpleSubModule(nnx.Module):
    """A simple submodule for the mock model."""

    def __init__(self, *, rngs: nnx.Rngs):
        self.linear = nnx.Linear(16, 16, rngs=rngs)

    def __call__(self, x):
        return self.linear(x)


class SimpleModel(nnx.Module):
    """A mock NNX model for testing quantization."""

    def __init__(self, *, rngs: nnx.Rngs):
        self.attn = SimpleSubModule(rngs=rngs)
        self.mlp = SimpleSubModule(rngs=rngs)
        self.other = nnx.Linear(16, 16, rngs=rngs)

    def __call__(self, input_ids, attention_metadata, kv_caches):
        # A dummy forward pass that uses inputs to be valid for tracing.
        x = jnp.ones((input_ids.shape[0], 16), dtype=jnp.float32)
        x = self.attn(x)
        x = self.mlp(x)
        x = self.other(x)
        # Dummy use of other args to prevent them being optimized away
        return x + jnp.sum(attention_metadata.input_positions) + jnp.sum(
            kv_caches['layer.0'].k_cache)


# --- Main Test Class for Model Quantization ---


class TestQwixQuantizeNnxModel(unittest.TestCase):
    """Tests for the main `qwix_quantize_nnx_model` function."""

    def setUp(self):
        """Set up a mock model, mesh, and common arguments."""
        self.mesh = Mesh(np.array(jax.devices()), ('dp'))
        self.rngs = nnx.Rngs(0)
        self.model = SimpleModel(rngs=self.rngs)
        self.temp_dir = tempfile.TemporaryDirectory()

        self.common_kwargs = {
            "model": self.model,
            "rng": jax.random.PRNGKey(0),
            "mesh": self.mesh,
            "num_hidden_layers": 1,
            "kv_cache_block_size": 16,
            "kv_cache_num_combined_kv_heads": 8,
            "kv_cache_head_size": 128,
        }

        self.valid_yaml_content = """
rules:
  - module_path: '.*attn.*'
    weight_qtype: 'int8'
  - module_path: '.*mlp.*'
    weight_qtype: 'int4'
    act_qtype: 'int4'
"""

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_argument_validation_error(self):
        """Verify ValueError when both dtype and rules file are provided."""
        rules_path = os.path.join(self.temp_dir.name, "rules.yaml")
        with open(rules_path, "w") as f:
            f.write(self.valid_yaml_content)

        with self.assertRaisesRegex(
                ValueError,
                "Cannot specify both quantization rules and quantization dtype"
        ):
            quantization_utils.qwix_quantize_nnx_model(
                **self.common_kwargs,
                quant_dtype='int8',
                rules_file_path=rules_path,
            )

    @patch(
        'tpu_commons.models.jax.utils.utils.quantization.quantization_utils.hbm_usage_gb',
        return_value="")
    def test_quantization_with_dtype_uses_default_rules(self):
        """Verify quantization with `quant_dtype` correctly applies default rules."""
        quantized_model = quantization_utils.qwix_quantize_nnx_model(
            **self.common_kwargs, quant_dtype='int8')

        # Default rules quantize modules with 'attn' and 'mlp' in their path
        self.assertIsInstance(quantized_model.attn.linear,
                              qwix.nnx.QuantizedLinear)
        self.assertIsInstance(quantized_model.mlp.linear,
                              qwix.nnx.QuantizedLinear)
        # The 'other' layer should not be quantized
        self.assertIsInstance(quantized_model.other, nnx.Linear)
        self.assertNotIsInstance(quantized_model.other,
                                 qwix.nnx.QuantizedLinear)

    @patch('tpu_commons.utils_jax.hbm_usage_gb', return_value="")
    def test_quantization_with_rules_file(self):
        """Verify quantization using a custom YAML rules file."""
        # Rule file that only quantizes 'mlp' layers
        rules_content = """
rules:
  - module_path: '.*mlp.*'
    weight_qtype: 'int8'
"""
        rules_path = os.path.join(self.temp_dir.name, "custom_rules.yaml")
        with open(rules_path, "w") as f:
            f.write(rules_content)

        quantized_model = quantization_utils.qwix_quantize_nnx_model(
            **self.common_kwargs, quant_dtype=None, rules_file_path=rules_path)

        # Only the 'mlp' layer should be quantized
        self.assertIsInstance(quantized_model.mlp.linear,
                              qwix.nnx.QuantizedLinear)
        # 'attn' and 'other' layers should remain un-quantized
        self.assertIsInstance(quantized_model.attn.linear, nnx.Linear)
        self.assertIsInstance(quantized_model.other, nnx.Linear)


if __name__ == '__main__':
    unittest.main()
