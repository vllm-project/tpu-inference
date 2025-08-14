# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import unittest
from unittest.mock import MagicMock, patch

import jax
import qwix
from flax import nnx
from jax.sharding import Mesh

import tpu_commons.models.jax.utils.quantization.quantization_utils as quantize_qwix  # noqa: E402
from tpu_commons.models.jax.utils.quantization.quantization_utils import (
    DEFAULT_MAX_NUM_BLOCKS_PER_REQ, DEFAULT_MAX_NUM_SEQS_FOR_MODEL_INPUTS,
    DEFAULT_NUM_TOKENS_FOR_MODEL_INPUTS)


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
        self.kv_cache_num_kv_heads = 4
        self.kv_cache_head_size = 64

        self.mock_kv_caches = [MagicMock(), MagicMock()]

    def test_quantization_call_with_correct_args(self, mock_quantize_model):
        """Test that qwix.quantize_model is called with the correct arguments."""
        quantized_model_mock = MagicMock(spec=nnx.Module)
        mock_quantize_model.return_value = quantized_model_mock

        with patch(
                "tpu_commons.models.jax.utils.quantization.quantization_utils.init_logger",
                return_value=MagicMock()
        ), patch(
                "tpu_commons.runner.utils.utils.hbm_usage_gb",
                return_value=[(0.0, 0.0), (0.0, 0.0)]
        ), patch(
                "tpu_commons.models.jax.utils.quantization.quantization_utils.create_kv_caches",
                return_value=self.mock_kv_caches
        ), patch(
                "tpu_commons.models.jax.utils.quantization.quantization_utils.quantization_config_file_path_to_dict",
                return_value=self.qwix_config):
            returned_model = quantize_qwix.qwix_quantize_nnx_model(
                model=self.model,
                qwix_config=self.qwix_config,
                rng=self.rng,
                mesh=self.mesh,
                num_hidden_layers=self.num_hidden_layers,
                kv_cache_block_size=self.kv_cache_block_size,
                kv_cache_num_kv_heads=self.kv_cache_num_kv_heads,
                kv_cache_head_size=self.kv_cache_head_size,
            )

        self.assertIs(returned_model, quantized_model_mock)
        mock_quantize_model.assert_called_once()
        args, kwargs = mock_quantize_model.call_args

        # Assert positional arguments for qwix.quantize_model
        self.assertIs(args[0], self.model)
        self.assertIsInstance(args[1], qwix.PtqProvider)

        # Assert keyword arguments (model inputs for tracing)
        self.assertIn("kv_caches", kwargs)
        self.assertEqual(kwargs["kv_caches"], self.mock_kv_caches)
        self.assertIn("input_ids", kwargs)
        self.assertEqual(kwargs["input_ids"].shape, (512, ))
        self.assertIn("attention_metadata", kwargs)
        attention_metadata = kwargs["attention_metadata"]

        assert attention_metadata.input_positions.shape == (
            DEFAULT_NUM_TOKENS_FOR_MODEL_INPUTS, )
        assert attention_metadata.block_tables.shape == (
            DEFAULT_MAX_NUM_SEQS_FOR_MODEL_INPUTS,
            DEFAULT_MAX_NUM_BLOCKS_PER_REQ)
        assert attention_metadata.seq_lens.shape == (
            DEFAULT_MAX_NUM_SEQS_FOR_MODEL_INPUTS, )
        assert attention_metadata.query_start_loc.shape == (
            DEFAULT_MAX_NUM_SEQS_FOR_MODEL_INPUTS + 1, )
        assert attention_metadata.request_distribution.shape == (3, )


if __name__ == '__main__':
    unittest.main()
