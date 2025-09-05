# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import unittest
from unittest.mock import MagicMock, mock_open, patch

import jax
import qwix
from flax import nnx
from jax.sharding import Mesh

import tpu_commons.models.jax.utils.quantization.quantization_utils as quantize_qwix  # noqa: E402
from tpu_commons.models.jax.model_loader import apply_qwix_quantization
from tpu_commons.models.jax.utils.quantization.quantization_utils import (
    DEFAULT_MAX_NUM_BLOCKS_PER_REQ, DEFAULT_MAX_NUM_SEQS_FOR_MODEL_INPUTS,
    DEFAULT_NUM_TOKENS_FOR_MODEL_INPUTS)

mock_nnx = MagicMock()
mock_jax = MagicMock()

module_mocks = {
    'flax': MagicMock(nnx=mock_nnx),
    'flax.nnx': mock_nnx,
    'jax': mock_jax,
    'jax.sharding': MagicMock(),
    'vllm': MagicMock(),
    'vllm.config': MagicMock(),
    'tpu_commons': MagicMock(),
    'tpu_commons.logger': MagicMock(init_logger=lambda name: MagicMock()),
    'tpu_commons.models.jax.utils.quantization.quantization_utils':
    MagicMock(),
}


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
                "tpu_commons.utils.hbm_usage_gb",
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
            DEFAULT_MAX_NUM_SEQS_FOR_MODEL_INPUTS *
            DEFAULT_MAX_NUM_BLOCKS_PER_REQ, )
        assert attention_metadata.seq_lens.shape == (
            DEFAULT_MAX_NUM_SEQS_FOR_MODEL_INPUTS, )
        assert attention_metadata.query_start_loc.shape == (
            DEFAULT_MAX_NUM_SEQS_FOR_MODEL_INPUTS + 1, )
        assert attention_metadata.request_distribution.shape == (3, )


@patch.dict('sys.modules', module_mocks)
class TestApplyQwixQuantization(unittest.TestCase):

    def setUp(self):
        """Set up common mock objects for all tests in this suite."""
        mock_nnx.reset_mock()
        mock_jax.reset_mock()

        self.mock_vllm_config = MagicMock()
        self.mock_vllm_config.additional_config = {}
        self.mock_vllm_config.cache_config.block_size = 16
        self.mock_vllm_config.model_config.get_head_size.return_value = 128
        self.mock_vllm_config.model_config.get_total_num_kv_heads.return_value = 8
        self.mock_vllm_config.model_config.hf_config.num_hidden_layers = 32

        self.mock_model = MagicMock(name="original_nnx_model",
                                    spec_set=nnx.Module)
        self.mock_rng = MagicMock(name="mock_rng")
        self.mock_mesh = MagicMock(name="mock_mesh")

    def test_no_quantization_config(self):
        """
        Test that the model is returned unchanged if no 'quantization' key exists.
        """
        result = apply_qwix_quantization(self.mock_vllm_config,
                                         self.mock_model,
                                         self.mock_rng,
                                         self.mock_mesh,
                                         apply_to_abstract_model=False)

        self.assertIs(result, self.mock_model,
                      "Model should be returned as-is.")
        mock_nnx.jit.assert_not_called()

    @patch('tpu_commons.models.jax.model_loader.nnx.jit')
    def test_quantization_applied_from_dict(self, mock_jit):
        """
        Test that quantization is applied correctly when the config is a dictionary.
        """
        qwix_rules = {"weights": "int8", "activations": None}
        self.mock_vllm_config.additional_config = {
            "quantization": {
                "qwix": {
                    "rules": qwix_rules
                }
            }
        }

        with patch('tpu_commons.utils.get_padded_num_heads', return_value=128):
            apply_qwix_quantization(self.mock_vllm_config,
                                    self.mock_model,
                                    self.mock_rng,
                                    self.mock_mesh,
                                    apply_to_abstract_model=False)
        mock_jit.assert_called_once()

    @patch('tpu_commons.models.jax.model_loader.nnx.jit')
    def test_quantization_applied_from_string_path(self, mock_jit):
        """
        Test that quantization is applied when the config is a string (file path).
        """
        config_path = "int8_default.yaml"
        self.mock_vllm_config.additional_config = {"quantization": config_path}
        with patch('tpu_commons.utils.get_padded_num_heads', return_value=128):
            apply_qwix_quantization(self.mock_vllm_config,
                                    self.mock_model,
                                    self.mock_rng,
                                    self.mock_mesh,
                                    apply_to_abstract_model=False)

        mock_jit.assert_called_once()


class TestQuantizationConfigFileToDict(unittest.TestCase):
    """Tests for the quantization_config_file_path_to_dict function."""

    @patch("os.listdir")
    @patch("os.path.join")
    def test_file_not_found_raises_value_error(self, mock_join, mock_listdir):
        """Test that a ValueError is raised if the config file is not found."""
        mock_listdir.return_value = ["another_file.yaml", "config.txt"]
        config_file_path = "non_existent.yaml"

        with self.assertRaisesRegex(
                ValueError,
                f"Could not find quantization config file with name '{config_file_path}'"
        ):
            quantize_qwix.quantization_config_file_path_to_dict(
                config_file_path)
        mock_listdir.assert_called_once_with(
            quantize_qwix.QUANTIZATION_CONFIG_PATH)

    @patch("os.listdir")
    @patch("os.path.join")
    @patch("builtins.open",
           new_callable=mock_open,
           read_data="qwix:\n  rules: []")
    def test_file_found_and_loaded_successfully(self, mock_file, mock_join,
                                                mock_listdir):
        """Test that the YAML file is correctly loaded when found."""
        config_filename = "my_quant_config.yaml"
        mock_listdir.return_value = ["another.yaml", config_filename]
        mock_join.return_value = f"/fake/path/{config_filename}"
        expected_dict = {"qwix": {"rules": []}}

        result = quantize_qwix.quantization_config_file_path_to_dict(
            config_filename)

        mock_listdir.assert_called_once_with(
            quantize_qwix.QUANTIZATION_CONFIG_PATH)
        mock_join.assert_called_once_with(
            quantize_qwix.QUANTIZATION_CONFIG_PATH, config_filename)
        mock_file.assert_called_once_with(f"/fake/path/{config_filename}", "r")
        self.assertEqual(result, expected_dict)


class TestApplyQwixQuantizationLogic(unittest.TestCase):
    """Tests the core logic of apply_qwix_quantization."""

    def setUp(self):
        self.mock_vllm_config = MagicMock()
        self.mock_vllm_config.additional_config = {}
        self.mock_vllm_config.cache_config.block_size = 16
        self.mock_vllm_config.model_config.get_head_size.return_value = 128
        self.mock_vllm_config.model_config.get_total_num_kv_heads.return_value = 8
        self.mock_vllm_config.model_config.hf_config.num_hidden_layers = 32
        self.mock_model = MagicMock(name="original_nnx_model")
        self.mock_rng = MagicMock(name="mock_rng")
        self.mock_mesh = MagicMock(name="mock_mesh", shape={"model": 1})

    def test_quantization_config_without_qwix_rules(self):
        """Test model is unchanged if the config lacks 'qwix' or 'rules'."""
        self.mock_vllm_config.additional_config = {"quantization": {}}
        result1 = quantize_qwix.apply_qwix_quantization(
            self.mock_vllm_config, self.mock_model, self.mock_rng,
            self.mock_mesh, False)
        self.assertIs(result1, self.mock_model)

        self.mock_vllm_config.additional_config = {
            "quantization": {
                "qwix": {}
            }
        }
        result2 = quantize_qwix.apply_qwix_quantization(
            self.mock_vllm_config, self.mock_model, self.mock_rng,
            self.mock_mesh, False)
        self.assertIs(result2, self.mock_model)

    @patch(
        'tpu_commons.models.jax.utils.quantization.quantization_utils.qwix_quantize_nnx_model'
    )
    @patch(
        'tpu_commons.models.jax.utils.quantization.quantization_utils.utils')
    def test_apply_to_abstract_model(self, mock_utils, mock_quantize_func):
        """Test quantization is correctly applied to an abstract model factory."""
        mock_utils.get_padded_num_heads.return_value = 8
        mock_utils.get_padded_head_dim.return_value = 128
        qwix_rules = [{"module_path": ".*", "weight_qtype": "int8"}]
        self.mock_vllm_config.additional_config = {
            "quantization": {
                "qwix": {
                    "rules": qwix_rules
                }
            }
        }
        mock_abstract_model = MagicMock(name="abstract_model")
        mock_model_fn = MagicMock(name="model_factory",
                                  return_value=mock_abstract_model)
        quantized_model = MagicMock(name="quantized_model")
        mock_quantize_func.return_value = quantized_model

        model_factory = quantize_qwix.apply_qwix_quantization(
            self.mock_vllm_config,
            mock_model_fn,
            self.mock_rng,
            self.mock_mesh,
            apply_to_abstract_model=True)

        self.assertTrue(callable(model_factory))
        result_model = model_factory()

        mock_model_fn.assert_called_once()
        mock_quantize_func.assert_called_once()
        call_kwargs = mock_quantize_func.call_args.kwargs
        self.assertIs(call_kwargs['model'], mock_abstract_model)
        self.assertIs(call_kwargs['rng'], self.mock_rng)
        self.assertIs(result_model, quantized_model)

    @patch(
        'tpu_commons.models.jax.utils.quantization.quantization_utils.qwix_quantize_nnx_model'
    )
    @patch(
        'tpu_commons.models.jax.utils.quantization.quantization_utils.utils')
    def test_apply_to_abstract_model_with_initialize_cache(
            self, mock_utils, mock_quantize_func):
        """Test abstract model quantization with 'initialize_cache' method."""
        mock_utils.get_padded_num_heads.return_value = 8
        mock_utils.get_padded_head_dim.return_value = 128
        qwix_rules = [{"module_path": ".*", "weight_qtype": "int8"}]
        self.mock_vllm_config.additional_config = {
            "quantization": {
                "qwix": {
                    "rules": qwix_rules
                }
            }
        }
        mock_abstract_model = MagicMock(name="abstract_model")
        mock_abstract_model.initialize_cache = MagicMock()
        mock_model_fn = MagicMock(name="model_factory",
                                  return_value=mock_abstract_model)

        model_factory = quantize_qwix.apply_qwix_quantization(
            self.mock_vllm_config,
            mock_model_fn,
            self.mock_rng,
            self.mock_mesh,
            apply_to_abstract_model=True)

        model_factory()

        mock_abstract_model.initialize_cache.assert_called_once()
        mock_quantize_func.assert_called_once()


class TestDetermineWhetherToApplyQwixOnAbstractModel(unittest.TestCase):
    """Tests for apply_qwix_on_abstract_model."""

    def setUp(self):
        self.mock_vllm_config = MagicMock()
        self.mock_vllm_config.additional_config = {
            "quantization": "some_config.yaml"
        }

        self.mock_vllm_config_no_additional_config = MagicMock()
        self.mock_vllm_config_no_additional_config.additional_config = {}

    @patch(
        "tpu_commons.models.jax.utils.quantization.quantization_utils.quantization_config_file_path_to_dict"
    )
    def test_returns_true_when_config_is_true(self, mock_load_dict):
        """Test it returns True when use_abstract_model is True in config."""
        mock_load_dict.return_value = {"qwix": {"use_abstract_model": True}}
        result = quantize_qwix.apply_qwix_on_abstract_model(
            self.mock_vllm_config)
        self.assertTrue(result)
        mock_load_dict.assert_called_once_with("some_config.yaml")

    @patch(
        "tpu_commons.models.jax.utils.quantization.quantization_utils.quantization_config_file_path_to_dict"
    )
    def test_returns_false_when_config_is_false(self, mock_load_dict):
        """Test it returns False when use_abstract_model is False in config."""
        mock_load_dict.return_value = {"qwix": {"use_abstract_model": False}}
        result = quantize_qwix.apply_qwix_on_abstract_model(
            self.mock_vllm_config)
        self.assertFalse(result)

    @patch(
        "tpu_commons.models.jax.utils.quantization.quantization_utils.quantization_config_file_path_to_dict"
    )
    def test_returns_false_when_key_is_missing(self, mock_load_dict):
        """Test it defaults to False when use_abstract_model key is missing."""
        mock_load_dict.return_value = {"qwix": {"rules": []}}
        result = quantize_qwix.apply_qwix_on_abstract_model(
            self.mock_vllm_config)
        self.assertFalse(result)

    def test_returns_false_when_additional_config_is_missing(self):
        """Test it returns False when additional_config is missing."""
        result = quantize_qwix.apply_qwix_on_abstract_model(
            self.mock_vllm_config_no_additional_config)
        self.assertFalse(result)


if __name__ == '__main__':
    unittest.main()
