# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import unittest
from unittest.mock import MagicMock, mock_open, patch

import jax
import jax.numpy as jnp
import qwix
from flax import nnx
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from qwix._src.providers import ptq

import tpu_inference.models.jax.utils.qwix.qwix_utils as quantize_qwix  # noqa: E402
from tpu_inference.models.common.model_loader import apply_qwix_quantization
from tpu_inference.models.jax.utils.qwix.qwix_utils import (
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
    'tpu_inference': MagicMock(),
    'tpu_inference.logger': MagicMock(init_logger=lambda name: MagicMock()),
    'tpu_inference.models.jax.utils.qwix.qwix_utils': MagicMock(),
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
        self.mesh = Mesh(jax.devices(), ('model', ))
        self.rng = jax.random.PRNGKey(0)
        self.model = SimpleModel(rngs=nnx.Rngs(0))
        self.model.vllm_config = MagicMock()
        self.model.vllm_config.model_config.use_mla = False

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
        self.model.vllm_config.sharding_config.total_dp_size = 1

        with patch(
                "tpu_inference.models.jax.utils.qwix.qwix_utils.init_logger",
                return_value=MagicMock()
        ), patch(
                "tpu_inference.utils.hbm_usage_gb",
                return_value=[(0.0, 0.0), (0.0, 0.0)]
        ), patch(
                "tpu_inference.models.jax.utils.qwix.qwix_utils.create_kv_caches",
                return_value=self.mock_kv_caches
        ), patch(
                "tpu_inference.models.jax.utils.qwix.qwix_utils.quantization_config_file_path_to_dict",
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
                kv_cache_dtype="auto")

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

    @patch('tpu_inference.models.common.model_loader.nnx.jit')
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

        with patch('tpu_inference.utils.get_padded_num_heads',
                   return_value=128):
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
        'tpu_inference.models.jax.utils.qwix.qwix_utils.qwix_quantize_nnx_model'
    )
    @patch('tpu_inference.models.jax.utils.qwix.qwix_utils.utils')
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
        'tpu_inference.models.jax.utils.qwix.qwix_utils.qwix_quantize_nnx_model'
    )
    @patch('tpu_inference.models.jax.utils.qwix.qwix_utils.utils')
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
            "quantization": {
                "qwix": {
                    "use_abstract_model": True,
                    "rules": [{
                        "module_path": ".*",
                        "weight_qtype": "int8"
                    }]
                }
            }
        }

        self.mock_vllm_config_no_abstract_model = MagicMock()
        self.mock_vllm_config_no_abstract_model.additional_config = {
            "quantization": {
                "qwix": {
                    "rules": [{
                        "module_path": ".*",
                        "weight_qtype": "int8"
                    }]
                }
            }
        }

        self.mock_vllm_config_no_additional_config = MagicMock()
        self.mock_vllm_config_no_additional_config.additional_config = {}

    def test_returns_false_when_additional_config_is_missing(self):
        """Test it returns False when additional_config is missing."""
        result = quantize_qwix.apply_qwix_on_abstract_model(
            self.mock_vllm_config_no_additional_config)
        self.assertFalse(result)

    def test_returns_true_when_additional_config_is_present(self):
        """Test it returns False when additional_config is missing."""
        result = quantize_qwix.apply_qwix_on_abstract_model(
            self.mock_vllm_config)
        self.assertTrue(result)

    def test_returns_false_when_use_abstract_model_is_false(self):
        """Test it returns False when use_abstract_model is False."""
        result = quantize_qwix.apply_qwix_on_abstract_model(
            self.mock_vllm_config_no_abstract_model)
        self.assertFalse(result)


class TestLoadRandomWeightsIntoQwixAbstractModel(unittest.TestCase):
    """Tests for the load_random_weights_into_qwix_abstract_model function."""

    def setUp(self):
        """Set up a mock environment for testing."""
        if not jax.devices():
            self.skipTest(
                "JAX device not found, skipping JAX-dependent tests.")

        self.rng = jax.random.PRNGKey(0)
        self.mesh = Mesh(jax.devices(), ('data', ))
        self.quantization_config = {
            "weight_block_size": [64, 1],
        }

        # Mock model structure
        self.model = MagicMock(spec=['weight_loader', 'initialize_cache'])
        self.model.weight_loader = MagicMock(
            spec=['scale_dtype', 'scale_shape_map_for_random_weight_loading'])
        self.model.weight_loader.scale_dtype = jnp.float16
        self.model.weight_loader.scale_shape_map_for_random_weight_loading = {}

    @patch('tpu_inference.models.jax.utils.qwix.qwix_utils.nnx.iter_graph')
    @patch(
        'tpu_inference.models.jax.utils.qwix.qwix_utils.get_random_sharded_array'
    )
    def test_successful_initialization(self, mock_get_random_array,
                                       mock_iter_graph):
        """Test that variables are correctly initialized."""
        # Setup mock graph elements
        mock_weight_param = nnx.Param(jnp.empty((128, 64), dtype=jnp.int8),
                                      sharding=P('data', None))
        mock_scale_var = nnx.Variable(jnp.empty((1, 1), dtype=jnp.float16))
        mock_rng_var = nnx.Variable(jax.random.PRNGKey(0))
        mock_random_array = jax.numpy.ones(1)
        mock_get_random_array.return_value = mock_random_array

        self.model.weight_loader.scale_shape_map_for_random_weight_loading = {
            'attention.wq': (1, 1)
        }

        mock_iter_graph.return_value = [
            (('layers', '0', 'attention', 'wq', 'kernel'), mock_weight_param),
            (('layers', '0', 'attention', 'wq', 'array', 'scale'),
             mock_scale_var),
            (('rng', 'params', 'key'), mock_rng_var),
        ]

        quantize_qwix.load_random_weights_into_qwix_abstract_model(
            self.rng, self.model, self.mesh, self.quantization_config)

        # Assert weight is updated
        self.assertIs(mock_weight_param.value, mock_random_array)
        # Assert scale is updated
        self.assertIs(mock_scale_var.value, mock_random_array)
        # Assert RNG key is updated with the passed-in RNG
        self.assertIs(mock_rng_var.value, self.rng)
        # Assert initialize_cache is called
        self.model.initialize_cache.assert_called_once()

    def test_invalid_config_raises_assertion_error(self):
        """Test that an invalid quantization_block_sizes config raises an error."""
        invalid_config = {"weight_block_size": [64]}  # Length is 1, not 2
        with self.assertRaisesRegex(AssertionError,
                                    "Expected only 2 quantization block"):
            quantize_qwix.load_random_weights_into_qwix_abstract_model(
                self.rng, self.model, self.mesh, invalid_config)

    @patch('tpu_inference.models.jax.utils.qwix.qwix_utils.nnx.iter_graph')
    def test_param_shape_setting_no_scale_map(self, mock_iter_graph):
        """Test correct scale shape calculation when not in the map."""
        old_weight_param_val = jnp.empty((128, 64))
        mock_weight_param = nnx.Param(old_weight_param_val, dtype=jnp.int8)
        old_scale_var_val = jnp.empty((0, 0))
        mock_scale_var = nnx.Variable(old_scale_var_val)

        mock_iter_graph.return_value = [
            (('layers', '0', 'attention', 'wq', 'kernel'), mock_weight_param),
            (('layers', '0', 'attention', 'wq', 'array', 'scale'),
             mock_scale_var),
        ]

        with self.assertRaises(ValueError):
            quantize_qwix.load_random_weights_into_qwix_abstract_model(
                self.rng, self.model, self.mesh, self.quantization_config)

    @patch('tpu_inference.models.jax.utils.qwix.qwix_utils.nnx.iter_graph')
    def test_param_shape_setting_with_scale_map(self, mock_iter_graph):
        """Test correct scale shape calculation when in the map."""
        old_weight_param_val = jnp.empty((128, 64))
        mock_weight_param = nnx.Param(old_weight_param_val, dtype=jnp.int8)
        old_scale_var_val = jnp.empty((0, 0))
        mock_scale_var = nnx.Variable(old_scale_var_val)

        expected_scale_shape = (55, 34)

        self.model.weight_loader.scale_shape_map_for_random_weight_loading = {
            'attention.wq': expected_scale_shape
        }

        mock_iter_graph.return_value = [
            (('layers', '0', 'attention', 'wq', 'kernel'), mock_weight_param),
            (('layers', '0', 'attention', 'wq', 'array', 'scale'),
             mock_scale_var),
        ]

        quantize_qwix.load_random_weights_into_qwix_abstract_model(
            self.rng, self.model, self.mesh, self.quantization_config)

        new_weight_param_val = mock_weight_param.value
        new_scale_var_val = mock_scale_var.value

        actual_scale_shape = new_scale_var_val.shape

        expected_weight_shape = (128, 64)
        actual_weight_shape = new_weight_param_val.shape

        self.assertEqual(expected_scale_shape, actual_scale_shape)
        self.assertEqual(expected_weight_shape, actual_weight_shape)
        self.assertNotEqual(old_scale_var_val.shape, new_scale_var_val.shape)
        assert jnp.not_equal(old_weight_param_val, new_weight_param_val).all()

    @patch('jax.random.randint')
    @patch('jax.random.normal')
    @patch('jax.make_array_from_callback')
    def test_get_random_sharded_array_dtype_dispatch(self, mock_make_array,
                                                     mock_normal,
                                                     mock_randint):
        """Test that integer dtypes call randint and floats call normal."""
        # Test integer
        quantize_qwix.get_random_sharded_array(
            self.rng, self.mesh, nnx.Param(jnp.empty((2, 2)), sharding=P()),
            (2, 2), jnp.int8, "int_param")
        mock_randint.assert_called_once()
        mock_normal.assert_not_called()

        mock_randint.reset_mock()
        mock_normal.reset_mock()

        # Test float
        quantize_qwix.get_random_sharded_array(
            self.rng, self.mesh, nnx.Param(jnp.empty((2, 2)), sharding=P()),
            (2, 2), jnp.float32, "float_param")
        mock_randint.assert_not_called()
        mock_normal.assert_called_once()

    @patch("tpu_inference.models.jax.utils.qwix.qwix_utils.logger.warning")
    @patch("jax.make_array_from_callback")
    def test_get_random_sharded_array_sharding_fallback(
            self, mock_make_array, mock_logger_warning):
        """Test that sharding failure logs a warning and uses a fallback."""
        # First call raises an error, second call (fallback) succeeds
        mock_make_array.side_effect = [
            ValueError("Sharding failed"),
            MagicMock()
        ]

        param = nnx.Param(jnp.empty((2, 2)), sharding=P('data', None))
        quantize_qwix.get_random_sharded_array(self.rng, self.mesh, param,
                                               (2, 2), jnp.float32,
                                               "test_param")

        # Check that a warning was logged
        mock_logger_warning.assert_called_once()
        self.assertIn("Could not create sharded scale for test_param",
                      mock_logger_warning.call_args[0][0])

        # Check that the fallback was attempted with an empty PartitionSpec
        fallback_call_args = mock_make_array.call_args_list[1]
        fallback_sharding = fallback_call_args.args[1]
        self.assertEqual(fallback_sharding, NamedSharding(self.mesh, P()))


class TestManualQwixQuantization(unittest.TestCase):
    """Tests for manual Qwix quantization functions."""

    def setUp(self):
        if not jax.devices():
            self.skipTest(
                "JAX device not found, skipping JAX-dependent tests.")
        self.weight = jnp.ones((4, 4))
        self.inputs = jnp.ones((8, 4))
        self.qtype = jnp.int8
        self.channelwise_axes = [0]
        self.tiled_axes = {}
        self.calibration_method = 'max'

    @patch(
        'tpu_inference.models.jax.utils.qwix.qwix_utils.ptq.create_quantized_param'
    )
    def test_manually_quantize_qwix_weight(self, mock_create_param):
        """Test that manually_quantize_qwix_weight calls ptq.create_quantized_param correctly."""
        quantize_qwix.manually_quantize_qwix_weight(
            weight=self.weight,
            qtype=self.qtype,
            channelwise_axes=self.channelwise_axes,
            tiled_axes=self.tiled_axes,
            calibration_method=self.calibration_method)

        mock_create_param.assert_called_once()
        args, _ = mock_create_param.call_args
        passed_weight, passed_how_to_quantize = args

        self.assertTrue(jnp.array_equal(passed_weight, self.weight))
        self.assertIsInstance(passed_how_to_quantize, ptq.qarray.HowToQuantize)
        self.assertEqual(passed_how_to_quantize.qtype, self.qtype)
        self.assertEqual(passed_how_to_quantize.channelwise_axes,
                         self.channelwise_axes)
        self.assertEqual(passed_how_to_quantize.tiled_axes, self.tiled_axes)
        self.assertEqual(passed_how_to_quantize.calibration_method,
                         self.calibration_method)

    @patch('tpu_inference.models.jax.utils.qwix.qwix_utils.ptq.quantize_act')
    @patch('qwix.pallas.get_current_rule')
    def test_manually_quantize_qwix_activation(self, mock_get_rule,
                                               mock_quantize_act):
        """Test that manually_quantize_qwix_activation calls ptq.quantize_act correctly."""
        mock_rule = MagicMock()
        mock_rule.act_static_scale = False
        mock_get_rule.return_value = mock_rule
        rule_name = "test_rule"

        quantize_qwix.manually_quantize_qwix_activation(
            inputs=self.inputs,
            rule_name=rule_name,
            qtype=self.qtype,
            channelwise_axes=self.channelwise_axes,
            tiled_axes=self.tiled_axes,
            calibration_method=self.calibration_method)

        mock_get_rule.assert_called_once_with(rule_name)
        mock_quantize_act.assert_called_once()

        args, _ = mock_quantize_act.call_args
        passed_inputs, passed_how, passed_rule, passed_act_name = args

        self.assertTrue(jnp.array_equal(passed_inputs, self.inputs))
        self.assertIsInstance(passed_how, ptq.qarray.HowToQuantize)
        self.assertEqual(passed_how.qtype, self.qtype)
        self.assertEqual(passed_how.channelwise_axes, self.channelwise_axes)
        self.assertEqual(passed_how.tiled_axes, self.tiled_axes)
        self.assertEqual(passed_how.calibration_method,
                         self.calibration_method)
        self.assertIs(passed_rule, mock_rule)
        self.assertEqual(passed_act_name, "")  # act_name is hardcoded to ""

    @patch('qwix.pallas.get_current_rule')
    def test_manually_quantize_qwix_activation_static_scale_raises_error(
            self, mock_get_rule):
        """Test that an assertion is raised if the rule has static scale."""
        mock_rule = MagicMock()
        mock_rule.act_static_scale = True
        mock_get_rule.return_value = mock_rule

        with self.assertRaisesRegex(AssertionError,
                                    "Static scale not supported right now"):
            quantize_qwix.manually_quantize_qwix_activation(
                inputs=self.inputs,
                rule_name="any_rule",
                qtype=self.qtype,
                channelwise_axes=self.channelwise_axes,
                tiled_axes=self.tiled_axes,
                calibration_method=self.calibration_method)


class TestGetQuantDtypeFromQwixConfig(unittest.TestCase):
    """Tests for the get_quant_dtype_from_qwix_config function."""

    def setUp(self):
        self.mock_vllm_config = MagicMock()
        self.mock_vllm_config.additional_config = {}

    def test_get_quant_dtype_success(self):
        """Test successful extraction of dtypes from a valid config."""
        self.mock_vllm_config.additional_config = {
            "quantization": {
                "qwix": {
                    "scale_dtype":
                    "float16",
                    "rules": [
                        {
                            "module_path": ".*mlp.*",
                            "weight_qtype": "int4"
                        },
                        {
                            "module_path": ".*",
                            "weight_qtype": "int8"
                        },
                    ],
                }
            }
        }
        scale_dtype, quant_dtype = quantize_qwix.get_quant_dtype_from_qwix_config(
            self.mock_vllm_config)
        self.assertEqual(scale_dtype, jnp.float16)
        self.assertEqual(quant_dtype, jnp.int8)

    def test_get_quant_dtype_default_scale(self):
        """Test that scale_dtype defaults to bfloat16 when not specified."""
        self.mock_vllm_config.additional_config = {
            "quantization": {
                "qwix": {
                    "rules": [{
                        "module_path": ".*",
                        "weight_qtype": "int8"
                    }]
                }
            }
        }
        scale_dtype, quant_dtype = quantize_qwix.get_quant_dtype_from_qwix_config(
            self.mock_vllm_config)
        self.assertEqual(scale_dtype, jnp.bfloat16)
        self.assertEqual(quant_dtype, jnp.int8)

    def test_no_quantization_config_returns_defaults(self):
        """Test that default dtypes are returned when config is missing."""
        self.mock_vllm_config.additional_config = {}
        scale_dtype, quant_dtype = quantize_qwix.get_quant_dtype_from_qwix_config(
            self.mock_vllm_config)
        self.assertEqual(scale_dtype, jnp.bfloat16)
        self.assertIsNone(quant_dtype)

    def test_get_quant_dtype_no_wildcard_rule_returns_none(self):
        """Test that quant_dtype is None if no wildcard rule is found."""
        self.mock_vllm_config.additional_config = {
            "quantization": {
                "qwix": {
                    "rules": [{
                        "module_path": ".*mlp.*",
                        "weight_qtype": "int4"
                    }]
                }
            }
        }
        scale_dtype, quant_dtype = quantize_qwix.get_quant_dtype_from_qwix_config(
            self.mock_vllm_config)
        self.assertEqual(scale_dtype, jnp.bfloat16)
        self.assertIsNone(quant_dtype)

    def test_get_quant_dtype_wildcard_rule_missing_qtype_raises_error(self):
        """Test that an assertion is raised if the wildcard rule is missing weight_qtype."""
        self.mock_vllm_config.additional_config = {
            "quantization": {
                "qwix": {
                    "rules": [{
                        "module_path": ".*"
                    }]
                }
            }
        }
        with self.assertRaisesRegex(AssertionError,
                                    "Quantization dtype not found"):
            quantize_qwix.get_quant_dtype_from_qwix_config(
                self.mock_vllm_config)

    def test_get_quant_dtype_no_rules_key_returns_none(self):
        """Test that quant_dtype is None if 'rules' key is missing."""
        self.mock_vllm_config.additional_config = {
            "quantization": {
                "qwix": {
                    "scale_dtype": "float16",
                }
            }
        }
        scale_dtype, quant_dtype = quantize_qwix.get_quant_dtype_from_qwix_config(
            self.mock_vllm_config)
        self.assertEqual(scale_dtype, jnp.float16)
        self.assertIsNone(quant_dtype)


class TestGetDefaultQwixQuantizationConfig(unittest.TestCase):
    """Tests for the get_default_qwix_quantization_config function."""

    def setUp(self):
        # Mocking the default configs that the function expects to find in the module
        self.mock_deepseek_config = {
            "qwix": {
                "rules": [{
                    "module_path": ".*",
                    "tile_size": 0
                }]
            }
        }
        self.mock_llama_config = {"qwix": {"rules": [{"name": "llama_rule"}]}}
        self.mock_gpt_oss_config = {"qwix": {"rules": [{"name": "gpt_rule"}]}}

        # Patch the constants in the module where the function resides
        self.patchers = [
            patch(
                "tpu_inference.models.jax.utils.qwix.qwix_utils.DEFAULT_DEEPSEEK_FP4_MLP_MOE_FP8_ATTN_CONFIG",
                self.mock_deepseek_config),
            patch(
                "tpu_inference.models.jax.utils.qwix.qwix_utils.DEFAULT_LLAMA4_FP8_CONFIG",
                self.mock_llama_config),
            patch(
                "tpu_inference.models.jax.utils.qwix.qwix_utils.DEFAULT_GPT_OSS_FP4_CONFIG",
                self.mock_gpt_oss_config),
            patch("tpu_inference.models.jax.utils.qwix.qwix_utils.logger",
                  MagicMock())
        ]
        for p in self.patchers:
            p.start()

    def tearDown(self):
        for p in self.patchers:
            p.stop()

    def test_skip_quantization_returns_none(self):
        """Test that skip_quantization=True returns None immediately."""
        result = quantize_qwix.get_default_qwix_quantization_config(
            MagicMock(), True)
        self.assertIsNone(result)

    def test_unsupported_model_returns_none(self):
        """Test that an unknown model type returns None."""
        hf_config = MagicMock()
        hf_config.model_type = "unknown_model"
        result = quantize_qwix.get_default_qwix_quantization_config(
            hf_config, False)
        self.assertIsNone(result)

    def test_deepseek_v3_success(self):
        """Test DeepSeek V3 config with valid weight_block_size."""
        hf_config = MagicMock()
        hf_config.model_type = "DeepSeek_V3"
        hf_config.quantization_config = {
            "quant_method": "fp8",
            "weight_block_size": [1, 128]
        }

        result = quantize_qwix.get_default_qwix_quantization_config(
            hf_config, False)

        # Check if tile_size was updated from 0 to 128
        self.assertEqual(result["qwix"]["rules"][0]["tile_size"], 128)
        # Ensure it's a deep copy (original mock shouldn't change)
        self.assertEqual(
            self.mock_deepseek_config["qwix"]["rules"][0]["tile_size"], 0)

    def test_deepseek_v3_invalid_block_size(self):
        """Test DeepSeek V3 raises ValueError on invalid block size format."""
        hf_config = MagicMock()
        hf_config.model_type = "deepseek_v3"
        hf_config.quantization_config = {
            "quant_method": "fp8",
            "weight_block_size": [128]
        }

        with self.assertRaisesRegex(ValueError, "Invalid weight_block_size"):
            quantize_qwix.get_default_qwix_quantization_config(
                hf_config, False)

    def test_deepseek_v3_invalid_block_size_2d_subchannel(self):
        """Test DeepSeek V3 raises ValueError on invalid block size format."""
        hf_config = MagicMock()
        hf_config.model_type = "deepseek_v3"
        hf_config.quantization_config = {
            "quant_method": "fp8",
            "weight_block_size": [512, 512]
        }

        with self.assertRaisesRegex(AssertionError,
                                    "Expected first dimension to be 1"):
            quantize_qwix.get_default_qwix_quantization_config(
                hf_config, False)

    def test_deepseek_v3_no_weight_block_size(self):
        """Test DeepSeek V3 config with valid weight_block_size."""
        hf_config = MagicMock()
        hf_config.model_type = "DeepSeek_V3"
        hf_config.quantization_config = {
            "quant_method": "fp8",
        }

        with self.assertRaisesRegex(
                AssertionError,
                "Expected weight_block_size in quantization_config"):

            quantize_qwix.get_default_qwix_quantization_config(
                hf_config, False)

    def test_deepseek_v3_tile_size_assertion(self):
        """Test DeepSeek V3 raises AssertionError if tile_size is <= 1."""
        hf_config = MagicMock()
        hf_config.model_type = "deepseek_v3"
        hf_config.quantization_config = {
            "quant_method": "fp8",
            "weight_block_size": [1, 1]
        }

        with self.assertRaises(AssertionError):
            quantize_qwix.get_default_qwix_quantization_config(
                hf_config, False)

    def test_llama4_success(self):
        """Test Llama 4 default config path."""
        hf_config = MagicMock()
        hf_config.model_type = "llama4"
        hf_config.quantization_config = {"quant_method": "compressed-tensors"}

        result = quantize_qwix.get_default_qwix_quantization_config(
            hf_config, False)
        self.assertEqual(result, self.mock_llama_config)

    def test_gpt_oss_success(self):
        """Test GPT-OSS default config path."""
        hf_config = MagicMock()
        hf_config.model_type = "gpt_oss"
        hf_config.quantization_config = {"quant_method": "mxfp4"}

        result = quantize_qwix.get_default_qwix_quantization_config(
            hf_config, False)
        self.assertEqual(result, self.mock_gpt_oss_config)

    def test_missing_attributes_handled(self):
        """Test that function handles hf_config objects missing model_type safely."""
        hf_config = object()  # No attributes
        result = quantize_qwix.get_default_qwix_quantization_config(
            hf_config, False)
        self.assertIsNone(result)


if __name__ == '__main__':
    unittest.main()
