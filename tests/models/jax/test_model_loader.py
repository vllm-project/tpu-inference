import sys
from types import ModuleType
from unittest.mock import MagicMock, patch

import jax
import jax.numpy as jnp
import pytest
from flax import nnx
from jax.sharding import Mesh
from transformers import PretrainedConfig
from vllm.config import VllmConfig


class MockCausalLM(nnx.Module):
    """A mock nnx.Module that mimics the behavior of a causal language model."""

    def __init__(self, vllm_config: VllmConfig, rng: jax.Array, mesh: Mesh):
        """Initializes a dummy parameter."""
        # Using the inputs to show they are passed correctly
        self.config = vllm_config
        self.mesh = mesh
        # Create a dummy parameter to ensure weight loading works
        self.w = nnx.Param(jax.random.normal(rng, (4, 4)))
        self.hidden_dim = 128
        self.vocab_size = 256

    def load_weights(self, rng: jax.Array):
        """Simulates loading weights by setting the parameter to a known value."""
        self.w.value = jnp.ones_like(self.w.value)

    def __call__(self, input_ids: jax.Array, positions: jax.Array,
                 kv_cache: jax.Array) -> tuple[jax.Array, jax.Array]:
        """Simulates a forward pass."""
        # Simulate some work and return dummy outputs with expected shapes/types
        batch_size, seq_len = input_ids.shape
        new_kv_cache = kv_cache + jnp.sum(self.w.value)  # Dummy op
        hidden_states = jnp.ones((batch_size, self.hidden_dim))
        return new_kv_cache, hidden_states

    def compute_logits(self, hidden_states: jax.Array) -> jax.Array:
        """Simulates computing logits from hidden states."""
        batch_size = hidden_states.shape[0]
        logits = jnp.ones((batch_size, self.vocab_size))
        return logits * jnp.mean(self.w.value)  # Dummy op


@pytest.fixture(scope="session", autouse=True)
def mock_dependencies(request):
    """
    This autouse fixture runs for the entire test session. It mocks modules
    that are imported locally within the functions of model_loader.py.
    This ensures that the module under test can be imported and used
    without having the actual dependencies present.
    """
    # Create mock modules for the models
    mock_llama_module = ModuleType("tpu_commons.models.jax.llama")
    setattr(mock_llama_module, "LlamaForCausalLM", MockCausalLM)

    mock_qwen2_module = ModuleType("tpu_commons.models.jax.qwen2")
    setattr(mock_qwen2_module, "Qwen2ForCausalLM", MockCausalLM)

    # Create a mock for the logger
    mock_logger_module = ModuleType("tpu_commons.logger")
    mock_logger = MagicMock()
    setattr(mock_logger_module, "init_logger",
            MagicMock(return_value=mock_logger))

    # Add the mock modules to sys.modules
    original_modules = sys.modules.copy()
    sys.modules["tpu_commons.models.jax.llama"] = mock_llama_module
    sys.modules["tpu_commons.models.jax.qwen2"] = mock_qwen2_module
    sys.modules["tpu_commons.logger"] = mock_logger_module

    # Allow the tests to import the module under test
    global model_loader
    from tpu_commons.models.jax import model_loader

    # Teardown: restore original sys.modules
    def fin():
        sys.modules.clear()
        sys.modules.update(original_modules)

    request.addfinalizer(fin)


@pytest.fixture(scope="module")
def mesh() -> Mesh:
    """Provides a JAX device mesh for sharding."""
    devices = jax.devices()
    # Pass the 1D list of devices directly. Its ndim will match len(axis_names).
    return Mesh(devices, axis_names=("model", ))


@pytest.fixture
def vllm_config() -> MagicMock:
    """Provides a mock VllmConfig object."""
    mock_config = MagicMock(spec=VllmConfig)
    mock_config.model_config.hf_config = PretrainedConfig(
        architectures=["LlamaForCausalLM"])
    return mock_config


# ==============================================================================
# >> Test Cases
# ==============================================================================


def test_get_model_architecture_supported(vllm_config):
    """
    Tests that _get_model_architecture returns the correct model class
    for a supported architecture.
    """
    config = vllm_config.model_config.hf_config
    model_class = model_loader._get_model_architecture(config)
    assert model_class == MockCausalLM


def test_get_model_architecture_unsupported():
    """
    Tests that _get_model_architecture raises a ValueError for an
    unsupported architecture.
    """
    config = PretrainedConfig(architectures=["UnsupportedModel"])
    with pytest.raises(ValueError, match="not supported"):
        model_loader._get_model_architecture(config)


# def test_get_flax_model(vllm_config, mesh):
#     """
#     An integration test for the main public function `get_flax_model`.
#     It verifies that the function returns two valid, JIT-compiled functions
#     that execute correctly and produce outputs with the expected sharding.
#     """
#     rng = jax.random.PRNGKey(42)

#     # 1. Get the compiled model and logit computation functions
#     model_fn, compute_logits_fn = model_loader.get_flax_model(
#         vllm_config, rng, mesh)

#     assert callable(model_fn)
#     assert callable(compute_logits_fn)


@pytest.fixture
def mock_causal_lm_instance(vllm_config, mesh) -> MockCausalLM:
    """Provides an instance of the MockCausalLM for testing."""
    rng = jax.random.PRNGKey(0)
    return MockCausalLM(vllm_config, rng, mesh)


def test_apply_qwix_quantization_no_config(vllm_config,
                                           mock_causal_lm_instance, mesh):
    """
    Tests that the model is returned unmodified when no 'quantization' key
    is in the vllm_config.
    """
    # Setup: Ensure no quantization config is present
    vllm_config.additional_config = {}
    rng = jax.random.PRNGKey(0)
    input_model = mock_causal_lm_instance

    # Mock the functions that should not be called
    with patch(
            "tpu_commons.models.jax.model_loader.quantization_config_file_path_to_dict"
    ) as mock_convert, patch(
            "tpu_commons.models.jax.model_loader.qwix_quantize_nnx_model"
    ) as mock_quantize:

        # Action
        output_model = model_loader._apply_qwix_quantization(
            vllm_config, input_model, rng, mesh)

        # Assertions
        assert output_model is input_model  # Should be the same object
        mock_convert.assert_not_called()
        mock_quantize.assert_not_called()


def test_apply_qwix_quantization_no_qwix_rules(vllm_config,
                                               mock_causal_lm_instance, mesh):
    """
    Tests that the model is returned unmodified when the quantization config
    is present but lacks qwix rules.
    """
    # Setup: Provide a quantization config file path
    vllm_config.additional_config = {}
    rng = jax.random.PRNGKey(0)
    input_model = mock_causal_lm_instance
    output_model = model_loader._apply_qwix_quantization(
        vllm_config, input_model, rng, mesh)

    assert output_model is input_model


def test_apply_qwix_quantization_is_applied(vllm_config,
                                            mock_causal_lm_instance, mesh):
    """
    Tests that qwix quantization is correctly applied when a valid config
    with qwix rules is provided.
    """
    # 1. Setup
    rng = jax.random.PRNGKey(0)
    input_model = mock_causal_lm_instance
    quantized_model_mock = MagicMock(spec=nnx.Module)

    # Configure vllm_config with all necessary parameters for quantization
    vllm_config.additional_config = {"quantization": "my_qwix_config.json"}
    vllm_config.cache_config = MagicMock()
    vllm_config.cache_config.block_size = 16
    vllm_config.model_config.get_head_size.return_value = 64
    vllm_config.model_config.get_total_num_kv_heads.return_value = 8
    vllm_config.model_config.hf_config.num_hidden_layers = 12

    # Mock the config that would be loaded from the file
    mock_qwix_rules = {
        "rules": [{
            "module_path": ".*",
            "weight_qtype": "int8",
            "act_qtype": "int8"
        }]
    }
    mock_config_dict = {"qwix": mock_qwix_rules}

    with patch(
            "tpu_commons.models.jax.model_loader.quantization_config_file_path_to_dict",
            return_value=mock_config_dict) as mock_convert, \
                    patch("flax.nnx.jit", return_value=quantized_model_mock) as mock_jit:

        model_loader._apply_qwix_quantization(vllm_config, input_model, rng,
                                              mesh)

        mock_convert.assert_called_once_with("my_qwix_config.json")

        mock_jit.assert_called_once()
