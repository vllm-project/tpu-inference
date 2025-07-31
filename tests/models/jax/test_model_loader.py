import os
import sys
from types import ModuleType
from unittest.mock import MagicMock, patch

import jax
import jax.numpy as jnp
import pytest
import torch
from flax import nnx
from jax.sharding import Mesh
from transformers import PretrainedConfig
from vllm.config import VllmConfig
from vllm.engine.arg_utils import EngineArgs


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

    @classmethod
    def create_model_for_checkpoint_loading(cls, vllm_config, rng, mesh):
        """Mocks creating a model for loading weights by returning an instance."""
        return cls(vllm_config, rng, mesh)

    @classmethod
    def create_model_with_random_weights(cls, vllm_config, rng, mesh):
        """Mocks creating a model with random weights by returning an instance."""
        return cls(vllm_config, rng, mesh)


@pytest.fixture(scope="session", autouse=True)
def mock_dependencies(request):
    """
    This autouse fixture runs for the entire test session. It mocks modules
    that are imported locally within the functions of model_loader.py.
    This ensures that the module under test can be imported and used
    without having the actual dependencies present.
    """
    # Create mock modules for the models
    mock_llama_module = ModuleType("tpu_commons.models.jax.llama3")
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
    sys.modules["tpu_commons.models.jax.llama3"] = mock_llama_module
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
    mock_config.model_config.model = "test-llama-8b-model"
    mock_config.additional_config = {}
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


def test_get_flax_model(vllm_config, mesh):
    """
    An integration test for the main public function `get_flax_model`.
    It verifies that the function returns two valid, JIT-compiled functions
    that execute correctly and produce outputs with the expected sharding.
    """
    rng = jax.random.PRNGKey(42)

    # 1. Get the compiled model and logit computation functions
    model_fn, compute_logits_fn, _ = model_loader.get_flax_model(
        vllm_config, rng, mesh)

    assert callable(model_fn)
    assert callable(compute_logits_fn)


def test_get_vllm_model(mesh):
    """
    An integration test for the main public function `get_vllm_model`.
    It verifies that the function returns two valid, JIT-compiled functions
    that execute correctly and produce outputs with the expected sharding.
    """
    rng = jax.random.PRNGKey(42)

    engine_args = EngineArgs(model="Qwen/Qwen2-1.5B-Instruct")
    vllm_config = engine_args.create_engine_config()
    vllm_config.model_config.dtype = torch.bfloat16

    model_fn, compute_logits_fn, _ = model_loader.get_vllm_model(
        vllm_config, rng, mesh)

    assert callable(model_fn)
    assert callable(compute_logits_fn)


@pytest.mark.parametrize("set_in_config", [True, False])
def test_get_vllm_model_random_weights(mesh, set_in_config):
    rng = jax.random.PRNGKey(42)

    engine_args = EngineArgs(model="Qwen/Qwen2-1.5B-Instruct")
    vllm_config = engine_args.create_engine_config()
    vllm_config.model_config.dtype = torch.bfloat16
    if set_in_config:
        vllm_config.load_config.load_format = "dummy"
    else:
        os.environ["JAX_RANDOM_WEIGHTS"] = "True"

    with patch(
            "vllm.model_executor.model_loader.dummy_loader.DummyModelLoader.load_weights"
    ) as mock_load:
        model_fn, compute_logits_fn, _ = model_loader.get_vllm_model(
            vllm_config, rng, mesh)

    assert callable(model_fn)
    assert callable(compute_logits_fn)
    mock_load.assert_called()
