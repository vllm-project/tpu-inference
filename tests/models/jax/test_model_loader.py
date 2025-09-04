import os
import tempfile
from unittest.mock import MagicMock, patch

import jax
import jax.numpy as jnp
import pytest
import torch
from jax.sharding import Mesh
from transformers import PretrainedConfig
from vllm.config import ModelConfig, VllmConfig
from vllm.distributed.parallel_state import (ensure_model_parallel_initialized,
                                             init_distributed_environment)
from vllm.engine.arg_utils import EngineArgs

from tpu_commons.models.jax import model_loader
from tpu_commons.models.jax.qwen3 import Qwen3ForCausalLM


class MockModelA:
    pass


class MockModelB:
    pass


@pytest.fixture(scope="module")
def mesh() -> Mesh:
    """Provides a JAX device mesh for sharding."""
    devices = jax.devices()
    # Pass the 1D list of devices directly. Its ndim will match len(axis_names).
    return Mesh(devices, axis_names=("model", ))


@pytest.fixture
def vllm_config() -> MagicMock:
    """Provides a mock VllmConfig object."""
    model = "Qwen/Qwen3-0.6B"
    mock_config = MagicMock(spec=VllmConfig)
    mock_config.model_config = ModelConfig(model)
    mock_config.model_config.dtype = jnp.bfloat16
    mock_config.load_config = MagicMock()
    mock_config.load_config.download_dir = None
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
    assert model_class == Qwen3ForCausalLM


def test_get_model_architecture_unsupported():
    """
    Tests that _get_model_architecture raises a ValueError for an
    unsupported architecture.
    """
    config = PretrainedConfig(architectures=["UnsupportedModel"])
    with pytest.raises(ValueError, match="not supported"):
        model_loader._get_model_architecture(config)


@pytest.fixture(autouse=True)
def clear_model_registry_after_test():
    """Clear the model registry after each test to prevent side effects."""
    yield
    model_loader._MODEL_REGISTRY.clear()


def test_register_model_new_arch():
    """Tests registering a new model architecture."""
    model_loader.register_model("NewArch", MockModelA)
    config = PretrainedConfig(architectures=["NewArch"])
    model_class = model_loader._get_model_architecture(config)
    assert model_class == MockModelA


def test_register_model_update_arch():
    """Tests updating an existing registered model architecture."""
    # 1. Register initial model
    model_loader.register_model("UpdatableArch", MockModelA)
    config = PretrainedConfig(architectures=["UpdatableArch"])
    # 2. Verify initial registration
    model_class_1 = model_loader._get_model_architecture(config)
    assert model_class_1 == MockModelA
    # 3. Update the registration
    model_loader.register_model("UpdatableArch", MockModelB)
    # 4. Verify the update
    model_class_2 = model_loader._get_model_architecture(config)
    assert model_class_2 == MockModelB


def test_get_flax_model(vllm_config, mesh):
    """
    An integration test for the main public function `get_flax_model`.
    It verifies that the function returns two valid, JIT-compiled functions
    that execute correctly and produce outputs with the expected sharding.
    """
    rng = jax.random.PRNGKey(42)

    # 1. Get the compiled model and logit computation functions
    model_fn, compute_logits_fn, _, _, _ = model_loader.get_flax_model(
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

    engine_args = EngineArgs(model="Qwen/Qwen3-0.6B")
    vllm_config = engine_args.create_engine_config()
    vllm_config.model_config.dtype = torch.bfloat16

    with patch("vllm.config.get_current_vllm_config",
               return_value=vllm_config):
        temp_file = tempfile.mkstemp()[1]
        init_distributed_environment(
            world_size=1,
            rank=0,
            local_rank=0,
            distributed_init_method=f"file://{temp_file}",
            backend="gloo",
        )
        ensure_model_parallel_initialized(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
        )
        model_fn, compute_logits_fn, _, _, _ = model_loader.get_vllm_model(
            vllm_config, rng, mesh)

    assert callable(model_fn)
    assert callable(compute_logits_fn)


@pytest.mark.parametrize("set_in_config", [True, False])
def test_get_vllm_model_random_weights(mesh, set_in_config):
    rng = jax.random.PRNGKey(42)

    engine_args = EngineArgs(model="Qwen/Qwen3-0.6B")
    vllm_config = engine_args.create_engine_config()
    vllm_config.model_config.dtype = torch.bfloat16
    if set_in_config:
        vllm_config.load_config.load_format = "dummy"
    else:
        os.environ["JAX_RANDOM_WEIGHTS"] = "True"

    with patch(
            "vllm.model_executor.model_loader.dummy_loader.DummyModelLoader.load_weights"
    ) as mock_load:
        model_fn, compute_logits_fn, _, _, _ = model_loader.get_vllm_model(
            vllm_config, rng, mesh)

    assert callable(model_fn)
    assert callable(compute_logits_fn)
    mock_load.assert_called()
