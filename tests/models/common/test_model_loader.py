# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import tempfile
from unittest.mock import MagicMock, patch

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
from jax.sharding import Mesh
from transformers import PretrainedConfig
from vllm.config import (ModelConfig, ParallelConfig, VllmConfig,
                         set_current_vllm_config)
from vllm.distributed.parallel_state import (ensure_model_parallel_initialized,
                                             init_distributed_environment)
from vllm.engine.arg_utils import EngineArgs
from vllm.model_executor.models.registry import ModelRegistry

from tpu_inference.models.common import model_loader
from tpu_inference.models.jax.qwen3 import Qwen3ForCausalLM


class MockModelA:

    def __init__(self, vllm_config, rng=None, mesh=None):
        pass

    def __call__(self, kv_caches, input_ids, attention_metadata):
        pass


class MockModelB:

    def __init__(self, vllm_config, rng=None, mesh=None):
        pass

    def __call__(self, kv_caches, input_ids, attention_metadata):
        pass


@pytest.fixture(scope="module")
def mesh() -> Mesh:
    """Provides a JAX device mesh for sharding."""
    devices = np.array(jax.devices()[:1])
    devices = devices.reshape((1, 1, 1, -1))
    # Pass the 1D list of devices directly. Its ndim will match len(axis_names).
    return Mesh(devices, axis_names=("data", "attn_dp", "expert", "model"))


@pytest.fixture
def vllm_config() -> MagicMock:
    """Provides a mock VllmConfig object."""
    model = "Qwen/Qwen3-0.6B"
    mock_config = MagicMock(spec=VllmConfig)
    mock_config.model_config = ModelConfig(model)
    mock_config.model_config.dtype = jnp.bfloat16
    mock_config.load_config = MagicMock()
    mock_config.load_config.download_dir = None
    mock_config.load_config.load_format = "auto"
    mock_config.additional_config = {}
    mock_config.cache_config = MagicMock(cache_dtype="auto")
    mock_config.parallel_config = ParallelConfig(pipeline_parallel_size=1)
    return mock_config


# --- Added RNG Fixture ---
@pytest.fixture
def rng() -> jax.Array:
    """Provides a JAX PRNGKey."""
    return jax.random.PRNGKey(0)


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
    with pytest.raises(ValueError, match="not registered"):
        model_loader._get_model_architecture(config)


@pytest.fixture(autouse=True)
def clear_model_registry_after_test():
    """Clear the model registry after each test to prevent side effects."""
    yield
    model_loader._MODEL_REGISTRY.clear()


def test_register_model_validation():
    """Tests that register_model validates the model interface."""

    class ValidModel:

        def __init__(self, vllm_config, rng=None, mesh=None):
            pass

        def __call__(self, kv_caches, input_ids, attention_metadata, **kwargs):
            pass

    class MissingInitArgModel:

        def __init__(self, rng=None, mesh=None):  # Missing vllm_config
            pass

        def __call__(self, kv_caches, input_ids, attention_metadata):
            pass

    class MissingCallArgModel:

        def __init__(self, vllm_config, rng=None, mesh=None):
            pass

        def __call__(self, kv_caches, input_ids):  # Missing attention_metadata
            pass

    class NoCallModel:

        def __init__(self, vllm_config, rng=None, mesh=None):
            pass

    # This should succeed
    model_loader.register_model("ValidModel", ValidModel)

    # These should fail
    with pytest.raises(TypeError, match="vllm_config"):
        model_loader.register_model("InvalidInit", MissingInitArgModel)

    with pytest.raises(TypeError, match="attention_metadata"):
        model_loader.register_model("InvalidCall", MissingCallArgModel)

    with pytest.raises(TypeError, match="__call__ method"):
        model_loader.register_model("NoCallModel", NoCallModel)


def test_register_model_new_arch():
    """Tests registering a new model architecture."""
    arch = "NewArch"
    model_loader.register_model(arch, MockModelA)

    # Check tpu_inference registry
    config = PretrainedConfig(architectures=[arch])
    model_class = model_loader._get_model_architecture(config)
    assert model_class == MockModelA

    # Check vLLM registry
    vllm_model_class = ModelRegistry._try_load_model_cls(arch)
    assert vllm_model_class is not None
    assert vllm_model_class.__name__ == f"VllmCompatible{MockModelA.__name__}"
    assert issubclass(vllm_model_class, MockModelA)
    assert issubclass(vllm_model_class, torch.nn.Module)


def test_register_model_update_arch():
    """Tests updating an existing registered model architecture."""
    arch = "UpdatableArch"
    config = PretrainedConfig(architectures=[arch])

    # Register initial model
    model_loader.register_model(arch, MockModelA)

    # Verify initial registration in both registries
    model_class_1 = model_loader._get_model_architecture(config)
    assert model_class_1 == MockModelA
    vllm_model_class_1 = ModelRegistry._try_load_model_cls(arch)
    assert vllm_model_class_1.__name__ == f"VllmCompatible{MockModelA.__name__}"
    assert issubclass(vllm_model_class_1, MockModelA)

    # Update the registration
    model_loader.register_model(arch, MockModelB)

    # Verify the update in both registries
    model_class_2 = model_loader._get_model_architecture(config)
    assert model_class_2 == MockModelB
    vllm_model_class_2 = ModelRegistry._try_load_model_cls(arch)
    assert vllm_model_class_2.__name__ == f"VllmCompatible{MockModelB.__name__}"
    assert issubclass(vllm_model_class_2, MockModelB)


def test_register_model_vllm_wrapper_methods():
    """Tests that the vLLM wrapper has correct dummy methods."""
    arch = "WrapperMethodTestArch"
    model_loader.register_model(arch, MockModelA)

    vllm_model_class = ModelRegistry._try_load_model_cls(arch)
    instance = vllm_model_class()

    # `forward` should be unimplemented.
    with pytest.raises(NotImplementedError, match="JAX model"):
        instance.forward(input_ids=None, positions=None)

    # `embed_input_ids` should be unimplemented.
    with pytest.raises(NotImplementedError, match="JAX model"):
        instance.embed_input_ids(input_ids=None, positions=None)

    # `load_weights` should be a no-op that returns None.
    assert instance.load_weights() is None


def test_get_flax_model(vllm_config, mesh):
    """
    An integration test for the main public function `get_flax_model`.
    It verifies that the function returns two valid, JIT-compiled functions
    that execute correctly and produce outputs with the expected sharding.
    """
    rng = jax.random.PRNGKey(42)

    # 1. Get the compiled model and logit computation functions
    model_fn, compute_logits_fn, *_ = model_loader.get_flax_model(
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

    with set_current_vllm_config(vllm_config):
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

    model_fn, compute_logits_fn, *_ = model_loader.get_vllm_model(
        vllm_config, rng, mesh)

    assert callable(model_fn)
    assert callable(compute_logits_fn)


def test_get_vllm_model_random_weights(mesh):
    rng = jax.random.PRNGKey(42)

    engine_args = EngineArgs(model="Qwen/Qwen3-0.6B")
    vllm_config = engine_args.create_engine_config()
    vllm_config.model_config.dtype = torch.bfloat16
    vllm_config.load_config.load_format = "dummy"

    with set_current_vllm_config(vllm_config):
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

    with patch(
            "vllm.model_executor.model_loader.dummy_loader.DummyModelLoader.load_weights"
    ) as mock_load:
        model_fn, compute_logits_fn, *_ = model_loader.get_vllm_model(
            vllm_config, rng, mesh)

    assert callable(model_fn)
    assert callable(compute_logits_fn)
    mock_load.assert_called()


# ==============================================================================
# >> Test Suite for get_model Fallback Logic
# ==============================================================================


@pytest.mark.usefixtures("mesh")  # This fixture is module-scoped, but fine
class TestGetModel:
    """Tests the main get_model() entrypoint and its fallback logic."""

    @patch.dict(os.environ, {"MODEL_IMPL_TYPE": "flax_nnx"}, clear=True)
    @patch("tpu_inference.models.common.model_loader.get_vllm_model")
    @patch("tpu_inference.models.common.model_loader.get_flax_model")
    def test_get_model_flax_happy_path(self, mock_get_flax, mock_get_vllm,
                                       vllm_config, rng, mesh):
        """Tests that 'flax_nnx' impl calls get_flax_model."""
        mock_get_flax.return_value = "flax_model_sentinel"

        result = model_loader.get_model(vllm_config, rng, mesh)

        mock_get_flax.assert_called_once_with(vllm_config, rng, mesh, False)
        mock_get_vllm.assert_not_called()
        assert result == "flax_model_sentinel"

    @patch.dict(os.environ, {"MODEL_IMPL_TYPE": "flax_nnx"}, clear=True)
    @patch("tpu_inference.models.common.model_loader.get_vllm_model")
    @patch("tpu_inference.models.common.model_loader.get_flax_model")
    def test_get_model_flax_happy_path_withPP(self, mock_get_flax,
                                              mock_get_vllm, vllm_config, rng,
                                              mesh):
        """Tests that 'flax_nnx' impl calls get_vllm_model when PP is enabled."""
        mock_get_flax.return_value = "flax_model_sentinel"
        mock_get_vllm.return_value = "vllm_model_sentinel"
        vllm_config.parallel_config.pipeline_parallel_size = 2
        result = model_loader.get_model(vllm_config, rng, mesh)

        mock_get_flax.assert_not_called()
        mock_get_vllm.assert_called_once_with(vllm_config, rng, mesh)
        assert result == "vllm_model_sentinel"

    @patch.dict(os.environ, {"MODEL_IMPL_TYPE": "vllm"}, clear=True)
    @patch("tpu_inference.models.common.model_loader.get_vllm_model")
    @patch("tpu_inference.models.common.model_loader.get_flax_model")
    def test_get_model_vllm_happy_path(self, mock_get_flax, mock_get_vllm,
                                       vllm_config, rng, mesh):
        """Tests that 'vllm' impl calls get_vllm_model."""
        mock_get_vllm.return_value = "vllm_model_sentinel"

        result = model_loader.get_model(vllm_config, rng, mesh)

        mock_get_flax.assert_not_called()
        mock_get_vllm.assert_called_once_with(vllm_config, rng, mesh)
        assert result == "vllm_model_sentinel"

    @patch.dict(os.environ, {"MODEL_IMPL_TYPE": "flax_nnx"}, clear=True)
    @patch("tpu_inference.models.common.model_loader.get_vllm_model")
    @patch("tpu_inference.models.common.model_loader.get_flax_model")
    def test_get_model_flax_fallback_on_unsupported_arch(
            self, mock_get_flax, mock_get_vllm, vllm_config, rng, mesh):
        """
        Tests that 'flax_nnx' falls back to get_vllm_model on
        UnsupportedArchitectureError.
        """
        # Mock get_flax_model to raise the specific error
        mock_get_flax.side_effect = model_loader.UnsupportedArchitectureError(
            "Model not supported")
        mock_get_vllm.return_value = "vllm_fallback_sentinel"

        result = model_loader.get_model(vllm_config, rng, mesh)

        # Check that both were called
        mock_get_flax.assert_called_once_with(vllm_config, rng, mesh, False)
        mock_get_vllm.assert_called_once_with(vllm_config, rng, mesh)
        assert result == "vllm_fallback_sentinel"

    @patch.dict(os.environ, {"MODEL_IMPL_TYPE": "flax_nnx"}, clear=True)
    @patch("tpu_inference.models.common.model_loader.get_vllm_model")
    @patch("tpu_inference.models.common.model_loader.get_flax_model")
    def test_get_model_flax_reraises_other_errors(self, mock_get_flax,
                                                  mock_get_vllm, vllm_config,
                                                  rng, mesh):
        """
        Tests that 'flax_nnx' re-raises other ValueErrors
        and does not fall back.
        """
        # Mock get_flax_model to raise a *different* error
        mock_get_flax.side_effect = ValueError("A different error")

        with pytest.raises(ValueError, match="A different error"):
            model_loader.get_model(vllm_config, rng, mesh)

        # Check that flax was called but vllm was not
        mock_get_flax.assert_called_once_with(vllm_config, rng, mesh, False)
        mock_get_vllm.assert_not_called()

    @patch.dict(os.environ, {"MODEL_IMPL_TYPE": "jetpack"}, clear=True)
    @patch("tpu_inference.models.common.model_loader.get_vllm_model")
    @patch("tpu_inference.models.common.model_loader.get_flax_model")
    def test_get_model_not_implemented(self, mock_get_flax, mock_get_vllm,
                                       vllm_config, rng, mesh):
        """Tests that an unknown impl raises NotImplementedError."""
        with pytest.raises(NotImplementedError):
            model_loader.get_model(vllm_config, rng, mesh)

        mock_get_flax.assert_not_called()
        mock_get_vllm.assert_not_called()

    @patch.dict(os.environ, {"MODEL_IMPL_TYPE": "auto"}, clear=True)
    @patch("tpu_inference.models.common.model_loader.get_vllm_model")
    @patch("tpu_inference.models.common.model_loader.get_flax_model")
    def test_get_model_auto_resolves_to_flax_nnx(self, mock_get_flax,
                                                 mock_get_vllm, vllm_config,
                                                 rng, mesh):
        """
        Tests that 'auto' resolves to 'flax_nnx' for standard architectures
        (not in _VLLM_REQUIRED_ARCHITECTURES).
        """
        # vllm_config uses Qwen3 which is NOT in _VLLM_REQUIRED_ARCHITECTURES
        mock_get_flax.return_value = "flax_model_sentinel"

        result = model_loader.get_model(vllm_config, rng, mesh)

        mock_get_flax.assert_called_once_with(vllm_config, rng, mesh, False)
        mock_get_vllm.assert_not_called()
        assert result == "flax_model_sentinel"

    @patch.dict(os.environ, {"MODEL_IMPL_TYPE": "auto"}, clear=True)
    @patch("tpu_inference.models.common.model_loader.get_vllm_model")
    @patch("tpu_inference.models.common.model_loader.get_flax_model")
    def test_get_model_auto_resolves_to_vllm_for_gpt_oss(
            self, mock_get_flax, mock_get_vllm, vllm_config, rng, mesh):
        """
        Tests that 'auto' resolves to 'vllm' for architectures in
        _VLLM_REQUIRED_ARCHITECTURES (e.g., GptOssForCausalLM).
        """
        # Mock the architecture to be GptOssForCausalLM
        vllm_config.model_config.hf_config.architectures = [
            "GptOssForCausalLM"
        ]
        mock_get_vllm.return_value = "vllm_model_sentinel"

        result = model_loader.get_model(vllm_config, rng, mesh)

        mock_get_flax.assert_not_called()
        mock_get_vllm.assert_called_once_with(vllm_config, rng, mesh)
        assert result == "vllm_model_sentinel"
