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
"""Unit tests for pathways_dummy_loader module."""

from unittest.mock import MagicMock, patch

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
from flax import nnx
from jax.sharding import (Mesh, NamedSharding, PartitionSpec,
                          SingleDeviceSharding)

from tpu_inference.layers.jax import JaxModule, JaxModuleList
from tpu_inference.models.common.pathways_dummy_loader import (
    _HIGH, _LOW, PathwaysDummyModelLoader,
    _process_weights_after_loading_jax, create_dummy_weights_on_tpu,
    is_pathways_dummy_load, load_dummy_weights_jax)

# ==============================================================================
# >> Fixtures
# ==============================================================================


@pytest.fixture(scope="module")
def mesh() -> Mesh:
    """Provides a single-device JAX mesh for testing."""
    devices = np.array(jax.devices()[:1]).reshape((1, 1))
    return Mesh(devices, axis_names=("data", "model"))


@pytest.fixture
def single_device_mesh() -> Mesh:
    """Provides a 1-device mesh with standard axis names."""
    devices = np.array(jax.devices()[:1]).reshape((1, ))
    return Mesh(devices, axis_names=("model", ))


# ==============================================================================
# >> Tests for is_pathways_dummy_load
# ==============================================================================


class TestIsPathwaysDummyLoad:
    """Tests for the is_pathways_dummy_load() helper function."""

    @patch("tpu_inference.models.common.pathways_dummy_loader.vllm_envs")
    def test_returns_false_when_not_using_pathways(self, mock_envs):
        """Should return False when VLLM_TPU_USING_PATHWAYS is False."""
        mock_envs.VLLM_TPU_USING_PATHWAYS = False
        assert is_pathways_dummy_load() is False

    @patch(
        "tpu_inference.models.common.pathways_dummy_loader.get_current_vllm_config"
    )
    @patch("tpu_inference.models.common.pathways_dummy_loader.vllm_envs")
    def test_returns_true_for_dummy_format(self, mock_envs, mock_get_config):
        """Should return True when using Pathways with load_format='dummy'."""
        mock_envs.VLLM_TPU_USING_PATHWAYS = True
        mock_config = MagicMock()
        mock_config.load_config.load_format = "dummy"
        mock_get_config.return_value = mock_config
        assert is_pathways_dummy_load() is True

    @patch(
        "tpu_inference.models.common.pathways_dummy_loader.get_current_vllm_config"
    )
    @patch("tpu_inference.models.common.pathways_dummy_loader.vllm_envs")
    def test_returns_true_for_pathways_dummy_format(self, mock_envs,
                                                    mock_get_config):
        """Should return True when using Pathways with load_format='pathways_dummy'."""
        mock_envs.VLLM_TPU_USING_PATHWAYS = True
        mock_config = MagicMock()
        mock_config.load_config.load_format = "pathways_dummy"
        mock_get_config.return_value = mock_config
        assert is_pathways_dummy_load() is True

    @patch(
        "tpu_inference.models.common.pathways_dummy_loader.get_current_vllm_config"
    )
    @patch("tpu_inference.models.common.pathways_dummy_loader.vllm_envs")
    def test_returns_false_for_other_load_formats(self, mock_envs,
                                                  mock_get_config):
        """Should return False for non-dummy load formats even with Pathways."""
        mock_envs.VLLM_TPU_USING_PATHWAYS = True
        mock_config = MagicMock()
        mock_config.load_config.load_format = "auto"
        mock_get_config.return_value = mock_config
        assert is_pathways_dummy_load() is False


# ==============================================================================
# >> Tests for create_dummy_weights_on_tpu
# ==============================================================================


class TestCreateDummyWeightsOnTpu:
    """Tests for the create_dummy_weights_on_tpu() function."""

    def test_returns_correct_shape(self, mesh):
        """Generated array should have the requested shape."""
        shape = (4, 8)
        sharding = NamedSharding(mesh, PartitionSpec())
        result = create_dummy_weights_on_tpu(sharding=sharding,
                                             weight_shape=shape,
                                             weight_dtype=jnp.float32)
        assert result.shape == shape

    def test_returns_correct_dtype(self, mesh):
        """Generated array should have the requested dtype."""
        sharding = NamedSharding(mesh, PartitionSpec())
        for dtype in [jnp.float32, jnp.bfloat16, jnp.float16]:
            result = create_dummy_weights_on_tpu(sharding=sharding,
                                                 weight_shape=(2, 3),
                                                 weight_dtype=dtype)
            assert result.dtype == dtype

    def test_values_within_range(self, mesh):
        """All values should be in [_LOW, _HIGH]."""
        sharding = NamedSharding(mesh, PartitionSpec())
        result = create_dummy_weights_on_tpu(sharding=sharding,
                                             weight_shape=(100, 100),
                                             weight_dtype=jnp.float32)
        values = np.array(result)
        assert np.all(values >= _LOW)
        assert np.all(values <= _HIGH)

    def test_deterministic_with_same_seed(self, mesh):
        """Calling twice should produce identical results (fixed seed)."""
        sharding = NamedSharding(mesh, PartitionSpec())
        a = create_dummy_weights_on_tpu(sharding=sharding,
                                        weight_shape=(4, 4),
                                        weight_dtype=jnp.float32)
        b = create_dummy_weights_on_tpu(sharding=sharding,
                                        weight_shape=(4, 4),
                                        weight_dtype=jnp.float32)
        np.testing.assert_array_equal(np.array(a), np.array(b))

    def test_sharding_is_applied(self, mesh):
        """Result should carry the requested sharding."""
        spec = PartitionSpec("data", "model")
        sharding = NamedSharding(mesh, spec)
        result = create_dummy_weights_on_tpu(sharding=sharding,
                                             weight_shape=(4, 8),
                                             weight_dtype=jnp.float32)
        assert result.sharding == sharding

    def test_scalar_shape(self, mesh):
        """Should handle scalar (empty-tuple) shapes."""
        sharding = NamedSharding(mesh, PartitionSpec())
        result = create_dummy_weights_on_tpu(sharding=sharding,
                                             weight_shape=(),
                                             weight_dtype=jnp.float32)
        assert result.shape == ()

    def test_1d_shape(self, single_device_mesh):
        """Should handle 1-D shapes."""
        sharding = NamedSharding(single_device_mesh, PartitionSpec("model"))
        result = create_dummy_weights_on_tpu(sharding=sharding,
                                             weight_shape=(16, ),
                                             weight_dtype=jnp.float32)
        assert result.shape == (16, )


# ==============================================================================
# >> Tests for load_dummy_weights_jax
# ==============================================================================


class _SimpleLinear(JaxModule):
    """A minimal JaxModule with a single weight param for testing."""

    def __init__(self, in_features: int, out_features: int, dtype=jnp.float32):
        super().__init__()
        abstract = jnp.zeros((in_features, out_features), dtype=dtype)
        self.weight = nnx.Param(abstract)
        self.weight.set_metadata("sharding", PartitionSpec())

    def __call__(self, x):
        return x @ self.weight.value


class _NestedModel(JaxModule):
    """A JaxModule containing a JaxModuleList of _SimpleLinear."""

    def __init__(self, dtype=jnp.float32):
        super().__init__()
        self.layers = JaxModuleList([
            _SimpleLinear(4, 8, dtype=dtype),
            _SimpleLinear(8, 4, dtype=dtype),
        ])

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class TestLoadDummyWeightsJax:
    """Tests for load_dummy_weights_jax()."""

    def test_fills_simple_model_weights(self, mesh):
        """All params should be filled with non-zero dummy data."""
        model = _SimpleLinear(4, 8)
        # Initially all zeros
        assert np.all(np.array(model.weight.value) == 0.0)

        with patch(
                "tpu_inference.models.common.pathways_dummy_loader._process_weights_after_loading_jax"
        ):
            load_dummy_weights_jax(model, mesh)

        # After loading, should not be all zeros
        values = np.array(model.weight.value)
        assert values.shape == (4, 8)
        assert np.all(values >= _LOW)
        assert np.all(values <= _HIGH)

    def test_fills_nested_model_weights(self, mesh):
        """Nested model params should all be filled."""
        model = _NestedModel()

        with patch(
                "tpu_inference.models.common.pathways_dummy_loader._process_weights_after_loading_jax"
        ):
            load_dummy_weights_jax(model, mesh)

        for _name, param in model.named_parameters():
            values = np.array(param.value)
            assert np.all(values >= _LOW)
            assert np.all(values <= _HIGH)

    def test_calls_process_weights_after_loading(self, mesh):
        """Should call _process_weights_after_loading_jax after loading."""
        model = _SimpleLinear(4, 8)

        with patch(
                "tpu_inference.models.common.pathways_dummy_loader._process_weights_after_loading_jax"
        ) as mock_process:
            load_dummy_weights_jax(model, mesh)
            mock_process.assert_called_once_with(model)

    def test_respects_param_dtype(self, mesh):
        """Loaded weights should have the same dtype as the original param."""
        model = _SimpleLinear(4, 8, dtype=jnp.bfloat16)

        with patch(
                "tpu_inference.models.common.pathways_dummy_loader._process_weights_after_loading_jax"
        ):
            load_dummy_weights_jax(model, mesh)

        assert model.weight.value.dtype == jnp.bfloat16

    def test_handles_named_sharding_spec(self, mesh):
        """Params with NamedSharding metadata should be handled correctly."""
        model = _SimpleLinear(4, 8)
        sharding = NamedSharding(mesh, PartitionSpec("data", "model"))
        model.weight.set_metadata("sharding", sharding)

        with patch(
                "tpu_inference.models.common.pathways_dummy_loader._process_weights_after_loading_jax"
        ):
            load_dummy_weights_jax(model, mesh)

        values = np.array(model.weight.value)
        assert values.shape == (4, 8)

    def test_handles_single_device_sharding_spec(self, mesh):
        """Params with SingleDeviceSharding metadata should be handled correctly."""
        model = _SimpleLinear(4, 8)
        device = jax.devices()[0]
        model.weight.set_metadata("sharding", SingleDeviceSharding(device))

        with patch(
                "tpu_inference.models.common.pathways_dummy_loader._process_weights_after_loading_jax"
        ):
            load_dummy_weights_jax(model, mesh)

        values = np.array(model.weight.value)
        assert values.shape == (4, 8)


# ==============================================================================
# >> Tests for _process_weights_after_loading_jax
# ==============================================================================


class TestProcessWeightsAfterLoadingJax:
    """Tests for _process_weights_after_loading_jax()."""

    def test_calls_quant_method_process_weights(self):
        """Should call process_weights_after_loading on quant method."""
        from tpu_inference.layers.jax.quantization import QuantizeMethodBase

        # Create a concrete subclass of QuantizeMethodBase so isinstance checks pass
        class FakeQuantMethod(QuantizeMethodBase):

            def create_weights(self, *args, **kwargs):
                pass

            def apply(self, *args, **kwargs):
                pass

            def process_weights_after_loading(self, module):
                pass

        mock_quant = MagicMock(spec=FakeQuantMethod)
        # Make isinstance(mock_quant, QuantizeMethodBase) return True
        mock_quant.__class__ = FakeQuantMethod

        module = MagicMock(spec=JaxModule)
        module.quant_method = mock_quant

        _process_weights_after_loading_jax(module)

        mock_quant.process_weights_after_loading.assert_called_once_with(
            module)

    def test_recurses_through_jax_module_list(self):
        """Should recursively process children in JaxModuleList."""
        child1 = MagicMock(spec=JaxModule)
        child1.quant_method = None
        child1.named_children.return_value = []

        child2 = MagicMock(spec=JaxModule)
        child2.quant_method = None
        child2.named_children.return_value = []

        module_list = JaxModuleList([child1, child2])

        # Should not raise
        _process_weights_after_loading_jax(module_list)

    def test_recurses_through_named_children(self):
        """Should recursively process named_children on non-list modules."""
        child = MagicMock(spec=JaxModule)
        child.quant_method = None
        child.named_children.return_value = []

        parent = MagicMock(spec=JaxModule)
        parent.quant_method = None
        parent.named_children.return_value = [("child", child)]

        # Should not raise
        _process_weights_after_loading_jax(parent)


# ==============================================================================
# >> Tests for PathwaysDummyModelLoader
# ==============================================================================


class TestPathwaysDummyModelLoader:
    """Tests for the PathwaysDummyModelLoader class."""

    def test_download_model_is_noop(self):
        """download_model should do nothing."""
        load_config = MagicMock()
        loader = PathwaysDummyModelLoader(load_config)
        # Should not raise
        loader.download_model(model_config=MagicMock())

    def test_load_weights_dispatches_to_jax_backend(self, mesh):
        """load_weights should call load_dummy_weights_jax for JaxModule models."""
        load_config = MagicMock()
        loader = PathwaysDummyModelLoader(load_config)
        model = _SimpleLinear(4, 8)

        with patch(
                "tpu_inference.models.common.pathways_dummy_loader.load_dummy_weights_jax"
        ) as mock_load_jax:
            with patch(
                    "tpu_inference.models.common.pathways_dummy_loader.jax.sharding.get_mesh",
                    return_value=mesh,
            ):
                loader.load_weights(model, model_config=MagicMock())

            mock_load_jax.assert_called_once_with(model, mesh)

            mock_load_jax.assert_called_once_with(model, mesh)

    def test_load_weights_noop_for_non_jax_model(self):
        """load_weights should be a no-op for non-JaxModule (torchax) models."""
        load_config = MagicMock()
        loader = PathwaysDummyModelLoader(load_config)
        model = MagicMock()  # Not a JaxModule

        with patch(
                "tpu_inference.models.common.pathways_dummy_loader.load_dummy_weights_jax"
        ) as mock_load_jax:
            loader.load_weights(model, model_config=MagicMock())
            mock_load_jax.assert_not_called()

    def test_load_weights_raises_without_mesh_for_jax(self):
        """load_weights should raise RuntimeError when no mesh is set for JaxModule."""
        load_config = MagicMock()
        loader = PathwaysDummyModelLoader(load_config)
        model = _SimpleLinear(4, 8)

        with patch(
                "tpu_inference.models.common.pathways_dummy_loader.jax.sharding.get_mesh",
                return_value=None,
        ):
            with pytest.raises(RuntimeError, match="active JAX mesh"):
                loader.load_weights(model, model_config=MagicMock())

    def test_load_model_initializes_and_processes_weights(self):
        """load_model should initialize model and call process_weights_after_loading."""
        load_config = MagicMock()
        load_config.device = None
        loader = PathwaysDummyModelLoader(load_config)

        mock_vllm_config = MagicMock()
        mock_vllm_config.device_config.device = "cpu"
        mock_vllm_config.load_config = load_config

        mock_model_config = MagicMock()
        mock_model_config.dtype = torch.float32

        mock_model = MagicMock()
        mock_model.eval.return_value = mock_model

        with patch(
                "tpu_inference.models.common.pathways_dummy_loader.initialize_model",
                return_value=mock_model,
        ) as mock_init:
            with patch(
                    "tpu_inference.models.common.pathways_dummy_loader.process_weights_after_loading"
            ) as mock_process:
                result = loader.load_model(
                    vllm_config=mock_vllm_config,
                    model_config=mock_model_config,
                )

        mock_init.assert_called_once()
        mock_process.assert_called_once()
        mock_model.eval.assert_called_once()
        assert result is mock_model


# ==============================================================================
# >> Registration test
# ==============================================================================


class TestRegistration:
    """Tests that the loader is properly registered."""

    def test_pathways_dummy_loader_is_registered(self):
        """PathwaysDummyModelLoader should be discoverable via the registry."""
        from vllm.model_executor.model_loader import get_model_loader

        load_config = MagicMock()
        load_config.load_format = "pathways_dummy"
        loader = get_model_loader(load_config)
        assert isinstance(loader, PathwaysDummyModelLoader)
