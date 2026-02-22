# Copyright 2026 Google LLC
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

import re
from unittest.mock import MagicMock

import jax
import numpy as np
import pytest
from flax.typing import PRNGKey
from jax import numpy as jnp
from jax.sharding import Mesh
from vllm.config import ModelConfig
from vllm.model_executor.model_loader import LoadConfig, register_model_loader
from vllm.model_executor.model_loader.default_loader import DefaultModelLoader


@pytest.fixture
def rng() -> PRNGKey:
    """Provides a reusable JAX PRNGKey."""
    return jax.random.PRNGKey(42)


@pytest.fixture(scope="package")
def mesh():
    """
    Creates a mesh with 1 device.
    """
    if not jax.devices():
        pytest.skip("No JAX devices available for mesh creation.")

    devices = np.array(jax.local_devices()[:1])
    num_devices = len(devices)
    assert num_devices == 1
    device_mesh = devices.reshape((num_devices, 1, 1, 1))

    with Mesh(device_mesh,
              axis_names=('data', 'attn_dp', 'expert', 'model')) as m:
        yield m


@pytest.fixture
def mock_vllm_config():

    class MockVllmConfig:

        def __init__(self, model: str, kv_cache_dtype: str):
            self.model_config = ModelConfig(model)
            self.model_config.dtype = jnp.bfloat16
            self.load_config = LoadConfig(load_format="auto")
            self.load_config.download_dir = None
            self.cache_config = MagicMock(cache_dtype=kv_cache_dtype)
            self.quant_config = None
            self.additional_config = {}

    return MockVllmConfig


@register_model_loader("skip_layers_model_loader_for_test")
class SkipLayersModelLoaderForTest(DefaultModelLoader):
    """Weight loader that skips layers beyond given limit.
    
    Some test are testing against weight loading, but it's meaningless
    to test all layers, assuming successfully loading the first few
    layers implies success of all layers. This special loader skips
    layers after given limit.
    """

    def __init__(self, load_config):
        self._num_layers_to_load = load_config.num_layers_to_load_for_test
        assert isinstance(self._num_layers_to_load, int)
        # `_prepare_weights` only recogonizes `load_format` from upstream.
        load_config.load_format = "auto"
        super().__init__(load_config)

    def get_all_weights(self, *args, **kwargs):
        for name, param in super().get_all_weights(*args, **kwargs):
            # If name matches "layers.\d+.", parse and skip if layer index is beyond limit
            match = re.search(r"layers\.(\d+)\.", name)
            if match:
                layer_idx = int(match.group(1))
                if layer_idx >= self._num_layers_to_load:
                    continue
            yield name, param
