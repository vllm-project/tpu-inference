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

from unittest.mock import MagicMock

import jax
import numpy as np
import pytest
from flax.typing import PRNGKey
from jax import numpy as jnp
from jax.sharding import Mesh
from vllm.config import ModelConfig


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
            self.load_config = MagicMock()
            self.load_config.download_dir = None
            self.cache_config = MagicMock(cache_dtype=kv_cache_dtype)
            self.quant_config = None
            self.additional_config = {}

    return MockVllmConfig
