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

import logging
import re
from contextlib import contextmanager
from typing import Optional
from unittest.mock import MagicMock

import jax
import numpy as np
import pytest
from flax import nnx
from flax.typing import PRNGKey
from jax import numpy as jnp
from jax.sharding import Mesh
from vllm.config import ModelConfig
from vllm.model_executor.model_loader import LoadConfig, register_model_loader
from vllm.model_executor.model_loader.default_loader import DefaultModelLoader

logger = logging.getLogger(__name__)

GBYTES = 1024 * 1024 * 1024


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


def _get_device_memory_bytes(device) -> Optional[int]:
    """Read current bytes_in_use from a device's memory_stats().

    Returns None if memory_stats() is unavailable (e.g., CPU-only devices).
    """
    try:
        stats = device.memory_stats()
        if not stats:
            return None
        return stats.get("bytes_in_use")
    except Exception:
        return None


def count_model_param_bytes(model) -> int:
    """Count total bytes of all parameters in a Flax nnx model."""
    _, state = nnx.split(model)
    total = 0
    for _, v in state.flat_state():
        if hasattr(v, 'value') and hasattr(v.value, 'nbytes'):
            total += v.value.nbytes
    return total


@contextmanager
def _assert_device_memory_bounded(
    max_increase_bytes: int,
    device=None,
    description: str = "operation",
):
    """Assert that device memory increase stays within a bound.

    Measures device bytes_in_use before and after the wrapped block.
    If the delta exceeds max_increase_bytes, raises AssertionError.

    When memory_stats() is unavailable (CPU-only environments), the
    check is silently skipped with a warning log.

    Args:
        max_increase_bytes: Maximum allowed increase in bytes.
        device: JAX device to monitor. Defaults to jax.local_devices()[0].
        description: Label for diagnostic messages.
    """
    if device is None:
        local_devs = jax.local_devices()
        if not local_devs:
            logger.warning("No local devices; skipping memory check for %s",
                           description)
            yield
            return
        device = local_devs[0]

    before = _get_device_memory_bytes(device)
    if before is None:
        logger.warning(
            "memory_stats() unavailable on %s; skipping memory check for %s",
            device, description)
        yield
        return

    yield

    after = _get_device_memory_bytes(device)
    if after is None:
        logger.warning("memory_stats() became unavailable during %s",
                       description)
        return

    delta = after - before
    logger.info(
        "Memory check [%s]: before=%.3f GB, after=%.3f GB, "
        "delta=%.3f GB, limit=%.3f GB", description, before / GBYTES,
        after / GBYTES, delta / GBYTES, max_increase_bytes / GBYTES)

    assert delta <= max_increase_bytes, (
        f"Device memory increase during {description} exceeded threshold: "
        f"delta={delta / GBYTES:.3f} GB > "
        f"max={max_increase_bytes / GBYTES:.3f} GB. "
        f"Before: {before / GBYTES:.3f} GB, After: {after / GBYTES:.3f} GB. "
        f"This may indicate weights or temporaries were placed on device "
        f"instead of being loaded on CPU first.")


@pytest.fixture
def assert_weight_loading_memory_bounded():
    """Fixture providing a function that monitors device memory during
    weight loading.

    Usage in tests::

        with assert_weight_loading_memory_bounded(model, "load_weights(...)"):
            loader.load_weights(model, model_config)
    """

    def _factory(model, description="operation"):
        model_param_bytes = count_model_param_bytes(model)
        max_memory_increase = int(model_param_bytes * 1.5)
        return _assert_device_memory_bounded(
            max_increase_bytes=max_memory_increase,
            description=description,
        )

    return _factory
