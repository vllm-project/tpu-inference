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
import threading
from contextlib import contextmanager
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

from tpu_inference.layers.jax import JaxModule

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
            self.parallel_config = None

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


def count_model_param_bytes(model: JaxModule) -> int:
    """Count total bytes of all parameters in a model."""
    total = 0
    for _, param in model.named_parameters():
        if hasattr(param.get_value(), 'nbytes'):
            total += param.get_value().nbytes
    return total


class _MemoryPoller:
    """Background thread that polls device bytes_in_use and tracks the max.

    Polling is ~1μs per call on TPU, so even at sub-millisecond intervals
    the overhead is negligible relative to weight loading (seconds).
    """

    def __init__(self, device, poll_interval_s: float = 0.001):
        self._device = device
        self._poll_interval = poll_interval_s
        self._peak: int = 0
        self._num_samples: int = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)

    def start(self, initial_bytes: int):
        self._peak = initial_bytes
        self._num_samples = 0
        self._stop.clear()
        self._thread.start()

    def stop(self) -> tuple[int, int]:
        """Stop polling and return (peak_bytes_observed, num_samples)."""
        self._stop.set()
        self._thread.join(timeout=5.0)
        return self._peak, self._num_samples

    def _poll_loop(self):
        while not self._stop.is_set():
            try:
                stats = self._device.memory_stats()
                if stats and "bytes_in_use" in stats:
                    current = stats["bytes_in_use"]
                    if current > self._peak:
                        self._peak = current
                    self._num_samples += 1
            except Exception:
                pass
            self._stop.wait(timeout=self._poll_interval)


def _get_bytes_in_use(device):
    """Read current bytes_in_use, or None if unavailable."""
    try:
        stats = device.memory_stats()
        return stats.get("bytes_in_use") if stats else None
    except Exception:
        return None


@pytest.fixture
def assert_weight_loading_memory_bounded():
    """Fixture that monitors device memory during weight loading.

    Spawns a background poller to capture peak bytes_in_use (window-scoped,
    not the cumulative high-water mark) and asserts peak delta stays within
    max(model_params * multiplier, min_threshold).

    Usage in tests::

        with assert_weight_loading_memory_bounded(model, "load_weights(...)"):
            loader.load_weights(model, model_config)
    """

    @contextmanager
    def _monitor(model,
                 description="operation",
                 threshold_multiplier=0.3,
                 min_threshold_bytes=2 * GBYTES):
        model_param_bytes = count_model_param_bytes(model)
        # Proportional threshold with absolute floor. PP-partitioned models
        # have fewer params but per-layer transients don't shrink
        # proportionally, so the floor prevents false failures.
        max_peak_delta = max(int(model_param_bytes * threshold_multiplier),
                             min_threshold_bytes)

        device = jax.local_devices()[0] if jax.local_devices() else None
        if device is None:
            logger.warning("No local devices; memory profiling skipped")
            yield
            return

        bytes_before = _get_bytes_in_use(device)
        if bytes_before is None:
            logger.warning("memory_stats() unavailable; profiling skipped")
            yield
            return

        poller = _MemoryPoller(device)
        poller.start(initial_bytes=bytes_before)

        yield

        peak_observed, num_samples = poller.stop()
        bytes_after = _get_bytes_in_use(device) or bytes_before
        peak_observed = max(peak_observed, bytes_after)
        peak_delta = peak_observed - bytes_before

        logger.info(
            "Memory profile [%s]: "
            "before=%.3f GB, after=%.3f GB, peak_observed=%.3f GB, "
            "peak_delta=%.3f GB, samples=%d, threshold=%.3f GB",
            description,
            bytes_before / GBYTES,
            bytes_after / GBYTES,
            peak_observed / GBYTES,
            peak_delta / GBYTES,
            num_samples,
            max_peak_delta / GBYTES,
        )

        assert peak_delta <= max_peak_delta, (
            f"Peak device memory during {description} exceeded threshold: "
            f"peak_delta={peak_delta / GBYTES:.3f} GB > "
            f"max={max_peak_delta / GBYTES:.3f} GB "
            f"(before={bytes_before / GBYTES:.3f} GB, "
            f"after={bytes_after / GBYTES:.3f} GB, "
            f"peak={peak_observed / GBYTES:.3f} GB, "
            f"{num_samples} samples)")

    return _monitor
