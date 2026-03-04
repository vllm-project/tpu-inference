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

import dataclasses
import logging
import re
import threading
from contextlib import contextmanager
from typing import Iterator, Optional
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


def count_model_param_bytes(model) -> int:
    """Count total bytes of all parameters in a Flax nnx model."""
    _, state = nnx.split(model)
    total = 0
    for _, v in state.flat_state():
        if hasattr(v, 'value') and hasattr(v.value, 'nbytes'):
            total += v.value.nbytes
    return total


@dataclasses.dataclass
class MemoryProfile:
    """Device memory profile captured over a code block.

    Tracks both settled memory (before/after) and peak memory observed
    by polling bytes_in_use from a background thread. The polled peak
    is window-scoped (not cumulative), so it accurately captures transient
    spikes during the block — which is what actually causes OOMs.
    """
    bytes_before: int
    bytes_after: int
    peak_bytes_observed: int  # max bytes_in_use seen by polling thread
    num_samples: int  # number of polls taken
    description: str

    @property
    def settled_delta(self) -> int:
        """Net memory change after the block completes."""
        return self.bytes_after - self.bytes_before

    @property
    def peak_delta(self) -> int:
        """Peak memory reached during the block, relative to starting baseline.

        This is the metric that matters for OOM risk: the highest point
        observed relative to where we started. Window-scoped via polling,
        so not affected by peaks from previous tests.
        """
        return self.peak_bytes_observed - self.bytes_before

    def log(self, threshold_bytes: Optional[int] = None):
        """Log the full memory profile. Called on every run, not just failures."""
        threshold_str = (f", threshold={threshold_bytes / GBYTES:.3f} GB"
                         if threshold_bytes is not None else "")
        logger.info(
            "Memory profile [%s]: "
            "before=%.3f GB, after=%.3f GB, peak_observed=%.3f GB, "
            "settled_delta=%.3f GB, peak_delta=%.3f GB, "
            "samples=%d%s",
            self.description,
            self.bytes_before / GBYTES,
            self.bytes_after / GBYTES,
            self.peak_bytes_observed / GBYTES,
            self.settled_delta / GBYTES,
            self.peak_delta / GBYTES,
            self.num_samples,
            threshold_str,
        )

    def assert_peak_bounded(self, max_bytes: int):
        """Assert peak memory delta stayed within threshold."""
        self.log(threshold_bytes=max_bytes)
        assert self.peak_delta <= max_bytes, (
            f"Peak device memory during {self.description} exceeded threshold: "
            f"peak_delta={self.peak_delta / GBYTES:.3f} GB > "
            f"max={max_bytes / GBYTES:.3f} GB. "
            f"before={self.bytes_before / GBYTES:.3f} GB, "
            f"after={self.bytes_after / GBYTES:.3f} GB, "
            f"peak_observed={self.peak_bytes_observed / GBYTES:.3f} GB "
            f"({self.num_samples} samples). "
            f"This may indicate an unexpected memory regression during "
            f"weight loading or processing.")

    def assert_settled_bounded(self, max_bytes: int):
        """Assert settled (net) memory delta stayed within threshold."""
        self.log(threshold_bytes=max_bytes)
        assert self.settled_delta <= max_bytes, (
            f"Settled device memory after {self.description} exceeded threshold: "
            f"settled_delta={self.settled_delta / GBYTES:.3f} GB > "
            f"max={max_bytes / GBYTES:.3f} GB. "
            f"before={self.bytes_before / GBYTES:.3f} GB, "
            f"after={self.bytes_after / GBYTES:.3f} GB.")


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


def _get_bytes_in_use(device) -> Optional[int]:
    """Read current bytes_in_use from a device's memory_stats()."""
    try:
        stats = device.memory_stats()
        if not stats:
            return None
        return stats.get("bytes_in_use")
    except Exception:
        return None


@contextmanager
def device_memory_profile(
    device=None,
    description: str = "operation",
    poll_interval_s: float = 0.001,
) -> Iterator[MemoryProfile]:
    """Profile device memory over a code block.

    Spawns a background thread that polls bytes_in_use at ~1ms intervals
    to capture the true peak within the block (not the cumulative
    high-water mark from process start).

    Yields a MemoryProfile that is populated after the block completes.
    The caller can then inspect the profile or call assert methods on it.

    When memory_stats() is unavailable (CPU-only environments), yields
    a zeroed-out profile and logs a warning.

    Usage::

        with device_memory_profile(description="load_weights") as prof:
            loader.load_weights(model, model_config)
        print(prof.peak_delta)  # inspect without asserting
        prof.assert_peak_bounded(max_bytes)  # or assert
    """
    if device is None:
        local_devs = jax.local_devices()
        if not local_devs:
            logger.warning("No local devices; memory profiling skipped for %s",
                           description)
            yield MemoryProfile(0, 0, 0, 0, description)
            return
        device = local_devs[0]

    bytes_before = _get_bytes_in_use(device)
    if bytes_before is None:
        logger.warning(
            "memory_stats() unavailable on %s; memory profiling skipped for %s",
            device, description)
        yield MemoryProfile(0, 0, 0, 0, description)
        return

    profile = MemoryProfile(
        bytes_before=bytes_before,
        bytes_after=0,
        peak_bytes_observed=bytes_before,
        num_samples=0,
        description=description,
    )

    poller = _MemoryPoller(device, poll_interval_s=poll_interval_s)
    poller.start(initial_bytes=bytes_before)

    yield profile

    peak_observed, num_samples = poller.stop()

    bytes_after = _get_bytes_in_use(device)
    if bytes_after is None:
        logger.warning("memory_stats() became unavailable during %s",
                       description)
        return

    profile.bytes_after = bytes_after
    # Take the max of the poller's peak and the final value, in case
    # the final read catches a higher point than the last poll.
    profile.peak_bytes_observed = max(peak_observed, bytes_after)
    profile.num_samples = num_samples


@contextmanager
def device_memory_profile_bounded(
    max_peak_bytes: int,
    device=None,
    description: str = "operation",
) -> Iterator[MemoryProfile]:
    """Profile device memory and assert peak delta is within threshold.

    Convenience wrapper around device_memory_profile that automatically
    calls assert_peak_bounded on exit.
    """
    with device_memory_profile(device=device, description=description) as prof:
        yield prof
    # Only assert if we got real measurements (non-zero means stats were available)
    if prof.bytes_before > 0 or prof.bytes_after > 0:
        prof.assert_peak_bounded(max_peak_bytes)


@pytest.fixture
def assert_weight_loading_memory_bounded():
    """Fixture that monitors device memory during weight loading.

    Usage in tests::

        with assert_weight_loading_memory_bounded(model, "load_weights(...)"):
            loader.load_weights(model, model_config)

        # With explicit threshold:
        with assert_weight_loading_memory_bounded(
            model, "load_weights(...)", threshold_multiplier=0.5,
        ):
            loader.load_weights(model, model_config)
    """

    def _factory(model,
                 description="operation",
                 threshold_multiplier=0.3,
                 min_threshold_bytes=2 * GBYTES):
        model_param_bytes = count_model_param_bytes(model)
        # Use proportional threshold, but enforce an absolute floor.
        # PP-partitioned models have fewer params but per-layer transients
        # don't shrink proportionally, so the floor prevents false failures.
        max_peak_bytes = max(int(model_param_bytes * threshold_multiplier),
                             min_threshold_bytes)
        return device_memory_profile_bounded(
            max_peak_bytes=max_peak_bytes,
            description=description,
        )

    return _factory
