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
"""
======================================================================
Opt-out of Global JAX Compilation Cache
======================================================================
By default, CI tests utilize a global GCS-backed JAX compilation cache
to accelerate execution. If a test needs to safely bypass this cache,
simply decorate the test function or class with `@pytest.mark.disable_jax_cache`.

Example Usage:

    import pytest

    # Apply to a single test function
    @pytest.mark.disable_jax_cache
    def test_fucntion():
        ...

    # Or apply to an entire test class
    @pytest.mark.disable_jax_cache
    class TestClass:
        ...
======================================================================
"""

import os
from unittest.mock import patch

import jax
import pytest


def pytest_configure(config):
    """
    Register the custom marker to prevent Pytest from throwing 'Unknown marker' warnings.
    """
    config.addinivalue_line(
        "markers",
        "disable_jax_cache: explicitly bypass the global JAX compilation cache for this test."
    )


@pytest.fixture(scope="session", autouse=True)
def _prewarm_jax_compilation_cache():
    """Initialize the JAX compilation cache once at session start.

    On JAX 0.10, JaxTestCase wraps each test in ``assert_global_configs_unchanged``,
    which checks that ``jax._src.compilation_cache._cache`` is the same object
    before and after the test. When ``JAX_COMPILATION_CACHE_DIR`` is set, the
    cache is lazily initialized on the first JIT call — so the first test to
    JIT transitions ``_cache`` from ``None`` to an ``LRUCache`` and the helper
    raises ``AssertionError: Test changed the compilation cache object``.

    Forcing one JIT here ensures ``_cache`` has a stable identity before any
    test body runs, so the assertion holds across the whole session. No-op if
    the env var is unset (cache stays ``None`` everywhere) or if JAX/devices
    are unavailable.
    """
    if not os.environ.get("JAX_COMPILATION_CACHE_DIR"):
        return
    try:
        jax.jit(lambda x: x + jax.numpy.float32(1.0))(jax.numpy.float32(0.0))
    except Exception:
        pass


@pytest.fixture(autouse=True)
def _handle_disable_jax_cache_marker(request):
    """
    A globally auto-used fixture that intercepts tests marked with @pytest.mark.disable_jax_cache.
    It temporarily disables the compilation cache at both the vLLM environment level and the JAX core level.
    """
    marker = request.node.get_closest_marker("disable_jax_cache")

    if marker:
        # Safely read the original JAX config state
        orig_jax_cache_dir = getattr(jax.config, "jax_compilation_cache_dir",
                                     None)

        # Force disable at the JAX level (for unit tests that bypass vLLM's CompilationManager)
        jax.config.update("jax_compilation_cache_dir", None)

        # Force disable at the vLLM level (for end-to-end tests that use CompilationManager)
        # Using patch.dict safely isolates the environment variable modification to the scope of this test
        with patch.dict(os.environ, {"VLLM_DISABLE_COMPILE_CACHE": "1"}):
            yield  # -> Hand over control to run the test

        # Teardown - Restore the original JAX config state for subsequent tests
        jax.config.update("jax_compilation_cache_dir", orig_jax_cache_dir)

    else:
        # If the marker is not present, just run the test normally
        yield
