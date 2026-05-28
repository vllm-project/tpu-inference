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
def _cleanup_tpu_zombies():
    """
    Automated self-healing fixture to clear lingering JAX/libtpu process locks.

    Orphaned 'VLLM::EngineCore' child workers or lingering JAX lockfiles from prior
    crashed/aborted test runs can block JAX backend initialization. This fixture
    reaps them both during session setup and session teardown.
    """
    import subprocess
    def reap():
        try:
            # Kill orphaned EngineCore child workers spawned under our user
            subprocess.run(["pkill", "-9", "-f", "VLLM::EngineCore"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            # Kill orphaned vLLM engine subprocesses
            subprocess.run(["pkill", "-9", "-f", "vllm"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            # Clear JAX libtpu shared memory lockfiles
            subprocess.run("rm -f /tmp/libtpu*", shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception:
            pass

    reap()
    yield
    reap()


@pytest.fixture(scope="session", autouse=True)
def _prewarm_jax_compilation_cache():
    """
    Pre-warms the JAX compilation cache singleton before any tests run.

    JaxTestCase wraps every test in `assert_global_configs_unchanged`, which
    asserts that `jax._src.compilation_cache._cache` is the same object before
    and after the test. With `JAX_COMPILATION_CACHE_DIR` set, the cache is
    lazily initialized on the first JIT call, so the first test to JIT would
    transition `_cache` from None to LRUCache and trip the assertion. Force
    that initialization at session setup so `_cache` has a stable identity
    for every test that follows. Tests that need to opt out can still use
    the `@pytest.mark.disable_jax_cache` marker.

    Use `compilation_cache._initialize_cache()` (JAX's own lazy-init helper)
    instead of a dummy `jax.jit(...)` because a JIT would also initialize the
    backend (libtpu on TPU) in the pytest parent process, which conflicts with
    tests that fork worker processes (e.g. Ray's RayDistributedExecutor, vLLM
    MultiProcExecutor) and produces `Internal error when accessing libtpu
    multi-process lockfile`. `_initialize_cache` only constructs the LRUCache
    object from the configured cache dir — no backend touched.
    """
    if jax.config.jax_compilation_cache_dir:
        from jax._src import compilation_cache
        compilation_cache._initialize_cache()


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
