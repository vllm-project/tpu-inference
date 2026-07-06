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

os.environ["LIBTPU_INIT_ARGS"] = (os.environ.get("LIBTPU_INIT_ARGS", "") +
                                  " --xla_tpu_scoped_vmem_limit_kib=65536")

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
    config.addinivalue_line(
        "markers",
        "bvt: per-push smoke case. Selected when BVT_ONLY=1; otherwise all tests run."
    )


def partition_bvt_only(items):
    """Split collected items into (keep, drop) for a BVT_ONLY run.

    For any module that declares at least one ``@pytest.mark.bvt`` item, keep
    only the bvt-marked items from that module; modules that do not use the
    marker are kept in full. So enabling BVT_ONLY only ever narrows files that
    opted in -- it can never empty an unrelated suite. Pure helper so the logic
    can be unit-tested directly.
    """
    bvt_modules = {
        item.nodeid.split("::")[0]
        for item in items if item.get_closest_marker("bvt")
    }
    if not bvt_modules:
        return list(items), []
    keep, drop = [], []
    for item in items:
        in_bvt_module = item.nodeid.split("::")[0] in bvt_modules
        if in_bvt_module and not item.get_closest_marker("bvt"):
            drop.append(item)
        else:
            keep.append(item)
    return keep, drop


def pytest_collection_modifyitems(config, items):
    """Gate the per-push BVT subset.

    Only narrows collection when BVT_ONLY=1 (set by the "- bvt" pipeline step).
    By default (BVT_ONLY unset) every test runs, so nightly, feature
    support-matrix, and release-tag builds -- none of which set BVT_ONLY -- run
    the full set with no extra configuration.
    """
    if os.environ.get("BVT_ONLY") != "1":
        return
    keep, drop = partition_bvt_only(items)
    if drop:
        config.hook.pytest_deselected(items=drop)
        items[:] = keep


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
