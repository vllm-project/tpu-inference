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
"""`_run_compilation` must AOT-lower anything that exposes `lower`.

The flax_nnx model path hands the runner `wrapped_model_fn`, a plain closure
over the jitted `run_model` that filters kwargs and forwards a `lower`
attribute. Before that attribute existed, `hasattr(fn, 'lower')` was False
and the backbone silently took the "AOT lower skipped (not a jit)" path --
compiled only inside warmup, reported no `memory_analysis`, and left OOMs
during warmup unattributable to a bucket. These tests pin the contract from
both sides: a `lower`-bearing wrapper gets its `lower` called and compiled,
and a bare closure still falls back to warmup-only (the control that keeps
the first assertion honest).
"""
import os
from unittest.mock import MagicMock, patch

import jax
import jax.numpy as jnp
import numpy as np
import pytest

os.environ.setdefault("NUM_PRECOMPILE_WORKERS", "1")

import tpu_inference.runner.compilation_manager as cm


@pytest.fixture
def manager():
    mesh = jax.sharding.Mesh(np.array(jax.devices()[:1]), ("x", ))
    runner = MagicMock()
    runner.mesh = mesh
    mgr = cm.CompilationManager(runner)
    # Force the synchronous path regardless of NUM_PRECOMPILE_WORKERS.
    mgr._compile_executor = None
    return mgr


def _wrapped_jit_pair():
    """A (wrapper, lower_calls) pair shaped like model_loader's
    wrapped_model_fn: a kwargs-filtering closure with a forwarded `lower`."""
    jitted = jax.jit(lambda a: a * 2 + 1)
    lower_calls = []

    def wrapper(*args, **kwargs):
        kwargs.pop("shared_attention_metadata", None)
        return jitted(*args, **kwargs)

    def _lower(*args, **kwargs):
        kwargs.pop("shared_attention_metadata", None)
        lower_calls.append(args)
        return jitted.lower(*args, **kwargs)

    wrapper.lower = _lower
    return wrapper, lower_calls


def test_lower_bearing_wrapper_is_aot_compiled(manager):
    wrapper, lower_calls = _wrapped_jit_pair()
    with patch.object(cm, "logger") as mock_logger:
        manager._run_compilation("wrapped_backbone", wrapper,
                                 jnp.ones((8, ), dtype=jnp.float32))
    assert len(lower_calls) == 1, "the wrapper's lower() was never invoked"
    skip_logs = [
        c for c in mock_logger.info.call_args_list
        if "AOT lower skipped" in str(c.args[0])
    ]
    assert not skip_logs, f"backbone took the skip path: {skip_logs}"
    # The warmup task must still be queued: AOT compilation primes the cache,
    # the warmup call populates the jit's own dispatch path.
    assert len(manager._warmup_tasks) == 1


def test_bare_closure_still_skips_aot(manager):
    jitted = jax.jit(lambda a: a * 2 + 1)

    def bare(*args, **kwargs):
        return jitted(*args, **kwargs)

    with patch.object(cm, "logger") as mock_logger:
        manager._run_compilation("bare_backbone", bare,
                                 jnp.ones((8, ), dtype=jnp.float32))
    skip_logs = [
        c for c in mock_logger.info.call_args_list
        if "AOT lower skipped" in str(c.args[0])
    ]
    assert len(skip_logs) == 1
    assert len(manager._warmup_tasks) == 1
