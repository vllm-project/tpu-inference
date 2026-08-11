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
"""Mamba/KDA state pools must be allocated zero-initialized.

As of jax 0.11.0, ``jnp.empty`` returns genuinely uninitialized memory on
platforms that support it (before 0.11.0 it was an alias for
``jnp.zeros``). A recurrent-state pool allocated uninitialized holds
recycled allocator bytes — including NaN patterns — and any step where
``has_initial_state=1`` meets pool contents the request did not write
consumes them: the output becomes argmax-over-NaN (token id 0) for every
remaining step of that request. The ``has_initial_state=0`` masking only
protects the first touch of a slot, so zero-init at allocation is the only
initialization the rest of the stack can rely on.

These tests pin the allocation contract of
``kv_cache_manager.allocate_mamba_pool`` directly, on 1 device (CPU or a
single TPU chip), independent of the model or pool sizing.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh, NamedSharding, PartitionSpec

from tpu_inference.runner.kv_cache_manager import allocate_mamba_pool


@pytest.fixture
def sharding():
    mesh = Mesh(np.array(jax.devices()[:1]), ("data", ))
    return NamedSharding(mesh, PartitionSpec())


@pytest.mark.parametrize(
    "shape,dtype",
    [
        ((65, 3, 384), jnp.bfloat16),  # conv window (KDA q/k/v)
        ((65, 3, 128, 128), jnp.float32),  # recurrent state
    ],
)
def test_pool_is_zero_initialized(shape, dtype, sharding, monkeypatch):
    monkeypatch.delenv("MAMBA_POOL_NAN_CANARY", raising=False)
    pool = np.asarray(allocate_mamba_pool(shape, dtype, sharding))
    assert pool.shape == shape
    # all-zero implies all-finite; assert both so a failure names the mode.
    assert np.isfinite(pool.astype(np.float32)).all(), (
        "freshly allocated mamba pool contains non-finite values — the "
        "allocation is uninitialized (jnp.empty under jax>=0.11.0?)")
    assert (pool.astype(np.float32) == 0).all(), (
        "freshly allocated mamba pool is not zero-initialized")


def test_canary_env_fills_nan(sharding, monkeypatch):
    monkeypatch.setenv("MAMBA_POOL_NAN_CANARY", "1")
    pool = np.asarray(allocate_mamba_pool((8, 3, 16), jnp.float32, sharding))
    assert np.isnan(pool).all()
