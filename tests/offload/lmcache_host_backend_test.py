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
"""Tests for LMCacheHostBackend (drop-in for LocalCPUBackend with LMCache spill).

Runs on jax[cpu]; no TPU required. Uses the reference File/Memory KV stores so
no LMCache install is needed to run these.
"""
import tempfile

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tpu_inference.offload.lmcache_host_backend import LMCacheHostBackend
from tpu_inference.offload.lmcache_kv_store import (FileKVStore,
                                                    InMemoryKVStore)

KV_SHAPE = (1, 4, 16, 8, 2, 32)


def _blk(seed, dtype=jnp.bfloat16, shape=KV_SHAPE):
    return jax.random.normal(jax.random.PRNGKey(seed), shape).astype(dtype)


def _eq(a, b):
    return np.asarray(jax.device_get(a)).tobytes() == np.asarray(jax.device_get(b)).tobytes()


def test_parity_no_store():
    """No LMCache tier -> behaves like LocalCPUBackend (in-memory only)."""
    be = LMCacheHostBackend(num_cpu_chunks=8, lmcache_store=None)
    blocks = {i: _blk(i) for i in range(5)}
    for i, b in blocks.items():
        assert be.add(i, b)
    assert be.num_saved_cpu_chunks == 5
    for i, b in blocks.items():
        assert _eq(be.get(i), b)
    assert be.get(99) is None


def test_invalid_chunk_id():
    be = LMCacheHostBackend(num_cpu_chunks=4, lmcache_store=None)
    with pytest.raises(ValueError):
        be.add(4, _blk(0))
    with pytest.raises(ValueError):
        be.add(-1, _blk(0))


def test_disk_spill_bit_exact(tmp_path):
    store = FileKVStore(str(tmp_path))
    be = LMCacheHostBackend(num_cpu_chunks=64, lmcache_store=store, hot_capacity=2)
    blocks = {i: _blk(100 + i) for i in range(6)}
    for i, b in blocks.items():
        be.add(i, b)
    assert len(be.cache) <= 2
    for i, b in blocks.items():
        got = be.get(i)
        assert got is not None and _eq(got, b)


def test_list_of_layers_spill():
    store = InMemoryKVStore()
    be = LMCacheHostBackend(num_cpu_chunks=64, lmcache_store=store, hot_capacity=1)
    layers_by_chunk = {
        i: [_blk(200 + i * 10 + L, shape=(1, 1, 16, 8, 2, 32)) for L in range(4)]
        for i in range(3)
    }
    for i, layers in layers_by_chunk.items():
        be.add(i, layers)
    for i, layers in layers_by_chunk.items():
        got = be.get(i)
        assert got is not None and len(got) == 4
        for L in range(4):
            assert _eq(got[L], layers[L])


def test_reclaim_unoccupied():
    store = InMemoryKVStore()
    be = LMCacheHostBackend(num_cpu_chunks=8, lmcache_store=store)
    for i in range(6):
        be.add(i, _blk(300 + i))
    be.reclaim_unoccupied_chunks([0, 1, 2])
    for i in range(6):
        got = be.get(i)
        if i in (0, 1, 2):
            assert got is not None
        else:
            assert got is None


def test_content_hash_hint(tmp_path):
    store = FileKVStore(str(tmp_path))
    be = LMCacheHostBackend(num_cpu_chunks=64, lmcache_store=store, hot_capacity=1)
    b = _blk(400)
    be.set_chunk_hash_hint(0, "deadbeefhash")
    be.add(0, b)
    be.add(1, _blk(401))  # forces hot eviction of chunk 0
    got = be.get(0)
    assert got is not None and _eq(got, b)
    assert store.contains("h:deadbeefhash")
