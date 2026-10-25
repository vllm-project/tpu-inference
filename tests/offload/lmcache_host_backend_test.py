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


# ---------------------------------------------------------------------------
# Session isolation and generation-aware persistence.
# ---------------------------------------------------------------------------


class _FlakyStore(InMemoryKVStore):
    """In-memory store whose ``put`` can be made to fail after N more puts."""

    def __init__(self) -> None:
        super().__init__()
        self.puts_before_failure = None  # None -> never fail

    def put(self, key: str, data: bytes) -> None:
        if self.puts_before_failure is not None:
            if self.puts_before_failure <= 0:
                raise OSError("injected put failure")
            self.puts_before_failure -= 1
        super().put(key, data)


class _WriteOnceStore(InMemoryKVStore):
    """Store that keeps the first value written under a key (dedup-on-put),
    as content-addressed caches commonly do."""

    def put(self, key: str, data: bytes) -> None:
        if not self.contains(key):
            super().put(key, data)


def _num_entries(store: InMemoryKVStore) -> int:
    return len(store._d)


def _factory_env(monkeypatch, root, hot_chunks=1):
    monkeypatch.setenv("TPU_OFFLOAD_LMCACHE", "1")
    monkeypatch.setenv("TPU_OFFLOAD_LMCACHE_BACKEND", "file")
    monkeypatch.setenv("TPU_OFFLOAD_LMCACHE_PATH", str(root))
    monkeypatch.setenv("TPU_OFFLOAD_LMCACHE_HOT_CHUNKS", str(hot_chunks))


def test_two_backends_same_root_do_not_share_slot_keys(monkeypatch, tmp_path):
    """Two engines built from the same env (same file root) must not read each
    other's spilled blocks for the same chunk_id."""
    from tpu_inference.offload.host_backend_factory import build_host_backend
    _factory_env(monkeypatch, tmp_path)
    be_a = build_host_backend(num_cpu_chunks=8)
    be_b = build_host_backend(num_cpu_chunks=8)
    a0, a1, b0, b1 = _blk(500), _blk(501), _blk(502), _blk(503)
    be_a.add(0, a0)
    be_b.add(0, b0)
    be_a.add(1, a1)  # hot capacity 1 -> chunk 0 of A spills
    be_b.add(1, b1)  # hot capacity 1 -> chunk 0 of B spills
    assert 0 not in be_a.cache and 0 not in be_b.cache
    got_a, got_b = be_a.get(0), be_b.get(0)
    assert got_a is not None and _eq(got_a, a0)
    assert got_b is not None and _eq(got_b, b0)


def test_reclaim_in_one_backend_keeps_other_backends_spill(
        monkeypatch, tmp_path):
    from tpu_inference.offload.host_backend_factory import build_host_backend
    _factory_env(monkeypatch, tmp_path)
    be_a = build_host_backend(num_cpu_chunks=8)
    be_b = build_host_backend(num_cpu_chunks=8)
    b0 = _blk(511)
    be_a.add(0, _blk(510))
    be_b.add(0, b0)
    be_a.add(1, _blk(512))
    be_b.add(1, _blk(513))
    be_a.reclaim_unoccupied_chunks([])  # A releases everything it owns
    got_b = be_b.get(0)
    assert got_b is not None and _eq(got_b, b0)


def test_hot_tier_eviction_is_lru_and_bounded():
    store = InMemoryKVStore()
    be = LMCacheHostBackend(num_cpu_chunks=8,
                            lmcache_store=store,
                            hot_capacity=2)
    blocks = {i: _blk(515 + i) for i in range(3)}
    for i, b in blocks.items():
        be.add(i, b)
    assert list(be.cache) == [1, 2]
    assert be.num_saved_cpu_chunks == 2
    got = be.get(0)  # reload + promote evicts the LRU chunk (1)
    assert got is not None and _eq(got, blocks[0])
    assert list(be.cache) == [2, 0]
    assert be.num_saved_cpu_chunks == 2


def test_unpersisted_chunks_stay_resident_up_to_chunk_id_space():
    """If the spill tier rejects writes, chunks are kept hot instead of being
    dropped; the hot tier never grows beyond the chunk_id space."""
    store = _FlakyStore()
    store.puts_before_failure = 0
    be = LMCacheHostBackend(num_cpu_chunks=4,
                            lmcache_store=store,
                            hot_capacity=1)
    blocks = {i: _blk(526 + i) for i in range(4)}
    for i, b in blocks.items():
        be.add(i, b)
    for i, b in blocks.items():
        got = be.get(i)
        assert got is not None and _eq(got, b)
    assert len(be.cache) == 4 and be.num_saved_cpu_chunks == 4
    assert _num_entries(store) == 0


def test_failed_overwrite_never_returns_stale_generation():
    store = _FlakyStore()
    be = LMCacheHostBackend(num_cpu_chunks=8,
                            lmcache_store=store,
                            hot_capacity=1)
    v1, v2, other = _blk(520), _blk(521), _blk(522)
    be.add(0, v1)  # generation 1 persisted
    store.puts_before_failure = 0
    be.add(0, v2)  # overwrite: persist fails, must not raise
    be.add(1, other)  # hot pressure on chunk 0
    got = be.get(0)
    assert got is not None, "current generation was lost"
    assert not _eq(got, v1), "stale generation returned as current data"
    assert _eq(got, v2)
    assert _eq(be.get(1), other)
    # Once the store recovers, new data is persisted and served normally.
    store.puts_before_failure = None
    v3 = _blk(523)
    be.add(0, v3)
    be.add(2, _blk(524))
    be.add(3, _blk(525))
    assert 0 not in be.cache
    got = be.get(0)
    assert got is not None and _eq(got, v3)


def test_failed_overwrite_of_layer_list_is_not_torn():
    store = _FlakyStore()
    be = LMCacheHostBackend(num_cpu_chunks=8,
                            lmcache_store=store,
                            hot_capacity=1)
    shape = (1, 1, 16, 8, 2, 32)
    old = [_blk(530 + L, shape=shape) for L in range(4)]
    new = [_blk(540 + L, shape=shape) for L in range(4)]
    be.add(0, old)
    store.puts_before_failure = 2  # layers 0-1 written, layer 2 fails
    be.add(0, new)
    store.puts_before_failure = None
    be.add(1, _blk(550))
    got = be.get(0)
    assert got is not None and len(got) == 4
    for L in range(4):
        assert _eq(got[L], new[L]), f"layer {L} is not from the current add"


def test_write_once_store_does_not_serve_previous_generation():
    """Reusing one persistent key per slot breaks with dedup-on-put stores:
    the overwrite is dropped and the old generation is served."""
    store = _WriteOnceStore()
    be = LMCacheHostBackend(num_cpu_chunks=8,
                            lmcache_store=store,
                            hot_capacity=1)
    v1, v2 = _blk(560), _blk(561)
    be.add(0, v1)
    be.add(0, v2)
    be.add(1, _blk(562))
    got = be.get(0)
    assert got is None or not _eq(got, v1), "stale generation returned"
    assert got is not None and _eq(got, v2)


def test_reload_after_eviction_and_overwrite_returns_latest():
    store = InMemoryKVStore()
    be = LMCacheHostBackend(num_cpu_chunks=8,
                            lmcache_store=store,
                            hot_capacity=1)
    v1, v2 = _blk(570), _blk(571)
    be.add(0, v1)
    be.add(1, _blk(572))  # evict 0
    got = be.get(0)  # reload + promote, evicts 1
    assert got is not None and _eq(got, v1)
    be.add(0, v2)  # overwrite of a reloaded chunk
    be.add(1, _blk(573))  # evict 0 again
    got = be.get(0)
    assert got is not None and _eq(got, v2)
    # Superseded generations are not left behind in the store.
    assert _num_entries(store) == 2


def test_restart_ignores_previous_session_files(tmp_path):
    """A new backend on a root that still holds another session's files (e.g.
    after a crash) never serves those files."""
    old_root, root = tmp_path / "old", tmp_path / "root"
    be_old = LMCacheHostBackend(num_cpu_chunks=8,
                                lmcache_store=FileKVStore(str(old_root)),
                                hot_capacity=1)
    for i in range(4):
        be_old.add(i, _blk(580 + i))
    import shutil
    shutil.copytree(old_root, root)  # leftover files from a dead session
    leftovers = sorted(p.name for p in root.iterdir())
    assert leftovers

    be_new = LMCacheHostBackend(num_cpu_chunks=8,
                                lmcache_store=FileKVStore(str(root)),
                                hot_capacity=1)
    for i in range(4):
        assert be_new.get(i) is None
    fresh = {i: _blk(590 + i) for i in range(4)}
    for i, b in fresh.items():
        be_new.add(i, b)
    for i, b in fresh.items():
        got = be_new.get(i)
        assert got is not None and _eq(got, b)
    # Cleaning up the new session leaves the other session's files alone.
    be_new.close()
    assert sorted(p.name for p in root.iterdir()) == leftovers


def test_close_removes_only_own_entries(tmp_path):
    be_a = LMCacheHostBackend(num_cpu_chunks=8,
                              lmcache_store=FileKVStore(str(tmp_path)),
                              hot_capacity=1)
    be_b = LMCacheHostBackend(num_cpu_chunks=8,
                              lmcache_store=FileKVStore(str(tmp_path)),
                              hot_capacity=1)
    b0 = _blk(600)
    be_a.add(0, _blk(601))
    be_a.add(1, _blk(602))
    be_b.add(0, b0)
    be_b.add(1, _blk(603))
    files_b = {p.name for p in tmp_path.iterdir()}
    be_a.close()
    files_after = {p.name for p in tmp_path.iterdir()}
    assert files_after and files_after < files_b
    assert be_a.get(0) is None and be_a.num_saved_cpu_chunks == 0
    got = be_b.get(0)
    assert got is not None and _eq(got, b0)
    be_b.close()
    assert not any(tmp_path.iterdir())


def test_garbage_collected_backend_cleans_up_its_files(tmp_path):
    import gc
    be = LMCacheHostBackend(num_cpu_chunks=8,
                            lmcache_store=FileKVStore(str(tmp_path)),
                            hot_capacity=1)
    for i in range(3):
        be.add(i, _blk(610 + i))
    assert any(tmp_path.iterdir())
    del be
    gc.collect()
    assert not any(tmp_path.iterdir())


def test_file_store_key_mapping_is_injective(tmp_path):
    store = FileKVStore(str(tmp_path))
    store.put("h:a_b", b"first")
    store.put("h:a:b", b"second")
    store.put("h:a/b", b"third")
    assert store.get("h:a_b") == b"first"
    assert store.get("h:a:b") == b"second"
    assert store.get("h:a/b") == b"third"


def test_flags_off_factory_returns_stock_backend(monkeypatch):
    from tpu_inference.offload.cpu_backend import LocalCPUBackend
    from tpu_inference.offload.host_backend_factory import build_host_backend
    for name in ("TPU_OFFLOAD_LMCACHE", "TPU_OFFLOAD_LMCACHE_BACKEND",
                 "TPU_OFFLOAD_LMCACHE_PATH", "TPU_OFFLOAD_LMCACHE_HOT_CHUNKS"):
        monkeypatch.delenv(name, raising=False)
    be = build_host_backend(num_cpu_chunks=8)
    assert type(be) is LocalCPUBackend
    assert be.max_num_cpu_chunks == 8
