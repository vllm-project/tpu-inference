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
"""LMCacheHostBackend: a drop-in replacement for ``LocalCPUBackend`` that adds
LMCache-backed storage tiers (disk spill today; remote/P2P/cross-instance next)
BEHIND the native ``TPUOffloadConnector``.

Rationale
---------
``TPUOffloadConnector`` already owns the TPU-specific hard part: moving KV blocks
between HBM (JAX device arrays) and host memory (host-sharded JAX arrays), the
scatter/gather staging, and the scheduler-side content hashing. Its host store is
``LocalCPUBackend`` — an in-memory ``OrderedDict`` capped at ``num_cpu_chunks``.

This class keeps that exact interface (``add`` / ``get`` /
``reclaim_unoccupied_chunks`` / ``num_saved_cpu_chunks``) so it is a one-line swap
in the connector worker, but backs it with:
  * a bounded hot CPU tier (identical semantics to LocalCPUBackend), and
  * an LMCache storage tier (local_disk today) that persists evicted blocks and
    serves them back on miss — so KV survives beyond RAM and (with a content-hash
    key, Depth 2) can be shared across TPU replicas / restarts.

The value bridge (``kv_bridge``) serializes the host JAX array to raw bytes
bit-exactly (incl. bfloat16) and reconstructs it on load. No CUDA. Runs on
jax[cpu] for local development and on real TPU unchanged.

Design notes
------------
* Depth-1 keying is by the connector's ephemeral ``chunk_id`` (an int slot). The
  connector guarantees a given content maps to a stable chunk_id for its lifetime
  via the scheduler ``LRUCacheManager``; within a process this is a correct cache.
* Depth-2 (cross-instance) requires threading the ``BlockHash`` to the worker so
  the LMCache key is content-addressed. ``set_chunk_hash_hint`` is the forward-
  compatible hook for that; when unset we fall back to chunk_id keying.
  Content-addressed entries are shared by design and are never removed by
  ``reclaim_unoccupied_chunks`` or ``close()``.
* A chunk_id is only meaningful inside one backend instance, and the same slot
  is reused for different content over time. Slot-keyed entries are therefore
  written under ``<session namespace>/c:<chunk_id>/g:<generation>``: the
  namespace is unique per backend instance (engines sharing a store root or a
  remote store never see each other's slots, and a restarted engine never sees
  a previous session's slots), and the generation changes on every ``add`` (a
  failed, dropped or torn write can never make an older generation readable as
  the current one).
* Reads only go through the per-chunk record of the last successfully
  persisted generation. ``add`` invalidates the previous record before writing.
  If persisting fails, the chunk stays resident in the hot tier: a chunk is
  never evicted without a persisted copy. The hot tier can then exceed
  ``hot_capacity``, but never the connector's chunk_id space, i.e. never more
  than ``LocalCPUBackend`` would hold.
* ``close()`` (also run when the backend is garbage collected or at interpreter
  exit) removes the slot-keyed entries owned by this instance.
* All LMCache tiers are optional. With ``lmcache_backend=None`` this class behaves
  exactly like ``LocalCPUBackend`` (pure in-memory), which makes it safe to enable
  by default and lets tests run without any LMCache install.
"""
from __future__ import annotations

import dataclasses
import itertools
import sys
import threading
import uuid
import weakref
from collections import OrderedDict
from typing import TYPE_CHECKING, Any, Optional, Union

from tpu_inference.logger import init_logger
from tpu_inference.offload.kv_bridge import (KVBlockSpec, bytes_to_jax_block,
                                             jax_block_to_bytes)

if TYPE_CHECKING:
    from tpu_inference.offload.lmcache_kv_store import LMCacheKVStore

logger = init_logger(__name__)

CpuChunkId = int


@dataclasses.dataclass(frozen=True)
class _PersistRecord:
    """Where the current generation of one chunk was persisted, and how."""
    keys: tuple[str, ...]
    spec: Union[KVBlockSpec, list[KVBlockSpec]]
    generation: int
    # True for slot-keyed entries, which are private to one backend instance.
    # False for content-addressed entries, which may be shared.
    owned: bool


def _remove_keys(store: Any, keys) -> None:
    for key in keys:
        try:
            store.remove(key)
        except Exception:  # pragma: no cover - defensive
            pass


def _remove_owned_entries(store: Any, records: dict) -> None:
    """Best-effort removal of every slot-keyed entry listed in ``records``."""
    for record in list(records.values()):
        if record.owned:
            _remove_keys(store, record.keys)
    records.clear()


def _finalize_backend(store: Any, records: dict, lock: threading.Lock) -> None:
    # Bounded wait so that interpreter shutdown never hangs on a busy lock.
    acquired = lock.acquire(timeout=5)
    try:
        _remove_owned_entries(store, records)
    finally:
        if acquired:
            lock.release()


class LMCacheHostBackend:
    """Drop-in for LocalCPUBackend with an LMCache-backed spill tier.

    Interface parity with ``tpu_inference.offload.cpu_backend.LocalCPUBackend``:
        add(chunk_id, value) -> bool
        get(chunk_id) -> Optional[value]
        reclaim_unoccupied_chunks(occupied_chunk_ids)
        property num_saved_cpu_chunks
    """

    def __init__(
        self,
        num_cpu_chunks: int,
        lmcache_store: Optional["LMCacheKVStore"] = None,
        hot_capacity: Optional[int] = None,
    ):
        # ``num_cpu_chunks`` is the connector's chunk_id space (upper bound on
        # valid chunk_ids), matching LocalCPUBackend. ``hot_capacity`` is how
        # many blocks we keep resident in host RAM; overflow spills to the
        # LMCache tier. When no LMCache tier is configured they are equal, so
        # behavior is identical to LocalCPUBackend.
        self.max_num_cpu_chunks = num_cpu_chunks
        if hot_capacity is None:
            hot_capacity = num_cpu_chunks
        self.hot_capacity = hot_capacity
        # Hot tier: identical to LocalCPUBackend (LRU OrderedDict of jax arrays).
        self.cache: "OrderedDict[CpuChunkId, Any]" = OrderedDict()
        self.current_size_bytes = 0
        self._num_saved_cpu_chunks = 0
        # Optional persistent tier (disk/remote) via LMCache.
        self._store = lmcache_store
        # chunk_id -> optional content hash hint (Depth 2). Falls back to
        # chunk_id keying when a hint is not provided.
        self._hash_hints: dict[CpuChunkId, str] = {}
        # chunk_id -> record of the last successfully persisted generation.
        # A chunk without a record has no readable persistent copy.
        self._records: dict[CpuChunkId, _PersistRecord] = {}
        self._generations = itertools.count(1)
        # Unique per instance: slot keys of different engines (or of an
        # earlier run of this engine) sharing one store can never collide.
        self.session_id = uuid.uuid4().hex
        self._namespace = f"tpu-offload/{self.session_id}"
        self._lock = threading.Lock()
        # Drop this instance's slot-keyed entries when it is garbage collected
        # or at interpreter exit. The finalizer must not reference ``self``.
        self._finalizer = None
        if self._store is not None:
            self._finalizer = weakref.finalize(self, _finalize_backend,
                                               self._store, self._records,
                                               self._lock)
        logger.info(
            "LMCacheHostBackend initialized. chunk_id space=%d, hot capacity=%d "
            "chunks, lmcache_store=%s, session=%s", self.max_num_cpu_chunks,
            self.hot_capacity,
            type(self._store).__name__ if self._store else None,
            self.session_id)

    # ---- interface parity helpers -------------------------------------------------
    @property
    def num_saved_cpu_chunks(self) -> int:
        return self._num_saved_cpu_chunks

    def _get_value_size(self, value: Any) -> int:
        if isinstance(value, list):
            return sum(v.nbytes for v in value if hasattr(v, "nbytes"))
        if hasattr(value, "nbytes"):
            return value.nbytes
        return sys.getsizeof(value)

    def _store_key(self, chunk_id: CpuChunkId, generation: int) -> str:
        """Content-hash key if a hint is set (Depth 2), else a slot key that is
        private to this instance and to this generation of the chunk."""
        h = self._hash_hints.get(chunk_id)
        if h is not None:
            return f"h:{h}"
        return f"{self._namespace}/c:{chunk_id}/g:{generation}"

    def set_chunk_hash_hint(self, chunk_id: CpuChunkId,
                            chunk_hash: str) -> None:
        """Depth-2 hook: associate a content hash with a chunk_id so the LMCache
        key is content-addressed (enables cross-instance / persistent reuse)."""
        self._hash_hints[chunk_id] = chunk_hash

    # ---- store ---------------------------------------------------------------------
    def add(self, chunk_id: CpuChunkId, value: Any) -> bool:
        if chunk_id < 0 or chunk_id >= self.max_num_cpu_chunks:
            raise ValueError(f"get invalid chunk_id: {chunk_id}")
        with self._lock:
            if chunk_id in self.cache:
                old = self.cache.pop(chunk_id)
                self.current_size_bytes -= self._get_value_size(old)
                del old
                self._num_saved_cpu_chunks -= 1

            self.cache[chunk_id] = value
            self._num_saved_cpu_chunks += 1
            self.current_size_bytes += self._get_value_size(value)

            # Persist to LMCache tier (best-effort, never blocks correctness of
            # the hot tier). Single jax arrays are bridged directly; lists are
            # stored per-layer under an index suffix.
            if self._store is not None:
                # The previous generation is stale from here on, whether or
                # not the new one can be persisted.
                old_record = self._records.pop(chunk_id, None)
                generation = next(self._generations)
                try:
                    self._records[chunk_id] = self._persist(
                        chunk_id, value, generation)
                except Exception as e:
                    logger.warning(
                        "LMCache persist failed for chunk %s: %s. Keeping it "
                        "resident in the hot tier.", chunk_id, e)
                if old_record is not None and old_record.owned:
                    _remove_keys(self._store, old_record.keys)

            # Evict hot-tier LRU beyond capacity; evicted blocks remain in the
            # LMCache tier and are transparently reloaded on get().
            self._evict_hot_if_needed()
        return True

    def _persist(self, chunk_id: CpuChunkId, value: Any,
                 generation: int) -> _PersistRecord:
        key = self._store_key(chunk_id, generation)
        owned = chunk_id not in self._hash_hints
        written: list[str] = []
        try:
            if isinstance(value, list):
                spec = []
                for i, layer in enumerate(value):
                    raw, layer_spec = jax_block_to_bytes(layer)
                    self._store.put(f"{key}#{i}", raw)
                    written.append(f"{key}#{i}")
                    spec.append(layer_spec)
            else:
                raw, spec = jax_block_to_bytes(value)
                self._store.put(key, raw)
                written.append(key)
        except Exception:
            # Do not leave a partially written generation behind.
            if owned:
                _remove_keys(self._store, written)
            raise
        return _PersistRecord(keys=tuple(written),
                              spec=spec,
                              generation=generation,
                              owned=owned)

    def _evict_hot_if_needed(self) -> None:
        # Keep the hot tier bounded to hot_capacity. Evicted blocks remain in
        # the LMCache tier and are transparently reloaded on get(). With a
        # persistent tier, only chunks whose current generation was persisted
        # are evictable.
        while len(self.cache) > self.hot_capacity:
            # Least recently used evictable chunk; normally the first one.
            evict_id = next((c for c in self.cache
                             if self._store is None or c in self._records),
                            None)
            if evict_id is None:
                return
            evict_val = self.cache.pop(evict_id)
            self.current_size_bytes -= self._get_value_size(evict_val)
            self._num_saved_cpu_chunks -= 1
            if self._store is None:
                # No persistent tier -> this is a real drop (matches capacity).
                logger.debug("Hot-tier drop chunk %s (no LMCache tier)",
                             evict_id)
            del evict_val

    # ---- load ----------------------------------------------------------------------
    def get(self, chunk_id: CpuChunkId) -> Optional[Any]:
        with self._lock:
            if chunk_id in self.cache:
                self.cache.move_to_end(chunk_id)
                return self.cache[chunk_id]
            # Hot miss: try the LMCache tier.
            if self._store is not None and chunk_id in self._records:
                try:
                    val = self._reload(self._records[chunk_id])
                except Exception as e:  # pragma: no cover - defensive
                    logger.warning("LMCache reload failed for chunk %s: %s",
                                   chunk_id, e)
                    return None
                if val is not None:
                    # Promote back into the hot tier.
                    self.cache[chunk_id] = val
                    self._num_saved_cpu_chunks += 1
                    self.current_size_bytes += self._get_value_size(val)
                    self._evict_hot_if_needed()
                    return val
            return None

    def _reload(self, record: _PersistRecord) -> Optional[Any]:
        if isinstance(record.spec, list):
            layers = []
            for key, s in zip(record.keys, record.spec):
                raw = self._store.get(key)
                if raw is None:
                    return None
                layers.append(bytes_to_jax_block(raw, s))
            return layers
        raw = self._store.get(record.keys[0])
        if raw is None:
            return None
        return bytes_to_jax_block(raw, record.spec)

    # ---- reclaim -------------------------------------------------------------------
    def reclaim_unoccupied_chunks(self, occupied_chunk_ids: list) -> None:
        with self._lock:
            occupied = set(occupied_chunk_ids)
            for chunk_id in [c for c in self.cache if c not in occupied]:
                val = self.cache.pop(chunk_id)
                self.current_size_bytes -= self._get_value_size(val)
                self._num_saved_cpu_chunks -= 1
                del val
            # Also drop persistent entries for unoccupied chunks (chunk_id keying;
            # under content-hash keying we keep them for cross-request reuse).
            if self._store is not None:
                for chunk_id in [
                        c for c in self._records if c not in occupied
                ]:
                    if self._records[chunk_id].owned:
                        record = self._records.pop(chunk_id)
                        _remove_keys(self._store, record.keys)

    # ---- cleanup -------------------------------------------------------------------
    def close(self) -> None:
        """Drop the hot tier and remove this instance's slot-keyed entries from
        the persistent tier. Content-addressed entries are left in place.
        Entries written by other instances are never touched."""
        with self._lock:
            self.cache.clear()
            self.current_size_bytes = 0
            self._num_saved_cpu_chunks = 0
            if self._store is not None:
                _remove_owned_entries(self._store, self._records)
