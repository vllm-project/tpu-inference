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
* All LMCache tiers are optional. With ``lmcache_backend=None`` this class behaves
  exactly like ``LocalCPUBackend`` (pure in-memory), which makes it safe to enable
  by default and lets tests run without any LMCache install.
"""
from __future__ import annotations

import sys
import threading
from collections import OrderedDict
from typing import Any, Optional

from tpu_inference.logger import init_logger
from tpu_inference.offload.kv_bridge import (KVBlockSpec, bytes_to_jax_block,
                                             jax_block_to_bytes)

logger = init_logger(__name__)

CpuChunkId = int


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
        # chunk_id -> KVBlockSpec, so we can reconstruct after a disk hit.
        self._specs: dict[CpuChunkId, KVBlockSpec] = {}
        self._lock = threading.Lock()
        logger.info(
            "LMCacheHostBackend initialized. chunk_id space=%d, hot capacity=%d "
            "chunks, lmcache_store=%s", self.max_num_cpu_chunks, self.hot_capacity,
            type(self._store).__name__ if self._store else None)

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

    def _store_key(self, chunk_id: CpuChunkId) -> str:
        """Content-hash key if a hint is set (Depth 2), else chunk_id key."""
        h = self._hash_hints.get(chunk_id)
        return f"h:{h}" if h is not None else f"c:{chunk_id}"

    def set_chunk_hash_hint(self, chunk_id: CpuChunkId, chunk_hash: str) -> None:
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
                try:
                    self._persist(chunk_id, value)
                except Exception as e:  # pragma: no cover - defensive
                    logger.warning("LMCache persist failed for chunk %s: %s",
                                   chunk_id, e)

            # Evict hot-tier LRU beyond capacity; evicted blocks remain in the
            # LMCache tier and are transparently reloaded on get().
            self._evict_hot_if_needed()
        return True

    def _persist(self, chunk_id: CpuChunkId, value: Any) -> None:
        key = self._store_key(chunk_id)
        if isinstance(value, list):
            specs = []
            for i, layer in enumerate(value):
                raw, spec = jax_block_to_bytes(layer)
                self._store.put(f"{key}#{i}", raw)
                specs.append(spec)
            self._specs[chunk_id] = specs  # type: ignore[assignment]
        else:
            raw, spec = jax_block_to_bytes(value)
            self._store.put(key, raw)
            self._specs[chunk_id] = spec

    def _evict_hot_if_needed(self) -> None:
        # Keep the hot tier bounded to hot_capacity. Evicted blocks remain in
        # the LMCache tier and are transparently reloaded on get().
        while len(self.cache) > self.hot_capacity:
            evict_id, evict_val = self.cache.popitem(last=False)
            self.current_size_bytes -= self._get_value_size(evict_val)
            self._num_saved_cpu_chunks -= 1
            if self._store is None:
                # No persistent tier -> this is a real drop (matches capacity).
                logger.debug("Hot-tier drop chunk %s (no LMCache tier)", evict_id)
            del evict_val

    # ---- load ----------------------------------------------------------------------
    def get(self, chunk_id: CpuChunkId) -> Optional[Any]:
        with self._lock:
            if chunk_id in self.cache:
                self.cache.move_to_end(chunk_id)
                return self.cache[chunk_id]
            # Hot miss: try the LMCache tier.
            if self._store is not None and chunk_id in self._specs:
                try:
                    val = self._reload(chunk_id)
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

    def _reload(self, chunk_id: CpuChunkId) -> Optional[Any]:
        key = self._store_key(chunk_id)
        spec = self._specs.get(chunk_id)
        if spec is None:
            return None
        if isinstance(spec, list):
            layers = []
            for i, s in enumerate(spec):
                raw = self._store.get(f"{key}#{i}")
                if raw is None:
                    return None
                layers.append(bytes_to_jax_block(raw, s))
            return layers
        raw = self._store.get(key)
        if raw is None:
            return None
        return bytes_to_jax_block(raw, spec)

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
                for chunk_id in [c for c in list(self._specs) if c not in occupied]:
                    if chunk_id not in self._hash_hints:
                        self._drop_persistent(chunk_id)

    def _drop_persistent(self, chunk_id: CpuChunkId) -> None:
        key = self._store_key(chunk_id)
        spec = self._specs.pop(chunk_id, None)
        if spec is None:
            return
        try:
            if isinstance(spec, list):
                for i in range(len(spec)):
                    self._store.remove(f"{key}#{i}")
            else:
                self._store.remove(key)
        except Exception:  # pragma: no cover - defensive
            pass
