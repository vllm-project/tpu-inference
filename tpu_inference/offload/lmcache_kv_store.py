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
"""Raw-bytes KV store protocol for the LMCache-on-TPU host backend.

``LMCacheHostBackend`` needs a simple, engine-agnostic persistence tier:
``put(key: str, data: bytes)`` / ``get(key) -> bytes | None`` / ``remove(key)``.

We keep this deliberately minimal (raw bytes, string keys) so it can be backed by:
  * an in-memory dict (reference / tests),
  * a plain filesystem directory (reference / single-host disk spill),
  * LMCache's real StorageManager (disk / remote / P2P / cross-instance) via the
    ``LMCacheStorageKVStore`` adapter in the lmcache fork.

Raw bytes (not torch tensors / MemoryObj) is the correct interface here because
the TPU KV blocks are JAX arrays; the value bridge already serializes them to a
flat, dtype-preserving byte buffer. This avoids forcing LMCache's torch-centric
MemoryObj assumptions onto the JAX path.
"""
from __future__ import annotations

import os
import threading
from typing import Optional, Protocol, runtime_checkable


@runtime_checkable
class LMCacheKVStore(Protocol):
    """Minimal raw-bytes persistence contract."""

    def put(self, key: str, data: bytes) -> None: ...

    def get(self, key: str) -> Optional[bytes]: ...

    def remove(self, key: str) -> None: ...

    def contains(self, key: str) -> bool: ...


class InMemoryKVStore:
    """Reference in-memory store (for tests and single-process spill)."""

    def __init__(self) -> None:
        self._d: dict[str, bytes] = {}
        self._lock = threading.Lock()

    def put(self, key: str, data: bytes) -> None:
        with self._lock:
            self._d[key] = data

    def get(self, key: str) -> Optional[bytes]:
        with self._lock:
            return self._d.get(key)

    def remove(self, key: str) -> None:
        with self._lock:
            self._d.pop(key, None)

    def contains(self, key: str) -> bool:
        with self._lock:
            return key in self._d


class FileKVStore:
    """Reference filesystem store: one file per key. Single-host disk spill.

    A safe, dependency-free stand-in that mirrors what LMCache's LocalDiskBackend
    does at the byte level, so the connector integration can be validated without
    a full LMCache install. Production uses LMCacheStorageKVStore instead.
    """

    def __init__(self, root: str) -> None:
        self._root = root
        os.makedirs(self._root, exist_ok=True)
        self._lock = threading.Lock()

    def _path(self, key: str) -> str:
        safe = key.replace("/", "-").replace(":", "_").replace("#", "__")
        return os.path.join(self._root, safe + ".kvb")

    def put(self, key: str, data: bytes) -> None:
        path = self._path(key)
        tmp = path + ".tmp"
        with self._lock:
            with open(tmp, "wb") as f:
                f.write(data)
            os.replace(tmp, path)  # atomic

    def get(self, key: str) -> Optional[bytes]:
        path = self._path(key)
        with self._lock:
            if not os.path.exists(path):
                return None
            with open(path, "rb") as f:
                return f.read()

    def remove(self, key: str) -> None:
        path = self._path(key)
        with self._lock:
            if os.path.exists(path):
                os.remove(path)

    def contains(self, key: str) -> bool:
        with self._lock:
            return os.path.exists(self._path(key))
