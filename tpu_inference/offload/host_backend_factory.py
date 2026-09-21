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
"""Factory that selects the TPUOffloadConnector host store.

Default (TPU_OFFLOAD_LMCACHE=0): the stock in-memory ``LocalCPUBackend`` — zero
behavior change. When TPU_OFFLOAD_LMCACHE=1: an ``LMCacheHostBackend`` with a
persistent spill tier selected by TPU_OFFLOAD_LMCACHE_BACKEND:
  * "file"    -> FileKVStore (reference filesystem spill, no LMCache install)
  * "memory"  -> InMemoryKVStore (reference, single-process)
  * "lmcache" -> LMCacheStorageKVStore (real LMCache StorageManager: disk/remote/
                 P2P/cross-instance) from the lmcache fork. Imported lazily so
                 tpu-inference has no hard dependency on lmcache.
"""
from __future__ import annotations

from typing import Any

from tpu_inference import envs
from tpu_inference.logger import init_logger

logger = init_logger(__name__)


def build_host_backend(num_cpu_chunks: int, model_name: str = "") -> Any:
    """Return the host KV store for the TPUOffloadConnector worker."""
    if not envs.TPU_OFFLOAD_LMCACHE:
        # Stock path: identical to upstream.
        from tpu_inference.offload.cpu_backend import LocalCPUBackend
        return LocalCPUBackend(num_cpu_chunks=num_cpu_chunks)

    from tpu_inference.offload.lmcache_host_backend import LMCacheHostBackend

    backend = envs.TPU_OFFLOAD_LMCACHE_BACKEND
    hot_chunks = envs.TPU_OFFLOAD_LMCACHE_HOT_CHUNKS or num_cpu_chunks

    store = _build_store(backend, model_name=model_name)
    logger.info(
        "TPU_OFFLOAD_LMCACHE enabled: backend=%s hot_chunks=%d chunk_id_space=%d",
        backend, hot_chunks, num_cpu_chunks)
    return LMCacheHostBackend(
        num_cpu_chunks=num_cpu_chunks,
        lmcache_store=store,
        hot_capacity=hot_chunks,
    )


def _build_store(backend: str, model_name: str = "") -> Any:
    from tpu_inference.offload.lmcache_kv_store import (FileKVStore,
                                                        InMemoryKVStore)
    if backend == "memory":
        return InMemoryKVStore()
    if backend == "file":
        return FileKVStore(root=envs.TPU_OFFLOAD_LMCACHE_PATH)
    if backend == "lmcache":
        # Real LMCache StorageManager (disk/remote/P2P/cross-instance). Lazy
        # import so lmcache is only required when this backend is selected.
        try:
            from lmcache.integration.tpu.lmcache_storage_kv_store import (
                LMCacheStorageKVStore)
        except Exception as e:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "TPU_OFFLOAD_LMCACHE_BACKEND=lmcache requires the lmcache fork "
                "with the TPU integration (lmcache.integration.tpu). "
                f"Import failed: {e}")
        return LMCacheStorageKVStore.from_env(model_name=model_name)
    raise ValueError(f"Unknown TPU_OFFLOAD_LMCACHE_BACKEND: {backend!r}")
