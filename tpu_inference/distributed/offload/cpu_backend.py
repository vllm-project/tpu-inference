# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
import sys
from collections import OrderedDict
from typing import Any, Optional

from tpu_inference.distributed.offload.utils import CpuChunkId
from tpu_inference.logger import init_logger

logger = init_logger(__name__)

GB = 1024**3
DEFAULT_CPU_CACHE_SIZE_BYTES = 1 * GB


# TODO(jcgu): creating independent cpu backends since scheduler & worker could be in different processes.
class LocalCPUBackend:
    """
    A singleton in-memory CPU backend for storing KV cache keys and values.

    This class uses the singleton pattern to ensure that the scheduler and the
    worker, running in the same process, can share the same cache.
    The scheduler reads from this to find cache hits, and the worker writes
    to it after saving KV blocks from the TPU.

    It implements an LRU (Least Recently Used) eviction policy with a maximum
    size limit and support for pinning cache entries to prevent eviction.
    """

    def __init__(self,
                 max_cpu_cache_size_bytes: int = DEFAULT_CPU_CACHE_SIZE_BYTES):
        env_cache_size_gb = os.getenv("TPU_OFFLOAD_CPU_CACHE_SIZE_GB")
        self.max_cpu_cache_size_bytes = (int(env_cache_size_gb) *
                                         GB if env_cache_size_gb is not None
                                         else max_cpu_cache_size_bytes)

        # The cache is an OrderedDict for LRU behavior.
        self.cache: OrderedDict[CpuChunkId, Any] = OrderedDict()
        self.current_size_bytes = 0
        logger.info("Singleton LocalCPUBackend initialized."
                    f"CPU cache size: {self.max_cpu_cache_size_bytes} bytes")

    def _get_value_size(self, value: Any) -> int:
        """Calculates the size of a cache value in bytes."""
        size_in_bytes = 0
        if isinstance(value, list):
            # The value is a list of JAX arrays (one per layer)
            size_in_bytes = sum(v.nbytes for v in value
                                if hasattr(v, 'nbytes'))
        elif hasattr(value, 'nbytes'):
            size_in_bytes = value.nbytes
        else:
            size_in_bytes = sys.getsizeof(value)
        return size_in_bytes

    def add(self, key: CpuChunkId, value: Any) -> bool:
        """
        Adds a key-value pair to the cache.

        If the cache is full, it evicts the least recently used, unpinned
        entries until there is enough space.
        """
        # Add the new item.
        if key in self.cache:
            old_value = self.cache.pop(key)
            self.current_size_bytes -= self._get_value_size(old_value)
            del old_value

        self.cache[key] = value
        value_size = self._get_value_size(value)
        self.current_size_bytes += value_size
        logger.info(f"Added key: {key} (size:{value_size}) to CPU backend.")
        logger.info(f"Cache size: {self.current_size_bytes} bytes / "
                    f"{self.max_cpu_cache_size_bytes} bytes")
        return True

    def get(self, key: CpuChunkId) -> Optional[Any]:
        """
        Gets the value for a given key and marks it as recently used.
        """
        if key in self.cache:
            return self.cache[key]
        return None

    def reclaim_unoccupied_chunks(self, occupied_chunk_ids: list[CpuChunkId]):
        chunk_ids = list(self.cache.keys())
        unoccupied_chunk_ids = [
            chunk_id for chunk_id in chunk_ids
            if chunk_id not in occupied_chunk_ids
        ]
        reclaimed_size_bytes = 0
        for chunk_id in unoccupied_chunk_ids:
            dummy_value = self.cache.pop(chunk_id)
            reclaimed_size_bytes += self._get_value_size(dummy_value)
            del dummy_value
        self.current_size_bytes -= reclaimed_size_bytes

        logger.info(
            f" Reclaimed {len(unoccupied_chunk_ids)} unoccupied chunks, "
            f"with {reclaimed_size_bytes} bytes.")
