# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
import sys
from collections import OrderedDict
from typing import Any, List, Optional, Tuple

from tpu_inference.logger import init_logger

from .cache_util import CacheKey

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
    _instance: Optional["LocalCPUBackend"] = None
    _initialized: bool = False

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(LocalCPUBackend, cls).__new__(cls)
        return cls._instance

    def __init__(self,
                 max_cpu_cache_size_bytes: int = DEFAULT_CPU_CACHE_SIZE_BYTES):
        if self._initialized:
            return

        env_cache_size_gb = os.getenv("TPU_OFFLOAD_CPU_CACHE_SIZE_GB")
        self.max_cpu_cache_size_bytes = (int(env_cache_size_gb) *
                                         GB if env_cache_size_gb is not None
                                         else max_cpu_cache_size_bytes)

        # The cache is an OrderedDict for LRU behavior.
        self.cache: OrderedDict[CacheKey, Any] = OrderedDict()
        self.current_size_bytes = 0
        self.pinned_keys: set[CacheKey] = set()
        self._initialized = True
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

    def add(self, key: CacheKey, value: Any):
        """
        Adds a key-value pair to the cache.

        If the cache is full, it evicts the least recently used, unpinned
        entries until there is enough space.
        """
        value_size = self._get_value_size(value)
        # Do not add if the item itself is larger than the cache capacity.
        if value_size > self.max_cpu_cache_size_bytes:
            logger.warning(
                f"Cannot add item of size {value_size} bytes to "
                f"cache with capacity {self.max_cpu_cache_size_bytes} bytes.")
            return

        # If key already exists, remove it to update its size and position.
        if key in self.cache:
            old_value = self.cache.pop(key)
            self.current_size_bytes -= self._get_value_size(old_value)

        # Evict old, unpinned entries until there is enough space for the new item.
        while self.current_size_bytes + value_size > self.max_cpu_cache_size_bytes:
            evicted_key = None
            # Find the first unpinned key from the LRU end of the cache.
            for k in self.cache:
                if k not in self.pinned_keys:
                    evicted_key = k
                    break

            # If no unpinned key can be evicted, we cannot make space.
            if evicted_key is None:
                logger.warning(
                    "Cache is full of pinned items. Cannot add new key "
                    f"({key.chunk_hash}) until some are unpinned.")
                # If we popped the key before, we need to decide what to do.
                # For simplicity, we just won't add the new value.
                return

            # Evict the found key.
            evicted_value = self.cache.pop(evicted_key)
            self.current_size_bytes -= self._get_value_size(evicted_value)
            logger.info(f"Evicted key {evicted_key.chunk_hash} to make space.")

        # Add the new item.
        self.cache[key] = value
        self.current_size_bytes += value_size
        logger.info(f"Added key to CPU backend. Hash: {key.chunk_hash}")
        logger.info(f"Cache size: {self.current_size_bytes} bytes / "
                    f"{self.max_cpu_cache_size_bytes} bytes")

    def get(self, key: CacheKey) -> Optional[Any]:
        """
        Gets the value for a given key and marks it as recently used.
        """
        if key in self.cache:
            # Mark as most recently used.
            self.cache.move_to_end(key)
            return self.cache[key]
        return None

    def contains(self, key: CacheKey, pin_on_hit: bool = False) -> bool:
        """
        Checks if a key exists in the cache.

        If the key is found, it's marked as recently used. If `pin_on_hit`
        is True, the key is also pinned to prevent eviction.
        """
        if key in self.cache:
            # Mark as most recently used, since this is an access.
            self.cache.move_to_end(key)
            if pin_on_hit:
                self.pinned_keys.add(key)
                logger.info(f"Pinned key on hit. Hash: {key.chunk_hash}")
            return True
        return False

    def unpin_keys(self, keys: List[CacheKey]) -> Tuple[int, int]:
        """Unpins a list of keys, making them eligible for eviction again."""
        unpinned_count = 0
        found_count = 0
        for key in keys:
            if key in self.pinned_keys:
                found_count += 1
                self.pinned_keys.remove(key)
                unpinned_count += 1
                logger.info(f"Unpinned key. Hash: {key.chunk_hash}")
        return unpinned_count, found_count
