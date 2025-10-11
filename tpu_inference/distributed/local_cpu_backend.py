# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any, Optional

from tpu_inference.logger import init_logger

from .util import CacheKey

logger = init_logger(__name__)


class LocalCPUBackend:
    """
    A singleton in-memory CPU backend for storing KV cache keys and values.

    This class uses the singleton pattern to ensure that the scheduler and the
    worker, running in the same process, can share the same cache.
    The scheduler reads from this to find cache hits, and the worker writes
    to it after saving KV blocks from the TPU.
    """
    _instance: Optional["LocalCPUBackend"] = None
    _initialized: bool = False

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(LocalCPUBackend, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        # The cache is now a dictionary mapping CacheKey -> KV cache data
        self.cache: dict[CacheKey, Any] = {}
        self.current_size_bytes = 0
        self._initialized = True
        logger.info(
            "Singleton LocalCPUBackend initialized as a Key-Value store.")

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
            import sys
            size_in_bytes = sys.getsizeof(value)
        logger.info(
            f"Calculated size for value of type {type(value)}: {size_in_bytes} bytes"
        )
        return size_in_bytes

    def add(self, key: CacheKey, value: Any):
        """Adds a key-value pair to the cache."""
        if key not in self.cache:
            self.cache[key] = value
            value_size = self._get_value_size(value)
            self.current_size_bytes += value_size
            logger.info(f"Added key to CPU backend. Hash: {key.chunk_hash}")
            logger.info(
                f"Cache size: {self.current_size_bytes / 1024**2:.2f} MB")

    def get(self, key: CacheKey) -> Optional[Any]:
        """Gets the value for a given key."""
        return self.cache.get(key)

    def contains(self, key: CacheKey) -> bool:
        """Checks if a key exists in the cache."""
        return key in self.cache
