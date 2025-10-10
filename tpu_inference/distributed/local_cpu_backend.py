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
        self._initialized = True
        logger.info(
            "Singleton LocalCPUBackend initialized as a Key-Value store.")

    def add(self, key: CacheKey, value: Any):
        """Adds a key-value pair to the cache."""
        if key not in self.cache:
            self.cache[key] = value
            logger.debug(f"Added key to CPU backend. Hash: {key.chunk_hash}")

    def get(self, key: CacheKey) -> Optional[Any]:
        """Gets the value for a given key."""
        return self.cache.get(key)

    def contains(self, key: CacheKey) -> bool:
        """Checks if a key exists in the cache."""
        return key in self.cache
