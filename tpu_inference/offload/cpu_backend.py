# SPDX-License-Identifier: Apache-2.0

import sys
from collections import OrderedDict
from typing import Any, Optional

from tpu_inference.logger import init_logger
from tpu_inference.offload.utils import CpuChunkId

logger = init_logger(__name__)


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

    def __init__(self, num_cpu_chunks: int):
        self.max_num_cpu_chunks = num_cpu_chunks
        self.cache: OrderedDict[CpuChunkId, Any] = OrderedDict()
        self.current_size_bytes = 0
        self._num_saved_cpu_chunks = 0
        logger.info(
            "LocalCPUBackend initialized."
            f"CPU cache capacity: {self.max_num_cpu_chunks} chunks / pages.")

    @property
    def num_saved_cpu_chunks(self) -> int:
        return self._num_saved_cpu_chunks

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

    def add(self, chunk_id: CpuChunkId, value: Any) -> bool:
        """
        Adds a key-value pair to the cache.

        If the cache is full, it evicts the least recently used, unpinned
        entries until there is enough space.
        """
        if chunk_id < 0 or chunk_id >= self.max_num_cpu_chunks:
            # TODO(jcgu): report failure when offload scheduler / worker
            # can handle failed operations.
            raise ValueError(f" get invalid chunk_id: {chunk_id}")

        # Add the new item.
        if chunk_id in self.cache:
            old_value = self.cache.pop(chunk_id)
            self.current_size_bytes -= self._get_value_size(old_value)
            del old_value
            self._num_saved_cpu_chunks -= 1

        self.cache[chunk_id] = value
        self._num_saved_cpu_chunks += 1
        value_size = self._get_value_size(value)
        self.current_size_bytes += value_size
        logger.info(
            f"Added chunk_id: {chunk_id} (size:{value_size}) to CPU backend.")
        logger.info(
            f"Cache: {self.current_size_bytes} bytes, {self._num_saved_cpu_chunks} occupied chunks."
        )
        return True

    def get(self, chunk_id: CpuChunkId) -> Optional[Any]:
        """
        Gets the value for a given chunk_id and marks it as recently used.
        """
        if chunk_id in self.cache:
            return self.cache[chunk_id]
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
