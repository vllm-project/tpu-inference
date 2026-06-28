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

import math
import os
import queue
import threading
import time
from typing import Any, Literal, Optional, Tuple

import tensorstore as ts

from tpu_inference.logger import init_logger
from tpu_inference.offload.cpu_backend import Backend, LocalCPUBackend
from tpu_inference.offload.offload_manager import (CacheManager, ChunkHash,
                                                   CPUChunk, LRUCacheManager)
from tpu_inference.offload.utils import CpuChunkId, FileMapper

logger = init_logger(__name__)

BASE_PATH_CONFIG = 'fs_offload_base_path'


class TieredCacheManager(CacheManager):
    """Manages a logical tiering of CPU and filesystem offloaded KV cache.

    Enable with fs_offload_base_path in kv_transfer_config.extra_config.

    This wraps an LRUCacheManager to track if a cache entry is already
    present in memory before doing the expensive lookup to storage.

    Movement of chunks from storage to CPU is handled by the backend.

    This lives in the scheduler of the TPUOffloadConnector, while the
    TieredBackend lives in the worker process.

    This manager tracks what blocks are availble, performing
    filesystem lookups when necessary for determining if cached blocks
    exist in the shared storage. Blocks passed back to vLLM are
    eventually routed to the worker, where the TieredBackend will
    assume that any blocks requested, exist in the cache. How this will
    deal with eviction and such is TBD (we'll probably want some kind
    of locking on the file names?)
    """

    def __init__(self, cpu_manager: LRUCacheManager, mapper: FileMapper):
        self.cpu_manager = cpu_manager
        self.mapper = mapper

    def lookup(self, chunk_hashes: list[ChunkHash]) -> int:
        """_summary_
        return the number of cache hit starting from the first chunk
        """
        cpu_count = self.cpu_manager.lookup(chunk_hashes)

        fs_count = 0
        for chunk_hash in chunk_hashes[cpu_count:]:
            # This should be async --- but in testing, this seemed to be called
            # with a single hash rather than a list, so the async needs to
            # happen further up.
            filename = self.mapper.get_file_name(chunk_hash)
            if not os.path.exists(filename):
                break
            fs_count += 1

        logger.info(f'Found {cpu_count} chunks on cpu and {fs_count} on disk')
        return cpu_count + fs_count

    def touch(self, chunk_hashes: list[ChunkHash]) -> int:
        """ access chunks for both save / load; and move them to the end."""
        # The cpu manager will ignore any chunks that aren't in memory.
        self.cpu_manager.touch(chunk_hashes)
        # TODO: eviction for the filesystem

    def allocate_for_save(
        self, chunk_hashes: list[ChunkHash]
    ) -> Tuple[list[CPUChunk], list[int]] | None:
        chunks, ids = self.cpu_manager.allocate_for_save(chunk_hashes)
        # There's nothing to allocate on the filesystem side, the
        # backend will offload all hashes it sees.
        return chunks, ids

    def prepare_load(self, chunk_hashes: list[ChunkHash]) -> list[CPUChunk]:
        fs_hashes = []
        cpu_hashes = []
        chunk_map = {}
        for chunk_hash in chunk_hashes:
            if not self.cpu_manager.lookup([chunk_hash]):
                fs_hashes.append(chunk_hash)
            else:
                cpu_hashes.append(chunk_hash)
        if cpu_hashes:
            prepared = self.cpu_manager.prepare_load(cpu_hashes)
            assert prepared
            assert len(prepared) == len(cpu_hashes)
            for hash, chunk in zip(cpu_hashes, prepared):
                chunk_map[hash] = chunk

        # todo: what if pulling in fs_chunks evicts cpu chunks?
        if fs_hashes:
            # The CPU manager silently ignores hashes that aren't in the cache. We'll do
            # the same for fs hashes, even though the lookup is expensive.
            fs_hashes = [hash for hash in fs_hashes if self.lookup([hash])]

        if fs_hashes:
            allocated = self.cpu_manager.allocate_for_save(fs_hashes)
            if allocated is None:
                raise Exception("yikes, couldn't prepare load")
            fs_chunks, chunk_indices = allocated
            self.cpu_manager.complete_save(fs_hashes)
            self.cpu_manager.prepare_load(fs_hashes)
            assert len(chunk_indices) == len(fs_hashes)
            assert all(i in chunk_indices for i in range(len(fs_hashes)))
            assert all(fs_chunks[i].chunk_hash == fs_hashes[i]
                       for i in chunk_indices)
            chunk_map.update((c.chunk_hash, c) for c in fs_chunks)
        assert sum(hash in chunk_map
                   for hash in chunk_hashes) == len(chunk_hashes)
        return list(chunk_map[hash] for hash in chunk_hashes)

    def complete_save(self, chunk_hashes: list[ChunkHash]) -> None:
        # Like allocate_for_save, this affects in-memory chunks only. FS save is async
        # and handled by the backend.
        self.cpu_manager.complete_save(chunk_hashes)

    def complete_load(self, chunk_hashes: list[ChunkHash]) -> None:
        # The cpu_manager hash has been updated in prepare_load to contain all chunks that
        # will eventually be loaded in. So this complete step is just passed through.
        self.cpu_manager.complete_load(chunk_hashes)

    def mark_completion(self, chunk_ids, operation: Literal['save',
                                                            'load']) -> None:
        # See the comments for complete_save and complete_load for why this is
        # pass-through.
        self.cpu_manager.mark_completion(chunk_ids, operation)


class TieredBackend(Backend):
    """
    A tiered backend for storing KV caches.

    Keeps chunks in memory, using a LocalCPUBackend. Offloads
    asynchronously to a filesystem cache, transparently loads on get.
    """

    def __init__(self, mapper: FileMapper, cpu_backend: LocalCPUBackend):
        self.mapper = mapper
        self.host_ram_backend = cpu_backend
        self.write_queue = queue.Queue(
        )  # Store tensorstore write futures, no idea what type that is.

        def write_queue_worker():
            while True:
                try:
                    self.write_queue.get().result()
                except Exception as e:
                    logger.warning(f'write failed: {e}')
                self.write_queue.task_done()

        threading.Thread(target=write_queue_worker, daemon=True).start()

    def add(self, chunk_id: CpuChunkId, chunk_hash: ChunkHash,
            value: Any) -> bool:
        """
        Adds a key-value pair to the cache.
        """
        self.host_ram_backend.add(chunk_id, chunk_hash, value)
        self._start_save_to_fs(chunk_hash, value)
        return True

    def get(self, chunk_id: CpuChunkId, hash: ChunkHash) -> Optional[Any]:
        """
        Gets the value for a given chunk_hash
        """
        chunk = self.host_ram_backend.get(chunk_id, hash)
        if chunk is not None:
            return chunk

        chunk = self._load_from_fs(hash)
        if chunk is not None:
            self.host_ram_backend.add(chunk_id, hash, chunk)
        return chunk

    def wait_for_add_background(self):
        self.write_queue.join()

    def _load_from_fs(self, hash: ChunkHash) -> Optional[Any]:
        """
        Synchronous load from the filesystem

        TODO: probably want this to be async so we can increase io depth.
        """
        # TODO: parametertize the spec
        spec = {
            'driver': 'zarr',
            'kvstore': {
                'driver': 'file',
                'path': self.mapper.get_file_name(hash),
            },
        }
        start = time.time()
        dataset = ts.open(spec).result()
        res = dataset.read().result()
        return res

    def _start_save_to_fs(self, chunk_hash: ChunkHash, chunk: Any):
        """
        Async save to the backend
        """
        # Set the chunk size to be like the input shape, but reduced to 0.5-1 Gb per
        # chunk, if necessary.
        chunk_bytes = math.prod(chunk.shape)
        factor = 750 * 2**20 / chunk_bytes
        if factor < 1:
            chunking = [max(1, math.floor(c * factor)) for c in chunk.shape]
        else:
            chunking = chunk.shape

        spec = {
            'driver': 'zarr',
            'kvstore': {
                'driver': 'file',
                'path': self.mapper.get_file_name(chunk_hash),
            },
            'metadata': {
                'shape': chunk.shape,
                'dtype': self.mapper.dtype,
                'chunks': chunking,
            },
        }

        tstore = ts.open(spec, create=True, delete_existing=True).result()
        op = tstore.write(chunk)
        op_start = time.time()
        self.write_queue.put_nowait(op)
