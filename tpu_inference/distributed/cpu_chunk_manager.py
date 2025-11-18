# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections import OrderedDict
from dataclasses import dataclass
from typing import Literal, Tuple

from vllm.v1.core.kv_cache_utils import BlockHash

from tpu_inference.logger import init_logger

logger = init_logger(__name__)

GB = 1024**3
DEFAULT_CPU_CACHE_SIZE_BYTES = 1 * GB

ChunkHash = BlockHash


@dataclass
class CPUChunk:
    chunk_id: int
    ref_cnt: int = -1
    _chunk_hash: ChunkHash | None = None

    @property
    def is_ready_to_load(self):
        return self.ref_cnt >= 0

    @property
    def is_ready_to_evict(self):
        return self.ref_cnt <= 0

    @property
    def is_in_use(self):
        return self.ref_cnt >= 1

    @property
    def chunk_hash(self):
        return self._chunk_hash

    def touch(self):
        self.ref_cnt += 1

    def untouch(self):
        self.ref_cnt -= 1

    def reset(self):
        self._chunk_hash = None
        self.ref_cnt = -1


class CPUChunkPool:

    def __init__(self, num_chunks: int):
        self.num_chunks: int = num_chunks
        self._num_allocated_chunks: int = 0
        self.free_chunk_list: list[CPUChunk] = [
            CPUChunk(idx) for idx in range(num_chunks - 1, -1, -1)
        ]
        # {allocated_chunk_id: chunk_hash}
        self.allocated_id_to_hash_map: dict[int, ChunkHash] = {}

    @property
    def num_free_chunks(self):
        return self.num_chunks - self._num_allocated_chunks

    @property
    def num_allocated_chunks(self):
        return self._num_allocated_chunks

    def allocate_chunks(self, chunk_hashes: list[ChunkHash]) -> list[CPUChunk]:
        num_required_chunks = len(chunk_hashes)
        if num_required_chunks > self.num_free_chunks:
            raise ValueError(
                f"Cannot get {num_required_chunks} free chunks from the pool")

        ret: list[CPUChunk] = [
            self.free_chunk_list.pop() for _ in range(num_required_chunks)
        ]
        for chunk, chunk_hash in zip(ret, chunk_hashes):
            chunk._chunk_hash = chunk_hash
            assert chunk.chunk_id not in self.allocated_id_to_hash_map
            self.allocated_id_to_hash_map[chunk.chunk_id] = chunk_hash

        return ret

    def release_chunks(self, chunks: list[CPUChunk]):
        for chunk in chunks:
            if not chunk.is_ready_to_evict:
                logger.warning(f"  Chunk[{chunk.chunk_id}] is still in use.")
            assert chunk.chunk_id in self.allocated_id_to_hash_map
            self.allocated_id_to_hash_map.pop(chunk.chunk_id)
            self.free_chunk_list.append(chunk)
            chunk.reset()
        self._num_allocated_chunks -= len(chunks)


class LRUOffloadingManager:

    def __init__(self, num_cpu_chunks: int):
        self.num_chunks = num_cpu_chunks
        self.chunk_pool = CPUChunkPool(self.num_chunks)

        self.cpu_cache: OrderedDict[ChunkHash, CPUChunk] = OrderedDict()

        # The cache is an OrderedDict for LRU behavior.
    def lookup(self, chunk_hashes: list[ChunkHash]) -> int:
        """_summary_
        return the number of cache hit starting from the first chunk
        """
        hit_count = 0
        for chunk_hash in chunk_hashes:
            chunk = self.cpu_cache.get(chunk_hash)
            if chunk is None or not chunk.is_ready_to_load:
                break
            hit_count += 1
        return hit_count

    def touch(self, chunk_hashes: list[ChunkHash]) -> int:
        """ access chunks for both save / load; and move them to the end."""
        for chunk_hash in reversed(chunk_hashes):
            if self.cpu_cache.get(chunk_hash):
                self.cpu_cache.move_to_end(chunk_hash)

    def allocate_for_save(
        self, chunk_hashes: list[ChunkHash]
    ) -> Tuple[list[CPUChunk], list[int]] | None:
        # filter out chunks that are already stored
        num_chunks = len(chunk_hashes)
        new_chunk_idxs = [
            i for i in range(num_chunks)
            if chunk_hashes[i] not in self.cpu_cache
        ]

        num_new_chunks = len(new_chunk_idxs)
        if num_new_chunks == 0:
            logger.info("No new chunks to allocate")
            return None
        num_chunks_to_evict = max(
            0, num_new_chunks - self.chunk_pool.num_free_chunks)

        # build list of chunks to evict / reuse
        to_evict = []
        if num_chunks_to_evict > 0:
            for chunk_hash, chunk in self.cpu_cache.items():
                if chunk.is_ready_to_evict:
                    to_evict.append(chunk_hash)
                    num_chunks_to_evict -= 1
                    if num_chunks_to_evict == 0:
                        break
            else:
                # we could not evict enough chunks
                return None

        # evict chunks
        self.chunk_pool.release_chunks([
            self.cpu_cache.pop(evicting_chunk_hash)
            for evicting_chunk_hash in to_evict
        ])

        new_chunk_hashes = [chunk_hashes[i] for i in new_chunk_idxs]
        # allocate
        try:
            new_chunks = self.chunk_pool.allocate_chunks(new_chunk_hashes)
            assert len(new_chunks) == len(new_chunk_hashes)
        except Exception as e:
            logger.warning(f" Failed to allocate {len(new_chunk_hashes)}: {e}")
            # NOTE(jcgu): should we return None or something else?
            return None
        for chunk_hash, chunk in zip(new_chunk_hashes, new_chunks):
            self.cpu_cache[chunk_hash] = chunk
        # newly-allocated chunks, chunk-idx in the given chunk_hashes list
        return new_chunks, new_chunk_idxs

    def prepare_load(self, chunk_hashes: list[ChunkHash]) -> list[CPUChunk]:
        chunks = []
        for chunk_hash in chunk_hashes:
            chunk = self.cpu_cache[chunk_hash]
            assert chunk.is_ready_to_load
            chunk.touch()
            chunks.append(chunk)
        return chunks

    def complete_save(self, chunk_hashes: list[ChunkHash]) -> None:
        """ After store completion, mark the chunk to be ready to load."""
        for chunk_hash in chunk_hashes:
            chunk = self.cpu_cache[chunk_hash]
            assert not chunk.is_ready_to_load
            # mark ready to load
            chunk.touch()
            assert chunk.is_ready_to_load

    def complete_load(self, chunk_hashes: list[ChunkHash]) -> None:
        for chunk_hash in chunk_hashes:
            chunk = self.cpu_cache[chunk_hash]
            assert chunk.is_in_use
            chunk.untouch()

    def mark_completion(self, chunk_ids, operation: Literal['save',
                                                            'load']) -> None:
        chunk_hashes = [
            self.chunk_pool.allocated_id_to_hash_map[chunk_id]
            for chunk_id in chunk_ids
        ]
        chunk_hashes = []
        unknown_chunk_ids = []
        for chunk_id in chunk_ids:
            if chunk_id in self.chunk_pool.allocated_id_to_hash_map:
                chunk_hashes.append(
                    self.chunk_pool.allocated_id_to_hash_map[chunk_id])
            else:
                unknown_chunk_ids.append(chunk_id)
        logger.warning(
            f"  Chunks[{unknown_chunk_ids}] are not found as allocated chunks in the pool."
        )

        if operation == 'save':
            self.complete_save(chunk_hashes)
        elif operation == 'load':
            self.complete_load(chunk_hashes)
        else:
            raise ValueError(f"Unknown operation: {operation}")
