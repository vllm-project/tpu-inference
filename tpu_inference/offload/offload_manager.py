# SPDX-License-Identifier: Apache-2.0

from collections import OrderedDict
from dataclasses import dataclass
from typing import Literal, Optional, Tuple

from vllm.v1.core.kv_cache_utils import BlockHash

from tpu_inference.logger import init_logger
from tpu_inference.offload.utils import CpuChunkId, ReqId

logger = init_logger(__name__)

ChunkHash = BlockHash


@dataclass
class CPUChunk:
    """
    ref_cnt:
    -1: init, not saved
    0: saved, ready_to_evict, ready_to_load
    >=1: loadings, ready_to_load, in_use
    """
    chunk_id: CpuChunkId
    ref_cnt: int = -1
    _chunk_hash: ChunkHash | None = None

    @property
    def is_ready_to_load(self):
        return self.ref_cnt >= 0

    @property
    def is_ready_to_evict(self):
        return self.ref_cnt == 0

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
        self.allocated_id_to_hash_map: dict[CpuChunkId, ChunkHash] = {}

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
        self._num_allocated_chunks += num_required_chunks
        for chunk, chunk_hash in zip(ret, chunk_hashes):
            chunk._chunk_hash = chunk_hash
            assert chunk.chunk_id not in self.allocated_id_to_hash_map
            self.allocated_id_to_hash_map[chunk.chunk_id] = chunk_hash

        return ret

    def release_chunk(self, chunk: CPUChunk) -> bool:
        if not chunk.is_ready_to_evict:
            logger.warning(f"  Chunk[{chunk.chunk_id}] is still in use.")
            return False
        assert chunk.chunk_id in self.allocated_id_to_hash_map
        self.allocated_id_to_hash_map.pop(chunk.chunk_id)
        chunk.reset()
        self.free_chunk_list.append(chunk)
        self._num_allocated_chunks -= 1
        return True


class LRUCacheManager:

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
        for evicting_chunk_hash in to_evict:
            evicting_chunk = self.cpu_cache.pop(evicting_chunk_hash)
            # always true, since all evicting chunks are ready to evict
            self.chunk_pool.release_chunk(evicting_chunk)

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
        try:
            chunk_hashes = [
                self.chunk_pool.allocated_id_to_hash_map[chunk_id]
                for chunk_id in chunk_ids
            ]
        except Exception as e:
            raise ValueError(f' failed to retrieve chunk hashes: {e}')

        chunk_hashes = []
        unknown_chunk_ids = []
        for chunk_id in chunk_ids:
            if chunk_id in self.chunk_pool.allocated_id_to_hash_map:
                chunk_hashes.append(
                    self.chunk_pool.allocated_id_to_hash_map[chunk_id])
            else:
                unknown_chunk_ids.append(chunk_id)
        if unknown_chunk_ids:
            logger.warning(
                f"  Chunks[{unknown_chunk_ids}] are not found as allocated chunks in the pool."
            )

        if operation == 'save':
            self.complete_save(chunk_hashes)
        elif operation == 'load':
            self.complete_load(chunk_hashes)
        else:
            raise ValueError(f"Unknown operation: {operation}")


class StagingBufferManager():
    """ Bookkeeping the staging buffer inside the connector scheduler.
    NOTE(jcgu): the operations (e.g., allocate, free, get) to staging buffer / blocks are NOT thread-safe.
    But it's okay since there is only one connector scheduler instance.
    """

    def __init__(self, num_blocks: int):
        self.num_blocks = num_blocks
        # {req_id: list(num_occupied_staging_blocks)}
        self._blocks_for_save: dict[ReqId, int] = {}
        self._blocks_for_load: dict[ReqId, int] = {}

        self._num_free_blocks: int = self.num_blocks
        # keep track of the total occupied staging blocks for save and load respectively
        self._num_blocks_for_save: int = 0
        self._num_blocks_for_load: int = 0

    def get_num_free_staging_blocks(self) -> int:
        return self._num_free_blocks

    def get_num_used_staging_blocks(self) -> int:
        return self._num_blocks_for_load + self._num_blocks_for_save

    def get_num_used_save_staging_blocks(self, req_id: ReqId) -> int:
        return self._blocks_for_save.get(req_id, 0)

    def get_num_used_load_staging_blocks(self, req_id: ReqId) -> int:
        return self._blocks_for_load.get(req_id, 0)

    def allocate(self, req_id: ReqId, num_blocks: int,
                 usage: Literal["load", "save"]) -> int:
        if num_blocks < 0:
            logger.warning(
                f"  get {num_blocks} staging blocks to allocate for Req:{req_id}."
            )
            return num_blocks
        if num_blocks > self._num_free_blocks:
            # do not have enough capacity, return 0
            return 0

        if usage == "load":
            if req_id in self._blocks_for_load:
                # NOTE(jcgu): before completing the previous load, new load
                # should not be triggered for the same request (is this correct?)
                raise ValueError(
                    f" Req({req_id}) already has {self._blocks_for_load[req_id]}, and should not have new loads."
                )
            else:
                self._blocks_for_load[req_id] = num_blocks
            self._num_blocks_for_load += num_blocks
        elif usage == "save":
            if req_id in self._blocks_for_save:
                self._blocks_for_save[req_id] += num_blocks
            else:
                self._blocks_for_save[req_id] = num_blocks
            self._num_blocks_for_save += num_blocks
        else:
            raise ValueError(
                f" Staging buffer manager should not get usage: {usage}")
        self._num_free_blocks -= num_blocks

        logger.info(
            f"  allocate {num_blocks} staging blocks to Req:{req_id} for {usage}."
        )
        return num_blocks

    def free(self,
             req_id: ReqId,
             usage: Literal["load", "save"],
             num_finished_blocks: Optional[int] = None) -> int:
        """
        when num_finished_blocks is not given, we will assume the request is finished and should be removed.
        """
        num_freed_blocks = 0
        # NOTE(jcgu): assuming FIFO execution order for a single request's save and
        # load operations respectively
        if usage == "load":
            if req_id not in self._blocks_for_load:
                logger.warning(
                    f" there is no record of staging buffer (usage: {usage}) for Req:{req_id}"
                )
                return 0
            if num_finished_blocks is None:
                num_freed_blocks = self._blocks_for_load[req_id]
            else:
                num_freed_blocks = num_finished_blocks
            if self._blocks_for_load[req_id] < num_freed_blocks:
                logger.warning(
                    f" Req({req_id}) has {num_finished_blocks} load staging buffer to free, but only has {self._blocks_for_load[req_id]} on record."
                )

            self._blocks_for_load[req_id] -= num_freed_blocks
            if self._blocks_for_load[req_id] <= 0:
                del self._blocks_for_load[req_id]
            self._num_blocks_for_load -= num_freed_blocks
        elif usage == "save":
            if req_id not in self._blocks_for_save:
                logger.warning(
                    f" there is no record of staging buffer (usage: {usage}) for Req:{req_id}"
                )
                return 0
            if num_finished_blocks is None:
                num_freed_blocks = self._blocks_for_save[req_id]
            else:
                num_freed_blocks = num_finished_blocks
            if self._blocks_for_save[req_id] < num_freed_blocks:
                logger.warning(
                    f" Req({req_id}) has {num_finished_blocks} save staging buffer to free, but only has {self._blocks_for_save[req_id]} on record."
                )

            self._blocks_for_save[req_id] -= num_freed_blocks
            if self._blocks_for_save[req_id] <= 0:
                del self._blocks_for_save[req_id]
            self._num_blocks_for_save -= num_freed_blocks
        else:
            raise ValueError(
                f" Staging buffer manager should not get usage: {usage}")
        self._num_free_blocks += num_freed_blocks

        logger.info(
            f"  free {num_freed_blocks} staging blocks (usage: {usage}) from Req:{req_id}"
        )
        return num_freed_blocks

    def get_usage(self, with_details: bool = False):
        usage_str = (f"Staging Buffer: total={self.num_blocks}, "
                     f"free={self._num_free_blocks}, "
                     f"used_for_load={self._num_blocks_for_load}, "
                     f"used_for_save={self._num_blocks_for_save};")
        if with_details:
            blocks_for_save_str = " save_details:{"
            for req, bn in self._blocks_for_save.items():
                blocks_for_save_str += f"{req}:{bn},"
            blocks_for_save_str += "} "

            blocks_for_load_str = " load_details:{"
            for req, bn in self._blocks_for_load.items():
                blocks_for_load_str += f"{req}:{bn},"
            blocks_for_load_str += "}."
            usage_str += blocks_for_save_str + blocks_for_load_str

        return usage_str
