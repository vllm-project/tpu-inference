# SPDX-License-Identifier: Apache-2.0
import pytest

from tpu_inference.logger import init_logger
from tpu_inference.offload.offload_manager import (CPUChunkPool,
                                                   LRUCacheManager,
                                                   StagingBufferManager)
from tpu_inference.offload.utils import ReqId

logger = init_logger(__name__)


class TestStagingBufferManager:

    def test_initialization(self):
        manager = StagingBufferManager(num_blocks=100)
        assert manager.num_blocks == 100
        assert manager.get_num_free_staging_blocks() == 100
        assert manager.get_num_used_staging_blocks() == 0

    def test_allocate_simple(self):
        manager = StagingBufferManager(num_blocks=100)
        req_id1: ReqId = "req1"
        req_id2: ReqId = "req2"

        allocated1 = manager.allocate(req_id1, 10, "load")
        assert allocated1 == 10
        assert manager.get_num_free_staging_blocks() == 90
        assert manager.get_num_used_staging_blocks() == 10
        assert manager._num_blocks_for_load == 10
        assert manager._num_blocks_for_save == 0

        allocated2 = manager.allocate(req_id2, 20, "save")
        assert allocated2 == 20
        assert manager.get_num_free_staging_blocks() == 70
        assert manager.get_num_used_staging_blocks() == 30
        assert manager._num_blocks_for_load == 10
        assert manager._num_blocks_for_save == 20

    def test_allocate_insufficient_capacity(self):
        manager = StagingBufferManager(num_blocks=10)
        req_id: ReqId = "req1"
        allocated = manager.allocate(req_id, 20, "load")
        assert allocated == 0
        assert manager.get_num_free_staging_blocks() == 10
        assert manager.get_num_used_staging_blocks() == 0

    def test_allocate_existing_load_request(self):
        manager = StagingBufferManager(num_blocks=100)
        req_id: ReqId = "req1"
        manager.allocate(req_id, 10, "load")
        with pytest.raises(ValueError):
            # multiple concurrent loads from a single request is not allowed.
            manager.allocate(req_id, 5, "load")

    def test_allocate_existing_save_request(self):
        manager = StagingBufferManager(num_blocks=100)
        req_id: ReqId = "req1"
        manager.allocate(req_id, 10, "save")
        assert manager._blocks_for_save[req_id] == 10
        manager.allocate(req_id, 5, "save")
        assert manager._blocks_for_save[req_id] == 15
        assert manager.get_num_free_staging_blocks() == 85
        assert manager.get_num_used_staging_blocks() == 15

    def test_allocate_negative_blocks(self):
        manager = StagingBufferManager(num_blocks=100)
        req_id: ReqId = "req1"
        allocated = manager.allocate(req_id, -5, "load")
        assert allocated == -5
        assert manager.get_num_free_staging_blocks() == 100

    def test_free_full(self):
        manager = StagingBufferManager(num_blocks=100)
        req_id: ReqId = "req1"
        manager.allocate(req_id, 10, "load")
        freed = manager.free(req_id, "load")
        assert freed == 10
        assert manager.get_num_free_staging_blocks() == 100
        assert manager.get_num_used_staging_blocks() == 0
        assert req_id not in manager._blocks_for_load

    def test_free_partial(self):
        manager = StagingBufferManager(num_blocks=100)
        req_id: ReqId = "req1"
        manager.allocate(req_id, 10, "save")
        freed = manager.free(req_id, "save", num_finished_blocks=4)
        assert freed == 4
        assert manager.get_num_free_staging_blocks() == 94
        assert manager.get_num_used_staging_blocks() == 6
        assert manager._blocks_for_save[req_id] == 6

    def test_free_more_than_allocated(self):
        manager = StagingBufferManager(num_blocks=100)
        req_id: ReqId = "req1"
        manager.allocate(req_id, 10, "load")
        manager.free(req_id, "load", num_finished_blocks=15)
        assert req_id not in manager._blocks_for_load

    def test_free_non_existent_request(self):
        manager = StagingBufferManager(num_blocks=100)
        req_id: ReqId = "req1"
        freed = manager.free(req_id, "load")
        assert freed == 0

    def test_complex_scenario(self):
        manager = StagingBufferManager(num_blocks=50)
        req1, req2, req3 = "req1", "req2", "req3"

        # req1 loads 10, req2 saves 15
        assert manager.allocate(req1, 10, "load") == 10
        assert manager.allocate(req2, 15, "save") == 15
        assert manager.get_num_free_staging_blocks() == 25
        assert manager.get_num_used_staging_blocks() == 25

        # req3 tries to load 30, fails
        assert manager.allocate(req3, 30, "load") == 0
        assert manager.get_num_free_staging_blocks() == 25

        # req1 finishes loading
        assert manager.free(req1, "load") == 10
        assert manager.get_num_free_staging_blocks() == 35

        # req3 can now load 20
        assert manager.allocate(req3, 20, "load") == 20
        assert manager.get_num_free_staging_blocks() == 15
        assert manager.get_num_used_staging_blocks(
        ) == 35  # 15 for save (req2) + 20 for load (req3)

        # req2 saves another 5
        assert manager.allocate(req2, 5, "save") == 5
        assert manager.get_num_free_staging_blocks() == 10
        assert manager._blocks_for_save[req2] == 20

        # req2 frees 8 blocks
        assert manager.free(req2, "save", 8) == 8
        assert manager.get_num_free_staging_blocks() == 18
        assert manager._blocks_for_save[req2] == 12

        # req2 and req3 finish
        assert manager.free(req2, "save") == 12
        assert manager.free(req3, "load") == 20
        assert manager.get_num_free_staging_blocks() == 50
        assert manager.get_num_used_staging_blocks() == 0


class TestCPUChunkPool:

    def test_initialization(self):
        pool = CPUChunkPool(num_chunks=10)
        assert pool.num_chunks == 10
        assert pool.num_free_chunks == 10
        assert pool.num_allocated_chunks == 0
        assert len(pool.free_chunk_list) == 10

    def test_allocate_chunks(self):
        pool = CPUChunkPool(num_chunks=10)
        chunk_hashes = [101, 102, 103]
        chunks = pool.allocate_chunks(chunk_hashes)

        assert len(chunks) == 3
        assert pool.num_free_chunks == 7
        assert pool.num_allocated_chunks == 3
        for i, chunk in enumerate(chunks):
            assert chunk.chunk_hash == chunk_hashes[i]
            assert chunk.chunk_id in pool.allocated_id_to_hash_map

    def test_allocate_chunks_insufficient_space(self):
        pool = CPUChunkPool(num_chunks=2)
        chunk_hashes = [101, 102, 103]
        with pytest.raises(ValueError):
            pool.allocate_chunks(chunk_hashes)

    def test_release_chunks(self):
        pool = CPUChunkPool(num_chunks=10)
        chunk_hashes = [101, 102, 103]
        chunks = pool.allocate_chunks(chunk_hashes)
        for chunk in chunks:
            chunk.touch()

        for chunk in chunks:
            pool.release_chunk(chunk)

        assert pool.num_free_chunks == 10
        assert pool.num_allocated_chunks == 0
        assert len(pool.free_chunk_list) == 10
        for chunk in chunks:
            assert chunk.chunk_id not in pool.allocated_id_to_hash_map
            assert chunk.chunk_hash is None
            assert chunk.ref_cnt == -1

    def test_release_chunks_in_use(self):
        pool = CPUChunkPool(num_chunks=10)
        chunk_hashes = [101]
        chunks = pool.allocate_chunks(chunk_hashes)
        chunks[0].touch()  # ref_cnt = 0: saved
        chunks[0].touch()  # ref_cnt = 1: loading

        assert not pool.release_chunk(chunks[0])


class TestLRUCacheManager:

    def test_initialization(self):
        manager = LRUCacheManager(num_cpu_chunks=20)
        assert manager.num_chunks == 20
        assert isinstance(manager.chunk_pool, CPUChunkPool)
        assert len(manager.cpu_cache) == 0

    def test_lookup(self):
        manager = LRUCacheManager(num_cpu_chunks=20)
        chunk_hashes = [101, 102, 103]

        # 1. Cache miss
        assert manager.lookup(chunk_hashes) == 0

        # 2. Cache hit
        # Manually add to cache for testing
        chunks = manager.chunk_pool.allocate_chunks(chunk_hashes)
        for chunk, h in zip(chunks, chunk_hashes):
            chunk.touch()  # Make it ready to load
            manager.cpu_cache[h] = chunk

        assert manager.lookup(chunk_hashes) == 3

        # 3. Partial hit
        assert manager.lookup([101, 102, 104]) == 2

    def test_touch(self):
        manager = LRUCacheManager(num_cpu_chunks=3)
        chunk_hashes = [101, 102, 103]
        chunks = manager.chunk_pool.allocate_chunks(chunk_hashes)
        for chunk, h in zip(chunks, chunk_hashes):
            manager.cpu_cache[h] = chunk

        manager.touch([101])
        assert list(manager.cpu_cache.keys()) == [102, 103, 101]

        manager.touch([102, 103])
        assert list(manager.cpu_cache.keys()) == [101, 103, 102]

    def test_allocate_for_save_simple(self):
        manager = LRUCacheManager(num_cpu_chunks=5)
        chunk_hashes = [101, 102]

        new_chunks, new_chunk_idxs = manager.allocate_for_save(chunk_hashes)

        assert len(new_chunks) == 2
        assert new_chunk_idxs == [0, 1]
        assert manager.chunk_pool.num_free_chunks == 3
        assert len(manager.cpu_cache) == 2

    def test_allocate_for_save_no_new_chunks(self):
        manager = LRUCacheManager(num_cpu_chunks=5)
        chunk_hashes = [101, 102]
        manager.allocate_for_save(chunk_hashes)

        result = manager.allocate_for_save(chunk_hashes)
        assert result is None

    def test_allocate_for_save_with_eviction(self):
        manager = LRUCacheManager(num_cpu_chunks=2)
        # Fill the cache
        manager.allocate_for_save([101, 102])
        # Mark as evictable
        manager.cpu_cache[101].touch()
        manager.cpu_cache[102].touch()

        manager.touch([101, 102])

        # This should evict 102
        new_chunks, new_chunk_idxs = manager.allocate_for_save([103])

        assert len(new_chunks) == 1
        assert new_chunk_idxs == [0]
        assert 102 not in manager.cpu_cache
        assert 101 in manager.cpu_cache
        assert 103 in manager.cpu_cache
        assert manager.chunk_pool.num_free_chunks == 0

    def test_allocate_for_save_cannot_evict(self):
        manager = LRUCacheManager(num_cpu_chunks=2)
        manager.allocate_for_save([101, 102])
        # Mark as in use, not evictable
        manager.cpu_cache[101].touch()
        manager.cpu_cache[101].touch()
        manager.cpu_cache[102].touch()
        manager.cpu_cache[102].touch()

        result = manager.allocate_for_save([103])
        assert result is None
        assert len(manager.cpu_cache) == 2

    def test_prepare_load(self):
        manager = LRUCacheManager(num_cpu_chunks=2)
        chunk_hashes = [101]
        manager.allocate_for_save(chunk_hashes)
        manager.complete_save(chunk_hashes)  # ref_cnt = 0

        chunks = manager.prepare_load(chunk_hashes)
        assert len(chunks) == 1
        assert chunks[0].is_in_use  # ref_cnt = 1

    def test_complete_save(self):
        manager = LRUCacheManager(num_cpu_chunks=2)
        chunk_hashes = [101]
        manager.allocate_for_save(chunk_hashes)

        chunk = manager.cpu_cache[101]
        assert not chunk.is_ready_to_load  # ref_cnt = -1

        manager.complete_save(chunk_hashes)
        assert chunk.is_ready_to_load  # ref_cnt = 0

    def test_complete_load(self):
        manager = LRUCacheManager(num_cpu_chunks=2)
        chunk_hashes = [101]
        manager.allocate_for_save(chunk_hashes)
        manager.complete_save(chunk_hashes)
        chunks = manager.prepare_load(chunk_hashes)

        assert chunks[0].is_in_use  # ref_cnt = 1
        manager.complete_load(chunk_hashes)
        assert not chunks[0].is_in_use  # ref_cnt = 0

    def test_mark_completion(self):
        manager = LRUCacheManager(num_cpu_chunks=2)
        chunk_hashes = [101]
        new_chunks, _ = manager.allocate_for_save(chunk_hashes)
        chunk_ids = [c.chunk_id for c in new_chunks]

        manager.mark_completion(chunk_ids, 'save')
        assert manager.cpu_cache[101].is_ready_to_load

        manager.prepare_load(chunk_hashes)
        assert manager.cpu_cache[101].is_in_use
        manager.mark_completion(chunk_ids, 'load')
        assert not manager.cpu_cache[101].is_in_use

    def test_mark_completion_unknown_id(self):
        manager = LRUCacheManager(num_cpu_chunks=2)
        with pytest.raises(ValueError):
            manager.mark_completion([999], 'save')
