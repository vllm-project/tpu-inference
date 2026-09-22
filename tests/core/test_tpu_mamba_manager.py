# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the tpu-inference project
"""Tests for TPUMambaManager.

The GDN kernel writes one checkpoint per forward pass, at
``(seq_len - 1) // block_size``, while standard vLLM prefix caching indexes
every full block boundary. TPUMambaManager restricts prefix cache registration
so that only blocks actually checkpointed by the GDN kernel are indexed in the
prefix cache (``cached_block_hash_to_block``).
"""

from unittest.mock import MagicMock

import pytest
import torch
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_utils import BlockHash, KVCacheBlock
from vllm.v1.kv_cache_interface import (FullAttentionSpec, KVCacheConfig,
                                        KVCacheGroupSpec, MambaSpec)
from vllm.v1.request import Request, RequestStatus

from tpu_inference.core.hybrid_coordinator import (
    TPUHybridKVCacheCoordinator, TPUMambaManager, set_mamba_num_blocks)

BLOCK_SIZE = 256


class FakeBlock:

    def __init__(self, block_id: int):
        self.block_id = block_id
        self.block_hash = None
        self.block_hash_num_tokens = None
        self.is_null = block_id == 0


class FakePool:
    """A minimal block pool matching BlockPool's cache interface."""

    def __init__(self):
        self.cached_block_hash_to_block: dict[BlockHash, FakeBlock] = {}
        self.null_block = FakeBlock(0)
        self.hash_block_size = BLOCK_SIZE

    def cache_full_blocks(
        self,
        request,
        blocks,
        num_cached_blocks,
        num_full_blocks,
        block_size,
        kv_cache_group_id,
        block_mask=None,
    ):
        for i in range(num_cached_blocks, num_full_blocks):
            if block_mask is not None and not block_mask[i - num_cached_blocks]:
                continue
            block = blocks[i]
            block_hash = BlockHash(f"hash_{i}".encode())
            block.block_hash = block_hash
            block.block_hash_num_tokens = (i + 1) * block_size
            self.cached_block_hash_to_block[block_hash] = block

    def get_cached_block(self, block_hash, kv_cache_group_ids):
        block = self.cached_block_hash_to_block.get(block_hash)
        return [block] * len(kv_cache_group_ids) if block else None


@pytest.fixture
def spec():
    s = MagicMock(spec=MambaSpec)
    s.block_size = BLOCK_SIZE
    s.mamba_cache_mode = "align"
    s.prefix_cacheable = True
    return s


def _make_manager(spec, pool):
    mgr = TPUMambaManager.__new__(TPUMambaManager)
    mgr.kv_cache_spec = spec
    mgr.block_size = spec.block_size
    mgr.mamba_cache_mode = "align"
    mgr.block_pool = pool
    mgr.req_to_blocks = {}
    mgr.num_cached_block = {}
    mgr.cached_blocks_this_step = set()
    mgr._pending_boundary_state_offloads = []
    mgr.kv_cache_group_id = 0
    mgr._checkpoints = {}
    mgr._producer_partial_tail_reqs = {}
    mgr._partial_hit_reqs = {}
    return mgr


def test_coordinator_instantiates_tpu_mamba_manager():
    attn_spec = FullAttentionSpec(
        block_size=16,
        num_kv_heads=8,
        head_size=128,
        dtype=torch.bfloat16,
    )
    mamba_spec = MambaSpec(
        shapes=((3, 64), (8, 64, 16)),
        dtypes=(torch.bfloat16, torch.float32),
        block_size=16,
        mamba_cache_mode="align",
    )
    groups = [
        KVCacheGroupSpec(kv_cache_spec=attn_spec, layer_names=["attn_0"]),
        KVCacheGroupSpec(kv_cache_spec=mamba_spec, layer_names=["mamba_0"]),
    ]
    set_mamba_num_blocks(50)
    cfg = KVCacheConfig(
        num_blocks=100,
        kv_cache_tensors=[],
        kv_cache_groups=groups,
    )
    coord = TPUHybridKVCacheCoordinator(
        kv_cache_config=cfg,
        max_model_len=1024,
        max_in_flight_tokens=128,
        use_eagle=False,
        enable_caching=True,
        enable_kv_cache_events=False,
        dcp_world_size=1,
        pcp_world_size=1,
        scheduler_block_size=16,
        hash_block_size=16,
    )
    assert isinstance(coord.single_type_managers[1], TPUMambaManager)


def test_tpu_mamba_manager_init_filters_unsupported_kwargs(spec):
    """Ensure TPUMambaManager drops args like max_in_flight_tokens and max_model_len
    for compatibility with older vLLM releases whose SingleTypeKVCacheManager.__init__
    does not accept them."""
    pool = FakePool()
    mgr = TPUMambaManager(
        kv_cache_spec=spec,
        block_pool=pool,
        enable_caching=True,
        kv_cache_group_id=0,
        scheduler_block_size=16,
        max_in_flight_tokens=128,
        max_model_len=2048,
    )
    assert mgr.kv_cache_spec is spec
    assert mgr.block_pool is pool


def test_cache_blocks_without_replay_boundaries(spec):
    """Calling cache_blocks without replay_boundaries (as standard vLLM coordinator does) succeeds."""
    pool = FakePool()
    mgr = _make_manager(spec, pool)
    blocks = [FakeBlock(100 + i) for i in range(8)]
    mgr.req_to_blocks["req1"] = blocks

    request = MagicMock()
    request.request_id = "req1"
    request.num_prompt_tokens = 2048

    mgr.cache_blocks(request, 2048)
    expected_hash = BlockHash(b"hash_7")
    assert list(pool.cached_block_hash_to_block.keys()) == [expected_hash]
    assert pool.cached_block_hash_to_block[expected_hash].block_id == 107


def test_cache_blocks_only_indexes_written_checkpoint(spec):
    """Chunk of 2048 tokens (8 blocks) must only index block 7 in the cache."""
    pool = FakePool()
    mgr = _make_manager(spec, pool)
    blocks = [FakeBlock(100 + i) for i in range(8)]
    mgr.req_to_blocks["req1"] = blocks

    request = MagicMock()
    request.request_id = "req1"
    request.num_prompt_tokens = 2048

    mgr.cache_blocks(request, 2048, replay_boundaries=[])

    # Only block 7 was checkpointed by the GDN kernel
    expected_hash = BlockHash(b"hash_7")
    assert list(pool.cached_block_hash_to_block.keys()) == [expected_hash]
    assert pool.cached_block_hash_to_block[expected_hash].block_id == 107
    # Intermediate blocks 0..6 must remain unindexed
    for i in range(7):
        assert blocks[i].block_hash is None


def test_cache_blocks_multiple_chunks(spec):
    """Chunked prefill across two passes must only index the end of each pass."""
    pool = FakePool()
    mgr = _make_manager(spec, pool)
    blocks = [FakeBlock(100 + i) for i in range(16)]
    mgr.req_to_blocks["req1"] = blocks

    request = MagicMock()
    request.request_id = "req1"
    request.num_prompt_tokens = 4096

    # Pass 1: first 2048 tokens -> checkpoints block 7
    mgr.cache_blocks(request, 2048, replay_boundaries=[])
    assert set(pool.cached_block_hash_to_block.keys()) == {BlockHash(b"hash_7")}

    # Pass 2: next 2048 tokens (total 4096) -> checkpoints block 15
    mgr.cache_blocks(request, 4096, replay_boundaries=[])
    assert set(pool.cached_block_hash_to_block.keys()) == {
        BlockHash(b"hash_7"), BlockHash(b"hash_15")
    }


def test_find_longest_cache_hit_rejects_unwritten_boundaries(spec):
    """Subsequent request of length 512 matches blocks 0 and 1, neither of which
    was checkpointed; cache hit must be 0 rather than returning dirty memory."""
    pool = FakePool()
    mgr = _make_manager(spec, pool)
    blocks = [FakeBlock(100 + i) for i in range(8)]
    mgr.req_to_blocks["req1"] = blocks

    request = MagicMock()
    request.request_id = "req1"
    request.num_prompt_tokens = 2048
    mgr.cache_blocks(request, 2048, replay_boundaries=[])

    # Request 2 with 512 prompt tokens
    hashes = [BlockHash(f"hash_{i}".encode()) for i in range(16)]
    computed, hit_length = TPUMambaManager.find_longest_cache_hit(
        block_hashes=hashes,
        max_length=512,
        kv_cache_group_ids=[0],
        block_pool=pool,
        kv_cache_spec=spec,
        drop_eagle_block=False,
        alignment_tokens=BLOCK_SIZE,
    )
    assert hit_length == 0
    assert computed[0] == []


def test_find_longest_cache_hit_matches_written_checkpoint(spec):
    """Subsequent request of length 2048 matches the checkpoint at block 7."""
    pool = FakePool()
    mgr = _make_manager(spec, pool)
    blocks = [FakeBlock(100 + i) for i in range(8)]
    mgr.req_to_blocks["req1"] = blocks

    request = MagicMock()
    request.request_id = "req1"
    request.num_prompt_tokens = 2048
    mgr.cache_blocks(request, 2048, replay_boundaries=[])

    hashes = [BlockHash(f"hash_{i}".encode()) for i in range(16)]
    computed, hit_length = TPUMambaManager.find_longest_cache_hit(
        block_hashes=hashes,
        max_length=2048,
        kv_cache_group_ids=[0],
        block_pool=pool,
        kv_cache_spec=spec,
        drop_eagle_block=False,
        alignment_tokens=BLOCK_SIZE,
    )
    assert hit_length == 2048
    assert computed[0][-1].block_id == 107


def test_find_longest_cache_hit_falls_back_to_earlier_written_checkpoint(spec):
    """Subsequent request of length 3000 (after 2048 and 4096 were written)
    falls back cleanly to token 2048 (block 7), without hitting unwritten
    blocks 8..10."""
    pool = FakePool()
    mgr = _make_manager(spec, pool)
    blocks = [FakeBlock(100 + i) for i in range(16)]
    mgr.req_to_blocks["req1"] = blocks

    request = MagicMock()
    request.request_id = "req1"
    request.num_prompt_tokens = 4096
    mgr.cache_blocks(request, 2048, replay_boundaries=[])
    mgr.cache_blocks(request, 4096, replay_boundaries=[])

    hashes = [BlockHash(f"hash_{i}".encode()) for i in range(16)]
    computed, hit_length = TPUMambaManager.find_longest_cache_hit(
        block_hashes=hashes,
        max_length=3000,
        kv_cache_group_ids=[0],
        block_pool=pool,
        kv_cache_spec=spec,
        drop_eagle_block=False,
        alignment_tokens=BLOCK_SIZE,
    )
    assert hit_length == 2048
    assert computed[0][-1].block_id == 107
