# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the tpu-inference project
"""The mamba written-boundary clamp.

The GDN kernel writes one checkpoint per forward pass, at
``(seq_len - 1) // block_size``, while ``cache_blocks`` registers every
boundary the retention mask keeps. These tests pin the gap: a hit may only
land on a block some pass actually checkpointed.
"""

from unittest.mock import MagicMock

import pytest
from vllm.v1.kv_cache_interface import MambaSpec

from tpu_inference.core.hybrid_coordinator import TPUMambaManager

BLOCK_SIZE = 256


class FakeBlock:

    def __init__(self, block_id: int):
        self.block_id = block_id
        self.is_null = block_id == 0


class FakePool:
    """A block pool whose cache is a plain {hash: block} map."""

    def __init__(self, cached: dict[int, int]):
        # block index -> block id
        self.cached = {i: FakeBlock(bid) for i, bid in cached.items()}
        self.null_block = FakeBlock(0)
        self.written_block_ids: set[int] = set()
        self.hash_block_size = BLOCK_SIZE

    def get_cached_block(self, block_hash, kv_cache_group_ids):
        block = self.cached.get(block_hash)
        return [block] * len(kv_cache_group_ids) if block else None


@pytest.fixture
def spec():
    # `find_longest_cache_hit` asserts on the spec type, and MagicMock(spec=)
    # satisfies isinstance.
    s = MagicMock(spec=MambaSpec)
    s.block_size = BLOCK_SIZE
    return s


def _hit(pool, spec, max_length, num_blocks=8):
    # `block_hashes[i]` is just `i`, so a block index doubles as its hash.
    return TPUMambaManager.find_longest_cache_hit(
        block_hashes=list(range(num_blocks)),
        max_length=max_length,
        kv_cache_group_ids=[0],
        block_pool=pool,
        kv_cache_spec=spec,
        drop_eagle_block=False,
        alignment_tokens=BLOCK_SIZE,
    )


def test_hit_on_written_boundary_is_kept(spec):
    pool = FakePool({5: 105})
    pool.written_block_ids.add(105)
    blocks, hit = _hit(pool, spec, 8 * BLOCK_SIZE)
    assert hit == 6 * BLOCK_SIZE
    assert blocks[0][-1].block_id == 105


def test_hit_on_unwritten_boundary_is_refused(spec):
    # Block 5 is advertised but no pass ever checkpointed it.
    pool = FakePool({5: 105})
    blocks, hit = _hit(pool, spec, 8 * BLOCK_SIZE)
    assert hit == 0
    assert blocks[0] == []


def test_falls_back_to_the_nearest_written_boundary(spec):
    # 2 and 5 advertised; only 2 written. Upstream would take 5.
    pool = FakePool({2: 102, 5: 105})
    pool.written_block_ids.add(102)
    blocks, hit = _hit(pool, spec, 8 * BLOCK_SIZE)
    assert hit == 3 * BLOCK_SIZE
    assert blocks[0][-1].block_id == 102


def test_walks_back_over_several_unwritten_boundaries(spec):
    pool = FakePool({1: 101, 4: 104, 5: 105, 6: 106})
    pool.written_block_ids.add(101)
    blocks, hit = _hit(pool, spec, 8 * BLOCK_SIZE)
    assert hit == 2 * BLOCK_SIZE
    assert blocks[0][-1].block_id == 101


def test_no_written_set_leaves_behaviour_unchanged(spec):
    # A pool without the attribute (e.g. a non-TPU pool) must not be clamped.
    pool = FakePool({5: 105})
    del pool.written_block_ids
    _, hit = _hit(pool, spec, 8 * BLOCK_SIZE)
    assert hit == 6 * BLOCK_SIZE


def test_cache_blocks_records_only_the_pass_end():
    """A chunk that ends at token 2048 checkpoints block 7 and nothing else,
    even though `cache_blocks` registered every boundary below it."""
    pool = FakePool({})
    mgr = TPUMambaManager.__new__(TPUMambaManager)
    mgr.mamba_cache_mode = "align"
    mgr.block_size = BLOCK_SIZE
    mgr.block_pool = pool
    mgr.req_to_blocks = {"r0": [FakeBlock(100 + i) for i in range(8)]}

    request = MagicMock()
    request.request_id = "r0"
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(TPUMambaManager, "cache_blocks",
                   TPUMambaManager.cache_blocks)
        # Skip the upstream body; only the recording half is under test.
        mp.setattr("vllm.v1.core.single_type_kv_cache_manager.MambaManager."
                   "cache_blocks", lambda *a, **k: None)
        TPUMambaManager.cache_blocks(mgr, request, 2048)

    assert pool.written_block_ids == {107}


def test_null_blocks_are_never_recorded_as_written():
    pool = FakePool({})
    mgr = TPUMambaManager.__new__(TPUMambaManager)
    mgr.mamba_cache_mode = "align"
    mgr.block_size = BLOCK_SIZE
    mgr.block_pool = pool
    mgr.req_to_blocks = {"r0": [FakeBlock(0)]}

    request = MagicMock()
    request.request_id = "r0"
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr("vllm.v1.core.single_type_kv_cache_manager.MambaManager."
                   "cache_blocks", lambda *a, **k: None)
        TPUMambaManager.cache_blocks(mgr, request, BLOCK_SIZE)

    assert pool.written_block_ids == set()
