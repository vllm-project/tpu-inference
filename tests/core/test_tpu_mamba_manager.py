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
from vllm.v1.core.kv_cache_utils import BlockHash
from vllm.v1.core.single_type_kv_cache_manager import MambaManager
from vllm.v1.kv_cache_interface import (FullAttentionSpec, KVCacheConfig,
                                        KVCacheGroupSpec, MambaSpec)
from vllm.v1.request import Request, RequestStatus

if not hasattr(MambaSpec, "prefix_cacheable"):
    MambaSpec.prefix_cacheable = True
if not hasattr(MambaManager, "_pending_boundary_state_offloads"):
    MambaManager._pending_boundary_state_offloads = []

from tpu_inference.core.hybrid_coordinator import (TPUHybridKVCacheCoordinator,
                                                   TPUKVCacheManager,
                                                   TPUMambaManager,
                                                   set_mamba_num_blocks)
from tpu_inference.core.sched.utils import \
    patch_vllm_scheduler_for_continue_decode

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
            if block_mask is not None and not block_mask[i -
                                                         num_cached_blocks]:
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
    assert set(
        pool.cached_block_hash_to_block.keys()) == {BlockHash(b"hash_7")}

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


def _make_hybrid_kv_cache_manager(
        block_size: int = 16,
        num_attn_blocks: int = 64,
        num_mamba_blocks: int = 32,
        max_model_len: int = 512) -> TPUKVCacheManager:
    attn_spec = FullAttentionSpec(
        block_size=block_size,
        num_kv_heads=4,
        head_size=64,
        dtype=torch.bfloat16,
    )
    mamba_spec = MambaSpec(
        shapes=((3, 64), (8, 64, 16)),
        dtypes=(torch.bfloat16, torch.float32),
        block_size=block_size,
        mamba_cache_mode="align",
    )
    groups = [
        KVCacheGroupSpec(kv_cache_spec=attn_spec, layer_names=["attn_0"]),
        KVCacheGroupSpec(kv_cache_spec=mamba_spec, layer_names=["mamba_0"]),
    ]
    set_mamba_num_blocks(num_mamba_blocks)
    cfg = KVCacheConfig(
        num_blocks=num_attn_blocks,
        kv_cache_tensors=[],
        kv_cache_groups=groups,
    )
    cfg.mamba_num_blocks = num_mamba_blocks
    from unittest.mock import patch
    with patch(
            "vllm.v1.core.kv_cache_manager.get_kv_cache_coordinator",
            side_effect=lambda *a, **kw: TPUHybridKVCacheCoordinator(*a, **kw),
    ):
        return TPUKVCacheManager(
            kv_cache_config=cfg,
            max_model_len=max_model_len,
            scheduler_block_size=block_size,
            enable_caching=True,
            use_eagle=False,
            log_stats=False,
            enable_kv_cache_events=False,
            dcp_world_size=1,
            pcp_world_size=1,
            hash_block_size=block_size,
        )


def _make_real_request(request_id: str,
                       prompt_token_ids: list[int],
                       block_size: int = 16,
                       max_tokens: int = 64) -> Request:
    import hashlib
    import pickle

    from vllm.sampling_params import SamplingParams
    from vllm.v1.core.kv_cache_utils import (get_request_block_hasher,
                                             init_none_hash)

    def hash_fn(x):
        return hashlib.sha256(pickle.dumps(x)).digest()

    init_none_hash(hash_fn)
    hasher = get_request_block_hasher(block_size, caching_hash_fn=hash_fn)
    req = Request(
        request_id=request_id,
        prompt_token_ids=list(prompt_token_ids),
        sampling_params=SamplingParams(max_tokens=max_tokens),
        pooling_params=None,
        block_hasher=hasher,
    )
    req.status = RequestStatus.RUNNING
    return req


def test_sync_continue_decode_mamba_align_boundary_cap_and_caching():
    """Verify TPUKVCacheManager + TPUMambaManager in sync continue_decode when
    TPUModelRunner._get_min_remaining_slots caps bursts at the mamba_block_size
    boundary:
    1. Prefill and decode ignore num_lookahead_tokens in MambaManager (only
       attention pre-allocates lookahead blocks).
    2. When a boundary-capped continue_decode burst stops at B * block_size
       (e.g. 14 -> 16 for block_size=16), block 0 is cached (written_block_idx=0)
       before the next step advances into block 1 and frees block 0.
    3. A subsequent request sharing the 16-token prefix hits block 0 in both
       attention and mamba pools.
    """
    block_size = 16
    kv_mgr = _make_hybrid_kv_cache_manager(block_size=block_size)
    mamba_mgr: TPUMambaManager = kv_mgr.coordinator.single_type_managers[1]

    # Prompt of 16 tokens (exact block boundary): with lookahead=9, prefill
    # only allocates 1 mamba block.
    req_aligned = _make_real_request("req_aligned",
                                     list(range(100, 116)),
                                     block_size=block_size)
    kv_mgr.allocate_slots(req_aligned,
                          num_new_tokens=16,
                          num_lookahead_tokens=9)
    assert len(mamba_mgr.req_to_blocks["req_aligned"]) == 1
    assert not mamba_mgr.req_to_blocks["req_aligned"][0].is_null
    kv_mgr.free(req_aligned)

    # 14-token prompt followed by a boundary-capped 2-token continue_decode
    # burst (seq_len 15, 16) stopping right at the block_size=16 boundary.
    prompt = list(range(1, 15))  # 14 tokens
    req1 = _make_real_request("req1", prompt, block_size=block_size)

    # Step 1: Prefill 14 tokens (with num_lookahead_tokens=9).
    kv_mgr.allocate_slots(req1, num_new_tokens=14, num_lookahead_tokens=9)
    assert len(mamba_mgr.req_to_blocks["req1"]) == 1
    b0 = mamba_mgr.req_to_blocks["req1"][0]
    assert not b0.is_null
    req1.num_computed_tokens = 14
    req1.append_output_token_ids(101)  # 15th token sampled after prefill

    # Step 2: Decode step 1 with continue_decode (lookahead=9 for attention).
    # num_tokens_main_model = 15 -> MambaManager keeps 1 mamba block (b0).
    # On-device _get_min_remaining_slots caps max_decode_steps at
    # 16 - ((15 - 1) % 16) = 2 steps (seq_len=15 and seq_len=16).
    kv_mgr.new_step_starts()
    kv_mgr.allocate_slots(req1, num_new_tokens=1, num_lookahead_tokens=9)
    assert len(mamba_mgr.req_to_blocks["req1"]) == 1
    assert mamba_mgr.req_to_blocks["req1"][0] is b0

    # Simulate continue_decode generating 2 tokens on-device ([102, 103]),
    # advancing num_computed_tokens to 16 (exact block boundary), and
    # patched_update_base caching blocks at the completed burst boundary.
    req1.num_computed_tokens = 15  # from _update_after_schedule
    req1.append_output_token_ids(102)
    req1.append_output_token_ids(103)
    req1.num_computed_tokens += 1  # from patched_update_base (total 16)
    kv_mgr.cache_blocks(req1, req1.num_computed_tokens)

    # Step 3: Next schedule() call at num_computed_tokens=16, num_new_tokens=1.
    # Standard MambaManager allocates block 1 (b1) while keeping b0 at
    # last_state_block_idx=0 for reading on step 3.
    kv_mgr.new_step_starts()
    kv_mgr.allocate_slots(req1, num_new_tokens=1, num_lookahead_tokens=9)
    assert len(mamba_mgr.req_to_blocks["req1"]) == 2
    assert mamba_mgr.req_to_blocks["req1"][0] is b0
    b1 = mamba_mgr.req_to_blocks["req1"][1]
    assert not b1.is_null and b1 is not b0
    assert b0.block_hash is not None

    # A new request sharing the first 16 tokens (14 prompt + [101, 102]) gets
    # a 16-token prefix cache hit in both attention and mamba groups!
    req2 = _make_real_request("req2",
                              prompt + [101, 102, 999, 1000],
                              block_size=block_size)
    computed_blocks, num_computed = kv_mgr.get_computed_blocks(req2)[:2]
    assert num_computed == 16
    assert computed_blocks.blocks[1][-1] is b0


def test_async_scheduler_and_continue_decode_mamba_align():
    """Verify AsyncScheduler lifecycle with TPUMambaManager in align mode when
    continue_decode is capped at max(1, mamba_remaining - 1) (plus stop-token
    trimming)."""
    from unittest import mock

    from vllm.v1.core.sched.async_scheduler import AsyncScheduler
    from vllm.v1.core.sched.scheduler import Scheduler

    originals = (Scheduler._update_request_with_output, Scheduler.__init__,
                 AsyncScheduler._update_request_with_output)
    try:
        patch_vllm_scheduler_for_continue_decode()

        block_size = 16
        kv_mgr = _make_hybrid_kv_cache_manager(block_size=block_size)
        mamba_mgr: TPUMambaManager = kv_mgr.coordinator.single_type_managers[1]

        scheduler = mock.MagicMock(spec=AsyncScheduler)
        scheduler.max_model_len = 512
        scheduler.__dict__["kv_cache_manager"] = kv_mgr

        # Prompt of 12 tokens.
        prompt = list(range(200, 212))
        req1 = _make_real_request("async_req1",
                                  prompt,
                                  block_size=block_size,
                                  max_tokens=6)

        # 1. Prefill (12 tokens): allocates b0.
        kv_mgr.allocate_slots(req1, num_new_tokens=12, num_lookahead_tokens=18)
        b0 = mamba_mgr.req_to_blocks["async_req1"][0]
        assert len(mamba_mgr.req_to_blocks["async_req1"]) == 1
        req1.num_computed_tokens = 12
        req1.num_in_flight_tokens = 12
        req1.num_output_placeholders = 1

        # 2. Schedule Decode Step 1 while prefill is in flight:
        # num_tokens_main_model = 13 -> still in b0.
        # On device: num_tokens=13 -> mamba_remaining=4 -> async cap = max(1, 4-1) = 3
        # steps (seq_len = 13, 14, 15, stopping 1 token before boundary 16).
        kv_mgr.new_step_starts()
        kv_mgr.allocate_slots(req1, num_new_tokens=1, num_lookahead_tokens=18)
        assert len(mamba_mgr.req_to_blocks["async_req1"]) == 1
        assert mamba_mgr.req_to_blocks["async_req1"][0] is b0
        req1.num_computed_tokens = 13
        req1.num_in_flight_tokens = 13
        req1.num_output_placeholders = 2

        # 3. Prefill output arrives (1 token [301]):
        req1.num_in_flight_tokens -= 12
        AsyncScheduler._update_request_with_output(scheduler,
                                                   req1, [301],
                                                   is_stale=False)
        assert req1.num_output_placeholders == 1
        assert req1.num_computed_tokens == 13
        assert req1.num_tokens == 13

        # 4. Schedule Decode Step 2 while Decode Step 1 is in flight:
        # Host still sees num_computed_tokens=13 -> num_tokens_main_model=14 -> b0.
        # On device: runner's input_batch.num_tokens was advanced by 3 to 16 ->
        # mamba_remaining=1 -> async cap = max(1, 1-1) = 1 step (executes the
        # single boundary token seq_len=16 into b0!).
        kv_mgr.new_step_starts()
        kv_mgr.allocate_slots(req1, num_new_tokens=1, num_lookahead_tokens=18)
        assert len(mamba_mgr.req_to_blocks["async_req1"]) == 1
        assert mamba_mgr.req_to_blocks["async_req1"][0] is b0
        req1.num_computed_tokens = 14
        req1.num_in_flight_tokens += 1
        req1.num_output_placeholders += 1  # now 2

        # 5. Decode Step 1 output arrives (3 tokens [302, 303, 304]):
        req1.num_in_flight_tokens -= 1
        res_tokens, stopped = AsyncScheduler._update_request_with_output(
            scheduler, req1, [302, 303, 304], is_stale=False)
        assert stopped is False
        assert res_tokens == [302, 303, 304]
        assert req1.num_output_placeholders == 1
        assert req1.num_computed_tokens == 16
        assert req1.num_tokens == 16
        # Step 2 (at seq_len=16) is still in flight (num_computed_tokens - placeholders = 15),
        # so b0 is not yet cached from _update_request_with_output.
        assert b0.block_hash is None

        # 6. Schedule Decode Step 3 while Decode Step 2 is in flight:
        # Now host has num_computed_tokens=16 -> num_tokens_main_model=17!
        # Standard MambaManager allocates b1 (block 1) without needing any lookahead!
        kv_mgr.new_step_starts()
        kv_mgr.allocate_slots(req1, num_new_tokens=1, num_lookahead_tokens=18)
        assert len(mamba_mgr.req_to_blocks["async_req1"]) == 2
        assert mamba_mgr.req_to_blocks["async_req1"][0] is b0
        b1 = mamba_mgr.req_to_blocks["async_req1"][1]
        assert not b1.is_null and b1 is not b0
        req1.num_computed_tokens = 17
        req1.num_in_flight_tokens += 1
        req1.num_output_placeholders += 1  # now 2

        # 7. Decode Step 2 output arrives (1 boundary token [305]), and then
        # Decode Step 3 output arrives with 3 tokens [306, 307, 308] which gets
        # trimmed by max_tokens=6 to [306]:
        req1.num_in_flight_tokens -= 1
        res_tokens2, stopped2 = AsyncScheduler._update_request_with_output(
            scheduler, req1, [305], is_stale=False)
        assert stopped2 is False
        assert res_tokens2 == [305]
        assert req1.num_output_placeholders == 1
        # Block 0 (covering tokens 0..15) is now cached!
        assert b0.block_hash is not None

        req1.num_in_flight_tokens -= 1
        res_tokens3, stopped3 = AsyncScheduler._update_request_with_output(
            scheduler, req1, [306, 307, 308], is_stale=False)
        assert stopped3 is True
        assert res_tokens3 == [306]
        assert req1.num_output_placeholders == 0

        # A second request sharing those 16 tokens (12 prompt + [301, 302, 303, 304])
        # hits block 0 in the prefix cache on the next step.
        kv_mgr.new_step_starts()
        req2 = _make_real_request("async_req2",
                                  prompt + [301, 302, 303, 304, 400, 401],
                                  block_size=block_size)
        computed_blocks, num_computed = kv_mgr.get_computed_blocks(req2)[:2]
        assert num_computed == 16
        assert computed_blocks.blocks[1][-1] is b0
    finally:
        (Scheduler._update_request_with_output, Scheduler.__init__,
         AsyncScheduler._update_request_with_output) = originals
        for cls in (Scheduler, AsyncScheduler):
            if "_continue_decode_patched" in cls.__dict__:
                del cls._continue_decode_patched


def test_sync_scheduler_update_base_caches_and_clamps_in_flight():
    """Verify patched Scheduler._update_request_with_output in sync mode:
    1. Clamps negative request.num_in_flight_tokens underflow (caused by
       _process_continue_decode_outputs rewriting num_scheduled_tokens to actual_len)
       back to 0.
    2. Calls kv_cache_manager.cache_blocks immediately when a boundary-capped
       continue_decode burst finishes a block at B * block_size, even when the
       request stops on that step.
    """
    from unittest import mock

    from vllm.v1.core.sched.async_scheduler import AsyncScheduler
    from vllm.v1.core.sched.scheduler import Scheduler

    originals = (Scheduler._update_request_with_output, Scheduler.__init__,
                 AsyncScheduler._update_request_with_output)
    try:
        patch_vllm_scheduler_for_continue_decode()

        block_size = 16
        kv_mgr = _make_hybrid_kv_cache_manager(block_size=block_size)
        mamba_mgr: TPUMambaManager = kv_mgr.coordinator.single_type_managers[1]

        scheduler = mock.MagicMock(spec=Scheduler)
        scheduler.max_model_len = 512
        scheduler.__dict__["kv_cache_manager"] = kv_mgr

        prompt = list(range(50, 63))  # 13 tokens
        req1 = _make_real_request("sync_stop_req1",
                                  prompt,
                                  block_size=block_size,
                                  max_tokens=4)

        # Prefill 13 tokens -> 1st output token [701] (14 tokens total)
        kv_mgr.allocate_slots(req1, num_new_tokens=13, num_lookahead_tokens=9)
        b0 = mamba_mgr.req_to_blocks["sync_stop_req1"][0]
        req1.num_computed_tokens = 13
        req1.append_output_token_ids(701)

        # Decode step with continue_decode capped at boundary 16 (2 steps: seq_len 14, 15 -> wait:
        # 13 prompt + 1 output = 14 tokens, so num_computed_tokens becomes 14 on schedule(),
        # and 2 steps on-device brings num_computed_tokens to 16!):
        kv_mgr.new_step_starts()
        kv_mgr.allocate_slots(req1, num_new_tokens=1, num_lookahead_tokens=9)
        req1.num_computed_tokens = 14
        # Simulate update_from_output subtracting actual_len=3 from num_in_flight_tokens=1:
        req1.num_in_flight_tokens = -2

        res_tokens, stopped = Scheduler._update_request_with_output(
            scheduler, req1, [702, 703, 704], is_stale=False)
        assert stopped is True
        assert res_tokens == [702, 703, 704]
        assert req1.num_in_flight_tokens == 0
        assert req1.num_computed_tokens == 16
        # Block 0 (covering tokens 0..15) was cached at num_computed_tokens=16
        # before _free_request runs!
        assert b0.block_hash is not None
        kv_mgr.free(req1)

        # Subsequent request hits block 0 in the prefix cache!
        kv_mgr.new_step_starts()
        req2 = _make_real_request("sync_stop_req2",
                                  prompt + [701, 702, 703, 800, 801],
                                  block_size=block_size)
        computed_blocks, num_computed = kv_mgr.get_computed_blocks(req2)[:2]
        assert num_computed == 16
        assert computed_blocks.blocks[1][-1] is b0
    finally:
        (Scheduler._update_request_with_output, Scheduler.__init__,
         AsyncScheduler._update_request_with_output) = originals
        for cls in (Scheduler, AsyncScheduler):
            if "_continue_decode_patched" in cls.__dict__:
                del cls._continue_decode_patched
