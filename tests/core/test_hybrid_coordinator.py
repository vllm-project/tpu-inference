# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock

import torch
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_utils import (BlockHash,
                                         make_block_hash_with_group_id)
from vllm.v1.kv_cache_interface import (FullAttentionSpec, KVCacheConfig,
                                        KVCacheGroupSpec, MambaSpec)
from vllm.v1.request import Request, RequestStatus

from tpu_inference.core.hybrid_coordinator import (
    TPUDualBlockPool, TPUHybridKVCacheCoordinator, TPUKVCacheManager,
    install_hybrid_coordinator_hooks, set_mamba_num_blocks)


def _make_mock_hybrid_kv_cache_config(
    num_attn_blocks: int = 500,
    mamba_num_blocks: int | None = 50,
    block_size: int = 16,
) -> KVCacheConfig:
    attn_spec = FullAttentionSpec(
        block_size=block_size,
        num_kv_heads=8,
        head_size=128,
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
    cfg = KVCacheConfig(
        num_blocks=num_attn_blocks,
        kv_cache_tensors=[],
        kv_cache_groups=groups,
    )
    if mamba_num_blocks is not None:
        set_mamba_num_blocks(mamba_num_blocks)
    return cfg


class TestTPUDualBlockPool:

    def test_routing_free_and_touch_to_origin_pools(self):
        pool_attn = BlockPool(num_gpu_blocks=100,
                              enable_caching=True,
                              hash_block_size=16)
        pool_mamba = BlockPool(num_gpu_blocks=20,
                               enable_caching=True,
                               hash_block_size=16)

        dual_pool = TPUDualBlockPool(pool_attn,
                                     pool_mamba,
                                     mamba_group_ids={1})

        # Allocate 2 from attn, 1 from mamba
        attn_blks = pool_attn.get_new_blocks(2)
        mamba_blks = pool_mamba.get_new_blocks(1)

        assert pool_attn.get_num_free_blocks() == 100 - 1 - 2  # 1 null block
        assert pool_mamba.get_num_free_blocks() == 20 - 1 - 1

        # Test touch via dual_pool
        dual_pool.touch([attn_blks[0], mamba_blks[0]])
        assert attn_blks[0].ref_cnt == 2
        assert mamba_blks[0].ref_cnt == 2

        # Reset ref_cnt back to 1 for clean free
        attn_blks[0].ref_cnt = 1
        mamba_blks[0].ref_cnt = 1

        # Free combined list via dual_pool
        mixed_blocks = [attn_blks[0], mamba_blks[0], attn_blks[1]]
        dual_pool.free_blocks(mixed_blocks)

        # Both pools should have all blocks returned
        assert pool_attn.get_num_free_blocks() == 100 - 1
        assert pool_mamba.get_num_free_blocks() == 20 - 1

    def test_reset_prefix_cache_clears_both(self):
        pool_attn = BlockPool(num_gpu_blocks=100,
                              enable_caching=True,
                              hash_block_size=16)
        pool_mamba = BlockPool(num_gpu_blocks=20,
                               enable_caching=True,
                               hash_block_size=16)
        dual_pool = TPUDualBlockPool(pool_attn,
                                     pool_mamba,
                                     mamba_group_ids={1})

        assert dual_pool.reset_prefix_cache() is True

    def test_get_cached_block_multi_group(self):
        pool_attn = BlockPool(num_gpu_blocks=100,
                              enable_caching=True,
                              hash_block_size=16)
        pool_mamba = BlockPool(num_gpu_blocks=20,
                               enable_caching=True,
                               hash_block_size=16)
        dual_pool = TPUDualBlockPool(pool_attn,
                                     pool_mamba,
                                     mamba_group_ids={1})

        pool_attn.get_cached_block = MagicMock()
        pool_mamba.get_cached_block = MagicMock()

        blk_attn = MagicMock()
        blk_mamba = MagicMock()

        # Both hit
        pool_attn.get_cached_block.return_value = [blk_attn]
        pool_mamba.get_cached_block.return_value = [blk_mamba]

        res = dual_pool.get_cached_block("hash1", [0, 1])
        assert res == [blk_attn, blk_mamba]
        pool_attn.get_cached_block.assert_called_with("hash1", [0])
        pool_mamba.get_cached_block.assert_called_with("hash1", [1])

        # Miss on mamba
        pool_mamba.get_cached_block.return_value = None
        assert dual_pool.get_cached_block("hash2", [0, 1]) is None

        # Miss on attn
        pool_attn.get_cached_block.return_value = None
        pool_mamba.get_cached_block.return_value = [blk_mamba]
        assert dual_pool.get_cached_block("hash3", [0, 1]) is None

    def test_evict_blocks_targets_attention_pool_only(self):
        pool_attn = BlockPool(num_gpu_blocks=100,
                              enable_caching=True,
                              hash_block_size=16)
        pool_mamba = BlockPool(num_gpu_blocks=20,
                               enable_caching=True,
                               hash_block_size=16)
        dual_pool = TPUDualBlockPool(pool_attn,
                                     pool_mamba,
                                     mamba_group_ids={1})

        pool_attn.evict_blocks = MagicMock()
        pool_mamba.evict_blocks = MagicMock()

        dual_pool.evict_blocks({5, 20, 200})
        pool_attn.evict_blocks.assert_called_once_with({5, 20})
        pool_mamba.evict_blocks.assert_not_called()


class TestTPUHybridKVCacheCoordinator:

    def test_decoupled_pool_initialization(self):
        cfg = _make_mock_hybrid_kv_cache_config(num_attn_blocks=500,
                                                mamba_num_blocks=50)
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

        assert coord.attention_block_pool.num_gpu_blocks == 500
        assert coord.mamba_block_pool.num_gpu_blocks == 50

        # Manager 0 (attn) points to attention pool
        assert coord.single_type_managers[
            0].block_pool is coord.attention_block_pool
        # Manager 1 (mamba) points to mamba pool
        assert coord.single_type_managers[
            1].block_pool is coord.mamba_block_pool

    def test_can_allocate_tokens_respects_both_pools(self):
        cfg = _make_mock_hybrid_kv_cache_config(num_attn_blocks=10,
                                                mamba_num_blocks=5)
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

        req = MagicMock(spec=Request)
        req.request_id = "req-1"
        req.num_computed_tokens = 0
        req.status = RequestStatus.RUNNING

        # Request 32 tokens (needs 2 attn blocks, 1 mamba block)
        can_fit = coord.can_allocate_tokens(
            request=req,
            num_tokens=32,
            new_computed_blocks=([], []),
            num_encoder_tokens=0,
            total_computed_tokens=0,
            num_local_computed_tokens=0,
            num_tokens_main_model=32,
            watermark_blocks=0,
            reserved_blocks=0,
        )
        assert can_fit is True

        # Now exhaust attention pool
        free_attn = coord.attention_block_pool.get_num_free_blocks()
        allocated_attn = coord.attention_block_pool.get_new_blocks(free_attn)
        can_fit_no_attn = coord.can_allocate_tokens(
            request=req,
            num_tokens=32,
            new_computed_blocks=([], []),
            num_encoder_tokens=0,
            total_computed_tokens=0,
            num_local_computed_tokens=0,
            num_tokens_main_model=32,
            watermark_blocks=0,
            reserved_blocks=0,
        )
        assert can_fit_no_attn is False

        # Free attention pool
        coord.attention_block_pool.free_blocks(allocated_attn)

        # Now exhaust mamba pool
        free_mamba = coord.mamba_block_pool.get_num_free_blocks()
        allocated_mamba = coord.mamba_block_pool.get_new_blocks(free_mamba)
        can_fit_no_mamba = coord.can_allocate_tokens(
            request=req,
            num_tokens=32,
            new_computed_blocks=([], []),
            num_encoder_tokens=0,
            total_computed_tokens=0,
            num_local_computed_tokens=0,
            num_tokens_main_model=32,
            watermark_blocks=0,
            reserved_blocks=0,
        )
        assert can_fit_no_mamba is False
        coord.mamba_block_pool.free_blocks(allocated_mamba)

    def test_find_longest_cache_hit_reconciles_minimum(self):
        cfg = _make_mock_hybrid_kv_cache_config(num_attn_blocks=500,
                                                mamba_num_blocks=50)
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

        hashes = [BlockHash(f"hash_{i}".encode("utf-8")) for i in range(10)]

        # Simulate: attn cached 5 blocks (80 tokens), mamba cached 3 blocks (48 tokens)
        # Attn pool caches hashes 0..4 (group_id = 0)
        attn_blocks = coord.attention_block_pool.get_new_blocks(5)
        for i, b in enumerate(attn_blocks):
            coord.attention_block_pool._insert_block_hash(
                make_block_hash_with_group_id(hashes[i], 0),
                b,
                num_tokens=(i + 1) * 16)

        # Mamba pool caches hashes 0..2 (group_id = 1)
        mamba_blocks = coord.mamba_block_pool.get_new_blocks(3)
        for i, b in enumerate(mamba_blocks):
            coord.mamba_block_pool._insert_block_hash(
                make_block_hash_with_group_id(hashes[i], 1),
                b,
                num_tokens=(i + 1) * 16)

        hit_blocks, hit_length, uncached = coord.find_longest_cache_hit(
            block_hashes=hashes,
            max_cache_hit_length=160,
        )

        # Reconciled hit length must be min(80, 48) = 48 tokens (3 blocks)
        assert hit_length == 48
        # Attention blocks must be truncated to 3 blocks
        assert len(hit_blocks[0]) == 3
        # Mamba block list has length 3 (with null blocks inserted before the match)
        assert len(hit_blocks[1]) == 3

        # Test find_longest_cache_hit_per_group inherited from upstream
        per_group_blocks, per_group_lengths = coord.find_longest_cache_hit_per_group(
            block_hashes=hashes,
            max_cache_hit_length=160,
        )
        assert per_group_lengths == (80, 48)
        assert len(per_group_blocks[0]) == 5
        assert len(per_group_blocks[1]) == 3


class TestHybridCoordinatorHooks:

    def test_install_hooks(self):
        import vllm.v1.core.kv_cache_coordinator as coord_mod
        import vllm.v1.core.kv_cache_manager as mgr_mod

        install_hybrid_coordinator_hooks()
        assert mgr_mod.KVCacheManager is TPUKVCacheManager
        assert callable(coord_mod.get_kv_cache_coordinator)

    def test_tpu_get_kv_cache_coordinator_resolves_from_global(self):
        from tpu_inference.core.hybrid_coordinator import (
            TPUHybridKVCacheCoordinator, set_mamba_num_blocks,
            tpu_get_kv_cache_coordinator)

        set_mamba_num_blocks(64)
        cfg = _make_mock_hybrid_kv_cache_config(num_attn_blocks=100,
                                                mamba_num_blocks=None)
        assert getattr(cfg, "mamba_num_blocks", None) is None

        coord_kwargs = dict(
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
        coord = tpu_get_kv_cache_coordinator(cfg, **coord_kwargs)
        assert isinstance(coord, TPUHybridKVCacheCoordinator)
        assert coord.mamba_num_blocks == 64

    def test_tpu_get_kv_cache_coordinator_raises_if_missing(self):
        import pytest

        import tpu_inference.core.hybrid_coordinator as hc_mod
        from tpu_inference.core.hybrid_coordinator import \
            tpu_get_kv_cache_coordinator
        hc_mod._GLOBAL_MAMBA_NUM_BLOCKS = None

        cfg = _make_mock_hybrid_kv_cache_config(num_attn_blocks=100,
                                                mamba_num_blocks=None)
        assert getattr(cfg, "mamba_num_blocks", None) is None

        coord_kwargs = dict(
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
        with pytest.raises(ValueError,
                           match="mamba_num_blocks must be registered"):
            tpu_get_kv_cache_coordinator(cfg, **coord_kwargs)


class TestTPUKVCacheManager:

    def test_allocate_slots_dual_pool_gating(self):
        set_mamba_num_blocks(5)
        cfg = _make_mock_hybrid_kv_cache_config(num_attn_blocks=10,
                                                mamba_num_blocks=5)
        manager = TPUKVCacheManager(
            kv_cache_config=cfg,
            max_model_len=1024,
            scheduler_block_size=16,
            hash_block_size=16,
            enable_caching=True,
        )
        assert isinstance(manager.coordinator, TPUHybridKVCacheCoordinator)

        req = Request(
            request_id="req-1",
            prompt_token_ids=list(range(32)),
            sampling_params=MagicMock(),
            pooling_params=None,
        )
        req.block_hashes = [BlockHash(f"h_{i}".encode()) for i in range(2)]
        req.status = RequestStatus.RUNNING

        # 1. Successful allocation
        blocks = manager.allocate_slots(request=req, num_new_tokens=32)
        assert blocks is not None
        assert len(blocks.blocks[0]) == 2  # 2 attn blocks
        assert len(blocks.blocks[1]) == 2  # 2 mamba blocks (1 null + 1 state)

        # 2. Attention pool exhaustion -> returns None for new request
        coord = manager.coordinator
        free_attn = coord.attention_block_pool.get_num_free_blocks()
        allocated_attn = coord.attention_block_pool.get_new_blocks(free_attn)
        req2 = Request(
            request_id="req-2",
            prompt_token_ids=list(range(16)),
            sampling_params=MagicMock(),
            pooling_params=None,
        )
        req2.block_hashes = [BlockHash(b"h_req2")]
        req2.status = RequestStatus.WAITING
        blocks_no_attn = manager.allocate_slots(request=req2,
                                                num_new_tokens=16)
        assert blocks_no_attn is None
        coord.attention_block_pool.free_blocks(allocated_attn)

        # 3. Mamba pool exhaustion -> returns None (prevents ValueError crash in get_new_blocks)
        free_mamba = coord.mamba_block_pool.get_num_free_blocks()
        allocated_mamba = coord.mamba_block_pool.get_new_blocks(free_mamba)
        req3 = Request(
            request_id="req-3",
            prompt_token_ids=list(range(16)),
            sampling_params=MagicMock(),
            pooling_params=None,
        )
        req3.block_hashes = [BlockHash(b"h_req3")]
        req3.status = RequestStatus.WAITING
        blocks_no_mamba = manager.allocate_slots(request=req3,
                                                 num_new_tokens=16)
        assert blocks_no_mamba is None
        coord.mamba_block_pool.free_blocks(allocated_mamba)
