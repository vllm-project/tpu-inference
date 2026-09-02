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
    install_hybrid_coordinator_hooks)


def _make_mock_hybrid_kv_cache_config(
    num_attn_blocks: int = 500,
    mamba_num_blocks: int = 50,
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
    cfg.mamba_num_blocks = mamba_num_blocks
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


class TestHybridCoordinatorHooks:

    def test_install_hooks(self):
        import vllm.v1.core.kv_cache_coordinator as coord_mod
        import vllm.v1.core.kv_cache_manager as mgr_mod

        install_hybrid_coordinator_hooks()
        assert mgr_mod.KVCacheManager is TPUKVCacheManager
        assert callable(coord_mod.get_kv_cache_coordinator)
