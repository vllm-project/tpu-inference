# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock

import pytest
import torch
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_utils import (BlockHash,
                                         make_block_hash_with_group_id)
from vllm.v1.kv_cache_interface import (FullAttentionSpec, KVCacheConfig,
                                        KVCacheGroupSpec, MambaSpec)
from vllm.v1.request import Request, RequestStatus

from tpu_inference.core.hybrid_coordinator import (
    MambaBlockPool, MirrorMambaBlockPool, TPUDualBlockPool,
    TPUHybridKVCacheCoordinator, TPUKVCacheManager, TPUMambaManager,
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


def _make_mock_vllm_config(dp_size: int, max_num_seqs: int) -> MagicMock:
    vllm_config = MagicMock()
    vllm_config.sharding_config.total_dp_size = dp_size
    vllm_config.scheduler_config.max_num_seqs = max_num_seqs
    vllm_config.speculative_config = None
    vllm_config.additional_config = {}
    return vllm_config


_COORD_KWARGS = dict(
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


class TestMirrorMambaBlockPool:
    """Mirroring makes G mamba groups cost one block id, not G.

    A pool of N ids otherwise holds N/G checkpoints while the memory behind it
    holds N: every mamba layer has N slots and a checkpoint needs one per
    layer.
    """

    def _pools(self, num_gpu_blocks=12, primary_group_id=1):
        primary = MambaBlockPool(num_gpu_blocks=num_gpu_blocks,
                                 enable_caching=True,
                                 hash_block_size=16,
                                 primary_group_id=primary_group_id)
        return primary, MirrorMambaBlockPool(primary)

    def test_mirror_reuses_the_primary_block_ids(self):
        primary, mirror = self._pools()

        first = primary.get_new_blocks(2)
        second = mirror.get_new_blocks(2)
        third = mirror.get_new_blocks(2)

        assert [b.block_id for b in second] == [b.block_id for b in first]
        assert [b.block_id for b in third] == [b.block_id for b in first]
        # Three groups genuinely reference each block, so the count must show
        # it or the first free would return a block another group still uses.
        assert all(b.ref_cnt == 3 for b in first)
        # Only 2 ids were consumed, not 6.
        assert primary.get_num_free_blocks() == 12 - 1 - 2

    def test_mirror_refuses_to_alias_when_groups_diverge(self):
        primary, mirror = self._pools()
        primary.get_new_blocks(2)

        # A different count means the groups are no longer in lockstep;
        # aliasing here would point a layer at an unrelated request's slot.
        with pytest.raises(AssertionError, match="lockstep"):
            mirror.get_new_blocks(3)

    def test_lookup_of_every_group_resolves_to_one_entry(self):
        primary, _ = self._pools(primary_group_id=1)

        req = Request(request_id="r0",
                      prompt_token_ids=list(range(16)),
                      sampling_params=MagicMock(),
                      pooling_params=None)
        req.block_hashes = [BlockHash(b"h0")]
        blocks = primary.get_new_blocks(1)
        primary.cache_full_blocks(request=req,
                                  blocks=blocks,
                                  num_cached_blocks=0,
                                  num_full_blocks=1,
                                  block_size=16,
                                  kv_cache_group_id=1)

        # One cached entry must satisfy a lookup spanning all three groups,
        # returning that block once per group.
        hit = primary.get_cached_block(BlockHash(b"h0"), [1, 2, 3])
        assert hit is not None
        assert len(hit) == 3
        assert {b.block_id for b in hit} == {blocks[0].block_id}
        assert len(primary.cached_block_hash_to_block) == 1

    def test_mirror_does_not_double_cache(self):
        primary, mirror = self._pools()
        req = Request(request_id="r0",
                      prompt_token_ids=list(range(16)),
                      sampling_params=MagicMock(),
                      pooling_params=None)
        req.block_hashes = [BlockHash(b"h0")]
        blocks = primary.get_new_blocks(1)

        mirror.cache_full_blocks(request=req,
                                 blocks=blocks,
                                 num_cached_blocks=0,
                                 num_full_blocks=1,
                                 block_size=16,
                                 kv_cache_group_id=2)

        assert len(primary.cached_block_hash_to_block) == 0

    def test_mirror_forwards_the_rest_of_the_pool_api(self):
        primary, mirror = self._pools()
        assert mirror.num_gpu_blocks == primary.num_gpu_blocks
        assert mirror.null_block is primary.null_block
        assert mirror.hash_block_size == primary.hash_block_size

        blocks = primary.get_new_blocks(1)
        mirror.get_new_blocks(1)  # second reference
        mirror.free_blocks(blocks)
        assert blocks[0].ref_cnt == 1

    def test_no_canonicalisation_without_a_primary_group(self):
        # A model whose mamba layers already form one group has nothing to
        # mirror, so the pool keeps vLLM's per-group keying: a lookup spanning
        # three group ids then needs three cached entries.
        pool = MambaBlockPool(num_gpu_blocks=12,
                              enable_caching=True,
                              hash_block_size=16)
        req = Request(request_id="r0",
                      prompt_token_ids=list(range(16)),
                      sampling_params=MagicMock(),
                      pooling_params=None)
        req.block_hashes = [BlockHash(b"h0")]
        blocks = pool.get_new_blocks(1)
        pool.cache_full_blocks(request=req,
                               blocks=blocks,
                               num_cached_blocks=0,
                               num_full_blocks=1,
                               block_size=16,
                               kv_cache_group_id=1)

        assert pool.get_cached_block(BlockHash(b"h0"), [1]) is not None
        assert pool.get_cached_block(BlockHash(b"h0"), [1, 2, 3]) is None


class TestTPUHybridKVCacheCoordinator:

    def _make_multi_mamba_group_config(self, num_mamba_groups=3):
        """The Qwen3.5 shape: vLLM splits the mamba layers across groups."""
        attn_spec = FullAttentionSpec(block_size=16,
                                      num_kv_heads=8,
                                      head_size=128,
                                      dtype=torch.bfloat16)
        mamba_spec = MambaSpec(shapes=((3, 64), (8, 64, 16)),
                               dtypes=(torch.bfloat16, torch.float32),
                               block_size=16,
                               mamba_cache_mode="align")
        groups = [KVCacheGroupSpec(["attn_0"], attn_spec)]
        groups += [
            KVCacheGroupSpec([f"mamba_{i}"], mamba_spec)
            for i in range(num_mamba_groups)
        ]
        set_mamba_num_blocks(50)
        return KVCacheConfig(num_blocks=500,
                             kv_cache_tensors=[],
                             kv_cache_groups=groups)

    def _make_coordinator(self, cfg):
        return TPUHybridKVCacheCoordinator(
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

    def test_multiple_mamba_groups_are_mirrored_by_default(self):
        coord = self._make_coordinator(self._make_multi_mamba_group_config())

        assert coord.mirror_mamba_groups is True
        assert coord.primary_mamba_group_id == 1  # group 0 is attention
        assert coord.mamba_block_pool.primary_group_id == 1
        # Only the primary group owns the pool; the rest are handed its ids.
        assert coord.single_type_managers[
            1].block_pool is coord.mamba_block_pool
        for i in (2, 3):
            assert isinstance(coord.single_type_managers[i].block_pool,
                              MirrorMambaBlockPool)

    def test_single_mamba_group_is_not_mirrored(self):
        coord = self._make_coordinator(
            self._make_multi_mamba_group_config(num_mamba_groups=1))

        # Nothing to share ids with, so keep vLLM's per-group keying.
        assert coord.mirror_mamba_groups is False
        assert coord.mamba_block_pool.primary_group_id is None
        assert coord.single_type_managers[
            1].block_pool is coord.mamba_block_pool

    def test_mirrored_groups_do_not_multiply_the_capacity_check(self):
        coord = self._make_coordinator(self._make_multi_mamba_group_config())
        req = Request(request_id="r0",
                      prompt_token_ids=list(range(32)),
                      sampling_params=MagicMock(),
                      pooling_params=None)
        req.block_hashes = [BlockHash(f"h{i}".encode()) for i in range(2)]

        # 3 mamba groups asking for the same blocks is one block's worth of
        # demand, not three: counting each would reject requests that fit.
        empty = tuple([] for _ in coord.kv_cache_config.kv_cache_groups)
        assert coord.can_allocate_tokens(
            request=req,
            num_tokens=32,
            new_computed_blocks=empty,
            num_encoder_tokens=0,
            total_computed_tokens=0,
            num_local_computed_tokens=0,
            num_tokens_main_model=32,
        ) is True

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

    def test_mamba_blocks_per_request(self):
        """One resident slot per request outside align mode; in align mode a
        second resident slot plus `custom_mamba_cache_multiplier` (default 8)
        checkpoint slots, never fewer than the resident minimum."""
        from tpu_inference.core.hybrid_coordinator import (
            DEFAULT_MAMBA_CACHE_MULTIPLIER, mamba_blocks_per_request)

        vllm_config = _make_mock_vllm_config(dp_size=2, max_num_seqs=8)
        assert mamba_blocks_per_request(
            vllm_config,
            is_align_mode=True) == (2, DEFAULT_MAMBA_CACHE_MULTIPLIER)
        assert mamba_blocks_per_request(vllm_config,
                                        is_align_mode=False) == (1, 1)

        vllm_config.additional_config = {"custom_mamba_cache_multiplier": 3}
        assert mamba_blocks_per_request(vllm_config,
                                        is_align_mode=True) == (2, 3)

        vllm_config.speculative_config = MagicMock(num_speculative_tokens=4)
        # Resident minimum (4 + 1 + 1 = 6) beats the multiplier of 3.
        assert mamba_blocks_per_request(vllm_config,
                                        is_align_mode=True) == (6, 6)

    def test_propagate_mamba_num_blocks_publishes_worker_value(self):
        import tpu_inference.core.hybrid_coordinator as hc_mod
        from tpu_inference.core.hybrid_coordinator import \
            propagate_mamba_num_blocks

        hc_mod._GLOBAL_MAMBA_NUM_BLOCKS = None
        cfg = _make_mock_hybrid_kv_cache_config(num_attn_blocks=100,
                                                mamba_num_blocks=None)
        vllm_config = _make_mock_vllm_config(dp_size=2, max_num_seqs=8)
        vllm_config.cache_config.mamba_num_blocks = None
        engine_core = MagicMock()
        engine_core.collective_rpc.return_value = [1160, 1160]

        assert propagate_mamba_num_blocks(engine_core, cfg,
                                          vllm_config) == 1160
        engine_core.collective_rpc.assert_called_once_with(
            "get_mamba_num_blocks")
        assert vllm_config.cache_config.mamba_num_blocks == 1160
        assert cfg.mamba_num_blocks == 1160
        assert hc_mod.get_mamba_num_blocks() == 1160

        # Disagreeing workers cannot be reconciled.
        engine_core.collective_rpc.return_value = [1160, 1024]
        with pytest.raises(ValueError, match="disagree"):
            propagate_mamba_num_blocks(engine_core, cfg, vllm_config)

        # Nothing reported for a model with mamba layers is an error.
        hc_mod._GLOBAL_MAMBA_NUM_BLOCKS = None
        engine_core.collective_rpc.return_value = [None, None]
        with pytest.raises(ValueError, match="No worker reported"):
            propagate_mamba_num_blocks(engine_core, cfg, vllm_config)
        assert hc_mod.get_mamba_num_blocks() is None

    def test_executor_mixin_publishes_after_workers_allocate(self):
        import tpu_inference.core.hybrid_coordinator as hc_mod
        from tpu_inference.core.hybrid_coordinator import \
            MambaPoolSyncExecutorMixin

        cfg = _make_mock_hybrid_kv_cache_config(num_attn_blocks=100,
                                                mamba_num_blocks=None)
        vllm_config = _make_mock_vllm_config(dp_size=1, max_num_seqs=8)
        vllm_config.cache_config.mamba_num_blocks = None
        calls = []

        class FakeBaseExecutor:

            def __init__(self, vllm_config):
                self.vllm_config = vllm_config

            def initialize_from_config(self, kv_cache_configs):
                calls.append(("workers_allocated", kv_cache_configs))

            def collective_rpc(self, method):
                assert method == "get_mamba_num_blocks"
                assert calls, "RPC must run after the workers allocated"
                return [640, 640]

        class FakeExecutor(MambaPoolSyncExecutorMixin, FakeBaseExecutor):
            pass

        hc_mod._GLOBAL_MAMBA_NUM_BLOCKS = None
        FakeExecutor(vllm_config).initialize_from_config([cfg])
        assert calls == [("workers_allocated", [cfg])]
        assert cfg.mamba_num_blocks == 640
        assert vllm_config.cache_config.mamba_num_blocks == 640
        assert hc_mod.get_mamba_num_blocks() == 640

    def test_tpu_get_kv_cache_coordinator_resolves_from_kv_cache_config(self):
        import tpu_inference.core.hybrid_coordinator as hc_mod
        from tpu_inference.core.hybrid_coordinator import \
            tpu_get_kv_cache_coordinator

        hc_mod._GLOBAL_MAMBA_NUM_BLOCKS = None
        cfg = _make_mock_hybrid_kv_cache_config(num_attn_blocks=100,
                                                mamba_num_blocks=None)
        cfg.mamba_num_blocks = 72
        coord = tpu_get_kv_cache_coordinator(cfg, **_COORD_KWARGS)
        assert coord.mamba_num_blocks == 72

    def test_maybe_install_hooks_gated_on_align_prefix_caching(self):
        from unittest.mock import patch

        import tpu_inference.core.hybrid_coordinator as hc_mod

        vllm_config = _make_mock_vllm_config(dp_size=1, max_num_seqs=8)
        vllm_config.cache_config.enable_prefix_caching = True
        vllm_config.cache_config.mamba_cache_mode = "align"
        with patch.object(hc_mod, "install_hybrid_coordinator_hooks") as inst:
            hc_mod.maybe_install_hybrid_coordinator_hooks(vllm_config)
            inst.assert_called_once_with(vllm_config)

        vllm_config.cache_config.mamba_cache_mode = "none"
        with patch.object(hc_mod, "install_hybrid_coordinator_hooks") as inst:
            hc_mod.maybe_install_hybrid_coordinator_hooks(vllm_config)
            inst.assert_not_called()

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


class _FakeBlock:

    def __init__(self, block_id: int):
        self.block_id = block_id
        self.block_hash = None
        self.block_hash_num_tokens = None
        self.is_null = block_id == 0


class _FakePool:
    """A minimal block pool matching BlockPool's cache interface."""

    def __init__(self, hash_block_size: int = 256):
        self.cached_block_hash_to_block: dict[BlockHash, _FakeBlock] = {}
        self.null_block = _FakeBlock(0)
        self.hash_block_size = hash_block_size

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

    def cache_partial_block(self, *args, **kwargs):
        return None


def _make_test_mamba_spec(block_size: int = 256) -> MambaSpec:
    spec = MambaSpec(
        shapes=((3, 64), (8, 64, 16)),
        dtypes=(torch.bfloat16, torch.float32),
        block_size=block_size,
        mamba_cache_mode="align",
    )
    return spec


def _make_test_mamba_manager(spec: MambaSpec, pool: _FakePool) -> TPUMambaManager:
    return TPUMambaManager(
        kv_cache_spec=spec,
        block_pool=pool,
        enable_caching=True,
        kv_cache_group_id=0,
        scheduler_block_size=spec.block_size,
    )


class TestTPUMambaManager:

    def test_coordinator_instantiates_tpu_mamba_manager(self):
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

    def test_tpu_mamba_manager_init_filters_unsupported_kwargs(self):
        """Ensure TPUMambaManager drops args like max_in_flight_tokens and max_model_len
        for compatibility with older vLLM releases whose SingleTypeKVCacheManager.__init__
        does not accept them."""
        spec = _make_test_mamba_spec()
        pool = _FakePool()
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

    def test_cache_blocks_without_replay_boundaries(self):
        """Calling cache_blocks without replay_boundaries (as standard vLLM coordinator does) succeeds."""
        spec = _make_test_mamba_spec()
        pool = _FakePool()
        mgr = _make_test_mamba_manager(spec, pool)
        blocks = [_FakeBlock(100 + i) for i in range(8)]
        mgr.req_to_blocks["req1"] = blocks

        request = MagicMock()
        request.request_id = "req1"
        request.num_prompt_tokens = 2048

        mgr.cache_blocks(request, 2048)

    def test_cache_blocks_only_indexes_written_checkpoint(self):
        """Chunk of 2048 tokens (8 blocks) must only index block 7 in the cache."""
        spec = _make_test_mamba_spec()
        pool = _FakePool()
        mgr = _make_test_mamba_manager(spec, pool)
        blocks = [_FakeBlock(100 + i) for i in range(8)]
        mgr.req_to_blocks["req1"] = blocks

        request = MagicMock()
        request.request_id = "req1"
        request.num_prompt_tokens = 2048

        mgr.cache_blocks(request, 2048, replay_boundaries=[])

        # Intermediate blocks 0..6 must NOT be indexed
        for i in range(7):
            h = BlockHash(f"hash_{i}".encode())
            assert h not in pool.cached_block_hash_to_block

        # Only block 7 was written and indexed
        h7 = BlockHash(b"hash_7")
        assert h7 in pool.cached_block_hash_to_block
        assert pool.cached_block_hash_to_block[h7].block_id == 107

    def test_cache_blocks_multiple_chunks(self):
        """Chunked prefill across two passes must only index the end of each pass."""
        spec = _make_test_mamba_spec()
        pool = _FakePool()
        mgr = _make_test_mamba_manager(spec, pool)
        blocks = [_FakeBlock(100 + i) for i in range(16)]
        mgr.req_to_blocks["req1"] = blocks

        request = MagicMock()
        request.request_id = "req1"
        request.num_prompt_tokens = 4096

        # Pass 1: first 2048 tokens -> checkpoints block 7
        mgr.cache_blocks(request, 2048, replay_boundaries=[])
        assert BlockHash(b"hash_7") in pool.cached_block_hash_to_block
        assert BlockHash(b"hash_6") not in pool.cached_block_hash_to_block

        # Pass 2: remaining 2048 tokens (total 4096) -> checkpoints block 15
        mgr.cache_blocks(request, 4096, replay_boundaries=[])
        assert BlockHash(b"hash_15") in pool.cached_block_hash_to_block
        # Intermediate blocks 8..14 must NOT be indexed
        for i in range(8, 15):
            assert BlockHash(f"hash_{i}".encode()) not in pool.cached_block_hash_to_block

    def test_find_longest_cache_hit_rejects_unwritten_boundaries(self):
        """Subsequent request of length 512 matches blocks 0 and 1, neither of which
        was checkpointed; cache hit must be 0 rather than returning dirty memory."""
        spec = _make_test_mamba_spec()
        pool = _FakePool()
        mgr = _make_test_mamba_manager(spec, pool)
        blocks = [_FakeBlock(100 + i) for i in range(8)]
        mgr.req_to_blocks["req1"] = blocks

        request = MagicMock()
        request.request_id = "req1"
        request.num_prompt_tokens = 2048
        mgr.cache_blocks(request, 2048, replay_boundaries=[])

        # A request needing only 512 tokens (blocks 0..1)
        hashes = [BlockHash(f"hash_{i}".encode()) for i in range(2)]
        computed, hit_length = TPUMambaManager.find_longest_cache_hit(
            block_hashes=hashes,
            max_length=512,
            kv_cache_group_ids=[0],
            block_pool=pool,
            kv_cache_spec=spec,
            drop_eagle_block=False,
            alignment_tokens=256,
        )
        assert hit_length == 0

    def test_find_longest_cache_hit_matches_written_checkpoint(self):
        """Subsequent request of length 2048 matches the checkpoint at block 7."""
        spec = _make_test_mamba_spec()
        pool = _FakePool()
        mgr = _make_test_mamba_manager(spec, pool)
        blocks = [_FakeBlock(100 + i) for i in range(8)]
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
            alignment_tokens=256,
        )
        assert hit_length == 2048
        assert computed[0][-1].block_id == 107

    def test_find_longest_cache_hit_falls_back_to_earlier_written_checkpoint(self):
        """Subsequent request of length 3000 (after 2048 and 4096 were written)
        falls back cleanly to token 2048 (block 7), without hitting unwritten
        blocks 8..10."""
        spec = _make_test_mamba_spec()
        pool = _FakePool()
        mgr = _make_test_mamba_manager(spec, pool)
        blocks = [_FakeBlock(100 + i) for i in range(16)]
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
            alignment_tokens=256,
        )
        assert hit_length == 2048
        assert computed[0][-1].block_id == 107
