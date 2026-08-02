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

from types import SimpleNamespace

import pytest

from tpu_inference.core.compact_mamba_pool import (
    _patch_compact_mamba_pool_classes, get_mamba_prefix_cache_num_blocks)


def _fake_vllm_classes():

    class Block:

        def __init__(self, block_id, *, is_null=False):
            self.block_id = block_id
            self.is_null = is_null

    class BlockPool:

        def __init__(
            self,
            num_gpu_blocks,
            enable_caching=True,
            hash_block_size=128,
            enable_kv_cache_events=False,
            metrics_collector=None,
        ):
            self.num_gpu_blocks = num_gpu_blocks
            self.enable_caching = enable_caching
            self.hash_block_size = hash_block_size
            self.enable_kv_cache_events = enable_kv_cache_events
            self.metrics_collector = metrics_collector
            self.null_block = Block(0, is_null=True)
            self.free_blocks = num_gpu_blocks - 1
            self.cached = {}
            self.evicted = []
            self.events = []
            self.reset_count = 0

        def get_num_free_blocks(self):
            return self.free_blocks

        def get_cached_block(self, block_hash, group_ids):
            blocks = []
            for group_id in group_ids:
                block = self.cached.get((block_hash, group_id))
                if block is None:
                    return None
                blocks.append(block)
            return blocks

        def evict_blocks(self, block_ids):
            self.evicted.append(set(block_ids))

        def reset_prefix_cache(self):
            self.reset_count += 1
            return True

        def take_events(self):
            events = self.events
            self.events = []
            return events

    class MambaManager:

        def __init__(self, block_pool, *, needed=0):
            self.block_pool = block_pool
            self._null_block = block_pool.null_block
            self.needed = needed
            self.mamba_cache_mode = "align"
            self.req_to_blocks = {}
            self.num_cached_block = {}
            self.last_state_block_idx = {}
            self._allocated_block_reqs = set()
            self.cached_blocks_this_step = set()

        def get_num_blocks_to_allocate(
            self,
            request_id,
            num_tokens,
            new_computed_blocks,
            total_computed_tokens,
            num_tokens_main_model,
            apply_admission_cap=False,
        ):
            return self.needed

        @classmethod
        def find_longest_cache_hit(
            cls,
            block_hashes,
            max_length,
            kv_cache_group_ids,
            block_pool,
            kv_cache_spec,
            use_eagle,
            alignment_tokens,
            dcp_world_size=1,
            pcp_world_size=1,
        ):
            return ("original", )

    class AttentionManager:

        def __init__(self, block_pool, *, needed=0):
            self.block_pool = block_pool
            self.needed = needed

        def get_num_blocks_to_allocate(
            self,
            request_id,
            num_tokens,
            new_computed_blocks,
            total_computed_tokens,
            num_tokens_main_model,
            apply_admission_cap=False,
        ):
            return self.needed

    class Coordinator:

        def __init__(self, block_pool, managers):
            self.block_pool = block_pool
            self.single_type_managers = tuple(managers)

        def get_num_blocks_to_allocate(
            self,
            request_id,
            num_tokens,
            new_computed_blocks,
            num_encoder_tokens,
            total_computed_tokens,
            num_tokens_main_model,
            apply_admission_cap=False,
        ):
            return sum(
                manager.get_num_blocks_to_allocate(
                    request_id=request_id,
                    num_tokens=num_tokens,
                    new_computed_blocks=new_computed_blocks[group_id],
                    total_computed_tokens=total_computed_tokens,
                    num_tokens_main_model=num_tokens_main_model,
                    apply_admission_cap=apply_admission_cap,
                )
                for group_id, manager in enumerate(self.single_type_managers))

    class HybridCoordinator(Coordinator):
        pass

    class KVCacheManager:

        def __init__(self, coordinator):
            self.coordinator = coordinator
            self.block_pool = coordinator.block_pool
            self.log_stats = True
            self.prefix_cache_stats = SimpleNamespace(reset=False)
            self.original_evict_calls = []
            self.original_reset_calls = 0
            self.original_event_calls = 0

        def evict_blocks(self, block_ids):
            self.original_evict_calls.append(set(block_ids))
            self.block_pool.evict_blocks(block_ids)

        def reset_prefix_cache(self):
            self.original_reset_calls += 1
            return True

        def take_events(self):
            self.original_event_calls += 1
            return ["original"]

    class Scheduler:

        def __init__(self, coordinator, max_num_seqs, vllm_config=None):
            self.vllm_config = vllm_config
            self.scheduler_config = SimpleNamespace(max_num_seqs=max_num_seqs)
            self.kv_cache_manager = KVCacheManager(coordinator)

    return SimpleNamespace(
        Block=Block,
        BlockPool=BlockPool,
        MambaManager=MambaManager,
        AttentionManager=AttentionManager,
        Coordinator=Coordinator,
        HybridCoordinator=HybridCoordinator,
        KVCacheManager=KVCacheManager,
        Scheduler=Scheduler,
    )


def _install(classes):
    _patch_compact_mamba_pool_classes(
        scheduler_cls=classes.Scheduler,
        coordinator_cls=classes.Coordinator,
        kv_cache_manager_cls=classes.KVCacheManager,
        hybrid_coordinator_cls=classes.HybridCoordinator,
        mamba_manager_cls=classes.MambaManager,
        block_pool_cls=classes.BlockPool,
    )


class TestCompactMambaPool:

    def test_block_count_uses_configured_multiplier(self, monkeypatch):
        monkeypatch.setenv("TPU_MAMBA_PREFIX_CACHE_BLOCK_MULTIPLIER", "4")

        assert get_mamba_prefix_cache_num_blocks(8) == 1 + 4 * 8

    @pytest.mark.parametrize("multiplier", ["-1", "0", "1"])
    def test_block_count_rejects_multiplier_below_correctness_minimum(
            self, monkeypatch, multiplier):
        monkeypatch.setenv("TPU_MAMBA_PREFIX_CACHE_BLOCK_MULTIPLIER",
                           multiplier)

        with pytest.raises(ValueError, match="must be at least 2"):
            get_mamba_prefix_cache_num_blocks(8)

    @pytest.mark.parametrize(
        ("config", "message"),
        [
            (SimpleNamespace(
                cache_config=SimpleNamespace(mamba_cache_mode="align"),
                kv_transfer_config=object(),
                speculative_config=None,
            ), "KV transfer"),
            (SimpleNamespace(
                cache_config=SimpleNamespace(mamba_cache_mode="align"),
                kv_transfer_config=None,
                speculative_config=object()), "speculative decoding"),
        ],
    )
    def test_rejects_unsupported_config_before_installing_pool(
            self, config, message):
        classes = _fake_vllm_classes()
        _install(classes)
        main_pool = classes.BlockPool(1000)
        mamba = classes.MambaManager(main_pool)
        coordinator = classes.HybridCoordinator(main_pool, [mamba])

        with pytest.raises(NotImplementedError, match=message):
            classes.Scheduler(coordinator, max_num_seqs=4, vllm_config=config)

        assert mamba.block_pool is main_pool
        assert not hasattr(main_pool, "_tpu_compact_mamba_pools")

    def test_each_mamba_group_gets_a_sized_private_pool(self):
        classes = _fake_vllm_classes()
        _install(classes)
        # Installing twice must not stack wrappers.
        _install(classes)

        main_pool = classes.BlockPool(1000)
        attention = classes.AttentionManager(main_pool)
        mamba_0 = classes.MambaManager(main_pool)
        mamba_1 = classes.MambaManager(main_pool)
        coordinator = classes.HybridCoordinator(main_pool,
                                                [attention, mamba_0, mamba_1])
        scheduler = classes.Scheduler(coordinator, max_num_seqs=4)

        pools = scheduler.kv_cache_manager._tpu_compact_mamba_pools
        assert set(pools) == {1, 2}
        assert pools[1].num_gpu_blocks == 1 + 2 * 4
        assert pools[2].num_gpu_blocks == 1 + 2 * 4
        assert pools[1] is not pools[2]
        assert pools[1] is not main_pool
        assert attention.block_pool is main_pool
        assert mamba_0.block_pool is pools[1]
        assert mamba_0._null_block is pools[1].null_block
        assert mamba_1.block_pool is pools[2]
        assert mamba_1._null_block is pools[2].null_block

    def test_pool_installation_preflights_every_mamba_group(self):
        classes = _fake_vllm_classes()
        _install(classes)
        main_pool = classes.BlockPool(1000)
        mamba_0 = classes.MambaManager(main_pool)
        mamba_1 = classes.MambaManager(main_pool)
        mamba_1.req_to_blocks["active"] = [main_pool.null_block]
        coordinator = classes.HybridCoordinator(main_pool, [mamba_0, mamba_1])

        with pytest.raises(RuntimeError, match="group 1 already owns"):
            classes.Scheduler(coordinator, max_num_seqs=4)

        assert mamba_0.block_pool is main_pool
        assert mamba_1.block_pool is main_pool

    def test_admission_checks_each_pool_without_charging_main_pool(self):
        classes = _fake_vllm_classes()
        _install(classes)
        main_pool = classes.BlockPool(1000)
        attention = classes.AttentionManager(main_pool, needed=7)
        mamba_0 = classes.MambaManager(main_pool, needed=2)
        mamba_1 = classes.MambaManager(main_pool, needed=3)
        coordinator = classes.HybridCoordinator(main_pool,
                                                [attention, mamba_0, mamba_1])
        classes.Scheduler(coordinator, max_num_seqs=4)

        args = dict(
            request_id="request-0",
            num_tokens=128,
            new_computed_blocks=([], [], []),
            num_encoder_tokens=0,
            total_computed_tokens=0,
            num_tokens_main_model=128,
        )
        assert coordinator.get_num_blocks_to_allocate(**args) == 7

        mamba_0.needed = mamba_0.block_pool.get_num_free_blocks() + 1
        assert (coordinator.get_num_blocks_to_allocate(
            **args) == main_pool.get_num_free_blocks() + 1)

    def test_cache_hit_must_exist_at_same_boundary_in_every_group(self):
        classes = _fake_vllm_classes()
        _install(classes)
        main_pool = classes.BlockPool(1000)
        mamba_0 = classes.MambaManager(main_pool)
        mamba_1 = classes.MambaManager(main_pool)
        coordinator = classes.HybridCoordinator(main_pool, [mamba_0, mamba_1])
        classes.Scheduler(coordinator, max_num_seqs=4)
        pools = main_pool._tpu_compact_mamba_pools
        block_0 = classes.Block(3)
        block_1 = classes.Block(5)
        pools[0].cached[("hash-2", 0)] = block_0
        pools[1].cached[("hash-2", 1)] = block_1
        spec = SimpleNamespace(block_size=2048)

        hit = classes.MambaManager.find_longest_cache_hit(
            ["hash-0", "hash-1", "hash-2"],
            3 * spec.block_size,
            [0, 1],
            main_pool,
            spec,
            False,
            128,
        )
        assert hit[0] == [pools[0].null_block, pools[0].null_block, block_0]
        assert hit[1] == [pools[1].null_block, pools[1].null_block, block_1]

        pools[1].cached = {("hash-1", 1): block_1}
        assert classes.MambaManager.find_longest_cache_hit(
            ["hash-0", "hash-1", "hash-2"],
            3 * spec.block_size,
            [0, 1],
            main_pool,
            spec,
            False,
            128,
        ) == ([], [])

    def test_global_lifecycle_operations_keep_pool_local_ids_separate(self):
        classes = _fake_vllm_classes()
        _install(classes)
        main_pool = classes.BlockPool(20, enable_kv_cache_events=True)
        mamba = classes.MambaManager(main_pool)
        coordinator = classes.HybridCoordinator(main_pool, [mamba])
        scheduler = classes.Scheduler(coordinator, max_num_seqs=2)
        manager = scheduler.kv_cache_manager
        private_pool = mamba.block_pool

        manager.evict_blocks({2, 12})
        assert main_pool.evicted == [{2, 12}]
        assert private_pool.evicted == []

        main_pool.events = ["attention"]
        private_pool.events = ["mamba"]
        assert manager.take_events() == ["attention", "mamba"]

        private_pool.free_blocks -= 1
        assert manager.reset_prefix_cache() is False
        assert main_pool.reset_count == 0
        assert private_pool.reset_count == 0

        private_pool.free_blocks += 1
        assert manager.reset_prefix_cache() is True
        assert main_pool.reset_count == 1
        assert private_pool.reset_count == 1
        assert manager.prefix_cache_stats.reset is True
