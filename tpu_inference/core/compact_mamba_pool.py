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
"""Give each align-mode Mamba group its own scheduler block pool.

vLLM normally gives every KV-cache group one shared ``BlockPool``. That is a
good fit when every group's physical cache has the same leading dimension, but
it prevents TPU hybrid attention/Mamba models from allocating a compact Mamba
state array: a Mamba block ID can otherwise span the full attention pool.

An align-mode Mamba request needs at most a prior/source state and a running
state. The default private pool therefore needs::

    1 + 2 * max_num_seqs

blocks per Mamba group, including that pool's null block. The multiplier is
configurable through ``TPU_MAMBA_PREFIX_CACHE_BLOCK_MULTIPLIER``; values above
two reserve additional LRU retention capacity. Cached states whose reference
count reaches zero remain in the private pool's normal free/LRU queue, so
prefix hits and eviction retain vLLM's native semantics.
"""

from __future__ import annotations

import inspect
from collections.abc import Mapping, Sequence
from functools import wraps
from typing import Any

from tpu_inference import envs
from tpu_inference.logger import init_logger

logger = init_logger(__name__)

_PRIVATE_POOLS_ATTR = "_tpu_compact_mamba_pools"


def get_mamba_prefix_cache_block_multiplier() -> int:
    """Return the validated number of Mamba blocks reserved per sequence."""
    multiplier = envs.TPU_MAMBA_PREFIX_CACHE_BLOCK_MULTIPLIER
    if multiplier < 2:
        raise ValueError(
            "TPU_MAMBA_PREFIX_CACHE_BLOCK_MULTIPLIER must be at least 2 "
            f"(cached source + running destination), got {multiplier}")
    return multiplier


def get_mamba_prefix_cache_num_blocks(max_num_seqs: int) -> int:
    """Return one DP rank's Mamba pool size, including its null block."""
    if max_num_seqs <= 0:
        raise ValueError(f"max_num_seqs must be positive, got {max_num_seqs}")
    return 1 + get_mamba_prefix_cache_block_multiplier() * max_num_seqs


def is_mamba_prefix_cache_enabled(vllm_config: Any) -> bool:
    """Whether this model needs scheduler-addressable Mamba prefix state."""
    model_config = getattr(vllm_config, "model_config", None)
    cache_config = getattr(vllm_config, "cache_config", None)
    return bool(model_config is not None
                and getattr(model_config, "is_hybrid", False)
                and cache_config is not None
                and getattr(cache_config, "enable_prefix_caching", False))


def validate_mamba_prefix_cache_config(vllm_config: Any) -> None:
    """Validate features that share scheduler-owned Mamba block IDs."""
    cache_mode = getattr(vllm_config.cache_config, "mamba_cache_mode", None)
    if cache_mode != "align":
        raise NotImplementedError(
            "TPU Mamba prefix caching requires mamba_cache_mode='align', "
            f"got {cache_mode!r}.")
    if getattr(vllm_config, "kv_transfer_config", None) is not None:
        raise NotImplementedError(
            "Compact Mamba prefix-cache pools do not support KV transfer "
            "connectors because their block IDs are pool-local.")
    if getattr(vllm_config, "speculative_config", None) is not None:
        raise NotImplementedError(
            "Compact Mamba prefix-cache pools do not support speculative "
            "decoding.")
    get_mamba_prefix_cache_block_multiplier()


def _new_block_pool(block_pool_cls: type, main_pool: Any,
                    num_blocks: int) -> Any:
    """Construct a private pool across the supported vLLM signatures."""
    available = {
        "enable_caching":
        getattr(main_pool, "enable_caching", True),
        "hash_block_size":
        main_pool.hash_block_size,
        "enable_kv_cache_events":
        getattr(main_pool, "enable_kv_cache_events", False),
        # A private pool must not double-count the main pool's metrics.
        "metrics_collector":
        None,
    }
    parameters = inspect.signature(block_pool_cls).parameters
    kwargs = {
        name: value
        for name, value in available.items() if name in parameters
    }
    return block_pool_cls(num_blocks, **kwargs)


def _manager_has_allocations(manager: Any) -> bool:
    tracked = (
        "req_to_blocks",
        "num_cached_block",
        "last_state_block_idx",
        "_allocated_block_reqs",
        "cached_blocks_this_step",
    )
    return any(bool(getattr(manager, name, ())) for name in tracked)


def _attach_private_mamba_pools(
    scheduler: Any,
    *,
    block_pool_cls: type,
    hybrid_coordinator_cls: type,
    mamba_manager_cls: type,
) -> None:
    kv_cache_manager = scheduler.kv_cache_manager
    coordinator = kv_cache_manager.coordinator
    if not isinstance(coordinator, hybrid_coordinator_cls):
        return

    main_pool = coordinator.block_pool
    if hasattr(main_pool, _PRIVATE_POOLS_ATTR):
        return
    if not getattr(main_pool, "enable_caching", True):
        return

    max_num_seqs = scheduler.scheduler_config.max_num_seqs
    mamba_managers = [
        (group_id, manager)
        for group_id, manager in enumerate(coordinator.single_type_managers)
        if isinstance(manager, mamba_manager_cls)
        and getattr(manager, "mamba_cache_mode", None) == "align"
    ]
    if not mamba_managers:
        return

    vllm_config = getattr(scheduler, "vllm_config", None)
    if vllm_config is not None:
        validate_mamba_prefix_cache_config(vllm_config)

    for group_id, manager in mamba_managers:
        if _manager_has_allocations(manager):
            raise RuntimeError(
                "Compact Mamba pools must be installed before scheduling "
                f"requests; group {group_id} already owns cache blocks.")

    num_blocks = get_mamba_prefix_cache_num_blocks(max_num_seqs)
    private_pools = {
        group_id: _new_block_pool(block_pool_cls, main_pool, num_blocks)
        for group_id, _ in mamba_managers
    }
    for group_id, manager in mamba_managers:
        pool = private_pools[group_id]
        manager.block_pool = pool
        # SingleTypeKVCacheManager caches the null block at construction.
        manager._null_block = pool.null_block

    if not private_pools:
        return

    setattr(coordinator, _PRIVATE_POOLS_ATTR, private_pools)
    # HybridKVCacheCoordinator.find_longest_cache_hit passes its main pool to
    # the Mamba classmethod. Tag it so that method can route each group lookup
    # to the corresponding private pool.
    setattr(main_pool, _PRIVATE_POOLS_ATTR, private_pools)
    setattr(kv_cache_manager, _PRIVATE_POOLS_ATTR, private_pools)
    logger.info(
        "Installed compact scheduler pools for Mamba groups %s: "
        "max_num_seqs=%d, blocks=%s",
        sorted(private_pools),
        max_num_seqs,
        {
            group_id: pool.num_gpu_blocks
            for group_id, pool in private_pools.items()
        },
    )


def _all_block_pools(kv_cache_manager: Any) -> tuple[Any, ...]:
    main_pool = kv_cache_manager.block_pool
    private_pools = getattr(kv_cache_manager, _PRIVATE_POOLS_ATTR, {})
    pools: list[Any] = []
    seen: set[int] = set()
    for pool in (main_pool, *private_pools.values()):
        if id(pool) in seen:
            continue
        seen.add(id(pool))
        pools.append(pool)
    return tuple(pools)


def _find_private_mamba_cache_hit(
    *,
    block_hashes: Sequence[Any],
    max_length: int,
    kv_cache_group_ids: Sequence[int],
    pools: Mapping[int, Any],
    kv_cache_spec: Any,
    alignment_tokens: int,
    dcp_world_size: int = 1,
    pcp_world_size: int = 1,
) -> tuple[list[Any], ...]:
    """Find one boundary present in every requested Mamba group."""
    if dcp_world_size != 1:
        raise NotImplementedError("DCP does not support Mamba prefix caching.")
    if pcp_world_size != 1:
        raise NotImplementedError("PCP does not support Mamba prefix caching.")

    computed_blocks: tuple[list[Any],
                           ...] = tuple([] for _ in kv_cache_group_ids)
    block_size = kv_cache_spec.block_size
    max_num_blocks = max_length // block_size
    for index in range(max_num_blocks - 1, -1, -1):
        if (block_size != alignment_tokens
                and (index + 1) * block_size % alignment_tokens != 0):
            continue

        hits: list[Any] = []
        for group_id in kv_cache_group_ids:
            cached = pools[group_id].get_cached_block(block_hashes[index],
                                                      [group_id])
            if not cached:
                break
            hits.append(cached[0])
        if len(hits) != len(kv_cache_group_ids):
            continue

        for computed, hit, group_id in zip(computed_blocks,
                                           hits,
                                           kv_cache_group_ids,
                                           strict=True):
            computed.extend([pools[group_id].null_block] * index)
            computed.append(hit)
        break
    return computed_blocks


def _patch_compact_mamba_pool_classes(
    *,
    scheduler_cls: type,
    coordinator_cls: type,
    kv_cache_manager_cls: type,
    hybrid_coordinator_cls: type,
    mamba_manager_cls: type,
    block_pool_cls: type,
) -> None:
    """Install the patch on supplied classes; split out for focused tests."""
    if not hasattr(scheduler_cls, "_tpu_orig_compact_mamba_init"):
        original_scheduler_init = scheduler_cls.__init__
        scheduler_cls._tpu_orig_compact_mamba_init = original_scheduler_init

        @wraps(original_scheduler_init)
        def scheduler_init(self, *args, **kwargs):
            original_scheduler_init(self, *args, **kwargs)
            _attach_private_mamba_pools(
                self,
                block_pool_cls=block_pool_cls,
                hybrid_coordinator_cls=hybrid_coordinator_cls,
                mamba_manager_cls=mamba_manager_cls,
            )

        scheduler_cls.__init__ = scheduler_init

    if not hasattr(coordinator_cls, "_tpu_orig_compact_mamba_get_num_blocks"):
        original_get_num_blocks = coordinator_cls.get_num_blocks_to_allocate
        coordinator_cls._tpu_orig_compact_mamba_get_num_blocks = original_get_num_blocks

        @wraps(original_get_num_blocks)
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
            total = int(
                original_get_num_blocks(
                    self,
                    request_id=request_id,
                    num_tokens=num_tokens,
                    new_computed_blocks=new_computed_blocks,
                    num_encoder_tokens=num_encoder_tokens,
                    total_computed_tokens=total_computed_tokens,
                    num_tokens_main_model=num_tokens_main_model,
                    apply_admission_cap=apply_admission_cap,
                ))
            private_pools = getattr(self, _PRIVATE_POOLS_ATTR, None)
            if not private_pools:
                return total

            private_total = 0
            for group_id, pool in private_pools.items():
                manager = self.single_type_managers[group_id]
                needed = manager.get_num_blocks_to_allocate(
                    request_id=request_id,
                    num_tokens=num_tokens,
                    new_computed_blocks=new_computed_blocks[group_id],
                    total_computed_tokens=total_computed_tokens,
                    num_tokens_main_model=num_tokens_main_model,
                    apply_admission_cap=apply_admission_cap,
                )
                if needed > pool.get_num_free_blocks():
                    # KVCacheManager compares this scalar with the main pool.
                    # Return a guaranteed failure while preserving its normal
                    # retry/preemption path.
                    return self.block_pool.get_num_free_blocks() + 1
                private_total += needed

            main_total = total - private_total
            if main_total < 0:
                raise RuntimeError(
                    "Private Mamba allocation exceeded total KV allocation: "
                    f"private={private_total}, total={total}")
            return main_total

        coordinator_cls.get_num_blocks_to_allocate = get_num_blocks_to_allocate

    if not hasattr(mamba_manager_cls,
                   "_tpu_orig_compact_mamba_find_cache_hit"):
        original_find_cache_hit = mamba_manager_cls.find_longest_cache_hit.__func__
        mamba_manager_cls._tpu_orig_compact_mamba_find_cache_hit = (
            original_find_cache_hit)

        @classmethod
        @wraps(original_find_cache_hit)
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
            main_pool = block_pool
            private_pools = getattr(main_pool, _PRIVATE_POOLS_ATTR, None)
            group_ids = kv_cache_group_ids
            if not private_pools or any(group_id not in private_pools
                                        for group_id in group_ids):
                return original_find_cache_hit(
                    cls,
                    block_hashes=block_hashes,
                    max_length=max_length,
                    kv_cache_group_ids=kv_cache_group_ids,
                    block_pool=block_pool,
                    kv_cache_spec=kv_cache_spec,
                    use_eagle=use_eagle,
                    alignment_tokens=alignment_tokens,
                    dcp_world_size=dcp_world_size,
                    pcp_world_size=pcp_world_size,
                )

            return _find_private_mamba_cache_hit(
                block_hashes=block_hashes,
                max_length=max_length,
                kv_cache_group_ids=group_ids,
                pools=private_pools,
                kv_cache_spec=kv_cache_spec,
                alignment_tokens=alignment_tokens,
                dcp_world_size=dcp_world_size,
                pcp_world_size=pcp_world_size,
            )

        mamba_manager_cls.find_longest_cache_hit = find_longest_cache_hit

    if not hasattr(kv_cache_manager_cls,
                   "_tpu_orig_compact_mamba_reset_prefix_cache"):
        original_reset_prefix_cache = kv_cache_manager_cls.reset_prefix_cache
        kv_cache_manager_cls._tpu_orig_compact_mamba_reset_prefix_cache = (
            original_reset_prefix_cache)

        @wraps(original_reset_prefix_cache)
        def reset_prefix_cache(self):
            if not getattr(self, _PRIVATE_POOLS_ATTR, None):
                return original_reset_prefix_cache(self)
            pools = _all_block_pools(self)
            if any(pool.num_gpu_blocks - pool.get_num_free_blocks() != 1
                   for pool in pools):
                logger.warning(
                    "Failed to reset prefix cache because a compact Mamba "
                    "or attention pool still has referenced blocks")
                return False
            for pool in pools:
                if not pool.reset_prefix_cache():
                    raise RuntimeError(
                        "KV block pool reset failed after a successful "
                        "all-pool preflight")
            if getattr(self, "log_stats", False):
                assert self.prefix_cache_stats is not None
                self.prefix_cache_stats.reset = True
            return True

        kv_cache_manager_cls.reset_prefix_cache = reset_prefix_cache

    if not hasattr(kv_cache_manager_cls,
                   "_tpu_orig_compact_mamba_take_events"):
        original_take_events = kv_cache_manager_cls.take_events
        kv_cache_manager_cls._tpu_orig_compact_mamba_take_events = original_take_events

        @wraps(original_take_events)
        def take_events(self):
            if not getattr(self, _PRIVATE_POOLS_ATTR, None):
                return original_take_events(self)
            events = []
            for pool in _all_block_pools(self):
                events.extend(pool.take_events())
            return events

        kv_cache_manager_cls.take_events = take_events


def install_compact_mamba_pool() -> None:
    """Install compact, scheduler-addressable Mamba pools into vLLM."""
    from vllm.v1.core.block_pool import BlockPool
    from vllm.v1.core.kv_cache_coordinator import (HybridKVCacheCoordinator,
                                                   KVCacheCoordinator)
    from vllm.v1.core.kv_cache_manager import KVCacheManager
    from vllm.v1.core.sched.scheduler import Scheduler
    from vllm.v1.core.single_type_kv_cache_manager import MambaManager

    _patch_compact_mamba_pool_classes(
        scheduler_cls=Scheduler,
        coordinator_cls=KVCacheCoordinator,
        kv_cache_manager_cls=KVCacheManager,
        hybrid_coordinator_cls=HybridKVCacheCoordinator,
        mamba_manager_cls=MambaManager,
        block_pool_cls=BlockPool,
    )
