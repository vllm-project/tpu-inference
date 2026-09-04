# SPDX-License-Identifier: Apache-2.0

from collections.abc import Iterable, Sequence
from typing import Any

from vllm.logger import init_logger
from vllm.utils.math_utils import cdiv
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_coordinator import (HybridKVCacheCoordinator,
                                               KVCacheCoordinator)
from vllm.v1.core.kv_cache_coordinator import \
    get_kv_cache_coordinator as orig_get_kv_cache_coordinator
from vllm.v1.core.kv_cache_manager import KVCacheBlocks, KVCacheManager
from vllm.v1.core.kv_cache_utils import BlockHash, KVCacheBlock
from vllm.v1.core.single_type_kv_cache_manager import (
    CrossAttentionManager, get_manager_for_kv_cache_spec)
from vllm.v1.kv_cache_interface import (FullAttentionSpec, KVCacheConfig,
                                        MambaSpec)
from vllm.v1.request import Request, RequestStatus

logger = init_logger(__name__)

# ==============================================================================
# Decoupled Mamba Block Capacity Lifecycle:
# ------------------------------------------------------------------------------
# 1. PRODUCER (tpu_inference/runner/kv_cache_manager.py):
#    During device HBM memory profiling, KVCacheManager calculates the physical
#    compact Mamba block capacity and writes it to two places:
#      - cache_config.mamba_num_blocks = int(mamba_num_blocks)
#      - set_mamba_num_blocks(int(mamba_num_blocks))
#
# 2. PARTITIONER / PROPAGATION:
#    - DP Serving (tpu_inference/core/sched/dp_scheduler.py):
#      DPScheduler reads the total capacity from `vllm_config.cache_config.mamba_num_blocks`
#      and shards it per rank:
#        rank_kv_config.num_blocks = kv_cache_config.num_blocks // dp_size
#        rank_kv_config.mamba_num_blocks = mamba_num_blocks // dp_size
#      In the worker subprocess (_scheduler_worker_process), each rank sets:
#        set_mamba_num_blocks(kv_cache_config.mamba_num_blocks)
#        cache_config.mamba_num_blocks = kv_cache_config.mamba_num_blocks
#    - Non-DP Serving (DP=1):
#      Runs in the same process where KVCacheManager registered the total capacity via
#      set_mamba_num_blocks().
#
# 3. CONSUMER (TPUHybridKVCacheCoordinator):
#    Resolves mamba_num_blocks directly from get_mamba_num_blocks() (or an explicit
#    argument) and initializes an independent Mamba BlockPool, failing fast if not set.
# ==============================================================================
_HOOKS_INSTALLED: bool = False
_GLOBAL_MAMBA_NUM_BLOCKS: int | None = None


def set_mamba_num_blocks(num_blocks: int) -> None:
    """Register the TPU runner's computed Mamba block pool capacity."""
    global _GLOBAL_MAMBA_NUM_BLOCKS
    _GLOBAL_MAMBA_NUM_BLOCKS = int(num_blocks)


def get_mamba_num_blocks() -> int | None:
    """Get the registered Mamba block pool capacity."""
    return _GLOBAL_MAMBA_NUM_BLOCKS


def is_mamba_spec(spec: Any) -> bool:
    """Check if a KV cache spec represents a Mamba layer."""
    if isinstance(spec, MambaSpec):
        return True
    if hasattr(spec, "kv_cache_specs"):
        return any(
            isinstance(s, MambaSpec) for s in spec.kv_cache_specs.values())
    if "Mamba" in type(spec).__name__:
        return True
    return False


def is_mamba_group(group: Any) -> bool:
    """Check if a KV cache group contains Mamba layers."""
    spec = getattr(group, "kv_cache_spec", group)
    return is_mamba_spec(spec)


class TPUDualBlockPool(BlockPool):
    """A composite BlockPool that coordinates separate Attention and Mamba pools.

    This ensures that:
    - Attention and Mamba managers allocate from distinct physical pools with decoupled sizes.
    - Freeing, evicting, and touching blocks routes each block to its originating pool without cross-contamination.
    - Any external consumer accessing block_pool from KVCacheManager or scheduler sees a coherent BlockPool interface.
    """

    def __init__(
        self,
        attention_pool: BlockPool,
        mamba_pool: BlockPool,
        mamba_group_ids: set[int] | None = None,
    ):
        self.attention_pool = attention_pool
        self.mamba_pool = mamba_pool
        self.mamba_group_ids = mamba_group_ids or set()
        self.mamba_block_identities: set[int] = {
            id(b)
            for b in mamba_pool.blocks
        }

        # Mirror essential attributes from attention_pool (primary pool)
        self.num_gpu_blocks = attention_pool.num_gpu_blocks
        self.enable_caching = attention_pool.enable_caching
        self.hash_block_size = attention_pool.hash_block_size
        self.enable_kv_cache_events = attention_pool.enable_kv_cache_events
        self.metrics_collector = attention_pool.metrics_collector
        self.null_block = attention_pool.null_block
        self.blocks = attention_pool.blocks
        self.free_block_queue = attention_pool.free_block_queue
        self.cached_block_hash_to_block = attention_pool.cached_block_hash_to_block
        self.cached_block_hashes_by_block = attention_pool.cached_block_hashes_by_block
        self.kv_event_queue = attention_pool.kv_event_queue

    def free_blocks(self, ordered_blocks: Iterable[KVCacheBlock]) -> None:
        attn_blocks: list[KVCacheBlock] = []
        mamba_blocks: list[KVCacheBlock] = []
        for block in ordered_blocks:
            if id(block) in self.mamba_block_identities:
                mamba_blocks.append(block)
            else:
                attn_blocks.append(block)
        if attn_blocks:
            self.attention_pool.free_blocks(attn_blocks)
        if mamba_blocks:
            self.mamba_pool.free_blocks(mamba_blocks)

    def touch(self, blocks: Sequence[KVCacheBlock]) -> None:
        attn_blocks: list[KVCacheBlock] = []
        mamba_blocks: list[KVCacheBlock] = []
        for block in blocks:
            if id(block) in self.mamba_block_identities:
                mamba_blocks.append(block)
            else:
                attn_blocks.append(block)
        if attn_blocks:
            self.attention_pool.touch(attn_blocks)
        if mamba_blocks:
            self.mamba_pool.touch(mamba_blocks)

    def get_num_free_blocks(self) -> int:
        return self.attention_pool.get_num_free_blocks()

    def get_usage(self) -> float:
        return max(self.attention_pool.get_usage(),
                   self.mamba_pool.get_usage())

    def reset_prefix_cache(self) -> bool:
        r1 = self.attention_pool.reset_prefix_cache()
        r2 = self.mamba_pool.reset_prefix_cache()
        return r1 and r2

    def take_events(self):
        return self.attention_pool.take_events() + self.mamba_pool.take_events(
        )

    def evict_blocks(self, block_ids: set[int]) -> None:
        """Evict invalid blocks from cache on KV transfer failures.

        Note: Block IDs are assumed to belong to attention_pool, as external
        KV transfer load failures only apply to Attention KV cache.
        """
        attn_ids = {
            bid
            for bid in block_ids if bid < len(self.attention_pool.blocks)
        }
        if attn_ids:
            self.attention_pool.evict_blocks(attn_ids)

    def get_cached_block(
            self, block_hash: BlockHash,
            kv_cache_group_ids: list[int]) -> list[KVCacheBlock] | None:
        cached_blocks: list[KVCacheBlock] = []
        for gid in kv_cache_group_ids:
            pool = (self.mamba_pool
                    if gid in self.mamba_group_ids else self.attention_pool)
            block = pool.get_cached_block(block_hash, [gid])
            if not block:
                return None
            cached_blocks.extend(block)
        return cached_blocks


class TPUHybridKVCacheCoordinator(HybridKVCacheCoordinator):
    """Decoupled KV cache coordinator for hybrid models under Mamba align mode on TPU.

    Attention layers use a large pool (e.g. ~34,000 blocks) to maximize context length
    and concurrency, while Mamba layers use a dedicated, compact pool (e.g. 2,048 blocks)
    for active recurrent states and prefix checkpoints.
    Prefix cache hits are independently queried from each pool and reconciled via min(),
    preventing any cache eviction desync.
    """

    def __init__(
        self,
        kv_cache_config: KVCacheConfig,
        *args,
        mamba_num_blocks: int | None = None,
        **kwargs,
    ):
        super().__init__(kv_cache_config, *args, **kwargs)

        # Base __init__ initialized self.block_pool with kv_cache_config.num_blocks (Attention pool)
        self.attention_block_pool = self.block_pool

        # Identify Mamba and Attention group IDs
        self.mamba_group_ids = {
            i
            for i, g in enumerate(kv_cache_config.kv_cache_groups)
            if is_mamba_group(g)
        }
        self.attention_group_ids = {
            i
            for i, g in enumerate(kv_cache_config.kv_cache_groups)
            if not is_mamba_group(g)
        }

        assert len(self.mamba_group_ids) > 0, (
            f"[TPUHybridKVCacheCoordinator] No Mamba groups identified in "
            f"kv_cache_groups: {kv_cache_config.kv_cache_groups}")

        # Resolve mamba_num_blocks
        if mamba_num_blocks is None:
            mamba_num_blocks = get_mamba_num_blocks()
        if mamba_num_blocks is None:
            raise ValueError(
                "[TPUHybridKVCacheCoordinator] mamba_num_blocks must be registered "
                "via set_mamba_num_blocks().")
        self.mamba_num_blocks = int(mamba_num_blocks)

        logger.info(
            "[TPUHybridKVCacheCoordinator] Initializing dual block pools: "
            "attn_blocks=%d, mamba_blocks=%d (mamba_groups=%s)",
            self.attention_block_pool.num_gpu_blocks,
            self.mamba_num_blocks,
            sorted(self.mamba_group_ids),
        )

        # Allocate dedicated Mamba block pool
        self.mamba_block_pool = BlockPool(
            num_gpu_blocks=self.mamba_num_blocks,
            enable_caching=self.enable_caching,
            hash_block_size=self.hash_block_size,
            enable_kv_cache_events=self.attention_block_pool.
            enable_kv_cache_events,
            metrics_collector=self.attention_block_pool.metrics_collector,
        )

        # Re-bind Mamba managers to mamba_block_pool
        new_managers = list(self.single_type_managers)
        for i in self.mamba_group_ids:
            old_mgr = self.single_type_managers[i]
            new_managers[i] = get_manager_for_kv_cache_spec(
                kv_cache_spec=kv_cache_config.kv_cache_groups[i].kv_cache_spec,
                max_in_flight_tokens=getattr(old_mgr, "max_in_flight_tokens",
                                             128),
                max_model_len=self.max_model_len,
                block_pool=self.mamba_block_pool,
                enable_caching=self.enable_caching,
                kv_cache_group_id=i,
                dcp_world_size=getattr(self, "dcp_world_size", 1),
                pcp_world_size=getattr(self, "pcp_world_size", 1),
                scheduler_block_size=self.scheduler_block_size,
                needs_kv_cache_zeroing=self.kv_cache_config.
                needs_kv_cache_zeroing,
            )
        self.single_type_managers = tuple(new_managers)

        for i in self.mamba_group_ids:
            assert self.single_type_managers[
                i].block_pool is self.mamba_block_pool, (
                    f"Manager {i} block_pool was not re-bound to mamba_block_pool!"
                )
        for i in self.attention_group_ids:
            assert self.single_type_managers[
                i].block_pool is self.attention_block_pool, (
                    f"Manager {i} block_pool is not attention_block_pool!")

        # Replace self.block_pool with TPUDualBlockPool
        self.block_pool = TPUDualBlockPool(
            self.attention_block_pool,
            self.mamba_block_pool,
            mamba_group_ids=self.mamba_group_ids,
        )

        # Re-verify and split groups so attention_groups binds to updated managers
        self.verify_and_split_kv_cache_groups()

    def find_longest_cache_hit(
        self,
        block_hashes: list[BlockHash],
        max_cache_hit_length: int,
    ) -> tuple[tuple[list[KVCacheBlock], ...], int, int]:
        num_groups = len(self.kv_cache_config.kv_cache_groups)
        hit_length = max_cache_hit_length
        longest_hit_length = 0
        hit_blocks_by_group: list[list[KVCacheBlock]
                                  | None] = [None] * num_groups
        hit_length_by_group: list[int] = [0] * num_groups

        is_simple_hybrid = len(self.attention_groups) == 2 and isinstance(
            self.attention_groups[0].spec, FullAttentionSpec)
        eagle_verified: set[int] = set()

        while True:
            curr_hit_length = hit_length

            for idx, (spec, group_ids, manager_cls,
                      use_eagle) in enumerate(self.attention_groups):
                first_group_id = group_ids[0]
                group_block_size = self.single_type_managers[
                    first_group_id].block_size
                cached_blocks = hit_blocks_by_group[first_group_id]
                if isinstance(spec,
                              FullAttentionSpec) and cached_blocks is not None:
                    curr_hit_length = min(curr_hit_length,
                                          hit_length_by_group[first_group_id])
                    continue

                drop_eagle_block = use_eagle and idx not in eagle_verified
                _max_length = curr_hit_length
                if drop_eagle_block and not is_mamba_spec(spec):
                    eagle_margin = (
                        self.hash_block_size if self.enable_partial_hash_hits
                        and manager_cls.supports_fine_grained_hash_lookup
                        and group_block_size > self.hash_block_size else
                        group_block_size)
                    _max_length = min(curr_hit_length + eagle_margin,
                                      max_cache_hit_length)

                pool = (self.mamba_block_pool
                        if is_mamba_spec(spec) else self.attention_block_pool)

                hit_blocks, _new_hit_length = manager_cls.find_longest_cache_hit(
                    block_hashes=block_hashes,
                    max_length=_max_length,
                    kv_cache_group_ids=group_ids,
                    block_pool=pool,
                    kv_cache_spec=spec,
                    drop_eagle_block=drop_eagle_block,
                    alignment_tokens=self._cache_hit_alignment_tokens,
                    dcp_world_size=(self.dcp_world_size if isinstance(
                        spec, FullAttentionSpec) else 1),
                )
                if drop_eagle_block:
                    eagle_verified.add(idx)
                elif _new_hit_length < curr_hit_length:
                    eagle_verified.clear()
                curr_hit_length = _new_hit_length
                for group_id, blocks in zip(group_ids, hit_blocks):
                    hit_blocks_by_group[group_id] = blocks
                    hit_length_by_group[group_id] = _new_hit_length

                longest_hit_length = max(longest_hit_length, curr_hit_length)

            if curr_hit_length >= hit_length:
                break
            hit_length = curr_hit_length
            if is_simple_hybrid:
                break

        # Truncate full attention blocks to final hit_length
        first_group = self.attention_groups[0]
        if isinstance(first_group.spec, FullAttentionSpec):
            group_block_size = self.single_type_managers[
                first_group.group_ids[0]].block_size
            num_blocks = cdiv(hit_length, group_block_size)
            for group_id in first_group.group_ids:
                if (blks := hit_blocks_by_group[group_id]) is not None:
                    del blks[num_blocks:]
                    hit_length_by_group[group_id] = hit_length

        num_uncached_common_prefix_tokens = longest_hit_length - hit_length
        cache_hit_blocks = tuple(blocks if blocks is not None else []
                                 for blocks in hit_blocks_by_group)
        return cache_hit_blocks, hit_length, num_uncached_common_prefix_tokens

    def find_longest_cache_hit_per_group(
        self,
        block_hashes: list[BlockHash],
        max_cache_hit_length: int,
    ) -> tuple[tuple[list[KVCacheBlock], ...], tuple[int, ...]]:
        num_groups = len(self.kv_cache_config.kv_cache_groups)
        hit_blocks: list[list[KVCacheBlock]] = [[] for _ in range(num_groups)]
        hit_lengths: list[int] = [0] * num_groups

        for spec, group_ids, manager_cls, use_eagle in self.attention_groups:
            pool = (self.mamba_block_pool
                    if is_mamba_spec(spec) else self.attention_block_pool)
            blocks, group_hit = manager_cls.find_longest_cache_hit(
                block_hashes=block_hashes,
                max_length=max_cache_hit_length,
                kv_cache_group_ids=group_ids,
                block_pool=pool,
                kv_cache_spec=spec,
                drop_eagle_block=use_eagle,
                alignment_tokens=self._cache_hit_alignment_tokens,
            )
            for gid, blks in zip(group_ids, blocks):
                hit_blocks[gid] = blks
                hit_lengths[gid] = group_hit

        return tuple(hit_blocks), tuple(hit_lengths)

    def can_allocate_tokens(
        self,
        request: Request,
        num_tokens: int,
        new_computed_blocks: tuple[Sequence[KVCacheBlock], ...],
        num_encoder_tokens: int,
        total_computed_tokens: int,
        num_local_computed_tokens: int,
        num_tokens_main_model: int,
        apply_admission_cap: bool = False,
        watermark_blocks: int = 0,
        reserved_blocks: int = 0,
    ) -> bool:
        attn_blocks_needed = 0
        mamba_blocks_needed = 0

        for i, manager in enumerate(self.single_type_managers):
            is_mamba = i in self.mamba_group_ids
            if isinstance(manager, CrossAttentionManager):
                needed = manager.get_num_blocks_to_allocate(
                    request.request_id,
                    num_encoder_tokens,
                    [],
                    0,
                    0,
                    num_encoder_tokens,
                    apply_admission_cap=apply_admission_cap,
                )
            else:
                needed = manager.get_num_blocks_to_allocate(
                    request.request_id,
                    num_tokens,
                    new_computed_blocks[i],
                    total_computed_tokens,
                    num_local_computed_tokens,
                    num_tokens_main_model,
                    apply_admission_cap=apply_admission_cap,
                )

            if is_mamba:
                mamba_blocks_needed += needed
            else:
                attn_blocks_needed += needed

        avail_attn = (self.attention_block_pool.get_num_free_blocks() -
                      reserved_blocks)
        if attn_blocks_needed + watermark_blocks > avail_attn:
            return False

        avail_mamba = self.mamba_block_pool.get_num_free_blocks()
        if mamba_blocks_needed > avail_mamba:
            return False

        return True

    def get_num_blocks_to_allocate(
        self,
        request_id: str,
        num_tokens: int,
        new_computed_blocks: tuple[Sequence[KVCacheBlock], ...],
        num_encoder_tokens: int,
        total_computed_tokens: int,
        num_local_computed_tokens: int,
        num_tokens_main_model: int,
        apply_admission_cap: bool = False,
    ) -> int:
        """Returns attention blocks needed. Used by scheduler for in-flight prefill reservation."""
        num_blocks_to_allocate = 0
        for i, manager in enumerate(self.single_type_managers):
            if i in self.mamba_group_ids:
                continue
            if isinstance(manager, CrossAttentionManager):
                num_blocks_to_allocate += manager.get_num_blocks_to_allocate(
                    request_id,
                    num_encoder_tokens,
                    [],
                    0,
                    0,
                    num_encoder_tokens,
                    apply_admission_cap=apply_admission_cap,
                )
            else:
                num_blocks_to_allocate += manager.get_num_blocks_to_allocate(
                    request_id,
                    num_tokens,
                    new_computed_blocks[i],
                    total_computed_tokens,
                    num_local_computed_tokens,
                    num_tokens_main_model,
                    apply_admission_cap=apply_admission_cap,
                )
        return num_blocks_to_allocate


class TPUKVCacheManager(KVCacheManager):
    """KVCacheManager subclass that coordinates allocation across decoupled pools."""

    def allocate_slots(
        self,
        request: Request,
        num_new_tokens: int,
        num_new_computed_tokens: int = 0,
        new_computed_blocks: KVCacheBlocks | None = None,
        num_lookahead_tokens: int = 0,
        num_external_computed_tokens: int = 0,
        delay_cache_blocks: bool = False,
        num_encoder_tokens: int = 0,
        full_sequence_must_fit: bool = False,
        reserved_blocks: int = 0,
        has_scheduled_reqs: bool = True,
    ) -> KVCacheBlocks | None:
        if not isinstance(self.coordinator, TPUHybridKVCacheCoordinator):
            return super().allocate_slots(
                request=request,
                num_new_tokens=num_new_tokens,
                num_new_computed_tokens=num_new_computed_tokens,
                new_computed_blocks=new_computed_blocks,
                num_lookahead_tokens=num_lookahead_tokens,
                num_external_computed_tokens=num_external_computed_tokens,
                delay_cache_blocks=delay_cache_blocks,
                num_encoder_tokens=num_encoder_tokens,
                full_sequence_must_fit=full_sequence_must_fit,
                reserved_blocks=reserved_blocks,
                has_scheduled_reqs=has_scheduled_reqs,
            )

        if num_new_tokens == 0 and num_external_computed_tokens == 0:
            raise ValueError(
                "num_new_tokens must be greater than 0 when there are no "
                "external computed tokens")

        if new_computed_blocks is not None:
            new_computed_block_list = new_computed_blocks.blocks
        else:
            new_computed_block_list = self.empty_kv_cache_blocks.blocks

        num_local_computed_tokens = (request.num_computed_tokens +
                                     num_new_computed_tokens)
        total_computed_tokens = min(
            num_local_computed_tokens + num_external_computed_tokens,
            self.max_model_len,
        )

        watermark_blocks = 0
        if has_scheduled_reqs and request.status in (
                RequestStatus.WAITING,
                RequestStatus.PREEMPTED,
        ):
            watermark_blocks = self.watermark_blocks

        if full_sequence_must_fit:
            full_num_tokens = min(request.num_tokens, self.max_model_len)
            can_fit = self.coordinator.can_allocate_tokens(
                request=request,
                num_tokens=full_num_tokens,
                new_computed_blocks=new_computed_block_list,
                num_encoder_tokens=num_encoder_tokens,
                total_computed_tokens=total_computed_tokens,
                num_local_computed_tokens=num_local_computed_tokens,
                num_tokens_main_model=full_num_tokens,
                apply_admission_cap=True,
                watermark_blocks=watermark_blocks,
                reserved_blocks=0,
            )
            if not can_fit:
                return None

        num_tokens_main_model = total_computed_tokens + num_new_tokens
        num_tokens_need_slot = min(
            num_tokens_main_model + num_lookahead_tokens, self.max_model_len)

        self.coordinator.remove_skipped_blocks(
            request.request_id,
            max(0, total_computed_tokens - request.num_in_flight_tokens),
            num_prompt_tokens=request.num_prompt_tokens,
        )

        can_fit = self.coordinator.can_allocate_tokens(
            request=request,
            num_tokens=num_tokens_need_slot,
            new_computed_blocks=new_computed_block_list,
            num_encoder_tokens=num_encoder_tokens,
            total_computed_tokens=num_local_computed_tokens +
            num_external_computed_tokens,
            num_local_computed_tokens=num_local_computed_tokens,
            num_tokens_main_model=num_tokens_main_model,
            apply_admission_cap=False,
            watermark_blocks=watermark_blocks,
            reserved_blocks=reserved_blocks,
        )
        if not can_fit:
            return None

        if (new_computed_block_list is not self.empty_kv_cache_blocks.blocks
                or num_external_computed_tokens > 0):
            self.coordinator.allocate_new_computed_blocks(
                request_id=request.request_id,
                new_computed_blocks=new_computed_block_list,
                num_local_computed_tokens=num_local_computed_tokens,
                num_external_computed_tokens=num_external_computed_tokens,
            )

        new_blocks = self.coordinator.allocate_new_blocks(
            request.request_id,
            num_tokens_need_slot,
            num_tokens_main_model,
            num_encoder_tokens,
        )

        if not self.enable_caching or delay_cache_blocks:
            return self.create_kv_cache_blocks(new_blocks)

        num_tokens_to_cache = min(
            total_computed_tokens + num_new_tokens,
            request.num_tokens,
        )
        self.coordinator.cache_blocks(request, num_tokens_to_cache)

        return self.create_kv_cache_blocks(new_blocks)


def tpu_get_kv_cache_coordinator(
    kv_cache_config: KVCacheConfig,
    *args,
    **kwargs,
) -> KVCacheCoordinator:
    enable_caching = kwargs.get("enable_caching", False)
    has_mamba = any(is_mamba_group(g) for g in kv_cache_config.kv_cache_groups)
    if enable_caching and has_mamba:
        return TPUHybridKVCacheCoordinator(kv_cache_config, *args, **kwargs)
    return orig_get_kv_cache_coordinator(kv_cache_config, *args, **kwargs)


def install_hybrid_coordinator_hooks(vllm_config: Any | None = None) -> None:
    """Installs hooks into vLLM to use TPUHybridKVCacheCoordinator and TPUKVCacheManager."""
    global _HOOKS_INSTALLED
    import sys

    import vllm.v1.core.kv_cache_coordinator as coord_mod
    import vllm.v1.core.kv_cache_manager as mgr_mod

    coord_mod.get_kv_cache_coordinator = tpu_get_kv_cache_coordinator
    mgr_mod.get_kv_cache_coordinator = tpu_get_kv_cache_coordinator
    mgr_mod.KVCacheManager = TPUKVCacheManager

    if "vllm.v1.core.sched.scheduler" in sys.modules:
        sys.modules[
            "vllm.v1.core.sched.scheduler"].KVCacheManager = TPUKVCacheManager
    if "vllm.v1.core.sched.async_scheduler" in sys.modules:
        sys.modules[
            "vllm.v1.core.sched.async_scheduler"].KVCacheManager = TPUKVCacheManager

    _HOOKS_INSTALLED = True
    logger.info(
        "[tpu_inference] Installed TPUHybridKVCacheCoordinator hooks into vLLM"
    )
