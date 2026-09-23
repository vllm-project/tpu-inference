# SPDX-License-Identifier: Apache-2.0

import inspect
from collections.abc import Iterable, Sequence
from typing import Any

from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_coordinator import (HybridKVCacheCoordinator,
                                               KVCacheCoordinator)
from vllm.v1.core.kv_cache_coordinator import \
    get_kv_cache_coordinator as orig_get_kv_cache_coordinator
from vllm.v1.core.kv_cache_manager import KVCacheBlocks, KVCacheManager
from vllm.v1.core.kv_cache_utils import BlockHash, KVCacheBlock
from vllm.v1.core.single_type_kv_cache_manager import (
    CrossAttentionManager, MambaManager, SingleTypeKVCacheManager,
    get_manager_for_kv_cache_spec)
from vllm.v1.kv_cache_interface import KVCacheConfig, MambaSpec
from vllm.v1.request import Request, RequestStatus

from tpu_inference.logger import init_logger

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
#    - Multi-process serving (Ray / multiproc executor): the runner is not in the
#      engine-core process, so the TPU executors mix in `MambaPoolSyncExecutorMixin`,
#      which fetches the allocated row count from the workers right after
#      `initialize_from_config` (before the scheduler is built) and writes it to the
#      same two places plus `kv_cache_config.mamba_num_blocks`.
#
# 3. CONSUMER (TPUHybridKVCacheCoordinator):
#    Resolves mamba_num_blocks from an explicit argument, `kv_cache_config`, or
#    get_mamba_num_blocks(), and initializes an independent Mamba BlockPool;
#    fails fast if none of them is known.
# ==============================================================================
_HOOKS_INSTALLED: bool = False
_GLOBAL_MAMBA_NUM_BLOCKS: int | None = None

# Mamba blocks reserved per request in align mode (prefix caching): one for
# the state a request generates from, the rest for the prefix checkpoints it
# leaves behind for later requests to resume from. Override per-run with
# `--additional-config '{"custom_mamba_cache_multiplier": N}'`.
DEFAULT_MAMBA_CACHE_MULTIPLIER = 8


def set_mamba_num_blocks(num_blocks: int) -> None:
    """Register the TPU runner's computed Mamba block pool capacity."""
    global _GLOBAL_MAMBA_NUM_BLOCKS
    _GLOBAL_MAMBA_NUM_BLOCKS = int(num_blocks)


def get_mamba_num_blocks() -> int | None:
    """Get the registered Mamba block pool capacity."""
    return _GLOBAL_MAMBA_NUM_BLOCKS


def mamba_blocks_per_request(vllm_config: Any,
                             is_align_mode: bool) -> tuple[int, int]:
    """Return `(min_blocks_per_req, blocks_per_req)` for the compact mamba pool.

    Every request keeps one resident slot per speculative position
    (`num_spec + 1`), plus one more in align mode for the state it generates
    from. Align mode also reserves `custom_mamba_cache_multiplier` (default
    `DEFAULT_MAMBA_CACHE_MULTIPLIER`) slots per request for prefix
    checkpoints, never fewer than the resident minimum.
    """
    num_spec = 0
    if vllm_config.speculative_config is not None:
        num_spec = vllm_config.speculative_config.num_speculative_tokens
    min_blocks_per_req = num_spec + 1 + (1 if is_align_mode else 0)
    if not is_align_mode:
        return min_blocks_per_req, min_blocks_per_req
    multiplier = int(
        vllm_config.additional_config.get("custom_mamba_cache_multiplier",
                                          DEFAULT_MAMBA_CACHE_MULTIPLIER))
    return min_blocks_per_req, max(multiplier, min_blocks_per_req)


def mamba_pool_size(max_num_reqs: int, blocks_per_req: int,
                    divisor: int) -> int:
    """`max_num_reqs * blocks_per_req` slots plus the null block, rounded up to
    a multiple of the sharding `divisor` so per-device shards are equal."""
    blocks = max_num_reqs * blocks_per_req + 1
    return ((blocks + divisor - 1) // divisor) * divisor


def is_mamba_spec(spec: Any) -> bool:
    """Check if a KV cache spec represents a Mamba layer."""
    if isinstance(spec, MambaSpec):
        return True
    if hasattr(spec, "kv_cache_specs"):
        return any(
            isinstance(s, MambaSpec) for s in spec.kv_cache_specs.values())
    return False


def is_mamba_group(group: Any) -> bool:
    """Check if a KV cache group contains Mamba layers."""
    spec = getattr(group, "kv_cache_spec", group)
    return is_mamba_spec(spec)


class MambaBlockPool(BlockPool):
    """The decoupled mamba pool, shared by every mamba kv-cache group.

    When there is more than one mamba group the coordinator mirrors them onto
    this pool's ids (see MirrorMambaBlockPool), and `primary_group_id` is the
    group those shared cache entries are keyed to.
    """

    def __init__(self, *args, primary_group_id: int | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        # When mamba groups are mirrored, every group's cache entry is keyed
        # to this one group id
        self.primary_group_id = primary_group_id
        # A one-slot buffer holding the block the
        # primary mamba group most recently pulled off the free queue
        self._last_allocation: list[KVCacheBlock] = []

    def replay_last_allocation(self, num_blocks: int) -> list[KVCacheBlock]:
        """Hand a mirrored group the ids the primary group just allocated."""
        if num_blocks != len(self._last_allocation):
            raise AssertionError(
                f"Mirrored mamba group asked for {num_blocks} blocks but the "
                f"primary group just allocated {len(self._last_allocation)}. "
                f"The groups are no longer in lockstep; mirroring would alias "
                f"unrelated state.")
        for block in self._last_allocation:
            block.ref_cnt += 1
        return list(self._last_allocation)

    def _canonical_group_ids(self, kv_cache_group_ids: list[int]) -> list[int]:
        if self.primary_group_id is None:
            return kv_cache_group_ids
        return [self.primary_group_id] * len(kv_cache_group_ids)

    def get_new_blocks(self, num_blocks: int) -> list[KVCacheBlock]:
        blocks = super().get_new_blocks(num_blocks)
        self._last_allocation = list(blocks)
        logger.info(
            "[KV_TRACE:MAMBA_POOL:ALLOC] gid=%s requested=%d allocated_bids=%s free_remaining=%d/%d",
            self.primary_group_id, num_blocks, [b.block_id for b in blocks],
            self.get_num_free_blocks(), len(self.blocks))
        return blocks

    def free_blocks(self, ordered_blocks: Iterable[KVCacheBlock]) -> None:
        ordered_list = list(ordered_blocks)
        bids = [b.block_id for b in ordered_list]
        super().free_blocks(ordered_list)
        logger.info(
            "[KV_TRACE:MAMBA_POOL:FREE] gid=%s freed_bids=%s free_now=%d/%d",
            self.primary_group_id, bids, self.get_num_free_blocks(), len(self.blocks))

    def _maybe_evict_cached_block(self, block: KVCacheBlock) -> bool:
        had_hash = block.block_hash is not None or bool(self.cached_block_hashes_by_block.get(block.block_id))
        evicted = super()._maybe_evict_cached_block(block)
        if evicted or had_hash:
            logger.warning(
                "[KV_TRACE:MAMBA_POOL:EVICT_FROM_CACHE] bid=%d had_hash=%s evicted=%s",
                block.block_id, had_hash, evicted)
        return evicted

    def get_cached_block(self, block_hash: BlockHash,
                         kv_cache_group_ids: list[int]):
        res = super().get_cached_block(
            block_hash, self._canonical_group_ids(kv_cache_group_ids))
        logger.info(
            "[KV_TRACE:MAMBA_POOL:LOOKUP] hash=%s gids=%s -> hit_bids=%s",
            block_hash, kv_cache_group_ids, [b.block_id for b in res] if res else None)
        return res


class MirrorMambaBlockPool:
    """A secondary mamba group's view of the primary group's block pool.
    """

    def __init__(self, primary: MambaBlockPool):
        self._primary = primary

    def __getattr__(self, name: str) -> Any:
        return getattr(self._primary, name)

    def get_new_blocks(self, num_blocks: int) -> list[KVCacheBlock]:
        return self._primary.replay_last_allocation(num_blocks)

    def cache_full_blocks(self, *args, **kwargs) -> None:
        # The primary group cached these blocks under the shared key already
        return None


class TPUMambaManager(MambaManager):
    """A Mamba manager for TPU that only caches checkpoints the GDN kernel wrote.

    The GDN kernel emits exactly one checkpoint per forward pass, at
    `(seq_len - 1) // block_size`. Intermediate blocks allocated during
    a chunked prefill are never written by the kernel and still hold uninitialized
    memory.
    """

    def __init__(
        self,
        kv_cache_spec: MambaSpec,
        block_pool: BlockPool,
        **kwargs,
    ) -> None:
        # Compatibility across vLLM versions: SingleTypeKVCacheManager.__init__
        # does not take arbitrary kwargs (e.g. max_in_flight_tokens, max_model_len).
        kwargs.pop("max_in_flight_tokens", None)
        kwargs.pop("max_model_len", None)
        valid_params = inspect.signature(
            SingleTypeKVCacheManager.__init__).parameters
        has_var_keyword = any(
            p.kind == inspect.Parameter.VAR_KEYWORD
            for p in valid_params.values())
        if not has_var_keyword:
            kwargs = {k: v for k, v in kwargs.items() if k in valid_params}
        super().__init__(kv_cache_spec, block_pool, **kwargs)

    def cache_blocks(
        self,
        request: Request,
        num_tokens: int,
        retention_interval: int | None = None,
        *,
        replay_boundaries: Sequence[int] = (),
        **kwargs,
    ) -> None:
        if num_tokens <= 0:
            return

        if self.mamba_cache_mode != "align":
            super().cache_blocks(
                request,
                num_tokens,
                retention_interval=retention_interval,
            )
            return

        if not self.kv_cache_spec.prefix_cacheable:
            return

        num_cached_blocks = self.num_cached_block.get(request.request_id, 0)
        num_full_blocks = num_tokens // self.block_size

        all_bids = [b.block_id for b in self.req_to_blocks.get(request.request_id, [])]
        if num_cached_blocks < num_full_blocks:
            # The GDN kernel writes exactly one checkpoint per forward pass, at
            # token `num_tokens - 1`. Only this block holds valid state;
            # intermediate blocks were never checkpointed.
            written_block_idx = (num_tokens - 1) // self.block_size
            block_mask = [
                (num_cached_blocks + i) == written_block_idx
                for i in range(num_full_blocks - num_cached_blocks)
            ]
            logger.info(
                "[KV_TRACE:MAMBA_MGR:CACHE] req=%s gid=%d tokens=%d cached_blks=%d full_blks=%d written_idx=%d mask=%s all_bids=%s",
                request.request_id, self.kv_cache_group_id, num_tokens, num_cached_blocks,
                num_full_blocks, written_block_idx, block_mask, all_bids)

            self.block_pool.cache_full_blocks(
                request=request,
                blocks=self.req_to_blocks[request.request_id],
                num_cached_blocks=num_cached_blocks,
                num_full_blocks=num_full_blocks,
                block_size=self.block_size,
                kv_cache_group_id=self.kv_cache_group_id,
                block_mask=block_mask,
            )

            blocks = self.req_to_blocks[request.request_id]
            for idx in range(num_cached_blocks, num_full_blocks):
                block = blocks[idx]
                if block.is_null or block.block_hash is None:
                    continue
                self.cached_blocks_this_step.add(block.block_hash)
                if block.block_hash_num_tokens is not None:
                    self._pending_boundary_state_offloads.append(
                        (
                            request.request_id,
                            self.kv_cache_group_id,
                            block,
                            block.block_hash_num_tokens,
                        )
                    )

            self.num_cached_block[request.request_id] = num_full_blocks
        else:
            logger.info(
                "[KV_TRACE:MAMBA_MGR:NO_NEW_CACHE] req=%s gid=%d tokens=%d cached_blks=%d full_blks=%d all_bids=%s",
                request.request_id, self.kv_cache_group_id, num_tokens, num_cached_blocks,
                num_full_blocks, all_bids)

        partial_hash = self._cache_partial_tail_block(request, num_tokens)
        if partial_hash is not None:
            self.cached_blocks_this_step.add(partial_hash)


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
        logger.info(
            "[KV_TRACE:DUAL_POOL:FREE] freeing attn_bids=%s mamba_bids=%s",
            [b.block_id for b in attn_blocks], [b.block_id for b in mamba_blocks])
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
        logger.warning("[KV_TRACE:DUAL_POOL:EVICT] block_ids=%s", block_ids)
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
            mamba_num_blocks = getattr(kv_cache_config, "mamba_num_blocks",
                                       None)
        if mamba_num_blocks is None:
            mamba_num_blocks = get_mamba_num_blocks()
        if mamba_num_blocks is None:
            raise ValueError(
                "[TPUHybridKVCacheCoordinator] mamba_num_blocks must be "
                "registered via set_mamba_num_blocks() or carried by "
                "kv_cache_config; the model runner's pool size never reached "
                "this process.")
        self.mamba_num_blocks = int(mamba_num_blocks)

        logger.info(
            "[TPUHybridKVCacheCoordinator] Initializing dual block pools: "
            "attn_blocks=%d, mamba_blocks=%d (mamba_groups=%s)",
            self.attention_block_pool.num_gpu_blocks,
            self.mamba_num_blocks,
            sorted(self.mamba_group_ids),
        )
        self.mirror_mamba_groups = len(self.mamba_group_ids) > 1
        self.primary_mamba_group_id = min(self.mamba_group_ids)

        # Allocate dedicated Mamba block pool
        self.mamba_block_pool = MambaBlockPool(
            num_gpu_blocks=self.mamba_num_blocks,
            enable_caching=self.enable_caching,
            hash_block_size=self.hash_block_size,
            enable_kv_cache_events=self.attention_block_pool.
            enable_kv_cache_events,
            metrics_collector=self.attention_block_pool.metrics_collector,
            primary_group_id=(self.primary_mamba_group_id
                              if self.mirror_mamba_groups else None),
        )
        self._mirror_pool = (MirrorMambaBlockPool(self.mamba_block_pool)
                             if self.mirror_mamba_groups else None)
        if self.mirror_mamba_groups:
            logger.info(
                "[TPUHybridKVCacheCoordinator] Mirroring %d mamba groups onto "
                "group %d's block ids: the %d-block pool now holds %d "
                "checkpoints instead of %d.", len(self.mamba_group_ids),
                self.primary_mamba_group_id, self.mamba_num_blocks,
                self.mamba_num_blocks,
                self.mamba_num_blocks // len(self.mamba_group_ids))

        # Unify null_block instance across both pools so TPUDualBlockPool,
        # attention pool, and Mamba managers share the exact same null block.
        self.mamba_block_pool.null_block = self.attention_block_pool.null_block

        # Re-bind Mamba managers to mamba_block_pool
        new_managers = list(self.single_type_managers)
        for i in self.mamba_group_ids:
            old_mgr = self.single_type_managers[i]
            pool = self.mamba_block_pool
            if self.mirror_mamba_groups and i != self.primary_mamba_group_id:
                pool = self._mirror_pool
            spec = kv_cache_config.kv_cache_groups[i].kv_cache_spec
            manager_kwargs = dict(
                block_pool=pool,
                enable_caching=self.enable_caching,
                kv_cache_group_id=i,
                dcp_world_size=getattr(self, "dcp_world_size", 1),
                pcp_world_size=getattr(self, "pcp_world_size", 1),
                scheduler_block_size=self.scheduler_block_size,
                needs_kv_cache_zeroing=self.kv_cache_config.
                needs_kv_cache_zeroing,
            )
            if is_mamba_spec(spec):
                # Use TPU-specific MambaManager that only indexes written checkpoints
                manager = TPUMambaManager(
                    kv_cache_spec=spec,
                    **manager_kwargs,
                )
                logger.info(
                    "[TPUHybridKVCacheCoordinator] group %d uses "
                    "checkpoint-aware TPUMambaManager", i)
            else:
                max_in_flight = getattr(old_mgr, "max_in_flight_tokens", 128)
                manager = get_manager_for_kv_cache_spec(
                    kv_cache_spec=spec,
                    max_in_flight_tokens=max_in_flight,
                    max_model_len=self.max_model_len,
                    **manager_kwargs,
                )
            new_managers[i] = manager
        self.single_type_managers = tuple(new_managers)

        for i in self.mamba_group_ids:
            expected = self.mamba_block_pool
            if self.mirror_mamba_groups and i != self.primary_mamba_group_id:
                expected = self._mirror_pool
            assert self.single_type_managers[i].block_pool is expected, (
                f"Manager {i} block_pool was not re-bound to the mamba pool!")
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
        cache_hit_blocks, hit_length, num_uncached = super().find_longest_cache_hit(
            block_hashes, max_cache_hit_length)
        logger.info(
            "[KV_TRACE:COORD:CACHE_HIT] hashes=%d max_len=%d -> hit_len=%d uncached=%d hit_blks=%s",
            len(block_hashes), max_cache_hit_length, hit_length, num_uncached,
            [[b.block_id for b in blist] for blist in cache_hit_blocks])
        return cache_hit_blocks, hit_length, num_uncached

    def free(self, request_id: str) -> None:
        all_blocks = self.get_blocks(request_id)
        if any(all_blocks):
            all_bids = [[b.block_id for b in blist] for blist in all_blocks]
            logger.info("[KV_TRACE:COORD:FREE_REQ] req=%s freed_blocks_per_group=%s",
                        request_id, all_bids)
        super().free(request_id)

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
                # Mirrored groups reuse the primary's ids
                if (self.mirror_mamba_groups
                        and i != self.primary_mamba_group_id):
                    continue
                mamba_blocks_needed += needed
            else:
                attn_blocks_needed += needed

        avail_attn = (self.attention_block_pool.get_num_free_blocks() -
                      reserved_blocks)
        avail_mamba = self.mamba_block_pool.get_num_free_blocks()
        can_fit = (attn_blocks_needed + watermark_blocks <= avail_attn) and (mamba_blocks_needed <= avail_mamba)
        if not can_fit or mamba_blocks_needed > 0:
            logger.info(
                "[KV_TRACE:COORD:CAN_ALLOC] req=%s tokens=%d attn_need=%d (avail=%d) mamba_need=%d (avail=%d) -> can_fit=%s",
                request.request_id, num_tokens, attn_blocks_needed, avail_attn, mamba_blocks_needed, avail_mamba, can_fit)
        if attn_blocks_needed + watermark_blocks > avail_attn:
            return False
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

        logger.info(
            "[KV_TRACE:COORD:ALLOC_SLOTS] req=%s status=%s new_tokens=%d computed_tokens=%d new_computed_blks=%s",
            request.request_id, request.status, num_new_tokens, num_new_computed_tokens,
            [[b.block_id for b in blist] for blist in new_computed_block_list] if new_computed_block_list else [])
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
        all_blocks = self.coordinator.get_blocks(request.request_id)
        logger.info(
            "[KV_TRACE:COORD:ALLOC_RESULT] req=%s new_blocks_per_group=%s all_blocks_per_group=%s",
            request.request_id,
            [[b.block_id for b in blist] for blist in new_blocks],
            [[b.block_id for b in blist] for blist in all_blocks])

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


def propagate_mamba_num_blocks(rpc_owner: Any, kv_cache_config: Any,
                               vllm_config: Any) -> int | None:
    """Fetch the mamba pool size the workers allocated (via
    `rpc_owner.collective_rpc`) and publish it in the engine-core process.
    Returns None when the model has no mamba layers."""
    if not any(is_mamba_group(g) for g in kv_cache_config.kv_cache_groups):
        return None
    reported = [
        v for v in rpc_owner.collective_rpc("get_mamba_num_blocks")
        if v is not None
    ]
    if not reported:
        raise ValueError(
            "[tpu_inference] No worker reported a compact mamba pool size "
            "for a model with mamba layers; the scheduler cannot size its "
            "mamba block pool.")
    if len(set(reported)) != 1:
        raise ValueError(
            "[tpu_inference] Workers disagree on the compact mamba pool "
            f"size: {reported}. The scheduler cannot size its mamba block "
            "pool consistently.")
    mamba_num_blocks = int(reported[0])
    vllm_config.cache_config.mamba_num_blocks = mamba_num_blocks
    kv_cache_config.mamba_num_blocks = mamba_num_blocks
    set_mamba_num_blocks(mamba_num_blocks)
    logger.info(
        "[tpu_inference] mamba_num_blocks=%d propagated from %d worker(s) "
        "to the engine core.", mamba_num_blocks, len(reported))
    return mamba_num_blocks


class MambaPoolSyncExecutorMixin:
    """Publish the workers' allocated mamba pool size in the engine-core
    process as soon as the caches exist: `initialize_from_config` runs there
    after every worker has allocated and before the scheduler is built."""

    def initialize_from_config(self, kv_cache_configs: Any) -> None:
        super().initialize_from_config(kv_cache_configs)
        if kv_cache_configs:
            propagate_mamba_num_blocks(self, kv_cache_configs[0],
                                       self.vllm_config)


def maybe_install_hybrid_coordinator_hooks(vllm_config: Any) -> None:
    """Install the hooks when mamba prefix caching (align mode) is on. Called
    from the executors' `_init_executor`, which runs in the engine-core
    process; the platform config hook does not."""
    cache_config = vllm_config.cache_config
    if (cache_config.enable_prefix_caching
            and getattr(cache_config, "mamba_cache_mode", "none") == "align"):
        install_hybrid_coordinator_hooks(vllm_config)


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
