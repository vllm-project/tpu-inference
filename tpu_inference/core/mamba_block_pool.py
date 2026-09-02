# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Dedicated block pools for mamba (GDN / linear-attention) KV cache groups.

Why
---
With prefix caching, hybrid GDN models run vLLM's mamba ``"align"`` mode,
which addresses recurrent state by block id. vLLM hands out block ids from
one pool to the attention group and to every mamba group, and on TPU every
layer owns its own array, so the uniform sizing must give *every* block id
a state slot in *every* mamba layer: an attention block id reserves (and
never touches) a mamba slot in all mamba layers, and a mamba id owned by
one group reserves slots in the other groups' layers too. For Qwen3.5
(15 attention + 45 GDN layers, block size 1024) each block id costs
~213 MiB of which ~183 MiB is mamba state that is ~95% idle, while a
running request only ever pins 2 mamba slots per group.

How
---
The worker (`runner/kv_cache_manager.py`) splits the KV budget into an
attention block pool and `S` mamba state slots per mamba layer, and hands
vLLM a `TPUMambaSpec` (a `MambaSpec` with ``num_blocks=S``). vLLM's spec
registry maps that spec to `TPUMambaManager`, which owns a private
`BlockPool` of `S` ids instead of the shared pool the coordinator passes
in. Mamba block ids therefore live in ``[0, S)`` per DP rank and index the
mamba arrays directly; the GDN op is unchanged.

Nothing in vLLM changes. The places where vLLM assumes a single pool are
handled inside the manager:

* admission: `get_num_blocks_to_allocate` reports the shared pool's size + 1
  (the sentinel `MambaManager` already uses) when the private pool cannot
  serve, so `KVCacheManager.allocate_slots` returns None and the scheduler
  waits or preempts;
* cache hits: the classmethod `find_longest_cache_hit` is called with the
  shared pool and a batch of group ids that share a spec; it looks each
  group up in its own private pool and reconciles the hit length;
* deferred frees: the scheduler returns popped blocks to the *shared* pool,
  so popped mamba blocks are kept here and released
  ``max_concurrent_batches + 1`` scheduler steps later, after any in-flight
  step that could still write them has completed;
* prefix-cache reset: there is no per-manager hook, so a reset of the shared
  pool (its hash map object is replaced) is detected at the next step and
  the private pool is reset too;
* partial-tail offload hand-offs and KV cache events are disabled for
  private pools (both would touch the shared pool with foreign blocks).
"""
from __future__ import annotations

import weakref
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass, fields
from typing import Any

from vllm.config import get_current_vllm_config_or_none
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_utils import KVCacheBlock
from vllm.v1.core.single_type_kv_cache_manager import MambaManager
from vllm.v1.kv_cache_interface import KVCacheSpec, MambaSpec
from vllm.v1.kv_cache_spec_registry import KVCacheSpecRegistry

from tpu_inference.logger import init_logger

logger = init_logger(__name__)


@dataclass(frozen=True, kw_only=True)
class TPUMambaSpec(MambaSpec):
    """`MambaSpec` whose group owns a dedicated block pool.

    ``num_blocks`` is the size of that pool for the scheduler that receives
    the spec (the DP scheduler divides it by the DP size per rank). ``None``
    keeps vLLM's behavior: the group shares the attention block pool.
    """
    num_blocks: int | None = None

    @classmethod
    def from_mamba_spec(cls, spec: MambaSpec,
                        num_blocks: int | None) -> "TPUMambaSpec":
        values = {f.name: getattr(spec, f.name) for f in fields(MambaSpec)}
        return cls(**values, num_blocks=num_blocks)


# shared pool -> {kv_cache_group_id: private pool}. `find_longest_cache_hit`
# is a classmethod that only receives the shared pool, so it finds the
# private pools through this map. Weak keys: pools die with their manager.
_PRIVATE_POOLS: "weakref.WeakKeyDictionary[BlockPool, dict[int, BlockPool]]" = (
    weakref.WeakKeyDictionary())


class TPUMambaManager(MambaManager):
    """`MambaManager` backed by a private block pool of `spec.num_blocks` ids."""

    def __init__(self, kv_cache_spec: KVCacheSpec, block_pool: BlockPool,
                 **kwargs: Any) -> None:
        self.shared_block_pool = block_pool
        num_blocks = getattr(kv_cache_spec, "num_blocks", None)
        self.has_private_pool = num_blocks is not None
        if self.has_private_pool:
            pool = BlockPool(
                num_gpu_blocks=num_blocks,
                enable_caching=kwargs.get("enable_caching", True),
                hash_block_size=block_pool.hash_block_size,
                enable_kv_cache_events=False,
            )
            _PRIVATE_POOLS.setdefault(block_pool,
                                      {})[kwargs.get("kv_cache_group_id",
                                                     0)] = pool
        else:
            pool = block_pool
        super().__init__(kv_cache_spec, pool, **kwargs)

        # Reset detection: `BlockPool.reset_prefix_cache` replaces this map.
        self._shared_hash_map = block_pool.cached_block_hash_to_block
        # Step-clocked deferred frees (see module docstring).
        self._step = 0
        vllm_config = get_current_vllm_config_or_none()
        max_in_flight = (getattr(vllm_config, "max_concurrent_batches", 1)
                         if vllm_config is not None else 1)
        self._free_delay_steps = max(int(max_in_flight), 1) + 1
        self._deferred_frees: deque[tuple[int, list[KVCacheBlock]]] = deque()

    # ------------------------------------------------------------ lookups
    @classmethod
    def find_longest_cache_hit(
        cls,
        block_hashes,
        max_length: int,
        kv_cache_group_ids: list[int],
        block_pool: BlockPool,
        kv_cache_spec: KVCacheSpec,
        drop_eagle_block: bool,
        alignment_tokens: int,
        dcp_world_size: int = 1,
        pcp_world_size: int = 1,
    ):
        private_pools = _PRIVATE_POOLS.get(block_pool)
        if not private_pools or any(gid not in private_pools
                                    for gid in kv_cache_group_ids):
            return super().find_longest_cache_hit(
                block_hashes=block_hashes,
                max_length=max_length,
                kv_cache_group_ids=kv_cache_group_ids,
                block_pool=block_pool,
                kv_cache_spec=kv_cache_spec,
                drop_eagle_block=drop_eagle_block,
                alignment_tokens=alignment_tokens,
                dcp_world_size=dcp_world_size,
                pcp_world_size=pcp_world_size,
            )

        # Each group hits in its own pool; a mamba hit is a single state
        # block at one boundary, so shrink the candidate length until every
        # group agrees (mirrors the hybrid coordinator's fixed point).
        hit_length = max_length
        while True:
            per_group = [
                super(TPUMambaManager, cls).find_longest_cache_hit(
                    block_hashes=block_hashes,
                    max_length=hit_length,
                    kv_cache_group_ids=[gid],
                    block_pool=private_pools[gid],
                    kv_cache_spec=kv_cache_spec,
                    drop_eagle_block=drop_eagle_block,
                    alignment_tokens=alignment_tokens,
                    dcp_world_size=dcp_world_size,
                    pcp_world_size=pcp_world_size,
                ) for gid in kv_cache_group_ids
            ]
            new_hit_length = min(length for _, length in per_group)
            if new_hit_length == hit_length or new_hit_length == 0:
                break
            hit_length = new_hit_length
        if new_hit_length == 0:
            return tuple([] for _ in kv_cache_group_ids), 0
        return tuple(blocks[0] for blocks, _ in per_group), new_hit_length

    # --------------------------------------------------------- admission
    def get_num_blocks_to_allocate(
        self,
        request_id: str,
        num_tokens: int,
        new_computed_blocks: Sequence[KVCacheBlock],
        total_computed_tokens: int,
        num_local_computed_tokens: int,
        num_tokens_main_model: int,
        apply_admission_cap: bool = False,
    ) -> int:
        num_blocks = super().get_num_blocks_to_allocate(
            request_id,
            num_tokens,
            new_computed_blocks,
            total_computed_tokens,
            num_local_computed_tokens,
            num_tokens_main_model,
            apply_admission_cap=apply_admission_cap,
        )
        if not self.has_private_pool:
            return num_blocks
        if num_blocks > self.block_pool.get_num_free_blocks():
            # The coordinator checks the sum against the *shared* pool, so
            # report more than it can ever have: the request is not admitted
            # (or a running request is preempted) until mamba slots free up.
            return self.shared_block_pool.num_gpu_blocks + 1
        # Served from the private pool: costs the shared pool nothing.
        return 0

    # -------------------------------------------------------------- frees
    def pop_blocks_for_free(self, request_id: str) -> list[KVCacheBlock]:
        blocks = super().pop_blocks_for_free(request_id)
        if not self.has_private_pool:
            return blocks
        # The scheduler would return these to the shared pool. Keep them
        # until every step that may still write them has completed.
        if blocks:
            self._deferred_frees.append(
                (self._step + self._free_delay_steps, blocks))
        return []

    def free(self, request_id: str) -> None:
        if not self.has_private_pool:
            super().free(request_id)
            return
        # Immediate free (the scheduler verified nothing is in flight).
        blocks = MambaManager.pop_blocks_for_free(self, request_id)
        self.block_pool.free_blocks(reversed(blocks))

    def _flush_deferred_frees(self, up_to_step: int | None = None) -> None:
        while self._deferred_frees and (up_to_step is None
                                        or self._deferred_frees[0][0]
                                        <= up_to_step):
            _, blocks = self._deferred_frees.popleft()
            self.block_pool.free_blocks(reversed(blocks))

    def new_step_starts(self) -> None:
        super().new_step_starts()
        if not self.has_private_pool:
            return
        self._step += 1
        self._flush_deferred_frees(self._step)
        shared_map = self.shared_block_pool.cached_block_hash_to_block
        if shared_map is not self._shared_hash_map:
            # The shared pool was reset (e.g. RL weight update): its used
            # block count was 1, so no request holds blocks and every
            # deferred free has passed its fence too.
            self._shared_hash_map = shared_map
            self._flush_deferred_frees()
            if not self.block_pool.reset_prefix_cache():
                logger.warning(
                    "Dedicated mamba block pool of group %d could not be "
                    "reset: %d blocks still in use.", self.kv_cache_group_id,
                    self.block_pool.num_gpu_blocks -
                    self.block_pool.get_num_free_blocks() - 1)

    # ----------------------------------------------- unsupported features
    def take_pending_partial_tail_offloads(self):
        if not self.has_private_pool:
            return super().take_pending_partial_tail_offloads()
        pending = super().take_pending_partial_tail_offloads()
        if pending:
            # `KVCacheManager` would pin these on the shared pool.
            logger.warning_once(
                "Dropping %d partial-tail offload hand-offs: not supported "
                "with dedicated mamba block pools.", len(pending))
        return []


def register_tpu_mamba_spec(vllm_config=None) -> None:
    """Map `TPUMambaSpec` to `TPUMambaManager` in vLLM's spec registry.

    vLLM fills the registry lazily and skips its built-ins once any spec is
    registered, so make sure they are in first (also reached through
    `TpuPlatform.register_custom_kv_cache_specs`). Re-registering the same
    pair is a no-op.
    """
    ensure_registered = getattr(KVCacheSpecRegistry, "_ensure_registered",
                                None)
    if ensure_registered is not None:
        ensure_registered(vllm_config)
    KVCacheSpecRegistry.register(TPUMambaSpec,
                                 TPUMambaManager,
                                 uniform_type_base_spec=MambaSpec)
