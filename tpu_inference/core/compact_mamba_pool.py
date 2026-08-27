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

By default native chunk-based Mamba caching remains enabled.
``TPU_MAMBA_CACHED_POSITIONS`` and ``TPU_MAMBA_AUTO_MM_CACHE_POSITIONS`` add
forced, hybrid-aligned scheduler boundaries so important states are
materialized alongside ordinary chunk endpoints. Set
``TPU_MAMBA_DISABLE_CHUNK_CACHE`` to retain only those selected positions. A
manual position inside an atomic multimodal placeholder is skipped for that
request; positions at either placeholder edge remain eligible.
"""

from __future__ import annotations

import hashlib
import inspect
from collections.abc import Mapping, Sequence
from functools import wraps
from typing import Any

from tpu_inference import envs
from tpu_inference.logger import init_logger

logger = init_logger(__name__)

_PRIVATE_POOLS_ATTR = "_tpu_compact_mamba_pools"
_CACHED_POSITIONS_ATTR = "_tpu_mamba_cached_positions"
_AUTO_CACHE_POSITIONS_ATTR = "_tpu_mamba_auto_cache_positions"
_DISABLE_CHUNK_CACHE_ATTR = "_tpu_mamba_disable_chunk_cache"
_ALIGNMENT_TOKENS_ATTR = "_tpu_mamba_alignment_tokens"
# Per-request guard so the content-safe decision trace logs once, not once per
# scheduling step. Set on the request object the first time it is composed.
_LOGGED_ATTR = "_tpu_mamba_decision_logged"


def _media_digest(feature: Any) -> str:
    """Content-safe fingerprint of a media span's identity, hashed so the log
    never carries the raw identifier.
    """
    identifier = getattr(feature, "identifier", None)
    if identifier is None:
        return "none"
    return hashlib.sha256(str(identifier).encode()).hexdigest()[:20]


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


def get_fixed_mamba_cached_positions() -> frozenset[int] | None:
    """Return configured Mamba boundary lengths, or ``None`` when unset."""
    positions = envs.TPU_MAMBA_CACHED_POSITIONS
    if not positions:
        return None
    if any(position <= 0 for position in positions):
        raise ValueError(
            "TPU_MAMBA_CACHED_POSITIONS must contain positive token positions, "
            f"got {positions}")
    return frozenset(positions)


def auto_mm_cache_positions(mm_features: Sequence[Any],
                            alignment_tokens: int) -> frozenset[int]:
    """Auto-detect block-aligned Mamba cache boundaries for a multimodal request.

    Yields up to two reusable boundaries so a later request can hit whichever
    prefix it shares:

    * ``text0``: ``offset // alignment_tokens * alignment_tokens``.
      ``offset`` is the media chunk's start index, which equals the text0 token
      count — i.e. the exclusive end of the leading text. Flooring keeps the
      boundary within the text region.
    * ``text0 + media0``: floor of the first contiguous media run's end to
      ``alignment_tokens``. This floor may land *inside* the media run; align
      mode supports stopping there (chunked MM input is required, so the Mamba
      state is materializable mid-placeholder), so partial-media caching is
      accepted in exchange for a reusable boundary.

    Returns an empty set when there is no media or no leading text (offset 0).
    The ``text0`` boundary is dropped when it floors to 0 (text shorter than one
    block); the ``text0 + media0`` boundary is dropped when it floors
    at/before the media's start (nothing of the media would be cached).

    TODO: only the first ``text0``/``text0 + media0`` pair is auto-detected
    today. Extend to chain later segments —
    ``text0 + media0 + text1 + media1 + ...`` — so more text/media prefix
    boundary becomes a reusable cache position. Until then, use
     ``TPU_MAMBA_CACHED_POSITIONS`` for further caching boundaries.
    """
    if not mm_features or alignment_tokens <= 0:
        return frozenset()
    # mm_features are ordered ascending by mm_position.offset (prompt order); no
    # consumer sorts them and multimodal_manager relies on it.
    first_offset = mm_features[0].mm_position.offset
    if first_offset <= 0:
        return frozenset()

    positions: set[int] = set()
    text0_boundary = first_offset // alignment_tokens * alignment_tokens
    if text0_boundary > 0:
        positions.add(text0_boundary)

    run_end = first_offset + mm_features[0].mm_position.length
    # Adjacent placeholders form one logical media run; extend across them so the
    # boundary is measured from the end of that run.
    for feature in mm_features[1:]:
        if feature.mm_position.offset != run_end:
            break
        run_end = feature.mm_position.offset + feature.mm_position.length
    pair_boundary = run_end // alignment_tokens * alignment_tokens
    if pair_boundary > first_offset:
        positions.add(pair_boundary)

    return frozenset(positions)


def mamba_auto_positions(obj: Any, request: Any) -> frozenset[int]:
    """Obtain auto-detected cache positions from the request.

    If ``TPU_MAMBA_AUTO_MM_CACHE_POSITIONS`` is set,
    return automatically detected cache positions from this request's media.
    If ``TPU_MAMBA_AUTO_MM_CACHE_POSITIONS`` is not set, return empty set.
    """
    if not getattr(obj, _AUTO_CACHE_POSITIONS_ATTR, False):
        return frozenset()
    alignment_tokens = getattr(obj, _ALIGNMENT_TOKENS_ATTR, 0)
    mm_features = getattr(request, "mm_features", None) or ()
    return auto_mm_cache_positions(mm_features, alignment_tokens)


def mamba_fixed_positions(obj: Any) -> frozenset[int] | None:
    """Obtain statically pinned cache positions.

    If ``TPU_MAMBA_CACHED_POSITIONS`` is set,
    return the positions configured through this env var.
    If not set, return ``None``.
    """
    return getattr(obj, _CACHED_POSITIONS_ATTR, None)


def _drop_mid_media_positions(positions: frozenset[int] | None,
                              request: Any) -> frozenset[int] | None:
    """Drop positions landing strictly inside an atomic media placeholder.

    A Mamba state cannot be materialized mid-placeholder, so such positions are
    not valid boundaries for this request; positions exactly at either edge are
    kept. ``None`` (native "every boundary") passes through untouched.
    """
    if positions is None:
        return None
    mm_features = getattr(request, "mm_features", None) or ()
    if not mm_features:
        return positions

    def is_inside_mm_placeholder(position: int) -> bool:
        return any(
            mm_feature.mm_position.offset < position <
            mm_feature.mm_position.offset + mm_feature.mm_position.length
            for mm_feature in mm_features)

    return frozenset(position for position in positions
                     if not is_inside_mm_placeholder(position))


def mamba_cache_positions(obj: Any, request: Any) -> frozenset[int] | None:
    """Obtain selected Mamba boundaries and optionally log the decision.

    These positions always force scheduler splits. They restrict cache
    insertion only when ``TPU_MAMBA_DISABLE_CHUNK_CACHE`` is enabled.
    """
    composed = _mamba_cache_positions_helper(obj, request)
    _log_cache_decision(obj, request, composed)
    return composed


def _mamba_cache_positions_helper(obj: Any,
                                  request: Any) -> frozenset[int] | None:
    """Helper function containing `mamba_cache_position` decision logic.

    * First try to detect automatically cache boundaries from request's text and media,
      block align and truncate mid-media if necessary.
    * then for rest of tokens to compute, fall back to use fixed cache positions,
      note fixed position does not support mid-media --  will skip such cache positions.
    """
    fixed = mamba_fixed_positions(obj)
    if not getattr(obj, _AUTO_CACHE_POSITIONS_ATTR, False):
        return _drop_mid_media_positions(fixed, request)
    auto = mamba_auto_positions(obj, request)
    if not auto:
        return _drop_mid_media_positions(fixed, request)
    if not fixed:
        return auto
    cutoff = max(auto)
    beyond = frozenset(position for position in fixed if position > cutoff)
    return auto | _drop_mid_media_positions(beyond, request)


def _describe_auto_detection(mm_features: Sequence[Any], alignment_tokens: int,
                             auto_on: bool) -> str:
    """Human-readable narration of the automatic cache position detection decision."""
    if not auto_on:
        return "auto off"
    if not mm_features:
        return "auto skip: no media spans"
    if alignment_tokens <= 0:
        return f"auto skip: non-positive alignment ({alignment_tokens})"
    first_offset = mm_features[0].mm_position.offset
    if first_offset <= 0:
        return f"auto skip: no prefix text, media first (first offset={first_offset})"

    parts = []
    text0 = first_offset // alignment_tokens * alignment_tokens
    parts.append(f"text0 offset {first_offset}->{text0}" if text0 >
                 0 else f"text0 dropped (offset {first_offset} floors to 0)")
    run_end = first_offset + mm_features[0].mm_position.length
    merged = 1
    for feature in mm_features[1:]:
        if feature.mm_position.offset != run_end:
            break
        run_end = feature.mm_position.offset + feature.mm_position.length
        merged += 1
    parts.append(f"media0 {merged} span(s) run_end={run_end}")
    pair = run_end // alignment_tokens * alignment_tokens
    if pair > first_offset:
        parts.append(f"text0+media0 {run_end}->{pair}" +
                     (" (inside media)" if pair < run_end else ""))
    else:
        parts.append(
            f"text0+media0 dropped ({pair}<=media start {first_offset})")
    return "auto: " + ", ".join(parts)


def _log_cache_decision(obj: Any, request: Any,
                        composed: frozenset[int] | None) -> None:
    """Config-gated, content-safe trace of a request's caching decision.

    Off unless ``TPU_MAMBA_LOG_AUTO_MM_CACHE_DECISIONS`` is set; logs once per request.
    Emits only metadata (offsets, lengths, positions, block size) and hashed
    media identities -- never raw prompt text, generated text, or media.
    """
    if not envs.TPU_MAMBA_LOG_AUTO_MM_CACHE_DECISIONS:
        return
    if getattr(request, _LOGGED_ATTR, False):
        return
    try:
        setattr(request, _LOGGED_ATTR, True)
    except (AttributeError, TypeError):
        pass

    auto_on = bool(getattr(obj, _AUTO_CACHE_POSITIONS_ATTR, False))
    alignment_tokens = getattr(obj, _ALIGNMENT_TOKENS_ATTR, 0)
    mm_features = getattr(request, "mm_features", None) or ()
    spans = (", ".join(
        "#%d off=%d len=%d id=%s" %
        (i, f.mm_position.offset, f.mm_position.length, _media_digest(f))
        for i, f in enumerate(mm_features)) or "none")
    fixed = mamba_fixed_positions(obj)

    selected_only = bool(getattr(obj, _DISABLE_CHUNK_CACHE_ATTR, False))
    logger.info(
        "mamba-cache: block=%d spans=[%s] | %s | auto=%s fixed=%s "
        "=> selected=%s chunk_cache=%s",
        alignment_tokens,
        spans,
        _describe_auto_detection(mm_features, alignment_tokens, auto_on),
        sorted(mamba_auto_positions(obj, request)),
        "native(None)" if fixed is None else sorted(fixed),
        "native(None)" if composed is None else sorted(composed),
        "disabled" if selected_only else "enabled",
    )


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
    get_fixed_mamba_cached_positions()


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
        # The compact pool (and its auto boundary detection) is align-mode only.
        # Never enable the feature outside align mode; surface the misconfig.
        if envs.TPU_MAMBA_AUTO_MM_CACHE_POSITIONS:
            logger.warning(
                "TPU_MAMBA_AUTO_MM_CACHE_POSITIONS is set but no align-mode "
                "Mamba cache group is present; the feature is disabled.")
        return

    fixed_cached_positions = get_fixed_mamba_cached_positions()
    alignment_tokens = coordinator.lcm_block_size
    if fixed_cached_positions is not None:
        misaligned_positions = sorted(position
                                      for position in fixed_cached_positions
                                      if position % alignment_tokens != 0)
        if misaligned_positions:
            raise ValueError(
                "TPU_MAMBA_CACHED_POSITIONS values must be multiples of the "
                f"hybrid cache alignment ({alignment_tokens} tokens), got "
                f"{misaligned_positions}")
    auto_mm_cache_positions = envs.TPU_MAMBA_AUTO_MM_CACHE_POSITIONS
    disable_chunk_cache = envs.TPU_MAMBA_DISABLE_CHUNK_CACHE

    setattr(scheduler, _CACHED_POSITIONS_ATTR, fixed_cached_positions)
    setattr(scheduler, _AUTO_CACHE_POSITIONS_ATTR, auto_mm_cache_positions)
    setattr(scheduler, _DISABLE_CHUNK_CACHE_ATTR, disable_chunk_cache)
    setattr(scheduler, _ALIGNMENT_TOKENS_ATTR, alignment_tokens)

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
        setattr(manager, _CACHED_POSITIONS_ATTR, fixed_cached_positions)
        setattr(manager, _AUTO_CACHE_POSITIONS_ATTR, auto_mm_cache_positions)
        setattr(manager, _DISABLE_CHUNK_CACHE_ATTR, disable_chunk_cache)
        setattr(manager, _ALIGNMENT_TOKENS_ATTR, alignment_tokens)

    if not private_pools:
        return

    setattr(coordinator, _PRIVATE_POOLS_ATTR, private_pools)
    # HybridKVCacheCoordinator.find_longest_cache_hit passes its main pool to
    # the Mamba classmethod. Tag it so that method can route each group lookup
    # to the corresponding private pool.
    setattr(main_pool, _PRIVATE_POOLS_ATTR, private_pools)
    # Native chunk caching uses unrestricted lookup. Selected-only mode can
    # restrict fixed-only lookup, while auto mode remains unrestricted because
    # its eligible positions vary per request.
    setattr(
        main_pool,
        _CACHED_POSITIONS_ATTR,
        (fixed_cached_positions
         if disable_chunk_cache and not auto_mm_cache_positions else None),
    )
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
    if auto_mm_cache_positions:
        logger.info(
            "Mamba prefix caching auto-targets the first text0/text0+media0 "
            "boundaries per request (alignment=%d tokens)%s; native chunk "
            "caching is %s",
            alignment_tokens,
            ("" if fixed_cached_positions is None else
             f"; also pinning {sorted(fixed_cached_positions)}"),
            "disabled" if disable_chunk_cache else "enabled",
        )
    elif fixed_cached_positions is not None:
        logger.info(
            "Mamba prefix caching adds token positions %s; native chunk "
            "caching is %s",
            sorted(fixed_cached_positions),
            "disabled" if disable_chunk_cache else "enabled",
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
    cached_positions: frozenset[int] | None = None,
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
        position = (index + 1) * block_size
        if cached_positions is not None and position not in cached_positions:
            continue
        if block_size != alignment_tokens and position % alignment_tokens != 0:
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
                cached_positions=getattr(main_pool, _CACHED_POSITIONS_ATTR,
                                         None),
                dcp_world_size=dcp_world_size,
                pcp_world_size=pcp_world_size,
            )

        mamba_manager_cls.find_longest_cache_hit = find_longest_cache_hit

    if not hasattr(mamba_manager_cls, "_tpu_orig_compact_mamba_cache_blocks"):
        original_cache_blocks = mamba_manager_cls.cache_blocks
        mamba_manager_cls._tpu_orig_compact_mamba_cache_blocks = original_cache_blocks
        supports_alignment_tokens = (
            "alignment_tokens"
            in inspect.signature(original_cache_blocks).parameters)

        @wraps(original_cache_blocks)
        def cache_blocks(self, request, num_tokens, alignment_tokens=None):
            cached_positions = mamba_cache_positions(self, request)
            selected_only = bool(
                getattr(self, _DISABLE_CHUNK_CACHE_ATTR, False))
            if not selected_only or cached_positions is None:
                if supports_alignment_tokens:
                    return original_cache_blocks(
                        self,
                        request,
                        num_tokens,
                        alignment_tokens=alignment_tokens,
                    )
                return original_cache_blocks(self, request, num_tokens)

            num_cached_blocks = self.num_cached_block.get(
                request.request_id, 0)
            num_full_blocks = num_tokens // self.block_size
            if num_cached_blocks >= num_full_blocks:
                return

            blocks = self.req_to_blocks[request.request_id]
            blocks_to_cache = list(blocks)
            for block_index in range(num_cached_blocks, num_full_blocks):
                position = (block_index + 1) * self.block_size
                selected = position in cached_positions
                if alignment_tokens is not None:
                    selected &= position % alignment_tokens == 0
                if not selected:
                    # Older vLLM BlockPool versions do not accept block_mask.
                    # A shallow list copy with nulls preserves the same sparse
                    # insertion semantics without mutating request ownership.
                    blocks_to_cache[block_index] = self._null_block

            self.block_pool.cache_full_blocks(
                request=request,
                blocks=blocks_to_cache,
                num_cached_blocks=num_cached_blocks,
                num_full_blocks=num_full_blocks,
                block_size=self.block_size,
                kv_cache_group_id=self.kv_cache_group_id,
            )
            self.num_cached_block[request.request_id] = num_full_blocks

            for block in blocks[num_cached_blocks:num_full_blocks]:
                if block.is_null or block.block_hash is None:
                    continue
                self.cached_blocks_this_step.add(block.block_hash)

        mamba_manager_cls.cache_blocks = cache_blocks

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
    from vllm.v1.core.kv_cache_coordinator import (
        HybridKVCacheCoordinator,
        KVCacheCoordinator,
    )
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
