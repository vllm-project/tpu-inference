# Copyright 2025 Google LLC
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
"""Lowers this worker's KV cache geometry into Raiden's byte-span pool IR.

Raiden's reshard pipeline is geometry-neutral: the controller validates and
re-keys byte spans, but it never derives them. Every mapping from "my mesh,
my page size, my head shard" to bytes is the caller's job -- see
``tpu_raiden/api/torch/pool_layout.py``:

    callers (e.g. the TPU vLLM connector) derive them from their own model
    and kernel layouts.

This module is that derivation. It produces two things:

  * :func:`build_pool_manifest` -- the static, per-worker declaration of
    where the live KV bytes sit inside each attention page. Registered once
    at worker init via ``register_work_unit(pool_manifest=...)``.
  * :func:`build_request_spans` -- the per-request, per-(src rank, dst rank)
    byte spans that say which of *my* bytes land where in *your* pages.
    Registered per request via ``register_request_blocks(pool_spans=...)``.

Because the two sides may run different tensor-parallel degrees and
different page sizes, neither side's page layout is assumed by the other.
The only thing that must agree is :func:`layout_fingerprint`, which covers
the model-invariant parts and deliberately excludes TP degree and page size
-- those are precisely what differ.

Scope:

  * Attention (``fa``) pools only. MLA and Mamba/GDN state pools raise.
  * No context parallelism (``KV_CONTEXT`` mesh product must be 1).
  * ``BATCH`` unsharded, so a rank's local page is dense and contiguous.
"""

from __future__ import annotations

import dataclasses
import hashlib
import math
from typing import Any, List, Optional, Sequence, Tuple

from tpu_raiden.api.torch.pool_layout import PoolSpec, RegionSpec
from tpu_sync.rpc.raiden_controller import RaidenId

from tpu_inference.logger import init_logger

logger = init_logger(__name__)

# Raiden pool tag for flash-attention KV pages. Must match on both sides;
# the controller filters plans by tag and treats it as an opaque string.
KV_POOL_TAG = "fa"

# `data_name` for every KV work unit. The controller keys pool identity on
# (tag, dtype_tag), so this is only a human-readable discriminator.
KV_DATA_NAME = "kv.fa"


@dataclasses.dataclass(frozen=True)
class PoolByteSpan:
    """One (source -> destination) byte range of a rank's declaration.

    Mirrors ``PoolByteSpan`` in Raiden's ``controller_service.proto``; field
    names match so the encoder can copy them across without a mapping table.
    A ``count`` above one repeats the range uniformly, advancing each side by
    its own stride, which is how a head slice interleaved through a
    token-major page is expressed without one entry per token.
    """

    src_block_ordinal: int
    src_offset_bytes: int
    dst_block_index: int
    dst_offset_bytes: int
    size_bytes: int
    src_stride_bytes: int = 0
    dst_stride_bytes: int = 0
    count: int = 1


@dataclasses.dataclass(frozen=True)
class PoolSpanRegistration:
    """One rank's byte declaration for one pool tag, for one request.

    ``declared_bytes`` is cross-checked against the sum of the spans, and
    over all contributing ranks against the destination's byte coverage, so
    a rank that owns none of a destination's heads still registers with an
    empty span list rather than staying silent.
    """

    tag: str
    block_ids: Tuple[int, ...]
    spans: Tuple[PoolByteSpan, ...]
    declared_bytes: int
    # 0: ``dst_block_index`` indexes the transfer's destination block list and
    # ``dst_offset_bytes`` is page-local. 1 is destination-page-agnostic and
    # cannot express a strided head slice, so this lowering always emits 0.
    dst_space_version: int = 0


@dataclasses.dataclass(frozen=True)
class AttentionKVGeometry:
    """One worker's byte-level view of its attention KV cache.

    All ``*_bytes`` fields describe the worker's **local** shard, which is
    what Raiden actually reads and writes. The logical (unsharded) array is
    only used to derive them.

    The kernel page layout is token-major, head-minor::

        (num_blocks, page_tokens, head_groups, packing, padded_head_dim)
           dim 0        dim 1        dim 2 (sharded)  dim 3      dim 4

    so within one page the bytes run ``token 0 [all local head groups] |
    token 1 [...] | ...``. That ordering is why a TP4 source contributes an
    interleaved half of every destination token slot rather than one
    contiguous run -- see :func:`build_request_spans`.
    """

    num_layers: int
    num_blocks: int
    page_tokens: int
    # Global (unsharded) dim-2 extent, and this rank's slice of it.
    head_groups: int
    head_groups_local: int
    packing: int
    padded_head_dim: int
    dtype_bits: int
    # Position of this worker within the KV_HEAD mesh axis.
    transfer_rank: int
    transfer_parallelism: int
    dtype_tag: str

    def __post_init__(self):
        if self.transfer_parallelism <= 0:
            raise ValueError("transfer_parallelism must be positive")
        if not 0 <= self.transfer_rank < self.transfer_parallelism:
            raise ValueError(
                f"transfer_rank {self.transfer_rank} is out of range for "
                f"parallelism {self.transfer_parallelism}")
        if self.head_groups != self.head_groups_local * self.transfer_parallelism:
            raise ValueError(
                f"head_groups {self.head_groups} is not "
                f"{self.head_groups_local} x {self.transfer_parallelism}")

    @property
    def group_bytes(self) -> int:
        """Bytes of one head group for one token."""
        return self.packing * self.padded_head_dim * self.dtype_bits // 8

    @property
    def per_token_bytes(self) -> int:
        """Local bytes occupied by one token slot in one page."""
        return self.head_groups_local * self.group_bytes

    @property
    def block_live_bytes(self) -> int:
        """Local live bytes in one page."""
        return self.page_tokens * self.per_token_bytes

    @property
    def head_group_range(self) -> Tuple[int, int]:
        """This rank's half-open slice of the global head-group axis."""
        start = self.transfer_rank * self.head_groups_local
        return (start, start + self.head_groups_local)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def geometry_to_dict(geometry: AttentionKVGeometry) -> dict:
    """JSON-safe form, for carrying a rank's geometry in ``kv_transfer_params``.

    The destination is the side that builds the spans -- it is the only side
    that knows both geometries -- so the producer publishes its own geometry
    in the handshake and the consumer reconstitutes it here.
    """
    return dataclasses.asdict(geometry)


def geometry_from_dict(payload: dict) -> AttentionKVGeometry:
    """Inverse of :func:`geometry_to_dict`. Validates via ``__post_init__``."""
    fields = {f.name for f in dataclasses.fields(AttentionKVGeometry)}
    unknown = set(payload) - fields
    _require(not unknown, f"Unknown geometry fields in handshake: {unknown}")
    return AttentionKVGeometry(**payload)


def work_unit_id(role: str, rank: int, instance: str = "") -> RaidenId:
    """The Raiden work-unit identity of one KV shard.

    One unit per **device shard**, not per process. A JAX vLLM worker is a
    single process holding N chips, but Raiden's byte-span planner requires
    one data-plane endpoint per work unit
    (``raiden_controller.py``: "Pool reshard planning requires one data-plane
    endpoint per work unit") and always emits ``dst_shard_idx = 0``. So each
    local shard gets its own ``KVCacheManager``, its own endpoint, and its own
    unit -- which is also exactly the shape
    :func:`contributing_src_ranks` assumes.

    ``job_replica_id`` must be the rank rendered as an int-parseable string:
    when several single-shard sources feed one destination, the controller
    facade keys the push schedules by ``int(src_unit.job_replica_id)``.

    ``instance`` separates engines that share a role and a controller. The
    controller's work-unit registry is keyed by the whole id, so two decode
    engines of the same parallelism would otherwise register the same units
    and the second would take over the first's endpoints -- every push then
    lands on one engine and the other's loads never complete. Rank alone
    cannot separate them: both engines hold ranks 0..TP-1. Pass a tag that is
    stable across an engine's own processes and distinct between engines; see
    :func:`engine_instance_tag`.
    """
    _require(rank >= 0, f"rank must be non-negative, got {rank}")
    return RaidenId(
        job_name=role if not instance else f"{role}@{instance}",
        job_replica_id=str(int(rank)),
        data_name=KV_DATA_NAME,
        data_replica_idx=int(rank),
    )


def engine_instance_tag(host: str, transfer_port: int | str) -> str:
    """A per-engine tag for :func:`work_unit_id`.

    The data-plane host and base transfer port identify an engine uniquely --
    two engines on one host must already differ in port to coexist -- and both
    are read from the same configuration in the scheduler and the worker, so
    the two processes derive the same tag without having to exchange one.
    """
    return f"{host}:{transfer_port}"


def unit_to_dict(unit: RaidenId) -> dict:
    """JSON-safe form of a work-unit id, for the handshake."""
    return {
        "job_name": unit.job_name,
        "job_replica_id": unit.job_replica_id,
        "data_name": unit.data_name,
        "data_replica_idx": int(unit.data_replica_idx),
    }


def unit_from_dict(payload: dict) -> RaidenId:
    """Inverse of :func:`unit_to_dict`."""
    return RaidenId(
        job_name=str(payload["job_name"]),
        job_replica_id=str(payload["job_replica_id"]),
        data_name=str(payload["data_name"]),
        data_replica_idx=int(payload["data_replica_idx"]),
    )


def dst_req_id(req_id: str, dst_rank: int) -> str:
    """Per-destination-rank request id.

    The controller's registration store is keyed ``(req_id, unit)`` while a
    span set addresses exactly one destination's byte space, so a request
    fanning out to K decode ranks needs K distinct req_ids -- reusing one
    would have each destination's spans overwrite the last.
    """
    return f"{req_id}#d{int(dst_rank)}"


_DST_RANK_MASK = 0xFF


def dst_uuid(uuid: int, dst_rank: int) -> int:
    """Per-destination-rank plan uuid.

    A source registers one active plan per transfer, keyed by uuid, and
    rejects a repeat with ``ALREADY_EXISTS``. Whenever a source feeds more
    than one destination -- any geometry with ``src_parallelism <
    dst_parallelism``, and every source in a partial fan-in such as 4 -> 2 --
    that same source registers K plans, so the uuid has to distinguish them
    just as :func:`dst_req_id` distinguishes the registrations.

    The rank goes in the low bits rather than being mixed in, so a uuid stays
    recognisably one request's across its destinations. That spends 8 of the
    uuid's 50 bits, which only has to keep *in-flight* requests apart.
    """
    _require(0 <= dst_rank <= _DST_RANK_MASK,
             f"dst_rank {dst_rank} does not fit in {_DST_RANK_MASK:#x}")
    return (int(uuid) & ~_DST_RANK_MASK) | int(dst_rank)


def geometry_from_mesh(
    mesh: Any,
    *,
    num_blocks: int,
    block_size: int,
    num_kv_heads: int,
    head_dim: int,
    kv_dtype: Any,
    num_layers: int,
    transfer_rank: int,
    use_mla: bool = False,
) -> AttentionKVGeometry:
    """Derives the local byte geometry from a live mesh.

    Reuses the runner's own shape helper rather than recomputing the kernel
    layout, so a kernel change cannot silently desynchronise the manifest
    from the actual allocation.
    """
    # Imported lazily: this pulls vLLM and the attention kernels, which the
    # pure-arithmetic paths below do not need.
    from jax._src import dtypes

    from tpu_inference import utils
    from tpu_inference.layers.common.sharding import ShardingAxisName
    from tpu_inference.runner.kv_cache import get_kv_cache_shape_with_mesh
    from tpu_inference.utils import to_jax_dtype

    if use_mla:
        raise NotImplementedError(
            "MLA KV pools are out of scope: the latent cache has no KV_HEAD "
            "axis, so head-slice resharding is not defined for it")

    context_cnt = utils.get_mesh_shape_product(mesh,
                                               ShardingAxisName.KV_CONTEXT)
    _require(
        context_cnt == 1,
        f"Context parallelism (KV_CONTEXT={context_cnt}) is out of scope; a "
        "page would then hold a strided token subset")
    model_cnt = utils.get_mesh_shape_product(mesh, ShardingAxisName.KV_HEAD)

    jax_dtype = to_jax_dtype(kv_dtype)
    shape = get_kv_cache_shape_with_mesh(
        mesh=mesh,
        total_num_pages=num_blocks,
        block_size=block_size,
        actual_num_kv_heads=num_kv_heads,
        actual_head_dim=head_dim,
        kv_dtype=jax_dtype,
        use_mla=False,
    )
    _, page_tokens, head_groups, packing, padded_head_dim = shape
    _require(
        head_groups % model_cnt == 0,
        f"Head-group extent {head_groups} is not divisible by the KV_HEAD "
        f"mesh product {model_cnt}")

    return AttentionKVGeometry(
        num_layers=num_layers,
        num_blocks=num_blocks,
        page_tokens=int(page_tokens),
        head_groups=int(head_groups),
        head_groups_local=int(head_groups) // model_cnt,
        packing=int(packing),
        padded_head_dim=int(padded_head_dim),
        dtype_bits=int(dtypes.itemsize_bits(jax_dtype)),
        transfer_rank=int(transfer_rank),
        transfer_parallelism=int(model_cnt),
        dtype_tag=str(jax_dtype.__name__ if hasattr(jax_dtype, "__name__"
                                                    ) else jax_dtype),
    )


def layout_fingerprint(
    *,
    num_layers: int,
    num_kv_heads: int,
    head_dim: int,
    dtype_tag: str,
    use_mla: bool = False,
) -> str:
    """Stable identity of the parts of the layout that must agree.

    The controller rejects a transfer whose source and destination
    fingerprints differ (``raiden_controller.py``: "Layout fingerprint
    mismatch between source and destination"), before any worker RPC.

    Deliberately excluded: **tensor-parallel degree** and **page size**.
    Those are exactly the axes this work makes heterogeneous, so folding
    them in would reject every transfer we care about. Padding and packing
    are excluded too because they are functions of ``head_dim`` and the
    dtype, which are included.
    """
    payload = "|".join([
        "tpu-inference.kv.v1",
        f"layers={int(num_layers)}",
        f"kv_heads={int(num_kv_heads)}",
        f"head_dim={int(head_dim)}",
        f"dtype={dtype_tag}",
        f"mla={int(bool(use_mla))}",
    ])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]


def build_pool_manifest(geometry: AttentionKVGeometry) -> List[PoolSpec]:
    """One :class:`PoolSpec` per attention layer, in layer order.

    The controller matches pools **positionally** by ``(tag, dtype_tag)``
    against the destination manifest, so both sides must publish the same
    number of pools in the same order. Layer count is model-invariant, so
    that is compatible with differing TP and page size.

    A rank's local shard is dense -- ``BATCH`` is unsharded and there is no
    context parallelism -- so each block is one contiguous live region and
    ``block_stride_bytes == live bytes``.
    """
    per_token = geometry.per_token_bytes
    return [
        PoolSpec(
            tag=KV_POOL_TAG,
            storage_index=layer_idx,
            base_offset_bytes=0,
            block_stride_bytes=geometry.block_live_bytes,
            num_blocks=geometry.num_blocks,
            dtype_tag=geometry.dtype_tag,
            regions=(RegionSpec(
                name=KV_POOL_TAG,
                offset_bytes=0,
                stride_bytes=per_token,
                unit_bytes=per_token,
                num_units=geometry.page_tokens,
                units_per_stride=1,
            ), ),
        ) for layer_idx in range(geometry.num_layers)
    ]


def head_group_overlap(
    src: AttentionKVGeometry,
    dst: AttentionKVGeometry,
) -> Optional[Tuple[int, int]]:
    """Half-open global head-group range that ``src`` owes ``dst``.

    Returns ``None`` when the two ranks share no heads, i.e. this source
    contributes nothing to this destination and must be left out of the
    transfer's ``src_units`` entirely.
    """
    _require(
        src.head_groups == dst.head_groups,
        "Source and destination disagree on the total head-group extent "
        f"({src.head_groups} vs {dst.head_groups}). Alignment padding is "
        "computed on the *local* head count, so it does not always scale "
        "linearly with TP degree; this reshard is not well defined.")
    _require(
        src.group_bytes == dst.group_bytes,
        f"Head-group byte size differs ({src.group_bytes} vs "
        f"{dst.group_bytes}); dtype or head_dim padding disagree")

    src_start, src_end = src.head_group_range
    dst_start, dst_end = dst.head_group_range
    start = max(src_start, dst_start)
    end = min(src_end, dst_end)
    return (start, end) if start < end else None


def contributing_src_ranks(
    src_geometries: Sequence[AttentionKVGeometry],
    dst: AttentionKVGeometry,
) -> List[int]:
    """Indices of the source ranks that own any of ``dst``'s heads.

    Every transfer targets exactly one destination rank (Raiden's byte-span
    planner allows one shard per work unit and one destination unit per
    plan, so ``dst_shard_idx`` is always 0 and a TP2 decode is two separate
    work units). Passing a non-contributing source rank would leave a hole
    in that rank's declared coverage and fail the controller's
    exactly-once destination coverage check.
    """
    return [
        index for index, src in enumerate(src_geometries)
        if head_group_overlap(src, dst) is not None
    ]


def source_schedule_key(
    transfer_rank: int,
    transfer_parallelism: int,
    dst_parallelism: int,
) -> int:
    """The Raiden ``node_id`` a source rank must carry to be routed correctly.

    A pool-reshard push carries the sender's ``node_id`` in its wire header
    (``block_transport.cc``: ``header.remote_id = block_delegate_->node_id()``)
    and the receiver uses it, and only it, to pick which source's schedule to
    scatter the payload with (``kv_cache_manager_base.cc``:
    ``found_src_shard = sender_node_id``). The controller keys those schedules
    by the source's **ordinal among the ranks that actually contribute to this
    destination** (``raiden_controller.py``: ``src_schedule_keys = {unit:
    ordinal for ordinal, unit in enumerate(active_src_units)}``), not by its
    global ``transfer_rank``.

    So ``node_id`` must be that ordinal. Leaving it at Raiden's default of 0 is
    silently correct while exactly one source feeds each destination -- which
    is every symmetric geometry -- and silently
    *wrong* the moment two do: all the pushes resolve to the first source's
    schedule, so every rank's bytes land in the first rank's head slots and the
    rest of the destination keeps stale KV. Nothing errors; the plan validates,
    the byte totals are exact, and the model emits garbage.

    Under uniform tensor parallelism the ordinal is a property of the source
    rank alone, so it can be fixed at worker init: destination ``d`` is fed by
    the contiguous run of sources ``[d * P // D, (d+1) * P // D)`` when
    ``P >= D``, and by the single source ``d * P // D`` when ``P < D``. The
    ordering is derived here from :func:`contributing_src_ranks` rather than
    open-coded, so it cannot drift from the spans it has to agree with.
    """
    _require(
        0 <= transfer_rank < transfer_parallelism,
        f"transfer_rank {transfer_rank} out of range for "
        f"parallelism {transfer_parallelism}")
    _require(dst_parallelism > 0, "dst_parallelism must be positive")

    # Any extent both sides divide evenly routes identically to the real one,
    # because ownership is the rank's equal share of it.
    extent = transfer_parallelism * dst_parallelism // math.gcd(
        transfer_parallelism, dst_parallelism)
    src_geometries = [
        _routing_geometry(rank, transfer_parallelism, extent)
        for rank in range(transfer_parallelism)
    ]
    for dst_rank in range(dst_parallelism):
        dst = _routing_geometry(dst_rank, dst_parallelism, extent)
        contributors = contributing_src_ranks(src_geometries, dst)
        if transfer_rank in contributors:
            return contributors.index(transfer_rank)
    raise ValueError(
        f"Source rank {transfer_rank}/{transfer_parallelism} feeds no "
        f"destination rank of a {dst_parallelism}-way decode; the head-group "
        "extent is not divisible by both parallelisms")


def _routing_geometry(rank: int, parallelism: int,
                      extent: int) -> AttentionKVGeometry:
    """A geometry that carries only what head-group routing depends on.

    Head-group ownership is fixed by ``(transfer_rank, transfer_parallelism)``
    and the global extent; page size, dtype and layer count do not enter it.
    Using the smallest extent both parallelisms divide keeps this independent
    of the model, so a producer can compute its schedule key before it has
    ever seen the consumer's geometry.
    """
    return AttentionKVGeometry(
        num_layers=1,
        num_blocks=1,
        page_tokens=1,
        head_groups=extent,
        head_groups_local=extent // parallelism,
        packing=1,
        padded_head_dim=1,
        dtype_bits=16,
        transfer_rank=rank,
        transfer_parallelism=parallelism,
        dtype_tag="bfloat16",
    )


def build_request_span_entries(
    src: AttentionKVGeometry,
    dst: AttentionKVGeometry,
    *,
    src_block_ids: Sequence[int],
    num_tokens: int,
) -> List[PoolSpanRegistration]:
    """``register_request_blocks(pool_spans=...)`` payload for one rank pair.

    Empty when this source owes this destination nothing. Register it anyway:
    the transfer must list **every** source rank in ``src_units`` because the
    controller requires ``transfer_rank`` values contiguous from zero
    (``raiden_controller.py``: "Source transfer_rank values must be contiguous
    from zero"), and it raises "Missing producer block registration" for any
    listed unit without a registration under the transfer's uuid. Ranks that
    contribute nothing therefore register an empty span list and are dropped
    from ``active_src_units`` when the plan is built.

    Because the registration store is keyed ``(req_id, unit)`` and a set of
    spans addresses one destination's byte space, a request fanning out to
    several decode ranks needs **a distinct req_id (and uuid) per destination
    rank** -- otherwise the second registration overwrites the first.
    """
    if head_group_overlap(src, dst) is None:
        return []
    return [
        build_request_spans(src,
                            dst,
                            src_block_ids=src_block_ids,
                            num_tokens=num_tokens)
    ]


def build_request_spans(
    src: AttentionKVGeometry,
    dst: AttentionKVGeometry,
    *,
    src_block_ids: Sequence[int],
    num_tokens: int,
) -> PoolSpanRegistration:
    """Byte spans carrying one source rank's share of one request to one
    destination rank.

    Under pure tensor parallelism every rank holds *all* tokens for its own
    head slice, so source page ``o`` covers tokens
    ``[o * src.page_tokens, (o+1) * src.page_tokens)`` and destination page
    ``k`` covers ``[k * dst.page_tokens, (k+1) * dst.page_tokens)``. A run of
    consecutive tokens that stays inside one page on both sides becomes a
    single uniformly strided span.

    The spans are emitted in the **v0** (page-indexed) form, not the
    ``dst_space_version=1`` global-compact form. v1 requires every span to
    be a plain contiguous range with no repeats, which cannot express a head
    reshard: because pages are token-major, a TP4 source owns an interleaved
    slice of every destination token slot, not one contiguous run. v1 stays
    correct for the token-split (context-parallel) case, which this path is
    not.
    """
    _require(num_tokens > 0, "num_tokens must be positive")
    overlap = head_group_overlap(src, dst)
    if overlap is None:
        raise ValueError(
            f"Source rank {src.transfer_rank}/{src.transfer_parallelism} owns "
            f"no heads of destination rank "
            f"{dst.transfer_rank}/{dst.transfer_parallelism}; it must be "
            "excluded from src_units rather than declaring empty spans")
    overlap_start, overlap_end = overlap

    group_bytes = src.group_bytes
    # Bytes this pair moves per token, and where they sit in each token slot.
    width = (overlap_end - overlap_start) * group_bytes
    src_in_token = (overlap_start - src.head_group_range[0]) * group_bytes
    dst_in_token = (overlap_start - dst.head_group_range[0]) * group_bytes
    src_token_bytes = src.per_token_bytes
    dst_token_bytes = dst.per_token_bytes

    expected_blocks = -(-num_tokens // src.page_tokens)  # ceil
    _require(
        len(src_block_ids) == expected_blocks,
        f"Expected {expected_blocks} source block ids for {num_tokens} tokens "
        f"at page_tokens={src.page_tokens}, got {len(src_block_ids)}")

    spans: List[PoolByteSpan] = []
    token = 0
    while token < num_tokens:
        src_block, src_token_off = divmod(token, src.page_tokens)
        dst_block, dst_token_off = divmod(token, dst.page_tokens)
        run = min(
            num_tokens - token,
            src.page_tokens - src_token_off,
            dst.page_tokens - dst_token_off,
        )
        src_offset = src_token_off * src_token_bytes + src_in_token
        dst_offset = dst_token_off * dst_token_bytes + dst_in_token

        if width == src_token_bytes and width == dst_token_bytes:
            # Same head slice on both sides: the strided form degenerates to
            # one contiguous copy. Emitting it that way keeps the symmetric
            # TP-N -> TP-N path cheap (one entry per run, not one per token).
            spans.append(
                PoolByteSpan(
                    src_block_ordinal=src_block,
                    src_offset_bytes=src_offset,
                    dst_block_index=dst_block,
                    dst_offset_bytes=dst_offset,
                    size_bytes=width * run,
                ))
        else:
            spans.append(
                PoolByteSpan(
                    src_block_ordinal=src_block,
                    src_offset_bytes=src_offset,
                    dst_block_index=dst_block,
                    dst_offset_bytes=dst_offset,
                    size_bytes=width,
                    src_stride_bytes=src_token_bytes,
                    dst_stride_bytes=dst_token_bytes,
                    count=run,
                ))
        token += run

    return PoolSpanRegistration(
        tag=KV_POOL_TAG,
        block_ids=tuple(int(block_id) for block_id in src_block_ids),
        spans=tuple(spans),
        declared_bytes=num_tokens * width,
        dst_space_version=0,
    )
