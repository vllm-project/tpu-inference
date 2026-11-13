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
"""Offline validation of the KV byte-span lowering against a real controller.

No TPU and no model: the reshard control plane runs as a sidecar process, and
stand-in worker listeners record the plan it dispatches instead of moving
bytes. That plans a TP4-prefill -> TP2-decode reshard with *different page
sizes* on both sides, and the byte accounting is checked against the
controller's own arithmetic rather than against a second derivation of this
module's.

Everything below the plan tests is pure lowering and needs no control plane.
"""

import json
import math
import os
import shutil
import socket
import subprocess
import tempfile
import threading
import time

import pytest

raiden_controller = pytest.importorskip(
    "tpu_sync.rpc.raiden_controller",
    reason="tpu-raiden is not installed in this environment",
)
raiden_service_pb2 = pytest.importorskip(
    "tpu_sync.rpc.raiden_service_pb2",
    reason="tpu-raiden is not installed in this environment",
)

from tpu_inference.distributed import kv_pool_layout as kvpl  # noqa: E402
from tpu_inference.distributed.raiden_reshard_client import \
    ReshardRequestBlockClient  # noqa: E402

# Raiden serves the reshard control plane from a standalone binary that no
# wheel ships and that ``build.sh`` does not build by default, so a source
# checkout built with ``//tpu_sync/kv_cache/reshard:reshard_sidecar`` is what
# makes the plan tests below runnable.
_SIDECAR_BINARY = "reshard_sidecar"
_SIDECAR_ENV = "TPU_KV_RESHARD_SIDECAR"

# Qwen3-0.6B-shaped attention, truncated to a few layers to keep the plan
# small. bf16 => packing 2; head_dim 128 is already 128-aligned.
NUM_KV_HEADS = 8
HEAD_DIM = 128
PACKING = 2
DTYPE_BITS = 16
DTYPE_TAG = "bfloat16"
# align_to(NUM_KV_HEADS * 2, PACKING) // PACKING
HEAD_GROUPS = 8
GROUP_BYTES = PACKING * HEAD_DIM * DTYPE_BITS // 8  # 512
NUM_LAYERS = 4
NUM_BLOCKS = 64

FINGERPRINT = kvpl.layout_fingerprint(
    num_layers=NUM_LAYERS,
    num_kv_heads=NUM_KV_HEADS,
    head_dim=HEAD_DIM,
    dtype_tag=DTYPE_TAG,
)


def _recv_exactly(sock, count):
    buf = b""
    while len(buf) < count:
        chunk = sock.recv(count - len(buf))
        if not chunk:
            return None
        buf += chunk
    return buf


class _RecordingWorker:
    """A control-plane listener that records the plan the controller sends.

    The controller arms the receiver and fires each sender over the framed
    ``ControlRequest`` wire at the address the work unit registered, and
    returns once every worker has acked. Acking without touching a TPU is
    enough to get the whole plan out of it.
    """

    def __init__(self):
        self._server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._server.bind(("127.0.0.1", 0))
        self._server.listen(16)
        self._server.settimeout(0.2)
        self.address = f"127.0.0.1:{self._server.getsockname()[1]}"
        self.requests = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()

    def _serve(self):
        while not self._stop.is_set():
            try:
                conn, _ = self._server.accept()
            except socket.timeout:
                continue
            except OSError:
                return
            with conn:
                try:
                    header = _recv_exactly(conn, 4)
                    if header is None:
                        continue
                    payload = _recv_exactly(conn,
                                            int.from_bytes(header, "big"))
                    if payload is None:
                        continue
                    req = raiden_service_pb2.ControlRequest()
                    req.ParseFromString(payload)
                    self.requests.append(req)
                    body = raiden_service_pb2.ControlResponse(
                        success=True).SerializeToString()
                    conn.sendall(len(body).to_bytes(4, "big") + body)
                except OSError:
                    pass

    def close(self):
        self._stop.set()
        self._thread.join(timeout=5)
        self._server.close()

    @property
    def start_transfer(self):
        """The single START_TRANSFER this worker was sent."""
        commands = [
            req.start_transfer_request for req in self.requests
            if req.command == raiden_service_pb2.ControlRequest.
            COMMAND_START_TRANSFER
        ]
        assert len(commands) == 1, f"expected 1 START_TRANSFER, got {commands}"
        return commands[0]


def _find_sidecar_binary():
    """Env var, then $PATH, then the Bazel output of a Raiden checkout."""
    candidates = [os.environ.get(_SIDECAR_ENV), shutil.which(_SIDECAR_BINARY)]
    import tpu_sync  # Raiden is a namespace package, so __file__ is None.
    for entry in list(getattr(tpu_sync, "__path__", ())):
        candidates.append(
            os.path.join(os.path.dirname(entry), "bazel-bin", "tpu_sync",
                         "kv_cache", "reshard", _SIDECAR_BINARY))
    for candidate in candidates:
        if candidate and os.access(candidate, os.X_OK):
            return candidate
    return None


@pytest.fixture(scope="module")
def controller_address():
    """A reshard control plane, shared by every plan test in this module."""
    binary = _find_sidecar_binary()
    if binary is None:
        pytest.skip(
            f"No executable {_SIDECAR_BINARY}: build "
            f"//tpu_sync/kv_cache/reshard:{_SIDECAR_BINARY} in a Raiden "
            f"checkout or set ${_SIDECAR_ENV}")

    handle, ready_file = tempfile.mkstemp(prefix="reshard_", suffix=".ready")
    os.close(handle)
    os.unlink(ready_file)
    process = subprocess.Popen([
        binary, "--port=0", "--advertise-host=127.0.0.1",
        f"--ready-file={ready_file}"
    ])
    try:
        deadline = time.monotonic() + 60.0
        ready = None
        while ready is None and time.monotonic() < deadline:
            assert process.poll() is None, (
                f"sidecar exited with {process.returncode} before binding")
            try:
                with open(ready_file, "r") as f:
                    line = f.read().strip()
                if line:
                    ready = json.loads(line)
            except (FileNotFoundError, json.JSONDecodeError):
                time.sleep(0.05)
        assert ready is not None, "sidecar never published its ready file"
        yield ready["address"]
    finally:
        process.terminate()
        process.wait(timeout=30)
        if os.path.exists(ready_file):
            os.unlink(ready_file)


def _geometry(*, tp, rank, page_tokens, head_groups=HEAD_GROUPS):
    return kvpl.AttentionKVGeometry(
        num_layers=NUM_LAYERS,
        num_blocks=NUM_BLOCKS,
        page_tokens=page_tokens,
        head_groups=head_groups,
        head_groups_local=head_groups // tp,
        packing=PACKING,
        padded_head_dim=HEAD_DIM,
        dtype_bits=DTYPE_BITS,
        transfer_rank=rank,
        transfer_parallelism=tp,
        dtype_tag=DTYPE_TAG,
    )


class _Plan:
    """The dispatched plan, read back off the worker listeners.

    The receiver is sent every source's schedule keyed by that source's
    ordinal among the ranks feeding it; each sender is sent its own schedule
    under key 0. Both views come from the controller, so the fields below are
    its arithmetic, not a second copy of the lowering's.
    """

    def __init__(self, dst_worker, src_workers, src_units):
        receiver = dst_worker.start_transfer
        self.src_units = [
            _to_raiden_id(unit) for unit in receiver.src_units
        ]
        group = receiver.pool_groups[0]
        self.dst_device_block_ids = list(group.dst_device_block_ids)
        self.dst_expected_extent_bytes = list(group.dst_expected_extent_bytes)
        self.transfer_pool_indices = list(receiver.transfer_pool_indices)
        self.expected_block_count = receiver.expected_block_count

        # A sender that owns none of this destination's heads is dropped from
        # the plan and never dispatched.
        self.shard_push_schedules = {}
        for unit, worker in zip(src_units, src_workers):
            if not worker.requests:
                continue
            schedules = worker.start_transfer.shard_push_schedules
            self.shard_push_schedules[unit] = schedules

        # Recover the receiver's key for each sender by matching the schedule
        # it was armed with against the one that sender was fired with. That
        # key is what the receiver scatters a push under, so it is the field a
        # wrong source ordinal corrupts silently.
        self.src_schedule_keys = {}
        for unit, schedules in self.shard_push_schedules.items():
            blob = schedules[0].SerializeToString()
            for key, armed in receiver.shard_push_schedules.items():
                if armed.SerializeToString() == blob:
                    self.src_schedule_keys[unit] = key
                    break

    @property
    def entries(self):
        for schedules in self.shard_push_schedules.values():
            for schedule in schedules.values():
                yield from schedule.entries


def _to_raiden_id(proto):
    return raiden_controller.RaidenId(
        job_name=proto.job_name,
        job_replica_id=proto.job_replica_id,
        data_name=proto.data_name,
        data_replica_idx=proto.data_replica_idx,
    )


_REQUEST_COUNTER = [0]


def _plan_one_destination(
    controller_address,
    *,
    num_tokens,
    src_tp,
    dst_tp,
    src_page,
    dst_page,
    dst_rank,
    src_fingerprint=FINGERPRINT,
    dst_fingerprint=FINGERPRINT,
    workers_out=None,
):
    """Registers both meshes and drives the plan for one decode rank."""
    facade = raiden_controller.RaidenControllerClientFacade(controller_address)
    blocks = ReshardRequestBlockClient(controller_address)

    src_geoms = [
        _geometry(tp=src_tp, rank=r, page_tokens=src_page)
        for r in range(src_tp)
    ]
    dst_geoms = [
        _geometry(tp=dst_tp, rank=d, page_tokens=dst_page)
        for d in range(dst_tp)
    ]

    _REQUEST_COUNTER[0] += 1
    generation = _REQUEST_COUNTER[0]

    src_workers = [_RecordingWorker() for _ in src_geoms]
    dst_workers = [_RecordingWorker() for _ in dst_geoms]
    if workers_out is not None:
        workers_out.extend(src_workers + dst_workers)

    try:
        # The work-unit directory is keyed by RaidenId and the sidecar outlives
        # any one test, so the geometry goes in the id: a later test's TP2
        # source must not inherit a TP4 registration.
        src_units = []
        for rank, geometry in enumerate(src_geoms):
            unit = raiden_controller.RaidenId(
                job_name="prefill",
                job_replica_id=f"engine-tp{src_tp}p{src_page}-rank{rank}",
                data_name="kv.fa",
                data_replica_idx=0,
            )
            src_units.append(unit)
            facade.register_work_unit(
                unit,
                [f"10.0.0.{rank + 1}:8000"],
                control_plane_rpc_address=src_workers[rank].address,
                pool_manifest=kvpl.build_pool_manifest(geometry),
                layout_fingerprint=src_fingerprint,
                page_tokens=geometry.page_tokens,
                transfer_parallelism=geometry.transfer_parallelism,
                transfer_rank=geometry.transfer_rank,
            )

        dst_units = []
        for rank, geometry in enumerate(dst_geoms):
            unit = raiden_controller.RaidenId(
                job_name="decode",
                job_replica_id=f"engine-tp{dst_tp}p{dst_page}-rank{rank}",
                data_name="kv.fa",
                data_replica_idx=0,
            )
            dst_units.append(unit)
            facade.register_work_unit(
                unit,
                [f"10.1.0.{rank + 1}:8000"],
                control_plane_rpc_address=dst_workers[rank].address,
                pool_manifest=kvpl.build_pool_manifest(geometry),
                layout_fingerprint=dst_fingerprint,
                page_tokens=geometry.page_tokens,
                transfer_parallelism=geometry.transfer_parallelism,
                transfer_rank=geometry.transfer_rank,
            )

        # One (req_id, uuid) pair per destination rank: a registration is keyed
        # (req_id, unit) and its spans address a single destination's byte
        # space. The generation keeps tests from colliding in the registry.
        req_id = f"request-{generation}-d{dst_rank}"
        uuid = 1000 * generation + dst_rank
        src_blocks_per_rank = math.ceil(num_tokens / src_page)
        src_block_ids = [[
            10 + rank * src_blocks_per_rank + i
            for i in range(src_blocks_per_rank)
        ] for rank in range(src_tp)]

        for rank, geometry in enumerate(src_geoms):
            entries = kvpl.build_request_span_entries(
                geometry,
                dst_geoms[dst_rank],
                src_block_ids=src_block_ids[rank],
                num_tokens=num_tokens,
            )
            blocks.register_request_blocks(
                req_id,
                uuid,
                src_units[rank],
                src_block_ids[rank],
                pool_spans=entries,
            )

        dst_ids = list(range(40, 40 + math.ceil(num_tokens / dst_page)))
        facade.coordinate_transfer(
            src_units=src_units,
            dst_units=[dst_units[dst_rank]],
            req_id=req_id,
            use_block_chunks=True,
            is_sender=True,
            uuid=uuid,
            src_controller_address=controller_address,
            dst_mem_type=raiden_controller.RaidenMemoryType.HBM,
            dst_device_block_ids=dst_ids,
            num_tokens=num_tokens,
            transfer_pool_tags=[kvpl.KV_POOL_TAG],
        )
        plan = _Plan(dst_workers[dst_rank], src_workers, src_units)
    finally:
        if workers_out is None:
            for worker in src_workers + dst_workers:
                worker.close()
    return controller_address, dst_workers, src_units, dst_units, dst_ids, plan


def _scheduled_bytes(plan):
    """Sums the size field of every emitted shard-push entry."""
    return sum(entry.size_bytes * max(entry.count, 1)
               for entry in plan.entries)


# --- end-to-end plans --------------------------------------------------------


@pytest.mark.parametrize("dst_rank", [0, 1])
def test_tp4_to_tp2_differing_page_sizes_plans_with_exact_bytes(
        controller_address, dst_rank):
    """The end-to-end case: the plan builds and the byte accounting is exact."""
    num_tokens = 256
    _, _, src_units, _, dst_ids, plan = _plan_one_destination(
        controller_address,
        num_tokens=num_tokens,
        src_tp=4,
        dst_tp=2,
        src_page=128,
        dst_page=64,
        dst_rank=dst_rank,
    )

    dst_per_token = (HEAD_GROUPS // 2) * GROUP_BYTES  # 2048
    src_per_token = (HEAD_GROUPS // 4) * GROUP_BYTES  # 1024
    assert dst_per_token == 2 * src_per_token

    # Every byte of this decode rank's slice arrives exactly once.
    assert _scheduled_bytes(plan) == num_tokens * dst_per_token
    assert sum(plan.dst_expected_extent_bytes) == num_tokens * dst_per_token

    # Only the two prefill ranks owning this decode rank's heads participate;
    # the other two registered empty spans and were dropped.
    expected = [src_units[2 * dst_rank], src_units[2 * dst_rank + 1]]
    assert plan.src_units == expected

    assert plan.dst_device_block_ids == dst_ids
    assert plan.transfer_pool_indices == list(range(NUM_LAYERS))

    # One shard per work unit, so dst_shard_idx is a constant, not a route.
    for shards in plan.shard_push_schedules.values():
        assert list(shards) == [0]
        for schedule in shards.values():
            assert {entry.dst_shard_idx for entry in schedule.entries} == {0}


def test_both_decode_ranks_together_cover_the_whole_request(
        controller_address):
    """Summed over decode ranks, the plans move the full KV footprint."""
    num_tokens = 256
    total = sum(
        _scheduled_bytes(
            _plan_one_destination(controller_address,
                                  num_tokens=num_tokens,
                                  src_tp=4,
                                  dst_tp=2,
                                  src_page=128,
                                  dst_page=64,
                                  dst_rank=d)[5]) for d in range(2))
    assert total == num_tokens * HEAD_GROUPS * GROUP_BYTES


def test_partial_final_destination_page_is_allowed(controller_address):
    """num_tokens need not be a multiple of the destination page size."""
    num_tokens = 200  # 200 = 3*64 + 8, and 200 = 1*128 + 72
    _, _, _, _, dst_ids, plan = _plan_one_destination(
        controller_address,
        num_tokens=num_tokens,
        src_tp=4,
        dst_tp=2,
        src_page=128,
        dst_page=64,
        dst_rank=0,
    )
    assert len(dst_ids) == 4
    dst_per_token = (HEAD_GROUPS // 2) * GROUP_BYTES
    assert _scheduled_bytes(plan) == num_tokens * dst_per_token
    # Only the last page is partial.
    extents = plan.dst_expected_extent_bytes
    assert extents[:-1] == [64 * dst_per_token] * 3
    assert extents[-1] == 8 * dst_per_token


# --- negative and structural checks -----------------------------------------


def test_fingerprint_mismatch_is_rejected_before_any_worker_rpc(
        controller_address):
    workers = []
    try:
        with pytest.raises(RuntimeError, match="fingerprint"):
            _plan_one_destination(
                controller_address,
                num_tokens=256,
                src_tp=4,
                dst_tp=2,
                src_page=128,
                dst_page=64,
                dst_rank=0,
                dst_fingerprint="a-different-model",
                workers_out=workers,
            )
        # Nothing was armed and nothing was fired: planning rejects the pair
        # before a single worker hears about the transfer.
        assert [worker.requests for worker in workers] == [[]] * len(workers)
    finally:
        for worker in workers:
            worker.close()


def test_fingerprint_ignores_tp_degree_and_page_size():
    """The two axes this work makes heterogeneous must not be in it."""
    base = dict(num_layers=NUM_LAYERS,
                num_kv_heads=NUM_KV_HEADS,
                head_dim=HEAD_DIM,
                dtype_tag=DTYPE_TAG)
    assert kvpl.layout_fingerprint(**base) == FINGERPRINT
    assert kvpl.layout_fingerprint(**{**base, "num_layers": 5}) != FINGERPRINT
    assert kvpl.layout_fingerprint(**{**base, "head_dim": 64}) != FINGERPRINT
    assert kvpl.layout_fingerprint(**{
        **base, "dtype_tag": "float8_e4m3"
    }) != FINGERPRINT


def test_contributing_src_ranks_partition_by_head_range():
    src = [_geometry(tp=4, rank=r, page_tokens=128) for r in range(4)]
    dst = [_geometry(tp=2, rank=d, page_tokens=64) for d in range(2)]
    assert kvpl.contributing_src_ranks(src, dst[0]) == [0, 1]
    assert kvpl.contributing_src_ranks(src, dst[1]) == [2, 3]


def test_head_reshard_emits_strided_spans():
    """TP4 -> TP2 interleaves, so spans must carry count and strides."""
    src = _geometry(tp=4, rank=1, page_tokens=128)
    dst = _geometry(tp=2, rank=0, page_tokens=64)
    registration = kvpl.build_request_spans(src,
                                            dst,
                                            src_block_ids=[10, 11],
                                            num_tokens=256)
    assert registration.dst_space_version == 0
    assert registration.declared_bytes == 256 * (HEAD_GROUPS //
                                                 4) * GROUP_BYTES
    # Runs are bounded by both page sizes: 256 tokens / min(128, 64) = 4.
    assert len(registration.spans) == 4
    for span in registration.spans:
        assert span.count == 64
        assert span.size_bytes == (HEAD_GROUPS // 4) * GROUP_BYTES
        assert span.src_stride_bytes == (HEAD_GROUPS // 4) * GROUP_BYTES
        assert span.dst_stride_bytes == (HEAD_GROUPS // 2) * GROUP_BYTES
        # Rank 1 is the upper half of decode rank 0's token slot.
        assert span.dst_offset_bytes % span.dst_stride_bytes == (
            HEAD_GROUPS // 4) * GROUP_BYTES


def test_symmetric_geometry_coalesces_to_contiguous_spans():
    """Equal TP and page size must not pay the per-token entry cost."""
    src = _geometry(tp=2, rank=0, page_tokens=64)
    dst = _geometry(tp=2, rank=0, page_tokens=64)
    registration = kvpl.build_request_spans(src,
                                            dst,
                                            src_block_ids=[10, 11, 12, 13],
                                            num_tokens=256)
    assert len(registration.spans) == 4
    for span in registration.spans:
        assert span.count == 1
        assert span.src_stride_bytes == 0
        assert span.dst_stride_bytes == 0
        assert span.size_bytes == 64 * (HEAD_GROUPS // 2) * GROUP_BYTES


def test_symmetric_tp_with_differing_page_size_still_plans(controller_address):
    """Isolates the page-size axis from the TP axis."""
    num_tokens = 256
    _, _, src_units, _, _, plan = _plan_one_destination(
        controller_address,
        num_tokens=num_tokens,
        src_tp=2,
        dst_tp=2,
        src_page=128,
        dst_page=64,
        dst_rank=1,
    )
    assert plan.src_units == [src_units[1]]
    assert _scheduled_bytes(plan) == num_tokens * (HEAD_GROUPS //
                                                   2) * GROUP_BYTES


def test_nonlinear_alignment_padding_is_rejected():
    """F3: padding is computed on the local head count, so it need not scale."""
    src = _geometry(tp=4, rank=0, page_tokens=128, head_groups=8)
    dst = _geometry(tp=2, rank=0, page_tokens=64, head_groups=12)
    with pytest.raises(ValueError, match="total head-group extent"):
        kvpl.head_group_overlap(src, dst)


def test_non_contributing_rank_registers_no_spans():
    src = _geometry(tp=4, rank=3, page_tokens=128)
    dst = _geometry(tp=2, rank=0, page_tokens=64)
    assert kvpl.build_request_span_entries(src,
                                           dst,
                                           src_block_ids=[10, 11],
                                           num_tokens=256) == []
    with pytest.raises(ValueError, match="owns no heads"):
        kvpl.build_request_spans(src,
                                 dst,
                                 src_block_ids=[10, 11],
                                 num_tokens=256)


def test_pool_manifest_is_one_dense_pool_per_layer():
    geometry = _geometry(tp=4, rank=0, page_tokens=128)
    pools = kvpl.build_pool_manifest(geometry)
    assert len(pools) == NUM_LAYERS
    for layer_idx, pool in enumerate(pools):
        pool.validate()
        assert pool.tag == kvpl.KV_POOL_TAG
        assert pool.storage_index == layer_idx
        assert pool.dtype_tag == DTYPE_TAG
        # A local shard is dense: live bytes fill the whole block stride.
        assert pool.live_bytes_per_block == pool.block_stride_bytes
        assert pool.block_stride_bytes == 128 * (HEAD_GROUPS //
                                                 4) * GROUP_BYTES


# --- source schedule keys ----------------------------------------------------
#
# The receiver picks which source's schedule to scatter a push with using the
# sender's Raiden node_id alone, and the controller keys those schedules by the
# source's ordinal among the ranks feeding that destination. If the connector
# leaves node_id at Raiden's default of 0, every rank's bytes land in the first
# rank's head slots: the plan still validates and the byte totals are still
# exact, so nothing errors and the model emits garbage. These lock our
# derivation against the controller's own.


@pytest.mark.parametrize("dst_rank", [0, 1])
def test_schedule_key_matches_the_controller_tp4_to_tp2(
        controller_address, dst_rank):
    _, _, src_units, _, _, plan = _plan_one_destination(
        controller_address,
        num_tokens=256,
        src_tp=4,
        src_page=128,
        dst_tp=2,
        dst_page=64,
        dst_rank=dst_rank,
    )
    for rank, unit in enumerate(src_units):
        if unit not in plan.src_schedule_keys:
            continue  # contributes nothing to this destination
        assert plan.src_schedule_keys[unit] == kvpl.source_schedule_key(
            rank, 4, 2)


def test_schedule_key_matches_the_controller_tp4_to_tp1(controller_address):
    _, _, src_units, _, _, plan = _plan_one_destination(
        controller_address,
        num_tokens=256,
        src_tp=4,
        src_page=128,
        dst_tp=1,
        dst_page=64,
        dst_rank=0,
    )
    # Every rank contributes, so the ordinal is the global rank.
    assert [plan.src_schedule_keys[unit] for unit in src_units] == [0, 1, 2, 3]
    for rank, unit in enumerate(src_units):
        assert plan.src_schedule_keys[unit] == kvpl.source_schedule_key(
            rank, 4, 1)


@pytest.mark.parametrize("src_tp,dst_tp", [(1, 1), (2, 2), (4, 4), (4, 2),
                                           (4, 1), (2, 1), (1, 2), (2, 4)])
def test_schedule_key_is_the_ordinal_among_contributors(src_tp, dst_tp):
    """Derived the same way the controller derives it, for every shape."""
    src = [
        _geometry(tp=src_tp, rank=r, page_tokens=128) for r in range(src_tp)
    ]
    for dst_rank in range(dst_tp):
        dst = _geometry(tp=dst_tp, rank=dst_rank, page_tokens=64)
        contributors = kvpl.contributing_src_ranks(src, dst)
        for ordinal, rank in enumerate(contributors):
            assert kvpl.source_schedule_key(rank, src_tp, dst_tp) == ordinal


def test_symmetric_geometry_keeps_the_default_key():
    """One source per destination means the ordinal is always 0."""
    for tp in (1, 2, 4):
        for rank in range(tp):
            assert kvpl.source_schedule_key(rank, tp, tp) == 0


def test_dst_uuid_separates_a_requests_destinations():
    """A source keys its active plan by uuid, so one per destination rank.

    Reusing one uuid makes every source that feeds more than one destination
    reject all but the first with ALREADY_EXISTS -- the remaining ranks then
    wait forever on a push that was never armed.
    """
    base = 0x3_FFFF_FFFF_FFFF
    keys = {kvpl.dst_uuid(base, rank) for rank in range(8)}
    assert len(keys) == 8


def test_dst_uuid_stays_within_the_uuids_width():
    """`get_uuid` trims to 50 bits so vLLM's encoder and Go agree on it."""
    for base in (0, 1, (1 << 50) - 1):
        for rank in range(8):
            tagged = kvpl.dst_uuid(base, rank)
            assert 0 <= tagged <= max(base, (1 << 50) - 1)
            assert tagged.bit_length() <= 50


def test_dst_uuid_keeps_one_request_recognisable():
    """Only the low bits move, so sibling transfers stay visibly related."""
    base = 0x2_ABCD_EF01_2345
    for rank in range(4):
        assert kvpl.dst_uuid(base, rank) >> 8 == base >> 8


def test_dst_uuid_rejects_a_rank_it_cannot_encode():
    with pytest.raises(ValueError):
        kvpl.dst_uuid(1 << 40, 256)
