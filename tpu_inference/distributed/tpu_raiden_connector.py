# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Proxy server routes the request to P with max_output_tokens=1

P workflow:
    P recives the request

    P scheduler checks if the prefill is full done in `request_finished()`
    If done:
        P puts the request-id in `scheduler_output.finished_req_ids`
            and puts the request in `scheduler_output.kv_connector_metadata.reqs_to_send`
        P responds the proxy server with `finished_req_ids` and the `kv_transfer_params`
        P worker gets `reqs_to_send` and runs async `_prepare_kv_and_wait()`
    Else:
        P schedules the prefill with multiple turns due to chunked-prefill.

    P worker checks if the request has been pulled by D
    If done:
        P worker puts the request-id in `done_sending()`
        P scheduler frees blocks for the requet in done sending.
    Else:
        P holds the blocks for the request until it's pulled by D

    (
        One scheduler step can finish:
            scheduler RUNNING -> connector reqs_to_send -> worker prefill -> output
        The waiting buffer will get freed after notified by D or expired.
    )

Proxy server recives the response from P and forwards it to D

D workflow:
    D recives the request

    D scheduler calculates the num of tokens needing to pull from P in `get_num_new_matched_tokens()`
    D checks if need to pull from P
    If true:
        D puts the request in `scheduler_output.kv_connector_metadata.reqs_to_load`
        D worker gets `reqs_to_load` and runs `_pull_and_write_kv()` in separate threads (to be async)
        D worker checks if the async loading is done:
            If done:
                D worker puts the request-id in `done_recving`.
                D scheduler then knows the request can be scheduled for decoding now. The model decode
                  will happen in the next scheduler step.
            Else:
                D worker handles other requests first.
    Else (too short prompt, full local prefix-cache):
        D still needs to puts the request in `reqs_to_load` but with None metadata, because D needs to
            notify P the prefilled KV cache is no longer needed and can be freed in P.

    (
        Two scheduler steps can finish:
            scheduler WAITING_FOR_REMOTE_KVS -> connector reqs_to_load -> worker wait for pulling
            worker pulling done, notify P to free blocks
            scheduler RUNNING -> connector reqs_to_load=None -> worker decode -> output
        The waiting buffer will get freed after notified by D or expired.
    )
"""

import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, Optional
from uuid import uuid4

import jax
from jax.sharding import Mesh
from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1, KVConnectorMetadata, KVConnectorRole)
from vllm.distributed.kv_transfer.kv_connector.v1.metrics import (
    KVConnectorPromMetrics, KVConnectorStats, PromMetric, PromMetricT)
from vllm.utils.math_utils import round_down
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.request import RequestStatus

if TYPE_CHECKING:
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.request import Request

try:
    from tpu_raiden.api.jax.kv_cache_manager import KVCacheManager
    from tpu_sync.rpc.raiden_controller import (RaidenControllerClientFacade,
                                                RaidenMemoryType)

    from tpu_inference.distributed import kv_pool_layout
    from tpu_inference.distributed.raiden_reshard_client import (
        ReshardRequestBlockClient)
    _RAIDEN_IMPORT_ERROR = None
except Exception as _exc:  # pylint: disable=broad-except
    KVCacheManager = None
    RaidenControllerClientFacade = None
    RaidenMemoryType = None
    kv_pool_layout = None
    ReshardRequestBlockClient = None
    _RAIDEN_IMPORT_ERROR = _exc

import tpu_inference.distributed.utils as dist_utils
from tpu_inference import envs
from tpu_inference.distributed.tpu_connector_stats import (
    TpuKVConnectorPromMetrics, TpuKVConnectorStats)
from tpu_inference.logger import init_logger
from tpu_inference.runner.tpu_runner import TPUModelRunner

ReqId = str

# `job_name` of the two sides' Raiden work units. Purely a label, but it must
# agree between the producer (which names the source units in the handshake)
# and the consumer (which names its own).
_PREFILL_ROLE = "prefill"
_DECODE_ROLE = "decode"
# How often the worker reports its running reshard-latency split.
_RESHARD_TIMING_EVERY = 50

# Feature requests:
# 1. support async pulling natively
# 2. partial pulling (like RDMA)
# 3. non-blocking jax array read/write

logger = init_logger(__name__)


@dataclass
class SendMeta:
    uuid: int
    # `list[int]`       used for non-HMA connector
    # `list[list[int]]` used for HMA connector (per-kv-cache-group)
    local_block_ids: list[int] | list[list[int]]
    expiration_time: float
    # Controller path only: the per-destination-rank req_ids this request
    # fans out to. The producer's blocks are done once every one of them has
    # been pushed; see `TPUConnectorWorker.get_finished`.
    dst_req_ids: list[str] | None = None


@dataclass
class LoadMeta:
    uuid: int
    # `list[int]`       used for non-HMA connector.
    # `list[list[int]]` used for HMA connector (per-kv-cache-group).
    local_block_ids: list[int] | list[list[int]] | None
    remote_block_ids: list[int] | list[list[int]] | None
    remote_host: str | list[str]
    remote_port: int | list[int]
    # Controller path only (all None on the symmetric fall-through). The
    # destination is the only side that knows both geometries, so it builds
    # the byte spans and drives the transfer; the producer publishes what it
    # needs to do that.
    src_units: list[dict] | None = None
    src_geometry: dict | None = None
    src_controller: str | None = None
    num_tokens: int | None = None
    dst_req_ids: list[str] | None = None


# The metadata used for communicating between scheduler and worker connectors.
@dataclass
class TPUConnectorMetadata(KVConnectorMetadata):
    reqs_to_send: dict[ReqId, SendMeta] = field(default_factory=dict)
    reqs_to_load: dict[ReqId, LoadMeta] = field(default_factory=dict)


class TPUConnector(KVConnectorBase_V1):

    def __init__(self, vllm_config: VllmConfig, role: KVConnectorRole,
                 kv_cache_config: KVCacheConfig):
        super().__init__(vllm_config, role, kv_cache_config)
        assert vllm_config.kv_transfer_config is not None
        self._connector_metadata = None

        if role == KVConnectorRole.SCHEDULER:
            self.connector_scheduler = \
                TPUConnectorScheduler(vllm_config)
            self.connector_worker = None
        elif role == KVConnectorRole.WORKER:
            self.connector_scheduler = None
            self.connector_worker = TPUConnectorWorker(vllm_config)

    ############################################################
    # Scheduler Side Methods
    ############################################################
    def get_num_new_matched_tokens(
            self, request: "Request",
            num_computed_tokens: int) -> tuple[int, bool]:
        assert self.connector_scheduler is not None
        return self.connector_scheduler.get_num_new_matched_tokens(
            request, num_computed_tokens)

    def update_state_after_alloc(self, request: "Request",
                                 blocks: "KVCacheBlocks",
                                 num_external_tokens: int):
        assert self.connector_scheduler is not None
        return self.connector_scheduler.update_state_after_alloc(
            request, blocks, num_external_tokens)

    def build_connector_meta(
        self,
        scheduler_output: SchedulerOutput,
    ) -> TPUConnectorMetadata:
        assert self.connector_scheduler is not None
        return self.connector_scheduler.build_connector_meta()

    def request_finished(
        self,
        request: "Request",
        block_ids: list[int],
    ) -> tuple[bool, Optional[dict[str, Any]]]:
        assert self.connector_scheduler is not None
        return self.connector_scheduler.request_finished(request, block_ids)

    def get_finished_count(self) -> int:
        assert self.connector_scheduler is not None
        return self.connector_scheduler.get_finished_count()

    ############################################################
    # Worker Side Methods
    ############################################################
    def register_kv_caches(self, kv_caches: list[jax.Array]):
        """
        We don't register kv_caches in connector, we call `register_runner` and
        use runner.kv_caches directly instead because the ref of runner.kv_caches
        would be reassigned during model forward.
        """
        pass

    def register_runner(self, runner: TPUModelRunner) -> None:
        assert self.connector_worker is not None
        self.connector_worker.register_runner(runner)

    def start_load_kv(self, _, **kwargs) -> None:
        assert self.connector_worker is not None
        assert isinstance(self._connector_metadata, TPUConnectorMetadata)
        self.connector_worker.process_send_load(self._connector_metadata)

    def wait_for_layer_load(self, layer_name: str) -> None:
        """TPU connector doesn't support layer wise load."""
        pass

    def save_kv_layer(self, *args, **kwargs) -> None:
        """TPU connector doesn't support layer wise save."""
        pass

    def wait_for_save(self):
        """
        Not useful for TPU, because by the design of vLLM KVConnectorModelRunnerMixin,
        this function is only called when scheduler_output.total_num_scheduled_tokens is not 0.
        But the reqs_to_send is only available after the req finished prefilling where the
        total_num_scheduled_tokens could be 0 if no other running reqs.
        So we run saving logic in `start_load_kv -> process_send_load` instead.
        """
        pass

    def get_finished(self,
                     finished_req_ids: set[str]) -> tuple[set[str], set[str]]:
        assert self.connector_worker is not None
        return self.connector_worker.get_finished()

    def get_kv_connector_stats(self) -> KVConnectorStats | None:
        """
        Get the KV transfer stats for the connector.
        """
        if self.connector_worker is None:
            return None
        return self.connector_worker.get_kv_connector_stats()

    @classmethod
    def build_kv_connector_stats(
            cls,
            data: dict[str, Any] | None = None) -> KVConnectorStats | None:
        return (TpuKVConnectorStats(
            data=data) if data is not None else TpuKVConnectorStats())

    @classmethod
    def build_prom_metrics(
        cls,
        vllm_config: VllmConfig,
        metric_types: dict[type[PromMetric], type[PromMetricT]],
        labelnames: list[str],
        per_engine_labelvalues: dict[int, list[object]],
    ) -> KVConnectorPromMetrics:
        return TpuKVConnectorPromMetrics(vllm_config, metric_types, labelnames,
                                         per_engine_labelvalues)


class TPUConnectorScheduler():

    def __init__(self, vllm_config: "VllmConfig"):
        self.vllm_config = vllm_config
        self.config = vllm_config.kv_transfer_config
        self.is_producer = self.config.is_kv_producer

        self.block_size = vllm_config.cache_config.block_size

        # This is updated in self.update_state_after_alloc() for D,
        # each request that needs to pull KV cache from remote will be added to it.
        self.reqs_to_send: dict[ReqId, SendMeta] = {}

        # This is updated in self.request_finished() for P,
        # each request that finished prefilling will be added to it.
        self.reqs_to_load: dict[ReqId, LoadMeta] = {}

        self.kv_ip = dist_utils.get_kv_ips()
        self.kv_port = dist_utils.get_kv_ports()

        # Controller path: when TPU_KV_CONTROLLER_ADDRESS is set the
        # handshake carries geometry and the transfer is planned by Raiden's
        # byte-span reshard planner, which tolerates a different TP degree and
        # page size on the two sides. Unset, everything below is inert and the
        # symmetric index-matched pull is unchanged.
        self.controller_address = dist_utils.get_kv_controller_address()
        self.use_controller = self.controller_address is not None
        self.num_kv_ranks = vllm_config.parallel_config.tensor_parallel_size
        self.num_decode_ranks = (dist_utils.get_kv_decode_tp_size(
            vllm_config.kv_transfer_config) or self.num_kv_ranks)
        # The handshake names this engine's own source units, which its workers
        # register from a different process. Both sides derive the tag from the
        # same configuration rather than exchanging it.
        self.instance_tag = kv_pool_layout.engine_instance_tag(
            dist_utils.get_host_ip(), dist_utils.get_kv_transfer_port())

        logger.info(
            f"TPUConnectorScheduler --> kv_ip={self.kv_ip} | kv_port={self.kv_port} | "
            f"controller={self.controller_address} | "
            f"kv_ranks={self.num_kv_ranks} | decode_ranks={self.num_decode_ranks}"
        )

    def _local_geometry(self) -> dict:
        """This engine's KV byte geometry, published by its worker at init.

        Derived from the live mesh and the runner's own page-shape helper, so
        it cannot drift from the actual allocation; the worker stashes it
        where the scheduler can reach it (see
        `dist_utils.set_local_kv_geometry`).
        """
        geometry = dist_utils.get_local_kv_geometry()
        if geometry is None:
            raise RuntimeError(
                "TPU_KV_CONTROLLER_ADDRESS is set but the worker never "
                "published its KV geometry. The worker connector publishes it "
                "in register_runner(); if the scheduler runs in a different "
                "process (Ray multi-host), the executor must forward it the "
                "same way it forwards get_node_kv_ip_port.")
        return geometry

    def get_num_new_matched_tokens(
        self,
        request: "Request",
        num_computed_tokens: int,
    ) -> tuple[int, bool]:
        """
        D workers use this to get the number of new tokens
        that can be loaded from remote P workers.
        No-op for P workers.

        Args:
            request (Request): the request object.
            num_computed_tokens (int): the number of locally
                computed tokens for this request

        Returns:
            A tuple with the following elements:
                - The number of tokens that will be loaded from the
                  external KV cache.
                - If async loading. Must be 'False' for TPU connector
                  because TPU pulls KV cache in a blocking way.

        """
        if self.is_producer or not request.kv_transfer_params:
            return 0, False

        # Only trigger 1 KV transfer per request.
        if request.kv_transfer_params.get("do_remote_prefill", True) is False:
            # logger.debug(f"TPUConnector Scheduler skip kv transfer for request {request.request_id} as it already pulled before.")
            return 0, False

        assert num_computed_tokens % self.block_size == 0
        # This rounding logic must be consistent with calculating
        # remote_block_ids in P's request_finished()
        rounded_num_prompt_tokens = round_down(len(request.prompt_token_ids),
                                               self.block_size)
        # ... which stops being true once the two sides page differently. The
        # producer drops its own trailing partial block, so it sends
        # `round_down(len(prompt), P.block_size)`; deriving the same number
        # from our block size gives a different answer for any prompt that is
        # not a multiple of both. At P=128 / D=64 a 200-token prompt is 128
        # tokens sent and 192 claimed, and tokens 128-191 would be decoded
        # against KV nobody ever wrote. Take the producer's count when it
        # tells us -- it is the one that bounds what actually arrives -- and
        # round it to our own pages, because vLLM requires the externally
        # computed prefix to be block-aligned here.
        sent = request.kv_transfer_params.get("num_tokens")
        if sent is not None:
            rounded_num_prompt_tokens = min(
                rounded_num_prompt_tokens,
                round_down(int(sent), self.block_size))
        count = max(rounded_num_prompt_tokens - num_computed_tokens, 0)
        # NOTE(xiang): Although the JAX P2P pulling is a blocking op, we will run it in a
        # separte thread to make it async, so we are safe to return True here.
        if count > 0:
            return count, True
        return 0, False

    def update_state_after_alloc(self, request: "Request",
                                 blocks: "KVCacheBlocks",
                                 num_external_tokens: int):
        """
        Update states after block allocation.
        No-op for P workers.

        Args:
            request (Request): the request object.
            blocks (KVCacheBlocks): the blocks allocated for the request.
            num_external_tokens (int): the number of tokens that will be
                loaded from the external KV cache.
        """
        if self.is_producer or not request.kv_transfer_params:
            return

        params = request.kv_transfer_params
        if num_external_tokens > 0:
            # We need to load KV-cache from remote (partial prefix cache hit).
            local_block_ids = blocks.get_block_ids()[0]

            # NOTE(xiang): D needs to pull the whole prefill blocks from the remote
            # regardless how much ratio the prefix cache hits.
            # The reason is JAX P2P doesn't work as RDMA, instead it works like:
            # P just prepares the whole prefilled data and waits for pulling, then D pulls the
            # whole data. Which means even with partial prefix cache hit on D, D cannot only
            # pull the remaining partial data from P.
            # Unless we implement a side channel to let P know the prefix cache hit info on D,
            # so P can prepare those non-hit KV only, with that we need to change to:
            # local_block_ids = blocks.get_unhashed_block_ids()

            self.reqs_to_load[request.request_id] = LoadMeta(
                uuid=params["uuid"],
                local_block_ids=local_block_ids,
                remote_block_ids=params["remote_block_ids"],
                remote_host=params["remote_host"],
                remote_port=params["remote_port"],
                **self._controller_load_fields(params),
            )
        else:
            # This branch means two cases:
            # 1. We don't need to load KV-cache from remote because of full local cache.
            # 2. The async pulling is done.
            # In both cases we need to send notification to let P free memory.
            self.reqs_to_load[request.request_id] = LoadMeta(
                uuid=params["uuid"],
                local_block_ids=blocks.get_block_ids()[0],
                remote_block_ids=None,
                remote_host=params["remote_host"],
                remote_port=params["remote_port"],
            )

        # Only trigger 1 KV transfer per request.
        params["do_remote_prefill"] = False

        logger.info(
            f"TPUConnector Scheduler update_state_after_alloc -->  reqs_to_load={self.reqs_to_load}"
        )

    def _controller_load_fields(self, params: dict) -> dict:
        """The controller-path half of a `LoadMeta`, or empty for the
        symmetric path.

        A producer that predates this change (or runs with the controller
        disabled) simply omits `src_units`, and the worker falls through to
        `start_read`. That keeps the two sides independently deployable.
        """
        if not params.get("src_units"):
            return {}
        dst_req_ids = params["dst_req_ids"]
        if len(dst_req_ids) != self.num_kv_ranks:
            raise ValueError(
                f"Producer expects {len(dst_req_ids)} decode ranks but this "
                f"decoder has {self.num_kv_ranks}. Set "
                f"--kv-transfer-config kv_connector_extra_config="
                f'{{"decode_tp_size": {self.num_kv_ranks}}} on the prefill '
                f"side.")
        return dict(
            src_units=params["src_units"],
            src_geometry=params["src_geometry"],
            src_controller=params["src_controller"],
            num_tokens=params["num_tokens"],
            dst_req_ids=dst_req_ids,
        )

    def build_connector_meta(self) -> TPUConnectorMetadata:
        """
        Build the scheduler metadata and pass to the downstream worker.

        This function should NOT modify fields in the scheduler_output.
        Also, calling this function will reset the state of the connector.
        """
        meta = TPUConnectorMetadata()

        if self.is_producer:
            meta.reqs_to_send = self.reqs_to_send
            self.reqs_to_send = {}
        else:
            meta.reqs_to_load = self.reqs_to_load
            self.reqs_to_load = {}

        return meta

    def get_finished_count(self) -> int:
        """
        Return how many workers need pull the kv cache and report back.
        """
        return len(self.kv_ip) if isinstance(self.kv_ip, list) else 1

    def request_finished(
        self,
        request: "Request",
        block_ids: list[int],
    ) -> tuple[bool, Optional[dict[str, Any]]]:
        """
        Called when a request has finished, before its blocks are freed.
        No-op for D workers.

        Args:
            request (Request): the request object.
            block_ids: The block IDs allocated for this request and need to be freed.
        Returns:
            True if the request is being saved/sent asynchronously and blocks
            should not be freed until the request_id is returned from
            get_finished().
            Optional KVTransferParams to be included in the request outputs
            returned by the kv_manager.
        """
        if not self.is_producer:
            return False, None

        # Mark the request finished only if the prefill is done and generates 1 output token.
        # The request's max_tokens has been reset to 1, so it must be finished by length capped.
        if request.status != RequestStatus.FINISHED_LENGTH_CAPPED:
            return False, None

        # NOTE(xiang): Get computed blocks rounded by block_size.
        # This indication means for the last partially filled block, we won't bother transfering
        # KV-cache, will just let D run prefill locally.
        all_full = request.num_computed_tokens % self.block_size == 0
        computed_block_ids = block_ids if all_full else block_ids[:-1]

        # If prompt < block_size, no transfer so free blocks immediately.
        delay_free_blocks = len(computed_block_ids) > 0
        if delay_free_blocks:
            uuid = get_uuid()
            expiration_time = time.perf_counter(
            ) + dist_utils.get_p2p_wait_pull_timeout()
            dst_req_ids = None
            kv_transfer_params = dict(uuid=uuid,
                                      remote_block_ids=computed_block_ids,
                                      remote_host=self.kv_ip,
                                      remote_port=self.kv_port)
            if self.use_controller:
                dst_req_ids = [
                    kv_pool_layout.dst_req_id(request.request_id, rank)
                    for rank in range(self.num_decode_ranks)
                ]
                # The destination builds the byte spans -- it is the only side
                # that sees both geometries -- so publish ours, every source
                # unit (the controller requires transfer_rank contiguous from
                # zero, so non-contributors are listed too and register empty
                # spans), and the controller that owns the plan.
                kv_transfer_params.update(
                    src_units=[
                        kv_pool_layout.unit_to_dict(
                            kv_pool_layout.work_unit_id(
                                _PREFILL_ROLE, rank, self.instance_tag))
                        for rank in range(self.num_kv_ranks)
                    ],
                    src_geometry=self._local_geometry(),
                    src_controller=self.controller_address,
                    num_tokens=len(computed_block_ids) * self.block_size,
                    dst_req_ids=dst_req_ids,
                )
            self.reqs_to_send[request.request_id] = SendMeta(
                uuid=uuid,
                local_block_ids=computed_block_ids,
                expiration_time=expiration_time,
                dst_req_ids=dst_req_ids)
            logger.info(
                f"TPUConnector Scheduler ---->  generated reqs_to_send={self.reqs_to_send} | "
                f"kv_transfer_params={kv_transfer_params}")
        else:
            kv_transfer_params = {}

        return delay_free_blocks, kv_transfer_params


class TPUConnectorWorker:

    def __init__(self, vllm_config: VllmConfig):
        self.vllm_config = vllm_config
        self.config = vllm_config.kv_transfer_config
        self.is_producer = self.config.is_kv_producer

        self.runner: TPUModelRunner = None
        self.mesh: Mesh = None
        self.multi_host = envs.TPU_MULTIHOST_BACKEND == "ray"
        # default value for none distributed scenario
        # when the topology is initialized, runner will update it
        # based on topology_order_id
        self.node_id = 0

        # The Raiden kv cache manager, constructed in register_runner() once the
        # runner's kv_caches exist. Replaces the jax.experimental.transfer
        # server + the ZMQ side channel + HostKVPool host staging.
        self.kv_manager = None
        # Consumer-side: req_ids for which a real pull (submit_load) was issued,
        # so the scheduler's later remote_block_ids=None notify step is a no-op.
        self._submitted: set[ReqId] = set()

        self.host_ip = dist_utils.get_host_ip()
        # Bind the kv_manager control socket to the same port the scheduler
        # advertises as remote_port (TPU_KV_TRANSFER_PORT).
        self.kv_transfer_port = int(dist_utils.get_kv_transfer_port())

        self.transfer_stats = TpuKVConnectorStats()

        # --- Controller path ---------------------------------------------
        # Raiden's byte-span planner is built around torch-vLLM's one process
        # per rank: it requires one data-plane endpoint per work unit and
        # always emits dst_shard_idx = 0. A JAX vLLM worker is instead one
        # process holding N chips, so we give each *device shard* its own
        # KVCacheManager over that shard's buffers -- one endpoint, one unit,
        # one shard each. The per-shard views alias the same device memory the
        # model computes into, so nothing is copied and nothing diverges.
        self.controller_address = dist_utils.get_kv_controller_address()
        self.use_controller = self.controller_address is not None
        self.listener_port = dist_utils.get_raiden_listener_port()
        self.role = _PREFILL_ROLE if self.is_producer else _DECODE_ROLE
        self.instance_tag = kv_pool_layout.engine_instance_tag(
            self.host_ip, self.kv_transfer_port)
        # Per local device shard, index-aligned:
        self.shard_managers: list = []
        self.shard_geometries: list = []
        self.shard_units: list = []
        self._transfer_pool = None
        # Consumer: req_id -> the per-destination-rank transfers still running,
        # and the per-destination-rank req_ids they will complete under.
        self._loads_in_flight: dict[ReqId, list] = {}
        self._load_req_ids: dict[ReqId, set[str]] = {}
        # Producer: req_id -> the per-destination-rank req_ids not yet pushed.
        self._sends_outstanding: dict[ReqId, set[str]] = {}
        # Completions drained from every shard manager. `poll_stats` consumes
        # its queue, so exactly one place may call it: `get_finished`.
        self._sent_ids: set[str] = set()
        self._recvd_ids: set[str] = set()
        self._failed_ids: set[str] = set()
        # Reshard latency, split by phase; see _record_reshard_timing.
        self._reshard_count = 0
        self._reshard_register_s = 0.0
        self._reshard_coordinate_s = 0.0
        # ... and the latency vLLM actually waits on: submission of the first
        # shard transfer to the step that reports the load finished.
        self._load_started_at: dict[ReqId, float] = {}
        self._load_count = 0
        self._load_visible_s = 0.0

        logger.info(f"TPUConnector Worker --> init | "
                    f"is_producer={self.is_producer} | ip={self.host_ip} | "
                    f"kv_transfer_port={self.kv_transfer_port} | "
                    f"controller={self.controller_address}")

    def register_runner(self, runner: TPUModelRunner):
        if KVCacheManager is None:
            raise ImportError(
                "KVCacheManager is not importable. Ensure tpu-raiden is correctly "
                "installed or added to PYTHONPATH so 'tpu_raiden.api.jax.kv_cache_manager' resolves "
                "(and set RAIDEN_PRELOAD_ENGINE=1 so sitecustomize.py preloads "
                f"the kv_cache_manager .so first). Original error: {_RAIDEN_IMPORT_ERROR}"
            )
        self.node_id = runner.topology_order_id
        self.runner = runner
        self.mesh = runner.mesh

        kv_caches = runner.kv_caches
        self.num_layers = len(kv_caches)
        self.sharding = kv_caches[0].sharding
        block_size = self.vllm_config.cache_config.block_size
        max_blocks = self.vllm_config.model_config.max_model_len // block_size
        num_slots = int(os.getenv("RAIDEN_NUM_SLOTS", "16"))
        # H2H transport sockets per transfer (1 = single socket). Higher values
        # parallelize the host-to-host pull to use more network bandwidth.
        parallelism = int(os.getenv("RAIDEN_TRANSPORT_PARALLELISM", "1"))
        # In the new tpu-raiden kv_cache_manager API, parallelism is a per-pull argument
        # to start_read() rather than a constructor arg; stash it here.
        self._parallelism = parallelism
        skip_lock = os.getenv("RAIDEN_UNSAFE_SKIP_BUFFER_LOCK",
                              "true").lower() in ("1", "true", "yes")

        manager_kwargs = dict(
            max_blocks=max_blocks,
            num_slots=num_slots,
            timeout_s=float(dist_utils.get_p2p_wait_pull_timeout()),
            unsafe_skip_buffer_lock=skip_lock,
            # The PUSH (producer H2hWrite) fans out across the manager's ctor
            # `parallelism` (default 4!)
            parallelism=parallelism,
        )

        if self.use_controller:
            self._register_controller_shards(kv_caches, manager_kwargs)
            return

        # The kv_cache_manager holds the physical KV buffers. The model forward updates
        # them in place (donation), so the kv_cache_manager always serves/writes the live
        # KV without re-registration (see plan, Blocker B).
        self.kv_manager = KVCacheManager(
            kv_caches=kv_caches,
            local_control_port=self.kv_transfer_port,
            **manager_kwargs,
        )
        logger.info(
            f"TPUConnector Worker {self.node_id} --> Raiden kv_cache_manager ready | "
            f"ip={self.host_ip} | "
            f"control_port={getattr(self.kv_manager, 'local_control_port', None)} | "
            f"data_port={getattr(self.kv_manager, 'local_data_port', None)} | "
            f"max_blocks={max_blocks} | num_slots={num_slots} | "
            f"parallelism={parallelism}")

    def _shard_layout(self, kv_caches) -> tuple[int, int, list[int]]:
        """(global head groups, local head groups, KV_HEAD rank per shard).

        The KV page is ``(num_blocks, page_tokens, head_groups, packing,
        padded_head_dim)`` with only dim 2 sharded, so a shard's position on
        the KV_HEAD axis is read straight off its slice of that dim. Reading
        the ranks from the arrays rather than from the mesh means a shard can
        never be mislabelled: the byte spans are then addressed to the same
        chip that holds those heads.
        """
        array = kv_caches[0]
        shards = array.addressable_shards
        global_groups = int(array.shape[2])
        local_groups = int(shards[0].data.shape[2])

        ranks: list[int] = []
        for shard in shards:
            index = shard.index
            for dim in (0, 1):
                span = index[dim]
                if span.start not in (None, 0) or span.stop not in (
                        None, array.shape[dim]):
                    raise NotImplementedError(
                        f"KV cache dim {dim} is sharded ({span}); the byte "
                        "lowering assumes BATCH unsharded and no context "
                        "parallelism, so a page is dense per rank")
            if int(shard.data.shape[2]) != local_groups:
                raise ValueError(
                    "Uneven head-group sharding is not supported: "
                    f"{shard.data.shape[2]} vs {local_groups}")
            start = int(index[2].start or 0)
            if start % local_groups:
                raise ValueError(
                    f"Head-group offset {start} is not a multiple of the "
                    f"local extent {local_groups}")
            ranks.append(start // local_groups)

        for layer, array_l in enumerate(kv_caches):
            if array_l.shape != array.shape:
                raise ValueError(
                    f"Layer {layer} KV shape {array_l.shape} differs from "
                    f"layer 0 {array.shape}; the pool manifest is one dense "
                    "pool per layer with a shared geometry")
        return global_groups, local_groups, ranks

    def _register_controller_shards(self, kv_caches, manager_kwargs) -> None:
        """One Raiden work unit per local device shard.

        Each unit gets its own `KVCacheManager` built over that shard's
        buffers, its own data-plane endpoint and its own control-plane
        listener, which is the only shape Raiden's byte-span planner accepts
        (`raiden_controller.py`: "Pool reshard planning requires one
        data-plane endpoint per work unit", and `dst_shard_idx` is always 0).
        """
        import jax.numpy as jnp

        array = kv_caches[0]
        num_blocks, page_tokens, _, packing, padded_head_dim = array.shape
        global_groups, local_groups, ranks = self._shard_layout(kv_caches)
        dtype_bits = jnp.dtype(array.dtype).itemsize * 8
        dtype_tag = jnp.dtype(array.dtype).name

        # A source's schedule key depends on the decode fan-out width, and it
        # is fixed here, long before any handshake -- so the producer has to be
        # told. `decode_tp_size` is the connector option the scheduler already
        # uses to mint one request id per destination rank; 0 means symmetric.
        # Only the producer pushes, so only the producer's key can matter.
        parallelism = global_groups // local_groups
        declared_peers = (dist_utils.get_kv_decode_tp_size(
            self.vllm_config.kv_transfer_config) if self.is_producer else 0)
        peer_ranks = declared_peers or parallelism

        # Guessing symmetric is right whenever it is, and silently wrong
        # otherwise: every rank would key to 0, every push would resolve to
        # source 0's schedule, and the reshard would scatter one rank's bytes
        # over all of them. That failure validates cleanly and is invisible
        # until the output is garbage, so say so here rather than there.
        if self.is_producer and not declared_peers and parallelism > 1:
            logger.warning(
                f"TPUConnector Worker {self.node_id} --> "
                f"decode_tp_size is unset; assuming decode is "
                f"TP{parallelism} like us. If decode runs a different TP "
                f"degree, the per-shard schedule keys computed here are wrong "
                f"and the reshard will silently corrupt KV.")

        model_config = self.vllm_config.model_config
        fingerprint = kv_pool_layout.layout_fingerprint(
            num_layers=self.num_layers,
            num_kv_heads=model_config.get_total_num_kv_heads(),
            head_dim=model_config.get_head_size(),
            dtype_tag=dtype_tag,
        )

        facade = RaidenControllerClientFacade(self.controller_address)
        for position, rank in enumerate(ranks):
            geometry = kv_pool_layout.AttentionKVGeometry(
                num_layers=self.num_layers,
                num_blocks=int(num_blocks),
                page_tokens=int(page_tokens),
                head_groups=global_groups,
                head_groups_local=local_groups,
                packing=int(packing),
                padded_head_dim=int(padded_head_dim),
                dtype_bits=int(dtype_bits),
                transfer_rank=rank,
                transfer_parallelism=global_groups // local_groups,
                dtype_tag=dtype_tag,
            )
            # `addressable_shards[i].data` aliases the same device buffer the
            # model writes -- verified pointer-identical against the global
            # array's own buffer -- so Raiden reads and writes live KV.
            shard_caches = [
                layer.addressable_shards[position].data for layer in kv_caches
            ]
            # The receiver routes an incoming push to a source schedule purely
            # by the sender's node_id, and the controller keys those schedules
            # by the source's ordinal among the ranks feeding that destination
            # -- see kv_pool_layout.source_schedule_key. Raiden defaults
            # node_id to 0, which is only right while one source feeds each
            # destination; with a head fan-in it silently misroutes every rank
            # but the first.
            schedule_key = kv_pool_layout.source_schedule_key(
                rank, parallelism, peer_ranks)
            manager = KVCacheManager(
                kv_caches=shard_caches,
                local_control_port=self.kv_transfer_port + position,
                listener_port=self.listener_port + position,
                node_id=schedule_key,
                **manager_kwargs,
            )
            manifest = kv_pool_layout.build_pool_manifest(geometry)
            summary = manager.register_pools(manifest)
            unit = kv_pool_layout.work_unit_id(self.role, rank,
                                               self.instance_tag)

            facade.register_work_unit(
                unit,
                [manager.transfer_address],
                control_plane_rpc_address=manager.listener_address,
                pool_manifest=manifest,
                layout_fingerprint=fingerprint,
                page_tokens=geometry.page_tokens,
                transfer_parallelism=geometry.transfer_parallelism,
                transfer_rank=rank,
            )

            self.shard_managers.append(manager)
            self.shard_geometries.append(geometry)
            self.shard_units.append(unit)
            logger.info(
                f"TPUConnector Worker {self.node_id} --> registered {unit} | "
                f"data={manager.transfer_address} | "
                f"control={manager.listener_address} | "
                f"heads={geometry.head_group_range} | "
                f"schedule_key={schedule_key} | "
                f"page_tokens={geometry.page_tokens} | "
                f"bytes/token={geometry.per_token_bytes} | pools={summary}")

        # The scheduler builds the handshake and has no mesh of its own.
        # Every local shard has the same geometry bar its rank, so publishing
        # shard 0's is enough for the peer to reconstruct all of them.
        dist_utils.set_local_kv_geometry(
            kv_pool_layout.geometry_to_dict(self.shard_geometries[0]))

        self._transfer_pool = ThreadPoolExecutor(
            max_workers=max(4, len(self.shard_managers)),
            thread_name_prefix="raiden-reshard")

    def _remote_endpoint(self, req_meta: "LoadMeta"):
        """Resolve the producer endpoint(s) for start_read.

        The producer's KVCacheManager binds one control port per NUMA
        sub-manager, consecutively from the advertised base port
        (TPU_KV_TRANSFER_PORT); each sub-manager serves a distinct shard set.
        With >1 sub-manager we MUST route each local sub-manager to the producer
        sub-manager that holds its shards. Passing a single "host:base" string
        instead hits start_read's broadcast overload, so every local sub-manager
        pulls from the base port (sub-manager 0) and the non-base sockets receive
        the wrong shards -- silent ~50% KV corruption.

        Returns a list of structured {endpoint, shards} descriptors when there is
        more than one sub-manager (so start_read does shard-matched routing), and
        a plain "host:port" string for the single-sub-manager case (unchanged).
        Both prefill and decode run identical hardware, so producer sub-manager i
        (port base+i) serves the same shards as our local sub-manager i.
        """
        host = req_meta.remote_host
        port = req_meta.remote_port
        if isinstance(host, list):
            assert isinstance(port, list) and len(host) == len(port)
            host = host[self.node_id]
            port = port[self.node_id]
        base = int(port)
        local_eps = self.kv_manager.get_local_endpoints()
        if len(local_eps) <= 1:
            return f"{host}:{base}"
        return [{
            "endpoint": f"{host}:{base + i}",
            "shards": list(ep["shards"])
        } for i, ep in enumerate(local_eps)]

    def process_send_load(self, metadata: TPUConnectorMetadata):
        """
        This is called in runner before calling model forward,
        whenever the scheduler_output.total_num_scheduled_tokens is empty or not.
        """
        reqs = metadata.reqs_to_send
        if reqs:
            assert self.is_producer
            logger.info(
                f"TPUConnector Worker {self.node_id} -->  reqs_to_send={reqs}")
        for req_id, req_meta in reqs.items():
            if req_meta.dst_req_ids is not None:
                # Controller path: the destination plans and drives the
                # transfer, and the controller fires our push over the
                # listener socket. Nothing to arm here -- just remember which
                # pushes must land before these blocks may be freed.
                self._sends_outstanding.setdefault(req_id, set()).update(
                    req_meta.dst_req_ids)
                continue
            self.kv_manager.register_read(req_id, req_meta.uuid,
                                          req_meta.local_block_ids)

        reqs = metadata.reqs_to_load
        if reqs:
            assert not self.is_producer
            logger.info(
                f"TPUConnector Worker {self.node_id} -->  reqs_to_load={reqs}")
        for req_id, req_meta in reqs.items():
            if req_meta.src_units is not None:
                if req_meta.remote_block_ids is not None:
                    self._start_controller_load(req_id, req_meta)
                # remote_block_ids None means the transfer already finished
                # (or never had to happen); the producer's blocks are freed by
                # its own push completion, so there is nothing to notify.
                continue

            if req_meta.remote_block_ids is not None:
                # Consumer: pull remote_block_ids straight into the local KV
                # cache at local_block_ids. The kv_manager does the H2H pull + H2D
                # write directly into kv_caches -- no separate insert_kv_chunks.
                # Replaces kv_transfer_server.connect + conn.pull + insert.
                if req_id in self._submitted:
                    # Pre-allocated blocks may be re-issued; submit only once.
                    continue
                self._submitted.add(req_id)
                # Resolved here, not above: the post-load notify pass arrives
                # with remote_block_ids=None and, because there is nothing left
                # to load, none of the controller fields either. Under the
                # controller there is no `kv_manager` to ask for endpoints, so
                # resolving one unconditionally would crash that pass.
                remote_endpoint = self._remote_endpoint(req_meta)
                self.kv_manager.start_read(
                    req_id=req_id,
                    uuid=req_meta.uuid,
                    remote_endpoint=remote_endpoint,
                    remote_block_ids=req_meta.remote_block_ids,
                    local_block_ids=req_meta.local_block_ids,
                    parallelism=self._parallelism,
                )
            else:
                # remote_block_ids is None => the async pull already finished
                # (the kv_manager wrote KV into local_block_ids and acked P during
                # submit_load) or there was no pull (full local prefix cache).
                # Nothing to do here: the producer is freed by the pull's own
                # ack, or by timeout if no pull happened. Do NOT issue a 0-block
                # submit_load -- the producer rejects a 0-block pull stream.
                self._submitted.discard(req_id)

    def _start_controller_load(self, req_id: ReqId,
                               req_meta: "LoadMeta") -> None:
        """Drive one reshard per local decode shard, off the engine loop.

        The destination is the side that owns the lowering: it is the only one
        that sees both geometries. It declares each source rank's spans on its
        behalf -- `register_request_blocks` is a controller-side declaration
        and never touches the producer process -- and then asks the producer's
        controller to plan, arm and fire.

        `coordinate_transfer` blocks until the whole transfer completes, so it
        runs on a thread pool; completion still surfaces through
        `poll_stats()` in `get_finished()`, leaving the vLLM-facing contract
        unchanged.
        """
        if req_id in self._loads_in_flight or req_id in self._load_req_ids:
            # Pre-allocated blocks may be re-issued; submit only once.
            return

        src_base = kv_pool_layout.geometry_from_dict(req_meta.src_geometry)
        src_units = [
            kv_pool_layout.unit_from_dict(unit) for unit in req_meta.src_units
        ]
        if len(src_units) != src_base.transfer_parallelism:
            raise ValueError(
                f"Producer published {len(src_units)} source units but a "
                f"transfer parallelism of {src_base.transfer_parallelism}; "
                "the controller requires every source rank to be listed.")
        # Tensor parallelism is uniform, so the peer's ranks differ from the
        # published geometry only in their position on the head axis.
        src_geometries = [
            replace(src_base, transfer_rank=rank)
            for rank in range(src_base.transfer_parallelism)
        ]

        futures = []
        req_ids = set()
        for position, geometry in enumerate(self.shard_geometries):
            shard_req_id = req_meta.dst_req_ids[geometry.transfer_rank]
            req_ids.add(shard_req_id)
            futures.append(
                self._transfer_pool.submit(self._reshard_one_shard,
                                           shard_req_id, req_meta, position,
                                           src_units, src_geometries))
        self._loads_in_flight[req_id] = futures
        self._load_req_ids[req_id] = req_ids
        self._load_started_at[req_id] = time.perf_counter()

    def _reshard_one_shard(self, shard_req_id: str, req_meta: "LoadMeta",
                           position: int, src_units: list,
                           src_geometries: list) -> None:
        dst_geometry = self.shard_geometries[position]
        # The scheduler admitted `round_down(sent, our page)` tokens as
        # externally computed (get_num_new_matched_tokens), so move exactly
        # that -- moving the producer's full count instead would write into a
        # page the scheduler never reserved.
        num_tokens = (int(req_meta.num_tokens) //
                      dst_geometry.page_tokens) * dst_geometry.page_tokens
        src_page = src_geometries[0].page_tokens
        src_blocks = list(
            req_meta.remote_block_ids)[:-(-num_tokens // src_page)]
        # One uuid per destination rank, not one per request: a source keys
        # its active plan by uuid and rejects a repeat with ALREADY_EXISTS, so
        # any source feeding more than one of us would arm only the first and
        # leave the rest waiting on a push that never comes.
        uuid = kv_pool_layout.dst_uuid(req_meta.uuid,
                                       dst_geometry.transfer_rank)
        facade = RaidenControllerClientFacade(req_meta.src_controller)
        blocks = ReshardRequestBlockClient(req_meta.src_controller)
        started = time.perf_counter()

        for src_unit, src_geometry in zip(src_units, src_geometries):
            # Sources that own none of this destination's heads still have to
            # be registered -- the controller wants transfer_rank contiguous
            # from zero and raises "Missing producer block registration" for
            # any listed unit without one -- but with no spans, which drops
            # them from the plan.
            entries = kv_pool_layout.build_request_span_entries(
                src_geometry,
                dst_geometry,
                src_block_ids=src_blocks,
                num_tokens=num_tokens)
            blocks.register_request_blocks(shard_req_id,
                                           uuid,
                                           src_unit,
                                           src_blocks,
                                           pool_spans=entries)

        registered = time.perf_counter()

        # The producer only transferred whole source pages, so trim our own
        # blocks to what those tokens actually fill at *our* page size.
        dst_blocks = list(
            req_meta.local_block_ids)[:-(-num_tokens //
                                         dst_geometry.page_tokens)]
        facade.coordinate_transfer(
            src_units=src_units,
            dst_units=[self.shard_units[position]],
            req_id=shard_req_id,
            use_block_chunks=True,
            is_sender=True,
            uuid=uuid,
            src_controller_address=req_meta.src_controller,
            dst_mem_type=RaidenMemoryType.HBM,
            dst_device_block_ids=dst_blocks,
            num_tokens=num_tokens,
            transfer_pool_tags=[kv_pool_layout.KV_POOL_TAG],
        )
        self._record_reshard_timing(started, registered, time.perf_counter())

    def _record_reshard_timing(self, started: float, registered: float,
                               done: float) -> None:
        """Split a reshard into declaration vs. plan-arm-push-wait.

        `coordinate_transfer` blocks through the whole pipeline, so without
        this split a TTFT regression is unattributable: it could be the span
        declarations, the controller's planning, or the wire. Reported as a
        running mean so the cost is one log line per `_RESHARD_TIMING_EVERY`
        transfers rather than per request.
        """
        self._reshard_count += 1
        self._reshard_register_s += registered - started
        self._reshard_coordinate_s += done - registered
        if self._reshard_count % _RESHARD_TIMING_EVERY:
            return
        count = self._reshard_count
        logger.info(
            f"TPUConnector Worker {self.node_id} --> reshard timing over "
            f"{count} transfers | register_request_blocks="
            f"{1e3 * self._reshard_register_s / count:.1f} ms | "
            f"coordinate_transfer="
            f"{1e3 * self._reshard_coordinate_s / count:.1f} ms (mean)")

    def _record_load_latency(self, started: Optional[float]) -> None:
        """The load latency vLLM is actually blocked on.

        Everything past the reshard itself -- the completion ack reaching
        `poll_stats`, and the engine step that next calls `get_finished` --
        lands in the gap between this and the reshard timings above.
        """
        if started is None:
            return
        self._load_count += 1
        self._load_visible_s += time.perf_counter() - started
        if self._load_count % _RESHARD_TIMING_EVERY:
            return
        logger.info(
            f"TPUConnector Worker {self.node_id} --> load latency over "
            f"{self._load_count} requests | submit-to-done="
            f"{1e3 * self._load_visible_s / self._load_count:.1f} ms (mean)")

    def _get_finished_controller(self) -> tuple[set[str], set[str]]:
        """Aggregate per-shard completions back to vLLM request ids.

        A request fans out to one transfer per destination rank, each under
        its own req_id, so a vLLM request is finished only when every one of
        them is. On the producer side the mapping from source rank to
        destination rank is the plan's business, not ours -- but every
        destination has at least one contributing source, so waiting for the
        union of the destination req_ids to be pushed is exact.
        """
        for manager in self.shard_managers:
            done_sending, done_recving, failed_recving = manager.poll_stats()
            self._sent_ids.update(done_sending)
            self._recvd_ids.update(done_recving)
            self._failed_ids.update(failed_recving)
        if self._failed_ids:
            logger.error(f"TPUConnector Worker {self.node_id} --> "
                         f"failed_recving={sorted(self._failed_ids)}")

        finished_sending: set[str] = set()
        for req_id, outstanding in list(self._sends_outstanding.items()):
            if outstanding <= self._sent_ids:
                finished_sending.add(req_id)
                del self._sends_outstanding[req_id]
                self._sent_ids -= outstanding

        finished_recving: set[str] = set()
        for req_id, futures in list(self._loads_in_flight.items()):
            if not all(future.done() for future in futures):
                continue
            errors = [f.exception() for f in futures if f.exception()]
            if errors:
                # Do NOT report a failed load as done: vLLM would decode
                # against KV that was never written. Leave it pending and let
                # the request time out at the API layer instead.
                logger.error(
                    f"TPUConnector Worker {self.node_id} --> reshard failed "
                    f"for {req_id}: {errors[0]}")
                del self._loads_in_flight[req_id]
                del self._load_req_ids[req_id]
                self._load_started_at.pop(req_id, None)
                continue
            shard_req_ids = self._load_req_ids[req_id]
            if shard_req_ids <= self._recvd_ids:
                finished_recving.add(req_id)
                del self._loads_in_flight[req_id]
                del self._load_req_ids[req_id]
                self._recvd_ids -= shard_req_ids
                self._record_load_latency(
                    self._load_started_at.pop(req_id, None))

        if finished_sending:
            logger.info(f"TPUConnector Worker {self.node_id} -->  "
                        f"done_sending={finished_sending}")
        if finished_recving:
            logger.info(f"TPUConnector Worker {self.node_id} -->  "
                        f"done_recving={finished_recving}")
        return finished_sending, finished_recving

    def get_kv_connector_stats(self) -> KVConnectorStats | None:
        """
        Get the KV transfer stats for the worker.
        """
        # Clear stats for next iteration
        if not self.transfer_stats.is_empty():
            return self.transfer_stats.clone_and_reset()
        return None

    def get_finished(self) -> tuple[set[str], set[str]]:
        # The kv_manager's control plane reports producer completion (done_sending,
        # after D acks) and consumer completion (done_recving, after H2H+H2D).
        # Replaces the reqs_wait_pull/reqs_pulling bookkeeping + ZMQ side channel.
        if self.use_controller:
            return self._get_finished_controller()
        if self.kv_manager is None:
            return set(), set()
        done_sending, done_recving, failed_recving = self.kv_manager.poll_stats(
        )
        if failed_recving:
            # Do NOT report failed receives as done_recving: vllm would then try
            # to decode with KV that was never written and hit an AssertionError
            # that kills the EngineCore (taking down all other requests). Leave
            # them pending; the request times out at the API layer instead.
            logger.error(
                f"TPUConnector Worker {self.node_id} --> failed_recving={failed_recving}"
            )
        if done_sending:
            logger.info(
                f"TPUConnector Worker {self.node_id} -->  done_sending={done_sending}"
            )
        if done_recving:
            logger.info(
                f"TPUConnector Worker {self.node_id} -->  done_recving={done_recving}"
            )
        return set(done_sending), set(done_recving)


def get_uuid() -> int:
    int128 = uuid4().int
    # Must be less than 64-bit int, otherwise vllm output encoder would raise error.
    # use 50 bit to avoid GO trunk the int when doing JSon serialization
    return int128 >> 78
