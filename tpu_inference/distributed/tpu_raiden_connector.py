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
from dataclasses import dataclass, field
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
    _RAIDEN_IMPORT_ERROR = None
except Exception as _exc:  # pylint: disable=broad-except
    KVCacheManager = None
    _RAIDEN_IMPORT_ERROR = _exc

import tpu_inference.distributed.utils as dist_utils
from tpu_inference import envs
from tpu_inference.distributed.tpu_connector_stats import (
    TpuKVConnectorPromMetrics, TpuKVConnectorStats)
from tpu_inference.logger import init_logger
from tpu_inference.runner.tpu_runner import TPUModelRunner

ReqId = str

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


@dataclass
class LoadMeta:
    uuid: int
    # `list[int]`       used for non-HMA connector.
    # `list[list[int]]` used for HMA connector (per-kv-cache-group).
    local_block_ids: list[int] | list[list[int]] | None
    remote_block_ids: list[int] | list[list[int]] | None
    remote_host: str | list[str]
    remote_port: int | list[int]


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
        logger.info(
            f"TPUConnectorScheduler --> kv_ip={self.kv_ip} | kv_port={self.kv_port}"
        )

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
            self.reqs_to_send[request.request_id] = SendMeta(
                uuid=uuid,
                local_block_ids=computed_block_ids,
                expiration_time=expiration_time)
            kv_transfer_params = dict(uuid=uuid,
                                      remote_block_ids=computed_block_ids,
                                      remote_host=self.kv_ip,
                                      remote_port=self.kv_port)
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

        logger.info(f"TPUConnector Worker --> init | "
                    f"is_producer={self.is_producer} | ip={self.host_ip} | "
                    f"kv_transfer_port={self.kv_transfer_port}")

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

        # The kv_cache_manager holds the physical KV buffers. The model forward updates
        # them in place (donation), so the kv_cache_manager always serves/writes the live
        # KV without re-registration (see plan, Blocker B).
        self.kv_manager = KVCacheManager(
            kv_caches=kv_caches,
            local_control_port=self.kv_transfer_port,
            max_blocks=max_blocks,
            num_slots=num_slots,
            timeout_s=float(dist_utils.get_p2p_wait_pull_timeout()),
            unsafe_skip_buffer_lock=skip_lock,
        )
        logger.info(
            f"TPUConnector Worker {self.node_id} --> Raiden kv_cache_manager ready | "
            f"ip={self.host_ip} | "
            f"control_port={getattr(self.kv_manager, 'local_control_port', None)} | "
            f"data_port={getattr(self.kv_manager, 'local_data_port', None)} | "
            f"max_blocks={max_blocks} | num_slots={num_slots} | "
            f"parallelism={parallelism}")

    def _remote_endpoint(self, req_meta: "LoadMeta") -> str:
        host = req_meta.remote_host
        port = req_meta.remote_port
        if isinstance(host, list):
            assert isinstance(port, list) and len(host) == len(port)
            return f"{host[self.node_id]}:{port[self.node_id]}"
        return f"{host}:{port}"

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
            # Producer: register the prefilled blocks so D can pull them.
            # Replaces select_from_kv_caches + kv_transfer_server.await_pull.
            self.kv_manager.register_read(req_id, req_meta.uuid,
                                          req_meta.local_block_ids)

        reqs = metadata.reqs_to_load
        if reqs:
            assert not self.is_producer
            logger.info(
                f"TPUConnector Worker {self.node_id} -->  reqs_to_load={reqs}")
        for req_id, req_meta in reqs.items():
            remote_endpoint = self._remote_endpoint(req_meta)
            if req_meta.remote_block_ids is not None:
                # Consumer: pull remote_block_ids straight into the local KV
                # cache at local_block_ids. The kv_manager does the H2H pull + H2D
                # write directly into kv_caches -- no separate insert_kv_chunks.
                # Replaces kv_transfer_server.connect + conn.pull + insert.
                if req_id in self._submitted:
                    # Pre-allocated blocks may be re-issued; submit only once.
                    continue
                self._submitted.add(req_id)
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
