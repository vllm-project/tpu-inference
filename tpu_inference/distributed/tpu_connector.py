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

import copy
import functools
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional
from uuid import uuid4

import jax
import jax.numpy as jnp
import numpy as np
import zmq
from jax.experimental.transfer import start_transfer_server
from jax.sharding import Mesh
from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1, KVConnectorMetadata, KVConnectorRole)
from vllm.utils.math_utils import round_down
from vllm.utils.network_utils import make_zmq_path, make_zmq_socket
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.request import RequestStatus

if TYPE_CHECKING:
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.request import Request

from tpu_inference import envs
from tpu_inference.distributed.utils import (get_host_ip, get_kv_ips,
                                             get_kv_ports,
                                             get_kv_transfer_port,
                                             get_side_channel_port)
from tpu_inference.logger import init_logger
from tpu_inference.runner.tpu_runner import TPUModelRunner
from tpu_inference.utils import device_array

ReqId = str

# Feature requests:
# 1. support async pulling natively
# 2. partial pulling (like RDMA)
# 3. non-blocking jax array read/write

# The await pull KV cache will be cleared after
# this time (in seconds) if no pulling occurred on it.
P2P_WAIT_PULL_TIMEOUT = 120

logger = init_logger(__name__)


@dataclass
class SendMeta:
    uuid: int
    local_block_ids: list[int]
    expiration_time: float


@dataclass
class LoadMeta:
    uuid: int
    local_block_ids: list[int]
    remote_block_ids: list[int]
    remote_host: str | list[str]
    remote_port: int | list[int]


@dataclass
class _kv_transfer_params:
    """
    P prepares this in request_finished() and responds to proxy server.
    D recieves this from proxy server and uses this to create LoadMeta.
    """
    uuid: int
    remote_block_ids: list[int]
    # A single IP for single-host, or a list of IPs for mult-host.
    remote_host: str | list[str]
    # A single port for single-host, or a list of ports for mult-host.
    remote_port: int | list[int]


# The metadata used for communicating between scheduler and worker connectors.
@dataclass
class TPUConnectorMetadata(KVConnectorMetadata):
    reqs_to_send: dict[ReqId, SendMeta] = field(default_factory=dict)
    reqs_to_load: dict[ReqId, LoadMeta] = field(default_factory=dict)


class TPUConnector(KVConnectorBase_V1):

    def __init__(self, vllm_config: VllmConfig, role: KVConnectorRole):
        assert vllm_config.kv_transfer_config is not None

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

    def save_kv_layer(self, **kwargs) -> None:
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

        self.kv_ip = get_kv_ips()
        self.kv_port = get_kv_ports()
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
                local_block_ids=None,
                remote_block_ids=None,
                remote_host=params["remote_host"],
                remote_port=params["remote_port"],
            )
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
            returned by the engine.
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
            expiration_time = time.perf_counter() + P2P_WAIT_PULL_TIMEOUT
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

        # req_id: (kv, expiration_time)
        self.reqs_wait_pull: dict[ReqId, list[list[jax.Array], float]] = {}
        # req_id: thread_future
        self.reqs_pulling: dict[ReqId, Future] = {}

        self.host_ip = get_host_ip()
        self.kv_transfer_port = get_kv_transfer_port()
        self.side_channel_port = get_side_channel_port()

        self.kv_transfer_server = None
        self.zmq_cxt = zmq.Context()
        if self.is_producer:
            ready_event = threading.Event()
            self.pull_notify_listener_t = threading.Thread(
                target=self._pull_notify_listener,
                args=(ready_event, ),
                daemon=True,
            )
            self.pull_notify_listener_t.start()
            ready_event.wait()
        else:
            self.pull_executor = ThreadPoolExecutor(max_workers=64)
            self.pull_conns: dict[str, Any] = {}
            self.notif_sockets: dict[str, zmq.Socket] = {}

        logger.info(f"TPUConnector Worker --> init | "
                    f"ip={self.host_ip} | "
                    f"kv_transfer_port={self.kv_transfer_port} | "
                    f"side_channel_port={self.side_channel_port}")

    def __del__(self):
        if self.is_producer:
            if hasattr(self, "pull_notify_listener_t"):
                self.pull_notify_listener_t.join(timeout=0)
        else:
            if hasattr(self, "pull_executor"):
                self.pull_executor.shutdown(wait=False)
        if hasattr(self, "zmq_cxt"):
            self.zmq_cxt.destroy(linger=0)

    def register_runner(self, runner: TPUModelRunner):
        self.node_id = runner.topology_order_id
        self.runner = runner
        self.mesh = runner.mesh

        # Get the spec of the kv_caches
        kv_caches = runner.kv_caches
        kv_layer = kv_caches[0]
        self.num_layers = len(kv_caches)
        self.shape = list(kv_layer.shape)
        self.dtype = kv_layer.dtype
        self.sharding = kv_layer.sharding
        logger.info(f"TPUConnector Worker --> register_runner | "
                    f"node_id={self.node_id} | "
                    f"ip={self.host_ip} | "
                    f"kv_transfer_port={self.kv_transfer_port}")
        self._maybe_start_p2p_server()

    def _maybe_start_p2p_server(self):
        if self.kv_transfer_server is not None:
            return
        server_addr = f"{self.host_ip}:{self.kv_transfer_port}"
        transport_addr = f'{self.host_ip}:0'
        self.kv_transfer_server = start_transfer_server(
            jax.local_devices()[0].client,
            server_addr,
            [transport_addr],
            max_num_parallel_copies=8,
            transfer_size=256 * 1024 * 1024,
            use_raw_buffers=False,
        )
        logger.info(
            f"TPUConnector Worker {self.node_id} --> kv transfer | addr={self.kv_transfer_server.address()}"
        )

    def _pull_notify_listener(self, ready_event: threading.Event):
        sock_path = make_zmq_path("tcp", "*", self.side_channel_port)
        sock = make_zmq_socket(ctx=self.zmq_cxt,
                               path=sock_path,
                               socket_type=zmq.ROUTER,
                               bind=True)
        ready_event.set()
        logger.info(
            f"TPUConnector Worker {self.node_id} --> zmq listener | sock_path={sock_path}"
        )

        while True:
            client_id, req_id_bytes = sock.recv_multipart()
            req_id = req_id_bytes.decode('utf-8')
            logger.info(
                f"TPUConnector Worker {self.node_id} --> zmq recieve | req_id={req_id}"
            )
            if req_id in self.reqs_wait_pull:
                # Set the expiration time of this request to -1, mark to be done
                self.reqs_wait_pull[req_id][1] = -1
            else:
                raise ValueError(
                    f"Disagg producer recives a non-exist pulling finished notification request {req_id}"
                )
            time.sleep(0)
            # The response is not really needed.
            # sock.send_multipart([client_id, b"", b"ACK"])

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
            self._prepare_kv_and_wait(req_id, req_meta)

        reqs = metadata.reqs_to_load
        if reqs:
            assert not self.is_producer
            logger.info(
                f"TPUConnector Worker {self.node_id} -->  reqs_to_load={reqs}")
        for req_id, req_meta in reqs.items():
            if req_meta.remote_block_ids is not None:
                # The request requires to pull KV from P, build the connection and pull
                # the data asyncly.
                conn = self._maybe_build_kv_connection(req_meta)
                self.reqs_pulling[req_id] = self.pull_executor.submit(
                    self._pull_kv, conn, req_meta)
            else:
                # The request has finished pulling the KV from remote, or it has full local
                # prefix cache, need to notify P to let it free blocks.
                socket = self._maybe_build_notif_socket(req_meta)
                self._notify_pull_done(socket, req_id)

    def _prepare_kv_and_wait(self, req_id: str, req_meta: SendMeta):
        local_block_ids = req_meta.local_block_ids
        # TODO(xiang): pad block_ids to avoid recompilation
        indices = device_array(self.mesh, np.array(local_block_ids))
        kv = select_from_kv_caches(self.runner.kv_caches, indices)
        # NOTE(xiang): We need to manually store the kv because:
        # Although we can set use_raw_buffers=True to let kv be safely destroyed after
        # calling await_pull, it could be a stranding buffer if D never pulls it.
        # So we have to set use_raw_buffers=False and stores the kv, then the kv buffer
        # will be safely destroyed by either D notifying or expiration.
        self.reqs_wait_pull[req_id] = [kv, req_meta.expiration_time]
        self.kv_transfer_server.await_pull(req_meta.uuid, kv)

    def _maybe_build_kv_connection(self, req_meta: LoadMeta) -> Any:
        if isinstance(req_meta.remote_host, list):
            assert len(req_meta.remote_host) == len(req_meta.remote_port)
            remote_addr = f"{req_meta.remote_host[self.node_id]}:{req_meta.remote_port[self.node_id]}"
        else:
            remote_addr = f"{req_meta.remote_host}:{req_meta.remote_port}"

        if remote_addr in self.pull_conns:
            conn = self.pull_conns[remote_addr]
        else:
            conn = self.kv_transfer_server.connect(remote_addr)
            self.pull_conns[remote_addr] = conn
            logger.info(
                f"Worker {self.node_id} --> kv transfer | connect={remote_addr}"
            )
        return conn

    def _pull_kv(self, conn: Any, req_meta: LoadMeta):
        # The local allocated blocks which don't hit prefix caching.
        local_block_ids = req_meta.local_block_ids
        # The remote computed blocks which need to pull from P.
        remote_block_ids = req_meta.remote_block_ids
        # Make sure they have the same num blocks because we don't care
        # if partial prefix cache hit now.
        assert len(local_block_ids) == len(remote_block_ids)

        kv_spec = self._get_kv_spec(len(remote_block_ids))
        # TODO(xiang): pad block_ids to avoid recompilation
        indices = device_array(self.mesh, np.array(local_block_ids))
        kv = conn.pull(req_meta.uuid, kv_spec)
        logger.info(
            f"Worker {self.node_id} --> kv transfer | pull uuid={req_meta.uuid}"
        )
        return kv, indices

    def _get_kv_spec(self, num_blocks: int) -> list[jax.ShapeDtypeStruct]:
        assert num_blocks <= self.shape[0]
        shape = copy.copy(self.shape)
        shape[0] = num_blocks
        return [
            jax.ShapeDtypeStruct(shape, self.dtype, sharding=self.sharding)
        ] * self.num_layers

    def _maybe_build_notif_socket(self, req_meta: LoadMeta) -> zmq.Socket:
        remote_host = req_meta.remote_host
        if isinstance(req_meta.remote_host, list):
            remote_host = req_meta.remote_host[self.node_id]

        sock_path = make_zmq_path("tcp", remote_host, self.side_channel_port)
        if sock_path in self.notif_sockets:
            sock = self.notif_sockets[sock_path]
        else:
            sock = make_zmq_socket(ctx=self.zmq_cxt,
                                   path=sock_path,
                                   socket_type=zmq.DEALER,
                                   bind=False)
            logger.info(
                f"Worker {self.node_id} --> zmq notify | sock_path={sock_path}"
            )
        return sock

    def _notify_pull_done(self, sock: zmq.Socket, req_id: str):
        logger.info(f"Worker {self.node_id} --> zmq notify | req_id={req_id}")
        sock.send_string(req_id)
        # The response is not really needed.
        # ack = sock.recv_string()

    def get_finished(self) -> tuple[set[str], set[str]]:
        done_sending: set[str] = set()
        done_recving: set[str] = set()
        if not self.reqs_wait_pull and not self.reqs_pulling:
            return done_sending, done_recving

        # Mark a req as done recieving after its pulling thread returns.
        # This req can then be scheduled for decoding in the next scheduler step.
        for req_id in list(self.reqs_pulling.keys()):
            future = self.reqs_pulling[req_id]
            if future.done():
                # NOTE(xiang): we do the scatter in main thread to avoid data racing.
                # The data racing is not for the kv_caches buffer, it's for the runner.kv_caches ref.
                kv, indices = future.result()
                self.runner.kv_caches = scatter_kv_slices(
                    self.runner.kv_caches, kv, indices)
                del self.reqs_pulling[req_id]
                done_recving.add(req_id)

        # Mark a req as done seding when it's expired.
        # This req can then be released blocks in the current scheduler step.
        now = time.perf_counter()
        for req_id in list(self.reqs_wait_pull):
            _, expires = self.reqs_wait_pull[req_id]
            if now > expires:
                del self.reqs_wait_pull[req_id]
                done_sending.add(req_id)
        if done_sending:
            logger.info(
                f"Worker {self.node_id} -->  done_sending={done_sending}")
        if done_recving:
            logger.info(
                f"Worker {self.node_id} -->  done_recving={done_recving}")
        return done_sending, done_recving


def get_uuid() -> int:
    int128 = uuid4().int
    # Must be less than 64-bit int, otherwise vllm output encoder would raise error.
    # use 50 bit to avoid GO trunk the int when doing JSon serialization
    return int128 >> 78


@jax.jit
def select_from_kv_caches(kv_caches: list[jax.Array],
                          indices: list[jax.Array]) -> list[jax.Array]:
    selected = [cache.at[indices].get() for cache in kv_caches]
    return selected


@functools.partial(
    jax.jit,
    donate_argnames=("kv_caches", ),
)
def scatter_kv_slices(kv_caches: list[jax.Array], kv_slices: list[jax.Array],
                      indices: list[jax.Array]) -> list[jax.Array]:
    num_indices = indices.shape[0]
    num_slices = kv_slices[0].shape[0]
    # indices might be padded
    assert num_slices <= num_indices

    new_kv_caches = []
    for cache, slice in zip(kv_caches, kv_slices):
        if num_slices < num_indices:
            slice = jnp.pad(slice, ((0, num_indices - num_slices), (0, 0),
                                    (0, 0), (0, 0)))
        new_cache = cache.at[indices].set(slice)
        new_kv_caches.append(new_cache)
    return new_kv_caches
