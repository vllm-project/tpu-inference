# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import functools
import os
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional
from uuid import uuid4

import jax
import jax.numpy as jnp
from jax.experimental.transfer import start_transfer_server
from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.factory import \
    KVConnectorFactory
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1, KVConnectorMetadata, KVConnectorRole)
from vllm.utils import get_ip, round_down
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.request import RequestStatus

if TYPE_CHECKING:
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.request import Request

from tpu_commons.logger import init_logger

EngineId = str
ReqId = str

# Feature requests:
# 1. support async pulling natively
# 2. partial pulling (like RDMA)
# 3. non-blocking jax array read/write

# The await pull KV cache will be cleared after
# this time (in seconds) if no pulling occurred on it.
P2P_WAIT_PULL_TIMEOUT = 120

P2P_SERVER_HOST = "0.0.0.0"
P2P_SERVER_PORT = os.getenv("TPU_KV_TRANSFER_PORT", 9527)

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
    remote_host: str
    remote_port: int


@dataclass
class _kv_transfer_params:
    """
    P prepares this in request_finished() and responds to proxy server.
    D recieves this from proxy server and uses this to create LoadMeta.
    """
    uuid: int
    remote_block_ids: list[int]
    remote_host: str
    remote_port: int


# The metadata used for communicating between scheduler and worker connectors.
@dataclass
class TPUConnectorMetadata(KVConnectorMetadata):
    reqs_to_send: dict[ReqId, SendMeta] = {}
    reqs_to_load: dict[ReqId, LoadMeta] = {}


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
    ) -> KVConnectorMetadata:
        assert self.connector_scheduler is not None
        return self.connector_scheduler.build_connector_meta(scheduler_output)

    def request_finished(
        self,
        request: "Request",
        block_ids: list[int],
    ) -> tuple[bool, Optional[dict[str, Any]]]:
        assert self.connector_scheduler is not None
        return self.connector_scheduler.request_finished(request, block_ids)

    ############################################################
    # Worker Side Methods
    ############################################################
    def register_kv_caches(self, kv_caches: dict[str, jax.Array]):
        assert self.connector_worker is not None
        self.connector_worker.register_kv_caches(kv_caches)

    def start_load_kv(self, _, **kwargs) -> None:
        assert self.connector_worker is not None
        assert isinstance(self._connector_metadata, TPUConnectorMetadata)
        self.connector_worker.start_load_kv(self._connector_metadata)

    def wait_for_save(self):
        assert self.connector_worker is not None
        assert isinstance(self._connector_metadata, TPUConnectorMetadata)
        self.connector_worker.wait_for_save(self._connector_metadata)

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

        self.kv_transfer_host = get_ip()
        self.kv_transfer_port = P2P_SERVER_PORT

        # This is updated in self.update_state_after_alloc() for D,
        # each request that needs to pull KV cache from remote will be added to it.
        self.reqs_to_send: dict[ReqId, SendMeta] = {}

        # This is updated in self.request_finished() for P,
        # each request that finished prefilling will be added to it.
        self.reqs_to_load: dict[ReqId, LoadMeta] = {}

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
        if self.is_producer:
            return 0, False

        assert num_computed_tokens % self.block_size == 0
        # This rounding logic must be consistent with calculating
        # remote_block_ids in P's request_finished()
        rounded_num_prompt_tokens = round_down(len(request.prompt_token_ids),
                                               self.block_size)
        count = max(rounded_num_prompt_tokens - num_computed_tokens, 0)
        # NOTE(xiang): Although the JAX P2P pulling is a blocking op, we will run it in a
        # separte thread to make it async, so we are safe to return True here.
        return count, True

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
        if self.is_producer:
            return

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
            # so P can prepare those non-hit KV only, this can be a TODO in the future. With that
            # we need to change to:
            # local_block_ids = blocks.get_unhashed_block_ids()

            params = request.kv_transfer_params
            self.reqs_to_load[request.request_id] = LoadMeta(
                uuid=params["uuid"],
                local_block_ids=local_block_ids,
                remote_block_ids=params["remote_block_ids"],
                remote_host=params["remote_host"],
                remote_port=params["remote_port"],
            )
        else:
            # We don't need to load KV-cache from remote (full prefix cache hit).
            # But We need to send notification to let P free memory.
            self.reqs_to_load[request.request_id] = None

    def build_connector_meta(self, _: SchedulerOutput) -> KVConnectorMetadata:
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
            self.reqs_to_load = self.reqs_to_load
            self.reqs_to_load = {}

        return meta

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
            uuid = uuid4()
            expiration_time = time.perf_counter() + P2P_WAIT_PULL_TIMEOUT
            self.reqs_to_send[request.request_id] = SendMeta(
                uuid=uuid,
                local_block_ids=computed_block_ids,
                expiration_time=expiration_time)
            kv_transfer_params = dict(uuid=uuid,
                                      remote_block_ids=computed_block_ids,
                                      remote_host=self.kv_transfer_host,
                                      remote_port=self.kv_transfer_port)
        else:
            kv_transfer_params = {}

        return delay_free_blocks, kv_transfer_params


class TPUConnectorWorker:

    def __init__(self, vllm_config: VllmConfig):
        self.vllm_config = vllm_config

        self.kv_caches: list[jax.Array] = None

        self.server = None
        # TODO(xiang): add cleanup
        self.conns: dict[str, Any] = {}

        # This can be different for P and D.
        self.tp_size = vllm_config.parallel_config.tensor_parallel_size

        # req_id: (kv, expiration_time)
        self.reqs_wait_pull: dict[ReqId, (list[jax.Array], float)] = {}

        self.pull_threads = ThreadPoolExecutor(max_workers=32)
        self.reqs_pulling: dict[ReqId, Future] = {}

    def _start_p2p_server(self):
        if self.server is None:
            server_addr = f"{P2P_SERVER_HOST}:{P2P_SERVER_PORT}"
            self.server = start_transfer_server(
                jax.local_devices()[0].client,
                server_addr,
                ['0.0.0.0:0'],
                max_num_parallel_copies=8,
                transfer_size=256 * 1024 * 1024,
                use_raw_buffers=False,
            )

    def register_kv_caches(self, kv_caches):
        self.kv_caches = kv_caches

    def start_load_kv(self, metadata: TPUConnectorMetadata):
        self._start_p2p_server()
        reqs = metadata.reqs_to_load

        for req_id, req_meta in reqs.items():
            if req_meta is None:
                # This request has full local prefix cache hit, no need to pull from remote.
                self.reqs_pulling[req_id] = dummy_future()
            else:
                conn = self._maybe_build_connection(req_meta)
                self.reqs_pulling[req_id] = self.pull_threads.submit(
                    self._pull_and_write_kv, conn, req_meta)

    def _maybe_build_connection(self, req_meta: LoadMeta) -> Any:
        remote_addr = f"{req_meta.remote_host}:{req_meta.remote_port}"
        if remote_addr not in self.conns:
            conn = self.server.connect(remote_addr)
            self.conns[remote_addr] = conn
        else:
            conn = self.conns[remote_addr]
        return conn

    def _pull_and_write_kv(self, conn: Any, req_meta: LoadMeta):
        # The local allocated blocks which don't hit prefix caching.
        local_block_ids = req_meta.local_block_ids
        # The remote computed blocks which need to pull from P.
        remote_block_ids = req_meta.remote_block_ids
        # Make sure they have the same num blocks because we don't care
        # if partial prefix cache hit now.
        assert len(local_block_ids) == len(remote_block_ids)

        kv_spec = self._get_kv_spec(len(remote_block_ids))
        kv = conn.pull(req_meta.uuid, kv_spec)
        # TODO(xiang): pad block_ids to avoid recompilation
        indices = jnp.array(local_block_ids)
        self.kv_caches = scatter_kv_slices(self.kv_caches, kv, indices)

    def _get_kv_spec(self, num_blocks: int) -> list[jax.ShapeDtypeStruct]:
        num_layers = len(num_blocks)
        kv_layer = self.kv_caches[0]
        shape = list(kv_layer.shape)
        assert num_blocks <= shape[0]
        shape[0] = num_blocks
        dtype = kv_layer.dtype
        sharding = kv_layer.sharding
        return [jax.ShapeDtypeStruct(shape, dtype, sharding=sharding)
                ] * num_layers

    def wait_for_save(self, metadata: TPUConnectorMetadata):
        self._start_p2p_server()
        reqs = metadata.reqs_to_send

        for req_id, req_meta in reqs.items():
            self._wait_pull_kv(req_id, req_meta)

    def _wait_pull_kv(self, req_id: str, req_meta: SendMeta):
        # TODO(xiang): pad block_ids to avoid recompilation
        indices = jnp.array(req_meta.local_block_ids)
        kv = select_from_kv_caches(self.kv_caches, indices)
        # NOTE(xiang): We need to manually store the kv because:
        # Although we can set use_raw_buffers=True to let kv be safely destroyed after
        # calling await_pull, it could be a stranding buffer if D never pulls it.
        # So we have to set use_raw_buffers=False and stores the kv, then the kv buffer
        # will be safely destroyed by either D notifying or expiration.
        self.reqs_wait_pull[req_id] = (kv, req_meta.expiration_time)
        self.server.await_pull(req_meta.uuid, self.reqs_wait_pull[req_id][0])

    def get_finished(self) -> tuple[set[str], set[str]]:
        done_sending: set[str] = set()
        done_recving: set[str] = set()

        # Mark a req as done recieving after its pulling thread returns.
        for req_id in list(self.reqs_pulling.keys()):
            result = self.reqs_pulling[req_id]
            if result.done():
                del self.reqs_pulling[req_id]
                done_recving.add(req_id)

        # TODO(xiang): Mark a req as done seding after its recieves a notification from D using zmq.

        # Mark a req as done seding when it's expired.
        now = time.perf_counter()
        for req_id in list(self.reqs_wait_pull):
            _, expires = self.reqs_wait_pull[req_id]
            if now < expires:
                del self.reqs_wait_pull[req_id]
                done_sending.add(req_id)


def dummy_future() -> Future:
    x = Future()
    x.set_result(1)
    return x


@jax.jit
def select_from_kv_caches(kv_caches: list[jax.Array],
                          indices: list[jax.Array]) -> list[jax.Array]:
    return [array.at[indices].get() for array in kv_caches]


@functools.partial(
    jax.jit,
    donate_argnames=("kv_caches", ),
)
def scatter_kv_slices(kv_caches: list[jax.Array], kv_slices: list[jax.Array],
                      indices: list[jax.Array]) -> list[jax.Array]:
    new_kv_caches = []
    for cache, slice in zip(kv_caches, kv_slices):
        new_cache = cache.at[indices].set(slice)
        new_kv_caches.append(new_cache)
    return new_kv_caches


KVConnectorFactory.register_connector("TPUConnector",
                                      "tpu_commons.distributed.tpu_connector",
                                      "TPUConnector")
