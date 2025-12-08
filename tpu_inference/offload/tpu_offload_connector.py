# SPDX-License-Identifier: Apache-2.0
"""
Scheduler side execution:
TPUOffloadConnectorScheduler manages the state of KV cache loading and saving for
each request. It acts as a state machine, tracking the progress of requests
across multiple scheduling steps and generating work orders (TPUReqMeta) for
the TPUOffloadConnectorWorker.

Core Components:
- RequestTracker: The primary state object for a request. It tracks the
    cumulative tokens and blocks processed, and how many of those tokens have
    been saved to the CPU cache. A tracker is created when a request is first
    scheduled and lives until the request is finished.

- LoadSpec: A temporary state object created when a new request has a prefix
    that matches data in the CPU cache (`get_num_new_matched_tokens`). It
    holds the number of matched tokens and a `can_load` flag, which is set
    to True only after the vLLM scheduler allocates the necessary blocks for
    the load (`update_state_after_alloc`).

- SaveSpec: A part of the work order sent to the worker. It instructs the
    worker to save a specific slice of the KV cache from TPU to CPU. It
    contains `num_skip_leading_tokens` to indicate which part of the request's
    KV cache is new and needs saving, and an `is_final_save` flag to signal
    the last save operation for a request.

- TPUReqMeta: The unified work order for a single request in a single step,
    sent from the scheduler to the worker. It can contain a `load_spec` (to
    load from CPU to TPU), a `save_spec` (to save from TPU to CPU), or both.

State Machine Flow (from the perspective of a request):

1.  RECEIVED -> AWAITING_ALLOCATION
    - A new request arrives.
    - `get_num_new_matched_tokens` checks the CPU backend for a matching
        token prefix.
    - If a match is found (N > 0 tokens), a `LoadSpec(num_matched_tokens=N, can_load=False)`
        is created. The request now waits for the vLLM scheduler to allocate
        physical blocks for these N tokens.

2.  AWAITING_ALLOCATION -> SCHEDULED
    - The vLLM scheduler allocates blocks for the request.
    - `update_state_after_alloc` is called. If a `LoadSpec` exists, its
        `can_load` flag is set to True, greenlighting the load operation.
        The request is now considered scheduled for processing in this step.

3.  SCHEDULED -> IN_FLIGHT or COMPLETED
    - This transition is handled by `build_connector_meta` which calls the
        central decision-making function, `_prepare_req_meta`.
    - LoadSpec Preparation: The `LoadSpec` (if it exists and `can_load`
        is True) is passed directly into the `TPUReqMeta`. The worker will
        use `num_matched_tokens` to slice the correct prefix from the request's
        `token_ids` and fetch the corresponding data from the CPU cache.
    - SaveSpec Preparation: `_prepare_req_meta` determines if a save is
        needed by comparing the total tokens processed so far
        (`len(tracker.token_ids)`) with the number of tokens already saved
        (`tracker.num_saved_tokens`).
        - If `len(token_ids) > num_saved_tokens`, a `SaveSpec` is created.
        - `num_skip_leading_tokens` is set to `tracker.num_saved_tokens`. This
            tells the worker to ignore the prefix that's already in the CPU
            cache and only save the new data.
        - The scheduler then *transactionally* updates `tracker.num_saved_tokens`
            to the new total length, ensuring this slice of data is not saved
            again.
    - If the scheduler has not finished the request, it transitions to
        IN_FLIGHT. Its tracker is updated for the next scheduling step.
    - If the scheduler has finished the request, it transitions to
        COMPLETED. The tracker is removed, and a final `SaveSpec` is
        generated.
        - is_final_save: This flag is set to `True` only when the
            scheduler marks a request as finished. It is a  signal
            for the worker, indicating that after this save is complete, the
            request's lifecycle is over and its resources
            can be safely freed.

Worker Side Execution:
- The TPUOffloadConnectorWorker receives the `TPUOffloadConnectorMetadata` containing the list of
    `TPUReqMeta` objects.
- `start_load_kv`: Iterates through the metadata. If a `meta.load_spec`
    exists, it reads the corresponding data from the CPU backend and copies it
    into the allocated blocks on the TPU. This is a blocking operation.
- `wait_for_save`: Iterates through the metadata. If a `meta.save_spec`
    exists, it submits an asynchronous task to copy the specified slice of
    KV data from TPU to CPU and update the CPU backend. It then waits for all
    submitted save tasks for the current step to complete.
"""
import copy
import math
import os
import time
from collections import defaultdict
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, Optional, get_args

import jax
import jax.numpy as jnp
from jax.sharding import Mesh
from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1, KVConnectorHandshakeMetadata, KVConnectorMetadata,
    KVConnectorRole)
from vllm.distributed.kv_transfer.kv_connector.v1.metrics import \
    KVConnectorStats
from vllm.v1.core.kv_cache_utils import BlockHash, get_uniform_page_size
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.outputs import KVConnectorOutput

if TYPE_CHECKING:
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.request import Request
    from vllm.forward_context import ForwardContext

from tpu_inference import envs, utils
from tpu_inference.logger import init_logger
from tpu_inference.offload.cpu_backend import LocalCPUBackend
from tpu_inference.offload.offload_manager import (LRUCacheManager,
                                                   StagingBufferManager)
from tpu_inference.offload.utils import (CPU_OFFLOADING_SWAP_OP_TYPE,
                                         CpuChunkId, KVCacheSwapFn, ReqId,
                                         get_kv_cache_swap_fn,
                                         jitted_insert_kv_cache_slices)
from tpu_inference.runner.kv_cache_manager import KVCacheManager
from tpu_inference.runner.tpu_runner import TPUModelRunner

logger = init_logger(__name__)

# kv cache layout needed by cpu offloading mechanism
REQUIRED_KV_CACHE_LAYOUT = "NHD"

BLOCK_SIZE_BUCKETS = [1, 2, 4, 8, 16]

# we keep our operations at vllm's block granularity,
# and want to provide the following three preferences when handling
# the last partial block during save:
# 1. [supported] drop: drop the entire partial block
# 2. pad: pad to a full block
# 3. dynamic: keep the partial block as is.
PARTIAL_BLOCK_SAVE_BEHAVIOR = Literal["drop"]


@dataclass
class OffloadMetadata(KVConnectorHandshakeMetadata):
    block_size: int
    # memory footprint of each kv cache block (single layer)
    block_size_in_bytes: int
    num_layers: int
    kv_cache_layout: str
    num_devices: int


@dataclass
class SaveSpec:
    """A confirmed work order for the worker to save KV data."""
    num_skip_leading_tokens: int
    # total processed tokens for matching / saving
    num_total_tokens: int
    src_blocks: list[int]
    dst_chunks: list[int]
    # final save for the (newly) finished request
    is_final_save: bool = False
    # A direct signal to the worker to skip the data transfer but still
    # process the completion signal if is_final_save is True.
    skip_save: bool = False


@dataclass
class LoadSpec:
    """Internal scheduler state for a potential load operation."""
    num_matched_tokens: int
    src_chunks: list[int]
    dst_blocks: list[int]
    can_load: bool = False
    num_skip_leading_tokens: int = 0


@dataclass
class TPUReqMeta:
    """A unified work order for a single request in a single step."""
    # The unique identifier for the request.
    req_id: str
    # For a load operation, this contains the prefix of tokens to be loaded
    # from the cache. For a save operation, this contains the new tokens
    # that have just been computed.
    token_ids: list[int]
    # The full list of physical blocks corresponding to the `token_ids`.
    local_block_ids: list[int]
    # An optional `SaveSpec` object. If present, it instructs the worker to
    # perform a save operation.
    save_spec: Optional[SaveSpec] = None
    # An optional `LoadSpec` object. If present, it instructs the worker to
    # perform a load operation.
    load_spec: Optional[LoadSpec] = None

    def __repr__(self) -> str:
        load_info = f"load_spec_exists={self.load_spec is not None}"
        if self.load_spec:
            load_info += (
                f", num_matched_tokens={self.load_spec.num_matched_tokens}, "
                f"can_load={self.load_spec.can_load}, "
                f"num_skip_leading_tokens={self.load_spec.num_skip_leading_tokens}, "
                f"src_chunks={self.load_spec.src_chunks}, "
                f"dst_blocks={self.load_spec.dst_blocks}")
        save_info = f"save_spec_exists={self.save_spec is not None}"
        if self.save_spec:
            save_info += (
                f", num_skip_leading_tokens={self.save_spec.num_skip_leading_tokens}, "
                f"num_total_tokens={self.save_spec.num_total_tokens}, "
                f"is_final_save={self.save_spec.is_final_save}, "
                f"skip_save={self.save_spec.skip_save}, "
                f"dst_chunks={self.save_spec.dst_chunks}, "
                f"src_blocks={self.save_spec.src_blocks}")

        return (f"TPUReqMeta(req_id={self.req_id}, "
                f"num_token_ids={len(self.token_ids)}, "
                f"num_local_block_ids={len(self.local_block_ids)}, "
                f"{load_info}, {save_info})")


@dataclass
class RequestTracker:
    """Tracks the evolving state of a single request across multiple scheduling steps."""
    # The unique identifier for the request.
    req_id: str
    # The total number of tokens in the original prompt.
    prompt_len: int
    # The full, cumulative list of physical block numbers allocated to this
    # request so far.
    block_ids: list[int]
    # The full, cumulative list of token IDs that have been processed for this
    # request so far. This list only contains the
    # tokens to be computed, not the prefix loaded from cache.
    token_ids: list[int]
    # The number of tokens that were a hit in the CPU cache at the beginning
    # of the request. This is constant for the lifetime of the request.
    num_external_hits: int = 0
    # A high-water mark indicating how many tokens from the start of the
    # computed tokens (`token_ids`) have already been saved to the CPU cache.
    save_watermark: int = 0
    # Whether the request is in the decoding phase (generating one token at a time).
    is_decode_phase: bool = False

    def update(self, new_block_ids: list[int], new_token_ids: list[int]):
        """Appends new block IDs and token IDs to the tracker."""
        if new_block_ids is None:
            new_block_ids = []
        elif len(new_block_ids) == 0:
            new_block_ids = []
        elif isinstance(new_block_ids, tuple):
            new_block_ids = new_block_ids[0]
        elif isinstance(new_block_ids, list):
            pass
        else:
            raise ValueError(
                f"Unsupported new_block_ids type {type(new_block_ids)}")
        self.block_ids.extend(new_block_ids)
        self.token_ids.extend(new_token_ids)

        # NOTE(jcgu): is it always true? will MTP affect this judgement?
        # When a request is scheduled again, and the number of new tokens
        # is 1 (excluding chunked prefill), the request is in decode phase.
        if len(new_token_ids) == 1:
            self.is_decode_phase = True

    def __repr__(self) -> str:
        output_str = "    - RequestTracker: " + \
                        f"req_id={self.req_id}, " + \
                        f"prompt_len={self.prompt_len}, " + \
                        f"num_tokens={len(self.token_ids)}, " + \
                        f"num_blocks={len(self.block_ids)}, " + \
                        f"save_watermark={self.save_watermark}"
        return output_str


@dataclass
class KVOffloadConnectorStats(KVConnectorStats):
    """Container for transfer performance metrics"""

    def __post_init__(self):
        if not self.data:
            # Empty container init, no data is passed in.
            self.reset()

    def reset(self):
        # Must be serializable
        self.data: dict[str, dict[str, list[int]]] = {
            "finished_save_chunks": dict(),
            "finished_load_chunks": dict(),
        }

    def record_save(self, req: ReqId, saved_chunk_ids: list[int]):
        if req not in self.data["finished_save_chunks"]:
            self.data["finished_save_chunks"][req] = []
        self.data["finished_save_chunks"][req].extend(
            copy.deepcopy(saved_chunk_ids))

    def record_load(self, req: ReqId, loaded_chunk_ids: list[int]):
        if req not in self.data["finished_load_chunks"]:
            self.data["finished_load_chunks"][req] = []
        self.data["finished_load_chunks"][req].extend(
            copy.deepcopy(loaded_chunk_ids))

    def clone_and_reset(self) -> "KVOffloadConnectorStats":
        old = copy.copy(self)
        self.reset()
        return old

    def is_empty(self) -> bool:
        return self.num_finished_blocks == 0

    def aggregate(self, other: KVConnectorStats) -> KVConnectorStats:
        return self

    def reduce(self) -> dict[str, int | float]:
        # Compute compact representative stats suitable for CLI logging
        if self.is_empty():
            return {
                "Num finished save blocks ": 0,
                "Num finished load blocks ": 0,
            }

        finished_save_chunks = sum(
            len(chunk_list)
            for chunk_list in self.data["finished_save_chunks"].values())
        finished_load_chunks = sum(
            len(chunk_list)
            for chunk_list in self.data["finished_load_chunks"].values())

        return {
            "Num finished save chunks ": finished_save_chunks,
            "Num finished load chunks": finished_load_chunks,
        }

    @property
    def num_finished_blocks(self) -> int:
        return len(self.data["finished_save_chunks"]) + len(
            self.data["finished_load_chunks"])


# The metadata used for communicating between scheduler and worker connectors.
@dataclass
class TPUOffloadConnectorMetadata(KVConnectorMetadata):
    requests_meta: list[TPUReqMeta] = field(default_factory=list)


class TPUOffloadConnector(KVConnectorBase_V1):

    def __init__(
        self,
        vllm_config: VllmConfig,
        role: KVConnectorRole,
        kv_cache_config: KVCacheConfig | None = None,
    ):
        super().__init__(vllm_config, role, kv_cache_config)
        logger.info("TPUOffloadConnector: Entering __init__")
        if role == KVConnectorRole.SCHEDULER:
            self.connector_scheduler = \
                TPUOffloadConnectorScheduler(vllm_config)
            self.connector_worker = None
        elif role == KVConnectorRole.WORKER:
            self.connector_scheduler = None
            # The worker needs a reference to the base connector to access
            # the metadata object set by the engine.
            self.connector_worker = TPUOffloadConnectorWorker(
                vllm_config, self)

    ############################################################
    # Class Methods
    ############################################################
    @classmethod
    def get_required_kvcache_layout(cls, vllm_config: VllmConfig):
        if vllm_config.model_config is None:
            logger.warning_once("Unable to detect current VLLM config. "
                                "Fallback to default kv cache layout.")
            return None

        # TODO(jcgu): test mla
        use_mla = vllm_config.model_config.use_mla
        if use_mla:
            # which fallback to the default behavior.
            return None

        logger.info_once(
            "TPUOffloadConnector currently only supports %s KV cache layout.",
            REQUIRED_KV_CACHE_LAYOUT)
        return REQUIRED_KV_CACHE_LAYOUT

    @classmethod
    def build_kv_connector_stats(
        cls,
        data: dict[str, dict[str, int]] | None = None
    ) -> KVConnectorStats | None:
        return (KVOffloadConnectorStats(
            data=data) if data is not None else KVOffloadConnectorStats())

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
    ) -> TPUOffloadConnectorMetadata:
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
    def register_kv_caches(self, kv_caches: list[jax.Array]):
        logger.info("TPUOffloadConnector: Entering register_kv_caches")
        """
        We don't register kv_caches in connector, we call `register_runner` and
        use runner.kv_caches directly instead because the ref of runner.kv_caches
        would be reassigned during model forward.
        """
        pass

    def register_runner(self, runner: TPUModelRunner) -> None:
        logger.info("TPUOffloadConnector: Entering register_runner")
        assert self.connector_worker is not None
        self.connector_worker.register_runner(runner)

    def start_load_kv(self, fwd_ctx: "ForwardContext") -> None:
        """Starts loading the KV cache for the given requests."""
        assert self.connector_worker is not None
        self.connector_worker.start_load_kv(fwd_ctx)

    def wait_for_layer_load(self, layer_name: str) -> None:
        logger.info("TPUOffloadConnector: Entering wait_for_layer_load")
        """TPU connector doesn't support layer wise load."""
        pass

    def save_kv_layer(self, **kwargs) -> None:
        logger.info("TPUOffloadConnector: Entering save_kv_layer")
        """TPU connector doesn't support layer wise save."""
        pass

    def wait_for_save(self):
        assert isinstance(self._connector_metadata,
                          TPUOffloadConnectorMetadata)
        self.connector_worker.wait_for_save()

    def get_finished(self,
                     finished_req_ids: set[str]) -> tuple[set[str], set[str]]:
        assert self.connector_worker is not None
        return self.connector_worker.get_finished()

    def update_connector_output(self, connector_output: KVConnectorOutput):
        assert self.connector_scheduler is not None
        self.connector_scheduler.update_connector_output(connector_output)

    def get_kv_connector_stats(self) -> KVConnectorStats | None:
        if self.connector_worker is None:
            return None
        return self.connector_worker.get_kv_connector_stats()

    def set_xfer_handshake_metadata(
            self, metadata: dict[int, KVConnectorHandshakeMetadata]) -> None:
        """
        Set offload metata from worker

        Args:
            metadata (dict): the handshake metadata to set.
        """
        assert self.connector_scheduler is not None
        self.connector_scheduler.set_offload_metadata(metadata)

    def get_handshake_metadata(self) -> KVConnectorHandshakeMetadata | None:
        """
        Get worker's offload metadata

        Returns:
            KVConnectorHandshakeMetadata: the handshake metadata.
            None if no handshake metadata is available.
        """
        assert self.connector_worker is not None
        return self.connector_worker.offload_metadata


class TPUOffloadConnectorScheduler():

    def __init__(self, vllm_config: "VllmConfig"):
        logger.info("TPUOffloadConnectorScheduler: Entering __init__")
        self.vllm_config = vllm_config
        self.block_size = vllm_config.cache_config.block_size

        self.offload_metadata: OffloadMetadata | None = None
        self.cpu_buffer_size_in_bytes = math.floor(
            envs.TPU_OFFLOAD_CPU_BUF_SIZE_GB * utils.GBYTES)
        self.staging_buffer_size_in_bytes = math.floor(
            envs.TPU_OFFLOAD_STAGING_BUF_SIZE_GB * utils.GBYTES)
        # offloading manager & staging buffer manager
        # initialize them after getting offload_metadata
        self.num_cpu_chunks = -1
        self.offload_manager = None
        self.num_staging_blocks = -1
        self.staging_buffer_manager = None

        self._request_trackers: dict[ReqId, RequestTracker] = {}
        # This dictionary holds the full vLLM Request object for all requests
        # that are currently in a running state (i.e., have been scheduled but
        # are not yet finished). It's used to access the complete prompt token
        # list when processing incremental updates for cached/running requests,
        # as the scheduler output for these requests is minimal.
        self._unfinished_requests: dict[ReqId, "Request"] = {}
        self.load_specs: dict[ReqId, LoadSpec] = {}
        # requests with load ops that have been considered by vllm scheduler,
        # not all of them will be scheduled, the scheduled ones will be
        # moved to load_specs.
        # it should be cleaned after ConnectorMetadata's creation
        self._pre_load_specs: dict[ReqId, LoadSpec] = {}

        # {reqid: total_num_matched_tokens_in_cpu_backend}
        self._external_cache_hits: dict[ReqId, int] = {}

        # request ID -> set(block hashes being saved/loaded)
        self._reqs_being_saved = defaultdict[ReqId, set[CpuChunkId]](set)
        self._reqs_being_loaded = defaultdict[ReqId, set[CpuChunkId]](set)

        model_name = self.vllm_config.model_config.model

        self.decode_save = envs.TPU_OFFLOAD_DECODE_SAVE
        # NOTE(jcgu): currently, let's make chunk_size == block_size
        # chunk_size == n * block_size lead to
        #  1. multi-size chunks
        #  2. complicated resize (split, concatenate) operations due to
        #     real-chunk-size in save and load
        self.cpu_chunk_size = self.block_size

        self.partial_block_save_behavior: PARTIAL_BLOCK_SAVE_BEHAVIOR = "drop"

        logger.info(
            f"TPUOffloadConnectorScheduler initialized with: "
            f"block_size={self.block_size}, "
            f"cpu_chunk_size={self.cpu_chunk_size}, "
            f"cpu_buffer_size ={self.cpu_buffer_size_in_bytes} Bytes, "
            f"staging_buffer_size ={self.staging_buffer_size_in_bytes} Bytes, "
            f"model_name={model_name}, "
            f"decode_save={self.decode_save}, "
            f"partial_block_save_behavior={self.partial_block_save_behavior}, "
            f"num_staging_blocks={self.num_staging_blocks}.")

    def set_offload_metadata(
            self, metadata: dict[int, KVConnectorHandshakeMetadata]) -> None:
        """
        Set offload metadata for this connector scheduler,
        figure out is cpu cache and staging buffer capacity,
        and initialize its offload manager and staging buffer manager.

        Args:
            metadata (dict): the handshake metadata to set.
        """
        # NOTE(jcgu): single host only
        assert len(metadata) == 1
        offload_metadata = list(metadata.values())[0]
        assert isinstance(offload_metadata, OffloadMetadata)
        self.offload_metadata = offload_metadata
        logger.info(f" received offload metadata: {offload_metadata}")

        block_size_in_bytes = self.offload_metadata.block_size_in_bytes
        num_layers = self.offload_metadata.num_layers
        self.num_cpu_chunks = self.cpu_buffer_size_in_bytes // block_size_in_bytes // num_layers
        self.num_staging_blocks = self.staging_buffer_size_in_bytes // block_size_in_bytes // num_layers
        if self.num_cpu_chunks > 0 and self.num_staging_blocks > 0:
            self.offload_manager = LRUCacheManager(
                num_cpu_chunks=self.num_cpu_chunks)
            self.staging_buffer_manager = StagingBufferManager(
                num_blocks=self.num_staging_blocks)

            logger.info(
                f" init offload scheduler with {self.num_cpu_chunks} cpu_chunks and {self.num_staging_blocks} staging blocks."
            )
        else:
            raise ValueError(
                f" get invalid num_cpu_chunks ({self.num_cpu_chunks}) or num_staging_blocks ({self.num_staging_blocks}). Both must be > 0."
            )

    def _get_request_block_hashes(self, req: "Request") -> list[BlockHash]:
        # request's original block_hashes do not include the last partial block
        # TODO(jcgu): add an option to use local token_processor
        return req.block_hashes

    def get_num_new_matched_tokens(
        self,
        request: "Request",
        num_computed_tokens: int,
    ) -> tuple[int, bool]:
        """
        Checks for external KV cache hit against the local CPU backend.
        """
        assert self.offload_manager and self.staging_buffer_manager
        assert num_computed_tokens % self.block_size == 0, f"{num_computed_tokens} % {self.block_size} != 0"
        # get block_hash
        block_hashes = self._get_request_block_hashes(request)
        num_total_blocks = len(block_hashes)
        prompt_token_ids = request.prompt_token_ids
        logger.info(f"Request {request.request_id}: Checking for cache hit. "
                    f"Prompt length: {len(prompt_token_ids)}, "
                    f"Block_hashes ({num_total_blocks}),"
                    f"Already computed tokens: {num_computed_tokens}. ")

        # look for blocks in the cache
        num_hits = self.offload_manager.lookup(block_hashes)
        matched_block_hashes = block_hashes[:num_hits]
        self.offload_manager.touch(block_hashes)
        num_matched_blocks = len(matched_block_hashes)
        # num_matched_tokens = min(num_matched_blocks * self.block_size,
        #                          len(prompt_token_ids))
        num_matched_tokens = num_matched_blocks * self.block_size
        assert num_matched_tokens <= len(prompt_token_ids)
        num_computed_blocks = num_computed_tokens // self.block_size
        num_blocks_to_load = max(num_matched_blocks - num_computed_blocks, 0)
        logger.info(
            f"Request {request.request_id}: Found {num_matched_tokens} (out of {len(prompt_token_ids)} prompt tokens) matched tokens ({num_matched_blocks} blocks) in CPU backend (computed_blocks: {num_computed_blocks}, blocks_to_load: {num_blocks_to_load})."
        )

        if num_blocks_to_load > 0:
            # planning staging blocks for load
            # NOTE(jcgu): do not worry about the inconsistency of the staging buffer status;
            # there is only one connector scheduler who is operating on it.
            num_avail_staging_blocks = self.staging_buffer_manager.get_num_free_staging_blocks(
            )
            if num_blocks_to_load > num_avail_staging_blocks:
                # reduce blocks_to_load (and matched tokens) when there are insufficient staging blocks.
                logger.info(
                    f" Req({request.request_id}) found {num_matched_blocks} blocks ({num_matched_tokens} tokens), but only {num_avail_staging_blocks} staging blocks available."
                )
                num_blocks_to_load = num_avail_staging_blocks
                num_matched_blocks = num_blocks_to_load + num_computed_blocks
                num_matched_tokens = num_matched_blocks * self.block_size

            # still have something to load
            if num_blocks_to_load > 0:
                # NOTE(jcgu): put dummy chunk / block ids;
                # fill real ids later when the requests gets scheduled
                src_chunk_ids = [-1] * num_blocks_to_load
                dummy_dst_blocks = [-1] * num_blocks_to_load
                self._pre_load_specs[request.request_id] = LoadSpec(
                    num_matched_tokens=num_matched_tokens,
                    src_chunks=src_chunk_ids,
                    dst_blocks=dummy_dst_blocks,
                    num_skip_leading_tokens=num_computed_tokens,
                )
                num_allocated_staging_blocks = self.staging_buffer_manager.allocate(
                    request.request_id,
                    num_blocks=num_blocks_to_load,
                    usage="load")
                assert num_allocated_staging_blocks == num_blocks_to_load >= 0, f" failed to allocate {num_allocated_staging_blocks} (load) staging blocks for request {request.request_id}, expected {num_blocks_to_load}."

        # record the matched tokens in the cache, it will be needed in
        # init save_spec
        self._external_cache_hits[request.request_id] = num_matched_tokens

        is_full_prefix_hit = (num_matched_tokens > 0
                              and num_matched_tokens == len(prompt_token_ids))
        num_matched_for_scheduler = num_matched_tokens
        if is_full_prefix_hit:
            # When the entire prompt is found in the CPU cache (a "full hit"),
            # report N-1 matched tokens to the vLLM scheduler instead
            # of the true N. If we report a 100% match (N
            # matched tokens for a prompt of length N), the scheduler sees
            # zero new tokens and may not schedule the request for a prefill
            # step at all and hits
            # https://github.com/vllm-project/vllm/blob/b8b302cde434df8c9289a2b465406b47ebab1c2d/vllm/v1/core/sched/scheduler.py#L438 assetion.
            # By reporting N-1, we ensure the scheduler allocates resources
            # for and schedules the computation of the "last" token of the
            # prompt. The worker (`start_load_kv`) still load the KV of N
            # matched tokens, but the final token'KV will not be used, but be
            # "re-computed" in the following forward pass (the loaded data in
            # the slot gets override.) And from there, the request can
            # seamlessly transition to the decoding phase.
            num_matched_for_scheduler = num_matched_tokens - 1
            logger.info(
                f"Request {request.request_id}: Full prompt hit. Reporting {num_matched_for_scheduler} matched tokens. Actual hit from backend is {num_matched_tokens} tokens"
            )

        # Note on unpinning for the full prefix hit case: Although we report N-1 tokens
        # to the scheduler, the RequestTracker (created later in
        # `build_connector_meta`) stores the true, full N prompt tokens.
        # The `get_finished` method on the worker side uses this complete
        # token list to regenerate the keys, ensuring that all N keys
        # originally pinned during this lookup are gracefully unpinned upon
        # request completion.
        # We don't need to load tokens that are already computed locally in vLLM
        num_to_load = max(0, num_matched_for_scheduler - num_computed_tokens)
        logger.info(
            f"Request {request.request_id}: After accounting for {num_computed_tokens} computed tokens, reporting {num_to_load} tokens to load."
        )

        # external_computed_tokens, load_kv_async
        return num_to_load, False

    def update_state_after_alloc(self, request: "Request",
                                 blocks: "KVCacheBlocks",
                                 num_external_tokens: int):
        """
        This hook is not used for the save logic.
        Update the dst_blocks in the load_spec
        """
        assert self.offload_manager and self.staging_buffer_manager, "please initialize offload / staging_buffer manager."
        logger.info(
            f"TPUOffloadConnectorScheduler: Entering update_state_after_alloc Request {request.request_id}: Scheduler allocated "
            f"{num_external_tokens} external tokens.")
        self._unfinished_requests[request.request_id] = request
        if num_external_tokens == 0:
            return

        # retrieve the load_spec
        load_spec = self._pre_load_specs.pop(request.request_id, None)
        if load_spec:
            assert load_spec.num_skip_leading_tokens % self.block_size == 0
            assert len(load_spec.src_chunks) == len(load_spec.dst_blocks)
            skip_leading_blocks = load_spec.num_skip_leading_tokens // self.block_size
            num_blocks_to_load = len(load_spec.src_chunks)
            num_matched_blocks = num_blocks_to_load + skip_leading_blocks
            assert num_matched_blocks == load_spec.num_matched_tokens // self.block_size, f"{num_matched_blocks} != {load_spec.num_matched_tokens} // {self.block_size}"

            block_hashes = self._get_request_block_hashes(request)
            all_blocks = blocks.get_block_ids()[0]
            logger.info(
                f"  Request: {request.request_id} has {len(all_blocks)} blocks / {len(block_hashes)} block hashes."
            )

            # get the src chunk ids to load
            block_hashes_to_load = block_hashes[
                skip_leading_blocks:num_matched_blocks]
            chunks_to_load = self.offload_manager.prepare_load(
                block_hashes_to_load)
            src_chunk_ids = [chunk.chunk_id for chunk in chunks_to_load]

            # get dst block ids
            dst_blocks = all_blocks[skip_leading_blocks:num_matched_blocks]

            # update load spec
            load_spec.src_chunks = src_chunk_ids
            load_spec.dst_blocks = dst_blocks
            load_spec.can_load = True
            self.load_specs[request.request_id] = load_spec
            self._reqs_being_loaded[request.request_id] |= set(
                load_spec.src_chunks)
            logger.info(
                f"Request {request.request_id} ({len(dst_blocks)} dst_blocks) is ready to load."
            )

    def _prepare_req_meta(
        self,
        tracker: RequestTracker,
        load_spec: Optional[LoadSpec],
        is_finished: bool,
    ) -> Optional[TPUReqMeta]:
        """
        Central decision-making function. Determines if a save or load is
        needed and prepares the metadata. Also performs the transactional
        update of the tracker's save state.
        """
        req_id = tracker.req_id
        _request = self._unfinished_requests[req_id]
        block_hashes = self._get_request_block_hashes(_request)
        self.offload_manager.touch(block_hashes)

        # only consider the tokens covered by block_hashes;
        # currently full blocks only
        num_total_blocks = len(block_hashes)
        num_total_tokens = min(num_total_blocks * self.block_size,
                               len(tracker.token_ids))
        num_full_blocks = num_total_tokens // self.block_size
        num_full_block_tokens = num_full_blocks * self.block_size
        adjusted_num_total_tokens = num_full_block_tokens
        adjusted_num_total_blocks = num_full_blocks
        assert adjusted_num_total_blocks <= len(tracker.block_ids)

        has_new_tokens = adjusted_num_total_tokens > tracker.save_watermark
        should_save = False
        # Determine if a save is needed for this step
        # when there are new token KVs:
        # 1. Prefill: always save
        # 2. Decode (with save_decode=True)
        #  2.1 regular decode (not finished): accumulate until getting a full block
        #  2.2 request finished: save
        if has_new_tokens:
            if not tracker.is_decode_phase:
                # Prefill: always save the new-computed blocks
                should_save = True
            elif self.decode_save:
                if is_finished:
                    # After decode, if there are new final new tokens to save
                    should_save = True
                else:
                    # During decode, we do not drop or pad, just accumulate tokens until the next block boundary
                    next_block_boundary = (
                        tracker.save_watermark // self.block_size +
                        1) * self.block_size
                    logger.info(
                        f"in decode phase, next_block_boundary: {next_block_boundary}, "
                    )
                    if num_total_tokens == next_block_boundary:
                        # always save the full block for decode (not affected by saving_behavior)
                        assert num_total_tokens == adjusted_num_total_tokens, f" decode_save: {num_total_tokens} != (adjusted) {adjusted_num_total_tokens}"
                        should_save = True

        logger.info(f"    - Preparing meta for req (save): {tracker.req_id}, "
                    f"is_finished={is_finished}, "
                    f"total_tokens={num_total_tokens}, "
                    f"adjusted_num_total_tokens={adjusted_num_total_tokens}, "
                    f"adjusted_num_total_blocks={adjusted_num_total_blocks}, "
                    f"saved_tokens={tracker.save_watermark}, "
                    f"has_new={has_new_tokens}, "
                    f"is_decode={tracker.is_decode_phase}, "
                    f"should_save={should_save}")

        # A SaveSpec is always prepared for a finished request to signal completion,
        # even if we don't save the underlying KV data. This is to ensure the TPUOffloadConnectorWorker
        # can correctly report finished request.
        save_spec = None
        if should_save:
            # get src block_ids for save
            # NOTE(jcgu): recompute skip_leading_blocks
            # if tracker.save_watermark has partial tokens in the last block
            # and we saved (i.e., pad) the entire block to cpu_backend, now we
            # want to save the kv of the new tokens in that block; because of
            # the new tokens in that block's token sequence, the block will
            # have a new key (hash value) in cpu_backend, so we should treat
            # the block as a new cache and save the entire block.
            # Example:
            # we have saved:
            # blocks:     [------b0------] [------b1------]
            # tokens:     [t0, t1, t2, t3] [t4, t5,]
            # cpu-backend:{key0: b0, key1:b1(2 tokens, padded)}
            #
            # Now, we have 2 new tokens in the sequence
            # blocks:     [------b0------] [------b1------]
            # tokens:     [t0, t1, t2, t3] [t4, t5, t6, t7]
            # cpu-backend:{key0: b0, key1:b1(2 tokens, padded),
            #              key1_2: b1_2(4 tokens)}
            # In cpu-backend, since b1's token-sequence has been changed, it
            # will have a new key.
            #
            # if we always drop the partial-filled block when saving, then there
            # will no such an issue.
            num_skip_leading_blocks = tracker.save_watermark // self.block_size
            num_skip_leading_tokens = num_skip_leading_blocks * self.block_size
            num_blocks_to_save = adjusted_num_total_blocks - num_skip_leading_blocks

            # planning staging blocks for save
            num_avail_staging_blocks = self.staging_buffer_manager.get_num_free_staging_blocks(
            )
            if num_blocks_to_save > num_avail_staging_blocks:
                # reduce blocks_to_save due to limited free staging blocks
                logger.info(
                    f" Req({tracker.req_id}) have {num_blocks_to_save} ({adjusted_num_total_blocks} - {num_skip_leading_blocks}) blocks to save, but only {num_avail_staging_blocks} staging blocks available."
                )
                num_blocks_to_save = num_avail_staging_blocks
                adjusted_num_total_blocks = num_skip_leading_blocks + num_blocks_to_save
                adjusted_num_total_tokens = adjusted_num_total_blocks * self.block_size

            if num_blocks_to_save > 0:
                block_hashes_to_save = block_hashes[
                    num_skip_leading_blocks:adjusted_num_total_blocks]
                allocate_output = self.offload_manager.allocate_for_save(
                    block_hashes_to_save)
                if allocate_output is not None:
                    # there are enough chunks to save
                    chunks_for_save, chunk_idxs = allocate_output
                    assert num_blocks_to_save == len(chunks_for_save)
                    src_block_ids = tracker.block_ids[
                        num_skip_leading_blocks:adjusted_num_total_blocks]

                    dst_chunks = [chunk.chunk_id for chunk in chunks_for_save]
                    src_blocks = [src_block_ids[idx] for idx in chunk_idxs]

                    # This is a real save operation.
                    save_spec = SaveSpec(
                        num_skip_leading_tokens=num_skip_leading_tokens,
                        num_total_tokens=adjusted_num_total_tokens,
                        is_final_save=is_finished,
                        skip_save=False,
                        src_blocks=src_blocks,
                        dst_chunks=dst_chunks,
                    )
                    self._reqs_being_saved[req_id] |= set(dst_chunks)
                    num_allocated_blocks = self.staging_buffer_manager.allocate(
                        tracker.req_id,
                        num_blocks=num_blocks_to_save,
                        usage="save")
                    assert num_allocated_blocks == num_blocks_to_save >= 0, f" failed to allocate {num_allocated_blocks} (save) staging blocks for request {tracker.req_id}, expected {num_blocks_to_save}."

                    if adjusted_num_total_tokens > tracker.save_watermark:
                        logger.info(
                            f"      -> Old watermark {tracker.save_watermark}, new save_watermark count: {adjusted_num_total_tokens}"
                        )
                        tracker.save_watermark = adjusted_num_total_tokens

        if is_finished and save_spec is None:
            # For finished requests, there must be a no-op save to update the state in the worker side.
            # This is a "completion-only" signal because should_save is False.
            # NOTE(jcgu): num_total_tokens will be used to unpin tokens;
            #  apply the number of saved tokens;
            # TODO(jcgu): rm the no-op save, since save status has been updated
            # through kv_connector_output.kv_connector_stats
            save_spec = SaveSpec(
                num_skip_leading_tokens=tracker.save_watermark,
                num_total_tokens=tracker.save_watermark,
                src_blocks=[],
                dst_chunks=[],
                is_final_save=True,
                skip_save=True,
            )

        # 2. Determine if a work order is needed.
        if not save_spec and not (load_spec and load_spec.can_load):
            return None

        # 3. Construct and return the final work order.
        return TPUReqMeta(
            req_id=tracker.req_id,
            token_ids=tracker.token_ids,
            local_block_ids=tracker.block_ids,
            save_spec=save_spec,
            load_spec=load_spec,
        )

    def build_connector_meta(
            self,
            scheduler_output: SchedulerOutput) -> TPUOffloadConnectorMetadata:
        assert self.offload_manager and self.staging_buffer_manager, "please initialize offload / staging_buffer manager."
        metadata = TPUOffloadConnectorMetadata()

        # Phase 1: Handle and clean up finished requests
        # This block handles requests that have completed their generation.
        # We pop their state from our tracking dictionaries and call _prepare_req_meta
        # one last time. This ensures any final, unsaved tokens are captured and
        # signals to the worker that this is the final save for the request.
        logger.info(
            f"Phase 1: Processing {len(scheduler_output.finished_req_ids)} finished requests."
        )
        for finished_req_id in scheduler_output.finished_req_ids:
            logger.info(f"  - Processing finished req: {finished_req_id}")
            tracker = self._request_trackers[finished_req_id]

            if not tracker:
                logger.warning(
                    f"  - No tracker found for finished req: {finished_req_id}. Skipping."
                )
                continue

            # Prepare one final metadata object if there's a final save needed.
            # `is_finished` is set to True to flag this as the last save operation.
            req_meta = self._prepare_req_meta(tracker,
                                              load_spec=None,
                                              is_finished=True)
            if req_meta:
                logger.info(
                    f"  - Creating final save metadata for req: {finished_req_id}"
                )
                metadata.requests_meta.append(req_meta)

            # Pop tracker and other state first.
            self._request_trackers.pop(finished_req_id, None)
            self._unfinished_requests.pop(finished_req_id, None)
            self.load_specs.pop(finished_req_id, None)

        # Phase 2: Process newly scheduled requests
        # This block handles requests being scheduled for the very first time.
        # It creates the initial RequestTracker and prepares the first work order.
        logger.info(
            f"Phase 2: Processing {len(scheduler_output.scheduled_new_reqs)} new requests."
        )
        for request in scheduler_output.scheduled_new_reqs:
            req_id = request.req_id

            _request = self._unfinished_requests[req_id]
            logger.info(
                f"  - Processing new req: {req_id}, {len(_request.block_hashes)} block_hashes."
            )
            num_new_scheduled_tokens = scheduler_output.num_scheduled_tokens[
                req_id]

            # Get the external cache hit count from our new, reliable source.
            num_external_hits = self._external_cache_hits.pop(req_id, 0)

            # Determine the total length of tokens the tracker should hold.
            # This is vLLM's already computed tokens + newly scheduled tokens.
            num_total_tokens_for_tracker = request.num_computed_tokens + num_new_scheduled_tokens
            tokens_for_tracker = request.prompt_token_ids[:
                                                          num_total_tokens_for_tracker]
            logger.info(
                f"    - num_new_scheduled_tokens: {num_new_scheduled_tokens}, num_vllm_computed: {request.num_computed_tokens}, num_external_hits: {num_external_hits}"
            )
            logger.info(
                f"    - Slicing prompt[:{num_total_tokens_for_tracker}] -> len(tokens_for_tracker): {len(tokens_for_tracker)}"
            )

            # Set the initial high-water mark for `save_watermark`.
            # This is the maximum of what vLLM has computed and what's in our external cache.
            initial_save_watermark = max(request.num_computed_tokens,
                                         num_external_hits)

            # Create and store the tracker, which will maintain the request's
            # state for its entire lifetime.
            assert req_id not in self._request_trackers, f"Request {req_id} already has a tracker."
            # TODO(jcgu): reduce duplicated info in request tracker
            tracker = RequestTracker(
                req_id=req_id,
                prompt_len=len(request.prompt_token_ids),
                block_ids=copy.deepcopy(request.block_ids[0]),
                token_ids=tokens_for_tracker,
                num_external_hits=num_external_hits,
                # The high-water mark for saved tokens starts after the cached prefix.
                save_watermark=initial_save_watermark,
            )
            self._request_trackers[req_id] = tracker
            logger.info(
                f"    - Created tracker for {req_id} with initial state: {tracker}"
            )

            # Immediately prepare metadata for this new request.
            # This could include both a load operation (for the cached part)
            # and a save operation (for the newly computed part).
            load_spec = self.load_specs.pop(req_id, None)
            req_meta = self._prepare_req_meta(tracker,
                                              load_spec,
                                              is_finished=False)
            if req_meta:
                logger.info(f"    - Creating metadata for new req: {req_id} "
                            f"(has_load={req_meta.load_spec is not None}, "
                            f"has_save={req_meta.save_spec is not None})")
                metadata.requests_meta.append(req_meta)

        # Phase 3: Process cached (running) requests
        # This block handles requests that have already been pre-filled at least
        # once and are now being processed again (e.g., for chunked prefill).
        cached_reqs = scheduler_output.scheduled_cached_reqs
        logger.info(
            f"Phase 3: Processing {len(cached_reqs.req_ids)} cached requests.")
        for i, req_id in enumerate(cached_reqs.req_ids):
            tracker = self._request_trackers[req_id]
            full_request = self._unfinished_requests.get(req_id)
            _block_hashes = full_request.block_hashes
            logger.info(
                f"  - Processing cached req: {req_id}, {len(_block_hashes)} block_hashes."
            )

            if full_request is None:
                logger.warning(
                    f"  - No full request found for cached req: {req_id}. Skipping."
                )
                continue

            # num_new_tokens: The number of *additional* tokens the scheduler is
            # processing in this step for this ongoing request.
            num_new_tokens = scheduler_output.num_scheduled_tokens[req_id]

            # current_token_count: This is the crucial calculation to find our
            # place in the full prompt. It's the length of the token prefix
            # already processed in previous steps.
            current_token_count = len(tracker.token_ids)

            logger.info(
                f"    - len(full_request.all_token_ids): {len(full_request.all_token_ids)}"
            )
            # new_token_ids: The slice of the full token sequence corresponding to the
            # new work being done in this step.
            new_token_ids = full_request.all_token_ids[
                current_token_count:current_token_count + num_new_tokens]

            # new_blocks: The new physical blocks allocated for the new_token_ids.
            new_blocks = cached_reqs.new_block_ids[i]
            if new_blocks is None:
                new_blocks = []

            logger.info(
                f"    - num_new_tokens: {num_new_tokens}, current_token_count: {current_token_count}"
            )
            logger.info(
                f"    - Slicing prompt -> len(new_token_ids): {len(new_token_ids)}"
            )
            logger.info(f"    - New blocks allocated: {len(new_blocks)}")

            # Update the tracker with the incremental data.
            tracker.update(new_blocks, new_token_ids)
            logger.info(f"    - Updated tracker for {req_id}: "
                        f"total_tokens={len(tracker.token_ids)}, "
                        f"total_blocks={len(tracker.block_ids)}")

            # for cached requests, whose kv pages get evicted, there will be
            # load operations.
            load_spec = self.load_specs.pop(req_id, None)
            req_meta = self._prepare_req_meta(tracker,
                                              load_spec=load_spec,
                                              is_finished=False)
            if req_meta:
                logger.info(
                    f"    - Creating metadata for cached req: {req_id} "
                    f"(has_save={req_meta.save_spec is not None})")
                metadata.requests_meta.append(req_meta)

        if metadata.requests_meta:
            logger.info(
                f"Prepared {len(metadata.requests_meta)} requests for worker.")

        # after building connector_metadata, all load_specs should be consumed
        assert len(
            self.load_specs
        ) == 0, f" load_specs still has {list(self.load_specs.keys())}"

        # clean up the temporary states of requests that are not scheduled
        for req_id, _load_spec in self._pre_load_specs.items():
            logger.info(f"non-scheduled-reuqest:{req_id}")
            _freed_num_staging_blocks = self.staging_buffer_manager.free(
                req_id, "load")
            assert _freed_num_staging_blocks == len(
                _load_spec.src_chunks
            ), f"{_freed_num_staging_blocks} != {len(_load_spec.src_chunks)}"
        self._pre_load_specs.clear()
        self._external_cache_hits.clear()

        return metadata

    def update_connector_output(self, connector_output: KVConnectorOutput):
        """
        Update KVConnector state from worker-side connectors output.

        Args:
            connector_output (KVConnectorOutput): the worker-side
                connectors output.
        """
        logger.info(
            f"TPUOffloadConnectorScheduler: getting workers' output: finished_sending: {connector_output.finished_sending}, finished_recving: {connector_output.finished_recving}"
        )

        # per iteration, update the finished staging blocks
        if connector_output.kv_connector_stats and connector_output.kv_connector_stats.data is not None:
            assert isinstance(connector_output.kv_connector_stats,
                              KVOffloadConnectorStats)
            assert "finished_save_chunks" in connector_output.kv_connector_stats.data
            assert "finished_load_chunks" in connector_output.kv_connector_stats.data
            for req_id, saved_chunk_ids in connector_output.kv_connector_stats.data[
                    "finished_save_chunks"].items():
                num_saved_chunks = len(saved_chunk_ids)
                logger.info(
                    f"  finished_save_chunks for {req_id}: {saved_chunk_ids}")
                # free staging blocks
                self.staging_buffer_manager.free(
                    req_id, usage="save", num_finished_blocks=num_saved_chunks)
                # update in-flight save
                for saved_chunk_id in saved_chunk_ids:
                    assert saved_chunk_id in self._reqs_being_saved[req_id]
                    self._reqs_being_saved[req_id].remove(saved_chunk_id)
                if len(self._reqs_being_saved[req_id]) == 0:
                    self._reqs_being_saved.pop(req_id, None)
                else:
                    logger.info(
                        f"  remaining_saving_blocks:{req_id}, { self._reqs_being_saved[req_id]}."
                    )

                # update the status of occupied cpu chunks
                self.offload_manager.mark_completion(saved_chunk_ids, "save")

            for req_id, loaded_chunk_ids in connector_output.kv_connector_stats.data[
                    "finished_load_chunks"].items():
                num_loaded_chunks = len(loaded_chunk_ids)
                logger.info(
                    f"  finished_load_chunks for {req_id}: {num_loaded_chunks}"
                )
                self.staging_buffer_manager.free(
                    req_id,
                    usage="load",
                    num_finished_blocks=num_loaded_chunks)
                # update in-flight save
                for loaded_chunk_id in loaded_chunk_ids:
                    assert loaded_chunk_id in self._reqs_being_loaded[req_id]
                    self._reqs_being_loaded[req_id].remove(loaded_chunk_id)
                if len(self._reqs_being_loaded[req_id]) == 0:
                    self._reqs_being_loaded.pop(req_id, None)
                # update the status of occupied cpu chunks
                self.offload_manager.mark_completion(loaded_chunk_ids, "load")

        # clean up the status of the finished requests
        # save
        for req_id in connector_output.finished_sending or []:
            if req_id in self._reqs_being_saved:
                assert len(self._reqs_being_saved[req_id]) == 0
                self._reqs_being_saved.pop(req_id)
            num_freed_blocks = self.staging_buffer_manager.free(req_id,
                                                                usage="save")
            logger.info(
                f"  freed {num_freed_blocks} staging blocks (save) from {req_id}"
            )

        # load
        for req_id in connector_output.finished_recving or []:
            if req_id in self._reqs_being_loaded:
                assert len(self._reqs_being_loaded[req_id]) == 0
                self._reqs_being_loaded.pop(req_id)
            num_freed_blocks = self.staging_buffer_manager.free(req_id,
                                                                usage="load")
            logger.info(
                f"  freed {num_freed_blocks} staging blocks (load) from {req_id}"
            )

    def request_finished(
        self,
        request: "Request",
        block_ids: list[int],
    ) -> tuple[bool, Optional[dict[str, Any]]]:
        """
        Called when a request has finished, before its blocks are freed.

        True if the request is being saved/sent asynchronously and blocks
        should not be freed until the request_id is returned from
        get_finished().
        Optional KVTransferParams to be included in the request outputs
        returned by the engine.
        return:
            delay_free_blocks, kv_xfer_params
        """
        logger.info(" Entering request_finished")
        # Return True to indicate the request is being saved asynchronously
        # and its blocks should not be freed yet.

        req_id = request.request_id
        if req_id in self._reqs_being_saved and len(
                self._reqs_being_saved[req_id]) > 0:
            logger.info(
                f"not_free_with_save:{req_id}, {self._reqs_being_saved[req_id]}"
            )
            return True, None
        if req_id in self._reqs_being_loaded and len(
                self._reqs_being_loaded[req_id]) > 0:
            logger.info(
                f"not_free_with_load:{req_id}, {self._reqs_being_loaded[req_id]}"
            )
            return True, None

        logger.info(f" finished request: {req_id}")
        self._reqs_being_saved.pop(req_id, None)
        self._reqs_being_loaded.pop(req_id, None)

        return False, None


class TPUOffloadConnectorWorker:

    def __init__(self, vllm_config: VllmConfig,
                 connector: "TPUOffloadConnector"):
        logger.info("TPUOffloadConnectorWorker: Entering __init__")
        self.vllm_config = vllm_config
        self.connector = connector
        self.block_size = vllm_config.cache_config.block_size

        self.runner: Optional[TPUModelRunner] = None
        self.mesh: Optional[Mesh] = None
        self.swap_in_fn: KVCacheSwapFn = None
        self.swap_out_fn: KVCacheSwapFn = None
        self.swap_op_type = envs.TPU_OFFLOAD_SWAP_OP_TYPE
        # TODO(jcgu): check libtpu compatibility for pallas dma kernel
        assert self.swap_op_type in get_args(CPU_OFFLOADING_SWAP_OP_TYPE)
        self.use_bucketed_swap_ops = not envs.TPU_OFFLOAD_SKIP_JAX_PRECOMPILE
        logger.info(f" swap operation type is {self.swap_op_type}, "
                    f"use_bucketed_swap_ops={self.use_bucketed_swap_ops}.")

        # metadata for kv cache offload (e.g., page_size_in_bytes)
        self.offload_metadata: OffloadMetadata | None = None
        # cpu cache capacity
        self.cpu_buffer_size_in_bytes = math.floor(
            envs.TPU_OFFLOAD_CPU_BUF_SIZE_GB * utils.GBYTES)
        self.num_cpu_chunks = -1
        self.cpu_backend = None

        # The worker needs its own token processor to generate keys.
        model_name = self.vllm_config.model_config.model
        logger.info(
            f"Model name is {model_name}, KV block_size={self.block_size}")

        self.cpu_chunk_size = self.block_size
        # Thread pool for asynchronous TPU->CPU copies
        self.num_save_threads = envs.TPU_OFFLOAD_SAVE_THREADS
        self.save_executor = ThreadPoolExecutor(
            max_workers=self.num_save_threads, thread_name_prefix="tpu_save_handler")
        self.finished_save_reqs: set[ReqId] = set()
        self.finished_load_reqs: set[ReqId] = set()
        # Tracks if wait_for_save has been called for the current step's metadata.
        self._processed_save_for_step = False

        # record finished save / load blocks (with req_ids) for each iteration
        self.offload_stats = KVOffloadConnectorStats()

    def __del__(self):
        logger.info("TPUOffloadConnectorWorker: Entering __del__")
        self.save_executor.shutdown(wait=True)

    def register_runner(self, runner: TPUModelRunner):
        logger.info("TPUOffloadConnectorWorker: Entering register_runner")
        self.runner = runner
        self.devices = runner.devices
        self.mesh = runner.mesh
        # Get the spec of the kv_caches
        kv_caches = runner.kv_caches
        if kv_caches:
            self.kv_cache_layout = runner.get_kv_cache_layout()
            kv_layer = kv_caches[0]
            self.num_layers = len(kv_caches)
            self.shape = list(kv_layer.shape)
            self.dtype = kv_layer.dtype
            self.device_sharding = kv_layer.sharding

            # NOTE(jcgu): needed when sliced-kv is [num_tokens, num_head, head_dim]
            self.flatten_device_sharding = jax.sharding.NamedSharding(
                mesh=self.device_sharding.mesh,
                spec=jax.sharding.PartitionSpec(None, "model"),
                memory_kind="device")

            self.flatten_host_sharding = jax.sharding.NamedSharding(
                mesh=self.device_sharding.mesh,
                spec=jax.sharding.PartitionSpec(None, "model"),
                memory_kind="pinned_host")

            self.swap_in_fn, self.swap_out_fn = get_kv_cache_swap_fn(
                self.swap_op_type,
                host_sharding=self.flatten_host_sharding,
                device_sharding=self.flatten_device_sharding)

            # collect offload metadata for the offload scheduler
            kv_cache_specs = runner.get_kv_cache_spec()
            # memory footprint of kv cache block of single layer
            self.block_size_in_bytes = get_uniform_page_size(
                list(kv_cache_specs.values()))

            self.offload_metadata = OffloadMetadata(
                block_size=self.block_size,
                block_size_in_bytes=self.block_size_in_bytes,
                num_layers=self.num_layers,
                kv_cache_layout=self.kv_cache_layout,
                num_devices=len(self.devices),
            )

            self.num_cpu_chunks = self.cpu_buffer_size_in_bytes // self.block_size_in_bytes // self.num_layers
            self.cpu_backend = LocalCPUBackend(
                num_cpu_chunks=self.num_cpu_chunks)
            logger.info(
                f" init cpu_backend with {self.num_cpu_chunks} chunks.")

            logger.info(
                "KV Cache details registered in TPUOffloadConnectorWorker:")
            logger.info(f"  - Num layers: {self.num_layers}")
            logger.info(f"  - Shape per layer: {self.shape}")
            logger.info(f"  - DType: {self.dtype}")
            logger.info(f"  - Device sharding: {self.device_sharding}")
            logger.info(
                f"  - Flatten Device sharding: {self.flatten_device_sharding}")
            logger.info(f"  - Layout: {self.kv_cache_layout}")
            logger.info(
                f"  - Block size (single layer): {self.block_size_in_bytes} Bytes"
            )
        else:
            raise ValueError(
                "TPUOffloadConnectorWorker registered with no KV caches.")

        # Pre-compile the JIT functions for KV cache swapping.
        if self.use_bucketed_swap_ops:
            self._precompile_kv_swap_operations()

    def _decompose_into_buckets(self, num_blocks: int) -> list[int]:
        """
        Decomposes a number into a sum of numbers from the BLOCK_SIZE_BUCKETS
        list using a greedy approach.
        """
        sorted_buckets = sorted(BLOCK_SIZE_BUCKETS, reverse=True)
        chunks = []
        remaining = num_blocks
        while remaining > 0:
            for bucket_size in sorted_buckets:
                if remaining >= bucket_size:
                    chunks.append(bucket_size)
                    remaining -= bucket_size
                    break
            else:
                # This should not happen if 1 is in the buckets
                raise ValueError(
                    "Could not decompose number with the given buckets.")
        return chunks

    def _precompile_kv_swap_operations(self):
        """
        Pre-compiles the JIT-compiled functions used for KV cache swapping
        with a variety of common block sizes to avoid runtime recompilation.
        """
        if os.getenv("TPU_OFFLOAD_SKIP_JAX_PRECOMPILE", "0") == "1":
            logger.info(
                "Skipping KV swap pre-compilation due to environment variable."
            )
            return

        logger.info("Starting pre-compilation of KV cache swap operations")
        start_time = time.time()
        paged_kv_for_compilation = self.runner.kv_caches
        for num_blocks in BLOCK_SIZE_BUCKETS:
            try:
                logger.info(f"  - Compiling for {num_blocks} blocks...")
                dummy_block_ids = jnp.arange(num_blocks)

                # 1. Pre-compile gather (used in save)
                flat_dummy_kv_caches_tpu = KVCacheManager._jitted_gather_kv_cache(
                    paged_kv_for_compilation, dummy_block_ids)
                jax.block_until_ready(flat_dummy_kv_caches_tpu)

                # 2. Pre-compile TPU -> CPU transfer (used in save)
                dummy_kv_cpu = self.swap_out_fn(flat_dummy_kv_caches_tpu)
                jax.block_until_ready(dummy_kv_cpu)

                # 3. Pre-compile CPU -> TPU transfer (used in load)
                split_size_list = [self.block_size] * num_blocks
                chunked_dummy_kv_cpu = jax.tree.map(
                    lambda flat_layer_cache: jax.lax.split(
                        flat_layer_cache, split_size_list, axis=0),
                    dummy_kv_cpu)

                chunked_dummy_kv_tpu = self.swap_in_fn(chunked_dummy_kv_cpu)
                jax.block_until_ready(chunked_dummy_kv_tpu)

                # 4. Pre-compile insert (used in load).
                # The result is passed to the next iteration's gather to avoid
                # using a "deleted" array.
                logger.info(
                    f"    - Calling jitted_insert_kv_cache_slices with paged_kv_for_compilation len: {len(paged_kv_for_compilation)}, first_element_shape: {paged_kv_for_compilation[0].shape}, "
                    f"chunked_dummy_kv_tpu len: {len(chunked_dummy_kv_tpu)}")
                paged_kv_for_compilation = jitted_insert_kv_cache_slices(
                    self.block_size, paged_kv_for_compilation,
                    chunked_dummy_kv_tpu, dummy_block_ids)
                jax.block_until_ready(paged_kv_for_compilation)
            except Exception as e:
                logger.warning(
                    f"    - Failed to pre-compile for {num_blocks} blocks: {e}",
                    exc_info=True)

        self.runner.kv_caches = paged_kv_for_compilation
        duration = time.time() - start_time
        logger.info("KV cache swap pre-compilation finished in %.2f [secs].",
                    duration)

    def _bucketed_gather_kv_cache(
        self,
        kv_caches: list[jax.Array],
        block_ids: jax.Array,
    ) -> list[jax.Array]:
        """
        Gathers KV cache data for the given block_ids by breaking the operation
        into bucket-aligned chunks to leverage JIT compilation cache.
        """
        num_blocks = len(block_ids)
        if num_blocks == 0:
            return []
        if num_blocks in BLOCK_SIZE_BUCKETS:
            return KVCacheManager._jitted_gather_kv_cache(kv_caches, block_ids)

        decomposed_block_sizes = self._decompose_into_buckets(num_blocks)
        logger.info(
            f"Decomposing gather for {num_blocks} blocks into bucket sizes {decomposed_block_sizes}"
        )
        gathered_chunks = []
        block_offset = 0
        for decomposed_block_size in decomposed_block_sizes:
            block_slice = jax.lax.dynamic_slice_in_dim(block_ids,
                                                       block_offset,
                                                       decomposed_block_size,
                                                       axis=0)
            gathered_chunk = KVCacheManager._jitted_gather_kv_cache(
                kv_caches, block_slice)
            gathered_chunks.append(gathered_chunk)
            block_offset += decomposed_block_size

        # Reassemble the results from all chunks
        return jax.tree.map(lambda *x: jnp.concatenate(x, axis=0),
                            *gathered_chunks)

    def _bucketed_swap_out_fn(
            self,
            flat_kv_caches_tpu: list[jax.Array]) -> list[list[jax.Array]]:
        """
        Swaps out KV cache data from TPU to CPU in bucket-aligned chunks,
        returning a list of block-sized chunks per layer.
        """
        num_tokens = flat_kv_caches_tpu[0].shape[0]
        num_blocks = num_tokens // self.block_size
        if num_blocks == 0:
            return [[] for _ in range(self.num_layers)]

        # Fast path: handle bucket-sized transfers
        if num_blocks in BLOCK_SIZE_BUCKETS:
            split_size_list = [self.block_size] * num_blocks
            flat_kv_caches_cpu = self.swap_out_fn(flat_kv_caches_tpu)
            jax.block_until_ready(flat_kv_caches_cpu)
            return jax.tree.map(
                lambda flat_layer_cache: jax.lax.split(
                    flat_layer_cache, split_size_list, axis=0),
                flat_kv_caches_cpu)

        # Bucket decomposition path
        decomposed_block_sizes = self._decompose_into_buckets(num_blocks)
        logger.info(
            f"Decomposing swap-out for {num_blocks} blocks into bucket sizes {decomposed_block_sizes}"
        )
        # This will be a list of lists, where each inner list holds the chunks
        # for a layer.
        final_chunks_per_layer = [[] for _ in range(self.num_layers)]
        token_offset = 0
        for decomposed_block_size in decomposed_block_sizes:
            chunk_size_in_tokens = decomposed_block_size * self.block_size

            # Slice the TPU tensor for the current bucket
            tpu_chunk = [
                jax.lax.dynamic_slice_in_dim(layer_cache,
                                             token_offset,
                                             chunk_size_in_tokens,
                                             axis=0)
                for layer_cache in flat_kv_caches_tpu
            ]

            # Swap the bucket to CPU, result is a flat tensor for this bucket. We are doing the chunking inside this function to avoid returning any jnp.concatenate
            # of kv cache for the the bucketed blocks
            cpu_chunk_flat_per_layer = self.swap_out_fn(tpu_chunk)
            jax.block_until_ready(cpu_chunk_flat_per_layer)
            # Split the flat bucket tensor into block-sized chunks and append
            split_size_list = [self.block_size] * decomposed_block_size
            for i, layer_cache in enumerate(cpu_chunk_flat_per_layer):
                chunks = jax.lax.split(layer_cache, split_size_list, axis=0)
                final_chunks_per_layer[i].extend(chunks)

            token_offset += chunk_size_in_tokens

        return final_chunks_per_layer

    def _bucketed_swap_in_fn(
        self,
        assembled_kv_on_cpu: list[list[jax.Array]],
    ) -> list[list[jax.Array]]:
        """
        Swaps in KV cache data from CPU to TPU in bucket-aligned chunks,
        assembling a complete staging buffer on the TPU.
        """
        num_blocks = len(assembled_kv_on_cpu[0])
        if num_blocks == 0:
            return [[] for _ in range(self.num_layers)]
        if num_blocks in BLOCK_SIZE_BUCKETS:
            return self.swap_in_fn(assembled_kv_on_cpu)

        decomposed_block_sizes = self._decompose_into_buckets(num_blocks)
        logger.info(
            f"Decomposing swap-in for {num_blocks} blocks into bucket sizes {decomposed_block_sizes}"
        )

        tpu_chunks_per_layer = [[] for _ in range(self.num_layers)]
        block_offset = 0
        for decomposed_block_size in decomposed_block_sizes:
            cpu_chunks_for_bucket = [
                layer_chunks[block_offset:block_offset + decomposed_block_size]
                for layer_chunks in assembled_kv_on_cpu
            ]
            tpu_chunks_for_bucket = self.swap_in_fn(cpu_chunks_for_bucket)
            for i in range(self.num_layers):
                tpu_chunks_per_layer[i].extend(tpu_chunks_for_bucket[i])
            block_offset += decomposed_block_size

        return tpu_chunks_per_layer

    def _bucketed_jitted_insert_kv_cache_slices(
        self,
        kv_caches: list[jax.Array],
        kv_cache_slices: list[list[jax.Array]],
        dst_blocks: jax.Array,
    ) -> list[jax.Array]:
        """
        Inserts KV cache slices into the main cache in bucket-aligned chunks.
        """
        num_blocks = len(dst_blocks)
        if num_blocks == 0:
            return kv_caches
        if num_blocks in BLOCK_SIZE_BUCKETS:
            return jitted_insert_kv_cache_slices(self.block_size, kv_caches,
                                                 kv_cache_slices, dst_blocks)

        decomposed_block_sizes = self._decompose_into_buckets(num_blocks)
        logger.info(
            f"Decomposing insert for {num_blocks} blocks into bucket sizes {decomposed_block_sizes}"
        )

        updated_kv_caches = kv_caches
        block_offset = 0
        for decomposed_block_size in decomposed_block_sizes:
            slices_for_bucket = [
                layer_slices[block_offset:block_offset + decomposed_block_size]
                for layer_slices in kv_cache_slices
            ]
            dst_blocks_for_bucket = jax.lax.dynamic_slice_in_dim(
                dst_blocks, block_offset, decomposed_block_size, axis=0)

            updated_kv_caches = jitted_insert_kv_cache_slices(
                self.block_size, updated_kv_caches, slices_for_bucket,
                dst_blocks_for_bucket)

            block_offset += decomposed_block_size

        return updated_kv_caches

    def _save_blocks_to_cpu(self, req_id: ReqId, full_block_ids: list[int],
                            full_token_ids: list[int],
                            save_spec: SaveSpec) -> ReqId:
        """
        Extracts KV cache blocks from TPU, copies them to CPU, and updates the
        CPU backend with the new cache keys and their corresponding token data.
        """
        if not self.runner or not self.runner.kv_caches:
            logger.error(f"Cannot save blocks for request {req_id}: runner or "
                         "KV caches not registered.")
            return req_id

        blocks_to_save = save_spec.src_blocks
        dst_chunks = save_spec.dst_chunks

        num_total_tokens = save_spec.num_total_tokens
        num_skip_leading_tokens = save_spec.num_skip_leading_tokens
        num_blocks_to_save = len(blocks_to_save)

        assert num_total_tokens <= len(
            full_token_ids), f"{num_total_tokens} > {len(full_token_ids)}"

        num_tokens_to_save = num_total_tokens - num_skip_leading_tokens
        if num_tokens_to_save <= 0 and not save_spec.is_final_save:
            logger.info(f"Request {req_id}: No new tokens to save.")
            return req_id

        process_token_ids = full_token_ids[:num_total_tokens]
        tokens_to_save = process_token_ids[num_skip_leading_tokens:]

        logger.info(
            f"Request {req_id} save details: "
            f"full_block_ids len={len(full_block_ids)}, "
            f"num_skip_leading_tokens={num_skip_leading_tokens}, "
            f"num_total_tokens={num_total_tokens}, "
            f"num_tokens_to_save={num_tokens_to_save}, "
            f"blocks_to_save({len(blocks_to_save)}: {blocks_to_save}), "
            f"dst_chunks({len(dst_chunks)}: {dst_chunks}) ")

        if not blocks_to_save and tokens_to_save:
            logger.warning(
                f"Request {req_id}: Tokens to save but no corresponding blocks found."
            )
            return req_id

        if not tokens_to_save:
            logger.info(
                f"Request {req_id}: No new tokens to save, but processing as final save."
            )
            return req_id

        # Verify if blocks_to_save is a contiguous subarray of full_block_ids
        first_src_block = blocks_to_save[0]
        last_src_block = blocks_to_save[-1]
        try:
            first_block_idx_in_full = full_block_ids.index(first_src_block)
            last_block_idx_in_full = full_block_ids.index(last_src_block)
            if not (last_block_idx_in_full - first_block_idx_in_full + 1
                    == len(blocks_to_save)):
                raise ValueError(
                    f"Request({req_id}): blocks_to_save {blocks_to_save} does not exist in full_block_ids {full_block_ids}"
                )
        except ValueError:
            raise ValueError(
                f"Request({req_id}): blocks_to_save {blocks_to_save} contains blocks not present in local_block_ids {full_block_ids}"
            )

        try:
            start_time = time.time()
            blocks_to_save = jnp.array(blocks_to_save)
            if self.use_bucketed_swap_ops:
                flat_kv_caches_tpu = self._bucketed_gather_kv_cache(
                    self.runner.kv_caches, blocks_to_save)
            else:
                flat_kv_caches_tpu = KVCacheManager._jitted_gather_kv_cache(
                    self.runner.kv_caches, blocks_to_save)

            jax.block_until_ready(flat_kv_caches_tpu)
            logger.info(
                f"extracted_blocks_tpu: {flat_kv_caches_tpu[0].shape}, {flat_kv_caches_tpu[0].sharding}"
            )

            chunks_on_cpu = None
            if self.use_bucketed_swap_ops:
                chunks_on_cpu = self._bucketed_swap_out_fn(flat_kv_caches_tpu)
            else:
                flat_kv_caches_cpu = self.swap_out_fn(flat_kv_caches_tpu)
                if flat_kv_caches_cpu:
                    jax.block_until_ready(flat_kv_caches_cpu)
                    # NOTE(jcgu): we keep cpu_chunk_size == block_size
                    split_size_list = [self.cpu_chunk_size
                                       ] * num_blocks_to_save
                    chunks_on_cpu = jax.tree.map(
                        lambda flat_layer_cache: jax.lax.split(
                            flat_layer_cache, split_size_list, axis=0),
                        flat_kv_caches_cpu)

            if chunks_on_cpu and chunks_on_cpu[0]:
                jax.block_until_ready(chunks_on_cpu)

            duration = time.time() - start_time
            logger.info(f"Successfully saved {len(blocks_to_save)} blocks for "
                        f"request {req_id} to CPU in {duration:.4f} seconds.")

            total_size_bytes = sum(
                sum(chunk.nbytes for chunk in layer_chunks)
                for layer_chunks in chunks_on_cpu)
            logger.info(
                f"Total size of chunks_on_cpu: {total_size_bytes / 1024**2:.2f} MB"
            )

            post_transfer_start_time = time.time()

            for i in range(num_blocks_to_save):
                chunk_id = dst_chunks[i]
                cur_chunk_cross_layers = [
                    chunks_on_cpu[j][i] for j in range(self.num_layers)
                ]
                self.cpu_backend.add(chunk_id, cur_chunk_cross_layers)
                logger.info(f"Request {req_id}: Saving to CPU chunk: "
                            f"chunk_id={chunk_id}, "
                            f" local_chunk_idx={i}")

            logger.info(
                f"Request {req_id}: Added {num_blocks_to_save} chunks to CPU backend."
            )

            post_transfer_duration = time.time() - post_transfer_start_time
            logger.info(
                f"Request {req_id}: e2e host processing of {num_blocks_to_save} chunks took {post_transfer_duration:.4f} seconds."
            )
        except Exception as e:
            logger.error(f"Error saving blocks for request {req_id}: {e}",
                         exc_info=True)

        return req_id

    def wait_for_save(self):
        """
        Initiates and waits for all pending asynchronous save operations for the
        current step to complete.
        """
        assert self.cpu_backend, "please initialize cpu_backend first."
        # This method is idempotent. If the save operations for the current
        # step's metadata have already been processed, we can exit early.
        if self._processed_save_for_step:
            return

        # logger.info("TPUOffloadConnectorWorker: Entering wait_for_save")
        metadata = self.connector._get_connector_metadata()
        if not isinstance(metadata, TPUOffloadConnectorMetadata):
            logger.info(
                "wait_for_save:not an instances of TPUOffloadConnectorMetadata"
            )
            self._processed_save_for_step = True
            return

        if not metadata.requests_meta:
            # logger.info("wait_for_save:no reqs to save")
            self._processed_save_for_step = True
            return

        pending_save_futures: list[tuple[Future, TPUReqMeta]] = []
        # Handle save requests
        for meta in metadata.requests_meta:
            if meta.save_spec:
                if meta.save_spec.skip_save:
                    logger.info(
                        f"Request {meta.req_id}: Scheduler signaled to skip save."
                    )
                    if meta.save_spec.is_final_save:
                        logger.info(
                            f"Request {meta.req_id}: Final save is a no-op. Marking as finished."
                        )
                        # self.finished_save_reqs.add(meta.req_id)
                    continue

                # If there are tokens to save, submit the task to the thread pool.
                logger.info(f"Submitting save task for request {meta.req_id}")
                future = self.save_executor.submit(self._save_blocks_to_cpu,
                                                   meta.req_id,
                                                   meta.local_block_ids,
                                                   meta.token_ids,
                                                   meta.save_spec)
                pending_save_futures.append((future, meta))

        if not pending_save_futures:
            self._processed_save_for_step = True
            return

        logger.info(f"Waiting for {len(pending_save_futures)} save "
                    "operations to complete...")
        start_time = time.time()

        for future, meta in pending_save_futures:
            try:
                # The result of _save_blocks_to_cpu is the request_id
                finished_req_id = future.result()
                logger.info(
                    f"Save operation completed for request {finished_req_id}")

                if len(meta.save_spec.src_blocks) > 0:
                    self.offload_stats.record_save(
                        req=finished_req_id,
                        saved_chunk_ids=meta.save_spec.dst_chunks)

                if meta.save_spec and meta.save_spec.is_final_save:
                    logger.info(
                        f"Request {finished_req_id}: Final save completed. Marking as finished."
                    )
                    self.finished_save_reqs.add(finished_req_id)

            except Exception as e:
                logger.error(f"A save operation failed: {e}", exc_info=True)

        duration = time.time() - start_time
        logger.info(f"All {len(pending_save_futures)} save operations "
                    f"completed in {duration:.4f} seconds.")
        self._processed_save_for_step = True

    def start_load_kv(self, fwd_ctx: "ForwardContext") -> None:
        """
        This function is the worker-side entry point for loading data from the
        local CPU backend into the TPU's sharded KV cache. It is a blocking
        operation that ensures the cache is fully updated before the model's
        forward pass begins.
        """
        assert self.cpu_backend, "please initialize cpu_backend first."
        # Reset the save processing flag at the start of a new step.
        self._processed_save_for_step = False
        metadata = self.connector._get_connector_metadata()
        if not isinstance(
                metadata,
                TPUOffloadConnectorMetadata) or not metadata.requests_meta:
            logger.info("No load operations scheduled for this step.")
            return

        if not self.device_sharding:
            raise RuntimeError(
                "KV cache sharding info not available. Was register_runner called?"
            )

        assert self.runner is not None and self.runner.kv_caches is not None

        # Process each request that needs its KV cache loaded
        load_times = []
        for meta in metadata.requests_meta:
            if not (meta.load_spec and meta.load_spec.can_load):
                continue

            request_load_start_time = time.time()
            logger.info(
                "TPUOffloadConnectorWorker: Starting KV cache load process.")
            dst_blocks = meta.load_spec.dst_blocks
            src_chunks = meta.load_spec.src_chunks
            num_blocks_to_load = len(dst_blocks)
            num_matched_tokens = meta.load_spec.num_matched_tokens
            num_skip_leading_tokens = meta.load_spec.num_skip_leading_tokens
            num_tokens_to_load_delta = num_matched_tokens - num_skip_leading_tokens
            assert num_skip_leading_tokens % self.block_size == 0, f"{num_skip_leading_tokens} % {self.block_size} != 0"

            if num_tokens_to_load_delta <= 0:
                logger.info(
                    f"Request {meta.req_id}: No new tokens to load. Skipping.")
                continue

            assert num_blocks_to_load > 0, f"Request({meta.req_id}) has no dst blocks to load."
            # Verify if dst_blocks is a contiguous subarray of meta.local_block_ids
            first_dst_block = dst_blocks[0]
            last_dst_block = dst_blocks[-1]
            try:
                first_block_idx_in_local = meta.local_block_ids.index(
                    first_dst_block)
                last_block_idx_in_local = meta.local_block_ids.index(
                    last_dst_block)
                if not (last_block_idx_in_local - first_block_idx_in_local + 1
                        == len(dst_blocks)):
                    raise ValueError(
                        f"Request({meta.req_id}): dst_blocks {dst_blocks} does not exist in local_block_ids {meta.local_block_ids}"
                    )
            except ValueError:
                raise ValueError(
                    f"Request({meta.req_id}): dst_blocks {dst_blocks} contains blocks not present in local_block_ids {meta.local_block_ids}"
                )

            logger.info(
                f"Processing KV load for request {meta.req_id}: "
                f"Total matched: {num_matched_tokens}, "
                f"Already computed: {num_skip_leading_tokens}. "
                f"Fetching delta of {num_tokens_to_load_delta} tokens from cache for "
                f"{num_blocks_to_load} blocks.")

            # Assemble the per-layer data for the delta tokens on the CPU.
            # We create a list of lists, where the outer list represents layers
            # and the inner lists will hold the data chunks for that layer.
            assembled_kv_on_cpu = [[] for _ in range(self.num_layers)]
            # Fetch and chunks from the backend.
            for i in range(num_blocks_to_load):
                src_chunk_id = src_chunks[i]
                cached_value = self.cpu_backend.get(src_chunk_id)
                if cached_value:
                    for j in range(self.num_layers):
                        assembled_kv_on_cpu[j].append(cached_value[j])
                else:
                    logger.error(
                        f"Chunk[{src_chunk_id}] not found in CPU backend for request {meta.req_id}. Inconsistent state detected."
                    )
                    return

            # swap-in
            # output: [[cpu_chunk_size * num_chunks] * num_layer]
            if self.use_bucketed_swap_ops:
                # Use the bucketed wrappers for a uniform two-step process
                raw_chunked_kv_on_tpu = self._bucketed_swap_in_fn(
                    assembled_kv_on_cpu)
            else:
                raw_chunked_kv_on_tpu = self.swap_in_fn(assembled_kv_on_cpu)
            jax.block_until_ready(raw_chunked_kv_on_tpu)

            if self.use_bucketed_swap_ops:
                self.runner.kv_caches = self._bucketed_jitted_insert_kv_cache_slices(
                    self.runner.kv_caches,
                    raw_chunked_kv_on_tpu,
                    jnp.array(dst_blocks),
                )
            else:
                self.runner.kv_caches = jitted_insert_kv_cache_slices(
                    self.block_size,
                    self.runner.kv_caches,
                    raw_chunked_kv_on_tpu,
                    jnp.array(dst_blocks),
                )
            jax.block_until_ready(self.runner.kv_caches)
            logger.info(
                f"Request {meta.req_id}: Loaded {num_tokens_to_load_delta} tokens into "
                f"{num_blocks_to_load} new blocks.")

            load_times.append(time.time() - request_load_start_time)
            self.finished_load_reqs.add(meta.req_id)
            if num_blocks_to_load > 0:
                self.offload_stats.record_load(req=meta.req_id,
                                               loaded_chunk_ids=src_chunks)

        if load_times:
            aggregate_load_time = sum(load_times)
            logger.info(
                f"TPUOffloadConnectorWorker: Aggregate KV cache load time for {len(load_times)} requests: {aggregate_load_time:.4f} seconds"
            )

    def get_kv_connector_stats(self) -> KVConnectorStats | None:
        """
        Get the KV transfer stats for the connector.
        """
        # Clear stats for next iteration
        if not self.offload_stats.is_empty():
            return self.offload_stats.clone_and_reset()
        return None

    def get_finished(self) -> tuple[set[str], set[str]]:
        """
        Returns the sets of request IDs for completed save and load operations.
        """
        # Safeguard call to wait_for_save().
        # In the final step for a request, the vLLM engine may not call
        # `worker.execute_model()` if there's no computation to be done.
        # This skips the usual `wait_for_save()` call, preventing the final
        # save operation (marked with `is_final_save=True`) from being
        # processed. Calling it here ensures that any pending save operations
        # for the current step's metadata are executed, and the finished
        # request IDs are correctly identified and reported back to the engine
        # for resource cleanup. The `wait_for_save` method is idempotent,
        # so this call is a no-op in the normal execution path.
        logger.info("TPUOffloadConnectorWorker: Entering get_finished")
        self.wait_for_save()

        finished_saves = self.finished_save_reqs
        self.finished_save_reqs = set()
        finished_loads = self.finished_load_reqs
        self.finished_load_reqs = set()
        logger.info(f"Finished saves: {finished_saves}, "
                    f"Finished loads: {finished_loads}")
        return finished_saves, finished_loads
