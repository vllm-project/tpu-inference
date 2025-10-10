# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Scheduler side execution:
TPUConnectorScheduler manages the state of KV cache loading and saving for
each request. It acts as a state machine, tracking the progress of requests
across multiple scheduling steps and generating work orders (TPUReqMeta) for
the TPUConnectorWorker.

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
    contains `skip_leading_tokens` to indicate which part of the request's
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
        - `skip_leading_tokens` is set to `tracker.num_saved_tokens`. This
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
- The TPUConnectorWorker receives the `TPUConnectorMetadata` containing the list of
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
import functools
import os
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional
from uuid import uuid4

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh
from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import \
    KVConnectorBase_V1, KVConnectorMetadata, KVConnectorRole
from vllm.utils import cdiv
from vllm.v1.core.sched.output import SchedulerOutput

if TYPE_CHECKING:
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.request import Request

from .local_cpu_backend import LocalCPUBackend
from .util import ChunkedTokens
from tpu_inference.logger import init_logger
from tpu_inference.runner.tpu_jax_runner import TPUModelRunner
from tpu_inference.utils import device_array

EngineId = str
ReqId = str

logger = init_logger(__name__)


@dataclass
class SaveSpec:
    """A confirmed work order for the worker to save KV data."""
    skip_leading_tokens: int
    is_final_save: bool = False


@dataclass
class LoadSpec:
    """Internal scheduler state for a potential load operation."""
    num_matched_tokens: int
    can_load: bool = False


@dataclass
class TPUReqMeta:
    """A unified work order for a single request in a single step."""
    req_id: str
    token_ids: list[int]
    local_block_ids: list[int]
    save_spec: Optional[SaveSpec] = None
    load_spec: Optional[LoadSpec] = None


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
    requests_meta: list[TPUReqMeta] = field(default_factory=list)


class TPUConnector(KVConnectorBase_V1):

    def __init__(self, vllm_config: VllmConfig, role: KVConnectorRole):
        logger.info("TPUConnector: Entering __init__")
        assert vllm_config.kv_transfer_config is not None

        if role == KVConnectorRole.SCHEDULER:
            self.connector_scheduler = \
                TPUConnectorScheduler(vllm_config)
            self.connector_worker = None
        elif role == KVConnectorRole.WORKER:
            self.connector_scheduler = None
            # The worker needs a reference to the base connector to access
            # the metadata object set by the engine.
            self.connector_worker = TPUConnectorWorker(vllm_config, self)

    ############################################################
    # Scheduler Side Methods
    ############################################################
    def get_num_new_matched_tokens(
            self, request: "Request",
            num_computed_tokens: int) -> tuple[int, bool]:
        # logger.info("TPUConnector: Entering get_num_new_matched_tokens")
        assert self.connector_scheduler is not None
        return self.connector_scheduler.get_num_new_matched_tokens(
            request,
            num_computed_tokens)

    def update_state_after_alloc(self, request: "Request",
                                 blocks: "KVCacheBlocks",
                                 num_external_tokens: int):
        assert self.connector_scheduler is not None
        return self.connector_scheduler.update_state_after_alloc(
            request,
            blocks,
            num_external_tokens)

    def build_connector_meta(
        self,
        scheduler_output: SchedulerOutput,
    ) -> TPUConnectorMetadata:
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
        logger.info("TPUConnector: Entering register_kv_caches")
        """
        We don't register kv_caches in connector, we call `register_runner` and
        use runner.kv_caches directly instead because the ref of runner.kv_caches
        would be reassigned during model forward.
        """
        pass

    def register_runner(self, runner: TPUModelRunner) -> None:
        logger.info("TPUConnector: Entering register_runner")
        assert self.connector_worker is not None
        self.connector_worker.register_runner(runner)

    def start_load_kv(self, fwd_ctx: "ForwardContext") -> None:
        """Starts loading the KV cache for the given requests."""
        assert self.connector_worker is not None
        self.connector_worker.start_load_kv(fwd_ctx)

    def wait_for_layer_load(self, layer_name: str) -> None:
        logger.info("TPUConnector: Entering wait_for_layer_load")
        """TPU connector doesn't support layer wise load."""
        pass

    def save_kv_layer(self, **kwargs) -> None:
        logger.info("TPUConnector: Entering save_kv_layer")
        """TPU connector doesn't support layer wise save."""
        pass

    def wait_for_save(self):
        assert isinstance(self._connector_metadata, TPUConnectorMetadata)
        self.connector_worker.wait_for_save()

    def get_finished(
            self,
            finished_req_ids: set[str]) -> tuple[set[str], set[str]]:
        assert self.connector_worker is not None
        return self.connector_worker.get_finished()


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
    # request so far. In contrast to LMCache, this list only contains the
    # tokens to be computed, not the prefix loaded from cache.
    token_ids: list[int]
    # The number of tokens that were a hit in the CPU cache at the beginning
    # of the request. This is constant for the lifetime of the request.
    num_cached_tokens: int = 0
    # A high-water mark indicating how many tokens from the start of the
    # computed tokens (`token_ids`) have already been saved to the CPU cache.
    num_saved_tokens: int = 0

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
            raise ValueError(f"Unsupported new_block_ids type {type(new_block_ids)}")
        self.block_ids.extend(new_block_ids)
        self.token_ids.extend(new_token_ids)


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


class TPUConnectorScheduler():
    def __init__(self, vllm_config: "VllmConfig"):
        logger.info("TPUConnectorScheduler: Entering __init__")
        self.vllm_config = vllm_config
        self.config = vllm_config.kv_transfer_config
        self.block_size = vllm_config.cache_config.block_size
        logger.info("Block size is %d", self.block_size)
        self.kv_transfer_host = self.config.kv_ip
        self.kv_transfer_port = self.config.kv_port

        self._request_trackers: dict[ReqId, RequestTracker] = {}
        # This dictionary holds the full vLLM Request object for all requests
        # that are currently in a running state (i.e., have been scheduled but
        # are not yet finished). It's used to access the complete prompt token
        # list when processing incremental updates for cached/running requests,
        # as the scheduler output for these requests is minimal.
        self._unfinished_requests: dict[ReqId, "Request"] = {}
        self.load_specs: dict[ReqId, LoadSpec] = {}
        self.cpu_backend = LocalCPUBackend()
        model_name = self.vllm_config.model_config.model
        logger.info("Model name is %s", model_name)
        self.token_processor = ChunkedTokens(model_name=model_name)

    def get_num_new_matched_tokens(
        self,
        request: "Request",
        num_computed_tokens: int,
    ) -> tuple[int, bool]:
        """
        Checks for external KV cache hit against the local CPU backend.
        """
        prompt_token_ids = request.prompt_token_ids
        logger.info(
            f"Request {request.request_id}: Checking for cache hit. "
            f"Prompt length: {len(prompt_token_ids)}, "
            f"Already computed tokens: {num_computed_tokens}."
        )

        # Generate keys for the incoming request's tokens
        request_keys = self.token_processor.process_tokens(prompt_token_ids)

        num_matched_tokens = 0
        # The generator needs to be consumed to count.
        keys = list(request_keys)
        for start_idx, end_idx, key in keys:
            logger.debug(f"  Processing chunk {start_idx}-{end_idx} with hash {key.chunk_hash}")
            if self.cpu_backend.contains(key):
                num_matched_tokens = end_idx
                logger.debug(f"  -> HIT. Total matched tokens so far: {num_matched_tokens}")
            else:
                # Stop at the first cache miss
                logger.debug("  -> MISS. Stopping search.")
                break
        
        logger.info(f"Request {request.request_id}: Found {num_matched_tokens} matched tokens in CPU backend.")
        # We don't need to load tokens that are already computed locally in vLLM
        num_to_load = max(0, num_matched_tokens - num_computed_tokens)
        logger.info(f"Request {request.request_id}: After accounting for {num_computed_tokens} computed tokens, {num_to_load} tokens will be loaded.")

        if num_to_load > 0:
            self.load_specs[request.request_id] = LoadSpec(
                num_matched_tokens=num_matched_tokens
            )
        return num_to_load, False

    def update_state_after_alloc(self, request: "Request",
                                 blocks: "KVCacheBlocks",
                                 num_external_tokens: int):
        """
        This hook is not used for the save logic. Trackers are created
        and managed within `build_connector_meta`.
        """
        logger.info(
            f"TPUConnectorScheduler: Entering update_state_after_alloc Request {request.request_id}: Scheduler allocated "
            f"{num_external_tokens} external tokens."
        )
        self._unfinished_requests[request.request_id] = request
        if num_external_tokens > 0:
            if request.request_id in self.load_specs:
                self.load_specs[request.request_id].can_load = True
                logger.info(
                    f"Request {request.request_id}: Marked as ready to load."
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
        save_spec = None
        
        # 1. Decide if a save is necessary.
        if len(tracker.token_ids) > tracker.num_saved_tokens:
            save_spec = SaveSpec(
                skip_leading_tokens=tracker.num_saved_tokens,
                is_final_save=is_finished,
            )
            # Transactional update: advance the state *before* dispatching.
            tracker.num_saved_tokens = len(tracker.token_ids)

        # Also consider a save if the request is finished, even with no new tokens,
        # to signal the worker this is the final step.
        elif is_finished:
             save_spec = SaveSpec(
                skip_leading_tokens=tracker.num_saved_tokens,
                is_final_save=True,
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
        self, scheduler_output: SchedulerOutput
    ) -> TPUConnectorMetadata:
        metadata = TPUConnectorMetadata()

        # Phase 1: Handle and clean up finished requests
        # This block handles requests that have completed their generation.
        # We pop their state from our tracking dictionaries and call _prepare_req_meta
        # one last time. This ensures any final, unsaved tokens are captured and
        # signals to the worker that this is the final save for the request.
        logger.info(f"Phase 1: Processing {len(scheduler_output.finished_req_ids)} finished requests.")
        for finished_req_id in scheduler_output.finished_req_ids:
            logger.info(f"  - Processing finished req: {finished_req_id}")
            # Pop tracker and other state first.
            tracker = self._request_trackers.pop(finished_req_id, None)
            self._unfinished_requests.pop(finished_req_id, None)
            self.load_specs.pop(finished_req_id, None)
            
            if not tracker:
                logger.warning(f"  - No tracker found for finished req: {finished_req_id}. Skipping.")
                continue

            # Prepare one final metadata object if there's a final save needed.
            # `is_finished` is set to True to flag this as the last save operation.
            req_meta = self._prepare_req_meta(tracker, load_spec=None, is_finished=True)
            if req_meta:
                logger.info(f"  - Creating final save metadata for req: {finished_req_id}")
                metadata.requests_meta.append(req_meta)

        # Phase 2: Process newly scheduled requests
        # This block handles requests being scheduled for the very first time.
        # It creates the initial RequestTracker and prepares the first work order.
        logger.info(f"Phase 2: Processing {len(scheduler_output.scheduled_new_reqs)} new requests.")
        for request in scheduler_output.scheduled_new_reqs:
            req_id = request.req_id
            logger.info(f"  - Processing new req: {req_id}")
            # num_new_tokens: The number of tokens the scheduler has decided to
            # process in this very first step for this request.
            num_new_tokens = scheduler_output.num_scheduled_tokens[req_id]
            
            # load_spec/num_cached_tokens: The result from the earlier
            # `get_num_new_matched_tokens` call. This tells us how many tokens
            # were found in the CPU cache.
            load_spec = self.load_specs.get(req_id)
            num_cached_tokens = load_spec.num_matched_tokens if load_spec else 0
            
            # start_idx/end_idx: These are calculated to slice the full prompt.
            # We only need to process the tokens that were *not* a cache hit.
            start_idx = num_cached_tokens
            end_idx = num_cached_tokens + num_new_tokens
            
            # initial_tokens: The actual slice of tokens that vLLM will compute
            # in this first step.
            tokens_for_tracker = request.prompt_token_ids[:end_idx]
            logger.info(f"    - num_new_tokens: {num_new_tokens}, num_cached_tokens: {num_cached_tokens}")
            logger.info(f"    - Slicing prompt[:{end_idx}] -> len(tokens_for_tracker): {len(tokens_for_tracker)}")

            # Create and store the tracker, which will maintain the request's
            # state for its entire lifetime.
            tracker = RequestTracker(
                req_id=req_id,
                prompt_len=len(request.prompt_token_ids),
                block_ids=request.block_ids[0],
                token_ids=tokens_for_tracker,
                num_cached_tokens=num_cached_tokens,
                # The high-water mark for saved tokens starts after the cached prefix.
                num_saved_tokens=num_cached_tokens,
            )
            self._request_trackers[req_id] = tracker
            logger.info(f"    - Created tracker for {req_id} with initial state: "
                        f"prompt_len={tracker.prompt_len}, "
                        f"num_tokens={len(tracker.token_ids)}, "
                        f"num_blocks={len(tracker.block_ids)}, "
                        f"num_saved={tracker.num_saved_tokens}")

            # Immediately prepare metadata for this new request. This could include
            # both a load operation (for the cached part) and a save operation
            # (for the newly computed part).
            req_meta = self._prepare_req_meta(tracker, load_spec, is_finished=False)
            if req_meta:
                logger.info(f"    - Creating metadata for new req: {req_id} "
                            f"(has_load={req_meta.load_spec is not None}, "
                            f"has_save={req_meta.save_spec is not None})")
                metadata.requests_meta.append(req_meta)

        # Phase 3: Process cached (running) requests
        # This block handles requests that have already been pre-filled at least
        # once and are now being processed again (e.g., for chunked prefill).
        cached_reqs = scheduler_output.scheduled_cached_reqs
        logger.info(f"Phase 3: Processing {len(cached_reqs.req_ids)} cached requests.")
        for i, req_id in enumerate(cached_reqs.req_ids):
            logger.info(f"  - Processing cached req: {req_id}")
            tracker = self._request_trackers[req_id]
            full_request = self._unfinished_requests.get(req_id)
            
            # num_new_tokens: The number of *additional* tokens the scheduler is
            # processing in this step for this ongoing request.
            num_new_tokens = scheduler_output.num_scheduled_tokens[req_id]
            
            # current_token_count: This is the crucial calculation to find our
            # place in the full prompt. It's the length of the token prefix
            # already processed in previous steps.
            current_token_count = len(tracker.token_ids)
            
            # new_token_ids: The slice of the full prompt corresponding to the
            # new work being done in this step.
            new_token_ids = full_request.prompt_token_ids[current_token_count : current_token_count + num_new_tokens]
            
            # new_blocks: The new physical blocks allocated for the new_token_ids.
            new_blocks = cached_reqs.new_block_ids[i]
            if new_blocks is None:
                new_blocks = []
            
            logger.info(f"    - num_new_tokens: {num_new_tokens}, current_token_count: {current_token_count}")
            logger.info(f"    - Slicing prompt -> len(new_token_ids): {len(new_token_ids)}")
            logger.info(f"    - New blocks allocated: {len(new_blocks)}")

            # Update the tracker with the incremental data.
            tracker.update(new_blocks, new_token_ids)
            logger.info(f"    - Updated tracker for {req_id}: "
                        f"total_tokens={len(tracker.token_ids)}, "
                        f"total_blocks={len(tracker.block_ids)}")

            # Immediately prepare metadata for this updated request. This will
            # typically be a save operation for the new tokens.
            req_meta = self._prepare_req_meta(tracker, load_spec=None, is_finished=False)
            if req_meta:
                logger.info(f"    - Creating metadata for cached req: {req_id} "
                            f"(has_save={req_meta.save_spec is not None})")
                metadata.requests_meta.append(req_meta)
        
        if metadata.requests_meta:
            logger.info(f"Prepared {len(metadata.requests_meta)} requests for worker.")
        return metadata

    def request_finished(
        self,
        request: "Request",
        block_ids: list[int],
    ) -> tuple[bool, Optional[dict[str, Any]]]:
        """
        Signals to the scheduler that the connector is handling the finished
        request asynchronously. The actual logic to prepare the save operation
        occurs in `build_connector_meta`.
        """
        logger.info("TPUConnectorScheduler: Entering request_finished")
        # Return True to indicate the request is being saved asynchronously
        # and its blocks should not be freed yet.
        return True, None


class TPUConnectorWorker:
    def __init__(self, vllm_config: VllmConfig, connector: "TPUConnector"):
        logger.info("TPUConnectorWorker: Entering __init__")
        self.vllm_config = vllm_config
        self.config = vllm_config.kv_transfer_config
        self.connector = connector
        self.block_size = vllm_config.cache_config.block_size

        self.runner: Optional[TPUModelRunner] = None
        self.mesh: Optional[Mesh] = None

        self.host = self.config.kv_ip
        self.kv_transfer_port = self.config.kv_port

        # Get the singleton instance of the CPU backend.
        self.cpu_backend = LocalCPUBackend()
        # The worker needs its own token processor to generate keys.
        model_name = self.vllm_config.model_config.model
        self.token_processor = ChunkedTokens(model_name=model_name)

        # Thread pool for asynchronous TPU->CPU copies
        self.save_executor = ThreadPoolExecutor(max_workers=4,
                                                thread_name_prefix="tpu_saver")
        self.finished_save_reqs: set[ReqId] = set()
        self.finished_load_reqs: set[ReqId] = set()

    def __del__(self):
        logger.info("TPUConnectorWorker: Entering __del__")
        self.save_executor.shutdown(wait=True)

    def _save_blocks_to_cpu(self, req_id: ReqId, full_block_ids: list[int], full_token_ids: list[int], save_spec: SaveSpec) -> ReqId:
        """
        Extracts KV cache blocks from TPU, copies them to CPU, and updates the
        CPU backend with the new cache keys and their corresponding token data.
        """
        if not self.runner or not self.runner.kv_caches:
            logger.error(f"Cannot save blocks for request {req_id}: runner or "
                         "KV caches not registered.")
            return req_id
        
        # --- NEW LOGIC: SLICE DATA BASED ON SPEC ---
        num_tokens_to_save = len(full_token_ids) - save_spec.skip_leading_tokens
        if num_tokens_to_save <= 0 and not save_spec.is_final_save:
            logger.info(f"Request {req_id}: No new tokens to save.")
            return req_id

        tokens_to_process = full_token_ids[save_spec.skip_leading_tokens:]
        
        # Calculate the block slice.
        start_block_idx = save_spec.skip_leading_tokens // self.block_size
        blocks_to_process = full_block_ids[start_block_idx:]

        if not blocks_to_process and tokens_to_process:
            logger.warning(f"Request {req_id}: Tokens to save but no corresponding blocks found.")
            return req_id
        
        if not tokens_to_process:
            logger.info(f"Request {req_id}: No new tokens to save, but processing as final save.")
            return req_id

        try:
            start_time = time.time()
            cpu_device = jax.devices("cpu")[0]

            # Extract blocks on TPU first
            extracted_blocks_tpu = [
                layer_cache_tpu[blocks_to_process, ...] 
                for layer_cache_tpu in self.runner.kv_caches
            ]

            # Initiate non-blocking copy to CPU
            kv_caches_on_cpu = [
                jax.device_put(extracted_blocks, cpu_device)
                for extracted_blocks in extracted_blocks_tpu
            ]

            # Block until the transfer is complete
            if kv_caches_on_cpu:
                jax.block_until_ready(kv_caches_on_cpu)

            duration = time.time() - start_time
            logger.info(f"Successfully saved {len(blocks_to_process)} blocks for "
                        f"request {req_id} to CPU in {duration:.4f} seconds.")

            if kv_caches_on_cpu:
                logger.info(f"Shape of a single layer on CPU before reshape (num_blocks, block_size, ...): {kv_caches_on_cpu[0].shape}")

            # Reshape per-layer data from (num_blocks, block_size, ...) to
            # a flat (total_tokens, ...) array for easy slicing.
            flat_kv_caches_on_cpu = [
                layer_cache.reshape(-1, *layer_cache.shape[2:])
                for layer_cache in kv_caches_on_cpu
            ]

            if flat_kv_caches_on_cpu:
                logger.info(f"Shape of a single layer after reshape (total_tokens, ...): {flat_kv_caches_on_cpu[0].shape}")

            # Generate keys and store them with their corresponding token data.
            keys_generator = self.token_processor.process_tokens(tokens_to_process)
            keys = list(keys_generator)
            if keys:
                for start_idx, end_idx, key in keys:
                    # For each key, slice the corresponding token data from each layer.
                    value_for_key = [
                        flat_layer_cache[start_idx:end_idx]
                        for flat_layer_cache in flat_kv_caches_on_cpu
                    ]
                    self.cpu_backend.add(key, value_for_key)

                logger.info(f"Updated CPU backend with {len(keys)} new key-value pairs for request {req_id}.")
            else:
                logger.info(f"No new keys generated for request {req_id} ({len(tokens_to_process)} tokens).")

        except Exception as e:
            logger.error(f"Error saving blocks for request {req_id}: {e}",
                         exc_info=True)

        return req_id

    def wait_for_save(self):
        """
        Initiates and waits for all pending asynchronous save operations for the
        current step to complete.
        """
        # logger.info("TPUConnectorWorker: Entering wait_for_save")
        metadata = self.connector._connector_metadata
        if not isinstance(metadata, TPUConnectorMetadata):
            logger.info("wait_for_save:not an instances of TPUConnectorMetadata")
            return
        
        if not metadata.requests_meta:
            # logger.info("wait_for_save:no reqs to save")
            return

        pending_save_futures: list[tuple[Future, TPUReqMeta]] = []
        # Handle save requests
        for meta in metadata.requests_meta:
            if meta.save_spec:
                logger.info(f"Submitting save task for request {meta.req_id}")
                future = self.save_executor.submit(self._save_blocks_to_cpu,
                                                   meta.req_id,
                                                   meta.local_block_ids,
                                                   meta.token_ids,
                                                   meta.save_spec)
                pending_save_futures.append((future, meta))

        if not pending_save_futures:
            return

        logger.info(f"Waiting for {len(pending_save_futures)} save "
                    "operations to complete...")
        start_time = time.time()
      
        for future, meta in pending_save_futures:
            try:
                # The result of _save_blocks_to_cpu is the request_id
                finished_req_id = future.result()
                if meta.save_spec and meta.save_spec.is_final_save:
                    logger.info(f"Request {finished_req_id}: Final save completed. Marking as finished.")
                    self.finished_save_reqs.add(finished_req_id)
            except Exception as e:
                logger.error(f"A save operation failed: {e}", exc_info=True)

        duration = time.time() - start_time
        logger.info(f"All {len(pending_save_futures)} save operations "
                    f"completed in {duration:.4f} seconds.")

    def register_runner(self, runner: TPUModelRunner):
        logger.info("TPUConnectorWorker: Entering register_runner")
        self.runner = runner
        self.mesh = runner.mesh
        # Get the spec of the kv_caches
        kv_caches = runner.kv_caches
        if kv_caches:
            kv_layer = kv_caches[0]
            self.num_layers = len(kv_caches)
            self.shape = list(kv_layer.shape)
            self.dtype = kv_layer.dtype
            self.sharding = kv_layer.sharding
            logger.info("KV Cache details registered in TPUConnectorWorker:")
            logger.info(f"  - Num layers: {self.num_layers}")
            logger.info(f"  - Shape per layer: {self.shape}")
            logger.info(f"  - DType: {self.dtype}")
            logger.info(f"  - Sharding: {self.sharding}")
        else:
            logger.warning("TPUConnectorWorker registered with no KV caches.")

    def start_load_kv(self, fwd_ctx: "ForwardContext") -> None:
        """
        This function is the worker-side entry point for loading data from the
        local CPU backend into the TPU's sharded KV cache. It is a blocking
        operation that ensures the cache is fully updated before the model's
        forward pass begins.
        """
        load_start_time = time.time()
        metadata = self.connector._get_connector_metadata()
        if not isinstance(metadata, TPUConnectorMetadata) or not metadata.requests_meta:
            logger.info("No load operations scheduled for this step.")
            return

        if not self.sharding:
            raise RuntimeError("KV cache sharding info not available. Was register_runner called?")

        assert self.runner is not None and self.runner.kv_caches is not None

        # Process each request that needs its KV cache loaded
        for meta in metadata.requests_meta:
            if not (meta.load_spec and meta.load_spec.can_load):
                continue

            logger.info("TPUConnectorWorker: Starting KV cache load process.")
            tokens_to_load = meta.token_ids[:meta.load_spec.num_matched_tokens]
            logger.info(
                f"Processing KV load for request {meta.req_id}: "
                f"{len(tokens_to_load)} tokens into "
                f"{len(meta.local_block_ids)} blocks."
            )

            # 1. Generate keys and fetch data chunks from the CPU backend.
            # The token processor regenerates the same keys that were used for saving.
            keys_generator = self.token_processor.process_tokens(tokens_to_load)
            keys = list(keys_generator)

            if not keys:
                logger.warning(f"Could not generate any keys for loading request {meta.req_id}. Aborting load.")
                continue

            # 2. Assemble the per-layer data on the CPU.
            # We create a list of lists, where the outer list represents layers
            # and the inner lists will hold the data chunks for that layer.
            assembled_kv_on_cpu = []
            for _ in range(self.num_layers):
                assembled_kv_on_cpu.append([])

            # Fetch all chunks for all layers from the backend.
            for start_idx, end_idx, key in keys:
                cached_value = self.cpu_backend.get(key)
                if cached_value:
                    for i in range(self.num_layers):
                        assembled_kv_on_cpu[i].append(cached_value[i])
                else:
                    logger.error(f"Cache key {key.chunk_hash} not found in CPU backend for request {meta.req_id}. Inconsistent state!")
                    return

            # Concatenate all chunks for each layer into a single, contiguous array.
            final_kv_on_cpu = [
                jnp.concatenate(layer_chunks, axis=0)
                for layer_chunks in assembled_kv_on_cpu
            ]

            if not final_kv_on_cpu:
                logger.warning(f"No cache data found in CPU backend for request {meta.req_id}")
                continue
            
            logger.info(f"Request {meta.req_id}: Assembled CPU data for one layer has shape {final_kv_on_cpu[0].shape}.")

            # 3. Pad the data to fill a full number of blocks.
            num_tokens_to_load = len(tokens_to_load)
            num_blocks_to_load = len(meta.local_block_ids)
            padded_token_len = num_blocks_to_load * self.block_size

            padded_kv_on_cpu = []
            for layer_data in final_kv_on_cpu:
                pad_width = padded_token_len - num_tokens_to_load
                if pad_width > 0:
                    padding = jnp.zeros((pad_width, *layer_data.shape[1:]), dtype=layer_data.dtype)
                    padded_kv_on_cpu.append(jnp.concatenate([layer_data, padding], axis=0))
                else:
                    padded_kv_on_cpu.append(layer_data)

            # 4. Reshape data back to block format for the update operation.
            block_shaped_kv_on_cpu = [
                layer_data.reshape(num_blocks_to_load, self.block_size, *layer_data.shape[1:])
                for layer_data in padded_kv_on_cpu
            ]
            
            logger.info(f"Request {meta.req_id}: Reshaped data for transfer to TPU. Shape for one layer: {block_shaped_kv_on_cpu[0].shape}.")

            # 5. Transfer to TPU, applying the correct sharding.
            loaded_kv_sharded_on_tpu = [
                jax.device_put(layer_data, device=self.sharding)
                for layer_data in block_shaped_kv_on_cpu
            ]

            # 6. Update the runner's KV cache with the correctly sharded data.
            destination_blocks = meta.local_block_ids
            for i in range(len(self.runner.kv_caches)):
                self.runner.kv_caches[i] = self.runner.kv_caches[i].at[
                    destination_blocks, ...
                ].set(loaded_kv_sharded_on_tpu[i])

            logger.info(f"Successfully loaded {len(destination_blocks)} blocks into TPU KV cache for request {meta.req_id}")
        
        load_duration = time.time() - load_start_time
        logger.info(f"TPUConnectorWorker: Finished KV cache load process in {load_duration:.4f} seconds.")

    def get_finished(self) -> tuple[set[str], set[str]]:
        """
        Returns the sets of request IDs for completed save and load operations.
        """
        finished_saves = self.finished_save_reqs
        self.finished_save_reqs = set()

        finished_loads = self.finished_load_reqs
        self.finished_load_reqs = set()
        return finished_saves, finished_loads
