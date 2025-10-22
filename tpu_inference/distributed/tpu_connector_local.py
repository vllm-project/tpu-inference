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
import os
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional, get_args

import jax
import jax.numpy as jnp
from jax.sharding import Mesh
from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1, KVConnectorMetadata, KVConnectorRole)
from vllm.v1.core.sched.output import SchedulerOutput

if TYPE_CHECKING:
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.request import Request
    from vllm.forward_context import ForwardContext

from tpu_inference.logger import init_logger
from tpu_inference.runner.tpu_jax_runner import TPUModelRunner

from .cache_util import CPU_OFFLOADING_SWAP_OP_TYPE, TokenProcessor, swap_ops
from .local_cpu_backend import LocalCPUBackend

EngineId = str
ReqId = str

logger = init_logger(__name__)

# kv cache layout needed by cpu offloading mechanism
REQUIRED_KV_CACHE_LAYOUT = "NHD"

# default swap op type
DEFAULT_HOST_HBM_SWAP_OP_TYPE = "pallas"


@dataclass
class SaveSpec:
    """A confirmed work order for the worker to save KV data."""
    skip_leading_tokens: int
    is_final_save: bool = False
    # A direct signal to the worker to skip the data transfer but still
    # process the completion signal if is_final_save is True.
    skip_save: bool = False


@dataclass
class LoadSpec:
    """Internal scheduler state for a potential load operation."""
    num_matched_tokens: int
    can_load: bool = False
    is_full_prefix_hit: bool = False


@dataclass
class TPUReqMeta:
    """A unified work order for a single request in a single step that bundles the Load and Save specs."""
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
            "TPUConnector currently only supports %s KV cache layout.",
            REQUIRED_KV_CACHE_LAYOUT)
        return REQUIRED_KV_CACHE_LAYOUT

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

    def get_finished(self,
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

        # When a request is scheduled again, and the number of new tokens
        # is 1 (excluding chunked prefill), the request is in decode phase.
        if len(new_token_ids) == 1:
            self.is_decode_phase = True


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
        self._external_cache_hits: dict[ReqId, int] = {}
        self.cpu_backend = LocalCPUBackend()
        model_name = self.vllm_config.model_config.model
        logger.info(
            f"Model name is {model_name}, KV block_size={self.block_size}")
        self.token_processor = TokenProcessor(model_name=model_name,
                                              chunk_size=self.block_size)
        self.decode_save = os.getenv("TPU_OFFLOAD_DECODE_SAVE", "0") == "1"
        logger.info(f"decode_save is {self.decode_save}")

    def get_num_new_matched_tokens(
        self,
        request: "Request",
        num_computed_tokens: int,
    ) -> tuple[int, bool]:
        """
        Checks for external KV cache hit against the local CPU backend.
        """
        prompt_token_ids = request.prompt_token_ids
        logger.info(f"Request {request.request_id}: Checking for cache hit. "
                    f"Prompt length: {len(prompt_token_ids)}, "
                    f"Already computed tokens: {num_computed_tokens}.")

        # Generate keys for the incoming request's tokens
        request_keys = self.token_processor.process_tokens(prompt_token_ids)

        num_matched_tokens = 0
        # The generator needs to be consumed to count.
        keys = list(request_keys)
        for start_idx, end_idx, key in keys:
            logger.debug(
                f"  Processing chunk {start_idx}-{end_idx} with hash {key.chunk_hash}"
            )
            if self.cpu_backend.contains(key, pin_on_hit=True):
                num_matched_tokens = end_idx
                logger.debug(
                    f"  -> HIT. Total matched tokens so far: {num_matched_tokens}"
                )
            else:
                # Stop at the first cache miss
                logger.debug("  -> MISS. Stopping search.")
                break

        logger.info(
            f"Request {request.request_id}: Found {num_matched_tokens} matched tokens in CPU backend."
        )

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
            # prompt. The worker-side logic (`start_load_kv`) is aware of this
            # state via `is_full_prefix_hit` flag in the load_spec.
            # It will load the true N tokens from the cache but only copy the first N-1 tokens'
            # worth of KV data to the TPU, fitting perfectly into the allocation.
            # The model then "re-computes" the final prompt token, and from there,
            # the request can seamlessly transition to the decoding phase.
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

        self._external_cache_hits[request.request_id] = num_matched_tokens

        if num_to_load > 0 or is_full_prefix_hit:
            self.load_specs[request.request_id] = LoadSpec(
                num_matched_tokens=num_matched_tokens,
                is_full_prefix_hit=is_full_prefix_hit)
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
            f"{num_external_tokens} external tokens.")
        self._unfinished_requests[request.request_id] = request
        if num_external_tokens > 0:
            if request.request_id in self.load_specs:
                self.load_specs[request.request_id].can_load = True
                logger.info(
                    f"Request {request.request_id}: Marked as ready to load.")

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
        num_total_tokens = len(tracker.token_ids)
        has_new_tokens = num_total_tokens > tracker.save_watermark
        # Determine if a save is needed for this step
        should_save = False
        if is_finished:
            # If the request finished during the decode phase, respect the decode_save flag for saving data.
            # Otherwise (e.g., finished after prefill), always save data.
            if tracker.is_decode_phase and not self.decode_save:
                should_save = False
                logger.info(
                    f"Request {tracker.req_id}: Will skip saving final tokens for decoded request because decode_save is False."
                )
            else:
                should_save = True
        elif has_new_tokens:
            if not tracker.is_decode_phase:
                should_save = True  # Prefill
            elif self.decode_save:
                # Decode: check for block boundary
                next_block_boundary = (tracker.save_watermark //
                                       self.block_size + 1) * self.block_size
                logger.info(
                    f"in decode phase, next_block_boundary: {next_block_boundary}, "
                )
                if num_total_tokens >= next_block_boundary:
                    should_save = True

        logger.info(f"    - Preparing meta for req: {tracker.req_id}, "
                    f"is_finished={is_finished}, "
                    f"total_tokens={num_total_tokens}, "
                    f"saved_tokens={tracker.save_watermark}, "
                    f"has_new={has_new_tokens}, "
                    f"is_decode={tracker.is_decode_phase}, "
                    f"should_save={should_save}")

        # A SaveSpec is always prepared for a finished request to signal completion,
        # even if we don't save the underlying KV data. This is to ensure the TPUConnectorWorker
        # can correctly report finished request.
        save_spec = None
        if should_save:
            # This is a real save operation.
            save_spec = SaveSpec(
                skip_leading_tokens=tracker.save_watermark,
                is_final_save=is_finished,
                skip_save=False,
            )
            if has_new_tokens:
                last_known_watermark = tracker.save_watermark
                tracker.save_watermark = num_total_tokens
                logger.info(
                    f"      -> Old watermark {last_known_watermark}, new save_watermark count: {tracker.save_watermark}"
                )
        elif is_finished:
            # This is a "completion-only" signal because should_save is False.
            save_spec = SaveSpec(
                skip_leading_tokens=tracker.save_watermark,
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
            self, scheduler_output: SchedulerOutput) -> TPUConnectorMetadata:
        metadata = TPUConnectorMetadata()

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
            # Pop tracker and other state first.
            tracker = self._request_trackers.pop(finished_req_id, None)
            self._unfinished_requests.pop(finished_req_id, None)
            self.load_specs.pop(finished_req_id, None)

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

        # Phase 2: Process newly scheduled requests
        # This block handles requests being scheduled for the very first time.
        # It creates the initial RequestTracker and prepares the first work order.
        logger.info(
            f"Phase 2: Processing {len(scheduler_output.scheduled_new_reqs)} new requests."
        )
        for request in scheduler_output.scheduled_new_reqs:
            req_id = request.req_id
            logger.info(f"  - Processing new req: {req_id}")
            num_new_scheduled_tokens = scheduler_output.num_scheduled_tokens[
                req_id]

            # Get the external cache hit count from our new, reliable source.
            num_external_hits = self._external_cache_hits.pop(req_id, 0)

            # Determine the total length of tokens the tracker should hold.
            # This is vLLM's already computed tokens + newly scheduled tokens.
            total_tokens_for_tracker = request.num_computed_tokens + num_new_scheduled_tokens
            tokens_for_tracker = request.prompt_token_ids[:
                                                          total_tokens_for_tracker]
            logger.info(
                f"    - num_new_scheduled_tokens: {num_new_scheduled_tokens}, num_vllm_computed: {request.num_computed_tokens}, num_external_hits: {num_external_hits}"
            )
            logger.info(
                f"    - Slicing prompt[:{total_tokens_for_tracker}] -> len(tokens_for_tracker): {len(tokens_for_tracker)}"
            )

            # Set the initial high-water mark for `save_watermark`.
            # This is the maximum of what vLLM has computed and what's in our external cache.
            initial_save_watermark = max(request.num_computed_tokens,
                                         num_external_hits)

            # Create and store the tracker, which will maintain the request's
            # state for its entire lifetime.
            tracker = RequestTracker(
                req_id=req_id,
                prompt_len=len(request.prompt_token_ids),
                block_ids=request.block_ids[0],
                token_ids=tokens_for_tracker,
                num_external_hits=num_external_hits,
                # The high-water mark for saved tokens starts after the cached prefix.
                save_watermark=initial_save_watermark,
            )
            self._request_trackers[req_id] = tracker
            logger.info(
                f"    - Created tracker for {req_id} with initial state: "
                f"prompt_len={tracker.prompt_len}, "
                f"num_tokens={len(tracker.token_ids)}, "
                f"num_blocks={len(tracker.block_ids)}, "
                f"save_watermark={tracker.save_watermark}")

            # Immediately prepare metadata for this new request. This could include
            # both a load operation (for the cached part) and a save operation
            # (for the newly computed part).
            load_spec = self.load_specs.get(req_id)
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
            logger.info(f"  - Processing cached req: {req_id}")
            tracker = self._request_trackers[req_id]
            full_request = self._unfinished_requests.get(req_id)

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

            # Immediately prepare metadata for this updated request. This will
            # typically be a save operation for the new tokens.
            req_meta = self._prepare_req_meta(tracker,
                                              load_spec=None,
                                              is_finished=False)
            if req_meta:
                logger.info(
                    f"    - Creating metadata for cached req: {req_id} "
                    f"(has_save={req_meta.save_spec is not None})")
                metadata.requests_meta.append(req_meta)

        if metadata.requests_meta:
            logger.info(
                f"Prepared {len(metadata.requests_meta)} requests for worker.")
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
        self.swap_op_type = os.getenv("TPU_KV_OFFLOADING_SWAP_OP_TYPE",
                                      default=DEFAULT_HOST_HBM_SWAP_OP_TYPE)
        assert self.swap_op_type in get_args(CPU_OFFLOADING_SWAP_OP_TYPE)
        # TODO(jcgu): check libtpu compatibility for pallas dma kernel
        # directly import jax._src.test_util crashes the process
        logger.info(
            f"(cpu offloading) swap operation type is {self.swap_op_type}")

        self.host = self.config.kv_ip
        self.kv_transfer_port = self.config.kv_port

        # Get the singleton instance of the CPU backend.
        self.cpu_backend = LocalCPUBackend()
        # The worker needs its own token processor to generate keys.
        model_name = self.vllm_config.model_config.model
        logger.info(
            f"Model name is {model_name}, KV block_size={self.block_size}")
        self.token_processor = TokenProcessor(model_name=model_name,
                                              chunk_size=self.block_size)

        # Thread pool for asynchronous TPU->CPU copies
        self.save_executor = ThreadPoolExecutor(max_workers=4,
                                                thread_name_prefix="tpu_saver")
        self.finished_save_reqs: set[ReqId] = set()
        self.finished_load_reqs: set[ReqId] = set()
        self._tokens_to_unpin: dict[ReqId, list[int]] = {}
        # Tracks if wait_for_save has been called for the current step's metadata.
        self._processed_save_for_step = False

    def __del__(self):
        logger.info("TPUConnectorWorker: Entering __del__")
        self.save_executor.shutdown(wait=True)

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

        num_tokens_to_save = len(
            full_token_ids) - save_spec.skip_leading_tokens
        if num_tokens_to_save <= 0 and not save_spec.is_final_save:
            logger.info(f"Request {req_id}: No new tokens to save.")
            return req_id

        tokens_to_process = full_token_ids[save_spec.skip_leading_tokens:]

        # Calculate the block slice.
        start_block_idx = save_spec.skip_leading_tokens // self.block_size
        blocks_to_process = full_block_ids[start_block_idx:]

        logger.info(f"Request {req_id} save details: "
                    f"full_block_ids len={len(full_block_ids)}, "
                    f"skip_leading_tokens={save_spec.skip_leading_tokens}, "
                    f"num_tokens_to_save={num_tokens_to_save}, "
                    f"start_block_idx={start_block_idx}, "
                    f"blocks_to_process len={len(blocks_to_process)}")

        if not blocks_to_process and tokens_to_process:
            logger.warning(
                f"Request {req_id}: Tokens to save but no corresponding blocks found."
            )
            return req_id

        if not tokens_to_process:
            logger.info(
                f"Request {req_id}: No new tokens to save, but processing as final save."
            )
            return req_id

        try:
            start_time = time.time()

            # Extract blocks on TPU first
            extracted_blocks_tpu = [
                layer_cache_tpu[blocks_to_process, ...]
                for layer_cache_tpu in self.runner.kv_caches
            ]

            # Initiate non-blocking copy to CPU
            kv_caches_on_cpu = [
                swap_ops(extracted_blocks, self.host_sharding, "d2h",
                         self.swap_op_type)
                for extracted_blocks in extracted_blocks_tpu
            ]

            # Block until the transfer is complete
            if kv_caches_on_cpu:
                jax.block_until_ready(kv_caches_on_cpu)

            duration = time.time() - start_time
            logger.info(
                f"Successfully saved {len(blocks_to_process)} blocks for "
                f"request {req_id} to CPU in {duration:.4f} seconds.")

            if kv_caches_on_cpu:
                logger.info(
                    f"Shape of a single layer on CPU before reshape (num_blocks, block_size, ...): {kv_caches_on_cpu[0].shape}"
                )

            post_transfer_start_time = time.time()
            # Reshape per-layer data from (num_blocks, block_size, ...) to
            # a flat (total_tokens, ...) array for easy slicing.
            flat_kv_caches_on_cpu = [
                layer_cache.reshape(-1, *layer_cache.shape[2:])
                for layer_cache in kv_caches_on_cpu
            ]
            # flat_kv_caches_on_cpu = [
            #     jax_einshape("ab...->(ab)...", layer_cache)
            #     for layer_cache in kv_caches_on_cpu
            # ]

            jax.block_until_ready(flat_kv_caches_on_cpu)

            if flat_kv_caches_on_cpu:
                total_size_bytes = sum(layer.nbytes
                                       for layer in flat_kv_caches_on_cpu)
                logger.info(
                    f"Total size of flat_kv_caches_on_cpu: {total_size_bytes / 1024**2:.2f} MB"
                )
                logger.info(
                    f"Shape of a single layer after reshape (total_tokens, ...): {flat_kv_caches_on_cpu[0].shape}"
                )

            # Generate keys for the entire token sequence to get absolute positions. This to ensure that the delta
            # tokens that is about to be captured in the cache are correctly mapped. These keys will be recreated
            # during get_finished() to unpin the correct keys.
            all_keys_generator = self.token_processor.process_tokens(
                full_token_ids)
            all_keys = list(all_keys_generator)

            # Filter for keys that correspond to the new data we are saving.
            relevant_keys = []
            for abs_start_idx, abs_end_idx, key in all_keys:
                if abs_start_idx >= save_spec.skip_leading_tokens:
                    relevant_keys.append((abs_start_idx, abs_end_idx, key))

            if relevant_keys:
                # The flat_kv_caches_on_cpu array corresponds to the new tokens,
                # so its indexing is relative to the start of the new data.
                for abs_start_idx, abs_end_idx, key in relevant_keys:
                    # Calculate indices relative to the start of our new data slice.
                    rel_start_idx = abs_start_idx - save_spec.skip_leading_tokens
                    rel_end_idx = abs_end_idx - save_spec.skip_leading_tokens

                    # Slice the data and add to the backend.
                    value_for_key = [
                        jax.lax.slice_in_dim(flat_layer_cache,
                                             rel_start_idx,
                                             rel_end_idx,
                                             axis=0)
                        for flat_layer_cache in flat_kv_caches_on_cpu
                    ]
                    jax.block_until_ready(value_for_key)
                    self.cpu_backend.add(key, value_for_key)

                logger.info(
                    f"Request {req_id}: Added {len(relevant_keys)} keys to CPU backend."
                )

            post_transfer_duration = time.time() - post_transfer_start_time
            logger.info(
                f"Request {req_id}: e2e host processing of {len(relevant_keys)} keys took {post_transfer_duration:.4f} seconds."
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
        # This method is idempotent. If the save operations for the current
        # step's metadata have already been processed, we can exit early.
        if self._processed_save_for_step:
            return

        # logger.info("TPUConnectorWorker: Entering wait_for_save")
        metadata = self.connector._connector_metadata
        if not isinstance(metadata, TPUConnectorMetadata):
            logger.info(
                "wait_for_save:not an instances of TPUConnectorMetadata")
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
                        self.finished_save_reqs.add(meta.req_id)
                        self._tokens_to_unpin[meta.req_id] = meta.token_ids
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
                if meta.save_spec and meta.save_spec.is_final_save:
                    logger.info(
                        f"Request {finished_req_id}: Final save completed. Marking as finished."
                    )
                    self.finished_save_reqs.add(finished_req_id)
                    self._tokens_to_unpin[finished_req_id] = meta.token_ids

            except Exception as e:
                logger.error(f"A save operation failed: {e}", exc_info=True)

        duration = time.time() - start_time
        logger.info(f"All {len(pending_save_futures)} save operations "
                    f"completed in {duration:.4f} seconds.")
        self._processed_save_for_step = True

    def register_runner(self, runner: TPUModelRunner):
        logger.info("TPUConnectorWorker: Entering register_runner")
        self.runner = runner
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
            # TODO(jcgu): handle SingleDeviceSharding
            self.host_sharding = jax.sharding.NamedSharding(
                mesh=self.device_sharding.mesh,
                spec=self.device_sharding.spec,
                memory_kind="pinned_host")

            logger.info("KV Cache details registered in TPUConnectorWorker:")
            logger.info(f"  - Num layers: {self.num_layers}")
            logger.info(f"  - Shape per layer: {self.shape}")
            logger.info(f"  - DType: {self.dtype}")
            logger.info(f"  - Device sharding: {self.device_sharding}")
            logger.info(f"  - Layout: {self.kv_cache_layout}")
        else:
            logger.warning("TPUConnectorWorker registered with no KV caches.")

    def start_load_kv(self, fwd_ctx: "ForwardContext") -> None:
        """
        This function is the worker-side entry point for loading data from the
        local CPU backend into the TPU's sharded KV cache. It is a blocking
        operation that ensures the cache is fully updated before the model's
        forward pass begins.
        """
        # Reset the save processing flag at the start of a new step.
        self._processed_save_for_step = False
        metadata = self.connector._get_connector_metadata()
        if not isinstance(metadata,
                          TPUConnectorMetadata) or not metadata.requests_meta:
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
            logger.info("TPUConnectorWorker: Starting KV cache load process.")
            # The number of tokens to fetch from the backend is the true number
            # of matched tokens stored in the spec.
            tokens_to_fetch = meta.token_ids[:meta.load_spec.
                                             num_matched_tokens]

            logger.info(
                f"Processing KV load for request {meta.req_id}: "
                f"Fetching {len(tokens_to_fetch)} tokens from cache for "
                f"{len(meta.local_block_ids)} blocks.")

            # 1. Generate keys and fetch data chunks from the CPU backend.
            # We use tokens_to_fetch to generate the correct keys that exist
            # in the backend.
            keys_generator = self.token_processor.process_tokens(
                tokens_to_fetch)
            keys = list(keys_generator)

            if not keys:
                logger.warning(
                    f"Could not generate any keys for loading request {meta.req_id}. Aborting load."
                )
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
                    logger.error(
                        f"Cache key {key.chunk_hash} not found in CPU backend for request {meta.req_id}. Inconsistent state!"
                    )
                    return

            # Concatenate all chunks for each layer into a single, contiguous array.
            final_kv_on_cpu = [
                jnp.concatenate(layer_chunks, axis=0)
                for layer_chunks in assembled_kv_on_cpu
            ]

            if not final_kv_on_cpu:
                logger.warning(
                    f"No cache data found in CPU backend for request {meta.req_id}"
                )
                continue

            jax.block_until_ready(final_kv_on_cpu)
            logger.info(
                f"Request {meta.req_id}: Assembled CPU data for one layer has shape {final_kv_on_cpu[0].shape}."
            )

            # If the full prefix hit state for a given request is active, we fetched N tokens but must
            # now truncate to N-1 before padding and loading, to match the
            # allocation made by the scheduler.
            if meta.load_spec.is_full_prefix_hit:
                final_kv_on_cpu = [
                    layer_data[:-1] for layer_data in final_kv_on_cpu
                ]
                logger.info(
                    f"Request {meta.req_id}: is_full_prefix_hit = {meta.load_spec.is_full_prefix_hit}"
                    "Truncated fetched cache data by 1 token. New shape: "
                    f"{final_kv_on_cpu[0].shape if final_kv_on_cpu else 'N/A'}"
                )

            # 3. Pad the data to fill a full number of blocks.
            num_tokens_to_load = final_kv_on_cpu[0].shape[
                0] if final_kv_on_cpu else 0
            num_blocks_to_load = len(meta.local_block_ids)
            padded_token_len = num_blocks_to_load * self.block_size

            padded_kv_on_cpu = []
            pad_width = padded_token_len - num_tokens_to_load
            if pad_width > 0:
                pad_value = jnp.array(0, dtype=self.dtype)
                padding_config = [
                    (0, pad_width, 0)
                ] + [(0, 0, 0)] * (len(final_kv_on_cpu[0].shape) - 1)
                padded_kv_on_cpu = [
                    jax.lax.pad(layer_cpu, pad_value, padding_config)
                    for layer_cpu in final_kv_on_cpu
                ]
                jax.block_until_ready(padded_kv_on_cpu)
                logger.info(
                    f"padded_kv_on_cpu[0]: {padded_kv_on_cpu[0].shape}, {padded_kv_on_cpu[0].sharding}"
                )
            else:
                padded_kv_on_cpu = final_kv_on_cpu

            # 4. Reshape data back to block format for the update operation.
            block_shaped_kv_on_cpu = [
                layer_data.reshape(num_blocks_to_load, self.block_size,
                                   *layer_data.shape[1:])
                for layer_data in padded_kv_on_cpu
            ]
            # jax_einshape("(nb)...->nb...", layer_cache, b=block_size)

            jax.block_until_ready(block_shaped_kv_on_cpu)
            logger.info(
                f"Request {meta.req_id}: Reshaped data for transfer to TPU. Shape for one layer: {block_shaped_kv_on_cpu[0].shape}."
            )

            # 5. Transfer to TPU, applying the correct sharding.
            loaded_kv_sharded_on_tpu = [
                swap_ops(layer_data, self.device_sharding, "h2d",
                         self.swap_op_type)
                for layer_data in block_shaped_kv_on_cpu
            ]
            jax.block_until_ready(loaded_kv_sharded_on_tpu)
            logger.info(
                f"loaded_kv_on_tpu[0]: {loaded_kv_sharded_on_tpu[0].shape}, {loaded_kv_sharded_on_tpu[0].sharding}"
            )

            # 6. Update the runner's KV cache with the correctly sharded data.
            destination_blocks = meta.local_block_ids
            for i in range(len(self.runner.kv_caches)):
                self.runner.kv_caches[i] = self.runner.kv_caches[i].at[
                    destination_blocks, ...].set(loaded_kv_sharded_on_tpu[i])
            jax.block_until_ready(self.runner.kv_caches)
            logger.info(
                f"Successfully loaded {len(destination_blocks)} blocks into TPU KV cache for request {meta.req_id}"
            )

            request_load_duration = time.time() - request_load_start_time
            load_times.append(request_load_duration)
            logger.info(
                f"Request {meta.req_id}: KV cache load completed in {request_load_duration:.4f} seconds."
            )

        if load_times:
            aggregate_load_time = sum(load_times)
            logger.info(
                f"TPUConnectorWorker: Aggregate KV cache load time for {len(load_times)} requests: {aggregate_load_time:.4f} seconds"
            )

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
        logger.info("TPUConnectorWorker: Entering get_finished")
        self.wait_for_save()

        finished_saves = self.finished_save_reqs
        logger.info(f"Finished saves to report: {finished_saves}")

        # Unpinning logic:
        # A finished request consists of N prompt tokens and M generated tokens.
        # The N prompt tokens were pinned during the initial lookup in
        # `get_num_new_matched_tokens`. The M generated tokens were never
        # pinned, as they were directly added to the cache.
        # Here, we generate keys for the full N+M sequence. The call to
        # `unpin_keys` will correctly unpin the N prompt keys and perform
        # a harmless no-op for the M generated keys, which were never in
        # the pinned set to begin with.
        keys_to_unpin = []
        for req_id in finished_saves:
            if req_id in self._tokens_to_unpin:
                tokens = self._tokens_to_unpin.pop(req_id)
                keys_generator = self.token_processor.process_tokens(tokens)
                unpin_keys = [key for _, _, key in keys_generator]
                keys_to_unpin.extend(unpin_keys)
                logger.info(
                    f"Generated {len(unpin_keys)} keys to unpin for request {req_id}."
                )

        if keys_to_unpin:
            self.cpu_backend.unpin_keys(keys_to_unpin)
            logger.info(f"Unpinned a total of {len(keys_to_unpin)} keys.")

        self.finished_save_reqs = set()

        finished_loads = self.finished_load_reqs
        self.finished_load_reqs = set()
        logger.info(f"Finished saves: {finished_saves}, "
                    f"Finished loads: {finished_loads}")
        return finished_saves, finished_loads
