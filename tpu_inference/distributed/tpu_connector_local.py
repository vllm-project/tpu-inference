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
import os
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, Optional, get_args

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
from tpu_inference.runner.kv_cache_manager import KVCacheManager
from tpu_inference.runner.tpu_jax_runner import TPUModelRunner

from .cache_util import (CPU_OFFLOADING_SWAP_OP_TYPE, KVCacheSwapFn,
                         TokenProcessor, get_kv_cache_swap_fn,
                         jitted_insert_kv_cache_slices)
from .local_cpu_backend import LocalCPUBackend

EngineId = str
ReqId = str

logger = init_logger(__name__)

# kv cache layout needed by cpu offloading mechanism
REQUIRED_KV_CACHE_LAYOUT = "NHD"

# default swap op type
DEFAULT_HOST_HBM_SWAP_OP_TYPE = "jax"

# we keep our operations at vllm's block granularity,
# and provide the following three preferences when handling
# the last partial block during save:
# 1. drop: drop the entire partial block
# 2. pad: pad to a full block
# 3. dynamic: keep the partial block as is.
PARTIAL_BLOCK_SAVE_BEHAVIOR = Literal["drop", "pad", "dynamic"]


@dataclass
class SaveSpec:
    """A confirmed work order for the worker to save KV data."""
    num_skip_leading_tokens: int
    # total processed tokens for matching / saving
    num_total_tokens: int
    src_blocks: list[int]
    # final save for the (newly) finished request
    is_final_save: bool = False
    # A direct signal to the worker to skip the data transfer but still
    # process the completion signal if is_final_save is True.
    skip_save: bool = False


@dataclass
class LoadSpec:
    """Internal scheduler state for a potential load operation."""
    num_matched_tokens: int
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
                f"dst_blocks={self.load_spec.dst_blocks}")
        save_info = f"save_spec_exists={self.save_spec is not None}"
        if self.save_spec:
            save_info += (
                f", num_skip_leading_tokens={self.save_spec.num_skip_leading_tokens}, "
                f"num_total_tokens={self.save_spec.num_total_tokens}, "
                f"is_final_save={self.save_spec.is_final_save}, "
                f"skip_save={self.save_spec.skip_save}, "
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

        # NOTE(jcgu): is it always true? will MTP affect this judegment?
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


class TPUConnectorScheduler():

    def __init__(self, vllm_config: "VllmConfig"):
        logger.info("TPUConnectorScheduler: Entering __init__")
        self.vllm_config = vllm_config
        self.config = vllm_config.kv_transfer_config
        self.block_size = vllm_config.cache_config.block_size

        self._request_trackers: dict[ReqId, RequestTracker] = {}
        # This dictionary holds the full vLLM Request object for all requests
        # that are currently in a running state (i.e., have been scheduled but
        # are not yet finished). It's used to access the complete prompt token
        # list when processing incremental updates for cached/running requests,
        # as the scheduler output for these requests is minimal.
        self._unfinished_requests: dict[ReqId, "Request"] = {}
        self.load_specs: dict[ReqId, LoadSpec] = {}
        # {reqid: total_num_matched_tokens_in_cpu_backend}
        self._external_cache_hits: dict[ReqId, int] = {}
        self.cpu_backend = LocalCPUBackend()
        model_name = self.vllm_config.model_config.model
        self.token_processor = TokenProcessor(model_name=model_name,
                                              chunk_size=self.block_size)
        self.decode_save = os.getenv("TPU_OFFLOAD_DECODE_SAVE", "0") == "1"
        # NOTE(jcgu): currently, let's nail on chunk_size == block_size
        # chunk_size == n * block_size lead to
        #  1. multi-size chunks
        #  2. complicated resize (split, concatenate) operations based on real-chunk-size in save and load
        self.cpu_chunk_size = self.block_size

        # define partial_block saving behavior
        self.partial_block_save_behavior: PARTIAL_BLOCK_SAVE_BEHAVIOR = \
            os.getenv("TPU_OFFLOAD_PARTIAL_BLOCK_SAVE_BEHAVIOR", "drop")
        assert self.partial_block_save_behavior in get_args(
            PARTIAL_BLOCK_SAVE_BEHAVIOR
        ), f"{self.partial_block_save_behavior} not in {get_args(PARTIAL_BLOCK_SAVE_BEHAVIOR)}"
        self.partial_block_dynamic_pad_lower_limit = \
            int(os.getenv("TPU_OFFLOAD_PARTIAL_BLOCK_DYNAMIC_PAD_LOWER_LIMIT", "0"))
        if self.partial_block_save_behavior == "dynamic":
            if self.partial_block_dynamic_pad_lower_limit <= 0:
                self.partial_block_save_behavior == "drop"
            elif self.partial_block_dynamic_pad_lower_limit >= self.block_size:
                self.partial_block_save_behavior == "pad"

        logger.info(
            f"TPUConnectorScheduler initialized with: "
            f"block_size={self.block_size}, "
            f"cpu_chunk_size={self.cpu_chunk_size}, "
            f"model_name={model_name}, "
            f"decode_save={self.decode_save}, "
            f"partial_block_save_behavior={self.partial_block_save_behavior}, "
            f"partial_block_dynamic_pad_lower_limit={self.partial_block_dynamic_pad_lower_limit}"
        )

    def get_num_new_matched_tokens(
        self,
        request: "Request",
        num_computed_tokens: int,
    ) -> tuple[int, bool]:
        """
        Checks for external KV cache hit against the local CPU backend.
        """
        assert num_computed_tokens % self.block_size == 0, f"{num_computed_tokens} % {self.block_size} != 0"
        prompt_token_ids = request.prompt_token_ids
        logger.info(f"Request {request.request_id}: Checking for cache hit. "
                    f"Prompt length: {len(prompt_token_ids)}, "
                    f"Already computed tokens: {num_computed_tokens}. ")

        # Generate keys for the incoming request's tokens
        request_keys = self.token_processor.process_tokens(prompt_token_ids)

        num_matched_tokens = 0
        # The generator needs to be consumed to count.
        keys = list(request_keys)
        for start_token_idx, end_token_idx, key in keys:
            logger.info(
                f"  Processing chunk {start_token_idx}-{end_token_idx} with hash {key.chunk_hash}"
            )
            if self.cpu_backend.contains(key, pin_on_hit=True):
                num_matched_tokens = end_token_idx
                logger.info(
                    f"  -> HIT. Total matched tokens so far: {num_matched_tokens}"
                )
            else:
                # Stop at the first cache miss
                logger.info("  -> MISS. Stopping search.")
                break

        logger.info(
            f"Request {request.request_id}: Found {num_matched_tokens} (out of {len(prompt_token_ids)} prompt tokens) matched tokens in CPU backend."
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

        self._external_cache_hits[request.request_id] = num_matched_tokens

        if num_matched_tokens > num_computed_tokens:
            # NOTE(jcgu): fill real dst_blocks later when blocks get allocated.
            self.load_specs[request.request_id] = LoadSpec(
                num_matched_tokens=num_matched_tokens,
                dst_blocks=[],
                num_skip_leading_tokens=num_computed_tokens,
            )
        return num_to_load, False

    def _adjust_last_partial_block(self,
                                   last_partial_block_num_tokens: int) -> bool:
        """
        adjust prompt token / len based on pre-configed save behavior
        when the last block of request's token is partially used.
        In order to keep all the saved kv be aligned with block_size,
        we may
         1. drop the partial block
         2. pad the partial block to be a full block
         3. drop or pad based on actual num_tokens in the last partial block

        Input: num of tokens in the last partial block (could be 0)
        Output: the last partial block should be kept (True) or dropped (False)
        """
        if self.partial_block_save_behavior == "pad":
            return True if last_partial_block_num_tokens > 0 else False
        elif self.partial_block_save_behavior == "drop":
            return False
        elif self.partial_block_save_behavior == "dynamic":
            return True if last_partial_block_num_tokens >= self.partial_block_dynamic_pad_lower_limit else False

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
                load_spec = self.load_specs[request.request_id]
                # Update loading block info.
                all_blocks = blocks.get_block_ids()[0]
                skip_leading_blocks = load_spec.num_skip_leading_tokens // self.block_size
                total_matched_blocks = load_spec.num_matched_tokens // self.block_size
                # adjust last partial block
                last_partial_block_num_tokens = load_spec.num_matched_tokens - total_matched_blocks * self.block_size
                need_last_block = self._adjust_last_partial_block(
                    last_partial_block_num_tokens)
                if need_last_block:
                    total_matched_blocks += 1
                assert total_matched_blocks <= len(
                    all_blocks), f"{total_matched_blocks} > {len(all_blocks)}"
                dst_blocks = all_blocks[
                    skip_leading_blocks:total_matched_blocks]
                load_spec.dst_blocks = dst_blocks
                load_spec.can_load = True
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
        num_total_tokens = len(tracker.token_ids)
        num_full_blocks = num_total_tokens // self.block_size
        num_full_blocks_tokens = num_full_blocks * self.block_size
        # adjust last partial block
        last_partial_block_num_tokens = num_total_tokens - num_full_blocks_tokens
        need_last_block = self._adjust_last_partial_block(
            last_partial_block_num_tokens)
        adjusted_num_total_tokens = num_total_tokens if need_last_block else num_full_blocks_tokens
        adjusted_num_total_blocks = num_full_blocks + (1 if need_last_block
                                                       else 0)
        assert adjusted_num_total_blocks <= len(tracker.block_ids)

        has_new_tokens = adjusted_num_total_tokens > tracker.save_watermark
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
                # NOTE(jcgu): for decode, we do not drop or pad, just accumulate tokens until the next block boundary
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
        # even if we don't save the underlying KV data. This is to ensure the TPUConnectorWorker
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
            src_block_ids = tracker.block_ids[
                num_skip_leading_blocks:adjusted_num_total_blocks]
            # This is a real save operation.
            save_spec = SaveSpec(
                num_skip_leading_tokens=num_skip_leading_tokens,
                num_total_tokens=adjusted_num_total_tokens,
                is_final_save=is_finished,
                skip_save=False,
                src_blocks=src_block_ids,
            )
            if has_new_tokens:
                last_known_watermark = tracker.save_watermark
                tracker.save_watermark = adjusted_num_total_tokens
                logger.info(
                    f"      -> Old watermark {last_known_watermark}, new save_watermark count: {tracker.save_watermark}"
                )
        elif is_finished:
            # This is a "completion-only" signal because should_save is False.
            # NOTE(jcgu): num_total_tokens will be used to unpin tokens;
            #  apply the number of saved tokens
            save_spec = SaveSpec(
                num_skip_leading_tokens=tracker.save_watermark,
                num_total_tokens=tracker.save_watermark,
                src_blocks=[],
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

            logger.info(f"---{new_token_ids}, {new_blocks}")
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
        self.swap_op_type = os.getenv("TPU_OFFLOAD_SWAP_OP_TYPE",
                                      default=DEFAULT_HOST_HBM_SWAP_OP_TYPE)
        assert self.swap_op_type in get_args(CPU_OFFLOADING_SWAP_OP_TYPE)
        # TODO(jcgu): check libtpu compatibility for pallas dma kernel
        logger.info(
            f"(cpu offloading) swap operation type is {self.swap_op_type}")

        self.swap_in_fn: KVCacheSwapFn = None
        self.swap_out_fn: KVCacheSwapFn = None

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

        self.cpu_chunk_size = self.block_size
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

            logger.info("KV Cache details registered in TPUConnectorWorker:")
            logger.info(f"  - Num layers: {self.num_layers}")
            logger.info(f"  - Shape per layer: {self.shape}")
            logger.info(f"  - DType: {self.dtype}")
            logger.info(f"  - Device sharding: {self.device_sharding}")
            logger.info(
                f"  - Flatten Device sharding: {self.flatten_device_sharding}")
            logger.info(f"  - Layout: {self.kv_cache_layout}")
        else:
            raise ValueError(
                "TPUConnectorWorker registered with no KV caches.")

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

        logger.info(f"Request {req_id} save details: "
                    f"full_block_ids len={len(full_block_ids)}, "
                    f"num_skip_leading_tokens={num_skip_leading_tokens}, "
                    f"num_total_tokens={num_total_tokens}, "
                    f"num_tokens_to_save={num_tokens_to_save}, "
                    f"blocks_to_save len={len(blocks_to_save)}")

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
            # gather and reshape blocks on TPU first: output_shape: [blocks_to_save * block_size, num_heads, 2, head_dim]
            flat_kv_caches_tpu = KVCacheManager._jitted_gather_kv_cache(
                self.runner.kv_caches, blocks_to_save)

            jax.block_until_ready(flat_kv_caches_tpu)
            logger.info(
                f"extracted_blocks_tpu: {flat_kv_caches_tpu[0].shape}, {flat_kv_caches_tpu[0].sharding}"
            )

            flat_kv_caches_cpu = self.swap_out_fn(flat_kv_caches_tpu)
            # Block until the transfer is complete
            if flat_kv_caches_cpu:
                jax.block_until_ready(flat_kv_caches_cpu)

            duration = time.time() - start_time
            logger.info(f"Successfully saved {len(blocks_to_save)} blocks for "
                        f"request {req_id} to CPU in {duration:.4f} seconds.")

            if flat_kv_caches_cpu:
                logger.info(
                    f"Shape of a single layer on CPU before reshape (num_blocks, block_size, ...): {flat_kv_caches_cpu[0].shape}"
                )

                total_size_bytes = sum(layer.nbytes
                                       for layer in flat_kv_caches_cpu)
                logger.info(
                    f"Total size of flat_kv_caches_cpu: {total_size_bytes / 1024**2:.2f} MB"
                )
                logger.info(
                    f"Shape of a single layer after reshape (total_tokens, ...): {flat_kv_caches_cpu[0].shape}"
                )

            post_transfer_start_time = time.time()

            # Generate keys for the entire token sequence to get absolute positions. This to ensure that the delta
            # tokens that is about to be captured in the cache are correctly mapped. These keys will be recreated
            # during get_finished() to unpin the correct keys.
            all_keys_generator = self.token_processor.process_tokens(
                process_token_ids)
            all_keys = list(all_keys_generator)

            # NOTE(jcgu): we keep cpu_chunk_size == block_size
            split_size_list = [self.cpu_chunk_size] * num_blocks_to_save
            chunks_on_cpu = [
                jax.lax.split(flat_layer_cache, split_size_list, axis=0)
                for flat_layer_cache in flat_kv_caches_cpu
            ]
            jax.block_until_ready(chunks_on_cpu)

            # Filter for keys that correspond to the new data we are saving.
            relevant_keys = []
            for abs_start_token_idx, abs_end_token_idx, key in all_keys:
                if abs_start_token_idx >= num_skip_leading_tokens:
                    relevant_keys.append(
                        (abs_start_token_idx, abs_end_token_idx, key))

            if relevant_keys:
                assert len(
                    relevant_keys
                ) == num_blocks_to_save, f"{len(relevant_keys)} != {num_blocks_to_save}"
                for i in range(num_blocks_to_save):
                    abs_start_token_idx, abs_end_token_idx, key = relevant_keys[
                        i]
                    cur_chunk_cross_layers = [
                        chunks_on_cpu[j][i] for j in range(self.num_layers)
                    ]
                    self.cpu_backend.add(key, cur_chunk_cross_layers)
                    logger.info(
                        f"Request {req_id}: Saving to CPU chunk: "
                        f"abs_start_token_idx={abs_start_token_idx}, abs_end_token_idx={abs_end_token_idx}, "
                        f"chunk_hash={key.chunk_hash}, "
                        f" local_chunk_idx={i}, chunk_size={split_size_list[i]}"
                    )

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
        metadata = self.connector._get_connector_metadata()
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
                        self._tokens_to_unpin[
                            meta.req_id] = meta.token_ids[:meta.save_spec.
                                                          num_total_tokens]
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
                    self._tokens_to_unpin[
                        finished_req_id] = meta.token_ids[:meta.save_spec.
                                                          num_total_tokens]

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
            dst_blocks = meta.load_spec.dst_blocks
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

            # 1. Generate keys for the entire matched prefix to find the right
            # chunks in the backend.
            keys_generator = self.token_processor.process_tokens(
                meta.token_ids[:num_matched_tokens])

            # 2. Assemble the per-layer data for the delta tokens on the CPU.
            # We create a list of lists, where the outer list represents layers
            # and the inner lists will hold the data chunks for that layer.
            assembled_kv_on_cpu = [[] for _ in range(self.num_layers)]
            # Fetch and slice chunks from the backend.
            for start_token_idx, end_token_idx, key in keys_generator:
                # This chunk is entirely before the delta, so we can skip it.
                if end_token_idx <= num_skip_leading_tokens:
                    continue
                # This chunk is entirely after the delta.
                if start_token_idx >= num_matched_tokens:
                    continue

                cached_value = self.cpu_backend.get(key)
                if cached_value:
                    # Calculate the precise slice needed from this specific chunk.
                    # rel_start_token_idx is the index within this chunk where the delta tokens begin.
                    if start_token_idx < num_skip_leading_tokens:
                        assert False, f"start_token_idx {start_token_idx} should not be less than num_skip_leading_tokens {num_skip_leading_tokens}, when cpu_chunk_size == block_size"
                        rel_start_token_idx = num_skip_leading_tokens - start_token_idx
                        rel_end_token_idx = end_token_idx - num_skip_leading_tokens
                        for i in range(self.num_layers):
                            # NOTE(jcgu): if only one block to load (and it's a padded block),
                            # then rel_end_token_idx will not be inaccurate (< block_size).
                            # Slice the jax array fetched from the backend.
                            sliced_chunk = jax.lax.slice_in_dim(
                                cached_value[i],
                                rel_start_token_idx,
                                rel_end_token_idx,
                                axis=0)
                            assembled_kv_on_cpu[i].append(sliced_chunk)
                    else:
                        for i in range(self.num_layers):
                            assembled_kv_on_cpu[i].append(cached_value[i])
                else:
                    logger.error(
                        f"Cache key {key.chunk_hash} not found in CPU backend for request {meta.req_id}. Inconsistent state detected."
                    )
                    return

            # swap-in
            # output: [[cpu_chunk_size * num_chunks] * num_layer]
            raw_chunked_kv_on_tpu = self.swap_in_fn(assembled_kv_on_cpu)
            jax.block_until_ready(raw_chunked_kv_on_tpu)

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
            unpinned_count, found_count = self.cpu_backend.unpin_keys(
                keys_to_unpin)
            logger.info(
                f"Unpinned {unpinned_count} out of {found_count} existing keys (Request to unpin {len(keys_to_unpin)} keys)."
            )

        self.finished_save_reqs = set()

        finished_loads = self.finished_load_reqs
        self.finished_load_reqs = set()
        logger.info(f"Finished saves: {finished_saves}, "
                    f"Finished loads: {finished_loads}")
        return finished_saves, finished_loads
