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

import copy
import multiprocessing.reduction
from collections import defaultdict, deque
from dataclasses import dataclass
from enum import Enum
from multiprocessing import Process, Queue
from time import time
from typing import Any, Dict, List, Optional, Tuple

import cloudpickle
import torch
from vllm.config import VllmConfig
from vllm.multimodal import MULTIMODAL_REGISTRY, MultiModalRegistry
from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.interface import SchedulerInterface
from vllm.v1.core.sched.output import (CachedRequestData, GrammarOutput,
                                       SchedulerOutput)
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.engine import EngineCoreOutputs
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.metrics.stats import PrefixCacheStats, SchedulerStats
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import Request
from vllm.v1.structured_output import StructuredOutputManager

from tpu_inference.logger import init_logger
from tpu_inference.utils import time_function

logger = init_logger(__name__)


class SchedulerCommand(Enum):
    """Enum for scheduler worker process commands."""
    ADD_REQUEST = "add_request"
    SCHEDULE = "schedule"
    FINISH_REQUESTS = "finish_requests"
    UPDATE_DRAFT_TOKEN_IDS = "update_draft_token_ids"
    UPDATE_FROM_OUTPUT = "update_from_output"
    GET_GRAMMAR_BITMASK = "get_grammar_bitmask"
    MAKE_STATS = "make_stats"
    RESET_PREFIX_CACHE = "reset_prefix_cache"
    GET_NUM_UNFINISHED_REQUESTS = "get_num_unfinished_requests"
    HAS_FINISHED_REQUESTS = "has_finished_requests"
    GET_REQUEST_COUNTS = "get_request_counts"
    GET_TOKEN_COUNT = "get_token_count"
    GET_COMPUTED_BLOCKS = "get_computed_blocks"
    SHUTDOWN = "shutdown"


class SchedulerWorkerError(Exception):
    """Exception raised when a scheduler worker process encounters an error."""

    def __init__(self, rank: int, message: str):
        self.rank = rank
        self.message = message
        super().__init__(f"Scheduler worker {rank} error: {message}")


# Monkey-patch multiprocessing to use cloudpickle
# Standard pickle fails to serialize the vLLM Request object.
_original_dumps = multiprocessing.reduction.ForkingPickler.dumps
_original_loads = multiprocessing.reduction.ForkingPickler.loads


def _cloudpickle_dumps(obj, protocol=None):
    """Use cloudpickle for serialization."""
    return cloudpickle.dumps(obj, protocol=protocol)


def _cloudpickle_loads(data):
    """Use cloudpickle for deserialization."""
    return cloudpickle.loads(data)


def _enable_cloudpickle():
    """Enable cloudpickle for multiprocessing queues."""
    multiprocessing.reduction.ForkingPickler.dumps = staticmethod(
        _cloudpickle_dumps)
    multiprocessing.reduction.ForkingPickler.loads = staticmethod(
        _cloudpickle_loads)


def _disable_cloudpickle():
    """Restore original pickle for multiprocessing."""
    multiprocessing.reduction.ForkingPickler.dumps = _original_dumps
    multiprocessing.reduction.ForkingPickler.loads = _original_loads


def _scheduler_worker_process(
    rank: int,
    input_queue: Queue,
    output_queues: Dict[str, Queue],
    vllm_config: Any,
    kv_cache_config: Any,
    structured_output_manager: Any,
    block_size: int,
    mm_registry: Any,
    include_finished_set: bool,
    log_stats: bool,
    original_scheduler_cls: type,
):
    """Worker process that manages a single scheduler instance."""
    # Initialize the scheduler in this process
    scheduler = original_scheduler_cls(
        vllm_config=vllm_config,
        kv_cache_config=kv_cache_config,
        structured_output_manager=structured_output_manager,
        block_size=block_size,
        mm_registry=mm_registry,
        include_finished_set=include_finished_set,
        log_stats=log_stats,
    )

    logger.debug(f"Scheduler worker process {rank} started")

    # Process commands from the input queue
    while True:
        try:
            command, data = input_queue.get()

            match command:
                case SchedulerCommand.ADD_REQUEST:
                    request = data
                    scheduler.add_request(request)
                    output_queues[command.value].put(None)  # Signal completion

                case SchedulerCommand.SCHEDULE:
                    output = scheduler.schedule()
                    output_queues[command.value].put(output)

                case SchedulerCommand.FINISH_REQUESTS:
                    request_ids, finished_status = data
                    scheduler.finish_requests(request_ids, finished_status)
                    output_queues[command.value].put(None)  # Signal completion

                case SchedulerCommand.UPDATE_DRAFT_TOKEN_IDS:
                    draft_token_ids = data
                    scheduler.update_draft_token_ids(draft_token_ids)
                    output_queues[command.value].put(None)  # Signal completion

                case SchedulerCommand.UPDATE_FROM_OUTPUT:
                    scheduler_output, model_runner_output = data
                    result = scheduler.update_from_output(
                        scheduler_output, model_runner_output)
                    output_queues[command.value].put(result)

                case SchedulerCommand.GET_GRAMMAR_BITMASK:
                    scheduler_output = data
                    result = scheduler.get_grammar_bitmask(scheduler_output)
                    output_queues[command.value].put(result)

                case SchedulerCommand.MAKE_STATS:
                    spec_decoding_stats, kv_connector_stats = data
                    result = scheduler.make_stats(spec_decoding_stats,
                                                  kv_connector_stats)
                    output_queues[command.value].put(result)

                case SchedulerCommand.RESET_PREFIX_CACHE:
                    result = scheduler.reset_prefix_cache()
                    output_queues[command.value].put(result)

                case SchedulerCommand.GET_NUM_UNFINISHED_REQUESTS:
                    result = scheduler.get_num_unfinished_requests()
                    output_queues[command.value].put(result)

                case SchedulerCommand.HAS_FINISHED_REQUESTS:
                    result = scheduler.has_finished_requests()
                    output_queues[command.value].put(result)

                case SchedulerCommand.GET_REQUEST_COUNTS:
                    running = len(scheduler.running)
                    waiting = len(scheduler.waiting)
                    output_queues[command.value].put((running, waiting))

                case SchedulerCommand.GET_TOKEN_COUNT:
                    # Calculate total tokens across running and waiting requests
                    total_tokens = 0
                    for req in scheduler.running:
                        total_tokens += len(req.all_token_ids)
                    for req in scheduler.waiting:
                        total_tokens += len(req.all_token_ids)
                    output_queues[command.value].put(total_tokens)

                case SchedulerCommand.GET_COMPUTED_BLOCKS:
                    request = data
                    blocks, cached_tokens = scheduler.kv_cache_manager.get_computed_blocks(
                        request)
                    output_queues[command.value].put((blocks, cached_tokens))

                case SchedulerCommand.SHUTDOWN:
                    scheduler.shutdown()
                    output_queues[command.value].put(None)  # Signal completion
                    break
                case _:
                    error = SchedulerWorkerError(
                        rank, f"Unknown command: {command}")
                    output_queues[command.value].put(error)
                    raise error

        except Exception as e:
            logger.error(f"Error in scheduler worker {rank}: {e}",
                         exc_info=True)
            error = SchedulerWorkerError(rank, str(e))
            output_queues[command.value].put(error)


@dataclass
class DPSchedulerOutput(SchedulerOutput):
    """Extended SchedulerOutput that includes DP rank assignments."""
    assigned_dp_rank: Optional[Dict[str, int]] = None

    def __init__(self, *args, assigned_dp_rank=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.assigned_dp_rank = assigned_dp_rank or {}


class DPScheduler(SchedulerInterface):
    """
    DPScheduler is used when DP size is >=2. Otherwise the default vLLM scheduler is used.

    The DPScheduler manages:
    1. Multiple vLLM Schedulers (one per DP rank)
    2. Request-to-scheduler assignment

    Each Scheduler manages its own logical KV cache shard and scheduling logic.

    **Load Balancing**

    For new requests:
    - If there is prefix cache hit, assigns request to the rank with the best hit
    - Otherwise, assigns request to the rank with the least total tokens

    Once a DP rank is assigned to a request, it remains fixed for the request's lifetime.
    A request will be freed from its assigned rank when it is completed or preempted.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        kv_cache_config: KVCacheConfig,
        structured_output_manager: StructuredOutputManager,
        block_size: int,
        mm_registry: MultiModalRegistry = MULTIMODAL_REGISTRY,
        include_finished_set: bool = False,
        log_stats: bool = False,
    ) -> None:
        self.vllm_config = vllm_config
        self.block_size = block_size
        self.log_stats = log_stats
        self.connector = None
        self.structured_output_manager = structured_output_manager

        # DP state
        self.dp_size = vllm_config.sharding_config.total_dp_size
        self.assigned_dp_rank: Dict[str, int] = {}  # req_id -> dp_rank
        self.cached_schedulers_output = deque()
        self._create_per_rank_configs(kv_cache_config)

        # The original scheduler class could be Scheduler or AsyncScheduler
        original_scheduler_cls = vllm_config.scheduler_config._original_scheduler_cls

        # Enable cloudpickle for multiprocessing to handle local functions
        _enable_cloudpickle()

        # Create worker processes with separate output queues for each command type
        import multiprocessing
        ctx = multiprocessing.get_context('fork')
        self.input_queues: List[Queue] = []
        self.output_queues: Dict[Tuple[int, str], Queue] = {}
        self.processes: List[Process] = []

        for rank in range(self.dp_size):
            input_queue = ctx.Queue()
            self.input_queues.append(input_queue)

            output_queues_for_rank: Dict[str, Queue] = {}
            for cmd in SchedulerCommand:
                output_queues_for_rank[cmd.value] = ctx.Queue()
                self.output_queues[(
                    rank, cmd.value)] = output_queues_for_rank[cmd.value]

            process = ctx.Process(
                target=_scheduler_worker_process,
                args=(
                    rank,
                    input_queue,
                    output_queues_for_rank,
                    self.vllm_config,
                    self.per_rank_kv_cache_configs[rank],
                    structured_output_manager,
                    block_size,
                    mm_registry,
                    include_finished_set,
                    log_stats,
                    original_scheduler_cls,
                ),
            )
            process.start()
            self.processes.append(process)

        logger.info(
            f"DPScheduler (Async = {self.vllm_config.scheduler_config.async_scheduling}) "
            f"started {self.dp_size} worker processes with cloudpickle. "
            f"Per-rank limits: max_seqs={self.vllm_config.scheduler_config.max_num_seqs}, "
            f"max_tokens={self.vllm_config.scheduler_config.max_num_batched_tokens}"
        )

    def _create_per_rank_configs(self, kv_cache_config: KVCacheConfig) -> None:
        self.per_rank_kv_cache_configs: List[KVCacheConfig] = []
        for _ in range(self.dp_size):
            rank_config = copy.deepcopy(kv_cache_config)
            rank_config.num_blocks = kv_cache_config.num_blocks // self.dp_size
            self.per_rank_kv_cache_configs.append(rank_config)

    def _get_result_from_queue(self, rank: int,
                               command: SchedulerCommand) -> Any:
        """Get result from the output queue for a specific rank and command type."""
        queue_obj = self.output_queues[(rank, command.value)]
        try:
            start_time = time()
            result = queue_obj.get()
            end_time = time()
            if end_time - start_time > 1.0:
                logger.warning(
                    f"Long wait time ({end_time - start_time:.2f}s) for rank {rank} "
                    f"command {command.value} response.")
        except EOFError as e:
            raise RuntimeError(
                f"Queue error for rank {rank} command {command.value}: "
                "Worker process terminated unexpectedly. "
                "This may indicate a crash in the scheduler worker process."
            ) from e
        if isinstance(result, SchedulerWorkerError):
            raise result
        return result

    def _get_rank_token_counts(self) -> Dict[int, int]:
        """Calculate total tokens currently assigned to each DP rank."""
        for rank in range(self.dp_size):
            self.input_queues[rank].put(
                (SchedulerCommand.GET_TOKEN_COUNT, None))

        rank_tokens = {}
        for rank in range(self.dp_size):
            token_count = self._get_result_from_queue(
                rank, SchedulerCommand.GET_TOKEN_COUNT)
            rank_tokens[rank] = token_count

        return rank_tokens

    def _find_best_rank_for_request(self, request: Request) -> int:
        """Find the best DP rank for a new request based on load balancing."""
        rank_tokens = self._get_rank_token_counts()

        # First, try to find a rank with prefix cache hit
        for rank in range(self.dp_size):
            self.input_queues[rank].put(
                (SchedulerCommand.GET_COMPUTED_BLOCKS, request))

        best_cache_rank = None
        best_cache_tokens = 0
        for rank in range(self.dp_size):
            blocks, cached_tokens = self._get_result_from_queue(
                rank, SchedulerCommand.GET_COMPUTED_BLOCKS)
            if cached_tokens > best_cache_tokens:
                best_cache_tokens = cached_tokens
                best_cache_rank = rank
        if best_cache_tokens > 0:
            return best_cache_rank

        # Otherwise, find rank with least tokens
        selected_rank = min(rank_tokens, key=rank_tokens.get)
        return selected_rank

    def add_request(self, request: Request) -> None:
        """
        Add a new request to the appropriate DP rank scheduler.

        This is the main entry point for new requests. The scheduler will:
        1. Determine the best DP rank for the request (load balancing + cache hits)
        2. Assign the request to that rank
        3. Add the request to the rank's scheduler
        """
        assert request.request_id not in self.assigned_dp_rank, (
            f"Request {request.request_id} already "
            f"assigned to rank {self.assigned_dp_rank[request.request_id]})")
        rank = self._find_best_rank_for_request(request)
        self.assigned_dp_rank[request.request_id] = rank

        self.input_queues[rank].put((SchedulerCommand.ADD_REQUEST, request))
        self._get_result_from_queue(rank, SchedulerCommand.ADD_REQUEST)

    @time_function
    def schedule(self) -> DPSchedulerOutput:
        """
        Main scheduling method that coordinates all DP rank schedulers.

        Process:
        1. Add any new requests to appropriate DP ranks
        2. Run each scheduler independently in parallel
        3. Combine outputs from all schedulers
        4. Return unified scheduling result
        """
        # Run each scheduler independently
        for rank in range(self.dp_size):
            self.input_queues[rank].put((SchedulerCommand.SCHEDULE, None))

        # Collect outputs from all workers (blocking)
        rank_outputs = []
        for rank in range(self.dp_size):
            output = self._get_result_from_queue(rank,
                                                 SchedulerCommand.SCHEDULE)
            rank_outputs.append(output)

        # Cache scheduler outputs to use in `update_from_output`
        self.cached_schedulers_output.append(rank_outputs)

        # Return combined scheduler outputs
        combined_output = self._combine_scheduler_outputs(rank_outputs)

        logger.debug(
            f"DPScheduler scheduled: "
            f"{combined_output.total_num_scheduled_tokens} total tokens, "
            f"{len(combined_output.scheduled_new_reqs)} new requests, "
            f"{len(combined_output.scheduled_cached_reqs.req_ids)} cached requests"
        )

        return combined_output

    def _combine_scheduler_outputs(
            self, rank_outputs: List[SchedulerOutput]) -> DPSchedulerOutput:
        """Combine outputs from all DP rank schedulers into a unified output."""

        # Combine new requests
        all_new_reqs = []
        for output in rank_outputs:
            all_new_reqs.extend(output.scheduled_new_reqs)

        # Combine cached request data
        combined_cached_data = self._combine_cached_request_data(rank_outputs)

        # Combine token counts and other metrics
        combined_num_scheduled_tokens = {}
        combined_spec_decode_tokens = {}
        combined_encoder_inputs = {}
        total_scheduled_tokens = 0

        for output in rank_outputs:
            combined_num_scheduled_tokens.update(output.num_scheduled_tokens)
            combined_spec_decode_tokens.update(
                output.scheduled_spec_decode_tokens)
            combined_encoder_inputs.update(output.scheduled_encoder_inputs)
            total_scheduled_tokens += output.total_num_scheduled_tokens

        # Combine finished request IDs
        combined_finished_req_ids = set()
        for output in rank_outputs:
            combined_finished_req_ids.update(output.finished_req_ids)

        # Combine other fields (take from first non-empty or use defaults)
        num_common_prefix_blocks = rank_outputs[
            0].num_common_prefix_blocks if rank_outputs else []

        # Create DP rank assignment mapping for scheduled requests
        assigned_dp_rank = {}
        for req_id in combined_num_scheduled_tokens.keys():
            assigned_dp_rank[req_id] = self.assigned_dp_rank[req_id]

        return DPSchedulerOutput(
            scheduled_new_reqs=all_new_reqs,
            scheduled_cached_reqs=combined_cached_data,
            num_scheduled_tokens=combined_num_scheduled_tokens,
            total_num_scheduled_tokens=total_scheduled_tokens,
            scheduled_spec_decode_tokens=combined_spec_decode_tokens,
            scheduled_encoder_inputs=combined_encoder_inputs,
            num_common_prefix_blocks=num_common_prefix_blocks,
            finished_req_ids=combined_finished_req_ids,
            free_encoder_mm_hashes=set(),
            assigned_dp_rank=assigned_dp_rank,
        )

    def _combine_cached_request_data(
            self, rank_outputs: List[SchedulerOutput]) -> CachedRequestData:
        """Combine cached request data from all DP rank schedulers."""
        combined_req_ids = []
        combined_resumed_req_ids = []
        combined_new_token_ids = []
        combined_all_token_ids = []
        combined_new_block_ids = []
        combined_num_computed_tokens = []
        combined_num_output_tokens = []

        for output in rank_outputs:
            cached_data = output.scheduled_cached_reqs

            combined_req_ids.extend(cached_data.req_ids)
            combined_resumed_req_ids.extend(cached_data.resumed_req_ids)
            combined_new_token_ids.extend(cached_data.new_token_ids)
            combined_all_token_ids.extend(cached_data.all_token_ids)
            combined_new_block_ids.extend(cached_data.new_block_ids)
            combined_num_computed_tokens.extend(
                cached_data.num_computed_tokens)
            combined_num_output_tokens.extend(cached_data.num_output_tokens)

        return CachedRequestData(
            req_ids=combined_req_ids,
            resumed_req_ids=combined_resumed_req_ids,
            new_token_ids=combined_new_token_ids,
            all_token_ids=combined_all_token_ids,
            new_block_ids=combined_new_block_ids,
            num_computed_tokens=combined_num_computed_tokens,
            num_output_tokens=combined_num_output_tokens,
        )

    def get_grammar_bitmask(
        self,
        scheduler_output: DPSchedulerOutput,
    ) -> GrammarOutput | None:
        """
        Generate grammar bitmask for structured output requests across all DP ranks.

        This method calls get_grammar_bitmask on each underlying scheduler and
        combines their outputs, similar to how other operations are handled.
        """
        # Use the most recent cached outputs from the schedule() call
        if not self.cached_schedulers_output:
            return None

        rank_scheduler_outputs = self.cached_schedulers_output[
            -1]  # Get the most recent

        combined_structured_output_request_ids = []
        combined_bitmasks = []

        # Get grammar bitmask from each DP rank scheduler
        for rank in range(self.dp_size):
            self.input_queues[rank].put((SchedulerCommand.GET_GRAMMAR_BITMASK,
                                         rank_scheduler_outputs[rank]))
        for rank in range(self.dp_size):
            grammar_output = self._get_result_from_queue(
                rank, SchedulerCommand.GET_GRAMMAR_BITMASK)
            if grammar_output is not None:
                combined_structured_output_request_ids.extend(
                    grammar_output.structured_output_request_ids)
                combined_bitmasks.append(grammar_output.grammar_bitmask)

        if not combined_structured_output_request_ids:
            return None

        # Combine bitmasks - concatenate along the batch dimension
        if len(combined_bitmasks) == 1:
            combined_bitmask = combined_bitmasks[0]
        else:
            combined_bitmask = torch.cat(combined_bitmasks, dim=0)

        return GrammarOutput(combined_structured_output_request_ids,
                             combined_bitmask)

    def update_from_output(
        self, scheduler_output: DPSchedulerOutput,
        model_runner_output: ModelRunnerOutput
    ) -> dict[int, EngineCoreOutputs]:
        """
        Update all DP rank schedulers based on model runner output.

        We need to route the model runner output to the appropriate scheduler
        based on which rank each request belongs to.
        """
        # Group model runner outputs by DP rank
        rank_model_outputs = self._split_model_output_by_rank(
            model_runner_output)
        rank_scheduler_outputs = self.cached_schedulers_output.popleft()
        # Update each scheduler with its portion of the output
        for rank in range(self.dp_size):
            self.input_queues[rank].put(
                (SchedulerCommand.UPDATE_FROM_OUTPUT,
                 (rank_scheduler_outputs[rank], rank_model_outputs[rank])))

        combined_engine_outputs = defaultdict(list)
        for rank in range(self.dp_size):
            rank_engine_outputs = self._get_result_from_queue(
                rank, SchedulerCommand.UPDATE_FROM_OUTPUT)
            for client_idx, engine_output in rank_engine_outputs.items():
                combined_engine_outputs[client_idx].append(engine_output)

        # Clean up finished requests from DP tracking
        self._cleanup_finished_requests(scheduler_output.finished_req_ids)

        # Return combined EngineCoreOutput
        for client_idx, engine_outputs in combined_engine_outputs.items():
            combined_output = EngineCoreOutputs()
            outputs = []
            finished_requests = set()
            for engine_output in engine_outputs:
                outputs.extend(engine_output.outputs)
                if engine_output.finished_requests:
                    finished_requests.update(engine_output.finished_requests)
            combined_output.engine_index = engine_outputs[0].engine_index
            combined_output.outputs = outputs
            combined_output.finished_requests = finished_requests
            combined_output.scheduler_stats = self.make_stats()
            combined_engine_outputs[client_idx] = combined_output

        return combined_engine_outputs

    def _split_model_output_by_rank(
            self,
            global_model_output: ModelRunnerOutput) -> List[ModelRunnerOutput]:
        """Split the model runner output by DP rank for individual scheduler updates."""
        outputs = [
            ModelRunnerOutput(
                req_ids=[],
                req_id_to_index=global_model_output.req_id_to_index,
                sampled_token_ids=global_model_output.sampled_token_ids,
                logprobs=global_model_output.logprobs,
                prompt_logprobs_dict=global_model_output.prompt_logprobs_dict,
                pooler_output=None,
                num_nans_in_logits=global_model_output.num_nans_in_logits,
                kv_connector_output=global_model_output.kv_connector_output,
            ) for _ in range(self.dp_size)
        ]

        for req_id in global_model_output.req_ids:
            rank = self.assigned_dp_rank[req_id]
            outputs[rank].req_ids.append(req_id)

        return outputs

    def _cleanup_finished_requests(self, finished_req_ids: set[str]) -> None:
        """Remove finished requests from our DP rank assignment tracking."""
        for req_id in finished_req_ids:
            if req_id in self.assigned_dp_rank:
                del self.assigned_dp_rank[req_id]

    def finish_requests(self, request_ids, finished_status) -> None:
        """Forward request finish signals to the appropriate DP rank schedulers."""
        if isinstance(request_ids, str):
            request_ids = [request_ids]

        # Route finish signals to appropriate schedulers
        rank_request_ids = defaultdict(list)
        for req_id in request_ids:
            rank = self.assigned_dp_rank[req_id]
            rank_request_ids[rank].append(req_id)

        # Forward to each scheduler
        for rank, req_ids in rank_request_ids.items():
            self.input_queues[rank].put(
                (SchedulerCommand.FINISH_REQUESTS, (req_ids, finished_status)))
            self._get_result_from_queue(rank, SchedulerCommand.FINISH_REQUESTS)

    def get_num_unfinished_requests(self) -> int:
        """Get total number of unfinished requests across all DP ranks."""
        for rank in range(self.dp_size):
            self.input_queues[rank].put(
                (SchedulerCommand.GET_NUM_UNFINISHED_REQUESTS, None))

        total = 0
        for rank in range(self.dp_size):
            count = self._get_result_from_queue(
                rank, SchedulerCommand.GET_NUM_UNFINISHED_REQUESTS)
            total += count
        return total

    def has_finished_requests(self) -> bool:
        """Check if any DP rank has finished requests."""
        for rank in range(self.dp_size):
            self.input_queues[rank].put(
                (SchedulerCommand.HAS_FINISHED_REQUESTS, None))

        has_finished_any = False
        for rank in range(self.dp_size):
            has_finished_any |= self._get_result_from_queue(
                rank, SchedulerCommand.HAS_FINISHED_REQUESTS)
        return has_finished_any

    def get_request_counts(self) -> Tuple[int, int]:
        """Get total (running, waiting) request counts across all DP ranks."""
        for rank in range(self.dp_size):
            self.input_queues[rank].put(
                (SchedulerCommand.GET_REQUEST_COUNTS, None))

        total_running = 0
        total_waiting = 0
        for rank in range(self.dp_size):
            running, waiting = self._get_result_from_queue(
                rank, SchedulerCommand.GET_REQUEST_COUNTS)
            total_running += running
            total_waiting += waiting
        return total_running, total_waiting

    def reset_prefix_cache(self) -> bool:
        """Reset prefix cache for all DP rank schedulers."""
        for rank in range(self.dp_size):
            self.input_queues[rank].put(
                (SchedulerCommand.RESET_PREFIX_CACHE, None))

        all_success = True
        for rank in range(self.dp_size):
            success = self._get_result_from_queue(
                rank, SchedulerCommand.RESET_PREFIX_CACHE)
            all_success &= success
        return all_success

    def make_stats(self,
                   spec_decoding_stats=None,
                   kv_connector_stats=None) -> Optional[SchedulerStats]:
        """Combine stats from all DP rank schedulers."""
        if not self.log_stats:
            return None

        # Aggregate stats from all schedulers
        total_running_reqs = 0
        total_waiting_reqs = 0
        total_kv_cache_usage = 0.0

        combined_prefix_cache_stats = PrefixCacheStats()
        combined_connector_prefix_cache_stats: Optional[
            PrefixCacheStats] = None

        for rank in range(self.dp_size):
            self.input_queues[rank].put(
                (SchedulerCommand.MAKE_STATS, (spec_decoding_stats,
                                               kv_connector_stats)))

        for rank in range(self.dp_size):
            rank_stats = self._get_result_from_queue(
                rank, SchedulerCommand.MAKE_STATS)
            if rank_stats is None:
                continue

            total_running_reqs += rank_stats.num_running_reqs
            total_waiting_reqs += rank_stats.num_waiting_reqs
            total_kv_cache_usage += rank_stats.kv_cache_usage

            # Combine prefix cache stats
            if rank_stats.prefix_cache_stats:
                combined_prefix_cache_stats.reset = rank_stats.prefix_cache_stats.reset
                combined_prefix_cache_stats.requests += rank_stats.prefix_cache_stats.requests
                combined_prefix_cache_stats.queries += rank_stats.prefix_cache_stats.queries
                combined_prefix_cache_stats.hits += rank_stats.prefix_cache_stats.hits

            # Combine connector prefix cache stats
            if rank_stats.connector_prefix_cache_stats:
                if combined_connector_prefix_cache_stats is None:
                    combined_connector_prefix_cache_stats = PrefixCacheStats()
                combined_connector_prefix_cache_stats.reset = rank_stats.connector_prefix_cache_stats.reset
                combined_connector_prefix_cache_stats.requests += rank_stats.connector_prefix_cache_stats.requests
                combined_connector_prefix_cache_stats.queries += rank_stats.connector_prefix_cache_stats.queries
                combined_connector_prefix_cache_stats.hits += rank_stats.connector_prefix_cache_stats.hits

        # Average KV cache usage across ranks
        avg_kv_cache_usage = total_kv_cache_usage / self.dp_size if self.dp_size else 0.0

        return SchedulerStats(
            num_running_reqs=total_running_reqs,
            num_waiting_reqs=total_waiting_reqs,
            kv_cache_usage=avg_kv_cache_usage,
            prefix_cache_stats=combined_prefix_cache_stats,
            connector_prefix_cache_stats=combined_connector_prefix_cache_stats,
            spec_decoding_stats=spec_decoding_stats,
            kv_connector_stats=kv_connector_stats.data
            if kv_connector_stats else None,
        )

    def update_draft_token_ids(self, draft_token_ids) -> None:
        """Forward draft token updates to the appropriate DP rank schedulers."""
        # Group draft tokens by DP rank based on request assignments
        rank_draft_tokens = defaultdict(lambda: {
            "req_ids": [],
            "draft_token_ids": []
        })

        for req_id, tokens in zip(draft_token_ids.req_ids,
                                  draft_token_ids.draft_token_ids):
            if req_id in self.assigned_dp_rank:
                rank = self.assigned_dp_rank[req_id]
                rank_draft_tokens[rank]["req_ids"].append(req_id)
                rank_draft_tokens[rank]["draft_token_ids"].append(tokens)

        for rank, draft_data in rank_draft_tokens.items():
            # Create a draft_token_ids object for this rank (mock structure)
            rank_draft_token_ids = type(draft_token_ids)(
                req_ids=draft_data["req_ids"],
                draft_token_ids=draft_data["draft_token_ids"])
            self.input_queues[rank].put(
                (SchedulerCommand.UPDATE_DRAFT_TOKEN_IDS,
                 rank_draft_token_ids))
            self._get_result_from_queue(
                rank, SchedulerCommand.UPDATE_DRAFT_TOKEN_IDS)

    def shutdown(self) -> None:
        """Shutdown all DP rank scheduler worker processes."""
        # Send shutdown command to all workers
        for rank in range(self.dp_size):
            self.input_queues[rank].put((SchedulerCommand.SHUTDOWN, None))

        # Wait for acknowledgment (blocking)
        for rank in range(self.dp_size):
            self._get_result_from_queue(rank, SchedulerCommand.SHUTDOWN)

        # Terminate and join all processes
        for process in self.processes:
            process.join(timeout=5.0)
            if process.is_alive():
                process.terminate()
                process.join()

        # Restore original pickle
        _disable_cloudpickle()


def update_vllm_config_for_dp_scheduler(vllm_config: Any) -> None:
    """
    Update vLLM configuration to use DPScheduler when DP size > 1.
    """
    dp_size = vllm_config.sharding_config.total_dp_size

    if dp_size > 1:
        if vllm_config.scheduler_config.async_scheduling:
            vllm_config.scheduler_config._original_scheduler_cls = AsyncScheduler
        else:
            vllm_config.scheduler_config._original_scheduler_cls = Scheduler

        vllm_config.scheduler_config.scheduler_cls = DPScheduler
