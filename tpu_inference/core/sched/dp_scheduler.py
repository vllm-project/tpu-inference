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
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import jax
import jax.experimental.colocated_python as colocated_python
import torch
from vllm.config import VllmConfig
from vllm.multimodal import MULTIMODAL_REGISTRY, MultiModalRegistry
from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.interface import PauseState, SchedulerInterface
from vllm.v1.core.sched.output import (CachedRequestData, GrammarOutput,
                                       SchedulerOutput)
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.engine import EngineCoreOutputs
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.metrics.stats import PrefixCacheStats, SchedulerStats
from vllm.v1.outputs import DraftTokenIds, ModelRunnerOutput
from vllm.v1.request import Request
from vllm.v1.structured_output import StructuredOutputManager

from tpu_inference.logger import init_logger
from tpu_inference.utils import time_function

logger = init_logger(__name__)


class SchedulerWorker:
    """Wraps a single vLLM Scheduler instance for one DP rank.

    This class is intended to be used with
    ``jax.experimental.colocated_python.colocated_python_class`` so that
    each instance lives on the colocated host CPU(s) rather than the
    single-controller CPU.  All methods are plain synchronous Python;
    the colocated-Python runtime takes care of shipping the calls to the
    correct host.
    """

    def __init__(
        self,
        rank: int,
        vllm_config: Any,
        kv_cache_config: Any,
        structured_output_manager: Any,
        block_size: int,
        mm_registry: Any,
        include_finished_set: bool,
        log_stats: bool,
        original_scheduler_cls: type,
    ):
        self.rank = rank
        self.scheduler = original_scheduler_cls(
            vllm_config=vllm_config,
            kv_cache_config=kv_cache_config,
            structured_output_manager=structured_output_manager,
            block_size=block_size,
            mm_registry=mm_registry,
            include_finished_set=include_finished_set,
            log_stats=log_stats,
        )
        logger.debug(f"SchedulerWorker (colocated) for rank {rank} created")

    # ---- forwarded scheduler methods ----

    def add_request(self, request: Request) -> None:
        self.scheduler.add_request(request)

    def schedule(self) -> SchedulerOutput:
        return self.scheduler.schedule()

    def finish_requests(self, request_ids, finished_status) -> None:
        self.scheduler.finish_requests(request_ids, finished_status)

    def update_draft_token_ids(self, draft_token_ids) -> None:
        self.scheduler.update_draft_token_ids(draft_token_ids)

    def update_from_output(self, scheduler_output, model_runner_output):
        return self.scheduler.update_from_output(scheduler_output,
                                                 model_runner_output)

    def get_grammar_bitmask(self, scheduler_output):
        return self.scheduler.get_grammar_bitmask(scheduler_output)

    def make_stats(self, spec_decoding_stats=None, kv_connector_stats=None):
        return self.scheduler.make_stats(spec_decoding_stats,
                                         kv_connector_stats)

    def reset_prefix_cache(self, reset_running_requests=False,
                           reset_connector=False) -> bool:
        return self.scheduler.reset_prefix_cache(
            reset_running_requests=reset_running_requests,
            reset_connector=reset_connector)

    def reset_encoder_cache(self) -> None:
        self.scheduler.reset_encoder_cache()

    def get_num_unfinished_requests(self) -> int:
        return self.scheduler.get_num_unfinished_requests()

    def has_finished_requests(self) -> bool:
        return self.scheduler.has_finished_requests()

    def get_request_counts(self) -> Tuple[int, int]:
        running = len(self.scheduler.running)
        waiting = len(self.scheduler.waiting)
        return (running, waiting)

    def get_token_count(self) -> int:
        total_tokens = 0
        for req in self.scheduler.running:
            total_tokens += len(req.all_token_ids)
        for req in self.scheduler.waiting:
            total_tokens += len(req.all_token_ids)
        return total_tokens

    def probe_computed_blocks(self, request: Request) -> int:
        kv_cache_mgr = self.scheduler.kv_cache_manager
        if not kv_cache_mgr.enable_caching or request.skip_reading_prefix_cache:
            return 0
        max_cache_hit_length = request.num_tokens - 1
        _, num_cached_tokens = (
            kv_cache_mgr.coordinator.find_longest_cache_hit(
                request.block_hashes, max_cache_hit_length))
        return num_cached_tokens

    def get_pause_state(self) -> PauseState:
        return self.scheduler.pause_state

    def set_pause_state(self, pause_state: PauseState) -> None:
        self.scheduler.set_pause_state(pause_state)

    def shutdown(self) -> None:
        logger.info(f"SchedulerWorker rank {self.rank}: Shutting down")
        self.scheduler.shutdown()


# Wrap SchedulerWorker so instances are created on the colocated host CPUs.
ColocatedSchedulerWorker = colocated_python.colocated_python_class(
    SchedulerWorker)


@dataclass
class DPSchedulerOutput(SchedulerOutput):
    """Extended SchedulerOutput that includes DP rank assignments."""
    assigned_dp_rank: Optional[Dict[str, int]] = None
    # The maximum number of tokens scheduled on any single DP rank in this step.
    # This is used by the Runner to calculate the global padded batch size
    # (padded_max * dp_size), ensuring consistent shapes across pipeline stages.
    max_num_scheduled_tokens_per_dp_rank: int = 0

    def __init__(self,
                 *args,
                 assigned_dp_rank=None,
                 max_num_scheduled_tokens_per_dp_rank=0,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.assigned_dp_rank = assigned_dp_rank or {}
        self.max_num_scheduled_tokens_per_dp_rank = max_num_scheduled_tokens_per_dp_rank


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
        print("WENXIN DEBUGGING")

        # Initialize NONE_HASH global before creating worker instances
        if vllm_config.cache_config.enable_prefix_caching:
            from vllm.utils.hashing import get_hash_fn_by_name
            from vllm.v1.core.kv_cache_utils import init_none_hash
            caching_hash_fn = get_hash_fn_by_name(
                vllm_config.cache_config.prefix_caching_hash_algo)
            init_none_hash(caching_hash_fn)

        # The original scheduler class could be Scheduler or AsyncScheduler
        original_scheduler_cls = vllm_config.scheduler_config._original_scheduler_cls

        # Obtain colocated CPU devices so that we can specialize every
        # worker method.  The SchedulerWorker methods exchange only plain
        # Python objects (no JAX arrays), so the colocated-Python runtime
        # cannot infer target devices from function arguments.  Explicit
        # specialization is therefore mandatory.
        cpu_devices = colocated_python.colocated_cpu_devices(jax.devices())

        # Create colocated SchedulerWorker instances – one per DP rank.
        # Each instance will be materialised on the colocated host CPU(s)
        # the first time one of its methods is called.
        self.workers: List[Any] = []
        for rank in range(self.dp_size):
            worker = ColocatedSchedulerWorker(
                rank=rank,
                vllm_config=self.vllm_config,
                kv_cache_config=self.per_rank_kv_cache_configs[rank],
                structured_output_manager=structured_output_manager,
                block_size=block_size,
                mm_registry=mm_registry,
                include_finished_set=include_finished_set,
                log_stats=log_stats,
                original_scheduler_cls=original_scheduler_cls,
            )
            # Specialize all public methods with devices so that
            # colocated_python knows where to execute them.
            for attr_name in dir(SchedulerWorker):
                if attr_name.startswith('_'):
                    continue
                attr = getattr(worker, attr_name, None)
                if attr is not None and hasattr(attr, 'specialize'):
                    setattr(worker, attr_name,
                            attr.specialize(devices=cpu_devices))
            self.workers.append(worker)

        logger.info(
            f"DPScheduler (Async = {self.vllm_config.scheduler_config.async_scheduling}) "
            f"created {self.dp_size} colocated scheduler workers. "
            f"Per-rank limits: max_seqs={self.vllm_config.scheduler_config.max_num_seqs}, "
            f"max_tokens={self.vllm_config.scheduler_config.max_num_batched_tokens}"
        )

    def _create_per_rank_configs(self, kv_cache_config: KVCacheConfig) -> None:
        self.per_rank_kv_cache_configs: List[KVCacheConfig] = []
        for _ in range(self.dp_size):
            rank_config = copy.deepcopy(kv_cache_config)
            rank_config.num_blocks = kv_cache_config.num_blocks // self.dp_size
            self.per_rank_kv_cache_configs.append(rank_config)

    def _get_rank_token_counts(self) -> Dict[int, int]:
        """Calculate total tokens currently assigned to each DP rank."""
        rank_tokens = {}
        for rank in range(self.dp_size):
            rank_tokens[rank] = self.workers[rank].get_token_count()
        return rank_tokens

    def _find_best_rank_for_request(self, request: Request) -> int:
        """Find the best DP rank for a new request based on load balancing."""
        rank_tokens = self._get_rank_token_counts()

        # First, try to find a rank with prefix cache hit.
        best_cache_rank = None
        best_cache_tokens = 0
        for rank in range(self.dp_size):
            cached_tokens = self.workers[rank].probe_computed_blocks(request)
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

        self.workers[rank].add_request(request)

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
        # Run each scheduler independently and collect outputs
        rank_outputs = []
        for rank in range(self.dp_size):
            output = self.workers[rank].schedule()
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
        # Track the maximum token load on any single rank to determine global padding.
        max_scheduled_tokens_per_rank = 0

        for output in rank_outputs:
            combined_num_scheduled_tokens.update(output.num_scheduled_tokens)
            combined_spec_decode_tokens.update(
                output.scheduled_spec_decode_tokens)
            combined_encoder_inputs.update(output.scheduled_encoder_inputs)
            total_scheduled_tokens += output.total_num_scheduled_tokens
            max_scheduled_tokens_per_rank = max(
                max_scheduled_tokens_per_rank,
                output.total_num_scheduled_tokens)

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
            max_num_scheduled_tokens_per_dp_rank=max_scheduled_tokens_per_rank,
        )

    def _combine_cached_request_data(
            self, rank_outputs: List[SchedulerOutput]) -> CachedRequestData:
        """Combine cached request data from all DP rank schedulers."""
        combined_req_ids = []
        combined_resumed_req_ids = []
        combined_new_token_ids = []
        combined_all_token_ids = {}
        combined_new_block_ids = []
        combined_num_computed_tokens = []
        combined_num_output_tokens = []

        for output in rank_outputs:
            cached_data = output.scheduled_cached_reqs

            combined_req_ids.extend(cached_data.req_ids)
            combined_resumed_req_ids.extend(cached_data.resumed_req_ids)
            combined_new_token_ids.extend(cached_data.new_token_ids)
            combined_all_token_ids.update(cached_data.all_token_ids)
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

    def _combine_scheduler_stats(
        self,
        rank_stats_list: List[Optional[SchedulerStats]],
    ) -> Optional[SchedulerStats]:
        """Combine SchedulerStats from all DP rank schedulers.

        The per-rank stats are extracted from the workers' update_from_output
        results, where the base scheduler's make_stats() already collected
        and reset the prefix cache stats.
        """
        total_running_reqs = 0
        total_waiting_reqs = 0
        total_kv_cache_usage = 0.0

        combined_prefix_cache_stats = PrefixCacheStats()
        combined_connector_prefix_cache_stats: Optional[
            PrefixCacheStats] = None
        has_any_stats = False

        for rank_stats in rank_stats_list:
            if rank_stats is None:
                continue
            has_any_stats = True

            total_running_reqs += rank_stats.num_running_reqs
            total_waiting_reqs += rank_stats.num_waiting_reqs
            total_kv_cache_usage += rank_stats.kv_cache_usage

            # Combine prefix cache stats
            if rank_stats.prefix_cache_stats:
                combined_prefix_cache_stats.reset = (
                    combined_prefix_cache_stats.reset
                    or rank_stats.prefix_cache_stats.reset)
                combined_prefix_cache_stats.requests += (
                    rank_stats.prefix_cache_stats.requests)
                combined_prefix_cache_stats.queries += (
                    rank_stats.prefix_cache_stats.queries)
                combined_prefix_cache_stats.hits += (
                    rank_stats.prefix_cache_stats.hits)

            # Combine connector prefix cache stats
            if rank_stats.connector_prefix_cache_stats:
                if combined_connector_prefix_cache_stats is None:
                    combined_connector_prefix_cache_stats = PrefixCacheStats()
                combined_connector_prefix_cache_stats.reset = (
                    rank_stats.connector_prefix_cache_stats.reset)
                combined_connector_prefix_cache_stats.requests += (
                    rank_stats.connector_prefix_cache_stats.requests)
                combined_connector_prefix_cache_stats.queries += (
                    rank_stats.connector_prefix_cache_stats.queries)
                combined_connector_prefix_cache_stats.hits += (
                    rank_stats.connector_prefix_cache_stats.hits)

        if not has_any_stats:
            return None

        # Average KV cache usage across ranks
        num_ranks = len(rank_stats_list)
        avg_kv_cache_usage = (total_kv_cache_usage /
                              num_ranks if num_ranks else 0.0)

        return SchedulerStats(
            num_running_reqs=total_running_reqs,
            num_waiting_reqs=total_waiting_reqs,
            kv_cache_usage=avg_kv_cache_usage,
            prefix_cache_stats=combined_prefix_cache_stats,
            connector_prefix_cache_stats=combined_connector_prefix_cache_stats,
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
            grammar_output = self.workers[rank].get_grammar_bitmask(
                rank_scheduler_outputs[rank])
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
        combined_engine_outputs = defaultdict(list)
        rank_scheduler_stats: List[Optional[SchedulerStats]] = []
        for rank in range(self.dp_size):
            rank_engine_outputs = self.workers[rank].update_from_output(
                rank_scheduler_outputs[rank], rank_model_outputs[rank])
            rank_stats = None
            for client_idx, engine_output in rank_engine_outputs.items():
                combined_engine_outputs[client_idx].append(engine_output)
                if engine_output.scheduler_stats is not None:
                    rank_stats = engine_output.scheduler_stats
            rank_scheduler_stats.append(rank_stats)

        # Combine scheduler stats from all DP ranks
        combined_stats = self._combine_scheduler_stats(rank_scheduler_stats)

        # Clean up finished requests from DP tracking
        self._cleanup_finished_requests(scheduler_output.finished_req_ids)

        # Return combined EngineCoreOutput
        stats_attached = False
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
            # Attach combined stats to only the first client output
            # (matching the base scheduler behavior)
            if not stats_attached and combined_stats is not None:
                combined_output.scheduler_stats = combined_stats
                stats_attached = True
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
            self.workers[rank].finish_requests(req_ids, finished_status)

    def get_num_unfinished_requests(self) -> int:
        """Get total number of unfinished requests across all DP ranks."""
        total = 0
        for rank in range(self.dp_size):
            total += self.workers[rank].get_num_unfinished_requests()
        return total

    def has_finished_requests(self) -> bool:
        """Check if any DP rank has finished requests."""
        has_finished_any = False
        for rank in range(self.dp_size):
            has_finished_any |= self.workers[rank].has_finished_requests()
        return has_finished_any

    def get_request_counts(self) -> Tuple[int, int]:
        """Get total (running, waiting) request counts across all DP ranks."""
        total_running = 0
        total_waiting = 0
        for rank in range(self.dp_size):
            running, waiting = self.workers[rank].get_request_counts()
            total_running += running
            total_waiting += waiting
        return total_running, total_waiting

    def reset_prefix_cache(
        self,
        reset_running_requests: bool = False,
        reset_connector: bool = False,
    ) -> bool:
        """Reset prefix cache for all DP rank schedulers."""
        all_success = True
        for rank in range(self.dp_size):
            success = self.workers[rank].reset_prefix_cache(
                reset_running_requests, reset_connector)
            all_success &= success
        return all_success

    def reset_encoder_cache(self) -> None:
        """Reset encoder cache for all DP rank schedulers."""
        for rank in range(self.dp_size):
            self.workers[rank].reset_encoder_cache()

    @property
    def pause_state(self) -> PauseState:
        """Get the pause state from the first DP rank scheduler.

        All ranks share the same pause state, so we only need to query one.
        """
        return self.workers[0].get_pause_state()

    def set_pause_state(self, pause_state: PauseState) -> None:
        """Set pause state for all DP rank schedulers."""
        for rank in range(self.dp_size):
            self.workers[rank].set_pause_state(pause_state)

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
            rank_stats = self.workers[rank].make_stats(
                spec_decoding_stats, kv_connector_stats)
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
            # Create a draft_token_ids object for this rank
            rank_draft_token_ids = type(draft_token_ids)(
                req_ids=draft_data["req_ids"],
                draft_token_ids=draft_data["draft_token_ids"])
            self.workers[rank].update_draft_token_ids(rank_draft_token_ids)

    def update_draft_token_ids_in_output(
            self, draft_token_ids: "DraftTokenIds",
            scheduler_output: "SchedulerOutput") -> None:
        """Not implemented for DPScheduler."""
        raise NotImplementedError(
            "update_draft_token_ids_in_output is not implemented for DPScheduler."
        )

    def shutdown(self) -> None:
        """Shutdown all colocated scheduler workers."""
        for rank in range(self.dp_size):
            self.workers[rank].shutdown()
        # Release references so the colocated instances can be garbage-collected
        # (which will trigger __del__ on the remote side).
        self.workers.clear()


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
