import copy
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

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

logger = init_logger(__name__)


@dataclass
class DPSchedulerOutput(SchedulerOutput):
    """Extended SchedulerOutput that includes DP rank assignments."""
    assigned_dp_rank: Optional[Dict[str, int]] = None

    def __init__(self, *args, assigned_dp_rank=None, rank_to_req_ids=None,scheduled_tokens_per_dp_rank = None,num_scheduled_tokens_per_dp_rank = None, num_req_per_dp_rank=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.assigned_dp_rank = assigned_dp_rank or {}
        self.rank_to_req_ids = rank_to_req_ids or {}
        self.scheduled_tokens_per_dp_rank = scheduled_tokens_per_dp_rank or {}
        self.num_scheduled_tokens_per_dp_rank = num_scheduled_tokens_per_dp_rank or {}
        self.num_req_per_dp_rank = num_req_per_dp_rank or {}
        


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
        self.schedulers: List[Scheduler] = []
        for rank in range(self.dp_size):
            scheduler = original_scheduler_cls(
                vllm_config=self.vllm_config,
                kv_cache_config=self.per_rank_kv_cache_configs[rank],
                structured_output_manager=structured_output_manager,
                block_size=block_size,
                mm_registry=mm_registry,
                include_finished_set=include_finished_set,
                log_stats=log_stats,
            )
            self.schedulers.append(scheduler)

        logger.info(
            f"DPScheduler (Async = {self.vllm_config.scheduler_config.async_scheduling}) "
            f"per-rank limits: max_seqs={self.vllm_config.scheduler_config.max_num_seqs}, "
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
        rank_tokens = {rank: 0 for rank in range(self.dp_size)}

        for rank, scheduler in enumerate(self.schedulers):
            for request in scheduler.running:
                rank_tokens[rank] += request.num_tokens
            for request in scheduler.waiting:
                rank_tokens[rank] += request.num_tokens

        return rank_tokens

    def _find_best_rank_for_request(self, request: Request) -> int:
        """Find the best DP rank for a new request based on load balancing."""
        rank_tokens = self._get_rank_token_counts()

        # First, try to find a rank with prefix cache hit
        best_cache_rank = None
        best_cache_tokens = 0
        for rank, scheduler in enumerate(self.schedulers):
            blocks, cached_tokens = scheduler.kv_cache_manager.get_computed_blocks(
                request)
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
        self.schedulers[rank].add_request(request)

    def schedule(self) -> DPSchedulerOutput:
        """
        Main scheduling method that coordinates all DP rank schedulers.

        Process:
        1. Add any new requests to appropriate DP ranks
        2. Run each scheduler independently
        3. Combine outputs from all schedulers
        4. Return unified scheduling result
        """
        # Run each scheduler independently
        rank_outputs = []
        for rank, scheduler in enumerate(self.schedulers):
            logger.debug(
                f"Running scheduler for rank {rank}: "
                f"{len(scheduler.running)} running, {len(scheduler.waiting)} waiting"
            )
            output = scheduler.schedule()
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
        rank_to_req_ids = {dp_rank: [] for dp_rank in range(self.dp_size)}
        num_scheduled_tokens_per_dp_rank = {dp_rank: 0 for dp_rank in range(self.dp_size)}
        scheduled_tokens_per_dp_rank = {
            dp_rank: []
            for dp_rank in range(self.dp_size)
        }
        num_req_per_dp_rank= {dp_rank: 0 for dp_rank in range(self.dp_size)}

        for req_id in combined_num_scheduled_tokens.keys():
            assigned_dp_rank[req_id] = self.assigned_dp_rank[req_id]
            rank_to_req_ids[self.assigned_dp_rank[req_id]].append(req_id)
            num_scheduled_tokens_per_dp_rank[self.assigned_dp_rank[req_id]] += combined_num_scheduled_tokens[req_id]
            scheduled_tokens_per_dp_rank[self.assigned_dp_rank[req_id]].append(
                combined_num_scheduled_tokens[req_id])
            num_req_per_dp_rank[self.assigned_dp_rank[req_id]] += 1
            
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
            rank_to_req_ids=rank_to_req_ids,
            num_scheduled_tokens_per_dp_rank=num_scheduled_tokens_per_dp_rank,
            scheduled_tokens_per_dp_rank=scheduled_tokens_per_dp_rank, 
            num_req_per_dp_rank=num_req_per_dp_rank,
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
        for rank, scheduler in enumerate(self.schedulers):
            rank_output = rank_scheduler_outputs[rank]
            grammar_output = scheduler.get_grammar_bitmask(rank_output)

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
        for rank, scheduler in enumerate(self.schedulers):
            rank_engine_outputs = scheduler.update_from_output(
                rank_scheduler_outputs[rank], rank_model_outputs[rank])
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
            self.schedulers[rank].finish_requests(req_ids, finished_status)

    def get_num_unfinished_requests(self) -> int:
        """Get total number of unfinished requests across all DP ranks."""
        return sum(scheduler.get_num_unfinished_requests()
                   for scheduler in self.schedulers)

    def has_finished_requests(self) -> bool:
        """Check if any DP rank has finished requests."""
        return any(scheduler.has_finished_requests()
                   for scheduler in self.schedulers)

    def get_request_counts(self) -> Tuple[int, int]:
        """Get total (running, waiting) request counts across all DP ranks."""
        total_running = sum(
            len(scheduler.running) for scheduler in self.schedulers)
        total_waiting = sum(
            len(scheduler.waiting) for scheduler in self.schedulers)
        return total_running, total_waiting

    def reset_prefix_cache(self) -> bool:
        """Reset prefix cache for all DP rank schedulers."""
        return all(scheduler.reset_prefix_cache()
                   for scheduler in self.schedulers)

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

        for scheduler in self.schedulers:
            rank_stats = scheduler.make_stats(spec_decoding_stats,
                                              kv_connector_stats)
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
        avg_kv_cache_usage = total_kv_cache_usage / len(
            self.schedulers) if self.schedulers else 0.0

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

        # Forward to each scheduler
        for rank, draft_data in rank_draft_tokens.items():
            # Create a draft_token_ids object for this rank (mock structure)
            rank_draft_token_ids = type(draft_token_ids)(
                req_ids=draft_data["req_ids"],
                draft_token_ids=draft_data["draft_token_ids"])
            self.schedulers[rank].update_draft_token_ids(rank_draft_token_ids)

    def shutdown(self) -> None:
        """Shutdown all DP rank schedulers."""
        for scheduler in self.schedulers:
            scheduler.shutdown()


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
