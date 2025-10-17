import copy
import itertools
from dataclasses import dataclass
from typing import Optional

from vllm.v1.core.kv_cache_manager import KVCacheManager, KVCacheBlocks
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.request import Request
from vllm.v1.metrics.stats import PrefixCacheStats, SchedulerStats
from vllm.v1.spec_decode.metrics import SpecDecodingStats
from vllm.distributed.kv_transfer.kv_connector.v1.metrics import KVConnectorStats
from tpu_inference.logger import init_logger
logger = init_logger(__name__)


@dataclass
class DPSchedulerOutput(SchedulerOutput):
    assigned_dp_rank: dict[str, int] = None
    
    def __init__(self, *args, assigned_dp_rank=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.assigned_dp_rank = assigned_dp_rank or {}

    
class DPScheduler(Scheduler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.dp_size = self.vllm_config.sharding_config.total_dp_size
        
        self._init_dp_state()
        self._replace_kv_cache_manager()
    
        logger.info(f"Scheduler initialized with DP size: {self.dp_size}")
        logger.info(f"Max requests per DP rank: {self.max_reqs_per_dp_rank}")
        logger.info(f"Max tokens per DP rank: {self.max_tokens_per_dp_rank}")
    
    def _log_dp_state(self):
        """Log current DP state for debugging."""
        logger.debug(f"DP rank token budget (cap {self.max_tokens_per_dp_rank}): {self.token_budget}")
        logger.debug(f"DP rank request budget (cap {self.max_reqs_per_dp_rank}): {self.request_budget}")
        logger.debug(f"Total assigned requests: {len(self.assigned_dp_rank)}")
    
    def _init_dp_state(self):
        """Initialize DP-specific state."""
        self.assigned_dp_rank: dict[str, int] = {} # req_id -> dp_rank
        self.num_scheduled_tokens: dict[str, int] = {} #req_id -> num_tokens
        
        self.round_robin_counter = itertools.cycle(range(self.dp_size))
        
        self.empty_kv_cache_blocks = KVCacheBlocks(
            tuple(() for _ in range(self.kv_cache_manager.num_kv_cache_groups))
        )

        # Budgets for each DP rank
        self.max_reqs_per_dp_rank = self.scheduler_config.max_num_seqs // self.dp_size
        self.max_tokens_per_dp_rank = self.scheduler_config.max_num_batched_tokens // self.dp_size

        self.request_budget: dict[int, int] = {i: self.max_reqs_per_dp_rank for i in range(self.dp_size)}
        self.token_budget: dict[int, int] = {i: self.max_tokens_per_dp_rank for i in range(self.dp_size)}

    def _replace_kv_cache_manager(self):
        """Replace single KV cache manager with multiple DP managers."""
        dp_kv_cache_config = copy.deepcopy(self.kv_cache_config)
        dp_kv_cache_config.num_blocks = self.kv_cache_config.num_blocks // self.dp_size
        
        self._original_kv_cache_manager = self.kv_cache_manager
        
        self.dp_kv_cache_managers = [
            KVCacheManager(
                kv_cache_config=dp_kv_cache_config,
                max_model_len=self.max_model_len,
                enable_caching=self.cache_config.enable_prefix_caching,
                use_eagle=self.use_eagle,
                log_stats=self.log_stats,
                enable_kv_cache_events=self.enable_kv_cache_events,
                dcp_world_size=self.dcp_world_size,
            ) for _ in range(self.dp_size)
        ]
        
        self._patch_kv_operations()
    
    def _patch_kv_operations(self):
        class KVCacheManagerProxy:
            def __init__(self, scheduler):
                self.scheduler: DPScheduler = scheduler
            
            def allocate_slots(self, request, num_new_tokens, *args, **kwargs):
                dp_rank = self.scheduler._get_dp_rank_for_allocation(request, num_new_tokens)
                req_id = request.request_id
                if dp_rank is None: 
                    logger.debug(f"Request {req_id} cannot be scheduled: no DP rank with capacity for {num_new_tokens} tokens")
                    if req_id in self.scheduler.assigned_dp_rank:
                        # Clean up any previous assignment if it exists. 
                        # This can happen if the request was previously assigned a DP rank during prefix cache.
                        dp_rank = self.scheduler.assigned_dp_rank[req_id]
                        self.scheduler.request_budget[dp_rank] += 1
                        self.scheduler.assigned_dp_rank.pop(req_id)
                    return None
                self.scheduler._update_dp_rank_budget(request, dp_rank, num_new_tokens)
                logger.debug(f"Request {req_id} allocated to DP rank {dp_rank} with {num_new_tokens} tokens")
                return self.scheduler.dp_kv_cache_managers[dp_rank].allocate_slots(request, num_new_tokens, *args, **kwargs)
            
            def get_computed_blocks(self, request):
                return self.scheduler._get_computed_blocks_dp(request)
            
            def get_blocks(self, request_id):
                dp_rank = self.scheduler.assigned_dp_rank[request_id]
                return self.scheduler.dp_kv_cache_managers[dp_rank].get_blocks(request_id)
            
            def get_block_ids(self, request_id):
                dp_rank = self.scheduler.assigned_dp_rank[request_id]  
                return self.scheduler.dp_kv_cache_managers[dp_rank].get_block_ids(request_id)
            
            def free(self, request):
                # Remove preempted requests
                req_id = request.request_id
                dp_rank = self.scheduler.assigned_dp_rank.get(req_id)
                if dp_rank is not None:
                    self.scheduler.dp_kv_cache_managers[dp_rank].free(request)
                    self.scheduler.request_budget[dp_rank] += 1
                    self.scheduler.token_budget[dp_rank] += self.scheduler.num_scheduled_tokens[req_id]
                    # Clean up state
                    self.scheduler.assigned_dp_rank.pop(req_id)
                    self.scheduler.num_scheduled_tokens.pop(req_id)
            
            def get_num_common_prefix_blocks(self, request_id):
                dp_rank = self.scheduler.assigned_dp_rank[request_id]
                return self.scheduler.dp_kv_cache_managers[dp_rank].get_num_common_prefix_blocks(request_id)
            
            def take_events(self):
                events = []
                for manager in self.scheduler.dp_kv_cache_managers:
                    events.extend(manager.take_events())
                return events
            
            def reset_prefix_cache(self):
                for manager in self.scheduler.dp_kv_cache_managers:
                    if not manager.reset_prefix_cache():
                        return False
                return True
            
            def make_prefix_cache_stats(self):
                combined_stats = PrefixCacheStats()
                for manager in self.scheduler.dp_kv_cache_managers:
                    stats = manager.make_prefix_cache_stats()
                    if stats:
                        combined_stats.reset = stats.reset
                        combined_stats.requests += stats.requests
                        combined_stats.queries += stats.queries
                        combined_stats.hits += stats.hits
                return combined_stats
            
            @property
            def usage(self):
                return sum(manager.usage for manager in self.scheduler.dp_kv_cache_managers) / len(self.scheduler.dp_kv_cache_managers)
        
        # Replace the KV cache manager with our proxy
        self.kv_cache_manager = KVCacheManagerProxy(self)
    
    def _get_dp_rank_for_allocation(self, request: Request, num_tokens: int = 0) -> Optional[int]:
        """Find an available DP rank for the incoming request. Each DP rank can only be assigned max_num_reqs // dp_size requests
        and max_num_batched_tokens // dp_size tokens. 
        
        Returns None if the request cannot be scheduled. 
        """
        assert num_tokens > 0
        req_id = request.request_id
        # Existing requests
        if req_id in self.assigned_dp_rank:
            dp_rank = self.assigned_dp_rank[req_id]
            if self._rank_has_capacity_for_existing_requests(dp_rank, num_tokens):
                return dp_rank
            return None
            
        # New requests
        for _ in range(self.dp_size):
            candidate_rank = next(self.round_robin_counter)
            if self._rank_has_capacity_for_new_requests(candidate_rank, num_tokens):
                return candidate_rank
        return None
    
    def _update_dp_rank_budget(self, request: Request, dp_rank: int, num_tokens: int) -> Optional[int]:
        req_id = request.request_id
        if req_id not in self.assigned_dp_rank:
            self.assigned_dp_rank[req_id] = dp_rank
        else:
            assert dp_rank == self.assigned_dp_rank[req_id], "DP rank mismatch in state update"
        
        self.request_budget[dp_rank] -= 1
        self.token_budget[dp_rank] -= num_tokens
        self.num_scheduled_tokens[req_id] = num_tokens

    def _rank_has_capacity_for_new_requests(self, dp_rank: int, num_tokens: int) -> bool:
        if (self.request_budget[dp_rank] > 0 and
            self.token_budget[dp_rank] >= num_tokens):
            return True
        return False
    
    def _rank_has_capacity_for_existing_requests(self, dp_rank: int, num_tokens: int) -> bool:
        if (self.token_budget[dp_rank]  >= num_tokens):
            return True
        return False
    
    def _get_computed_blocks_dp(self, request: Request):
        """Get computed blocks with DP logic - check all ranks for cache hits."""
        assert request.request_id not in self.assigned_dp_rank, "Request already has assigned DP rank"
        best_blocks = None
        best_tokens = 0
        best_rank = None
        
        # Check all DP managers for the best cache hit
        for rank in range(self.dp_size):
            # Only consider this rank if it has request capacity
            # We cannot check for token capacity until we find out the number of new tokens needed
            if self.request_budget[rank] > 0:
                blocks, tokens = self.dp_kv_cache_managers[rank].get_computed_blocks(request)
                if tokens > best_tokens:
                    best_blocks = blocks
                    best_tokens = tokens
                    best_rank = rank
        
        if best_rank is not None and best_tokens > 0:
            # Only assign rank  
            # Token accounting will happen in allocate_slots
            self.assigned_dp_rank[request.request_id] = best_rank
            self.request_budget[best_rank] -= 1

            logger.debug(f"{request.request_id}. prefix cache hit: {best_rank}")
            return best_blocks, best_tokens
        else:
            return self.empty_kv_cache_blocks, 0
    
    def schedule(self) -> SchedulerOutput:
        """Override schedule to return DP-specific output."""
        self.request_budget: dict[int, int] = {i: self.max_reqs_per_dp_rank for i in range(self.dp_size)}
        self.token_budget: dict[int, int] = {i: self.max_tokens_per_dp_rank for i in range(self.dp_size)}
        self.num_scheduled_tokens = {}
        

        base_output = super().schedule()
        
        # Extract scheduled request IDs and create assigned_dp_rank mapping
        scheduled_requests = []
        scheduled_requests.extend([req_data.req_id for req_data in base_output.scheduled_new_reqs])
        scheduled_requests.extend(base_output.scheduled_cached_reqs.req_ids)
        
        assigned_dp_rank = {
            req_id: self.assigned_dp_rank[req_id]
            for req_id in scheduled_requests
            if req_id in self.assigned_dp_rank
        }
        logger.debug(f"Scheduled requests DP ranks: {assigned_dp_rank}")
        
        # Log state after scheduling
        self._log_dp_state()

        return DPSchedulerOutput(**base_output.__dict__, assigned_dp_rank=assigned_dp_rank)
    
    def _free_blocks(self, request: Request):
        """Override to handle DP-specific block freeing."""
        assert request.is_finished()
        req_dp_rank = self.assigned_dp_rank.get(request.request_id)
        if req_dp_rank is not None:
            self.dp_kv_cache_managers[req_dp_rank].free(request)
        del self.requests[request.request_id]
    
    def make_stats(self, spec_decoding_stats: Optional[SpecDecodingStats] = None, kv_connector_stats: Optional[KVConnectorStats] = None) -> Optional[SchedulerStats]:
        """Override to handle DP-specific stats."""
        if not self.log_stats:
            return None
        
        # Combine stats from all DP managers  
        combined_prefix_cache_stats = PrefixCacheStats()
        combined_usage = 0
        
        for kv_cache_mgr in self.dp_kv_cache_managers:
            prefix_cache_stats = kv_cache_mgr.make_prefix_cache_stats()
            if prefix_cache_stats:
                combined_prefix_cache_stats.reset = prefix_cache_stats.reset
                combined_prefix_cache_stats.requests += prefix_cache_stats.requests
                combined_prefix_cache_stats.queries += prefix_cache_stats.queries
                combined_prefix_cache_stats.hits += prefix_cache_stats.hits
            combined_usage += kv_cache_mgr.usage
            
        return SchedulerStats(
            num_running_reqs=len(self.running),
            num_waiting_reqs=len(self.waiting),
            kv_cache_usage=combined_usage / len(self.dp_kv_cache_managers),
            prefix_cache_stats=combined_prefix_cache_stats,
            spec_decoding_stats=spec_decoding_stats,
            num_corrupted_reqs=sum(req.is_output_corrupted for req in self.running),
            kv_connector_stats=kv_connector_stats.data if kv_connector_stats else None
        )


