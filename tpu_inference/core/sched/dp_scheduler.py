import copy
import itertools
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

from vllm.distributed.kv_transfer.kv_connector.v1.metrics import \
    KVConnectorStats
from vllm.v1.core.kv_cache_manager import KVCacheBlocks, KVCacheManager
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.metrics.stats import PrefixCacheStats, SchedulerStats
from vllm.v1.request import Request
from vllm.v1.spec_decode.metrics import SpecDecodingStats

from tpu_inference.logger import init_logger

logger = init_logger(__name__)


@dataclass
class DPSchedulerOutput(SchedulerOutput):    
    assigned_dp_rank: Optional[Dict[str, int]] = None

    def __init__(self, *args, assigned_dp_rank=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.assigned_dp_rank = assigned_dp_rank or {}


class DPScheduler(Scheduler):
    """
    DPScheduler extends the base vLLM Scheduler to support data-parallel (DP) execution.
    It manages request distribution and KV cache allocation across multiple data-parallel ranks.

    **DP Rank**: Each rank has its own logical KV cache shard and processes a subset
    of the total requests.

    The DPScheduler manages:
    1. Multiple KV cache managers (one per DP rank)
    2. Request-to-rank assignment tracking
    3. Per-rank resource budgets (requests budget and tokens budget)

    **Request Assignment**

    For new requests:
    - Uses round-robin to find an available DP rank
    - A DP rank is available if it has capacity for both new requests and new tokens
    - If there is prefix cache hit, assigns request to the rank with the best hit

    Once a rank is assigned to a request, it remains fixed for the request's lifetime.
    A request will be freed from its assigned rank when it is completed or preempted.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.dp_size: int = self.vllm_config.sharding_config.total_dp_size

        self._init_dp_state()
        self._replace_kv_cache_manager()

        logger.info(f"Scheduler initialized with DP size: {self.dp_size}")
        logger.info(f"Max requests per DP rank: {self.max_reqs_per_dp_rank}")

    def _init_dp_state(self) -> None:
        
        self.assigned_dp_rank: Dict[str, int] = {}  # req_id -> dp_rank
        
        self.num_scheduled_tokens: Dict[str, int] = {}  # req_id -> num_tokens
        
        self.round_robin_counter: Iterator[int] = itertools.cycle(range(self.dp_size))
        
        # Empty KV cache blocks placeholder for requests with no cache hits
        self.empty_kv_cache_blocks: KVCacheBlocks = KVCacheBlocks(
            tuple(() for _ in range(self.kv_cache_manager.num_kv_cache_groups))
        )

        self.max_reqs_per_dp_rank: int = self.scheduler_config.max_num_seqs // self.dp_size
        self.max_tokens_per_dp_rank: int = self.scheduler_config.max_num_batched_tokens // self.dp_size
        
        # Initialize per-rank budgets
        self.request_budget: Dict[int, int] = {
            i: self.max_reqs_per_dp_rank
            for i in range(self.dp_size)
        }
        self.token_budget: Dict[int, int] = {
            i: self.max_tokens_per_dp_rank
            for i in range(self.dp_size)
        }

    def get_dynamic_token_budget(
        self, 
        request: Request,
        available_budget: int
    ) -> int:
        """Override token budget based on DP rank constraints.

        For DP scheduling, we need to constrain the token budget based on:
        1. For new requests: the maximum token budget across all DP ranks
        2. For existing requests: the token budget specific to its DP rank
        
        Args:
            request: The request to get the token budget for.
            available_budget: The total available token budget from 
                the base scheduler.
            
        Returns:
            The token budget for this request considering DP constraints.
        """
        if request.request_id in self.assigned_dp_rank:
            dp_rank = self.assigned_dp_rank[request.request_id]
            assert available_budget >= self.token_budget[dp_rank], (
                f"Available budget {available_budget} is less than "
                f"DP rank {dp_rank} budget {self.token_budget[dp_rank]}"
            )
            return self.token_budget[dp_rank]
        else:
            max_budget = max(self.token_budget.values())
            assert available_budget >= max_budget, (
                f"Available budget {available_budget} is less than "
                f"max DP rank budget {max_budget}"
            )
            return max_budget

    def _replace_kv_cache_manager(self) -> None:
        """Replace `self.kv_cache_manager` with `KVCacheManagerProxy`.
        
        This allows us to intercept KV cache operations and route them to the 
        appropriate DP rank's KV cache manager. Essentially, we want to route 
        `self.kv_cache_manager.fn()` calls to `self.dp_kv_cache_managers[dp_rank].fn()`.
        """
        dp_kv_cache_config = copy.deepcopy(self.kv_cache_config)
        dp_kv_cache_config.num_blocks //= self.dp_size

        # Initialize KV cache managers for each DP rank
        self.dp_kv_cache_managers: List[KVCacheManager] = [
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

    def _patch_kv_operations(self) -> None:
        """Create KVCacheManagerProxy to intercept KV operations."""

        class KVCacheManagerProxy:

            def __init__(self, scheduler: "DPScheduler") -> None:
                self.scheduler = scheduler

            def allocate_slots(
                self, 
                request: Request, 
                num_new_tokens: int,
                *args: Any, 
                **kwargs: Any
            ) -> Optional[KVCacheBlocks]:
                """Allocate KV cache slots for the given request.
                
                This overrides the `self.kv_cache_manager.allocate_slots` function in the
                base scheduler. For new requests, it finds an available DP rank.
                For existing requests, it checks if the assigned DP rank has token capacity.
                
                Args:
                    request: The request to allocate slots for.
                    num_new_tokens: Number of new tokens to allocate.
                    *args: Additional positional arguments.
                    **kwargs: Additional keyword arguments.
                    
                Returns:
                    KVCacheBlocks if allocation succeeds, None otherwise.
                """
                dp_rank = self.scheduler._get_or_find_dp_rank(request, num_new_tokens)
                req_id = request.request_id
                
                if dp_rank is None:
                    # Request cannot be scheduled - check if we need to clean up
                    if (req_id in self.scheduler.assigned_dp_rank and 
                        self.scheduler.num_scheduled_tokens.get(req_id, 0) == 0):
                        # This can happen if the request was previously assigned 
                        # a DP rank during prefix cache, but no tokens are scheduled 
                        # yet. Remove the assignment to allow future rescheduling.
                        assigned_rank = self.scheduler.assigned_dp_rank[req_id]
                        self.scheduler.request_budget[assigned_rank] += 1
                        self.scheduler.assigned_dp_rank.pop(req_id)
                    return None

                # Assign rank and update budgets
                self.scheduler._assign_rank_and_update_budget(request, dp_rank, num_new_tokens)
                
                # Allocate slots from the assigned DP rank's KV cache manager
                new_blocks = self.scheduler.dp_kv_cache_managers[dp_rank].allocate_slots(
                    request, num_new_tokens, *args, **kwargs
                )
                return new_blocks

            def get_computed_blocks(self, request: Request) -> Tuple[KVCacheBlocks, int]:
                """Get computed blocks for a request from the appropriate DP rank."""
                return self.scheduler._get_computed_blocks_dp(request)

            def get_blocks(self, request_id: str) -> Any:
                """Get blocks for a request from its assigned DP rank."""
                dp_rank = self.scheduler.assigned_dp_rank[request_id]
                return self.scheduler.dp_kv_cache_managers[dp_rank].get_blocks(request_id)

            def get_block_ids(self, request_id: str) -> Any:
                """Get block IDs for a request from its assigned DP rank."""
                dp_rank = self.scheduler.assigned_dp_rank[request_id]
                return self.scheduler.dp_kv_cache_managers[dp_rank].get_block_ids(request_id)

            def free(self, request: Request) -> None:
                """Free the KV cache blocks associated with a completed or preempted request."""
                req_id = request.request_id
                dp_rank = self.scheduler.assigned_dp_rank.get(req_id)
                
                logger.debug(
                    f"Freeing request {req_id} (DP rank {dp_rank})"
                )
                
                assert dp_rank is not None, f"Request {req_id} has no assigned DP rank"
                
                self.scheduler.assigned_dp_rank.pop(req_id)
                self.scheduler.request_budget[dp_rank] += 1                
                if req_id in self.scheduler.num_scheduled_tokens:
                    tokens_to_return = self.scheduler.num_scheduled_tokens[req_id]
                    self.scheduler.token_budget[dp_rank] += tokens_to_return
                    self.scheduler.num_scheduled_tokens.pop(req_id)
                self.scheduler.dp_kv_cache_managers[dp_rank].free(request)

            def get_num_common_prefix_blocks(self, request_id: str) -> int:
                """Get number of common prefix blocks for a request from its assigned DP rank."""
                dp_rank = self.scheduler.assigned_dp_rank[request_id]
                return self.scheduler.dp_kv_cache_managers[dp_rank].get_num_common_prefix_blocks(request_id)

            def take_events(self) -> List[Any]:
                """Collect events from all DP rank KV cache managers."""
                events = []
                for manager in self.scheduler.dp_kv_cache_managers:
                    events.extend(manager.take_events())
                return events

            def reset_prefix_cache(self) -> bool:
                """Reset prefix cache for all DP ranks."""
                for manager in self.scheduler.dp_kv_cache_managers:
                    if not manager.reset_prefix_cache():
                        return False
                return True

            def make_prefix_cache_stats(self) -> PrefixCacheStats:
                """Combine prefix cache stats from all DP ranks."""
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
            def usage(self) -> float:
                """Get average KV cache usage across all DP ranks."""
                return sum(
                    manager.usage for manager in self.scheduler.dp_kv_cache_managers
                ) / len(self.scheduler.dp_kv_cache_managers)

        # Replace the original KV cache manager with our proxy
        self.kv_cache_manager = KVCacheManagerProxy(self)

    def _get_or_find_dp_rank(
        self,
        request: Request,
        num_tokens: int = 0
    ) -> Optional[int]:
        """Find an available DP rank for the incoming request.
        
        Args:
            request: The request to find a DP rank for.
            num_tokens: Number of tokens needed for this request.
            
        Returns:
            The DP rank if available, None if no ranks have capacity.
        """
        assert num_tokens > 0

        req_id = request.request_id
        
        # Handle existing requests - check if assigned rank has capacity
        if req_id in self.assigned_dp_rank:
            dp_rank = self.assigned_dp_rank[req_id]
            if self._rank_has_capacity_for_existing_requests(dp_rank, num_tokens):
                return dp_rank
            logger.debug(
                f"Running request {req_id} cannot be scheduled: DP rank {dp_rank} "
                f"lacks capacity for {num_tokens} tokens. "
                f"Current token budget: {self.token_budget[dp_rank]}"
            )
            return None

        # Handle new requests - use round-robin to find available rank
        for _ in range(self.dp_size):
            candidate_rank = next(self.round_robin_counter)
            if self._rank_has_capacity_for_new_requests(candidate_rank, num_tokens):
                return candidate_rank
                
        logger.debug(
            f"New request {req_id} cannot be scheduled: no DP rank with "
            f"capacity for {num_tokens} tokens. Current token budgets: {self.token_budget}"
        )
        return None

    def _assign_rank_and_update_budget(
        self, 
        request: Request, 
        dp_rank: int,
        num_tokens: int
    ) -> None:
        """Assign the DP rank to the request and update resource budgets.
        
        Args:
            request: The request to assign a rank to.
            dp_rank: The DP rank to assign to the request.
            num_tokens: Number of tokens being allocated.
        """
        req_id = request.request_id
        
        if req_id not in self.assigned_dp_rank:
            self.assigned_dp_rank[req_id] = dp_rank
        else:
            assert dp_rank == self.assigned_dp_rank[req_id], (
                f"DP rank mismatch: expected {self.assigned_dp_rank[req_id]}, "
                f"got {dp_rank} for request {req_id}"
            )

        self.request_budget[dp_rank] -= 1
        self.token_budget[dp_rank] -= num_tokens
        self.num_scheduled_tokens[req_id] = num_tokens

    def _rank_has_capacity_for_new_requests(self, dp_rank: int, num_tokens: int) -> bool:
        """Check if a DP rank has capacity for new requests.
        
        Args:
            dp_rank: The DP rank to check.
            num_tokens: Number of tokens needed.
            
        Returns:
            True if the rank can accept new requests with the given token count.
        """
        return (self.request_budget[dp_rank] > 0 and 
                self.token_budget[dp_rank] >= num_tokens)

    def _rank_has_capacity_for_existing_requests(self, dp_rank: int, num_tokens: int) -> bool:
        """Check if a DP rank has token capacity for existing requests.
        
        Args:
            dp_rank: The DP rank to check.
            num_tokens: Number of tokens needed.
            
        Returns:
            True if the rank has sufficient token capacity.
        """
        return self.token_budget[dp_rank] >= num_tokens

    def _get_computed_blocks_dp(self, request: Request) -> Tuple[KVCacheBlocks, int]:
        """Get cached blocks and tokens for the request with DP-aware prefix caching.
        
        If the request is already assigned to a DP rank, only checks that rank. 
        Otherwise, checks all DP ranks and assigns the request to the rank with 
        the best cache hit.
        
        Args:
            request: The request to get computed blocks for.
            
        Returns:
            Tuple of (KVCacheBlocks, number of cached tokens).
        """
        req_id = request.request_id
        
        # If request is already assigned, use that rank
        if req_id in self.assigned_dp_rank:
            rank = self.assigned_dp_rank[req_id]
            return self.dp_kv_cache_managers[rank].get_computed_blocks(request)

        # Search all DP ranks for the best cache hit
        best_blocks = None
        best_tokens = 0
        best_rank = None

        for rank in range(self.dp_size):
            # Only consider ranks with request capacity
            # Token capacity will be checked later in allocate_slots
            if self.request_budget[rank] > 0:
                blocks, tokens = self.dp_kv_cache_managers[rank].get_computed_blocks(request)
                if tokens > best_tokens:
                    best_blocks = blocks
                    best_tokens = tokens
                    best_rank = rank
                    
        # Assign the best rank if we found a cache hit
        if best_rank is not None and best_tokens > 0:
            self.assigned_dp_rank[req_id] = best_rank
            self.request_budget[best_rank] -= 1
            logger.debug(
                f"Assigned DP rank {best_rank} to request {req_id} "
                f"with {best_tokens} cached tokens."
            )
            return best_blocks, best_tokens
        else:
            return self.empty_kv_cache_blocks, 0

    def schedule(self) -> DPSchedulerOutput:
        """Schedule requests across DP ranks and return extended output.
        
        Returns:
            DPSchedulerOutput containing both base scheduler results and DP rank assignments.
        """
        # Reset budgets for each scheduling round
        self.request_budget = {
            i: self.max_reqs_per_dp_rank
            for i in range(self.dp_size)
        }
        self.token_budget = {
            i: self.max_tokens_per_dp_rank
            for i in range(self.dp_size)
        }
        self.num_scheduled_tokens = {}

        # Run base scheduler logic
        base_output = super().schedule()

        assigned_dp_rank = {
            req_id: self.assigned_dp_rank[req_id]
            for req_id in base_output.num_scheduled_tokens.keys()
        }

        self._log_dp_state()
        
        return DPSchedulerOutput(
            **base_output.__dict__,
            assigned_dp_rank=assigned_dp_rank
        )

    def make_stats(
        self,
        spec_decoding_stats: Optional[SpecDecodingStats] = None,
        kv_connector_stats: Optional[KVConnectorStats] = None
    ) -> Optional[SchedulerStats]:
        """Create scheduler stats by combining stats from all DP ranks."""
        if not self.log_stats:
            return None

        # Combine stats from all DP KV cache managers
        combined_prefix_cache_stats = PrefixCacheStats()
        combined_usage = 0.0

        for kv_cache_mgr in self.dp_kv_cache_managers:
            prefix_cache_stats = kv_cache_mgr.make_prefix_cache_stats()
            if prefix_cache_stats:
                combined_prefix_cache_stats.reset = prefix_cache_stats.reset
                combined_prefix_cache_stats.requests += prefix_cache_stats.requests
                combined_prefix_cache_stats.queries += prefix_cache_stats.queries
                combined_prefix_cache_stats.hits += prefix_cache_stats.hits
            combined_usage += kv_cache_mgr.usage

        average_usage = combined_usage / len(self.dp_kv_cache_managers)

        return SchedulerStats(
            num_running_reqs=len(self.running),
            num_waiting_reqs=len(self.waiting),
            kv_cache_usage=average_usage,
            prefix_cache_stats=combined_prefix_cache_stats,
            spec_decoding_stats=spec_decoding_stats,
            num_corrupted_reqs=sum(req.is_output_corrupted for req in self.running),
            kv_connector_stats=kv_connector_stats.data if kv_connector_stats else None
        )

    def reset_prefix_cache(self) -> bool:
        """Reset prefix cache for all DP ranks.
        
        Returns:
            True if all DP ranks successfully reset their prefix cache, False otherwise.
        """
        for manager in self.dp_kv_cache_managers:
            if not manager.reset_prefix_cache():
                return False
        return True

    def _log_dp_state(self) -> None:
        """Log current DP scheduler state for debugging purposes."""
        logger.debug(
            f"DP rank token budgets (cap {self.max_tokens_per_dp_rank}): {self.token_budget}"
        )
        logger.debug(
            f"DP rank request budgets (cap {self.max_reqs_per_dp_rank}): {self.request_budget}"
        )
        logger.debug(f"Total assigned requests: {len(self.assigned_dp_rank)}")


def update_vllm_config_for_dp_scheduler(vllm_config: Any) -> None:
    """Modifies the vLLM configuration to use the DPScheduler when
    data parallelism size is greater than 1. It also scales the scheduler
    limits appropriately for the DP configuration.
    
    Args:
        vllm_config: The vLLM configuration object to update.
    """
    dp_size = vllm_config.sharding_config.total_dp_size
    
    if dp_size > 1:
        from tpu_inference.core.sched.dp_scheduler import DPScheduler
        
        vllm_config.scheduler_config.max_num_seqs *= dp_size
        vllm_config.scheduler_config.max_num_batched_tokens *= dp_size
        vllm_config.scheduler_config.scheduler_cls = DPScheduler
        
        logger.info(f"DP size ({dp_size}) >= 2, using DPScheduler")
    else:
        logger.info(f"DP size ({dp_size}) < 2, using default Scheduler")
