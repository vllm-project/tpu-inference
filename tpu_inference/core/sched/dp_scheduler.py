import itertools
from dataclasses import dataclass
from typing import Optional

from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.request import Request
from vllm.v1.metrics.stats import PrefixCacheStats, SchedulerStats
from vllm.v1.spec_decode.metrics import SpecDecodingStats
from vllm.distributed.kv_transfer.kv_connector.v1.metrics import KVConnectorStats


@dataclass
class DPSchedulerOutput(SchedulerOutput):
    assigned_dp_rank: dict[str, int] = None


def create_dp_scheduler(base_scheduler_class=Scheduler):
    """Factory function to create a DP-enabled scheduler class."""
    
    class DPScheduler(base_scheduler_class):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            
            # Initialize DP configuration
            self._init_dp_config()
            
            if self.dp_size > 1:
                self._init_dp_state()
                self._replace_kv_cache_manager()
        
        def _init_dp_config(self):
            """Initialize DP configuration from vllm_config."""
            try:
                self.dp_size = self.vllm_config.additional_config["sharding"]["sharding_strategy"]["data_parallelism"]
                print("scheduler dp_size from config", self.dp_size)
            except (KeyError, TypeError, AttributeError):
                self.dp_size = 1
            
            # DP attention calculation
            if hasattr(self.vllm_config, 'model_config') and hasattr(self.vllm_config.model_config, 'hf_config'):
                num_kv_heads = self.vllm_config.model_config.hf_config.num_key_value_heads
                tp = self.vllm_config.parallel_config.tensor_parallel_size
                attn_dp = max(tp // num_kv_heads, 1)
                self.dp_size = self.dp_size * attn_dp
            
            print("wenxin: scheduler dp_size", self.dp_size)
        
        def _init_dp_state(self):
            """Initialize DP-specific state."""
            self.assigned_dp_rank: dict[str, int] = {}
            self.round_robin_counter = itertools.cycle(range(self.dp_size))
        
        def _replace_kv_cache_manager(self):
            """Replace single KV cache manager with multiple DP managers."""
            # Create DP-specific KV cache config
            dp_kv_cache_config = self.kv_cache_config
            dp_kv_cache_config.num_blocks = self.kv_cache_config.num_blocks // self.dp_size
            
            # Store original manager for reference
            self._original_kv_cache_manager = self.kv_cache_manager
            
            # Create multiple KV cache managers
            self.kv_cache_managers = [
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
            
            # Patch KV cache manager methods to handle DP logic
            self._patch_kv_operations()
        
        def _patch_kv_operations(self):
            """Patch KV cache operations to use DP logic."""
            # Create a proxy object that handles method calls
            class KVCacheManagerProxy:
                def __init__(self, scheduler):
                    self.scheduler = scheduler
                
                def allocate_slots(self, request, *args, **kwargs):
                    dp_rank = self.scheduler._get_or_assign_dp_rank(request)
                    return self.scheduler.kv_cache_managers[dp_rank].allocate_slots(request, *args, **kwargs)
                
                def get_computed_blocks(self, request):
                    return self.scheduler._get_computed_blocks_dp(request)
                
                def get_blocks(self, request_id):
                    dp_rank = self.scheduler.assigned_dp_rank[request_id]
                    return self.scheduler.kv_cache_managers[dp_rank].get_blocks(request_id)
                
                def get_block_ids(self, request_id):
                    dp_rank = self.scheduler.assigned_dp_rank[request_id]  
                    return self.scheduler.kv_cache_managers[dp_rank].get_block_ids(request_id)
                
                def free(self, request):
                    dp_rank = self.scheduler.assigned_dp_rank.get(request.request_id)
                    if dp_rank is not None:
                        self.scheduler.kv_cache_managers[dp_rank].free(request)
                
                def get_num_common_prefix_blocks(self, request_id):
                    dp_rank = self.scheduler.assigned_dp_rank[request_id]
                    return self.scheduler.kv_cache_managers[dp_rank].get_num_common_prefix_blocks(request_id)
                
                def take_events(self):
                    events = []
                    for manager in self.scheduler.kv_cache_managers:
                        events.extend(manager.take_events())
                    return events
                
                def reset_prefix_cache(self):
                    for manager in self.scheduler.kv_cache_managers:
                        if not manager.reset_prefix_cache():
                            return False
                    return True
                
                def make_prefix_cache_stats(self):
                    combined_stats = PrefixCacheStats()
                    for manager in self.scheduler.kv_cache_managers:
                        stats = manager.make_prefix_cache_stats()
                        if stats:
                            combined_stats.reset = stats.reset
                            combined_stats.requests += stats.requests
                            combined_stats.queries += stats.queries
                            combined_stats.hits += stats.hits
                    return combined_stats
                
                @property
                def usage(self):
                    return sum(manager.usage for manager in self.scheduler.kv_cache_manager) / len(self.scheduler.kv_cache_manager)
            
            # Replace the KV cache manager with our proxy
            self.kv_cache_manager = KVCacheManagerProxy(self)
        
        def _get_or_assign_dp_rank(self, request: Request) -> int:
            """Get or assign DP rank for a request."""
            req_id = request.request_id
            if req_id in self.assigned_dp_rank:
                return self.assigned_dp_rank[req_id]
            self.assigned_dp_rank[req_id] = next(self.round_robin_counter)
            return self.assigned_dp_rank[req_id]
        
        def _get_computed_blocks_dp(self, request: Request):
            """Get computed blocks with DP logic - check all managers for cache hits."""
            best_blocks = None
            best_tokens = 0
            best_rank = None
            
            # Check all DP managers for the best cache hit
            for rank in range(self.dp_size):
                blocks, tokens = self.kv_cache_managers[rank].get_computed_blocks(request)
                if len(blocks.blocks[0]) > best_tokens:
                    best_blocks = blocks
                    best_tokens = tokens
                    best_rank = rank
            
            if best_rank is not None and best_tokens > 0:
                self.assigned_dp_rank[request.request_id] = best_rank
                print(f"{request.request_id}. prefix cache hit: {best_rank}")
                return best_blocks, best_tokens
            else:
                # No cache hit, assign new rank and get empty blocks
                rank = self._get_or_assign_dp_rank(request)
                return self.kv_cache_managers[rank].get_computed_blocks(request)
        
        def schedule(self) -> SchedulerOutput:
            """Override schedule to return DP-specific output."""
            base_output = super().schedule()
            
            if self.dp_size <= 1:
                return base_output
            
            # Create DP-specific output
            scheduled_requests = []
            if hasattr(base_output, 'scheduled_new_reqs'):
                scheduled_requests.extend([req_data.req_id for req_data in base_output.scheduled_new_reqs])
            if hasattr(base_output, 'scheduled_cached_reqs'):
                scheduled_requests.extend(base_output.scheduled_cached_reqs.req_ids)
            
            assigned_dp_rank = {
                req_id: self.assigned_dp_rank[req_id]
                for req_id in scheduled_requests
                if req_id in self.assigned_dp_rank
            }
            
            return DPSchedulerOutput(
                scheduled_new_reqs=base_output.scheduled_new_reqs,
                scheduled_cached_reqs=base_output.scheduled_cached_reqs,
                num_scheduled_tokens=base_output.num_scheduled_tokens,
                total_num_scheduled_tokens=base_output.total_num_scheduled_tokens,
                scheduled_spec_decode_tokens=base_output.scheduled_spec_decode_tokens,
                scheduled_encoder_inputs=base_output.scheduled_encoder_inputs,
                num_common_prefix_blocks=base_output.num_common_prefix_blocks,
                finished_req_ids=base_output.finished_req_ids,
                free_encoder_mm_hashes=base_output.free_encoder_mm_hashes,
                structured_output_request_ids=base_output.structured_output_request_ids,
                grammar_bitmask=base_output.grammar_bitmask,
                kv_connector_metadata=getattr(base_output, 'kv_connector_metadata', None),
                assigned_dp_rank=assigned_dp_rank,
            )
        
        def _free_blocks(self, request: Request):
            """Override to handle DP-specific block freeing."""
            if self.dp_size > 1:
                assert request.is_finished()
                req_dp_rank = self.assigned_dp_rank.get(request.request_id)
                if req_dp_rank is not None:
                    self.kv_cache_managers[req_dp_rank].free(request)
                del self.requests[request.request_id]
            else:
                super()._free_blocks(request)
        
        def make_stats(self, spec_decoding_stats: Optional[SpecDecodingStats] = None, kv_connector_stats: Optional[KVConnectorStats] = None) -> Optional[SchedulerStats]:
            """Override to handle DP-specific stats."""
            if self.dp_size > 1:
                if not self.log_stats:
                    return None
                
                # Combine stats from all DP managers  
                combined_prefix_cache_stats = PrefixCacheStats()
                combined_usage = 0
                
                for kv_cache_mgr in self.kv_cache_managers:
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
                    kv_cache_usage=combined_usage / len(self.kv_cache_managers),
                    prefix_cache_stats=combined_prefix_cache_stats,
                    spec_decoding_stats=spec_decoding_stats,
                    num_corrupted_reqs=sum(req.is_output_corrupted for req in self.running),
                    kv_connector_stats=kv_connector_stats.data if kv_connector_stats else None
                )
            else:
                return super().make_stats(spec_decoding_stats, kv_connector_stats)
    
    return DPScheduler


# Create the DP scheduler class
DPScheduler = create_dp_scheduler(Scheduler)
