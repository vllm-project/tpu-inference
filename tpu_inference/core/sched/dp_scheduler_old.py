import itertools
import os
import time
from dataclasses import dataclass
from typing import Optional

from vllm.distributed.kv_events import KVEventBatch
from vllm.logger import init_logger
from vllm.v1.core.kv_cache_manager import KVCacheBlocks, KVCacheManager
from vllm.v1.core.sched.output import NewRequestData, SchedulerOutput
from vllm.v1.core.sched.request_queue import (SchedulingPolicy,
                                              create_request_queue)
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.engine import EngineCoreEventType
from vllm.v1.request import Request, RequestStatus
from vllm.v1.metrics.stats import PrefixCacheStats
from vllm.v1.metrics.stats import SchedulerStats
from vllm.v1.spec_decode.metrics import SpecDecodingStats
from vllm.distributed.kv_transfer.kv_connector.v1.metrics import (
    KVConnectorStats)

logger = init_logger(__name__)


@dataclass
class DPSchedulerOutput(SchedulerOutput):
    assigned_dp_rank: dict[str, int] = None


@dataclass
class DPScheduler(Scheduler):

    def __init__(self, *args, **kwargs):
        super(DPScheduler, self).__init__(*args, **kwargs)

        self.assigned_dp_rank: dict[str, int] = {}
        try:
            # breakpoint()
            self.dp_size = self.vllm_config.additional_config["sharding"]["sharding_strategy"]["data_parallelism"]   
            print("scheduler dp_size from config", self.dp_size)
        except KeyError:
            self.dp_size = 1
        
        # DP attention.
        num_kv_heads = self.vllm_config.model_config.hf_config.num_key_value_heads
        tp = self.vllm_config.parallel_config.tensor_parallel_size
        attn_dp = max(tp // num_kv_heads, 1)
        self.dp_size = self.dp_size * attn_dp
        print("wenxin: scheduler dp:_size", self.dp_size)
        
        # assert self.dp_size >= 2, "Data parallel size must be at least 2." # Debugging purpose
        self.dp_kv_cache_config = self.kv_cache_config
        self.dp_kv_cache_config.num_blocks = (
            self.kv_cache_config.num_blocks // self.dp_size)
        self.round_robin_counter = itertools.cycle(range(self.dp_size))

        self.kv_cache_manager = [
            KVCacheManager(
                kv_cache_config=self.dp_kv_cache_config,
                max_model_len=self.max_model_len,
                enable_caching=self.cache_config.enable_prefix_caching,
                use_eagle=self.use_eagle,
                log_stats=self.log_stats,
                enable_kv_cache_events=self.enable_kv_cache_events,
                dcp_world_size=self.dcp_world_size,
            ) for _ in range(self.dp_size)
        ]

    def assign_dp_rank(self, request: Request) -> Optional[int]:
        """Assign a data parallel rank to the request based on round robin.
        TODO(wenxindong): consider more advanced load balancing strategies.
        """
        req_id = request.request_id
        if req_id in self.assigned_dp_rank:
            return self.assigned_dp_rank[req_id]
        self.assigned_dp_rank[req_id] = next(self.round_robin_counter)
        return self.assigned_dp_rank[req_id]

    def schedule(self) -> SchedulerOutput:
        # NOTE(woosuk) on the scheduling algorithm:
        # There's no "decoding phase" nor "prefill phase" in the scheduler.
        # Each request just has the num_computed_tokens and
        # num_tokens_with_spec. num_tokens_with_spec =
        # len(prompt_token_ids) + len(output_token_ids) + len(spec_token_ids).
        # At each step, the scheduler tries to assign tokens to the requests
        # so that each request's num_computed_tokens can catch up its
        # num_tokens_with_spec. This is general enough to cover
        # chunked prefills, prefix caching, speculative decoding,
        # and the "jump decoding" optimization in the future.

        scheduled_new_reqs: list[Request] = []
        scheduled_resumed_reqs: list[Request] = []
        scheduled_running_reqs: list[Request] = []
        preempted_reqs: list[Request] = []

        req_to_new_blocks: dict[str, KVCacheBlocks] = {}
        num_scheduled_tokens: dict[str, int] = {}
        token_budget = self.max_num_scheduled_tokens
        # Encoder-related.
        scheduled_encoder_inputs: dict[str, list[int]] = {}
        encoder_compute_budget = self.max_num_encoder_input_tokens
        # Spec decode-related.
        scheduled_spec_decode_tokens: dict[str, list[int]] = {}

        # For logging.
        scheduled_timestamp = time.monotonic()

        # First, schedule the RUNNING requests.
        req_index = 0
        while req_index < len(self.running) and token_budget > 0:
            request = self.running[req_index]
            num_new_tokens = (request.num_tokens_with_spec +
                              request.num_output_placeholders -
                              request.num_computed_tokens)
            if 0 < self.scheduler_config.long_prefill_token_threshold < num_new_tokens:
                num_new_tokens = self.scheduler_config.long_prefill_token_threshold
            num_new_tokens = min(num_new_tokens, token_budget)

            # Make sure the input position does not exceed the max model len.
            # This is necessary when using spec decoding.
            num_new_tokens = min(
                num_new_tokens,
                self.max_model_len - 1 - request.num_computed_tokens)

            # Schedule encoder inputs.
            encoder_inputs_to_schedule = None
            new_encoder_compute_budget = encoder_compute_budget
            if request.has_encoder_inputs:
                (
                    encoder_inputs_to_schedule,
                    num_new_tokens,
                    new_encoder_compute_budget,
                ) = self._try_schedule_encoder_inputs(
                    request,
                    request.num_computed_tokens,
                    num_new_tokens,
                    encoder_compute_budget,
                )

            if num_new_tokens == 0:
                # The request cannot be scheduled because one of the following
                # reasons:
                # 1. No new tokens to schedule. This may happen when
                #    (1) PP>1 and we have already scheduled all prompt tokens
                #    but they are not finished yet.
                #    (2) Async scheduling and the request has reached to either
                #    its max_total_tokens or max_model_len.
                # 2. The encoder budget is exhausted.
                # 3. The encoder cache is exhausted.
                # NOTE(woosuk): Here, by doing `continue` instead of `break`,
                # we do not strictly follow the FCFS scheduling policy and
                # allow the lower-priority requests to be scheduled.
                req_index += 1
                continue

            while True:
                req_dp_rank = self.assigned_dp_rank[request.request_id]

                new_blocks = self.kv_cache_manager[req_dp_rank].allocate_slots(
                    request,
                    num_new_tokens,
                    num_lookahead_tokens=self.num_lookahead_tokens,
                )
                if new_blocks is None:
                    # The request cannot be scheduled.
                    # Preempt the lowest-priority request.
                    if self.policy == SchedulingPolicy.PRIORITY:
                        preempted_req = max(
                            self.running,
                            key=lambda r: (r.priority, r.arrival_time),
                        )
                        self.running.remove(preempted_req)
                        if preempted_req in scheduled_running_reqs:
                            scheduled_running_reqs.remove(preempted_req)
                    else:
                        preempted_req = self.running.pop()

                    self.kv_cache_manager[req_dp_rank].free(preempted_req)
                    self.encoder_cache_manager.free(preempted_req)
                    preempted_req.status = RequestStatus.PREEMPTED
                    preempted_req.num_computed_tokens = 0
                    if self.log_stats:
                        preempted_req.record_event(
                            EngineCoreEventType.PREEMPTED, scheduled_timestamp)

                    self.waiting.prepend_request(preempted_req)
                    preempted_reqs.append(preempted_req)
                    if preempted_req == request:
                        # No more request to preempt.
                        can_schedule = False
                        break
                else:
                    # The request can be scheduled.
                    can_schedule = True
                    break
            if not can_schedule:
                break
            assert new_blocks is not None

            # Schedule the request.
            scheduled_running_reqs.append(request)
            req_to_new_blocks[request.request_id] = new_blocks
            num_scheduled_tokens[request.request_id] = num_new_tokens
            token_budget -= num_new_tokens
            req_index += 1

            # Speculative decode related.
            if request.spec_token_ids:
                num_scheduled_spec_tokens = (num_new_tokens +
                                             request.num_computed_tokens -
                                             request.num_tokens)
                if num_scheduled_spec_tokens > 0:
                    # Trim spec_token_ids list to num_scheduled_spec_tokens.
                    del request.spec_token_ids[num_scheduled_spec_tokens:]
                    scheduled_spec_decode_tokens[request.request_id] = (
                        request.spec_token_ids)

            # Encoder-related.
            if encoder_inputs_to_schedule:
                scheduled_encoder_inputs[request.request_id] = (
                    encoder_inputs_to_schedule)
                # Allocate the encoder cache.
                for i in encoder_inputs_to_schedule:
                    self.encoder_cache_manager.allocate(request, i)
                encoder_compute_budget = new_encoder_compute_budget

        # Record the LoRAs in scheduled_running_reqs
        scheduled_loras: set[int] = set()
        if self.lora_config:
            scheduled_loras = set(
                req.lora_request.lora_int_id for req in scheduled_running_reqs
                if req.lora_request and req.lora_request.lora_int_id > 0)
            assert len(scheduled_loras) <= self.lora_config.max_loras

        # Use a temporary RequestQueue to collect requests that need to be
        # skipped and put back at the head of the waiting queue later
        skipped_waiting_requests = create_request_queue(self.policy)
        # Next, schedule the WAITING requests.
        if not preempted_reqs:
            while self.waiting and token_budget > 0:
                if len(self.running) == self.max_num_running_reqs:
                    break

                request = self.waiting.peek_request()

                # KVTransfer: skip request if still waiting for remote kvs.
                if request.status == RequestStatus.WAITING_FOR_REMOTE_KVS:
                    is_ready = self._update_waiting_for_remote_kv(request)
                    if is_ready:
                        request.status = RequestStatus.WAITING
                    else:
                        logger.debug(
                            "%s is still in WAITING_FOR_REMOTE_KVS state.",
                            request.request_id,
                        )
                        self.waiting.pop_request()
                        skipped_waiting_requests.prepend_request(request)
                        continue

                # Skip request if the structured output request is still waiting
                # for FSM compilation.
                if request.status == RequestStatus.WAITING_FOR_FSM:
                    structured_output_req = request.structured_output_request
                    if structured_output_req and structured_output_req.grammar:
                        request.status = RequestStatus.WAITING
                    else:
                        self.waiting.pop_request()
                        skipped_waiting_requests.prepend_request(request)
                        continue

                # Check that adding the request still respects the max_loras
                # constraint.
                if (self.lora_config and request.lora_request and
                    (len(scheduled_loras) == self.lora_config.max_loras and
                     request.lora_request.lora_int_id not in scheduled_loras)):
                    # Scheduling would exceed max_loras, skip.
                    self.waiting.pop_request()
                    skipped_waiting_requests.prepend_request(request)
                    continue

                num_external_computed_tokens = 0
                load_kv_async = False

                # Get already-cached tokens.
                if request.num_computed_tokens == 0:

                    dp_rank = None
                    max_num_local_computed_tokens = -1
                    new_computed_blocks, num_new_local_computed_tokens = None, None
                    for rank in range(self.dp_size):
                        # Get locally-cached tokens.
                        # DO NOT SUBMIT: this is where prefill cache hit happens
                        _new_computed_blocks, _num_new_local_computed_tokens = (
                            kv_cache_managers[rank].get_computed_blocks(
                                request))
                        if (len(_new_computed_blocks.blocks[0])
                                > max_num_local_computed_tokens):
                            if len(_new_computed_blocks.blocks[0]) > 0:
                                dp_rank = rank
                            max_num_local_computed_tokens = len(
                                _new_computed_blocks.blocks[0])
                            new_computed_blocks = _new_computed_blocks
                            num_new_local_computed_tokens = (
                                _num_new_local_computed_tokens)
                    if dp_rank is None:
                        dp_rank = self.assign_dp_rank(request)
                    else:
                        self.assigned_dp_rank[request.request_id] = dp_rank
                        print(f"{request.request_id}. prefix cache hit: {dp_rank}")

                    # Get externally-cached tokens if using a KVConnector.
                    if self.connector is not None:
                        num_external_computed_tokens, load_kv_async = (
                            self.connector.get_num_new_matched_tokens(
                                request, num_new_local_computed_tokens))

                        if num_external_computed_tokens is None:
                            # The request cannot be scheduled because
                            # the KVConnector couldn't determine
                            # the number of matched tokens.
                            self.waiting.pop_request()
                            skipped_waiting_requests.prepend_request(request)
                            continue

                    # Total computed tokens (local + external).
                    num_computed_tokens = (num_new_local_computed_tokens +
                                           num_external_computed_tokens)
                # KVTransfer: WAITING reqs have num_computed_tokens > 0
                # after async KV recvs are completed.
                else:
                    dp_rank = self.assigned_dp_rank[request.request_id]
                    new_computed_blocks = self.kv_cache_manager[
                        dp_rank].create_empty_block_list()
                    num_new_local_computed_tokens = 0
                    num_computed_tokens = request.num_computed_tokens

                encoder_inputs_to_schedule = None
                new_encoder_compute_budget = encoder_compute_budget

                # KVTransfer: loading remote KV, do not allocate for new work.
                if load_kv_async:
                    assert num_external_computed_tokens > 0
                    num_new_tokens = 0
                # Number of tokens to be scheduled.
                else:
                    # We use `request.num_tokens` instead of
                    # `request.num_prompt_tokens` to consider the resumed
                    # requests, which have output tokens.
                    num_new_tokens = request.num_tokens - num_computed_tokens
                    if (0 < self.scheduler_config.long_prefill_token_threshold
                            < num_new_tokens):
                        num_new_tokens = (
                            self.scheduler_config.long_prefill_token_threshold)

                    # chunked prefill has to be enabled explicitly to allow
                    # pooling requests to be chunked
                    if (not self.scheduler_config.chunked_prefill_enabled
                            and num_new_tokens > token_budget):
                        self.waiting.pop_request()
                        skipped_waiting_requests.prepend_request(request)
                        continue

                    num_new_tokens = min(num_new_tokens, token_budget)
                    assert num_new_tokens > 0

                    # Schedule encoder inputs.
                    if request.has_encoder_inputs:
                        (
                            encoder_inputs_to_schedule,
                            num_new_tokens,
                            new_encoder_compute_budget,
                        ) = self._try_schedule_encoder_inputs(
                            request,
                            num_computed_tokens,
                            num_new_tokens,
                            encoder_compute_budget,
                        )
                        if num_new_tokens == 0:
                            # The request cannot be scheduled.
                            break

                # Handles an edge case when P/D Disaggregation
                # is used with Spec Decoding where an
                # extra block gets allocated which
                # creates a mismatch between the number
                # of local and remote blocks.
                effective_lookahead_tokens = (0 if request.num_computed_tokens
                                              == 0 else
                                              self.num_lookahead_tokens)

                # Determine if we need to allocate cross-attention blocks.
                if self.is_encoder_decoder and request.has_encoder_inputs:
                    # TODO(russellb): For Whisper, we know that the input is
                    # always padded to the maximum length. If we support other
                    # encoder-decoder models, this will need to be updated if we
                    # want to only allocate what is needed.
                    num_encoder_tokens = (
                        self.scheduler_config.max_num_encoder_input_tokens)
                else:
                    num_encoder_tokens = 0
                
                req_dp_rank = self.assigned_dp_rank[request.request_id]
                new_blocks = self.kv_cache_manager[req_dp_rank].allocate_slots(
                    request,
                    num_new_tokens + num_external_computed_tokens,
                    num_new_local_computed_tokens,
                    new_computed_blocks,
                    num_lookahead_tokens=effective_lookahead_tokens,
                    delay_cache_blocks=load_kv_async,
                    num_encoder_tokens=num_encoder_tokens,
                )

                if new_blocks is None:
                    # The request cannot be scheduled.
                    break

                # KVTransfer: the connector uses this info to determine
                # if a load is needed. Note that
                # This information is used to determine if a load is
                # needed for this request.
                if self.connector is not None:
                    self.connector.update_state_after_alloc(
                        request,
                        new_computed_blocks + new_blocks,
                        num_external_computed_tokens,
                    )

                # Request was already popped from self.waiting
                # unless it was re-added above due to new_blocks being None.
                request = self.waiting.pop_request()
                if load_kv_async:
                    # If loading async, allocate memory and put request
                    # into the WAITING_FOR_REMOTE_KV state.
                    skipped_waiting_requests.prepend_request(request)
                    request.status = RequestStatus.WAITING_FOR_REMOTE_KVS
                    continue

                req_index += 1
                self.running.append(request)
                if self.log_stats:
                    request.record_event(EngineCoreEventType.SCHEDULED,
                                         scheduled_timestamp)
                if request.status == RequestStatus.WAITING:
                    scheduled_new_reqs.append(request)
                elif request.status == RequestStatus.PREEMPTED:
                    scheduled_resumed_reqs.append(request)
                else:
                    raise RuntimeError(
                        f"Invalid request status: {request.status}")

                if self.lora_config and request.lora_request:
                    scheduled_loras.add(request.lora_request.lora_int_id)
                req_to_new_blocks[request.request_id] = self.kv_cache_manager[
                    req_dp_rank].get_blocks(request.request_id)
                num_scheduled_tokens[request.request_id] = num_new_tokens
                token_budget -= num_new_tokens
                request.status = RequestStatus.RUNNING
                request.num_computed_tokens = num_computed_tokens
                # Count the number of prefix cached tokens.
                if request.num_cached_tokens < 0:
                    request.num_cached_tokens = num_computed_tokens
                # Encoder-related.
                if encoder_inputs_to_schedule:
                    scheduled_encoder_inputs[request.request_id] = (
                        encoder_inputs_to_schedule)
                    # Allocate the encoder cache.
                    for i in encoder_inputs_to_schedule:
                        self.encoder_cache_manager.allocate(request, i)
                    encoder_compute_budget = new_encoder_compute_budget

        # Put back any skipped requests at the head of the waiting queue
        if skipped_waiting_requests:
            self.waiting.prepend_requests(skipped_waiting_requests)

        # Check if the scheduling constraints are satisfied.
        total_num_scheduled_tokens = sum(num_scheduled_tokens.values())
        assert total_num_scheduled_tokens <= self.max_num_scheduled_tokens
        assert token_budget >= 0
        assert len(self.running) <= self.max_num_running_reqs
        # Since some requests in the RUNNING queue may not be scheduled in
        # this step, the total number of scheduled requests can be smaller than
        # len(self.running).
        assert len(scheduled_new_reqs) + len(scheduled_resumed_reqs) + len(
            scheduled_running_reqs) <= len(self.running)

        # Get the longest common prefix among all requests in the running queue.
        # This can be potentially used for cascade attention.
        num_common_prefix_blocks = [0] * len(
            self.kv_cache_config.kv_cache_groups)
        if self.running:
            any_request = self.running[0]
            num_common_prefix_blocks = self.kv_cache_manager[
                req_dp_rank].get_num_common_prefix_blocks(
                    any_request.request_id)

        # Construct the scheduler output.
        new_reqs_data = [
            NewRequestData.from_request(
                req, req_to_new_blocks[req.request_id].get_block_ids())
            for req in scheduled_new_reqs
        ]
        cached_reqs_data = self._make_cached_request_data(
            scheduled_running_reqs,
            scheduled_resumed_reqs,
            num_scheduled_tokens,
            scheduled_spec_decode_tokens,
            req_to_new_blocks,
        )
        scheduled_requests = (scheduled_new_reqs + scheduled_running_reqs +
                              scheduled_resumed_reqs)
        structured_output_request_ids, grammar_bitmask = self.get_grammar_bitmask(
            scheduled_requests, scheduled_spec_decode_tokens)
        assigned_dp_rank = {
            req.request_id: self.assigned_dp_rank[req.request_id]
            for req in scheduled_requests
        }

        scheduler_output = DPSchedulerOutput(
            scheduled_new_reqs=new_reqs_data,
            scheduled_cached_reqs=cached_reqs_data,
            num_scheduled_tokens=num_scheduled_tokens,
            total_num_scheduled_tokens=total_num_scheduled_tokens,
            scheduled_spec_decode_tokens=scheduled_spec_decode_tokens,
            scheduled_encoder_inputs=scheduled_encoder_inputs,
            num_common_prefix_blocks=num_common_prefix_blocks,
            # finished_req_ids is an existing state in the scheduler,
            # instead of being newly scheduled in this step.
            # It contains the request IDs that are finished in between
            # the previous and the current steps.
            finished_req_ids=self.finished_req_ids,
            free_encoder_mm_hashes=self.encoder_cache_manager.
            get_freed_mm_hashes(),
            structured_output_request_ids=structured_output_request_ids,
            grammar_bitmask=grammar_bitmask,
            assigned_dp_rank=assigned_dp_rank,
        )

        # NOTE(Kuntai): this function is designed for multiple purposes:
        # 1. Plan the KV cache store
        # 2. Wrap up all the KV cache load / save ops into an opaque object
        # 3. Clear the internal states of the connector
        if self.connector is not None:
            meta = self.connector.build_connector_meta(scheduler_output)
            scheduler_output.kv_connector_metadata = meta

        # collect KV cache events from KV cache manager
        events = []
        for dp_rank in range(self.dp_size):
            events.extend(self.kv_cache_managers[dp_rank].take_events())

        # collect KV cache events from connector
        if self.connector is not None:
            connector_events = self.connector.take_events()
            if connector_events:
                if events is None:
                    events = list(connector_events)
                else:
                    events.extend(connector_events)

        # publish collected KV cache events
        if events:
            batch = KVEventBatch(ts=time.time(), events=events)
            self.kv_event_publisher.publish(batch)

        self._update_after_schedule(scheduler_output)
        return scheduler_output

    def _update_after_schedule(
        self,
        scheduler_output: SchedulerOutput,
    ) -> None:
        # Advance the number of computed tokens for the request AFTER
        # the request is scheduled.
        # 1. The scheduler_output of the current step has to include the
        #    original number of scheduled tokens to determine input IDs.
        # 2. Advance the number of computed tokens here allowing us to
        #    schedule the prefill request again immediately in the next
        #    scheduling step.
        # 3. If some tokens (e.g. spec tokens) are rejected later, the number of
        #    computed tokens will be adjusted in update_from_output.
        num_scheduled_tokens = scheduler_output.num_scheduled_tokens
        for req_id, num_scheduled_token in num_scheduled_tokens.items():
            request = self.requests[req_id]
            request.num_computed_tokens += num_scheduled_token

            # NOTE: _free_encoder_inputs relies on num_computed_tokens, which
            # may be updated again in _update_from_output for speculative
            # decoding. However, it is safe to call the method here because
            # encoder inputs are always part of the prompt, not the output,
            # and thus are unaffected by speculative decoding.
            if request.has_encoder_inputs:
                self._free_encoder_inputs(request)

        # Clear the finished request IDs.
        # NOTE: We shouldn't do self.finished_req_ids.clear() here because
        # it will also affect the scheduler output.
        self.finished_req_ids = set()
    
    def _free_blocks(self, request: Request):
        assert request.is_finished()
        req_dp_rank = self.assigned_dp_rank.get(request.request_id)
        self.kv_cache_manager[req_dp_rank].free(request)
        del self.requests[request.request_id]

    def reset_prefix_cache(self) -> bool:
        for kv_cache_mgr in self.kv_cache_manager:
            if not kv_cache_mgr.reset_prefix_cache():
                return False
        return True
    
    def make_stats(
        self,
        spec_decoding_stats: Optional[SpecDecodingStats] = None,
        kv_connector_stats: Optional[KVConnectorStats] = None,
    ) -> Optional[SchedulerStats]:
        if not self.log_stats:
            return None
        combined_prefix_cache_stats = PrefixCacheStats()
        for kv_cache_mgr in self.kv_cache_manager:
            prefix_cache_stats = kv_cache_mgr.make_prefix_cache_stats()
            assert prefix_cache_stats is not None
            combined_prefix_cache_stats.reset = prefix_cache_stats.reset
            combined_prefix_cache_stats.requests += (
                prefix_cache_stats.requests)
            combined_prefix_cache_stats.queries += prefix_cache_stats.queries
            combined_prefix_cache_stats.hits += (
                prefix_cache_stats.hits)
            
        return SchedulerStats(num_running_reqs=len(self.running),
                            num_waiting_reqs=len(self.waiting),
                            kv_cache_usage=kv_cache_mgr.usage,
                            prefix_cache_stats=combined_prefix_cache_stats,
                            spec_decoding_stats=spec_decoding_stats,
                            num_corrupted_reqs=sum(req.is_output_corrupted
                                                    for req in self.running),
                            kv_connector_stats=kv_connector_stats.data
                            if kv_connector_stats else None)
