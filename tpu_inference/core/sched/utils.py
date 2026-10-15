# Copyright 2026 Google LLC
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

DEFAULT_MAX_DECODE_STEPS = 10


def patch_vllm_scheduler_for_continue_decode():
    """Monkeypatches vLLM's Scheduler and AsyncScheduler for Continue Decode.

    In Continue Decode, the host schedules 1 step while the TPU runner executes
    up to max_decode_steps (N) decode iterations on-device in a single step.

    This function applies four patches:
    1. patched_init: Forces KVCacheManager to reserve enough KV cache blocks for
       all N tokens during schedule() (num_lookahead_tokens = max_decode_steps - 1).
    2. patched_update_base: Reconciles request.num_computed_tokens on host by
       adding the extra (N - 1) tokens generated on-device once model output returns.
    3. patched_update_from_output: When routed_experts are returned, temporarily
       sets scheduler_output.num_scheduled_tokens[req_id] to the sampled token
       count (and pre-compensates num_in_flight_tokens / num_stale_output_tokens)
       so vLLM's _extract_routed_experts slices all N tokens, then restores
       num_scheduled_tokens in a finally block.
    4. patched_async_update_request_with_output: Pre-compensates in-flight
       num_output_placeholders in AsyncScheduler by adding (N - 1) before
       subtracting N, preventing placeholder underflow in async mode.
    """
    from vllm.v1.core.sched.scheduler import Scheduler

    # Avoid patching multiple times
    if not getattr(Scheduler, "_continue_decode_patched", False):
        original_update_base = Scheduler._update_request_with_output

        def patched_update_base(scheduler_self,
                                request,
                                new_token_ids,
                                is_stale=False,
                                **kwargs):
            # Original update appends new_token_ids to request output and trims on stop token.
            res_token_ids, stopped = original_update_base(scheduler_self,
                                                          request,
                                                          new_token_ids,
                                                          is_stale=is_stale,
                                                          **kwargs)

            # schedule() only incremented num_computed_tokens by 1. Advance by the remaining
            # (N - 1) tokens generated on-device so host-side num_computed_tokens is accurate.
            # A stale delivery predates the preemption rollback of
            # num_computed_tokens and must not advance it (mirrors the
            # `if not output_is_stale` guards in vLLM's update_from_output).
            # AsyncScheduler's original calls super() WITHOUT forwarding
            # is_stale, so the async wrapper threads staleness through
            # _cd_stale_in_flight instead — check both.
            stale = is_stale or getattr(scheduler_self, "_cd_stale_in_flight",
                                        False)
            diff = len(res_token_ids) - 1
            if diff > 0 and not stale:
                request.num_computed_tokens += diff
                if getattr(scheduler_self, "_cd_in_async_update", False):
                    # In AsyncScheduler, _update_after_schedule() added 1 in-flight placeholder token.
                    # original_async_update_req subtracts len(res_token_ids) (after stop-token
                    # trimming) from num_output_placeholders. Pre-compensate by adding diff here
                    # using the trimmed res_token_ids so num_output_placeholders decrements by 1.
                    request.num_output_placeholders += diff

            return res_token_ids, stopped

        Scheduler._update_request_with_output = patched_update_base

        original_update_from_output = Scheduler.update_from_output

        def patched_update_from_output(scheduler_self, scheduler_output,
                                       model_runner_output, *args, **kwargs):
            orig_num_scheduled = {}
            sampled_token_counts = {}
            routing_offsets = {}
            if (getattr(model_runner_output, "routed_experts", None)
                    is not None and model_runner_output.sampled_token_ids):
                cached_data = getattr(scheduler_output,
                                      "scheduled_cached_reqs", None)
                num_output_tokens_dict = (dict(
                    zip(cached_data.req_ids, cached_data.num_output_tokens))
                                          if cached_data is not None else {})
                spec_decode_tokens = getattr(scheduler_output,
                                             "scheduled_spec_decode_tokens",
                                             None) or {}
                for req_id, req_idx in model_runner_output.req_id_to_index.items(
                ):
                    if (num_output_tokens_dict.get(req_id, 0) > 0
                            and not spec_decode_tokens.get(req_id)):
                        num_sampled = len(
                            model_runner_output.sampled_token_ids[req_idx])
                        orig = scheduler_output.num_scheduled_tokens.get(
                            req_id)
                        if orig is not None and num_sampled > orig:
                            orig_num_scheduled[req_id] = orig
                            sampled_token_counts[req_id] = num_sampled
                            scheduler_output.num_scheduled_tokens[
                                req_id] = num_sampled
                            extra = num_sampled - orig
                            request = scheduler_self.requests.get(req_id)
                            if request is not None:
                                request.num_in_flight_tokens += extra
                                if request.num_stale_output_tokens > 0:
                                    request.num_stale_output_tokens += extra
                if orig_num_scheduled:
                    offset = 0
                    for req_id in (getattr(model_runner_output, "req_ids",
                                           None)
                                   or model_runner_output.req_id_to_index):
                        routing_offsets[req_id] = offset
                        offset += scheduler_output.num_scheduled_tokens.get(
                            req_id, 0)
            try:
                outputs = original_update_from_output(scheduler_self,
                                                      scheduler_output,
                                                      model_runner_output,
                                                      *args, **kwargs)
                if (orig_num_scheduled and getattr(
                        scheduler_self, "enable_return_routed_experts", False)
                        and isinstance(outputs, dict)):
                    routing_data = getattr(model_runner_output.routed_experts,
                                           "routing_data", None)
                    if routing_data is not None:
                        for engine_outputs in outputs.values():
                            for out in getattr(engine_outputs, "outputs", ()):
                                req_id = out.request_id
                                if (req_id in orig_num_scheduled
                                        and out.routed_experts is not None):
                                    num_sampled = sampled_token_counts.get(
                                        req_id, 0)
                                    num_kept = len(out.new_token_ids)
                                    if 0 < num_kept < num_sampled:
                                        req_offset = routing_offsets[req_id]
                                        out.routed_experts = routing_data[
                                            req_offset:req_offset +
                                            num_kept].astype(
                                                out.routed_experts.dtype,
                                                copy=False)
                return outputs
            finally:
                if orig_num_scheduled:
                    scheduler_output.num_scheduled_tokens.update(
                        orig_num_scheduled)

        Scheduler.update_from_output = patched_update_from_output

        original_init = Scheduler.__init__

        def patched_init(scheduler_self, vllm_config, *args, **kwargs):
            original_init(scheduler_self, vllm_config, *args, **kwargs)

            additional_config = getattr(vllm_config, "additional_config", {})
            max_decode_steps = additional_config.get("max_decode_steps",
                                                     DEFAULT_MAX_DECODE_STEPS)
            # Reserve max_decode_steps - 1 lookahead tokens so KVCacheManager allocates
            # sufficient blocks for up to max_decode_steps tokens before execution on TPU.
            # Under async_scheduling, step T+1 is scheduled while step T's burst is still
            # in flight, so reserve 2 * (max_decode_steps - 1) lookahead tokens.
            is_async = getattr(
                getattr(vllm_config, "scheduler_config", None),
                "async_scheduling",
                False,
            )
            lookahead = (2 * (max_decode_steps - 1) if is_async else
                         (max_decode_steps - 1))
            scheduler_self.num_lookahead_tokens = max(
                scheduler_self.num_lookahead_tokens, lookahead)

        Scheduler.__init__ = patched_init

    from vllm.v1.core.sched.async_scheduler import AsyncScheduler

    if not getattr(AsyncScheduler, "_continue_decode_patched", False):
        original_async_update_req = AsyncScheduler._update_request_with_output

        def patched_async_update_request_with_output(scheduler_self,
                                                     request,
                                                     new_token_ids,
                                                     is_stale=False,
                                                     **kwargs):
            # The original's super() call does not forward is_stale, so the
            # patched base can't see it as a parameter. Thread it (and the
            # async-update marker) through instance flags for this call.
            scheduler_self._cd_in_async_update = True
            scheduler_self._cd_stale_in_flight = is_stale
            try:
                return original_async_update_req(scheduler_self,
                                                 request,
                                                 new_token_ids,
                                                 is_stale=is_stale,
                                                 **kwargs)
            finally:
                scheduler_self._cd_in_async_update = False
                scheduler_self._cd_stale_in_flight = False

        AsyncScheduler._update_request_with_output = patched_async_update_request_with_output
        AsyncScheduler._continue_decode_patched = True

    Scheduler._continue_decode_patched = True
