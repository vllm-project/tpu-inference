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

    This function applies three patches:
    1. patched_init: Forces KVCacheManager to reserve enough KV cache blocks for
       all N tokens during schedule() (num_lookahead_tokens = max_decode_steps - 1).
    2. patched_update_base: Reconciles request.num_computed_tokens on host by
       adding the extra (N - 1) tokens generated on-device once model output returns.
    3. patched_async_update_request_with_output: Pre-compensates in-flight
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
            from vllm.v1.request import RequestStatus

            status_before_update = getattr(request, "status", None)
            # Original update appends new_token_ids to request output and trims on stop token.
            res_token_ids, stopped = original_update_base(scheduler_self,
                                                          request,
                                                          new_token_ids,
                                                          is_stale=is_stale,
                                                          **kwargs)

            # In synchronous continue-decode, _process_continue_decode_outputs
            # rewrites scheduler_output.num_scheduled_tokens[req_id] from 1 to
            # actual_len, so Scheduler.update_from_output subtracts actual_len
            # from num_in_flight_tokens (underflowing it to 1 - actual_len).
            # Clamp any negative underflow back to 0 so subsequent
            # remove_skipped_blocks calls do not inflate processed_computed_tokens.
            num_in_flight = getattr(request, "num_in_flight_tokens", None)
            if isinstance(num_in_flight, int) and num_in_flight < 0:
                request.num_in_flight_tokens = 0

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
                    # Pre-compensate by the trimmed (N - 1) tokens right before
                    # AsyncScheduler._update_request_with_output subtracts
                    # len(new_token_ids) (which is len(res_token_ids) after
                    # check_stop trimming).
                    request.num_output_placeholders += diff
                elif status_before_update == RequestStatus.RUNNING:
                    # In synchronous Scheduler, cache any blocks completed during
                    # the continue-decode burst now (before _free_request on EOS
                    # or remove_skipped_blocks on the next schedule() call).
                    kv_cache_manager = getattr(scheduler_self, "__dict__",
                                               {}).get("kv_cache_manager")
                    if kv_cache_manager is not None:
                        num_tokens = getattr(request, "num_tokens", None)
                        in_flight = getattr(request, "num_in_flight_tokens", 0)
                        if not isinstance(in_flight, int):
                            in_flight = 0
                        computed = max(0,
                                       request.num_computed_tokens - in_flight)
                        num_tokens_to_cache = (min(
                            computed, num_tokens) if isinstance(
                                num_tokens, int) else computed)
                        kv_cache_manager.cache_blocks(request,
                                                      num_tokens_to_cache)

            return res_token_ids, stopped

        Scheduler._update_request_with_output = patched_update_base

        original_init = Scheduler.__init__

        def patched_init(scheduler_self, vllm_config, *args, **kwargs):
            original_init(scheduler_self, vllm_config, *args, **kwargs)

            additional_config = getattr(vllm_config, "additional_config", {})
            max_decode_steps = additional_config.get("max_decode_steps",
                                                     DEFAULT_MAX_DECODE_STEPS)
            # Reserve lookahead tokens so KVCacheManager allocates sufficient
            # blocks before execution on TPU. In AsyncScheduler, one continue-decode
            # step may already be in flight while the next step is being scheduled,
            # so up to 2 * (max_decode_steps - 1) extra tokens can be generated
            # beyond the host's scheduled token count.
            from vllm.v1.core.sched.async_scheduler import AsyncScheduler
            lookahead_tokens = (2 * (max_decode_steps - 1) if isinstance(
                scheduler_self, AsyncScheduler) else max_decode_steps - 1)
            scheduler_self.num_lookahead_tokens = max(
                scheduler_self.num_lookahead_tokens, lookahead_tokens)

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
            # patched base can't see it as a parameter. Thread it (and the async
            # update marker so placeholder pre-compensation uses the post-stop
            # trimmed length) through instance flags for the duration of this call.
            scheduler_self._cd_stale_in_flight = is_stale
            scheduler_self._cd_in_async_update = True
            try:
                return original_async_update_req(scheduler_self,
                                                 request,
                                                 new_token_ids,
                                                 is_stale=is_stale,
                                                 **kwargs)
            finally:
                scheduler_self._cd_stale_in_flight = False
                scheduler_self._cd_in_async_update = False

        AsyncScheduler._update_request_with_output = patched_async_update_request_with_output
        AsyncScheduler._continue_decode_patched = True

    Scheduler._continue_decode_patched = True
