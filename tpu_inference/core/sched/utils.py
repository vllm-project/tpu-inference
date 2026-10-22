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
       all N tokens during schedule() (num_lookahead_tokens = max_decode_steps - 1,
       or 2 * (max_decode_steps - 1) under async_scheduling).
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
            # AsyncScheduler's original may call super() without forwarding
            # is_stale, so check _cd_stale_in_flight as well.
            stale = is_stale or getattr(scheduler_self, "_cd_stale_in_flight",
                                        False)
            computed_before = request.num_computed_tokens
            # Original update appends new_token_ids to request output and trims on stop token.
            res_token_ids, stopped = original_update_base(scheduler_self,
                                                          request,
                                                          new_token_ids,
                                                          is_stale=stale,
                                                          **kwargs)

            # schedule() only incremented num_computed_tokens (and in-flight
            # placeholders under AsyncScheduler) by 1. Reconcile against the
            # trimmed output count, accounting for any stop-trimmed subtraction
            # performed by the base scheduler. A stale delivery predates
            # preemption rollback and must not advance either counter.
            if res_token_ids and not stale:
                delta = (computed_before + len(res_token_ids) -
                         1) - request.num_computed_tokens
                if delta > 0:
                    request.num_computed_tokens += delta
                    if getattr(scheduler_self, "_cd_in_async_update", False):
                        request.num_output_placeholders += delta

            return res_token_ids, stopped

        Scheduler._update_request_with_output = patched_update_base

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
