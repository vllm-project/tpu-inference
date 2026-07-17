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

        def patched_update_base(scheduler_self, request, new_token_ids):
            # Original update appends new_token_ids to request output and trims on stop token.
            res_token_ids, stopped = original_update_base(
                scheduler_self, request, new_token_ids)

            # schedule() only incremented num_computed_tokens by 1. Advance by the remaining
            # (N - 1) tokens generated on-device so host-side num_computed_tokens is accurate.
            diff = len(res_token_ids) - 1
            if diff > 0:
                request.num_computed_tokens += diff

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
            scheduler_self.num_lookahead_tokens = max(
                scheduler_self.num_lookahead_tokens, max_decode_steps - 1)

        Scheduler.__init__ = patched_init

    from vllm.v1.core.sched.async_scheduler import AsyncScheduler

    if not getattr(AsyncScheduler, "_continue_decode_patched", False):
        original_async_update_req = AsyncScheduler._update_request_with_output

        def patched_async_update_request_with_output(scheduler_self, request,
                                                     new_token_ids):
            if len(new_token_ids) > 1:
                # In AsyncScheduler, _update_after_schedule() added 1 in-flight placeholder token.
                # When N tokens return, original_async_update_req will subtract N from
                # num_output_placeholders. Pre-compensate by adding (N - 1) first so that
                # num_output_placeholders cleanly decrements by 1 without underflowing < 0.
                request.num_output_placeholders += (len(new_token_ids) - 1)
            return original_async_update_req(scheduler_self, request,
                                             new_token_ids)

        AsyncScheduler._update_request_with_output = patched_async_update_request_with_output
        AsyncScheduler._continue_decode_patched = True

    Scheduler._continue_decode_patched = True
