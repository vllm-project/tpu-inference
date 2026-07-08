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
    # Monkeypatch vLLM Scheduler to support continue decode multi-step scheduling
    from vllm.v1.core.sched.scheduler import Scheduler

    # Avoid patching multiple times
    if not getattr(Scheduler, "_continue_decode_patched", False):
        original_update_base = Scheduler._update_request_with_output

        def patched_update_base(scheduler_self, request, new_token_ids):
            # Call original first (which trims new_token_ids in-place if stopped)
            res_token_ids, stopped = original_update_base(
                scheduler_self, request, new_token_ids)

            # Update num_computed_tokens using the trimmed token length
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
            # We need max_decode_steps - 1 lookahead tokens to ensure we have enough blocks.
            scheduler_self.num_lookahead_tokens = max(
                scheduler_self.num_lookahead_tokens, max_decode_steps - 1)

        Scheduler.__init__ = patched_init

    # Scheduler.update_from_output does not need to be overridden as AsyncScheduler._update_after_schedule
    # and modify_prev_results_continue_decode handle the continue-decode token count lifecycle correctly.

    from vllm.v1.core.sched.async_scheduler import AsyncScheduler

    if not getattr(AsyncScheduler, "_continue_decode_patched", False):
        original_async_update_req = AsyncScheduler._update_request_with_output

        def patched_async_update_request_with_output(scheduler_self, request,
                                                     new_token_ids):
            if len(new_token_ids) > 1:
                # In continue-decode, 1 scheduling step generated N tokens on device.
                # AsyncScheduler._update_request_with_output subtracts len(new_token_ids) from num_output_placeholders.
                # We add back (len(new_token_ids) - 1) so that exactly 1 placeholder step is decremented.
                request.num_output_placeholders += (len(new_token_ids) - 1)
            return original_async_update_req(scheduler_self, request,
                                             new_token_ids)

        AsyncScheduler._update_request_with_output = patched_async_update_request_with_output
        AsyncScheduler._continue_decode_patched = True

    Scheduler._continue_decode_patched = True
