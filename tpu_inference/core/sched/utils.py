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


def patch_vllm_scheduler_for_continue_decode(vllm_config):
    # Monkeypatch vLLM Scheduler to support continue decode multi-step scheduling
    from vllm.v1.core.sched.scheduler import Scheduler
    from vllm.v1.request import Request, RequestStatus

    max_decode_steps = vllm_config.additional_config.get(
        "max_decode_steps", 10)

    if not getattr(Scheduler, "_continue_decode_property_patched", False):
        # 1. Monkeypatch Request.num_tokens_with_spec
        original_num_tokens_with_spec = Request.num_tokens_with_spec

        @property
        def hacked_num_tokens_with_spec(self):
            # If the request is running and not in prefill chunk, force scheduling of max_decode_steps tokens
            if self.status == RequestStatus.RUNNING and not self.is_prefill_chunk:
                return len(self._all_token_ids) + max_decode_steps
            return original_num_tokens_with_spec.__get__(self)

        Request.num_tokens_with_spec = hacked_num_tokens_with_spec

        # 2. Monkeypatch Scheduler.update_from_output to rollback state if TPU early terminated
        original_update_from_output = Scheduler.update_from_output

        def custom_update_from_output(self, scheduler_output,
                                      model_runner_output):
            # Read execution step count returned from TPU
            actual_steps = getattr(model_runner_output, "actual_steps", None)

            if actual_steps is not None:
                # Prior to executing standard update_from_output (which reads num_computed_tokens),
                # we correct the placeholder and computed token counts.
                for req_id, scheduled_tokens in scheduler_output.num_scheduled_tokens.items(
                ):
                    request = self.requests.get(req_id)
                    if request is not None:
                        # 1. Dynamic alignment of placeholders for async scheduling
                        # Because the CPU scheduler only incremented placeholders by +1 during scheduling,
                        # but the TPU actually generated actual_steps (e.g., K) tokens,
                        # we must pre-advance placeholders by +(K - 1) so that the standard
                        # update_from_output's -= K subtraction evaluates to exactly 0 without going negative.
                        if request.num_output_placeholders > 0:
                            request.num_output_placeholders += (actual_steps -
                                                                1)

                        # 2. Standard rollback for early termination mismatch (K < M)
                        mismatch = scheduled_tokens - actual_steps
                        if mismatch > 0:
                            request.num_computed_tokens -= mismatch
                            if request.num_output_placeholders > 0:
                                request.num_output_placeholders -= mismatch

            return original_update_from_output(self, scheduler_output,
                                               model_runner_output)

        Scheduler.update_from_output = custom_update_from_output
        Scheduler._continue_decode_property_patched = True
