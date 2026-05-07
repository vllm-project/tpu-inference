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


def update_vllm_scheduler_for_exporting_expert_ids():
    # Monkeypatch vLLM Scheduler to attach expert indices to outputs
    from vllm.v1.core.sched.scheduler import Scheduler

    if getattr(Scheduler, "_expert_ids_patched", False):
        return
    Scheduler._expert_ids_patched = True

    from vllm.model_executor.layers.fused_moe.routed_experts_capturer import \
        RoutedExpertsReader

    class DummyRoutedExpertsReader:

        @staticmethod
        def create():
            return DummyRoutedExpertsReader()

        def attach_buffer(self, *args, **kwargs):
            pass

        def get_routed_experts(self, *args, **kwargs):
            return None

    # Since we are reusing the upstream enable_return_routed_experts flag,
    # we need to stub out the actual RoutedExpertsReader class which the
    # upstream scheduler tries to create.
    RoutedExpertsReader.create = DummyRoutedExpertsReader.create

    original_update_from_output = Scheduler.update_from_output

    def custom_update_from_output(self, scheduler_output, model_runner_output):
        expert_indices = getattr(model_runner_output, "expert_indices", None)

        if expert_indices is not None:
            num_scheduled_tokens = scheduler_output.num_scheduled_tokens
            current_token_offset = 0
            for req_id, num_tokens_scheduled in num_scheduled_tokens.items():
                start_idx = current_token_offset
                end_idx = start_idx + num_tokens_scheduled
                current_token_offset = end_idx

                request = self.requests.get(req_id)
                if request is not None:
                    step_experts = expert_indices[:, start_idx:
                                                  end_idx, :].transpose(
                                                      1, 0, 2)
                    if not hasattr(request, "_accumulated_routed_experts"):
                        request._accumulated_routed_experts = []
                    request._accumulated_routed_experts.append(step_experts)

        return original_update_from_output(self, scheduler_output,
                                           model_runner_output)

    Scheduler.update_from_output = custom_update_from_output

    original_get_routed_experts = Scheduler._get_routed_experts

    def custom_get_routed_experts(self, request):
        if hasattr(request, "_accumulated_routed_experts"
                   ) and request._accumulated_routed_experts:
            import numpy as np
            return np.concatenate(request._accumulated_routed_experts, axis=0)
        return original_get_routed_experts(self, request)

    Scheduler._get_routed_experts = custom_get_routed_experts
