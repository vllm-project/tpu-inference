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

import torch
import types
from tpu_inference.models.vllm.vllm_model_wrapper_context import get_vllm_model_wrapper_context
from vllm.model_executor.layers.fused_moe import FusedMoE



def _custom_maybe_reduce_final_output(self, states: torch.Tensor, trunc_size: int) -> torch.Tensor:
    try:
        context = get_vllm_model_wrapper_context()
        vllm_config = context.vllm_config
    except AssertionError:
        vllm_config = None

    is_dp = False
    if vllm_config is not None:
        sharding_strategy = vllm_config.additional_config.get("sharding", {}).get("sharding_strategy", {})
        is_dp = sharding_strategy.get("enable_dp_attention", False)

    if is_dp:
        return states[..., :trunc_size]

    return self._original_maybe_reduce_final_output(states, trunc_size)

@FusedMoE.register_oot
class VllmFusedMoE(FusedMoE):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.runner._original_maybe_reduce_final_output = self.runner._maybe_reduce_final_output
        self.runner._maybe_reduce_final_output = types.MethodType(
            _custom_maybe_reduce_final_output, self.runner
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        return super().forward(hidden_states, router_logits)
