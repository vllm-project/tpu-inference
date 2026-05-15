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
from vllm.model_executor.layers.fused_moe import FusedMoE


@FusedMoE.register_oot
class VllmFusedMoE(FusedMoE):

    def forward(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        # This plugin is needed to pass through Hash MoE layers'
        # input_ids through the **kwargs argument.
        # The actual hash routing logicand precomputed topk_ids) lives in three other files:
        #   1. fused_moe_gmm.py — (TODO) support (optional) precomputed_topk_ids in fused_moe_func
        #   2. interface/moe.py — (TODO) thread precomputed_topk_ids through vllm_moe_apply
        #   3. deepseek_v4_fp8.py — detect hash layers in
        #      apply_monolithic, compute hash_table[input_ids], pass to vllm_moe_apply
        return super().forward(hidden_states, router_logits, **kwargs)
