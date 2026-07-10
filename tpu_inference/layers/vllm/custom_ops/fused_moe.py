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
from vllm.model_executor.layers.fused_moe.runner.moe_runner import MoERunner

from tpu_inference.layers.common.sharding import ShardingAxisName
from tpu_inference.models.vllm.vllm_model_wrapper_context import \
    get_vllm_model_wrapper_context
from tpu_inference.utils import get_mesh_shape_product


@MoERunner.register_oot
class VllmMoERunner(MoERunner):

    def _maybe_reduce_final_output(self, states: torch.Tensor,
                                   trunc_size: int) -> torch.Tensor:
        try:
            context = get_vllm_model_wrapper_context()
            mesh = context.mesh
        except AssertionError:
            mesh = None

        is_dp = False
        if mesh is not None:
            attn_dp_size = get_mesh_shape_product(mesh,
                                                  ShardingAxisName.ATTN_DATA)
            dp_size = get_mesh_shape_product(mesh, ShardingAxisName.MLP_DATA)
            is_dp = (attn_dp_size // dp_size) > 1

        if is_dp:
            return states[..., :trunc_size]

        return super()._maybe_reduce_final_output(states, trunc_size)
