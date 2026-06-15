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
import vllm.model_executor.kernels.mhc as mhc_kernels
from vllm.model_executor.layers.mhc import (HCHeadOp, MHCFusedPostPreOp,
                                            MHCPostOp, MHCPreOp)

from tpu_inference.logger import init_logger

logger = init_logger(__name__)


@MHCPreOp.register_oot
class VllmMHCPreOp(MHCPreOp):

    @classmethod
    def enabled(cls) -> bool:
        return True

    def forward_tpu(
        self,
        residual: torch.Tensor,
        fn: torch.Tensor,
        hc_scale: torch.Tensor,
        hc_base: torch.Tensor,
        rms_eps: float,
        hc_pre_eps: float,
        hc_sinkhorn_eps: float,
        hc_post_mult_value: float,
        sinkhorn_repeat: int,
        n_splits: int = 1,
        norm_weight: torch.Tensor | None = None,
        norm_eps: float = 0.0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return mhc_kernels.mhc_pre_torch(
            residual,
            fn,
            hc_scale,
            hc_base,
            rms_eps,
            hc_pre_eps,
            hc_sinkhorn_eps,
            hc_post_mult_value,
            sinkhorn_repeat,
        )


@MHCPostOp.register_oot
class VllmMHCPostOp(MHCPostOp):

    @classmethod
    def enabled(cls) -> bool:
        return True

    def forward_tpu(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
        post_layer_mix: torch.Tensor,
        comb_res_mix: torch.Tensor,
    ) -> torch.Tensor:
        return mhc_kernels.mhc_post_torch(
            x,
            residual,
            post_layer_mix,
            comb_res_mix,
        )


@HCHeadOp.register_oot
class VllmHCHeadOp(HCHeadOp):

    @classmethod
    def enabled(cls) -> bool:
        return True

    def forward_tpu(
        self,
        hidden_states: torch.Tensor,
        hc_fn: torch.Tensor,
        hc_scale: torch.Tensor,
        hc_base: torch.Tensor,
        rms_norm_eps: float,
        hc_eps: float,
    ) -> torch.Tensor:
        
        hc_mult, hidden_size = hidden_states.shape[-2:]
        outer_shape = hidden_states.shape[:-2]
        residual = hidden_states.view(-1, hc_mult, hidden_size)

        residual_flat = residual.flatten(-2).float()
        residual_norm = residual_flat * torch.rsqrt(
            residual_flat.square().mean(dim=-1, keepdim=True) + rms_norm_eps
        )
        pre_mix = torch.nn.functional.linear(residual_norm, hc_fn)
        pre_mix = torch.sigmoid(pre_mix * hc_scale + hc_base) + hc_eps
        out = torch.sum(pre_mix.unsqueeze(-1) * residual.float(), dim=-2).bfloat16()
        return out.view(*outer_shape, hidden_size)


@MHCFusedPostPreOp.register_oot
class VllmMHCFusedPostPreOp(MHCFusedPostPreOp):

    @classmethod
    def enabled(cls) -> bool:
        return True

    def forward_tpu(self, *args, **kwargs):
        raise NotImplementedError(
            "Native implementation of mhc_fused_post_pre is not available")
