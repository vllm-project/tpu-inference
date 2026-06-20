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
import torch.nn.functional as F
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
    """TPU implementation of HCHeadOp."""

    @classmethod
    def enabled(cls) -> bool:
        """Returns whether this operation is enabled."""
        return True

    def forward_tpu(
        self,
        hidden_states: torch.Tensor,  # [batch_size, hc_mult, hidden_size]
        hc_fn: torch.Tensor,  # [hc_mult, hc_mult * hidden_size]
        hc_scale: torch.Tensor,  # [hc_mult]
        hc_base: torch.Tensor,  # [hc_mult]
        rms_norm_eps: float,
        hc_eps: float,
    ) -> torch.Tensor:
        """Applies the TPU forward pass for the op."""
        # Using .flatten(start_dim=-2) avoids XLA contiguity RuntimeErrors.
        hs_flat = hidden_states.flatten(start_dim=-2)

        # Upcast to float32 for stable variance computation on TPUs.
        hs_flat_fp32 = hs_flat.float()
        variance = hs_flat_fp32.pow(2).mean(dim=-1, keepdim=True)
        hs_norm = (hs_flat_fp32 * torch.rsqrt(variance + rms_norm_eps)).to(
            hidden_states.dtype)

        # Compute mixing gates, apply scale/base, and calculate sigmoid + epsilon.
        gates = F.linear(hs_norm, hc_fn)
        gates = torch.sigmoid((gates * hc_scale) + hc_base) + hc_eps

        # Collapse multi-stream residuals into 1 stream via weighted sum.
        gates = gates.unsqueeze(-1)
        out = (hidden_states * gates).sum(dim=-2).bfloat16()

        return out


@MHCFusedPostPreOp.register_oot
class VllmMHCFusedPostPreOp(MHCFusedPostPreOp):

    @classmethod
    def enabled(cls) -> bool:
        return True

    def forward_tpu(self, *args, **kwargs):
        raise NotImplementedError(
            "Native implementation of mhc_fused_post_pre is not available")
