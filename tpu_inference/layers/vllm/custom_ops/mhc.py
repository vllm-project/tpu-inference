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
"""TPU overrides for DeepSeek-V4 Multi-Head Compression (MHC) CustomOps.

GPU paths (vllm/vllm/model_executor/kernels/mhc/tilelang.py) use tilelang /
DeepGEMM CUDA kernels. On TPU we delegate to the pure-PyTorch reference
implementations in vllm/vllm/model_executor/kernels/mhc/torch.py.

"""

import torch
import torch.nn.functional as F
from vllm.model_executor.kernels.mhc.torch import mhc_post_torch, mhc_pre_torch
from vllm.model_executor.layers.mhc import (
    HCHeadOp,
    MHCFusedPostPreOp,
    MHCPostOp,
    MHCPreOp,
)


@MHCPreOp.register_oot
class TpuMHCPreOp(MHCPreOp):
    """TPU override for the MHC pre block.

    GPU kernel implementation: vllm/vllm/model_executor/kernels/mhc/tilelang.py

    Reference torch implementation: vllm/vllm/model_executor/kernels/mhc/torch.py
    """

    def forward_tpu(
        self,
        residual: torch.Tensor, # [T, hc_mult, D]
        fn: torch.Tensor,  # [hc_mult3, hc_mult*D]
        hc_scale: torch.Tensor, # [3]
        hc_base: torch.Tensor, # [hc_mult3]
        rms_eps: float,
        hc_pre_eps: float,
        hc_sinkhorn_eps: float,
        hc_post_mult_value: float,
        sinkhorn_repeat: int,
        n_splits: int = 1,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Returns tuple:
        # (post_mix [T, hc_mult, 1], 
        #  comb_mix [T, hc_mult, hc_mult],
        #  layer_input [T, D])
        return mhc_pre_torch(
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
class TpuMHCPostOp(MHCPostOp):
    """TPU override for the MHC post block.

    GPU kernel implementation: vllm/vllm/model_executor/kernels/mhc/tilelang.py

    Reference torch implementation: vllm/vllm/model_executor/kernels/mhc/torch.py
    """

    def forward_tpu(
        self,
        x: torch.Tensor, # [T, D]
        residual: torch.Tensor, # [T, hc_mult, D]
        post_layer_mix: torch.Tensor, # [T, hc_mult, 1]
        comb_res_mix: torch.Tensor, # [T, hc_mult, hc_mult]
    ) -> torch.Tensor: # [T, hc_mult, D]
        return mhc_post_torch(x, residual, post_layer_mix, comb_res_mix)


@HCHeadOp.register_oot
class TpuHCHeadOp(HCHeadOp):
    """TPU override for the HC head reduction (applied once after the final layer).

    Collapses the multi-head residual [T, hc_mult, D] → [T, D] by weighting
    each head with a sigmoid gate, then summing.

    GPU kernel implementation: vllm/vllm/model_executor/kernels/mhc/tilelang.py

    Reference torch implementation: vllm/vllm/model_executor/kernels/mhc/torch.py called by
    vllm/vllm/model_executor/kernels/mhc/triton.py::hc_head_reduce_triton_kernel
    """

    def forward_tpu(
        self,
        hidden_states: torch.Tensor, # [T, hc_mult, D]
        hc_fn: torch.Tensor, # [hc_mult3, hc_mult*D]
        hc_scale: torch.Tensor, # [3]
        hc_base: torch.Tensor, # [hc_mult3]
        rms_norm_eps: float,
        hc_eps: float,
    ) -> torch.Tensor: # [T, D]
        hc_mult = hidden_states.shape[-2]
        outer_shape = hidden_states.shape[:-2]

        x_flat = hidden_states.flatten(-2).float()  # [..., hc_mult*D]
        # RMSnorm
        rsqrt = torch.rsqrt(x_flat.square().mean(-1, keepdim=True) + rms_norm_eps)
        x_norm = x_flat * rsqrt
        # Projection
        mixes = F.linear(x_norm, hc_fn)  # [..., hc_mult3]
        
        # Use the "pre" slice as head gates.
        gates = (
            torch.sigmoid(mixes[..., :hc_mult] * hc_scale[0] + hc_base[:hc_mult])
            + hc_eps
        )  # [..., hc_mult]
        
        # Weighted sum over heads
        y = torch.sum(
            gates.unsqueeze(-1) * hidden_states.float(), dim=-2
        )  # [..., D]
        
        # Unpack to expected hidden_states shape
        return y.to(hidden_states.dtype).view(*outer_shape, hidden_states.shape[-1])


@MHCFusedPostPreOp.register_oot
class TpuMHCFusedPostPreOp(MHCFusedPostPreOp):
    """TPU override for the fused MHC post+pre op (used between consecutive layers).

    GPU kernel version: torch.ops.vllm.mhc_fused_post_pre_tilelang

    TPU leverage mhc_post_torch and mhc_pre_torch from: vllm/vllm/model_executor/kernels/mhc/torch.py
    """

    def forward_tpu(
        self,
        x: torch.Tensor, # [T, D]
        residual: torch.Tensor, # [T, hc_mult, D]
        post_layer_mix: torch.Tensor, # [T, hc_mult, 1]
        comb_res_mix: torch.Tensor, # [T, hc_mult, hc_mult]
        fn: torch.Tensor, # [hc_mult3, hc_mult*D]
        hc_scale: torch.Tensor, # [3]
        hc_base: torch.Tensor, # [hc_mult3]
        rms_eps: float,
        hc_pre_eps: float,
        hc_sinkhorn_eps: float,
        hc_post_mult_value: float,
        sinkhorn_repeat: int,
        n_splits: int = 1,
        tile_n: int = 1,

        # Returns tuple:
        # (new_residual [T, hc_mult, D],
        #  post_mix [T, hc_mult, 1],
        #  comb_mix [T, hc_mult, hc_mult],
        #  layer_input [T, D])
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        new_residual = mhc_post_torch(x, residual, post_layer_mix, comb_res_mix)
        post_mix, comb_mix, layer_input = mhc_pre_torch(
            new_residual,
            fn,
            hc_scale,
            hc_base,
            rms_eps,
            hc_pre_eps,
            hc_sinkhorn_eps,
            hc_post_mult_value,
            sinkhorn_repeat,
        )
        return new_residual, post_mix, comb_mix, layer_input