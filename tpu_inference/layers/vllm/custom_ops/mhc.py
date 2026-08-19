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

import jax
import torch
import torch.nn.functional as F
from jax.sharding import PartitionSpec as P
from torchax.interop import jax_view, torch_view
from vllm.model_executor.layers.mhc import (HCHeadOp, MHCFusedPostPreOp,
                                            MHCPostOp, MHCPreOp)

from tpu_inference.kernels.experimental.deepseek_v4.mhc import (
    fused_post_pre_kernel, post_kernel, pre_kernel)
from tpu_inference.layers.common.sharding import ShardingAxisName
from tpu_inference.logger import init_logger
from tpu_inference.models.vllm.vllm_model_wrapper_context import \
    get_vllm_model_wrapper_context

logger = init_logger(__name__)


def _sharded_kernel(fn, n_token_args, n_token_outs):
    """Wrap a Pallas mHC kernel call for SPMD execution.

    Mosaic custom calls cannot be auto-partitioned by GSPMD; in the
    sharded model graph they must run per-device on local shards via
    shard_map. Every mHC tensor is token-major, so the first
    ``n_token_args`` inputs and all outputs shard on the attention-DP
    token axis; trailing inputs (fn / hc_scale / hc_base) replicate.
    Falls back to a direct call outside the model wrapper context
    (single-device tests, standalone use).
    """
    try:
        ctx = get_vllm_model_wrapper_context()
    except AssertionError:
        # Outside the model wrapper (single-device tests, standalone use).
        return fn
    if ctx is None or getattr(ctx, "mesh", None) is None:
        return fn
    data = P(ShardingAxisName.ATTN_DATA)

    def wrapped(*args):
        in_specs = tuple(data for _ in range(n_token_args)) + tuple(
            P() for _ in range(len(args) - n_token_args))
        out_specs = tuple(data for _ in range(n_token_outs))
        return jax.shard_map(fn,
                             mesh=ctx.mesh,
                             in_specs=in_specs,
                             out_specs=out_specs if n_token_outs > 1 else data,
                             check_vma=False)(*args)

    return wrapped


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
        assert norm_weight is None, (
            "the TPU Pallas mHC path does not support fused RMSNorm")

        # Only the chain endpoint reaches this op: the layer-0 pre, which
        # has no preceding post to fuse with.
        def _pre(residual_jax, fn_jax, hc_scale_jax, hc_base_jax):
            return pre_kernel.mhc_pre(residual_jax, fn_jax, hc_scale_jax,
                                      hc_base_jax, rms_eps, hc_pre_eps,
                                      hc_sinkhorn_eps, hc_post_mult_value,
                                      sinkhorn_repeat)

        post_mix, comb_mix, layer_input = _sharded_kernel(_pre, 1, 3)(
            jax_view(residual), jax_view(fn), jax_view(hc_scale),
            jax_view(hc_base))
        return (torch_view(post_mix), torch_view(comb_mix),
                torch_view(layer_input))


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
        # Only the chain endpoint reaches this op.
        post_fn = _sharded_kernel(post_kernel.mhc_post, 4, 1)
        return torch_view(
            post_fn(jax_view(x), jax_view(residual), jax_view(post_layer_mix),
                    jax_view(comb_res_mix)))


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

    def forward_tpu(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
        post_layer_mix: torch.Tensor,
        comb_res_mix: torch.Tensor,
        fn: torch.Tensor,
        hc_scale: torch.Tensor,
        hc_base: torch.Tensor,
        rms_eps: float,
        hc_pre_eps: float,
        hc_sinkhorn_eps: float,
        hc_post_mult_value: float,
        sinkhorn_repeat: int,
        n_splits: int = 1,
        tile_n: int = 1,
        norm_weight: torch.Tensor | None = None,
        norm_eps: float = 0.0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        assert norm_weight is None, (
            "the TPU fused mHC path does not support fused RMSNorm")

        # Takes the jax_view'd arrays (what shard_map maps over) and
        # captures the eps scalars, which are static kernel parameters
        # rather than traced arrays.
        def _fused(x_jax, residual_jax, post_layer_mix_jax, comb_res_mix_jax,
                   fn_jax, hc_scale_jax, hc_base_jax):
            return fused_post_pre_kernel.mhc_fused_post_pre(
                x_jax, residual_jax, post_layer_mix_jax, comb_res_mix_jax,
                fn_jax, hc_scale_jax, hc_base_jax, rms_eps, hc_pre_eps,
                hc_sinkhorn_eps, hc_post_mult_value, sinkhorn_repeat)

        residual_cur, post_mix_cur, comb_mix_cur, layer_input_cur = (
            _sharded_kernel(_fused, 4, 4)(
                jax_view(x),
                jax_view(residual),
                jax_view(post_layer_mix),
                jax_view(comb_res_mix),
                jax_view(fn),
                jax_view(hc_scale),
                jax_view(hc_base),
            ))
        return (torch_view(residual_cur), torch_view(post_mix_cur),
                torch_view(comb_mix_cur), torch_view(layer_input_cur))
