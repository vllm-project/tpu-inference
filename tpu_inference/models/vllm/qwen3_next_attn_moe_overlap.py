# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Qwen3Next and Qwen3.5 attention-to-MoE collective overlap monkey-patch."""

from __future__ import annotations

import logging
import os
from typing import Any

import torch
from vllm.distributed import (
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_gather,
    tensor_model_parallel_all_reduce,
    tensor_model_parallel_reduce_scatter,
)
from vllm.model_executor.models.qwen3_5 import Qwen3_5DecoderLayer
from vllm.model_executor.models.qwen3_next import (
    Qwen3NextDecoderLayer,
    Qwen3NextSparseMoeBlock,
)

import tpu_inference.envs as envs

try:
    from tpu_inference.layers.adapter import input_norm_quant_op
except ImportError:
    input_norm_quant_op = None

logger = logging.getLogger(__name__)

_PATCH_APPLIED = False

# --- QX prefill experiments (env-gated; unset/0 == original behaviour) -------
# QX_MOE_H_FP8_AG: 0 = bf16 AG of h (original); 1 = per-row fp8 quant of h_loc,
#   AG the fp8 payload bitcast to bf16 (same collective dtype as today) plus an
#   f32 per-row scale, dequant to bf16 after the gather; 2 = same but AG fp8
#   natively.
# QX_MOE_ROUTE_LOCAL: 1 = softmax/top-k/renorm on the T/tp local rows and AG a
#   packed f32[T, 2*topk] (weights | ids) instead of the bf16[T, E] logits.
#   Exact (same ops, same order); unpacked in moe_routing.route().
# Both only act when num_tokens >= QX_PREFILL_MIN_TOKENS (default 1024).
_QX_H_FP8_AG = int(os.environ.get("QX_MOE_H_FP8_AG", "0") or "0")
_QX_ROUTE_LOCAL = os.environ.get("QX_MOE_ROUTE_LOCAL", "0").strip().lower() in (
    "1",
    "true",
    "yes",
    "on",
)
_QX_MIN_TOKENS = int(os.environ.get("QX_PREFILL_MIN_TOKENS", "1024") or "1024")
_FP8_MAX = 448.0


def _qx_quant_rows_fp8(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-row symmetric e4m3 quant (amax/448, scale=1 for all-zero rows)."""
    xf = x.float()
    amax = xf.abs().amax(dim=-1, keepdim=True)
    scale = torch.where(amax > 0, amax / _FP8_MAX, torch.ones_like(amax))
    q = (xf / scale).clamp(-_FP8_MAX, _FP8_MAX).to(torch.float8_e4m3fn)
    return q, scale


def _qx_all_gather_h_fp8(h_loc: torch.Tensor, full_num_tokens: int) -> torch.Tensor:
    q, scale = _qx_quant_rows_fp8(h_loc)
    if _QX_H_FP8_AG == 1:
        q_g = tensor_model_parallel_all_gather(q.view(torch.bfloat16), 0).view(
            torch.float8_e4m3fn
        )
    else:
        q_g = tensor_model_parallel_all_gather(q, 0)
    s_g = tensor_model_parallel_all_gather(scale, 0)
    q_g = q_g.narrow(0, 0, full_num_tokens)
    s_g = s_g.narrow(0, 0, full_num_tokens)
    return (q_g.float() * s_g).to(h_loc.dtype)


def _qx_route_local_packed(self: Any, logits_loc: torch.Tensor) -> torch.Tensor:
    """Same math as moe_routing.select_experts (softmax, no bias, no groups)."""
    scores = logits_loc.float().softmax(dim=-1)
    w, ids = torch.topk(scores, self._qx_topk, dim=-1)
    if self._qx_renorm:
        w = w / torch.clamp(w.sum(dim=-1, keepdim=True), min=1e-20)
    return torch.cat([w.float(), ids.float()], dim=-1)
_orig_qwen3_next_decoder_init = Qwen3NextDecoderLayer.__init__
_orig_qwen3_5_decoder_init = Qwen3_5DecoderLayer.__init__
_orig_decoder_forward = Qwen3NextDecoderLayer.forward
_orig_moe_forward = Qwen3NextSparseMoeBlock.forward


def _configure_decoder_layer(self: Any, prefix: str = "") -> None:
    if getattr(self, "_tpu_attn_moe_overlap_configured", False):
        return
    self._tpu_attn_moe_overlap_configured = True

    if (
        "mtp" in prefix
        or not isinstance(getattr(self, "mlp", None), Qwen3NextSparseMoeBlock)
        or get_tensor_model_parallel_world_size() <= 1
    ):
        self._tpu_attn_moe_overlap = False
        return

    assert not getattr(self.mlp, "replicate_shared_expert", False)

    proj = (
        self.linear_attn.out_proj
        if self.layer_type == "linear_attention"
        else self.self_attn.o_proj
    )
    assert getattr(proj, "reduce_results", False) is True
    proj.reduce_results = False

    self.mlp.experts.gate = None
    self.mlp.experts.moe_config.skip_final_all_reduce = True
    assert self.mlp.experts.moe_config.skip_final_all_reduce is True

    if not hasattr(self.mlp.experts, "maybe_all_reduce_tensor_model_parallel"):
        self.mlp.experts.maybe_all_reduce_tensor_model_parallel = lambda x: (
            tensor_model_parallel_all_reduce(x)
        )

    self.mlp._tpu_attn_moe_overlap = True
    self._tpu_attn_moe_overlap = True


def _qx_set_routing_cfg(self: Any, vllm_config: Any) -> None:
    self._qx_route_local = False
    if not (_QX_ROUTE_LOCAL and getattr(self, "_tpu_attn_moe_overlap", False)):
        return
    mc = vllm_config.model_config
    cfg = getattr(mc, "hf_text_config", None) or mc.hf_config
    topk = getattr(cfg, "num_experts_per_tok", None)
    if topk is None or getattr(cfg, "scoring_func", "softmax") != "softmax":
        return
    self._qx_topk = int(topk)
    self._qx_renorm = bool(getattr(cfg, "norm_topk_prob", True))
    self._qx_route_local = True


def _patched_qwen3_next_decoder_init(
    self: Any,
    vllm_config: Any,
    layer_type: str,
    prefix: str = "",
) -> None:
    _orig_qwen3_next_decoder_init(self, vllm_config, layer_type, prefix=prefix)
    _configure_decoder_layer(self, prefix)
    _qx_set_routing_cfg(self, vllm_config)


def _patched_qwen3_5_decoder_init(
    self: Any,
    vllm_config: Any,
    layer_type: str,
    prefix: str = "",
) -> None:
    _orig_qwen3_5_decoder_init(self, vllm_config, layer_type, prefix=prefix)
    _configure_decoder_layer(self, prefix)
    _qx_set_routing_cfg(self, vllm_config)


def _patched_moe_forward(
    self: Qwen3NextSparseMoeBlock,
    hidden_states: torch.Tensor,
    already_sequence_parallel: bool = False,
    router_logits: torch.Tensor | None = None,
    fold_rows: torch.Tensor | None = None,
    ffn_scale: torch.Tensor | None = None,
) -> torch.Tensor:
    if not getattr(self, "_tpu_attn_moe_overlap", False):
        return _orig_moe_forward(
            self,
            hidden_states,
            already_sequence_parallel=already_sequence_parallel,
        )

    orig_shape = hidden_states.shape
    hidden_dim = hidden_states.shape[-1]
    hidden_states = hidden_states.view(-1, hidden_dim)
    if router_logits is None:
        gate_out = self.gate(hidden_states)
        router_logits = gate_out[0] if isinstance(gate_out, tuple) else gate_out
    final_hidden_states = self.experts(
        hidden_states=hidden_states,
        router_logits=router_logits,
    )
    if (
        getattr(self, "replicate_shared_expert", False)
        and getattr(self, "shared_expert", None) is not None
    ):
        replicated_shared_output = self.shared_expert(hidden_states)
        final_hidden_states = final_hidden_states + replicated_shared_output
    if ffn_scale is not None:
        final_hidden_states = final_hidden_states * ffn_scale
    if fold_rows is not None:
        final_hidden_states = final_hidden_states + fold_rows
    if self.tp_size > 1:
        if hasattr(self.experts, "maybe_all_reduce_tensor_model_parallel"):
            final_hidden_states = self.experts.maybe_all_reduce_tensor_model_parallel(
                final_hidden_states
            )
        else:
            final_hidden_states = tensor_model_parallel_all_reduce(final_hidden_states)
    return final_hidden_states.view(orig_shape)


def _patched_decoder_forward(
    self: Qwen3NextDecoderLayer,
    hidden_states: torch.Tensor,
    residual: torch.Tensor | None,
    positions: torch.Tensor,
    **kwargs: Any,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    if not getattr(self, "_tpu_attn_moe_overlap", False):
        return _orig_decoder_forward(
            self,
            hidden_states,
            residual,
            positions,
            **kwargs,
        )

    full_num_tokens = positions.shape[-1]
    prequant = None
    if (
        input_norm_quant_op is not None
        and getattr(self, getattr(input_norm_quant_op, "LAYER_ATTR", ""), False)
        and full_num_tokens > getattr(envs, "TPU_INPUT_NORM_QUANT_MIN_ROWS", 8192)
    ):
        # TPU_INPUT_NORM_QUANT: the norm also emits the fp8 rows of its output
        # for the input projection (normed rows only where a bf16 consumer
        # needs them: GDN's in_proj_ba).
        hidden_states, residual, prequant = input_norm_quant_op.norm_quant(
            self, hidden_states, residual
        )
    elif residual is None:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
    else:
        hidden_states, residual = self.input_layernorm(hidden_states, residual)

    if self.layer_type == "linear_attention":
        if prequant is None:
            hidden_states = self.linear_attn(hidden_states=hidden_states)
        else:
            hidden_states = self.linear_attn(
                hidden_states=hidden_states, prequant=prequant
            )
    elif self.layer_type == "full_attention":
        if prequant is None:
            hidden_states = self.self_attn(
                hidden_states=hidden_states,
                positions=positions,
            )
        else:
            hidden_states = input_norm_quant_op.attention_forward(
                self.self_attn, positions, prequant
            )
    else:
        raise ValueError(f"Invalid layer_type: {self.layer_type}")

    min_tokens = max(envs.TPU_ATTN_MOE_OVERLAP_MIN_TOKENS, 16)
    if full_num_tokens < min_tokens:
        hidden_states = tensor_model_parallel_all_reduce(hidden_states)
        if self.layer_scale:
            hidden_states = hidden_states * (
                self.attn_layer_scale.to(hidden_states.dtype)[0] + 1
            )
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        if self.layer_scale:
            hidden_states = hidden_states * (
                self.ffn_layer_scale.to(hidden_states.dtype)[0] + 1
            )
        return hidden_states, residual

    if self.layer_scale:
        hidden_states = hidden_states * (
            self.attn_layer_scale.to(hidden_states.dtype)[0] + 1
        )

    tp_size = self.mlp.tp_size
    a_partial = hidden_states
    x_loc = tensor_model_parallel_reduce_scatter(
        a_partial + residual.to(a_partial.dtype) * (1.0 / tp_size), 0
    )
    h_loc = self.post_attention_layernorm(x_loc)
    gate_out = self.mlp.gate(h_loc)
    logits_loc = gate_out[0] if isinstance(gate_out, tuple) else gate_out
    qx_on = full_num_tokens >= _QX_MIN_TOKENS
    if qx_on and getattr(self, "_qx_route_local", False):
        packed_loc = _qx_route_local_packed(self, logits_loc)
        logits = tensor_model_parallel_all_gather(packed_loc, 0).narrow(
            0, 0, full_num_tokens
        )
    else:
        logits = tensor_model_parallel_all_gather(logits_loc, 0).narrow(
            0, 0, full_num_tokens
        )
    if qx_on and _QX_H_FP8_AG in (1, 2):
        h = _qx_all_gather_h_fp8(h_loc, full_num_tokens)
    else:
        h = tensor_model_parallel_all_gather(h_loc, 0).narrow(0, 0, full_num_tokens)
    ffn_scale = (self.ffn_layer_scale.to(h.dtype)[0] + 1) if self.layer_scale else None
    out = self.mlp(
        h,
        router_logits=logits,
        fold_rows=a_partial,
        ffn_scale=ffn_scale,
    )
    return out, residual


def apply_qwen3_next_attn_moe_overlap_patch() -> None:
    """Monkey-patch Qwen3Next and Qwen3.5 decoder layer for attention-MoE overlap."""
    global _PATCH_APPLIED
    if _PATCH_APPLIED:
        return

    tpu_moe_sp = getattr(envs, "TPU_MOE_SEQUENCE_PARALLEL", False) or (
        os.environ.get("TPU_MOE_SEQUENCE_PARALLEL", "0").strip().lower()
        in ("1", "true", "yes", "on")
    )
    use_moe_fused_ep = getattr(envs, "USE_MOE_FUSED_EP_KERNEL", False) or (
        os.environ.get("USE_MOE_FUSED_EP_KERNEL", "0").strip().lower()
        in ("1", "true", "yes", "on")
    )
    if tpu_moe_sp or use_moe_fused_ep:
        raise ValueError(
            "TPU_ATTN_MOE_OVERLAP=1 is only supported on the default TP MoE path "
            "(TPU_MOE_SEQUENCE_PARALLEL=0, USE_MOE_FUSED_EP_KERNEL=0)."
        )

    Qwen3NextDecoderLayer.__init__ = _patched_qwen3_next_decoder_init
    Qwen3_5DecoderLayer.__init__ = _patched_qwen3_5_decoder_init
    Qwen3NextDecoderLayer.forward = _patched_decoder_forward
    Qwen3NextSparseMoeBlock.forward = _patched_moe_forward

    if _QX_H_FP8_AG or _QX_ROUTE_LOCAL:
        logger.warning(
            "[QX-PREFILL] active: QX_MOE_H_FP8_AG=%d QX_MOE_ROUTE_LOCAL=%d "
            "QX_PREFILL_MIN_TOKENS=%d",
            _QX_H_FP8_AG,
            int(_QX_ROUTE_LOCAL),
            _QX_MIN_TOKENS,
        )
        print(
            f"[QX-PREFILL] active: QX_MOE_H_FP8_AG={_QX_H_FP8_AG} "
            f"QX_MOE_ROUTE_LOCAL={int(_QX_ROUTE_LOCAL)} "
            f"QX_PREFILL_MIN_TOKENS={_QX_MIN_TOKENS}",
            flush=True,
        )

    _PATCH_APPLIED = True
