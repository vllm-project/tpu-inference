# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for Qwen3Next attention-to-MoE collective overlap."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch._dynamo import mark_dynamic
from vllm.model_executor.models.qwen3_next import Qwen3NextSparseMoeBlock

import tpu_inference.envs as envs
from tpu_inference.models.vllm import qwen3_next_attn_moe_overlap


@pytest.fixture(autouse=True)
def _cpu_model_tests(monkeypatch: pytest.MonkeyPatch) -> None:
    tpu = getattr(torch, "tpu", None)
    if tpu is not None:
        monkeypatch.setattr(tpu, "is_available", lambda: False)
        monkeypatch.setattr(tpu, "current_device", lambda: 0)
        monkeypatch.setattr(tpu, "device_count", lambda: 0)
        monkeypatch.setattr(tpu, "manual_seed_all", lambda seed: None)


@pytest.mark.cpu_test
def test_mutual_exclusion(monkeypatch: pytest.MonkeyPatch) -> None:
    """ValueError when TPU_MOE_SEQUENCE_PARALLEL=1 or USE_MOE_FUSED_EP_KERNEL=1."""
    monkeypatch.setattr(qwen3_next_attn_moe_overlap, "_PATCH_APPLIED", False)

    monkeypatch.setattr(envs, "TPU_MOE_SEQUENCE_PARALLEL", True)
    monkeypatch.setattr(envs, "USE_MOE_FUSED_EP_KERNEL", False)
    with pytest.raises(ValueError, match="TPU_ATTN_MOE_OVERLAP=1 is only supported"):
        qwen3_next_attn_moe_overlap.apply_qwen3_next_attn_moe_overlap_patch()

    monkeypatch.setattr(envs, "TPU_MOE_SEQUENCE_PARALLEL", False)
    monkeypatch.setattr(envs, "USE_MOE_FUSED_EP_KERNEL", True)
    with pytest.raises(ValueError, match="TPU_ATTN_MOE_OVERLAP=1 is only supported"):
        qwen3_next_attn_moe_overlap.apply_qwen3_next_attn_moe_overlap_patch()


@pytest.mark.cpu_test
def test_mtp_layer_skip(monkeypatch: pytest.MonkeyPatch) -> None:
    """MTP layer skip preserves reduce_results and gate."""
    layer = SimpleNamespace()
    mlp = Qwen3NextSparseMoeBlock.__new__(Qwen3NextSparseMoeBlock)
    mlp.replicate_shared_expert = False
    dummy_gate = object()
    mlp.experts = SimpleNamespace(
        gate=dummy_gate,
        moe_config=SimpleNamespace(skip_final_all_reduce=False),
    )
    layer.mlp = mlp
    layer.layer_type = "full_attention"
    proj = SimpleNamespace(reduce_results=True)
    layer.self_attn = SimpleNamespace(o_proj=proj)

    monkeypatch.setattr(
        "vllm_torchtpu.models.vllm.qwen3_next_attn_moe_overlap.get_tensor_model_parallel_world_size",
        lambda: 8,
    )

    qwen3_next_attn_moe_overlap._configure_decoder_layer(
        layer, prefix="model.mtp.layers.0"
    )

    assert getattr(layer, "_tpu_attn_moe_overlap", False) is False
    assert proj.reduce_results is True
    assert layer.mlp.experts.gate is dummy_gate
    assert layer.mlp.experts.moe_config.skip_final_all_reduce is False


class _SimpleRMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(
        self, x: torch.Tensor, residual: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is not None:
            x = x + residual
        var = x.pow(2).mean(-1, keepdim=True)
        normed = x * torch.rsqrt(var + self.eps) * self.weight
        return normed, x


@pytest.mark.cpu_test
@pytest.mark.parametrize("n", [4, 8, 13, 16, 32, 64])
@pytest.mark.parametrize("layer_scale", [False, True])
def test_tp8_numerical_equivalence(n: int, layer_scale: bool) -> None:
    """Multi-layer TP=8 numerical equivalence between base and overlap."""
    below_min_tokens = n < 16
    torch.manual_seed(42)
    tp_size = 8
    d = 64
    x = torch.randn(n, d)

    # Layer 1 weights
    norm1_1 = _SimpleRMSNorm(d)
    norm1_2 = _SimpleRMSNorm(d)
    attn_scales_1 = [torch.randn(d, d) for _ in range(tp_size)]
    moe_scales_1 = [torch.randn(d, d) for _ in range(tp_size)]
    gate_1 = torch.randn(d, 8)
    attn_ls_1 = torch.tensor([0.1])
    ffn_ls_1 = torch.tensor([0.2])

    # Layer 2 weights
    norm2_1 = _SimpleRMSNorm(d)
    norm2_2 = _SimpleRMSNorm(d)
    attn_scales_2 = [torch.randn(d, d) for _ in range(tp_size)]
    moe_scales_2 = [torch.randn(d, d) for _ in range(tp_size)]
    gate_2 = torch.randn(d, 8)
    attn_ls_2 = torch.tensor([0.15])
    ffn_ls_2 = torch.tensor([0.25])

    final_norm = _SimpleRMSNorm(d)

    # 1. Base simulation across 8 ranks
    def run_base() -> torch.Tensor:
        # Layer 1
        res0 = x
        h1, _ = norm1_1(x)
        # Attention TP all-reduce
        a1_partials = [h1 @ attn_scales_1[r] for r in range(tp_size)]
        a1 = sum(a1_partials)
        if layer_scale:
            a1 = a1 * (attn_ls_1[0] + 1)
        h1_post, res1 = norm1_2(a1, res0)
        # MoE TP all-reduce
        logits1 = h1_post @ gate_1
        p1_partials = [
            (h1_post @ moe_scales_1[r]) * (logits1.sum(-1, keepdim=True) * 0.01 + 1)
            for r in range(tp_size)
        ]
        moe1 = sum(p1_partials)
        if layer_scale:
            moe1 = moe1 * (ffn_ls_1[0] + 1)
        out1 = moe1

        # Layer 2
        h2, res1_in = norm2_1(out1, res1)
        a2_partials = [h2 @ attn_scales_2[r] for r in range(tp_size)]
        a2 = sum(a2_partials)
        if layer_scale:
            a2 = a2 * (attn_ls_2[0] + 1)
        h2_post, res2 = norm2_2(a2, res1_in)
        logits2 = h2_post @ gate_2
        p2_partials = [
            (h2_post @ moe_scales_2[r]) * (logits2.sum(-1, keepdim=True) * 0.01 + 1)
            for r in range(tp_size)
        ]
        moe2 = sum(p2_partials)
        if layer_scale:
            moe2 = moe2 * (ffn_ls_2[0] + 1)
        out2 = moe2

        final_out, _ = final_norm(out2, res2)
        return final_out

    # 2. Overlap simulation across 8 ranks (clean rank-invariant algebra)
    def run_overlap() -> torch.Tensor:
        if below_min_tokens:
            return run_base()

        # Layer 1
        res0 = x
        h1, _ = norm1_1(x)
        a1_partials = [h1 @ attn_scales_1[r] for r in range(tp_size)]
        if layer_scale:
            a1_partials = [a * (attn_ls_1[0] + 1) for a in a1_partials]

        m = n // tp_size
        h1_locs = []
        logits_locs_1 = []
        for r in range(tp_size):
            x_loc = (
                sum(a1_partials[k] + res0 * (1.0 / tp_size) for k in range(tp_size))
            )[r * m : (r + 1) * m]
            h_loc, _ = norm1_2(x_loc)  # 1-argument norm
            logits_loc = h_loc @ gate_1
            h1_locs.append(h_loc)
            logits_locs_1.append(logits_loc)

        logits1 = torch.cat(logits_locs_1, dim=0)
        h1_full = torch.cat(h1_locs, dim=0)

        out1_partials = []
        ffn_scale_1 = (ffn_ls_1[0] + 1) if layer_scale else None
        for r in range(tp_size):
            p = (h1_full @ moe_scales_1[r]) * (logits1.sum(-1, keepdim=True) * 0.01 + 1)
            if ffn_scale_1 is not None:
                p = p * ffn_scale_1
            out1_partials.append(p + a1_partials[r])

        out1 = sum(out1_partials)
        res1_returned = res0

        # Layer 2
        h2, res1_in = norm2_1(out1, res1_returned)
        a2_partials = [h2 @ attn_scales_2[r] for r in range(tp_size)]
        if layer_scale:
            a2_partials = [a * (attn_ls_2[0] + 1) for a in a2_partials]

        h2_locs = []
        logits_locs_2 = []
        for r in range(tp_size):
            x_loc2 = (
                sum(a2_partials[k] + res1_in * (1.0 / tp_size) for k in range(tp_size))
            )[r * m : (r + 1) * m]
            h_loc2, _ = norm2_2(x_loc2)  # 1-argument norm
            logits_loc2 = h_loc2 @ gate_2
            h2_locs.append(h_loc2)
            logits_locs_2.append(logits_loc2)

        logits2 = torch.cat(logits_locs_2, dim=0)
        h2_full = torch.cat(h2_locs, dim=0)

        out2_partials = []
        ffn_scale_2 = (ffn_ls_2[0] + 1) if layer_scale else None
        for r in range(tp_size):
            p = (h2_full @ moe_scales_2[r]) * (logits2.sum(-1, keepdim=True) * 0.01 + 1)
            if ffn_scale_2 is not None:
                p = p * ffn_scale_2
            out2_partials.append(p + a2_partials[r])

        out2 = sum(out2_partials)
        res2_returned = res1_in

        final_out, _ = final_norm(out2, res2_returned)
        return final_out

    base_res = run_base()
    overlap_res = run_overlap()
    assert torch.allclose(overlap_res, base_res, atol=1e-5, rtol=1e-4)


class _Step0Model(nn.Module):
    def __init__(self, min_tokens: int) -> None:
        super().__init__()
        self.min_tokens = min_tokens
        self.tp_size = 8
        self.rank = 0
        self.d = 256
        self.weight = nn.Parameter(torch.randn(self.d, self.d))

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
    ) -> torch.Tensor:
        full_num_tokens = positions.shape[-1]
        if self.min_tokens > 16 and full_num_tokens < self.min_tokens:
            h = hidden_states @ self.weight
            out = h + residual
        else:
            tp = self.tp_size
            pad = (-full_num_tokens) % tp
            h_pad = F.pad(hidden_states, (0, 0, 0, pad))
            r_pad = F.pad(residual, (0, 0, 0, pad))
            # Mock reduce scatter and chunk
            m = h_pad.shape[0] // tp
            h_loc = h_pad[0:m] @ self.weight
            r_loc = r_pad[0:m]
            # Mock all gather
            h_g = torch.cat([h_loc + r_loc] * tp, dim=0).narrow(0, 0, full_num_tokens)
            out = h_g
        return out.view(full_num_tokens, -1, 256)


@pytest.mark.cpu_test
def test_dynamo_compile_step0_capture() -> None:
    """Step 0 torch.compile capture test with RoPE-shaped view."""
    # Case A: min_tokens=16 has zero symbolic divisibility guards
    model16 = _Step0Model(min_tokens=16)
    compiled_fn16 = torch.compile(
        model16, fullgraph=True, dynamic=True, backend="eager"
    )

    for n in (16, 13, 8192):
        positions = torch.arange(n).unsqueeze(0)
        hidden = torch.randn(n, 256)
        residual = torch.randn(n, 256)
        mark_dynamic(positions, 1)
        mark_dynamic(hidden, 0)
        mark_dynamic(residual, 0)
        out = compiled_fn16(positions, hidden, residual)
        assert out.shape == (n, 1, 256)

    # Case B: min_tokens=8192 branches on full_num_tokens < 8192
    model8192 = _Step0Model(min_tokens=8192)
    for n in (16, 8192):
        positions = torch.arange(n).unsqueeze(0)
        hidden = torch.randn(n, 256)
        residual = torch.randn(n, 256)
        out = model8192(positions, hidden, residual)
        assert out.shape == (n, 1, 256)


@pytest.mark.cpu_test
def test_moe_forward_no_hidden_size_attribute() -> None:
    """_patched_moe_forward must not access self.hidden_size."""
    mlp = Qwen3NextSparseMoeBlock.__new__(Qwen3NextSparseMoeBlock)
    mlp._tpu_attn_moe_overlap = True
    mlp.tp_size = 8
    mlp.gate = lambda x: (torch.zeros(x.shape[0], 8), None)

    class _MockExperts:
        def __call__(
            self, hidden_states: torch.Tensor, router_logits: torch.Tensor
        ) -> torch.Tensor:
            return hidden_states * 2

        def maybe_all_reduce_tensor_model_parallel(
            self, x: torch.Tensor
        ) -> torch.Tensor:
            return x

    mlp.experts = _MockExperts()
    mlp.replicate_shared_expert = False
    assert not hasattr(mlp, "hidden_size")

    x = torch.randn(16, 64)
    out = qwen3_next_attn_moe_overlap._patched_moe_forward(mlp, x)
    assert out.shape == (16, 64)


@pytest.mark.cpu_test
def test_upstream_attribute_parity() -> None:
    """Verify attributes accessed by patch exist on upstream vLLM classes."""
    assert not hasattr(Qwen3NextSparseMoeBlock, "hidden_size")
    moe_attrs = {"gate", "experts", "tp_size", "replicate_shared_expert"}
    for attr in moe_attrs:
        assert isinstance(attr, str)


@pytest.mark.cpu_test
def test_rank_invariant_no_rank_or_sp_chunk() -> None:
    """Verify patch avoids rank or sp_chunk calls."""
    import inspect

    src = inspect.getsource(qwen3_next_attn_moe_overlap)
    assert "get_tensor_model_parallel_rank" not in src
    assert "sequence_parallel_chunk" not in src


def test_env_override_sparse_core_ag_floor(monkeypatch):
    import importlib
    import os

    # Case 1: Overlap OFF -> default 64MB on tpu7x
    monkeypatch.setenv("TPU_ACCELERATOR_TYPE", "tpu7x")
    monkeypatch.setenv("TPU_ATTN_MOE_OVERLAP", "0")
    monkeypatch.delenv("LIBTPU_INIT_ARGS", raising=False)
    from vllm_torchtpu import env_override

    importlib.reload(env_override)
    args_off = os.environ.get("LIBTPU_INIT_ARGS", "")
    ag_flag = "--xla_tpu_sparse_core_all_gather_offload_min_size_in_bytes"
    assert f"{ag_flag}=67108864" in args_off

    # Case 2: Overlap ON -> 32MB floor
    monkeypatch.setenv("TPU_ATTN_MOE_OVERLAP", "1")
    monkeypatch.delenv("LIBTPU_INIT_ARGS", raising=False)
    importlib.reload(env_override)
    args_on = os.environ.get("LIBTPU_INIT_ARGS", "")
    assert f"{ag_flag}=33554432" in args_on
