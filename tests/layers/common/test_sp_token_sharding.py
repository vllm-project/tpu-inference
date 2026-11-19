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
"""Tests for the sequence-parallel residual stream: `enable_sp` under
attention DP with expert parallelism and one of two MoE backends:

- GMM_EP: USE_MOE_EP_KERNEL=0 (and USE_FUSED_EP_MOE_V2 / USE_ROUTING_MOE
  unset).
- ROUTING_MOE: USE_ROUTING_MOE=1 (with USE_MOE_EP_KERNEL=0 and
  USE_FUSED_EP_MOE_V2 unset, as both take priority over it).

The sharded-matmul and MoE tests need 8 devices (e.g. CPU with
XLA_FLAGS=--xla_force_host_platform_device_count=8, or a TPU v7x-8). On hosts
with fewer than 8 devices, `test_sp_8_device_suite_on_cpu` runs them in a
subprocess with 8 virtual CPU devices; the guard tests use a stand-in mesh and
run in-process anywhere.
"""

import itertools
import os
import re
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from vllm.model_executor.layers.linear import (ColumnParallelLinear,
                                               ReplicatedLinear,
                                               RowParallelLinear)

from tpu_inference.layers.common import fused_moe_gmm
from tpu_inference.layers.common import linear as common_linear
from tpu_inference.layers.common import quant_methods
from tpu_inference.layers.common import sharding as common_sharding
from tpu_inference.layers.common.moe import MoEBackend
from tpu_inference.layers.common.sharding import (MESH_AXIS_NAMES,
                                                  ShardingAxisName,
                                                  ShardingAxisNameBase)
from tpu_inference.layers.vllm.custom_ops import fused_moe as vllm_fused_moe
from tpu_inference.layers.vllm.interface import moe as vllm_moe_interface
from tpu_inference.layers.vllm.quantization.configs import \
    VllmQuantLinearConfig

requires_8_devices = pytest.mark.skipif(len(jax.devices()) < 8,
                                        reason="needs 8 devices")


@pytest.fixture(autouse=True)
def base_sharding_axes(monkeypatch):
    """Multi-axis sharding names (attn_dp, ...) and a clean env."""
    monkeypatch.setattr(ShardingAxisName, "_cls", ShardingAxisNameBase)
    monkeypatch.delenv("MOE_GMM_EP_RAGGED_A2A", raising=False)
    for flag in ("USE_MOE_EP_KERNEL", "USE_FUSED_EP_MOE_V2",
                 "USE_ROUTING_MOE"):
        monkeypatch.delenv(flag, raising=False)
    yield


def _make_mesh(**sizes):
    shape = tuple(sizes.get(a, 1) for a in MESH_AXIS_NAMES)
    return Mesh(np.array(jax.devices()[:8]).reshape(shape), MESH_AXIS_NAMES)


def _shape_only_mesh(**sizes):
    """Stand-in for the guard tests: they only read the mesh shape."""
    return SimpleNamespace(shape={a: sizes.get(a, 1)
                                  for a in MESH_AXIS_NAMES},
                           axis_names=MESH_AXIS_NAMES)


def test_keeps_tokens_sharded_guards():

    def keeps(mesh, enable_sp=True, moe_keeps_tokens=True):
        return common_sharding.keeps_tokens_sharded(
            mesh, enable_sp, moe_keeps_tokens=moe_keeps_tokens)

    mesh = _shape_only_mesh(attn_dp=2, model=4)
    # Enabled under attention DP + attention TP > 1 when the MoE keeps tokens
    # sharded (GMM_EP or ROUTING_MOE).
    assert keeps(mesh)
    assert not keeps(mesh, enable_sp=False)
    assert not keeps(None)
    # Any other MoE backend ignores keep_token_sharding.
    assert not keeps(mesh, moe_keeps_tokens=False)
    # Only under attention DP with attention TP > 1.
    assert not keeps(_shape_only_mesh(model=8))
    assert not keeps(_shape_only_mesh(attn_dp=8))
    # `expert` / `pcp` > 1: their order in EXPERT_DATA doesn't match the
    # ATTN_HEAD gather / reduce-scatter inside an ATTN_DATA block.
    assert not keeps(_shape_only_mesh(attn_dp=2, model=2, expert=2))
    assert not keeps(_shape_only_mesh(attn_dp=2, model=2, pcp=2))
    assert keeps(_shape_only_mesh(attn_dp=2, model=2, dcp=2))


_MOE_BACKEND_FLAGS = ("USE_MOE_EP_KERNEL", "USE_FUSED_EP_MOE_V2",
                      "USE_ROUTING_MOE")


@pytest.mark.parametrize("use_ep", [False, True])
@pytest.mark.parametrize("flags", [
    flags for r in range(len(_MOE_BACKEND_FLAGS) + 1)
    for flags in itertools.combinations(_MOE_BACKEND_FLAGS, r)
])
def test_select_moe_backend_matches_fused_moe_config(monkeypatch, use_ep,
                                                     flags):
    """The env-only selection used by the SP gate and the backend the MoE
    layers actually use are the same for every flag combination."""
    for flag in flags:
        monkeypatch.setenv(flag, "1")
    expected = vllm_moe_interface.select_moe_backend_from_fused_moe_config(
        SimpleNamespace(use_ep=use_ep))
    assert vllm_moe_interface.select_moe_backend(use_ep) == expected
    assert (expected == MoEBackend.GMM_EP) == (use_ep and not flags)


def _vllm_config(*,
                 quantization=quant_methods.FP8,
                 enable_sp=True,
                 enable_expert_parallel=True):
    return SimpleNamespace(
        additional_config={"enable_sp": enable_sp},
        parallel_config=SimpleNamespace(
            enable_expert_parallel=enable_expert_parallel,
            tensor_parallel_size=8),
        model_config=SimpleNamespace(model="test-model",
                                     quantization=quantization),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=8192))


def test_vllm_keeps_tokens_sharded(monkeypatch):
    """The one gate the linear configs and the MoE share: enable_sp, GMM_EP,
    ROUTING_MOE or FUSED_EP_MOE_V2, and a quantization method validated with
    the token-sharded stream (FP8 only)."""
    keeps = vllm_moe_interface.vllm_keeps_tokens_sharded
    mesh = _shape_only_mesh(attn_dp=2, model=4)
    assert keeps(_vllm_config(quantization=quant_methods.FP8), mesh)
    # Not validated with the token-sharded stream (some don't thread
    # sp_token_axes through their linears): plain enable_sp behavior.
    for quantization in (quant_methods.UNQUANTIZED,
                         quant_methods.COMPRESSED_TENSORS, quant_methods.AWQ,
                         quant_methods.NVFP4, quant_methods.MXFP4,
                         quant_methods.DSV4_FP8):
        assert not keeps(_vllm_config(quantization=quantization), mesh)
    assert not keeps(_vllm_config(enable_sp=False), mesh)
    assert not keeps(None, mesh)
    # Expert parallelism off: GMM_TP.
    assert not keeps(_vllm_config(enable_expert_parallel=False), mesh)
    # Another EP backend selected that ignores keep_token_sharding.
    monkeypatch.setenv("USE_MOE_EP_KERNEL", "1")
    assert not keeps(_vllm_config(), mesh)
    monkeypatch.delenv("USE_MOE_EP_KERNEL")
    # ROUTING_MOE and FUSED_EP_MOE_V2 honor keep_token_sharding.
    for flag in ("USE_ROUTING_MOE", "USE_FUSED_EP_MOE_V2"):
        monkeypatch.setenv(flag, "1")
        assert keeps(_vllm_config(), mesh), flag
        monkeypatch.delenv(flag)
    assert keeps(_vllm_config(), mesh)


def _linear_config(layer_cls,
                   weight_sharding,
                   *,
                   keep_tokens_sharded,
                   enable_sp=True,
                   defer_all_reduce=False):
    """A VllmQuantLinearConfig without a vLLM layer / VllmConfig: only the
    fields the SP sharding logic reads, on an attn_dp=2 x model=4 mesh
    (EXPERT_DATA and MLP_TENSOR both 8)."""
    cfg = VllmQuantLinearConfig.__new__(VllmQuantLinearConfig)
    cfg.mesh = _shape_only_mesh(attn_dp=2, model=4)
    cfg.enable_sp = enable_sp
    cfg.tp_size = 8
    cfg.keep_tokens_sharded = keep_tokens_sharded
    cfg.weight_sharding = weight_sharding
    cfg.input_sharding, cfg.output_sharding = cfg._sp_shardings(
        layer_cls.__new__(layer_cls))
    # Set after __init__ in practice (VllmRowParallelLinear).
    cfg.defer_all_reduce = defer_all_reduce
    return cfg


def _tokens(n):
    return SimpleNamespace(shape=(n, 128))


def _weight_specs():
    """(row, column, replicated) weight shardings. Built inside the tests,
    after the fixture has set the multi-axis sharding names."""
    return (P(ShardingAxisName.ATTN_HEAD,
              None), P(None, ShardingAxisName.ATTN_HEAD), P(None, None))


def test_linear_config_sp_shardings():
    row_ws, col_ws, rep_ws = _weight_specs()
    tok = P(ShardingAxisName.EXPERT_DATA, None)
    mlp = P(ShardingAxisName.MLP_TENSOR, None)

    # Token-sharded residual stream: the ends follow _sp_layout.
    row = _linear_config(RowParallelLinear, row_ws, keep_tokens_sharded=True)
    col = _linear_config(ColumnParallelLinear,
                         col_ws,
                         keep_tokens_sharded=True)
    # A replicated shared expert (row- or column-parallel layer type).
    rep = _linear_config(RowParallelLinear, rep_ws, keep_tokens_sharded=True)
    for n in (8, 64):
        # 8 tokens: one per device. Still token-sharded, like the shard_map
        # specs (no MLP_TENSOR row-count threshold in this mode).
        x = _tokens(n)
        assert (row.get_input_sharding(x),
                row.get_output_sharding(x)) == (None, tok)
        assert (col.get_input_sharding(x),
                col.get_output_sharding(x)) == (tok, None)
        assert (rep.get_input_sharding(x), rep.get_output_sharding(x)) == (tok,
                                                                           tok)
    # Token count not divisible by EXPERT_DATA: ATTN_DATA fallback, no
    # constraint (sp_token_axes is None, as in sharded_matmul).
    for cfg in (row, col, rep):
        assert cfg.get_input_sharding(_tokens(4)) is None
        assert cfg.get_output_sharding(_tokens(4)) is None

    # Plain enable_sp (no token-sharded residual stream): MLP_TENSOR ends
    # above the TPU_SECOND_LAST_MINOR row threshold.
    row = _linear_config(RowParallelLinear, row_ws, keep_tokens_sharded=False)
    col = _linear_config(ColumnParallelLinear,
                         col_ws,
                         keep_tokens_sharded=False)
    rep = _linear_config(ReplicatedLinear, rep_ws, keep_tokens_sharded=False)
    x = _tokens(64)
    assert (row.get_input_sharding(x), row.get_output_sharding(x)) == (None,
                                                                       mlp)
    assert (col.get_input_sharding(x), col.get_output_sharding(x)) == (mlp,
                                                                       None)
    assert (rep.get_input_sharding(x), rep.get_output_sharding(x)) == (None,
                                                                       None)
    assert row.get_output_sharding(_tokens(8)) is None
    assert col.get_input_sharding(_tokens(8)) is None

    # enable_sp off: no constraints.
    for layer_cls, ws in ((RowParallelLinear, row_ws), (ColumnParallelLinear,
                                                        col_ws)):
        cfg = _linear_config(layer_cls,
                             ws,
                             keep_tokens_sharded=False,
                             enable_sp=False)
        assert cfg.get_input_sharding(x) is None
        assert cfg.get_output_sharding(x) is None


def test_linear_config_deferred_all_reduce_not_token_sharded():
    """A TP-sharded row-parallel linear with reduce_results=False returns
    partial sums on ATTN_DATA; its output must not be token-sharded."""
    row_ws, _, rep_ws = _weight_specs()
    x = _tokens(64)
    row = _linear_config(RowParallelLinear,
                         row_ws,
                         keep_tokens_sharded=True,
                         defer_all_reduce=True)
    assert row.get_output_sharding(x) is None
    # A replicated (shared-expert) weight has no partial sums, so its output
    # stays token-sharded even with reduce_results=False.
    rep = _linear_config(RowParallelLinear,
                         rep_ws,
                         keep_tokens_sharded=True,
                         defer_all_reduce=True)
    assert rep.get_output_sharding(x) == P(ShardingAxisName.EXPERT_DATA, None)


def _shared_expert_block(prefix, enable_sp, quantization):
    """A shared-expert MLP (gate_up_proj + down_proj) whose linear configs are
    built by the real VllmQuantLinearConfig.__init__ from the layer prefix, on
    an attn_dp=2 x model=4 mesh with GMM_EP. The vLLM layers' own __init__
    (weights, TP groups) is skipped."""
    mesh = _shape_only_mesh(attn_dp=2, model=4)
    vllm_config = _vllm_config(quantization=quantization, enable_sp=enable_sp)

    def linear(layer_cls, name):
        layer = layer_cls.__new__(layer_cls)
        torch.nn.Module.__init__(layer)
        layer.prefix = f"model.layers.0{prefix}{name}"
        layer.output_size = 128
        layer.quant_method = SimpleNamespace(
            linear_config=VllmQuantLinearConfig(vllm_config, mesh, layer))
        return layer

    block = torch.nn.Module()
    block.gate_up_proj = linear(ColumnParallelLinear, "gate_up_proj")
    block.down_proj = linear(RowParallelLinear, "down_proj")
    # reduce_results=False, as set by VllmRowParallelLinear after __init__.
    block.down_proj.quant_method.linear_config.defer_all_reduce = True
    return mesh, block


@pytest.mark.parametrize(
    "prefix, enable_sp, quantization, replicated",
    [
        (".mlp.shared_expert.", True, quant_methods.FP8, True),  # Qwen naming
        # DeepSeek V2/V3 naming.
        (".mlp.shared_experts.", True, quant_methods.FP8, True),
        # Unmatched name: stays TP-sharded with deferred partial sums.
        (".mlp.shared_mlp.", True, quant_methods.FP8, False),
        # Token-sharded residual stream off: nothing is replicated.
        (".mlp.shared_expert.", False, quant_methods.FP8, False),
        # Quantization not validated for it: plain enable_sp, not replicated.
        (".mlp.shared_expert.", True, quant_methods.COMPRESSED_TENSORS, False),
        (".mlp.shared_expert.", True, quant_methods.UNQUANTIZED, False),
    ])
def test_shared_expert_reduce_follows_linear_configs(monkeypatch, prefix,
                                                     enable_sp, quantization,
                                                     replicated):
    """The MoE runner skips the shared-expert all-reduce exactly when the
    linear configs replicated the shared expert, so it is never reduced twice
    (replicated: already complete) or zero times (TP-sharded: partial sums)."""
    mesh, block = _shared_expert_block(prefix, enable_sp, quantization)
    gate_up = block.gate_up_proj.quant_method.linear_config
    down = block.down_proj.quant_method.linear_config
    row_ws, col_ws, rep_ws = _weight_specs()
    assert gate_up.weight_sharding == (rep_ws if replicated else col_ws)
    assert down.weight_sharding == (rep_ws if replicated else row_ws)

    # A TP-sharded down_proj's deferred partial sums are never token-sharded
    # over EXPERT_DATA; a replicated one's complete output is.
    if down.keep_tokens_sharded:
        expected = P(ShardingAxisName.EXPERT_DATA,
                     None) if replicated else None
        assert down.get_output_sharding(_tokens(64)) == expected

    reduced = object()
    all_reduce = mock.Mock(return_value=reduced)
    monkeypatch.setattr(vllm_fused_moe, "_get_mesh", lambda: mesh)
    monkeypatch.setattr(vllm_fused_moe, "_all_reduce_over_tp", all_reduce)
    runner = vllm_fused_moe.VllmMoERunner.__new__(vllm_fused_moe.VllmMoERunner)
    # vLLM wraps the shared-expert module (SharedExperts); modules() still
    # reaches its linears.
    object.__setattr__(runner, "_shared_experts", block)
    object.__setattr__(runner, "moe_config",
                       SimpleNamespace(is_sequence_parallel=False))

    shared_output = object()
    out = runner._maybe_reduce_shared_expert_output(
        shared_output, fused_output_is_reduced=True)
    if replicated:
        assert out is shared_output
        all_reduce.assert_not_called()
    else:
        assert out is reduced
        all_reduce.assert_called_once_with(shared_output, mesh,
                                           ShardingAxisName.ATTN_HEAD)

    assert not common_sharding.shared_expert_is_replicated(None)
    assert not common_sharding.shared_expert_is_replicated(object())


def test_sp_gather_weight_selection():
    """The column-parallel weight-gather path is taken only for small weights
    that also receive fewer bytes than the token all-gather."""
    pick = common_linear._sp_gather_weight_is_cheaper
    bf16 = jnp.bfloat16
    # Tiny projection (in_proj_ba-like), prefill: M=8192, K=8192, N=128,
    # TP=8. Full weight 2 MiB; ~0.3 MB vs 16.8 MB received.
    assert pick((1024, 8192), bf16, (8192, 16), bf16, 8)
    # Normal projection, prefill: M=8192, K=8192, N=6144, TP=8. Fewer bytes
    # with all_to_all (~14 MB vs 16.8 MB), but the full weight is ~100 MB:
    # over the cap, so tokens are gathered.
    assert not pick((1024, 8192), bf16, (8192, 768), bf16, 8)
    # Tiny projection, decode: M=16, TP=2. Gathering 16 token rows is cheaper.
    assert not pick((8, 8192), bf16, (8192, 64), bf16, 2)
    # Extra leading batch dims count as token rows.
    assert pick((512, 2, 8192), bf16, (8192, 16), bf16, 8)


@requires_8_devices
@pytest.mark.parametrize("mesh_sizes", [
    dict(attn_dp=2, model=4),
    dict(attn_dp=2, model=2, dcp=2),
],
                         ids=["model4", "model2_dcp2"])
def test_sp_token_sharding_linears(mesh_sizes):
    """Replicated / column / row-parallel matmuls with token-sharded ends
    match the ATTN_DATA path; bf16 and FP8 (FP8 all-gather) weights. The
    model2_dcp2 mesh makes ATTN_HEAD a multi-axis TP group (model, dcp)."""
    mesh = _make_mesh(**mesh_sizes)
    t, h, qkv_out, ba_out = 64, 128, 256, 8
    keys = jax.random.split(jax.random.PRNGKey(42), 6)
    x = jax.random.normal(keys[0], (t, h), jnp.bfloat16)
    w_rep = jax.random.normal(keys[1], (h, 16), jnp.bfloat16)
    w_ba = jax.random.normal(keys[2], (h, ba_out), jnp.bfloat16)
    w_qkv = jax.random.normal(keys[3], (h, qkv_out),
                              jnp.float32).astype(jnp.float8_e4m3fn)
    s_qkv = jnp.ones((qkv_out, ), dtype=jnp.float32) * 0.1
    w_out = jax.random.normal(keys[4], (qkv_out, h),
                              jnp.float32).astype(jnp.float8_e4m3fn)
    s_out = jnp.ones((h, ), dtype=jnp.float32) * 0.1

    tok_spec = P(ShardingAxisName.EXPERT_DATA, None)
    attn_spec = P(ShardingAxisName.ATTN_DATA, None)
    col_spec = P(None, ShardingAxisName.ATTN_HEAD)
    row_spec = P(ShardingAxisName.ATTN_HEAD, None)
    rep_spec = P(None, None)

    def make_run_block(sp_token_axes):

        def run_block(x_in):
            kw = {"mesh": mesh, "sp_token_axes": sp_token_axes}
            y_rep = common_linear.sharded_matmul(x_in, w_rep, rep_spec, **kw)
            # ba_out / TP = 2 columns per shard: weight-gather + all_to_all.
            y_ba = common_linear.sharded_matmul(x_in, w_ba, col_spec, **kw)
            y_qkv = common_linear.sharded_quantized_matmul(
                x_in, w_qkv, s_qkv, col_spec, **kw)
            y_out = common_linear.sharded_quantized_matmul(
                y_qkv, w_out, s_out, row_spec, **kw)
            return y_rep, y_ba, y_qkv, y_out

        return run_block

    refs = jax.jit(make_run_block(None))(jax.device_put(
        x, NamedSharding(mesh, attn_spec)))
    sps = jax.jit(make_run_block(ShardingAxisName.EXPERT_DATA))(jax.device_put(
        x, NamedSharding(mesh, tok_spec)))

    tok_sharding = NamedSharding(mesh, tok_spec)
    # Replicated and row-parallel outputs stay token-sharded.
    assert sps[0].sharding.is_equivalent_to(tok_sharding, 2)
    assert sps[3].sharding.is_equivalent_to(tok_sharding, 2)
    for sp, ref, tol in zip(sps, refs, (1e-2, 1e-2, 1e-2, 2e-2)):
        np.testing.assert_allclose(np.asarray(sp, dtype=np.float32),
                                   np.asarray(ref, dtype=np.float32),
                                   atol=tol,
                                   rtol=tol)


@requires_8_devices
def test_sp_row_parallel_defer_all_reduce_keeps_attn_data():
    # A deferred all-reduce leaves partial sums, so no reduce-scatter.
    mesh = _make_mesh(attn_dp=2, model=4)
    keys = jax.random.split(jax.random.PRNGKey(3), 2)
    x = jax.random.normal(keys[0], (64, 256), jnp.float32)
    w = jax.random.normal(keys[1], (256, 128), jnp.float32)
    row_spec = P(ShardingAxisName.ATTN_HEAD, None)
    out = jax.jit(lambda x: common_linear.sharded_matmul(
        x,
        w,
        row_spec,
        mesh=mesh,
        defer_all_reduce=True,
        sp_token_axes=ShardingAxisName.EXPERT_DATA))(x)
    ref = jax.jit(lambda x: common_linear.sharded_matmul(
        x, w, row_spec, mesh=mesh, defer_all_reduce=True))(x)
    assert out.sharding.is_equivalent_to(ref.sharding, 2)


def _ref_gmm(lhs,
             rhs,
             rhs_scale,
             rhs_bias,
             group_sizes,
             group_offset,
             fuse_act=None,
             preferred_element_type=None):
    """Pure-jnp stand-in for gmm_wrapper so fused_moe_func runs on CPU."""
    del rhs_scale, rhs_bias
    start_group = group_offset[0]
    num_local = rhs.shape[0]
    local_sizes = jax.lax.dynamic_slice_in_dim(group_sizes,
                                               start_group,
                                               num_local,
                                               axis=0)
    ends = jnp.cumsum(local_sizes)
    start_row = jnp.sum(
        jnp.where(
            jnp.arange(group_sizes.shape[0]) < start_group, group_sizes, 0))
    pos = jnp.arange(lhs.shape[0]) - start_row
    gid = jnp.searchsorted(ends, pos, side="right")
    gid = jnp.clip(gid, 0, num_local - 1)
    out = jnp.einsum("mh,mhf->mf", lhs.astype(jnp.float32),
                     rhs[gid].astype(jnp.float32))
    if fuse_act:
        f = out.shape[1] // 2
        out = jax.nn.silu(out[:, :f]) * out[:, f:]
    return out.astype(preferred_element_type or lhs.dtype)


@requires_8_devices
@pytest.mark.parametrize("tc_permute", [False, True])
@pytest.mark.parametrize("all_gather_fp8", [False, True])
@pytest.mark.parametrize("fold_routing", [False, True])
def test_sp_token_sharding_standard_gmm_ep(monkeypatch, tc_permute,
                                           all_gather_fp8, fold_routing):
    """Standard expert_parallel_gmm (including MOE_TC_PERMUTE) with
    keep_token_sharding=True matches keep_token_sharding=False and returns
    EXPERT_DATA-sharded output."""
    monkeypatch.setenv("MOE_TC_PERMUTE", "1" if tc_permute else "0")
    monkeypatch.setenv("MOE_FOLD_ROUTING_INTO_DISPATCH",
                       "1" if fold_routing else "0")
    mesh = _make_mesh(attn_dp=2, model=4)
    num_tokens, hidden, inter, num_experts, topk = 64, 128, 32, 16, 4
    k1, k2, k3, k4 = jax.random.split(jax.random.PRNGKey(7), 4)
    x = jax.random.normal(k1, (num_tokens, hidden), jnp.bfloat16)
    gating = jax.random.normal(k2, (num_tokens, num_experts), jnp.bfloat16)
    w1 = jax.random.normal(k3, (num_experts, hidden, 2 * inter),
                           jnp.bfloat16) * 0.1
    w2 = jax.random.normal(k4,
                           (num_experts, inter, hidden), jnp.bfloat16) * 0.1

    attn_spec = NamedSharding(mesh, P(ShardingAxisName.ATTN_DATA, None))
    tok_spec = NamedSharding(mesh, P(ShardingAxisName.EXPERT_DATA, None))
    ep_spec = NamedSharding(mesh, P(ShardingAxisName.EXPERT, None, None))
    w1_s = jax.device_put(w1, ep_spec)
    w2_s = jax.device_put(w2, ep_spec)

    def _run(keep_sharded):
        in_sharding = tok_spec if keep_sharded else attn_spec
        x_in = jax.device_put(x, in_sharding)
        g_in = jax.device_put(gating, in_sharding)
        return fused_moe_gmm.fused_moe_func(
            hidden_states=x_in,
            w1=w1_s,
            w2=w2_s,
            w1_scale=None,
            w2_scale=None,
            w1_bias=None,
            w2_bias=None,
            gating_output=g_in,
            topk=topk,
            renormalize=True,
            mesh=mesh,
            use_ep=True,
            activation="silu",
            scoring_fn="softmax",
            all_gather_fp8=all_gather_fp8,
            onehot_moe_permute_threshold=0 if tc_permute else 10**8,
            scatter_results=True,
            keep_token_sharding=keep_sharded,
        )

    with mock.patch.object(fused_moe_gmm, "gmm_wrapper", _ref_gmm):
        jax.clear_caches()
        ref = _run(keep_sharded=False)
        out = _run(keep_sharded=True)

    assert ref.sharding.is_equivalent_to(attn_spec, 2)
    assert out.sharding.is_equivalent_to(tok_spec, 2)
    np.testing.assert_allclose(np.asarray(out, np.float32),
                               np.asarray(ref, np.float32),
                               rtol=1e-2,
                               atol=1e-2)


@requires_8_devices
@pytest.mark.parametrize("tc_permute", [False, True])
def test_sp_token_sharding_small_batch_fallback(monkeypatch, tc_permute):
    """When num_tokens is divisible by ATTN_DATA (2) but not EXPERT_DATA (8),
    e.g. small decode batches (16 tokens across 4 DP x 8 TP = 32 devices),
    fused_moe_func falls back to ATTN_DATA instead of failing psum_scatter."""
    monkeypatch.setenv("MOE_TC_PERMUTE", "1" if tc_permute else "0")
    mesh = _make_mesh(attn_dp=2, model=4)
    # 4 tokens: divisible by ATTN_DATA=2, not divisible by EXPERT_DATA=8;
    # 4 * topk(4) = 16 satisfies the GMM row-alignment check.
    num_tokens, hidden, inter, num_experts, topk = 4, 128, 32, 16, 4
    k1, k2, k3, k4 = jax.random.split(jax.random.PRNGKey(11), 4)
    x = jax.random.normal(k1, (num_tokens, hidden), jnp.bfloat16)
    gating = jax.random.normal(k2, (num_tokens, num_experts), jnp.bfloat16)
    w1 = jax.random.normal(k3, (num_experts, hidden, 2 * inter),
                           jnp.bfloat16) * 0.1
    w2 = jax.random.normal(k4,
                           (num_experts, inter, hidden), jnp.bfloat16) * 0.1

    attn_spec = NamedSharding(mesh, P(ShardingAxisName.ATTN_DATA, None))
    ep_spec = NamedSharding(mesh, P(ShardingAxisName.EXPERT, None, None))
    x_in = jax.device_put(x, attn_spec)
    g_in = jax.device_put(gating, attn_spec)
    w1_s = jax.device_put(w1, ep_spec)
    w2_s = jax.device_put(w2, ep_spec)

    assert common_sharding.sp_token_axes(
        mesh,
        common_sharding.keeps_tokens_sharded(mesh, True,
                                             moe_keeps_tokens=True),
        num_tokens) is None

    def _run(keep_sharded):
        return fused_moe_gmm.fused_moe_func(
            hidden_states=x_in,
            w1=w1_s,
            w2=w2_s,
            w1_scale=None,
            w2_scale=None,
            w1_bias=None,
            w2_bias=None,
            gating_output=g_in,
            topk=topk,
            renormalize=True,
            mesh=mesh,
            use_ep=True,
            activation="silu",
            scoring_fn="softmax",
            all_gather_fp8=True,
            onehot_moe_permute_threshold=0 if tc_permute else 10**8,
            scatter_results=True,
            keep_token_sharding=keep_sharded,
        )

    with mock.patch.object(fused_moe_gmm, "gmm_wrapper", _ref_gmm):
        jax.clear_caches()
        ref = _run(keep_sharded=False)
        out = _run(keep_sharded=True)

    assert out.sharding.is_equivalent_to(attn_spec, 2)
    np.testing.assert_allclose(np.asarray(out, np.float32),
                               np.asarray(ref, np.float32),
                               rtol=1e-2,
                               atol=1e-2)


@pytest.mark.skipif(len(jax.devices()) >= 8,
                    reason="8 devices already available in-process")
def test_sp_8_device_suite_on_cpu():
    """On single-chip CI hosts (< 8 devices), run the @requires_8_devices tests
    in a child process with 8 virtual CPU devices so they are never silently
    skipped."""
    repo_root = str(Path(__file__).resolve().parents[3])
    env = os.environ.copy()
    for var in ("COVERAGE_FILE", "COVERAGE_PROCESS_START",
                "BUILDKITE_ANALYTICS_TOKEN", "JAX_PLATFORM_NAME",
                "JAX_COMPILATION_CACHE_DIR", "BVT_ONLY"):
        env.pop(var, None)
    env["VLLM_DISABLE_COMPILE_CACHE"] = "1"
    py_path = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{repo_root}:{py_path}" if py_path else repo_root
    env["JAX_PLATFORMS"] = "cpu"
    env["PJRT_DEVICE"] = "CPU"
    xla_flags = re.sub(r"--xla_force_host_platform_device_count=\S+", "",
                       env.get("XLA_FLAGS", "")).strip()
    env["XLA_FLAGS"] = (
        f"{xla_flags} --xla_force_host_platform_device_count=8".strip())
    expr = ("test_sp_token_sharding_linears or "
            "test_sp_row_parallel_defer_all_reduce_keeps_attn_data or "
            "test_sp_token_sharding_standard_gmm_ep or "
            "test_sp_token_sharding_small_batch_fallback")
    proc = subprocess.run(
        [
            sys.executable, "-m", "pytest", "-v", "-o", "addopts=", __file__,
            "-k", expr
        ],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, (
        f"8-device CPU suite failed (rc={proc.returncode}):\n"
        f"{proc.stdout}\n{proc.stderr}")
    assert re.search(r"\b13 passed\b", proc.stdout), proc.stdout
    assert not re.search(r"\b\d+ skipped\b", proc.stdout), proc.stdout
