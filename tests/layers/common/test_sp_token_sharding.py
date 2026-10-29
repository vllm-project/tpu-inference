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
attention DP with the GMM_EP MoE (USE_MOE_EP_KERNEL=0).

The sharded-matmul tests need 8 devices (e.g. CPU with
XLA_FLAGS=--xla_force_host_platform_device_count=8, or a TPU v7x-8); the
guard tests run anywhere.
"""

from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from tpu_inference.layers.common import fused_moe_gmm
from tpu_inference.layers.common import linear as common_linear
from tpu_inference.layers.common import sharding as common_sharding
from tpu_inference.layers.common.sharding import (MESH_AXIS_NAMES,
                                                  ShardingAxisName,
                                                  ShardingAxisNameBase)

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


@requires_8_devices
def test_keeps_tokens_sharded_guards(monkeypatch):
    keeps = common_sharding.keeps_tokens_sharded
    mesh = _make_mesh(attn_dp=2, model=4)
    # Enabled under attention DP + attention TP > 1 with GMM_EP (default).
    assert keeps(mesh, enable_sp=True)
    assert not keeps(mesh, enable_sp=False)
    assert not keeps(None, enable_sp=True)
    # Disabled when another EP MoE backend is selected; they ignore
    # keep_token_sharding.
    for flag in ("USE_MOE_EP_KERNEL", "USE_FUSED_EP_MOE_V2",
                 "USE_ROUTING_MOE"):
        monkeypatch.setenv(flag, "1")
        assert not keeps(mesh, enable_sp=True), flag
        monkeypatch.delenv(flag, raising=False)
    assert keeps(mesh, enable_sp=True)
    # Only under attention DP with attention TP > 1.
    assert not keeps(_make_mesh(model=8), enable_sp=True)
    assert not keeps(_make_mesh(attn_dp=8), enable_sp=True)


@requires_8_devices
def test_sp_token_axes():
    # What VllmQuantLinearConfig.get_sp_token_axes returns.
    mesh = _make_mesh(attn_dp=2, model=4)
    get_axes = common_sharding.sp_token_axes
    assert get_axes(mesh, True, 64) == ShardingAxisName.EXPERT_DATA
    # Token count not divisible by EXPERT_DATA (8): usual ATTN_DATA.
    assert get_axes(mesh, True, 12) is None
    assert get_axes(mesh, False, 64) is None


@requires_8_devices
def test_sp_token_sharding_linears():
    """Replicated / column / row-parallel matmuls with token-sharded ends
    match the ATTN_DATA path; bf16 and FP8 (FP8 all-gather) weights."""
    mesh = _make_mesh(attn_dp=2, model=4)
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
            # ba_out / TP = 2 columns per shard: small-weight path.
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
        mesh, common_sharding.keeps_tokens_sharded(mesh, True),
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
