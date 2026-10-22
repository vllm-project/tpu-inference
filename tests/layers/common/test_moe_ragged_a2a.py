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
"""Tests for MOE_GMM_EP_RAGGED_A2A, the GMM_EP ragged all-to-all MoE dispatch
and combine.

The GMM kernel is replaced with a jnp reference so the routing, the ragged
all-to-all bookkeeping (including the dropless extra rounds of
MOE_RAGGED_A2A_DROPLESS) and the combine are checked against a dense MoE
reference. The tests need 8 devices: on TPU they use the real
ragged_all_to_all; on CPU (e.g. XLA_FLAGS=--xla_force_host_platform_device_count=8)
ragged_all_to_all is emulated with an all_gather, since XLA:CPU may not
implement it.
"""

from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from tpu_inference.layers.common import fused_moe_gmm
from tpu_inference.layers.common.sharding import (MESH_AXIS_NAMES,
                                                  ShardingAxisName,
                                                  ShardingAxisNameBase)

_NUM_DEVICES = len(jax.devices())
_ON_CPU = jax.devices()[0].platform == "cpu"

requires_8_devices = pytest.mark.skipif(_NUM_DEVICES < 8,
                                        reason="needs 8 devices")

# hidden, intermediate, experts, top-k. 16 experts over 8 devices gives 2
# local experts per device.
_H, _F, _E, _K = 128, 32, 16, 4


def _ref_gmm(lhs,
             rhs,
             rhs_scale,
             rhs_bias,
             group_sizes,
             group_offset,
             fuse_act=None,
             preferred_element_type=None):
    """jnp stand-in for gmm_wrapper: rows are grouped by group_sizes."""
    del rhs_scale, rhs_bias, group_offset
    ends = jnp.cumsum(group_sizes)
    gid = jnp.searchsorted(ends, jnp.arange(lhs.shape[0]), side="right")
    gid = jnp.minimum(gid, rhs.shape[0] - 1)
    out = jnp.einsum("mh,mhf->mf", lhs.astype(jnp.float32),
                     rhs[gid].astype(jnp.float32))
    if fuse_act:
        f = out.shape[1] // 2
        out = jax.nn.silu(out[:, :f]) * out[:, f:]
    return out.astype(preferred_element_type or lhs.dtype)


def _dense_ref(x, ids, wts, w1, w2):
    x = x.astype(jnp.float32)
    out = jnp.zeros_like(x)
    for k in range(ids.shape[1]):
        h = jnp.einsum("th,thf->tf", x, w1[ids[:, k]].astype(jnp.float32))
        f = h.shape[1] // 2
        h = jax.nn.silu(h[:, :f]) * h[:, f:]
        y = jnp.einsum("tf,tfh->th", h, w2[ids[:, k]].astype(jnp.float32))
        out = out + wts[:, k:k + 1] * y
    return out


def _emulated_ragged_all_to_all(operand, output, input_offsets, send_sizes,
                                output_offsets, recv_sizes, *, axis_name):
    """ragged_all_to_all via all_gather + masked scatter.

    Slice j of each sender goes to device j // slices_per_device.
    """
    del recv_sizes
    n = jax.lax.axis_size(axis_name)
    slices_per_device = input_offsets.shape[0] // n
    me = jax.lax.axis_index(axis_name)
    all_op = jax.lax.all_gather(operand, axis_name)
    all_in = jax.lax.all_gather(input_offsets, axis_name)
    all_size = jax.lax.all_gather(send_sizes, axis_name)
    all_out = jax.lax.all_gather(output_offsets, axis_name)
    r = jnp.arange(operand.shape[0])
    for s in range(n):
        for jj in range(slices_per_device):
            j = me * slices_per_device + jj
            size, src, dst = all_size[s, j], all_in[s, j], all_out[s, j]
            src_rows = jnp.minimum(src + r, operand.shape[0] - 1)
            idx = jnp.where(r < size, dst + r, output.shape[0])
            output = output.at[idx].set(all_op[s][src_rows], mode="drop")
    return output


@pytest.fixture(autouse=True)
def patched_moe(monkeypatch):
    """Multi-axis sharding names, jnp GMM and (on CPU) emulated a2a."""
    monkeypatch.setattr(ShardingAxisName, "_cls", ShardingAxisNameBase)
    monkeypatch.setattr(fused_moe_gmm, "gmm_wrapper", _ref_gmm)
    if _ON_CPU:
        monkeypatch.setattr(jax.lax, "ragged_all_to_all",
                            _emulated_ragged_all_to_all)
    for name in ("MOE_GMM_EP_RAGGED_A2A", "MOE_RAGGED_A2A_CAPACITY_FACTOR",
                 "MOE_RAGGED_A2A_DROPLESS", "FORCE_MOE_RANDOM_ROUTING"):
        monkeypatch.delenv(name, raising=False)
    yield


def _force_row_split(monkeypatch, row_split):
    # _H is too small for the hardware-derived split, so force it.
    monkeypatch.setattr(fused_moe_gmm, "_ragged_a2a_row_split",
                        lambda hidden_size: row_split)


def _make_mesh():
    shape = tuple({
        "attn_dp": 2,
        "model": 4
    }.get(a, 1) for a in MESH_AXIS_NAMES)
    return Mesh(np.array(jax.devices()[:8]).reshape(shape), MESH_AXIS_NAMES)


def _make_inputs(t, skew=False):
    keys = jax.random.split(jax.random.PRNGKey(0), 4)
    x = jax.random.normal(keys[0], (t, _H), jnp.float32)
    logits = jax.random.normal(keys[1], (t, _E))
    if skew:
        # Every token picks experts 0 and 1, both on device 0, so device 0
        # receives 2 * t rows.
        logits = logits.at[:, :2].add(10.0)
    wts, ids = jax.lax.top_k(jax.nn.softmax(logits), _K)
    wts = wts / wts.sum(-1, keepdims=True)
    w1 = jax.random.normal(keys[2], (_E, _H, 2 * _F)) / np.sqrt(_H)
    w2 = jax.random.normal(keys[3], (_E, _F, _H)) / np.sqrt(_F)
    return x, logits, ids, wts, w1, w2


def _run_ragged(mesh, x, ids, wts, w1, w2):
    tok = NamedSharding(mesh, P(ShardingAxisName.EXPERT_DATA, None))
    ep = NamedSharding(mesh, P(ShardingAxisName.EXPERT))

    @jax.jit
    def run(x, ids, wts, w1, w2):
        return fused_moe_gmm._ragged_a2a_expert_parallel_gmm(x,
                                                             ids,
                                                             wts,
                                                             w1,
                                                             None,
                                                             None,
                                                             w2,
                                                             None,
                                                             None,
                                                             mesh=mesh,
                                                             activation="silu",
                                                             topk=_K)

    return np.asarray(
        run(jax.device_put(x, tok), jax.device_put(ids, tok),
            jax.device_put(wts, tok), jax.device_put(w1, ep),
            jax.device_put(w2, ep)))


@requires_8_devices
@pytest.mark.parametrize("t", [64, 1024],
                         ids=["per_peer_slices", "per_expert_slices"])
@pytest.mark.parametrize("row_split", [1, 8])
@pytest.mark.parametrize("capacity_factor, padding", [
    (2.0, False),
    (2.0, True),
    (8.0, False),
],
                         ids=["default", "padding", "large_capacity"])
def test_matches_dense(monkeypatch, capacity_factor, padding, row_split, t):
    # t=1024 -> 128 tokens / device takes the per-expert-slice (prefill)
    # path; t=64 takes the per-peer-slice (decode) path.
    _force_row_split(monkeypatch, row_split)
    monkeypatch.setenv("MOE_RAGGED_A2A_CAPACITY_FACTOR", str(capacity_factor))
    x, _, ids, wts, w1, w2 = _make_inputs(t)
    ref_ids = ids
    if padding:
        valid = (jnp.arange(t) < t - 5)[:, None]
        ids = jnp.where(valid, ids, _E)
        wts = jnp.where(valid, wts, 0.0)
    ref = np.asarray(_dense_ref(x, ref_ids, wts, w1, w2))

    out = _run_ragged(_make_mesh(), x, ids, wts, w1, w2)
    np.testing.assert_allclose(out, ref, atol=1e-4, rtol=1e-4)


@requires_8_devices
@pytest.mark.parametrize("t", [512, 1024],
                         ids=["per_peer_slices", "per_expert_slices"])
@pytest.mark.parametrize("row_split", [1, 8])
def test_dropless_skewed_routing(monkeypatch, row_split, t):
    # Capacity factor 0.5 gives a receive capacity of 512 rows (rounded up to
    # a multiple of 512) while device 0 receives 2 * t rows: t=512 needs 2
    # rounds, t=1024 needs 4.
    _force_row_split(monkeypatch, row_split)
    monkeypatch.setenv("MOE_RAGGED_A2A_CAPACITY_FACTOR", "0.5")
    mesh = _make_mesh()
    x, _, ids, wts, w1, w2 = _make_inputs(t, skew=True)
    assert np.all(np.asarray(ids[:, :2]) < 2)
    ref = np.asarray(_dense_ref(x, ids, wts, w1, w2))

    out = _run_ragged(mesh, x, ids, wts, w1, w2)
    np.testing.assert_allclose(out, ref, atol=1e-4, rtol=1e-4)

    # Without the extra rounds, rows beyond the capacity are dropped and the
    # output no longer matches.
    monkeypatch.setenv("MOE_RAGGED_A2A_DROPLESS", "0")
    dropped = _run_ragged(mesh, x, ids, wts, w1, w2)
    assert np.max(np.abs(dropped - ref)) > 1e-2


@requires_8_devices
@pytest.mark.parametrize("all_gather_fp8", [False, True])
@pytest.mark.parametrize("padding", [False, True])
def test_fused_moe_func_dispatch(monkeypatch, padding, all_gather_fp8):
    """End to end through fused_moe_func with MOE_GMM_EP_RAGGED_A2A=1.

    An FP8 dispatch request is not supported by the ragged path and runs in
    the input dtype (with a warning), so it matches the reference as tightly
    as the plain case.
    """
    _force_row_split(monkeypatch, 8)
    monkeypatch.setenv("MOE_GMM_EP_RAGGED_A2A", "1")
    t = 64
    x, logits, ids, wts, w1, w2 = _make_inputs(t)
    num_valid = t - 6 if padding else t
    valid = (jnp.arange(t) < num_valid)[:, None]
    ref = np.asarray(_dense_ref(x, ids, jnp.where(valid, wts, 0.0), w1, w2))

    fused_moe_gmm.fused_moe_func.clear_cache()
    try:
        out = fused_moe_gmm.fused_moe_func(
            hidden_states=x,
            w1=w1,
            w2=w2,
            w1_scale=None,
            w2_scale=None,
            w1_bias=None,
            w2_bias=None,
            gating_output=logits,
            topk=_K,
            renormalize=True,
            mesh=_make_mesh(),
            use_ep=True,
            activation="silu",
            scoring_fn="softmax",
            all_gather_fp8=all_gather_fp8,
            num_valid_tokens=jnp.int32(num_valid) if padding else None,
        )
    finally:
        fused_moe_gmm.fused_moe_func.clear_cache()
    out = np.asarray(out)
    assert out.shape == (t, _H)
    np.testing.assert_allclose(out, ref, atol=1e-4, rtol=1e-4)
    if padding:
        np.testing.assert_array_equal(out[num_valid:], 0.0)


@pytest.mark.parametrize(
    "sublanes, lanes, hidden, expected",
    [
        (8, 128, 8192, 8),  # TPU v5e / v6e / v7x
        (16, 128, 8192, 16),  # 16-sublane TPUs
        (8, 128, 2048, 8),
        (16, 128, 1024, 8),  # 1024 / 16 = 64 < 128 lanes -> halve
        (8, 128, 384, 1),  # no power-of-two split keeps 128-lane rows
    ])
def test_row_split_from_tpu_info(sublanes, lanes, hidden, expected):
    info = mock.Mock(num_sublanes=sublanes, num_lanes=lanes)
    with mock.patch.object(fused_moe_gmm.pltpu,
                           "get_tpu_info",
                           return_value=info):
        assert fused_moe_gmm._ragged_a2a_row_split(hidden) == expected


def test_row_split_without_tpu():
    with mock.patch.object(fused_moe_gmm.pltpu,
                           "get_tpu_info",
                           side_effect=ValueError("no TPU")):
        assert fused_moe_gmm._ragged_a2a_row_split(8192) == 8
