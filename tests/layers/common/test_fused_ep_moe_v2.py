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
"""Tests for MoEBackend.FUSED_EP_MOE_V2, tokamax's fused EP MoE kernel.

The kernel quantizes differently from the GMM path (fp8 token rows, fp8
intermediate, fp8 results), so tests compare it to a dense f32 reference instead
of matching GMM bit for bit. moe_apply raises for layers the kernel can't run.
"""

from types import SimpleNamespace
from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from tests.layers.common.test_moe_hierarchical_dispatch import (
    _make_mesh, _mesh_id, requires_8_devices)
from tpu_inference.layers.common import fused_ep_moe_v2, moe
from tpu_inference.layers.common.moe import MoEBackend, moe_apply
from tpu_inference.layers.common.process_weights.moe_weights import (
    FusedMoEWeights, _get_moe_weight_shardings)
from tpu_inference.layers.common.sharding import (ShardingAxisName,
                                                  ShardingAxisNameBase)

NUM_EXPERTS, HIDDEN, INTER, TOPK = 16, 256, 256, 2
_SUPPORTED_MESHES = [
    dict(attn_dp=2, model=4),
    dict(attn_dp=4, model=2),
    dict(attn_dp=2, attn_dp_expert=2, model=2),
]
# fp8 on every mesh and bf16 on one: the weight dtype doesn't change how rows
# move between devices.
_KERNEL_CASES = ([(sizes, True) for sizes in _SUPPORTED_MESHES] +
                 [(_SUPPORTED_MESHES[0], False)])


@pytest.fixture(autouse=True)
def base_sharding_axes():
    """Use the multi-axis names (attn_dp, ...), as the other MoE tests do."""
    with mock.patch.object(ShardingAxisName, "_cls", ShardingAxisNameBase):
        yield


@pytest.fixture
def tokamax_kernel():
    """Skip tests that run the kernel if the installed tokamax doesn't have
    it."""
    pytest.importorskip("tokamax._src.ops.experimental.fused_moe.layer")


def _weights(mesh: Mesh, fp8: bool, block_scales: bool = False):
    """Random w1 [E, H, 2I] (gate | up) and w2 [E, I, H], sharded over the
    experts. fp8 weights also get scales [E, blocks, 1, N]; one block means
    per-channel."""
    k1, k2 = jax.random.split(jax.random.key(0))
    w1 = jax.random.normal(k1, (NUM_EXPERTS, HIDDEN, 2 * INTER)) / HIDDEN**0.5
    w2 = jax.random.normal(k2, (NUM_EXPERTS, INTER, HIDDEN)) / INTER**0.5
    ep = NamedSharding(mesh, P(ShardingAxisName.EXPERT))
    if not fp8:
        return [jax.device_put(w.astype(jnp.bfloat16), ep)
                for w in (w1, w2)] + [None, None]
    blocks = 2 if block_scales else 1

    def quantize(w):
        e, k, n = w.shape
        wb = w.reshape(e, blocks, k // blocks, n)
        scale = jnp.max(jnp.abs(wb), axis=2, keepdims=True) / 448.0
        q = (wb / scale).astype(jnp.float8_e4m3fn).reshape(e, k, n)
        return q, scale  # scale: [E, blocks, 1, N]

    (q1, s1), (q2, s2) = quantize(w1), quantize(w2)
    return [jax.device_put(a, ep) for a in (q1, q2, s1, s2)]


def _inputs(mesh: Mesh, num_tokens: int):
    k1, k2 = jax.random.split(jax.random.key(1))
    rows = NamedSharding(mesh, P(ShardingAxisName.ATTN_DATA, None))
    x = jax.random.normal(k1, (num_tokens, HIDDEN), jnp.bfloat16)
    gating = jax.random.normal(k2, (num_tokens, NUM_EXPERTS), jnp.bfloat16)
    return jax.device_put(x, rows), jax.device_put(gating, rows)


def _dense_reference(x, gating, w1, w2, s1, s2):
    """Reference MoE in f32: softmax, top-k, renormalize, then every expert on
    every token."""
    x = np.asarray(x, np.float32)
    w1, w2 = np.asarray(w1, np.float32), np.asarray(w2, np.float32)
    if s1 is not None:  # dequantize: repeat each block's scale over its rows
        s1, s2 = np.asarray(s1), np.asarray(s2)
        w1 = w1 * np.repeat(s1[:, :, 0], HIDDEN // s1.shape[1], axis=1)
        w2 = w2 * np.repeat(s2[:, :, 0], INTER // s2.shape[1], axis=1)
    g = np.asarray(gating, np.float32)
    p = np.exp(g - g.max(-1, keepdims=True))
    p /= p.sum(-1, keepdims=True)
    idx = np.argsort(-p, axis=-1, kind="stable")[:, :TOPK]
    wt = np.take_along_axis(p, idx, -1)
    wt /= wt.sum(-1, keepdims=True)
    out = np.zeros_like(x)
    for e in range(NUM_EXPERTS):
        a = x @ w1[e]
        h = a[:, :INTER] / (1 + np.exp(-a[:, :INTER])) * a[:, INTER:]
        out += np.sum(np.where(idx == e, wt, 0.0), -1)[:, None] * (h @ w2[e])
    return out


def _run_kernel(mesh: Mesh, x, gating, w1, w2, s1, s2):
    with jax.set_mesh(mesh):
        f = jax.jit(lambda *a: fused_ep_moe_v2.fused_ep_moe_v2_func(
            *a, topk=TOPK, renormalize=True, mesh=mesh, scoring_fn="softmax"))
        return jax.block_until_ready(f(x, w1, w2, s1, s2, None, None, gating))


def _rel_l2(a, want):
    a = np.asarray(a, np.float64)
    return float(np.linalg.norm(a - want) / np.linalg.norm(want))


def test_weights_like():
    """FUSED_EP_MOE_V2 uses GMM_EP's weight layout; every backend that isn't
    a GMM_EP-layout kernel uses its own."""
    assert MoEBackend.FUSED_EP_MOE_V2.weights_like() is MoEBackend.GMM_EP
    for backend in MoEBackend:
        if backend not in (MoEBackend.FUSED_EP_MOE_V2, MoEBackend.ROUTING_MOE):
            assert backend.weights_like() is backend
    assert MoEBackend.FUSED_EP_MOE_V2 in MoEBackend.fused_moe_backends()


@requires_8_devices
def test_weight_shardings_match_gmm_ep():
    mesh = _make_mesh(attn_dp=4, model=2)
    w = FusedMoEWeights(*(None, ) * 6)
    assert (_get_moe_weight_shardings(w, MoEBackend.FUSED_EP_MOE_V2,
                                      mesh) == _get_moe_weight_shardings(
                                          w, MoEBackend.GMM_EP, mesh))


@requires_8_devices
@pytest.mark.parametrize("sizes, groups", [
    (dict(attn_dp=2, model=4), ((0, 1, 2, 3), (4, 5, 6, 7))),
    (dict(attn_dp=4, model=2), ((0, 1), (2, 3), (4, 5), (6, 7))),
    (dict(attn_dp=2, attn_dp_expert=2, model=2), ((0, 1), (2, 3), (4, 5),
                                                  (6, 7))),
],
                         ids=lambda v: _mesh_id(v)
                         if isinstance(v, dict) else "")
def test_layout(sizes, groups):
    """The 1-D mesh keeps the device order, and devices with the same
    attention-data coordinates form a replica group."""
    mesh = _make_mesh(**sizes)
    mesh_1d, got, reason = fused_ep_moe_v2.ep_layout(mesh)
    assert reason is None
    assert list(mesh_1d.devices) == list(mesh.devices.reshape(-1))
    assert got == groups


@requires_8_devices
@pytest.mark.usefixtures("tokamax_kernel")
@pytest.mark.parametrize("sizes, fp8",
                         _KERNEL_CASES,
                         ids=lambda v: _mesh_id(v)
                         if isinstance(v, dict) else ("fp8" if v else "bf16"))
def test_matches_dense_reference(sizes, fp8):
    """The output is close to a dense f32 reference and sharded like the
    input."""
    mesh = _make_mesh(**sizes)
    x, gating = _inputs(mesh, 2048)
    w = _weights(mesh, fp8)
    want = _dense_reference(x, gating, *w)
    out = _run_kernel(mesh, x, gating, *w)
    assert out.shape == x.shape and out.dtype == x.dtype
    assert out.sharding.is_equivalent_to(x.sharding, x.ndim)
    error = _rel_l2(out, want)
    assert error < (0.07 if fp8 else 0.01), error


@requires_8_devices
@pytest.mark.parametrize("case, want", [
    (None, None),
    ("block_scales", "one weight scale per output channel"),
    ("w2_block_scales", "one weight scale per output channel"),
    ("dtypes_differ", "dtypes differ"),
    ("bf16_scaled", "unscaled"),
    ("hidden", "hidden size"),
    ("inter_padded", "intermediate size"),
    ("inter_mismatch", "differs"),
    ("experts", "split evenly"),
    ("data_axis", "expert axes"),
    ("hash_topk", "by hash"),
    ("random_routing", "FORCE_MOE_RANDOM_ROUTING"),
    ("approx_topk", "MOE_APPROX_TOPK"),
    ("activation", "only silu"),
    ("scoring_fn", "doesn't score"),
])
def test_unsupported_reason(case, want):
    sizes = (dict(data=2, attn_dp=2, model=2)
             if case == "data_axis" else dict(attn_dp=4, model=2))
    mesh = _make_mesh(**sizes)
    w1, w2, s1, s2 = _weights(mesh, True, block_scales=case == "block_scales")
    if case == "w2_block_scales":
        s2 = _weights(mesh, True, block_scales=True)[3]
    elif case == "dtypes_differ":
        w2 = w2.astype(jnp.bfloat16)
    elif case == "bf16_scaled":
        w1, w2 = w1.astype(jnp.bfloat16), w2.astype(jnp.bfloat16)
    elif case == "hidden":
        w1 = w1[:, :HIDDEN - 64]
    elif case == "inter_padded":
        # w13 padded to a multiple of 128 and w2 not, as GMM EP loads them.
        w2 = w2[:, :INTER - 64]
    elif case == "inter_mismatch":
        w2 = w2[:, :INTER - 128]
    elif case == "experts":
        w1 = w1[:NUM_EXPERTS - 4]
    with (mock.patch("tpu_inference.envs.FORCE_MOE_RANDOM_ROUTING",
                     case == "random_routing"),
          mock.patch("tpu_inference.envs.MOE_APPROX_TOPK",
                     case == "approx_topk")):
        reason = fused_ep_moe_v2.unsupported_reason(
            mesh, w1, w2, s1, s2,
            "swigluoai" if case == "activation" else "silu",
            "noaux_tc" if case == "scoring_fn" else "softmax",
            jnp.zeros((8, TOPK), jnp.int32) if case == "hash_topk" else None)
    if want is None:
        assert reason is None, reason
    else:
        assert reason is not None and want in reason, reason


def _layer():
    return SimpleNamespace(_get_name=lambda: "experts",
                           activation="silu",
                           top_k=TOPK,
                           renormalize=True,
                           scoring_func="softmax",
                           use_ep=True)


@requires_8_devices
def test_moe_apply_dispatch():
    """moe_apply runs the kernel for layers it supports."""
    mesh = _make_mesh(attn_dp=4, model=2)
    x, gating = _inputs(mesh, 512)
    w1, w2, s1, s2 = _weights(mesh, True)
    weights = FusedMoEWeights(w1, s1, None, w2, s2, None)
    with (mock.patch.object(fused_ep_moe_v2,
                            "fused_ep_moe_v2_func",
                            return_value="kernel") as
          kernel, mock.patch.object(moe, "fused_moe_func") as gmm):
        out = moe_apply(_layer(), x, gating, weights,
                        MoEBackend.FUSED_EP_MOE_V2, mesh, {})
    assert out == "kernel"
    assert kernel.called and not gmm.called


@requires_8_devices
@pytest.mark.parametrize("case, reason", [
    ("block_scales", "one weight scale per output channel"),
    ("hash_topk", "not by hash"),
    ("swiglu_limit", "not silu_and_mul_with_clamp"),
])
def test_moe_apply_refuses(case, reason):
    """moe_apply raises with the reason and runs neither the kernel nor the
    GMM path."""
    mesh = _make_mesh(attn_dp=4, model=2)
    x, gating = _inputs(mesh, 512)
    w1, w2, s1, s2 = _weights(mesh, True, block_scales=case == "block_scales")
    weights = FusedMoEWeights(w1, s1, None, w2, s2, None)
    layer = _layer()
    if case == "swiglu_limit":
        layer.swiglu_limit = 10.0
    extra = ({
        "hash_based_topk_indices": jnp.zeros((512, TOPK), jnp.int32)
    } if case == "hash_topk" else {})
    with (mock.patch.object(fused_ep_moe_v2, "fused_ep_moe_v2_func") as
          kernel, mock.patch.object(moe, "fused_moe_func") as
          gmm, pytest.raises(ValueError, match=reason)):
        moe_apply(layer, x, gating, weights, MoEBackend.FUSED_EP_MOE_V2, mesh,
                  extra)
    assert not kernel.called and not gmm.called
