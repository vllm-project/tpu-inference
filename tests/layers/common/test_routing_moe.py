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
"""Tests for MoEBackend.ROUTING_MOE, tokamax's routing_moe kernel.

With fp8 weights the kernel quantizes the tokens to fp8 too, so tests compare
it to a dense f32 reference instead of matching the GMM path bit for bit.
moe_apply raises for layers the kernel can't run.
"""

from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh

from tests.layers.common.test_fused_ep_moe_v2 import (HIDDEN, INTER,
                                                      NUM_EXPERTS, TOPK,
                                                      _dense_reference,
                                                      _inputs, _layer, _rel_l2,
                                                      _weights)
from tests.layers.common.test_moe_hierarchical_dispatch import (_make_mesh,
                                                                _mesh_id)
from tpu_inference.layers.common import moe, routing_moe
from tpu_inference.layers.common.fused_moe_gmm import select_experts
from tpu_inference.layers.common.moe import MoEBackend, moe_apply
from tpu_inference.layers.common.process_weights.moe_weights import (
    FusedMoEWeights, _get_moe_weight_shardings)
from tpu_inference.layers.common.sharding import (ShardingAxisName,
                                                  ShardingAxisNameBase)

requires_4_devices = pytest.mark.skipif(len(jax.devices()) < 4,
                                        reason="needs 4 devices")

# One attention-data axis with one replica axis, attention data only, replicas
# only, and two replica axes.
_SUPPORTED_MESHES = [
    dict(attn_dp=2, model=2),
    dict(attn_dp=4),
    dict(model=4),
    dict(expert=2, model=2),
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
    pytest.importorskip(
        "tokamax._src.ops.experimental.routing_moe.layer.modular")


def _run_kernel(mesh: Mesh, x, gating, w1, w2, s1, s2, **routing):
    with jax.set_mesh(mesh):
        f = jax.jit(
            lambda *a: routing_moe.routing_moe_func(*a,
                                                    topk=TOPK,
                                                    renormalize=True,
                                                    mesh=mesh,
                                                    activation="silu",
                                                    scoring_fn="softmax",
                                                    **routing))
        return jax.block_until_ready(f(x, w1, w2, s1, s2, gating))


def test_weights_like():
    """ROUTING_MOE uses GMM_EP's weight layout."""
    assert MoEBackend.ROUTING_MOE.weights_like() is MoEBackend.GMM_EP
    assert MoEBackend.ROUTING_MOE in MoEBackend.fused_moe_backends()


def test_weight_shardings_match_gmm_ep():
    mesh = _make_mesh()
    w = FusedMoEWeights(*(None, ) * 6)
    assert (_get_moe_weight_shardings(w, MoEBackend.ROUTING_MOE,
                                      mesh) == _get_moe_weight_shardings(
                                          w, MoEBackend.GMM_EP, mesh))


@requires_4_devices
@pytest.mark.parametrize("sizes, replica", [
    (dict(attn_dp=2, model=2), ("model", )),
    (dict(attn_dp=4), ()),
    (dict(model=4), ("model", )),
    (dict(expert=2, model=2), ("expert", "model")),
],
                         ids=lambda v: _mesh_id(v)
                         if isinstance(v, dict) else "")
def test_layout(sizes, replica):
    """The 1-D mesh keeps the device order; the replica axes are the used
    axes that aren't attention-data axes."""
    mesh = _make_mesh(**sizes)
    mesh_1d, got, reason = routing_moe.ep_layout(mesh)
    assert reason is None
    assert list(mesh_1d.devices) == list(mesh.devices.reshape(-1))
    assert got == replica


@requires_4_devices
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


@requires_4_devices
@pytest.mark.usefixtures("tokamax_kernel")
def test_rows_not_divisible_by_replicas():
    """150 tokens leave 75 rows per attention-data shard, so the rows are
    padded before each replica takes its half."""
    mesh = _make_mesh(attn_dp=2, model=2)
    x, gating = _inputs(mesh, 150)
    w = _weights(mesh, True)
    out = _run_kernel(mesh, x, gating, *w)
    assert out.shape == x.shape
    assert _rel_l2(out, _dense_reference(x, gating, *w)) < 0.07


@pytest.mark.parametrize("case, want", [
    (None, None),
    ("hidden_dtype", "bf16 hidden states"),
    ("block_scales", "one weight scale per output channel"),
    ("w2_block_scales", "one weight scale per output channel"),
    ("dtypes_differ", "dtypes differ"),
    ("bf16_scaled", "unscaled"),
    ("bias", "biases"),
    ("hidden", "hidden size"),
    ("inter_padded", "intermediate sizes"),
    ("inter_128", "intermediate sizes"),
    ("experts", "split evenly"),
    ("data_axis", "expert axes"),
    ("hash_topk", "hash routing"),
    ("approx_topk", "MOE_APPROX_TOPK"),
    ("random_routing", "FORCE_MOE_RANDOM_ROUTING"),
    ("activation", "only silu"),
])
def test_unsupported_reason(case, want):
    """Each check refuses its case with its reason; one device is enough for
    all but the two mesh checks."""
    sizes = {
        "data_axis": dict(data=2),
        "experts": dict(attn_dp=2)
    }.get(case, {})
    if len(jax.devices()) < max(sizes.values(), default=1):
        pytest.skip("needs 2 devices")
    mesh = _make_mesh(**sizes)
    w1, w2, s1, s2 = _weights(mesh, True, block_scales=case == "block_scales")
    if case == "w2_block_scales":
        s2 = _weights(mesh, True, block_scales=True)[3]
    elif case == "dtypes_differ":
        w2 = w2.astype(jnp.bfloat16)
    elif case == "bf16_scaled":
        w1, w2 = w1.astype(jnp.bfloat16), w2.astype(jnp.bfloat16)
    elif case == "hidden":
        w1 = w1[:, :HIDDEN - 128]
    elif case == "inter_padded":
        # w13 padded to a multiple of 128 and w2 not, as GMM EP loads them.
        w2 = w2[:, :INTER - 64]
    elif case == "inter_128":
        # A multiple of 128 that isn't one of 256: GMM EP leaves it as is.
        w1 = jnp.concatenate([w1[..., :INTER - 128], w1[..., INTER:-128]],
                             axis=-1)
        w2 = w2[:, :INTER - 128]
    elif case == "experts":
        w1 = w1[:NUM_EXPERTS - 1]
    with (mock.patch("tpu_inference.envs.FORCE_MOE_RANDOM_ROUTING",
                     case == "random_routing"),
          mock.patch("tpu_inference.envs.MOE_APPROX_TOPK",
                     case == "approx_topk")):
        reason = routing_moe.unsupported_reason(
            mesh, jnp.float32 if case == "hidden_dtype" else jnp.bfloat16, w1,
            w2, s1, s2,
            jnp.zeros((NUM_EXPERTS, 1, 2 * INTER)) if case == "bias" else None,
            None, "swigluoai" if case == "activation" else "silu",
            jnp.zeros((8, TOPK), jnp.int32) if case == "hash_topk" else None)
    if want is None:
        assert reason is None, reason
    else:
        assert reason is not None and want in reason, reason


def test_moe_apply_dispatch():
    """moe_apply runs the kernel for a layer it supports, with the layer's
    weights and routing settings."""
    mesh = _make_mesh()
    x, gating = _inputs(mesh, 512)
    w1, w2, s1, s2 = _weights(mesh, True)
    weights = FusedMoEWeights(w1, s1, None, w2, s2, None)
    extra = dict(e_score_correction_bias=jnp.zeros((NUM_EXPERTS, )))
    with (mock.patch.object(routing_moe,
                            "routing_moe_func",
                            autospec=True,
                            return_value="kernel") as
          kernel, mock.patch.object(moe, "fused_moe_func") as gmm):
        out = moe_apply(_layer(), x, gating, weights, MoEBackend.ROUTING_MOE,
                        mesh, extra)
    assert out == "kernel" and not gmm.called
    kwargs = kernel.call_args.kwargs
    assert kwargs["w1"] is w1 and kwargs["w2_scale"] is s2
    assert kwargs["gating_output"] is gating and kwargs["topk"] == TOPK
    assert kwargs["mesh"] is mesh and kwargs["activation"] == "silu"
    assert kwargs["scoring_fn"] == "softmax"
    assert (kwargs["expert_score_correction_bias"]
            is extra["e_score_correction_bias"])


@pytest.mark.parametrize("extra, want", [
    ({}, "per output channel"),
    ({
        "hash_based_topk_indices": jnp.zeros((8, TOPK), jnp.int32)
    }, "hash routing"),
])
def test_moe_apply_refuses_an_unsupported_layer(extra, want):
    """A layer the kernel can't run raises, naming the reason, rather than
    running on another backend."""
    mesh = _make_mesh()
    x, gating = _inputs(mesh, 512)
    w1, w2, s1, s2 = _weights(mesh, True, block_scales=not extra)
    weights = FusedMoEWeights(w1, s1, None, w2, s2, None)
    with (mock.patch.object(moe, "fused_moe_func") as
          gmm, pytest.raises(ValueError, match=want)):
        moe_apply(_layer(), x, gating, weights, MoEBackend.ROUTING_MOE, mesh,
                  extra)
    assert not gmm.called


def _fake_kernel(x, topk_weights, topk_indices, *args, **kwargs):
    """Stands in for tokamax's routing_moe: writes each row's routing into its
    output row, so the test can see which routing reached which token."""
    del args, kwargs
    out = jnp.zeros(x.shape, jnp.float32)
    out = out.at[:, :TOPK].set(topk_indices.astype(jnp.float32))
    return out.at[:, TOPK:2 * TOPK].set(topk_weights.astype(jnp.float32))


@requires_4_devices
@pytest.mark.usefixtures("tokamax_kernel")
@pytest.mark.parametrize("sizes", _SUPPORTED_MESHES, ids=_mesh_id)
@pytest.mark.parametrize("score_bias", [False, True])
def test_each_token_keeps_its_routing(sizes, score_bias):
    """Routing after the split gives each token the experts and weights
    select_experts picks for it on the unsplit rows."""
    from tokamax._src.ops.experimental.routing_moe.layer import modular
    mesh = _make_mesh(**sizes)
    # Rows that don't divide evenly among the replicas, where the mesh allows.
    num_tokens = 152 if sizes.get("attn_dp", 1) == 4 else 150
    x, gating = _inputs(mesh, num_tokens)
    # f32 logits, so the weights can be compared without bf16 rounding.
    gating = gating.astype(jnp.float32)
    w1, w2, s1, s2 = _weights(mesh, True)
    kwargs = {}
    if score_bias:
        kwargs["expert_score_correction_bias"] = jax.random.normal(
            jax.random.key(2), (NUM_EXPERTS, ))
    with mock.patch.object(modular, "routing_moe", _fake_kernel):
        out = _run_kernel(mesh, x, gating, w1, w2, s1, s2, **kwargs)
    want_weights, want_indices = select_experts(gating, TOPK, True, "softmax",
                                                **kwargs)
    np.testing.assert_array_equal(out[:, :TOPK], want_indices)
    np.testing.assert_allclose(out[:, TOPK:2 * TOPK], want_weights, rtol=1e-6)
