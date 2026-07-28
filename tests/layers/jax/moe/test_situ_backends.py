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
"""SiTU reaches the eager MoE expert paths.

The unfused (DENSE_MAT) backend is what Kimi-K3 routes through, so its two
forward variants have to apply SiTU to both the gate and the up branch. The
fused/GMM kernels only take a unary activation name today, hence these tests
pin the eager paths.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P

from tpu_inference.layers.common.process_weights.moe_weights import \
    UnfusedMoEWeights
from tpu_inference.layers.jax.activation import (SITU_BETA, SITU_LINEAR_BETA,
                                                 apply_gated_activation,
                                                 situ_params)
from tpu_inference.layers.jax.moe.dense_moe import (
    dense_moe_fwd, dense_moe_fwd_preapply_router_weights)

T, D, E, F, TOPK = 6, 8, 4, 16, 2


def _situ_numpy(gate, up, beta=SITU_BETA, linear_beta=SITU_LINEAR_BETA):
    """Straight transcription of HF moonshotai/Kimi-K3 SituAndMul, fp64."""
    gate = np.asarray(gate, dtype=np.float64)
    up = np.asarray(up, dtype=np.float64)
    situ = beta * np.tanh(gate / beta) * (1.0 / (1.0 + np.exp(-gate)))
    if linear_beta is not None:
        up = linear_beta * np.tanh(up / linear_beta)
    return situ * up


def _silu_numpy(gate, up):
    gate = np.asarray(gate, dtype=np.float64)
    up = np.asarray(up, dtype=np.float64)
    return gate / (1.0 + np.exp(-gate)) * up


class _FakeLayer:
    """Minimal stand-in for a JaxMoE carrying the SiTU betas."""

    def __init__(self, **attrs):
        for k, v in attrs.items():
            setattr(self, k, v)


def _fixture(seed=0):
    rng = np.random.default_rng(seed)
    x_TD = rng.standard_normal((T, D), dtype=np.float32)
    w1 = rng.standard_normal((E, D, F), dtype=np.float32)
    w2 = rng.standard_normal((E, D, F), dtype=np.float32)
    w3 = rng.standard_normal((E, F, D), dtype=np.float32)
    # Dense router weights: top-2 of 4 experts, renormalized.
    scores = rng.random((T, E))
    keep = np.argsort(-scores, axis=1)[:, :TOPK]
    full_TE = np.zeros((T, E), dtype=np.float32)
    for t in range(T):
        w = scores[t, keep[t]]
        full_TE[t, keep[t]] = w / w.sum()
    weights = UnfusedMoEWeights(
        w1_weight=jnp.asarray(w1),
        w1_weight_scale=None,
        w1_bias=None,
        w2_weight=jnp.asarray(w2),
        w2_weight_scale=None,
        w2_bias=None,
        w3_weight=jnp.asarray(w3),
        w3_weight_scale=None,
        w3_bias=None,
    )
    return x_TD, w1, w2, w3, full_TE, weights


def _reference(x_TE_D, w1, w2, w3, act_fn):
    """Expert MLPs in fp64: x[.,E,D] @ w1/w2 -> act -> @ w3."""
    gate = np.einsum('TED,EDF->TEF', x_TE_D, w1.astype(np.float64))
    up = np.einsum('TED,EDF->TEF', x_TE_D, w2.astype(np.float64))
    fuse = act_fn(gate, up)
    return np.einsum('TEF,EFD->TED', fuse, w3.astype(np.float64))


@pytest.fixture(scope="module")
def mesh():
    devices = np.array(jax.devices()[:1]).reshape(1, 1)
    return Mesh(devices, axis_names=("data", "model"))


@pytest.fixture(autouse=True)
def exact_matmuls():
    """These are activation-parity tests, not matmul-precision tests.

    TPU einsums default to a reduced-precision fp32 algorithm (~1e-1 abs error
    at these magnitudes for silu and situ alike), which would swamp the
    activation difference we are measuring.
    """
    with jax.default_matmul_precision("highest"):
        yield


@pytest.mark.parametrize("hidden_act,act_fn", [("situ", _situ_numpy),
                                               ("silu", _silu_numpy)])
def test_dense_moe_fwd_activation(mesh, hidden_act, act_fn):
    x_TD, w1, w2, w3, full_TE, weights = _fixture()
    with mesh:
        got = dense_moe_fwd(weights=weights,
                            x_TD=jnp.asarray(x_TD),
                            cast_dtype=jnp.float32,
                            activation_ffw_td=P(None, None),
                            hidden_act=hidden_act,
                            full_weights_TE=jnp.asarray(full_TE),
                            mesh=mesh)
    x_TED = np.repeat(x_TD[:, None, :].astype(np.float64), E, axis=1)
    down = _reference(x_TED, w1, w2, w3, act_fn)
    ref = np.einsum('TED,TE->TD', down, full_TE.astype(np.float64))
    err = np.abs(np.asarray(got, dtype=np.float64) - ref).max()
    print(f"[observed] dense_moe_fwd {hidden_act} max_abs={err:.3e} "
          f"(ref max_abs={np.abs(ref).max():.3e})")
    np.testing.assert_allclose(np.asarray(got), ref, rtol=2e-4, atol=2e-4)


@pytest.mark.parametrize("hidden_act,act_fn", [("situ", _situ_numpy),
                                               ("silu", _silu_numpy)])
def test_dense_moe_fwd_preapply_activation(mesh, hidden_act, act_fn):
    x_TD, w1, w2, w3, full_TE, weights = _fixture(seed=1)
    with mesh:
        got = dense_moe_fwd_preapply_router_weights(
            weights=weights,
            x_TD=jnp.asarray(x_TD),
            cast_dtype=jnp.float32,
            activation_ffw_ted=P(None, None, None),
            hidden_act=hidden_act,
            full_weights_TE=jnp.asarray(full_TE),
            mesh=mesh)
    x_TED = (np.repeat(x_TD[:, None, :].astype(np.float64), E, axis=1) *
             full_TE.astype(np.float64)[..., None])
    ref = _reference(x_TED, w1, w2, w3, act_fn).sum(axis=1)
    err = np.abs(np.asarray(got, dtype=np.float64) - ref).max()
    print(f"[observed] dense_moe_fwd_preapply {hidden_act} max_abs={err:.3e} "
          f"(ref max_abs={np.abs(ref).max():.3e})")
    np.testing.assert_allclose(np.asarray(got), ref, rtol=2e-4, atol=2e-4)


def test_situ_is_not_silu_on_the_dense_path(mesh):
    """Anti-vacuity: the parity tests above cannot both pass by accident."""
    x_TD, _, _, _, full_TE, weights = _fixture()
    kwargs = dict(weights=weights,
                  x_TD=jnp.asarray(x_TD),
                  cast_dtype=jnp.float32,
                  activation_ffw_td=P(None, None),
                  full_weights_TE=jnp.asarray(full_TE),
                  mesh=mesh)
    with mesh:
        situ = np.asarray(dense_moe_fwd(hidden_act="situ", **kwargs))
        silu = np.asarray(dense_moe_fwd(hidden_act="silu", **kwargs))
    rel = np.abs(situ - silu).max() / max(np.abs(silu).max(), 1e-9)
    print(f"[observed] situ-vs-silu relative gap={rel:.3f}")
    assert rel > 0.1, "situ and silu outputs are indistinguishable"


def test_dense_moe_fwd_honors_custom_betas(mesh):
    """Betas must be threaded, not silently defaulted to the K3 constants."""
    x_TD, w1, w2, w3, full_TE, weights = _fixture(seed=2)
    beta, linear_beta = 1.0, 2.0
    with mesh:
        got = dense_moe_fwd(weights=weights,
                            x_TD=jnp.asarray(x_TD),
                            cast_dtype=jnp.float32,
                            activation_ffw_td=P(None, None),
                            hidden_act="situ",
                            full_weights_TE=jnp.asarray(full_TE),
                            mesh=mesh,
                            situ_beta=beta,
                            situ_linear_beta=linear_beta)
    x_TED = np.repeat(x_TD[:, None, :].astype(np.float64), E, axis=1)
    down = _reference(
        x_TED, w1, w2, w3,
        lambda g, u: _situ_numpy(g, u, beta=beta, linear_beta=linear_beta))
    ref = np.einsum('TED,TE->TD', down, full_TE.astype(np.float64))
    err = np.abs(np.asarray(got, dtype=np.float64) - ref).max()
    print(f"[observed] custom-beta dense_moe_fwd max_abs={err:.3e}")
    np.testing.assert_allclose(np.asarray(got), ref, rtol=2e-4, atol=2e-4)
    # And the K3 defaults would give a materially different answer.
    default = _reference(x_TED, w1, w2, w3, _situ_numpy)
    default = np.einsum('TED,TE->TD', default, full_TE.astype(np.float64))
    assert np.abs(default - ref).max() > 1e-2


def test_apply_gated_activation_dispatch():
    rng = np.random.default_rng(3)
    gate = jnp.asarray(rng.standard_normal((5, 7), dtype=np.float32) * 3)
    up = jnp.asarray(rng.standard_normal((5, 7), dtype=np.float32) * 3)

    situ = np.asarray(apply_gated_activation("situ", gate, up))
    np.testing.assert_allclose(situ, _situ_numpy(gate, up), atol=1e-4)

    silu = np.asarray(apply_gated_activation("silu", gate, up))
    np.testing.assert_allclose(silu, _silu_numpy(gate, up), atol=1e-4)

    # Unary activations ignore the SiTU betas entirely.
    same = np.asarray(
        apply_gated_activation("silu",
                               gate,
                               up,
                               situ_beta=0.5,
                               situ_linear_beta=0.5))
    np.testing.assert_array_equal(silu, same)


def test_situ_params_reads_layer_then_defaults():
    assert situ_params(_FakeLayer()) == (SITU_BETA, SITU_LINEAR_BETA)
    assert situ_params(_FakeLayer(situ_beta=1.5)) == (1.5, SITU_LINEAR_BETA)
    assert situ_params(_FakeLayer(situ_beta=1.5,
                                  situ_linear_beta=9.0)) == (1.5, 9.0)
