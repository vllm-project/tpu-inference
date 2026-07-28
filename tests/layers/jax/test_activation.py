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
"""SiTU activation parity against the reference formula."""

import jax.numpy as jnp
import numpy as np
import pytest

from tpu_inference.layers.jax.activation import (SITU_BETA, SITU_LINEAR_BETA,
                                                 situ_and_mul)


def _situ_numpy(gate, up, beta=SITU_BETA, linear_beta=SITU_LINEAR_BETA):
    """Straight transcription of HF moonshotai/Kimi-K3 SituAndMul, fp64."""
    gate = gate.astype(np.float64)
    up = up.astype(np.float64)
    situ = beta * np.tanh(gate / beta) * (1.0 / (1.0 + np.exp(-gate)))
    if linear_beta is not None:
        up = linear_beta * np.tanh(up / linear_beta)
    return situ * up


def test_matches_reference_formula():
    rng = np.random.default_rng(0)
    gate = rng.standard_normal((64, 128), dtype=np.float32) * 3
    up = rng.standard_normal((64, 128), dtype=np.float32) * 3
    got = np.asarray(situ_and_mul(jnp.asarray(gate), jnp.asarray(up)))
    ref = _situ_numpy(gate, up)
    err = np.abs(got - ref).max()
    print(f"[observed] situ max_abs={err:.3e}")
    np.testing.assert_allclose(got, ref, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("scale", [1e2, 1e4])
def test_both_branches_soft_clamped(scale):
    """Large inputs must saturate rather than blow up.

    The product is bounded by beta * linear_beta, approached from below as
    the input grows (at gate=up=100 the up branch is only
    25*tanh(4) = 24.98, so the bound is not yet reached).
    """
    limit = SITU_BETA * SITU_LINEAR_BETA
    big = jnp.full((8, 8), scale, dtype=jnp.float32)
    out = np.asarray(situ_and_mul(big, big))
    assert np.isfinite(out).all()
    assert (out <= limit + 1e-4).all(), f"exceeded soft clamp {limit}"
    np.testing.assert_allclose(out,
                               _situ_numpy(np.asarray(big), np.asarray(big)),
                               rtol=1e-5)

    neg = jnp.full((8, 8), -scale, dtype=jnp.float32)
    out_neg = np.asarray(situ_and_mul(neg, neg))
    assert np.isfinite(out_neg).all()
    # gate branch -> -beta * sigmoid(-inf) = 0
    np.testing.assert_allclose(out_neg, 0.0, atol=1e-5)


def test_saturates_toward_bound_as_input_grows():
    limit = SITU_BETA * SITU_LINEAR_BETA
    vals = [
        np.asarray(
            situ_and_mul(jnp.full((1, ), s, jnp.float32),
                         jnp.full((1, ), s, jnp.float32)))[0]
        for s in (10.0, 100.0, 1000.0)
    ]
    assert vals[0] < vals[1] < vals[2] <= limit + 1e-4
    assert abs(vals[2] - limit) < 1e-3


def test_fp32_internals_under_bf16_io():
    """bf16 in / bf16 out, but the math runs in fp32."""
    rng = np.random.default_rng(1)
    gate = rng.standard_normal((32, 64), dtype=np.float32)
    up = rng.standard_normal((32, 64), dtype=np.float32)
    out = situ_and_mul(jnp.asarray(gate, jnp.bfloat16),
                       jnp.asarray(up, jnp.bfloat16))
    assert out.dtype == jnp.bfloat16
    ref = _situ_numpy(gate, up)
    # bf16 has ~3 decimal digits; just check it tracks the reference.
    np.testing.assert_allclose(np.asarray(out, np.float32),
                               ref,
                               atol=5e-2,
                               rtol=5e-2)


def test_linear_beta_none_leaves_up_untouched():
    rng = np.random.default_rng(2)
    gate = rng.standard_normal((16, 16), dtype=np.float32)
    up = rng.standard_normal((16, 16), dtype=np.float32)
    got = np.asarray(
        situ_and_mul(jnp.asarray(gate), jnp.asarray(up), linear_beta=None))
    ref = _situ_numpy(gate, up, linear_beta=None)
    np.testing.assert_allclose(got, ref, atol=1e-5, rtol=1e-5)
