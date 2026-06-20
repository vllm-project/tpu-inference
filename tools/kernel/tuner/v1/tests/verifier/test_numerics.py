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
"""Unit tests for the multi-tier numerics check."""

import math

import jax.numpy as jnp
import numpy as np
import pytest

from tools.kernel.tuner.v1.verifier.numerics import (COSINE_FLOOR_DEFAULT,
                                                     check_many, check_one)
from tools.kernel.tuner.v1.verifier.reference_oracle import rpa_v3_tolerance


def test_passes_identical_arrays():
    a = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    rep = check_one(a, a, atol=1e-5, rtol=1e-5)
    assert rep.passed
    assert rep.max_abs_diff == 0.0
    assert rep.cosine == pytest.approx(1.0)


def test_rejects_shape_mismatch():
    a = np.zeros((4, ))
    b = np.zeros((5, ))
    rep = check_one(a, b, atol=1.0, rtol=1.0)
    assert not rep.passed
    assert "shape mismatch" in rep.reason


def test_rejects_nan():
    actual = np.array([1.0, 2.0, np.nan, 4.0], dtype=np.float32)
    ref = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    rep = check_one(actual, ref, atol=1.0, rtol=1.0)
    assert not rep.passed
    assert rep.nan_count == 1
    assert rep.max_abs_diff == math.inf
    assert "NaN" in rep.reason


def test_rejects_inf():
    actual = np.array([1.0, 2.0, np.inf, 4.0], dtype=np.float32)
    ref = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    rep = check_one(actual, ref, atol=1.0, rtol=1.0)
    assert not rep.passed
    assert rep.inf_count == 1
    assert "Inf" in rep.reason


def test_rejects_dropped_layer_via_cosine():
    """A kernel that returns the first half of the reference (rest zeros)
    still has a finite max-abs-diff that allclose with a loose atol might
    accept. The cosine floor catches it."""
    ref = np.linspace(1.0, 10.0, 100, dtype=np.float32)
    actual = ref.copy()
    actual[50:] = 0.0
    # Use a loose tolerance that would otherwise pass.
    rep = check_one(actual,
                    ref,
                    atol=10.0,
                    rtol=10.0,
                    cosine_floor=COSINE_FLOOR_DEFAULT)
    assert not rep.passed
    assert "cosine" in rep.reason
    assert rep.cosine < COSINE_FLOOR_DEFAULT


def test_rejects_when_allclose_fails_but_cosine_passes():
    """A scaled output (close cosine, far in magnitude) is rejected by
    allclose even when cosine ≈ 1."""
    ref = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    actual = ref * 10.0
    rep = check_one(actual, ref, atol=0.01, rtol=0.01)
    assert not rep.passed
    assert "allclose" in rep.reason
    assert rep.cosine == pytest.approx(1.0)


def test_check_many_short_circuits_on_first_failure():
    good = np.array([1.0, 2.0], dtype=np.float32)
    bad = np.array([1.0, 100.0], dtype=np.float32)
    ref = np.array([1.0, 2.0], dtype=np.float32)
    rep = check_many([bad, good], [ref, ref], atol=0.01, rtol=0.01)
    assert not rep.passed
    # Either tier (cosine or allclose) is a valid first-failure point; what
    # matters is that the offending pair is the first one (max_abs_diff
    # corresponds to the bad pair).
    assert rep.max_abs_diff == pytest.approx(98.0)
    assert ("cosine" in rep.reason) or ("allclose" in rep.reason)


def test_check_many_short_circuits_at_allclose_tier():
    """When cosine is near-1 but allclose fails (close-direction, wrong
    magnitude), the failure should be attributed to the allclose tier."""
    good = np.array([1.0, 2.0], dtype=np.float32)
    bad = np.array([1.0 + 5.0, 2.0 + 10.0], dtype=np.float32)
    ref = np.array([1.0, 2.0], dtype=np.float32)
    rep = check_many([bad, good], [ref, ref], atol=0.01, rtol=0.01)
    assert not rep.passed
    assert "allclose" in rep.reason


def test_check_many_aggregates_on_success():
    a = np.array([1.0, 2.0], dtype=np.float32)
    b = np.array([3.0, 4.0], dtype=np.float32)
    ref_a = np.array([1.0001, 2.0], dtype=np.float32)
    ref_b = np.array([3.0, 4.0002], dtype=np.float32)
    rep = check_many([a, b], [ref_a, ref_b], atol=0.01, rtol=0.01)
    assert rep.passed
    assert rep.max_abs_diff >= 0.0002 - 1e-6


def test_check_many_raises_on_length_mismatch():
    a = np.zeros((4, ))
    with pytest.raises(ValueError, match="same length"):
        check_many([a, a], [a], atol=1.0, rtol=1.0)


@pytest.mark.parametrize(
    "dtype,expected_atol",
    [
        (jnp.float32, 0.15),
        (jnp.bfloat16, 0.2),
        (jnp.float8_e4m3fn, 0.2),
        # No int4 dtype in jnp; the table includes 4-bit for completeness.
    ],
)
def test_rpa_v3_tolerance_matches_test_table(dtype, expected_atol):
    atol, rtol = rpa_v3_tolerance(dtype)
    assert atol == expected_atol
    assert rtol == expected_atol


def test_rpa_v3_tolerance_rejects_unsupported_dtype():
    with pytest.raises(ValueError, match="Unsupported dtype"):
        rpa_v3_tolerance(jnp.float64)


def test_cosine_handles_zero_vectors():
    z = np.zeros((4, ), dtype=np.float32)
    rep = check_one(z, z, atol=1e-5, rtol=1e-5)
    assert rep.passed
    # The cosine convention here is 1.0 for zero-vs-zero.
    assert rep.cosine == pytest.approx(1.0)
