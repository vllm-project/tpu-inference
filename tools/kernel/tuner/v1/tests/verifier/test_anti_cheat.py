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
"""Unit tests for the anti-cheat detectors."""

import numpy as np

from tools.kernel.tuner.v1.verifier.anti_cheat import (
    AntiCheatGuard, cross_trial_independence, detect_constant_output,
    detect_returns_input, detect_zero_output)


def test_detects_zero_output():
    out = np.zeros((4, 4), dtype=np.float32)
    assert detect_zero_output(out)
    out2 = np.array([0.0, 0.0, 0.0, 1e-20], dtype=np.float32)
    assert detect_zero_output(out2)
    out3 = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    assert not detect_zero_output(out3)


def test_detects_constant_output():
    out = np.full((4, 4), 3.14, dtype=np.float32)
    assert detect_constant_output(out)
    # A tiny perturbation should not register as constant for tol=1e-6.
    out2 = out.copy()
    out2[0, 0] += 1e-2
    assert not detect_constant_output(out2)


def test_constant_detector_passes_normal_outputs():
    rng = np.random.default_rng(0)
    out = rng.normal(0, 1, size=(16, 16)).astype(np.float32)
    assert not detect_constant_output(out)


def test_detects_returns_input_bytewise_match():
    inputs = {
        "q": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        "k": np.array([[9.0, 8.0], [7.0, 6.0]], dtype=np.float32),
    }
    # If the kernel returns q verbatim, the detector catches it.
    out = inputs["q"]
    assert detect_returns_input(out, inputs) == "q"


def test_detects_returns_input_respects_skip_keys():
    inputs = {
        "kv_cache": np.array([1.0, 2.0, 3.0], dtype=np.float32),
        "queries": np.array([1.0, 2.0, 3.0], dtype=np.float32),
    }
    # Output matches both but we skip kv_cache. Detector should report
    # 'queries' (since it isn't skipped).
    out = inputs["kv_cache"].copy()
    assert detect_returns_input(out, inputs,
                                skip_keys=["kv_cache"]) == "queries"


def test_detects_returns_input_ignores_shape_mismatch():
    inputs = {"q": np.zeros((4, 4), dtype=np.float32)}
    out = np.zeros((4, ), dtype=np.float32)
    assert detect_returns_input(out, inputs) is None


def test_anti_cheat_guard_composes_checks():
    guard = AntiCheatGuard()
    rng = np.random.default_rng(0)
    good = rng.normal(0, 1, size=(8, 8)).astype(np.float32)
    inputs = {
        "q": rng.normal(0, 1, size=(8, 8)).astype(np.float32),
    }
    assert guard.inspect(good, inputs).passed

    # Zero output trips the zero detector.
    zero_out = np.zeros((8, 8), dtype=np.float32)
    rep = guard.inspect(zero_out, inputs)
    assert not rep.passed
    assert "all-zero" in rep.reason

    # Aliased output trips the returns-input detector.
    rep = guard.inspect(inputs["q"], inputs)
    assert not rep.passed
    assert "input 'q'" in rep.reason


def test_cross_trial_independence_detects_identical_outputs():
    out = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    rep = cross_trial_independence(out, out.copy())
    assert not rep.passed
    assert "bit-identical" in rep.reason


def test_cross_trial_independence_passes_different_outputs():
    a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    b = np.array([1.1, 2.0, 3.0], dtype=np.float32)
    rep = cross_trial_independence(a, b)
    assert rep.passed


def test_cross_trial_independence_passes_different_shapes():
    a = np.zeros((4, ), dtype=np.float32)
    b = np.zeros((5, ), dtype=np.float32)
    assert cross_trial_independence(a, b).passed
