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
"""Tests for the statistical-significance harness."""

import random

from tools.kernel.evolve.stats.significance import (PairedComparison,
                                                    paired_t_test,
                                                    welch_t_test,
                                                    wilcoxon_signed_rank)


def test_paired_t_test_detects_large_real_difference():
    rng = random.Random(0)
    a = [1000.0 + rng.gauss(0, 10) for _ in range(20)]
    b = [1100.0 + rng.gauss(0, 10) for _ in range(20)]
    comp = PairedComparison(label_a="a", label_b="b", a_values=a, b_values=b)
    t = paired_t_test(comp)
    assert t["significant_at_005"]
    assert t["mean_diff"] > 50
    assert t["ci95_low"] > 50
    assert t["p_value_approx"] < 0.01


def test_paired_t_test_does_not_detect_noise():
    rng = random.Random(0)
    a = [1000.0 + rng.gauss(0, 50) for _ in range(15)]
    # Identical mean — only noise.
    b = [1000.0 + rng.gauss(0, 50) for _ in range(15)]
    comp = PairedComparison(label_a="a", label_b="b", a_values=a, b_values=b)
    t = paired_t_test(comp)
    assert not t["significant_at_005"]


def test_paired_t_test_returns_error_for_too_few_pairs():
    comp = PairedComparison(label_a="a",
                            label_b="b",
                            a_values=[1.0],
                            b_values=[1.1])
    t = paired_t_test(comp)
    assert "error" in t


def test_paired_t_test_confidence_interval_brackets_mean():
    rng = random.Random(0)
    a = [1000.0 + rng.gauss(0, 5) for _ in range(20)]
    b = [1050.0 + rng.gauss(0, 5) for _ in range(20)]
    comp = PairedComparison(label_a="a", label_b="b", a_values=a, b_values=b)
    t = paired_t_test(comp)
    assert t["ci95_low"] <= t["mean_diff"] <= t["ci95_high"]


def test_welch_t_test_independent_samples():
    rng = random.Random(0)
    a = [100.0 + rng.gauss(0, 5) for _ in range(15)]
    b = [120.0 + rng.gauss(0, 5) for _ in range(15)]
    r = welch_t_test(a, b)
    assert r["significant_at_005"]
    assert r["mean_b"] > r["mean_a"]


def test_wilcoxon_detects_consistent_improvement():
    rng = random.Random(0)
    a = [100.0 + rng.gauss(0, 2) for _ in range(15)]
    # +5% on every paired round.
    b = [a[i] * 1.05 + rng.gauss(0, 0.5) for i in range(15)]
    comp = PairedComparison(label_a="a", label_b="b", a_values=a, b_values=b)
    w = wilcoxon_signed_rank(comp)
    assert w["significant_at_005"]


def test_wilcoxon_handles_pure_noise():
    rng = random.Random(0)
    a = [100.0 + rng.gauss(0, 10) for _ in range(20)]
    b = [100.0 + rng.gauss(0, 10) for _ in range(20)]
    comp = PairedComparison(label_a="a", label_b="b", a_values=a, b_values=b)
    w = wilcoxon_signed_rank(comp)
    assert not w["significant_at_005"]


def test_wilcoxon_requires_minimum_samples():
    comp = PairedComparison(label_a="a",
                            label_b="b",
                            a_values=[1.0, 2.0],
                            b_values=[1.1, 2.1])
    w = wilcoxon_signed_rank(comp)
    assert "error" in w
