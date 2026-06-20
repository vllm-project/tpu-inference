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
"""Tests for statistical winner-promotion."""

import random

from tools.kernel.evolve.archive import Archive
from tools.kernel.evolve.genome import Genome, GenomeStatus
from tools.kernel.evolve.stats.promote import (format_promotion_report,
                                               promote_top_k)


def _make_archive_with_candidates(n: int = 3) -> Archive:
    arc = Archive(baseline=Genome.baseline("x"), num_islands=1)
    arc.baseline.fitness = 1000.0
    arc.baseline.status = GenomeStatus.VERIFIED
    for i in range(n):
        g = Genome.new(
            diff=f"--- a/x\n+++ b/x\n@@ -1,1 +1,1 @@\n-a{i}\n+b{i}\n",
            baseline_path="x",
            parent_ids=[],
            generation=1,
            island_id=0)
        g.fitness = 900.0 - i * 50
        g.status = GenomeStatus.VERIFIED
        arc.insert(g)
    return arc


def test_promote_top_k_flags_significant_win():
    arc = _make_archive_with_candidates(1)
    rng = random.Random(0)

    def rebench_baseline():
        return [1000.0 + rng.gauss(0, 5) for _ in range(15)]

    def rebench_candidate(g):
        return [800.0 + rng.gauss(0, 5) for _ in range(15)]

    results = promote_top_k(arc,
                            top_k=3,
                            n_rounds=15,
                            rebench_fn=rebench_candidate,
                            rebench_baseline_fn=rebench_baseline)
    assert len(results) == 1
    r = results[0]
    assert r.promoted
    assert r.paired_t_significant
    assert r.speedup > 1.2


def test_promote_top_k_demotes_within_noise():
    arc = _make_archive_with_candidates(1)
    rng = random.Random(0)

    def rebench_baseline():
        return [1000.0 + rng.gauss(0, 50) for _ in range(15)]

    def rebench_candidate(g):
        # Tiny mean improvement entirely within noise.
        return [995.0 + rng.gauss(0, 50) for _ in range(15)]

    results = promote_top_k(arc,
                            top_k=3,
                            n_rounds=15,
                            rebench_fn=rebench_candidate,
                            rebench_baseline_fn=rebench_baseline)
    assert len(results) == 1
    r = results[0]
    assert not r.promoted


def test_promote_top_k_handles_empty_archive():
    arc = Archive(baseline=Genome.baseline("x"), num_islands=1)
    # baseline never verified.
    results = promote_top_k(arc,
                            top_k=3,
                            n_rounds=10,
                            rebench_fn=lambda g: [],
                            rebench_baseline_fn=lambda: [])
    assert results == []


def test_promotion_report_renders():
    arc = _make_archive_with_candidates(2)
    rng = random.Random(0)
    results = promote_top_k(
        arc,
        top_k=2,
        n_rounds=15,
        rebench_fn=lambda g: [900.0 + rng.gauss(0, 5) for _ in range(15)],
        rebench_baseline_fn=lambda:
        [1000.0 + rng.gauss(0, 5) for _ in range(15)])
    txt = format_promotion_report(results)
    assert "speedup" in txt
    assert "p" in txt
    assert "verdict" in txt
