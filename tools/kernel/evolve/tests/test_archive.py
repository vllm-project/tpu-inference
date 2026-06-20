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
"""Tests for the island-based archive."""

import math

import numpy as np

from tools.kernel.evolve.archive import Archive, Island
from tools.kernel.evolve.genome import Genome, GenomeStatus


def _g(diff: str,
       *,
       fitness: float,
       generation: int = 1,
       island_id: int = 0,
       parent_ids=None) -> Genome:
    genome = Genome.new(
        diff=diff,
        baseline_path="x",
        parent_ids=parent_ids or [],
        generation=generation,
        island_id=island_id,
    )
    genome.fitness = fitness
    genome.status = (GenomeStatus.VERIFIED
                     if math.isfinite(fitness) else GenomeStatus.FAILED_RUN)
    return genome


def test_island_insert_caps_population():
    isl = Island(id=0, cap=3)
    for f in [100.0, 200.0, 300.0, 50.0, 150.0]:
        isl.insert(_g(str(f), fitness=f))
    assert len(isl.members) == 3
    assert [g.fitness for g in isl.members] == [50.0, 100.0, 150.0]


def test_island_best_finite_only():
    isl = Island(id=0)
    isl.insert(_g("a", fitness=math.inf))
    isl.insert(_g("b", fitness=200.0))
    isl.insert(_g("c", fitness=math.inf))
    best = isl.best()
    assert best is not None
    assert best.fitness == 200.0


def test_island_best_returns_none_when_empty():
    isl = Island(id=0)
    assert isl.best() is None


def test_archive_insert_dedupes_by_id():
    arc = Archive(baseline=Genome.baseline("x"), num_islands=2)
    g = _g("dup", fitness=100.0, island_id=0)
    assert arc.insert(g)
    assert not arc.insert(g)
    assert arc.size() == 2  # baseline + one


def test_archive_tournament_falls_back_to_baseline():
    arc = Archive(baseline=Genome.baseline("x"), num_islands=2)
    arc.baseline.fitness = 1_000_000.0
    arc.baseline.status = GenomeStatus.VERIFIED
    rng = np.random.default_rng(0)
    pick = arc.select_parent(0, tournament_k=3, rng=rng)
    assert pick.id == "baseline"


def test_archive_tournament_picks_best_among_pool():
    arc = Archive(baseline=Genome.baseline("x"), num_islands=1)
    for f in [100.0, 200.0, 300.0]:
        arc.insert(_g(str(f), fitness=f, island_id=0))
    rng = np.random.default_rng(0)
    # tournament_k=3 over 3 individuals: deterministically picks all of them
    pick = arc.select_parent(0, tournament_k=3, rng=rng)
    assert pick.fitness == 100.0


def test_archive_migration_propagates_top_k():
    arc = Archive(baseline=Genome.baseline("x"), num_islands=3)
    # Seed island 0 with the best genome.
    g_best = _g("best", fitness=100.0, island_id=0)
    g_mid = _g("mid", fitness=200.0, island_id=0)
    arc.insert(g_best)
    arc.insert(g_mid)
    # Seed island 1 with a worse genome.
    arc.insert(_g("worse1", fitness=500.0, island_id=1))
    # Seed island 2 with a worse genome.
    arc.insert(_g("worse2", fitness=400.0, island_id=2))
    rng = np.random.default_rng(42)
    arc.migrate(top_k=1, rng=rng)
    # The top-1 from island 0 (fitness=100) should now appear in at least one
    # other island as a clone.
    found_clone = False
    for isl in arc.islands:
        if isl.id == 0:
            continue
        for g in isl.members:
            if g.id == g_best.id and g.island_id != 0:
                found_clone = True
    assert found_clone


def test_archive_best_genome_across_islands():
    arc = Archive(baseline=Genome.baseline("x"), num_islands=3)
    arc.insert(_g("a", fitness=300.0, island_id=0))
    arc.insert(_g("b", fitness=150.0, island_id=1))
    arc.insert(_g("c", fitness=200.0, island_id=2))
    best = arc.best_genome()
    assert best is not None
    assert best.fitness == 150.0


def test_archive_jsonl_round_trip(tmp_path):
    path = tmp_path / "archive.jsonl"
    arc = Archive(
        baseline=Genome.baseline("x"),
        num_islands=2,
        persist_path=path,
    )
    arc.insert(_g("p1", fitness=100.0, island_id=0))
    arc.insert(_g("p2", fitness=200.0, island_id=1))
    # Simulate restart.
    arc2 = Archive(
        baseline=Genome.baseline("x"),
        num_islands=2,
        persist_path=path,
    )
    ids = sorted([g.id for isl in arc2.islands for g in isl.members])
    expected = sorted([g.id for isl in arc.islands for g in isl.members])
    assert ids == expected


def test_archive_summary_reports_speedup():
    arc = Archive(baseline=Genome.baseline("x"), num_islands=1)
    arc.baseline.fitness = 1000.0
    arc.baseline.status = GenomeStatus.VERIFIED
    arc.insert(_g("better", fitness=500.0, island_id=0))
    s = arc.summary()
    assert s["speedup_vs_baseline"] == 2.0
    assert s["best_fitness_ns"] == 500.0
