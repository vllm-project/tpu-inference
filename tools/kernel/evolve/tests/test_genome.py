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
"""Tests for Genome serialization, status, and id stability."""

import math

from tools.kernel.evolve.genome import Genome, GenomeStatus


def test_genome_new_id_stable_for_same_inputs():
    g1 = Genome.new(
        diff="--- a/x\n+++ b/x\n@@ -1,1 +1,1 @@\n-a\n+b\n",
        baseline_path="x",
        parent_ids=["aa", "bb"],
        generation=3,
        island_id=1,
    )
    g2 = Genome.new(
        diff="--- a/x\n+++ b/x\n@@ -1,1 +1,1 @@\n-a\n+b\n",
        baseline_path="x",
        parent_ids=["bb", "aa"],  # order shouldn't matter
        generation=3,
        island_id=1,
    )
    assert g1.id == g2.id
    assert len(g1.id) == 12


def test_genome_new_id_differs_when_diff_differs():
    g1 = Genome.new(diff="a",
                    baseline_path="x",
                    parent_ids=[],
                    generation=0,
                    island_id=0)
    g2 = Genome.new(diff="b",
                    baseline_path="x",
                    parent_ids=[],
                    generation=0,
                    island_id=0)
    assert g1.id != g2.id


def test_genome_to_dict_handles_inf_fitness():
    g = Genome(id="abc",
               diff="",
               baseline_path="x",
               status=GenomeStatus.PENDING)
    g.fitness = math.inf
    d = g.to_dict()
    assert d["fitness"] == "inf"
    g.fitness = 1234.0
    assert g.to_dict()["fitness"] == 1234.0


def test_genome_round_trip_serialization():
    g = Genome.new(diff="d",
                   baseline_path="x",
                   parent_ids=["p"],
                   generation=1,
                   island_id=2)
    g.fitness = 500_000.0
    g.status = GenomeStatus.VERIFIED
    g.metrics = {"p50_ns": 480_000, "cosine": 0.9999}
    d = g.to_dict()
    g2 = Genome.from_dict(d)
    assert g2.id == g.id
    assert g2.fitness == g.fitness
    assert g2.status == GenomeStatus.VERIFIED
    assert g2.metrics["p50_ns"] == 480_000


def test_status_helpers():
    assert not GenomeStatus.PENDING.is_terminal
    assert not GenomeStatus.EVALUATING.is_terminal
    assert GenomeStatus.VERIFIED.is_terminal
    assert GenomeStatus.VERIFIED.is_success
    assert GenomeStatus.FAILED_NUMERICS.is_terminal
    assert not GenomeStatus.FAILED_NUMERICS.is_success


def test_baseline_has_special_id():
    g = Genome.baseline(baseline_path="path/to/kernel.py")
    assert g.id == "baseline"
    assert g.diff == ""
    assert g.parent_ids == []
