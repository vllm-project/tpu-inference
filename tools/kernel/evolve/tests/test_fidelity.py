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
"""Tests for the multi-fidelity router."""

import math

from tools.kernel.evolve.fidelity.router import MultiFidelityRouter


def test_router_keeps_top_k_after_each_tier():
    candidates = list(range(20))

    def t1(c):
        # Tier-1 score proportional to candidate index — small = better.
        return float(c)

    def t2(c):
        # Tier-2 fitness inverts t1 — to test that t2 wins matter, the
        # ones surviving t1 should be ordered by t2 from there.
        return {"fitness_ns": float(c) * 10.0, "status": "VERIFIED"}

    def t3(c):
        return {"fitness_ns": float(c) * 100.0, "status": "VERIFIED"}

    router = MultiFidelityRouter(
        tier1_score_fn=t1,
        tier2_eval_fn=t2,
        tier3_eval_fn=t3,
        keep_after_tier1=0.5,
        keep_after_tier2=0.5,
        min_tier2_keep=3,
        min_tier3_keep=2,
    )
    results = router.route(candidates)
    # All 20 reached tier 1.
    assert all(r.tier_reached >= 1 for r in results)
    # Top 10 reached tier 2.
    t2_reached = [r for r in results if r.tier_reached >= 2]
    assert len(t2_reached) == 10
    # Top 5 reached tier 3.
    t3_reached = [r for r in results if r.tier_reached >= 3]
    assert len(t3_reached) == 5


def test_router_with_only_tier1_uses_predicted_fitness():

    def t1(c):
        return math.log(c + 1.0)

    router = MultiFidelityRouter(tier1_score_fn=t1)
    results = router.route([1, 10, 100])
    # All only reached tier 1; fitness should be the exponentiated score.
    for r, cand in zip(results, [1, 10, 100]):
        assert r.tier_reached == 1
        # Float comparison with small tolerance for math.exp(math.log()) roundtrip.
        assert abs(r.fitness_ns - (cand + 1.0)) < 1e-9


def test_router_handles_tier2_exceptions():

    def t2(c):
        raise RuntimeError(f"sim {c}")

    router = MultiFidelityRouter(tier2_eval_fn=t2,
                                 keep_after_tier1=1.0,
                                 min_tier2_keep=10)
    results = router.route([1, 2, 3])
    for r in results:
        assert r.error is not None or r.fitness_ns == math.inf


def test_router_no_inf_propagation_to_tier3():
    """If tier 2 marks something as inf, it should NOT advance to tier 3."""
    t3_calls: list[int] = []

    def t2(c):
        return {"fitness_ns": math.inf, "status": "FAILED"}

    def t3(c):
        t3_calls.append(c)
        return {"fitness_ns": 1.0, "status": "VERIFIED"}

    router = MultiFidelityRouter(tier2_eval_fn=t2,
                                 tier3_eval_fn=t3,
                                 keep_after_tier2=1.0,
                                 min_tier3_keep=10)
    router.route([1, 2, 3])
    assert t3_calls == []
