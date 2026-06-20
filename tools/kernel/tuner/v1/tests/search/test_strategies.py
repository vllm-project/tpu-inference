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
"""Unit tests for the search strategies against a synthetic 4-D quadratic.

Each strategy must find a configuration within 10% of the known optimum
within its documented trial budget. The synthetic problem mirrors the actual
parametric kernel search space: integer-valued options over four block-size
axes.
"""

import math

import pytest

from tools.kernel.tuner.v1.search import (EvolutionarySearch, GridSearch,
                                          IntChoice, IntLog2Range, TpeSearch)


def _make_space():
    return {
        "bq_sz": IntChoice("bq_sz", options=[16, 32, 64, 128]),
        "bkv_sz": IntLog2Range("bkv_sz", min_val=128, max_val=1024),
        "bq_csz": IntChoice("bq_csz", options=[8, 16, 32, 64]),
        "bkv_csz": IntLog2Range("bkv_csz", min_val=128, max_val=1024),
    }


_OPTIMUM = {"bq_sz": 64, "bkv_sz": 512, "bq_csz": 32, "bkv_csz": 512}


def _quadratic_score(params):
    # Latency-like score; lower is better; minimum at _OPTIMUM == 0.
    return sum((math.log2(params[k] / _OPTIMUM[k]))**2 for k in _OPTIMUM)


def _drive(strategy, evaluator):
    while not strategy.done():
        p = strategy.suggest()
        score = evaluator(p)
        strategy.observe(p, score, {})
    return strategy.best()


def test_int_log2_range_enumerates_powers_of_two():
    r = IntLog2Range("x", min_val=128, max_val=1024)
    assert r.values() == [128, 256, 512, 1024]


def test_int_log2_range_rejects_invalid_bounds():
    with pytest.raises(ValueError):
        IntLog2Range("x", min_val=0, max_val=8)
    with pytest.raises(ValueError):
        IntLog2Range("x", min_val=64, max_val=32)


def test_int_choice_neighbours():
    c = IntChoice("x", options=[8, 16, 32, 64])
    assert c.neighbours(16) == [8, 32]
    assert c.neighbours(8) == [16]
    assert c.neighbours(64) == [32]
    assert c.neighbours(99) == [8, 16, 32, 64]


def test_grid_finds_optimum_with_full_enumeration():
    space = _make_space()
    n_total = 4 * 4 * 4 * 4
    strategy = GridSearch(space=space, trial_budget=n_total)
    best_params, best_score = _drive(strategy, _quadratic_score)
    assert best_params == _OPTIMUM
    assert best_score == pytest.approx(0.0)


def test_grid_terminates_when_exhausted():
    space = {"x": IntChoice("x", options=[1, 2])}
    strategy = GridSearch(space=space, trial_budget=999)
    seen = []
    while not strategy.done():
        p = strategy.suggest()
        seen.append(p["x"])
        strategy.observe(p, float(p["x"]), {})
        if len(seen) > 10:
            pytest.fail("GridSearch did not terminate on exhaustion")
    assert sorted(seen) == [1, 2]


def test_evolutionary_finds_near_optimum():
    space = _make_space()
    strategy = EvolutionarySearch(space=space, trial_budget=200, seed=1)
    best_params, best_score = _drive(strategy, _quadratic_score)
    # Within (log2(2))**2 = 1 of optimum on any single axis is acceptable.
    assert best_score < 1.0
    assert best_params is not None


def test_evolutionary_ignores_infinite_scores():
    space = {
        "x": IntChoice("x", options=[1, 2, 4, 8, 16, 32, 64]),
    }

    def evaluator(p):
        return math.inf if p["x"] <= 4 else float(p["x"])

    strategy = EvolutionarySearch(space=space, trial_budget=100, seed=2)
    best_params, best_score = _drive(strategy, evaluator)
    # The optimum among finite candidates is x=8 (score 8); the inf branch
    # must not pollute the population.
    assert best_score == pytest.approx(8.0)
    assert best_params == {"x": 8}


def test_tpe_finds_near_optimum():
    space = _make_space()
    strategy = TpeSearch(space=space,
                         trial_budget=80,
                         n_startup_trials=10,
                         seed=42)
    best_params, best_score = _drive(strategy, _quadratic_score)
    assert best_score < 1.0
    assert best_params is not None


def test_tpe_handles_failed_trials():
    space = _make_space()
    strategy = TpeSearch(space=space,
                         trial_budget=40,
                         n_startup_trials=10,
                         seed=7)

    def evaluator(p):
        if p["bq_sz"] == 128:
            return math.inf
        return _quadratic_score(p)

    best_params, best_score = _drive(strategy, evaluator)
    assert best_params is not None
    assert best_params["bq_sz"] != 128
