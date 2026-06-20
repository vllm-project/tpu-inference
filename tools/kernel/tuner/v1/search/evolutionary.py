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
"""Simple (μ+λ)-evolutionary search with tournament selection and elitism.

Operators:

* **Crossover** — uniform per-gene.
* **Mutation** — for each gene, with probability ``mutation_rate`` swap to a
  random neighbour from ``ParamRange.neighbours(x)``.
* **Selection** — tournament of size ``tournament_k``; ``inf``-scored
  individuals are skipped so failed trials don't survive into the next
  generation.
* **Elitism** — top ``elite`` individuals are carried into the next gen.

Reasonable defaults for the ~10⁴–10⁶ parametric kernel-tuning space described
in the plan's *Phase 1* section.
"""

import logging
import math
from typing import Any

import numpy as np

from tools.kernel.tuner.v1.search.strategy import SearchSpace, SearchStrategy

logger = logging.getLogger(__name__)


class EvolutionarySearch(SearchStrategy):

    def __init__(
        self,
        *,
        space: SearchSpace,
        trial_budget: int,
        population_size: int = 8,
        offspring_size: int = 24,
        elite: int = 2,
        tournament_k: int = 3,
        mutation_rate: float = 0.5,
        seed: int = 0,
    ) -> None:
        super().__init__(space=space, trial_budget=trial_budget)
        if population_size < 2:
            raise ValueError(
                f"population_size must be >= 2, got {population_size}")
        if offspring_size < 1:
            raise ValueError(
                f"offspring_size must be >= 1, got {offspring_size}")
        if elite < 0 or elite > population_size:
            raise ValueError(
                f"elite must satisfy 0 <= elite <= population_size, got "
                f"{elite} (population_size={population_size})")
        if tournament_k < 1:
            raise ValueError(f"tournament_k must be >= 1, got {tournament_k}")
        self.population_size = population_size
        self.offspring_size = offspring_size
        self.elite = elite
        self.tournament_k = tournament_k
        self.mutation_rate = mutation_rate
        self._rng = np.random.default_rng(seed)
        self._names = sorted(space.keys())
        self._population: list[tuple[dict[str, Any], float]] = []
        self._pending: list[dict[str, Any]] = []
        for _ in range(population_size):
            self._pending.append(self._random_individual())

    def _random_individual(self) -> dict[str, Any]:
        return {n: self.space[n].sample(self._rng) for n in self._names}

    def _finite(self) -> list[tuple[dict[str, Any], float]]:
        return [p for p in self._population if math.isfinite(p[1])]

    def _tournament(self) -> dict[str, Any]:
        pool = self._finite()
        if not pool:
            return self._random_individual()
        k = min(self.tournament_k, len(pool))
        idx = self._rng.choice(len(pool), size=k, replace=False)
        best = min((pool[int(i)] for i in idx), key=lambda p: p[1])
        return dict(best[0])

    def _uniform_crossover(
        self,
        a: dict[str, Any],
        b: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            n: (a[n] if self._rng.random() < 0.5 else b[n])
            for n in self._names
        }

    def _mutate(self, p: dict[str, Any]) -> dict[str, Any]:
        out = dict(p)
        for n in self._names:
            if self._rng.random() < self.mutation_rate:
                neighbours = self.space[n].neighbours(out[n])
                if neighbours:
                    out[n] = neighbours[int(
                        self._rng.integers(0, len(neighbours)))]
        return out

    def _spawn_generation(self) -> None:
        finite = self._finite()
        finite.sort(key=lambda p: p[1])
        survivors = finite[:self.population_size]
        # Keep failures so anti-cheat / debugging can inspect them, but never
        # let them flow into selection (filtered in `_finite`).
        self._population = survivors + [
            p for p in self._population if not math.isfinite(p[1])
        ]
        offspring: list[dict[str, Any]] = list(
            dict(p[0]) for p in finite[:self.elite])
        while len(offspring) < self.offspring_size:
            a = self._tournament()
            b = self._tournament()
            child = self._uniform_crossover(a, b)
            child = self._mutate(child)
            offspring.append(child)
        self._pending = offspring

    def suggest(self) -> dict[str, Any]:
        if not self._pending:
            self._spawn_generation()
        return self._pending.pop(0)

    def observe(
        self,
        params: dict[str, Any],
        score: float,
        aux: dict[str, Any] | None = None,
    ) -> None:
        super().observe(params, score, aux)
        self._population.append((dict(params), score))
