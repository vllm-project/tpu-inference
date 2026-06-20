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
"""Multi-fidelity hierarchical evaluation router.

Three tiers of fidelity, each ~10× more expensive than the prior:

* Tier 1 (cost-model): score N candidates in milliseconds via the learned
  surrogate. Keep top-K1.
* Tier 2 (microbench): run K1 candidates on TPU at the kernel level for
  ~5s each via the existing ``evaluate_genome``. Keep top-K2.
* Tier 3 (full model): run K2 candidates through Qwen3-0.6B end-to-end
  for ~3min each via subprocess. Final ranking.

The router orchestrates these tiers transparently — the caller hands it a
population of candidate diffs and gets back fitness scores at the highest
fidelity they reached.
"""

from __future__ import annotations

import dataclasses
import logging
import math
import time
from typing import Any, Callable

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class FidelityResult:
    """Per-candidate result across fidelity tiers."""
    candidate_id: str
    tier_reached: int  # 1, 2, or 3
    fitness_ns: float = math.inf
    status: str = "UNKNOWN"
    tier1_score: float | None = None
    tier2_fitness_ns: float | None = None
    tier3_fitness_ns: float | None = None
    error: str | None = None
    wall_time_s: float = 0.0


class MultiFidelityRouter:
    """Schedule candidates through the three-tier evaluation pipeline."""

    def __init__(
        self,
        *,
        tier1_score_fn: Callable[[Any], float] | None = None,
        tier2_eval_fn: Callable[[Any], dict] | None = None,
        tier3_eval_fn: Callable[[Any], dict] | None = None,
        keep_after_tier1: float = 0.3,
        keep_after_tier2: float = 0.2,
        min_tier2_keep: int = 4,
        min_tier3_keep: int = 1,
    ) -> None:
        self.tier1_score_fn = tier1_score_fn
        self.tier2_eval_fn = tier2_eval_fn
        self.tier3_eval_fn = tier3_eval_fn
        self.keep_after_tier1 = keep_after_tier1
        self.keep_after_tier2 = keep_after_tier2
        self.min_tier2_keep = min_tier2_keep
        self.min_tier3_keep = min_tier3_keep

    def route(self, candidates: list[Any]) -> list[FidelityResult]:
        """Run ``candidates`` through the tiers; return per-candidate result."""
        results = {
            i: FidelityResult(candidate_id=str(i), tier_reached=0)
            for i in range(len(candidates))
        }

        # Tier 1 — surrogate prediction.
        survivors_t1 = list(range(len(candidates)))
        if self.tier1_score_fn is not None and candidates:
            t0 = time.time()
            scored: list[tuple[int, float]] = []
            for i, c in enumerate(candidates):
                try:
                    s = float(self.tier1_score_fn(c))
                except Exception:
                    s = math.inf
                scored.append((i, s))
                results[i].tier1_score = s
                results[i].tier_reached = max(results[i].tier_reached, 1)
            scored.sort(key=lambda t: t[1])
            keep_n = max(self.min_tier2_keep,
                         int(len(scored) * self.keep_after_tier1))
            survivors_t1 = [i for i, _ in scored[:keep_n]]
            elapsed = time.time() - t0
            logger.info("tier1: scored %d, kept %d (%.2fs)", len(scored),
                        len(survivors_t1), elapsed)

        # Tier 2 — microbench (kernel-level).
        survivors_t2 = list(survivors_t1)
        if self.tier2_eval_fn is not None and survivors_t1:
            t0 = time.time()
            t2_results: list[tuple[int, float]] = []
            for i in survivors_t1:
                t1 = time.time()
                try:
                    r = self.tier2_eval_fn(candidates[i])
                    fit = float(r.get("fitness_ns", math.inf))
                    status = r.get("status", "UNKNOWN")
                    results[i].tier2_fitness_ns = fit
                    results[i].status = status
                    results[i].tier_reached = max(results[i].tier_reached, 2)
                    t2_results.append((i, fit))
                except Exception as err:
                    results[i].error = str(err)
                    t2_results.append((i, math.inf))
                results[i].wall_time_s += time.time() - t1
            t2_results.sort(key=lambda t: t[1])
            keep_n = max(self.min_tier3_keep,
                         int(len(t2_results) * self.keep_after_tier2))
            survivors_t2 = [
                i for i, f in t2_results[:keep_n] if math.isfinite(f)
            ]
            elapsed = time.time() - t0
            logger.info("tier2: evaluated %d, kept %d (%.1fs)",
                        len(t2_results), len(survivors_t2), elapsed)

        # Tier 3 — full-model eval.
        if self.tier3_eval_fn is not None and survivors_t2:
            t0 = time.time()
            for i in survivors_t2:
                t1 = time.time()
                try:
                    r = self.tier3_eval_fn(candidates[i])
                    fit = float(r.get("fitness_ns", math.inf))
                    status = r.get("status", "UNKNOWN")
                    results[i].tier3_fitness_ns = fit
                    results[i].status = status
                    results[i].tier_reached = max(results[i].tier_reached, 3)
                except Exception as err:
                    results[i].error = str(err)
                results[i].wall_time_s += time.time() - t1
            elapsed = time.time() - t0
            logger.info("tier3: evaluated %d (%.1fs)", len(survivors_t2),
                        elapsed)

        # Final fitness = highest-tier observed.
        for i, r in results.items():
            if r.tier3_fitness_ns is not None:
                r.fitness_ns = r.tier3_fitness_ns
            elif r.tier2_fitness_ns is not None:
                r.fitness_ns = r.tier2_fitness_ns
            elif r.tier1_score is not None:
                # Surrogate predicts log-fitness; exponentiate.
                r.fitness_ns = math.exp(r.tier1_score)
        return [results[i] for i in range(len(candidates))]
