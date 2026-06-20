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
"""Statistical winner-promotion step.

After the orchestrator finishes its evolution loop, its top-K candidates
each get re-benched N times (paired with the baseline) and gated by a
paired t-test at p<0.05. Within-noise candidates are demoted; only
significant wins make it into the auto-PR.

This is the third moat: even after the numerics gate and the LLM critic
gate, the *measured speedup* must be larger than per-round noise to be
called a win.
"""

from __future__ import annotations

import dataclasses
from typing import Callable, Sequence

from tools.kernel.evolve.archive import Archive
from tools.kernel.evolve.genome import Genome, GenomeStatus
from tools.kernel.evolve.stats.significance import (PairedComparison,
                                                    paired_t_test,
                                                    wilcoxon_signed_rank)


@dataclasses.dataclass
class PromotionResult:
    """Per-candidate promotion outcome."""
    genome_id: str
    baseline_mean_ns: float
    candidate_mean_ns: float
    speedup: float
    n_rounds: int
    p_value: float
    cohens_d: float
    ci95_low_ns: float
    ci95_high_ns: float
    paired_t_significant: bool
    wilcoxon_significant: bool
    promoted: bool


def promote_top_k(
    archive: Archive,
    *,
    top_k: int,
    n_rounds: int,
    rebench_fn: Callable[[Genome], list[float]],
    rebench_baseline_fn: Callable[[], list[float]],
    p_value_threshold: float = 0.05,
    speedup_threshold: float = 1.005,
) -> list[PromotionResult]:
    """Re-bench the top-K verified candidates and gate by paired t-test.

    Args:
        archive: source of verified candidates.
        top_k: number of fastest candidates to consider.
        n_rounds: per-config rounds; ≥10 recommended.
        rebench_fn: takes a Genome, returns a per-round latency vector.
        rebench_baseline_fn: returns a baseline latency vector.
        p_value_threshold: paired-t p threshold (default 0.05).
        speedup_threshold: floor for the candidate's mean speedup
            (default 1.005 — half a percent).
    """
    finite = sorted(archive.all_finite(), key=lambda g: g.fitness)[:top_k]
    if not finite:
        return []
    baseline_vec = rebench_baseline_fn()
    if not baseline_vec or any(v <= 0 for v in baseline_vec):
        return []
    baseline_mean = sum(baseline_vec) / len(baseline_vec)
    results: list[PromotionResult] = []
    for g in finite:
        cand_vec = rebench_fn(g)
        if not cand_vec or len(cand_vec) < 2:
            continue
        cand_mean = sum(cand_vec) / len(cand_vec)
        comp = PairedComparison(label_a="baseline",
                                label_b=g.id,
                                a_values=baseline_vec,
                                b_values=cand_vec)
        t = paired_t_test(comp)
        w = wilcoxon_signed_rank(comp)
        speedup = (baseline_mean /
                   cand_mean if cand_mean > 0 else float("nan"))
        promoted = (not isinstance(t, dict) or "error" not in t) and t.get(
            "p_value_approx",
            1.0) < p_value_threshold and speedup > speedup_threshold
        results.append(
            PromotionResult(
                genome_id=g.id,
                baseline_mean_ns=baseline_mean,
                candidate_mean_ns=cand_mean,
                speedup=speedup,
                n_rounds=t.get("n_pairs", 0),
                p_value=t.get("p_value_approx", float("nan")),
                cohens_d=t.get("cohens_d", float("nan")),
                ci95_low_ns=t.get("ci95_low", float("nan")),
                ci95_high_ns=t.get("ci95_high", float("nan")),
                paired_t_significant=bool(t.get("significant_at_005", False)),
                wilcoxon_significant=bool(w.get("significant_at_005", False))
                if isinstance(w, dict) else False,
                promoted=promoted,
            ))
        # Side-effect: tag the genome so the auto-PR step can filter on it.
        if promoted:
            g.metrics["stats_promoted"] = True
            g.metrics["p_value"] = t.get("p_value_approx")
            g.metrics["speedup_rebench"] = speedup
        else:
            g.metrics["stats_promoted"] = False
            # Demote within-noise candidates so the archive's "best" reflects
            # only statistically validated wins.
            if isinstance(t, dict) and "error" not in t:
                g.fitness = baseline_mean  # neutralize
                g.status = GenomeStatus.VERIFIED  # keep verified, but demoted
    return results


def format_promotion_report(results: Sequence[PromotionResult]) -> str:
    if not results:
        return "(no candidates re-benched)\n"
    lines = [
        f"{'genome':14s}  {'speedup':>9s}  {'95% CI':>22s}  {'p':>9s}  "
        f"{'cohens d':>9s}  {'verdict':<22s}",
        "-" * 100,
    ]
    for r in sorted(results, key=lambda x: -x.speedup):
        v = ("✓ SIGNIFICANT win" if r.promoted else "✗ significantly worse" if
             r.paired_t_significant and r.speedup < 1.0 else "✗ within noise")
        ci_lo = (r.baseline_mean_ns -
                 r.candidate_mean_ns) - (r.ci95_high_ns - r.ci95_low_ns) / 2.0
        ci_hi = ci_lo + (r.ci95_high_ns - r.ci95_low_ns)
        lines.append(f"{r.genome_id:14s}  {r.speedup:>8.4f}x  "
                     f"[{ci_lo:+8.0f}, {ci_hi:+8.0f}]  {r.p_value:>9.4f}  "
                     f"{r.cohens_d:>+9.3f}  {v:<22s}")
    return "\n".join(lines) + "\n"
