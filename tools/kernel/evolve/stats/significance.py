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
"""Statistical-significance tests for benchmark comparisons.

The user's POC sweeps measured "wins" in the 1–6% range, which sit inside
the per-round variance band of vLLM Qwen3-0.6B throughput. To distinguish
real wins from noise we need:

* **Paired comparisons** — same prompts, same warmup, same hardware, just
  with the kernel parameter toggled. Pairs cancel cross-run drift.
* **Hypothesis tests** — paired-t for parametric, Wilcoxon signed-rank for
  the small-sample / non-normal regime.
* **Confidence intervals** — 95% CI on the speedup ratio.
* **Effect-size proxy** — Cohen's d, so "+5% with d=0.2" is reported as
  "small effect" rather than just a single number.

No SciPy dependency: closed-form / table-based for the small-N cases.
"""

from __future__ import annotations

import dataclasses
import math
import statistics
from typing import Sequence

# Two-sided critical values for Student's t at α=0.05.
# Indexed by degrees of freedom (df). df>30 → use z=1.96.
_T_CRIT_005 = {
    1: 12.706,
    2: 4.303,
    3: 3.182,
    4: 2.776,
    5: 2.571,
    6: 2.447,
    7: 2.365,
    8: 2.306,
    9: 2.262,
    10: 2.228,
    11: 2.201,
    12: 2.179,
    13: 2.160,
    14: 2.145,
    15: 2.131,
    16: 2.120,
    17: 2.110,
    18: 2.101,
    19: 2.093,
    20: 2.086,
    21: 2.080,
    22: 2.074,
    23: 2.069,
    24: 2.064,
    25: 2.060,
    26: 2.056,
    27: 2.052,
    28: 2.048,
    29: 2.045,
    30: 2.042,
}


def _t_crit_05_two_sided(df: int) -> float:
    if df <= 0:
        return float("inf")
    if df in _T_CRIT_005:
        return _T_CRIT_005[df]
    return 1.96  # asymptotic z


@dataclasses.dataclass
class PairedComparison:
    """Paired comparison of two configs measured over N rounds each."""
    label_a: str
    label_b: str
    a_values: list[float]
    b_values: list[float]
    metric: str = "throughput_tok_s"

    @property
    def n(self) -> int:
        return min(len(self.a_values), len(self.b_values))

    @property
    def mean_a(self) -> float:
        return statistics.mean(self.a_values) if self.a_values else 0.0

    @property
    def mean_b(self) -> float:
        return statistics.mean(self.b_values) if self.b_values else 0.0

    @property
    def speedup(self) -> float:
        return self.mean_b / self.mean_a if self.mean_a > 0 else float("nan")


def paired_t_test(comp: PairedComparison) -> dict:
    """Two-sided paired t-test on the per-round differences (b − a).

    Returns a dict with: mean_diff, std_diff, t, df, p_value (estimated),
    ci95_low, ci95_high, significant_at_005.
    """
    n = comp.n
    if n < 2:
        return {"error": "need at least 2 paired rounds"}
    diffs = [comp.b_values[i] - comp.a_values[i] for i in range(n)]
    mean_d = statistics.mean(diffs)
    std_d = statistics.stdev(diffs) if n > 1 else 0.0
    se = std_d / math.sqrt(n) if n > 0 else float("inf")
    t = mean_d / se if se > 0 else float("inf")
    df = n - 1
    t_crit = _t_crit_05_two_sided(df)
    ci95_low = mean_d - t_crit * se
    ci95_high = mean_d + t_crit * se
    # Approximate p-value (Welch-Satterthwaite-style, asymptotic z).
    z = abs(t)
    p_approx = max(2.0 * (1.0 - _stdnorm_cdf(z)), 1e-300)
    cohens_d = mean_d / std_d if std_d > 0 else float("inf")
    return {
        "n_pairs": n,
        "mean_diff": mean_d,
        "std_diff": std_d,
        "se_diff": se,
        "t": t,
        "df": df,
        "p_value_approx": p_approx,
        "ci95_low": ci95_low,
        "ci95_high": ci95_high,
        "cohens_d": cohens_d,
        "significant_at_005": abs(t) > t_crit,
    }


def welch_t_test(a: Sequence[float], b: Sequence[float]) -> dict:
    """Two-sample Welch's t-test (unequal variances)."""
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return {"error": "need at least 2 samples per group"}
    ma, mb = statistics.mean(a), statistics.mean(b)
    va = statistics.variance(a)
    vb = statistics.variance(b)
    se = math.sqrt(va / na + vb / nb)
    t = (mb - ma) / se if se > 0 else float("inf")
    # Welch–Satterthwaite df.
    denom = (va / na)**2 / (na - 1) + (vb / nb)**2 / (nb - 1)
    df = int(((va / na + vb / nb)**2 /
              denom)) if denom > 0 else max(na + nb - 2, 1)
    t_crit = _t_crit_05_two_sided(df)
    z = abs(t)
    p = max(2.0 * (1.0 - _stdnorm_cdf(z)), 1e-300)
    return {
        "n_a": na,
        "n_b": nb,
        "mean_a": ma,
        "mean_b": mb,
        "se": se,
        "t": t,
        "df": df,
        "p_value_approx": p,
        "significant_at_005": abs(t) > t_crit,
    }


def wilcoxon_signed_rank(comp: PairedComparison) -> dict:
    """Wilcoxon signed-rank: non-parametric alternative to paired-t.

    Use when the per-round distribution is non-normal (e.g. long tails
    from JIT compile costs). Returns the W statistic and an asymptotic
    p-value.
    """
    n = comp.n
    if n < 5:
        return {"error": "need at least 5 paired rounds for the asymptotic"}
    diffs = [comp.b_values[i] - comp.a_values[i] for i in range(n)]
    nonzero = [d for d in diffs if d != 0]
    if not nonzero:
        return {"error": "all paired diffs are zero"}
    nz_n = len(nonzero)
    # Rank by absolute value (average rank for ties).
    sorted_abs = sorted([(abs(d), i) for i, d in enumerate(nonzero)])
    ranks: list[float] = [0.0] * nz_n
    i = 0
    while i < nz_n:
        j = i
        # Group ties on absolute value.
        while j + 1 < nz_n and sorted_abs[j + 1][0] == sorted_abs[i][0]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0  # 1-based
        for k in range(i, j + 1):
            ranks[sorted_abs[k][1]] = avg_rank
        i = j + 1
    w_plus = sum(ranks[i] for i, d in enumerate(nonzero) if d > 0)
    w_minus = sum(ranks[i] for i, d in enumerate(nonzero) if d < 0)
    w = min(w_plus, w_minus)
    # Asymptotic normal approximation.
    mu = nz_n * (nz_n + 1) / 4.0
    sigma = math.sqrt(nz_n * (nz_n + 1) * (2 * nz_n + 1) / 24.0)
    z = (w - mu) / sigma if sigma > 0 else 0.0
    p = max(2.0 * (1.0 - _stdnorm_cdf(abs(z))), 1e-300)
    return {
        "n_pairs": nz_n,
        "W": w,
        "W_plus": w_plus,
        "W_minus": w_minus,
        "z": z,
        "p_value_approx": p,
        "significant_at_005": p < 0.05,
    }


def _stdnorm_cdf(x: float) -> float:
    """Standard-normal CDF via the error function approximation."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def summarize_comparison(comp: PairedComparison) -> str:
    """Pretty-print summary suitable for inline benchmark reports."""
    t = paired_t_test(comp)
    if "error" in t:
        return f"{comp.label_a} vs {comp.label_b}: {t['error']}"
    w = wilcoxon_signed_rank(comp)
    sig = "✓ SIGNIFICANT" if t["significant_at_005"] else "✗ not significant"
    wsig = (" / wilcoxon ✓" if w.get("significant_at_005") else
            " / wilcoxon ✗" if "error" not in w else "")
    return (f"{comp.label_a} ({comp.mean_a:.1f}) vs "
            f"{comp.label_b} ({comp.mean_b:.1f})\n"
            f"  speedup: {comp.speedup:.4f}x  "
            f"(mean_diff={t['mean_diff']:+.2f} "
            f"95%CI=[{t['ci95_low']:+.2f}, {t['ci95_high']:+.2f}])\n"
            f"  t({t['df']})={t['t']:+.3f}  "
            f"p≈{t['p_value_approx']:.4f}  "
            f"d={t['cohens_d']:+.3f}  "
            f"{sig}{wsig}\n")
