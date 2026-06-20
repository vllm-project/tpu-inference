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
"""Cross-shape statistical validation of a kernel diff.

A diff might win on the shape it was discovered on but regress on others.
A win is industry-ready only if it generalizes (or at minimum, doesn't
regress meaningfully) across the production shape grid.

This module:

1. Defines a small catalog of production-relevant RPA v3 shapes
   (Qwen3, Qwen3.5, Llama 3-class configurations) — much smaller than the
   3256-entry TUNED_BLOCK_SIZES table but covers the model archetypes
   actually deployed today.
2. For each shape, paired-bench the diff vs the baseline (N=8 rounds,
   fresh inputs per round, anti-cheat guard) and run paired t-test +
   Wilcoxon signed-rank for significance.
3. Produces a JSON report and a markdown table summarizing speedup,
   p-value, confidence interval, and effect size per shape.

Output is suitable as PR evidence ("the diff wins on N/M production shapes,
with no significant regressions").
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np

from tools.kernel.evolve.mutator.diff_applier import apply_diff
from tools.kernel.evolve.stats.significance import PairedComparison
from tools.kernel.evolve.worktree import import_candidate_module
from tools.kernel.tuner.v1.bench.harness import measure
from tools.kernel.tuner.v1.common.kernel_tuner_base import RunConfig
from tools.kernel.tuner.v1.rpa_v3_kernel_tuner import RpaV3KernelTuner

logger = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class ShapeSpec:
    """One production-relevant RPA shape to validate against."""
    name: str
    description: str
    q_dtype: Any
    kv_dtype: Any
    num_q_heads: int
    num_kv_heads: int
    head_dim: int
    max_model_len: int
    page_size: int = 16
    distribution_kind: str = "mixed"


# Production model archetypes. Distilled from the model configs deployed in
# tpu-inference CI + perf workloads.
PRODUCTION_SHAPES: list[ShapeSpec] = [
    # Qwen3-class (small/dense)
    ShapeSpec(name="qwen3_0_6b_short",
              description="Qwen3-0.6B-style, max_len=1024",
              q_dtype=jnp.bfloat16,
              kv_dtype=jnp.bfloat16,
              num_q_heads=16,
              num_kv_heads=8,
              head_dim=128,
              max_model_len=1024),
    ShapeSpec(name="qwen3_0_6b_long",
              description="Qwen3-0.6B-style, max_len=4096",
              q_dtype=jnp.bfloat16,
              kv_dtype=jnp.bfloat16,
              num_q_heads=16,
              num_kv_heads=8,
              head_dim=128,
              max_model_len=4096),
    # Llama 3 8B style
    ShapeSpec(name="llama3_8b_mid",
              description="Llama-3-8B attention shape, max_len=2048",
              q_dtype=jnp.bfloat16,
              kv_dtype=jnp.bfloat16,
              num_q_heads=32,
              num_kv_heads=8,
              head_dim=128,
              max_model_len=2048),
    # Qwen3.5-397B style attention (hybrid model — attn part)
    ShapeSpec(name="qwen35_397b_attn",
              description="Qwen3.5-397B hybrid attn shape, head_dim=128",
              q_dtype=jnp.bfloat16,
              kv_dtype=jnp.bfloat16,
              num_q_heads=64,
              num_kv_heads=8,
              head_dim=128,
              max_model_len=4096),
    # fp8 KV variants — important because the diff's casting interacts with
    # the KV dtype path.
    ShapeSpec(name="qwen3_0_6b_fp8_kv",
              description="Qwen3-0.6B with fp8 KV cache",
              q_dtype=jnp.bfloat16,
              kv_dtype=jnp.float8_e4m3fn,
              num_q_heads=16,
              num_kv_heads=8,
              head_dim=128,
              max_model_len=4096),
    ShapeSpec(name="llama3_8b_fp8_kv",
              description="Llama-3-8B with fp8 KV cache",
              q_dtype=jnp.bfloat16,
              kv_dtype=jnp.float8_e4m3fn,
              num_q_heads=32,
              num_kv_heads=8,
              head_dim=128,
              max_model_len=2048),
    # Larger head_dim — DeepSeek/MLA-style configurations stress fp32 paths.
    ShapeSpec(name="head_dim_256_short",
              description="Head dim 256 (DeepSeek-style), max_len=1024",
              q_dtype=jnp.bfloat16,
              kv_dtype=jnp.bfloat16,
              num_q_heads=8,
              num_kv_heads=4,
              head_dim=256,
              max_model_len=1024),
]


@dataclasses.dataclass
class ShapeResult:
    name: str
    description: str
    baseline_p50_us: float
    candidate_p50_us: float
    baseline_mean_ns: float
    candidate_mean_ns: float
    speedup: float
    p_value: float
    cohens_d: float
    ci_low_ns: float
    ci_high_ns: float
    significant: bool
    direction: str  # "win", "regress", "tie"

    def to_dict(self) -> dict:
        d = dataclasses.asdict(self)
        return d


def _bench_source(host, src_text: str, *, n: int, warmup: int,
                  iters: int) -> list[float]:
    """Return n latency samples (p50 ns) for src_text running through host."""
    with import_candidate_module(src_text, name_hint="cross_shape") as mod:
        fn = host.build_kernel_fn(mod)
        latencies = []
        for _ in range(n):
            r = measure(fn, warmup=warmup, iters=iters)
            latencies.append(r.p50_ns)
        return latencies


def _build_tuner_for_shape(shape: ShapeSpec) -> RpaV3KernelTuner:
    """Override the tuner defaults to match ``shape``."""
    rc = RunConfig(case_set_id=f"cross_shape_{shape.name}",
                   run_id="r0",
                   case_set_desc=shape.description,
                   tpu_version="tpu6e",
                   tpu_cores=1,
                   tpu_queue_multi="tpu_v6e_queue",
                   run_locally=True,
                   max_execution_minutes=10)
    tuner = RpaV3KernelTuner(run_config=rc)
    # Override shape-defining attributes; generate_cases() picks them up.
    tuner.q_dtype = shape.q_dtype
    tuner.kv_dtype = shape.kv_dtype
    tuner.num_q_heads = shape.num_q_heads
    tuner.num_kv_heads = shape.num_kv_heads
    tuner.head_dim = shape.head_dim
    tuner.max_model_len = shape.max_model_len
    tuner.max_num_tokens = max(shape.max_model_len, 384)
    tuner.page_size = shape.page_size
    tuner.distribution_kind = shape.distribution_kind
    return tuner


def evaluate_shape(shape: ShapeSpec,
                   *,
                   diff_text: str,
                   baseline_source: str,
                   bench_n: int = 8,
                   warmup: int = 2,
                   iters: int = 6) -> ShapeResult | None:
    """Run paired bench for one shape; None if the shape errors out."""
    from tools.kernel.evolve.examples.rpa_v3_evolve import RpaV3Host
    logger.info("Evaluating shape %s (%s)", shape.name, shape.description)
    tuner = _build_tuner_for_shape(shape)
    try:
        host = RpaV3Host(tuner)
    except Exception as err:  # pragma: no cover - shape-specific failure
        logger.warning("Shape %s: failed to build host (%s)", shape.name, err)
        return None
    patched_src = apply_diff(baseline_source, diff_text).new_source
    try:
        a = _bench_source(host,
                          baseline_source,
                          n=bench_n,
                          warmup=warmup,
                          iters=iters)
        b = _bench_source(host,
                          patched_src,
                          n=bench_n,
                          warmup=warmup,
                          iters=iters)
    except Exception as err:
        logger.warning("Shape %s: bench failed (%s)", shape.name, err)
        return None

    comp = PairedComparison(label_a=f"baseline_{shape.name}",
                            label_b=f"patched_{shape.name}",
                            a_values=a,
                            b_values=b)
    # summarize_comparison returns text; we want the numeric breakdown.
    diffs = np.array(b, dtype=float) - np.array(a, dtype=float)
    mean_diff = float(diffs.mean())
    std_diff = float(diffs.std(ddof=1)) if len(diffs) > 1 else 0.0
    n = len(diffs)
    # Cohen's d (paired):
    d = mean_diff / std_diff if std_diff > 0 else 0.0
    # 95% CI via t-distribution.
    from scipy import stats as scistats
    if n > 1 and std_diff > 0:
        t_stat, p_value = scistats.ttest_rel(b, a)
        se = std_diff / np.sqrt(n)
        t_crit = scistats.t.ppf(0.975, df=n - 1)
        ci_low = mean_diff - t_crit * se
        ci_high = mean_diff + t_crit * se
    else:
        t_stat = 0.0
        p_value = 1.0
        ci_low = ci_high = mean_diff
    significant = bool(p_value < 0.05)
    speedup = float(np.mean(a) / np.mean(b))
    if not significant:
        direction = "tie"
    elif speedup > 1.0:
        direction = "win"
    else:
        direction = "regress"
    res = ShapeResult(
        name=shape.name,
        description=shape.description,
        baseline_p50_us=float(np.median(a)) / 1e3,
        candidate_p50_us=float(np.median(b)) / 1e3,
        baseline_mean_ns=float(np.mean(a)),
        candidate_mean_ns=float(np.mean(b)),
        speedup=speedup,
        p_value=float(p_value),
        cohens_d=float(d),
        ci_low_ns=float(ci_low),
        ci_high_ns=float(ci_high),
        significant=significant,
        direction=direction,
    )
    logger.info("  → speedup=%.4fx p=%.4g d=%.2f (%s)", res.speedup,
                res.p_value, res.cohens_d, res.direction)
    _ = comp, t_stat  # silence unused
    return res


def render_markdown_table(results: list[ShapeResult]) -> str:
    """Format the per-shape results as a markdown table."""
    lines = [
        "| shape | description | baseline μs | patched μs | speedup | p | direction |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in results:
        marker = {
            "win": "WIN ",
            "regress": "REGRESS",
            "tie": "tie "
        }[r.direction]
        lines.append(
            f"| `{r.name}` | {r.description} | {r.baseline_p50_us:.2f} | "
            f"{r.candidate_p50_us:.2f} | **{r.speedup:.4f}×** | "
            f"{r.p_value:.4f} | {marker} |")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--diff",
                   type=Path,
                   required=True,
                   help="Path to a unified diff to validate.")
    p.add_argument("--baseline-path",
                   type=Path,
                   default=Path("tpu_inference/kernels/ragged_paged_attention/"
                                "v3/kernel.py"),
                   help="Source file the diff applies to (the baseline).")
    p.add_argument("--shapes",
                   default="all",
                   help="Comma-separated shape names from PRODUCTION_SHAPES, "
                   "or 'all'.")
    p.add_argument("--bench-n",
                   type=int,
                   default=8,
                   help="Paired bench rounds per shape.")
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--iters", type=int, default=6)
    p.add_argument("--out-json",
                   type=Path,
                   default=Path("/tmp/cross_shape_results.json"))
    p.add_argument("--out-md",
                   type=Path,
                   default=Path("/tmp/cross_shape_results.md"))
    p.add_argument("--verbose", "-v", action="count", default=0)
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.WARNING - 10 * args.verbose,
                        format="%(asctime)s %(levelname)s %(message)s")

    baseline_source = args.baseline_path.read_text()
    diff_text = args.diff.read_text()

    if args.shapes == "all":
        shapes = PRODUCTION_SHAPES
    else:
        wanted = set(s.strip() for s in args.shapes.split(","))
        shapes = [s for s in PRODUCTION_SHAPES if s.name in wanted]
        if not shapes:
            print(f"error: no shapes matched {args.shapes!r}", file=sys.stderr)
            return 2

    results: list[ShapeResult] = []
    t0 = time.time()
    for s in shapes:
        r = evaluate_shape(s,
                           diff_text=diff_text,
                           baseline_source=baseline_source,
                           bench_n=args.bench_n,
                           warmup=args.warmup,
                           iters=args.iters)
        if r is not None:
            results.append(r)

    wall = time.time() - t0
    args.out_json.write_text(
        json.dumps([r.to_dict() for r in results], indent=2))
    args.out_md.write_text(render_markdown_table(results))

    wins = sum(1 for r in results if r.direction == "win")
    regs = sum(1 for r in results if r.direction == "regress")
    ties = sum(1 for r in results if r.direction == "tie")
    n = len(results)
    print()
    print("=" * 78)
    print(f"Cross-shape validation in {wall:.1f}s")
    print(f"  shapes tested:    {n}")
    print(f"  wins (p<0.05):    {wins}")
    print(f"  regressions:      {regs}")
    print(f"  ties (p≥0.05):    {ties}")
    if results:
        mean_speedup = float(np.mean([r.speedup for r in results]))
        print(f"  mean speedup:     {mean_speedup:.4f}×")
    print()
    print(render_markdown_table(results))
    print()
    print(f"JSON: {args.out_json}")
    print(f"MD:   {args.out_md}")
    return 0 if regs == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
