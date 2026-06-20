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
"""Statistically-rigorous Qwen3 benchmark comparison.

For each ``(bkv_p, bq)`` candidate config:

1. Run the Qwen3-0.6B benchmark with ``--num-measure-rounds=N`` for both
   the baseline and the candidate. Each pair is a single subprocess
   invocation, so JIT compile cost happens once per pair and gets warmed
   out before timing.
2. Collect the per-round throughput vectors.
3. Run a paired t-test + Wilcoxon signed-rank on the per-round pairs.
4. Report mean speedup, 95% CI, p-value, Cohen's d.

Designed to replace ad-hoc single-mean comparisons.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

from tools.kernel.evolve.mutator.diff_applier import apply_diff
from tools.kernel.evolve.stats.significance import (PairedComparison,
                                                    paired_t_test,
                                                    summarize_comparison,
                                                    wilcoxon_signed_rank)

_TUNED_PATH = Path(
    "/home/qizzzh_google_com/tpu-inference/tpu_inference/kernels/"
    "ragged_paged_attention/v3/tuned_block_sizes.py")


def _find_value_line(head_key: str,
                     extra_key: str,
                     dtype_key: str = "q_bfloat16_kv_bfloat16") -> int | None:
    lines = _TUNED_PATH.read_text().splitlines(keepends=True)
    in_dtype = False
    in_head = False
    head_depth = 0
    for i, line in enumerate(lines):
        if f"'{dtype_key}'" in line and "{" in line:
            in_dtype = True
            continue
        if not in_dtype:
            continue
        if f"'{head_key}'" in line and "{" in line:
            in_head = True
            head_depth = 1
            continue
        if in_head:
            head_depth += line.count("{") - line.count("}")
            if head_depth <= 0:
                in_head = False
                continue
            if f"'{extra_key}'" in line:
                return i
    return None


def _make_replace_diff(line_idx: int, bkv: int, bq: int) -> str:
    lines = _TUNED_PATH.read_text().splitlines(keepends=True)
    old_line = lines[line_idx].rstrip("\n")
    new_line = re.sub(r"\(\s*\d+\s*,\s*\d+\s*\)",
                      f"({bkv}, {bq})",
                      old_line,
                      count=1)
    if new_line == old_line:
        raise ValueError("could not rewrite tuple")
    rel = str(_TUNED_PATH).replace("/home/qizzzh_google_com/tpu-inference/",
                                   "")
    return (f"--- a/{rel}\n+++ b/{rel}\n"
            f"@@ -{line_idx + 1},1 +{line_idx + 1},1 @@\n"
            f"-{old_line}\n+{new_line}\n")


def _run_bench(label: str, max_model_len: int, max_tokens: int, rounds: int,
               warmup: int, out_path: Path) -> list[float] | None:
    cmd = [
        sys.executable,
        "-m",
        "tools.kernel.evolve.examples.qwen3_bench",
        "--label",
        label,
        "--output",
        str(out_path),
        "--max-model-len",
        str(max_model_len),
        "--max-tokens",
        str(max_tokens),
        "--num-warmup-rounds",
        str(warmup),
        "--num-measure-rounds",
        str(rounds),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
    if proc.returncode != 0:
        return None
    try:
        data = json.loads(out_path.read_text())
        return [
            t / w for t, w in zip(data["per_round_tokens"],
                                  data["per_round_wall_times_s"])
        ]
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--head-key", default="q_head-16_kv_head-8_head-128")
    p.add_argument("--max-model-len", type=int, default=1024)
    p.add_argument("--max-tokens", type=int, default=64)
    p.add_argument("--rounds", type=int, default=12)
    p.add_argument("--warmup", type=int, default=4)
    p.add_argument("--candidates",
                   default="8:32,8:128,4:32,16:32",
                   help="Comma-separated 'bkv:bq' pairs.")
    p.add_argument("--out-dir",
                   type=Path,
                   default=Path("/tmp/qwen3_stats_bench"))
    args = p.parse_args(argv)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    extra_key = f"max_model_len-{args.max_model_len}-sw-None"
    line_idx = _find_value_line(args.head_key, extra_key)
    if line_idx is None:
        print(f"Could not locate line for {args.head_key}/{extra_key}",
              file=sys.stderr)
        return 2

    # Snapshot the production line so we can restore.
    original_src = _TUNED_PATH.read_text()
    original_line = original_src.splitlines()[line_idx]
    cur_m = re.search(r"\((\d+),\s*(\d+)\)", original_line)
    if cur_m is None:
        print(f"Could not parse current tuple from {original_line!r}",
              file=sys.stderr)
        return 2
    cur_pair = (int(cur_m.group(1)), int(cur_m.group(2)))
    print(f"Production value: {cur_pair} at line {line_idx + 1}")

    candidates: list[tuple[int, int]] = [
        tuple(int(x) for x in c.split(":")) for c in args.candidates.split(",")
    ]

    # ---- Baseline run.
    print(f"\n[baseline {cur_pair}] {args.rounds} rounds...")
    baseline_path = args.out_dir / "baseline.json"
    baseline_vec = _run_bench("baseline", args.max_model_len, args.max_tokens,
                              args.rounds, args.warmup, baseline_path)
    if baseline_vec is None:
        print("Baseline run failed.", file=sys.stderr)
        return 2
    print(f"  baseline mean = {sum(baseline_vec)/len(baseline_vec):.1f} tok/s "
          f"(n={len(baseline_vec)})")

    # ---- Candidate runs.
    comparisons: list[PairedComparison] = []
    summary_rows: list[dict] = []
    for (bkv, bq) in candidates:
        if (bkv, bq) == cur_pair:
            continue
        label = f"bkv{bkv}_bq{bq}"
        print(f"\n[{label}] {args.rounds} rounds...")
        diff = _make_replace_diff(line_idx, bkv, bq)
        r = apply_diff(original_src, diff)
        if not r.success or r.new_source is None:
            print(f"  diff apply failed: {r.error}")
            continue
        try:
            _TUNED_PATH.write_text(r.new_source)
            out = args.out_dir / f"{label}.json"
            cand_vec = _run_bench(label, args.max_model_len, args.max_tokens,
                                  args.rounds, args.warmup, out)
        finally:
            _TUNED_PATH.write_text(original_src)
        if cand_vec is None:
            print(f"  {label} run failed.")
            continue
        comp = PairedComparison(label_a=f"baseline {cur_pair}",
                                label_b=f"{(bkv, bq)}",
                                a_values=baseline_vec,
                                b_values=cand_vec)
        comparisons.append(comp)
        t = paired_t_test(comp)
        w = wilcoxon_signed_rank(comp)
        print(summarize_comparison(comp))
        summary_rows.append({
            "candidate": (bkv, bq),
            "baseline_mean": comp.mean_a,
            "candidate_mean": comp.mean_b,
            "speedup": comp.speedup,
            "n_pairs": t.get("n_pairs"),
            "p_value_approx": t.get("p_value_approx"),
            "cohens_d": t.get("cohens_d"),
            "ci95_low": t.get("ci95_low"),
            "ci95_high": t.get("ci95_high"),
            "paired_t_significant": t.get("significant_at_005"),
            "wilcoxon_significant": w.get("significant_at_005"),
            "baseline_vec": baseline_vec,
            "candidate_vec": cand_vec,
        })

    summary_path = args.out_dir / "stats_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "head_key": args.head_key,
                "max_model_len": args.max_model_len,
                "current": cur_pair,
                "rounds_per_config": args.rounds,
                "warmup_rounds": args.warmup,
                "comparisons": summary_rows
            },
            indent=2,
            default=str))

    # ---- Final ranked report.
    print()
    print("=" * 78)
    print(f"Stats-bench: {args.head_key} / max_model_len={args.max_model_len}")
    print(
        f"  production={cur_pair} measured at {sum(baseline_vec)/len(baseline_vec):.1f} tok/s "
        f"(n={len(baseline_vec)} rounds)")
    print()
    print(f"  {'candidate':12s}  {'mean':>10s}  {'speedup':>9s}  "
          f"{'95% CI':>22s}  {'p_t':>9s}  {'wilcoxon':>9s}  {'verdict':<20s}")
    print("  " + "-" * 110)
    sig_wins = []
    for row in sorted(summary_rows, key=lambda r: -r["speedup"]):
        c = row["candidate"]
        verdict = "✗ within noise"
        if row.get("paired_t_significant") and row["speedup"] > 1.0:
            verdict = "✓ SIGNIFICANT win"
            sig_wins.append(row)
        elif row.get("paired_t_significant") and row["speedup"] < 1.0:
            verdict = "✗ significantly worse"
        print(f"  {str(c):12s}  {row['candidate_mean']:>10.1f}  "
              f"{row['speedup']:>8.4f}x  "
              f"[{row['ci95_low']:+8.2f}, {row['ci95_high']:+8.2f}]  "
              f"{row['p_value_approx']:>9.4f}  "
              f"{'✓' if row['wilcoxon_significant'] else '✗':>9s}  "
              f"{verdict:<20s}")
    if sig_wins:
        winner = max(sig_wins, key=lambda r: r["speedup"])
        print(f"\n  Significant winner: {winner['candidate']} "
              f"@ {winner['speedup']:.4f}x "
              f"(95% CI lower bound = +{winner['ci95_low']:.2f} tok/s)")
    else:
        print(
            f"\n  No candidate beat baseline with p<0.05 over {args.rounds} rounds."
        )
    print(f"\nFull JSON: {summary_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
