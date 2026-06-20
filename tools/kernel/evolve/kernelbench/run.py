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
"""CLI entrypoint for the KernelBench-TPU subset run."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from tools.kernel.evolve.kernelbench.runner import fast_p, run_subset
from tools.kernel.evolve.kernelbench.tasks import TASKS


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--output",
                   type=Path,
                   default=Path("/tmp/kernelbench_tpu.json"))
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--iters", type=int, default=10)
    p.add_argument("--levels",
                   type=str,
                   default="1,2",
                   help="Comma-separated level filter.")
    args = p.parse_args(argv)
    levels = {int(lv) for lv in args.levels.split(",")}
    tasks = [t for t in TASKS if t.level in levels]

    print(
        f"Running KernelBench-TPU subset: {len(tasks)} tasks "
        f"(levels={sorted(levels)})",
        file=sys.stderr)
    results = run_subset(tasks=tasks, warmup=args.warmup, iters=args.iters)

    # Pretty table.
    print()
    print(f"{'task':24s}  {'lvl':>3s}  {'baseline_us':>11s}  "
          f"{'candidate_us':>12s}  {'speedup':>8s}  {'correct':>7s}  "
          f"{'cosine':>8s}")
    print("-" * 90)
    for r in results:
        bp = r.baseline_p50_ns / 1e3
        cp = r.candidate_p50_ns / 1e3 if r.candidate_p50_ns else None
        cp_s = f"{cp:>11.2f}" if cp is not None else "        n/a"
        speedup_s = f"{r.speedup:>7.3f}x" if r.speedup is not None else "       n/a"
        cosine_s = f"{r.cosine:.6f}" if r.cosine is not None else "      n/a"
        print(f"{r.task_name:24s}  {r.level:>3d}  {bp:>11.2f}   {cp_s}  "
              f"{speedup_s}  {'✓' if r.correct else '✗':>7s}  {cosine_s}")
    print()
    print(f"fast_1 = {fast_p(results, 1.0):.2%}  "
          f"fast_2 = {fast_p(results, 2.0):.2%}  "
          f"total_tasks = {len(results)}  "
          f"correct = {sum(1 for r in results if r.correct)}")

    args.output.write_text(
        json.dumps(
            {
                "results": [vars(r) for r in results],
                "fast_1": fast_p(results, 1.0),
                "fast_2": fast_p(results, 2.0)
            },
            indent=2,
            default=str))
    print(f"\nResult JSON: {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
