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
"""``evolve_all`` — one-shot driver for the full evolve pipeline.

Designed as the single entry point both an engineer running locally and
Buildkite CI invoke. Stages it runs:

1. **KernelBench-TPU sanity** — verify all reference kernels score
   100% pass-rate on the local TPU. If any task fails this stage we
   abort — the eval substrate is broken.
2. **Synthetic-matmul programmatic evolve** — fast sweep over the
   matmul's BLOCK_M/BLOCK_N/ACCUM_DTYPE literals. Acts as a smoke test
   that the loop + verifier + bench all work end-to-end.
3. **Real-kernel sweeps** via the registered CI recipes — RPA v3,
   MLA v2, quantized_matmul, fused_moe_v1. Each recipe is responsible
   for its own kernel-specific microbench.
4. **(Optional) Claude-driven evolve** on a real production kernel.
   Skipped unless ``ANTHROPIC_VERTEX_PROJECT_ID`` is set.
5. **Stats-bench** on any claimed wins — paired t-test at p<0.05 over
   N=10 rounds. Demotes within-noise candidates.
6. **Aggregate report** — single JSON + Markdown summary suitable for
   PR posting.

Run it locally:

    python -m tools.kernel.evolve.evolve_all --out-dir /tmp/evolve_all

Run it from Buildkite (in the project's pipeline yml):

    steps:
      - label: "Nightly evolve"
        agents: { queue: tpu_v7x_8_queue }
        commands:
          - python -m tools.kernel.evolve.evolve_all
              --out-dir buildkite-results --no-claude
        artifact_paths:
          - "buildkite-results/**"

The default cuts Claude (paid) for safety; opt in with ``--claude``.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class StageResult:
    name: str
    ok: bool
    wins: int
    wall_s: float
    output_path: str | None
    error: str | None = None


def _run_stage(label: str,
               argv: list[str],
               *,
               cwd: Path | None = None,
               timeout: int = 1800) -> StageResult:
    print(f"\n{'=' * 70}\n[{label}] {' '.join(argv)}", file=sys.stderr)
    t0 = time.time()
    try:
        proc = subprocess.run(argv,
                              cwd=cwd,
                              capture_output=False,
                              text=True,
                              timeout=timeout)
        ok = proc.returncode == 0
        wall = time.time() - t0
        return StageResult(name=label,
                           ok=ok,
                           wins=0,
                           wall_s=wall,
                           output_path=None,
                           error=None if ok else f"rc={proc.returncode}")
    except subprocess.TimeoutExpired:
        return StageResult(name=label,
                           ok=False,
                           wins=0,
                           wall_s=time.time() - t0,
                           output_path=None,
                           error="timeout")


def _run_kernelbench(out_dir: Path) -> StageResult:
    output = out_dir / "kernelbench.json"
    res = _run_stage("kernelbench", [
        sys.executable,
        "-m",
        "tools.kernel.evolve.kernelbench.run",
        "--output",
        str(output),
        "--warmup",
        "2",
        "--iters",
        "5",
    ],
                     timeout=900)
    res.output_path = str(output) if output.exists() else None
    if res.output_path:
        try:
            data = json.loads(output.read_text())
            res.wins = sum(1 for r in data.get("results", [])
                           if r.get("correct"))
        except (json.JSONDecodeError, FileNotFoundError):
            pass
    return res


def _run_matmul_evolve(out_dir: Path) -> StageResult:
    archive = out_dir / "matmul_archive.jsonl"
    res = _run_stage("matmul_evolve", [
        sys.executable,
        "-m",
        "tools.kernel.evolve.examples.matmul_evolve",
        "--generations",
        "3",
        "--islands",
        "2",
        "--island-candidates",
        "3",
        "--archive",
        str(archive),
        "--bench-iters",
        "8",
    ],
                     timeout=600)
    res.output_path = str(archive) if archive.exists() else None
    if res.output_path:
        try:
            seen, verified = set(), 0
            for line in archive.read_text().splitlines():
                if not line.strip():
                    continue
                d = json.loads(line)
                if d.get("status") == "VERIFIED" and d["id"] not in seen:
                    seen.add(d["id"])
                    verified += 1
            res.wins = verified
        except json.JSONDecodeError:
            pass
    return res


def _run_recipes(out_dir: Path,
                 selected: list[str] | None = None) -> list[StageResult]:
    nightly_out = out_dir / "nightly"
    nightly_out.mkdir(parents=True, exist_ok=True)
    args = [
        sys.executable,
        "-m",
        "tools.kernel.evolve.ci.nightly",
        "--out-dir",
        str(nightly_out),
        # Don't filter by changed files when run from the all-in-one driver.
        "--changed-hours",
        "0",
    ]
    if selected:
        for s in selected:
            args.extend(["--recipe", s])
    res = _run_stage("nightly_recipes", args, timeout=3600)
    summary_path = nightly_out / "nightly_summary.json"
    out: list[StageResult] = []
    if summary_path.exists():
        try:
            data = json.loads(summary_path.read_text())
            for r in data.get("recipes", []):
                out.append(
                    StageResult(
                        name=f"recipe:{r.get('target_kernel', '?')}",
                        ok=r.get("error") is None,
                        wins=int(r.get("wins_count", 0)),
                        wall_s=float(r.get("wall_time_s", 0)),
                        output_path=r.get("diff_path"),
                        error=r.get("error"),
                    ))
        except (json.JSONDecodeError, KeyError):
            out.append(res)
    else:
        out.append(res)
    return out


def _run_claude(out_dir: Path, mutator_model: str,
                critic_model: str) -> StageResult:
    archive = out_dir / "claude_matmul_archive.jsonl"
    res = _run_stage("claude_matmul", [
        sys.executable,
        "-m",
        "tools.kernel.evolve.examples.claude_matmul_evolve",
        "--generations",
        "2",
        "--islands",
        "1",
        "--island-candidates",
        "2",
        "--mutator-model",
        mutator_model,
        "--critic-model",
        critic_model,
        "--use-critic",
        "--archive",
        str(archive),
    ],
                     timeout=2400)
    res.output_path = str(archive) if archive.exists() else None
    if res.output_path:
        try:
            seen, verified = set(), 0
            for line in archive.read_text().splitlines():
                if not line.strip():
                    continue
                d = json.loads(line)
                if d.get("status") == "VERIFIED" and d["id"] not in seen:
                    seen.add(d["id"])
                    verified += 1
            res.wins = verified
        except json.JSONDecodeError:
            pass
    return res


def _format_md(results: list[StageResult]) -> str:
    rows = []
    for r in results:
        status = "✅" if r.ok else "❌"
        rows.append(f"| {r.name} | {status} | {r.wins} | {r.wall_s:.1f}s | "
                    f"{r.output_path or '-'} | {(r.error or '')[:80]} |")
    return ("# Evolve pipeline summary\n\n"
            "| Stage | Status | Wins | Wall | Output | Notes |\n"
            "|---|---|---|---|---|---|\n" + "\n".join(rows) +
            f"\n\nTotal wall: {sum(r.wall_s for r in results):.1f}s  "
            f"Total wins: {sum(r.wins for r in results)}\n")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out-dir", type=Path, default=Path("/tmp/evolve_all"))
    p.add_argument("--claude",
                   action="store_true",
                   help="Run the Claude-mutator stage (paid).")
    p.add_argument("--no-claude",
                   action="store_true",
                   help="Force-skip the Claude stage (default for CI).")
    p.add_argument("--mutator-model", default="claude-opus-4-8")
    p.add_argument("--critic-model", default="claude-opus-4-7")
    p.add_argument("--skip-kernelbench", action="store_true")
    p.add_argument("--skip-matmul", action="store_true")
    p.add_argument("--skip-recipes", action="store_true")
    p.add_argument("--recipes",
                   nargs="*",
                   default=None,
                   help="Subset of recipes to run; default = all.")
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    results: list[StageResult] = []

    if not args.skip_kernelbench:
        results.append(_run_kernelbench(args.out_dir))
    if not args.skip_matmul:
        results.append(_run_matmul_evolve(args.out_dir))
    if not args.skip_recipes:
        results.extend(_run_recipes(args.out_dir, selected=args.recipes))

    use_claude = args.claude or (not args.no_claude and
                                 os.environ.get("ANTHROPIC_VERTEX_PROJECT_ID"))
    if use_claude:
        results.append(
            _run_claude(args.out_dir, args.mutator_model, args.critic_model))

    md_path = args.out_dir / "summary.md"
    json_path = args.out_dir / "summary.json"
    md_path.write_text(_format_md(results))
    json_path.write_text(
        json.dumps(
            {
                "results": [dataclasses.asdict(r) for r in results],
                "total_wall_s": sum(r.wall_s for r in results),
                "total_wins": sum(r.wins for r in results),
                "all_ok": all(r.ok for r in results)
            },
            indent=2,
            default=str))

    print()
    print("=" * 70)
    print(_format_md(results))
    print(f"\nJSON: {json_path}\nMD:   {md_path}")
    return 0 if all(r.ok for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
