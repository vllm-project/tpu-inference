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
"""``nightly_evolve``: end-to-end CI runner.

Designed to be invoked by Buildkite (or any cron) once per night:

* Detect kernels changed in the last 24h (git log) — narrow target set.
* For each target kernel, pull a kernel-specific `recipe` describing the
  rule library, the shape sweep, and the candidate budget.
* Run the orchestrator (or the sub-optimal sweep, depending on recipe).
* Collect verified wins.
* Produce a single mergeable patch file and a Markdown PR body.
* Print a one-line status to stdout for the CI agent to parse.

The recipes live in ``tools/kernel/evolve/ci/recipes/`` — see the example
recipe for ``ragged_paged_attention/v3``.
"""

from __future__ import annotations

import argparse
import dataclasses
import importlib
import json
import logging
import subprocess
import sys
import time
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class NightlyResult:
    target_kernel: str
    wins_count: int
    diff_path: Path | None
    summary_path: Path | None
    telemetry_path: Path | None
    wall_time_s: float
    error: str | None = None


def _changed_kernels_since(hours: int) -> list[str]:
    """Use git log to find files under kernels/ changed in the last N hours."""
    proc = subprocess.run(
        [
            "git", "-C", "/home/qizzzh_google_com/tpu-inference", "log",
            "--since", f"{hours} hours ago", "--name-only", "--pretty=format:",
            "tpu_inference/kernels/"
        ],
        capture_output=True,
        text=True,
    )
    paths = {ln.strip() for ln in proc.stdout.splitlines() if ln.strip()}
    # Map file paths to high-level kernel-family directories.
    kernel_families: set[str] = set()
    for p in paths:
        parts = Path(p).parts
        if len(parts) >= 3 and parts[0] == "tpu_inference" and parts[
                1] == "kernels":
            kernel_families.add(parts[2])
    return sorted(kernel_families)


def _run_recipe(recipe_path: Path, out_dir: Path) -> NightlyResult:
    """Execute a recipe — a Python module exposing a ``run(out_dir)`` callable."""
    spec = importlib.util.spec_from_file_location("_recipe", recipe_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    t0 = time.time()
    try:
        result = mod.run(out_dir=out_dir)
    except Exception as err:
        return NightlyResult(
            target_kernel=recipe_path.stem,
            wins_count=0,
            diff_path=None,
            summary_path=None,
            telemetry_path=None,
            wall_time_s=time.time() - t0,
            error=str(err),
        )
    return NightlyResult(
        target_kernel=result.get("target_kernel", recipe_path.stem),
        wins_count=int(result.get("wins_count", 0)),
        diff_path=Path(result["diff_path"])
        if result.get("diff_path") else None,
        summary_path=(Path(result["summary_path"])
                      if result.get("summary_path") else None),
        telemetry_path=(Path(result["telemetry_path"])
                        if result.get("telemetry_path") else None),
        wall_time_s=time.time() - t0,
        error=result.get("error"),
    )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--recipe",
                   action="append",
                   default=None,
                   help="Recipe file paths to run. May be repeated.")
    p.add_argument("--recipes-dir",
                   type=Path,
                   default=Path(__file__).parent / "recipes")
    p.add_argument("--out-dir", type=Path, default=Path("/tmp/evolve_nightly"))
    p.add_argument("--changed-hours",
                   type=int,
                   default=24,
                   help="Run recipes for kernels changed in the last N hours. "
                   "0 = run all recipes.")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.recipe:
        recipe_paths = [Path(r) for r in args.recipe]
    else:
        if not args.recipes_dir.exists():
            print(f"recipes dir {args.recipes_dir} not found", file=sys.stderr)
            return 2
        recipe_paths = sorted(args.recipes_dir.glob("*.py"))
        recipe_paths = [p for p in recipe_paths if not p.stem.startswith("_")]
        if args.changed_hours > 0:
            changed = set(_changed_kernels_since(args.changed_hours))
            recipe_paths = [
                p for p in recipe_paths if any(c in p.stem for c in changed)
            ]
            print(f"Changed kernels in last {args.changed_hours}h: {changed}")
            print(f"Recipes to run: {[p.name for p in recipe_paths]}")
    if not recipe_paths:
        print("No recipes to run.")
        return 0

    if args.dry_run:
        for r in recipe_paths:
            print(f"  would run: {r}")
        return 0

    results: list[NightlyResult] = []
    for r in recipe_paths:
        print(f"\n=== {r.name} ===")
        results.append(_run_recipe(r, args.out_dir))

    # Print final status line + structured summary.
    total_wins = sum(r.wins_count for r in results)
    print()
    print("=" * 70)
    print(
        f"nightly_evolve: {len(results)} recipes, {total_wins} verified wins")
    for r in results:
        status = "OK" if r.error is None else f"ERR: {r.error}"
        print(f"  {r.target_kernel:24s}  wins={r.wins_count:3d}  "
              f"wall={r.wall_time_s:5.1f}s  {status}")
    summary_path = args.out_dir / "nightly_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "recipes": [{
                    "target_kernel": r.target_kernel,
                    "wins_count": r.wins_count,
                    "diff_path": str(r.diff_path) if r.diff_path else None,
                    "wall_time_s": r.wall_time_s,
                    "error": r.error,
                } for r in results],
                "total_wins":
                total_wins,
                "wall_time_s":
                sum(r.wall_time_s for r in results)
            },
            indent=2,
            default=str))
    print(f"\nSummary: {summary_path}")
    return 0 if total_wins > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
