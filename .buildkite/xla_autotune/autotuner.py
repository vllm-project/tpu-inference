#!/usr/bin/env python3
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
"""OFAT autotuner for XLA / libtpu flags.

For each shard (1-based ``--slice-index`` of ``--slice-count``):
runs ``--baseline-runs`` stock baselines, then one trial per candidate flag
in the shard's contiguous slice of ``--flag-list-file``.  Each trial's record
is written to ``<artifact-dir>/<trial_id>.json`` and appended to
``summary.jsonl`` as it finishes, so the host watcher can ship partial
progress.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import sys
import time
import traceback
from datetime import datetime
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vllm_test_framework import VLLMTestFramework, VLLMTestParam  # noqa: E402


@dataclasses.dataclass
class Trial:
    trial_id: str
    kind: str  # "baseline" | "candidate"
    flag: Optional[str]  # None for baseline


def slice_flags(flags: List[str], idx: int, count: int) -> List[str]:
    """Return contiguous shard ``idx`` of ``count`` (1-based)."""
    if not 1 <= idx <= count:
        raise ValueError(f"slice_index {idx} out of [1, {count}]")
    chunk = math.ceil(len(flags) / count)
    start = (idx - 1) * chunk
    end = len(flags) if idx == count else start + chunk
    return flags[start:end]


def plan(
    candidates: List[str],
    baseline_runs: int,
    skip_candidates: int = 0,
) -> List[Trial]:
    """Baselines + the candidate slice with the first ``skip_candidates``
    dropped.  Baselines always run fresh; trial indices stay stable across
    runs (e.g. ``skip_candidates=12`` resumes at ``cand_013``)."""
    trials = [
        Trial(f"baseline_{i+1:02d}", "baseline", None)
        for i in range(baseline_runs)
    ]
    for i, flag in enumerate(candidates):
        if i < skip_candidates:
            continue
        trials.append(Trial(f"cand_{i+1:03d}", "candidate", flag))
    return trials


def load_lines(path: str) -> List[str]:
    with open(path) as f:
        return [
            ln.strip() for ln in f
            if ln.strip() and not ln.lstrip().startswith("#")
        ]


def apply_overrides(param: VLLMTestParam, overrides: Dict[str, Any]) -> None:
    for k, v in overrides.items():
        if k.startswith("_"):
            continue
        if not hasattr(param, k):
            raise ValueError(f"unknown VLLMTestParam field {k!r}")
        setattr(param, k, v)


def run_trial(
    trial: Trial,
    model: str,
    overrides: Dict[str, Any],
    artifact_dir: str,
    target_metric: str,
    dry_run: bool,
    summary_fp,
) -> Dict[str, Any]:
    """Run one trial, persist its record, never raise."""
    rec: Dict[str, Any] = {
        "trial_id": trial.trial_id,
        "kind": trial.kind,
        "flag": trial.flag,
        "extra_flags": [trial.flag] if trial.flag else [],
        "model": model,
        "target_metric": target_metric,
        "started_utc": datetime.utcnow().isoformat(),
        "success": False,
        "metrics": {},
        "target_value": None,
        "error": "",
    }
    t0 = time.time()
    logs_root = os.path.join(artifact_dir, "logs")
    os.makedirs(logs_root, exist_ok=True)
    exp_dir: Optional[str] = None

    try:
        p = VLLMTestParam(
            model_name=model,
            extra_libtpu_init_args=rec["extra_flags"],
            base_log_dir=logs_root,
            tag=f"autotune_{trial.trial_id}",
        )
        apply_overrides(p, overrides)
        fw = VLLMTestFramework(p, dry_run=dry_run)
        exp_dir = fw.exp_dir
        r = fw.run_benchmark()
        rec.update(success=r.success, metrics=r.metrics, error=r.error)
        for sub in r.metrics.values():
            if isinstance(sub, dict) and target_metric in sub:
                rec["target_value"] = sub[target_metric]
                break
    except Exception as e:  # noqa: BLE001
        rec["error"] = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"

    rec["duration_sec"] = round(time.time() - t0, 2)
    rec["finished_utc"] = datetime.utcnow().isoformat()

    out_path = os.path.join(artifact_dir, f"{trial.trial_id}.json")
    with open(out_path, "w") as f:
        json.dump(rec, f, indent=2)
    summary_fp.write(json.dumps(rec) + "\n")
    summary_fp.flush()

    if exp_dir and os.path.exists(exp_dir):
        _write_tag(exp_dir, rec)
        # ".done" marker tells the host watcher this log bundle is complete
        # and safe to upload as a unit.
        open(exp_dir + ".done", "w").close()

    print(
        f"[autotune] {trial.trial_id} kind={trial.kind} success={rec['success']} "
        f"{target_metric}={rec['target_value']} dur={rec['duration_sec']}s",
        flush=True,
    )
    return rec


def _write_tag(exp_dir: str, rec: Dict[str, Any]) -> None:
    err = rec["error"].splitlines()[0] if rec["error"] else ""
    with open(os.path.join(exp_dir, "_tag.txt"), "w") as f:
        for k in (
                "trial_id",
                "kind",
                "flag",
                "model",
                "target_metric",
                "target_value",
                "success",
                "duration_sec",
                "started_utc",
                "finished_utc",
        ):
            f.write(f"{k}: {rec.get(k)}\n")
        f.write(f"error: {err}\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--flag-list-file", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--benchmark-args-json",
                    default=None,
                    help="Optional JSON file of VLLMTestParam overrides.")
    ap.add_argument("--target-metric", default="total_token_throughput")
    ap.add_argument("--slice-index", type=int, default=1)
    ap.add_argument("--slice-count", type=int, default=1)
    ap.add_argument("--baseline-runs", type=int, default=2)
    ap.add_argument("--skip-candidates",
                    type=int,
                    default=0,
                    help="Drop the first N candidate trials from this shard's "
                    "slice (for resume).  Baselines always run fresh; "
                    "trial indices stay stable.")
    ap.add_argument("--artifact-dir", required=True)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    flags = load_lines(args.flag_list_file)
    sliced = slice_flags(flags, args.slice_index, args.slice_count)
    overrides = json.load(open(
        args.benchmark_args_json)) if args.benchmark_args_json else {}
    trials = plan(sliced, args.baseline_runs, args.skip_candidates)

    os.makedirs(args.artifact_dir, exist_ok=True)
    with open(os.path.join(args.artifact_dir, "manifest.json"), "w") as f:
        json.dump(
            {
                "model": args.model,
                "scheduler": "ofat",
                "slice_index": args.slice_index,
                "slice_count": args.slice_count,
                "total_flags": len(flags),
                "sliced_flags": sliced,
                "baseline_runs": args.baseline_runs,
                "skip_candidates": args.skip_candidates,
                "target_metric": args.target_metric,
                "trial_count": len(trials),
                "started_utc": datetime.utcnow().isoformat(),
            },
            f,
            indent=2)

    print(
        f"[autotune] shard {args.slice_index}/{args.slice_count}: "
        f"{len(sliced)} candidates + {args.baseline_runs} baselines = "
        f"{len(trials)} trials",
        flush=True,
    )

    failures = 0
    with open(os.path.join(args.artifact_dir, "summary.jsonl"),
              "w") as summary_fp:
        for trial in trials:
            rec = run_trial(
                trial,
                args.model,
                overrides,
                args.artifact_dir,
                args.target_metric,
                args.dry_run,
                summary_fp,
            )
            failures += 0 if rec["success"] else 1

    print(
        f"[autotune] DONE shard {args.slice_index}/{args.slice_count}: "
        f"{len(trials)} trials, {failures} failure(s)",
        flush=True,
    )
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
