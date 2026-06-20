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
"""Optional outer eval-harness gate for the final winning kernel config.

The numerics gate catches component-level drift; eval tasks catch the cases
where a candidate passes per-token allclose but degrades downstream model
quality. vLLM's ``.buildkite/lm-eval-harness/`` adopted the same pattern after
their FP8 KV-cache incident dropped needle-in-haystack from 91% to 13% with
passing unit tests.

This module is a thin wrapper around the ``lm_eval`` CLI (kept out of the
import graph to avoid pulling the heavy dep when ``--final-eval`` isn't set).
"""

import dataclasses
import json
import logging
import shutil
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class LmEvalResult:
    task: str
    metric: str
    score: float
    baseline: float
    delta_pp: float  # percentage points; negative = regression
    passed: bool


def lm_eval_available() -> bool:
    """Return True if a usable ``lm_eval`` CLI is on ``$PATH``."""
    return (shutil.which("lm_eval") is not None
            or shutil.which("lm-eval") is not None)


def _lm_eval_binary() -> str:
    if shutil.which("lm_eval") is not None:
        return "lm_eval"
    if shutil.which("lm-eval") is not None:
        return "lm-eval"
    raise RuntimeError(
        "lm_eval CLI not on $PATH. Install with `pip install lm-eval`.")


def _pick_primary_metric(task_data: dict) -> str | None:
    """Pick the first non-stderr, non-alias key as the primary metric."""
    keys = [
        k for k in task_data
        if not k.endswith("_stderr") and not k.startswith("alias")
    ]
    return sorted(keys)[0] if keys else None


def run_lm_eval(
    *,
    model_args: str,
    tasks: list[str],
    baselines: dict[str, float],
    delta_pp_tolerance: float = 0.5,
    limit: int = 50,
    output_dir: str | Path = "/tmp/lm_eval_out",
    extra_args: list[str] | None = None,
) -> list[LmEvalResult]:
    """Run lm-eval and compare per-task scores against ``baselines``.

    Args:
        model_args: ``--model_args`` payload for lm-eval (e.g.
            ``"pretrained=Qwen/Qwen3-0.6B"``).
        tasks: task names accepted by ``lm_eval --tasks`` (e.g. ``gsm8k``).
        baselines: expected per-task score (same key as ``tasks``).
        delta_pp_tolerance: max allowed regression in percentage points.
        limit: per-task sample cap for quick turnaround.
        output_dir: where lm-eval writes its results JSON.
        extra_args: extra CLI args appended verbatim.
    """
    bin_name = _lm_eval_binary()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        bin_name,
        "--model",
        "vllm",
        "--model_args",
        model_args,
        "--tasks",
        ",".join(tasks),
        "--limit",
        str(limit),
        "--output_path",
        str(output_dir),
    ]
    if extra_args:
        cmd.extend(extra_args)
    logger.info("Running lm-eval: %s", " ".join(cmd))
    proc = subprocess.run(cmd, check=True, capture_output=True, text=True)

    candidates = sorted(
        output_dir.glob("**/results*.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise RuntimeError(
            f"lm-eval finished but produced no results JSON under {output_dir}."
            f"\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}")
    with candidates[0].open("r") as f:
        payload = json.load(f)
    results_data = payload.get("results", {})

    reports: list[LmEvalResult] = []
    for task in tasks:
        task_data = results_data.get(task, {})
        metric = _pick_primary_metric(task_data)
        if metric is None:
            logger.warning("lm-eval has no metric for task %r", task)
            continue
        score = float(task_data[metric])
        baseline = float(baselines.get(task, score))
        delta_pp = (score - baseline) * 100.0
        reports.append(
            LmEvalResult(
                task=task,
                metric=metric,
                score=score,
                baseline=baseline,
                delta_pp=delta_pp,
                passed=delta_pp >= -delta_pp_tolerance,
            ))
    return reports
