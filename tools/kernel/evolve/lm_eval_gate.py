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
"""Outermost correctness gate — lm_eval before/after a kernel diff.

A passing numerics verifier + paired-t + cross-shape is *necessary* for
shipping, but not sufficient. vLLM/FP8/H100 famously had unit tests pass
while needle-in-haystack dropped from 91% → 13%. The lm_eval gate is the
last defense: actually run the model on real benchmarks (gsm8k,
mmlu_pro) and reject the patch if the delta exceeds a tolerance.

This wraps the existing ``tests/e2e/check_lm_eval.sh`` flow. The wrapper:

1. Snapshot the target kernel file.
2. Apply the diff in-place.
3. Run lm_eval via vLLM (subprocess).
4. Revert the kernel file.
5. Run lm_eval again with the baseline (or read a cached baseline).
6. Compare and emit a JSON report.

The wrapper is paranoid about restoring the working tree — uses
``try/finally`` and double-checks with a sha256 after.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import logging
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from tools.kernel.evolve.mutator.diff_applier import apply_diff

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class LmEvalResult:
    task: str
    score_baseline: float
    score_patched: float
    delta: float  # patched - baseline
    sample_size: int
    raw_baseline: dict[str, Any]
    raw_patched: dict[str, Any]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _build_lm_eval_cmd(*, model: str, task: str, limit: int,
                       tensor_parallel: int, max_model_len: int,
                       block_size: int, output_path: Path) -> list[str]:
    """Compose the lm_eval CLI command."""
    model_args = (f"pretrained={model},"
                  f"tensor_parallel_size={tensor_parallel},"
                  f"max_model_len={max_model_len},"
                  f"block_size={block_size},"
                  f"trust_remote_code=true,"
                  f"dtype=bfloat16")
    return [
        "lm_eval",
        "--model",
        "vllm",
        "--model_args",
        model_args,
        "--tasks",
        task,
        "--num_fewshot",
        "5" if task == "gsm8k" else "5",
        "--limit",
        str(limit),
        "--batch_size",
        "auto",
        "--output_path",
        str(output_path),
    ]


def _run_lm_eval(*,
                 model: str,
                 task: str,
                 limit: int,
                 tensor_parallel: int,
                 max_model_len: int,
                 block_size: int,
                 output_dir: Path,
                 timeout_sec: int = 1800) -> dict:
    """Run a single lm_eval invocation; return the parsed result block."""
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{task}.json"
    cmd = _build_lm_eval_cmd(model=model,
                             task=task,
                             limit=limit,
                             tensor_parallel=tensor_parallel,
                             max_model_len=max_model_len,
                             block_size=block_size,
                             output_path=output_path)
    logger.info("Running: %s", " ".join(cmd))
    t0 = time.time()
    proc = subprocess.run(cmd,
                          capture_output=True,
                          text=True,
                          timeout=timeout_sec)
    wall = time.time() - t0
    if proc.returncode != 0:
        raise RuntimeError(
            f"lm_eval failed (exit {proc.returncode}, wall={wall:.1f}s)\n"
            f"stderr tail:\n{proc.stderr[-2000:]}")
    # lm_eval writes one file per task under output_path or a timestamped
    # subdir. Be permissive about layout.
    candidates = list(output_dir.rglob("results_*.json"))
    if not candidates:
        # Some versions write a single results.json
        candidates = list(output_dir.rglob("results.json"))
    if not candidates:
        # Last resort — parse stdout (lm_eval prints final scores)
        return _parse_scores_from_stdout(proc.stdout, task=task)
    candidates.sort(key=lambda p: p.stat().st_mtime)
    return json.loads(candidates[-1].read_text())


def _parse_scores_from_stdout(text: str, *, task: str) -> dict:
    """Fallback when lm_eval doesn't write a JSON report."""
    pattern = re.compile(rf"\|\s*{re.escape(task)}\s*\|.*?\|\s*([0-9.]+)")
    m = pattern.search(text)
    if m:
        return {
            "results": {
                task: {
                    "exact_match,strict-match": float(m.group(1))
                }
            }
        }
    return {"results": {task: {}}}


def _extract_primary_score(data: dict, task: str) -> float:
    """Pull the headline score out of an lm_eval results dict."""
    results = data.get("results", {})
    task_res = results.get(task, {})
    # Common score keys in lm_eval, in preference order.
    for key in [
            "exact_match,strict-match", "exact_match,flexible-extract",
            "acc,none", "acc_norm,none", "pass@1,none"
    ]:
        if key in task_res:
            return float(task_res[key])
    # Last resort: first numeric value.
    for v in task_res.values():
        try:
            return float(v)
        except (TypeError, ValueError):
            continue
    return float("nan")


def _restore_kernel(kernel_path: Path, backup_bytes: bytes,
                    original_sha: str) -> None:
    """Restore the kernel file from a backup; verify SHA matches."""
    kernel_path.write_bytes(backup_bytes)
    restored_sha = _sha256(kernel_path)
    if restored_sha != original_sha:
        raise RuntimeError(
            f"lm_eval_gate: SHA mismatch after restore — expected "
            f"{original_sha}, got {restored_sha}")


def run_lm_eval_gate(*,
                     model: str,
                     kernel_path: Path,
                     diff_path: Path,
                     tasks: list[str],
                     limit: int,
                     tensor_parallel: int,
                     max_model_len: int,
                     block_size: int,
                     output_dir: Path,
                     tolerance: float = 0.005) -> list[LmEvalResult]:
    """Run baseline + patched lm_eval; return per-task results.

    Idempotent w.r.t. the working tree — restores the kernel file even on
    error. ``tolerance`` is the maximum allowed score regression per task
    (default 0.5pt = 0.005).
    """
    backup_bytes = kernel_path.read_bytes()
    original_sha = _sha256(kernel_path)
    diff_text = diff_path.read_text()

    baseline_results: dict[str, dict] = {}
    patched_results: dict[str, dict] = {}

    try:
        # 1. Baseline pass — kernel file untouched
        logger.info("--- BASELINE lm_eval pass ---")
        for task in tasks:
            baseline_results[task] = _run_lm_eval(
                model=model,
                task=task,
                limit=limit,
                tensor_parallel=tensor_parallel,
                max_model_len=max_model_len,
                block_size=block_size,
                output_dir=output_dir / "baseline" / task)

        # 2. Apply the diff
        logger.info("--- Applying diff to %s ---", kernel_path)
        new_src = apply_diff(kernel_path.read_text(), diff_text).new_source
        kernel_path.write_text(new_src)

        # 3. Patched pass
        logger.info("--- PATCHED lm_eval pass ---")
        for task in tasks:
            patched_results[task] = _run_lm_eval(
                model=model,
                task=task,
                limit=limit,
                tensor_parallel=tensor_parallel,
                max_model_len=max_model_len,
                block_size=block_size,
                output_dir=output_dir / "patched" / task)
    finally:
        # 4. Restore kernel — VERY important; guarantees clean tree.
        _restore_kernel(kernel_path, backup_bytes, original_sha)

    out: list[LmEvalResult] = []
    for task in tasks:
        s_base = _extract_primary_score(baseline_results.get(task, {}), task)
        s_patch = _extract_primary_score(patched_results.get(task, {}), task)
        out.append(
            LmEvalResult(
                task=task,
                score_baseline=s_base,
                score_patched=s_patch,
                delta=s_patch - s_base,
                sample_size=limit,
                raw_baseline=baseline_results.get(task, {}),
                raw_patched=patched_results.get(task, {}),
            ))
    return out


def render_summary(results: list[LmEvalResult],
                   tolerance: float = 0.005) -> str:
    lines = [
        "| task | baseline | patched | Δ | tolerance | verdict |",
        "|---|---|---|---|---|---|",
    ]
    for r in results:
        ok = abs(r.delta) <= tolerance
        verdict = "PASS" if ok else "FAIL"
        lines.append(
            f"| {r.task} | {r.score_baseline:.4f} | {r.score_patched:.4f} | "
            f"{r.delta:+.4f} | ±{tolerance:.4f} | {verdict} |")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--diff", type=Path, required=True)
    p.add_argument("--kernel-path", type=Path, required=True)
    p.add_argument("--model", default="Qwen/Qwen3-0.6B")
    p.add_argument("--tasks",
                   default="gsm8k",
                   help="Comma-separated lm_eval task names.")
    p.add_argument("--limit", type=int, default=200)
    p.add_argument("--tensor-parallel", type=int, default=2)
    p.add_argument("--max-model-len", type=int, default=1024)
    p.add_argument("--block-size", type=int, default=64)
    p.add_argument("--output-dir",
                   type=Path,
                   default=Path("/tmp/lm_eval_gate"))
    p.add_argument("--out-json",
                   type=Path,
                   default=Path("/tmp/lm_eval_results.json"))
    p.add_argument("--tolerance", type=float, default=0.005)
    p.add_argument("--verbose", "-v", action="count", default=0)
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.WARNING - 10 * args.verbose,
                        format="%(asctime)s %(levelname)s %(message)s")
    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    t0 = time.time()
    results = run_lm_eval_gate(model=args.model,
                               kernel_path=args.kernel_path,
                               diff_path=args.diff,
                               tasks=tasks,
                               limit=args.limit,
                               tensor_parallel=args.tensor_parallel,
                               max_model_len=args.max_model_len,
                               block_size=args.block_size,
                               output_dir=args.output_dir,
                               tolerance=args.tolerance)
    wall = time.time() - t0
    payload = [{
        "task": r.task,
        "baseline_score": r.score_baseline,
        "patched_score": r.score_patched,
        "delta": r.delta,
        "limit": r.sample_size,
    } for r in results]
    args.out_json.write_text(json.dumps(payload, indent=2))
    print(f"Done in {wall:.1f}s")
    print()
    print(render_summary(results, tolerance=args.tolerance))
    print()
    print(f"JSON: {args.out_json}")
    any_fail = any(abs(r.delta) > args.tolerance for r in results)
    return 1 if any_fail else 0


if __name__ == "__main__":
    sys.exit(main())
