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
"""End-to-end throughput benchmark — verifies a kernel diff moves the dial.

A +N% kernel speedup is only worth shipping if it translates to user-visible
throughput in the actual vLLM serving stack. There are subtle reasons a
kernel win might not surface (e.g. the kernel isn't on the critical path
for a given workload).

This wraps vLLM's offline_inference benchmark. The flow mirrors
``lm_eval_gate.py``:

1. Snapshot the kernel.
2. Bench at baseline (N trials).
3. Apply the diff.
4. Bench patched (N trials).
5. Restore the kernel.
6. Report throughput delta with paired-t.

The bench measures **prefill + decode** end-to-end. The "lift" reported
here is the user-relevant number a PR description should cite.
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

from tools.kernel.evolve.mutator.diff_applier import apply_diff
from tools.kernel.evolve.stats.significance import (PairedComparison,
                                                    summarize_comparison)

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class E2EResult:
    """Throughput numbers across paired trials."""
    label: str
    throughput_tok_per_s: list[float]
    mean_tok_per_s: float
    p50_tok_per_s: float
    n_trials: int


@dataclasses.dataclass
class E2EComparison:
    baseline: E2EResult
    patched: E2EResult
    speedup_mean: float
    p_value: float
    summary_text: str


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _build_bench_cmd(*,
                     model: str,
                     tensor_parallel: int,
                     max_model_len: int,
                     max_tokens: int,
                     num_prompts: int,
                     dataset_name: str = "random",
                     input_len: int = 1024) -> list[str]:
    """Compose the offline benchmark CLI command.

    We use vLLM's benchmark_throughput entrypoint via Python -m so the
    measured time excludes vLLM init overhead and reflects steady-state
    throughput.
    """
    return [
        "vllm",
        "bench",
        "throughput",
        "--model",
        model,
        "--tensor-parallel-size",
        str(tensor_parallel),
        "--max-model-len",
        str(max_model_len),
        "--max-num-batched-tokens",
        str(max_model_len),
        "--input-len",
        str(input_len),
        "--output-len",
        str(max_tokens),
        "--num-prompts",
        str(num_prompts),
        "--dataset-name",
        dataset_name,
        "--dtype",
        "bfloat16",
        "--trust-remote-code",
    ]


_THROUGHPUT_RE = re.compile(
    r"Throughput:\s*([\d.]+)\s*requests/s,\s*([\d.]+)\s*total tokens/s,"
    r"\s*([\d.]+)\s*output tokens/s", re.IGNORECASE)


def _parse_throughput(stdout: str) -> float | None:
    """Extract output tokens/sec from vLLM benchmark stdout."""
    m = _THROUGHPUT_RE.search(stdout)
    if m:
        return float(m.group(3))
    # Fallback: look for any "tokens/s" line
    alt = re.search(r"([\d.]+)\s*tokens/s", stdout)
    if alt:
        return float(alt.group(1))
    return None


def _bench_trial(cmd: list[str], *, timeout_sec: int = 600) -> float:
    """One bench trial → output tokens/sec."""
    logger.info("$ %s", " ".join(cmd))
    proc = subprocess.run(cmd,
                          capture_output=True,
                          text=True,
                          timeout=timeout_sec)
    if proc.returncode != 0:
        raise RuntimeError(f"vllm bench failed (exit {proc.returncode}):\n"
                           f"stderr tail:\n{proc.stderr[-2000:]}")
    out = _parse_throughput(proc.stdout)
    if out is None:
        raise RuntimeError(
            "vllm bench: couldn't parse throughput from stdout. "
            f"Last 500 chars: {proc.stdout[-500:]}")
    return out


def _bench_n_trials(*,
                    model: str,
                    tensor_parallel: int,
                    max_model_len: int,
                    max_tokens: int,
                    num_prompts: int,
                    n_trials: int,
                    timeout_sec: int = 600) -> list[float]:
    cmd = _build_bench_cmd(model=model,
                           tensor_parallel=tensor_parallel,
                           max_model_len=max_model_len,
                           max_tokens=max_tokens,
                           num_prompts=num_prompts)
    return [
        _bench_trial(cmd, timeout_sec=timeout_sec) for _ in range(n_trials)
    ]


def _restore(kernel_path: Path, backup: bytes, original_sha: str) -> None:
    kernel_path.write_bytes(backup)
    new_sha = _sha256(kernel_path)
    if new_sha != original_sha:
        raise RuntimeError(
            f"e2e_benchmark: SHA mismatch after restore — expected "
            f"{original_sha}, got {new_sha}")


def run_e2e_benchmark(*,
                      model: str,
                      kernel_path: Path,
                      diff_path: Path,
                      tensor_parallel: int = 1,
                      max_model_len: int = 1024,
                      max_tokens: int = 128,
                      num_prompts: int = 32,
                      n_trials: int = 3,
                      timeout_sec: int = 600) -> E2EComparison:
    """Bench baseline vs patched; return comparison + paired-t."""
    backup = kernel_path.read_bytes()
    original_sha = _sha256(kernel_path)
    diff_text = diff_path.read_text()
    try:
        logger.info("--- BASELINE bench ---")
        baseline_tps = _bench_n_trials(model=model,
                                       tensor_parallel=tensor_parallel,
                                       max_model_len=max_model_len,
                                       max_tokens=max_tokens,
                                       num_prompts=num_prompts,
                                       n_trials=n_trials,
                                       timeout_sec=timeout_sec)
        logger.info("--- Applying diff ---")
        new_src = apply_diff(kernel_path.read_text(), diff_text).new_source
        kernel_path.write_text(new_src)
        logger.info("--- PATCHED bench ---")
        patched_tps = _bench_n_trials(model=model,
                                      tensor_parallel=tensor_parallel,
                                      max_model_len=max_model_len,
                                      max_tokens=max_tokens,
                                      num_prompts=num_prompts,
                                      n_trials=n_trials,
                                      timeout_sec=timeout_sec)
    finally:
        _restore(kernel_path, backup, original_sha)

    import numpy as np
    base = E2EResult(label="baseline",
                     throughput_tok_per_s=baseline_tps,
                     mean_tok_per_s=float(np.mean(baseline_tps)),
                     p50_tok_per_s=float(np.median(baseline_tps)),
                     n_trials=len(baseline_tps))
    patch = E2EResult(label="patched",
                      throughput_tok_per_s=patched_tps,
                      mean_tok_per_s=float(np.mean(patched_tps)),
                      p50_tok_per_s=float(np.median(patched_tps)),
                      n_trials=len(patched_tps))
    speedup = patch.mean_tok_per_s / base.mean_tok_per_s
    # Higher = better here, but the stats helper assumes lower=better.
    # We pass reciprocals (per-trial inverse-throughput = seconds/token)
    # so the comparison framing matches.
    comp = PairedComparison(
        label_a="baseline_s_per_tok",
        label_b="patched_s_per_tok",
        a_values=[1.0 / t for t in baseline_tps],
        b_values=[1.0 / t for t in patched_tps],
    )
    summary = summarize_comparison(comp)
    p_val_match = re.search(r"p[≈=]([\d.]+)", summary)
    p_value = float(p_val_match.group(1)) if p_val_match else 1.0
    return E2EComparison(
        baseline=base,
        patched=patch,
        speedup_mean=speedup,
        p_value=p_value,
        summary_text=summary,
    )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--diff", type=Path, required=True)
    p.add_argument("--kernel-path", type=Path, required=True)
    p.add_argument("--model", default="Qwen/Qwen3-0.6B")
    p.add_argument("--tensor-parallel", type=int, default=1)
    p.add_argument("--max-model-len", type=int, default=1024)
    p.add_argument("--max-tokens", type=int, default=128)
    p.add_argument("--num-prompts", type=int, default=32)
    p.add_argument("--n-trials", type=int, default=3)
    p.add_argument("--timeout-sec", type=int, default=900)
    p.add_argument("--out-json",
                   type=Path,
                   default=Path("/tmp/e2e_benchmark.json"))
    p.add_argument("--verbose", "-v", action="count", default=0)
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.WARNING - 10 * args.verbose,
                        format="%(asctime)s %(levelname)s %(message)s")
    t0 = time.time()
    cmp = run_e2e_benchmark(model=args.model,
                            kernel_path=args.kernel_path,
                            diff_path=args.diff,
                            tensor_parallel=args.tensor_parallel,
                            max_model_len=args.max_model_len,
                            max_tokens=args.max_tokens,
                            num_prompts=args.num_prompts,
                            n_trials=args.n_trials,
                            timeout_sec=args.timeout_sec)
    wall = time.time() - t0
    payload = {
        "model": args.model,
        "baseline_mean_tok_per_s": cmp.baseline.mean_tok_per_s,
        "patched_mean_tok_per_s": cmp.patched.mean_tok_per_s,
        "speedup_mean": cmp.speedup_mean,
        "p_value": cmp.p_value,
        "n_trials": cmp.baseline.n_trials,
        "baseline_trials": cmp.baseline.throughput_tok_per_s,
        "patched_trials": cmp.patched.throughput_tok_per_s,
    }
    args.out_json.write_text(json.dumps(payload, indent=2))
    print()
    print(f"E2E benchmark in {wall:.1f}s")
    print(f"  baseline:  {cmp.baseline.mean_tok_per_s:.2f} tok/s")
    print(f"  patched:   {cmp.patched.mean_tok_per_s:.2f} tok/s")
    print(f"  speedup:   {cmp.speedup_mean:.4f}× (p≈{cmp.p_value:.4f})")
    print()
    print(cmp.summary_text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
