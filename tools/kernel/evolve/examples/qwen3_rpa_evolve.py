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
"""Real battle: evolve RPA v3 block sizes for Qwen3-0.6B's exact shape.

Qwen3-0.6B at ``max_model_len=1024`` lands on the
``q_head-16_kv_head-8_head-128`` key of the v3 kernel's tuning table. That
key has **no tuned entry** under the ``q_bfloat16_kv_bfloat16`` block, so
the kernel falls back to ``(bkv_p=4096/page_size, bq=32) = (32, 32)`` on
TPU v7.

This script:

1. Uses our ``apply_diff`` infrastructure to insert a candidate tuned entry
   into ``tuned_block_sizes.py`` for the exact Qwen3 shape.
2. Spawns ``qwen3_bench.py`` as a subprocess to measure throughput.
3. Restores the file after every trial.
4. Compares the candidate to a recorded baseline (also measured via
   subprocess so the comparison is apples-to-apples).

Each trial is ~3-4 minutes of wall time (model load + compile + measure).
Default sweep is 5 candidates over a focused (bkv_p, bq) grid.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import re
import subprocess
import sys
import time
from pathlib import Path

from tools.kernel.evolve.mutator.diff_applier import (apply_diff,
                                                      validate_python)

_TUNED_PATH = Path(
    "/home/qizzzh_google_com/tpu-inference/tpu_inference/kernels/"
    "ragged_paged_attention/v3/tuned_block_sizes.py")

# The exact head-config block we target inside ``q_bfloat16_kv_bfloat16``.
# We INSERT this entry — there's currently nothing for Qwen3's shape there.
_QWEN3_HEAD_KEY = "'q_head-16_kv_head-8_head-128'"

# Use the `q_head-16_kv_head-2_head-128` block (line 390) as the *anchor*.
# Our diff inserts a new sibling block right before this anchor. That way
# the diff has a unique, stable context and applies cleanly regardless of
# other file changes.

_DEFAULT_CANDIDATES = [
    (4, 32),  # smaller bkv_p than default
    (8, 32),  # what nearby head configs use
    (16, 32),  # medium
    (32, 32),  # current fallback default (sanity)
    (8, 64),  # bigger bq
]


@dataclasses.dataclass
class Trial:
    label: str
    bkv_p: int
    bq: int
    throughput_tok_s: float | None
    wall_time_s: float | None
    error: str | None = None


def _make_insert_diff(bkv_p: int, bq: int) -> str:
    """Build a unified diff that inserts a tuned entry for Qwen3's shape.

    Anchors on the ``'q_head-16_kv_head-2_head-128':`` line — that line
    exists exactly once under the ``q_bfloat16_kv_bfloat16`` block we want
    to mutate, and the resulting diff is unambiguous.
    """
    # Find the anchor line in the current file.
    text = _TUNED_PATH.read_text()
    lines = text.splitlines(keepends=True)

    anchor_idx = None
    in_bf16_block = False
    for i, line in enumerate(lines):
        if "'q_bfloat16_kv_bfloat16':" in line:
            in_bf16_block = True
            continue
        if in_bf16_block and "'q_head-16_kv_head-2_head-128':" in line:
            anchor_idx = i
            break
    if anchor_idx is None:
        raise RuntimeError(
            "Could not find anchor 'q_head-16_kv_head-2_head-128' within "
            "the q_bfloat16_kv_bfloat16 block. The tuned_block_sizes.py "
            "structure changed; re-derive the anchor.")

    anchor = lines[anchor_idx].rstrip("\n")
    indent_match = re.match(r"^(\s*)", anchor)
    indent = indent_match.group(1) if indent_match else "                "

    # Build the new block (a single max_model_len-1024 entry; reusing the
    # surrounding text-format conventions of the table).
    new_block = (
        f"{indent}{_QWEN3_HEAD_KEY}: {{\n"
        f"{indent}    'max_model_len-1024-sw-None': ({bkv_p}, {bq}),\n"
        f"{indent}}},\n")

    new_lines = new_block.splitlines(keepends=True)
    # Build a unified diff that adds these lines just before anchor_idx.
    line_no = anchor_idx + 1  # 1-based
    hunk_old_count = 1  # the anchor line stays in place after the insertion
    hunk_new_count = len(new_lines) + 1
    hunk_lines = [f"+{ln.rstrip(chr(10))}" for ln in new_lines]
    hunk_lines.append(f" {anchor}")
    diff = (
        f"--- a/{_rel(_TUNED_PATH)}\n"
        f"+++ b/{_rel(_TUNED_PATH)}\n"
        f"@@ -{line_no},{hunk_old_count} +{line_no},{hunk_new_count} @@\n" +
        "\n".join(hunk_lines) + "\n")
    return diff


def _rel(p: Path) -> str:
    return str(p).replace("/home/qizzzh_google_com/tpu-inference/", "")


def _apply_candidate(bkv_p: int, bq: int) -> tuple[str, str]:
    """Apply the candidate diff to tuned_block_sizes.py.

    Returns ``(original_source, diff)`` so the caller can restore.
    """
    original = _TUNED_PATH.read_text()
    diff = _make_insert_diff(bkv_p, bq)
    result = apply_diff(original, diff)
    if not result.success or result.new_source is None:
        raise RuntimeError(
            f"diff apply failed for ({bkv_p}, {bq}): {result.error}")
    ok, parse_err = validate_python(result.new_source)
    if not ok:
        raise RuntimeError(
            f"mutated tuned_block_sizes.py fails to parse: {parse_err}")
    _TUNED_PATH.write_text(result.new_source)
    return original, diff


def _restore(original: str) -> None:
    _TUNED_PATH.write_text(original)


def _run_qwen3_bench(*, label: str, output: Path, max_model_len: int,
                     max_tokens: int) -> dict:
    """Spawn qwen3_bench.py as a fresh subprocess so the source change is read."""
    cmd = [
        sys.executable,
        "-m",
        "tools.kernel.evolve.examples.qwen3_bench",
        "--label",
        label,
        "--output",
        str(output),
        "--max-model-len",
        str(max_model_len),
        "--max-tokens",
        str(max_tokens),
        "--num-warmup-rounds",
        "1",
        "--num-measure-rounds",
        "2",
    ]
    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - t0
    if proc.returncode != 0:
        return {
            "ok": False,
            "stderr_tail": proc.stderr[-2000:],
            "wall_time": elapsed,
        }
    try:
        data = json.loads(output.read_text())
    except (json.JSONDecodeError, FileNotFoundError) as err:
        return {
            "ok": False,
            "stderr_tail":
            f"could not parse output: {err}\n{proc.stderr[-1000:]}",
            "wall_time": elapsed,
        }
    data["ok"] = True
    data["subprocess_wall_time"] = elapsed
    return data


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--max-model-len", type=int, default=1024)
    p.add_argument("--max-tokens", type=int, default=64)
    p.add_argument("--out-dir", type=Path, default=Path("/tmp/qwen3_evolve"))
    p.add_argument("--candidates",
                   type=str,
                   default="",
                   help="Comma-separated 'bkv_p:bq' pairs. "
                   "Default: 4:32,8:32,16:32,32:32,8:64")
    p.add_argument("--baseline-json",
                   type=Path,
                   default=Path("/tmp/qwen3_baseline.json"),
                   help="Path to an existing baseline JSON (re-measured if "
                   "not present).")
    p.add_argument("--verbose", "-v", action="count", default=0)
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.WARNING - 10 * args.verbose,
                        format="%(asctime)s %(levelname)s %(message)s")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.candidates:
        candidates = [
            tuple(int(x) for x in c.split(":"))
            for c in args.candidates.split(",")
        ]
    else:
        candidates = _DEFAULT_CANDIDATES

    # ---- Baseline (re-measure if not present)
    if args.baseline_json.exists():
        baseline = json.loads(args.baseline_json.read_text())
        print(
            f"Using cached baseline from {args.baseline_json}: "
            f"{baseline['throughput_tokens_per_s']:.2f} tok/s",
            file=sys.stderr)
    else:
        print("Measuring baseline...", file=sys.stderr)
        baseline = _run_qwen3_bench(
            label="baseline",
            output=args.baseline_json,
            max_model_len=args.max_model_len,
            max_tokens=args.max_tokens,
        )
        if not baseline.get("ok", False):
            print(f"Baseline run failed: {baseline.get('stderr_tail')}",
                  file=sys.stderr)
            return 2

    # ---- Sweep candidates
    trials: list[Trial] = [
        Trial(
            label="baseline",
            bkv_p=32,
            bq=32,  # fallback values
            throughput_tok_s=baseline["throughput_tokens_per_s"],
            wall_time_s=baseline["mean_wall_time_s"]),
    ]
    for bkv_p, bq in candidates:
        label = f"bkv_p_{bkv_p}_bq_{bq}"
        print(f"\n[trial] {label}", file=sys.stderr)
        original = None
        try:
            original, diff = _apply_candidate(bkv_p, bq)
            (args.out_dir / f"{label}.diff").write_text(diff)
            out_json = args.out_dir / f"{label}.json"
            result = _run_qwen3_bench(
                label=label,
                output=out_json,
                max_model_len=args.max_model_len,
                max_tokens=args.max_tokens,
            )
            if result.get("ok"):
                trials.append(
                    Trial(
                        label=label,
                        bkv_p=bkv_p,
                        bq=bq,
                        throughput_tok_s=result["throughput_tokens_per_s"],
                        wall_time_s=result["mean_wall_time_s"],
                    ))
            else:
                trials.append(
                    Trial(
                        label=label,
                        bkv_p=bkv_p,
                        bq=bq,
                        throughput_tok_s=None,
                        wall_time_s=None,
                        error=result.get("stderr_tail", "unknown"),
                    ))
        except Exception as err:
            trials.append(
                Trial(label=label,
                      bkv_p=bkv_p,
                      bq=bq,
                      throughput_tok_s=None,
                      wall_time_s=None,
                      error=str(err)))
        finally:
            if original is not None:
                _restore(original)

    # ---- Report
    print()
    print("=" * 78)
    print("Qwen3-0.6B RPA v3 block-size sweep on TPU v7")
    print(
        f"  shape: q_head-16 kv_head-8 head_dim-128, max_model_len={args.max_model_len}"
    )
    print(f"  max_tokens={args.max_tokens}, num_prompts=10, "
          f"greedy decoding, mean of 2 measure rounds")
    print()
    print(f"  {'label':24s}  {'bkv_p':>5}  {'bq':>4}  "
          f"{'tok/s':>9}  {'vs baseline':>11}")
    base_tps = trials[0].throughput_tok_s or 1.0
    for t in trials:
        if t.throughput_tok_s is None:
            print(f"  {t.label:24s}  {t.bkv_p:>5}  {t.bq:>4}  "
                  f"{'FAILED':>9}  ({(t.error or '')[:30]})")
            continue
        speed = t.throughput_tok_s / base_tps
        print(f"  {t.label:24s}  {t.bkv_p:>5}  {t.bq:>4}  "
              f"{t.throughput_tok_s:>9.2f}  {speed:>10.3f}x")
    best = max((t for t in trials if t.throughput_tok_s is not None),
               key=lambda x: x.throughput_tok_s)
    print()
    print(f"Winner: {best.label}  (bkv_p={best.bkv_p}, bq={best.bq})  "
          f"@ {best.throughput_tok_s:.2f} tok/s "
          f"({best.throughput_tok_s/base_tps:.3f}x baseline)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
