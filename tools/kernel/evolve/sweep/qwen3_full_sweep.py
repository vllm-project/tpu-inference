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
"""End-to-end automated sweep:
1. Scan ``tuned_block_sizes.py`` for missing entries (popular Qwen3 family).
2. For each missing entry, run a Qwen3 benchmark with several candidate
   ``(bkv_p, bq)`` values, picking the fastest verified one.
3. Build a single mergeable auto-PR diff.
4. Emit telemetry per trial.

Result is an auto-PR diff at ``--out-diff`` plus a JSONL telemetry log at
``--out-telemetry``. Run nightly via CI or manually with ``--dry-run`` for
inspection.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

from tools.kernel.evolve.mutator.diff_applier import apply_diff
from tools.kernel.evolve.sweep.auto_pr import TunedWin, build_auto_pr
from tools.kernel.evolve.sweep.missing_entries import (POPULAR_MODELS,
                                                       find_missing_entries)
from tools.kernel.evolve.telemetry.writer import (TelemetryEvent,
                                                  TelemetryWriter)

# Candidate (bkv_p, bq) values for the per-shape sweep. Picked from the
# experimentally validated Qwen3 sweep (Phase 2): bkv_p=8 and bq=32 are the
# strongest reference points; adjacent values explore the local optimum.
_CANDIDATE_BLOCK_SIZES = [
    (4, 32),
    (8, 32),
    (8, 64),
    (16, 32),
]


def _ml_from_extra(extra_key: str) -> int | None:
    m = re.match(r"max_model_len-(\d+)-sw-", extra_key)
    return int(m.group(1)) if m else None


def _make_insert_diff_for(
    tuned_path: Path,
    head_key: str,
    extra_key: str,
    bkv_p: int,
    bq: int,
) -> str:
    """Programmatic diff: insert a single tuned entry into the q_bf16_kv_bf16
    block. Mirrors the proven approach from ``qwen3_rpa_evolve``.
    """
    text = tuned_path.read_text()
    lines = text.splitlines(keepends=True)
    in_bf16 = False
    anchor_idx: int | None = None
    head_pat = re.compile(r"'q_head-\d+_kv_head-\d+_head-\d+'")
    for i, line in enumerate(lines):
        if "'q_bfloat16_kv_bfloat16'" in line and "{" in line:
            in_bf16 = True
            continue
        if not in_bf16:
            continue
        if f"'{head_key}'" in line and "{" in line:
            anchor_idx = i
            break
        if head_pat.search(line) and "{" in line:
            # Anchor before this sibling block.
            if anchor_idx is None:
                anchor_idx = i
    if anchor_idx is None:
        # Bail: we'll let the caller skip this entry.
        raise ValueError(
            f"could not locate anchor for {head_key} in q_bfloat16_kv_bfloat16"
        )

    anchor_line = lines[anchor_idx]
    if f"'{head_key}'" in anchor_line:
        # Existing block: insert the extra_key inside it just before the close.
        depth = 0
        end_idx = None
        for j in range(anchor_idx, len(lines)):
            depth += lines[j].count("{") - lines[j].count("}")
            if depth == 0 and j > anchor_idx:
                end_idx = j
                break
        if end_idx is None:
            raise ValueError("could not find close of existing head block")
        # Use indent from the line right after the opening brace.
        body_indent = re.match(r"^(\s*)", lines[anchor_idx + 1]).group(
            1) if anchor_idx + 1 < end_idx else "                    "
        new_line = (f"{body_indent}'{extra_key}': ({bkv_p}, {bq}),\n")
        return _format_one_hunk_insert(tuned_path, end_idx, new_line)
    else:
        # New head block.
        indent = re.match(r"^(\s*)", anchor_line).group(1)
        body_indent = indent + "    "
        block = (f"{indent}'{head_key}': {{\n"
                 f"{body_indent}'{extra_key}': ({bkv_p}, {bq}),\n"
                 f"{indent}}},\n")
        return _format_multi_line_insert(tuned_path, anchor_idx, block)


def _format_one_hunk_insert(path: Path, before_line_index: int,
                            new_line: str) -> str:
    """Insert a single line before ``before_line_index``."""
    line_no_1based = before_line_index + 1
    anchor_text = path.read_text().splitlines(
        keepends=True)[before_line_index].rstrip("\n")
    return (f"--- a/{_rel(path)}\n"
            f"+++ b/{_rel(path)}\n"
            f"@@ -{line_no_1based},1 +{line_no_1based},2 @@\n"
            f"+{new_line.rstrip(chr(10))}\n"
            f" {anchor_text}\n")


def _format_multi_line_insert(path: Path, before_line_index: int,
                              block: str) -> str:
    line_no = before_line_index + 1
    anchor_text = path.read_text().splitlines(
        keepends=True)[before_line_index].rstrip("\n")
    new_lines = block.splitlines(keepends=True)
    hunk = []
    for nl in new_lines:
        hunk.append(f"+{nl.rstrip(chr(10))}")
    hunk.append(f" {anchor_text}")
    return (f"--- a/{_rel(path)}\n"
            f"+++ b/{_rel(path)}\n"
            f"@@ -{line_no},1 +{line_no},{len(new_lines) + 1} @@\n" +
            "\n".join(hunk) + "\n")


def _rel(p: Path) -> str:
    return str(p).replace("/home/qizzzh_google_com/tpu-inference/", "")


def _run_qwen3_bench(label: str, max_model_len: int, max_tokens: int,
                     output: Path) -> dict | None:
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
        "2",
        "--num-measure-rounds",
        "3",
    ]
    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
    wall = time.time() - t0
    if proc.returncode != 0:
        return {"ok": False, "wall": wall, "err": proc.stderr[-1500:]}
    try:
        data = json.loads(output.read_text())
        data["ok"] = True
        data["wall"] = wall
        return data
    except (FileNotFoundError, json.JSONDecodeError):
        return {"ok": False, "wall": wall, "err": "no result JSON"}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--tuned-path",
                   type=Path,
                   default=Path("/home/qizzzh_google_com/tpu-inference/"
                                "tpu_inference/kernels/ragged_paged_attention/"
                                "v3/tuned_block_sizes.py"))
    p.add_argument("--out-diff",
                   type=Path,
                   default=Path("/tmp/qwen3_auto_pr.diff"))
    p.add_argument("--out-telemetry",
                   type=Path,
                   default=Path("/tmp/qwen3_sweep_telemetry.jsonl"))
    p.add_argument("--out-summary",
                   type=Path,
                   default=Path("/tmp/qwen3_sweep_summary.json"))
    p.add_argument("--shapes",
                   type=int,
                   default=2,
                   help="Number of (model, context-length) shapes to sweep")
    p.add_argument("--candidates",
                   type=str,
                   default="",
                   help="Comma-separated 'bkv_p:bq' pairs to try per shape; "
                   "default tests 4 strong candidates.")
    p.add_argument("--context-lengths",
                   type=str,
                   default="1024",
                   help="Comma-separated max_model_len values to sweep.")
    p.add_argument("--models",
                   type=str,
                   default="Qwen3-0.6B",
                   help="Comma-separated model names from the popular set.")
    p.add_argument("--dry-run",
                   action="store_true",
                   help="Print the missing entries without running anything.")
    p.add_argument("--max-tokens", type=int, default=64)
    args = p.parse_args(argv)

    if args.candidates:
        candidates = [
            tuple(int(x) for x in c.split(":"))
            for c in args.candidates.split(",")
        ]
    else:
        candidates = _CANDIDATE_BLOCK_SIZES

    context_lengths = [int(x) for x in args.context_lengths.split(",")]
    model_names = [m.strip() for m in args.models.split(",")]
    selected = [m for m in POPULAR_MODELS if m.name in model_names]
    if not selected:
        print(f"No matching models in: {model_names}", file=sys.stderr)
        return 2

    missing = find_missing_entries(
        tuned_block_sizes_path=args.tuned_path,
        models=selected,
        context_lengths=context_lengths,
        devices=["TPU v7"],
        page_sizes=[128],
    )
    print(f"Found {len(missing)} missing entries.")
    missing = missing[:args.shapes]
    for me in missing:
        print(f"  {me.device}/{me.page_size}/{me.dtype_key}/{me.head_key}/"
              f"{me.extra_key} (models: {', '.join(me.referencing_models)})")

    if args.dry_run:
        return 0

    args.out_telemetry.parent.mkdir(parents=True, exist_ok=True)
    args.out_diff.parent.mkdir(parents=True, exist_ok=True)
    wins: list[TunedWin] = []
    with TelemetryWriter(args.out_telemetry) as tel:
        run_id = f"sweep_{int(time.time())}"
        for me in missing:
            ml = _ml_from_extra(me.extra_key)
            if ml is None:
                continue
            print(f"\n[sweep] {me.head_key} / max_model_len={ml}")

            # Baseline first.
            bl_path = args.out_diff.parent / f"baseline_{me.head_key}_{ml}.json"
            baseline = _run_qwen3_bench("baseline", ml, args.max_tokens,
                                        bl_path)
            tel.emit(
                TelemetryEvent(
                    timestamp=time.time(),
                    run_id=run_id,
                    kernel="rpa_v3",
                    shape_key=f"{me.head_key}/ml{ml}",
                    genome_id="baseline",
                    parent_ids=[],
                    generation=0,
                    island_id=0,
                    diff_summary="",
                    status="VERIFIED"
                    if baseline and baseline.get("ok") else "BASELINE_FAIL",
                    fitness_ns=None,
                    p50_ns=None,
                    p95_ns=None,
                    cosine=None,
                    max_abs_diff=None,
                    wall_time_s=baseline.get("wall", 0) if baseline else 0,
                    rule_name="baseline",
                    extra={
                        "throughput_tok_s":
                        baseline.get("throughput_tokens_per_s")
                        if baseline else None
                    },
                ))
            if not baseline or not baseline.get("ok"):
                print("  baseline failed; skipping shape")
                continue
            base_tps = baseline["throughput_tokens_per_s"]
            print(f"  baseline: {base_tps:.2f} tok/s")

            best_win: TunedWin | None = None
            for (bkv_p, bq) in candidates:
                label = f"bkv{bkv_p}_bq{bq}"
                orig = args.tuned_path.read_text()
                try:
                    diff = _make_insert_diff_for(args.tuned_path, me.head_key,
                                                 me.extra_key, bkv_p, bq)
                    result = apply_diff(orig, diff)
                    if not result.success or result.new_source is None:
                        raise RuntimeError(result.error or "apply failed")
                    args.tuned_path.write_text(result.new_source)
                    out_path = args.out_diff.parent / f"{label}_{me.head_key}_{ml}.json"
                    data = _run_qwen3_bench(label, ml, args.max_tokens,
                                            out_path)
                    if data and data.get("ok"):
                        tps = data["throughput_tokens_per_s"]
                        speedup = tps / base_tps if base_tps else None
                        status = "VERIFIED"
                        if speedup is None or speedup < 1.005:
                            status = "VERIFIED_NO_WIN"
                        tel.emit(
                            TelemetryEvent(
                                timestamp=time.time(),
                                run_id=run_id,
                                kernel="rpa_v3",
                                shape_key=f"{me.head_key}/ml{ml}",
                                genome_id=f"{me.head_key}_ml{ml}_{label}",
                                parent_ids=["baseline"],
                                generation=1,
                                island_id=0,
                                diff_summary=diff[:200],
                                status=status,
                                fitness_ns=(data["mean_wall_time_s"] * 1e9)
                                if status == "VERIFIED" else None,
                                p50_ns=None,
                                p95_ns=None,
                                cosine=None,
                                max_abs_diff=None,
                                wall_time_s=data.get("wall", 0),
                                rule_name=f"rpa_v3_block_{bkv_p}_{bq}",
                                extra={
                                    "throughput_tok_s": tps,
                                    "speedup": speedup
                                },
                            ))
                        print(f"  {label}: {tps:.2f} tok/s "
                              f"({speedup:.3f}x)"
                              if speedup else f"  {label}: {tps:.2f} tok/s")
                        if (speedup is not None and speedup > 1.01 and
                            (best_win is None
                             or tps > (1.0 / best_win.fitness_ns * 1e9))):
                            best_win = TunedWin(
                                device=me.device,
                                page_size=me.page_size,
                                dtype_key=me.dtype_key,
                                head_key=me.head_key,
                                extra_key=me.extra_key,
                                bkv_p=bkv_p,
                                bq=bq,
                                fitness_ns=data["mean_wall_time_s"] * 1e9,
                                speedup_vs_baseline=speedup,
                                referencing_models=list(me.referencing_models),
                            )
                    else:
                        tel.emit(
                            TelemetryEvent(
                                timestamp=time.time(),
                                run_id=run_id,
                                kernel="rpa_v3",
                                shape_key=f"{me.head_key}/ml{ml}",
                                genome_id=f"{me.head_key}_ml{ml}_{label}",
                                parent_ids=["baseline"],
                                generation=1,
                                island_id=0,
                                diff_summary=diff[:200],
                                status="FAILED_RUN",
                                fitness_ns=None,
                                p50_ns=None,
                                p95_ns=None,
                                cosine=None,
                                max_abs_diff=None,
                                wall_time_s=data.get("wall", 0) if data else 0,
                                rule_name=f"rpa_v3_block_{bkv_p}_{bq}",
                                extra={},
                            ))
                        print(f"  {label}: FAILED")
                except Exception as err:
                    tel.emit(
                        TelemetryEvent(
                            timestamp=time.time(),
                            run_id=run_id,
                            kernel="rpa_v3",
                            shape_key=f"{me.head_key}/ml{ml}",
                            genome_id=f"{me.head_key}_ml{ml}_{label}",
                            parent_ids=["baseline"],
                            generation=1,
                            island_id=0,
                            diff_summary="",
                            status="FAILED_DIFF",
                            fitness_ns=None,
                            p50_ns=None,
                            p95_ns=None,
                            cosine=None,
                            max_abs_diff=None,
                            wall_time_s=0,
                            rule_name=f"rpa_v3_block_{bkv_p}_{bq}",
                            extra={"error": str(err)},
                        ))
                    print(f"  {label}: SKIP ({err})")
                finally:
                    args.tuned_path.write_text(orig)

            if best_win is not None:
                wins.append(best_win)
                print(f"  WIN for {me.head_key}/ml{ml}: "
                      f"({best_win.bkv_p}, {best_win.bq}) → "
                      f"{best_win.speedup_vs_baseline:.3f}x")

    # Build auto-PR.
    if wins:
        new_source, pr_body = build_auto_pr(wins=wins,
                                            tuned_path=args.tuned_path)
        orig = args.tuned_path.read_text()
        # Write both files and use system diff for a proper unified diff.
        from tempfile import NamedTemporaryFile
        with NamedTemporaryFile("w", suffix=".old.py", delete=False) as fa:
            fa.write(orig)
            pa = fa.name
        with NamedTemporaryFile("w", suffix=".new.py", delete=False) as fb:
            fb.write(new_source)
            pb = fb.name
        proc = subprocess.run([
            "diff", "-u", "--label", f"a/{_rel(args.tuned_path)}", "--label",
            f"b/{_rel(args.tuned_path)}", pa, pb
        ],
                              capture_output=True,
                              text=True)
        args.out_diff.write_text(proc.stdout)
        args.out_summary.write_text(
            json.dumps({
                "wins": [w.__dict__ for w in wins],
                "pr_body": pr_body
            },
                       indent=2,
                       default=str))
        print(f"\nAuto-PR diff: {args.out_diff}")
        print(f"PR body / summary: {args.out_summary}")
    else:
        args.out_diff.write_text("")
        args.out_summary.write_text(json.dumps({"wins": []}))
        print("\nNo wins found.")

    # Print final summary
    print(f"\n{'=' * 70}")
    print(f"Sweep complete. {len(wins)} verified wins.")
    for w in sorted(wins, key=lambda x: -(x.speedup_vs_baseline or 0)):
        print(f"  {w.head_key}/{w.extra_key}: ({w.bkv_p},{w.bq}) "
              f"-> {w.speedup_vs_baseline:.3f}x")
    return 0


if __name__ == "__main__":
    sys.exit(main())
