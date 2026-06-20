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
"""Sweep EXISTING tuned table entries to find sub-optimal ones.

Strategy: for each (model, max_model_len) of interest:
* Look up the current tuned ``(bkv_p, bq)``.
* Try a small set of candidate ``(bkv_p, bq)`` perturbations.
* If any candidate beats the current value by more than a noise margin,
  record a verified win.

The auto-PR diff then *replaces* (not inserts) the value in the table.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import subprocess
import sys
import time
from pathlib import Path

from tools.kernel.evolve.sweep.missing_entries import (POPULAR_MODELS,
                                                       _dtype_key, _extra_key,
                                                       _head_key)
from tools.kernel.evolve.telemetry.writer import (TelemetryEvent,
                                                  TelemetryWriter)


def _load_table(path: Path) -> dict:
    spec = importlib.util.spec_from_file_location("_tmp", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.TUNED_BLOCK_SIZES


def _current_entry(table: dict, device: str, page_size: int, dtype_key: str,
                   head_key: str, extra_key: str) -> tuple[int, int] | None:
    try:
        v = table[device][page_size][dtype_key][head_key][extra_key]
        return tuple(v)
    except KeyError:
        return None


def _find_value_line(path: Path, head_key: str, extra_key: str,
                     dtype_key: str) -> int | None:
    """Return the 0-based line index of the entry inside the right block.

    Walks the file linearly, tracking which (dtype, head) block we're in.
    """
    text = path.read_text()
    lines = text.splitlines(keepends=True)
    in_dtype = False
    in_head = False
    head_depth = 0  # 0 = outside, >0 inside head block
    for i, line in enumerate(lines):
        if f"'{dtype_key}'" in line and "{" in line:
            in_dtype = True
            continue
        if not in_dtype:
            continue
        # Detect leaving the dtype block (a sibling-level dtype starts).
        if in_dtype and not in_head and "'q_" in line and "_kv_" in line:
            in_dtype_dtype_marker = line.lstrip().startswith("'q_bfloat") or \
                                     line.lstrip().startswith("'q_float") or \
                                     line.lstrip().startswith("'q_int") or \
                                     line.lstrip().startswith("'q_uint")
            if in_dtype_dtype_marker and f"'{dtype_key}'" not in line:
                in_dtype = False
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


def _make_replace_diff(path: Path, line_idx: int, new_bkv_p: int,
                       new_bq: int) -> str:
    """Build a unified diff that replaces just one value on the given line."""
    lines = path.read_text().splitlines(keepends=True)
    old_line = lines[line_idx].rstrip("\n")
    # Replace the tuple ``(X, Y),`` after the colon.
    new_line = re.sub(
        r"\(\s*\d+\s*,\s*\d+\s*\)",
        f"({new_bkv_p}, {new_bq})",
        old_line,
        count=1,
    )
    if new_line == old_line:
        raise ValueError(f"could not rewrite tuple on line: {old_line!r}")
    rel = str(path).replace("/home/qizzzh_google_com/tpu-inference/", "")
    return (f"--- a/{rel}\n"
            f"+++ b/{rel}\n"
            f"@@ -{line_idx + 1},1 +{line_idx + 1},1 @@\n"
            f"-{old_line}\n"
            f"+{new_line}\n")


def _bench(label: str, max_model_len: int, max_tokens: int,
           out_path: Path) -> dict | None:
    """Run a Qwen3-0.6B benchmark and return the parsed result."""
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
        "2",
        "--num-measure-rounds",
        "3",
    ]
    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
    wall = time.time() - t0
    if proc.returncode != 0:
        return {"ok": False, "err": proc.stderr[-1500:], "wall": wall}
    try:
        d = json.loads(out_path.read_text())
        d["ok"] = True
        d["wall"] = wall
        return d
    except (FileNotFoundError, json.JSONDecodeError):
        return {"ok": False, "err": "no result JSON", "wall": wall}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--tuned-path",
                   type=Path,
                   default=Path("/home/qizzzh_google_com/tpu-inference/"
                                "tpu_inference/kernels/ragged_paged_attention/"
                                "v3/tuned_block_sizes.py"))
    p.add_argument("--out-diff",
                   type=Path,
                   default=Path("/tmp/suboptimal_auto_pr.diff"))
    p.add_argument("--out-summary",
                   type=Path,
                   default=Path("/tmp/suboptimal_summary.json"))
    p.add_argument("--out-telemetry",
                   type=Path,
                   default=Path("/tmp/suboptimal_telemetry.jsonl"))
    p.add_argument("--models", type=str, default="Qwen3-0.6B")
    p.add_argument("--context-lengths", type=str, default="1024")
    p.add_argument("--device", default="TPU v7")
    p.add_argument("--page-size", type=int, default=128)
    p.add_argument("--max-tokens", type=int, default=64)
    p.add_argument("--candidates",
                   type=str,
                   default="",
                   help="Comma-separated bkv:bq pairs. "
                   "Default: a strong 4-candidate set.")
    p.add_argument("--min-win-margin",
                   type=float,
                   default=1.005,
                   help="Speedup floor to call a win (default 0.5%%).")
    args = p.parse_args(argv)

    candidates = ([
        tuple(int(x) for x in c.split(":")) for c in args.candidates.split(",")
    ] if args.candidates else [(4, 32), (8, 32), (8, 64), (16, 32), (8, 16)])
    context_lengths = [int(x) for x in args.context_lengths.split(",")]
    model_names = [m.strip() for m in args.models.split(",")]
    selected = [m for m in POPULAR_MODELS if m.name in model_names]

    table = _load_table(args.tuned_path)
    targets: list[tuple[str, str, tuple[int, int]]] = []
    for m in selected:
        dt = _dtype_key(m)
        hd = _head_key(m)
        for ml in context_lengths:
            ek = _extra_key(ml, m.sliding_window)
            cur = _current_entry(table, args.device, args.page_size, dt, hd,
                                 ek)
            if cur is None:
                continue
            targets.append((hd, ek, cur))

    if not targets:
        print("No targets found.")
        return 1

    print(f"Targets: {len(targets)}")
    for hd, ek, cur in targets:
        print(f"  {hd}/{ek} current={cur}")

    args.out_telemetry.parent.mkdir(parents=True, exist_ok=True)
    wins: list[dict] = []
    with TelemetryWriter(args.out_telemetry) as tel:
        run_id = f"subopt_{int(time.time())}"

        for hd, ek, cur in targets:
            line_idx = _find_value_line(args.tuned_path, hd, ek,
                                        "q_bfloat16_kv_bfloat16")
            if line_idx is None:
                print(f"  [skip] could not locate line for {hd}/{ek}")
                continue
            ml_match = re.match(r"max_model_len-(\d+)-sw-", ek)
            if not ml_match:
                continue
            ml = int(ml_match.group(1))

            print(f"\n[target] {hd}/{ek} current={cur}")
            # Baseline: leave file alone, measure current production value.
            bl = _bench("baseline", ml, args.max_tokens,
                        args.out_diff.parent / f"baseline_{hd}_{ml}.json")
            tel.emit(
                TelemetryEvent(
                    timestamp=time.time(),
                    run_id=run_id,
                    kernel="rpa_v3",
                    shape_key=f"{hd}/ml{ml}",
                    genome_id="baseline",
                    parent_ids=[],
                    generation=0,
                    island_id=0,
                    diff_summary="",
                    status="VERIFIED"
                    if bl and bl.get("ok") else "BASELINE_FAIL",
                    fitness_ns=(bl["mean_wall_time_s"] *
                                1e9 if bl and bl.get("ok") else None),
                    p50_ns=None,
                    p95_ns=None,
                    cosine=None,
                    max_abs_diff=None,
                    wall_time_s=bl["wall"] if bl else 0,
                    rule_name="baseline",
                    extra={
                        "throughput_tok_s":
                        bl.get("throughput_tokens_per_s") if bl else None,
                        "value":
                        cur
                    },
                ))
            if not bl or not bl.get("ok"):
                print("  baseline failed")
                continue
            base_tps = bl["throughput_tokens_per_s"]
            print(f"  baseline {cur}: {base_tps:.2f} tok/s")

            best_label = None
            best_tps = base_tps
            best_pair = cur
            best_speedup = 1.0
            for (bkv, bq) in candidates:
                if (bkv, bq) == cur:
                    continue
                label = f"bkv{bkv}_bq{bq}"
                try:
                    diff = _make_replace_diff(args.tuned_path, line_idx, bkv,
                                              bq)
                except ValueError as err:
                    print(f"  {label}: SKIP ({err})")
                    continue
                orig = args.tuned_path.read_text()
                from tools.kernel.evolve.mutator.diff_applier import apply_diff
                applied = apply_diff(orig, diff)
                if not applied.success or applied.new_source is None:
                    print(f"  {label}: diff failed — {applied.error}")
                    continue
                args.tuned_path.write_text(applied.new_source)
                try:
                    r = _bench(
                        label, ml, args.max_tokens,
                        args.out_diff.parent / f"{label}_{hd}_{ml}.json")
                    if r and r.get("ok"):
                        tps = r["throughput_tokens_per_s"]
                        speedup = tps / base_tps
                        status = ("VERIFIED" if speedup >= args.min_win_margin
                                  else "VERIFIED_NO_WIN")
                        tel.emit(
                            TelemetryEvent(
                                timestamp=time.time(),
                                run_id=run_id,
                                kernel="rpa_v3",
                                shape_key=f"{hd}/ml{ml}",
                                genome_id=f"{hd}_ml{ml}_{label}",
                                parent_ids=["baseline"],
                                generation=1,
                                island_id=0,
                                diff_summary=diff[:200],
                                status=status,
                                fitness_ns=r["mean_wall_time_s"] * 1e9,
                                p50_ns=None,
                                p95_ns=None,
                                cosine=None,
                                max_abs_diff=None,
                                wall_time_s=r["wall"],
                                rule_name=f"rpa_v3_replace_{bkv}_{bq}",
                                extra={
                                    "throughput_tok_s": tps,
                                    "speedup": speedup,
                                    "current": cur,
                                    "value": (bkv, bq)
                                },
                            ))
                        print(f"  {label}: {tps:.2f} tok/s ({speedup:.3f}x)")
                        if speedup > best_speedup:
                            best_speedup = speedup
                            best_tps = tps
                            best_pair = (bkv, bq)
                            best_label = label
                    else:
                        print(f"  {label}: FAILED")
                        tel.emit(
                            TelemetryEvent(
                                timestamp=time.time(),
                                run_id=run_id,
                                kernel="rpa_v3",
                                shape_key=f"{hd}/ml{ml}",
                                genome_id=f"{hd}_ml{ml}_{label}",
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
                                wall_time_s=r["wall"] if r else 0,
                                rule_name=f"rpa_v3_replace_{bkv}_{bq}",
                                extra={"value": (bkv, bq)},
                            ))
                finally:
                    args.tuned_path.write_text(orig)

            if best_label and best_speedup >= args.min_win_margin:
                wins.append({
                    "head_key": hd,
                    "extra_key": ek,
                    "current": cur,
                    "new": best_pair,
                    "speedup": best_speedup,
                    "baseline_tok_s": base_tps,
                    "new_tok_s": best_tps,
                    "line_index": line_idx,
                })
                print(f"  WIN: {cur} → {best_pair}  ({best_speedup:.3f}x)")

    args.out_summary.write_text(
        json.dumps(
            {
                "wins": wins,
                "candidates_tested": candidates,
                "min_win_margin": args.min_win_margin
            },
            indent=2,
            default=str))

    if wins:
        # Build a single accumulated diff by applying ALL wins in sequence.
        from tools.kernel.evolve.mutator.diff_applier import apply_diff
        orig = args.tuned_path.read_text()
        cur_src = orig
        for w in wins:
            d = _make_replace_diff(args.tuned_path, w["line_index"],
                                   w["new"][0], w["new"][1])
            r = apply_diff(cur_src, d)
            if r.success and r.new_source is not None:
                cur_src = r.new_source
        from tempfile import NamedTemporaryFile
        with NamedTemporaryFile("w", suffix=".old.py", delete=False) as fa:
            fa.write(orig)
            pa = fa.name
        with NamedTemporaryFile("w", suffix=".new.py", delete=False) as fb:
            fb.write(cur_src)
            pb = fb.name
        proc = subprocess.run([
            "diff", "-u", "--label",
            "a/tpu_inference/kernels/ragged_paged_attention/v3/tuned_block_sizes.py",
            "--label",
            "b/tpu_inference/kernels/ragged_paged_attention/v3/tuned_block_sizes.py",
            pa, pb
        ],
                              capture_output=True,
                              text=True)
        args.out_diff.write_text(proc.stdout)
        print(f"\nAuto-PR diff: {args.out_diff}")
    else:
        args.out_diff.write_text("")
        print("\nNo wins above margin.")

    print(f"\n{'=' * 70}")
    print(f"Sub-optimal sweep complete. {len(wins)} verified wins.")
    for w in sorted(wins, key=lambda x: -x["speedup"]):
        print(f"  {w['head_key']}/{w['extra_key']}: "
              f"{w['current']} -> {w['new']} ({w['speedup']:.3f}x, "
              f"{w['baseline_tok_s']:.0f} -> {w['new_tok_s']:.0f} tok/s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
