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
"""Shared logic for tuned-table sweep recipes (per kernel)."""

from __future__ import annotations

import dataclasses
import importlib.util
import json
import time
from pathlib import Path
from typing import Callable

from tools.kernel.evolve.telemetry.writer import (TelemetryEvent,
                                                  TelemetryWriter)


@dataclasses.dataclass
class TunedTableRecipe:
    """Describes a per-kernel sub-optimal-entry sweep."""
    kernel: str
    tuned_path: Path
    table_attr: str
    out_dir: Path
    microbench_fn: Callable[[dict | tuple], dict]
    entry_extractor: Callable[[dict], list[tuple[tuple, tuple]]]
    # entry_extractor takes the loaded TUNED dict and returns a list of
    # (key_tuple, value_tuple) entries to consider sweeping.
    candidates: list[tuple] = dataclasses.field(default_factory=list)


def _load_attr(path: Path, attr: str):
    spec = importlib.util.spec_from_file_location("_tuned", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return getattr(mod, attr)


def run_sweep(recipe: TunedTableRecipe,
              *,
              candidates_per_key: list[tuple] | None = None,
              max_keys: int = 8,
              min_win_margin: float = 1.01) -> dict:
    """Generic sub-optimal-entry sweep over the recipe's tuned table.

    Returns a CI summary in the shape expected by ``ci.nightly``.
    """
    table = _load_attr(recipe.tuned_path, recipe.table_attr)
    out_dir = recipe.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    telemetry_path = out_dir / f"{recipe.kernel}_telemetry.jsonl"
    summary_path = out_dir / f"{recipe.kernel}_summary.json"
    diff_path = out_dir / f"{recipe.kernel}_auto_pr.diff"

    targets = recipe.entry_extractor(table)[:max_keys]
    if not targets:
        return {
            "target_kernel": recipe.kernel,
            "wins_count": 0,
            "diff_path": None,
            "summary_path": str(summary_path),
            "telemetry_path": str(telemetry_path),
            "error": "no targets"
        }
    candidates = candidates_per_key or recipe.candidates
    wins: list[dict] = []
    with TelemetryWriter(telemetry_path) as tel:
        run_id = f"{recipe.kernel}_{int(time.time())}"
        for key, cur_value in targets:
            print(f"\n[{recipe.kernel}] target key={key} current={cur_value}")
            baseline = recipe.microbench_fn({
                "key": key,
                "value": cur_value,
                "label": "baseline"
            })
            tel.emit(
                TelemetryEvent(
                    timestamp=time.time(),
                    run_id=run_id,
                    kernel=recipe.kernel,
                    shape_key=str(key),
                    genome_id="baseline",
                    parent_ids=[],
                    generation=0,
                    island_id=0,
                    diff_summary="",
                    status=baseline.get("status", "BASELINE_FAIL"),
                    fitness_ns=baseline.get("latency_ns"),
                    p50_ns=baseline.get("p50_ns"),
                    p95_ns=baseline.get("p95_ns"),
                    cosine=baseline.get("cosine"),
                    max_abs_diff=baseline.get("max_abs_diff"),
                    wall_time_s=baseline.get("wall_s", 0.0),
                    rule_name="baseline",
                    extra={"value": cur_value},
                ))
            if baseline.get("status") != "VERIFIED":
                continue
            base_lat = float(baseline["latency_ns"])

            best: dict | None = None
            for cand in candidates:
                if tuple(cand) == tuple(cur_value):
                    continue
                label = f"cand_{cand}"
                result = recipe.microbench_fn({
                    "key": key,
                    "value": tuple(cand),
                    "label": label,
                })
                speedup = (base_lat / result["latency_ns"]
                           if result.get("status") == "VERIFIED"
                           and result.get("latency_ns") else None)
                tel.emit(
                    TelemetryEvent(
                        timestamp=time.time(),
                        run_id=run_id,
                        kernel=recipe.kernel,
                        shape_key=str(key),
                        genome_id=f"{recipe.kernel}_{key}_{cand}",
                        parent_ids=["baseline"],
                        generation=1,
                        island_id=0,
                        diff_summary=f"{cur_value} -> {cand}",
                        status=result.get("status", "UNKNOWN"),
                        fitness_ns=result.get("latency_ns"),
                        p50_ns=result.get("p50_ns"),
                        p95_ns=result.get("p95_ns"),
                        cosine=result.get("cosine"),
                        max_abs_diff=result.get("max_abs_diff"),
                        wall_time_s=result.get("wall_s", 0.0),
                        rule_name=f"replace_{cand}",
                        extra={
                            "speedup": speedup,
                            "current": cur_value,
                            "value": cand
                        },
                    ))
                if speedup is not None and speedup >= min_win_margin:
                    if best is None or speedup > best["speedup"]:
                        best = {
                            "key": list(key),
                            "current": list(cur_value),
                            "new": list(cand),
                            "speedup": speedup,
                            "baseline_ns": base_lat,
                            "new_ns": result["latency_ns"]
                        }
            if best is not None:
                wins.append(best)
                print(f"  WIN: {best['current']} -> {best['new']} "
                      f"({best['speedup']:.3f}x)")

    summary_path.write_text(json.dumps({"wins": wins}, indent=2, default=str))
    if wins:
        diff_path.write_text(_format_summary_diff(recipe, wins))
    else:
        diff_path.write_text("")
    return {
        "target_kernel": recipe.kernel,
        "wins_count": len(wins),
        "diff_path": str(diff_path) if wins else None,
        "summary_path": str(summary_path),
        "telemetry_path": str(telemetry_path)
    }


def _format_summary_diff(recipe: TunedTableRecipe, wins: list[dict]) -> str:
    """Render a human-readable patch summary (not a real applyable diff)."""
    lines = [
        f"# Sub-optimal-entry sweep results for {recipe.kernel}",
        f"# Source table: {recipe.tuned_path}", ""
    ]
    for w in wins:
        lines.append(f"# key={w['key']}  {w['current']} -> {w['new']}  "
                     f"speedup={w['speedup']:.3f}x")
    return "\n".join(lines) + "\n"
