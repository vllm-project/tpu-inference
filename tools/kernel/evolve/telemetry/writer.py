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
"""Append-only JSONL telemetry writer.

One event per trial: timestamp, kernel, host shape, mutation diff hash,
fitness, status. Designed for downstream analysis by the dashboard CLI
and for the failure-learning feedback loop.
"""

from __future__ import annotations

import dataclasses
import json
import os
from pathlib import Path
from typing import Any


@dataclasses.dataclass
class TelemetryEvent:
    timestamp: float
    run_id: str
    kernel: str
    shape_key: str | None
    genome_id: str
    parent_ids: list[str]
    generation: int
    island_id: int
    diff_summary: str  # first 200 chars of the diff
    status: str
    fitness_ns: float | None
    p50_ns: int | None
    p95_ns: int | None
    cosine: float | None
    max_abs_diff: float | None
    wall_time_s: float
    rule_name: str | None = None  # which mutation rule produced this
    extra: dict[str, Any] = dataclasses.field(default_factory=dict)

    def to_jsonable(self) -> dict[str, Any]:
        d = dataclasses.asdict(self)
        if d["fitness_ns"] is not None and not (d["fitness_ns"]
                                                == d["fitness_ns"]):
            d["fitness_ns"] = "nan"
        return d


class TelemetryWriter:
    """Append-only JSONL writer with crash safety."""

    def __init__(self, path: str | os.PathLike) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fp = self.path.open("a")

    def emit(self, event: TelemetryEvent) -> None:
        self._fp.write(json.dumps(event.to_jsonable(), default=str) + "\n")
        self._fp.flush()

    def close(self) -> None:
        try:
            self._fp.close()
        except Exception:
            pass

    def __enter__(self) -> "TelemetryWriter":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def load_events(path: str | os.PathLike) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    p = Path(path)
    if not p.exists():
        return out
    with p.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def summarize(events: list[dict[str, Any]]) -> dict[str, Any]:
    """Group by kernel × shape × status; report verified-rate and best fitness."""
    by_kernel: dict[str, dict[str, Any]] = {}
    for ev in events:
        k = ev.get("kernel", "?")
        shape = ev.get("shape_key", "?")
        b = by_kernel.setdefault(k, {})
        s = b.setdefault(shape, {
            "total": 0,
            "verified": 0,
            "best_fitness_ns": None,
            "by_status": {},
        })
        s["total"] += 1
        st = ev.get("status", "?")
        s["by_status"][st] = s["by_status"].get(st, 0) + 1
        if ev.get("status") == "VERIFIED":
            s["verified"] += 1
            f = ev.get("fitness_ns")
            if f is not None and isinstance(f, (int, float)):
                if s["best_fitness_ns"] is None or f < s["best_fitness_ns"]:
                    s["best_fitness_ns"] = f
    return {"kernels": by_kernel}
