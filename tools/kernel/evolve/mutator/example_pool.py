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
"""Curated positive-example pool — the RLAIF analogue.

Every verified winning diff (numerics-passing + statistically significant
when stats-bench has run) goes into a persistent pool keyed on kernel
name. When building the next mutation prompt, the top-K examples for the
target kernel are injected as few-shot exemplars.

Why this matters: a single winning diff is wasted information without
this. Once the system finds a real structural mutation that worked on
RPA v3, every subsequent prompt for any related kernel can use that
example as evidence that "this family of change is acceptable here."

This is in-context RL: the reward signal (verified + significant win) is
re-fed into the policy (the LLM's next prompt). No model fine-tuning
required.
"""

from __future__ import annotations

import dataclasses
import json
import math
import os
from pathlib import Path


@dataclasses.dataclass
class PositiveExample:
    """One verified mutation worth showing the LLM later.

    ``family_tags`` optionally records the perf-skill family letters this
    example instantiates (e.g. ``['H', 'A']`` for a dtype-policy win that
    also fuses an op). When set, the example becomes available to other
    kernels via ``ExamplePool.for_family()`` — the cross-kernel knowledge
    transfer path. Without tags, the example is per-kernel only.
    """
    kernel: str
    diff: str
    speedup: float
    p_value: float | None  # set if it passed stats-bench
    hypothesis: str  # 1-2 line description
    added_at: float
    source_run_id: str = ""
    family_tags: list[str] = dataclasses.field(default_factory=list)

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "PositiveExample":
        return cls(kernel=d["kernel"],
                   diff=d["diff"],
                   speedup=float(d.get("speedup", 1.0)),
                   p_value=d.get("p_value"),
                   hypothesis=d.get("hypothesis", ""),
                   added_at=float(d.get("added_at", 0.0)),
                   source_run_id=d.get("source_run_id", ""),
                   family_tags=list(d.get("family_tags", [])))


class ExamplePool:
    """Append-only JSON-backed positive example store."""

    def __init__(self,
                 *,
                 persist_path: str | os.PathLike | None = None,
                 max_per_kernel: int = 16) -> None:
        self.persist_path = (Path(persist_path)
                             if persist_path is not None else None)
        self.max_per_kernel = max_per_kernel
        self._by_kernel: dict[str, list[PositiveExample]] = {}
        if self.persist_path is not None and self.persist_path.exists():
            self._load()

    def add(self, example: PositiveExample) -> bool:
        """Insert a new example. Caps per-kernel; demotes lowest-speedup
        when over capacity. Returns True if inserted."""
        bucket = self._by_kernel.setdefault(example.kernel, [])
        # De-dup by diff hash.
        for existing in bucket:
            if existing.diff == example.diff:
                return False
        bucket.append(example)
        bucket.sort(key=lambda e: -e.speedup)
        if len(bucket) > self.max_per_kernel:
            bucket[:] = bucket[:self.max_per_kernel]
        self._save()
        return True

    def for_kernel(self,
                   kernel: str,
                   *,
                   top_k: int = 3,
                   min_speedup: float = 1.01) -> list[PositiveExample]:
        out: list[PositiveExample] = []
        for e in self._by_kernel.get(kernel, []):
            if e.speedup < min_speedup:
                continue
            out.append(e)
            if len(out) >= top_k:
                break
        return out

    def for_family(self,
                   family: str,
                   *,
                   exclude_kernel: str | None = None,
                   top_k: int = 2,
                   min_speedup: float = 1.02) -> list[PositiveExample]:
        """Return cross-kernel examples that share a family tag.

        Used to surface a verified family-H win on RPA v3 as inspiration
        when evolving MLA v2 or any other kernel. The exclusion keeps the
        per-kernel pool authoritative for its own kernel.
        """
        candidates: list[PositiveExample] = []
        for kernel, items in self._by_kernel.items():
            if exclude_kernel is not None and kernel == exclude_kernel:
                continue
            for e in items:
                if family not in e.family_tags:
                    continue
                if e.speedup < min_speedup:
                    continue
                candidates.append(e)
        candidates.sort(key=lambda e: -e.speedup)
        return candidates[:top_k]

    def all_kernels(self) -> list[str]:
        return sorted(self._by_kernel.keys())

    def size(self) -> int:
        return sum(len(v) for v in self._by_kernel.values())

    def _save(self) -> None:
        if self.persist_path is None:
            return
        with self.persist_path.open("w") as f:
            data = {
                kernel: [e.to_dict() for e in items]
                for kernel, items in self._by_kernel.items()
            }
            json.dump(data, f, indent=2, default=str)

    def _load(self) -> None:
        try:
            with self.persist_path.open("r") as f:
                data = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            return
        for kernel, items in data.items():
            self._by_kernel[kernel] = [
                PositiveExample.from_dict(d) for d in items
            ]


def render_examples_for_prompt(examples: list[PositiveExample], ) -> str:
    """Format an example list as a prompt section.

    The format is meant to be unambiguous to the LLM: each example is its
    own fenced ``diff`` block with a numbered header showing the measured
    speedup. The LLM is then asked to draw inspiration without copying
    verbatim.
    """
    if not examples:
        return ""
    lines = [
        "\n## Past verified wins (for inspiration — don't copy verbatim)\n"
    ]
    for i, e in enumerate(examples, start=1):
        speed = f"{e.speedup:.3f}x"
        p_tag = (f", p={e.p_value:.3f}"
                 if e.p_value is not None and math.isfinite(e.p_value) else "")
        head = (f"\n### Example {i} — verified +{speed} on "
                f"`{e.kernel}`{p_tag}\n")
        hypo_line = f"_Hypothesis: {e.hypothesis.strip()}_\n" if e.hypothesis else ""
        lines.append(f"{head}{hypo_line}\n```diff\n{e.diff.strip()}\n```\n")
    return "".join(lines)
