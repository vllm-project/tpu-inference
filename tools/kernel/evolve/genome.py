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
"""Genome — a single candidate in the evolution archive.

A genome is a unified diff against a baseline kernel source file plus its
measured fitness and provenance. Diffs (rather than whole source files) keep
the mutation surface small, make wins attributable to specific changes, and
let the archive be persisted compactly.
"""

import dataclasses
import enum
import hashlib
import math
from typing import Any


class GenomeStatus(str, enum.Enum):
    """Stages a genome goes through. Persistable as a plain string."""
    PENDING = "PENDING"
    EVALUATING = "EVALUATING"
    VERIFIED = "VERIFIED"
    REJECTED_CRITIC = "REJECTED_CRITIC"
    FAILED_DIFF = "FAILED_DIFF"
    FAILED_COMPILE = "FAILED_COMPILE"
    FAILED_RUN = "FAILED_RUN"
    FAILED_NUMERICS = "FAILED_NUMERICS"
    FAILED_ANTI_CHEAT = "FAILED_ANTI_CHEAT"

    @property
    def is_terminal(self) -> bool:
        return self != GenomeStatus.PENDING and self != GenomeStatus.EVALUATING

    @property
    def is_success(self) -> bool:
        return self == GenomeStatus.VERIFIED


def _hash_diff(diff: str, parent_ids: list[str], generation: int) -> str:
    """Stable 12-char id derived from diff + parents + generation."""
    h = hashlib.sha256()
    h.update(diff.encode("utf-8"))
    for p in sorted(parent_ids):
        h.update(p.encode("utf-8"))
    h.update(str(generation).encode("utf-8"))
    return h.hexdigest()[:12]


@dataclasses.dataclass
class Genome:
    """A candidate solution: diff against a baseline + provenance + fitness."""
    id: str
    diff: str  # unified-diff text; empty for the baseline genome
    baseline_path: str  # repo-relative path the diff applies to
    parent_ids: list[str] = dataclasses.field(default_factory=list)
    generation: int = 0
    island_id: int = 0
    status: GenomeStatus = GenomeStatus.PENDING
    fitness: float = math.inf  # avg_latency_ns; lower is better
    error: str | None = None
    metrics: dict[str, Any] = dataclasses.field(default_factory=dict)
    created_at: float = 0.0
    evaluated_at: float | None = None
    # Optional structured fields populated by the evaluator.
    mutated_source_preview: str | None = None  # first ~40 lines after apply

    @classmethod
    def new(
        cls,
        *,
        diff: str,
        baseline_path: str,
        parent_ids: list[str],
        generation: int,
        island_id: int,
        created_at: float = 0.0,
    ) -> "Genome":
        """Construct a pending genome with a stable id."""
        return cls(
            id=_hash_diff(diff, parent_ids, generation),
            diff=diff,
            baseline_path=baseline_path,
            parent_ids=list(parent_ids),
            generation=generation,
            island_id=island_id,
            created_at=created_at,
        )

    @classmethod
    def baseline(cls, baseline_path: str) -> "Genome":
        """The un-mutated baseline. fitness will be set after first eval."""
        return cls(
            id="baseline",
            diff="",
            baseline_path=baseline_path,
            parent_ids=[],
            generation=0,
            island_id=0,
            status=GenomeStatus.PENDING,
        )

    def to_dict(self) -> dict[str, Any]:
        d = dataclasses.asdict(self)
        d["status"] = self.status.value
        if not math.isfinite(self.fitness):
            d["fitness"] = "inf"
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Genome":
        status = GenomeStatus(d["status"])
        fitness = d["fitness"]
        if isinstance(fitness, str):
            fitness = math.inf
        return cls(
            id=d["id"],
            diff=d["diff"],
            baseline_path=d["baseline_path"],
            parent_ids=list(d.get("parent_ids", [])),
            generation=int(d.get("generation", 0)),
            island_id=int(d.get("island_id", 0)),
            status=status,
            fitness=float(fitness),
            error=d.get("error"),
            metrics=dict(d.get("metrics", {})),
            created_at=float(d.get("created_at", 0.0)),
            evaluated_at=d.get("evaluated_at"),
            mutated_source_preview=d.get("mutated_source_preview"),
        )

    def short(self) -> str:
        """One-line summary suitable for log output."""
        f = "inf" if not math.isfinite(
            self.fitness) else f"{self.fitness:.0f}ns"
        return (f"g{self.generation:>2} i{self.island_id} "
                f"{self.id} {self.status.value:<18} fitness={f}")
