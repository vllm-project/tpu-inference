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
"""Failure-learning feedback loop.

Tracks per-mutation-rule reject rates and verified-rates. Two effects:

* Mutators that consult ``FailureLog`` can down-weight rules whose
  verified-rate falls below a floor (lifting the overall verified-rate).
* The orchestrator can render a "anti-patterns to avoid" hint section in
  the LLM mutation prompt so the model doesn't repeat past failures.
"""

from __future__ import annotations

import dataclasses
import json
import math
from collections import defaultdict
from pathlib import Path


@dataclasses.dataclass
class RuleStats:
    rule_name: str
    proposed: int = 0
    verified: int = 0
    by_failure_status: dict[str, int] = dataclasses.field(default_factory=dict)
    best_fitness_ns: float | None = None

    @property
    def verified_rate(self) -> float:
        return self.verified / self.proposed if self.proposed else 0.0

    def to_dict(self) -> dict:
        return {
            "rule_name": self.rule_name,
            "proposed": self.proposed,
            "verified": self.verified,
            "verified_rate": self.verified_rate,
            "by_failure_status": dict(self.by_failure_status),
            "best_fitness_ns": self.best_fitness_ns,
        }


class FailureLog:
    """In-memory tracker with optional JSON persistence."""

    def __init__(self, *, persist_path: str | Path | None = None) -> None:
        self._by_rule: dict[str, RuleStats] = {}
        self.persist_path = Path(persist_path) if persist_path else None
        if self.persist_path is not None and self.persist_path.exists():
            self._load()

    def record(self,
               *,
               rule_name: str,
               status: str,
               fitness_ns: float | None = None) -> None:
        rs = self._by_rule.setdefault(rule_name, RuleStats(rule_name))
        rs.proposed += 1
        if status == "VERIFIED":
            rs.verified += 1
            if fitness_ns is not None and math.isfinite(fitness_ns):
                if rs.best_fitness_ns is None or fitness_ns < rs.best_fitness_ns:
                    rs.best_fitness_ns = fitness_ns
        else:
            rs.by_failure_status[status] = (
                rs.by_failure_status.get(status, 0) + 1)
        self._save()

    def verified_rate(self, rule_name: str) -> float:
        rs = self._by_rule.get(rule_name)
        return rs.verified_rate if rs else 0.0

    def is_dead_rule(self,
                     rule_name: str,
                     *,
                     min_observations: int = 8,
                     min_verified_rate: float = 0.05) -> bool:
        """A rule is considered dead if it's been observed enough times AND
        its verified rate is below the floor."""
        rs = self._by_rule.get(rule_name)
        if rs is None or rs.proposed < min_observations:
            return False
        return rs.verified_rate < min_verified_rate

    def anti_patterns(self) -> list[str]:
        """Human-readable summary suitable for LLM mutator prompt."""
        out: list[str] = []
        for rs in sorted(self._by_rule.values(), key=lambda x: -x.proposed):
            if rs.proposed < 4:
                continue
            top_fail = max(rs.by_failure_status.items(),
                           key=lambda kv: kv[1],
                           default=(None, 0))
            if top_fail[0] is None or rs.verified_rate >= 0.5:
                continue
            out.append(
                f"- Rule {rs.rule_name!r}: {rs.verified}/{rs.proposed} verified "
                f"({rs.verified_rate:.0%}), most-common-failure={top_fail[0]} "
                f"({top_fail[1]}x).")
        return out

    def all_stats(self) -> list[RuleStats]:
        return sorted(self._by_rule.values(), key=lambda rs: rs.rule_name)

    def _save(self) -> None:
        if self.persist_path is None:
            return
        with self.persist_path.open("w") as f:
            json.dump(
                {rs.rule_name: rs.to_dict()
                 for rs in self._by_rule.values()},
                f,
                indent=2,
                default=str)

    def _load(self) -> None:
        with self.persist_path.open("r") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                return
        for name, d in data.items():
            rs = RuleStats(
                rule_name=name,
                proposed=int(d.get("proposed", 0)),
                verified=int(d.get("verified", 0)),
                by_failure_status=defaultdict(
                    int, dict(d.get("by_failure_status", {}))),
                best_fitness_ns=d.get("best_fitness_ns"),
            )
            self._by_rule[name] = rs
