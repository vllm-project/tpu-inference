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
"""Abstract base for search strategies and the typed parameter ranges they
operate over.

A strategy proposes ``params: dict[str, Any]`` and is told the
``score: float`` for each evaluation. Score semantics:

* finite, lower-is-better (e.g. latency in nanoseconds);
* ``math.inf`` means the candidate failed for any reason (OOM, verifier
  failure, cost-model rejection). Strategies should treat ``inf`` as
  "no information" rather than "very bad" — TPE prunes such trials,
  EA filters them out of tournament selection.
"""

import abc
import dataclasses
import math
from typing import Any


@dataclasses.dataclass
class ParamRange:
    """Typed range over a single tunable parameter.

    Subclasses implement ``sample`` (random draw, used by EA seeding),
    ``values`` (full enumeration, used by grid search), and ``neighbours``
    (mutation step, used by EA).
    """
    name: str

    def sample(self, rng) -> Any:  # pragma: no cover - subclass implements
        raise NotImplementedError

    def values(self) -> list[Any]:  # pragma: no cover - subclass implements
        raise NotImplementedError

    def neighbours(self, x: Any) -> list[Any]:
        """Mutation neighbours for ``x``. Default: all values."""
        return self.values()


@dataclasses.dataclass
class IntChoice(ParamRange):
    """An explicit list of integer options."""
    options: list[int] = dataclasses.field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.options:
            raise ValueError(
                f"{self.name}: IntChoice needs at least one option")

    def sample(self, rng) -> int:
        return int(self.options[int(rng.integers(0, len(self.options)))])

    def values(self) -> list[int]:
        return list(self.options)

    def neighbours(self, x: int) -> list[int]:
        if x not in self.options:
            return list(self.options)
        i = self.options.index(x)
        out = []
        if i - 1 >= 0:
            out.append(self.options[i - 1])
        if i + 1 < len(self.options):
            out.append(self.options[i + 1])
        return out


@dataclasses.dataclass
class IntLog2Range(ParamRange):
    """Powers of two ``[min_val, max_val]`` (inclusive)."""
    min_val: int = 1
    max_val: int = 1024

    def __post_init__(self) -> None:
        if self.min_val < 1:
            raise ValueError(f"{self.name}: min_val must be >= 1")
        if self.max_val < self.min_val:
            raise ValueError(
                f"{self.name}: max_val ({self.max_val}) < min_val "
                f"({self.min_val})")
        opts: list[int] = []
        v = 1 << (self.min_val - 1).bit_length() if self.min_val > 1 else 1
        while v <= self.max_val:
            opts.append(v)
            v *= 2
        if not opts:
            raise ValueError(
                f"{self.name}: empty range [{self.min_val}, {self.max_val}]")
        self._opts = opts

    def sample(self, rng) -> int:
        return int(self._opts[int(rng.integers(0, len(self._opts)))])

    def values(self) -> list[int]:
        return list(self._opts)

    def neighbours(self, x: int) -> list[int]:
        if x not in self._opts:
            return list(self._opts)
        i = self._opts.index(x)
        out = []
        if i - 1 >= 0:
            out.append(self._opts[i - 1])
        if i + 1 < len(self._opts):
            out.append(self._opts[i + 1])
        return out


SearchSpace = dict[str, ParamRange]


class SearchStrategy(abc.ABC):
    """Generator API for search.

    Drives the inner loop in ``kernel_tuner_runner.run_smart_search``:

    ::

        while not strategy.done():
            params = strategy.suggest()
            score = evaluate(params)  # inf on any failure
            strategy.observe(params, score, aux)
        best_params, best_score = strategy.best()
    """

    def __init__(self, *, space: SearchSpace, trial_budget: int) -> None:
        if trial_budget < 1:
            raise ValueError(f"trial_budget must be >= 1, got {trial_budget}")
        if not space:
            raise ValueError("space must be non-empty")
        self.space = space
        self.trial_budget = trial_budget
        self._best_params: dict[str, Any] | None = None
        self._best_score: float = math.inf
        self._trials_observed: int = 0

    @abc.abstractmethod
    def suggest(self) -> dict[str, Any]:
        ...

    def observe(
        self,
        params: dict[str, Any],
        score: float,
        aux: dict[str, Any] | None = None,
    ) -> None:
        self._trials_observed += 1
        if math.isfinite(score) and score < self._best_score:
            self._best_score = score
            self._best_params = dict(params)

    def done(self) -> bool:
        return self._trials_observed >= self.trial_budget

    def best(self) -> tuple[dict[str, Any] | None, float]:
        return self._best_params, self._best_score

    @property
    def trials_observed(self) -> int:
        return self._trials_observed
