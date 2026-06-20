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
"""Best-of-N sampler over an ``LLMClient`` mutator.

Equivalent in expectation to GRPO-style sampling without training a model:
ask the LLM for ``N`` candidates per turn at varied temperatures, score
each candidate offline (critic + cost-model surrogate), return the
highest-scoring one for TPU evaluation.

Three knobs:

* ``n`` — number of candidates per call. Inference cost scales linearly.
* ``temperatures`` — sequence of temperatures to fan out across (default
  ``[0.3, 0.5, 0.7, 0.9]``). Diverse temperatures explore the model's
  output distribution more thoroughly than identical calls.
* ``ranker`` — a callable ``ranker(diff_text, hypothesis) -> float`` that
  scores candidates; ``None`` falls back to a length heuristic (prefers
  shorter, more focused diffs).

The wrapper *also* returns the discarded siblings via ``last_siblings``,
which the orchestrator can persist as telemetry for failure-learning
without burning TPU time on them.
"""

from __future__ import annotations

import dataclasses
import inspect
import logging
from typing import Callable, Protocol

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class BonCandidate:
    raw: str
    temperature: float
    score: float
    extracted_diff: str | None = None
    rejected_reason: str | None = None  # critic verdict if rejected


class _LikeLLMClient(Protocol):

    @property
    def model_id(self) -> str:
        ...

    def chat(self,
             *,
             system: str,
             user: str,
             max_tokens: int = 4096,
             temperature: float | None = None) -> str:
        ...


def _default_ranker(raw: str, hypothesis: str | None = None) -> float:
    """Prefer focused diffs: fewer hunks, shorter total length.

    Tie-break by presence of a hypothesis sentence before the fenced block.
    """
    from tools.kernel.evolve.mutator.diff_applier import extract_diff
    try:
        diff = extract_diff(raw)
    except ValueError:
        return -float("inf")
    hunks = diff.count("\n@@ ")
    chars = len(diff)
    has_hypo = "hypothesis" in raw.lower() or "because" in raw.lower()
    # Lower is better; flip sign so higher = better.
    score = -chars - 200 * hunks + (50 if has_hypo else 0)
    return score


class BestOfNMutator:
    """LLMClient-shaped wrapper that does best-of-N over an inner client."""

    def __init__(
        self,
        inner: _LikeLLMClient,
        *,
        n: int = 4,
        temperatures: list[float] | None = None,
        critic: _LikeLLMClient | None = None,
        ranker: Callable[[str, str | None], float] | None = None,
        # If `True`, log each candidate with its score for downstream
        # failure-learning. The orchestrator pulls this via
        # ``last_siblings``.
        log_siblings: bool = True,
    ) -> None:
        self.inner = inner
        self.critic = critic
        self.n = n
        self.temperatures = temperatures or [0.3, 0.5, 0.7, 0.9]
        self.ranker = ranker or _default_ranker
        self.log_siblings = log_siblings
        self._last_siblings: list[BonCandidate] = []

    @property
    def model_id(self) -> str:
        return f"bon{self.n}({self.inner.model_id})"

    @property
    def last_siblings(self) -> list[BonCandidate]:
        return list(self._last_siblings)

    def chat(self,
             *,
             system: str,
             user: str,
             max_tokens: int = 4096,
             temperature: float | None = None) -> str:
        """Generate ``n`` candidates, score them, return the winner.

        ``temperature`` (if given) is used as the *first* candidate's temp.
        Subsequent candidates cycle through ``self.temperatures``.
        """
        candidates: list[BonCandidate] = []
        temps = list(self.temperatures)
        if temperature is not None:
            temps = [temperature] + temps
        for i in range(self.n):
            t = temps[i % len(temps)]
            try:
                raw = self.inner.chat(system=system,
                                      user=user,
                                      max_tokens=max_tokens,
                                      temperature=t)
            except TypeError:
                # Inner client doesn't accept temperature kwarg.
                raw = self.inner.chat(system=system,
                                      user=user,
                                      max_tokens=max_tokens)
            except Exception as err:
                logger.warning("BoN candidate %d (t=%.2f) failed: %s", i, t,
                               err)
                continue
            cand = BonCandidate(raw=raw,
                                temperature=t,
                                score=self.ranker(raw, user))
            candidates.append(cand)

        if not candidates:
            raise RuntimeError("BoN: all inner calls failed")

        # Optional critic filtering — drop candidates the critic refutes.
        if self.critic is not None:
            from tools.kernel.evolve.mutator.critic import critique_diff
            from tools.kernel.evolve.mutator.diff_applier import extract_diff

            critic_sig = inspect.signature(critique_diff)
            for cand in candidates:
                try:
                    diff = extract_diff(cand.raw)
                except ValueError:
                    cand.score = -float("inf")
                    cand.rejected_reason = "no_fenced_diff"
                    continue
                cand.extracted_diff = diff
                # Only call critic if we'd actually use it.
                if cand.score == -float("inf"):
                    continue
                try:
                    crit_kw = {
                        "diff": diff,
                        "kernel_name": "candidate",
                        "baseline_path": "(see prompt)",
                        "baseline_source": user,
                    }
                    if "max_tokens" in critic_sig.parameters:
                        crit_kw["max_tokens"] = 256
                    verdict = critique_diff(self.critic, **crit_kw)
                    if verdict.verdict == "likely_broken":
                        cand.score = -1e9 + cand.score  # heavily penalize
                        cand.rejected_reason = (
                            f"critic_pre_filter: {verdict.reason[:80]}")
                except Exception as err:
                    logger.warning("BoN critic pre-filter failed: %s", err)

        candidates.sort(key=lambda c: c.score, reverse=True)
        winner = candidates[0]
        if self.log_siblings:
            self._last_siblings = candidates[1:]
        return winner.raw
