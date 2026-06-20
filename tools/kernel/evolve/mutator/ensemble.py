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
"""Multi-LLM ensemble client.

Different model families exhibit different mutation styles: Opus 4.8 tends
toward sophisticated structural rewrites; Opus 4.7 is more conservative;
Sonnet 4.6 is faster and explores the prompt space differently. An ensemble
that rotates across them gives the BoN sampler genuine policy diversity
(not just temperature diversity within one policy).

Drop this between ``BestOfNMutator`` and the underlying Vertex/Anthropic
clients:

    ensemble = EnsembleClient([
        VertexAnthropicClient(model="claude-opus-4-8"),
        VertexAnthropicClient(model="claude-opus-4-7"),
    ])
    mutator = BestOfNMutator(ensemble, n=4)

The ensemble preserves the ``LLMClient`` Protocol, so it works wherever a
single client did.
"""

from __future__ import annotations

import itertools
import logging
from typing import Iterable, Protocol

logger = logging.getLogger(__name__)


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


class EnsembleClient:
    """Round-robin (or weighted) wrapper over multiple LLM clients.

    Each ``chat()`` call picks the next inner client per the configured
    strategy. ``last_used`` is exposed so callers (e.g. orchestrator
    telemetry) can log which model produced each candidate.

    Strategies:
    * ``round_robin`` — strictly cycle through clients in order. Best for
      A/B comparing models when you want equal samples from each.
    * ``weighted`` — emit clients in proportion to their weight. Best when
      one model is known stronger but you still want diversity.
    """

    def __init__(
        self,
        clients: Iterable[_LikeLLMClient],
        *,
        strategy: str = "round_robin",
        weights: list[float] | None = None,
    ) -> None:
        self.clients = list(clients)
        if not self.clients:
            raise ValueError(
                "EnsembleClient: at least one inner client required")
        if strategy not in {"round_robin", "weighted"}:
            raise ValueError(f"EnsembleClient: unknown strategy {strategy!r}")
        self.strategy = strategy
        if strategy == "weighted":
            if weights is None or len(weights) != len(self.clients):
                raise ValueError(
                    "EnsembleClient: weighted strategy needs one weight per "
                    "client")
            if any(w <= 0 for w in weights):
                raise ValueError("EnsembleClient: weights must be positive")
            # Build an integer-tick schedule so weights of e.g. [2,1,1] produce
            # AABC AABC AABC… across calls.
            denom_factor = 100
            ticks: list[int] = []
            for i, w in enumerate(weights):
                ticks.extend([i] * max(1, int(round(w * denom_factor))))
            self._tick_schedule: list[int] = ticks
            self._tick_iter = itertools.cycle(self._tick_schedule)
        else:
            self._tick_schedule = []
            self._tick_iter = itertools.cycle(range(len(self.clients)))
        self._last_used: _LikeLLMClient | None = None
        self._call_counts = [0] * len(self.clients)

    @property
    def model_id(self) -> str:
        names = "+".join(c.model_id for c in self.clients)
        return f"ensemble({self.strategy}:{names})"

    @property
    def last_used(self) -> _LikeLLMClient | None:
        return self._last_used

    @property
    def call_counts(self) -> list[int]:
        """Per-client cumulative call counter. Indexed by client position."""
        return list(self._call_counts)

    def chat(
        self,
        *,
        system: str,
        user: str,
        max_tokens: int = 4096,
        temperature: float | None = None,
    ) -> str:
        idx = next(self._tick_iter)
        client = self.clients[idx]
        self._last_used = client
        self._call_counts[idx] += 1
        try:
            return client.chat(
                system=system,
                user=user,
                max_tokens=max_tokens,
                temperature=temperature,
            )
        except TypeError:
            # Inner doesn't accept temperature — pass through without it.
            return client.chat(
                system=system,
                user=user,
                max_tokens=max_tokens,
            )
