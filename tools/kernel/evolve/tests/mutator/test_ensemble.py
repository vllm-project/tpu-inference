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
"""Tests for ``EnsembleClient``."""

import pytest

from tools.kernel.evolve.mutator.bon import BestOfNMutator
from tools.kernel.evolve.mutator.ensemble import EnsembleClient
from tools.kernel.evolve.mutator.llm_client import StubClient

_GOOD_DIFF = ("Hypothesis: tiny\n```diff\n--- a/x\n+++ b/x\n"
              "@@ -1,1 +1,1 @@\n-a\n+b\n```\n")


class _NamedStub(StubClient):

    def __init__(self, name: str, responses: list[str]) -> None:
        super().__init__(responses)
        self._name = name

    @property
    def model_id(self) -> str:
        return f"stub:{self._name}"


def test_ensemble_round_robin_alternates():
    a = _NamedStub("a", [_GOOD_DIFF])
    b = _NamedStub("b", [_GOOD_DIFF])
    e = EnsembleClient([a, b], strategy="round_robin")
    for _ in range(4):
        e.chat(system="s", user="u")
    assert e.call_counts == [2, 2]


def test_ensemble_weighted_proportions():
    a = _NamedStub("a", [_GOOD_DIFF])
    b = _NamedStub("b", [_GOOD_DIFF])
    e = EnsembleClient([a, b], strategy="weighted", weights=[3.0, 1.0])
    # 400 calls; expect ~300 to a, ~100 to b
    for _ in range(400):
        e.chat(system="s", user="u")
    # Allow small drift from rounding in the tick schedule
    assert abs(e.call_counts[0] - 300) < 10
    assert abs(e.call_counts[1] - 100) < 10


def test_ensemble_model_id_lists_all_members():
    a = _NamedStub("alpha", [_GOOD_DIFF])
    b = _NamedStub("beta", [_GOOD_DIFF])
    e = EnsembleClient([a, b])
    mid = e.model_id
    assert "alpha" in mid
    assert "beta" in mid
    assert "round_robin" in mid


def test_ensemble_inside_bon_gets_diversity():
    """BoN over an ensemble should spread calls across inner clients."""
    a = _NamedStub("a", [_GOOD_DIFF])
    b = _NamedStub("b", [_GOOD_DIFF])
    e = EnsembleClient([a, b])
    bon = BestOfNMutator(e, n=4)
    bon.chat(system="s", user="u")
    # 4 candidates → 2 per client under round-robin.
    assert e.call_counts == [2, 2]


def test_ensemble_rejects_zero_clients():
    with pytest.raises(ValueError, match="at least one"):
        EnsembleClient([])


def test_ensemble_rejects_bad_weights():
    a = _NamedStub("a", [_GOOD_DIFF])
    b = _NamedStub("b", [_GOOD_DIFF])
    with pytest.raises(ValueError, match="one weight per client"):
        EnsembleClient([a, b], strategy="weighted", weights=[1.0])
    with pytest.raises(ValueError, match="weights must be positive"):
        EnsembleClient([a, b], strategy="weighted", weights=[1.0, -1.0])


def test_ensemble_last_used_tracks_caller():
    a = _NamedStub("a", [_GOOD_DIFF])
    b = _NamedStub("b", [_GOOD_DIFF])
    e = EnsembleClient([a, b])
    e.chat(system="s", user="u")
    assert e.last_used is a
    e.chat(system="s", user="u")
    assert e.last_used is b


def test_ensemble_falls_back_when_inner_rejects_temperature():
    """Inner client without temperature kwarg should still route correctly."""

    class _OldClient:
        model_id = "old"

        def chat(self, *, system, user, max_tokens=4096):
            return _GOOD_DIFF

    e = EnsembleClient([_OldClient()])
    out = e.chat(system="s", user="u", temperature=0.5)
    assert _GOOD_DIFF in out
