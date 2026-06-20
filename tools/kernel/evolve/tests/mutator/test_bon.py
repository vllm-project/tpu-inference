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
"""Tests for ``BestOfNMutator``."""

from tools.kernel.evolve.mutator.bon import BestOfNMutator
from tools.kernel.evolve.mutator.llm_client import StubClient

_GOOD_DIFF = (
    "Hypothesis: tiny\n```diff\n--- a/x\n+++ b/x\n@@ -1,1 +1,1 @@\n-a\n+b\n```\n"
)
_BAD_NO_FENCE = "I have no diff for you."
_HUGE_DIFF = ("Hypothesis: long\n```diff\n" +
              "--- a/x\n+++ b/x\n@@ -1,1 +1,1 @@\n-a\n+b\n" +
              "@@ -2,1 +2,1 @@\n-x\n+y\n" * 20 + "```\n")


def test_bon_returns_one_of_n_proposals():
    inner = StubClient([_GOOD_DIFF, _BAD_NO_FENCE, _HUGE_DIFF])
    bon = BestOfNMutator(inner, n=3)
    out = bon.chat(system="s", user="u")
    # Should pick the short well-formed diff over the huge multi-hunk one.
    assert "```diff" in out
    assert "Hypothesis: tiny" in out


def test_bon_logs_siblings():
    inner = StubClient([_GOOD_DIFF, _BAD_NO_FENCE, _HUGE_DIFF])
    bon = BestOfNMutator(inner, n=3, log_siblings=True)
    bon.chat(system="s", user="u")
    sibs = bon.last_siblings
    assert len(sibs) == 2  # winner is removed; siblings are the rest


def test_bon_handles_n_equals_one():
    inner = StubClient([_GOOD_DIFF])
    bon = BestOfNMutator(inner, n=1)
    out = bon.chat(system="s", user="u")
    assert "```diff" in out


def test_bon_cycles_through_temperatures():
    calls = []

    class _SpyClient:
        model_id = "spy"

        def chat(self, *, system, user, max_tokens=4096, temperature=None):
            calls.append(temperature)
            return _GOOD_DIFF

    bon = BestOfNMutator(_SpyClient(), n=4, temperatures=[0.1, 0.5, 0.9])
    bon.chat(system="s", user="u")
    # First 3 calls cycle through; 4th wraps to index 0.
    assert calls == [0.1, 0.5, 0.9, 0.1]


def test_bon_falls_back_when_inner_rejects_temperature_kwarg():
    """Inner clients that don't take ``temperature`` should still work."""

    class _OldClient:
        model_id = "old"

        def chat(self, *, system, user, max_tokens=4096):
            return _GOOD_DIFF

    bon = BestOfNMutator(_OldClient(), n=2)
    out = bon.chat(system="s", user="u")
    assert "```diff" in out


def test_bon_model_id_includes_inner():
    inner = StubClient(["x"])
    bon = BestOfNMutator(inner, n=4)
    assert "bon4" in bon.model_id
    assert "stub" in bon.model_id
