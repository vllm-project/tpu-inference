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
"""Tests for the critic verdict parser."""

from tools.kernel.evolve.mutator.critic import critique_diff
from tools.kernel.evolve.mutator.llm_client import StubClient

_DIFF = ("--- a/f.py\n"
         "+++ b/f.py\n"
         "@@ -1,1 +1,1 @@\n"
         "-x = 1\n"
         "+x = 2\n")


def test_parses_likely_broken_verdict():
    client = StubClient([
        "I think this is wrong because the constant shouldn't change.\n"
        "VERDICT: likely_broken reason: changes the constant semantics."
    ])
    c = critique_diff(client,
                      diff=_DIFF,
                      kernel_name="k",
                      baseline_path="f.py",
                      baseline_source="x = 1\n")
    assert c.verdict == "likely_broken"


def test_parses_likely_correct_verdict():
    client = StubClient(["VERDICT: likely_correct reason: harmless tweak."])
    c = critique_diff(client,
                      diff=_DIFF,
                      kernel_name="k",
                      baseline_path="f.py",
                      baseline_source="x = 1\n")
    assert c.verdict == "likely_correct"


def test_parses_unsure_verdict():
    client = StubClient(["VERDICT: unsure reason: hard to tell."])
    c = critique_diff(client,
                      diff=_DIFF,
                      kernel_name="k",
                      baseline_path="f.py",
                      baseline_source="x = 1\n")
    assert c.verdict == "unsure"


def test_defaults_to_unsure_when_no_verdict_marker():
    client = StubClient(["I don't have a verdict tag."])
    c = critique_diff(client,
                      diff=_DIFF,
                      kernel_name="k",
                      baseline_path="f.py",
                      baseline_source="x = 1\n")
    assert c.verdict == "unsure"


def test_case_insensitive_verdict_match():
    client = StubClient(["verdict: LIKELY_BROKEN. it deletes a guard."])
    c = critique_diff(client,
                      diff=_DIFF,
                      kernel_name="k",
                      baseline_path="f.py",
                      baseline_source="x = 1\n")
    assert c.verdict == "likely_broken"
