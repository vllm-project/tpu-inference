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
"""Tests for the RLAIF positive-example pool."""

import time

from tools.kernel.evolve.mutator.example_pool import (
    ExamplePool, PositiveExample, render_examples_for_prompt)


def _ex(*,
        kernel="k",
        diff="d1",
        speedup=1.05,
        p_value=0.01,
        hypothesis="h") -> PositiveExample:
    return PositiveExample(kernel=kernel,
                           diff=diff,
                           speedup=speedup,
                           p_value=p_value,
                           hypothesis=hypothesis,
                           added_at=time.time())


def test_pool_persists_across_instances(tmp_path):
    p = tmp_path / "pool.json"
    pool = ExamplePool(persist_path=p)
    assert pool.add(_ex(diff="A"))
    assert pool.add(_ex(diff="B", speedup=1.10))
    pool2 = ExamplePool(persist_path=p)
    assert pool2.size() == 2
    top = pool2.for_kernel("k", top_k=2)
    # Higher speedup is listed first.
    assert top[0].diff == "B"


def test_pool_dedupes_by_diff():
    pool = ExamplePool()
    assert pool.add(_ex(diff="X"))
    assert not pool.add(_ex(diff="X", speedup=1.20))


def test_pool_caps_per_kernel():
    pool = ExamplePool(max_per_kernel=3)
    for i, sp in enumerate([1.01, 1.10, 1.05, 1.03, 1.20]):
        pool.add(_ex(diff=f"d{i}", speedup=sp))
    top = pool.for_kernel("k", top_k=10)
    # 3 highest speedups should survive: 1.20, 1.10, 1.05
    speeds = sorted([e.speedup for e in top], reverse=True)
    assert speeds == [1.20, 1.10, 1.05]


def test_pool_filters_by_min_speedup():
    pool = ExamplePool()
    pool.add(_ex(diff="weak", speedup=1.005))
    pool.add(_ex(diff="strong", speedup=1.10))
    out = pool.for_kernel("k", top_k=10, min_speedup=1.05)
    assert len(out) == 1
    assert out[0].diff == "strong"


def test_render_examples_produces_diff_blocks():
    pool = ExamplePool()
    pool.add(
        _ex(diff="--- a/x\n+++ b/x\n@@ -1,1 +1,1 @@\n-a\n+b\n",
            hypothesis="tweak the constant"))
    text = render_examples_for_prompt(pool.for_kernel("k"))
    assert "```diff" in text
    assert "tweak the constant" in text
    assert "verified" in text.lower()


def test_render_examples_empty_returns_empty():
    assert render_examples_for_prompt([]) == ""


def _tagged(*, kernel, diff, speedup, tags):
    return PositiveExample(kernel=kernel,
                           diff=diff,
                           speedup=speedup,
                           p_value=0.01,
                           hypothesis="h",
                           added_at=time.time(),
                           family_tags=tags)


def test_for_family_pulls_cross_kernel():
    """A family-H win on RPA v3 should surface for MLA v2."""
    pool = ExamplePool()
    pool.add(_tagged(kernel="rpa_v3", diff="diffH1", speedup=1.17, tags=["H"]))
    pool.add(_tagged(kernel="rpa_v3", diff="diffJ1", speedup=1.04, tags=["J"]))
    pool.add(
        _tagged(kernel="fused_moe",
                diff="diffH2",
                speedup=1.10,
                tags=["H", "C"]))
    out = pool.for_family("H", exclude_kernel="mla_v2", top_k=5)
    assert len(out) == 2
    assert out[0].speedup >= out[1].speedup


def test_for_family_respects_exclude_kernel():
    pool = ExamplePool()
    pool.add(_tagged(kernel="rpa_v3", diff="own_H", speedup=1.17, tags=["H"]))
    pool.add(_tagged(kernel="mla_v2", diff="other_H", speedup=1.05,
                     tags=["H"]))
    out = pool.for_family("H", exclude_kernel="rpa_v3")
    # Only mla_v2 — rpa_v3's own example is excluded.
    assert len(out) == 1
    assert out[0].kernel == "mla_v2"


def test_for_family_skips_untagged_examples():
    pool = ExamplePool()
    pool.add(_ex(diff="untagged", speedup=1.20))
    pool.add(_tagged(kernel="k", diff="tagged", speedup=1.05, tags=["A"]))
    out = pool.for_family("A")
    assert len(out) == 1
    assert out[0].diff == "tagged"


def test_for_family_filters_by_min_speedup():
    pool = ExamplePool()
    pool.add(_tagged(kernel="k1", diff="weak", speedup=1.01, tags=["H"]))
    pool.add(_tagged(kernel="k1", diff="strong", speedup=1.10, tags=["H"]))
    out = pool.for_family("H", min_speedup=1.05)
    assert len(out) == 1
    assert out[0].diff == "strong"


def test_family_tags_roundtrip_through_json(tmp_path):
    p = tmp_path / "pool.json"
    pool = ExamplePool(persist_path=p)
    pool.add(_tagged(kernel="k", diff="d1", speedup=1.10, tags=["H", "J"]))
    pool2 = ExamplePool(persist_path=p)
    out = pool2.for_kernel("k")
    assert out[0].family_tags == ["H", "J"]
