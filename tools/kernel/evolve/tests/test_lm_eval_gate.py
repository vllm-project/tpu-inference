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
"""Tests for the lm_eval gate.

We mock the actual lm_eval CLI invocation — these tests exercise the
restore-on-error logic, score extraction, and tolerance gating, which
are the parts that don't need a TPU.
"""

import pytest

from tools.kernel.evolve import lm_eval_gate
from tools.kernel.evolve.lm_eval_gate import (LmEvalResult,
                                              _extract_primary_score,
                                              render_summary, run_lm_eval_gate)

_TINY_DIFF = ("--- a/x/y.py\n"
              "+++ b/x/y.py\n"
              "@@ -1,3 +1,3 @@\n"
              " def f():\n"
              "-    return 1\n"
              "+    return 2\n")


def test_extract_primary_score_prefers_strict_match():
    d = {
        "results": {
            "gsm8k": {
                "exact_match,strict-match": 0.85,
                "exact_match,flexible-extract": 0.87,
            }
        }
    }
    assert _extract_primary_score(d, "gsm8k") == 0.85


def test_extract_primary_score_falls_back_to_acc():
    d = {"results": {"mmlu_pro": {"acc,none": 0.5}}}
    assert _extract_primary_score(d, "mmlu_pro") == 0.5


def test_extract_primary_score_handles_missing():
    assert _extract_primary_score({"results": {}}, "gsm8k") != \
        _extract_primary_score({"results": {}}, "gsm8k")  # NaN != NaN


def test_render_summary_marks_failures():
    results = [
        LmEvalResult(task="gsm8k",
                     score_baseline=0.85,
                     score_patched=0.84,
                     delta=-0.01,
                     sample_size=200,
                     raw_baseline={},
                     raw_patched={}),
        LmEvalResult(task="mmlu_pro",
                     score_baseline=0.50,
                     score_patched=0.501,
                     delta=0.001,
                     sample_size=200,
                     raw_baseline={},
                     raw_patched={}),
    ]
    out = render_summary(results, tolerance=0.005)
    # gsm8k delta=-0.01 > 0.005 → FAIL
    assert "FAIL" in out
    # mmlu_pro delta=+0.001 → PASS
    assert "PASS" in out


def test_run_gate_restores_kernel_on_subprocess_failure(tmp_path, monkeypatch):
    """If lm_eval fails mid-run, the kernel file must be restored."""
    kernel = tmp_path / "x" / "y.py"
    kernel.parent.mkdir()
    original = "def f():\n    return 1\n"
    kernel.write_text(original)
    diff = tmp_path / "d.diff"
    diff.write_text(_TINY_DIFF)

    def _fake_run(**kw):
        raise RuntimeError("lm_eval exploded")

    monkeypatch.setattr(lm_eval_gate, "_run_lm_eval", _fake_run)
    with pytest.raises(RuntimeError, match="exploded"):
        run_lm_eval_gate(model="fake",
                         kernel_path=kernel,
                         diff_path=diff,
                         tasks=["gsm8k"],
                         limit=10,
                         tensor_parallel=1,
                         max_model_len=128,
                         block_size=16,
                         output_dir=tmp_path / "out")
    # KERNEL FILE MUST BE PRISTINE
    assert kernel.read_text() == original


def test_run_gate_restores_kernel_when_patched_pass_fails(
        tmp_path, monkeypatch):
    """Even if baseline succeeds and patched fails, restore."""
    kernel = tmp_path / "x" / "y.py"
    kernel.parent.mkdir()
    original = "def f():\n    return 1\n"
    kernel.write_text(original)
    diff = tmp_path / "d.diff"
    diff.write_text(_TINY_DIFF)
    calls = []

    def _fake_run(**kw):
        calls.append(kw["task"])
        if len(calls) == 1:
            return {"results": {"gsm8k": {"exact_match,strict-match": 0.85}}}
        raise RuntimeError("lm_eval (patched) blew up")

    monkeypatch.setattr(lm_eval_gate, "_run_lm_eval", _fake_run)
    with pytest.raises(RuntimeError, match="blew up"):
        run_lm_eval_gate(model="fake",
                         kernel_path=kernel,
                         diff_path=diff,
                         tasks=["gsm8k"],
                         limit=10,
                         tensor_parallel=1,
                         max_model_len=128,
                         block_size=16,
                         output_dir=tmp_path / "out")
    assert kernel.read_text() == original


def test_run_gate_happy_path_returns_deltas(tmp_path, monkeypatch):
    kernel = tmp_path / "x" / "y.py"
    kernel.parent.mkdir()
    kernel.write_text("def f():\n    return 1\n")
    diff = tmp_path / "d.diff"
    diff.write_text(_TINY_DIFF)

    def _fake_run(**kw):
        if "baseline" in str(kw.get("output_dir", "")):
            return {
                "results": {
                    kw["task"]: {
                        "exact_match,strict-match": 0.85
                    }
                }
            }
        else:
            return {
                "results": {
                    kw["task"]: {
                        "exact_match,strict-match": 0.86
                    }
                }
            }

    monkeypatch.setattr(lm_eval_gate, "_run_lm_eval", _fake_run)
    results = run_lm_eval_gate(model="fake",
                               kernel_path=kernel,
                               diff_path=diff,
                               tasks=["gsm8k"],
                               limit=10,
                               tensor_parallel=1,
                               max_model_len=128,
                               block_size=16,
                               output_dir=tmp_path / "out")
    assert len(results) == 1
    assert results[0].task == "gsm8k"
    assert results[0].score_baseline == pytest.approx(0.85)
    assert results[0].score_patched == pytest.approx(0.86)
    assert results[0].delta == pytest.approx(0.01)
    # kernel was restored
    assert kernel.read_text() == "def f():\n    return 1\n"
