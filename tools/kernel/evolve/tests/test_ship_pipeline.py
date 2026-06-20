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
"""Tests for the ship pipeline orchestrator — mock TPU-bound gates."""

import pytest

from tools.kernel.evolve import ship_pipeline
from tools.kernel.evolve.ship_pipeline import (GateOutcome, PipelineReport,
                                               run_pipeline)

_TINY_DIFF = ("--- a/x/y.py\n"
              "+++ b/x/y.py\n"
              "@@ -1,3 +1,3 @@\n"
              " def f():\n"
              "-    return 1\n"
              "+    return 2\n")


def _passing(name):
    return GateOutcome(name=name, passed=True, summary=f"{name} ok")


def _failing(name):
    return GateOutcome(name=name, passed=False, summary=f"{name} fail")


def _skipped(name):
    return GateOutcome(name=name, passed=True, summary="skipped", skipped=True)


def _setup_paths(tmp_path):
    diff = tmp_path / "x.diff"
    diff.write_text(_TINY_DIFF)
    kernel = tmp_path / "y.py"
    kernel.write_text("def f():\n    return 1\n")
    return diff, kernel


def _stub_all_gates(monkeypatch, *, stats=True, cross=True, lm=True, e2e=True):

    def _stub(name, ok):

        def _f(**kw):
            out = kw.get("out_json")
            if out:
                out.write_text("{}")
            return _passing(name) if ok else _failing(name)

        return _f

    monkeypatch.setattr(ship_pipeline, "_stats_gate", _stub("stats", stats))
    monkeypatch.setattr(ship_pipeline, "_cross_shape_gate",
                        _stub("cross_shape", cross))
    monkeypatch.setattr(ship_pipeline, "_lm_eval_gate", _stub("lm_eval", lm))
    monkeypatch.setattr(ship_pipeline, "_e2e_gate", _stub("e2e", e2e))


def test_pipeline_all_passing(tmp_path, monkeypatch):
    diff, kernel = _setup_paths(tmp_path)
    _stub_all_gates(monkeypatch)
    report = run_pipeline(
        diff_path=diff,
        kernel_path=kernel,
        host_factory=lambda: object(),
        kernel_name="kern",
        hypothesis="hope",
        model="fake",
        lm_eval_tasks=["gsm8k"],
        lm_eval_limit=10,
        tensor_parallel=1,
        max_model_len=128,
        lm_eval_block_size=16,
        max_tokens=8,
        num_prompts=4,
        work_dir=tmp_path / "w",
        emit_pr_on_pass=False,
    )
    assert report.overall_pass is True
    assert len(report.gates) == 4
    assert all(g.passed for g in report.gates)


def test_pipeline_fails_when_one_gate_fails(tmp_path, monkeypatch):
    diff, kernel = _setup_paths(tmp_path)
    _stub_all_gates(monkeypatch, lm=False)
    report = run_pipeline(
        diff_path=diff,
        kernel_path=kernel,
        host_factory=lambda: object(),
        kernel_name="kern",
        hypothesis="hope",
        model="fake",
        lm_eval_tasks=["gsm8k"],
        lm_eval_limit=10,
        tensor_parallel=1,
        max_model_len=128,
        lm_eval_block_size=16,
        max_tokens=8,
        num_prompts=4,
        work_dir=tmp_path / "w",
        emit_pr_on_pass=False,
    )
    assert report.overall_pass is False
    failed_names = [g.name for g in report.gates if not g.passed]
    assert "lm_eval" in failed_names


def test_pipeline_skips_when_flagged(tmp_path, monkeypatch):
    diff, kernel = _setup_paths(tmp_path)
    _stub_all_gates(monkeypatch)
    report = run_pipeline(
        diff_path=diff,
        kernel_path=kernel,
        host_factory=lambda: object(),
        kernel_name="kern",
        hypothesis="hope",
        model="fake",
        lm_eval_tasks=["gsm8k"],
        lm_eval_limit=10,
        tensor_parallel=1,
        max_model_len=128,
        lm_eval_block_size=16,
        max_tokens=8,
        num_prompts=4,
        work_dir=tmp_path / "w",
        emit_pr_on_pass=False,
        skip_lm_eval=True,
        skip_e2e=True,
        skip_cross_shape=True,
    )
    assert report.overall_pass is True
    skipped = [g.name for g in report.gates if g.skipped]
    assert set(skipped) == {"cross_shape", "lm_eval", "e2e"}


def test_pipeline_emits_pr_artifact_on_pass(tmp_path, monkeypatch):
    diff, kernel = _setup_paths(tmp_path)
    _stub_all_gates(monkeypatch)
    # Track auto_pr.emit_pr_branch invocation
    captured = {}

    def _fake_emit(*, evidence, repo_root, branch_prefix, push, open_pr,
                   dry_run):
        captured["evidence"] = evidence
        return {
            "branch": f"{branch_prefix}/test-abc",
            "dry_run": True,
            "pr_body": "fake body"
        }

    monkeypatch.setattr("tools.kernel.evolve.auto_pr.emit_pr_branch",
                        _fake_emit)
    report = run_pipeline(
        diff_path=diff,
        kernel_path=kernel,
        host_factory=lambda: object(),
        kernel_name="kern",
        hypothesis="hope",
        model="fake",
        lm_eval_tasks=["gsm8k"],
        lm_eval_limit=10,
        tensor_parallel=1,
        max_model_len=128,
        lm_eval_block_size=16,
        max_tokens=8,
        num_prompts=4,
        work_dir=tmp_path / "w",
        dry_run_pr=True,
        emit_pr_on_pass=True,
    )
    assert report.overall_pass is True
    assert report.pr_artifact is not None
    assert report.pr_artifact["dry_run"] is True
    assert captured["evidence"].kernel == "kern"


def test_pipeline_does_not_emit_pr_on_fail(tmp_path, monkeypatch):
    diff, kernel = _setup_paths(tmp_path)
    _stub_all_gates(monkeypatch, stats=False)
    monkeypatch.setattr("tools.kernel.evolve.auto_pr.emit_pr_branch",
                        lambda **kw: pytest.fail("should not emit PR on fail"))
    report = run_pipeline(
        diff_path=diff,
        kernel_path=kernel,
        host_factory=lambda: object(),
        kernel_name="kern",
        hypothesis="hope",
        model="fake",
        lm_eval_tasks=["gsm8k"],
        lm_eval_limit=10,
        tensor_parallel=1,
        max_model_len=128,
        lm_eval_block_size=16,
        max_tokens=8,
        num_prompts=4,
        work_dir=tmp_path / "w",
        emit_pr_on_pass=True,
    )
    assert report.overall_pass is False
    assert report.pr_artifact is None


def test_report_to_dict_round_trip(tmp_path):
    r = PipelineReport(
        diff_path=tmp_path / "d.diff",
        gates=[_passing("a"), _failing("b"),
               _skipped("c")],
        overall_pass=False,
        wall_sec=42.0,
    )
    d = r.to_dict()
    assert d["overall_pass"] is False
    assert len(d["gates"]) == 3
    assert d["gates"][0]["passed"] is True
    assert d["gates"][1]["passed"] is False
    assert d["gates"][2]["skipped"] is True
