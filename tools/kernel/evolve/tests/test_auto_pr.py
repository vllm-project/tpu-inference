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
"""Tests for ``tools.kernel.evolve.auto_pr``."""

import json
import subprocess

import pytest

from tools.kernel.evolve.auto_pr import (CrossShapeRow, Evidence,
                                         LmEvalEvidence, StatsEvidence,
                                         emit_pr_branch, render_pr_body)

_TINY_DIFF = ("--- a/x/y.py\n"
              "+++ b/x/y.py\n"
              "@@ -1,3 +1,3 @@\n"
              " def f():\n"
              "-    return 1\n"
              "+    return 2\n")


def _stats(speedup=1.05, p=0.01):
    return StatsEvidence(label_a="baseline",
                         label_b="patched",
                         mean_a_ns=500_000.0,
                         mean_b_ns=476_190.0,
                         speedup=speedup,
                         p_value=p,
                         cohens_d=-1.2,
                         ci_low_ns=-30_000,
                         ci_high_ns=-10_000,
                         n=8)


def _cross(name, sp, p, dir_):
    return CrossShapeRow(name=name,
                         description=f"{name} desc",
                         speedup=sp,
                         p_value=p,
                         direction=dir_,
                         baseline_p50_us=500.0,
                         candidate_p50_us=476.0)


def test_render_pr_body_contains_required_sections():
    ev = Evidence(kernel="my_kernel",
                  hypothesis="be faster",
                  diff_text=_TINY_DIFF,
                  stats=_stats())
    body = render_pr_body(ev)
    assert "## Summary" in body
    assert "### Hypothesis" in body
    assert "## Stats-bench" in body
    assert "1.0500×" in body
    assert "p < 0.05" in body or "statistically significant" in body
    assert "## Test plan" in body
    assert "## Auto-generation notes" in body


def test_render_pr_body_marks_cross_shape_regressions():
    ev = Evidence(
        kernel="my_kernel",
        hypothesis="hopes and dreams",
        diff_text=_TINY_DIFF,
        stats=_stats(),
        cross_shape=[
            _cross("good", 1.10, 0.001, "win"),
            _cross("bad", 0.90, 0.001, "regress"),
            _cross("nope", 1.00, 0.5, "tie"),
        ],
    )
    body = render_pr_body(ev)
    assert "Cross-shape validation (1 wins / 1 regressions / 1 ties" in body
    assert "REGRESS" in body
    assert "WIN" in body
    # Test plan should reflect the failure — cross-shape NOT [x]
    assert "[ ] Cross-shape" in body
    assert "GATE FAILURE" in body


def test_render_pr_body_passes_gate_when_all_good():
    ev = Evidence(
        kernel="k",
        hypothesis="hope",
        diff_text=_TINY_DIFF,
        stats=_stats(speedup=1.05, p=0.01),
        cross_shape=[
            _cross("a", 1.10, 0.001, "win"),
            _cross("b", 1.05, 0.01, "win")
        ],
    )
    body = render_pr_body(ev)
    # All passing — no GATE FAILURE block
    assert "GATE FAILURE" not in body
    assert "[x] Cross-shape" in body
    assert "[x] Paired t-test" in body


def test_render_pr_body_marks_stats_fail():
    ev = Evidence(
        kernel="k",
        hypothesis="hope",
        diff_text=_TINY_DIFF,
        stats=_stats(speedup=1.001, p=0.3),  # not significant
    )
    body = render_pr_body(ev)
    assert "[ ] Paired t-test" in body
    assert "GATE FAILURE" in body


def test_render_pr_body_includes_lm_eval_fail_tag():
    ev = Evidence(
        kernel="x",
        hypothesis="h",
        diff_text=_TINY_DIFF,
        stats=_stats(),
        lm_eval=[
            LmEvalEvidence(task="gsm8k",
                           baseline_score=0.85,
                           patched_score=0.83,
                           delta=-0.02,
                           limit=100)
        ],
    )
    body = render_pr_body(ev)
    assert "gsm8k" in body
    assert "(FAIL)" in body  # |delta|=0.02 > 0.005 threshold


def test_evidence_from_paths_roundtrip(tmp_path):
    diff = tmp_path / "x.diff"
    diff.write_text(_TINY_DIFF)
    stats = tmp_path / "stats.json"
    stats.write_text(
        json.dumps({
            "label_a": "a",
            "label_b": "b",
            "mean_a_ns": 1000,
            "mean_b_ns": 900,
            "speedup": 1.111,
            "p_value": 0.01,
            "cohens_d": -0.9,
            "ci_low_ns": -50,
            "ci_high_ns": -10,
            "n": 6,
        }))
    cs = tmp_path / "cs.json"
    cs.write_text(
        json.dumps([
            {
                "name": "s1",
                "description": "d",
                "speedup": 1.1,
                "p_value": 0.01,
                "direction": "win",
                "baseline_p50_us": 1.0,
                "candidate_p50_us": 0.9
            },
        ]))
    ev = Evidence.from_paths(
        kernel="kern",
        hypothesis="hop",
        diff_path=diff,
        stats_json=stats,
        cross_shape_json=cs,
    )
    assert ev.stats.speedup == pytest.approx(1.111)
    assert len(ev.cross_shape) == 1
    assert ev.cross_shape[0].name == "s1"


def test_dry_run_does_not_touch_git(tmp_path):
    """Dry run returns the body but doesn't shell out to git."""
    target = tmp_path / "x" / "y.py"
    target.parent.mkdir()
    target.write_text("def f():\n    return 1\n")
    ev = Evidence(kernel="kern",
                  hypothesis="hope",
                  diff_text=_TINY_DIFF,
                  stats=_stats())
    result = emit_pr_branch(evidence=ev, repo_root=tmp_path, dry_run=True)
    assert result["dry_run"] is True
    assert "kern" in result["branch"]
    assert "Stats-bench" in result["pr_body"]


def test_emit_pr_branch_in_real_git_repo(tmp_path):
    """End-to-end: init a git repo, run emit_pr_branch, verify commit."""
    repo = tmp_path
    subprocess.run(["git", "init", "-q", "-b", "main"], cwd=repo, check=True)
    target = repo / "x" / "y.py"
    target.parent.mkdir()
    target.write_text("def f():\n    return 1\n")
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run([
        "git", "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-q",
        "-m", "init"
    ],
                   cwd=repo,
                   check=True)
    ev = Evidence(kernel="kern",
                  hypothesis="hope",
                  diff_text=_TINY_DIFF,
                  stats=_stats())
    result = emit_pr_branch(
        evidence=ev,
        repo_root=repo,
        branch_prefix="evolve-test",
        push=False,
        open_pr=False,
        dry_run=False,
    )
    assert "commit_sha" in result
    branch_name = result["branch"]
    assert branch_name.startswith("evolve-test/kern-")
    # File should have been mutated
    assert "return 2" in target.read_text()
    # Branch checkout
    branch_proc = subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"],
                                 cwd=repo,
                                 capture_output=True,
                                 text=True,
                                 check=True)
    assert branch_proc.stdout.strip() == branch_name


def test_emit_pr_branch_refuses_dirty_tree(tmp_path):
    """If working tree has uncommitted changes, refuse to branch."""
    repo = tmp_path
    subprocess.run(["git", "init", "-q", "-b", "main"], cwd=repo, check=True)
    (repo / "init.txt").write_text("hi")
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run([
        "git", "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-q",
        "-m", "init"
    ],
                   cwd=repo,
                   check=True)
    (repo / "x" / "y.py").parent.mkdir()
    (repo / "x" / "y.py").write_text("def f():\n    return 1\n")
    # Stage but don't commit — dirty tree
    subprocess.run(["git", "add", "x/y.py"], cwd=repo, check=True)
    subprocess.run([
        "git", "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-q",
        "-m", "stage"
    ],
                   cwd=repo,
                   check=True)
    # Now make a non-staged change to simulate "dirty"
    (repo / "x" / "y.py").write_text("def f():\n    return 99\n")
    ev = Evidence(kernel="kern",
                  hypothesis="hope",
                  diff_text=_TINY_DIFF,
                  stats=_stats())
    with pytest.raises(RuntimeError, match="uncommitted changes"):
        emit_pr_branch(evidence=ev, repo_root=repo, dry_run=False)
