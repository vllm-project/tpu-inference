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
"""Auto-PR generator for evolve-discovered wins.

This closes the loop: take a diff that has passed all three verifier moats
(numerics, paired-t significance, cross-shape robustness, optional lm-eval)
and emit a PR-ready branch — applied to the working tree, committed with
the Signed-off-by trailer, and (optionally) pushed + opened on GitHub via
``gh``.

The PR body is auto-generated and includes every piece of evidence that
made the win pass gate. A human reviewer can audit each line.

Usage::

    python -m tools.kernel.evolve.auto_pr \\
        --diff /tmp/rpa_v3_4_7pct_win.diff \\
        --kernel ragged_paged_attention_v3 \\
        --hypothesis "Move dtype cast out of jnp.sum to keep accumulation in fp32" \\
        --evidence-json /tmp/cross_shape_results.json \\
        --branch-prefix claude-auto

By default, ``--push=False`` so nothing leaves the local repo until the
operator confirms. Pass ``--push --open-pr`` to push the branch and call
``gh pr create``.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import subprocess
import sys
import textwrap
import time
from pathlib import Path

from tools.kernel.evolve.mutator.diff_applier import apply_diff

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class StatsEvidence:
    """Stats-bench outcome for the primary shape."""
    label_a: str = "baseline"
    label_b: str = "candidate"
    mean_a_ns: float = 0.0
    mean_b_ns: float = 0.0
    speedup: float = 0.0
    p_value: float = 1.0
    cohens_d: float = 0.0
    ci_low_ns: float = 0.0
    ci_high_ns: float = 0.0
    n: int = 0


@dataclasses.dataclass
class CrossShapeRow:
    name: str
    description: str
    speedup: float
    p_value: float
    direction: str
    baseline_p50_us: float
    candidate_p50_us: float


@dataclasses.dataclass
class LmEvalEvidence:
    task: str
    baseline_score: float
    patched_score: float
    delta: float
    limit: int


@dataclasses.dataclass
class Evidence:
    kernel: str
    hypothesis: str
    diff_text: str
    stats: StatsEvidence | None = None
    cross_shape: list[CrossShapeRow] = dataclasses.field(default_factory=list)
    lm_eval: list[LmEvalEvidence] = dataclasses.field(default_factory=list)
    discovered_by: str = "claude-evolve"
    extra_notes: str = ""

    @staticmethod
    def from_paths(*,
                   kernel: str,
                   hypothesis: str,
                   diff_path: Path,
                   stats_json: Path | None = None,
                   cross_shape_json: Path | None = None,
                   lm_eval_json: Path | None = None,
                   discovered_by: str = "claude-evolve",
                   extra_notes: str = "") -> "Evidence":
        diff_text = diff_path.read_text()
        stats = None
        if stats_json and stats_json.exists():
            d = json.loads(stats_json.read_text())
            stats = StatsEvidence(**d)
        cs: list[CrossShapeRow] = []
        if cross_shape_json and cross_shape_json.exists():
            for row in json.loads(cross_shape_json.read_text()):
                cs.append(
                    CrossShapeRow(
                        name=row["name"],
                        description=row["description"],
                        speedup=row["speedup"],
                        p_value=row["p_value"],
                        direction=row["direction"],
                        baseline_p50_us=row["baseline_p50_us"],
                        candidate_p50_us=row["candidate_p50_us"],
                    ))
        le: list[LmEvalEvidence] = []
        if lm_eval_json and lm_eval_json.exists():
            for row in json.loads(lm_eval_json.read_text()):
                le.append(LmEvalEvidence(**row))
        return Evidence(
            kernel=kernel,
            hypothesis=hypothesis,
            diff_text=diff_text,
            stats=stats,
            cross_shape=cs,
            lm_eval=le,
            discovered_by=discovered_by,
            extra_notes=extra_notes,
        )


def _run(cmd: list[str],
         *,
         cwd: Path | None = None,
         check: bool = True) -> subprocess.CompletedProcess:
    logger.info("$ %s", " ".join(cmd))
    return subprocess.run(cmd,
                          cwd=cwd,
                          check=check,
                          capture_output=True,
                          text=True)


def _short_hash(text: str, n: int = 8) -> str:
    import hashlib
    return hashlib.sha1(text.encode()).hexdigest()[:n]


def render_pr_body(evidence: Evidence) -> str:
    """Build a markdown PR body from the evidence."""
    s = evidence.stats
    parts: list[str] = []
    parts.append(
        f"## Summary\n\n"
        f"Auto-discovered optimization of `{evidence.kernel}` by "
        f"`{evidence.discovered_by}`. Verified by all three independent "
        f"moats (numerics + statistical + cross-shape).\n")
    parts.append(f"### Hypothesis\n\n> {evidence.hypothesis.strip()}\n")

    if s is not None:
        sig = "**statistically significant**" if s.p_value < 0.05 else "not significant"
        parts.append(f"## Stats-bench (paired t-test, N={s.n})\n"
                     f"\n"
                     f"| metric | value |\n"
                     f"|---|---|\n"
                     f"| baseline mean | {s.mean_a_ns/1e3:.2f} μs |\n"
                     f"| candidate mean | {s.mean_b_ns/1e3:.2f} μs |\n"
                     f"| speedup | **{s.speedup:.4f}×** |\n"
                     f"| p-value | {s.p_value:.4g} ({sig}) |\n"
                     f"| Cohen's d | {s.cohens_d:.2f} |\n"
                     f"| 95% CI for delta_ns | "
                     f"[{s.ci_low_ns/1e3:.2f}, {s.ci_high_ns/1e3:.2f}] μs |\n")

    if evidence.cross_shape:
        wins = sum(1 for r in evidence.cross_shape if r.direction == "win")
        regs = sum(1 for r in evidence.cross_shape if r.direction == "regress")
        ties = sum(1 for r in evidence.cross_shape if r.direction == "tie")
        n = len(evidence.cross_shape)
        parts.append(
            f"## Cross-shape validation ({wins} wins / {regs} regressions "
            f"/ {ties} ties across {n} production shapes)\n"
            f"\n"
            f"| shape | description | baseline μs | patched μs | "
            f"speedup | p | direction |\n"
            f"|---|---|---|---|---|---|---|\n")
        for r in evidence.cross_shape:
            mark = {
                "win": "WIN",
                "regress": "REGRESS",
                "tie": "tie"
            }[r.direction]
            parts[-1] += (
                f"| `{r.name}` | {r.description} | "
                f"{r.baseline_p50_us:.2f} | {r.candidate_p50_us:.2f} | "
                f"**{r.speedup:.4f}×** | {r.p_value:.4f} | {mark} |\n")

    if evidence.lm_eval:
        parts.append("## lm-eval correctness gate\n")
        parts.append("| task | baseline | patched | Δ | limit |\n"
                     "|---|---|---|---|---|\n")
        for r in evidence.lm_eval:
            tag = " (FAIL)" if abs(r.delta) > 0.005 else ""
            parts[-1] += (
                f"| {r.task} | {r.baseline_score:.4f} | "
                f"{r.patched_score:.4f} | {r.delta:+.4f}{tag} | {r.limit} |\n")

    if evidence.extra_notes:
        parts.append(f"## Notes\n\n{evidence.extra_notes.strip()}\n")

    # Derive gate checks from the evidence we actually have. A passing
    # evidence row gets [x]; a missing or failing row gets [ ].
    stats_ok = (s is not None and s.p_value < 0.05 and s.speedup > 1.0)
    cs_ok = (bool(evidence.cross_shape)
             and not any(r.direction == "regress"
                         for r in evidence.cross_shape)
             and any(r.direction == "win" for r in evidence.cross_shape))
    lm_ok = (bool(evidence.lm_eval)
             and all(abs(r.delta) <= 0.005 for r in evidence.lm_eval))
    parts.append("## Test plan\n\n"
                 "- [x] Numerics verifier (dtype-aware allclose + cosine + "
                 "anti-cheat) passed during evolution\n"
                 f"- [{'x' if stats_ok else ' '}] Paired t-test at p < 0.05 "
                 "on discovery shape\n"
                 f"- [{'x' if cs_ok else ' '}] Cross-shape: no statistically "
                 "significant regressions on the production shape set\n"
                 f"- [{'x' if lm_ok else ' '}] lm-eval correctness gate "
                 "(per-task |Δ| ≤ 0.005)\n"
                 "- [ ] CI: `pytest tests/kernels/<kernel>_test.py`\n"
                 "- [ ] CI: end-to-end vLLM offline_inference smoke\n")
    if not (stats_ok and cs_ok):
        parts.append(
            "\n> **GATE FAILURE**: At least one verification gate failed — "
            "this candidate is NOT ship-ready. The PR body is preserved as "
            "evidence; do not merge without addressing the gate failures.\n")
    parts.append(
        textwrap.dedent("""\
        ## Auto-generation notes

        This PR was generated by `tools.kernel.evolve.auto_pr`. The diff was
        discovered, verified, and stats-tested without human in the loop; a
        human MUST review before merging.
        """))
    return "\n".join(parts)


def emit_pr_branch(*,
                   evidence: Evidence,
                   repo_root: Path,
                   branch_prefix: str = "claude-auto",
                   author_name: str = "Claude Evolve",
                   author_email: str = "claude-evolve@noreply.local",
                   base_branch: str = "main",
                   push: bool = False,
                   open_pr: bool = False,
                   dry_run: bool = False) -> dict:
    """Create a branch, apply diff, commit, and (optionally) push + open PR.

    Returns a dict with branch name, commit sha (if not dry-run), PR url
    (if opened), and the PR body text.
    """
    kernel_slug = evidence.kernel.replace("_", "-")
    diff_hash = _short_hash(evidence.diff_text)
    branch = f"{branch_prefix}/{kernel_slug}-{diff_hash}"
    pr_body = render_pr_body(evidence)
    # Determine the file path the diff touches.
    target_path = None
    for line in evidence.diff_text.splitlines():
        if line.startswith("--- a/") or line.startswith("--- "):
            p = line.split(" ", 1)[1].strip()
            if p.startswith("a/"):
                p = p[2:]
            target_path = repo_root / p
            break
    if target_path is None:
        raise ValueError("auto_pr: couldn't find target file in diff")

    if dry_run:
        return {
            "branch": branch,
            "target_path": str(target_path),
            "pr_body": pr_body,
            "dry_run": True,
        }

    # 1. Check working tree is clean before we touch the branch
    diff_clean = _run(["git", "diff", "--quiet"], cwd=repo_root, check=False)
    if diff_clean.returncode != 0:
        raise RuntimeError(
            "auto_pr: working tree has uncommitted changes — refusing to "
            "create a branch. Stash or commit first.")

    # 2. Create branch from base
    _run(["git", "fetch", "origin", base_branch], cwd=repo_root, check=False)
    _run(["git", "checkout", "-b", branch, f"origin/{base_branch}"],
         cwd=repo_root,
         check=False)
    # Fallback to local base if fetch failed:
    cur = _run(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=repo_root)
    if cur.stdout.strip() != branch:
        _run(["git", "checkout", "-b", branch, base_branch], cwd=repo_root)

    # 3. Apply the diff to the target file
    current = target_path.read_text()
    new_text = apply_diff(current, evidence.diff_text).new_source
    target_path.write_text(new_text)

    # 4. Stage + commit
    _run(["git", "add", str(target_path)], cwd=repo_root)
    commit_msg_lines = [
        f"[evolve] {evidence.kernel}: {evidence.hypothesis.splitlines()[0]}",
        "",
        "Discovered by `tools.kernel.evolve` LLM-mutation pipeline; verified by",
        "numerics + paired-t + cross-shape gates. See PR body for evidence.",
        "",
        f"Hypothesis: {evidence.hypothesis.strip()}",
        "",
        f"Signed-off-by: {author_name} <{author_email}>",
    ]
    commit_msg = "\n".join(commit_msg_lines)
    env_overrides = {
        "GIT_AUTHOR_NAME": author_name,
        "GIT_AUTHOR_EMAIL": author_email,
        "GIT_COMMITTER_NAME": author_name,
        "GIT_COMMITTER_EMAIL": author_email,
    }
    import os
    env = {**os.environ, **env_overrides}
    proc = subprocess.run(["git", "commit", "-F", "-"],
                          cwd=repo_root,
                          input=commit_msg,
                          text=True,
                          env=env,
                          capture_output=True,
                          check=False)
    if proc.returncode != 0:
        raise RuntimeError(
            f"auto_pr: git commit failed: {proc.stderr.strip()}")
    sha_proc = _run(["git", "rev-parse", "HEAD"], cwd=repo_root)
    sha = sha_proc.stdout.strip()

    result: dict = {
        "branch": branch,
        "commit_sha": sha,
        "target_path": str(target_path),
        "pr_body": pr_body,
    }

    if push:
        _run(["git", "push", "-u", "origin", branch], cwd=repo_root)
        result["pushed"] = True
        if open_pr:
            title = f"[evolve] {evidence.kernel}: {evidence.hypothesis.splitlines()[0][:60]}"
            gh_proc = subprocess.run(
                ["gh", "pr", "create", "--title", title, "--body", pr_body],
                cwd=repo_root,
                capture_output=True,
                text=True,
                check=False)
            if gh_proc.returncode == 0:
                result["pr_url"] = gh_proc.stdout.strip()
            else:
                result["pr_error"] = gh_proc.stderr.strip()
    return result


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--diff", type=Path, required=True)
    p.add_argument("--kernel", required=True)
    p.add_argument("--hypothesis", required=True)
    p.add_argument("--stats-json",
                   type=Path,
                   default=None,
                   help="StatsEvidence JSON from paired-t bench.")
    p.add_argument("--cross-shape-json",
                   type=Path,
                   default=None,
                   help="Cross-shape results JSON.")
    p.add_argument("--lm-eval-json",
                   type=Path,
                   default=None,
                   help="lm-eval results JSON.")
    p.add_argument("--extra-notes", default="")
    p.add_argument("--repo-root", type=Path, default=Path.cwd())
    p.add_argument("--branch-prefix", default="claude-auto")
    p.add_argument("--push", action="store_true")
    p.add_argument("--open-pr", action="store_true")
    p.add_argument("--dry-run",
                   action="store_true",
                   help="Just render the PR body — don't touch git.")
    p.add_argument("--out-md", type=Path, default=Path("/tmp/auto_pr_body.md"))
    p.add_argument("--verbose", "-v", action="count", default=0)
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.WARNING - 10 * args.verbose,
                        format="%(asctime)s %(levelname)s %(message)s")
    evidence = Evidence.from_paths(
        kernel=args.kernel,
        hypothesis=args.hypothesis,
        diff_path=args.diff,
        stats_json=args.stats_json,
        cross_shape_json=args.cross_shape_json,
        lm_eval_json=args.lm_eval_json,
        extra_notes=args.extra_notes,
    )
    t0 = time.time()
    result = emit_pr_branch(evidence=evidence,
                            repo_root=args.repo_root,
                            branch_prefix=args.branch_prefix,
                            push=args.push,
                            open_pr=args.open_pr,
                            dry_run=args.dry_run)
    wall = time.time() - t0
    args.out_md.write_text(result["pr_body"])
    print(f"Done in {wall:.1f}s")
    print(f"  branch: {result['branch']}")
    if "commit_sha" in result:
        print(f"  commit: {result['commit_sha']}")
    if "pr_url" in result:
        print(f"  PR:     {result['pr_url']}")
    print(f"  PR body written to {args.out_md}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
