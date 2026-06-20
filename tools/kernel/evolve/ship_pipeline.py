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
"""End-to-end ship pipeline for an evolve-discovered kernel diff.

Runs the FULL six-gate verification chain a candidate must pass before it
becomes a PR:

1. **Numerics gate** — already enforced by the verifier during evolution.
2. **Critic gate** — already enforced during evolution (semantic refutation).
3. **Stats-bench** — paired-t bench (N rounds, p < 0.05) on the discovery
   shape. Quantifies the win against noise.
4. **Cross-shape** — bench on the production shape catalog; reject if any
   shape regresses significantly.
5. **lm-eval** — gsm8k/mmlu_pro deltas within tolerance on the target model.
6. **E2E throughput** — vLLM offline_inference; the user-visible number.

If every gate passes, the pipeline calls into ``auto_pr.emit_pr_branch`` to
write a branch + commit + (optional) push + open PR via gh.

This is the "press the button" path. Operators use it after the evolution
loop produces a promising candidate.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class GateOutcome:
    """One stage's pass/fail + evidence."""
    name: str
    passed: bool
    summary: str
    artifact_path: Path | None = None
    skipped: bool = False
    error: str | None = None


@dataclasses.dataclass
class PipelineReport:
    diff_path: Path
    gates: list[GateOutcome]
    overall_pass: bool
    wall_sec: float
    pr_artifact: dict[str, Any] | None = None

    def to_dict(self) -> dict:
        return {
            "diff_path":
            str(self.diff_path),
            "overall_pass":
            self.overall_pass,
            "wall_sec":
            self.wall_sec,
            "gates": [{
                "name":
                g.name,
                "passed":
                g.passed,
                "summary":
                g.summary,
                "artifact_path":
                str(g.artifact_path) if g.artifact_path else None,
                "skipped":
                g.skipped,
                "error":
                g.error,
            } for g in self.gates],
            "pr_artifact":
            self.pr_artifact,
        }


def _stats_gate(*, diff_path: Path, kernel_path: Path, host_factory,
                bench_n: int, warmup: int, iters: int,
                out_json: Path) -> GateOutcome:
    """Paired-t bench on the discovery shape. host_factory() -> RpaV3Host-like."""
    import numpy as np
    from scipy import stats as scistats

    from tools.kernel.evolve.mutator.diff_applier import apply_diff
    from tools.kernel.evolve.worktree import import_candidate_module
    from tools.kernel.tuner.v1.bench.harness import measure

    try:
        host = host_factory()
        src = kernel_path.read_text()
        diff = diff_path.read_text()
        patched = apply_diff(src, diff).new_source

        def _bench(text):
            with import_candidate_module(
                    text, name_hint="ship_pipeline_stats") as mod:
                fn = host.build_kernel_fn(mod)
                return [
                    measure(fn, warmup=warmup, iters=iters).p50_ns
                    for _ in range(bench_n)
                ]

        baseline = _bench(src)
        candidate = _bench(patched)
        diffs = np.array(candidate, float) - np.array(baseline, float)
        if len(diffs) > 1 and diffs.std(ddof=1) > 0:
            _, p_value = scistats.ttest_rel(candidate, baseline)
        else:
            p_value = 1.0
        speedup = float(np.mean(baseline) / np.mean(candidate))
        std_diff = float(diffs.std(ddof=1)) if len(diffs) > 1 else 0.0
        cohens_d = float(diffs.mean() / std_diff) if std_diff > 0 else 0.0
        n = len(diffs)
        if n > 1 and std_diff > 0:
            se = std_diff / np.sqrt(n)
            t_crit = scistats.t.ppf(0.975, df=n - 1)
            ci_low = float(diffs.mean() - t_crit * se)
            ci_high = float(diffs.mean() + t_crit * se)
        else:
            ci_low = ci_high = float(diffs.mean())

        payload = {
            "label_a": "baseline",
            "label_b": "candidate",
            "mean_a_ns": float(np.mean(baseline)),
            "mean_b_ns": float(np.mean(candidate)),
            "speedup": speedup,
            "p_value": float(p_value),
            "cohens_d": cohens_d,
            "ci_low_ns": ci_low,
            "ci_high_ns": ci_high,
            "n": n,
        }
        out_json.write_text(json.dumps(payload, indent=2))
        passed = bool(p_value < 0.05 and speedup > 1.0)
        return GateOutcome(
            name="stats",
            passed=passed,
            summary=(f"speedup={speedup:.4f}× p={p_value:.4g} "
                     f"d={cohens_d:.2f} N={n}"),
            artifact_path=out_json,
        )
    except Exception as e:
        logger.exception("stats gate failed")
        return GateOutcome(name="stats",
                           passed=False,
                           summary=f"error: {e}",
                           error=str(e))


def _cross_shape_gate(*,
                      diff_path: Path,
                      baseline_path: Path,
                      out_json: Path,
                      out_md: Path,
                      bench_n: int,
                      warmup: int,
                      iters: int,
                      allow_regressions: bool = False) -> GateOutcome:
    from tools.kernel.evolve.cross_shape import (PRODUCTION_SHAPES,
                                                 evaluate_shape,
                                                 render_markdown_table)
    try:
        baseline_source = baseline_path.read_text()
        diff_text = diff_path.read_text()
        results = []
        for shape in PRODUCTION_SHAPES:
            r = evaluate_shape(shape,
                               diff_text=diff_text,
                               baseline_source=baseline_source,
                               bench_n=bench_n,
                               warmup=warmup,
                               iters=iters)
            if r is not None:
                results.append(r)
        out_json.write_text(
            json.dumps([r.to_dict() for r in results], indent=2))
        out_md.write_text(render_markdown_table(results))
        regs = sum(1 for r in results if r.direction == "regress")
        wins = sum(1 for r in results if r.direction == "win")
        ties = sum(1 for r in results if r.direction == "tie")
        if allow_regressions:
            passed = wins > 0  # at least one win, regressions tolerated
        else:
            passed = regs == 0 and wins > 0
        return GateOutcome(
            name="cross_shape",
            passed=passed,
            summary=f"{wins} wins / {regs} regressions / {ties} ties "
            f"(across {len(results)} shapes)",
            artifact_path=out_json,
        )
    except Exception as e:
        logger.exception("cross-shape gate failed")
        return GateOutcome(name="cross_shape",
                           passed=False,
                           summary=f"error: {e}",
                           error=str(e))


def _lm_eval_gate(*, diff_path: Path, kernel_path: Path, model: str,
                  tasks: list[str], limit: int, tensor_parallel: int,
                  max_model_len: int, block_size: int, output_dir: Path,
                  out_json: Path, tolerance: float) -> GateOutcome:
    from tools.kernel.evolve.lm_eval_gate import (render_summary,
                                                  run_lm_eval_gate)
    try:
        results = run_lm_eval_gate(model=model,
                                   kernel_path=kernel_path,
                                   diff_path=diff_path,
                                   tasks=tasks,
                                   limit=limit,
                                   tensor_parallel=tensor_parallel,
                                   max_model_len=max_model_len,
                                   block_size=block_size,
                                   output_dir=output_dir,
                                   tolerance=tolerance)
        payload = [{
            "task": r.task,
            "baseline_score": r.score_baseline,
            "patched_score": r.score_patched,
            "delta": r.delta,
            "limit": r.sample_size,
        } for r in results]
        out_json.write_text(json.dumps(payload, indent=2))
        any_fail = any(abs(r.delta) > tolerance for r in results)
        return GateOutcome(
            name="lm_eval",
            passed=not any_fail,
            summary=render_summary(results, tolerance=tolerance),
            artifact_path=out_json,
        )
    except Exception as e:
        logger.exception("lm_eval gate failed")
        return GateOutcome(name="lm_eval",
                           passed=False,
                           summary=f"error: {e}",
                           error=str(e))


def _e2e_gate(*, diff_path: Path, kernel_path: Path, model: str,
              tensor_parallel: int, max_model_len: int, max_tokens: int,
              num_prompts: int, n_trials: int, out_json: Path) -> GateOutcome:
    from tools.kernel.evolve.e2e_benchmark import run_e2e_benchmark
    try:
        cmp = run_e2e_benchmark(model=model,
                                kernel_path=kernel_path,
                                diff_path=diff_path,
                                tensor_parallel=tensor_parallel,
                                max_model_len=max_model_len,
                                max_tokens=max_tokens,
                                num_prompts=num_prompts,
                                n_trials=n_trials)
        payload = {
            "model": model,
            "baseline_mean_tok_per_s": cmp.baseline.mean_tok_per_s,
            "patched_mean_tok_per_s": cmp.patched.mean_tok_per_s,
            "speedup_mean": cmp.speedup_mean,
            "p_value": cmp.p_value,
            "n_trials": cmp.baseline.n_trials,
        }
        out_json.write_text(json.dumps(payload, indent=2))
        passed = bool(cmp.speedup_mean > 1.0 and cmp.p_value < 0.05)
        return GateOutcome(
            name="e2e",
            passed=passed,
            summary=(f"baseline={cmp.baseline.mean_tok_per_s:.2f} tok/s; "
                     f"patched={cmp.patched.mean_tok_per_s:.2f} tok/s; "
                     f"speedup={cmp.speedup_mean:.4f}× "
                     f"p≈{cmp.p_value:.4g}"),
            artifact_path=out_json,
        )
    except Exception as e:
        logger.exception("e2e gate failed")
        return GateOutcome(name="e2e",
                           passed=False,
                           summary=f"error: {e}",
                           error=str(e))


def run_pipeline(
        *,
        diff_path: Path,
        kernel_path: Path,
        host_factory,  # () -> EvolutionHost-shaped
        kernel_name: str,
        hypothesis: str,
        model: str,
        lm_eval_tasks: list[str],
        lm_eval_limit: int,
        tensor_parallel: int,
        max_model_len: int,
        lm_eval_block_size: int,
        max_tokens: int,
        num_prompts: int,
        stats_bench_n: int = 8,
        cross_shape_bench_n: int = 6,
        e2e_trials: int = 3,
        warmup: int = 2,
        iters: int = 6,
        lm_eval_tolerance: float = 0.005,
        work_dir: Path = Path("/tmp/ship_pipeline"),
        skip_lm_eval: bool = False,
        skip_e2e: bool = False,
        skip_cross_shape: bool = False,
        allow_regressions: bool = False,
        repo_root: Path = Path.cwd(),
        branch_prefix: str = "claude-auto",
        push: bool = False,
        open_pr: bool = False,
        dry_run_pr: bool = True,
        emit_pr_on_pass: bool = True) -> PipelineReport:
    """Run all gates; if all pass, emit a PR branch."""
    work_dir.mkdir(parents=True, exist_ok=True)
    gates: list[GateOutcome] = []
    t0 = time.time()

    # 1. Stats-bench (paired t)
    logger.info("=== Gate 1/4: Stats-bench ===")
    stats_out = work_dir / "stats.json"
    gates.append(
        _stats_gate(diff_path=diff_path,
                    kernel_path=kernel_path,
                    host_factory=host_factory,
                    bench_n=stats_bench_n,
                    warmup=warmup,
                    iters=iters,
                    out_json=stats_out))
    logger.info("  → %s", gates[-1].summary)

    # 2. Cross-shape
    cs_out = work_dir / "cross_shape.json"
    cs_md = work_dir / "cross_shape.md"
    if skip_cross_shape:
        gates.append(
            GateOutcome(name="cross_shape",
                        passed=True,
                        summary="skipped",
                        skipped=True))
    else:
        logger.info("=== Gate 2/4: Cross-shape ===")
        gates.append(
            _cross_shape_gate(diff_path=diff_path,
                              baseline_path=kernel_path,
                              out_json=cs_out,
                              out_md=cs_md,
                              bench_n=cross_shape_bench_n,
                              warmup=warmup,
                              iters=iters,
                              allow_regressions=allow_regressions))
        logger.info("  → %s", gates[-1].summary)

    # 3. lm-eval
    lm_out = work_dir / "lm_eval.json"
    if skip_lm_eval:
        gates.append(
            GateOutcome(name="lm_eval",
                        passed=True,
                        summary="skipped",
                        skipped=True))
    else:
        logger.info("=== Gate 3/4: lm-eval correctness ===")
        gates.append(
            _lm_eval_gate(diff_path=diff_path,
                          kernel_path=kernel_path,
                          model=model,
                          tasks=lm_eval_tasks,
                          limit=lm_eval_limit,
                          tensor_parallel=tensor_parallel,
                          max_model_len=max_model_len,
                          block_size=lm_eval_block_size,
                          output_dir=work_dir / "lm_eval_runs",
                          out_json=lm_out,
                          tolerance=lm_eval_tolerance))
        logger.info("  → %s", gates[-1].summary.splitlines()[0])

    # 4. E2E throughput
    e2e_out = work_dir / "e2e.json"
    if skip_e2e:
        gates.append(
            GateOutcome(name="e2e",
                        passed=True,
                        summary="skipped",
                        skipped=True))
    else:
        logger.info("=== Gate 4/4: E2E throughput ===")
        gates.append(
            _e2e_gate(diff_path=diff_path,
                      kernel_path=kernel_path,
                      model=model,
                      tensor_parallel=tensor_parallel,
                      max_model_len=max_model_len,
                      max_tokens=max_tokens,
                      num_prompts=num_prompts,
                      n_trials=e2e_trials,
                      out_json=e2e_out))
        logger.info("  → %s", gates[-1].summary)

    overall = all(g.passed for g in gates if not g.skipped)
    wall = time.time() - t0

    pr_artifact = None
    if overall and emit_pr_on_pass:
        from tools.kernel.evolve.auto_pr import Evidence, emit_pr_branch
        ev = Evidence.from_paths(
            kernel=kernel_name,
            hypothesis=hypothesis,
            diff_path=diff_path,
            stats_json=stats_out if stats_out.exists() else None,
            cross_shape_json=cs_out if cs_out.exists() else None,
            lm_eval_json=lm_out if lm_out.exists() else None,
        )
        pr_artifact = emit_pr_branch(evidence=ev,
                                     repo_root=repo_root,
                                     branch_prefix=branch_prefix,
                                     push=push,
                                     open_pr=open_pr,
                                     dry_run=dry_run_pr)

    report = PipelineReport(diff_path=diff_path,
                            gates=gates,
                            overall_pass=overall,
                            wall_sec=wall,
                            pr_artifact=pr_artifact)
    return report


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--diff", type=Path, required=True)
    p.add_argument("--kernel-path", type=Path, required=True)
    p.add_argument("--kernel-name", required=True)
    p.add_argument("--hypothesis", required=True)
    p.add_argument("--target",
                   default="rpa_v3",
                   choices=("rpa_v3", "mla_v2", "quantized_matmul",
                            "fused_moe_v1"),
                   help="Which evolve host to use for stats-bench.")
    p.add_argument("--model", default="Qwen/Qwen3-0.6B")
    p.add_argument("--lm-eval-tasks", default="gsm8k")
    p.add_argument("--lm-eval-limit", type=int, default=200)
    p.add_argument("--tensor-parallel", type=int, default=1)
    p.add_argument("--max-model-len", type=int, default=1024)
    p.add_argument("--lm-eval-block-size", type=int, default=64)
    p.add_argument("--max-tokens", type=int, default=128)
    p.add_argument("--num-prompts", type=int, default=32)
    p.add_argument("--stats-bench-n", type=int, default=8)
    p.add_argument("--cross-shape-bench-n", type=int, default=6)
    p.add_argument("--e2e-trials", type=int, default=3)
    p.add_argument("--lm-eval-tolerance", type=float, default=0.005)
    p.add_argument("--work-dir", type=Path, default=Path("/tmp/ship_pipeline"))
    p.add_argument("--skip-lm-eval", action="store_true")
    p.add_argument("--skip-e2e", action="store_true")
    p.add_argument("--skip-cross-shape", action="store_true")
    p.add_argument("--allow-regressions", action="store_true")
    p.add_argument("--repo-root", type=Path, default=Path.cwd())
    p.add_argument("--branch-prefix", default="claude-auto")
    p.add_argument("--push", action="store_true")
    p.add_argument("--open-pr", action="store_true")
    p.add_argument("--dry-run-pr", action="store_true", default=True)
    p.add_argument("--verbose", "-v", action="count", default=0)
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.WARNING - 10 * args.verbose,
                        format="%(asctime)s %(levelname)s %(message)s")

    def _host_factory():
        if args.target == "rpa_v3":
            from tools.kernel.evolve.examples.rpa_v3_evolve import RpaV3Host
            from tools.kernel.tuner.v1.common.kernel_tuner_base import \
                RunConfig
            from tools.kernel.tuner.v1.rpa_v3_kernel_tuner import \
                RpaV3KernelTuner
            rc = RunConfig(case_set_id="ship",
                           run_id="r0",
                           case_set_desc="ship pipeline",
                           tpu_version="tpu6e",
                           tpu_cores=1,
                           tpu_queue_multi="tpu_v6e_queue",
                           run_locally=True,
                           max_execution_minutes=10)
            return RpaV3Host(RpaV3KernelTuner(run_config=rc))
        raise NotImplementedError(f"target {args.target!r}")

    tasks = [t.strip() for t in args.lm_eval_tasks.split(",") if t.strip()]
    report = run_pipeline(
        diff_path=args.diff,
        kernel_path=args.kernel_path,
        host_factory=_host_factory,
        kernel_name=args.kernel_name,
        hypothesis=args.hypothesis,
        model=args.model,
        lm_eval_tasks=tasks,
        lm_eval_limit=args.lm_eval_limit,
        tensor_parallel=args.tensor_parallel,
        max_model_len=args.max_model_len,
        lm_eval_block_size=args.lm_eval_block_size,
        max_tokens=args.max_tokens,
        num_prompts=args.num_prompts,
        stats_bench_n=args.stats_bench_n,
        cross_shape_bench_n=args.cross_shape_bench_n,
        e2e_trials=args.e2e_trials,
        lm_eval_tolerance=args.lm_eval_tolerance,
        work_dir=args.work_dir,
        skip_lm_eval=args.skip_lm_eval,
        skip_e2e=args.skip_e2e,
        skip_cross_shape=args.skip_cross_shape,
        allow_regressions=args.allow_regressions,
        repo_root=args.repo_root,
        branch_prefix=args.branch_prefix,
        push=args.push,
        open_pr=args.open_pr,
        dry_run_pr=args.dry_run_pr,
    )
    report_path = args.work_dir / "report.json"
    report_path.write_text(json.dumps(report.to_dict(), indent=2))
    print()
    print("=" * 78)
    for g in report.gates:
        tag = ("SKIP " if g.skipped else ("PASS " if g.passed else "FAIL "))
        print(f"  [{tag}] {g.name:<14} {g.summary.splitlines()[0]}")
    print()
    print(f"  overall: {'PASS' if report.overall_pass else 'FAIL'}")
    print(f"  wall:    {report.wall_sec:.1f}s")
    if report.pr_artifact:
        print(f"  branch:  {report.pr_artifact.get('branch')}")
        if report.pr_artifact.get("commit_sha"):
            print(f"  commit:  {report.pr_artifact['commit_sha']}")
        if report.pr_artifact.get("pr_url"):
            print(f"  PR url:  {report.pr_artifact['pr_url']}")
    print(f"  report:  {report_path}")
    return 0 if report.overall_pass else 1


if __name__ == "__main__":
    sys.exit(main())
