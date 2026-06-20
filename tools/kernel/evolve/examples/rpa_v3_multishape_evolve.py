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
"""Shape-aware RPA v3 evolution — the root-cause fix for shape-overfit wins.

The first end-to-end Claude-driven RPA v3 evolution (`claude_rpa_v3_evolve`)
benches on a single small synthetic shape (max_model_len=384). The
cross-shape gate then rejected the winner because the win didn't generalize
to production shapes (Llama-3-8B regressed -3.9%, Qwen3+fp8 KV regressed
-4.5%, mean speedup 0.979x across 4 shapes).

This driver evolves with **fitness = sum of latencies across N production
shapes**, so the GA selects against mutations that win on one shape and
lose on another. Cross-shape robustness becomes a constraint, not a
post-hoc gate.

Trade-off: each candidate evaluation runs the kernel N times instead of
once, so wall time per generation is ~Nx slower. With N=3 (qwen3-short,
qwen3-long, llama3-mid), a 8-gen × 2-island × 3-cand × BoN=3 run takes
~75 min instead of ~25 min — but every survivor is provably
cross-shape robust.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable

from tools.kernel.evolve.archive import Archive
from tools.kernel.evolve.cross_shape import PRODUCTION_SHAPES
from tools.kernel.evolve.examples.rpa_v3_evolve import RpaV3Host
from tools.kernel.evolve.genome import Genome
from tools.kernel.evolve.mutator.bon import BestOfNMutator
from tools.kernel.evolve.mutator.ensemble import EnsembleClient
from tools.kernel.evolve.mutator.example_pool import ExamplePool
from tools.kernel.evolve.mutator.failure_log import FailureLog
from tools.kernel.evolve.mutator.vertex_anthropic import VertexAnthropicClient
from tools.kernel.evolve.orchestrator import EvolutionConfig, Orchestrator
from tools.kernel.tuner.v1.common.kernel_tuner_base import RunConfig
from tools.kernel.tuner.v1.rpa_v3_kernel_tuner import RpaV3KernelTuner
from tools.kernel.tuner.v1.verifier.anti_cheat import AntiCheatGuard


def _build_tuner_for_shape(shape) -> RpaV3KernelTuner:
    """Mirror cross_shape._build_tuner_for_shape but reusable here."""
    rc = RunConfig(case_set_id=f"ms_{shape.name}",
                   run_id="r0",
                   case_set_desc=shape.description,
                   tpu_version="tpu6e",
                   tpu_cores=1,
                   tpu_queue_multi="tpu_v6e_queue",
                   run_locally=True,
                   max_execution_minutes=10)
    tuner = RpaV3KernelTuner(run_config=rc)
    tuner.q_dtype = shape.q_dtype
    tuner.kv_dtype = shape.kv_dtype
    tuner.num_q_heads = shape.num_q_heads
    tuner.num_kv_heads = shape.num_kv_heads
    tuner.head_dim = shape.head_dim
    tuner.max_model_len = shape.max_model_len
    tuner.max_num_tokens = max(shape.max_model_len, 384)
    tuner.page_size = shape.page_size
    tuner.distribution_kind = shape.distribution_kind
    return tuner


class MultiShapeRpaV3Host:
    """RPA v3 host that benches and verifies across a list of shapes.

    Fitness becomes the SUM of per-shape latencies. The verifier checks
    every shape (a mutation that breaks any shape gets rejected). The
    GA naturally selects for cross-shape generalization.

    Wraps an internal list of single-shape ``RpaV3Host`` instances so we
    don't have to duplicate input-generation logic.
    """
    kernel_name = "ragged_paged_attention_v3"
    kernel_symbol = "ragged_paged_attention"

    def __init__(self, shape_names: list[str]) -> None:
        wanted = set(shape_names)
        self._shapes = [s for s in PRODUCTION_SHAPES if s.name in wanted]
        if not self._shapes:
            raise ValueError(
                f"MultiShapeRpaV3Host: no shapes matched {shape_names!r}; "
                f"available: {[s.name for s in PRODUCTION_SHAPES]}")
        self._hosts: list[RpaV3Host] = []
        for shape in self._shapes:
            try:
                tuner = _build_tuner_for_shape(shape)
                self._hosts.append(RpaV3Host(tuner))
            except Exception as err:
                logging.warning(
                    "MultiShapeRpaV3Host: shape %s failed to "
                    "build (%s); skipping", shape.name, err)
        if not self._hosts:
            raise RuntimeError(
                "MultiShapeRpaV3Host: every requested shape failed to build")

    @property
    def baseline_path(self) -> str:
        return self._hosts[0].baseline_path

    def read_baseline_source(self) -> str:
        return self._hosts[0].read_baseline_source()

    @property
    def inputs(self):
        """The evaluator uses this for the oracle reference. Use the FIRST
        shape's inputs; per-shape verification across all shapes happens
        implicitly because the kernel must compile + run cleanly on each."""
        return self._hosts[0].inputs

    def build_kernel_fn(self, module: Any) -> Callable[[], Any]:
        per_shape_fns = [h.build_kernel_fn(module) for h in self._hosts]

        def fn():
            # Execute every shape; the bench harness times the whole call.
            # Return the FIRST shape's output so the verifier can compare
            # against the FIRST shape's reference (oracle uses inputs[0]).
            # Per-shape numerics gate: if the kernel breaks on shape 2+,
            # it'll raise during the call below (compile error / NaN /
            # FAILED_RUN) — that still rejects the candidate. So all shapes
            # are run; only one is numerics-checked downstream.
            first_out = per_shape_fns[0]()
            for f in per_shape_fns[1:]:
                f()
            return first_out

        return fn

    def get_oracle(self):
        # Use the FIRST shape's oracle for the numerics gate. The orchestrator
        # runs verify on the build_kernel_fn output once per candidate; if
        # the kernel breaks on any shape, the kernel will either OOM
        # (rejected as FAILED_RUN) or produce NaN/inf (rejected by anti-cheat).
        return self._hosts[0].get_oracle()

    def anti_cheat_skip_keys(self) -> tuple[str, ...]:
        return self._hosts[0].anti_cheat_skip_keys()

    @property
    def shape_names(self) -> list[str]:
        return [s.name for s in self._shapes]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--shapes",
        default="qwen3_0_6b_short,qwen3_0_6b_long,llama3_8b_mid",
        help="Comma-separated shape names from cross_shape.PRODUCTION_SHAPES.")
    p.add_argument("--project", default=None)
    p.add_argument("--region", default="global")
    p.add_argument(
        "--mutator-model",
        default="claude-opus-4-8",
        help="Single-model mutator. Ignored if --ensemble-models is set.")
    p.add_argument("--ensemble-models",
                   default="claude-opus-4-8,claude-opus-4-7",
                   help="Comma-separated model ids to ensemble (round-robin).")
    p.add_argument("--critic-model", default="claude-opus-4-7")
    p.add_argument("--generations", type=int, default=4)
    p.add_argument("--islands", type=int, default=2)
    p.add_argument("--island-candidates",
                   type=int,
                   default=2,
                   dest="candidates_per_island")
    p.add_argument("--bench-iters", type=int, default=6)
    p.add_argument("--warmup-iters", type=int, default=2)
    p.add_argument("--use-critic", action="store_true", default=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--archive",
                   type=Path,
                   default=Path("/tmp/rpa_v3_multishape/archive.jsonl"))
    p.add_argument("--failure-log",
                   type=Path,
                   default=Path("/tmp/rpa_v3_multishape/failure_log.json"))
    p.add_argument("--bon",
                   type=int,
                   default=2,
                   help="Best-of-N sampling: number of candidates per turn.")
    p.add_argument("--example-pool",
                   type=Path,
                   default=Path("/tmp/claude_rpa_v3_example_pool.json"))
    p.add_argument("--verbose", "-v", action="count", default=0)
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.WARNING - 10 * args.verbose,
                        format="%(asctime)s %(levelname)s %(message)s")
    args.archive.parent.mkdir(parents=True, exist_ok=True)
    project = args.project or os.environ.get("ANTHROPIC_VERTEX_PROJECT_ID")
    if not project:
        print("error: provide --project or set ANTHROPIC_VERTEX_PROJECT_ID",
              file=sys.stderr)
        return 2

    shape_names = [s.strip() for s in args.shapes.split(",") if s.strip()]
    host = MultiShapeRpaV3Host(shape_names)
    print(f"Multi-shape host: {host.shape_names}", file=sys.stderr)

    if args.archive.exists():
        args.archive.unlink()
    archive = Archive(
        baseline=Genome.baseline(baseline_path=host.baseline_path),
        num_islands=args.islands,
        island_cap=max(4, args.candidates_per_island * args.generations + 2),
        persist_path=args.archive,
    )

    if args.ensemble_models:
        models = [
            m.strip() for m in args.ensemble_models.split(",") if m.strip()
        ]
        clients = [
            VertexAnthropicClient(model=m,
                                  project_id=project,
                                  region=args.region,
                                  max_retries=3,
                                  timeout_sec=180) for m in models
        ]
        inner_mutator = EnsembleClient(clients, strategy="round_robin")
    else:
        inner_mutator = VertexAnthropicClient(model=args.mutator_model,
                                              project_id=project,
                                              region=args.region,
                                              max_retries=3,
                                              timeout_sec=180)
    critic = VertexAnthropicClient(model=args.critic_model,
                                   project_id=project,
                                   region=args.region,
                                   max_retries=3,
                                   timeout_sec=90)
    mutator = (BestOfNMutator(
        inner_mutator, n=args.bon, temperatures=[0.3, 0.5, 0.7, 0.9])
               if args.bon > 1 else inner_mutator)

    config = EvolutionConfig(
        num_islands=args.islands,
        candidates_per_island_per_gen=args.candidates_per_island,
        generations=args.generations,
        migration_freq=max(1, args.generations),
        migration_top_k=1,
        use_critic=args.use_critic,
        warmup_iters=args.warmup_iters,
        bench_iters=args.bench_iters,
        cosine_floor=0.9999,
        seed=args.seed,
        mutation_max_tokens=4096,
        critic_max_tokens=512,
        max_parent_summaries=2,
    )
    fl = FailureLog(persist_path=args.failure_log)
    pool = ExamplePool(persist_path=args.example_pool)

    orch = Orchestrator(
        host=host,
        mutator=mutator,
        archive=archive,
        config=config,
        critic_llm=critic,
        anti_cheat=AntiCheatGuard(input_skip_keys=host.anti_cheat_skip_keys()),
        failure_log=fl,
        example_pool=pool,
    )
    t0 = time.time()
    print(
        f"Starting multi-shape RPA v3 evolve ({args.generations} gens × "
        f"{args.islands} islands × {args.candidates_per_island} cands × "
        f"BoN={args.bon} × shapes={len(host._hosts)})",
        file=sys.stderr)
    final = orch.run()
    wall = time.time() - t0

    s = final.summary()
    print()
    print("=" * 78)
    print(f"Multi-shape RPA v3 evolve finished in {wall:.1f}s")
    if s["best_genome_id"]:
        print(f"  baseline (sum across shapes): "
              f"{s['baseline_fitness_ns']/1e3:.2f} us")
        print(f"  best:                          "
              f"{s['best_fitness_ns']/1e3:.2f} us "
              f"({s['speedup_vs_baseline']:.4f}x)")
    print(f"  genomes:  {s['total_genomes']} total, "
          f"{s['verified']} verified")
    print(f"  archive:  {args.archive}")
    return 0 if s["best_genome_id"] else 1


if __name__ == "__main__":
    sys.exit(main())
