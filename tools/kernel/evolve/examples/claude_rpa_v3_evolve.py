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
"""Claude-driven evolve on ``ragged_paged_attention/v3`` — production kernel.

This is the *real* target: a 1933-LOC Pallas kernel that Qwen3 uses for
attention. Claude sees the full source via the orchestrator's mutation
prompt, proposes a unified diff, the verifier compares against the eager
reference, and only verified candidates survive.

Token cost per call is meaningful (~50k input × $15/M ≈ $0.75/call). Default
budget here is conservative: 2 generations × 1 island × 2 candidates =
~4-6 LLM calls + critic = ~$10-15 of usage.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

from tools.kernel.evolve.archive import Archive
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


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--project", default=None)
    p.add_argument("--region", default="global")
    p.add_argument(
        "--mutator-model",
        default="claude-opus-4-8",
        help="Single-model mutator. Ignored if --ensemble-models is set.")
    p.add_argument("--ensemble-models",
                   default=None,
                   help="Comma-separated model ids to ensemble (round-robin). "
                   "Overrides --mutator-model. Example: "
                   "'claude-opus-4-8,claude-opus-4-7'.")
    p.add_argument("--critic-model", default="claude-opus-4-7")
    p.add_argument("--generations", type=int, default=2)
    p.add_argument("--islands", type=int, default=1)
    p.add_argument("--island-candidates",
                   type=int,
                   default=2,
                   dest="candidates_per_island")
    p.add_argument("--bench-iters", type=int, default=6)
    p.add_argument("--warmup-iters", type=int, default=2)
    p.add_argument("--use-critic", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--archive",
                   type=Path,
                   default=Path("/tmp/claude_rpa_v3_archive.jsonl"))
    p.add_argument("--failure-log",
                   type=Path,
                   default=Path("/tmp/claude_rpa_v3_failure_log.json"),
                   help="Persistent failure log; anti-patterns fed into "
                   "subsequent mutator prompts. Pass /dev/null to disable.")
    p.add_argument("--bon",
                   type=int,
                   default=1,
                   help="Best-of-N sampling: number of candidates per turn.")
    p.add_argument("--example-pool",
                   type=Path,
                   default=Path("/tmp/claude_rpa_v3_example_pool.json"),
                   help="Persistent RLAIF positive-example pool.")
    p.add_argument("--verbose", "-v", action="count", default=0)
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.WARNING - 10 * args.verbose,
                        format="%(asctime)s %(levelname)s %(message)s")
    project = args.project or os.environ.get("ANTHROPIC_VERTEX_PROJECT_ID")
    if not project:
        print("error: provide --project or set ANTHROPIC_VERTEX_PROJECT_ID",
              file=sys.stderr)
        return 2

    rc = RunConfig(
        case_set_id="claude_rpa_v3",
        run_id="r0",
        case_set_desc="claude-driven evolve",
        tpu_version="tpu6e",
        tpu_cores=1,
        tpu_queue_multi="tpu_v6e_queue",
        run_locally=True,
        max_execution_minutes=30,
    )
    tuner = RpaV3KernelTuner(run_config=rc)
    host = RpaV3Host(tuner)

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
    if args.bon > 1:
        mutator = BestOfNMutator(inner_mutator,
                                 n=args.bon,
                                 temperatures=[0.3, 0.5, 0.7, 0.9])
    else:
        mutator = inner_mutator
    print(f"Mutator: {mutator.model_id}", file=sys.stderr)
    print(f"Critic:  {critic.model_id}", file=sys.stderr)
    print(f"Target:  {host.kernel_name} ({host.baseline_path})",
          file=sys.stderr)

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
    fl = (FailureLog(persist_path=args.failure_log)
          if str(args.failure_log) != "/dev/null" else FailureLog())
    pool = (ExamplePool(persist_path=args.example_pool)
            if str(args.example_pool) != "/dev/null" else ExamplePool())
    print(f"FailureLog: {len(fl.all_stats())} rules tracked across runs",
          file=sys.stderr)
    print(f"ExamplePool: {pool.size()} past verified wins available",
          file=sys.stderr)

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
        f"Starting Claude-driven RPA v3 evolve "
        f"({args.generations} gens × {args.islands} islands × "
        f"{args.candidates_per_island} cands)...\n",
        file=sys.stderr)
    final = orch.run()
    wall = time.time() - t0

    s = final.summary()
    print()
    print("=" * 78)
    print(f"Claude RPA v3 evolve finished in {wall:.1f}s.")
    print(
        f"  baseline: {s['baseline_fitness_ns']/1e3 if s['baseline_fitness_ns'] else 'n/a':.2f} us"
    )
    if s["best_genome_id"]:
        print(f"  best:     {s['best_fitness_ns']/1e3:.2f} us "
              f"({s['speedup_vs_baseline']:.4f}x)")
    print(f"  genomes:  {s['total_genomes']} total, "
          f"{s['verified']} verified")
    print()
    top = sorted(final.all_finite(), key=lambda g: g.fitness)[:5]
    for g in top:
        print(f"  {g.short()}")
    print(f"\nArchive: {args.archive}")
    return 0 if s["best_genome_id"] else 1


if __name__ == "__main__":
    sys.exit(main())
