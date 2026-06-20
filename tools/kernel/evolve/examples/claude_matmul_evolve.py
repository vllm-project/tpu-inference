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
"""Real LLM-driven evolve loop: Claude (Vertex) on ``synthetic_matmul``.

This is the warm-up demonstration that connects the full stack:

    Claude mutator → diff applier → worktree → bench harness → verifier →
    archive → critic gate → telemetry

The target is the synthetic Pallas matmul. Small (~80 LOC) so Claude can
see the whole file in one prompt; the verifier compares against pure
``jnp.matmul``; the bench harness measures real TPU latency.

Set ``ANTHROPIC_VERTEX_PROJECT_ID`` (or pass via flag). Auth via gcloud
application-default credentials.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

from tools.kernel.evolve.archive import Archive
from tools.kernel.evolve.examples.matmul_evolve import MatmulHost
from tools.kernel.evolve.genome import Genome
from tools.kernel.evolve.mutator.bon import BestOfNMutator
from tools.kernel.evolve.mutator.example_pool import ExamplePool
from tools.kernel.evolve.mutator.failure_log import FailureLog
from tools.kernel.evolve.mutator.vertex_anthropic import VertexAnthropicClient
from tools.kernel.evolve.orchestrator import EvolutionConfig, Orchestrator
from tools.kernel.tuner.v1.verifier.anti_cheat import AntiCheatGuard


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--project",
                   default=None,
                   help="GCP project_id; defaults to env "
                   "ANTHROPIC_VERTEX_PROJECT_ID.")
    p.add_argument("--region",
                   default="global",
                   help="Vertex region. 'global' is the documented default "
                   "for Anthropic-on-Vertex auto-routing.")
    p.add_argument("--mutator-model",
                   default="claude-opus-4-8",
                   help="Claude model id for the mutation step.")
    p.add_argument(
        "--critic-model",
        default="claude-opus-4-7",
        help="Claude model id for the adversarial critic. Defaults "
        "to 4-7 (one tier below mutator) for faster/cheaper refute.")
    p.add_argument("--M", type=int, default=2048)
    p.add_argument("--N", type=int, default=2048)
    p.add_argument("--K", type=int, default=2048)
    p.add_argument("--generations", type=int, default=3)
    p.add_argument("--islands", type=int, default=2)
    p.add_argument("--island-candidates",
                   type=int,
                   default=3,
                   dest="candidates_per_island")
    p.add_argument("--bench-iters", type=int, default=10)
    p.add_argument("--warmup-iters", type=int, default=3)
    p.add_argument("--use-critic",
                   action="store_true",
                   help="Enable adversarial critic gate.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--archive",
                   type=Path,
                   default=Path("/tmp/claude_matmul_archive.jsonl"))
    p.add_argument("--telemetry",
                   type=Path,
                   default=Path("/tmp/claude_matmul_telemetry.jsonl"))
    p.add_argument("--bon",
                   type=int,
                   default=1,
                   help="Best-of-N sampling: number of candidates per turn.")
    p.add_argument(
        "--example-pool",
        type=Path,
        default=Path("/tmp/claude_example_pool.json"),
        help="Persistent RLAIF positive-example pool. /dev/null to disable.")
    p.add_argument("--failure-log",
                   type=Path,
                   default=Path("/tmp/claude_failure_log.json"),
                   help="Persistent failure log. /dev/null to disable.")
    p.add_argument("--verbose", "-v", action="count", default=0)
    args = p.parse_args(argv)

    log_level = logging.WARNING - 10 * args.verbose
    logging.basicConfig(
        level=max(log_level, logging.DEBUG),
        format="%(asctime)s %(levelname)s %(name)s %(message)s")

    project = args.project or os.environ.get("ANTHROPIC_VERTEX_PROJECT_ID")
    if not project:
        print("error: provide --project or set ANTHROPIC_VERTEX_PROJECT_ID",
              file=sys.stderr)
        return 2

    host = MatmulHost(M=args.M, N=args.N, K=args.K, seed=args.seed)
    if args.archive.exists():
        args.archive.unlink()  # fresh archive each run
    archive = Archive(
        baseline=Genome.baseline(baseline_path=host.baseline_path),
        num_islands=args.islands,
        island_cap=max(8, args.candidates_per_island * args.generations + 2),
        persist_path=args.archive,
    )

    inner_mutator = VertexAnthropicClient(model=args.mutator_model,
                                          project_id=project,
                                          region=args.region,
                                          max_retries=3,
                                          timeout_sec=120)
    critic = VertexAnthropicClient(model=args.critic_model,
                                   project_id=project,
                                   region=args.region,
                                   max_retries=3,
                                   timeout_sec=60)
    if args.bon > 1:
        mutator = BestOfNMutator(inner_mutator,
                                 n=args.bon,
                                 temperatures=[0.3, 0.5, 0.7, 0.9])
        print(f"Mutator: {mutator.model_id} (best-of-{args.bon})",
              file=sys.stderr)
    else:
        mutator = inner_mutator
        print(f"Mutator: {mutator.model_id}", file=sys.stderr)
    print(f"Critic:  {critic.model_id}", file=sys.stderr)

    fl = (FailureLog(persist_path=args.failure_log)
          if str(args.failure_log) != "/dev/null" else FailureLog())
    pool = (ExamplePool(persist_path=args.example_pool)
            if str(args.example_pool) != "/dev/null" else ExamplePool())
    print(f"FailureLog: {len(fl.all_stats())} rules tracked", file=sys.stderr)
    print(f"ExamplePool: {pool.size()} past wins available", file=sys.stderr)
    print(f"Target:  {host.kernel_name} ({host.baseline_path})",
          file=sys.stderr)
    print(f"Shape:   ({args.M}, {args.K}) @ ({args.K}, {args.N}) bf16",
          file=sys.stderr)

    config = EvolutionConfig(
        num_islands=args.islands,
        candidates_per_island_per_gen=args.candidates_per_island,
        generations=args.generations,
        migration_freq=max(1, args.generations // 2),
        migration_top_k=1,
        use_critic=args.use_critic,
        warmup_iters=args.warmup_iters,
        bench_iters=args.bench_iters,
        cosine_floor=0.999,
        seed=args.seed,
        mutation_max_tokens=2048,
        critic_max_tokens=512,
        max_parent_summaries=2,
    )
    orch = Orchestrator(
        host=host,
        mutator=mutator,
        archive=archive,
        config=config,
        critic_llm=critic,
        anti_cheat=AntiCheatGuard(),
        failure_log=fl,
        example_pool=pool,
    )

    t0 = time.time()
    print(
        f"\nStarting Claude-driven evolve "
        f"({args.generations} gens × {args.islands} islands × "
        f"{args.candidates_per_island} candidates)...\n",
        file=sys.stderr)
    final = orch.run()
    wall = time.time() - t0

    s = final.summary()
    print()
    print("=" * 78)
    print(f"Claude evolve finished in {wall:.1f}s.")
    print(
        f"  baseline: {s['baseline_fitness_ns']/1e3 if s['baseline_fitness_ns'] else 'n/a':.2f} us"
    )
    if s["best_genome_id"]:
        print(f"  best:     {s['best_fitness_ns']/1e3:.2f} us "
              f"({s['speedup_vs_baseline']:.4f}x)")
    print(f"  genomes:  {s['total_genomes']} total, "
          f"{s['verified']} verified")
    print()
    print("Top 5 candidates:")
    top = sorted(final.all_finite(), key=lambda g: g.fitness)[:5]
    for g in top:
        print(f"  {g.short()}")
    print()
    print(f"Archive: {args.archive}")
    return 0 if s["best_genome_id"] else 1


if __name__ == "__main__":
    sys.exit(main())
