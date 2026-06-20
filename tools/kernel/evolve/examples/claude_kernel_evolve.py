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
"""Generic Claude-driven evolve driver — works for any ``GenericHost``.

Subsumes the per-kernel drivers (`claude_rpa_v3_evolve`,
`claude_matmul_evolve`) for the common case where the host is built by
a factory function in ``kernel_hosts.py``. Pass ``--target`` to pick the
factory; everything else (ensemble, BoN, critic, ExamplePool, FailureLog)
is wired through.

Available targets:
* ``quantized_matmul`` -> ``make_quantized_matmul_host``
* ``fused_moe_v1``     -> ``make_fused_moe_v1_host``
* ``mla_v2``           -> ``make_mla_v2_host``

To add a kernel: write a ``make_<kernel>_host`` factory in
``kernel_hosts.py`` and register it in ``_TARGETS`` below.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

from tools.kernel.evolve.archive import Archive
from tools.kernel.evolve.examples.kernel_hosts import (
    make_fused_moe_v1_host, make_mla_v2_host, make_quantized_matmul_host)
from tools.kernel.evolve.genome import Genome
from tools.kernel.evolve.mutator.bon import BestOfNMutator
from tools.kernel.evolve.mutator.ensemble import EnsembleClient
from tools.kernel.evolve.mutator.example_pool import ExamplePool
from tools.kernel.evolve.mutator.failure_log import FailureLog
from tools.kernel.evolve.mutator.vertex_anthropic import VertexAnthropicClient
from tools.kernel.evolve.orchestrator import EvolutionConfig, Orchestrator
from tools.kernel.tuner.v1.verifier.anti_cheat import AntiCheatGuard

# Map --target to (host_factory, default kwargs).
_TARGETS = {
    "quantized_matmul": (
        make_quantized_matmul_host,
        dict(n_batch=512, n_in=2048, n_out=2048),
    ),
    "fused_moe_v1": (
        make_fused_moe_v1_host,
        dict(num_tokens=64,
             hidden_size=512,
             intermediate_size=1024,
             num_experts=8,
             topk=2),
    ),
    "mla_v2": (
        make_mla_v2_host,
        dict(seq_lens=((1, 128), (1, 256), (4, 384)),
             num_heads=8,
             lkv_dim=512,
             r_dim=64,
             page_size=64),
    ),
}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--target", required=True, choices=sorted(_TARGETS.keys()))
    p.add_argument("--project", default=None)
    p.add_argument("--region", default="global")
    p.add_argument("--mutator-model", default="claude-opus-4-8")
    p.add_argument("--ensemble-models",
                   default="claude-opus-4-8,claude-opus-4-7",
                   help="Comma-separated model ids to ensemble (round-robin).")
    p.add_argument("--critic-model", default="claude-opus-4-7")
    p.add_argument("--generations", type=int, default=3)
    p.add_argument("--islands", type=int, default=2)
    p.add_argument("--island-candidates",
                   type=int,
                   default=2,
                   dest="candidates_per_island")
    p.add_argument("--bon",
                   type=int,
                   default=2,
                   help="Best-of-N sampling: candidates per turn.")
    p.add_argument("--use-critic", action="store_true", default=True)
    p.add_argument("--bench-iters", type=int, default=6)
    p.add_argument("--warmup-iters", type=int, default=2)
    p.add_argument("--cosine-floor",
                   type=float,
                   default=0.9999,
                   help="Numerics cosine-similarity floor (default 0.9999). "
                   "Some kernels (MoE with per-token reference) need a "
                   "looser floor like 0.997.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--archive",
                   type=Path,
                   help="Path for JSONL archive (default: per-target /tmp).")
    p.add_argument("--failure-log",
                   type=Path,
                   help="Path for failure log (default: per-target /tmp).")
    p.add_argument("--example-pool",
                   type=Path,
                   help="Path for example pool (default: per-target /tmp).")
    p.add_argument("--verbose", "-v", action="count", default=0)
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.WARNING - 10 * args.verbose,
                        format="%(asctime)s %(levelname)s %(message)s")
    project = args.project or os.environ.get("ANTHROPIC_VERTEX_PROJECT_ID")
    if not project:
        print("error: provide --project or set ANTHROPIC_VERTEX_PROJECT_ID",
              file=sys.stderr)
        return 2

    factory, defaults = _TARGETS[args.target]
    host = factory(**defaults)
    print(f"Target host: {host.kernel_name} symbol={host.kernel_symbol}",
          file=sys.stderr)

    arc_dir = Path(f"/tmp/claude_{args.target}")
    arc_dir.mkdir(parents=True, exist_ok=True)
    archive_path = args.archive or arc_dir / "archive.jsonl"
    failure_path = args.failure_log or arc_dir / "failure_log.json"
    pool_path = args.example_pool or arc_dir / "example_pool.json"

    if archive_path.exists():
        archive_path.unlink()
    archive = Archive(
        baseline=Genome.baseline(baseline_path=host.baseline_path),
        num_islands=args.islands,
        island_cap=max(4, args.candidates_per_island * args.generations + 2),
        persist_path=archive_path,
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
        inner = EnsembleClient(clients, strategy="round_robin")
    else:
        inner = VertexAnthropicClient(model=args.mutator_model,
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
        inner, n=args.bon, temperatures=[0.3, 0.5, 0.7, 0.9])
               if args.bon > 1 else inner)
    print(f"Mutator: {mutator.model_id}", file=sys.stderr)
    print(f"Critic:  {critic.model_id}", file=sys.stderr)

    config = EvolutionConfig(
        num_islands=args.islands,
        candidates_per_island_per_gen=args.candidates_per_island,
        generations=args.generations,
        migration_freq=max(1, args.generations),
        migration_top_k=1,
        use_critic=args.use_critic,
        warmup_iters=args.warmup_iters,
        bench_iters=args.bench_iters,
        cosine_floor=args.cosine_floor,
        seed=args.seed,
        mutation_max_tokens=4096,
        critic_max_tokens=512,
        max_parent_summaries=2,
    )
    fl = FailureLog(persist_path=failure_path)
    pool = ExamplePool(persist_path=pool_path)
    print(f"FailureLog: {len(fl.all_stats())} rules tracked", file=sys.stderr)
    print(f"ExamplePool: {pool.size()} past wins available", file=sys.stderr)

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
        f"Starting Claude-driven {args.target} evolve "
        f"({args.generations} gens × {args.islands} islands × "
        f"{args.candidates_per_island} cands × BoN={args.bon})",
        file=sys.stderr)
    final = orch.run()
    wall = time.time() - t0
    s = final.summary()
    print()
    print("=" * 78)
    print(f"Claude {args.target} evolve finished in {wall:.1f}s")
    if s["best_genome_id"]:
        print(f"  baseline: {s['baseline_fitness_ns']/1e3:.2f} us")
        print(f"  best:     {s['best_fitness_ns']/1e3:.2f} us "
              f"({s['speedup_vs_baseline']:.4f}x)")
    print(f"  genomes: {s['total_genomes']} total, "
          f"{s['verified']} verified")
    print(f"  archive:  {archive_path}")
    return 0 if s["best_genome_id"] else 1


if __name__ == "__main__":
    sys.exit(main())
