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
"""Production-shape RPA v3 evolution with profile-data-in-prompt.

Two things this driver does differently from `rpa_v3_multishape_evolve`:

1. **Production-scale shapes**. The default test fixture (3 sequences,
   ~7 query tokens) underflows kernel dispatch overhead — roofline
   analysis there reports <1% HBM util and 0% MXU util because nothing
   real is being computed. This driver uses
   `cross_shape.PRODUCTION_SCALE_DECODE_SHAPES` (256 decode seqs × kv=2048
   etc.) where the kernel is genuinely MXU- or HBM-bound and there is
   real headroom to attack.

2. **Profile-data-in-prompt**. Before evolution starts, this driver runs
   `tools.kernel.evolve.roofline` on the chosen shapes and injects the
   diagnosis ("kernel is MXU-bound at 35% util, 65% headroom — attack
   family B pipelining") into every mutator system prompt via a thin
   wrapper around the inner LLM client. This tells Claude WHICH FAMILY
   of mutation to attempt, so it stops grinding on family-H precision
   policy (which can't help an MXU-bound kernel by definition).

Result: in initial validation, this driver discovers structural
mutations (pipelining, op reordering, regime specialization) instead of
single-line precision casts — the kind of mutation that produces
double-digit speedups.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

from tools.kernel.evolve.archive import Archive
from tools.kernel.evolve.examples.rpa_v3_multishape_evolve import \
    MultiShapeRpaV3Host
from tools.kernel.evolve.genome import Genome
from tools.kernel.evolve.mutator.bon import BestOfNMutator
from tools.kernel.evolve.mutator.ensemble import EnsembleClient
from tools.kernel.evolve.mutator.example_pool import ExamplePool
from tools.kernel.evolve.mutator.failure_log import FailureLog
from tools.kernel.evolve.mutator.vertex_anthropic import VertexAnthropicClient
from tools.kernel.evolve.orchestrator import EvolutionConfig, Orchestrator
from tools.kernel.evolve.roofline import bench_baseline_at_shape, render_md
from tools.kernel.tuner.v1.verifier.anti_cheat import AntiCheatGuard

DEFAULT_PRODUCTION_SHAPES = ("qwen3_0_6b_decode_conc256,"
                             "llama3_8b_decode_conc256,"
                             "qwen35_decode_conc128")


def _build_roofline_hint(shape_names: list[str]) -> str:
    """Run roofline on each shape, return a one-paragraph diagnosis the
    mutator can act on."""
    from tools.kernel.evolve.cross_shape import PRODUCTION_SHAPES
    name_to_shape = {s.name: s for s in PRODUCTION_SHAPES}
    results = []
    for n in shape_names:
        if n not in name_to_shape:
            print(f"warning: unknown shape {n}", file=sys.stderr)
            continue
        try:
            r = bench_baseline_at_shape(name_to_shape[n])
            results.append(r)
        except Exception as e:
            print(f"warning: shape {n} failed roofline: {e}", file=sys.stderr)
    if not results:
        return ""
    md = render_md(results)
    # Compose a directional hint for the mutator system prompt.
    hbm = [r.hbm_util_frac for r in results]
    mxu = [r.mxu_util_frac for r in results]
    avg_hbm = sum(hbm) / len(hbm)
    avg_mxu = sum(mxu) / len(mxu)
    regimes = {r.regime for r in results}
    main_regime = (next(iter(regimes)) if len(regimes) == 1 else max(
        regimes, key=lambda x: sum(1 for r in results if r.regime == x)))
    if main_regime == "mxu-bound":
        directive = (
            "The kernel is MXU-BOUND at production scale "
            f"(avg MXU util {avg_mxu*100:.1f}%; HBM util "
            f"{avg_hbm*100:.1f}%). Headroom: ~{(1.0/max(avg_mxu, 0.01)):.1f}x "
            "if MXU could be saturated.\n\n"
            "**PRIORITIZE these mutation classes (perf-skill families B+D+J)**:\n"
            "- B: PIPELINING — overlap QK·softmax with PV (#2282 pattern: "
            "split the inner loop into separate flash_attention_qk_softmax "
            "and flash_attention_pv calls; pipeline so chunk N's softmax "
            "runs concurrently with chunk N-1's PV matmul).\n"
            "- B: software-pipeline the KV DMA with compute "
            "(prologue prefetch / N+1 prefetch / wait N / compute / "
            "store N-2).\n"
            "- D: regime specialization — if the decode and prefill paths "
            "use the same block sizes and code, split them so each runs "
            "the kernel best suited to its shape (#1820 split into 3 "
            "specialized launches).\n"
            "- J: WORK ELIMINATION — causal-skip "
            "(`end_bkv_idx = cdiv(min(kv_len, processed+bq), bkv)`), "
            "algebraic identity (collapse rank-1 matmul into scalar dot, "
            "#2498), avoid materializing intermediates the next op only "
            "needs partially.\n\n"
            "**DEPRIORITIZE family H (dtype/precision policy)**: the kernel "
            f"is using only {avg_hbm*100:.1f}% of HBM bandwidth, so saving "
            "a few HBM bytes via cast policy will not move the dial. The "
            "MXU is the bottleneck.")
    elif main_regime == "hbm-bound":
        directive = (
            "The kernel is HBM-BOUND at production scale "
            f"(avg HBM util {avg_hbm*100:.1f}%; MXU util "
            f"{avg_mxu*100:.1f}%). Headroom: ~{(1.0/max(avg_hbm, 0.01)):.1f}x "
            "if HBM bandwidth could be saturated.\n\n"
            "**PRIORITIZE family A (memory hierarchy & fusion)**: donation "
            "(`donate_argnames=`), in-VMEM dequant, fuse adjacent ops to "
            "keep intermediates in VMEM, eliminate redundant HBM "
            "round-trips. **DEPRIORITIZE family B**: the MXU is not the "
            "bottleneck.")
    else:
        directive = (
            "The kernel is VPU/SCALAR-BOUND at production scale "
            f"(HBM util {avg_hbm*100:.1f}%, MXU util {avg_mxu*100:.1f}% — "
            "both below 10%). The kernel is dispatch- or scalar-op-bound. "
            "PRIORITIZE family I (host/dispatch overhead) and family J "
            "(work elimination, algebraic identities, fuse small VPU ops).")
    full = (f"## CURRENT BOTTLENECK (roofline analysis at production "
            f"shapes)\n\n{directive}\n\n### Per-shape roofline\n\n{md}\n")
    return full


class _RooflineInjectingClient:
    """Wraps an inner LLM client; prepends a fixed roofline hint to every
    user prompt. Implements the ``LLMClient`` Protocol."""

    def __init__(self, inner, hint: str) -> None:
        self.inner = inner
        self.hint = hint

    @property
    def model_id(self) -> str:
        return f"roofline+{self.inner.model_id}"

    def chat(self,
             *,
             system: str,
             user: str,
             max_tokens: int = 4096,
             temperature: float | None = None) -> str:
        injected = f"{self.hint}\n\n---\n\n{user}" if self.hint else user
        try:
            return self.inner.chat(system=system,
                                   user=injected,
                                   max_tokens=max_tokens,
                                   temperature=temperature)
        except TypeError:
            return self.inner.chat(system=system,
                                   user=injected,
                                   max_tokens=max_tokens)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--shapes",
                   default=DEFAULT_PRODUCTION_SHAPES,
                   help="Comma-separated production shape names from "
                   "cross_shape.PRODUCTION_SHAPES.")
    p.add_argument("--project", default=None)
    p.add_argument("--region", default="global")
    p.add_argument("--ensemble-models",
                   default="claude-opus-4-8,claude-opus-4-7")
    p.add_argument("--critic-model", default="claude-opus-4-7")
    p.add_argument("--generations", type=int, default=6)
    p.add_argument("--islands", type=int, default=2)
    p.add_argument("--island-candidates",
                   type=int,
                   default=2,
                   dest="candidates_per_island")
    p.add_argument("--bon", type=int, default=2)
    p.add_argument("--use-critic", action="store_true", default=True)
    p.add_argument("--bench-iters", type=int, default=4)
    p.add_argument("--warmup-iters", type=int, default=2)
    p.add_argument("--cosine-floor", type=float, default=0.9999)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--archive",
                   type=Path,
                   default=Path("/tmp/rpa_v3_production/archive.jsonl"))
    p.add_argument("--failure-log",
                   type=Path,
                   default=Path("/tmp/rpa_v3_production/failure_log.json"))
    p.add_argument("--example-pool",
                   type=Path,
                   default=Path("/tmp/claude_rpa_v3_example_pool.json"))
    p.add_argument(
        "--skip-roofline",
        action="store_true",
        help="Skip auto-roofline analysis (use stored hint if any).")
    p.add_argument("--hint-file",
                   type=Path,
                   default=None,
                   help="Use a stored roofline hint instead of running.")
    p.add_argument("--verbose", "-v", action="count", default=0)
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.WARNING - 10 * args.verbose,
                        format="%(asctime)s %(levelname)s %(message)s")
    project = args.project or os.environ.get("ANTHROPIC_VERTEX_PROJECT_ID")
    if not project:
        print("error: provide --project or set ANTHROPIC_VERTEX_PROJECT_ID",
              file=sys.stderr)
        return 2
    args.archive.parent.mkdir(parents=True, exist_ok=True)
    shape_names = [s.strip() for s in args.shapes.split(",") if s.strip()]

    # Build roofline hint
    if args.hint_file and args.hint_file.exists():
        hint = args.hint_file.read_text()
        print(f"Using stored roofline hint from {args.hint_file}",
              file=sys.stderr)
    elif args.skip_roofline:
        hint = ""
    else:
        print(f"Running roofline analysis on shapes: {shape_names} ...",
              file=sys.stderr)
        hint = _build_roofline_hint(shape_names)
        hint_path = args.archive.parent / "roofline_hint.md"
        hint_path.write_text(hint)
        print(f"Roofline hint saved to {hint_path}", file=sys.stderr)

    host = MultiShapeRpaV3Host(shape_names)
    print(f"Production host: {host.shape_names}", file=sys.stderr)

    if args.archive.exists():
        args.archive.unlink()
    archive = Archive(
        baseline=Genome.baseline(baseline_path=host.baseline_path),
        num_islands=args.islands,
        island_cap=max(4, args.candidates_per_island * args.generations + 2),
        persist_path=args.archive,
    )
    models = [m.strip() for m in args.ensemble_models.split(",") if m.strip()]
    clients = [
        VertexAnthropicClient(model=m,
                              project_id=project,
                              region=args.region,
                              max_retries=3,
                              timeout_sec=180) for m in models
    ]
    inner = EnsembleClient(clients, strategy="round_robin")
    # Wrap in the roofline-injecting client BEFORE BoN; so every BoN
    # candidate gets the hint and Claude's diversity is over candidate
    # mutations, not over whether it sees the hint.
    injected = _RooflineInjectingClient(inner, hint)
    critic = VertexAnthropicClient(model=args.critic_model,
                                   project_id=project,
                                   region=args.region,
                                   max_retries=3,
                                   timeout_sec=90)
    mutator = (BestOfNMutator(
        injected, n=args.bon, temperatures=[0.3, 0.5, 0.7, 0.9])
               if args.bon > 1 else injected)
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
    fl = FailureLog(persist_path=args.failure_log)
    pool = ExamplePool(persist_path=args.example_pool)
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
        f"Starting production-shape RPA v3 evolve "
        f"({args.generations} gens × {args.islands} islands × "
        f"{args.candidates_per_island} cands × BoN={args.bon} × "
        f"shapes={len(host._hosts)})",
        file=sys.stderr)
    final = orch.run()
    wall = time.time() - t0
    s = final.summary()
    print()
    print("=" * 78)
    print(f"Production-shape RPA v3 evolve finished in {wall:.1f}s")
    if s["best_genome_id"]:
        print(f"  baseline (sum across shapes): "
              f"{s['baseline_fitness_ns']/1e3:.2f} us")
        print(f"  best: {s['best_fitness_ns']/1e3:.2f} us "
              f"({s['speedup_vs_baseline']:.4f}x)")
    print(f"  genomes: {s['total_genomes']}; verified: {s['verified']}")
    print(f"  archive: {args.archive}")
    return 0 if s["best_genome_id"] else 1


if __name__ == "__main__":
    sys.exit(main())
