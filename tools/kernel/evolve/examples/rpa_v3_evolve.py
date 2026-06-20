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
"""End-to-end demo: evolve ``ragged_paged_attention`` v3.

Usage::

    # With the Anthropic API (live LLM mutation):
    export ANTHROPIC_API_KEY=...
    python -m tools.kernel.evolve.examples.rpa_v3_evolve \\
        --mutator anthropic --generations 5 --island-candidates 4 \\
        --archive /tmp/evolve_rpa_v3.jsonl

    # With the deterministic stub (no API needed; for CI / smoke tests):
    python -m tools.kernel.evolve.examples.rpa_v3_evolve \\
        --mutator stub --generations 2 --island-candidates 2 \\
        --archive /tmp/evolve_rpa_v3_stub.jsonl

The host reuses the same workload that
``tools/kernel/tuner/v1/rpa_v3_kernel_tuner.py`` ships with as the default
(test-derived three chunked-prefill sequences), which is known to match the
eager reference within fp8/bf16 tolerance.
"""

from __future__ import annotations

import argparse
import logging
import sys
import textwrap
import time
from pathlib import Path
from typing import Any, Callable

import jax.numpy as jnp

from tools.kernel.evolve.archive import Archive
from tools.kernel.evolve.genome import Genome
from tools.kernel.evolve.mutator.llm_client import (AnthropicClient,
                                                    CachingClient, LLMClient,
                                                    StubClient)
from tools.kernel.evolve.orchestrator import EvolutionConfig, Orchestrator
from tools.kernel.tuner.v1.common.kernel_tuner_base import RunConfig
from tools.kernel.tuner.v1.rpa_v3_kernel_tuner import (VMEM_LIMIT_BYTES,
                                                       RpaV3KernelTuner)
from tools.kernel.tuner.v1.verifier.anti_cheat import AntiCheatGuard
from tools.kernel.tuner.v1.verifier.reference_oracle import RpaV3Oracle

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[4]
_RPA_V3_SOURCE_REL = "tpu_inference/kernels/ragged_paged_attention/v3/kernel.py"


class RpaV3Host:
    """``KernelHost`` impl for ``ragged_paged_attention`` v3.

    Encapsulates inputs, oracle, kernel symbol, and the per-call closure
    that the evaluator times.
    """
    kernel_name = "ragged_paged_attention_v3"
    kernel_symbol = "ragged_paged_attention"

    def __init__(self, tuner: RpaV3KernelTuner) -> None:
        self._tuner = tuner
        tk = tuner.get_default_tuning_key()
        self._tuning_key = tk
        self.inputs = tuner.generate_inputs(tk)

    @property
    def baseline_path(self) -> str:
        return _RPA_V3_SOURCE_REL

    def read_baseline_source(self) -> str:
        return (_REPO_ROOT / _RPA_V3_SOURCE_REL).read_text()

    def build_kernel_fn(self, module: Any) -> Callable[[], Any]:
        kernel = getattr(module, self.kernel_symbol)
        tk = self._tuning_key
        q = self.inputs["q"]
        k = self.inputs["k"]
        v = self.inputs["v"]
        kv_cache = self.inputs["kv_cache"]
        kv_lens = self.inputs["kv_lens"]
        page_indices = self.inputs["page_indices"]
        cu_q_lens = self.inputs["cu_q_lens"]
        distribution = self.inputs["distribution"]

        def fn():
            # The kernel returns (output, updated_kv_cache). The kv_cache has
            # NaN padding by design (mirrors the v3 unit test setup); the
            # verifier ignores it via the oracle returning only the output.
            out, _ = kernel(
                jnp.copy(q),
                jnp.copy(k),
                jnp.copy(v),
                jnp.copy(kv_cache),
                kv_lens,
                page_indices,
                cu_q_lens,
                distribution,
                sliding_window=tk.sliding_window,
                vmem_limit_bytes=VMEM_LIMIT_BYTES,
            )
            return out

        return fn

    def get_oracle(self):
        return RpaV3Oracle(semantic_kwargs={
            "sliding_window": self._tuning_key.sliding_window,
        })

    def anti_cheat_skip_keys(self) -> tuple[str, ...]:
        # kv_cache is legitimately the same tensor going in and (modified)
        # coming out — the kernel returns (output, updated_kv_cache). Don't
        # let the anti-cheat detector flag this as input-aliasing.
        return ("kv_cache", )


# Stub mutator responses — used when ``--mutator stub`` is set.
# Each is a complete LLM "response" string with a hypothesis sentence and a
# fenced diff. The first one is a no-op comment-only mutation that proves
# the pipeline can apply, compile, run, and verify a diff against the real
# v3 kernel. The second is intentionally syntactically broken so the demo
# exercises the FAILED_DIFF classification. The third is a numerics-correct
# whitespace tweak.
_STUB_DIFFS = [
    textwrap.dedent('''\
        Hypothesis: comment-only mutation to prove the diff pipeline plumbs end-to-end
        against the real v3 kernel without changing semantics.

        ```diff
        --- a/tpu_inference/kernels/ragged_paged_attention/v3/kernel.py
        +++ b/tpu_inference/kernels/ragged_paged_attention/v3/kernel.py
        @@ -66,1 +66,1 @@
        -def ref_ragged_paged_attention(
        +def ref_ragged_paged_attention(  # evolve: candidate-A
        ```
        '''),
    textwrap.dedent('''\
        Hypothesis: intentionally invalid mutation to exercise FAILED_DIFF.

        ```diff
        --- a/tpu_inference/kernels/ragged_paged_attention/v3/kernel.py
        +++ b/tpu_inference/kernels/ragged_paged_attention/v3/kernel.py
        @@ -1,1 +1,1 @@
        -A_LINE_THAT_DOES_NOT_EXIST_IN_THE_BASELINE
        +nope
        ```
        '''),
    textwrap.dedent('''\
        Hypothesis: a second harmless comment elsewhere so we sample multiple verified
        descendants in the archive.

        ```diff
        --- a/tpu_inference/kernels/ragged_paged_attention/v3/kernel.py
        +++ b/tpu_inference/kernels/ragged_paged_attention/v3/kernel.py
        @@ -1586,1 +1586,1 @@
        -def ragged_paged_attention(
        +def ragged_paged_attention(  # evolve: candidate-B
        ```
        '''),
]

_STUB_CRITIC = "VERDICT: likely_correct reason: cosmetic-only change."


def _make_mutator(name: str, *, cache_dir: Path | None) -> LLMClient:
    if name == "stub":
        return StubClient(_STUB_DIFFS, model_id="stub-rpa-v3")
    if name == "anthropic":
        client: LLMClient = AnthropicClient()
        if cache_dir is not None:
            cache = cache_dir / "anthropic_cache.jsonl"
            client = CachingClient(client, cache_path=cache)
        return client
    raise ValueError(f"unknown mutator backend: {name!r}")


def _make_critic(name: str, *, cache_dir: Path | None) -> LLMClient:
    if name == "stub":
        return StubClient([_STUB_CRITIC], model_id="stub-critic")
    if name == "anthropic":
        client: LLMClient = AnthropicClient(model="claude-haiku-4-5-20251001")
        if cache_dir is not None:
            cache = cache_dir / "anthropic_critic_cache.jsonl"
            client = CachingClient(client, cache_path=cache)
        return client
    if name == "none":
        return StubClient([_STUB_CRITIC])
    raise ValueError(f"unknown critic backend: {name!r}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--mutator", choices=("stub", "anthropic"), default="stub")
    p.add_argument(
        "--critic",
        choices=("stub", "anthropic", "none"),
        default="stub",
        help="Adversarial pre-filter LLM. 'none' disables the critic.")
    p.add_argument("--generations", type=int, default=2)
    p.add_argument("--islands", type=int, default=2)
    p.add_argument("--island-candidates",
                   type=int,
                   default=2,
                   dest="candidates_per_island")
    p.add_argument("--archive",
                   type=Path,
                   default=Path("/tmp/evolve_rpa_v3.jsonl"))
    p.add_argument("--cache-dir",
                   type=Path,
                   default=Path("/tmp/evolve_rpa_v3_cache"),
                   help="LLM response cache directory (anthropic only).")
    p.add_argument("--bench-iters", type=int, default=10)
    p.add_argument("--warmup-iters", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--verbose", "-v", action="count", default=0)
    args = p.parse_args(argv)

    log_level = logging.WARNING - 10 * args.verbose
    logging.basicConfig(
        level=max(log_level, logging.DEBUG),
        format="%(asctime)s %(levelname)s %(name)s %(message)s")

    # Build the tuner host. Reuses the test-derived workload that's
    # numerically verified to match the eager reference.
    run_config = RunConfig(
        case_set_id="evolve_demo",
        run_id="r0",
        case_set_desc="evolve demo",
        tpu_version="tpu6e",
        tpu_cores=1,
        tpu_queue_multi="tpu_v6e_queue",
        run_locally=True,
        max_execution_minutes=15,
    )
    tuner = RpaV3KernelTuner(run_config=run_config)
    host = RpaV3Host(tuner)

    archive_dir = args.archive.parent
    archive_dir.mkdir(parents=True, exist_ok=True)
    if args.cache_dir is not None:
        args.cache_dir.mkdir(parents=True, exist_ok=True)

    mutator = _make_mutator(args.mutator, cache_dir=args.cache_dir)
    critic = _make_critic(args.critic, cache_dir=args.cache_dir)

    archive = Archive(
        baseline=Genome.baseline(baseline_path=host.baseline_path),
        num_islands=args.islands,
        island_cap=max(8, args.candidates_per_island * args.generations + 2),
        persist_path=args.archive,
    )
    config = EvolutionConfig(
        num_islands=args.islands,
        candidates_per_island_per_gen=args.candidates_per_island,
        generations=args.generations,
        migration_freq=max(1, args.generations // 2),
        migration_top_k=1,
        use_critic=(args.critic != "none"),
        warmup_iters=args.warmup_iters,
        bench_iters=args.bench_iters,
        seed=args.seed,
    )
    orch = Orchestrator(
        host=host,
        mutator=mutator,
        archive=archive,
        config=config,
        critic_llm=critic,
        anti_cheat=AntiCheatGuard(input_skip_keys=host.anti_cheat_skip_keys()),
    )

    t0 = time.time()
    print(f"Starting evolve on {host.kernel_name} ({host.baseline_path})",
          file=sys.stderr)
    print(
        f"  islands={args.islands}, candidates/island/gen={args.candidates_per_island}, "
        f"generations={args.generations}, mutator={args.mutator}, critic={args.critic}",
        file=sys.stderr)
    final = orch.run()
    wall = time.time() - t0

    summary = final.summary()
    print()
    print("=" * 78)
    print(f"Evolve done in {wall:.1f}s.")
    print(
        f"  baseline (id=baseline): "
        f"{summary['baseline_fitness_ns'] / 1e3 if summary['baseline_fitness_ns'] else 'n/a':.2f} us avg"
    )
    print(f"  total genomes: {summary['total_genomes']}, "
          f"verified: {summary['verified']}")
    if summary["best_genome_id"] is not None:
        speed = summary["speedup_vs_baseline"]
        print(f"  best:  {summary['best_genome_id']}   "
              f"{summary['best_fitness_ns'] / 1e3:.2f} us "
              f"({speed:.3f}x baseline)")
    print()
    print("Top 5 verified candidates:")
    top = sorted(final.all_finite(), key=lambda g: g.fitness)[:5]
    if not top:
        print("  (none — all candidates failed)")
    for g in top:
        print(f"  {g.short()}")
    print()
    print(f"Archive persisted at: {args.archive}")
    return 0 if summary["best_genome_id"] is not None else 1


if __name__ == "__main__":
    sys.exit(main())
