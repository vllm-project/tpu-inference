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
"""Real-TPU evolve demo on a synthetic Pallas matmul (no LLM API).

Drives the full orchestrator → verifier → bench loop against the synthetic
matmul in ``synthetic_matmul.py`` using the ``ProgrammaticMutator``. Every
candidate is a real source diff; every fitness is a real TPU measurement.
The reference is a pure-JAX ``jnp.matmul`` so the verifier catches the
``BLOCK_M``-not-divisible-by-shape and similar real failure modes.

Usage::

    python -m tools.kernel.evolve.examples.matmul_evolve \\
        --M 2048 --N 2048 --K 2048 \\
        --generations 4 --islands 2 --island-candidates 4 \\
        --archive /tmp/matmul_evolve.jsonl

The synthetic kernel's tunable literals (``BLOCK_M``, ``BLOCK_N``,
``ACCUM_DTYPE``) define the search space. The mutator enumerates fresh
``(name, value)`` pairs each turn; the archive de-dupes by genome hash so
no two trials repeat.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Any, Callable

import jax
import jax.numpy as jnp
import numpy as np

from tools.kernel.evolve.archive import Archive
from tools.kernel.evolve.genome import Genome
from tools.kernel.evolve.mutator.llm_client import StubClient
from tools.kernel.evolve.mutator.programmatic import (LiteralRewriteRule,
                                                      ProgrammaticMutator)
from tools.kernel.evolve.orchestrator import EvolutionConfig, Orchestrator
from tools.kernel.tuner.v1.verifier.anti_cheat import AntiCheatGuard

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[4]
_TARGET_REL = "tools/kernel/evolve/examples/synthetic_matmul.py"


class _MatmulOracle:
    """Pure-JAX reference; the verifier scores actual outputs against this."""

    def compute(self, inputs: dict[str, Any]) -> jax.Array:
        from tools.kernel.evolve.examples.synthetic_matmul import \
            matmul_reference
        return matmul_reference(inputs["x"], inputs["y"])

    def dtype_tolerance(self, dtype: Any) -> tuple[float, float]:
        bits = jnp.dtype(dtype).itemsize * 8
        # bf16 matmul → fp32 accumulate: a few ULPs of expected drift.
        # fp32 matmul → fp32 accumulate: ~ULP.
        # bf16 matmul → bf16 accumulate (the candidate-of-interest mutation):
        # noticeably more drift, but still tight enough that "wrong answer"
        # fails. Use the standard inference-grade tolerances from
        # ragged_paged_attention_kernel_v3_test.py.
        return ({32: (0.05, 0.05), 16: (0.1, 0.1)}.get(bits, (0.2, 0.2)))


class MatmulHost:
    """``KernelHost`` for the synthetic matmul."""
    kernel_name = "synthetic_matmul"
    kernel_symbol = "matmul"

    def __init__(self,
                 *,
                 M: int,
                 N: int,
                 K: int,
                 dtype=jnp.bfloat16,
                 seed: int = 0) -> None:
        rng = np.random.default_rng(seed)
        self._x = jnp.asarray(
            rng.normal(0, 1, size=(M, K)).astype(np.float32)).astype(dtype)
        self._y = jnp.asarray(
            rng.normal(0, 1, size=(K, N)).astype(np.float32)).astype(dtype)
        self.inputs = {"x": self._x, "y": self._y}

    @property
    def baseline_path(self) -> str:
        return _TARGET_REL

    def read_baseline_source(self) -> str:
        return (_REPO_ROOT / _TARGET_REL).read_text()

    def build_kernel_fn(self, module: Any) -> Callable[[], Any]:
        kernel = getattr(module, self.kernel_symbol)
        x = self._x
        y = self._y

        def fn():
            return kernel(jnp.copy(x), jnp.copy(y))

        return fn

    def get_oracle(self):
        return _MatmulOracle()

    def anti_cheat_skip_keys(self) -> tuple[str, ...]:
        return ()


def _build_mutator(*, baseline_path: str, seed: int) -> ProgrammaticMutator:
    """Rule library: real Pallas tuning levers for matmul.

    * Block sizes (M, N) sweep over powers of two that divide the test
      shape. The mutator detects and avoids duplicate proposals.
    * Accumulator dtype trades bf16/fp32 precision for speed.
    """
    return ProgrammaticMutator(
        baseline_path=baseline_path,
        literal_rules=[
            LiteralRewriteRule(
                name="BLOCK_M",
                values=["64", "128", "256", "512", "1024"],
                description="row tile size (rows per kernel block)",
            ),
            LiteralRewriteRule(
                name="BLOCK_N",
                values=["64", "128", "256", "512", "1024"],
                description="col tile size",
            ),
            LiteralRewriteRule(
                name="ACCUM_DTYPE",
                values=["jnp.float32", "jnp.bfloat16"],
                description="accumulator dtype",
            ),
        ],
        seed=seed,
        model_id="programmatic-matmul",
    )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--M", type=int, default=2048)
    p.add_argument("--N", type=int, default=2048)
    p.add_argument("--K", type=int, default=2048)
    p.add_argument("--generations", type=int, default=4)
    p.add_argument("--islands", type=int, default=2)
    p.add_argument("--island-candidates",
                   type=int,
                   default=3,
                   dest="candidates_per_island")
    p.add_argument("--bench-iters", type=int, default=10)
    p.add_argument("--warmup-iters", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--archive",
                   type=Path,
                   default=Path("/tmp/matmul_evolve.jsonl"))
    p.add_argument("--verbose", "-v", action="count", default=0)
    args = p.parse_args(argv)

    log_level = logging.WARNING - 10 * args.verbose
    logging.basicConfig(
        level=max(log_level, logging.DEBUG),
        format="%(asctime)s %(levelname)s %(name)s %(message)s")

    host = MatmulHost(M=args.M, N=args.N, K=args.K, seed=args.seed)
    args.archive.parent.mkdir(parents=True, exist_ok=True)
    if args.archive.exists():
        args.archive.unlink()  # start fresh each run

    archive = Archive(
        baseline=Genome.baseline(baseline_path=host.baseline_path),
        num_islands=args.islands,
        island_cap=max(8, args.candidates_per_island * args.generations + 2),
        persist_path=args.archive,
    )
    mutator = _build_mutator(baseline_path=host.baseline_path, seed=args.seed)
    critic = StubClient(
        ["VERDICT: likely_correct reason: deterministic rule."])
    config = EvolutionConfig(
        num_islands=args.islands,
        candidates_per_island_per_gen=args.candidates_per_island,
        generations=args.generations,
        migration_freq=max(1, args.generations // 2),
        migration_top_k=1,
        use_critic=False,  # deterministic mutator: critic adds no signal
        warmup_iters=args.warmup_iters,
        bench_iters=args.bench_iters,
        cosine_floor=0.999,  # bf16 accum is loose; cosine stays high.
        seed=args.seed,
    )
    orch = Orchestrator(
        host=host,
        mutator=mutator,
        archive=archive,
        config=config,
        critic_llm=critic,
        anti_cheat=AntiCheatGuard(),
    )

    print(f"Evolve matmul ({args.M}×{args.K}) @ ({args.K}×{args.N}) bfloat16",
          file=sys.stderr)
    print(
        f"  islands={args.islands} candidates/island/gen="
        f"{args.candidates_per_island} generations={args.generations}",
        file=sys.stderr)

    t0 = time.time()
    final = orch.run()
    wall = time.time() - t0

    summary = final.summary()
    print()
    print("=" * 78)
    print(
        f"Matmul evolve finished in {wall:.1f}s on {jax.devices()[0].platform}."
    )
    base_us = (summary['baseline_fitness_ns'] /
               1e3 if summary['baseline_fitness_ns'] else None)
    best_us = (summary['best_fitness_ns'] /
               1e3 if summary['best_fitness_ns'] else None)
    print(f"  baseline:  {base_us:7.2f} us")
    if best_us is not None:
        speed = summary['speedup_vs_baseline']
        print(f"  best:      {best_us:7.2f} us  ({speed:.3f}x)")
    print(f"  archive:   {summary['total_genomes']} genomes, "
          f"{summary['verified']} verified")
    print()
    print("Top 5 verified candidates (by latency):")
    top = sorted(final.all_finite(), key=lambda g: g.fitness)[:5]
    for g in top:
        m = g.metrics
        cos = m.get("cosine", float("nan"))
        print(f"  {g.short()} p50={m.get('p50_ns', 0)/1e3:.2f}us "
              f"cosine={cos:.6f}")
    print()
    print(f"Archive persisted at: {args.archive}")
    return 0 if best_us is not None else 1


if __name__ == "__main__":
    sys.exit(main())
