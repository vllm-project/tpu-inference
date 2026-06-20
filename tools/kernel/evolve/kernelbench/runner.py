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
"""KernelBench-TPU runner.

Each task's *baseline candidate* is the JAX reference itself running under
``jax.jit``. The evolution loop is *not* required for the benchmark — we
report each task's measured latency and verified correctness vs. the
oracle to produce a comparable ``fast_p`` scorecard.

``fast_p`` = (#tasks where candidate is correct AND ≥ p× faster than
baseline) / total. Stanford's report uses p=1 (any speedup) and p=2.
"""

from __future__ import annotations

import dataclasses
import logging
from typing import Callable

import jax
import jax.numpy as jnp

from tools.kernel.evolve.kernelbench.tasks import TASKS, KernelTask
from tools.kernel.tuner.v1.bench.harness import block_all, measure
from tools.kernel.tuner.v1.verifier.numerics import check_many

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class KernelBenchResult:
    task_name: str
    level: int
    baseline_p50_ns: int
    candidate_p50_ns: int | None
    speedup: float | None
    correct: bool
    cosine: float | None
    max_abs_diff: float | None
    error: str | None = None


def _materialize(inputs: tuple) -> tuple:
    """Move generator outputs to device, block, return."""
    out = jax.tree_util.tree_map(jnp.asarray, inputs)
    block_all(out)
    return out


def _bench_callable(fn, *, warmup: int = 3, iters: int = 10):
    """Returns ``(p50_ns, output)`` for a zero-arg callable."""
    res = measure(fn, warmup=warmup, iters=iters)
    return res.p50_ns, res.output


class TpuKernelBench:
    """Runs the KernelBench task suite on TPU."""

    def __init__(self,
                 *,
                 seed: int = 0,
                 warmup: int = 3,
                 iters: int = 10) -> None:
        self.rng_seed = seed
        self.warmup = warmup
        self.iters = iters

    def _bench_baseline(self,
                        task: KernelTask) -> tuple[int, jnp.ndarray | tuple]:
        key = jax.random.PRNGKey(self.rng_seed)
        inputs = _materialize(task.make_inputs(key))
        ref_jit = jax.jit(task.reference)

        def fn():
            return ref_jit(*inputs)

        p50_ns, out = _bench_callable(fn, warmup=self.warmup, iters=self.iters)
        return p50_ns, out

    def _bench_candidate(
        self,
        task: KernelTask,
        candidate_fn: Callable,
    ) -> tuple[int | None, jnp.ndarray | tuple | None, str | None]:
        key = jax.random.PRNGKey(self.rng_seed)
        inputs = _materialize(task.make_inputs(key))
        try:
            cand_jit = jax.jit(candidate_fn)

            def fn():
                return cand_jit(*inputs)

            p50_ns, out = _bench_callable(fn,
                                          warmup=self.warmup,
                                          iters=self.iters)
            return p50_ns, out, None
        except Exception as err:
            return None, None, f"{err!s}"

    def verify(
        self,
        task: KernelTask,
        candidate_out,
        reference_out,
    ) -> tuple[bool, float | None, float | None]:
        ca = candidate_out if isinstance(candidate_out, (tuple, list)) \
            else (candidate_out,)
        ra = reference_out if isinstance(reference_out, (tuple, list)) \
            else (reference_out,)
        report = check_many(ca,
                            ra,
                            atol=task.atol,
                            rtol=task.rtol,
                            cosine_floor=task.cosine_floor)
        return report.passed, report.cosine, report.max_abs_diff

    def run_task(self,
                 task: KernelTask,
                 candidate_fn: Callable | None = None) -> KernelBenchResult:
        """Benchmark a single task. ``candidate_fn=None`` means use reference
        itself (the trivial case — speedup ≈ 1.0)."""
        bp50, bout = self._bench_baseline(task)
        if candidate_fn is None:
            return KernelBenchResult(task_name=task.name,
                                     level=task.level,
                                     baseline_p50_ns=int(bp50),
                                     candidate_p50_ns=int(bp50),
                                     speedup=1.0,
                                     correct=True,
                                     cosine=1.0,
                                     max_abs_diff=0.0)
        cp50, cout, err = self._bench_candidate(task, candidate_fn)
        if cp50 is None or cout is None:
            return KernelBenchResult(task_name=task.name,
                                     level=task.level,
                                     baseline_p50_ns=int(bp50),
                                     candidate_p50_ns=None,
                                     speedup=None,
                                     correct=False,
                                     cosine=None,
                                     max_abs_diff=None,
                                     error=err)
        passed, cosine, mad = self.verify(task, cout, bout)
        speedup = bp50 / cp50 if cp50 > 0 else None
        return KernelBenchResult(task_name=task.name,
                                 level=task.level,
                                 baseline_p50_ns=int(bp50),
                                 candidate_p50_ns=int(cp50),
                                 speedup=speedup,
                                 correct=passed,
                                 cosine=cosine,
                                 max_abs_diff=mad,
                                 error=None if passed else "verifier rejected")


def run_subset(
    *,
    tasks: list[KernelTask] | None = None,
    candidates: dict[str, Callable] | None = None,
    seed: int = 0,
    warmup: int = 3,
    iters: int = 10,
) -> list[KernelBenchResult]:
    """Run a subset of tasks. ``candidates`` maps task.name → kernel callable.

    A task without an entry in ``candidates`` is benchmarked against
    itself (speedup=1.0) so the reference latency lands in the result for
    downstream comparison.
    """
    tasks = tasks or TASKS
    candidates = candidates or {}
    bench = TpuKernelBench(seed=seed, warmup=warmup, iters=iters)
    return [bench.run_task(t, candidates.get(t.name)) for t in tasks]


def fast_p(results: list[KernelBenchResult], p: float = 1.0) -> float:
    """Stanford-style ``fast_p`` score: fraction of correct ≥ p× wins."""
    if not results:
        return 0.0
    wins = sum(1 for r in results
               if r.correct and r.speedup is not None and r.speedup >= p)
    return wins / len(results)
