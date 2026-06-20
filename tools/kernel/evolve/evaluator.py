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
"""Evaluate a Genome: apply diff -> import mutated module -> bench + verify.

Reuses the Phase 0 + 1 verifier/bench stack unchanged. The evaluator's job
is to wire the mutated kernel callable into a ``KernelHost`` (an interface
the example provides) so the candidate can be run with the host's pinned
inputs and its oracle/anti-cheat config.

The fitness signal is ``BenchResult.mean_ns`` on real TPU (or wall-clock
when no TPU is available, useful for CI smoke tests).
"""

from __future__ import annotations

import dataclasses
import math
import time
import traceback
from typing import Any, Callable, Protocol, runtime_checkable

from tools.kernel.evolve.genome import Genome, GenomeStatus
from tools.kernel.evolve.mutator.diff_applier import (apply_diff,
                                                      validate_python)
from tools.kernel.evolve.worktree import import_candidate_module
from tools.kernel.tuner.v1.bench.harness import BenchResult, measure
from tools.kernel.tuner.v1.verifier.anti_cheat import AntiCheatGuard
from tools.kernel.tuner.v1.verifier.numerics import (COSINE_FLOOR_DEFAULT,
                                                     NumericsReport,
                                                     check_many)


@runtime_checkable
class KernelHost(Protocol):
    """The example provides one of these to plug a mutated kernel into the loop.

    Responsibilities:
    * ``baseline_path`` — repo-relative source the diffs apply to.
    * ``read_baseline_source`` — returns the current file content.
    * ``build_kernel_fn(module)`` — given an imported module, return a
      zero-arg callable that runs the kernel with the host's pinned inputs.
    * ``inputs`` — the inputs dict (for the oracle and anti-cheat).
    * ``oracle`` — a ``ReferenceOracle``-like with ``compute(inputs)`` and
      ``dtype_tolerance(dtype)``.
    * ``kernel_symbol`` — the name of the kernel function inside the
      mutated module (e.g. ``"ragged_paged_attention"``).
    """

    @property
    def kernel_name(self) -> str:
        ...

    @property
    def baseline_path(self) -> str:
        ...

    @property
    def kernel_symbol(self) -> str:
        ...

    @property
    def inputs(self) -> dict[str, Any]:
        ...

    def read_baseline_source(self) -> str:
        ...

    def build_kernel_fn(self, module: Any) -> Callable[[], Any]:
        ...

    def get_oracle(self) -> Any:
        ...

    def anti_cheat_skip_keys(self) -> tuple[str, ...]:
        ...


@dataclasses.dataclass
class EvaluationResult:
    status: GenomeStatus
    fitness: float  # avg_latency_ns; math.inf on failure
    bench: BenchResult | None = None
    numerics: NumericsReport | None = None
    error: str | None = None
    mutated_source_preview: str | None = None


def evaluate_genome(
    genome: Genome,
    host: KernelHost,
    *,
    warmup: int = 2,
    iters: int = 10,
    cosine_floor: float = COSINE_FLOOR_DEFAULT,
    anti_cheat: AntiCheatGuard | None = None,
) -> EvaluationResult:
    """Run the candidate end-to-end.

    Pipeline:
    1. Apply the genome's diff to the host's baseline source.
    2. Validate the result parses as Python.
    3. Import the mutated module in an isolated workspace.
    4. Build the zero-arg kernel callable; run through ``bench.measure``.
    5. Run the verifier (numerics + anti-cheat) against the host's oracle.
    """
    started = time.time()
    baseline_source = host.read_baseline_source()

    # ---- Baseline shortcut: empty diff means "evaluate the baseline as-is".
    if not genome.diff.strip():
        return _evaluate_baseline(host,
                                  warmup=warmup,
                                  iters=iters,
                                  cosine_floor=cosine_floor,
                                  anti_cheat=anti_cheat,
                                  started=started)

    # ---- Apply diff
    diff_result = apply_diff(baseline_source, genome.diff)
    if not diff_result.success or diff_result.new_source is None:
        return EvaluationResult(
            status=GenomeStatus.FAILED_DIFF,
            fitness=math.inf,
            error=diff_result.error or "apply_diff returned no source",
        )
    ok, parse_err = validate_python(diff_result.new_source)
    if not ok:
        return EvaluationResult(
            status=GenomeStatus.FAILED_DIFF,
            fitness=math.inf,
            error=parse_err,
            mutated_source_preview=_preview(diff_result.new_source),
        )

    # ---- Import + build kernel fn + bench
    try:
        with import_candidate_module(diff_result.new_source,
                                     name_hint=host.kernel_name) as module:
            if not hasattr(module, host.kernel_symbol):
                return EvaluationResult(
                    status=GenomeStatus.FAILED_COMPILE,
                    fitness=math.inf,
                    error=(f"mutated module missing kernel symbol "
                           f"{host.kernel_symbol!r}"),
                    mutated_source_preview=_preview(diff_result.new_source),
                )
            try:
                kernel_fn = host.build_kernel_fn(module)
            except Exception as err:
                return EvaluationResult(
                    status=GenomeStatus.FAILED_COMPILE,
                    fitness=math.inf,
                    error=f"build_kernel_fn raised: {err}",
                    mutated_source_preview=_preview(diff_result.new_source),
                )
            try:
                bench = measure(kernel_fn, warmup=warmup, iters=iters)
            except Exception as err:
                msg = str(err)
                return EvaluationResult(
                    status=GenomeStatus.FAILED_RUN,
                    fitness=math.inf,
                    error=f"bench raised: {msg[:300]}",
                    mutated_source_preview=_preview(diff_result.new_source),
                )
    except Exception as err:
        return EvaluationResult(
            status=GenomeStatus.FAILED_COMPILE,
            fitness=math.inf,
            error=(f"module import raised: {err}\n"
                   f"{traceback.format_exc()[-500:]}"),
            mutated_source_preview=_preview(diff_result.new_source),
        )

    # ---- Verifier
    actual_outputs = bench.output if isinstance(bench.output,
                                                (tuple,
                                                 list)) else (bench.output, )
    oracle = host.get_oracle()
    reference = oracle.compute(host.inputs)
    if not isinstance(reference, (tuple, list)):
        reference = (reference, )
    atol, rtol = oracle.dtype_tolerance(actual_outputs[0].dtype)
    numerics = check_many(actual_outputs,
                          reference,
                          atol=atol,
                          rtol=rtol,
                          cosine_floor=cosine_floor)
    if not numerics.passed:
        return EvaluationResult(
            status=GenomeStatus.FAILED_NUMERICS,
            fitness=math.inf,
            bench=bench,
            numerics=numerics,
            error=numerics.reason,
            mutated_source_preview=_preview(diff_result.new_source),
        )

    if anti_cheat is not None:
        ac = anti_cheat.inspect(actual_outputs[0], host.inputs)
        if not ac.passed:
            return EvaluationResult(
                status=GenomeStatus.FAILED_ANTI_CHEAT,
                fitness=math.inf,
                bench=bench,
                numerics=numerics,
                error=ac.reason,
                mutated_source_preview=_preview(diff_result.new_source),
            )

    return EvaluationResult(
        status=GenomeStatus.VERIFIED,
        fitness=float(bench.mean_ns),
        bench=bench,
        numerics=numerics,
        mutated_source_preview=_preview(diff_result.new_source),
    )


def _evaluate_baseline(
    host: KernelHost,
    *,
    warmup: int,
    iters: int,
    cosine_floor: float,
    anti_cheat: AntiCheatGuard | None,
    started: float,
) -> EvaluationResult:
    """Evaluate the un-mutated baseline by importing the file directly."""
    src = host.read_baseline_source()
    try:
        with import_candidate_module(
                src, name_hint=f"{host.kernel_name}_baseline") as module:
            kernel_fn = host.build_kernel_fn(module)
            bench = measure(kernel_fn, warmup=warmup, iters=iters)
    except Exception as err:
        return EvaluationResult(
            status=GenomeStatus.FAILED_RUN,
            fitness=math.inf,
            error=f"baseline run raised: {err}",
        )
    actual_outputs = bench.output if isinstance(bench.output,
                                                (tuple,
                                                 list)) else (bench.output, )
    oracle = host.get_oracle()
    reference = oracle.compute(host.inputs)
    if not isinstance(reference, (tuple, list)):
        reference = (reference, )
    atol, rtol = oracle.dtype_tolerance(actual_outputs[0].dtype)
    numerics = check_many(actual_outputs,
                          reference,
                          atol=atol,
                          rtol=rtol,
                          cosine_floor=cosine_floor)
    if not numerics.passed:
        return EvaluationResult(
            status=GenomeStatus.FAILED_NUMERICS,
            fitness=math.inf,
            bench=bench,
            numerics=numerics,
            error=("baseline failed verifier — fix the host's inputs/oracle "
                   "before running evolve: " + (numerics.reason or "")),
        )
    return EvaluationResult(
        status=GenomeStatus.VERIFIED,
        fitness=float(bench.mean_ns),
        bench=bench,
        numerics=numerics,
    )


def _preview(source: str, n_lines: int = 40) -> str:
    return "\n".join(source.splitlines()[:n_lines])
