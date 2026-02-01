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
"""Benchmarking strategies for TPU kernels."""

import dataclasses
import enum
import statistics
import time
from typing import Any, Callable, Dict, List, Tuple

import jax

from tpu_inference.tools.autotune import utils

# Optional import for XProf
try:
    from jax.profiler import ProfileData
except ImportError:
    ProfileData = None


class BenchmarkMethod(str, enum.Enum):
    AMORTIZED = "amortized"
    XPROF = "xprof"


@dataclasses.dataclass
class BenchmarkResult:
    mean_time_ns: float
    std_time_ns: float
    min_time_ns: float
    max_time_ns: float
    samples_ns: List[float]
    metadata: Dict[str, Any]


def _benchmark_amortized(
    compiled_fn: Any,
    args: Tuple[Any, ...],
    num_iterations: int,
    num_repeats: int,
) -> List[float]:
    """Benchmarks using repeated amortized loops."""
    samples_ns = []

    # Prepare amortized function
    # Note: compiled_fn is already the amortized_jit function from the caller
    # but strictly speaking, the caller constructs the amortized function.
    # To keep it generic, we expect the caller to pass a function that
    # executes 'num_iterations' internally.

    # Warmup
    outputs = compiled_fn(*args)
    utils.block_until_ready(outputs)

    for _ in range(num_repeats):
        start = time.perf_counter_ns()
        outputs = compiled_fn(*args)
        utils.block_until_ready(outputs)
        end = time.perf_counter_ns()

        total_time_ns = end - start
        avg_per_iter = total_time_ns / num_iterations
        samples_ns.append(avg_per_iter)

    return samples_ns


def _benchmark_xprof(
    compiled_fn: Any,
    args: Tuple[Any, ...],
    num_repeats: int,
) -> List[float]:
    """Benchmarks using XProf traces."""
    samples_ns = []

    # Warmup
    outputs = compiled_fn(*args)
    utils.block_until_ready(outputs)

    for _ in range(num_repeats):
        try:
            with utils.XprofProfileSession() as session:
                outputs = compiled_fn(*args)
                utils.block_until_ready(outputs)

            # session.total_op_time is now in ns (from utils update)
            samples_ns.append(session.total_op_time)

        except Exception as e:
            # Fallback for profiling failures
            print(f"Profiling failed: {e}")
            pass

    return samples_ns


def benchmark_kernel(
    benchmark_fn: Callable,
    args: Tuple[Any, ...],
    num_iterations: int = 100,
    num_repeats: int = 5,
    method: BenchmarkMethod = BenchmarkMethod.AMORTIZED,
) -> BenchmarkResult:
    """Main entry point for benchmarking."""

    if method == BenchmarkMethod.AMORTIZED:
        # Wrap in amortized loop
        amortized_fn = utils.amortized_wrapper(benchmark_fn,
                                               n_iter=num_iterations)
        amortized_jit = jax.jit(amortized_fn)
        # Pass the JIT-ed amortized function
        samples_ns = _benchmark_amortized(amortized_jit, args, num_iterations,
                                          num_repeats)

    elif method == BenchmarkMethod.XPROF:
        # For XProf, we benchmark the single kernel call (un-amortized)
        # But we verify if the user passed an amortized fn or single fn.
        # Usually we want to trace the SINGLE execution to avoid huge traces.
        compiled_fn = jax.jit(benchmark_fn)
        samples_ns = _benchmark_xprof(compiled_fn, args, num_repeats)

    else:
        raise ValueError(f"Unknown benchmarking method: {method}")

    # Calculate stats
    mean_ns = statistics.mean(samples_ns)
    std_ns = statistics.stdev(samples_ns) if len(samples_ns) > 1 else 0.0
    min_ns = min(samples_ns)
    max_ns = max(samples_ns)

    return BenchmarkResult(mean_time_ns=mean_ns,
                           std_time_ns=std_ns,
                           min_time_ns=min_ns,
                           max_time_ns=max_ns,
                           samples_ns=samples_ns,
                           metadata={
                               "method": method.value,
                               "repeats": num_repeats,
                               "iterations": num_iterations
                           })
