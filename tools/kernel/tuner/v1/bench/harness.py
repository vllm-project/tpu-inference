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
"""Reproducible benchmark harness for a single kernel call.

Differences from the loop in ``KernelTunerBase.measure_latency``:

* ``block_until_ready`` runs over every leaf of the kernel output (the
  existing tuner only blocks on the first element of the returned tuple).
* The cold-start iteration is excluded from the timing summary.
* Reports both p50 and p95, not just mean.
* Returns the last iteration's output to the verifier so the numerics gate
  can run without a second kernel call.
"""

import dataclasses
import time
from typing import Any, Callable

import jax


@dataclasses.dataclass
class BenchResult:
    """Per-trial latency summary plus the last output for verification."""
    p50_ns: int
    p95_ns: int
    mean_ns: int
    iters: int
    cold_start_excluded: bool
    output: Any


def block_all(x: Any) -> None:
    """Recursively call ``jax.block_until_ready`` on every leaf of ``x``."""
    if isinstance(x, (tuple, list)):
        for elem in x:
            block_all(elem)
    elif isinstance(x, dict):
        for elem in x.values():
            block_all(elem)
    else:
        jax.block_until_ready(x)


def _percentile_ns(xs: list[int], pct: float) -> int:
    if not xs:
        return 0
    s = sorted(xs)
    k = max(0, min(len(s) - 1, int(round((pct / 100.0) * (len(s) - 1)))))
    return s[k]


def measure(
    kernel_fn: Callable[[], Any],
    *,
    warmup: int = 3,
    iters: int = 20,
    exclude_cold_start: bool = True,
) -> BenchResult:
    """Run ``kernel_fn()`` ``warmup`` + ``iters`` times and return timings.

    Args:
        kernel_fn: zero-arg callable that runs the kernel. The caller pins
            the inputs (capture them in a closure) — this lets each candidate
            share compiled-once inputs while the harness owns timing.
        warmup: untimed warmup iterations that still block on outputs to flush
            any deferred work.
        iters: timed iterations.
        exclude_cold_start: drop the first timed iter from the summary
            (covers residual JIT cost when ``warmup`` is small).

    Returns:
        ``BenchResult`` carrying p50 / p95 / mean and the last output.
    """
    if iters < 1:
        raise ValueError(f"iters must be >= 1, got {iters}")
    for _ in range(warmup):
        out = kernel_fn()
        block_all(out)

    timings_ns: list[int] = []
    last_out: Any = None
    for _ in range(iters):
        t0 = time.perf_counter_ns()
        out = kernel_fn()
        block_all(out)
        t1 = time.perf_counter_ns()
        timings_ns.append(t1 - t0)
        last_out = out
    excluded = exclude_cold_start and len(timings_ns) > 1
    sample = timings_ns[1:] if excluded else timings_ns
    mean_ns = sum(sample) // len(sample)
    return BenchResult(
        p50_ns=_percentile_ns(sample, 50),
        p95_ns=_percentile_ns(sample, 95),
        mean_ns=mean_ns,
        iters=len(sample),
        cold_start_excluded=excluded,
        output=last_out,
    )
