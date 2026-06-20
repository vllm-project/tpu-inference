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
"""Parallel candidate evaluator that fans out across TPU cores.

Drives N concurrent subprocesses, each pinned to a distinct TPU core via
``TPU_VISIBLE_DEVICES``. Each subprocess receives a serialized work item
(diff + host config) on stdin, runs the same evaluate-genome pipeline, and
returns an ``EvaluationResult`` on stdout. Coordinator merges results into
the archive serially.

Why subprocess-per-core rather than threads: JAX/XLA compilation state is
process-global. Two concurrent runs of two different kernel variants in the
same process would clobber each other's compile caches. Subprocesses give
clean isolation with negligible overhead (a few hundred ms per spawn,
amortized over multi-second kernel runs).
"""

from __future__ import annotations

import concurrent.futures
import dataclasses
import json
import logging
import os
import subprocess
import sys
import time
from typing import Any, Iterable

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class TpuVisibleDevices:
    """One slice of the local TPU mesh for a worker subprocess."""
    indices: tuple[int, ...]

    @property
    def env_value(self) -> str:
        return ",".join(str(i) for i in self.indices)


def auto_partition_devices(num_workers: int) -> list[TpuVisibleDevices]:
    """Detect locally-available TPU cores and split evenly across workers.

    Falls back to a single-worker partition that uses whatever JAX sees if
    the partition would be empty (e.g. on CPU smoke tests).
    """
    try:
        import jax
        n = len(jax.devices())
    except Exception:
        n = 1
    if n < num_workers:
        num_workers = max(1, n)
    if n == 0:
        return [TpuVisibleDevices(indices=())]
    per = max(1, n // num_workers)
    out = []
    cursor = 0
    for w in range(num_workers):
        end = cursor + per if w < num_workers - 1 else n
        out.append(TpuVisibleDevices(indices=tuple(range(cursor, end))))
        cursor = end
    return out


@dataclasses.dataclass
class ParallelWorkItem:
    """Serialized unit-of-work for a worker subprocess."""
    genome_id: str
    diff: str
    host_module: str  # importable host class qualified name
    host_kwargs: dict[str, Any]
    warmup: int
    iters: int


@dataclasses.dataclass
class ParallelResult:
    genome_id: str
    fitness: float
    status: str
    error: str | None
    p50_ns: int | None
    p95_ns: int | None
    mean_ns: int | None
    cosine: float | None
    max_abs_diff: float | None
    wall_time_s: float


class ParallelEvaluator:
    """Concurrent evaluator that fans out across N TPU partitions."""

    def __init__(
        self,
        *,
        num_workers: int = 4,
        partitions: list[TpuVisibleDevices] | None = None,
        worker_module: str = "tools.kernel.evolve.parallel.worker",
        python_exe: str | None = None,
        env_overrides: dict[str, str] | None = None,
    ) -> None:
        self.num_workers = num_workers
        self.partitions = partitions or auto_partition_devices(num_workers)
        self.worker_module = worker_module
        self.python_exe = python_exe or sys.executable
        self.env_overrides = dict(env_overrides or {})

    def evaluate_batch(
        self,
        items: Iterable[ParallelWorkItem],
    ) -> list[ParallelResult]:
        items = list(items)
        if not items:
            return []
        # Distribute items round-robin across partitions.
        with concurrent.futures.ThreadPoolExecutor(
                max_workers=len(self.partitions)) as pool:
            futures = []
            for i, item in enumerate(items):
                part = self.partitions[i % len(self.partitions)]
                futures.append(pool.submit(self._run_one, item, part))
            return [f.result() for f in futures]

    def _run_one(
        self,
        item: ParallelWorkItem,
        partition: TpuVisibleDevices,
    ) -> ParallelResult:
        env = os.environ.copy()
        env.update(self.env_overrides)
        if partition.indices:
            env["TPU_VISIBLE_DEVICES"] = partition.env_value
        payload = json.dumps(dataclasses.asdict(item))
        t0 = time.time()
        try:
            proc = subprocess.run(
                [self.python_exe, "-m", self.worker_module],
                input=payload,
                capture_output=True,
                text=True,
                env=env,
                timeout=600,
            )
        except subprocess.TimeoutExpired:
            return ParallelResult(
                genome_id=item.genome_id,
                fitness=float("inf"),
                status="TIMEOUT",
                error="worker subprocess timed out",
                p50_ns=None,
                p95_ns=None,
                mean_ns=None,
                cosine=None,
                max_abs_diff=None,
                wall_time_s=time.time() - t0,
            )
        wall = time.time() - t0
        if proc.returncode != 0:
            return ParallelResult(
                genome_id=item.genome_id,
                fitness=float("inf"),
                status="WORKER_FAIL",
                error=f"rc={proc.returncode}\n{proc.stderr[-1500:]}",
                p50_ns=None,
                p95_ns=None,
                mean_ns=None,
                cosine=None,
                max_abs_diff=None,
                wall_time_s=wall,
            )
        try:
            data = json.loads(proc.stdout.strip().splitlines()[-1])
        except (json.JSONDecodeError, IndexError) as err:
            return ParallelResult(
                genome_id=item.genome_id,
                fitness=float("inf"),
                status="PARSE_FAIL",
                error=f"could not parse worker output: {err}\n"
                f"stdout tail: {proc.stdout[-500:]}\n"
                f"stderr tail: {proc.stderr[-500:]}",
                p50_ns=None,
                p95_ns=None,
                mean_ns=None,
                cosine=None,
                max_abs_diff=None,
                wall_time_s=wall,
            )
        return ParallelResult(
            genome_id=item.genome_id,
            fitness=float(data.get("fitness", float("inf"))),
            status=data.get("status", "UNKNOWN"),
            error=data.get("error"),
            p50_ns=data.get("p50_ns"),
            p95_ns=data.get("p95_ns"),
            mean_ns=data.get("mean_ns"),
            cosine=data.get("cosine"),
            max_abs_diff=data.get("max_abs_diff"),
            wall_time_s=wall,
        )
