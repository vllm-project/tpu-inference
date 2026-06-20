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
"""Subprocess entrypoint for ``ParallelEvaluator``.

Reads a serialized ``ParallelWorkItem`` from stdin, resolves the host
class, runs the evaluator, prints a single JSON line on stdout.
"""

from __future__ import annotations

import importlib
import json
import math
import sys
import time
import traceback

from tools.kernel.evolve.evaluator import evaluate_genome
from tools.kernel.evolve.genome import Genome


def _resolve(qualified_name: str):
    """Resolve 'pkg.module:ClassName' or 'pkg.module.ClassName'."""
    if ":" in qualified_name:
        mod_name, attr = qualified_name.split(":", 1)
    else:
        parts = qualified_name.rsplit(".", 1)
        if len(parts) != 2:
            raise ValueError(f"bad qualified name {qualified_name!r}")
        mod_name, attr = parts
    mod = importlib.import_module(mod_name)
    obj = mod
    for piece in attr.split("."):
        obj = getattr(obj, piece)
    return obj


def main() -> int:
    payload = json.loads(sys.stdin.read())
    host_cls = _resolve(payload["host_module"])
    host = host_cls(**payload["host_kwargs"])
    genome = Genome.new(
        diff=payload["diff"],
        baseline_path=host.baseline_path,
        parent_ids=[],
        generation=0,
        island_id=0,
        created_at=time.time(),
    )
    genome.id = payload["genome_id"]
    try:
        result = evaluate_genome(
            genome,
            host,
            warmup=int(payload.get("warmup", 2)),
            iters=int(payload.get("iters", 10)),
        )
    except Exception as err:
        out = {
            "genome_id": payload["genome_id"],
            "fitness": math.inf,
            "status": "WORKER_EXCEPTION",
            "error": f"{err}\n{traceback.format_exc()[-1500:]}",
        }
        print(json.dumps(out))
        return 1
    out = {
        "genome_id": payload["genome_id"],
        "status": result.status.value,
        "fitness":
        (None if not math.isfinite(result.fitness) else result.fitness),
        "error": result.error,
    }
    if result.bench is not None:
        out["p50_ns"] = int(result.bench.p50_ns)
        out["p95_ns"] = int(result.bench.p95_ns)
        out["mean_ns"] = int(result.bench.mean_ns)
    if result.numerics is not None:
        out["cosine"] = result.numerics.cosine
        out["max_abs_diff"] = result.numerics.max_abs_diff
    if out.get("fitness") is None:
        out["fitness"] = "inf"
    print(json.dumps(out, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
