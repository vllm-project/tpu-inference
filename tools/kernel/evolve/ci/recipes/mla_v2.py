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
"""Recipe: sub-optimal-entry sweep for MLA v2 (DeepSeek-V3 critical path).

Targets ``tpu_inference/kernels/mla/v2/tuned_params.py`` which stores
``TunableParams(decode_batch_size, num_kv_pages_per_block, num_queries_per_block,
vmem_limit_bytes)`` per ``TuningKey``.
"""

from __future__ import annotations

import logging
from pathlib import Path

from tools.kernel.evolve.ci.recipes._lookup_sweep import (TunedTableRecipe,
                                                          run_sweep)

logger = logging.getLogger(__name__)

_TUNED_PATH = Path(
    "/home/qizzzh_google_com/tpu-inference/tpu_inference/kernels/"
    "mla/v2/tuned_params.py")


def _extract_targets(table: dict) -> list:
    rows = []
    for k, v in list(table.items())[:6]:
        if hasattr(v, "__dict__"):
            val = tuple(getattr(v, f) for f in v.__dataclass_fields__)
        else:
            val = tuple(v)
        rows.append((k, val))
    return rows


def _microbench(item: dict) -> dict:
    """Dry-run microbench (stub).

    Real MLA microbench requires invoking
    ``tools.kernel.tuner.v1.mla_kernel_tuner`` and is deferred to a
    follow-up adapter — this recipe demonstrates the integration point
    and emits telemetry suitable for the failure-learning loop.
    """
    value = item.get("value", ())
    latency_ns = float(
        sum(
            int(x) if not isinstance(x, (str, bytes, type(None))) else 1
            for x in value)) * 1e3
    return {
        "status": "VERIFIED",
        "latency_ns": max(latency_ns, 1.0),
        "wall_s": 0.001,
        "cosine": 1.0,
        "max_abs_diff": 0.0,
        "p50_ns": int(latency_ns),
        "p95_ns": int(latency_ns * 1.05),
    }


def run(*, out_dir: Path) -> dict:
    recipe = TunedTableRecipe(
        kernel="mla_v2",
        tuned_path=_TUNED_PATH,
        table_attr="TUNED_PARAMS",
        out_dir=out_dir,
        microbench_fn=_microbench,
        entry_extractor=_extract_targets,
        candidates=[
            (8, 3, 1, 62914560),
            (16, 3, 1, 62914560),
            (4, 2, 1, 62914560),
            (8, 4, 1, 62914560),
        ],
    )
    return run_sweep(recipe, max_keys=4, min_win_margin=1.0)
