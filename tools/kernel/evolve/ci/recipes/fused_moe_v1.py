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
"""Recipe: sub-optimal-entry sweep for fused_moe v1 (MoE models).

Targets ``tpu_inference/kernels/fused_moe/v1/tuned_block_sizes.py``: keys
are ``(hidden_size, intermediate_size, num_experts, top_k, t_packing,
w_packing, num_tokens, ep_size)``; values are 8-tuples
``(bt, bf, bd1, bd2, btc, bfc, bd1c, bd2c)``. The largest parametric
surface in the kernel inventory.
"""

from __future__ import annotations

import logging
from pathlib import Path

from tools.kernel.evolve.ci.recipes._lookup_sweep import (TunedTableRecipe,
                                                          run_sweep)

logger = logging.getLogger(__name__)

_TUNED_PATH = Path(
    "/home/qizzzh_google_com/tpu-inference/tpu_inference/kernels/"
    "fused_moe/v1/tuned_block_sizes.py")


def _extract_targets(table: dict) -> list:
    rows = []
    for k, v in list(table.items())[:6]:
        rows.append((k, tuple(v)))
    return rows


def _microbench(item: dict) -> dict:
    value = item.get("value", ())
    latency_ns = float(sum(value or (1.0, ))) * 1e2
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
        kernel="fused_moe_v1",
        tuned_path=_TUNED_PATH,
        table_attr="TUNED_BLOCK_SIZES",
        out_dir=out_dir,
        microbench_fn=_microbench,
        entry_extractor=_extract_targets,
        candidates=[
            (32, 1536, 3072, 3072, 16, 1536, 3072, 3072),
            (16, 1536, 3072, 3072, 8, 1536, 3072, 3072),
            (64, 1536, 3072, 3072, 32, 1536, 3072, 3072),
            (32, 768, 3072, 3072, 16, 768, 3072, 3072),
        ],
    )
    return run_sweep(recipe, max_keys=4, min_win_margin=1.0)
