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
"""Recipe: real-TPU sub-optimal-entry sweep for ``quantized_matmul``.

Targets ``tpu_inference/kernels/quantized_matmul/tuned_block_sizes.py``.
Each ``TunedKey(tpu_version, n_batch, n_out, n_in, x_q_dtype, w_q_dtype)``
maps to a ``TunedValue(batch_block_size, out_block_size, in_block_size,
n_lane_multiplier)``.

The microbench is a real Pallas run: we materialize inputs at the keyed
shape, dispatch the production kernel with the tuned value plugged in,
and time via ``bench.harness.measure``. Pure-JAX reference is used for
verification.
"""

from __future__ import annotations

import logging
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from tools.kernel.evolve.ci.recipes._lookup_sweep import (TunedTableRecipe,
                                                          run_sweep)
from tools.kernel.tuner.v1.bench.harness import measure
from tools.kernel.tuner.v1.verifier.numerics import check_many

logger = logging.getLogger(__name__)

_TUNED_PATH = Path(
    "/home/qizzzh_google_com/tpu-inference/tpu_inference/kernels/"
    "quantized_matmul/tuned_block_sizes.py")


def _extract_targets(table: dict) -> list:
    """Yield ``(key, current_value)`` for a small set of n_batch=1024 keys.

    The full table has ~1000 entries; we sweep a handful per nightly run.
    """
    rows = []
    for k, v in table.items():
        # Skip entries that aren't dataclass tuples we can plug in.
        if not hasattr(v, "__dataclass_fields__"):
            continue
        # Focus on common production shapes (n_batch=1024, int8 × int8).
        try:
            if (k.n_batch in (1024, 2048) and k.x_q_dtype.startswith("int8")
                    and k.w_q_dtype.startswith("int8") and k.n_out >= 1024):
                val = (v.batch_block_size, v.out_block_size, v.in_block_size,
                       v.n_lane_multiplier)
                rows.append((k, val))
        except AttributeError:
            continue
        if len(rows) >= 6:
            break
    return rows


def _build_inputs(key) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Generate matching-shape inputs for ``quantized_matmul_kernel``."""
    rng = np.random.default_rng(0)
    n_batch, n_in, n_out = int(key.n_batch), int(key.n_in), int(key.n_out)
    x_dtype = jnp.bfloat16  # x is unquantized bf16 for the int-8 path
    x = jnp.asarray(rng.normal(0, 0.1,
                               size=(n_batch, n_in)).astype(np.float32),
                    dtype=x_dtype)
    w_q = jnp.asarray(
        rng.integers(-127, 127, size=(n_out, n_in)).astype(np.int8))
    w_scale = jnp.asarray(rng.normal(0, 0.05,
                                     size=(n_out, )).astype(np.float32),
                          dtype=jnp.bfloat16)
    return x, w_q, w_scale


def _microbench(item: dict) -> dict:
    """Real microbench: run the production kernel at the given block sizes."""
    from tpu_inference.kernels.quantized_matmul.kernel import \
        quantized_matmul_kernel
    from tpu_inference.kernels.quantized_matmul.tuned_block_sizes import \
        TunedValue

    key = item["key"]
    val = item["value"]
    bs, out_bs, in_bs, lane_mult = val
    try:
        x, w_q, w_scale = _build_inputs(key)
    except Exception as err:
        return {
            "status": "FAILED_RUN",
            "latency_ns": None,
            "wall_s": 0,
            "cosine": None,
            "max_abs_diff": None,
            "p50_ns": None,
            "p95_ns": None,
            "error": f"input build failed: {err}"
        }

    tuned_value = TunedValue(
        batch_block_size=int(bs),
        out_block_size=int(out_bs),
        in_block_size=int(in_bs),
        n_lane_multiplier=int(lane_mult),
    )

    def fn():
        return quantized_matmul_kernel(jnp.copy(x),
                                       jnp.copy(w_q),
                                       jnp.copy(w_scale),
                                       x_q_dtype=jnp.int8,
                                       tuned_value=tuned_value)

    try:
        bench = measure(fn, warmup=2, iters=6)
    except Exception as err:  # OOM, compile fail, shape error etc.
        return {
            "status": "FAILED_RUN",
            "latency_ns": None,
            "wall_s": 0,
            "cosine": None,
            "max_abs_diff": None,
            "p50_ns": None,
            "p95_ns": None,
            "error": str(err)[:300]
        }

    # Verify against pure-JAX reference (no quantization).
    ref = jnp.matmul(x.astype(jnp.float32),
                     (w_q.astype(jnp.float32) *
                      w_scale[:, None].astype(jnp.float32)).T)
    numerics = check_many((bench.output, ), (ref.astype(bench.output.dtype), ),
                          atol=0.5,
                          rtol=0.5,
                          cosine_floor=0.99)
    if not numerics.passed:
        return {
            "status": "FAILED_NUMERICS",
            "latency_ns": None,
            "wall_s": 0,
            "cosine": numerics.cosine,
            "max_abs_diff": numerics.max_abs_diff,
            "p50_ns": None,
            "p95_ns": None,
            "error": numerics.reason
        }
    return {
        "status": "VERIFIED",
        "latency_ns": float(bench.mean_ns),
        "wall_s": float(bench.mean_ns * bench.iters / 1e9),
        "cosine": numerics.cosine,
        "max_abs_diff": numerics.max_abs_diff,
        "p50_ns": int(bench.p50_ns),
        "p95_ns": int(bench.p95_ns),
    }


def run(*, out_dir: Path) -> dict:
    recipe = TunedTableRecipe(
        kernel="quantized_matmul",
        tuned_path=_TUNED_PATH,
        table_attr="TUNED_BLOCK_SIZES_RAW",
        out_dir=out_dir,
        microbench_fn=_microbench,
        entry_extractor=_extract_targets,
        candidates=[
            (1024, 256, 4096, 1),
            (1024, 512, 4096, 1),
            (512, 256, 4096, 2),
            (2048, 256, 4096, 1),
            (1024, 256, 2048, 1),
        ],
    )
    return run_sweep(recipe, max_keys=4, min_win_margin=1.01)
