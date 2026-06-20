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
"""Roofline analyzer for ragged_paged_attention v3 at production shapes.

Tells you what fraction of TPU peak the baseline kernel is actually
achieving, so you know whether +1% is the ceiling or there's 5x headroom.

For each shape:
* Compute theoretical bytes moved (Q + K + V + KV cache read + write).
* Compute theoretical FLOPs (QK^T + softmax + PV).
* Measure achieved time (paired-bench, p50).
* Report:
  - Achieved HBM bandwidth as % of peak (3.7 TB/s on TPU v7x)
  - Achieved arithmetic intensity (FLOPs / bytes)
  - MXU utilization as % of peak (1.155 PFLOPS bf16)
  - Roofline regime: HBM-bound, MXU-bound, or VPU/scalar-bound

The output is the bottleneck diagnosis that should drive mutator
strategy (HBM-bound -> attack data movement; MXU-bound -> attack
compute scheduling; VPU/scalar-bound -> attack precision policy /
elementwise op count).
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import sys
from pathlib import Path

import jax.numpy as jnp
import numpy as np

logger = logging.getLogger(__name__)

# TPU v7x per-chip peaks queried from pltpu.get_tpu_info() — keeping
# inline as constants for portability.
_TPU_V7X_HBM_BW_BPS = 3.7e12  # bytes/sec
_TPU_V7X_BF16_FLOPS = 1.155e15  # ops/sec
_TPU_V7X_FP8_FLOPS = 2.3e15  # ops/sec
_TPU_V7X_VMEM_BYTES = 64 * 1024 * 1024


@dataclasses.dataclass
class RooflineResult:
    shape_name: str
    description: str
    bench_ns_p50: int
    bench_ns_p95: int
    bytes_moved: int  # bytes read+write from/to HBM per call
    flops: int  # arithmetic ops per call
    hbm_bw_achieved: float  # bytes/sec achieved
    mxu_flops_achieved: float  # flops/sec achieved
    hbm_util_frac: float  # achieved / peak
    mxu_util_frac: float  # achieved / peak (bf16)
    arith_intensity: float  # flops / bytes
    ridge_arith_intensity: float  # peak_flops / peak_bw
    regime: str  # "hbm-bound" | "mxu-bound" | "vpu/scalar-bound"

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


def _bytes_per_dtype(dtype) -> int:
    return jnp.dtype(dtype).itemsize


def _estimate_rpa_v3_bytes(*, num_q_heads: int, num_kv_heads: int,
                           head_dim: int, total_q_len: int, total_kv_len: int,
                           page_size: int, num_pages: int, q_dtype,
                           kv_dtype) -> int:
    """Theoretical HBM bytes for one RPA call.

    Read:
      Q: total_q_len * num_q_heads * head_dim * sizeof(q_dtype)
      K,V from cache (each page): num_pages * page_size * num_kv_heads * 2
        * head_dim * sizeof(kv_dtype) -- but only the cu_q_lens worth
        of pages are touched, approximated by total_kv_len.
      New K, V (written): total_q_len * num_kv_heads * 2 * head_dim * sizeof(kv_dtype)
    Write:
      Output: total_q_len * num_q_heads * head_dim * sizeof(q_dtype)
      Updated KV cache: same total_kv_len worth as the read.
    """
    q_bytes = _bytes_per_dtype(q_dtype)
    kv_bytes = _bytes_per_dtype(kv_dtype)
    q_read = total_q_len * num_q_heads * head_dim * q_bytes
    kv_read = total_kv_len * num_kv_heads * 2 * head_dim * kv_bytes
    new_kv_write = total_q_len * num_kv_heads * 2 * head_dim * kv_bytes
    output_write = total_q_len * num_q_heads * head_dim * q_bytes
    return q_read + kv_read + new_kv_write + output_write


def _estimate_rpa_v3_flops(*, num_q_heads: int, num_kv_heads: int,
                           head_dim: int, q_kv_pairs: list[tuple[int,
                                                                 int]]) -> int:
    """Theoretical FLOPs for one RAGGED RPA call.

    Each sequence attends only to its OWN kv (not all kvs cross-batch).
    Correct formula sums per-seq products, NOT total_q_len * total_kv_len.

    Per seq i: 2 * q_len[i] * kv_len[i] * head_dim FLOPs (QK^T) and same
    again for PV. Softmax: ~5 * q_len[i] * kv_len[i] VPU ops.
    Multiply by num_q_heads (queries broadcast across heads).
    """
    per_seq_pairs = sum(q * kv for q, kv in q_kv_pairs)
    qk = 2 * num_q_heads * per_seq_pairs * head_dim
    pv = 2 * num_q_heads * per_seq_pairs * head_dim
    softmax = 5 * num_q_heads * per_seq_pairs
    return qk + pv + softmax


def classify_regime(hbm_util: float, mxu_util: float, ai: float,
                    ridge_ai: float) -> str:
    """Pick the dominant bottleneck."""
    # Simple heuristic from roofline theory:
    # - if arith intensity < ridge, kernel is on the HBM line
    # - if both utils very low, neither MXU nor HBM is busy -> VPU/scalar
    if hbm_util < 0.1 and mxu_util < 0.1:
        return "vpu/scalar-bound"
    if ai < ridge_ai:
        return "hbm-bound"
    else:
        return "mxu-bound"


def bench_baseline_at_shape(shape_spec) -> RooflineResult:
    """Bench RPA v3 baseline at a production shape; compute roofline."""
    from tools.kernel.evolve.cross_shape import _build_tuner_for_shape
    from tools.kernel.evolve.examples.rpa_v3_evolve import RpaV3Host
    from tools.kernel.tuner.v1.bench.harness import measure
    tuner = _build_tuner_for_shape(shape_spec)
    host = RpaV3Host(tuner)
    fn = host.build_kernel_fn(
        __import__("tpu_inference.kernels.ragged_paged_attention.v3.kernel",
                   fromlist=["ragged_paged_attention"]))
    # Bench
    samples = []
    for _ in range(8):
        r = measure(fn, warmup=2, iters=6)
        samples.append(r.p50_ns)
    p50_ns = int(np.median(samples))
    p95_ns = int(np.percentile(samples, 95))

    # Pull dims for byte/FLOP estimate
    q = host.inputs["q"]
    kv_cache = host.inputs["kv_cache"]
    kv_lens = host.inputs["kv_lens"]
    cu_q_lens = host.inputs["cu_q_lens"]
    total_q_len = int(cu_q_lens[-1])
    total_kv_len = int(jnp.sum(kv_lens))
    total_num_pages = int(kv_cache.shape[0])

    q_dtype = q.dtype
    kv_dtype = kv_cache.dtype
    num_q_heads = int(q.shape[1])
    num_kv_heads = shape_spec.num_kv_heads
    head_dim = shape_spec.head_dim
    page_size = shape_spec.page_size

    # Reconstruct per-seq (q_len, kv_len) pairs for correct ragged FLOPs.
    # cu_q_lens[i+1] - cu_q_lens[i] = q_len_i
    q_lens_list = [
        int(cu_q_lens[i + 1] - cu_q_lens[i])
        for i in range(len(cu_q_lens) - 1)
    ]
    kv_lens_list = [int(x) for x in kv_lens]
    q_kv_pairs = list(zip(q_lens_list, kv_lens_list))
    bytes_moved = _estimate_rpa_v3_bytes(num_q_heads=num_q_heads,
                                         num_kv_heads=num_kv_heads,
                                         head_dim=head_dim,
                                         total_q_len=total_q_len,
                                         total_kv_len=total_kv_len,
                                         page_size=page_size,
                                         num_pages=total_num_pages,
                                         q_dtype=q_dtype,
                                         kv_dtype=kv_dtype)
    flops = _estimate_rpa_v3_flops(num_q_heads=num_q_heads,
                                   num_kv_heads=num_kv_heads,
                                   head_dim=head_dim,
                                   q_kv_pairs=q_kv_pairs)
    sec = p50_ns / 1e9
    hbm_bw = bytes_moved / sec if sec > 0 else 0.0
    mxu_flops = flops / sec if sec > 0 else 0.0
    hbm_util = hbm_bw / _TPU_V7X_HBM_BW_BPS
    mxu_util = mxu_flops / _TPU_V7X_BF16_FLOPS
    ai = flops / bytes_moved if bytes_moved > 0 else 0.0
    ridge_ai = _TPU_V7X_BF16_FLOPS / _TPU_V7X_HBM_BW_BPS
    regime = classify_regime(hbm_util, mxu_util, ai, ridge_ai)

    return RooflineResult(
        shape_name=shape_spec.name,
        description=shape_spec.description,
        bench_ns_p50=p50_ns,
        bench_ns_p95=p95_ns,
        bytes_moved=bytes_moved,
        flops=flops,
        hbm_bw_achieved=hbm_bw,
        mxu_flops_achieved=mxu_flops,
        hbm_util_frac=hbm_util,
        mxu_util_frac=mxu_util,
        arith_intensity=ai,
        ridge_arith_intensity=ridge_ai,
        regime=regime,
    )


def render_md(results: list[RooflineResult]) -> str:
    lines = [
        "# Roofline analysis — RPA v3 baseline on TPU v7x\n",
        f"Per-chip peaks: HBM {_TPU_V7X_HBM_BW_BPS/1e12:.1f} TB/s; "
        f"bf16 {_TPU_V7X_BF16_FLOPS/1e12:.0f} TFLOPS; "
        f"VMEM {_TPU_V7X_VMEM_BYTES/1024/1024:.0f} MB; "
        f"ridge AI (flops/byte) = {_TPU_V7X_BF16_FLOPS/_TPU_V7X_HBM_BW_BPS:.0f}\n",
        "",
        "| shape | p50 μs | HBM BW achieved | HBM util | MXU achieved | "
        "MXU util | arith intensity | regime |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        lines.append(f"| `{r.shape_name}` | {r.bench_ns_p50/1e3:.1f} | "
                     f"{r.hbm_bw_achieved/1e9:.1f} GB/s | "
                     f"**{r.hbm_util_frac*100:.1f}%** | "
                     f"{r.mxu_flops_achieved/1e12:.2f} TFLOPS | "
                     f"**{r.mxu_util_frac*100:.1f}%** | "
                     f"{r.arith_intensity:.1f} | **{r.regime}** |")
    lines.append("")
    lines.append("## Diagnosis")
    lines.append("")
    counts = {"hbm-bound": 0, "mxu-bound": 0, "vpu/scalar-bound": 0}
    for r in results:
        counts[r.regime] = counts.get(r.regime, 0) + 1
    dom = max(counts, key=counts.get)
    lines.append(f"Dominant regime across {len(results)} production shapes: "
                 f"**{dom}** ({counts[dom]}/{len(results)} shapes).\n")
    if dom == "hbm-bound":
        lines.append("→ Mutator should attack **data movement** "
                     "(perf-skill family A): donation, dequant-in-VMEM, "
                     "fuse ops to keep intermediates in VMEM, eliminate "
                     "redundant HBM round-trips.")
    elif dom == "mxu-bound":
        lines.append("→ Mutator should attack **MXU scheduling** "
                     "(perf-skill family B): pipelining, op reordering "
                     "to hide latency, regime-specialized launches.")
    else:
        lines.append("→ Mutator should attack **VPU/scalar overhead** "
                     "(perf-skill family J): algebraic identities, "
                     "fuse small VPU ops, reduce elementwise op counts.")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--shapes",
                   default="all",
                   help="Comma-separated shape names from "
                   "cross_shape.PRODUCTION_SHAPES, or 'all'.")
    p.add_argument("--out-json",
                   type=Path,
                   default=Path("/tmp/rpa_v3_roofline.json"))
    p.add_argument("--out-md",
                   type=Path,
                   default=Path("/tmp/rpa_v3_roofline.md"))
    p.add_argument("--verbose", "-v", action="count", default=0)
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.WARNING - 10 * args.verbose,
                        format="%(asctime)s %(levelname)s %(message)s")
    from tools.kernel.evolve.cross_shape import PRODUCTION_SHAPES
    if args.shapes == "all":
        shapes = PRODUCTION_SHAPES
    else:
        wanted = set(s.strip() for s in args.shapes.split(","))
        shapes = [s for s in PRODUCTION_SHAPES if s.name in wanted]
    results: list[RooflineResult] = []
    for s in shapes:
        try:
            r = bench_baseline_at_shape(s)
            print(
                f"  {r.shape_name}: p50={r.bench_ns_p50/1e3:.1f}us  "
                f"HBM_util={r.hbm_util_frac*100:.1f}%  "
                f"MXU_util={r.mxu_util_frac*100:.1f}%  "
                f"regime={r.regime}",
                file=sys.stderr)
            results.append(r)
        except Exception as e:
            logger.warning("Shape %s failed: %s", s.name, e)
    args.out_json.write_text(
        json.dumps([r.to_dict() for r in results], indent=2))
    md = render_md(results)
    args.out_md.write_text(md)
    print()
    print(md)
    print(f"\nJSON: {args.out_json}")
    print(f"MD:   {args.out_md}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
