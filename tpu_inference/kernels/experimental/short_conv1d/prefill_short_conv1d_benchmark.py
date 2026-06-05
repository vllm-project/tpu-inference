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
"""Benchmark prefill_short_conv1d.

Single prefill sequence covering all ``N`` tokens, with initial conv state.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os

import jax
import jax.numpy as jnp
import numpy as np

from tpu_inference.kernels.experimental.short_conv1d import short_conv1d
from tpu_inference.kernels.benchmark_utils import (benchmark, get_device_name,
                                                   get_peak_mem_bw_gbs)

BASELINE_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "prefill_short_conv1d_benchmark_results.json",
)
DEFAULT_HEAD_NUM = 8
DEFAULT_ITERS = 100


@dataclasses.dataclass
class BenchmarkResult:
    N: int
    H: int
    D: int
    W: int
    block_n: int
    dtype: str
    mean_ms: float
    mem_bytes: int
    mem_bw_gbs: float
    mbu: float


def compute_mem_bytes(x: jax.Array, weight: jax.Array,
                      conv_state: jax.Array) -> int:
    state_slot_elems = conv_state.shape[1] * conv_state.shape[
        2] * conv_state.shape[3]
    read_bytes = (
        x.size * x.dtype.itemsize +
        weight.size * weight.dtype.itemsize +
        state_slot_elems * conv_state.dtype.itemsize)
    write_bytes = (
        x.size * x.dtype.itemsize +
        state_slot_elems * conv_state.dtype.itemsize)
    return read_bytes + write_bytes


def _build_inputs(n: int, h: int, d: int, w: int, dtype, *, seed: int = 0):
    max_reqs = 4
    n_states = 8
    cu_seqlens = jnp.asarray(
        np.array([0, n] + [n] * (max_reqs - 1), dtype=np.int32))
    state_indices = jnp.asarray(np.arange(max_reqs, dtype=np.int32))
    has_initial_state = jnp.asarray(
        np.array([1] + [0] * (max_reqs - 1), dtype=np.int32))
    distribution = jnp.asarray(np.array([0, 1], dtype=np.int32))

    key_x, key_w, key_state = jax.random.split(jax.random.key(seed), 3)
    x = jax.random.normal(key_x, (n, h, d), dtype=dtype) * 0.1
    weight = jax.random.normal(key_w, (w, h, d), dtype=dtype) * 0.1
    conv_state = (
        jax.random.normal(key_state, (n_states, w - 1, h, d), dtype=dtype) *
        0.1)
    return (
        x,
        weight,
        conv_state,
        cu_seqlens,
        state_indices,
        distribution,
        has_initial_state,
    )


def run_one(
    n: int,
    h: int,
    d: int,
    w: int,
    block_n: int | None,
    *,
    dtype=jnp.bfloat16,
    iters: int = DEFAULT_ITERS,
    trace_dir: str | None = None,
) -> BenchmarkResult:
    (
        x,
        weight,
        conv_state,
        cu_seqlens,
        state_indices,
        distribution,
        has_initial_state,
    ) = _build_inputs(n, h, d, w, dtype)

    mem_bytes = compute_mem_bytes(x, weight, conv_state)
    peak_bw = get_peak_mem_bw_gbs()
    dtype_str = str(dtype.dtype.name)

    fn = lambda: short_conv1d(
        jnp.copy(x),
        weight,
        jnp.copy(conv_state),
        cu_seqlens,
        state_indices,
        distribution,
        has_initial_state,
        prefill_block_n=block_n,
    )

    sub_trace = os.path.join(trace_dir, f"N{n}_H{h}_D{d}") if trace_dir else None
    mean_ms = benchmark(
        fn,
        iters=iters,
        trace_dir=sub_trace,
        event_name="jit_short_conv1d",
    )
    mem_bw_gbs = mem_bytes / (mean_ms / 1000.0) / 1e9
    mbu = mem_bw_gbs / peak_bw * 100 if peak_bw else float("nan")

    if block_n is None:
        from tpu_inference.kernels.experimental.short_conv1d.prefill_short_conv1d_kernel import \
            get_default_block_sizes

        block_n = get_default_block_sizes(h, d, w, dtype)

    return BenchmarkResult(
        N=n,
        H=h,
        D=d,
        W=w,
        block_n=block_n,
        dtype=dtype_str,
        mean_ms=mean_ms,
        mem_bytes=mem_bytes,
        mem_bw_gbs=mem_bw_gbs,
        mbu=mbu,
    )


def load_baseline(path: str) -> dict[tuple[int, int], BenchmarkResult]:
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        data = json.load(f)
    fields = {field.name for field in dataclasses.fields(BenchmarkResult)}
    baseline = {}
    for result in data["results"]:
        benchmark_result = BenchmarkResult(
            **{key: value for key, value in result.items() if key in fields})
        baseline[(benchmark_result.N, benchmark_result.H)] = benchmark_result
    return baseline


def print_results(
    results: list[BenchmarkResult],
    baseline: dict[tuple[int, int], BenchmarkResult],
) -> None:
    has_baseline = bool(baseline)
    header = (
        f"{'N':>5} {'H':>3} {'D':>4} {'W':>2} {'bt':>5} "
        f"{'Time(us)':>10} {'BW(GB/s)':>10} {'MBU(%)':>7}")
    if has_baseline:
        header += f"  {'old(us)':>10} {'delta':>8}"
    print(header)
    print("-" * len(header))
    for result in results:
        line = (
            f"{result.N:>5} {result.H:>3} {result.D:>4} {result.W:>2} "
            f"{result.block_n:>5} {result.mean_ms * 1000:>10.2f} "
            f"{result.mem_bw_gbs:>10.1f} {result.mbu:>7.2f}")
        if has_baseline:
            old = baseline.get((result.N, result.H))
            if old:
                delta_pct = (result.mean_ms - old.mean_ms) / old.mean_ms * 100
                sign = "+" if delta_pct >= 0 else ""
                line += f"  {old.mean_ms * 1000:>10.2f} {sign}{delta_pct:>6.1f}%"
            else:
                line += f"  {'n/a':>10} {'':>8}"
        print(line)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark prefill_short_conv1d kernel")
    parser.add_argument("--iters", type=int, default=DEFAULT_ITERS)
    parser.add_argument("--baseline", type=str, default=BASELINE_PATH)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--trace-dir", type=str, default=None)
    parser.add_argument("--head-num", type=int, default=DEFAULT_HEAD_NUM)
    args = parser.parse_args()

    print(f"Device: {get_device_name()}")
    print("Benchmarking prefill_short_conv1d "
          "(dtype=bfloat16, single prefill seq)")
    print(f"Using head_num={args.head_num}")
    print(f"Using iters={args.iters}")
    print("Using trace-based latency extraction")
    if args.trace_dir:
        print(f"Saving traces to {args.trace_dir}")
    print()

    cases = [(8192, args.head_num, 128, 4, None)]
    dtype = jnp.bfloat16

    results: list[BenchmarkResult] = []
    for n, h, d, w, block_n in cases:
        result = run_one(
            n,
            h,
            d,
            w,
            block_n,
            dtype=dtype,
            iters=args.iters,
            trace_dir=args.trace_dir,
        )
        results.append(result)
        print(
            f"N={result.N:>5}, H={result.H:>3}, bt={result.block_n}: "
            f"{result.mean_ms * 1000:.2f} us, "
            f"{result.mem_bw_gbs:.1f} GB/s, MBU={result.mbu:.2f}%")

    baseline = load_baseline(args.baseline)
    print("\n-- Summary --\n")
    print_results(results, baseline)

    data = {
        "device": get_device_name(),
        "results": [dataclasses.asdict(result) for result in results],
    }
    out_path = args.output or args.baseline
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)
        f.write("\n")
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
