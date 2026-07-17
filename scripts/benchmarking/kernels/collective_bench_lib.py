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
"""Shared harness for the collective-matmul microbenchmarks.

Holds what the benchmarks share — the mesh, per-call timing, and the sweep/
report driver — so each benchmark script is just its own collective: input
shardings, the einsum, and an IMPLEMENTATIONS registry. Not runnable on its own.
"""

import argparse
import contextlib
import time

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh

AXIS = "x"


def build_mesh(order):
    """Auto-axis mesh (not jax.make_mesh's explicit axes) so the auto-partitioner
    inserts the collective for the sharded einsum, as the serving path does.
    `order` is an optional device-index permutation for ring-sensitive kernels."""
    devices = jax.devices()
    if order:
        devices = [devices[i] for i in order]
    return Mesh(np.asarray(devices), (AXIS, ))


def time_call(fn, inputs, *, reps, warmup):
    """Mean per-call latency (ms): dispatch `reps` calls and block once. The
    calls pipeline on device, so wall / reps tracks device time once the device
    is the bottleneck — within ~7% of the profiler's module time at small M
    (dispatch overhead), under 1% at large M; raise --reps to tighten."""
    for _ in range(warmup):
        fn(*inputs)
    jax.block_until_ready(fn(*inputs))
    start = time.perf_counter()
    out = None
    for _ in range(reps):
        out = fn(*inputs)
    jax.block_until_ready(out)
    return (time.perf_counter() - start) / reps * 1e3


def run(*, make_inputs, implementations, default_n):
    """Sweep M for one collective and print a per-M table.

    make_inputs(mesh, m, k, n, dtype) -> operands; implementations maps
    name -> builder(mesh, inputs) -> callable(*inputs), first entry the
    reference; default_n is the --n default.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--m",
                        default="16,32,64,128,256,512,1024,2048,4096,8192",
                        help="comma-separated token counts to sweep")
    parser.add_argument("--k", type=int, default=8192, help="hidden dim")
    parser.add_argument("--n", type=int, default=default_n)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--reps",
                        type=int,
                        default=100,
                        help="dispatched calls per cell (blocked once)")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument(
        "--device-order",
        default=None,
        help="comma-separated device-index permutation for the "
        "mesh; default is natural order (order-sensitive ring "
        "kernels want a ring-friendly permutation)")
    parser.add_argument(
        "--profile-dir",
        default=None,
        help="if set, capture a JAX/xprof trace of the sweep into "
        "this dir and print the path; otherwise no profiling")
    args = parser.parse_args()

    assert jax.devices()[0].platform == "tpu", "requires a TPU host"
    tp = jax.device_count()
    if args.n % tp != 0:
        parser.error(f"--n ({args.n}) must be divisible by {tp} devices")
    order = ([int(i) for i in args.device_order.split(",")]
             if args.device_order else None)
    mesh = build_mesh(order)
    dtype = jnp.dtype(args.dtype)
    names = list(implementations)
    ref = names[0]

    print(f"devices={tp} ({jax.devices()[0].device_kind}), k={args.k}, "
          f"n={args.n} (n/tp={args.n // tp}), dtype={dtype.name}, "
          f"order={args.device_order or 'natural'}")
    cols = (["M", "TFLOP/s/c"] + [f"{n} us" for n in names] +
            [f"{n} x" for n in names[1:]])
    print(" | ".join(f"{c:>9}" for c in cols))

    prof = (jax.profiler.trace(args.profile_dir)
            if args.profile_dir else contextlib.nullcontext())
    with prof:
        for m in [int(v) for v in args.m.split(",")]:
            if m % tp != 0:
                print(f"{m:>9} | (skip: not divisible by {tp} devices)")
                continue
            inputs = make_inputs(mesh, m, args.k, args.n, dtype)
            times, errs = {}, {}
            for name in names:
                try:
                    fn = implementations[name](mesh, inputs)
                    times[name] = time_call(fn,
                                            inputs,
                                            reps=args.reps,
                                            warmup=args.warmup)
                except Exception as e:
                    times[name], errs[name] = None, " ".join(
                        str(e).split())[:70]
            base = times[ref]
            # Per-core matmul FLOPs = 2 * M * K * (N // tp), the same for both
            # patterns (AG: [M, K] @ [K, N//tp]; RS: [M, N//tp] @ [N//tp, K]).
            tflops = (2 * m * args.k * (args.n // tp) / (base * 1e-3) /
                      1e12 if base else None)
            cells = [f"{m:>9}", f"{tflops:>9.1f}" if tflops else f"{'--':>9}"]
            cells += [
                f"{times[n] * 1e3:>9.1f}" if times[n] else f"{'FAIL':>9}"
                for n in names
            ]
            cells += [
                f"{base / times[n]:>9.2f}"
                if base and times[n] else f"{'--':>9}" for n in names[1:]
            ]
            line = " | ".join(cells)
            if errs:
                line += "  " + "; ".join(f"{n}: {errs[n]}" for n in errs)
            print(line)

    if args.profile_dir:
        print(f"xprof trace written to {args.profile_dir}")
