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
import glob
import os
import shutil
import statistics
import tempfile

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh

AXIS = "x"


def build_mesh():
    """Auto-axis mesh (not jax.make_mesh's explicit axes) so the auto-partitioner
    inserts the collective for the sharded einsum, as the serving path does."""
    return Mesh(np.asarray(jax.devices()), (AXIS, ))


def _event_dur_ps(event):
    for name, value in event.stats:
        if name == "device_duration_ps":
            return value
    return 0.0


def _median_dur_ps(plane, line_name, event_name=None):
    durs = [
        _event_dur_ps(e) for line in plane.lines if line.name == line_name
        for e in line.events if event_name is None or e.name == event_name
    ]
    return statistics.median(durs) if durs else 0.0


def device_time(fn, inputs, *, reps, warmup, save_dir=None):
    """Median per-call device time (ms) from the JAX profiler, minus the
    `barrier-cores` span — the collective's cross-core wait for peers, not kernel
    work. Traces `reps` calls into save_dir if given (kept for inspection), else
    a temp dir; the per-call XLA-module time and barrier span are read back from
    the /device:TPU:0 plane."""
    for _ in range(warmup):
        fn(*inputs)
    jax.block_until_ready(fn(*inputs))
    trace_dir = save_dir or tempfile.mkdtemp()
    try:
        with jax.profiler.trace(trace_dir):
            out = None
            for _ in range(reps):
                out = fn(*inputs)
            jax.block_until_ready(out)
        pbs = glob.glob(os.path.join(trace_dir, "**", "*.xplane.pb"),
                        recursive=True)
        if not pbs:
            raise RuntimeError("profiler wrote no trace")
        newest = max(pbs,
                     key=os.path.getmtime)  # this run's, if save_dir reused
        plane = jax.profiler.ProfileData.from_file(
            newest).find_plane_with_name("/device:TPU:0")
        if plane is None:
            raise RuntimeError("no /device:TPU:0 plane in the trace")
        module = _median_dur_ps(plane, "XLA Modules")
        if not module:
            raise RuntimeError("no XLA-module device time in the profile")
        barrier = _median_dur_ps(plane, "XLA TraceMe", "barrier-cores")
        if barrier >= module:
            raise RuntimeError("barrier-cores span >= module time: "
                               "unexpected trace shape")
        return (module - barrier) / 1e9  # ps -> ms
    finally:
        if save_dir is None:
            shutil.rmtree(trace_dir, ignore_errors=True)


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
                        default=20,
                        help="traced calls per cell (median)")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument(
        "--profile-dir",
        default=None,
        help="if set, keep each cell's xprof trace under this dir "
        "(named <impl>_m<M>) for inspection; otherwise discarded")
    args = parser.parse_args()

    assert jax.devices()[0].platform == "tpu", "requires a TPU host"
    tp = jax.device_count()
    if args.n % tp != 0:
        parser.error(f"--n ({args.n}) must be divisible by {tp} devices")
    mesh = build_mesh()
    dtype = jnp.dtype(args.dtype)
    names = list(implementations)
    ref = names[0]

    print(f"devices={tp} ({jax.devices()[0].device_kind}), k={args.k}, "
          f"n={args.n} (n/tp={args.n // tp}), dtype={dtype.name}")
    cols = (["M", "TFLOP/s/c"] + [f"{n} us" for n in names] +
            [f"{n} x" for n in names[1:]])
    print(" | ".join(f"{c:>9}" for c in cols))

    for m in [int(v) for v in args.m.split(",")]:
        if m % tp != 0:
            print(f"{m:>9} | (skip: not divisible by {tp} devices)")
            continue
        inputs = make_inputs(mesh, m, args.k, args.n, dtype)
        times, errs = {}, {}
        for name in names:
            save_dir = (os.path.join(args.profile_dir, f"{name}_m{m}")
                        if args.profile_dir else None)
            try:
                fn = implementations[name](mesh, inputs)
                times[name] = device_time(fn,
                                          inputs,
                                          reps=args.reps,
                                          warmup=args.warmup,
                                          save_dir=save_dir)
            except Exception as e:
                times[name], errs[name] = None, " ".join(str(e).split())[:70]
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
            f"{base / times[n]:>9.2f}" if base and times[n] else f"{'--':>9}"
            for n in names[1:]
        ]
        line = " | ".join(cells)
        if errs:
            line += "  " + "; ".join(f"{n}: {errs[n]}" for n in errs)
        print(line)

    if args.profile_dir:
        print(f"xprof traces kept under {args.profile_dir}")
