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
"""JAX twin of the torch_tpu 112-MiB TP all-reduce microbenchmark.

Every device holds a full bf16 [tokens, width] tensor (default [8192, 7168],
112 MiB); the collective sums it across all devices of the slice. Modes:

  ar_1d     lax.psum over a 1-D mesh of every device (torch TP32 equivalent)
  ar_4d     lax.psum over all four axes of the physical (x, y, z, c) mesh
  rs_ag_1d  lax.psum_scatter (dim 0) + lax.all_gather on the 1-D mesh
  rs_ag_4d  the same on the 4-D mesh
  chunks2   psum of a tuple of two [tokens, width/2] halves on the 1-D mesh

Timing mirrors the torch script: 10 warmups, then N iterations each
synchronized with block_until_ready. Process 0 also records a pipelined
figure (N back-to-back calls, one sync), writes the optimized HLO of each
mode, and captures one xprof trace per mode. Output lines are prefixed
ALLREDUCE_JAX_RESULT (JSON).
"""

import argparse
import json
import os
import statistics
import sys
import time
import traceback
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.experimental import multihost_utils
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

try:
    shard_map = jax.shard_map
except AttributeError:
    from jax.experimental.shard_map import shard_map


def log(msg):
    if jax.process_index() == 0:
        print(msg, flush=True)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tokens", type=int, default=8192)
    p.add_argument("--width", type=int, default=7168)
    p.add_argument("--warmups", type=int, default=10)
    p.add_argument("--iterations", type=int, default=100)
    p.add_argument("--modes", default="ar_1d,ar_4d,rs_ag_1d,rs_ag_4d,chunks2")
    p.add_argument(
        "--orders",
        default="iota",
        help="comma-separated 1-D mesh device orders for the *_1d and chunks2 "
        "modes: iota (jax.devices() order), mesh_utils "
        "(mesh_utils.create_device_mesh ring order), or explicit:<device ids "
        "joined by '-'>; 4-D modes ignore this",
    )
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--out-dir",
                   default=os.environ.get("ALLREDUCE_JAX_OUT", "/results"))
    p.add_argument("--no-profile", action="store_true")
    return p.parse_args()


def smap(f, mesh, in_specs, out_specs):
    try:
        return shard_map(f,
                         mesh=mesh,
                         in_specs=in_specs,
                         out_specs=out_specs,
                         check_rep=False)
    except TypeError:
        return shard_map(f,
                         mesh=mesh,
                         in_specs=in_specs,
                         out_specs=out_specs,
                         check_vma=False)


def order_devices(order):
    devs = jax.devices()
    if order == "iota":
        return list(devs)
    if order == "mesh_utils":
        from jax.experimental import mesh_utils
        return list(mesh_utils.create_device_mesh((len(devs), )).flat)
    if order.startswith("explicit:"):
        by_id = {d.id: d for d in devs}
        ids = [int(v) for v in order[len("explicit:"):].split("-")]
        assert sorted(ids) == sorted(by_id), (ids, sorted(by_id))
        return [by_id[i] for i in ids]
    raise ValueError(order)


def build_meshes(order="iota"):
    devs = jax.devices()
    mesh_1d = Mesh(np.array(order_devices(order), dtype=object), ("tp", ))
    keyed = {}
    for d in devs:
        c = tuple(getattr(d, "coords", ()))
        if not c:
            return mesh_1d, None, None
        core = int(getattr(d, "core_on_chip", 0))
        keyed[c if len(c) == 4 else c + (core, )] = d
    dims = tuple(max(k[i] for k in keyed) + 1 for i in range(4))
    if int(np.prod(dims)) != len(devs):
        return mesh_1d, None, dims
    arr = np.empty(dims, dtype=object)
    for key, d in keyed.items():
        arr[key] = d
    return mesh_1d, Mesh(arr, ("x", "y", "z", "c")), dims


def linear_index(axes):
    """Row-major device index over the given mesh axes, as a traced int."""
    idx = lax.axis_index(axes[0])
    for a in axes[1:]:
        idx = idx * lax.axis_size(a) + lax.axis_index(a)
    return idx


def build_fn(mode, mesh, axes, world, dtype):
    axis_arg = axes if len(axes) > 1 else axes[0]
    rep = P(*([None] * 2)) if mesh is not None else None

    def contribution(x):
        # Same convention as the torch script: device r contributes
        # (r + 1) / world to every element, so the sum is (world + 1) / 2.
        r = linear_index(axes).astype(jnp.float32)
        return x + ((r + 1.0) / world).astype(dtype)

    if mode.startswith("ar_"):

        def inner(x):
            return lax.psum(contribution(x), axis_arg) + 1

    elif mode.startswith("rs_ag_"):

        def inner(x):
            s = lax.psum_scatter(contribution(x),
                                 axis_arg,
                                 scatter_dimension=0,
                                 tiled=True)
            return lax.all_gather(s, axis_arg, axis=0, tiled=True) + 1

    elif mode == "chunks2":

        def inner(x):
            v = contribution(x)
            half = v.shape[1] // 2
            a, b = lax.psum((v[:, :half], v[:, half:]), axis_arg)
            return jnp.concatenate([a, b], axis=1) + 1

    else:
        raise ValueError(mode)

    return jax.jit(smap(inner, mesh, rep, rep))


def main():
    args = parse_args()
    try:
        jax.distributed.initialize()
    except Exception as e:  # single-host runs don't need it
        print(f"jax.distributed.initialize skipped: {e}", flush=True)
    dtype = getattr(jnp, args.dtype)
    world = jax.device_count()
    try:
        import jax.extend as jex
        backend_ver = jex.backend.get_backend().platform_version
    except Exception:
        backend_ver = "unknown"
    log(f"jax {jax.__version__}, backend {backend_ver}")
    log(f"process {jax.process_index()}/{jax.process_count()}, "
        f"{jax.local_device_count()} local / {world} global devices")

    _, mesh_4d, dims = build_meshes()
    meshes_1d = {o: build_meshes(o)[0] for o in args.orders.split(",")}
    for o, m in meshes_1d.items():
        log(f"physical dims (x,y,z,c) = {dims}; 1-D mesh order {o} = "
            f"{[d.id for d in m.devices.flat]}")

    out_dir = args.out_dir if jax.process_index() == 0 else None
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "process0_host.txt"), "w") as f:
            f.write(os.uname().nodename + "\n")

    message_bytes = args.tokens * args.width * jnp.dtype(dtype).itemsize
    expected = (world + 1) / 2 + 1

    runs = []
    for mode in args.modes.split(","):
        if mode.endswith("_4d"):
            runs.append((mode, "physical"))
        else:
            runs.extend((mode, o) for o in meshes_1d)
    for mode, order in runs:
        if mode.endswith("_4d"):
            if mesh_4d is None:
                log(f"skip {mode}: no 4-D mesh")
                continue
            mesh, axes = mesh_4d, ("x", "y", "z", "c")
        else:
            mesh, axes = meshes_1d[order], ("tp", )
        tag = mode if order in ("physical", "iota") else f"{mode}@{order}"
        fn = build_fn(mode, mesh, axes, world, dtype)
        sh = NamedSharding(mesh, P())
        x = jax.device_put(jnp.zeros((args.tokens, args.width), dtype), sh)

        multihost_utils.sync_global_devices("pre_" + tag)
        y = fn(x)
        jax.block_until_ready(y)
        observed = float(np.asarray(y[0, 0]))
        tol = 0.0 if mode != "rs_ag_1d" and mode != "rs_ag_4d" else 0.5
        if abs(observed - expected) > tol:
            raise RuntimeError(f"{mode}: observed {observed}, expected "
                               f"{expected}")

        if out_dir:
            lowered = fn.lower(x)
            with open(os.path.join(out_dir, f"{tag}.after_optimizations.txt"),
                      "w") as f:
                f.write(lowered.compile().as_text())
            with open(os.path.join(out_dir, f"{tag}.before_optimizations.txt"),
                      "w") as f:
                f.write(lowered.as_text())

        for _ in range(args.warmups):
            jax.block_until_ready(fn(x))
        lat = []
        for _ in range(args.iterations):
            t0 = time.perf_counter_ns()
            jax.block_until_ready(fn(x))
            lat.append((time.perf_counter_ns() - t0) / 1e6)
        t0 = time.perf_counter_ns()
        for _ in range(args.iterations):
            y = fn(x)
        jax.block_until_ready(y)
        pipelined_ms = (time.perf_counter_ns() - t0) / 1e6 / args.iterations

        if not args.no_profile:
            multihost_utils.sync_global_devices("prof_" + tag)
            if out_dir:
                pdir = os.path.join(out_dir, "profile", tag)
                jax.profiler.start_trace(pdir)
                jax.block_until_ready(fn(x))
                jax.profiler.stop_trace()
            else:
                jax.block_until_ready(fn(x))

        median = statistics.median(lat)
        medians = multihost_utils.process_allgather(jnp.array(
            [median], dtype=jnp.float32),
                                                    tiled=True)
        rec = {
            "mode": tag,
            "order": order,
            "device_order": [int(d.id) for d in mesh.devices.flat],
            "world_size": world,
            "shape": [args.tokens, args.width],
            "dtype": args.dtype,
            "message_bytes": message_bytes,
            "message_mib": message_bytes / (1 << 20),
            "mesh_axes": list(axes),
            "physical_dims": dims,
            "warmups": args.warmups,
            "iterations": args.iterations,
            "median_ms": median,
            "mean_ms": statistics.fmean(lat),
            "min_ms": min(lat),
            "p90_ms": sorted(lat)[int(0.9 * (len(lat) - 1))],
            "max_ms": max(lat),
            "pipelined_ms": pipelined_ms,
            "process_medians_ms": [float(v) for v in np.asarray(medians)],
            "algorithm_gbps": message_bytes / (median / 1e3) / 1e9,
            "observed": observed,
            "expected": expected,
        }
        log("ALLREDUCE_JAX_RESULT " + json.dumps(rec))
        if out_dir:
            with open(os.path.join(out_dir, "results.jsonl"), "a") as f:
                f.write(json.dumps(rec) + "\n")

    multihost_utils.sync_global_devices("done")
    log("ALLREDUCE_JAX_DONE")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("ALLREDUCE_JAX_FAILED", flush=True)
        traceback.print_exc()
        sys.exit(1)
