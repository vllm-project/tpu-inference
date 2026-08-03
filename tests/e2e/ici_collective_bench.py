# Copyright 2025 Google LLC
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
"""ICI collective microbenchmark for tpu7x multi-host slices (e.g. 2x2x4).

Runs the same SPMD program on every host of the slice. Measures per-axis
collective latency (ppermute / all-gather / reduce-scatter / all-reduce) so
achieved link bandwidth can be derived per topology dimension, including:
  - D2D (chiplet pair inside one package) vs ICI (chip-to-chip) links
  - full-duplex behavior on size-2 dimensions
  - torus-vs-mesh behavior of the size-4 dimension (wraparound detection)

Results are printed by process 0 as lines prefixed with ICIBENCH_RESULT.
"""

import json
import os
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

ROWS = 32  # leading dim, divisible by every axis-group size we shard over
ITERS = 30


def log(msg):
    if jax.process_index() == 0:
        print(msg, flush=True)


def result(record):
    log("ICIBENCH_RESULT " + json.dumps(record))


def build_mesh():
    devs = jax.devices()
    # Debug/CI-smoke escape hatch for backends without .coords (e.g. CPU with
    # xla_force_host_platform_device_count): ICIBENCH_DIMS="2,2,4,2".
    if os.environ.get("ICIBENCH_DIMS"):
        dims = tuple(int(v) for v in os.environ["ICIBENCH_DIMS"].split(","))
        assert int(np.prod(dims)) == len(devs), (dims, len(devs))
        arr = np.array(devs, dtype=object).reshape(dims)
        return Mesh(arr, ("x", "y", "z", "c")), dims
    keyed = {}
    for d in devs:
        c = tuple(d.coords)
        core = int(getattr(d, "core_on_chip", 0))
        key = c if len(c) == 4 else c + (core, )
        keyed[key] = d
    dims = tuple(max(k[i] for k in keyed) + 1 for i in range(4))
    assert int(np.prod(dims)) == len(devs), (dims, len(devs))
    arr = np.empty(dims, dtype=object)
    for key, d in keyed.items():
        arr[key] = d
    assert not (arr == None).any()  # noqa: E711
    return Mesh(arr, ("x", "y", "z", "c")), dims


def make_input(mesh, cols, spec):
    sh = NamedSharding(mesh, spec)

    @partial(jax.jit, out_shardings=sh)
    def f():
        n = ROWS * cols
        return jnp.reshape(
            jnp.arange(n, dtype=jnp.float32) % 1024.0, (ROWS, cols))

    return f()


def bench(name, fn, x, meta):
    multihost_utils.sync_global_devices("pre_" + name)
    y = fn(x)
    jax.block_until_ready(y)
    y = fn(x)
    jax.block_until_ready(y)
    t0 = time.perf_counter()
    for _ in range(ITERS):
        y = fn(x)
    jax.block_until_ready(y)
    dt_ms = (time.perf_counter() - t0) / ITERS * 1e3
    rec = dict(meta)
    rec.update(name=name, avg_ms=round(dt_ms, 4), iters=ITERS)
    result(rec)


def axis_arg(axes):
    return axes if len(axes) > 1 else axes[0]


def smap(f, mesh, in_specs, out_specs):
    # check_rep/check_vma can't statically infer replication over the
    # non-participating axes here; the specs are still correct.
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


def pp_fn(mesh, axis, perm):

    def inner(a):
        return lax.ppermute(a, axis, perm)

    return jax.jit(smap(inner, mesh, P(axis), P(axis)))


def ag_fn(mesh, axes):

    def inner(a):
        return lax.all_gather(a, axis_arg(axes), axis=0, tiled=True)

    return jax.jit(smap(inner, mesh, P(axes), P(None, None)))


def rs_fn(mesh, axis):

    def inner(a):
        a = a + lax.axis_index(axis).astype(a.dtype)
        return lax.psum_scatter(a, axis, scatter_dimension=0, tiled=True)

    return jax.jit(smap(inner, mesh, P(None, None), P(axis)))


def ar_fn(mesh, axes):

    def inner(a):
        a = a + lax.axis_index(axes[0]).astype(a.dtype)
        return lax.psum(a, axes)

    return jax.jit(smap(inner, mesh, P(None, None), P(None, None)))


def main():
    try:
        jax.distributed.initialize()
    except Exception as e:  # single-host runs don't need it
        print(f"jax.distributed.initialize skipped: {e}", flush=True)
    try:
        import jax.extend as jex
        backend_ver = jex.backend.get_backend().platform_version
    except Exception:
        backend_ver = "unknown"
    log(f"jax {jax.__version__}, backend {backend_ver}")
    log(f"process {jax.process_index()}/{jax.process_count()}, "
        f"{jax.local_device_count()} local / {jax.device_count()} global "
        "devices")
    for d in jax.devices():
        log(f"  device id={d.id} proc={d.process_index} "
            f"coords={getattr(d, 'coords', None)} "
            f"core={getattr(d, 'core_on_chip', None)} kind={d.device_kind}")

    mesh, dims = build_mesh()
    log(f"mesh dims (x,y,z,c) = {dims}")

    scale = int(os.environ.get("ICIBENCH_SCALE", "1"))
    mib = 2**20 // scale

    def global_mib(n_bytes):
        return n_bytes / 2**20

    # ppermute tests: input sharded on the permuted axis; per-device block =
    # global / axis_size. Sized so every per-device block is 128 MiB.
    pp_tests = []
    for axis in ("c", "x", "y"):
        if mesh.shape[axis] < 2:
            continue
        nbytes = 128 * mib * mesh.shape[axis]
        pp_tests += [
            (f"pp_{axis}_uni", axis, [(0, 1)], nbytes),
            (f"pp_{axis}_bidi", axis, [(0, 1), (1, 0)], nbytes),
        ]
    s = mesh.shape["z"]
    if s >= 3:
        # size-s dim: chain vs wraparound behavior
        nbytes = 128 * mib * s
        chain = [(i, i + 1) for i in range(s - 1)]
        cyclic = [(i, (i + 1) % s) for i in range(s)]
        # full-duplex probes: pairwise swaps (each device once as src/dst)
        swap_pairs = [(i, i + 1) for i in range(0, s - 1, 2)]
        swap_pairs += [(b, a) for a, b in swap_pairs]
        pp_tests += [
            ("pp_z_hop01", "z", [(0, 1)], nbytes),
            ("pp_z_hop_last_rev", "z", [(s - 1, s - 2)], nbytes),
            ("pp_z_chain", "z", chain, nbytes),
            ("pp_z_cyclic", "z", cyclic, nbytes),
            ("pp_z_wrap_only", "z", [(s - 1, 0)], nbytes),
            ("pp_z_swap_pairs", "z", swap_pairs, nbytes),
        ]
        if s >= 4:
            swap_wrap = [(1, 2), (2, 1), (s - 1, 0), (0, s - 1)]
            pp_tests.append(("pp_z_swap_wrap", "z", swap_wrap, nbytes))
    for name, axis, perm, nbytes in pp_tests:
        cols = nbytes // 4 // ROWS
        x = make_input(mesh, cols, P(axis))
        size = mesh.shape[axis]
        bench(
            name, pp_fn(mesh, axis, perm), x, {
                "op": "ppermute",
                "axis": axis,
                "perm": perm,
                "global_mib": global_mib(nbytes),
                "per_device_mib": global_mib(nbytes) / size,
            })

    ag_axes = [
        ("c", ),
        ("x", ),
        ("z", ),
        ("x", "y"),
        ("z", "c"),
        ("x", "y", "z"),
        ("x", "y", "z", "c"),
    ]
    ag_axes = [t for t in ag_axes if all(mesh.shape[a] >= 2 for a in t)]
    for axes in ag_axes:
        group = int(np.prod([mesh.shape[a] for a in axes]))
        for nbytes in (64 * mib, 512 * mib):
            cols = nbytes // 4 // ROWS
            x = make_input(mesh, cols, P(axes))
            bench(
                "ag_" + "".join(axes), ag_fn(mesh, axes), x, {
                    "op": "all_gather",
                    "axes": axes,
                    "group": group,
                    "global_mib": global_mib(nbytes),
                    "per_device_in_mib": global_mib(nbytes) / group,
                })

    for axis in ("c", "x", "z"):
        if mesh.shape[axis] < 2:
            continue
        for nbytes in (64 * mib, 512 * mib):
            cols = nbytes // 4 // ROWS
            x = make_input(mesh, cols, P(None, None))
            bench(
                "rs_" + axis, rs_fn(mesh, axis), x, {
                    "op": "reduce_scatter",
                    "axes": (axis, ),
                    "group": mesh.shape[axis],
                    "global_mib": global_mib(nbytes),
                    "per_device_in_mib": global_mib(nbytes),
                })

    ar_axes = [("z", ), ("x", "y"), ("x", "y", "z"), ("x", "y", "z", "c")]
    ar_axes = [t for t in ar_axes if all(mesh.shape[a] >= 2 for a in t)]
    for axes in ar_axes:
        group = int(np.prod([mesh.shape[a] for a in axes]))
        for nbytes in (64 * mib, 512 * mib):
            cols = nbytes // 4 // ROWS
            x = make_input(mesh, cols, P(None, None))
            bench(
                "ar_" + "".join(axes), ar_fn(mesh, axes), x, {
                    "op": "all_reduce",
                    "axes": axes,
                    "group": group,
                    "global_mib": global_mib(nbytes),
                    "per_device_in_mib": global_mib(nbytes),
                })

    multihost_utils.sync_global_devices("done")
    log("ICIBENCH_DONE")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("ICIBENCH_FAILED", flush=True)
        traceback.print_exc()
        sys.exit(1)
