#!/usr/bin/env python
# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Toy: per-DP-rank `Runner` (sidecar, CPU-only) + `Executor` (controller, TPU).

This is the smallest possible end-to-end demonstration of the v3-direction
split used by `colocated_dp_engine.py`:

  - **`Runner`** lives inside a `colocated_python_class` on a colocated CPU
    host. It holds a queue and turns Python `Request`-like objects into a
    CPU-resident JAX array (the "prepared inputs"). Pure CPU; cannot address
    TPU.
  - **`Executor`** lives in the controller process. It takes the CPU array
    produced by the matching `Runner`, transfers it to its TPU mesh with
    `jax.device_put` (bypassing the controller CPU via the
    `transfer_guard_device_to_host("disallow_explicit")` pattern from
    `spike_devices_inside_sidecar.py`), runs a jitted toy computation, and
    returns the result.

Run on Pathways multi-host (`JAX_PLATFORMS=proxy`). On a single host this
demo still runs with `dp_size=1` but the split is less interesting; on
multi-host you'll see one Runner created per Pathways host.

Compared to the original pseudocode in `spike_devices_inside_sidecar.py`,
this file fixes:
  - `colocated_python_class(name, cls)` → `colocated_python_class(cls)` (the
    first form raises ``TypeError``).
  - method args + returns must be `jax.Array` only — Python objects are
    cloudpickled into a uint8 JAX blob (see `_pack` / `_unpack`).
  - the first call on each Runner uses a pin array on **that host's CPU
    sharding** so colocated_python actually constructs the instance on the
    right host.
  - methods that previously returned `None` now return a JAX array (the pin
    or an output blob) so the controller can `jax.block_until_ready`.
"""

from __future__ import annotations

import sys
from typing import Any, List, Sequence

import cloudpickle
import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import colocated_python
from jax.sharding import Mesh, NamedSharding, PartitionSpec
import pathwaysutils 
pathwaysutils.initialize()

# ---------------------------------------------------------------------------
# cloudpickle ↔ uint8 jax.Array helpers (same idea as `_pack`/`_unpack` in
# `tpu_inference/core/colocated_dp_engine.py`).
# ---------------------------------------------------------------------------


def _pack(obj: Any, sharding: NamedSharding) -> jax.Array:
    data = cloudpickle.dumps(obj)
    arr = np.frombuffer(bytearray(data), dtype=np.uint8)
    return jax.device_put(arr, sharding)


def _unpack(blob: jax.Array) -> Any:
    arr = np.asarray(blob)
    return cloudpickle.loads(arr.tobytes())


# ---------------------------------------------------------------------------
# Sidecar class — wrapped by colocated_python_class. Pure CPU.
# ---------------------------------------------------------------------------


class Runner:
    """Per-rank sidecar: holds the request queue and prepares input arrays."""

    def __init__(self, host_id: int):
        print(f"[host {host_id}] Runner created",
              flush=True)
        self.host_id = host_id
        self.requests: List[Any] = []

    def add_request(self, blob: jax.Array) -> jax.Array:
        """Append one request (cloudpickled in `blob`) to the queue.
        Returns the same blob so the caller can `block_until_ready`."""
        request = _unpack(blob)
        self.requests.append(request)
        print(f"[host {self.host_id}] added {request!r} "
              f"(queue depth={len(self.requests)})", flush=True)
        return blob

    def schedule_and_prepare(self, prep_pin: jax.Array) -> jax.Array:
        """Pop one request, produce a CPU-resident float32 array sharded
        across the local colocated CPU devices the *exact same way* as
        ``prep_pin``.

        `prep_pin` is a controller-supplied template — a JAX array on the
        rank's per-chip CPU sharding (`NamedSharding(cpu_mesh, P("x"))`
        where `cpu_mesh` is built from `colocated_cpu_devices(tpu_group)`,
        i.e. one logical CPU device per TPU chip).  By matching its
        sharding here, the controller's subsequent `jax.device_put` to the
        TPU mesh becomes a direct CPU-shard-to-TPU-chip copy on each host
        — no controller-host detour, so the strict `transfer_guard_*`
        check in `Executor.execute` passes.

        Constructed with `make_array_from_single_device_arrays`, the same
        idiom as the bottom of `spike_devices_inside_sidecar.py`.
        """
        n = 32
        if not self.requests:
            full = np.zeros((n, ), dtype=np.float32)
            print(f"[host {self.host_id}] no requests; emitting zeros({n})",
                  flush=True)
        else:
            request = self.requests.pop(0)
            full = np.full((n, ), float(request), dtype=np.float32)
            print(f"[host {self.host_id}] prepared shape={full.shape} "
                  f"value={full[0]} from {request!r}", flush=True)

        # Build per-shard arrays for each colocated CPU device addressable
        # from this host. `shard.index` tells us which slice of the full
        # logical array this shard owns.
        per_shard = []
        for shard in prep_pin.addressable_shards:
            piece = np.ascontiguousarray(full[shard.index])
            per_shard.append(jax.device_put(piece, shard.device))
        return jax.make_array_from_single_device_arrays(
            shape=(n, ),
            sharding=prep_pin.sharding,
            arrays=per_shard,
        )


# ---------------------------------------------------------------------------
# Controller class — NOT wrapped. Lives in the controller process; can call
# jax.device_put onto TPU and jit-dispatch.
# ---------------------------------------------------------------------------


class Executor:
    """Per-rank controller-side TPU executor."""

    def __init__(self, label: str, tpu_mesh: Mesh):
        self.label = label
        self.tpu_mesh = tpu_mesh
        self.tpu_sharding = NamedSharding(tpu_mesh, PartitionSpec("x"))

        @jax.jit
        def _forward_pass(x):
            return x + x

        self.model_fn = _forward_pass
        print(f"[controller] Executor {label} ready on {tpu_mesh}")

    def execute(self, cpu_input: jax.Array) -> jax.Array:
        with jax.transfer_guard_device_to_host("disallow_explicit"), \
             jax.transfer_guard_host_to_device("disallow_explicit"):
            tpu_input = jax.device_put(cpu_input, self.tpu_sharding)
        result = self.model_fn(tpu_input)
        jax.block_until_ready(result)
        return result


# ---------------------------------------------------------------------------
# Wire it up: one (Runner sidecar, Executor controller) pair per Pathways host.
# ---------------------------------------------------------------------------




def main() -> int:
    all_tpu = jax.devices()
    groups = [all_tpu[:8], all_tpu[8:]]
    print(f"Creating {len(groups)} Runner/Executor pair(s)")

    RunnerCls = colocated_python.colocated_python_class(Runner)

    runners: List[Any] = []
    executors: List[Executor] = []
    blob_shardings: List[NamedSharding] = []  # for cloudpickled pins / blobs
    prep_pins: List[jax.Array] = []           # template for Runner outputs

    for i, tpu_group in enumerate(groups):

        runners.append(RunnerCls(host_id=i))

        cpu_devs = colocated_python.colocated_cpu_devices(tpu_group)
        cpu_mesh = Mesh(np.asarray(cpu_devs), ("x", ))
        tpu_mesh = Mesh(np.asarray(tpu_group), ("x", ))

        # Two CPU shardings per rank:
        #  - blob_sharding: replicated. Used for the cloudpickled `add_request`
        #    blob (variable-length uint8 — can't be sharded by "x" because
        #    pickle byte length isn't divisible by mesh size).
        #  - prep_sharding: P("x"), matches `tpu_sharding`. Used for the
        #    `schedule_and_prepare` output so the controller's device_put to
        #    TPU is a direct copy (no host detour).
        blob_shardings.append(NamedSharding(cpu_mesh, PartitionSpec()))
        prep_sharding = NamedSharding(cpu_mesh, PartitionSpec("x"))
        # The prep_pin is a *template* — its sharding tells the Runner exactly
        # how to lay out its output via make_array_from_single_device_arrays.
        prep_pins.append(
            jax.device_put(np.zeros((32, ), dtype=np.float32),
                           prep_sharding))

        executors.append(Executor(label=f"exec-{i}", tpu_mesh=tpu_mesh))

    # --- Drive a few iterations to show the split end-to-end ----------------

    # 1) Push toy requests round-robin into the runners' queues. Blobs go on
    #    the replicated CPU sharding (variable-length).
    for k in range(len(runners) * 2):
        i = k % len(runners)
        blob = _pack(k * 10 + i, blob_shardings[i])
        jax.block_until_ready(runners[i].add_request(blob))

    # 2) For each (Runner, Executor) pair: prepare on sidecar, run on TPU.
    # TODO: This really should be happening in parallel via threading. 
    for i, (runner, executor) in enumerate(zip(runners, executors)):
        cpu_input = runner.schedule_and_prepare(prep_pins[i])
        jax.block_until_ready(cpu_input)
        result = executor.execute(cpu_input)
        # `result` is shape (32,) — model_fn is just `x + x`. Pull to host
        # for a quick eyeball; mean is a cheap one-number summary.
        host_result = jax.device_get(result)
        print(f"pair {i}: result shape={host_result.shape} "
              f"mean={float(host_result.mean()):.3f} "
              f"first8={host_result[:8].tolist()}")

    print("OK: per-host Runner/Executor split round-tripped data sidecar→TPU "
          "(strict transfer guards held)")
    return 0


if __name__ == "__main__":
    sys.exit(main())


# Open questions
# Function arg byte serialization is time consuming, is there a better way to pass structured data (e.g. a batch of requests) from the controller to the sidecar? 
# The controller process will need to spin up N threads to drive each of the sidecar instances independently in production setting, is there a better way to do this? 
