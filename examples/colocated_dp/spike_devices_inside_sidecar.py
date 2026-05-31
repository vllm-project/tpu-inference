# #!/usr/bin/env python
# # Copyright 2026 Google LLC
# #
# # Licensed under the Apache License, Version 2.0 (the "License");
# # you may not use this file except in compliance with the License.
# # You may obtain a copy of the License at
# #
# #     http://www.apache.org/licenses/LICENSE-2.0
# """Debug: what does a colocated-python sidecar see, and can we recover TPU
# device handles from an input array's sharding?

# Background: an earlier deployment hit ``AttributeError`` on ``device.coords``
# inside the sidecar, suggesting ``jax.devices()`` there returns CPU devices,
# not TPU. This script probes that question and tests whether passing a TPU
# array IN gives the sidecar usable TPU handles (with ``.coords``) — which
# would unblock the "drive TPU from inside colocated python" design.

# Run on Pathways multi-host (≥ 2 hosts of TPU chips)::

#     JAX_PLATFORMS=proxy python examples/colocated_dp/spike_devices_inside_sidecar.py

# For each Pathways host, the script:
#   1. Pins a function call to that host's colocated CPU device, passing in
#      - a small CPU "pin" array (selects the dispatch host)
#      - a TPU array sharded over that host's local TPU chips
#   2. Inside the sidecar, prints diagnostics on:
#        jax.devices(), jax.local_devices(), jax.devices('tpu'),
#        the TPU-array's sharding.device_set, .coords presence,
#        and whether we can build a Mesh + jit a matmul on those devices.
#   3. The script does NOT depend on tpu-inference.

# Read the per-host stdout to answer:
#   A. Does ``jax.devices()`` inside return TPU or CPU?
#   B. Does the input TPU array expose usable TPU handles via
#      ``array.sharding.device_set``?
#   C. Do those handles have ``.coords``?
#   D. Can a jitted TPU op (matmul) run on them from inside?
# """

from __future__ import annotations

import concurrent.futures as cf
import sys
import traceback
from typing import Any, List

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import colocated_python
from jax.sharding import Mesh, NamedSharding, PartitionSpec
import pathwaysutils

pathwaysutils.initialize()

# def _summarize_device(d: Any) -> str:
#     bits = [f"id={getattr(d, 'id', '?')}",
#             f"kind={getattr(d, 'device_kind', '?')}",
#             f"platform={getattr(d, 'platform', '?')}",
#             f"process_index={getattr(d, 'process_index', '?')}"]
#     if hasattr(d, "coords"):
#         try:
#             bits.append(f"coords={d.coords}")
#         except Exception as e:
#             bits.append(f"coords=<raised {type(e).__name__}>")
#     else:
#         bits.append("coords=<missing>")
#     return "(" + ", ".join(bits) + ")"


# def _print_section(title: str) -> None:
#     bar = "=" * 70
#     print(f"\n{bar}\n{title}\n{bar}", flush=True)


# # ---------------------------------------------------------------------------
# # Colocated functions: runs on the host of the input devices.
# #
# # colocated_python infers dispatch from the input arg's device list and rejects
# # mixing two different device lists in a single call, so we test the two cases
# # separately:
# #   probe_cpu(arr)  — input on COLOCATED CPU devices  (canonical pattern)
# #   probe_tpu(arr)  — input on TPU devices            (the question we want
# #                                                      answered: does it work?)
# #
# # Each variant prints the same diagnostics. Compare the two for each host.
# # ---------------------------------------------------------------------------


# def _make_probe(label: str):
#     @colocated_python.colocated_python
#     def _probe(arr: jax.Array) -> jax.Array:
#         import os
#         import sys as _sys
#         import traceback as _tb
#         import jax as _jax
#         import jax.numpy as _jnp
#         from jax.sharding import Mesh as _Mesh, NamedSharding as _NS, PartitionSpec as _P
#         import numpy as _np

#         def _summary(d):
#             bits = [f"id={getattr(d,'id','?')}",
#                     f"kind={getattr(d,'device_kind','?')}",
#                     f"platform={getattr(d,'platform','?')}",
#                     f"process_index={getattr(d,'process_index','?')}"]
#             try:
#                 if hasattr(d, "logical_task"):
#                     bits.append(f"logical_task={d.logical_task}")
#             except Exception:
#                 pass
#             if hasattr(d, "coords"):
#                 try:
#                     bits.append(f"coords={d.coords}")
#                 except Exception as e:
#                     bits.append(f"coords=<raised {type(e).__name__}>")
#             else:
#                 bits.append("coords=<missing>")
#             return "(" + ", ".join(bits) + ")"

#         tag = f"[sidecar PID={os.getpid()} {label}]"

#         def L(msg):
#             line = f"{tag} {msg}"
#             print(line, file=_sys.stderr, flush=True)
#             print(line, flush=True)

#         L(f"---- probe entered (input is {label}) ----")
#         L(f"jax.__version__={_jax.__version__}")

#         # (A) what does jax.devices() / local_devices() return inside?
#         for name, fn in [("jax.devices()", _jax.devices),
#                          ("jax.local_devices()", _jax.local_devices)]:
#             try:
#                 ds = fn()
#                 L(f"{name} -> {len(ds)} device(s):")
#                 for d in ds:
#                     L(f"   {_summary(d)}")
#             except Exception as e:
#                 L(f"{name} raised {type(e).__name__}: {e}")

#         # (A2) explicit platform filter — does asking for 'tpu' work?
#         for plat in ("tpu", "cpu"):
#             try:
#                 ds = _jax.devices(plat)
#                 L(f"jax.devices({plat!r}) -> {len(ds)} device(s):")
#                 for d in ds:
#                     L(f"   {_summary(d)}")
#             except Exception as e:
#                 L(f"jax.devices({plat!r}) raised {type(e).__name__}: {e}")

#         # (B) inspect the input array — what device handles do we recover?
#         L(f"arr: shape={arr.shape}, dtype={arr.dtype}")
#         try:
#             L(f"arr.sharding: {arr.sharding}")
#         except Exception as e:
#             L(f"arr.sharding raised {type(e).__name__}: {e}")
#         try:
#             ds = list(arr.sharding.device_set)
#             L(f"arr.sharding.device_set -> {len(ds)} device(s):")
#             for d in ds:
#                 L(f"   {_summary(d)}")
#         except Exception as e:
#             L(f"arr.sharding.device_set raised {type(e).__name__}: {e}")
#         try:
#             ds = list(arr.devices())
#             L(f"arr.devices() -> {len(ds)} device(s):")
#             for d in ds:
#                 L(f"   {_summary(d)}")
#         except Exception as e:
#             L(f"arr.devices() raised {type(e).__name__}: {e}")

#         # (C) does .coords work on the recovered handles?
#         recovered = []
#         try:
#             recovered = sorted(arr.sharding.device_set,
#                                key=lambda d: getattr(d, "id", 0))
#             coords_ok = all(hasattr(d, "coords") for d in recovered)
#             L(f"all recovered devices have .coords? {coords_ok}")
#         except Exception as e:
#             L(f"coords probe raised {type(e).__name__}: {e}")

#         # (D) probe operations on the input array directly (no controller-side
#         # mesh; just exercise what's already inferred from the input).
#         try:
#             @_jax.jit
#             def add_one(a):
#                 return a + 1.0

#             y = add_one(arr)
#             _jax.block_until_ready(y)
#             L(f"jit (a + 1) OK; out shape={y.shape}, sharding={y.sharding}")
#         except Exception as e:
#             L(f"jit (a + 1) raised {type(e).__name__}: {e}")

#         # (E) try to BUILD a fresh Mesh from the recovered handles & jit a matmul.
#         try:
#             if not recovered:
#                 L("skip Mesh+matmul probe — no recovered devices")
#             else:
#                 mesh = _Mesh(_np.asarray(recovered), ("model",))
#                 sharding = _NS(mesh, _P(None, "model"))
#                 x = _jax.device_put(
#                     _jnp.ones(arr.shape, _jnp.float32), sharding)

#                 @_jax.jit
#                 def matmul(a):
#                     return _jnp.einsum("ij,jk->ik", a, a.T)

#                 z = matmul(x)
#                 _jax.block_until_ready(z)
#                 L(f"Mesh+jit matmul OK; out shape={z.shape}, "
#                   f"sharding={z.sharding}")
#         except Exception as e:
#             L(f"Mesh+jit matmul raised {type(e).__name__}: {e}")
#             L(_tb.format_exc())

#         L(f"---- probe returning ({label}) ----")
#         return arr

#     return _probe


# probe_cpu = _make_probe("CPU_INPUT")
# probe_tpu = _make_probe("TPU_INPUT")


# # ---------------------------------------------------------------------------
# # Controller side: build per-host CPU + TPU arrays and dispatch.
# # ---------------------------------------------------------------------------


# def main() -> int:
#     _print_section("CONTROLLER")
#     all_devs = jax.devices()
#     for d in all_devs[:8]:
#         print(f"  {_summarize_device(d)}")
#     if len(all_devs) > 8:
#         print(f"  ... +{len(all_devs)-8} more")

#     # Group TPU devices by host.
#     hosts = sorted({d.id//8 for d in all_devs})
#     if len(hosts) < 1:
#         print("ERROR: no devices at all", file=sys.stderr)
#         return 2
#     host_tpu = {h: [d for d in all_devs if d.id // 8 == h]
#                 for h in hosts}

#     # Build, for each host: a colocated-CPU array AND a TPU-sharded array.
#     # We will dispatch the two probe variants separately because colocated_python
#     # requires all input args to live on the same device list.
#     cpu_payloads: List[jax.Array] = []
#     tpu_payloads: List[jax.Array] = []
#     for h in hosts:
#         tpu = host_tpu[h]
#         cpu = colocated_python.colocated_cpu_devices(tpu)
#         print(f"host={h}: found {len(cpu)} colocated CPU devices for "
#               f"{len(tpu)} TPU devices")

#         cpu_mesh = Mesh(np.asarray(cpu), ("x",))
#         cpu_sharding = NamedSharding(cpu_mesh, PartitionSpec())
#         cpu_arr = jax.device_put(np.zeros(1, np.uint8), cpu_sharding)
#         cpu_payloads.append(cpu_arr)

#         tpu_mesh = Mesh(np.asarray(tpu), ("model",))
#         tpu_sharding = NamedSharding(tpu_mesh, PartitionSpec(None, "model"))
#         shape = (64, 64 * len(tpu))
#         tpu_arr = jax.device_put(
#             jnp.ones(shape, dtype=jnp.float32), tpu_sharding)
#         tpu_payloads.append(tpu_arr)
#         print(f"host={h}: TPU chips={len(tpu)}, tpu_arr shape={shape}")

#     # ---- variant 1: input on colocated CPU (canonical pattern) ----------
#     _print_section("DISPATCHING probe_cpu() TO EACH HOST (input on CPU)")
#     with cf.ThreadPoolExecutor(max_workers=len(hosts)) as ex:
#         futs = {h: ex.submit(probe_cpu, cpu_payloads[i])
#                 for i, h in enumerate(hosts)}
#         for h, f in futs.items():
#             try:
#                 out = f.result()
#                 jax.block_until_ready(out)
#                 print(f"host={h}: probe_cpu() returned OK")
#             except Exception as e:
#                 print(f"host={h}: probe_cpu() raised {type(e).__name__}: {e}")
#                 traceback.print_exc()

#     # ---- variant 2: input on TPU (the design we want) -------------------
#     _print_section("DISPATCHING probe_tpu() TO EACH HOST (input on TPU)")
#     with cf.ThreadPoolExecutor(max_workers=len(hosts)) as ex:
#         futs = {h: ex.submit(probe_tpu, tpu_payloads[i])
#                 for i, h in enumerate(hosts)}
#         for h, f in futs.items():
#             try:
#                 out = f.result()
#                 jax.block_until_ready(out)
#                 print(f"host={h}: probe_tpu() returned OK")
#             except Exception as e:
#                 print(f"host={h}: probe_tpu() raised {type(e).__name__}: {e}")
#                 traceback.print_exc()

#     _print_section("DONE — read sidecar logs and compare CPU_INPUT vs TPU_INPUT")
#     print("Per host, for each variant, check:")
#     print("  A. jax.devices() inside -> TPU or CPU?  (the central question)")
#     print("  B. arr.sharding.device_set -> usable handles?")
#     print("  C. recovered handles have .coords?")
#     print("  D. jit(a + 1) on the input array — OK?")
#     print("  E. fresh-Mesh + jit matmul on recovered devices — OK?")
#     return 0


# if __name__ == "__main__":
#     sys.exit(main())


# from jax.experimental import colocated_python

# @colocated_python.colocated_python
# def twice(x):
#   pid = jax.process_index()
#   local_tpu_devices = jax.devices()[pid * 8:(pid + 1) * 8]
#   tpu_devices_mesh = jax.sharding.Mesh(local_tpu_devices, "x")
#   y = jnp.array([1] * 8)
#   y = jax.device_put(
#       y, jax.NamedSharding(tpu_devices_mesh, jax.sharding.PartitionSpec("x"))
#   )
#   @jax.jit
#   def matmul(a):
#       return a+a
#   y = matmul(y)

#   jax.block_until_ready(y)
  
#   # dummy logic
#   out_arrays = []
#   for shard in x.addressable_shards:
#     out_arrays.append(2 * shard.data)
#   return jax.make_array_from_single_device_arrays(
#       sharding=x.sharding, shape=x.shape, arrays=out_arrays
#   )

# tpu_devices = jax.devices()
# print("tpu_devices", tpu_devices)
# cpu_devices = colocated_python.colocated_cpu_devices(tpu_devices)
# cpu_devices_mesh = jax.sharding.Mesh(cpu_devices, "x")
# tpu_devices_mesh = jax.sharding.Mesh(tpu_devices, "x")

# # Construct input that is sharded across all cpu_devices
# x = jnp.array([1] * len(cpu_devices))
# x = jax.device_put(
#     x, jax.NamedSharding(cpu_devices_mesh, jax.sharding.PartitionSpec("x"))
# )

# # Get output that is sharded across all cpu_devices
# out = twice(x)

# # Copy output from cpu_devices into corresponding tpu_devices (without going through the pathways client)
# with (
#     jax.transfer_guard_device_to_host("disallow_explicit"),
#     jax.transfer_guard_host_to_device("disallow_explicit"),
# ):
#   out_tpus = jax.device_put(out, jax.NamedSharding(tpu_devices_mesh, jax.sharding.PartitionSpec("x")))

# print(str(out_tpus))


tpu_devices = jax.devices()
print("tpu_devices", tpu_devices)
cpu_devices = colocated_python.colocated_cpu_devices(tpu_devices)
cpu_devices_mesh = jax.sharding.Mesh(cpu_devices, "x")
tpu_devices_mesh = jax.sharding.Mesh(tpu_devices, "x")

dummy_args = jax.device_put(1, jax.NamedSharding(cpu_devices_mesh, jax.sharding.PartitionSpec()))


class Runner:
  
  def __init__(self):
    print('Runner created')
    self.requests = []

  def add_request(self, x):
    print('request added')
    self.requests.append(x)
    
  def schedule_and_prepare_inputs(self, dummy_arg):
    # dummy logic mimicking sampling. 
    out_arrays = [1]
    for shard in x.addressable_shards:
      out_arrays.append(2 * shard.data)
    return jax.make_array_from_single_device_arrays(
        sharding=x.sharding, shape=x.shape, arrays=out_arrays
    )

class Executor:
  
  def __init__(self, mesh):
    print('Executor created')
    self.mesh = mesh

  def __del__(self):
    print('Executor destroyed')
    
  def execute_on_tpu(self, cpu_inputs):
    tpu_inputs = jax.device_put(cpu_inputs, jax.NamedSharding(self.mesh, jax.sharding.PartitionSpec("x")))
    return tpu_inputs + tpu_inputs
    
    
meshA = jax.sharding.Mesh(tpu_devices[:8], "x")
meshB = jax.sharding.Mesh(tpu_devices[8:], "x")

RunnerA = colocated_python.colocated_python_class("RunnerA", Runner)
RunnerB = colocated_python.colocated_python_class("RunnerB", Runner)

ExecutorA = Executor(meshA)
ExecutorB = Executor(meshB)

requests = []

def user_adds_request():
  requests.append("what is the answer to the ultimate question of life?")
  
prepared_inputsA = RunnerA.add_request(1)  
prepared_inputsB = RunnerB.add_request(2)  


outputA = ExecutorA.execute_on_tpu(prepared_inputsA)
outputB = ExecutorB.execute_on_tpu(prepared_inputsB)



# class Adder:

#   def __init__(self, increment):
#     print('Adder created')
#     self.increment = increment

#   def __del__(self):
#     print('Adder destroyed')

#   def add(self, x):
#     return x + self.increment


# Adder = colocated_python.colocated_python_class(Adder)
# adder = Adder(1)
# x = jax.device_put(1, jax.NamedSharding(cpu_devices_mesh, jax.sharding.PartitionSpec()))
# y = adder.add(x)
# print(y)
