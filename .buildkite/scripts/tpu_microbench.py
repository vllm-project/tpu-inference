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
"""Time the same small JAX programs on kube and on bare metal.

Each probe isolates one resource a serving step depends on: chip compute, HBM
bandwidth, ICI all-reduce, host dispatch and host-to-device transfer. If a
probe differs between the two, that resource is where the node differs.
"""
import json
import statistics
import time

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import NamedSharding, PartitionSpec as P

devs = jax.devices()
d0 = devs[0]
out = {"devices": len(devs), "kind": d0.device_kind}


def per_call(f, *args, n):
    """Mean seconds per call with calls queued back to back."""
    f(*args).block_until_ready()
    t = time.perf_counter()
    for _ in range(n):
        r = f(*args)
    r.block_until_ready()
    return (time.perf_counter() - t) / n


def round_trips(f, x, n):
    """p50 and p90 microseconds of dispatch plus wait, one call at a time."""
    f(x).block_until_ready()
    ts = []
    for _ in range(n):
        t = time.perf_counter()
        f(x).block_until_ready()
        ts.append((time.perf_counter() - t) * 1e6)
    ts.sort()
    return round(ts[len(ts) // 2], 1), round(ts[int(len(ts) * 0.9)], 1)


# Compute: an 8192^3 bf16 matmul on one chip.
a = jax.device_put(jnp.ones((8192, 8192), jnp.bfloat16), d0)
t = per_call(jax.jit(lambda x: x @ x), a, n=50)
out["matmul_tflops"] = round(2 * 8192**3 / t / 1e12, 1)

# HBM: read and write 1 GiB on one chip.
b = jax.device_put(jnp.ones((1 << 29, ), jnp.bfloat16), d0)
t = per_call(jax.jit(lambda x: x * 1.0001), b, n=50)
out["hbm_gbps"] = round(2 * (1 << 30) / t / 1e9, 1)
del a, b

# ICI: all-reduce of one step's activations over every chip - 256 tokens (a
# decode step) and 2048 tokens (a full prefill step) of hidden size 4096.
mesh = jax.make_mesh((len(devs), ), ("i", ))
sharded = NamedSharding(mesh, P("i"))
replicated = NamedSharding(mesh, P())
allreduce = jax.jit(lambda x: x.sum(axis=0, keepdims=True),
                    out_shardings=replicated)
for tokens in (256, 2048):
    x = jax.device_put(jnp.ones((len(devs), tokens * 4096), jnp.bfloat16),
                       sharded)
    out[f"allreduce_{tokens}tok_us"] = round(
        per_call(allreduce, x, n=200) * 1e6, 1)

# Host dispatch: a scalar op on one chip, and the same over all chips (every
# TP=8 serving program is an 8-device launch).
inc = jax.jit(lambda x: x + 1)
out["dispatch_1chip_p50_p90_us"] = round_trips(
    inc, jax.device_put(jnp.zeros((), jnp.int32), d0), 2000)
out["dispatch_1chip_queued_us"] = round(
    per_call(inc, jax.device_put(jnp.zeros((), jnp.int32), d0), n=2000) * 1e6,
    1)
z8 = jax.device_put(jnp.zeros((len(devs), ), jnp.int32), sharded)
out["dispatch_8chip_p50_p90_us"] = round_trips(inc, z8, 2000)
out["dispatch_8chip_queued_us"] = round(per_call(inc, z8, n=2000) * 1e6, 1)

# Host to device: the per-step metadata blob size, then bulk bandwidth.
small = np.zeros(16 << 10, np.int32)
ts = []
for _ in range(500):
    t = time.perf_counter()
    jax.device_put(small, d0).block_until_ready()
    ts.append((time.perf_counter() - t) * 1e6)
out["h2d_64KiB_p50_us"] = round(statistics.median(ts), 1)
big = np.ones(64 << 20, np.int32)
jax.device_put(big, d0).block_until_ready()
t = time.perf_counter()
for _ in range(5):
    jax.device_put(big, d0).block_until_ready()
out["h2d_256MiB_gbps"] = round(5 * big.nbytes / (time.perf_counter() - t) / 1e9,
                               1)

print("MICROBENCH " + json.dumps(out))
