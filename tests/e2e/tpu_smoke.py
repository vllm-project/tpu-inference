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
"""Multi-host TPU smoke test: every host prints its JAX devices and runs one
tiny psum across the whole slice. Lines are prefixed TPU_SMOKE."""

import os
import sys
import traceback

import jax
import jax.numpy as jnp


def main():
    try:
        jax.distributed.initialize()
    except Exception as e:  # single-host runs don't need it
        print(f"TPU_SMOKE jax.distributed.initialize skipped: {e}", flush=True)
    host = os.uname().nodename
    print(f"TPU_SMOKE host={host} jax={jax.__version__} "
          f"process={jax.process_index()}/{jax.process_count()} "
          f"local={jax.local_device_count()} global={jax.device_count()}",
          flush=True)
    for d in jax.local_devices():
        print(f"TPU_SMOKE host={host} device id={d.id} kind={d.device_kind} "
              f"coords={getattr(d, 'coords', None)} "
              f"core={getattr(d, 'core_on_chip', None)}", flush=True)
    if jax.process_index() == 0:
        print(f"TPU_SMOKE all devices: {jax.devices()}", flush=True)
    # One collective across every device: sum of ones must equal device count.
    x = jnp.ones((jax.device_count(), ), dtype=jnp.float32)
    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
    mesh = Mesh(jax.devices(), ("d", ))
    xs = jax.device_put(x, NamedSharding(mesh, P("d")))
    total = float(jax.jit(lambda a: jnp.sum(a))(xs))
    ok = total == jax.device_count()
    print(f"TPU_SMOKE host={host} psum_total={total} expected="
          f"{jax.device_count()} {'OK' if ok else 'MISMATCH'}", flush=True)
    if not ok:
        sys.exit(1)
    print(f"TPU_SMOKE host={host} DONE", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("TPU_SMOKE FAILED", flush=True)
        traceback.print_exc()
        sys.exit(1)
