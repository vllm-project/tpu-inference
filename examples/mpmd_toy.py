# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Toy MPMD worker: one independent JAX process on a slice of TPU chips.

This script is a *worker only*. It reads its TPU chip assignment from the
libtpu env vars that the caller sets, so you launch it directly with the
env vars on the command line:

    TPU_VISIBLE_CHIPS=0,1 TPU_CHIPS_PER_PROCESS_BOUNDS=1,2,1 \\
    TPU_PROCESS_BOUNDS=1,1,1 TPU_PROCESS_PORT=8476 \\
    python examples/mpmd_toy.py

Launch two of them on disjoint chip sets (different TPU_VISIBLE_CHIPS and
TPU_PROCESS_PORT) to get Multiple-Program Multiple-Data: two independent
JAX runtimes on one host, with no collectives between them. See
examples/run_mpmd_toy.sh for a launcher that starts two side by side.

The worker keeps a tensor resident and recomputes it in a loop for
MPMD_DURATION_S seconds (default 30), so concurrently launched workers
overlap in time instead of one finishing before the other starts.
"""

import os
import time

TAG = os.environ.get("MPMD_TASK_ID", "?")
DURATION_S = float(os.environ.get("MPMD_DURATION_S", "30"))


def log(msg: str) -> None:
    print(f"[worker {TAG}] {msg}", flush=True)


def main() -> None:
    import jax
    import jax.numpy as jnp
    from jax.sharding import Mesh, NamedSharding
    from jax.sharding import PartitionSpec as P

    devices = jax.devices()
    log(f"TPU_VISIBLE_CHIPS={os.environ.get('TPU_VISIBLE_CHIPS')!r} "
        f"process_count={jax.process_count()} "
        f"devices={[d.id for d in devices]}")

    # A tensor sharded across this worker's own chips.
    mesh = Mesh(devices, ('x', ))
    sharding = NamedSharding(mesh, P('x'))
    x = jax.device_put(jnp.ones((len(devices), 1024)), sharding)

    @jax.jit
    def step(t):
        return jnp.sin(t) + 1.0

    log(f"running for {DURATION_S:.0f}s ...")
    start = time.time()
    it = 0
    while time.time() - start < DURATION_S:
        x = jax.block_until_ready(step(x))
        it += 1
        if it % 50 == 0:
            log(f"iter {it}, elapsed {time.time() - start:.1f}s, "
                f"x[0,0]={float(x[0, 0]):.4f}")
        time.sleep(0.05)
    log(f"done after {it} iters, {time.time() - start:.1f}s")


if __name__ == "__main__":
    main()
