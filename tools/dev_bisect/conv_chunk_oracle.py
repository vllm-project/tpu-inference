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
"""Oracle test: chunked ragged_conv1d must equal a direct causal conv.

Threads one logical sequence through ragged_conv1d_mixed_prefill in chunks
(including 1- and 2-token chunks — the gsm8k-at-conc-8 remainder shape),
carrying conv_state between chunks, and compares every chunk's output and
the final state against a straight causal depthwise conv of the whole
sequence. Run with PYTHONPATH at the stack under test.
"""
import os
import sys

import numpy as np

if os.environ.get("ORACLE_TPU") != "1":
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
import jax.numpy as jnp

from tpu_inference.layers.common.ragged_conv1d_jax import ragged_conv1d

K = 4  # kernel_size
DIM = int(os.environ.get("ORACLE_DIM", "1024"))


def direct_causal_conv(x, w):
    # x: (T, DIM); w: (DIM, 1, K)  depthwise causal
    T = x.shape[0]
    xp = np.concatenate([np.zeros((K - 1, DIM), np.float32), x], axis=0)
    out = np.zeros((T, DIM), np.float32)
    for t in range(T):
        window = xp[t:t + K]  # (K, DIM)
        out[t] = np.einsum("kd,dk->d", window, w[:, 0, :])
    return out


def run_chunked(x, w, schedule):
    """Feed x through ragged_conv1d in chunks of the given sizes; each call
    is a single-sequence batch (num_seqs=1) like one request's continuation
    chunk inside a mixed batch."""
    conv_state = jnp.zeros((3, K - 1, DIM), jnp.float32)  # slots 0..2
    sidx = jnp.array([1], jnp.int32)
    outs = []
    pos = 0
    for ci, size in enumerate(schedule):
        chunk = jnp.asarray(x[pos:pos + size])
        qsl = jnp.array([0, size], jnp.int32)
        dist = jnp.array([0, 0, 1], jnp.int32)  # one prefill seq
        has_init = jnp.array([1 if ci > 0 else 0], jnp.int32)
        out, conv_state = ragged_conv1d(chunk,
                                        conv_state,
                                        jnp.asarray(w),
                                        None,
                                        qsl,
                                        sidx,
                                        dist,
                                        has_init,
                                        kernel_size=K)
        outs.append(np.asarray(out))
        pos += size
    return np.concatenate(outs, axis=0), np.asarray(conv_state[1])


rng = np.random.default_rng(0)
w = rng.normal(size=(DIM, 1, K)).astype(np.float32) * 0.3

fail = 0
schedules = [
    [7, 1, 6],
    [5, 2, 1, 8],
    [1, 1, 1, 1],
    [12],
    [3, 1, 2, 1, 5],
    [2, 2, 9, 1],
]
for si, sched in enumerate(schedules):
    T = sum(sched)
    x = rng.normal(size=(T, DIM)).astype(np.float32) * 0.5
    ref = direct_causal_conv(x, w)
    got, final_state = run_chunked(x, w, sched)
    err = np.abs(got - ref).max()
    ref_state = np.concatenate([np.zeros((K - 1, DIM), np.float32), x],
                               axis=0)[-(K - 1):]
    serr = np.abs(final_state - ref_state).max()
    status = "OK " if err < 1e-4 and serr < 1e-4 else "FAIL"
    if status == "FAIL":
        fail += 1
    print(f"[conv-oracle] sched={sched} out_err={err:.3e} "
          f"state_err={serr:.3e} {status}")
print(
    f"[conv-oracle] DIM={DIM} RESULT: {'FAIL' if fail else 'PASS'} ({fail} bad)"
)
sys.exit(1 if fail else 0)
