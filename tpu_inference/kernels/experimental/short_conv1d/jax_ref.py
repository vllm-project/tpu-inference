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
"""Pure-JAX reference for short_conv1d kernels (prefill and decode).

The prefill formulation handles decoding for free: a decode token is
just a sequence of length 1.  Build ``cu_seqlens = jnp.arange(N+1)``
(plus any trailing pad-to-``max_reqs+1``) and the same reference
produces the decode result.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


def short_conv1d_reference(
    x: jax.Array,
    weight: jax.Array,
    conv_state: jax.Array,
    cu_seqlens: jax.Array,
    state_indices: jax.Array,
    distribution: jax.Array,
    has_initial_state: jax.Array,
    *,
    region_start_idx: int = 1,
):
    """Pure-JAX reference matching ``prefill_short_conv1d`` semantics.

    Iterates the sequences in
    ``[distribution[region_start_idx], distribution[region_start_idx + 1])``;
    for each sequence ``s`` (length ``seq_len = cu_seqlens[s+1] - cu_seqlens[s]``):

      * loads ``conv_state[state_indices[s]]`` if ``has_initial_state[s]``
        is set, else uses zeros;
      * convolves ``[state | x_seq]`` with ``weight``;
      * writes the convolved ``seq_len`` rows back to ``out``;
      * stores the trailing ``W-1`` rows of ``[state | x_seq]`` into
        ``new_conv_state[state_indices[s]]``.

    Decode-only callers can pass ``cu_seqlens = jnp.arange(N+1, ...)``
    (one token per sequence) and ``distribution = [0, 0, N]`` with
    ``region_start_idx=1``; the reference handles ``seq_len == 1``
    correctly via the same merged convolution.

    Args:
        x: ``[T, H, D]`` packed tokens.
        weight: ``[W, H, D]``.
        conv_state: ``[N_states, W-1, H, D]``.
        cu_seqlens: ``[max_reqs+1]`` int.
        state_indices: ``[max_reqs]`` int.
        distribution: int array; ``distribution[region_start_idx]`` and
            ``distribution[region_start_idx + 1]`` give the sequence range.
        has_initial_state: ``[max_reqs]`` int/bool.
        region_start_idx: which slice of ``distribution`` to iterate.

    Returns:
        ``(out, new_conv_state)`` with shapes matching the inputs.
    """
    T, H, D = x.shape
    W = weight.shape[0]
    seq_lo = int(distribution[region_start_idx])
    seq_hi = int(distribution[region_start_idx + 1])

    out = jnp.zeros_like(x)
    new_conv_state = conv_state

    for s in range(seq_lo, seq_hi):
        seq_start = int(cu_seqlens[s])
        seq_end = int(cu_seqlens[s + 1])
        seq_len = seq_end - seq_start
        state_idx = int(state_indices[s])

        x_seq = x[seq_start:seq_end]
        if bool(has_initial_state[s]):
            state_seq = conv_state[state_idx]
        else:
            state_seq = jnp.zeros((W - 1, H, D), dtype=x.dtype)

        full = jnp.concatenate([state_seq, x_seq], axis=0)

        out_seq = jnp.zeros((seq_len, H, D), jnp.float32)
        for k in range(W):
            out_seq = out_seq + full[k:k + seq_len].astype(
                jnp.float32) * weight[k]
        out = out.at[seq_start:seq_end].set(out_seq.astype(x.dtype))

        new_state_seq = full[-(W - 1):]
        new_conv_state = new_conv_state.at[state_idx].set(new_state_seq)

    return out, new_conv_state
