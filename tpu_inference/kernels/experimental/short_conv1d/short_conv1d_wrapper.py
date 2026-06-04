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
"""Short causal 1D convolution wrapper -- dispatch and public API."""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp

from tpu_inference.kernels.experimental.short_conv1d.decoding_short_conv1d_kernel import \
    decoding_short_conv1d
from tpu_inference.kernels.experimental.short_conv1d.prefill_short_conv1d_kernel import \
    prefill_short_conv1d


@functools.partial(
    jax.jit,
    static_argnames=("decoding_block_n", "prefill_block_n"),
    donate_argnames=("x", "conv_state"),
)
def short_conv1d(
    x: jax.Array,
    weight: jax.Array,
    conv_state: jax.Array,
    cu_seqlens: jax.Array,
    state_indices: jax.Array,
    distribution: jax.Array,
    has_initial_state: jax.Array,
    *,
    decoding_block_n: int | None = None,
    prefill_block_n: int | None = None,
):
    """Short causal 1D conv with mixed decode + prefill dispatch.

    Args:
        x: ``[T, H, D]`` packed token activations.
        weight: ``[W, H, D]`` causal kernel.
        conv_state: ``[N_states, W-1, H, D]`` history slots.
        cu_seqlens: ``[num_seqs+1]`` int32 cumulative sequence lengths.
        state_indices: ``[max_reqs]`` int32, request -> conv_state slot.
        distribution: ``[2]`` int32 ``[decoding, total]``: ``[0, decoding)``
            is dispatched to the decode kernel, ``[decoding, total)`` to
            the prefill kernel.
        has_initial_state: ``[max_reqs]`` int32 / bool.
        decoding_block_n: chunk size for the decode kernel (``None`` ->
            its default).
        prefill_block_n: chunk size for the prefill kernel (``None`` ->
            its default).

    Returns:
        ``(out, new_conv_state)``.  Decode and prefill regions touch
        disjoint sets of ``conv_state`` slots, so the chained donation
        leaves both regions correctly written.
    """
    # Each sub-kernel aliases its input ``x`` to its output, so x is
    # overwritten in place in the active region only:
    #   * decoding kernel writes ``x[:decode_end]`` in place
    #   * prefill kernel writes ``x[decode_end:total]`` in place
    # ``lax.cond`` skips the entire pallas_call dispatch (incl. its setup
    # cost ~5us) when the corresponding region is empty.  The kernels are
    # also internally no-ops via ``grid_dim=0`` -- the cond is purely a
    # host-side optimization.
    decode_end = distribution[0]
    total = distribution[1]

    def _do_decode(operands):
        x_, weight_, conv_state_, state_indices_, distribution_, has_init_ = operands
        return decoding_short_conv1d(
            x_,
            weight_,
            conv_state_,
            state_indices_,
            distribution_,
            has_init_,
            block_n=decoding_block_n,
        )

    def _skip_decode(operands):
        x_, _, conv_state_, _, _, _ = operands
        return x_, conv_state_

    x, state_after_decode = jax.lax.cond(
        decode_end > 0,
        _do_decode,
        _skip_decode,
        (x, weight, conv_state, state_indices, distribution, has_initial_state),
    )

    def _do_prefill(operands):
        (
            x_,
            weight_,
            state_,
            cu_,
            si_,
            distribution_,
            has_init_,
        ) = operands
        return prefill_short_conv1d(
            x_,
            weight_,
            state_,
            cu_,
            si_,
            distribution_,
            has_init_,
            block_n=prefill_block_n,
            region_start_idx=0,
        )

    def _skip_prefill(operands):
        x_, _, state_, _, _, _, _ = operands
        return x_, state_

    out, state_out = jax.lax.cond(
        decode_end < total,
        _do_prefill,
        _skip_prefill,
        (
            x,
            weight,
            state_after_decode,
            cu_seqlens,
            state_indices,
            distribution,
            has_initial_state,
        ),
    )

    return out, state_out
