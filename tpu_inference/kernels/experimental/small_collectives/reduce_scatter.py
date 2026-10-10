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
"""Low-latency Pallas ICI reduce-scatter kernel for small token batches.

Uses direct remote VMEM RDMA copies and hardware barrier semaphores to minimize
collective latency on decode-sized token tensors.
"""

import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from tpu_inference.kernels.experimental.small_collectives.common import (
    LANES, barrier, call, push, split_rank, sum_slots, wait_bytes_of)

# A barrier group distinct from the all-reduce kernel's.
COLLECTIVE_ID = 7
# Each block's rows are padded to this multiple: Mosaic refuses a row slice
# of the input that is not aligned to the sublane tile.
SUBLANES = 8


def reference(blocks, core):
    """[cores, cores * rows, hidden] -> core ``core``'s output."""
    rows = blocks.shape[1] // blocks.shape[0]
    mine = blocks[:, core * rows:(core + 1) * rows, :]
    return jnp.sum(mine.astype(jnp.float32), axis=0).astype(blocks.dtype)


def small_reduce_scatter(x,
                         *,
                         num_cores,
                         axis,
                         rank=None,
                         form=None,
                         collective_id=COLLECTIVE_ID):
    """Block r of ``x`` summed onto core r of mesh axis ``axis``, inside a
    shard_map over that axis; ``rank`` as in ``small_all_reduce``."""
    total, hidden = x.shape
    n = num_cores
    if hidden % LANES:
        raise ValueError(f"hidden {hidden} is not a multiple of {LANES} lanes")
    if total % n:
        raise ValueError(f"{total} rows do not split over {n} cores")
    rows = total // n
    padded = -(-rows // SUBLANES) * SUBLANES
    impl = (_hier_reduce_scatter if
            (form or
             ("hier" if n == 8 else "flat")) == "hier" else _reduce_scatter)
    if padded == rows:
        return impl(x, n, axis, rank, collective_id)
    x = jnp.pad(x.reshape(n, rows, hidden),
                ((0, 0), (0, padded - rows), (0, 0)))
    out = impl(x.reshape(n * padded, hidden), n, axis, rank, collective_id)
    return out[:rows]


def _hier_reduce_scatter(x, n, axis, rank, collective_id):
    """2-stage topology-aware Reduce-Scatter on 8 cores (4 chips x 2 cores/chip):
    Stage 1 (Intra-chip): Core `me` pushes the 4 blocks belonging to `partner = me ^ 1`'s
      subcore parity (`r = 2*c + (1 - subcore)` for c in 0..3) to `s1_recv` on `partner`,
      and sums its own 4 parity blocks (`r = 2*c + subcore`) with `s1_recv` into `chip_blocks`.
    Stage 2 (Inter-chip): Core `me` pushes `chip_blocks[c]` for `c != chip` to
      `(c << 1) | subcore` (3 inter-chip DMAs vs 6 in flat reduce-scatter) and sums the 4
      chip blocks into `o_ref`.
    """
    total, hidden = x.shape
    rows = total // n
    dict_ids = rank is not None

    def kernel(*refs):
        me, (
            x_ref,
            o_ref,
            s1_recv,
            chip_blocks,
            s2_recv,
            send1,
            recv1,
            send2,
            recv2,
        ) = split_rank(refs, dict_ids, axis)
        barrier(n, axis, dict_ids)

        subcore = me & jnp.int32(1)
        chip = me >> jnp.int32(1)
        partner = me ^ jnp.int32(1)

        # Stage 1: Push partner's 4 parity blocks over the intra-chip link
        @pl.when(subcore == 0)
        def _():
            for c in range(4):
                push(
                    x_ref.at[pl.ds((2 * c + 1) * rows, rows)],
                    s1_recv.at[c],
                    send1,
                    recv1,
                    partner,
                    axis,
                    dict_ids,
                )

        @pl.when(subcore == 1)
        def _():
            for c in range(4):
                push(
                    x_ref.at[pl.ds((2 * c) * rows, rows)],
                    s1_recv.at[c],
                    send1,
                    recv1,
                    partner,
                    axis,
                    dict_ids,
                )

        wait_bytes_of(s1_recv, recv1)

        @pl.when(subcore == 0)
        def _():
            for c in range(4):
                chip_blocks[c] = (x_ref[pl.ds(
                    (2 * c) * rows, rows), :].astype(jnp.float32) +
                                  s1_recv[c].astype(jnp.float32)).astype(
                                      x_ref.dtype)

        @pl.when(subcore == 1)
        def _():
            for c in range(4):
                chip_blocks[c] = (x_ref[pl.ds(
                    (2 * c + 1) * rows, rows), :].astype(jnp.float32) +
                                  s1_recv[c].astype(jnp.float32)).astype(
                                      x_ref.dtype)

        # Stage 2: Push chip_blocks[c] to peer `(c << 1) | subcore` for c != chip
        for c in range(4):

            @pl.when(jnp.int32(c) != chip)
            def _(c=c):
                push(
                    chip_blocks.at[c],
                    s2_recv.at[chip],
                    send2,
                    recv2,
                    (jnp.int32(c) << jnp.int32(1)) | subcore,
                    axis,
                    dict_ids,
                )

        s2_recv[pl.ds(chip, 1)] = chip_blocks[pl.ds(chip, 1)]
        wait_bytes_of(s2_recv.at[pl.ds(0, 3)], recv2)
        o_ref[...] = sum_slots(s2_recv, 4).astype(o_ref.dtype)

        wait_bytes_of(s1_recv, send1)
        wait_bytes_of(s2_recv.at[pl.ds(0, 3)], send2)

    scratch = [
        pltpu.VMEM((4, rows, hidden), x.dtype),
        pltpu.VMEM((4, rows, hidden), x.dtype),
        pltpu.VMEM((4, rows, hidden), x.dtype),
        *[pltpu.SemaphoreType.DMA] * 4,
    ]
    return call(
        kernel,
        x,
        rows,
        rank,
        scratch,
        collective_id,
        f"small_reduce_scatter_hier_r{rows}",
    )


def _reduce_scatter(x, n, axis, rank, collective_id):
    total, hidden = x.shape
    rows = total // n
    dict_ids = rank is not None

    def kernel(*refs):
        me, (x_ref, o_ref, slices, send_sem,
             recv_sem) = split_rank(refs, dict_ids, axis)
        barrier(n, axis, dict_ids)
        # block r into slot `me` of core r's slices
        for r in range(n):

            @pl.when(jnp.int32(r) != me)
            def _(r=r):
                push(
                    x_ref.at[pl.ds(r * rows, rows)],
                    slices.at[me],
                    send_sem,
                    recv_sem,
                    r,
                    axis,
                    dict_ids,
                )

        # my own block: a static row slice picked by a scalar compare
        for r in range(n):

            @pl.when(jnp.int32(r) == me)
            def _(r=r):
                slices[pl.ds(me, 1)] = x_ref[pl.ds(r * rows, rows), :][None]

        wait_bytes_of(slices.at[pl.ds(0, n - 1)], recv_sem)
        o_ref[...] = sum_slots(slices, n).astype(o_ref.dtype)
        wait_bytes_of(slices.at[pl.ds(0, n - 1)], send_sem)

    scratch = [
        pltpu.VMEM((n, rows, hidden), x.dtype),
        pltpu.SemaphoreType.DMA,
        pltpu.SemaphoreType.DMA,
    ]
    return call(kernel, x, rows, rank, scratch, collective_id,
                f"small_reduce_scatter_r{rows}")
