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
"""All-reduce of a small bf16 [rows, hidden] block as one Pallas kernel.

    one-shot      Every core pushes its whole block to every other core and
                  sums the arrivals locally (XOR static-slot addressing on 8
                  cores eliminates @pl.when branches and self-copies).
    hier-halves   On 8 cores (4 chips x 2 cores/chip), intra-chip reduce-scatter
                  of (rows, hidden/2) with twin `me ^ 1`, 4-chip all-reduce of
                  (rows, hidden/2) across `me ^ {2, 4, 6}`, then intra-chip
                  all-gather with `me ^ 1` (5 DMAs total, used at rows == 8).
    two-shot      Core j owns columns [j * own, (j + 1) * own). Every core
                  pushes each owner its columns; the owner sums them and pushes
                  the sum back to everyone (used for 8 < rows < 24).
    hier-2stage   On 8 cores (rows >= 24), 4-step hierarchical reduce-scatter +
                  all-gather: intra-chip RS (rows, hidden/2) with `me ^ 1` ->
                  4-chip RS (rows, hidden/8) with `me ^ {2, 4, 6}` -> 4-chip AG
                  -> intra-chip AG. Cuts inter-chip DMA count and inter-chip
                  bytes in half compared to flat two-shot.
"""

import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from tpu_inference.kernels.experimental.small_collectives.common import (
    LANES, barrier, call, push, split_rank, sum_slots, wait_bytes_of)

# One-shot up to this many bytes per core, two-shot above: one-shot transfers
# full buffers in a single round, whereas two-shot halves per-link volume across
# two rounds (crossing over near ~6 rows of H=4096 bf16 across 8 cores).
ONE_SHOT_MAX_BYTES = 32 * 1024
# Row threshold at and above which 8-core all-reduce switches from flat
# two-shot to 4-step topology-aware intra-chip/inter-chip hierarchical RS+AG.
HIER_2STAGE_MIN_ROWS = 24
# Barrier semaphore group. Every kernel compiled with the same id in one
# program shares it; the fused EP MoE kernels use 0.
COLLECTIVE_ID = 5


def form_for(rows, hidden, itemsize):
    """The form for a block of rows x hidden elements of itemsize bytes."""
    return "oneshot" if rows * hidden * itemsize <= ONE_SHOT_MAX_BYTES else "twoshot"


def dispatch_form_for(rows, hidden, itemsize, num_cores=8):
    """Topology-aware form selector for `num_cores == 8` (tpu7x-8)."""
    if rows * hidden * itemsize <= ONE_SHOT_MAX_BYTES:
        return "oneshot_xor" if num_cores == 8 else "oneshot"
    if num_cores == 8:
        if rows == 8:
            return "hier_halves"
        if rows >= HIER_2STAGE_MIN_ROWS:
            return "hier_2stage"
    return "twoshot"


form_for_8cores = dispatch_form_for


def reference(blocks):
    """The float32 sum over the cores' blocks [cores, rows, hidden], rounded
    once to the blocks' dtype."""
    return jnp.sum(blocks.astype(jnp.float32), axis=0).astype(blocks.dtype)


def _oneshot(x, n, axis, rank, collective_id):
    rows, hidden = x.shape
    dict_ids = rank is not None

    def kernel(*refs):
        me, (x_ref, o_ref, arrivals, send_sem,
             recv_sem) = split_rank(refs, dict_ids, axis)
        barrier(n, axis, dict_ids)
        for j in range(n):  # my block into slot `me` of every other core

            @pl.when(jnp.int32(j) != me)
            def _(j=j):
                push(x_ref, arrivals.at[me], send_sem, recv_sem, j, axis,
                     dict_ids)

        arrivals[pl.ds(me, 1)] = x_ref[...][None]
        wait_bytes_of(arrivals.at[pl.ds(0, n - 1)], recv_sem)
        o_ref[...] = sum_slots(arrivals, n).astype(o_ref.dtype)
        wait_bytes_of(arrivals.at[pl.ds(0, n - 1)], send_sem)

    scratch = [
        pltpu.VMEM((n, rows, hidden), x.dtype),
        pltpu.SemaphoreType.DMA,
        pltpu.SemaphoreType.DMA,
    ]
    return call(
        kernel,
        x,
        rows,
        rank,
        scratch,
        collective_id,
        f"small_all_reduce_oneshot_r{rows}",
    )


def _oneshot_xor(x, n, axis, rank, collective_id):
    """1-stage flat all-reduce using XOR peer addressing (`me ^ k` for k in 1..n-1).
    Eliminates all @pl.when branches, dynamic VMEM slot offsets, and self-copies.
    """
    rows, hidden = x.shape
    dict_ids = rank is not None

    def kernel(*refs):
        me, (x_ref, o_ref, arrivals, send_sem,
             recv_sem) = split_rank(refs, dict_ids, axis)
        barrier(n, axis, dict_ids)
        for k in range(1, n):
            push(
                x_ref,
                arrivals.at[k - 1],
                send_sem,
                recv_sem,
                me ^ jnp.int32(k),
                axis,
                dict_ids,
            )
        acc = x_ref[...].astype(jnp.float32)
        wait_bytes_of(arrivals, recv_sem)
        for k in range(n - 1):
            acc = acc + arrivals[k].astype(jnp.float32)
        o_ref[...] = acc.astype(o_ref.dtype)
        wait_bytes_of(arrivals, send_sem)

    scratch = [
        pltpu.VMEM((n - 1, rows, hidden), x.dtype),
        pltpu.SemaphoreType.DMA,
        pltpu.SemaphoreType.DMA,
    ]
    return call(
        kernel,
        x,
        rows,
        rank,
        scratch,
        collective_id,
        f"small_all_reduce_oneshot_xor_r{rows}",
    )


def _hier_halves_col_zcopy(x, n, axis, rank, collective_id):
    """3-stage topology-aware column-half (rows, hidden // 2) all-reduce on 8 cores:
    Stage 1: Intra-chip RS of (rows, half) with `me ^ 1` directly from `x_ref`.
    Stage 2: Inter-chip 4-way all-reduce of (rows, half) across `me ^ {2, 4, 6}`.
    Stage 3: Intra-chip AG of (rows, half) with `me ^ 1` directly into `o_ref`.
    """
    rows, hidden = x.shape
    half = hidden // 2
    dict_ids = rank is not None

    def kernel(*refs):
        me, (
            x_ref,
            o_ref,
            s1_recv,
            s2_send,
            s2_recv,
            send1,
            recv1,
            send2,
            recv2,
            send3,
            recv3,
        ) = split_rank(refs, dict_ids, axis)
        barrier(n, axis, dict_ids)

        subcore = me & jnp.int32(1)
        partner = me ^ jnp.int32(1)

        @pl.when(subcore == 0)
        def _():
            push(
                x_ref.at[:, pl.ds(half, half)],
                s1_recv,
                send1,
                recv1,
                partner,
                axis,
                dict_ids,
            )

        @pl.when(subcore == 1)
        def _():
            push(
                x_ref.at[:, pl.ds(0, half)],
                s1_recv,
                send1,
                recv1,
                partner,
                axis,
                dict_ids,
            )

        wait_bytes_of(s1_recv, recv1)

        @pl.when(subcore == 0)
        def _():
            s2_send[...] = (x_ref[:, pl.ds(0, half)].astype(jnp.float32) +
                            s1_recv[...].astype(jnp.float32)).astype(
                                x_ref.dtype)

        @pl.when(subcore == 1)
        def _():
            s2_send[...] = (x_ref[:, pl.ds(half, half)].astype(jnp.float32) +
                            s1_recv[...].astype(jnp.float32)).astype(
                                x_ref.dtype)

        for s in (1, 2, 3):
            push(
                s2_send,
                s2_recv.at[s - 1],
                send2,
                recv2,
                me ^ jnp.int32(2 * s),
                axis,
                dict_ids,
            )
        acc = s2_send[...].astype(jnp.float32)
        wait_bytes_of(s2_recv, recv2)
        for s in range(3):
            acc = acc + s2_recv[s].astype(jnp.float32)
        final_half = acc.astype(o_ref.dtype)

        @pl.when(subcore == 0)
        def _():
            o_ref[:, pl.ds(0, half)] = final_half
            push(
                o_ref.at[:, pl.ds(0, half)],
                o_ref.at[:, pl.ds(0, half)],
                send3,
                recv3,
                partner,
                axis,
                dict_ids,
            )

        @pl.when(subcore == 1)
        def _():
            o_ref[:, pl.ds(half, half)] = final_half
            push(
                o_ref.at[:, pl.ds(half, half)],
                o_ref.at[:, pl.ds(half, half)],
                send3,
                recv3,
                partner,
                axis,
                dict_ids,
            )

        wait_bytes_of(s1_recv, recv3)
        wait_bytes_of(s1_recv, send1)
        wait_bytes_of(s2_recv, send2)
        wait_bytes_of(s1_recv, send3)

    scratch = [
        pltpu.VMEM((rows, half), x.dtype),
        pltpu.VMEM((rows, half), x.dtype),
        pltpu.VMEM((3, rows, half), x.dtype),
        *[pltpu.SemaphoreType.DMA] * 6,
    ]
    return call(
        kernel,
        x,
        rows,
        rank,
        scratch,
        collective_id,
        f"small_all_reduce_hier_halves_r{rows}",
    )


def _twoshot(x, n, axis, rank, collective_id):
    rows, hidden = x.shape
    own = hidden // n
    dict_ids = rank is not None

    def kernel(*refs):
        me, refs = split_rank(refs, dict_ids, axis)
        x_ref, o_ref, mine, slices, gathered, send1, recv1, send2, recv2 = refs
        barrier(n, axis, dict_ids)
        for j in range(n):  # slot j: the columns core j owns
            mine[j] = x_ref[:, pl.ds(j * own, own)]
        # round 1: owner j's columns into slot `me` of its slices
        for j in range(n):

            @pl.when(jnp.int32(j) != me)
            def _(j=j):
                push(mine.at[j], slices.at[me], send1, recv1, j, axis,
                     dict_ids)

        slices[pl.ds(me, 1)] = mine[pl.ds(me, 1)]
        wait_bytes_of(slices.at[pl.ds(0, n - 1)], recv1)
        gathered[pl.ds(me, 1)] = sum_slots(slices,
                                           n).astype(gathered.dtype)[None]
        # round 2: my summed columns into slot `me` of every other core
        for j in range(n):

            @pl.when(jnp.int32(j) != me)
            def _(j=j):
                push(gathered.at[me], gathered.at[me], send2, recv2, j, axis,
                     dict_ids)

        wait_bytes_of(gathered.at[pl.ds(0, n - 1)], recv2)
        for j in range(n):
            o_ref[:, pl.ds(j * own, own)] = gathered[j]
        wait_bytes_of(slices.at[pl.ds(0, n - 1)], send1)
        wait_bytes_of(gathered.at[pl.ds(0, n - 1)], send2)

    scratch = [
        pltpu.VMEM((n, rows, own), x.dtype),
        pltpu.VMEM((n, rows, own), x.dtype),
        pltpu.VMEM((n, rows, own), x.dtype),
        *[pltpu.SemaphoreType.DMA] * 4,
    ]
    return call(
        kernel,
        x,
        rows,
        rank,
        scratch,
        collective_id,
        f"small_all_reduce_twoshot_r{rows}",
    )


def _hier_2stage_rs_ag(x, n, axis, rank, collective_id):
    """4-step hierarchical Reduce-Scatter + All-Gather on 8 cores (4 chips x 2 cores/chip):
    Step 1 (Intra-chip RS): exchange (rows, hidden/2) with `me ^ 1` (1 intra-chip DMA) -> sum (rows, hidden/2).
    Step 2 (4-Chip RS): split (rows, hidden/2) into 4x (rows, hidden/8), exchange 3 slices with
      `me ^ {2, 4, 6}` (3 inter-chip DMAs) -> sum (rows, hidden/8).
    Step 3 (4-Chip AG): push (rows, hidden/8) to `me ^ {2, 4, 6}` (3 inter-chip DMAs) -> assemble (rows, hidden/2).
    Step 4 (Intra-chip AG): push (rows, hidden/2) to `me ^ 1` (1 intra-chip DMA) -> write (rows, hidden).
    """
    rows, hidden = x.shape
    half = hidden // 2
    own = hidden // n
    dict_ids = rank is not None

    def kernel(*refs):
        me, (
            x_ref,
            o_ref,
            s1_send,
            s1_recv,
            s1_sum,
            s2_mine,
            s2_slices,
            s3_gathered,
            s4_send,
            s4_recv,
            send1,
            recv1,
            send2,
            recv2,
            send3,
            recv3,
            send4,
            recv4,
        ) = split_rank(refs, dict_ids, axis)
        barrier(n, axis, dict_ids)

        subcore = me & jnp.int32(1)
        chip = me >> jnp.int32(1)
        partner = me ^ jnp.int32(1)

        # Step 1: Intra-chip RS of (rows, half)
        @pl.when(subcore == 0)
        def _():
            s1_send[...] = x_ref[:, pl.ds(half, half)]

        @pl.when(subcore == 1)
        def _():
            s1_send[...] = x_ref[:, pl.ds(0, half)]

        push(s1_send, s1_recv, send1, recv1, partner, axis, dict_ids)
        wait_bytes_of(s1_recv, recv1)

        @pl.when(subcore == 0)
        def _():
            s1_sum[...] = (x_ref[:, pl.ds(0, half)].astype(jnp.float32) +
                           s1_recv[...].astype(jnp.float32)).astype(
                               x_ref.dtype)

        @pl.when(subcore == 1)
        def _():
            s1_sum[...] = (x_ref[:, pl.ds(half, half)].astype(jnp.float32) +
                           s1_recv[...].astype(jnp.float32)).astype(
                               x_ref.dtype)

        # Step 2: 4-Chip RS of (rows, own) across the 4 chips
        for c in range(4):
            s2_mine[c] = s1_sum[:, pl.ds(c * own, own)]
        for c in range(4):

            @pl.when(jnp.int32(c) != chip)
            def _(c=c):
                push(
                    s2_mine.at[c],
                    s2_slices.at[chip],
                    send2,
                    recv2,
                    (jnp.int32(c) << jnp.int32(1)) | subcore,
                    axis,
                    dict_ids,
                )

        s2_slices[pl.ds(chip, 1)] = s2_mine[pl.ds(chip, 1)]
        wait_bytes_of(s2_slices.at[pl.ds(0, 3)], recv2)
        s3_gathered[pl.ds(chip, 1)] = sum_slots(s2_slices,
                                                4).astype(x_ref.dtype)[None]

        # Step 3: 4-Chip AG of (rows, own) across the 4 chips
        for c in range(4):

            @pl.when(jnp.int32(c) != chip)
            def _(c=c):
                push(
                    s3_gathered.at[chip],
                    s3_gathered.at[chip],
                    send3,
                    recv3,
                    (jnp.int32(c) << jnp.int32(1)) | subcore,
                    axis,
                    dict_ids,
                )

        wait_bytes_of(s3_gathered.at[pl.ds(0, 3)], recv3)
        for c in range(4):
            s4_send[:, pl.ds(c * own, own)] = s3_gathered[c]

        # Step 4: Intra-chip AG of (rows, half) with partner
        push(s4_send, s4_recv, send4, recv4, partner, axis, dict_ids)

        @pl.when(subcore == 0)
        def _():
            o_ref[:, pl.ds(0, half)] = s4_send[...]

        @pl.when(subcore == 1)
        def _():
            o_ref[:, pl.ds(half, half)] = s4_send[...]

        wait_bytes_of(s4_recv, recv4)

        @pl.when(subcore == 0)
        def _():
            o_ref[:, pl.ds(half, half)] = s4_recv[...]

        @pl.when(subcore == 1)
        def _():
            o_ref[:, pl.ds(0, half)] = s4_recv[...]

        wait_bytes_of(s1_send, send1)
        wait_bytes_of(s2_slices.at[pl.ds(0, 3)], send2)
        wait_bytes_of(s3_gathered.at[pl.ds(0, 3)], send3)
        wait_bytes_of(s4_send, send4)

    scratch = [
        pltpu.VMEM((rows, half), x.dtype),
        pltpu.VMEM((rows, half), x.dtype),
        pltpu.VMEM((rows, half), x.dtype),
        pltpu.VMEM((4, rows, own), x.dtype),
        pltpu.VMEM((4, rows, own), x.dtype),
        pltpu.VMEM((4, rows, own), x.dtype),
        pltpu.VMEM((rows, half), x.dtype),
        pltpu.VMEM((rows, half), x.dtype),
        *[pltpu.SemaphoreType.DMA] * 8,
    ]
    return call(
        kernel,
        x,
        rows,
        rank,
        scratch,
        collective_id,
        f"small_all_reduce_hier_2stage_r{rows}",
    )


def small_all_reduce(x,
                     *,
                     num_cores,
                     axis,
                     rank=None,
                     form=None,
                     collective_id=COLLECTIVE_ID):
    """The sum of ``x`` over the ``num_cores`` cores of mesh axis ``axis``,
    inside a shard_map over that axis.

    Args:
      x: [rows, hidden], this core's block; hidden a multiple of
        num_cores * LANES.
      rank: int32 [1], this core's index on the axis as data (the torch
        bridge), or None to read it from the axis.
      form: "oneshot", "oneshot_xor", "hier_halves", "twoshot", or
        "hier_2stage"; by default ``dispatch_form_for`` the block.
    """
    rows, hidden = x.shape
    if hidden % (num_cores * LANES):
        raise ValueError(
            f"hidden {hidden} is not a multiple of {num_cores} cores x {LANES} lanes"
        )
    form = form or dispatch_form_for(rows, hidden,
                                     jnp.dtype(x.dtype).itemsize, num_cores)
    if num_cores != 8:
        if form in ("hier_halves", "hier_2stage"):
            form = "twoshot"
        elif form == "oneshot_xor":
            form = "oneshot"
    build = {
        "oneshot": _oneshot,
        "oneshot_xor": _oneshot_xor,
        "hier_halves": _hier_halves_col_zcopy,
        "twoshot": _twoshot,
        "hier_2stage": _hier_2stage_rs_ag,
    }[form]
    return build(x, num_cores, axis, rank, collective_id)
