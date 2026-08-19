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
"""Pallas kernel body for TensorCore Reduce-Scatter.

`hier_rs_kernel` is the pallas_call body: it sequences phase 1 (intra-chip D2D)
and phase 2 (inter-chip hypercube) across micro-batches, driving the DmaManager.
`make_unified_scratch_shapes` declares the scratch/BufferedRef layout the body
unpacks positionally -- the two MUST be kept in lockstep.
"""

import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from tpu_inference.kernels.collectives.hierrs_tc.config import (
    SCALE_LANE, Config, next_multiple_of)
from tpu_inference.kernels.collectives.hierrs_tc.dma_pipeline import (
    DmaManager, RemoteWaitBufferedRef)
from tpu_inference.kernels.collectives.hierrs_tc.topology import (ChunkLocator,
                                                                  Topology)


def hier_rs_kernel(
    input_ref,
    output_ref,
    running_sum_ref,
    recv_buf_ref,
    *args,
    config: Config,
    axis_name: str = "x",
    fp8_comm: bool = False,
    fp8_static_scale: float | None = None,
):
    p2_recv_buf_ref = None
    if fp8_comm:
        (
            fp8_send_ref,
            fp8_recv_ref,
            scale_send_ref,
            scale_recv_ref,
            recv_bref,
            run_bref,
            out_bref,
            phase1_send_sems,
            phase1_recv_sems,
            phase2_send_sems,
            phase2_recv_sems,
            fp8_p2_send_sems,
            fp8_p2_recv_sems,
            scale_p2_send_sems,
            scale_p2_recv_sems,
            fp8_send_bref,
            scale_send_bref,
            fp8_recv_bref,
            scale_bref,
        ) = args
    else:
        # Phase 2's bf16 wire gets its OWN landing buffer, appended ahead of
        # the scratch shapes. Without it an incoming phase-2 chunk can
        # overwrite phase-1 bytes the receiver has not drained yet -- a
        # cross-device WAR that produced wrong answers in 157/200 runs at
        # 512 rows / mb=4. Not optional; the fp8 wire has always had its own
        # landing buffer, which is exactly why it never reproduced that bug.
        p2_recv_buf_ref, *args = args
        (
            recv_bref,
            run_bref,
            out_bref,
            phase1_send_sems,
            phase1_recv_sems,
            phase2_send_sems,
            phase2_recv_sems,
        ) = args
        fp8_send_ref = fp8_recv_ref = scale_send_ref = scale_recv_ref = None
        fp8_p2_send_sems = fp8_p2_recv_sems = None
        scale_p2_send_sems = scale_p2_recv_sems = None
        fp8_send_bref = scale_send_bref = None
        fp8_recv_bref = scale_bref = None

    # Phase-2's bf16 landing zone. This is a
    # dedicated buffer, so incoming phase-2 chunks cannot land on phase-1 bytes
    # the receiver has not consumed yet (the cross-device WAR that corrupted
    # bf16 at num_micro_batches=4). Otherwise it aliases recv_buf_ref, the old
    # racy behaviour, kept for reproducing the bug.
    p2_buf = p2_recv_buf_ref if p2_recv_buf_ref is not None else recv_buf_ref

    topo = Topology(axis_name)
    locator = ChunkLocator(config, topo)
    dma = DmaManager(
        config,
        topo,
        locator,
        recv_bref,
        run_bref,
        out_bref,
        phase1_send_sems,
        phase1_recv_sems,
        phase2_send_sems,
        phase2_recv_sems,
        fp8_send_buf=fp8_send_ref,
        fp8_recv_buf=fp8_recv_ref,
        scale_send_buf=scale_send_ref,
        scale_recv_buf=scale_recv_ref,
        fp8_p2_send_sems=fp8_p2_send_sems,
        fp8_p2_recv_sems=fp8_p2_recv_sems,
        scale_p2_send_sems=scale_p2_send_sems,
        scale_p2_recv_sems=scale_p2_recv_sems,
        fp8_send_bref=fp8_send_bref,
        scale_send_bref=scale_send_bref,
        fp8_recv_bref=fp8_recv_bref,
        scale_bref=scale_bref,
        fp8_static_scale=fp8_static_scale,
    )

    all_phase1_ops = []
    all_phase2_ops = []

    # =========================================================================================================
    #                                  HIERARCHICAL REDUCE-SCATTER TIMELINE (D2D + C2C Step 0)
    # =========================================================================================================
    #
    # Time -------->  t0                  t1                                      t2                                      t3
    #                 | Global Prologue   |              Loop m=0                 |              Loop m=1                 |
    #                 |                   |                                       |                                       |
    # D2D/DMA (P1)    [A]========[B]      |       [D]========[E]                  |       [D]========[E]                  |
    #                 |   P1 MB0          |       |   P1 MB1                      |       |   P1 MB2                      |
    #                 |                   |       |                               |       |                               |
    # C2C (P2)        |                   [C]=====================================[G]                                     |
    #                 |                   |             P2 MB0                    |                                       |
    #                 |                   |                                       [F]=====================================[G]
    #                 |                   |                                       |             P2 MB1                    |
    #                 |                   |                                       |                                       [F]========> (to t4)
    #                 |                   |                                       |                                       |  P2 MB2
    # Accumulate      |          [B]======|                  [E]======|           [G]======|                 [I]======|   [J]======|
    #                 |            AC P1  |                    AC P1  |           |  AC P2 |                   AC P1  |   |  AC P2 |
    #                 |            (MB0)  |                    (MB1)  |           |  (MB0) |                   (MB2)  |   |  (MB1) |
    # =========================================================================================================

    # =========== Global Prologue: PHASE 1 Micro-Batch 0 D2D REDUCTIONS ===========
    # [Step A]: Start remote D2D copies for micro-batch 0
    all_phase1_ops.extend(
        dma.start_phase1_d2d_copies(src=input_ref,
                                    dst=recv_buf_ref,
                                    mb_idx=0))

    # [Step B]: Wait for micro-batch 0 copies to finish, and accumulate locally
    dma.run_phase1_accumulate_pipeline(
        src1=recv_buf_ref,
        src2=input_ref,
        dst=running_sum_ref,
        in_index_fn=locator.make_phase1_index_fn(mb_idx=0),
        out_index_fn=locator.make_phase1_index_fn(mb_idx=0),
        hbm_index_fn=locator.make_phase1_in_index_fn_with_recv_sem(
            mb_idx=0),
        block_size=config.mb_size,
        mb_idx=0,
    )

    # [Step C]: Start Phase 2 Ring ICI copies for micro-batch 0
    if fp8_comm:
        dma.quantize_chunks_to_fp8_staging(running_sum_ref,
                                           mb_idx=0,
                                           step_idx=0)
        mb_ops_0 = dma.start_phase2_c2c_copies_fp8(mb_idx=0, step_idx=0)
        all_phase2_ops.extend([item[0] for item in mb_ops_0])
        all_phase2_ops.extend([i[1] for i in mb_ops_0 if i[1] is not None])
    else:
        mb_ops_0 = dma.start_phase2_c2c_copies(src=running_sum_ref,
                                               dst=p2_buf,
                                               mb_idx=0,
                                               step_idx=0)
        all_phase2_ops.extend([item[0] for item in mb_ops_0])

    def _start_phase2_step0(mb):
        """[Step F] body: start the phase-2 step-0 C2C copies for micro-batch mb."""
        if fp8_comm:
            dma.quantize_chunks_to_fp8_staging(running_sum_ref,
                                               mb,
                                               step_idx=0)
            ops = dma.start_phase2_c2c_copies_fp8(mb, step_idx=0)
            all_phase2_ops.extend([item[0] for item in ops])
            all_phase2_ops.extend([i[1] for i in ops if i[1] is not None])
        else:
            ops = dma.start_phase2_c2c_copies(src=running_sum_ref,
                                              dst=p2_buf,
                                              mb_idx=mb,
                                              step_idx=0)
            all_phase2_ops.extend([item[0] for item in ops])

    for m in range(config.num_micro_batches):

        if m < config.num_micro_batches - 1:
            # [Step D]: Start overlap Phase 1 D2D copies for next micro-batch
            all_phase1_ops.extend(
                dma.start_phase1_d2d_copies(src=input_ref,
                                            dst=recv_buf_ref,
                                            mb_idx=m + 1))

            # [Step E]: Wait and Accumulate Phase 1 for next micro-batch
            dma.run_phase1_accumulate_pipeline(
                src1=recv_buf_ref,
                src2=input_ref,
                dst=running_sum_ref,
                in_index_fn=locator.make_phase1_index_fn(m + 1),
                out_index_fn=locator.make_phase1_index_fn(m + 1),
                hbm_index_fn=locator.
                make_phase1_in_index_fn_with_recv_sem(m + 1),
                block_size=config.mb_size,
                mb_idx=m + 1,
            )

            # [Step F]: Pre-start next micro-batch Phase 2 Ring ICI copies
            #
            # RACE (suspected, cross-device): start_phase2_c2c_copies issues remote
            # copies into the NEIGHBOR's recv_buf_ref -- the same buffer [Step D]
            # fills with phase-1 data and [Step E] consumes. Nothing tells the
            # neighbor that it has finished draining phase 1 for this micro-batch,
            # so a device that runs ahead can land phase-2 bytes on top of its
            # neighbor's unread phase-1 bytes.
            #
            # If that is the mechanism, local reordering CANNOT fix it -- it only
            # moves the window. RS_STEPF_DELAY=1 defers this start until after
            # [Step G] purely as a diagnostic: if deferring "fixes" mb=4, the fix is
            # timing, not ordering, and the race is still latent.
            _start_phase2_step0(m + 1)

        # [Phase 2, Step 1] Pre-start Step 1 MB0 during the last iteration of Step 0
        #
        # RACE: this reads running_sum_ref for micro-batch 0, which was produced
        # by [Step G] of an earlier iteration of this same loop. That producer is
        # an emit_pipeline whose output lands in HBM asynchronously, so the read
        # here is not ordered against it. Symptom: non-deterministic corruption of
        # the last micro-batch's tile -- identical input, identical code,
        # different answers run to run. Affects the bf16 and fp8 wires both;
        # num_micro_batches == 1 is always correct because that case does the
        # start AFTER the loop (see below), which is exactly the ordering the
        # multi-micro-batch path is missing.
        #
        # RS_UNSAFE_STEP1_PRESTART=1 restores the old in-loop start to reproduce
        # the bug; default is the safe ordering.
        # [Step G]: Wait and Accumulate Phase 2 Step 0 for current micro-batch
        if config.num_hcube_dims >= 1:
            is_last = config.num_hcube_dims == 1
            if fp8_comm:
                dma.run_phase2_dequant_accumulate_pipeline(
                    running_sum_ref,
                    output_ref,
                    m,
                    step_idx=0,
                    is_last_step=is_last)
            else:
                dma.run_phase2_accumulate_pipeline(
                    src1=p2_buf,
                    src2=running_sum_ref,
                    dst=output_ref if is_last else running_sum_ref,
                    in_index_fn=locator.make_phase2_index_fn(0, m),
                    out_index_fn=locator.make_phase2_out_index_fn(0, m)
                    if is_last else locator.make_phase2_index_fn(0, m),
                    hbm_index_fn=locator.
                    make_phase2_in_index_fn_with_recv_sem(0, m),
                    block_size=config.hc_chunk_size,
                    mb_idx=m,
                    step_idx=0,
                )

        # [Step F], deferred variant -- diagnostic only (see the RACE note above).
    # Start Step 1 MB0 here, after the Step-0 loop has fully drained, so the read
    # of running_sum_ref is ordered against the [Step G] pipeline that wrote it.
    # This was previously done only for num_micro_batches == 1 ("we couldn't
    # pre-start Step 1 MB0 in the loop due to data dependency") -- the same data
    # dependency exists for every micro-batch count; it was simply hidden by
    # timing when other micro-batches gave the write time to land.
    if fp8_comm:
        dma.quantize_chunks_to_fp8_staging(running_sum_ref, 0, step_idx=1)
        mb_ops = dma.start_phase2_c2c_copies_fp8(0, step_idx=1)
        all_phase2_ops.extend([item[0] for item in mb_ops])
        all_phase2_ops.extend([i[1] for i in mb_ops if i[1] is not None])
    else:
        mb_ops = dma.start_phase2_c2c_copies(src=running_sum_ref,
                                             dst=p2_buf,
                                             mb_idx=0,
                                             step_idx=1)
        all_phase2_ops.extend([item[0] for item in mb_ops])

    # ================= STEP 1 LOOP =================
    for m in range(config.num_micro_batches):

        if m < config.num_micro_batches - 1:
            if fp8_comm:
                dma.quantize_chunks_to_fp8_staging(running_sum_ref,
                                                   m + 1,
                                                   step_idx=1)
                mb_ops = dma.start_phase2_c2c_copies_fp8(m + 1,
                                                         step_idx=1)
                all_phase2_ops.extend([item[0] for item in mb_ops])
                all_phase2_ops.extend(
                    [i[1] for i in mb_ops if i[1] is not None])
            else:
                mb_ops = dma.start_phase2_c2c_copies(
                    src=running_sum_ref,
                    dst=p2_buf,
                    mb_idx=m + 1,
                    step_idx=1)
                all_phase2_ops.extend([item[0] for item in mb_ops])

        # Accumulate Step 1
        if config.num_hcube_dims > 1:
            if fp8_comm:
                dma.run_phase2_dequant_accumulate_pipeline(
                    running_sum_ref,
                    output_ref,
                    m,
                    step_idx=1,
                    is_last_step=True)
            else:
                dma.run_phase2_accumulate_pipeline(
                    src1=p2_buf,
                    src2=running_sum_ref,
                    dst=output_ref,
                    in_index_fn=locator.make_phase2_index_fn(1, m),
                    out_index_fn=locator.make_phase2_out_index_fn(1, m),
                    hbm_index_fn=locator.
                    make_phase2_in_index_fn_with_recv_sem(1, m),
                    block_size=config.hc_chunk_size,
                    mb_idx=m,
                    step_idx=1,
                )

    for op in all_phase1_ops:
        op.wait_send()
    for op in all_phase2_ops:
        op.wait_send()


def make_unified_scratch_shapes(
    seq_chunk_size,
    mb_size,
    dtype,
    num_chips,
    num_hcube_dims,
    num_micro_batches,
    fp8_comm: bool = False,
):
    block_spec = pl.BlockSpec(block_shape=(seq_chunk_size, mb_size),
                              index_map=lambda *args: (0, 0))
    recv_bref = pltpu.BufferedRef.input(block_spec, dtype, buffer_count=2)
    run_bref = pltpu.BufferedRef.input(block_spec, dtype, buffer_count=2)
    out_bref = pltpu.BufferedRef.output(block_spec, dtype, buffer_count=2)

    p2_sem_shape = (
        num_hcube_dims,
        num_micro_batches,
        num_hcube_dims,
        1 << max(0, num_hcube_dims - 1),
    )

    scratch_shapes = [
        recv_bref,
        run_bref,
        out_bref,
        pltpu.SemaphoreType.DMA((num_chips, num_micro_batches)),  # p1 send
        pltpu.SemaphoreType.DMA((num_chips, num_micro_batches)),  # p1 recv
        pltpu.SemaphoreType.DMA(p2_sem_shape),  # p2 send
        pltpu.SemaphoreType.DMA(p2_sem_shape),  # p2 recv
    ]

    if fp8_comm:
        fp8_block_spec = pl.BlockSpec(block_shape=(seq_chunk_size, mb_size),
                                      index_map=lambda *a: (0, 0))
        fp8_recv_bref = pltpu.BufferedRef.input(fp8_block_spec,
                                                jnp.float8_e4m3fn,
                                                buffer_count=2)
        # Send-side staging is now pipelined: emit_pipeline double-buffers these
        # OUTPUT BufferedRefs (BF16->FP8 quantize) instead of the old serial
        # load/quantize/store VMEM scratch + DMA-sem chain. The BF16 input reuses
        # run_bref, so net VMEM is slightly lower than the serial scratch.
        fp8_send_bref = pltpu.BufferedRef.output(fp8_block_spec,
                                                 jnp.float8_e4m3fn,
                                                 buffer_count=2)
        scale_block_spec = pl.BlockSpec(block_shape=(1, SCALE_LANE),
                                        index_map=lambda *a: (0, 0))
        scale_bref = pltpu.BufferedRef.input(scale_block_spec,
                                             jnp.float32,
                                             buffer_count=2)
        scale_send_bref = pltpu.BufferedRef.output(scale_block_spec,
                                                   jnp.float32,
                                                   buffer_count=2)
        scratch_shapes += [
            pltpu.SemaphoreType.DMA(p2_sem_shape),  # fp8_p2_send
            pltpu.SemaphoreType.DMA(p2_sem_shape),  # fp8_p2_recv
            pltpu.SemaphoreType.DMA(p2_sem_shape),  # scale_p2_send
            pltpu.SemaphoreType.DMA(p2_sem_shape),  # scale_p2_recv
            fp8_send_bref,  # fp8_send_bref (send-side pipeline output)
            scale_send_bref,  # scale_send_bref (send-side pipeline output)
            fp8_recv_bref,  # fp8_recv_bref
            scale_bref,  # scale_bref
        ]

        # Appended LAST so the existing positional unpacking in hier_rs_kernel is
        # undisturbed; the kernel unpacks these two off the tail.
    return scratch_shapes
