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
"""Pallas kernels for Reduce-Scatter."""

import jax
from jax.experimental import pallas as pl

from tpu_inference.kernels.collectives.hierrs_sc.config import Config
from tpu_inference.kernels.collectives.hierrs_sc.dma_pipeline import (
    LocalDmaManager, RemoteDmaManager)
from tpu_inference.kernels.collectives.hierrs_sc.topology import Topology

# ==============================================================================
#                 HIERARCHICAL REDUCE-SCATTER TIMELINE (D2D + C2C Step 0)
# ==============================================================================
# Hardware Execution Mapping:
#   - SCS (SparseCore Sequencer): Controls all asynchronous RDMA transfers
#     (D2D/C2C).
#   - TEC (Tile Core): Conducts all mathematical operations (Accumulation).
#
# Time -> t0            t1                      t2                      t3
#         | Prologue    |  Loop m=0             |  Loop m=1             |
#         |             |                       |                       |
# [SCS]   |             |                       |                       |
# D2D/DMA [A]====[B]    |  [D]====[E]           |  [D]====[E]           |
# (P1)    | P1 MB0      |  | P1 MB1             |  | P1 MB2             |
#         |             |  |                    |  |                    |
# [SCS]   |             |                       |                       |
# C2C     |             [C]=====================[G]                     |
# (P2)    |             |       P2 MB0          |                       |
#         |             |                       [F]=====================[G]
#         |             |                       |       P2 MB1          |
#         |             |                       |                       [F]====>
#         |             |                       |                       | P2 MB2
# [TEC]   |             |                       |                       |
# Accum   |      [B]====|          [E]====|     [G]====|         [I]====|
#         |        AC P1|            AC P1|     | AC P2|           AC P1|
#         |        (MB0)|            (MB1)|     | (MB0)|           (MB2)|
# ==============================================================================


def scs_kernel(
    # Inputs
    x_ref: jax.Ref,
    # Outputs.
    _: jax.Ref,  # output_ref, unused for SCS.
    running_sum_ref: jax.Ref,
    recv_buf_ref: jax.Ref,
    *,
    config: Config,
    axis_name: str | tuple[str, ...],
    # Scratch
    scs_to_tec: jax.Ref,
    tec_to_scs: jax.Ref,
    p1_recv_sem: jax.Ref,
    p2_recv_sem: jax.Ref,
    p1_send_sem: jax.Ref,
    p2_send_sem: jax.Ref,
    **unused_scratch,
):
    """Executes SparseCore Sequencer (SCS) execution logic for Reduce-Scatter.

  SCS is in charge of handling D2D and C2C ICI operations.
  """
    core_idx = jax.lax.axis_index("core")
    topo = Topology(axis_name)
    dma_manager = RemoteDmaManager(
        config,
        topo,
        core_idx,
        p1_send_sem=p1_send_sem,
        p2_send_sem=p2_send_sem,
        p1_recv_sem=p1_recv_sem,
        p2_recv_sem=p2_recv_sem,
    )

    def _signal_and_wait_tec(mb_idx, step):
        slot = mb_idx % 2
        for s in range(config.num_subcores):
            pl.semaphore_signal(scs_to_tec.at[slot, step],
                                device_id={"subcore": s})
        pl.semaphore_wait(tec_to_scs.at[slot, step], value=config.num_subcores)

    ############################################################################
    #                             PROLOGUE                                     #
    ############################################################################

    # [Step A - P1 MB0]: Start computing the initial prologue for pipeline by
    # firing D2D transfer for the very first micro-batch (MB), copying the first
    # `mb_size` block (MB 0) from local HBM over to the partner chiplet.
    dma_manager.start_phase1_d2d_copies(
        mb_idx=0,
        src=x_ref,
        dst=recv_buf_ref.at[0],
    )
    # [Step B - P1 MB0 Done]: Wait/block until the D2D transfer for MB 0
    # successfully done and the data is available in `recv_buf_ref[0]`, ensuring
    # the TEC has full data to begin accumulate phase 1.
    dma_manager.wait_phase1_d2d_copies(mb_idx=0,
                                       src=x_ref,
                                       dst=recv_buf_ref.at[0])

    # [Accumulate P1 MB0]: Signal the TEC to start local accumulation for MB 0
    # using the D2D data just received, and block SCS execution until TEC finishes
    # updating `running_sum_ref`.
    _signal_and_wait_tec(mb_idx=0, step=0)

    # [Step C - P2-S0 MB0]: Kick off Phase 2's first step ICI transfers. Copies
    # the partially reduced `running_sum_ref` out to the neighor chip, storing the
    # data into `recv_buf_ref[1]`.
    dma_manager.start_phase2_c2c_copies(
        mb_idx=0,
        step_idx=0,
        src=running_sum_ref,
        dst=recv_buf_ref.at[1],
    )

    ############################################################################
    #                  MAIN PIPELINE LOOP (P1 + P2 Step 0)                     #
    ############################################################################
    @pl.loop(0, config.num_micro_batches - 1)
    def step0_loop(mb_idx):
        # [Step D - P1 MB i+1]: While SEC+TEC processes the CURRENT micro-batch,
        # let's say MB i, it asynchronously triggers D2D transfers for the NEXT
        # micro-batch, (MB i+1), perfectly overlapping D2D with C2C and
        # accumulation.
        dma_manager.start_phase1_d2d_copies(
            mb_idx=mb_idx + 1,
            src=x_ref,
            dst=recv_buf_ref.at[0],
        )
        # [Step E - P1 MB i+1 Done]: Block until the D2D transfer for MB i+1 is
        # complete. This is safe because this isn't on the critical path of the
        # pipeline.
        dma_manager.wait_phase1_d2d_copies(mb_idx=mb_idx + 1,
                                           src=x_ref,
                                           dst=recv_buf_ref.at[0])

        # [Accumulate P1 MB i+1]: Signal the TEC to start accumulating MB i+1, and
        # wait here until the subcores finish accumulation and storing the reduced
        # chunk into `running_sum_ref`.
        _signal_and_wait_tec(mb_idx=mb_idx + 1, step=0)

        # [Step F - P2-S0 MB i+1]: Immediately queue Phase 2 Step 0 ICI transfers
        # for the NEXT micro-batch (MB i+1).
        dma_manager.start_phase2_c2c_copies(
            mb_idx=mb_idx + 1,
            step_idx=0,
            src=running_sum_ref,
            dst=recv_buf_ref.at[1],
        )

        # [Step G - P2-S0 MB i Done]: Block until the Phase 2 Step 0 ICI transfers
        # for the CURRENT micro-batch (MB i) is complete. We expect the newly
        # reduced chunk to arriving into `recv_buf_ref[1]`, then signal and wait for
        # the TEC to accumulate those chunks.
        dma_manager.wait_phase2_c2c_copies(mb_idx=mb_idx,
                                           step_idx=0,
                                           src=running_sum_ref,
                                           dst=recv_buf_ref.at[1])
        _signal_and_wait_tec(mb_idx=mb_idx, step=1)

    ############################################################################
    #                 EPILOGUE AND PRE-START P2-S1                             #
    ############################################################################
    last_mb_idx = config.num_micro_batches - 1
    # [P2-S1 MB 0]: Pre-start Phase 2 Step 1 Ring ICI transfers for MB 0,
    # overlapping with final MB execution for better pipeline saturation.
    # This will be done later if there's only 1 MB due to data depedency,
    # specifically accumulation for P2-S0 MB0 is not yet done.
    if config.num_micro_batches > 1:
        dma_manager.start_phase2_c2c_copies(
            mb_idx=0,
            step_idx=1,
            src=running_sum_ref,
            dst=recv_buf_ref.at[2],
        )
    # [Step G - P2-S0 MB Last Done]: Wait/block until the Phase 2 Step 0 hypercube
    # transfers finish arriving for the last micro-batch.
    dma_manager.wait_phase2_c2c_copies(
        mb_idx=last_mb_idx,
        step_idx=0,
        src=running_sum_ref,
        dst=recv_buf_ref.at[1],
    )
    # [Accumulate P2-S0 MB Last]: Signal the TEC to start accumulating the
    # last MB, and wait here until subcores finish accumulation.
    _signal_and_wait_tec(mb_idx=last_mb_idx, step=1)

    # [P2-S1 MB 0]: When there's only 1 micro-batch, start ICI (Phase 2) for step
    # 1 once accumulation for Phase 2 Step 0 is done.
    if config.num_micro_batches == 1:
        dma_manager.start_phase2_c2c_copies(
            mb_idx=0,
            step_idx=1,
            dst=recv_buf_ref.at[2],
            src=running_sum_ref,
        )

    ############################################################################
    #                           PHASE 2 STEP 1+ LOOP                           #
    ############################################################################
    for step_idx in range(1, config.num_hcube_dims):
        #
        # PIPELINE ICI and ACCUMULATION (P2 Step i)
        #
        @pl.loop(0, config.num_micro_batches - 1)
        def step_loop(mb_idx):
            # [P2-S_step MB i Done]: Block until ICI transfers finish for the CURRENT
            # MB (MB i)
            dma_manager.wait_phase2_c2c_copies(
                mb_idx=mb_idx,
                step_idx=step_idx,
                src=running_sum_ref,
                dst=recv_buf_ref.at[step_idx + 1],
            )
            # [P2-S_step MB i+1]: Immediately queue Phase 2 for ICI transfers
            # for the NEXT micro-batch across the ICI ring to minimize latency
            # stalling.
            dma_manager.start_phase2_c2c_copies(
                mb_idx=mb_idx + 1,
                step_idx=step_idx,
                src=running_sum_ref,
                dst=recv_buf_ref.at[step_idx + 1],
            )
            # [Accumulate P2-S_step MB i]: Signal the TEC to accumulate the newly
            # arrived chunk, and wait here until the subcores finish accumulation.
            _signal_and_wait_tec(mb_idx, step_idx + 1)

        #
        # EPILOGUE AND PRE-START P2 STEP i+1
        #
        last_mb_idx = config.num_micro_batches - 1
        # [P2-S_step MB Last Done]: Block until the ICI transfers finish for the
        # last MB and the reduced chunk is available.
        dma_manager.wait_phase2_c2c_copies(
            mb_idx=last_mb_idx,
            step_idx=step_idx,
            src=running_sum_ref,
            dst=recv_buf_ref.at[step_idx + 1],
        )

        # [P2-S_step+1 MB 0]: Pre-start Phase 2 Step i+1 ICI for MB 0, overlapping
        # with final MB accumulation for maximize ICI bandwidth saturation.
        # This will be done later if there's only 1 MB due to data depedency,
        # specifically accumulation for P2-S_i MB0 is not yet done.
        if config.num_micro_batches > 1 and step_idx < config.num_hcube_dims - 1:
            dma_manager.start_phase2_c2c_copies(
                mb_idx=0,
                step_idx=step_idx + 1,
                src=running_sum_ref,
                dst=recv_buf_ref.at[step_idx + 2],
            )
        # [Accumulate P2-S_step MB Last]: Signal the TEC to accumulate the received
        # chunk, and wait here until the subcores finish acumulation.
        _signal_and_wait_tec(last_mb_idx, step_idx + 1)

        # [P2-S_step+1 MB 0]: When there's only 1 micro-batch, start ICI for the
        # NEXT step once accumulation for the current step's MB0 is done.
        if config.num_micro_batches == 1 and step_idx < config.num_hcube_dims - 1:
            dma_manager.start_phase2_c2c_copies(
                mb_idx=0,
                step_idx=step_idx + 1,
                src=running_sum_ref,
                dst=recv_buf_ref.at[step_idx + 2],
            )

    ############################################################################
    #                 CLEAN UP UN-WAITED SEND SEMAPHORES                       #
    ############################################################################
    # Resolve un-waited send semaphores
    @pl.loop(0, config.num_micro_batches)
    def wait_p1_sends_loop(mb_idx):
        dma_manager.wait_phase1_d2d_copies(mb_idx=mb_idx,
                                           src=x_ref,
                                           dst=recv_buf_ref.at[0],
                                           wait_send=True)

    @pl.loop(0, config.num_hcube_dims)
    def wait_p2_sends_step_loop(step_idx):

        @pl.loop(0, config.num_micro_batches)
        def wait_p2_sends_mb_loop(mb_idx):
            dma_manager.wait_phase2_c2c_copies(
                mb_idx=mb_idx,
                step_idx=step_idx,
                src=running_sum_ref,
                dst=recv_buf_ref.at[step_idx + 1],
                wait_send=True,
            )


def tec_kernel(
    # Inputs
    x_ref: jax.Ref,
    # Outputs
    output_ref: jax.Ref,
    running_sum_ref: jax.Ref,
    recv_buf_ref: jax.Ref,
    *,
    config: Config,
    axis_name: str | tuple[str, ...],
    # Scratch
    scs_to_tec: jax.Ref,
    tec_to_scs: jax.Ref,
    **unused_scratch,
):
    """Kernel execution impl that runs on the Tile Core (TEC).

  TEC is responsible for handling the computation logic needed for reduce
  scatter, which is basically the local accumulation of the incoming chunks of
  data (running sum) and corresponding chunks.
  """

    num_hcube_dims = config.num_hcube_dims
    core_idx = jax.lax.axis_index("core")
    subcore_idx = jax.lax.axis_index("subcore")

    topo = Topology(axis_name)
    dma_manager = LocalDmaManager(
        config,
        topo,
        core_idx,
        subcore_idx,
    )

    ############################################################################
    #                             PROLOGUE                                     #
    ############################################################################
    # [Step B - P1 MB0 Done]: Wait/block until SCS confirms the very first D2D
    # transfer (MB 0) is done and the data is available in `recv_buf_ref[0]`.
    pl.semaphore_wait(scs_to_tec.at[0, 0], value=1)
    # [Accumulate P1 MB0]: Conduct local accumulation for MB 0, which is
    # equivalent to (running_sum_ref = x_ref + recv_buf_ref). And signal SCS that
    # accumulation is done.
    dma_manager.run_phase1_accumulate_pipeline(
        mb_idx=0,
        src1_ref=x_ref,
        src2_ref=recv_buf_ref.at[0],
        out_ref=running_sum_ref,
    )
    pl.semaphore_signal(tec_to_scs.at[0, 0])

    ############################################################################
    #                  MAIN PIPELINE LOOP (P1 + P2 Step 0)                     #
    ############################################################################
    @pl.loop(0, config.num_micro_batches - 1)
    def step0_loop(mb_idx):
        curr_slot = mb_idx % 2
        next_slot = 1 - curr_slot

        # [Step E - P1 MB i+1 Done]: Block until SCS confirms the overlapped D2D
        # payload (MB i+1) has safely arrived and is available in `recv_buf_ref[0]`.
        pl.semaphore_wait(scs_to_tec.at[next_slot, 0], value=1)
        # [Step E - Accumulate P1 MB i+1]: Accumulate the recevied Phase 1 chunks
        # for MB i+1, which is equivalent to (running_sum_ref = x_ref +
        # recv_buf_ref). And signal SCS that accumulation is done.
        dma_manager.run_phase1_accumulate_pipeline(
            mb_idx=mb_idx + 1,
            src1_ref=x_ref,
            src2_ref=recv_buf_ref.at[0],
            out_ref=running_sum_ref,
        )
        pl.semaphore_signal(tec_to_scs.at[next_slot, 0])

        # [Step G - P2-S0 MB i Done]: Block until the Phase 2 Step 0 ICI payload has
        # fully arrived for the CURRENT micro-batch (MB i).
        pl.semaphore_wait(scs_to_tec.at[curr_slot, 1], value=1)
        # [Step G - Accumulate P2-S0 MB i]: Accumulate the received Phase 2 Step 0
        # chunks for MB i, which is equivalent to (running_sum_ref += recv_buf_ref).
        # And signal SCS that accumulation is done.
        dma_manager.run_phase2_accumulate_pipeline(
            mb_idx=mb_idx,
            step_idx=0,
            src1_ref=running_sum_ref,
            src2_ref=recv_buf_ref,
            final_out_ref=output_ref,
        )
        pl.semaphore_signal(tec_to_scs.at[curr_slot, 1])

    last_mb_idx = config.num_micro_batches - 1
    curr_slot = last_mb_idx % 2
    # [Step G - P2-S0 MB Last Done]: Block until the Phase 2 Step 0 ICI network
    # payload has fully arrived for the final micro-batch.
    pl.semaphore_wait(scs_to_tec.at[curr_slot, 1], value=1)
    # [Accumulate P2-S0 MB Last]: Accumulate the received Phase 2 Step 0 chunks
    # for MB Last, which is equivalent to (running_sum_ref += recv_buf_ref). And
    # signal SCS that accumulation is done.
    dma_manager.run_phase2_accumulate_pipeline(
        mb_idx=last_mb_idx,
        step_idx=0,
        src1_ref=running_sum_ref,
        src2_ref=recv_buf_ref,
        final_out_ref=output_ref,
    )
    pl.semaphore_signal(tec_to_scs.at[curr_slot, 1])

    ############################################################################
    #                     PHASE 2 STEP 1+ ACCUMULATION LOOP                    #
    ############################################################################
    def do_phase2_step(mb_idx, step_idx):
        curr_slot = mb_idx % 2
        # [P2-S_step MB i Done]: Block until Phase 2 Step `step_idx` ICI payload has
        # safely arrived for the CURRENT micro-batch (MB i).
        pl.semaphore_wait(scs_to_tec.at[curr_slot, 1 + step_idx], value=1)
        # [Accumulate P2-S_step MB i]: Accumulate the received Phase 2 Step
        # `step_idx` chunks for MB i, which is equivalent to (running_sum_ref +=
        # recv_buf_ref). And signal SCS that accumulation is done.
        dma_manager.run_phase2_accumulate_pipeline(
            mb_idx=mb_idx,
            step_idx=step_idx,
            src1_ref=running_sum_ref,
            src2_ref=recv_buf_ref,
            final_out_ref=output_ref,
        )
        pl.semaphore_signal(tec_to_scs.at[curr_slot, 1 + step_idx])

    for step_idx in range(1, num_hcube_dims):

        @pl.loop(0, config.num_micro_batches)
        def step_loop(mb_idx):
            do_phase2_step(mb_idx, step_idx)
