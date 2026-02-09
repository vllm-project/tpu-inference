# SPDX-License-Identifier: Apache-2.0
"""Bidirectional Reduce-Scatter Matmul with M-Split Algorithm.

This implementation uses BOTH left and right neighbors simultaneously to double
the effective communication bandwidth. The key insight is to split the M dimension
into N blocks (one per device), and each block into TOP and BOT halves:
  - LEFT direction handles all TOP halves
  - RIGHT direction handles all BOT halves

This avoids the collision problem where both directions would compute for the
same shard at the midpoint step.

Setup (8 devices example):
- Each device has: x[M, K_shard], y[N, K_shard] where K is sharded
- M is split into 8 blocks, each block split into TOP and BOT halves
- Output: each device gets its M_block (TOP + BOT) with full K reduction
- D0 owns Block 0 (rows 0 to M/8-1)
- D1 owns Block 1 (rows M/8 to 2M/8-1)
- etc.

Bidirectional ring:
- LEFT direction: D0 → D7 → D6 → D5 → D4 → D3 → D2 → D1 → D0 (send to left)
- RIGHT direction: D0 → D1 → D2 → D3 → D4 → D5 → D6 → D7 → D0 (send to right)

M-SPLIT BIDIRECTIONAL ALGORITHM:
================================

Split M into N blocks, each block into TOP (first half) and BOT (second half):
  - LEFT handles: B0_TOP, B1_TOP, B2_TOP, ..., B(N-1)_TOP
  - RIGHT handles: B0_BOT, B1_BOT, B2_BOT, ..., B(N-1)_BOT

Each direction does COMPLETE reduce-scatter (N-1 steps) for its halves.

From Device 0's perspective (8 devices):

  Step 0 (Prologue):
    Barrier with both neighbors
    LEFT:  Compute P₀(B1_TOP) for block 1's top half
    RIGHT: Compute P₀(B7_BOT) for block 7's bot half
    Send both to neighbors

  Steps 1 to N-2: For each step s:
    Signal neighbors, wait for capacity
    Start bidirectional DMA:
      - LEFT: send to D7, receive from D1
      - RIGHT: send to D1, receive from D7
    DELAYED COMPUTATION (overlapped with DMA):
      - LEFT:  Compute P₀(B(s+1)_TOP) for target block's top half
      - RIGHT: Compute P₀(B(7-s)_BOT) for target block's bot half
    Wait for DMAs
    Accumulate computation results to received data

  Final Step:
    Final DMA exchange
    Compute own block contributions:
      - LEFT:  Compute P₀(B0_TOP) for own block's top half
      - RIGHT: Compute P₀(B0_BOT) for own block's bot half
    Accumulate to received data
    Write output:
      - TOP half from LEFT direction
      - BOT half from RIGHT direction

KEY INSIGHTS:
1. NO COLLISION: LEFT always computes TOP halves, RIGHT always computes BOT halves
2. Even at midpoint (step 3 for 8 devices), they compute DIFFERENT halves of same block
3. PERFECTLY BALANCED: Every step has exactly 2 half-block matmuls
4. NO IDLE STEPS: Both directions always have compute work
5. 2X BANDWIDTH: Both ICI directions fully utilized
6. GOOD OVERLAP: Compute overlaps with bidirectional DMA
"""

import functools
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

P = jax.sharding.PartitionSpec
Ref = Any


def mod(x: jax.Array, n: int) -> jax.Array:
    """Modulo operation that works with JAX arrays."""
    return lax.rem(x + n, n)


class KernelConfig(NamedTuple):
    """Configuration for the kernel."""

    num_devices: int
    m_block: int  # M rows per device = M // num_devices
    m_half_block: int  # Half of m_block = m_block // 2
    # Tile sizes for matmul
    bm: int = 128  # Block size for M dimension
    bn: int = 128  # Block size for N dimension
    bk: int = 128  # Block size for K dimension


def tiled_matmul_hbm(
    x_hbm_ref: Ref,  # [M, K_shard] in HBM (full buffer)
    y_hbm_ref: Ref,  # [N, K_shard] in HBM
    out_hbm_ref: Ref,  # [M_half_block, N] in HBM
    # VMEM scratch buffers
    x_vmem_ref: Ref,  # [bm, bk] in VMEM
    y_vmem_ref: Ref,  # [bn, bk] in VMEM
    acc_vmem_ref: Ref,  # [bm, bn] in VMEM (float32 for accumulation)
    out_vmem_ref: Ref,  # [bm, bn] in VMEM
    # Semaphores
    copy_sem: Ref,  # DMA semaphore
    *,
    m_block_idx: int
    | jax.Array,  # Which M block (multiply by bm to get offset)
    m_size: int,  # Number of M rows to process
    bm: int,
    bn: int,
    bk: int,
):
    """Tiled matmul: out = x[m_block_idx*bm:m_block_idx*bm+m_size, :] @ y.T.

    This function tiles the matmul computation using async_copy for HBM<->VMEM transfers.
    The result OVERWRITES the output buffer (does not accumulate to existing values).

    IMPORTANT: Uses block indices (multiplied by bm) instead of raw offsets to help
    the Mosaic compiler prove tile alignment at compile time.

    Args:
        x_hbm_ref: Full X buffer in HBM [M, K_shard]
        y_hbm_ref: Y buffer in HBM [N, K_shard]
        out_hbm_ref: Output buffer in HBM [m_size, N]
        x_vmem_ref: VMEM scratch for x tile [bm, bk]
        y_vmem_ref: VMEM scratch for y tile [bn, bk]
        acc_vmem_ref: VMEM scratch for accumulator [bm, bn] in float32
        out_vmem_ref: VMEM scratch for output tile [bm, bn]
        copy_sem: DMA semaphore for async copies
        m_block_idx: Starting block index (offset = m_block_idx * bm)
        m_size: Number of rows to process
        bm: Block size for M dimension
        bn: Block size for N dimension
        bk: Block size for K dimension
    """
    _, k_shard = x_hbm_ref.shape
    n_total, _ = y_hbm_ref.shape

    num_m_tiles = m_size // bm
    num_n_tiles = n_total // bn
    num_k_tiles = k_shard // bk

    for m_tile in range(num_m_tiles):
        # Compute global m index using block index arithmetic
        # This form helps the compiler prove alignment since it's (block_idx + tile) * bm
        global_m_tile = m_block_idx + m_tile
        for n_tile in range(num_n_tiles):
            n_start = n_tile * bn

            # Zero the accumulator
            acc_vmem_ref[...] = jnp.zeros((bm, bn), dtype=jnp.float32)

            # Iterate over K tiles
            for k_tile in range(num_k_tiles):
                k_start = k_tile * bk

                # Copy x tile from HBM to VMEM
                # Use global_m_tile * bm for the offset (compiler can prove alignment)
                x_copy = pltpu.make_async_copy(
                    src_ref=x_hbm_ref.at[pl.ds(global_m_tile * bm, bm),
                                         pl.ds(k_start, bk)],
                    dst_ref=x_vmem_ref,
                    sem=copy_sem,
                )
                x_copy.start()
                x_copy.wait()

                # Copy y tile from HBM to VMEM
                y_copy = pltpu.make_async_copy(
                    src_ref=y_hbm_ref.at[pl.ds(n_start, bn),
                                         pl.ds(k_start, bk)],
                    dst_ref=y_vmem_ref,
                    sem=copy_sem,
                )
                y_copy.start()
                y_copy.wait()

                # Compute partial result and accumulate
                x_f32 = x_vmem_ref[...].astype(jnp.float32)
                y_f32 = y_vmem_ref[...].astype(jnp.float32)
                acc_vmem_ref[...] = acc_vmem_ref[...] + jnp.dot(x_f32, y_f32.T)

            # Convert accumulator to output dtype and store
            out_vmem_ref[...] = acc_vmem_ref[...].astype(out_hbm_ref.dtype)

            # Copy result from VMEM to HBM
            # Use m_tile * bm for output offset (within the output buffer)
            out_copy = pltpu.make_async_copy(
                src_ref=out_vmem_ref,
                dst_ref=out_hbm_ref.at[pl.ds(m_tile * bm, bm),
                                       pl.ds(n_start, bn)],
                sem=copy_sem,
            )
            out_copy.start()
            out_copy.wait()


def tiled_add_hbm(
    src_hbm_ref: Ref,  # [M_half_block, N] in HBM
    dst_hbm_ref: Ref,  # [M_half_block, N] in HBM (will be modified in place)
    # VMEM scratch buffers
    src_vmem_ref: Ref,  # [bm, bn] in VMEM
    dst_vmem_ref: Ref,  # [bm, bn] in VMEM
    # Semaphores
    copy_sem: Ref,  # DMA semaphore
    *,
    bm: int,
    bn: int,
):
    """Tiled addition: dst += src, with HBM inputs.

    Adds the source buffer to the destination buffer in-place using async_copy.
    """
    m_size, n_total = src_hbm_ref.shape

    num_m_tiles = m_size // bm
    num_n_tiles = n_total // bn

    for m_tile in range(num_m_tiles):
        m_start = m_tile * bm
        for n_tile in range(num_n_tiles):
            n_start = n_tile * bn

            # Copy src tile from HBM to VMEM
            src_copy = pltpu.make_async_copy(
                src_ref=src_hbm_ref.at[pl.ds(m_start, bm),
                                       pl.ds(n_start, bn)],
                dst_ref=src_vmem_ref,
                sem=copy_sem,
            )
            src_copy.start()
            src_copy.wait()

            # Copy dst tile from HBM to VMEM
            dst_copy = pltpu.make_async_copy(
                src_ref=dst_hbm_ref.at[pl.ds(m_start, bm),
                                       pl.ds(n_start, bn)],
                dst_ref=dst_vmem_ref,
                sem=copy_sem,
            )
            dst_copy.start()
            dst_copy.wait()

            # Add and store back to dst_vmem
            result = src_vmem_ref[...].astype(
                jnp.float32) + dst_vmem_ref[...].astype(jnp.float32)
            dst_vmem_ref[...] = result.astype(dst_hbm_ref.dtype)

            # Copy result from VMEM to HBM
            out_copy = pltpu.make_async_copy(
                src_ref=dst_vmem_ref,
                dst_ref=dst_hbm_ref.at[pl.ds(m_start, bm),
                                       pl.ds(n_start, bn)],
                sem=copy_sem,
            )
            out_copy.start()
            out_copy.wait()


def _kernel(
    # Inputs (in HBM)
    x_ref: Ref,  # [M, K_shard]
    y_ref: Ref,  # [N, K_shard]
    # Output (in HBM)
    out_ref: Ref,  # [M_block, N] - this device's output block
    # Scratch space for communication (in HBM)
    # 4 slots: [0,1] for LEFT direction, [2,3] for RIGHT direction
    scratch_ref: Ref,  # [4, M_half_block, N]
    # Scratch space for computation results (in HBM)
    computation_scratch_ref: Ref,  # [2, M_half_block, N]
    # VMEM scratch buffers for tiled matmul
    x_vmem_ref: Ref,  # [bm, bk] in VMEM
    y_vmem_ref: Ref,  # [bn, bk] in VMEM
    acc_vmem_ref: Ref,  # [bm, bn] in VMEM
    out_vmem_ref: Ref,  # [bm, bn] in VMEM
    # VMEM scratch for tiled add
    add_vmem_ref: Ref,  # [bm, bn] in VMEM for tiled_add src
    # Semaphores
    send_sem: Ref,  # DMA semaphore for sending
    recv_sem: Ref,  # DMA semaphore for receiving
    copy_sem: Ref,  # DMA semaphore for local copies
    left_capacity_sem: Ref,  # REGULAR semaphore for flow control (left)
    right_capacity_sem: Ref,  # REGULAR semaphore for flow control (right)
    *,
    config: KernelConfig,
    axis_name: str,
):
    """Bidirectional Reduce-Scatter Matmul Kernel with M-split algorithm.

    Grid: (num_devices,) where each iteration is one ring step.

    Key insight: Split M into N blocks, each block into TOP and BOT halves.
    - LEFT direction handles all TOP halves (reduced via left ring)
    - RIGHT direction handles all BOT halves (reduced via right ring)

    This ensures:
    - No collision at midpoint (different halves)
    - Perfect load balance (every step has 2 half-block matmuls)
    - Full bandwidth utilization (both directions active)
    """
    num_devices = config.num_devices
    m_block = config.m_block
    m_half_block = config.m_half_block
    bm, bn, bk = config.bm, config.bn, config.bk

    # Grid index = ring step
    ring_step = pl.program_id(0)

    # Device topology
    my_id = lax.axis_index(axis_name)
    left_neighbor = mod(my_id - 1, num_devices)
    right_neighbor = mod(my_id + 1, num_devices)

    # Double buffer management for bidirectional communication
    # Left direction: slots 0 and 1
    # Right direction: slots 2 and 3
    left_working_slot = lax.rem(ring_step, 2)  # 0 or 1
    left_receiving_slot = 1 - left_working_slot
    right_working_slot = 2 + lax.rem(ring_step, 2)  # 2 or 3
    right_receiving_slot = 5 - right_working_slot  # 3 or 2

    # Computation scratch slots (double buffered)
    left_compute_slot = 0
    right_compute_slot = 1

    m_total, k_shard = x_ref.shape
    n_total, _ = y_ref.shape

    num_steps = num_devices
    is_first_step = ring_step == 0
    is_last_step = ring_step == num_steps - 1

    # Semaphores are now passed as parameters from scratch_shapes
    # (send_sem, recv_sem, copy_sem, left_capacity_sem, right_capacity_sem)

    # =========================================================================
    # Helper functions
    # =========================================================================

    def get_left_target_block(step):
        """LEFT direction: compute TOP half of block (my_id + step + 1) % N."""
        return mod(my_id + step + 1, num_devices)

    def get_right_target_block(step):
        """RIGHT direction: compute BOT half of block (my_id - step - 1) % N."""
        return mod(my_id - step - 1, num_devices)

    def compute_matmul_top_half(block_idx, out_slot):
        """Compute matmul for TOP half of specified block.

        Result: x[block_top, :] @ y.T → computation_scratch_ref[out_slot]
        """
        # Convert to block index: m_start = block_idx * m_block
        # m_block_idx = m_start // bm = block_idx * (m_block // bm)
        m_block_idx = block_idx * (m_block // bm)

        tiled_matmul_hbm(
            x_hbm_ref=x_ref,
            y_hbm_ref=y_ref,
            out_hbm_ref=computation_scratch_ref.at[out_slot],
            x_vmem_ref=x_vmem_ref,
            y_vmem_ref=y_vmem_ref,
            acc_vmem_ref=acc_vmem_ref,
            out_vmem_ref=out_vmem_ref,
            copy_sem=copy_sem,
            m_block_idx=m_block_idx,
            m_size=m_half_block,
            bm=bm,
            bn=bn,
            bk=bk,
        )

    def compute_matmul_bot_half(block_idx, out_slot):
        """Compute matmul for BOT half of specified block.

        Result: x[block_bot, :] @ y.T → computation_scratch_ref[out_slot]
        """
        # Convert to block index: m_start = block_idx * m_block + m_half_block
        # m_block_idx = m_start // bm = block_idx * (m_block // bm) + (m_half_block // bm)
        m_block_idx = block_idx * (m_block // bm) + (m_half_block // bm)

        tiled_matmul_hbm(
            x_hbm_ref=x_ref,
            y_hbm_ref=y_ref,
            out_hbm_ref=computation_scratch_ref.at[out_slot],
            x_vmem_ref=x_vmem_ref,
            y_vmem_ref=y_vmem_ref,
            acc_vmem_ref=acc_vmem_ref,
            out_vmem_ref=out_vmem_ref,
            copy_sem=copy_sem,
            m_block_idx=m_block_idx,
            m_size=m_half_block,
            bm=bm,
            bn=bn,
            bk=bk,
        )

    def accumulate_computation_to_slot(compute_slot, dst_slot):
        """Add computation_scratch_ref[compute_slot] to scratch_ref[dst_slot]."""
        tiled_add_hbm(
            src_hbm_ref=computation_scratch_ref.at[compute_slot],
            dst_hbm_ref=scratch_ref.at[dst_slot],
            src_vmem_ref=add_vmem_ref,
            dst_vmem_ref=out_vmem_ref,
            copy_sem=copy_sem,
            bm=bm,
            bn=bn,
        )

    def copy_computation_to_slot(compute_slot, dst_slot):
        """Copy computation_scratch_ref[compute_slot] to scratch_ref[dst_slot]."""
        local_copy = pltpu.make_async_copy(
            src_ref=computation_scratch_ref.at[compute_slot],
            dst_ref=scratch_ref.at[dst_slot],
            sem=copy_sem,
        )
        local_copy.start()
        local_copy.wait()

    def local_barrier():
        """Barrier with both neighbors using double-barrier pattern."""
        barrier_sem = pltpu.get_barrier_semaphore()

        pltpu.semaphore_signal(
            barrier_sem,
            inc=1,
            device_id=(left_neighbor, ),
            device_id_type=pltpu.DeviceIdType.MESH,
        )
        pltpu.semaphore_signal(
            barrier_sem,
            inc=1,
            device_id=(right_neighbor, ),
            device_id_type=pltpu.DeviceIdType.MESH,
        )
        pltpu.semaphore_wait(barrier_sem, 2)

        @functools.partial(pl.run_scoped,
                           second_barrier=pltpu.SemaphoreType.REGULAR)
        def _(second_barrier):
            pltpu.semaphore_signal(
                second_barrier,
                inc=1,
                device_id=(left_neighbor, ),
                device_id_type=pltpu.DeviceIdType.MESH,
            )
            pltpu.semaphore_signal(
                second_barrier,
                inc=1,
                device_id=(right_neighbor, ),
                device_id_type=pltpu.DeviceIdType.MESH,
            )
            pltpu.semaphore_wait(second_barrier, 2)

    def signal_left_neighbor():
        """Signal left neighbor that we are ready to receive from them."""
        pltpu.semaphore_signal(
            left_capacity_sem,
            inc=1,
            device_id=(left_neighbor, ),
            device_id_type=pltpu.DeviceIdType.MESH,
        )

    def signal_right_neighbor():
        """Signal right neighbor that we are ready to receive from them."""
        pltpu.semaphore_signal(
            right_capacity_sem,
            inc=1,
            device_id=(right_neighbor, ),
            device_id_type=pltpu.DeviceIdType.MESH,
        )

    # Get target blocks for current step
    left_target_block = get_left_target_block(ring_step)
    right_target_block = get_right_target_block(ring_step)

    # =========================================================================
    # STEP 0: Prologue - Barrier, compute, prepare for sending
    # =========================================================================
    @pl.when(is_first_step)
    def _prologue():
        # Barrier with both neighbors
        local_barrier()

        # Compute TOP half of left target block (for LEFT direction)
        compute_matmul_top_half(left_target_block, left_compute_slot)
        # Compute BOT half of right target block (for RIGHT direction)
        compute_matmul_bot_half(right_target_block, right_compute_slot)

        # Copy computation results to scratch slots for sending
        # LEFT result goes to left_working_slot (will be sent left in step 1)
        # RIGHT result goes to right_working_slot (will be sent right in step 1)
        copy_computation_to_slot(left_compute_slot, left_working_slot)
        copy_computation_to_slot(right_compute_slot, right_working_slot)

    # =========================================================================
    # STEPS 1 to N-1: Main loop with bidirectional DMA and compute
    # =========================================================================
    @pl.when(~is_first_step)
    def _main_loop():
        # Signal both neighbors that we're ready to receive from them
        signal_left_neighbor()
        signal_right_neighbor()

        # Wait for both neighbors to be ready to receive from us
        pltpu.semaphore_wait(left_capacity_sem, 1)
        pltpu.semaphore_wait(right_capacity_sem, 1)

        # Bidirectional DMA:
        # - Send left_receiving_slot to left neighbor, receive from right into left_working_slot
        # - Send right_receiving_slot to right neighbor, receive from left into right_working_slot
        remote_copy_to_left = pltpu.make_async_remote_copy(
            src_ref=scratch_ref.at[left_receiving_slot],
            dst_ref=scratch_ref.at[left_working_slot],
            send_sem=send_sem,
            recv_sem=recv_sem,
            device_id=(left_neighbor, ),
            device_id_type=pltpu.DeviceIdType.MESH,
        )
        remote_copy_to_left.start()

        remote_copy_to_right = pltpu.make_async_remote_copy(
            src_ref=scratch_ref.at[right_receiving_slot],
            dst_ref=scratch_ref.at[right_working_slot],
            send_sem=send_sem,
            recv_sem=recv_sem,
            device_id=(right_neighbor, ),
            device_id_type=pltpu.DeviceIdType.MESH,
        )
        remote_copy_to_right.start()

        # DELAYED COMPUTATION (overlapped with DMA):
        # - LEFT: compute TOP half of current target block
        # - RIGHT: compute BOT half of current target block
        compute_matmul_top_half(left_target_block, left_compute_slot)
        compute_matmul_bot_half(right_target_block, right_compute_slot)

        # Wait for both DMAs to complete
        remote_copy_to_left.wait()
        remote_copy_to_right.wait()

        # At last step: write output
        @pl.when(is_last_step)
        def _epilogue():
            # Accumulate own contribution to received data
            accumulate_computation_to_slot(left_compute_slot,
                                           left_working_slot)
            accumulate_computation_to_slot(right_compute_slot,
                                           right_working_slot)

            # Write output:
            # - TOP half (from LEFT direction) → out_ref[0:m_half_block, :]
            # - BOT half (from RIGHT direction) → out_ref[m_half_block:m_block, :]
            top_copy = pltpu.make_async_copy(
                src_ref=scratch_ref.at[left_working_slot],
                dst_ref=out_ref.at[pl.ds(0, m_half_block), :],
                sem=copy_sem,
            )
            top_copy.start()
            top_copy.wait()

            bot_copy = pltpu.make_async_copy(
                src_ref=scratch_ref.at[right_working_slot],
                dst_ref=out_ref.at[pl.ds(m_half_block, m_half_block), :],
                sem=copy_sem,
            )
            bot_copy.start()
            bot_copy.wait()

        # For non-last steps: accumulate and continue
        @pl.when(~is_last_step)
        def _accumulate():
            accumulate_computation_to_slot(left_compute_slot,
                                           left_working_slot)
            accumulate_computation_to_slot(right_compute_slot,
                                           right_working_slot)


def bidirectional_reduce_scatter_matmul(
    x: jax.Array,  # [M, K_shard] - K is sharded
    y: jax.Array,  # [N, K_shard] - K is sharded
    *,
    axis_name: str = "x",
    bm: int = 128,
    bn: int = 128,
    bk: int = 128,
) -> jax.Array:
    """Bidirectional reduce-scatter matmul with M-split algorithm.

    Computes: reduce_scatter(x @ y.T, scatter_dim=0)

    Where:
    - x @ y.T is computed with K sharded across devices
    - Result is scattered on M dimension (each device gets M/num_devices rows)

    The algorithm splits M into N blocks (one per device), and each block into
    TOP and BOT halves. LEFT direction handles TOP halves, RIGHT handles BOT halves.
    This achieves:
    - 2x ICI bandwidth utilization
    - Perfect load balance (every step has compute work)
    - No collision (directions always process different halves)

    Args:
        x: Input tensor [M, K_shard] where K is sharded across devices
        y: Weight tensor [N, K_shard] where K is sharded across devices
        axis_name: Name of the device axis for collective operations
        bm: Block size for M dimension (must divide M_half_block)
        bn: Block size for N dimension (must divide N)
        bk: Block size for K dimension (must divide K_shard)

    Returns:
        Output tensor [M_block, N] where M is scattered across devices
    """
    num_devices = lax.psum(1, axis_name)

    m_total, k_shard = x.shape
    n_total, _ = y.shape

    assert (
        m_total % num_devices == 0
    ), f"M ({m_total}) must be divisible by num_devices ({num_devices})"

    m_block = m_total // num_devices

    assert m_block % 2 == 0, f"M_block ({m_block}) must be divisible by 2"

    m_half_block = m_block // 2

    # Validate tile sizes
    assert (m_half_block % bm == 0
            ), f"M_half_block ({m_half_block}) must be divisible by bm ({bm})"
    assert n_total % bn == 0, f"N ({n_total}) must be divisible by bn ({bn})"
    assert k_shard % bk == 0, f"K_shard ({k_shard}) must be divisible by bk ({bk})"

    config = KernelConfig(
        num_devices=num_devices,
        m_block=m_block,
        m_half_block=m_half_block,
        bm=bm,
        bn=bn,
        bk=bk,
    )

    # Output shape: [M_block, N]
    out_shape = jax.ShapeDtypeStruct((m_block, n_total), x.dtype)

    # Scratch shapes for communication (4 slots for bidirectional double buffering)
    scratch_shape = jax.ShapeDtypeStruct((4, m_half_block, n_total), x.dtype)

    # Scratch shapes for computation (2 slots)
    computation_scratch_shape = jax.ShapeDtypeStruct(
        (2, m_half_block, n_total), x.dtype)

    # VMEM scratch shapes
    x_vmem_shape = pltpu.VMEM((bm, bk), x.dtype)
    y_vmem_shape = pltpu.VMEM((bn, bk), y.dtype)
    acc_vmem_shape = pltpu.VMEM((bm, bn), jnp.float32)
    out_vmem_shape = pltpu.VMEM((bm, bn), x.dtype)
    add_vmem_shape = pltpu.VMEM((bm, bn), x.dtype)

    # Grid: one iteration per ring step
    grid = (num_devices, )

    def kernel_fn(x_ref, y_ref, out_ref, scratch_ref, computation_scratch_ref,
                  x_vmem_ref, y_vmem_ref, acc_vmem_ref, out_vmem_ref,
                  add_vmem_ref, send_sem, recv_sem, copy_sem,
                  left_capacity_sem, right_capacity_sem):
        _kernel(
            x_ref=x_ref,
            y_ref=y_ref,
            out_ref=out_ref,
            scratch_ref=scratch_ref,
            computation_scratch_ref=computation_scratch_ref,
            x_vmem_ref=x_vmem_ref,
            y_vmem_ref=y_vmem_ref,
            acc_vmem_ref=acc_vmem_ref,
            out_vmem_ref=out_vmem_ref,
            add_vmem_ref=add_vmem_ref,
            send_sem=send_sem,
            recv_sem=recv_sem,
            copy_sem=copy_sem,
            left_capacity_sem=left_capacity_sem,
            right_capacity_sem=right_capacity_sem,
            config=config,
            axis_name=axis_name,
        )

    out, _, _ = pl.pallas_call(
        kernel_fn,
        out_shape=(out_shape, scratch_shape, computation_scratch_shape),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            in_specs=[
                pl.BlockSpec(memory_space=pltpu.ANY),  # x
                pl.BlockSpec(memory_space=pltpu.ANY),  # y
            ],
            out_specs=[
                pl.BlockSpec(memory_space=pltpu.ANY),  # out
                pl.BlockSpec(memory_space=pltpu.ANY),  # scratch
                pl.BlockSpec(memory_space=pltpu.ANY),  # computation_scratch
            ],
            scratch_shapes=[
                x_vmem_shape,
                y_vmem_shape,
                acc_vmem_shape,
                out_vmem_shape,
                add_vmem_shape,
                # DMA semaphores
                pltpu.SemaphoreType.DMA,  # send_sem
                pltpu.SemaphoreType.DMA,  # recv_sem
                pltpu.SemaphoreType.DMA,  # copy_sem
                # Flow control semaphores
                pltpu.SemaphoreType.REGULAR,  # left_capacity_sem
                pltpu.SemaphoreType.REGULAR,  # right_capacity_sem
            ],
            grid=grid,
        ),
        compiler_params=pltpu.CompilerParams(
            collective_id=0,
            dimension_semantics=("arbitrary", ),
        ),
    )(x, y)

    return out


# Algorithm diagram for reference
ALGORITHM_DIAGRAM = """
BIDIRECTIONAL REDUCE-SCATTER MATMUL WITH M-SPLIT ALGORITHM
===========================================================

M dimension split into N blocks, each block split into TOP and BOT halves:

Full M dimension (8 devices):
┌─────────┬─────────┬─────────┬─────────┬─────────┬─────────┬─────────┬─────────┐
│ Block 0 │ Block 1 │ Block 2 │ Block 3 │ Block 4 │ Block 5 │ Block 6 │ Block 7 │
└─────────┴─────────┴─────────┴─────────┴─────────┴─────────┴─────────┴─────────┘

Each block split:
┌────┬────┬────┬────┬────┬────┬────┬────┬────┬────┬────┬────┬────┬────┬────┬────┐
│B0_T│B0_B│B1_T│B1_B│B2_T│B2_B│B3_T│B3_B│B4_T│B4_B│B5_T│B5_B│B6_T│B6_B│B7_T│B7_B│
└────┴────┴────┴────┴────┴────┴────┴────┴────┴────┴────┴────┴────┴────┴────┴────┘
  │         │         │         │         │         │         │         │
  └─────────┴─────────┴─────────┴─────────┴─────────┴─────────┴─────────┘
                            LEFT (all TOP halves)
        └─────────┴─────────┴─────────┴─────────┴─────────┴─────────┴─────────┘
                            RIGHT (all BOT halves)

Direction assignment:
  LEFT:  All TOP halves → reduces via left ring (D0→D7→D6→...→D1→D0)
  RIGHT: All BOT halves → reduces via right ring (D0→D1→D2→...→D7→D0)

Device 0's computation schedule (8 devices):
┌──────────┬─────────────────────────┬─────────────────────────┐
│   Step   │     LEFT (TOP half)     │    RIGHT (BOT half)     │
├──────────┼─────────────────────────┼─────────────────────────┤
│    0     │ P₀(B1_TOP) → send to D7 │ P₀(B7_BOT) → send to D1 │
│    1     │ P₀(B2_TOP) + accum      │ P₀(B6_BOT) + accum      │
│    2     │ P₀(B3_TOP) + accum      │ P₀(B5_BOT) + accum      │
│    3     │ P₀(B4_TOP) + accum      │ P₀(B4_BOT) + accum      │  ← Same block, DIFFERENT halves!
│    4     │ P₀(B5_TOP) + accum      │ P₀(B3_BOT) + accum      │
│    5     │ P₀(B6_TOP) + accum      │ P₀(B2_BOT) + accum      │
│    6     │ P₀(B7_TOP) + accum      │ P₀(B1_BOT) + accum      │
│  Final   │ P₀(B0_TOP) → output TOP │ P₀(B0_BOT) → output BOT │
└──────────┴─────────────────────────┴─────────────────────────┘

Pipeline timeline (Device 0):
Time ────────────────────────────────────────────────────────────────────────────►

Step:     0      1      2      3      4      5      6     Final
         ───    ───    ───    ───    ───    ───    ───    ─────

LEFT:    ████   ████   ████   ████   ████   ████   ████   ████
(TOP)    B1_T   B2_T   B3_T   B4_T   B5_T   B6_T   B7_T   B0_T

RIGHT:   ████   ████   ████   ████   ████   ████   ████   ████
(BOT)    B7_B   B6_B   B5_B   B4_B   B3_B   B2_B   B1_B   B0_B

L-DMA:          ════   ════   ════   ════   ════   ════   ════
                →D7    →D7    →D7    →D7    →D7    →D7    →D7

R-DMA:          ════   ════   ════   ════   ════   ════   ════
                →D1    →D1    →D1    →D1    →D1    →D1    →D1

Matmuls: [2]    [2]    [2]    [2]    [2]    [2]    [2]    [2]

KEY BENEFITS:
✓ NO COLLISION: LEFT and RIGHT always process DIFFERENT halves
✓ PERFECT BALANCE: Every step has exactly 2 half-block matmuls
✓ NO IDLE STEPS: Both directions always have compute work
✓ 2X BANDWIDTH: Both ICI directions fully utilized
✓ GOOD OVERLAP: Compute overlaps with bidirectional DMA

Final output at Device 0:
┌─────────────────────────────────────────────────────────────┐
│  Block 0 TOP (from LEFT):  P₀ + P₁ + P₂ + ... + P₇        │
├─────────────────────────────────────────────────────────────┤
│  Block 0 BOT (from RIGHT): P₀ + P₁ + P₂ + ... + P₇        │
└─────────────────────────────────────────────────────────────┘
Combined: Complete Block 0 with full reduction from all 8 devices ✓
"""

if __name__ == "__main__":
    print(ALGORITHM_DIAGRAM)
