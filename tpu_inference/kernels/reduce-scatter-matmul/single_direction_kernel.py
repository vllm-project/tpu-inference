# SPDX-License-Identifier: Apache-2.0
"""Single-Direction Reduce-Scatter Matmul with Delayed Computation.

This is a simplified, educational implementation showing the core concept of
delayed computation in a reduce-scatter fused with matmul.

Setup (4 devices example):
- Each device has: x[M, K_shard], y[N, K_shard] where K is sharded
- Output: each device gets its M_shard portion with full K reduction
- D0 owns M_shard 0 (rows 0 to M/4-1)
- D1 owns M_shard 1 (rows M/4 to M/2-1)
- D2 owns M_shard 2, D3 owns M_shard 3

Ring direction: D0 → D1 → D2 → D3 → D0 (send to right neighbor)

DELAYED COMPUTATION ALGORITHM:
==============================

The key insight: DON'T compute your local partial upfront. Instead, compute it
WHEN the accumulator for that shard arrives.

From Device 0's perspective (4 devices, D0 owns shard 0):

  Step 0: Compute P₀(shard 3) - partial for shard 3 using D0's K_shard
          Send P₀(shard 3) → D1
          [NEW MATMUL WORK: computing for shard 3]

  Step 1: Receive accumulator A(shard 2) from D3
          A(shard 2) contains: P₃(shard 2)
          NOW compute P₀(shard 2) ← DELAYED!
          Accumulate: A(shard 2) + P₀(shard 2)
          Send accumulated result → D1
          [NEW MATMUL WORK: computing for shard 2]

  Step 2: Receive accumulator A(shard 1) from D3
          A(shard 1) contains: P₂(shard 1) + P₃(shard 1)
          NOW compute P₀(shard 1) ← DELAYED!
          Accumulate and send → D1
          [NEW MATMUL WORK: computing for shard 1]

  Step 3 (Final): Receive accumulator A(shard 0) from D3
          A(shard 0) contains: P₁ + P₂ + P₃ for shard 0
          NOW compute P₀(shard 0) ← DELAYED!
          Final result: P₀ + P₁ + P₂ + P₃ for shard 0
          Write to output (this is D0's result!)
          [NEW MATMUL WORK: computing for shard 0]

EVERY STEP HAS NEW MATMUL WORK because we delay computing until needed!
"""

# TODO(rupengliu) allow run ahead with recv_sem
# TODO(rupengliu) add bidirectional support
# TODO(rupengliu) increase pipeline depth through m and n tiling

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
    m_shard: int  # M rows per device = M // num_devices
    # Tile sizes for matmul
    bm: int = 128  # Block size for M dimension
    bn: int = 128  # Block size for N dimension
    bk: int = 128  # Block size for K dimension


# TODO(rupengliu) improve matmul
def tiled_matmul_hbm(
    x_hbm_ref: Ref,  # [M_shard, K_shard] in HBM
    y_hbm_ref: Ref,  # [N, K_shard] in HBM
    out_hbm_ref: Ref,  # [M_shard, N] in HBM
    # VMEM scratch buffers
    x_vmem_ref: Ref,  # [bm, bk] in VMEM
    y_vmem_ref: Ref,  # [bn, bk] in VMEM
    acc_vmem_ref: Ref,  # [bm, bn] in VMEM (float32 for accumulation)
    out_vmem_ref: Ref,  # [bm, bn] in VMEM
    # Semaphores
    copy_sem: Ref,  # DMA semaphore
    *,
    bm: int,
    bn: int,
    bk: int,
):
    """Tiled matmul: out = x @ y.T, with HBM inputs and output.

    This function tiles the matmul computation using async_copy for HBM<->VMEM transfers.

    The result OVERWRITES the output buffer (does not accumulate to existing values).

    Args:
        x_hbm_ref: Input x in HBM [M_shard, K_shard]
        y_hbm_ref: Input y in HBM [N, K_shard]
        out_hbm_ref: Output in HBM [M_shard, N]
        x_vmem_ref: VMEM scratch for x tile [bm, bk]
        y_vmem_ref: VMEM scratch for y tile [bn, bk]
        acc_vmem_ref: VMEM scratch for accumulator [bm, bn] in float32
        out_vmem_ref: VMEM scratch for output tile [bm, bn]
        copy_sem: DMA semaphore for async copies
        bm: Block size for M dimension
        bn: Block size for N dimension
        bk: Block size for K dimension
    """
    m_shard, k_shard = x_hbm_ref.shape
    n_total, _ = y_hbm_ref.shape

    num_m_tiles = m_shard // bm
    num_n_tiles = n_total // bn
    num_k_tiles = k_shard // bk

    for m_tile in range(num_m_tiles):
        m_start = m_tile * bm
        for n_tile in range(num_n_tiles):
            n_start = n_tile * bn

            # Zero the accumulator
            acc_vmem_ref[...] = jnp.zeros((bm, bn), dtype=jnp.float32)

            # Iterate over K tiles
            for k_tile in range(num_k_tiles):
                k_start = k_tile * bk

                # Copy x tile from HBM to VMEM
                x_copy = pltpu.make_async_copy(
                    src_ref=x_hbm_ref.at[pl.ds(m_start, bm),
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
            out_copy = pltpu.make_async_copy(
                src_ref=out_vmem_ref,
                dst_ref=out_hbm_ref.at[pl.ds(m_start, bm),
                                       pl.ds(n_start, bn)],
                sem=copy_sem,
            )
            out_copy.start()
            out_copy.wait()


# TODO(rupengliu) improve add
def tiled_add_hbm(
    src_hbm_ref: Ref,  # [M_shard, N] in HBM
    dst_hbm_ref: Ref,  # [M_shard, N] in HBM (will be modified in place)
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

    Args:
        src_hbm_ref: Source buffer in HBM [M_shard, N]
        dst_hbm_ref: Destination buffer in HBM [M_shard, N] (modified in-place)
        src_vmem_ref: VMEM scratch for src tile [bm, bn]
        dst_vmem_ref: VMEM scratch for dst tile [bm, bn]
        copy_sem: DMA semaphore for async copies
        bm: Block size for M dimension
        bn: Block size for N dimension
    """
    m_shard, n_total = src_hbm_ref.shape

    num_m_tiles = m_shard // bm
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


def _single_dir_reduce_scatter_matmul_kernel(
    # Inputs (in HBM)
    x_ref: Ref,  # [M, K_shard]
    y_ref: Ref,  # [N, K_shard]
    # Outputs (in HBM)
    out_ref: Ref,  # [M_shard, N] - this device's output portion
    # Scratch buffer for ring communication - double buffered (in HBM for DMA)
    scratch_ref: Ref,  # [2, M_shard, N]
    # Scratch buffer for computation result (separate from communication buffers)
    computation_scratch_ref: Ref,  # [M_shard, N] - stores matmul result
    # Semaphores (order must match scratch_shapes)
    send_sem: Ref,  # DMA semaphore for sending
    recv_sem: Ref,  # DMA semaphore for receiving
    copy_sem: Ref,  # DMA semaphore for local copies
    capacity_sem: Ref,  # REGULAR semaphore for flow control
    # VMEM scratch buffers for tiled matmul
    x_vmem_ref: Ref,  # [bm, bk] in VMEM
    y_vmem_ref: Ref,  # [bn, bk] in VMEM
    acc_vmem_ref: Ref,  # [bm, bn] in VMEM (float32 for accumulation)
    out_vmem_ref: Ref,  # [bm, bn] in VMEM
    # VMEM scratch for tiled add
    add_vmem_ref: Ref,  # [bm, bn] in VMEM for tiled_add src
    *,
    config: KernelConfig,
    axis_name: str,
):
    """Single-direction reduce-scatter matmul kernel with delayed compute.

    Grid: (num_devices,) where each iteration is one ring step.

    Ring direction: send to RIGHT neighbor (device (my_id + 1) % N)
    Receive from LEFT neighbor.

    This implementation follows the pattern from the working reduce_scatter kernel:
    - Barrier + initial signal at step 0
    - Wait for capacity before processing received data
    - Signal capacity after forwarding data
    """
    num_devices = config.num_devices
    m_shard = config.m_shard
    bm, bn, bk = config.bm, config.bn, config.bk

    # Grid index = ring step
    ring_step = pl.program_id(0)

    # Device topology
    my_id = lax.axis_index(axis_name)
    right_neighbor = mod(my_id + 1, num_devices)
    left_neighbor = mod(my_id - 1, num_devices)

    # Double buffer management
    working_slot = lax.rem(ring_step, 2)
    receiving_slot = 1 - working_slot

    m_total, k_shard = x_ref.shape
    n_total, _ = y_ref.shape

    is_first_step = ring_step == 0
    is_last_step = ring_step == num_devices - 1

    # =========================================================================
    # Helper functions
    # =========================================================================

    def get_target_shard(step):
        """Determine which M_shard we're working on at this step."""
        return mod(my_id - step - 1, num_devices)

    def get_m_start_for_shard(shard_idx):
        """Get the starting M row index for a given shard."""
        return shard_idx * m_shard

    def compute_matmul_for_shard(shard_idx):
        """Compute matmul for a specific shard using tiled computation.

        Result is written to computation_scratch_ref (separate from comm buffers).
        This allows computation to overlap with communication without buffer conflicts.
        """
        m_start = get_m_start_for_shard(shard_idx)

        tiled_matmul_hbm(
            x_hbm_ref=x_ref.at[pl.ds(m_start, m_shard), :],
            y_hbm_ref=y_ref,
            out_hbm_ref=computation_scratch_ref,
            x_vmem_ref=x_vmem_ref,
            y_vmem_ref=y_vmem_ref,
            acc_vmem_ref=acc_vmem_ref,
            out_vmem_ref=out_vmem_ref,
            copy_sem=copy_sem,
            bm=bm,
            bn=bn,
            bk=bk,
        )

    def accumulate_computation_to_slot(dst_slot):
        """Add computation_scratch_ref to scratch_ref[dst_slot] in place.

        dst_slot += computation_scratch_ref
        """
        tiled_add_hbm(
            src_hbm_ref=computation_scratch_ref,
            dst_hbm_ref=scratch_ref.at[dst_slot],
            src_vmem_ref=add_vmem_ref,
            dst_vmem_ref=out_vmem_ref,
            copy_sem=copy_sem,
            bm=bm,
            bn=bn,
        )

    def copy_computation_to_slot(dst_slot):
        """Copy computation_scratch_ref to scratch_ref[dst_slot]."""
        tiled_add_hbm(
            src_hbm_ref=computation_scratch_ref,
            dst_hbm_ref=scratch_ref.at[dst_slot],
            src_vmem_ref=add_vmem_ref,
            dst_vmem_ref=out_vmem_ref,
            copy_sem=copy_sem,
            bm=bm,
            bn=bn,
        )

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
        """Signal left neighbor that we are ready to receive from them.

        This is a flow control mechanism - we tell our left neighbor (who sends to us)
        that we are ready to receive. Without this, our left neighbor could get ahead
        and overwrite our buffer before we've processed it.

        The receiver signals the sender, not the other way around.
        """
        pltpu.semaphore_signal(
            capacity_sem,
            inc=1,
            device_id=(left_neighbor, ),
            device_id_type=pltpu.DeviceIdType.MESH,
        )

    # Which shard are we computing for this step?
    target_shard = get_target_shard(ring_step)

    # =========================================================================
    # STEP 0: Barrier, compute, initial send
    # =========================================================================
    @pl.when(is_first_step)
    def _prologue():
        # Barrier with both neighbors
        local_barrier()
        # TODO(rupengliu): allow compute_matmul_for_shard to directly write a selected hbm ref to avoid the local copy
        # Compute our partial for target_shard -> computation_scratch_ref
        compute_matmul_for_shard(target_shard)

        # Copy computation result to working_slot for sending in next step
        final_copy = pltpu.make_async_copy(
            src_ref=computation_scratch_ref,
            dst_ref=scratch_ref.at[working_slot],
            sem=copy_sem,
        )
        final_copy.start()
        final_copy.wait()

    # =========================================================================
    # STEPS 1 to N-1: Receive, accumulate, forward (or write output)
    # =========================================================================
    @pl.when(~is_first_step)
    def _main_loop():
        # Signal left neighbor that we're ready to receive from them
        # (flow control - receiver tells sender they're ready)
        # This must happen BEFORE we wait and send, following the all_reduce pattern.
        signal_left_neighbor()

        # Block until our right neighbor is ready to receive from us.
        # Our right neighbor signals us (their left neighbor) when they're ready.
        pltpu.semaphore_wait(capacity_sem, 1)

        # Now we can safely send/receive - our right neighbor is ready.
        # Send receiving_slot to right neighbor, receive from left neighbor into working_slot.
        # Note: receiving_slot contains the accumulated result from the previous step.
        remote_copy = pltpu.make_async_remote_copy(
            src_ref=scratch_ref.at[receiving_slot],
            dst_ref=scratch_ref.at[working_slot],
            send_sem=send_sem,
            recv_sem=recv_sem,
            device_id=(right_neighbor, ),
            device_id_type=pltpu.DeviceIdType.MESH,
        )
        remote_copy.start()

        # DELAYED COMPUTATION: Compute matmul to computation_scratch_ref
        # This is now in a SEPARATE buffer, so no conflict with working_slot!
        # Computation overlaps with communication.
        compute_matmul_for_shard(target_shard)
        # TODO(rupengliu) check if we could move the remote_copy later
        # Wait for communication to complete
        remote_copy.wait()

        # Accumulate: add our computation result to received data in working_slot
        # working_slot now contains the received accumulated partial from left neighbor
        accumulate_computation_to_slot(dst_slot=working_slot)

        @pl.when(is_last_step)
        def _epilogue():
            # Copy final result to output (accumulated result is in working_slot)
            final_copy = pltpu.make_async_copy(
                src_ref=scratch_ref.at[working_slot],
                dst_ref=out_ref,
                sem=copy_sem,
            )
            final_copy.start()
            final_copy.wait()


def single_dir_reduce_scatter_matmul(
    x: jax.Array,  # [M, K_shard] - K is sharded
    y: jax.Array,  # [N, K_shard] - K is sharded
    *,
    axis_name: str = "x",
    bm: int = 128,
    bn: int = 128,
    bk: int = 128,
) -> jax.Array:
    """Single-direction reduce-scatter matmul with delayed computation.

    Computes: reduce_scatter(x @ y.T, scatter_dim=0)

    Where:
    - x @ y.T is computed with K sharded across devices
    - Result is scattered on M dimension (each device gets M/num_devices rows)

    The matmul is computed using a tiled approach with emit_pipeline to handle
    large matrices that don't fit entirely in VMEM.

    Args:
        x: Input tensor [M, K_shard] where K is sharded across devices
        y: Weight tensor [N, K_shard] where K is sharded across devices
        axis_name: Name of the device axis for collective operations
        bm: Block size for M dimension (must divide M_shard)
        bn: Block size for N dimension (must divide N)
        bk: Block size for K dimension (must divide K_shard)

    Returns:
        Output tensor [M_shard, N] where M is scattered across devices
    """
    num_devices = lax.psum(1, axis_name)

    m_total, k_shard = x.shape
    n_total, _ = y.shape

    assert (
        m_total % num_devices == 0
    ), f"M ({m_total}) must be divisible by num_devices ({num_devices})"

    m_shard = m_total // num_devices

    # Validate tile sizes
    assert m_shard % bm == 0, f"M_shard ({m_shard}) must be divisible by bm ({bm})"
    assert n_total % bn == 0, f"N ({n_total}) must be divisible by bn ({bn})"
    assert k_shard % bk == 0, f"K_shard ({k_shard}) must be divisible by bk ({bk})"

    config = KernelConfig(
        num_devices=num_devices,
        m_shard=m_shard,
        bm=bm,
        bn=bn,
        bk=bk,
    )

    # Output shape: [M_shard, N]
    out_shape = jax.ShapeDtypeStruct((m_shard, n_total), x.dtype)

    # Scratch shape: [2, M_shard, N] for double buffering (communication)
    scratch_shape = jax.ShapeDtypeStruct((2, m_shard, n_total), x.dtype)

    # Computation scratch: [M_shard, N] for storing matmul result (separate from comm)
    computation_scratch_shape = jax.ShapeDtypeStruct((m_shard, n_total),
                                                     x.dtype)

    # Grid: one iteration per ring step
    grid = (num_devices, )

    grid_spec = pltpu.PrefetchScalarGridSpec(
        num_scalar_prefetch=0,
        in_specs=[
            pl.BlockSpec(memory_space=pltpu.ANY),  # x (HBM)
            pl.BlockSpec(memory_space=pltpu.ANY),  # y (HBM)
        ],
        out_specs=[
            pl.BlockSpec(memory_space=pltpu.ANY),  # output (HBM)
            pl.BlockSpec(memory_space=pltpu.ANY),  # scratch (HBM for DMA)
            pl.BlockSpec(memory_space=pltpu.ANY),  # computation_scratch (HBM)
        ],
        grid=grid,
        scratch_shapes=(
            # DMA semaphores
            pltpu.SemaphoreType.DMA,  # send_sem
            pltpu.SemaphoreType.DMA,  # recv_sem
            pltpu.SemaphoreType.DMA,  # copy_sem
            # Flow control semaphore
            pltpu.SemaphoreType.REGULAR,  # capacity_sem
            # VMEM scratch buffers for tiled matmul
            pltpu.VMEM((bm, bk), x.dtype),  # x_vmem_ref
            pltpu.VMEM((bn, bk), y.dtype),  # y_vmem_ref
            pltpu.VMEM((bm, bn),
                       jnp.float32),  # acc_vmem_ref (float32 for accumulation)
            pltpu.VMEM((bm, bn), x.dtype),  # out_vmem_ref
            # VMEM scratch for tiled add
            pltpu.VMEM((bm, bn), x.dtype),  # add_vmem_ref
        ),
    )

    kernel_fn = pl.pallas_call(
        functools.partial(
            _single_dir_reduce_scatter_matmul_kernel,
            config=config,
            axis_name=axis_name,
        ),
        out_shape=(out_shape, scratch_shape, computation_scratch_shape),
        grid_spec=grid_spec,
        compiler_params=pltpu.CompilerParams(collective_id=0),
    )

    out, _, _ = kernel_fn(x, y)
    return out


def single_dir_reduce_scatter_matmul_sharded(
    x: jax.Array,
    y: jax.Array,
    mesh: jax.sharding.Mesh,
    *,
    axis_name: str = "x",
) -> jax.Array:
    """Sharded version using shard_map.

    Args:
        x: Input [M, K] with K sharded as P(None, axis_name)
        y: Weight [N, K] with K sharded as P(None, axis_name)
        mesh: JAX mesh
        axis_name: Name of the sharding axis

    Returns:
        Output [M, N] with M sharded as P(axis_name, None)
    """
    from jax.experimental.shard_map import shard_map

    @functools.partial(
        shard_map,
        mesh=mesh,
        in_specs=(P(None, axis_name), P(None, axis_name)),
        out_specs=P(axis_name, None),
        check_rep=False,
    )
    def _sharded_fn(x_shard, y_shard):
        return single_dir_reduce_scatter_matmul(
            x_shard,
            y_shard,
            axis_name=axis_name,
        )

    return _sharded_fn(x, y)


# ============================================================================
# Reference implementation for correctness testing
# ============================================================================


def reduce_scatter_matmul_reference(
    x: jax.Array,
    y: jax.Array,
    axis_name: str = "x",
) -> jax.Array:
    """Reference implementation using JAX collectives.

    Computes: reduce_scatter(x @ y.T, axis=M)
    """
    # Local matmul: [M, K_shard] @ [K_shard, N] = [M, N] (partial)
    local_result = jnp.dot(x, y.T)

    # Reduce-scatter along M dimension
    result = lax.psum_scatter(local_result,
                              axis_name,
                              scatter_dimension=0,
                              tiled=True)

    return result


# ============================================================================
# Visualization of the algorithm
# ============================================================================

ALGORITHM_DIAGRAM = """
SINGLE-DIRECTION REDUCE-SCATTER MATMUL (4 devices)
==================================================

Setup:
- Each device has x[M, K/4] and y[N, K/4]
- Output: each device gets [M/4, N] with full K reduction
- Ring: D0 → D1 → D2 → D3 → D0

Device ownership:
- D0 owns M_shard 0 (rows 0 to M/4-1)
- D1 owns M_shard 1 (rows M/4 to M/2-1)
- D2 owns M_shard 2
- D3 owns M_shard 3

STEP 0 - Initial compute and send:
┌─────────────────────────────────────────────────────────────────────┐
│  D0                  D1                  D2                  D3     │
│  ────                ────                ────                ────   │
│  Compute P₀(s3)      Compute P₁(s0)      Compute P₂(s1)      P₃(s2) │
│       ↓                   ↓                   ↓                ↓    │
│  Send to D1          Send to D2          Send to D3       Send to D0│
└─────────────────────────────────────────────────────────────────────┘

STEP 1 - Receive, accumulate, forward:
┌─────────────────────────────────────────────────────────────────────┐
│  D0                  D1                  D2                  D3     │
│  ────                ────                ────                ────   │
│  Recv P₃(s2)         Recv P₀(s3)         Recv P₁(s0)         P₂(s1) │
│  Compute P₀(s2)      Compute P₁(s3)      Compute P₂(s0)      P₃(s1) │
│  Accum + Send        Accum + Send        Accum + Send        Accum  │
│  (P₀+P₃)(s2)→D1      (P₀+P₁)(s3)→D2      (P₁+P₂)(s0)→D3      →D0    │
└─────────────────────────────────────────────────────────────────────┘

STEP 2 - Receive, accumulate, forward:
┌─────────────────────────────────────────────────────────────────────┐
│  D0                  D1                  D2                  D3     │
│  ────                ────                ────                ────   │
│  Recv(P₂+P₃)(s1)     Recv(P₀+P₃)(s2)     Recv(P₀+P₁)(s3)    Recv... │
│  Compute P₀(s1)      Compute P₁(s2)      Compute P₂(s3)      P₃(s0) │
│  Accum + Send        Accum + Send        Accum + Send        Accum  │
│  (P₀+P₂+P₃)(s1)      (P₀+P₁+P₃)(s2)      (P₀+P₁+P₂)(s3)      →D0    │
└─────────────────────────────────────────────────────────────────────┘

STEP 3 (FINAL) - Receive, accumulate, write output:
┌─────────────────────────────────────────────────────────────────────┐
│  D0                  D1                  D2                  D3     │
│  ────                ────                ────                ────   │
│  Recv for s0         Recv for s1         Recv for s2         s3     │
│  (P₁+P₂+P₃)(s0)      (P₀+P₂+P₃)(s1)      (P₀+P₁+P₃)(s2)      ...    │
│  Compute P₀(s0)      Compute P₁(s1)      Compute P₂(s2)      P₃(s3) │
│  FINAL:              FINAL:              FINAL:              FINAL: │
│  (P₀+P₁+P₂+P₃)(s0)   (P₀+P₁+P₂+P₃)(s1)   (P₀+P₁+P₂+P₃)(s2)   (s3)   │
│       ↓                   ↓                   ↓                ↓    │
│  Write to output     Write to output     Write to output     Write  │
└─────────────────────────────────────────────────────────────────────┘

KEY INSIGHT: Every step has NEW matmul work!
- Step 0: Compute for shard (my_id - 1) mod N
- Step 1: Compute for shard (my_id - 2) mod N
- Step 2: Compute for shard (my_id - 3) mod N
- Step 3: Compute for shard my_id (our own!)

This is DELAYED COMPUTATION - we don't compute our local partial until the
accumulator for that shard arrives!
"""

if __name__ == "__main__":
    print(ALGORITHM_DIAGRAM)
