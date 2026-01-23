# SPDX-License-Identifier: Apache-2.0
"""Reduce-scatter matmul kernel.

This kernel performs matmul followed by reduce-scatter, which is useful for
tensor parallelism where weights are sharded along the output dimension.

Pattern:
  Input: x [m, k] (replicated), y [k, n_per_device] (sharded by n)
  Output: reduce_scatter(x @ y) -> [m_per_device, n] (sharded by m)

This is the inverse of all-gather matmul and is typically used in row-parallel
linear layers in tensor parallel training/inference.
"""

import functools

import jax
import jax.numpy as jnp
from jax import lax
from jax._src import dtypes
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

P = jax.sharding.PartitionSpec

# -----------------------------------------------------------------------------
# Reference Implementation
# -----------------------------------------------------------------------------


def ref_reduce_scatter_matmul(
    x: jax.Array,
    y: jax.Array,
    mesh: jax.sharding.AbstractMesh,
    axis_name: str,
    lhs_transpose: bool = False,
) -> jax.Array:
    """Reference implementation of reduce-scatter matmul using pure JAX.

    The operation computes:
        output[device_i] = (x @ y)[device_i * m_per_device : (device_i+1) * m_per_device]

    Each device computes the full matmul and takes its shard of rows.
    This is a scatter operation along the m dimension.

    Note: In this pattern, x and y are replicated on all devices, so each device
    computes the same result. The "scatter" distributes different row ranges to
    different devices. There's no actual "reduce" (summation) because all devices
    compute identical values.

    Args:
        x: LHS of the matmul [m, k] (replicated across devices).
        y: RHS of the matmul [k, n_per_device] (local shard, same on all devices).
        mesh: JAX mesh for sharding.
        axis_name: Name of the axis to scatter over.
        lhs_transpose: If True, x is transposed (shape [k, m]).

    Returns:
        matmul(x, y) scattered across devices along m dimension.
        Shape: [m_per_device, n_per_device] per device.

    Example:
        ```python
        import jax
        import jax.numpy as jnp
        from jax.sharding import Mesh, PartitionSpec as P

        # Setup mesh with 4 devices
        mesh = Mesh(jax.devices()[:4], ('x',))

        # Input shapes
        m, k, n = 1024, 4096, 8192
        tp_size = 4
        n_per_device = n // tp_size

        # x is replicated, y is local shard (each device has same copy)
        x = jnp.ones((m, k), dtype=jnp.bfloat16)
        y = jnp.ones((k, n_per_device), dtype=jnp.bfloat16)

        # Compute reference result
        out = ref_reduce_scatter_matmul(x, y, mesh, 'x')
        # out shape per device: [m // tp_size, n_per_device] = [256, 2048]
        ```
    """
    tp_size = mesh.shape[axis_name]

    if lhs_transpose:
        k, m = x.shape
    else:
        m, k = x.shape
    # y is expected to be passed with local shape [k, n_per_device]
    # The shard_map will replicate this on each device since we use P(None, None)
    _, n_per_device = y.shape
    m_per_device = m // tp_size

    def _matmul_reduce_scatter(x, y):
        # Compute local matmul: x @ y
        # x: [m, k], y: [k, n_per_device] -> partial: [m, n_per_device]
        if lhs_transpose:
            # x is [k, m], need to transpose for matmul
            partial = jnp.dot(x.T, y)
        else:
            partial = jnp.dot(x, y)

        # Get device index to select appropriate output shard
        device_idx = lax.axis_index(axis_name)

        # The kernel computes x[device_idx * m_per_device : (device_idx+1) * m_per_device] @ y
        # for its final output (after ring communication settles, each device outputs
        # the slice corresponding to its device index)
        start_idx = device_idx * m_per_device
        output = lax.dynamic_slice(partial, (start_idx, 0),
                                   (m_per_device, n_per_device))
        return output

    # Wrap with shard_map
    # x: replicated (each device has full copy)
    # y: replicated (already local shape [k, n_per_device])
    # output: sharded along m dimension (each device has m_per_device rows)
    sharded_fn = jax.shard_map(
        _matmul_reduce_scatter,
        mesh=mesh,
        in_specs=(P(None, None), P(None, None)),  # x and y both replicated
        out_specs=P(axis_name, None),  # output sharded by m
    )

    return jax.jit(sharded_fn)(x, y)


def ref_reduce_scatter_matmul_naive(
    x: jax.Array,
    y: jax.Array,
    tp_size: int,
    lhs_transpose: bool = False,
) -> jax.Array:
    """Naive reference implementation without mesh/sharding (single device).

    This is the simplest possible implementation, useful for understanding
    the mathematical operation and for testing on a single device.

    Args:
        x: LHS of the matmul [m, k].
        y: RHS of the matmul [k, n]. Note: full n, not n_per_device.
        tp_size: Number of devices (used to compute output shape).
        lhs_transpose: If True, x is transposed (shape [k, m]).

    Returns:
        List of tp_size arrays, each of shape [m_per_device, n_per_device].
        This simulates what each device would receive.

    Example:
        ```python
        import jax.numpy as jnp

        m, k, n = 1024, 4096, 8192
        tp_size = 4

        x = jnp.ones((m, k), dtype=jnp.float32)
        y = jnp.ones((k, n), dtype=jnp.float32)

        # Get what each device would compute
        outputs = ref_reduce_scatter_matmul_naive(x, y, tp_size)
        # outputs[i] has shape [256, 2048] for device i
        ```
    """
    if lhs_transpose:
        k, m = x.shape
    else:
        m, k = x.shape
    k_from_y, n = y.shape

    assert k == k_from_y, f"k mismatch: {k} vs {k_from_y}"

    m_per_device = m // tp_size
    n_per_device = n // tp_size

    # Step 1: Compute what each device would compute locally
    # Each device has y[:, i*n_per_device:(i+1)*n_per_device]
    partial_results = []
    for i in range(tp_size):
        y_shard = y[:, i * n_per_device:(i + 1) * n_per_device]
        if lhs_transpose:
            partial = jnp.dot(x.T, y_shard)  # [m, n_per_device]
        else:
            partial = jnp.dot(x, y_shard)  # [m, n_per_device]
        partial_results.append(partial)

    # Step 2: Reduce-scatter
    # Sum all partial results and scatter along m dimension
    # Device i gets rows [i*m_per_device:(i+1)*m_per_device] of the sum

    outputs = []
    for i in range(tp_size):
        y_shard = y[:, i * n_per_device:(i + 1) * n_per_device]
        if lhs_transpose:
            full_output = jnp.dot(x.T, y_shard)
        else:
            full_output = jnp.dot(x, y_shard)

        # Scatter: device i gets m_shard i
        output_shard = full_output[i * m_per_device:(i + 1) * m_per_device, :]
        outputs.append(output_shard)

    return outputs


# -----------------------------------------------------------------------------
# Pallas Kernel Implementation
# -----------------------------------------------------------------------------


def _cdiv(x, y):
    """Ceiling division."""
    return (x + y - 1) // y


def _local_barrier(left_neighbor, right_neighbor):
    """Performs a barrier with neighbors on the global barrier semaphore."""
    barrier_sem = pltpu.get_barrier_semaphore()
    for neighbor in [left_neighbor, right_neighbor]:
        pltpu.semaphore_signal(
            barrier_sem,
            inc=1,
            device_id=(neighbor, ),
            device_id_type=pltpu.DeviceIdType.MESH,
        )
    pltpu.semaphore_wait(barrier_sem, 2)


def _reduce_scatter_matmul_kernel(
    # Inputs
    x_hbm_ref,  # [m, k]
    y_hbm_ref,  # [k, n_per_device]
    # Outputs
    o_hbm_ref,  # [m_per_device, n_per_device]
    o_hbm_scratch_ref,  # [num_devices - 1, m_per_device, n_per_device]
    # Scratches
    x_local_copy_sem,
    y_local_copy_sem,
    o_local_copy_sem,
    send_sems,  # [2, num_devices - 1] for left and right
    recv_sems,  # [2, num_devices - 1] for left and right
    x_vmem_scratch_ref,  # [m, bk]
    y_vmem_scratch_ref,  # [bk, n_per_device]
    o_vmem_scratch_ref,  # [2, m_per_device, n_per_device]
    acc_vmem_scratch_ref,  # [m, n_per_device] of jnp.float32
    *,
    axis_name: str,
    bn: int,
    bk: int,
    debug_mode: bool = False,
    lhs_transpose: bool = False,
):
    """Pallas kernel for reduce-scatter matmul.

    This kernel computes matmul in blocks and overlaps the reduce-scatter
    communication with computation using a bidirectional ring topology.

    The computation proceeds in phases:
    1. Compute local partial results for each device's portion of output
    2. Send partial results to neighbors while computing next block
    3. Accumulate received partial results
    4. Final output is the reduced sum scattered across devices

    Args:
        x_hbm_ref: LHS input tensor in HBM.
        y_hbm_ref: RHS input tensor (local shard) in HBM.
        o_hbm_ref: Output tensor in HBM.
        o_hbm_scratch_ref: HBM scratch for remote DMA operations.
        x_local_copy_sem: Semaphore for x HBM-VMEM copy.
        y_local_copy_sem: Semaphore for y HBM-VMEM copy.
        o_local_copy_sem: Semaphore for output HBM-VMEM copy.
        send_sems: Send semaphores for bidirectional ring.
        recv_sems: Receive semaphores for bidirectional ring.
        x_vmem_scratch_ref: VMEM scratch for x.
        y_vmem_scratch_ref: VMEM scratch for y.
        o_vmem_scratch_ref: Double-buffer VMEM scratch for partial outputs.
        acc_vmem_scratch_ref: VMEM scratch for accumulation.
        axis_name: Name of the sharding axis.
        bn: Block size for n dimension.
        bk: Block size for k dimension.
        debug_mode: Whether to print debug messages.
        lhs_transpose: Whether the LHS is transposed.
    """
    # Grid dimensions: (num_devices + 2, grid_n, grid_k)
    # The +2 is for prologue and epilogue phases
    num_devices = pl.num_programs(0) - 2
    grid_n = pl.num_programs(1)
    grid_k = pl.num_programs(2)
    outer_step = pl.program_id(0)
    bn_i = pl.program_id(1)
    bk_i = pl.program_id(2)

    global_step_id = outer_step * grid_n * grid_k + bn_i * grid_k + bk_i
    mxu_total_steps = num_devices * grid_n * grid_k
    gn_by_gk = grid_n * grid_k

    my_id = lax.axis_index(axis_name)
    left_neighbor = lax.rem(my_id + num_devices - 1, jnp.int32(num_devices))
    right_neighbor = lax.rem(my_id + 1, jnp.int32(num_devices))

    # Slot management for double-buffering
    o_hbm_receiving_slot = outer_step
    o_hbm_working_slot = outer_step - 1
    o_vmem_receiving_slot = outer_step % 2
    o_vmem_working_slot = (global_step_id - 1) // gn_by_gk % 2

    # When lhs_transpose=True, x has shape [k, m]; otherwise x has shape [m, k]
    if lhs_transpose:
        _, m = x_hbm_ref.shape
    else:
        m, _ = x_hbm_ref.shape
    _, n_per_device = y_hbm_ref.shape
    m_per_device = m // num_devices
    m_per_device_per_direction = m_per_device // 2

    def debug_print(msg, *args):
        if debug_mode:

            @pl.when(my_id == 0)
            def _debug_print():
                pl.debug_print(msg, *args)

    def _start_or_wait_copy(op, wait: bool = False):
        if wait:
            op.wait()
        else:
            op.start()

    # --- X local copy operations ---
    def _do_x_local_copy(wait: bool = False):
        debug_print(
            "[RSMM debug, wait={}] do x local copy, bk_i={}",
            int(wait),
            bk_i,
        )
        k_slice = pl.ds(bk_i * bk, bk)
        if lhs_transpose:
            x_local_copy_op = pltpu.make_async_copy(
                src_ref=x_hbm_ref.at[k_slice, :],
                dst_ref=x_vmem_scratch_ref.at[k_slice, :],
                sem=x_local_copy_sem,
            )
        else:
            x_local_copy_op = pltpu.make_async_copy(
                src_ref=x_hbm_ref.at[:, k_slice],
                dst_ref=x_vmem_scratch_ref.at[:, k_slice],
                sem=x_local_copy_sem,
            )
        _start_or_wait_copy(x_local_copy_op, wait)

    # --- Y local copy operations ---
    def _do_y_local_copy(wait: bool = False):
        debug_print(
            "[RSMM debug, wait={}] do y local copy, bk_i={}, bn_i={}",
            int(wait),
            bk_i,
            bn_i,
        )
        k_slice = pl.ds(bk_i * bk, bk)
        n_slice = pl.ds(bn_i * bn, bn)
        y_local_copy_op = pltpu.make_async_copy(
            src_ref=y_hbm_ref.at[k_slice, n_slice],
            dst_ref=y_vmem_scratch_ref.at[k_slice, n_slice],
            sem=y_local_copy_sem,
        )
        _start_or_wait_copy(y_local_copy_op, wait)

    # --- Remote copy operations for reduce-scatter ---
    # In reduce-scatter, we send our partial results to neighbors
    # Left half goes left, right half goes right
    def _do_first_left_remote_copy(wait: bool = False):
        debug_print(
            "[RSMM debug, wait={}] do first left remote copy, o_hbm_receiving_slot={}",
            int(wait),
            o_hbm_receiving_slot,
        )
        left_remote_copy_op = pltpu.make_async_remote_copy(
            src_ref=o_vmem_scratch_ref.at[
                o_vmem_working_slot, :m_per_device_per_direction],
            dst_ref=o_hbm_scratch_ref.at[
                o_hbm_receiving_slot, :m_per_device_per_direction],
            send_sem=send_sems.at[0, outer_step],
            recv_sem=recv_sems.at[0, outer_step],
            device_id=(left_neighbor, ),
            device_id_type=pltpu.DeviceIdType.MESH,
        )
        _start_or_wait_copy(left_remote_copy_op, wait)

    def _do_first_right_remote_copy(wait: bool = False):
        debug_print(
            "[RSMM debug, wait={}] do first right remote copy, o_hbm_receiving_slot={}",
            int(wait),
            o_hbm_receiving_slot,
        )
        right_remote_copy_op = pltpu.make_async_remote_copy(
            src_ref=o_vmem_scratch_ref.at[o_vmem_working_slot,
                                          m_per_device_per_direction:],
            dst_ref=o_hbm_scratch_ref.at[o_hbm_receiving_slot,
                                         m_per_device_per_direction:],
            send_sem=send_sems.at[1, outer_step],
            recv_sem=recv_sems.at[1, outer_step],
            device_id=(right_neighbor, ),
            device_id_type=pltpu.DeviceIdType.MESH,
        )
        _start_or_wait_copy(right_remote_copy_op, wait)

    def _do_subsequent_left_remote_copy(wait: bool = False):
        debug_print(
            "[RSMM debug, wait={}] do subsequent left remote copy, "
            "o_hbm_receiving_slot={}, o_hbm_working_slot={}",
            int(wait),
            o_hbm_receiving_slot,
            o_hbm_working_slot,
        )
        left_remote_copy_op = pltpu.make_async_remote_copy(
            src_ref=o_hbm_scratch_ref.at[
                o_hbm_working_slot, :m_per_device_per_direction],
            dst_ref=o_hbm_scratch_ref.at[
                o_hbm_receiving_slot, :m_per_device_per_direction],
            send_sem=send_sems.at[0, outer_step],
            recv_sem=recv_sems.at[0, outer_step],
            device_id=(left_neighbor, ),
            device_id_type=pltpu.DeviceIdType.MESH,
        )
        _start_or_wait_copy(left_remote_copy_op, wait)

    def _do_subsequent_right_remote_copy(wait: bool = False):
        debug_print(
            "[RSMM debug, wait={}] do subsequent right remote copy, "
            "o_hbm_receiving_slot={}, o_hbm_working_slot={}",
            int(wait),
            o_hbm_receiving_slot,
            o_hbm_working_slot,
        )
        right_remote_copy_op = pltpu.make_async_remote_copy(
            src_ref=o_hbm_scratch_ref.at[o_hbm_working_slot,
                                         m_per_device_per_direction:],
            dst_ref=o_hbm_scratch_ref.at[o_hbm_receiving_slot,
                                         m_per_device_per_direction:],
            send_sem=send_sems.at[1, outer_step],
            recv_sem=recv_sems.at[1, outer_step],
            device_id=(right_neighbor, ),
            device_id_type=pltpu.DeviceIdType.MESH,
        )
        _start_or_wait_copy(right_remote_copy_op, wait)

    # --- MXU computation ---
    def _do_mxu():
        """Perform the matmul computation for the current block."""
        working_global_step_id = global_step_id - 1
        working_bk_i = working_global_step_id % grid_k
        working_bn_i = working_global_step_id % gn_by_gk // grid_k

        debug_print(
            "[RSMM debug] do mxu, working_bk_i={}, working_bn_i={}, my_id={}",
            working_bk_i,
            working_bn_i,
            my_id,
        )

        k_slice = pl.ds(working_bk_i * bk, bk)
        n_slice = pl.ds(working_bn_i * bn, bn)

        # Each device computes its own m_slice based on device index (my_id)
        # This ensures device i outputs slice i
        device_m_offset = my_id * m_per_device
        m_slice = pl.ds(device_m_offset, m_per_device)

        if grid_k == 1:
            # No accumulation needed across k blocks
            if lhs_transpose:
                lhs = x_vmem_scratch_ref.at[:, m_slice][...]
                rhs = y_vmem_scratch_ref.at[:, n_slice][...]
                o_vmem_scratch_ref.at[o_vmem_receiving_slot, :,
                                      n_slice][...] = lax.dot_general(
                                          lhs,
                                          rhs,
                                          dimension_numbers=(((0, ), (0, )),
                                                             ((), ())),
                                          preferred_element_type=jnp.float32,
                                      ).astype(x_vmem_scratch_ref.dtype)
            else:
                o_vmem_scratch_ref.at[o_vmem_receiving_slot, :, n_slice][
                    ...] = jnp.dot(
                        x_vmem_scratch_ref.at[m_slice, :][...],
                        y_vmem_scratch_ref.at[:, n_slice][...],
                        preferred_element_type=jnp.float32,
                    ).astype(x_vmem_scratch_ref.dtype)
        else:
            # Accumulate across k blocks
            if lhs_transpose:
                lhs = x_vmem_scratch_ref.at[k_slice, m_slice][...]
                rhs = y_vmem_scratch_ref.at[k_slice, n_slice][...]
                acc_vmem_scratch_ref.at[m_slice,
                                        n_slice][...] += lax.dot_general(
                                            lhs,
                                            rhs,
                                            dimension_numbers=(((0, ), (0, )),
                                                               ((), ())),
                                            preferred_element_type=jnp.float32,
                                        )
            else:
                acc_vmem_scratch_ref.at[m_slice, n_slice][...] += jnp.dot(
                    x_vmem_scratch_ref.at[m_slice, k_slice][...],
                    y_vmem_scratch_ref.at[k_slice, n_slice][...],
                    preferred_element_type=jnp.float32,
                )

            @pl.when(working_bk_i == grid_k - 1)
            def _update():
                debug_print(
                    "[RSMM debug] update, o_vmem_receiving_slot={}",
                    o_vmem_receiving_slot,
                )
                o_vmem_scratch_ref.at[o_vmem_receiving_slot, :,
                                      n_slice][...] = (acc_vmem_scratch_ref.at[
                                          m_slice, n_slice][...].astype(
                                              x_vmem_scratch_ref.dtype))
                acc_vmem_scratch_ref.at[m_slice, n_slice][...] = jnp.zeros(
                    (m_per_device, bn), dtype=jnp.float32)

    # --- Output accumulation from received partial results ---
    def _do_o_accumulate(wait: bool = False):
        """Accumulate received partial results and copy to final output."""
        working_global_step_id = global_step_id - grid_k - 1
        working_bn_i = (working_global_step_id % gn_by_gk) // grid_k
        _ = pl.ds(working_bn_i * bn, bn)  # n_slice for potential future use

        # Calculate which device's partial result we're accumulating
        offset = (global_step_id - 2) // gn_by_gk

        debug_print(
            "[RSMM debug, wait={}] do o accumulate, offset={}",
            int(wait),
            offset,
        )

        # Load received partial from HBM scratch and accumulate
        o_accum_copy_op = pltpu.make_async_copy(
            src_ref=o_hbm_scratch_ref.at[o_hbm_working_slot],
            dst_ref=o_vmem_scratch_ref.at[o_vmem_working_slot],
            sem=o_local_copy_sem,
        )
        _start_or_wait_copy(o_accum_copy_op, wait)

        if not wait:
            return

        # Accumulate the received partial result with local computation
        # This is the reduce part of reduce-scatter
        @pl.when(offset > 0)
        def _accumulate():
            # Add received partial to our running sum
            o_vmem_scratch_ref.at[o_vmem_receiving_slot][...] += (
                o_vmem_scratch_ref.at[o_vmem_working_slot][...])

    def _do_o_local_copy(wait: bool = False):
        """Copy final accumulated result to output."""
        working_global_step_id = global_step_id - grid_k - 1
        working_bn_i = (working_global_step_id % gn_by_gk) // grid_k
        n_slice = pl.ds(working_bn_i * bn, bn)

        debug_print(
            "[RSMM debug, wait={}] do o local copy, working_bn_i={}",
            int(wait),
            working_bn_i,
        )

        o_local_copy_op = pltpu.make_async_copy(
            src_ref=o_vmem_scratch_ref.at[o_vmem_working_slot, :, n_slice],
            dst_ref=o_hbm_ref.at[:, n_slice],
            sem=o_local_copy_sem,
        )
        _start_or_wait_copy(o_local_copy_op, wait)

    ### ------- Kernel start ------- ###
    debug_print(
        "===== starting a grid, outer_step={}, bn_i={}, bk_i={} =====",
        outer_step,
        bn_i,
        bk_i,
    )

    # --- Prologue: Initialize and start first copies ---
    @pl.when(global_step_id == 0)
    @jax.named_scope("_init")
    def _init():
        if grid_k > 1:
            acc_vmem_scratch_ref[...] = jnp.zeros_like(acc_vmem_scratch_ref)
        _local_barrier(left_neighbor, right_neighbor)

    # Start x local copy
    @pl.when(outer_step == 0)
    @jax.named_scope("_start_x_local_copy")
    def _start_x_local_copy():
        _do_x_local_copy(wait=False)

    # Start y local copy
    @pl.when(outer_step == 0)
    @jax.named_scope("_start_y_local_copy")
    def _start_y_local_copy():
        _do_y_local_copy(wait=False)

    # Wait for x local copy
    @pl.when(outer_step == 0)
    @jax.named_scope("_wait_x_local_copy")
    def _wait_x_local_copy():
        _do_x_local_copy(wait=True)

    # Wait for y local copy
    @pl.when(outer_step == 0)
    @jax.named_scope("_wait_y_local_copy")
    def _wait_y_local_copy():
        _do_y_local_copy(wait=True)

    # --- Main computation loop ---
    @pl.when(
        jnp.logical_and(global_step_id >= 1, global_step_id
                        < 1 + mxu_total_steps))
    @jax.named_scope("_mxu")
    def _mxu():
        _do_mxu()

    # --- Start remote copies after first computation completes ---
    cond_start_first_remote_copy = jnp.logical_and(
        global_step_id == gn_by_gk,
        num_devices > 1,
    )

    @pl.when(cond_start_first_remote_copy)
    @jax.named_scope("_start_first_remote_copy")
    def _start_first_remote_copy():
        _do_first_left_remote_copy(wait=False)
        _do_first_right_remote_copy(wait=False)

    cond_start_subsequent_remote_copy = jnp.logical_and(
        jnp.logical_and(outer_step > 1, outer_step < num_devices),
        global_step_id % gn_by_gk == 0,
    )

    @pl.when(cond_start_subsequent_remote_copy)
    @jax.named_scope("_start_subsequent_remote_copy")
    def _start_subsequent_remote_copy():
        _do_subsequent_left_remote_copy(wait=False)
        _do_subsequent_right_remote_copy(wait=False)

    # --- Wait for remote copies ---
    cond_wait_first_remote_copy = jnp.logical_and(
        global_step_id == 2 * gn_by_gk - 1,
        num_devices > 1,
    )

    @pl.when(cond_wait_first_remote_copy)
    @jax.named_scope("_wait_first_remote_copy")
    def _wait_first_remote_copy():
        _do_first_left_remote_copy(wait=True)
        _do_first_right_remote_copy(wait=True)

    cond_wait_subsequent_remote_copy = jnp.logical_and(
        jnp.logical_and(outer_step > 1, outer_step < num_devices),
        global_step_id % gn_by_gk == gn_by_gk - 1,
    )

    @pl.when(cond_wait_subsequent_remote_copy)
    @jax.named_scope("_wait_subsequent_remote_copy")
    def _wait_subsequent_remote_copy():
        _do_subsequent_left_remote_copy(wait=True)
        _do_subsequent_right_remote_copy(wait=True)

    # --- Epilogue: Write final output ---
    def _get_o_local_copy_cond():
        if grid_k == 1:
            return jnp.logical_and(
                global_step_id >= mxu_total_steps + 1,
                global_step_id < mxu_total_steps + gn_by_gk + 1,
            )
        else:
            return jnp.logical_and(
                jnp.logical_and(
                    global_step_id >= mxu_total_steps + grid_k,
                    global_step_id < mxu_total_steps + gn_by_gk + grid_k,
                ),
                global_step_id % grid_k == 0,
            )

    @pl.when(_get_o_local_copy_cond())
    @jax.named_scope("_o_local_copy")
    def _o_local_copy():
        _do_o_local_copy(wait=False)
        _do_o_local_copy(wait=True)

    ### ------- Kernel end ------- ###


def get_vmem_estimate_bytes(
    m: int,
    n: int,
    k: int,  # noqa: ARG001 - kept for API consistency
    bn: int,  # noqa: ARG001 - kept for API consistency
    bk: int,
    acc_bytes: int,
    tp_size: int,
    x_dtype: jnp.dtype,
    y_dtype: jnp.dtype,
    out_dtype: jnp.dtype,
) -> int:
    """Returns the estimated VMEM bytes used by the kernel."""
    n_per_device = n // tp_size
    m_per_device = m // tp_size

    x_vmem_bytes = (m * bk * (dtypes.bit_width(x_dtype) if hasattr(
        dtypes, "bit_width") else dtypes.itemsize_bits(x_dtype)) // 8)
    y_vmem_bytes = (bk * n_per_device * (dtypes.bit_width(y_dtype) if hasattr(
        dtypes, "bit_width") else dtypes.itemsize_bits(y_dtype)) // 8)
    o_vmem_bytes = (
        2 * m_per_device *
        n_per_device * (dtypes.bit_width(out_dtype) if hasattr(
            dtypes, "bit_width") else dtypes.itemsize_bits(out_dtype)) // 8)

    total_bytes = x_vmem_bytes + y_vmem_bytes + o_vmem_bytes + acc_bytes
    return total_bytes


def validate_inputs(x, y, tp_size, lhs_transpose=False):
    """Validates the inputs to the reduce_scatter_matmul kernel."""
    if x.ndim != 2 or y.ndim != 2:
        raise ValueError(
            f"Inputs must be 2D, got shapes {x.shape} and {y.shape}.")
    if x.dtype != y.dtype:
        raise ValueError(
            f"Input dtypes must match, got {x.dtype} and {y.dtype}.")

    if lhs_transpose:
        k, m = x.shape
    else:
        m, k = x.shape
    k_from_y, n = y.shape

    if k != k_from_y:
        raise ValueError(
            f"Incompatible shapes for matmul: contracting dimension mismatch: "
            f"{x.shape} and {y.shape}.")

    if k % 128 != 0:
        raise ValueError(f"k ({k}) must be divisible by 128.")

    if n % 128 != 0:
        raise ValueError(f"n ({n}) must be divisible by 128.")

    m_per_device = m // tp_size
    m_per_device_per_direction = m_per_device // 2
    if m_per_device_per_direction % 8 != 0:
        raise ValueError(
            f"m ({m}) must be divisible by tp_size * 2 * 8 = {tp_size * 2 * 8}."
        )

    if m % (tp_size * 2) != 0:
        raise ValueError(
            f"x.shape[{'1' if lhs_transpose else '0'}] ({m}) must be divisible "
            f"by tp_size * 2 ({tp_size * 2}).")


def reduce_scatter_matmul(
    x: jax.Array,
    y: jax.Array,
    mesh: jax.sharding.AbstractMesh,
    axis_name: str,
    collective_id: int | None = 0,
    bn: int | None = None,
    bk: int | None = None,
    lhs_transpose: bool = False,
) -> jax.Array:
    """Performs matmul followed by reduce-scatter.

    This is useful for tensor parallelism where weights are sharded along the
    output dimension. Each device computes a partial result, then reduce-scatter
    sums the partials and distributes shards to each device.

    Computation pattern:
        Input:  x [m, k] (replicated), y [k, n_per_device] (sharded by n)
        Local:  partial = x @ y_local  [m, n_per_device]
        Output: reduce_scatter(partial, axis=0) -> [m_per_device, n_per_device]

    Args:
        x: LHS of the matmul (replicated across devices).
        y: RHS of the matmul (sharded along columns).
        mesh: JAX mesh for sharding.
        axis_name: Name of the axis to reduce-scatter over.
        collective_id: An integer used for barrier semaphore allocation.
        bn: Block size for n dimension.
        bk: Block size for k dimension.
        lhs_transpose: If True, x is transposed (shape [k, m]).

    Returns:
        matmul(x, y) with output reduce-scattered across devices.
        Shape: [m_per_device, n_per_device] per device.
    """
    tp_size = mesh.shape[axis_name]
    validate_inputs(x, y, tp_size, lhs_transpose)

    if lhs_transpose:
        k, m = x.shape
        x_in_spec = P(None, None)  # x is replicated
    else:
        m, k = x.shape
        x_in_spec = P(None, None)  # x is replicated

    _, n_per_device = y.shape
    m_per_device = m // tp_size

    # Default block sizes
    if bn is None:
        bn = min(n_per_device, 1024)
    if bk is None:
        bk = min(k, 1024)

    # Ensure block sizes are multiples of 128
    bn = ((bn + 127) // 128) * 128
    bk = ((bk + 127) // 128) * 128
    bn = min(bn, n_per_device)
    bk = min(bk, k)

    grid_n = _cdiv(n_per_device, bn)
    grid_k = _cdiv(k, bk)

    acc_shape = (m, n_per_device)
    if grid_k == 1:
        acc_shape = (8, 128)  # Minimal shape when no k accumulation needed

    acc_bytes = (
        acc_shape[0] *
        acc_shape[1] * (dtypes.bit_width(jnp.float32) if hasattr(
            dtypes, "bit_width") else dtypes.itemsize_bits(jnp.float32)) // 8)

    x_vmem_shape = (k, m) if lhs_transpose else (m, k)
    estimated_vmem_bytes = get_vmem_estimate_bytes(
        m,
        n_per_device * tp_size,
        k,
        bn,
        bk,
        acc_bytes,
        tp_size,
        x.dtype,
        y.dtype,
        x.dtype,
    )

    out_shape = [
        jax.ShapeDtypeStruct((m_per_device, n_per_device), x.dtype),  # output
        jax.ShapeDtypeStruct((tp_size - 1, m_per_device, n_per_device),
                             x.dtype),  # o HBM scratch
    ]

    grid_spec = pltpu.PrefetchScalarGridSpec(
        num_scalar_prefetch=0,
        in_specs=[
            pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM),  # x
            pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM),  # y
        ],
        out_specs=[
            pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM),  # output
            pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM),  # o HBM scratch
        ],
        scratch_shapes=(
            pltpu.SemaphoreType.DMA,  # x_local_copy_sem
            pltpu.SemaphoreType.DMA,  # y_local_copy_sem
            pltpu.SemaphoreType.DMA,  # o_local_copy_sem
            pltpu.SemaphoreType.DMA((2, tp_size - 1)),  # send semaphores
            pltpu.SemaphoreType.DMA((2, tp_size - 1)),  # recv semaphores
            pltpu.VMEM(x_vmem_shape, x.dtype),  # x vmem scratch
            pltpu.VMEM((k, n_per_device), y.dtype),  # y vmem scratch
            pltpu.VMEM((2, m_per_device, n_per_device),
                       x.dtype),  # o vmem scratch
            pltpu.VMEM(acc_shape, jnp.float32),  # acc vmem scratch
        ),
        grid=(tp_size + 2, grid_n, grid_k),
    )

    flops = 2 * m * k * n_per_device
    bytes_accessed = x.dtype.itemsize * (m * k + k * n_per_device +
                                         m_per_device * n_per_device)
    cost_estimate = pl.CostEstimate(flops=flops,
                                    bytes_accessed=bytes_accessed,
                                    transcendentals=0)

    @functools.partial(jax.jit, static_argnames=["bn", "bk", "lhs_transpose"])
    def _reduce_scatter_matmul_call(x, y, bn, bk, lhs_transpose):
        return pl.pallas_call(
            functools.partial(
                _reduce_scatter_matmul_kernel,
                bn=bn,
                bk=bk,
                axis_name=axis_name,
                lhs_transpose=lhs_transpose,
            ),
            out_shape=out_shape,
            grid_spec=grid_spec,
            compiler_params=pltpu.CompilerParams(
                collective_id=collective_id,
                vmem_limit_bytes=estimated_vmem_bytes + 8 * 1024 * 1024,
            ),
            cost_estimate=cost_estimate,
            name=get_kernel_name(bn, bk, lhs_transpose),
        )(x, y)[0]

    # Note: check_vma=False is required because we're using pallas_call inside shard_map
    # and the ShapeDtypeStruct outputs don't have vma annotations
    shard_map_kernel = jax.jit(
        jax.shard_map(
            functools.partial(
                _reduce_scatter_matmul_call,
                bn=bn,
                bk=bk,
                lhs_transpose=lhs_transpose,
            ),
            mesh=mesh,
            in_specs=(x_in_spec, P(
                None,
                None)),  # x replicated, y replicated (already local shape)
            out_specs=P(axis_name, None),  # output sharded by m
            check_vma=False,
        ), )

    return shard_map_kernel(x, y)


def get_kernel_name(bn: int, bk: int, lhs_transpose: bool) -> str:
    """Generate a unique kernel name for profiling."""
    return f"reduce_scatter_matmul_kernel_bn_{bn}_bk_{bk}_lhs_transpose_{lhs_transpose}"
