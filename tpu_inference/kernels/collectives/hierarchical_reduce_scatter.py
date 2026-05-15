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
"""Hierarchical Recursive Doubling Reduce-Scatter Implementation."""

import math

import jax
from jax.experimental import pallas as pl
from jax.experimental import shard_map
from jax.experimental.pallas import tpu as pltpu


def _next_multiple_of(val, multiple):
    return ((val + multiple - 1) // multiple) * multiple


# Asynchronously accumulates received data into a running sum using VMEM buffers.
def _accumulate(
    recv_hbm,
    run_hbm,
    out_hbm,
    vmem_recv,
    vmem_run,
    sync_sems,
    vmem_idx=0,
):
    load_recv_sem, load_run_sem, store_run_sem = sync_sems

    load_recv_op = pltpu.make_async_copy(recv_hbm, vmem_recv.at[vmem_idx],
                                         load_recv_sem)
    load_run_op = pltpu.make_async_copy(run_hbm, vmem_run.at[vmem_idx],
                                        load_run_sem)
    load_recv_op.start()
    load_run_op.start()
    load_recv_op.wait()
    load_run_op.wait()

    vmem_run[vmem_idx,
             ...] = vmem_run[vmem_idx, ...] + vmem_recv[vmem_idx, ...]

    if out_hbm is not None:
        store_run_op = pltpu.make_async_copy(vmem_run.at[vmem_idx], out_hbm,
                                             store_run_sem)
    else:
        store_run_op = pltpu.make_async_copy(vmem_run.at[vmem_idx], run_hbm,
                                             store_run_sem)

    store_run_op.start()
    return store_run_op


# Calculates the specific chunk index representing a hypercube node's data.
#
#   This function determines which chunk of the global tensor this device should
#   operate on during a specific step of the hypercube algorithm. It constructs
#   the chunk index by combining bits from:
#   1. The device's own ID for dimensions already processed (prev_dims).
#   2. The loop index for dimensions yet to be processed (future_dims).
#   3. The target dimension's value (dim_val).
def _get_hypercube_chunk_idx(loop_idx, future_dims, prev_dims, my_chip_id,
                             target_dim, dim_val):
    base = 0
    for d in prev_dims:
        bit = (my_chip_id >> d) & 1
        base = base | (bit << d)
    for bit_pos, d in enumerate(future_dims):
        bit = (loop_idx >> bit_pos) & 1
        base = base | (bit << d)
    base = base | (dim_val << target_dim)
    return base


# TODO(dawnhan): Put inter/intra operations into private helper functions.
def hier_rs_kernel(
    input_ref,
    output_ref,
    running_sum_ref,
    recv_buf_ref,
    vmem_recv_ref,
    vmem_run_ref,
    final_copy_sem,
    phase1_sync_sems,
    load_recv_sem,
    load_run_sem,
    store_run_sem,
    *phase2_sync_sems_args,
    num_chips: int,
    num_hypercube_dims: int,
    num_micro_batches: int,
    hidden_size_dim: int,
    final_chunk_size: int,
    full_chunk_size: int,
    mb_size: int,
    axis_name: str = 'x',
):
    sync_sems = (load_recv_sem, load_run_sem, store_run_sem)

    if phase2_sync_sems_args:
        sync_sems_c2c = phase2_sync_sems_args[0]
    else:
        sync_sems_c2c = None

    cur_id = jax.lax.axis_index(axis_name)
    cur_chip_id = cur_id // 2
    cur_chiplet_bit = cur_id % 2
    is_even = cur_chiplet_bit == 0
    twin_id = jax.lax.select(is_even, cur_id + 1, cur_id - 1)

    vmem_pipeline_depth = 4  # Outstanding VMEM accumulation stages

    # Phase 1: Intra-chip Reduce-Scatter (Fast Link)
    with jax.named_scope('phase1_intra_chip'):
        phase1_mb_size = _next_multiple_of(
            hidden_size_dim // num_micro_batches, 128)

        phase1_ops = []
        for micro_batch_idx in range(num_micro_batches):
            mb_start = micro_batch_idx * phase1_mb_size
            mb_size_actual = min(phase1_mb_size, hidden_size_dim - mb_start)
            if mb_size_actual <= 0:
                phase1_ops.append(None)
                continue

            mb_ops = []
            for pair_idx in range(num_chips):
                c_neigh = pair_idx * 2 + (1 - cur_chiplet_bit)

                op = pltpu.make_async_remote_copy(
                    src_ref=input_ref.at[
                        pl.ds(c_neigh * final_chunk_size, final_chunk_size),
                        pl.ds(mb_start, mb_size_actual),
                    ],
                    dst_ref=recv_buf_ref.at[
                        pl.ds(c_neigh * final_chunk_size, final_chunk_size),
                        pl.ds(mb_start, mb_size_actual),
                    ],
                    send_sem=phase1_sync_sems.at[pair_idx, micro_batch_idx],
                    recv_sem=phase1_sync_sems.at[pair_idx, micro_batch_idx],
                    device_id=twin_id,
                    device_id_type=pltpu.DeviceIdType.LOGICAL,
                )
                op.start()
                mb_ops.append(op)
            phase1_ops.append(mb_ops)

        # Wait and accumulate sequentially
        for micro_batch_idx in range(num_micro_batches):
            mb_start = micro_batch_idx * phase1_mb_size
            mb_size_actual = min(phase1_mb_size, hidden_size_dim - mb_start)
            if mb_size_actual <= 0:
                break

            store_ops = []
            for pair_idx in range(num_chips):
                c_me = pair_idx * 2 + cur_chiplet_bit

                phase1_ops[micro_batch_idx][pair_idx].wait()

                s_op = _accumulate(
                    recv_hbm=recv_buf_ref.at[
                        pl.ds(c_me * final_chunk_size, final_chunk_size),
                        pl.ds(mb_start, mb_size_actual),
                    ],
                    run_hbm=input_ref.at[
                        pl.ds(c_me * final_chunk_size, final_chunk_size),
                        pl.ds(mb_start, mb_size_actual),
                    ],
                    out_hbm=running_sum_ref.at[
                        pl.ds(c_me * final_chunk_size, final_chunk_size),
                        pl.ds(mb_start, mb_size_actual),
                    ],
                    vmem_recv=vmem_recv_ref.at[:,
                                               pl.ds(0, final_chunk_size),
                                               pl.ds(0, mb_size_actual)],
                    vmem_run=vmem_run_ref.at[:,
                                             pl.ds(0, final_chunk_size),
                                             pl.ds(0, mb_size_actual)],
                    sync_sems=sync_sems,
                    vmem_idx=pair_idx % vmem_pipeline_depth,
                )
                store_ops.append(s_op)
            for s_op in store_ops:
                s_op.wait()

    # Phase 2: Inter-chip Reduce-Scatter (3D Concurrent with Micro-batching)
    # Perform reduce-scatter across chips using a hypercube algorithm.
    # In each step, chips exchange data along one dimension of the hypercube.
    # By the end of all steps, each chip holds a fully reduced chunk of the data.
    # Example for a 4-chip setup (2 hypercube dimensions):
    # Chip IDs in binary: 00, 01, 10, 11
    # Step 0 (dim=0):
    #   - Chip 00 exchanges with 01 (differs in bit 0)
    #   - Chip 10 exchanges with 11
    # Step 1 (dim=1):
    #   - Chip 00 exchanges with 10 (differs in bit 1)
    #   - Chip 01 exchanges with 11
    for phase_step in range(num_hypercube_dims):
        with jax.named_scope(f'phase2_step_{phase_step}'):
            all_mb_ops = []
            for micro_batch_idx in range(num_micro_batches):
                mb_ops = []

                num_ops_in_step = 2**(num_hypercube_dims - 1 - phase_step)
                for op_idx in range(num_ops_in_step):

                    for hypercube_dim_idx in range(num_hypercube_dims):
                        # Rotate the dimension we operate on based on the phase step.
                        # This helps in structuring the recursive doubling.
                        dim = (hypercube_dim_idx +
                               phase_step) % num_hypercube_dims
                        # Find the neighbor chip that differs only in the 'dim' bit.
                        neighbor_chip_id = cur_chip_id ^ (1 << dim)
                        my_dim_bit = (cur_chip_id >> dim) & 1
                        neigh_dim_bit = 1 - my_dim_bit

                        # Keep track of resolved (prev) and unresolved (future) dimensions
                        # to calculate which chunk of data to operate on.
                        prev_dims = [
                            (hypercube_dim_idx + j) % num_hypercube_dims
                            for j in range(phase_step)
                        ]
                        future_dims = [
                            (hypercube_dim_idx + j) % num_hypercube_dims
                            for j in range(phase_step + 1, num_hypercube_dims)
                        ]

                        chunk_start = hypercube_dim_idx * full_chunk_size
                        chunk_end = min(chunk_start + full_chunk_size,
                                        hidden_size_dim)

                        mb_start_idx = min(
                            chunk_start + (micro_batch_idx * mb_size),
                            chunk_end)
                        mb_end_idx = min(mb_start_idx + mb_size, chunk_end)
                        k_size = mb_end_idx - mb_start_idx

                        my_base_chunk_idx = _get_hypercube_chunk_idx(
                            op_idx,
                            future_dims,
                            prev_dims,
                            cur_chip_id,
                            target_dim=dim,
                            dim_val=my_dim_bit,
                        )
                        neighbor_base_chunk_idx = _get_hypercube_chunk_idx(
                            op_idx,
                            future_dims,
                            prev_dims,
                            cur_chip_id,
                            target_dim=dim,
                            dim_val=neigh_dim_bit,
                        )

                        my_chunk_idx = my_base_chunk_idx * 2 + cur_chiplet_bit
                        neighbor_chunk_idx = neighbor_base_chunk_idx * 2 + cur_chiplet_bit

                        if k_size > 0:
                            op = pltpu.make_async_remote_copy(
                                src_ref=running_sum_ref.at[
                                    pl.ds(
                                        neighbor_chunk_idx * final_chunk_size,
                                        final_chunk_size,
                                    ),
                                    pl.ds(mb_start_idx, k_size),
                                ],
                                dst_ref=recv_buf_ref.at[
                                    pl.ds(
                                        neighbor_chunk_idx * final_chunk_size,
                                        final_chunk_size,
                                    ),
                                    pl.ds(mb_start_idx, k_size),
                                ],
                                send_sem=sync_sems_c2c.at[  # pytype: disable=attribute-error
                                    phase_step, micro_batch_idx,
                                    hypercube_dim_idx, op_idx],
                                recv_sem=sync_sems_c2c.at[  # pytype: disable=attribute-error
                                    phase_step, micro_batch_idx,
                                    hypercube_dim_idx, op_idx],
                                device_id=neighbor_chip_id * 2 +
                                cur_chiplet_bit,
                                device_id_type=pltpu.DeviceIdType.LOGICAL,
                            )
                            op.start()
                            mb_ops.append(
                                (op, my_chunk_idx, mb_start_idx, k_size))

                all_mb_ops.append(mb_ops)

            for micro_batch_idx in range(num_micro_batches):
                store_ops = []
                for op_id, (op, my_chunk_idx, c_start,
                            k_sz) in enumerate(all_mb_ops[micro_batch_idx]):
                    op.wait()
                    s_op = _accumulate(
                        recv_hbm=recv_buf_ref.at[
                            pl.ds(my_chunk_idx *
                                  final_chunk_size, final_chunk_size),
                            pl.ds(c_start, k_sz),
                        ],
                        run_hbm=running_sum_ref.at[
                            pl.ds(my_chunk_idx *
                                  final_chunk_size, final_chunk_size),
                            pl.ds(c_start, k_sz),
                        ],
                        out_hbm=None,
                        vmem_recv=vmem_recv_ref.at[:,
                                                   pl.ds(0, final_chunk_size),
                                                   pl.ds(0, k_sz)],
                        vmem_run=vmem_run_ref.at[:,
                                                 pl.ds(0, final_chunk_size),
                                                 pl.ds(0, k_sz)],
                        sync_sems=sync_sems,
                        vmem_idx=op_id % vmem_pipeline_depth,
                    )
                    store_ops.append(s_op)
                for s_op in store_ops:
                    s_op.wait()

    # TODO(dawnhan): Update the algorithm to avoid the final copy.
    # Scatter Phase: Each device copies its designated shard of the fully
    # reduced data from the running sum to the final output buffer, determined
    # by its unique logical ID.
    final_copy_op = pltpu.make_async_copy(
        src_ref=running_sum_ref.at[
            pl.ds(cur_id * final_chunk_size, final_chunk_size), :],
        dst_ref=output_ref,
        sem=final_copy_sem,
    )
    final_copy_op.start()
    final_copy_op.wait()


def hierarchical_reduce_scatter_local(
    local_x: jax.Array,
    num_devices: int,
    num_micro_batches: int = 2,
    axis_name: str | tuple[str, ...] = 'x',
) -> jax.Array:
    num_chips = num_devices // 2
    num_hypercube_dims = int(math.log2(num_chips)) if num_chips > 0 else 0

    seq_len_dim = local_x.shape[0]
    hidden_size_dim = local_x.shape[1]

    chunk_size_raw = (hidden_size_dim // num_hypercube_dims
                      if num_hypercube_dims > 0 else hidden_size_dim)
    full_chunk_size = _next_multiple_of(chunk_size_raw, 128)
    mb_size_raw = full_chunk_size // num_micro_batches
    mb_size = _next_multiple_of(mb_size_raw, 128)
    final_chunk_size = seq_len_dim // num_devices

    out_shape = jax.ShapeDtypeStruct((final_chunk_size, hidden_size_dim),
                                     local_x.dtype)
    running_sum_shape = jax.ShapeDtypeStruct((seq_len_dim, hidden_size_dim),
                                             local_x.dtype)
    recv_buf_shape = jax.ShapeDtypeStruct((seq_len_dim, hidden_size_dim),
                                          local_x.dtype)

    # Size VMEM buffers to hold only one micro-batch at a time to avoid OOM,
    # aligned to 128 bytes for efficient TPU DMA transfers.
    max_vmem_mb_size = _next_multiple_of(hidden_size_dim // num_micro_batches,
                                         128)

    scratch_shapes = [
        pltpu.VMEM((4, final_chunk_size, max_vmem_mb_size), local_x.dtype),
        pltpu.VMEM((4, final_chunk_size, max_vmem_mb_size), local_x.dtype),
        pltpu.SemaphoreType.DMA,
        pltpu.SemaphoreType.DMA((num_chips, num_micro_batches)),
        pltpu.SemaphoreType.DMA,
        pltpu.SemaphoreType.DMA,
        pltpu.SemaphoreType.DMA,
    ]

    if num_hypercube_dims > 0:
        scratch_shapes.append(
            pltpu.SemaphoreType.DMA((
                num_hypercube_dims,
                num_micro_batches,
                num_hypercube_dims,
                2**(num_hypercube_dims - 1),
            )))

    grid_spec = pltpu.PrefetchScalarGridSpec(
        num_scalar_prefetch=0,
        in_specs=[pl.BlockSpec(memory_space=pl.ANY)],
        out_specs=(
            pl.BlockSpec(memory_space=pl.ANY),
            pl.BlockSpec(memory_space=pl.ANY),
            pl.BlockSpec(memory_space=pl.ANY),
        ),
        scratch_shapes=tuple(scratch_shapes),
        grid=(1, ),
    )

    kernel_fn = jax.tree_util.Partial(
        hier_rs_kernel,
        num_chips=num_chips,
        num_hypercube_dims=num_hypercube_dims,
        num_micro_batches=num_micro_batches,
        hidden_size_dim=hidden_size_dim,
        final_chunk_size=final_chunk_size,
        full_chunk_size=full_chunk_size,
        mb_size=mb_size,
        axis_name=axis_name,
    )

    hier_rs = pl.pallas_call(
        kernel_fn,
        out_shape=(out_shape, running_sum_shape, recv_buf_shape),
        grid_spec=grid_spec,
    )

    final_out, _, _ = hier_rs(local_x)
    return final_out


def hierarchical_reduce_scatter(
    x: jax.Array,
    *,
    mesh: jax.sharding.Mesh,
    in_specs: jax.sharding.PartitionSpec = jax.sharding.PartitionSpec(
        'x', None),
    num_micro_batches: int = 2,
) -> jax.Array:
    num_devices = mesh.devices.size

    def inner(local_x):
        return hierarchical_reduce_scatter_local(local_x, num_devices,
                                                 num_micro_batches)

    return shard_map.shard_map(
        inner,
        mesh=mesh,
        in_specs=in_specs,
        out_specs=in_specs,
        check_rep=False,
    )(x)
