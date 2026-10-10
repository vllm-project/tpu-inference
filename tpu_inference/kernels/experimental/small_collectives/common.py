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
"""Shared ring topology, barrier semaphore, and VMEM scratch utilities for small collectives.
"""

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

LANES = 128
MESH = pl.DeviceIdType.MESH


def peer(j, axis, dict_ids=True):
    """Map a linear core index `j` into a device_id mesh coordinate.

    If `axis` is a single string, returns `{axis: jnp.int32(j)}`.
    If `axis` is a tuple/list of strings, maps linear core index `j` across the mesh
    axes in row-major order.
    """
    if isinstance(axis, str):
        return {axis: jnp.int32(j)}
    if len(axis) == 1:
        return {axis[0]: jnp.int32(j)}

    sizes = [lax.axis_size(a) for a in axis]
    strides = []
    stride = 1
    for s in reversed(sizes[1:]):
        stride *= s
        strides.insert(0, stride)
    strides.append(1)

    coords = {
        a: jnp.int32((j // strd) % s)
        for a, s, strd in zip(axis, sizes, strides)
    }
    return coords


def barrier(num_cores, axis, dict_ids=True):
    """Every core signals every core once, itself included, and waits for all.

    One barrier per call is enough. A core finishes call k only after every
    peer's data of call k has landed, and a peer sends that data only after
    passing its own barrier of call k, so no core can signal call k + 1 while
    another still waits at the barrier of call k.
    """
    sem = pltpu.get_barrier_semaphore()
    for j in range(num_cores):
        pl.semaphore_signal(sem,
                            1,
                            device_id=peer(j, axis, dict_ids),
                            device_id_type=MESH)
    pl.semaphore_wait(sem, num_cores)


def barrier_peers(me, xor_masks, axis, dict_ids=True):
    """Every core signals its direct XOR communication peers ``me ^ m`` once
    and waits for all of them. Sufficient whenever the set of direct DMA peers
    is symmetric (`b == a ^ m <=> a == b ^ m`).
    """
    sem = pltpu.get_barrier_semaphore()
    for m in xor_masks:
        pl.semaphore_signal(
            sem,
            1,
            device_id=peer(me ^ jnp.int32(m), axis, dict_ids),
            device_id_type=MESH,
        )
    pl.semaphore_wait(sem, len(xor_masks))


def wait_bytes_of(ref_slice, sem):
    """Block until ``sem`` has counted the bytes of ``ref_slice``."""
    pltpu.make_async_copy(ref_slice, ref_slice, sem).wait()


def push(src, dst, send_sem, recv_sem, j, axis, dict_ids=True):
    pltpu.make_async_remote_copy(
        src_ref=src,
        dst_ref=dst,
        send_sem=send_sem,
        recv_sem=recv_sem,
        device_id=peer(j, axis, dict_ids),
        device_id_type=MESH,
    ).start()


def split_rank(refs, rank_operand, axis):
    """This core's index and the remaining refs."""
    if rank_operand:
        rank_ref, *refs = refs
        return rank_ref[0], refs
    if isinstance(axis, str):
        return lax.axis_index(axis), refs
    if len(axis) == 1:
        return lax.axis_index(axis[0]), refs
    linear_idx = jnp.int32(0)
    for a in axis:
        linear_idx = linear_idx * lax.axis_size(a) + lax.axis_index(a)
    return linear_idx, refs


def sum_slots(slots, n):
    """Float32 sum over the leading axis of a VMEM ref, rounded once."""
    acc = slots[0].astype(jnp.float32)
    for j in range(1, n):
        acc = acc + slots[j].astype(jnp.float32)
    return acc


def call(kernel, x, out_rows, rank, scratch, collective_id, name):
    rank_specs = [pl.BlockSpec(
        memory_space=pltpu.SMEM)] if rank is not None else []
    args = (() if rank is None else (rank.astype(jnp.int32), )) + (x, )
    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((out_rows, x.shape[1]), x.dtype),
        in_specs=[*rank_specs, pl.BlockSpec()],
        scratch_shapes=scratch,
        compiler_params=pltpu.CompilerParams(collective_id=collective_id),
        name=name,
    )(*args)
