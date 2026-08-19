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
"""In-kernel PCP KV ring for the batched RPA CACHE_ONLY cache phase.

The schedule encodes ring rounds into k_idx (block * cp_group_size + round);
only round 0 fetches this rank's block from HBM, later rounds emit zero-size
fetches and receive the block from the previous rank over ICI instead. Every
rank runs rank 0's block count (the longest shard under page-interleaving) so
all devices stay in lock-step, and masking uses the source rank's shard
length, so one online softmax accumulates the full cache across ranks with no
LSE merge.

Because the batched kernel serializes consecutive k blocks of a task across
the batch lanes of a step (one online-softmax chain), each lane is one ring
micro-step. Each lane's round decides whether the resident block is the
lane's own HBM fetch (round 0, copied into a private ring scratch for
sending) or a received block, which is staged back into the pipeline's lane
buffer so the compute path is unchanged. Ring slots are therefore only ever
touched by DMAs — never by in-flight vector reads — so freeing a slot for
the previous rank only requires waiting the previous micro-step's send.

Protocol invariants (see PcpRing.stage):
- Slot parity strictly alternates across valid lanes: rounds increment
  within a block, blocks and tasks always end on round cp_group_size - 1,
  and cp_group_size is even.
- Masked lanes only ever appear at the schedule tail (final flush) and skip
  the ring entirely, identically on every device, keeping credits balanced.
- The credit release at micro-step u and the credit wait at micro-step u use
  the same predicate on identical schedules, so the sync semaphore is
  balanced and ends at zero.

Requires an even cp_group_size, AttentionScope.CACHE_ONLY, and the
HEAD_ALONG_SUBLANE layout (validated in configs.RpaConfigs.validate_inputs).
"""

import dataclasses

import jax
import jax.experimental.pallas as pl
import jax.experimental.pallas.tpu as pltpu
import jax.numpy as jnp
from jax import lax

from tpu_inference.kernels.experimental.batched_rpa import utils

# Configuration and schedule-side helpers (static; no ring state involved).


def compiler_params(cfgs) -> dict:
    """Extra pallas_call compiler params required by the ring."""
    if not cfgs.ring_enabled:
        return {}
    # The ring's startup barrier needs a barrier semaphore.
    return {"collective_id": 0}


def decode_k(k_idx, cfgs):
    """Decode a ring-encoded schedule k_idx into (block, is_round0)."""
    ring_block = k_idx // cfgs.serve.cp_group_size
    ring_is_round0 = k_idx % cfgs.serve.cp_group_size == 0
    return ring_block, ring_is_round0


def gate_fetch(dma_sz, ring_is_round0):
    """Only round 0 fetches this rank's block from HBM; later rounds receive
    the block from the previous rank over the ring."""
    return jnp.where(ring_is_round0, dma_sz, 0)


def block_range(cache_len, cfgs):
    """(start_k_idx, end_k_idx) of the ring-encoded block loop.

    Every rank must run the same number of steps, so size the block loop by
    rank 0's shard (the longest under page-interleaving) and run
    cp_group_size rounds per block; short ranks' tails are masked in the
    kernel.
    """
    rank0_cache_len = utils.cp_local_cache_len(cache_len,
                                               cfgs.serve.cp_group_size, 0,
                                               cfgs.serve.page_size)
    num_ring_blocks = pl.cdiv(rank0_cache_len, cfgs.bkv_sz)
    return 0, num_ring_blocks * cfgs.serve.cp_group_size


# Kernel-side allocations.


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class RingRefs:
    """The ring's scratch refs, passed through the pipeline as one pytree."""

    kv: jax.Ref  # [2, *kv_vmem_shape[1:]]: one lane block per slot
    dma_sems: jax.Ref  # [2] send/recv
    sync_sem: jax.Ref  # credit: "the slot your next send targets is free"
    local_sem: jax.Ref  # [1] local staging copies


def scratch_refs(cfgs) -> tuple:
    """RingRefs scratch shapes to append to the kernel's scoped scratches."""
    if not cfgs.ring_enabled:
        return ()
    return (RingRefs(
        kv=pltpu.VMEM(
            (2, *cfgs.kv_vmem_shape[1:]),
            dtype=cfgs.serve.dtype_kv,
        ),
        dma_sems=pltpu.SemaphoreType.DMA((2, )),
        sync_sem=pltpu.SemaphoreType.REGULAR,
        local_sem=pltpu.SemaphoreType.DMA((1, )),
    ), )


def zero_init(ring_refs: RingRefs):
    """Zero the ring slots, like the pipeline KV window: whatever a previous
    program left at this address must not be interpretable as plausible KV
    data."""
    num_lanes = pltpu.get_tpu_info().num_lanes
    ring_kv_flat = ring_refs.kv.bitcast(jnp.uint32).reshape(-1, num_lanes)
    ring_kv_flat[...] = jnp.zeros_like(ring_kv_flat)


# Kernel-body side.


class PcpRing:
    """Per-step ring state and the per-lane rotation protocol."""

    def __init__(self, cfgs, refs: RingRefs, *, step, chunk_start):
        self.cfgs = cfgs
        self.refs = refs
        self.size = cfgs.serve.cp_group_size
        self.my_id = lax.axis_index(cfgs.serve.pcp_ring_axis_name)
        self.next_id = self._device_id(lax.rem(self.my_id + 1, self.size))
        self.prev_id = self._device_id(
            lax.rem(self.my_id + self.size - 1, self.size))
        # Schedule chunks are 128-step aligned (max_steps_ub is a multiple of
        # num_lanes), so prefix_steps is always 0 and this is the global step
        # across chunks. The ring semaphores are scoped outside the chunk
        # loop, so the credit protocol carries across chunk boundaries.
        self.step_global = chunk_start + step
        self._rounds = []
        self._valids = []

    def _device_id(self, rank):
        if self.cfgs.serve.pcp_ring_mesh_axis_names is None:
            return (rank, )
        return tuple(
            rank if name == self.cfgs.serve.pcp_ring_axis_name else lax.
            axis_index(name)
            for name in self.cfgs.serve.pcp_ring_mesh_axis_names)

    def lane(self, k_idx, is_valid):
        """Decode one lane's ring-encoded k_idx, recording its micro-step.

        Returns (block k_idx, round). Must be called once per lane, in lane
        order, before stage().
        """
        round_b = lax.rem(k_idx, self.size)
        self._rounds.append(round_b)
        self._valids.append(is_valid)
        return k_idx // self.size, round_b

    def src_rank(self, round_b):
        """Rank the resident block originated from after round_b hops; its
        shard length is what masking must use instead of our own."""
        return lax.rem(self.my_id + self.size - round_b, self.size)

    def stage(self, kv_in_vref):
        """Run the per-lane rotation protocol for this step.

        For each lane (= ring micro-step), in order: wait the previous
        micro-step's send, release the freed slot's credit to the previous
        rank, wait for this round's incoming block, fill slot 0 on round 0,
        wait for the next rank's credit, send, and stage the received block
        into the pipeline's lane buffer. The ordering is load-bearing; see
        the per-op comments.
        """
        assert len(self._rounds) == self.cfgs.batch_size

        # Startup rendezvous with both neighbors before any remote traffic:
        # on a cold first execution a fast device's RDMA could otherwise land
        # on a neighbor that is still loading the program and lose part of
        # the block to its startup initialization.
        @pl.when(self.step_global == 0)
        def ring_startup_barrier():
            barrier_sem = pltpu.get_barrier_semaphore()
            pl.semaphore_signal(
                barrier_sem,
                1,
                device_id=self.next_id,
                device_id_type=pl.DeviceIdType.MESH,
            )
            pl.semaphore_signal(
                barrier_sem,
                1,
                device_id=self.prev_id,
                device_id_type=pl.DeviceIdType.MESH,
            )
            pl.semaphore_wait(barrier_sem, 2)

        for b_idx in range(self.cfgs.batch_size):
            valid_b = self._valids[b_idx]
            round_b = self._rounds[b_idx]
            slot_b = lax.rem(round_b, 2)
            sends_b = jnp.logical_and(valid_b, round_b != self.size - 1)
            receives_b = jnp.logical_and(valid_b, round_b > 0)
            # At micro-step 0 of the whole schedule every slot is known free,
            # so the credit exchange is skipped on both sides.
            if b_idx == 0:
                not_first_micro = self.step_global > 0
            else:
                not_first_micro = True

            remote_op = pltpu.make_async_remote_copy(
                src_ref=self.refs.kv.at[slot_b],
                dst_ref=self.refs.kv.at[1 - slot_b],
                send_sem=self.refs.dma_sems.at[0],
                recv_sem=self.refs.dma_sems.at[1],
                device_id=self.next_id,
                device_id_type=pl.DeviceIdType.MESH,
            )

            # The previous micro-step's send is the last reader of the slot
            # the credit below frees (its other reader, the staging copy, was
            # waited inline there). All slots have identical shapes, so this
            # descriptor waits the matching byte count.
            @pl.when(receives_b)
            def wait_prev_send():
                remote_op.wait_send()

            # Credit for the send the previous rank issues at this micro-step
            # (it lands in the slot our previous micro-step just vacated).
            @pl.when(jnp.logical_and(sends_b, not_first_micro))
            def release_prev_slot():
                pl.semaphore_signal(
                    self.refs.sync_sem,
                    1,
                    device_id=self.prev_id,
                    device_id_type=pl.DeviceIdType.MESH,
                )

            @pl.when(receives_b)
            def wait_ring_recv():
                remote_op.wait_recv()

            @pl.when(jnp.logical_and(valid_b, round_b == 0))
            def fill_ring_slot0():
                cp = pltpu.make_async_copy(kv_in_vref.at[b_idx],
                                           self.refs.kv.at[0],
                                           self.refs.local_sem.at[0])
                cp.start()
                cp.wait()

            @pl.when(jnp.logical_and(sends_b, not_first_micro))
            def wait_ring_sync():
                pl.semaphore_wait(self.refs.sync_sem, 1)

            @pl.when(sends_b)
            def start_rotate():
                remote_op.start()

            # Stage the received block into the pipeline's lane buffer (its
            # HBM fetch for rounds > 0 is zero-size) so compute reads every
            # lane the same way.
            @pl.when(receives_b)
            def stage_received_block():
                cp = pltpu.make_async_copy(self.refs.kv.at[slot_b],
                                           kv_in_vref.at[b_idx],
                                           self.refs.local_sem.at[0])
                cp.start()
                cp.wait()
