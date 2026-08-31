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

The schedule encodes ring rounds into k_idx (block * cp group_size + round);
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
  within a block, blocks and tasks always end on round group_size - 1, and
  the group size is even.
- Masked lanes only ever appear at the schedule tail (final flush) and skip
  the ring entirely, identically on every device, keeping credits balanced.
- The credit release at micro-step u and the credit wait at micro-step u use
  the same predicate on identical schedules, so the sync semaphore is
  balanced and ends at zero.

Length bookkeeping: the schedule sizes this rank's round-0 fetches by its
own shard length (kv_cache_lens, localized by the wrapper) but sizes the block
loop by rank 0's shard, and the kernel masks each resident block by the shard
length of the rank it came from. Both of the latter derive from the GLOBAL
cache length, which the wrapper passes alongside: to the schedule kernel as an
extra scalar (RingMetadataComputer) and to the attention kernel in place of
kv_cache_lens (RingStepMetadataComputer localizes it per lane).

The two computer classes mirror the kernel's plug-in points: the schedule's
RingMetadataComputer (a schedule_cp.CPMetadataComputer) emits ring-encoded
k indices and gates HBM fetches to round 0, and the attention kernel's
RingStepMetadataComputer (a kernel.StepMetadataComputer) decodes them, masks
by the source rank's shard, and rotates the blocks. Requires an even cp
group_size, AttentionScope.CACHE_ONLY, and the HEAD_ALONG_SUBLANE layout
(validated in configs.RpaConfigs.validate_inputs).
"""

import dataclasses

import jax
import jax.experimental.pallas as pl
import jax.experimental.pallas.tpu as pltpu
import jax.numpy as jnp
from jax import lax

from tpu_inference.kernels.experimental.batched_rpa import (configs, kernel,
                                                            schedule_cp, utils)

# Schedule side.


def decode_k(k_idx, cfgs):
    """Decode a ring-encoded schedule k_idx into (block, is_round0)."""
    ring_block = k_idx // cfgs.serve.cp.group_size
    ring_is_round0 = k_idx % cfgs.serve.cp.group_size == 0
    return ring_block, ring_is_round0


def block_range(global_cache_len, cfgs):
    """(start_k_idx, end_k_idx) of the ring-encoded block loop.

    Every rank must run the same number of steps, so size the block loop by
    rank 0's shard (the longest under page-interleaving) and run group_size
    rounds per block; short ranks' tails are masked in the kernel.
    """
    rank0_cache_len = utils.cp_local_cache_len(global_cache_len,
                                               cfgs.serve.cp.group_size, 0,
                                               cfgs.serve.page_size)
    num_ring_blocks = pl.cdiv(rank0_cache_len, cfgs.bkv_sz)
    return 0, num_ring_blocks * cfgs.serve.cp.group_size


class RingMetadataComputer(schedule_cp.CPMetadataComputer):
    """CP schedule with ring-encoded k blocks.

    extra_refs = (cp_rank, new_kv_starts, global_kv_cache_lens): the base CP
    computer keeps rank-owned DMA sizing from the wrapper-localized
    kv_cache_lens; the ring only changes which k indices a q block iterates
    (rank 0's block count times group_size rounds) and gates the HBM fetch to
    round 0.
    """

    @property
    def global_kv_cache_lens_ref(self):
        return self.extra_refs[2]

    def k_idx_range(self, *, s_idx, **kwargs):
        del kwargs
        return block_range(self.global_kv_cache_lens_ref[s_idx], self.cfgs)

    def decode_k(self, k_idx):
        # k_idx is ring-encoded; only round 0 fetches this rank's block from
        # HBM, later rounds receive it from the previous rank over the ring.
        return decode_k(k_idx, self.cfgs)


# Kernel side.


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class RingRefs:
    """The ring's scratch refs, passed through the pipeline as one pytree."""

    kv: jax.Ref  # [2, *kv_vmem_shape[1:]]: one lane block per slot
    dma_sems: jax.Ref  # [2] send/recv
    sync_sem: jax.Ref  # credit: "the slot your next send targets is free"
    local_sem: jax.Ref  # [1] local staging copies


class PcpRing:
    """Per-step ring state and the per-lane rotation protocol."""

    def __init__(self, cfgs, refs: RingRefs, *, step, chunk_start):
        self.cfgs = cfgs
        self.refs = refs
        self.size = cfgs.serve.cp.group_size
        self.my_id = lax.axis_index(cfgs.serve.cp.ring_axis_name)
        self.next_id = self._device_id(lax.rem(self.my_id + 1, self.size))
        self.prev_id = self._device_id(
            lax.rem(self.my_id + self.size - 1, self.size))
        # Schedule chunks are 128-step aligned (max_steps_ub is a multiple of
        # num_lanes), so prefix_steps is always 0 and this is the global step
        # across chunks. The ring semaphores are scoped outside the chunk
        # loop, so the credit protocol carries across chunk boundaries.
        self.step_global = chunk_start + step

    def _device_id(self, rank):
        if self.cfgs.serve.cp.ring_mesh_axis_names is None:
            return (rank, )
        return tuple(
            rank if name ==
            self.cfgs.serve.cp.ring_axis_name else lax.axis_index(name)
            for name in self.cfgs.serve.cp.ring_mesh_axis_names)

    def src_rank(self, round_b):
        """Rank the resident block originated from after round_b hops; its
        shard length is what masking must use instead of our own."""
        return lax.rem(self.my_id + self.size - round_b, self.size)

    def src_cache_len(self, global_cache_len, round_b):
        """Shard length of the rank the resident block came from."""
        return utils.cp_local_cache_len(global_cache_len, self.size,
                                        self.src_rank(round_b),
                                        self.cfgs.serve.page_size)

    def stage(self, kv_in_vref, rounds, valids):
        """Run the per-lane rotation protocol for this step.

        rounds/valids hold each lane's ring round and validity, in lane
        order. For each lane (= ring micro-step), in order: wait the previous
        micro-step's send, release the freed slot's credit to the previous
        rank, wait for this round's incoming block, fill slot 0 on round 0,
        wait for the next rank's credit, send, and stage the received block
        into the pipeline's lane buffer. The ordering is load-bearing; see
        the per-op comments.
        """
        assert len(rounds) == len(valids) == self.cfgs.batch_size

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
            valid_b = valids[b_idx]
            round_b = rounds[b_idx]
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


def ring_fetch_step_metadata(
    step: jax.Array,
    schedule_ref,
    cu_q_lens_ref: jax.Ref,
    q_offsets_ref: jax.Ref,
    kv_cache_lens_ref: jax.Ref,
    kv_new_lens_ref: jax.Ref,
    *,
    cfgs: configs.RpaConfigs,
    pcp_ring: PcpRing,
):
    """kernel.fetch_step_metadata for ring-encoded schedules.

    Decodes each lane's (block, round), masks the resident block by its
    SOURCE rank's shard length (kv_cache_lens_ref holds the global cache
    length under the ring), and returns the per-lane rounds/valids for
    PcpRing.stage. CACHE_ONLY only: kv_new_lens is all zeros.
    """
    causal_offset_list = []
    bkv_sz_frm_cache_list = []
    new_kv_len_start_list = []
    q_sz_list = []
    is_valid_list = []
    local_k_end_list = []
    rounds = []
    valids = []

    for b_idx in range(cfgs.batch_size):
        s_idx = schedule_ref.s_idx[step, b_idx]
        is_valid = s_idx != -1
        q_idx = schedule_ref.q_idx[step, b_idx]
        # k_idx is ring-encoded (see decode_k); split it before any block
        # arithmetic.
        k_idx = schedule_ref.k_idx[step, b_idx]
        ring_round = lax.rem(k_idx, pcp_ring.size)
        k_idx = k_idx // pcp_ring.size
        rounds.append(ring_round)
        valids.append(is_valid)

        k_id = jnp.where(is_valid, k_idx * cfgs.bkv_sz, 0)
        # The resident block came from another rank; mask it by that rank's
        # shard length.
        kv_cache_len_val = jnp.where(
            is_valid,
            pcp_ring.src_cache_len(kv_cache_lens_ref[s_idx], ring_round), 0)
        kv_new_len_val = jnp.where(is_valid, kv_new_lens_ref[s_idx], 0)
        q_offset = jnp.where(is_valid, q_offsets_ref[s_idx], 0)
        q_end = jnp.where(is_valid, cu_q_lens_ref[s_idx + 1], 0)
        q_sz = jnp.where(is_valid, schedule_ref.dma_q[step, b_idx, 1], 0)

        total_kv_len = kv_cache_len_val + kv_new_len_val

        # Causal base offset: K_base - Q_base
        q_base = q_idx * cfgs.bq_sz + q_offset
        causal_offset_list.append(k_id - q_base)
        q_sz_list.append(q_sz)
        is_valid_list.append(is_valid)
        local_k_end_list.append(kv_cache_len_val - k_id)

        # Stitching metadata (unused by HEAD_ALONG_SUBLANE, the only layout
        # the ring supports; kept for StepMetadata's shape).
        kv_left = jnp.maximum(total_kv_len - k_id, 0)
        kv_left_frm_cache = jnp.maximum(kv_cache_len_val - k_id, 0)
        kv_left_frm_new = jnp.maximum(kv_left - kv_left_frm_cache, 0)
        bkv_sz_frm_cache_list.append(
            jnp.minimum(kv_left_frm_cache, cfgs.bkv_sz))
        new_kv_len_start_list.append(q_end - kv_left_frm_new)

    meta = kernel.StepMetadata(
        causal_offset=causal_offset_list,
        bkv_sz_frm_cache=bkv_sz_frm_cache_list,
        new_kv_len_start=new_kv_len_start_list,
        q_sz=q_sz_list,
        is_valid=is_valid_list,
        local_k_start=None,
        local_k_end=local_k_end_list,
    )
    return meta, rounds, valids


class RingStepMetadataComputer(kernel.StepMetadataComputer):
    """StepMetadataComputer that decodes ring-encoded k indices and rotates
    the KV blocks before rpa_body's compute reads kv_in_vref."""

    @classmethod
    def scratch_shapes(cls, cfgs: configs.RpaConfigs) -> tuple:
        return (RingRefs(
            kv=pltpu.VMEM(
                (2, *cfgs.kv_vmem_shape[1:]),
                dtype=cfgs.serve.dtype_kv,
            ),
            dma_sems=pltpu.SemaphoreType.DMA((2, )),
            sync_sem=pltpu.SemaphoreType.REGULAR,
            local_sem=pltpu.SemaphoreType.DMA((1, )),
        ), )

    @classmethod
    def compiler_params(cls, cfgs: configs.RpaConfigs) -> dict:
        # The ring's startup barrier needs a barrier semaphore.
        return {"collective_id": 0}

    def init(self, ring_refs: RingRefs):
        """Zero the ring slots, like the pipeline KV window: whatever a
        previous program left at this address must not be interpretable as
        plausible KV data."""
        num_lanes = pltpu.get_tpu_info().num_lanes
        ring_kv_flat = ring_refs.kv.bitcast(jnp.uint32).reshape(-1, num_lanes)
        ring_kv_flat[...] = jnp.zeros_like(ring_kv_flat)

    def fetch_step_metadata(self, step, schedule_ref, kv_in_vref,
                            extra_scratches, chunk_start, *, cu_q_lens_ref,
                            q_offsets_ref, kv_cache_lens_ref, kv_new_lens_ref):
        (ring_refs, ) = extra_scratches
        pcp_ring = PcpRing(self.cfgs,
                           ring_refs,
                           step=step,
                           chunk_start=chunk_start)
        meta, rounds, valids = ring_fetch_step_metadata(
            step,
            schedule_ref,
            cu_q_lens_ref,
            q_offsets_ref,
            kv_cache_lens_ref,
            kv_new_lens_ref,
            cfgs=self.cfgs,
            pcp_ring=pcp_ring,
        )
        pcp_ring.stage(kv_in_vref, rounds, valids)
        return meta
