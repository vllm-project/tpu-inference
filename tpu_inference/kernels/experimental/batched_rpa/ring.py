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

import dataclasses

import jax
import jax.experimental.pallas as pl
import jax.experimental.pallas.tpu as pltpu
import jax.numpy as jnp
from jax import lax

from tpu_inference.kernels.experimental.batched_rpa import (configs, kernel,
                                                            schedule_cp, utils)


class RingMetadataComputer(schedule_cp.CPMetadataComputer):

    @property
    def global_kv_cache_lens_ref(self):
        return self.extra_refs[2]

    def k_idx_range(self, *, s_idx, **kwargs):
        # Every rank must run the same number of steps, so size the block
        # loop by rank 0's shard (the longest under page-interleaving) and
        # run group_size rounds per block; short ranks' tails are masked in
        # the kernel.
        del kwargs
        group_size = self.cfgs.serve.cp.group_size
        rank0_cache_len = utils.cp_local_cache_len(
            self.global_kv_cache_lens_ref[s_idx], group_size, 0,
            self.cfgs.serve.page_size)
        num_ring_blocks = pl.cdiv(rank0_cache_len, self.cfgs.bkv_sz)
        return 0, num_ring_blocks * group_size

    def decode_k(self, k_idx):
        # k_idx is ring-encoded (block * group_size + round); only round 0
        # fetches this rank's block from HBM, later rounds receive it from
        # the previous rank over the ring.
        group_size = self.cfgs.serve.cp.group_size
        return k_idx // group_size, k_idx % group_size == 0


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class RingSems:
    """The ring's semaphores, passed through the pipeline as one pytree."""

    dma_sems: jax.Ref  # [2] send/recv
    sync_sem: jax.Ref  # per-step handshake (see the module docstring)


class PcpRing:
    """Per-step ring state and the per-lane rotation protocol."""

    def __init__(self, cfgs, sems: RingSems, *, step, chunk_id, kv_window_ref):
        self.cfgs = cfgs
        self.sems = sems
        self.size = cfgs.serve.cp.group_size
        self.my_id = lax.axis_index(cfgs.serve.cp.ring_axis_name)
        self.next_id = self._device_id(lax.rem(self.my_id + 1, self.size))
        self.prev_id = self._device_id(
            lax.rem(self.my_id + self.size - 1, self.size))
        self.step_local = step
        self.step_global = chunk_id * cfgs.max_steps_ub + step
        self.kv_window = kv_window_ref
        self._pending_sends = []

    def _device_id(self, rank):
        if self.cfgs.serve.cp.ring_mesh_axis_names is None:
            return (rank, )
        return tuple(
            rank if name ==
            self.cfgs.serve.cp.ring_axis_name else lax.axis_index(name)
            for name in self.cfgs.serve.cp.ring_mesh_axis_names)

    def stage(self, rounds, valids):
        """Run the per-lane rotation for this step.

        rounds/valids hold each lane's ring round and validity, in lane
        order. Entering the step: startup barrier (first step only), then the
        per-step handshake. Per lane (= ring micro-step), in order: wait for
        this round's incoming block to land in the pipeline's window lane,
        then send that lane onward to the successor micro-step's window lane
        on the next rank. Sends are waited at the end of the step by
        finalize(). See the module docstring for why this is safe.
        """
        assert len(rounds) == len(valids) == self.cfgs.batch_size
        assert not self._pending_sends

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

        # Per-step handshake: tell the previous rank its sends of this step
        # may land (our regions last read at earlier steps are free), then
        # wait for the next rank's matching release before any send. Signal
        # first so every rank releases before it blocks.
        pl.semaphore_signal(
            self.sems.sync_sem,
            1,
            device_id=self.prev_id,
            device_id_type=pl.DeviceIdType.MESH,
        )
        pl.semaphore_wait(self.sems.sync_sem, 1)

        n_buffer = self.cfgs.n_buffer
        slot = lax.rem(self.step_local, n_buffer)
        # The successor of this step's last lane is the next step's lane 0;
        # the window's slot sequence restarts at zero on a new chunk.
        # Full chunks have exactly max_steps_ub steps; only the last chunk is
        # shorter, and its final step never sends (schedules end on the last
        # round or masked lanes), so the past-the-end value is never used.
        next_step = self.step_local + 1
        next_slot = jnp.where(next_step < self.cfgs.max_steps_ub,
                              lax.rem(next_step, n_buffer), 0)

        for b_idx in range(self.cfgs.batch_size):
            valid_b = valids[b_idx]
            round_b = rounds[b_idx]
            sends_b = jnp.logical_and(valid_b, round_b != self.size - 1)
            receives_b = jnp.logical_and(valid_b, round_b > 0)

            lane_ref = self.kv_window.at[slot, b_idx]
            if b_idx + 1 < self.cfgs.batch_size:
                dst_ref = self.kv_window.at[slot, b_idx + 1]
            else:
                dst_ref = self.kv_window.at[next_slot, 0]
            remote_op = pltpu.make_async_remote_copy(
                src_ref=lane_ref,
                dst_ref=dst_ref,
                send_sem=self.sems.dma_sems.at[0],
                recv_sem=self.sems.dma_sems.at[1],
                device_id=self.next_id,
                device_id_type=pl.DeviceIdType.MESH,
            )

            # For rounds > 0 the lane's HBM fetch was zero-size; the block
            # lands here over ICI instead, sent by the previous rank's
            # predecessor micro-step. All lanes have identical shapes, so
            # this descriptor waits the matching byte count.
            @pl.when(receives_b)
            def wait_ring_recv():
                remote_op.wait_recv()

            # Round 0 sends this rank's own HBM fetch (complete: the
            # pipeline's wait_in ran before the body); later rounds forward
            # the block that just landed above.
            @pl.when(sends_b)
            def start_rotate():
                remote_op.start()

            self._pending_sends.append((sends_b, remote_op))

    def finalize(self):
        """Wait this step's sends, at the end of the step.

        Must run before the body returns: the next iteration's copy_in phase
        may issue a local HBM fetch into this step's window slot, which would
        otherwise race an in-flight send still reading it.
        """
        for sends_b, remote_op in self._pending_sends:

            @pl.when(sends_b)
            def wait_ring_send():
                remote_op.wait_send()

        self._pending_sends = []


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
        # k_idx is ring-encoded (block * group_size + round); split it
        # before any block arithmetic.
        k_idx = schedule_ref.k_idx[step, b_idx]
        ring_round = lax.rem(k_idx, pcp_ring.size)
        k_idx = k_idx // pcp_ring.size
        rounds.append(ring_round)
        valids.append(is_valid)

        k_id = jnp.where(is_valid, k_idx * cfgs.bkv_sz, 0)
        # After ring_round hops the resident block came from rank
        # (my_id - ring_round); mask it by that rank's shard length, not ours
        # (kv_cache_lens_ref holds the global length under the ring).
        src_rank = lax.rem(pcp_ring.my_id + pcp_ring.size - ring_round,
                           pcp_ring.size)
        src_cache_len = utils.cp_local_cache_len(kv_cache_lens_ref[s_idx],
                                                 pcp_ring.size, src_rank,
                                                 cfgs.serve.page_size)
        kv_cache_len_val = jnp.where(is_valid, src_cache_len, 0)
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
    the KV blocks directly between the ranks' pipeline window buffers."""

    def fetch_step_metadata(self, step, schedule_ref, kv_in_vref,
                            extra_scratches, chunk_id, *, cu_q_lens_ref,
                            q_offsets_ref, kv_cache_lens_ref, kv_new_lens_ref,
                            kv_window_ref):
        del kv_in_vref  # the ring addresses the window by slot directly
        (ring_sems, ) = extra_scratches
        pcp_ring = PcpRing(self.cfgs,
                           ring_sems,
                           step=step,
                           chunk_id=chunk_id,
                           kv_window_ref=kv_window_ref)
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
        pcp_ring.stage(rounds, valids)
        # end_step waits this step's sends before the body returns, ahead of
        # the pipeline's next copy_in into their source slots.
        return meta, pcp_ring.finalize
