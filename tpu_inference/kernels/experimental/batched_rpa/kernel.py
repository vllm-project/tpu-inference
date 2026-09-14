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
import functools
from collections.abc import Sequence

import jax
import jax.experimental.pallas as pl
import jax.experimental.pallas.tpu as pltpu
import jax.numpy as jnp
from jax import lax

# yapf: disable
from tpu_inference.kernels.experimental.batched_rpa import (bref_override,
                                                            configs,
                                                            flash_attention,
                                                            schedule,
                                                            stitch_utils,
                                                            utils)

# yapf: enable


def strided_load_bkv(
    kv_in_vref: jax.Ref,
    b_idx: int,
    start: int,
    *,
    cfgs: configs.RpaConfigs,
) -> list[tuple[jax.Array, jax.Array]]:
    assert start % cfgs.serve.packing_kv == 0
    start //= cfgs.serve.packing_kv
    kv_u32_ref = kv_in_vref.at[b_idx].bitcast(jnp.uint32)
    kv_ref = kv_u32_ref.reshape(-1, cfgs.aligned_kv_head_dim)

    if cfgs.serve.packing_kv == 1:
        k = utils.strided_load(
            kv_ref,
            start,
            cfgs.bkv_sz * cfgs.bkv_stride,
            cfgs.bkv_stride,
            dtype=cfgs.serve.dtype_kv,
        )
        v = utils.strided_load(
            kv_ref,
            start + 1,
            cfgs.bkv_sz * cfgs.bkv_stride,
            cfgs.bkv_stride,
            dtype=cfgs.serve.dtype_kv,
        )
        return [(k, v)]

    kv = utils.strided_load(kv_ref, start, cfgs.bkv_sz * cfgs.bkv_stride,
                            cfgs.bkv_stride)
    bitwidth = jax.dtypes.itemsize_bits(cfgs.serve.dtype_kv)

    return utils.convert_to_target_bitwidth(kv,
                                            target_bitwidth=bitwidth,
                                            kv_dtype=cfgs.serve.dtype_kv)


def calculate_and_store_out(
    step_idx: jax.Array,
    schedule_ref: schedule.RpaSchedule,
    acc_list: list[jax.Array],
    l_list: list[jax.Array],
    m_list: list[jax.Array],
    o_vref: jax.Ref,
    lse_o_vref: jax.Ref | None,
    *,
    cfgs: configs.RpaConfigs,
):

    def _accum(b_idx: jax.Array, batch_acc: jax.Array, batch_l: jax.Array):
        batch_l = utils.broadcast_minor(batch_l, batch_acc.shape)

        if (cfgs.serve.dtype_out == jnp.float32
                or cfgs.serve.dtype_out == batch_l.dtype == jnp.bfloat16):
            result = lax.div(batch_acc, batch_l)
        else:
            result = batch_acc * pl.reciprocal(batch_l, approx=True)
        out = result.astype(cfgs.serve.dtype_out)

        o_u32_vref = o_vref.at[b_idx].bitcast(jnp.uint32)
        out_ref = o_u32_vref.reshape(-1, cfgs.aligned_q_head_dim)
        pad_width = [[0, 0] for _ in range(out.ndim)]
        pad_width[-1][-1] = cfgs.aligned_q_head_dim - cfgs.compute_kv_head_dim
        out = jnp.pad(out, pad_width, constant_values=0)
        out = pltpu.bitcast(out, out_ref.dtype).reshape(out_ref.shape)
        utils.strided_store(out_ref, 0, out_ref.shape[0], 1, out)

    def _stage_lse(b_idx: int, batch_m: jax.Array, batch_l: jax.Array):
        lse_val = batch_m + jnp.log(jnp.maximum(batch_l, 1e-9))
        lse_val = lse_val.astype(cfgs.serve.dtype_out)
        pad_rows = cfgs.lse_rows_per_token - cfgs.aligned_num_q_heads_per_kv_head
        if pad_rows:
            kv_heads, _, lanes = lse_val.shape
            lse_val = jnp.pad(
                lse_val.reshape(kv_heads, cfgs.bq_sz,
                                cfgs.aligned_num_q_heads_per_kv_head, lanes),
                ((0, 0), (0, 0), (0, pad_rows), (0, 0)),
            ).reshape(kv_heads, cfgs.bq_sz * cfgs.lse_rows_per_token, lanes)
        lse_o_vref[b_idx] = lse_val

    if cfgs.fuse_accum:
        for b in range(cfgs.batch_size):
            _accum(b, acc_list[b], l_list[b])
    else:
        # Adding a conditional causes a scheduling barrier. In prefill, we often
        # use small block sizes, so it's not worth executing the accumulation
        # on every block. In decode, because of the large block sizes / and or
        # batch sizes, we almost always use accumulation on every block. Please
        # tune `fuse_accum` for your use case.
        for b in range(cfgs.batch_size):
            is_last_k = schedule_ref.is_last_k[step_idx, b] == 1
            acc_val = acc_list[b]
            l_val = l_list[b]
            accum_named_call = jax.named_call(_accum, name=f"accum_{b}")
            jax.lax.cond(is_last_k, accum_named_call, lambda *_: None, b,
                         acc_val, l_val)

    if cfgs.serve.return_lse:
        for b in range(cfgs.batch_size):
            is_last_k = schedule_ref.is_last_k[step_idx, b] == 1
            m_val = m_list[b]
            l_val = l_list[b]
            jax.lax.cond(is_last_k, _stage_lse, lambda *_: None, b, m_val,
                         l_val)


def get_scale_factors(
    k: jax.Array,
    v: jax.Array,
    *,
    cfgs: configs.RpaConfigs,
):
    if not cfgs.serve.per_token_scale:
        return k, v, cfgs.serve.scale_k, cfgs.serve.scale_v

    b = k.shape[0]
    num_heads = k.shape[1]
    # Number of scale channels in VMEM is determined by the scale factor bitwidth
    # and the VMEM kv bitwidth (k.dtype). However, we cannot use the
    # scale_channels configs.py member variable because of the case of unpacking
    # uint8_t to fp4_e2m1fn.
    if cfgs.serve.per_token_scale_dtype is None:
        scale_channels = 0
    else:
        scale_bits = jax.dtypes.itemsize_bits(cfgs.serve.per_token_scale_dtype)
        kv_bits = jax.dtypes.itemsize_bits(k.dtype)
        scale_channels = max(1, scale_bits // kv_bits)

    # Extract multi-channel scale factor slices from the trailing sublanes
    k_scale_slice = k[:, :, cfgs.model.head_dim:cfgs.model.head_dim +
                      scale_channels, :]
    v_scale_slice = v[:, :, cfgs.model.head_dim:cfgs.model.head_dim +
                      scale_channels, :]

    # If there are multiple scale channels, pltpu.bitcast operates on the
    # second to last dimension, which is the head dimension, which is the same
    # as the scale_channels dimension, so this works out of the box.
    if scale_channels > 1:
        k_scale = pltpu.bitcast(k_scale_slice,
                                cfgs.serve.per_token_scale_dtype)
        v_scale = pltpu.bitcast(v_scale_slice,
                                cfgs.serve.per_token_scale_dtype)

    # If there is only one scale channel, just change dtype to the scale dtype.
    else:
        k_scale = k_scale_slice.astype(cfgs.serve.per_token_scale_dtype)
        v_scale = v_scale_slice.astype(cfgs.serve.per_token_scale_dtype)

    # In QK multiplication, we always scale after the qk matmul, so we can
    # unify per-token and per-tensor scaling by broadcasting the scale factor
    # to the head dimension outside of the flash attention kernel.
    k_scale = k_scale.reshape(b, num_heads, cfgs.bkv_sz)
    # To match the accumulation dtype of qk.
    k_scale = k_scale.astype(jnp.float32)
    # Add a new dimension for the head dimension.
    k_scale = k_scale[:, :, jnp.newaxis, :]

    # In PV multiplication, we scale the p rather than the pv for per-token
    # scaling (as it is mathematically necessary to scale before the matmul for
    # per-token scaling) so we can NOT unify per-token and per-tensor scaling,
    # so we broadcast the scale factor along the head dimension inside the
    # flash attention kernel itself and do not do so here.
    v_scale = v_scale.reshape(b, num_heads, cfgs.bkv_sz)

    k = k[:, :, :cfgs.compute_kv_head_dim, :]
    v = v[:, :, :cfgs.compute_kv_head_dim, :]

    return k, v, k_scale, v_scale


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class StepMetadata:
    """Metadata and scalars extracted for the current execution step."""

    causal_offset: list[jax.Array]
    bkv_sz_frm_cache: list[jax.Array]
    new_kv_len_start: list[jax.Array]
    local_k_start: list[jax.Array] | None = None
    local_k_end: list[jax.Array] | None = None


def fetch_step_metadata(
    step: jax.Array,
    schedule_ref: schedule.RpaSchedule,
    cu_q_lens_ref: jax.Ref,
    q_positions_ref: jax.Ref,
    global_kv_cache_lens_ref: jax.Ref,
    kv_new_lens_ref: jax.Ref,
    cp_rank_ref: jax.Ref,
    *,
    cfgs: configs.RpaConfigs,
) -> StepMetadata:
    """Fetches metadata and handles scalar & mask interval values for the current step."""
    causal_offset_list = []
    bkv_sz_frm_cache_list = []
    new_kv_len_start_list = []
    ring = cfgs.ring_enabled
    cp_rank = cp_rank_ref[0]
    # The ring makes every rank walk rank 0's stripe so they stay in lockstep.
    scope_rank = 0 if ring else cp_rank
    if ring:
        cp_size = cfgs.serve.cp_group_size

    local_k_start_list = ([] if cfgs.serve.attention_scope
                          == configs.AttentionScope.NEW_TOKENS_ONLY else None)
    local_k_end_list = ([] if cfgs.serve.attention_scope
                        == configs.AttentionScope.CACHE_ONLY else None)

    for b_idx in range(cfgs.batch_size):
        s_idx = schedule_ref.s_idx[step, b_idx]
        is_valid = s_idx != -1
        q_idx = schedule_ref.q_idx[step, b_idx]
        k_idx = schedule_ref.k_idx[step, b_idx]
        if ring:
            # k_idx packs (local block, hop); see CPMetadataComputer. After
            # `hop` hops the buffer holds the stripe of the rank that far back
            # around the ring.
            hop = lax.rem(k_idx, cp_size)
            k_idx = k_idx // cp_size
            src_rank = lax.rem(cp_rank + cp_size - hop, cp_size)
        # A rank's stripe is contiguous in its own pages, so the block offset
        # stays local even though the data belongs to `src_rank`.
        k_id = jnp.where(is_valid, k_idx * cfgs.bkv_sz, 0)
        global_kv_cache_len = jnp.where(is_valid,
                                        global_kv_cache_lens_ref[s_idx], 0)
        kv_cache_len_val = cfgs.local_kv_cache_len(global_kv_cache_len,
                                                   scope_rank)
        kv_new_len_val = jnp.where(is_valid, kv_new_lens_ref[s_idx], 0)
        # Without a page table the new KV is the sequence's own Q.
        kv_new_start_val = jnp.where(is_valid, cu_q_lens_ref[s_idx], 0)
        q_position = jnp.where(is_valid, q_positions_ref[s_idx], 0)

        total_kv_len = kv_cache_len_val + kv_new_len_val

        # Causal base offset: K_base - Q_base
        q_base = q_idx * cfgs.bq_sz + q_position
        causal_offset = k_id - q_base
        causal_offset_list.append(causal_offset)

        # New tokens start at `kv_cache_len` of the sequence's KV index space.
        # (Not `q_position`: under PCP a head/tail chunk's Q sits inside the
        # all-gathered current KV, so its position is past the cache.)
        if local_k_start_list is not None:
            local_k_start_list.append(kv_cache_len_val - k_id)
        if local_k_end_list is not None:
            scope_end = kv_cache_len_val
            if ring:
                # The buffer holds `src_rank`'s block, so its stripe is what
                # bounds it. Ranks differ by at most one page and every rank
                # runs rank 0's (longest) block count, so a short rank's tail
                # is masked off here.
                scope_end = cfgs.local_kv_cache_len(global_kv_cache_len,
                                                    src_rank)
            local_k_end_list.append(scope_end - k_id)

        # Stitching metadata
        kv_left = jnp.maximum(total_kv_len - k_id, 0)
        kv_left_frm_cache = jnp.maximum(kv_cache_len_val - k_id, 0)
        kv_left_frm_new = jnp.maximum(kv_left - kv_left_frm_cache, 0)
        bkv_sz_frm_cache = jnp.minimum(kv_left_frm_cache, cfgs.bkv_sz)
        new_kv_len_start = kv_new_start_val + kv_new_len_val - kv_left_frm_new

        bkv_sz_frm_cache_list.append(bkv_sz_frm_cache)
        new_kv_len_start_list.append(new_kv_len_start)

    return StepMetadata(
        causal_offset=causal_offset_list,
        bkv_sz_frm_cache=bkv_sz_frm_cache_list,
        new_kv_len_start=new_kv_len_start_list,
        local_k_start=local_k_start_list,
        local_k_end=local_k_end_list,
    )


def generate_mask(
    shape: tuple[int, int, int, int],
    *,
    bq_start: int,
    step_meta: StepMetadata,
    cfgs: configs.RpaConfigs,
) -> jax.Array:
    """Generates causal, sliding window, and attention scope mask for QK computation."""
    b, k_heads, tq, s = shape

    kv_iota = lax.broadcasted_iota(jnp.int32, (k_heads, tq, s), 2)
    q_iota = lax.broadcasted_iota(jnp.int32, (k_heads, tq, s), 1)
    q_iota //= cfgs.aligned_num_q_heads_per_kv_head
    q_kv_diff = q_iota - kv_iota

    masks = []
    for b_idx in range(b):
        # NOTE: Goal is to compute q_len >= kv_len. But we want to utilize scalar
        # compute as much as possible before involving vector compute. Therefore, we
        # break down a computational steps into following equations to separate out
        # scalar and vector compute.
        # q_len = q_iota + (bq_start + processed_q_len)
        # kv_len = kv_iota + processed_kv_len
        # Step 1: We already preprocessed causal_offset
        #   causal_offset = kv_len - q_len
        # Step 2
        #   offset = causal_offset - bq_start
        offset = step_meta.causal_offset[b_idx] - bq_start
        mask_b = q_kv_diff >= offset
        if (sliding_window := cfgs.model.sliding_window) is not None:
            mask_b = jnp.logical_and(mask_b, q_kv_diff
                                     < sliding_window + offset)

        if step_meta.local_k_start is not None:
            mask_b = jnp.logical_and(mask_b, kv_iota
                                     >= step_meta.local_k_start[b_idx])
        if step_meta.local_k_end is not None:
            mask_b = jnp.logical_and(mask_b, kv_iota
                                     < step_meta.local_k_end[b_idx])

        masks.append(mask_b)

    return masks

@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class RingSems:
    """The ring's semaphores, passed through the pipeline as one pytree."""

    dma_sems: jax.Ref  # [2] send/recv
    sync_sem: jax.Ref  # per-step handshake


class RingAttention:
    """Rotates the KV window around the PCP ring during the cache phase.

    Each rank owns one page-stripe of the KV cache. Step `t` is hop `t % P` of
    one KV block: at hop 0 the rank fetches its own block from HBM, and at each
    later hop the block it holds arrived from the previous rank, so after P
    hops its Q has attended the whole cache under a single online softmax. Only
    KV moves, which is the cheap side under GQA -- a block is
    `num_kv_heads * 2 * head_dim` wide, against `num_q_heads * head_dim` for
    the Q-gathering alternative.

    The ring runs one lane per step (`CPBlockSizeCalculator`), so the rotation
    can move the whole buffer in a single copy without worrying about lanes
    sitting on different hops.

    The rotation writes straight into the pipeline's KV buffer. Step `t`
    computes on slot `t % n_buffer` (`emit_pipeline` waits once per step, so the
    slot is a pure function of the step index), and every rank runs the same
    schedule -- the wrapper gives them all rank 0's cache length, the longest --
    so the sender knows exactly which slot the receiver will read next.

    Flow control is a single counting semaphore. The sender writes slots and
    the receiver frees them in the same order, so counting is enough to keep
    the sender from overrunning a slot still in use -- it never has to be told
    *which* slot is free:

      * each rank starts by granting its predecessor `n_buffer` credits, since
        every slot is free;
      * the sender spends one before each rotation;
      * the receiver grants one back after consuming a rotated block.

    A send at hop `r < P - 1` is always matched by a receive at hop `r + 1`, so
    grants and spends balance over the run and the kernel drains the initial
    `n_buffer` on the way out.
    """

    def __init__(self, cfgs: configs.RpaConfigs, sems: RingSems, *,
                 kv_window_ref: jax.Ref):
        self.cfgs = cfgs
        self.size = cfgs.serve.cp_group_size
        assert self.size is not None
        assert cfgs.serve.pcp_ring_axis_name is not None
        self.kv_window = kv_window_ref
        self.sems = sems
        my_id = lax.axis_index(cfgs.serve.pcp_ring_axis_name)
        self.next_id = self._device_id(lax.rem(my_id + 1, self.size))
        self.prev_id = self._device_id(
            lax.rem(my_id + self.size - 1, self.size))
        self._pending_send = None

    def _device_id(self, rank):
        names = self.cfgs.serve.pcp_ring_mesh_axis_names
        if names is None:
            return (rank, )
        return tuple(
            rank if name == self.cfgs.serve.pcp_ring_axis_name else
            lax.axis_index(name) for name in names)

    def initial_handshake(self):
        """Tell the predecessor every slot is free. Call before the pipeline."""
        pl.semaphore_signal(
            self.sems.sync_sem,
            self.cfgs.n_buffer,
            device_id=self.prev_id,
            device_id_type=pl.DeviceIdType.MESH,
        )

    def drain_credits(self):
        """Consume the successor's initial grants. Call after the pipeline."""
        pl.semaphore_wait(self.sems.sync_sem, self.cfgs.n_buffer)

    def _remote_copy(self, src_slot, dst_slot):
        return pltpu.make_async_remote_copy(
            src_ref=self.kv_window.at[src_slot],
            dst_ref=self.kv_window.at[dst_slot],
            send_sem=self.sems.dma_sems.at[0],
            recv_sem=self.sems.dma_sems.at[1],
            device_id=self.next_id,
            device_id_type=pl.DeviceIdType.MESH,
        )

    def receive_and_forward(self, step, schedule_ref):
        """Take this step's rotated block, then pass it on.

        Forwarding before the attention math lets the hop overlap compute; the
        DMA only reads the buffer the body also reads.
        """
        n_buffer = self.cfgs.n_buffer
        slot = lax.rem(step, n_buffer)
        # The next step restarts at slot 0 when it belongs to the next schedule
        # chunk, which is a fresh pipeline invocation. `num_programs` is this
        # invocation's grid, so it is exactly the chunk's step count.
        next_step = step + 1
        next_slot = jnp.where(next_step < pl.num_programs(0),
                              lax.rem(next_step, n_buffer), 0)

        # One lane per step under the ring, and k_idx packs (local block,
        # hop) -- how far this block has travelled to reach this rank.
        is_valid = schedule_ref.s_idx[step, 0] != -1
        hop = lax.rem(schedule_ref.k_idx[step, 0], self.size)
        receives = jnp.logical_and(is_valid, hop > 0)
        sends = jnp.logical_and(is_valid, hop < self.size - 1)

        @pl.when(receives)
        def _wait_recv():
            self._remote_copy(slot, slot).wait_recv()

        send_op = self._remote_copy(slot, next_slot)

        @pl.when(sends)
        def _start_send():
            # Spend a credit before writing into the receiver's buffer.
            pl.semaphore_wait(self.sems.sync_sem, 1)
            send_op.start()

        self._pending_send = (sends, receives, send_op, slot)

    def finish(self):
        """Close out the step's hop: drain the send, release the slot."""
        assert self._pending_send is not None
        sends, receives, send_op, slot = self._pending_send
        self._pending_send = None

        @pl.when(sends)
        def _wait_send():
            send_op.wait_send()

        @pl.when(receives)
        def _release_slot():
            pl.semaphore_signal(
                self.sems.sync_sem,
                1,
                device_id=self.prev_id,
                device_id_type=pl.DeviceIdType.MESH,
            )


def rpa_body(
    # Inputs.
    q_vref: jax.Ref,
    kv_in_vref: jax.Ref,
    # Outputs
    o_vref: jax.Ref,
    lse_o_vref: jax.Ref | None,
    # Scratches.
    schedule_ref: schedule.RpaSchedule,
    m_scratch_ref: jax.Ref,
    l_scratch_ref: jax.Ref,
    acc_scratch_ref: jax.Ref,
    *,
    # Passed refs
    cu_q_lens_ref: jax.Ref,
    q_positions_ref: jax.Ref,
    global_kv_cache_lens_ref: jax.Ref,
    kv_new_lens_ref: jax.Ref,
    cp_rank_ref: jax.Ref,
    # Configs.
    cfgs: configs.RpaConfigs,
    ring: "RingAttention | None" = None,
):
    step = pl.program_id(0)

    # Step 1: Fetch metadata.
    step_meta = fetch_step_metadata(
        step,
        schedule_ref,
        cu_q_lens_ref,
        q_positions_ref,
        global_kv_cache_lens_ref,
        kv_new_lens_ref,
        cp_rank_ref,
        cfgs=cfgs,
    )

    # Take this step's rotated KV block and forward it to the next rank, so
    # the hop overlaps the attention math below.
    if ring is not None:
        ring.receive_and_forward(step, schedule_ref)

    # Step 2: Fetch inputs.
    q_p = cfgs.aligned_num_q_heads_per_kv_head // cfgs.serve.packing_q
    q_ref = q_vref.bitcast(jnp.uint32).reshape(-1, cfgs.aligned_q_head_dim)
    q_loaded = utils.strided_load(
        q_ref,
        0,
        cfgs.batch_size * cfgs.model.num_kv_heads * cfgs.bq_sz * q_p,
        1,
        dtype=cfgs.serve.dtype_q,
    )
    q = q_loaded.reshape(
        cfgs.batch_size,
        cfgs.model.num_kv_heads,
        cfgs.bq_sz * cfgs.aligned_num_q_heads_per_kv_head,
        cfgs.aligned_q_head_dim,
    )
    q = q[..., :cfgs.compute_kv_head_dim]

    # We want to load k, v from (batch, bkv_sz, bkv_stride, kv_packing, d)
    # where bkv_stride ~= num_kv_heads * 2 // kv_packing
    # to 2x (batch, num_kv_heads, bkv_sz, d)
    # We use strided_load to avoid the expensive transpose.
    k_b = []
    v_b = []

    if cfgs.serve.kv_layout == configs.KVLayout.SEQ_ALONG_LANE:
        stitch_results = []
        for b_idx in range(cfgs.batch_size):
            res = stitch_utils.stitch_new_kv_lane(
                kv_in_vref,
                b_idx,
                step_meta.bkv_sz_frm_cache[b_idx],
                step_meta.new_kv_len_start[b_idx],
                cfgs=cfgs,
            )
            stitch_results.append(res)
        for b_idx in range(cfgs.batch_size):
            stitch_utils.store_new_kv_lane(
                kv_in_vref,
                b_idx,
                stitch_results[b_idx],
                cfgs=cfgs,
            )
        for b_idx in range(cfgs.batch_size):
            ks = []
            vs = []
            for kv_head in range(cfgs.model.num_kv_heads):
                k_slice = kv_in_vref.at[b_idx, kv_head * 2].bitcast(jnp.uint32)
                v_slice = kv_in_vref.at[b_idx,
                                        kv_head * 2 + 1].bitcast(jnp.uint32)

                target_shape = (-1, cfgs.bkv_sz + 2 * cfgs.serve.page_size)
                k_head_ref = k_slice.reshape(target_shape)
                v_head_ref = v_slice.reshape(target_shape)
                pack_dim = cfgs.aligned_kv_head_dim // cfgs.serve.packing_kv

                # Load as uint32 to avoid dtype conversion during strided load
                k_head_u32 = utils.strided_load(k_head_ref, 0, pack_dim, 1)
                v_head_u32 = utils.strided_load(v_head_ref, 0, pack_dim, 1)

                if cfgs.serve.dtype_kv == jnp.uint8:
                    fp4 = jnp.float4_e2m1fn
                    k_head = pltpu.bitcast(k_head_u32, fp4)[:, :cfgs.bkv_sz]
                    v_head = pltpu.bitcast(v_head_u32, fp4)[:, :cfgs.bkv_sz]
                    ks.append(k_head)
                    vs.append(v_head)
                else:
                    k_head_loaded = pltpu.bitcast(k_head_u32,
                                                  cfgs.serve.dtype_kv)
                    v_head_loaded = pltpu.bitcast(v_head_u32,
                                                  cfgs.serve.dtype_kv)
                    k_head = k_head_loaded[:, :cfgs.bkv_sz]
                    v_head = v_head_loaded[:, :cfgs.bkv_sz]
                    ks.append(
                        k_head.reshape(cfgs.aligned_kv_head_dim, cfgs.bkv_sz))
                    vs.append(
                        v_head.reshape(cfgs.aligned_kv_head_dim, cfgs.bkv_sz))
            k_b.append(jnp.stack(ks, axis=0))
            v_b.append(jnp.stack(vs, axis=0))
    else:
        for b_idx in range(cfgs.batch_size):
            heads_per_load = pl.cdiv(cfgs.serve.packing_kv, 2)
            ks = []
            vs = []
            for kv_head_start in range(0, cfgs.model.num_kv_heads,
                                       heads_per_load):
                bkv_lst = strided_load_bkv(
                    kv_in_vref,
                    b_idx,
                    kv_head_start * 2,
                    cfgs=cfgs,
                )
                ks.append(jnp.stack([k for k, _ in bkv_lst], axis=0))
                vs.append(jnp.stack([v for _, v in bkv_lst], axis=0))
            k, v = jnp.concat(ks, axis=0), jnp.concat(vs, axis=0)
            k = k.reshape(-1, cfgs.bkv_sz, cfgs.aligned_kv_head_dim)
            v = v.reshape(-1, cfgs.bkv_sz, cfgs.aligned_kv_head_dim)

            k = k[:cfgs.model.num_kv_heads]
            v = v[:cfgs.model.num_kv_heads]
            k_b.append(k)
            v_b.append(v)
    # Stack to (batch, num_heads, bkv_sz, num_lanes)
    k = jnp.stack(k_b, axis=0)
    v = jnp.stack(v_b, axis=0)

    k, v, k_scale, v_scale = get_scale_factors(k, v, cfgs=cfgs)

    # Step 3: Perform compute.
    m_val = m_scratch_ref[...]
    l_val = l_scratch_ref[...]
    acc_val = acc_scratch_ref[...]

    l_new_list = []
    m_new_list = []
    acc_new_list = []

    prev_p = prev_alpha_list = prev_q_slice = None
    for bq_start in range(0, cfgs.bq_sz, cfgs.bq_c_sz):
        bq_end = min(bq_start + cfgs.bq_c_sz, cfgs.bq_sz)
        q_start = bq_start * cfgs.aligned_num_q_heads_per_kv_head
        q_end = bq_end * cfgs.aligned_num_q_heads_per_kv_head
        q_slice = slice(q_start, q_end)

        custom_mask = generate_mask(
            shape=(
                cfgs.batch_size,
                cfgs.model.num_kv_heads,
                q_end - q_start,
                cfgs.bkv_sz,
            ),
            bq_start=bq_start,
            step_meta=step_meta,
            cfgs=cfgs,
        )

        p, alpha_list, m_next, l_next, m_carry = flash_attention.flash_attention_qk_softmax(
            step,
            q[:, :, q_slice],
            k,
            m_val[:, q_slice],
            l_val[:, q_slice],
            schedule_ref.is_last_k,
            custom_mask=custom_mask,
            cfgs=cfgs,
            bq_start=bq_start,
            k_scale=k_scale,
        )
        m_scratch_ref[:, q_slice] = m_carry
        l_scratch_ref[:, q_slice] = l_next[-1]
        if cfgs.serve.return_lse:
            m_new_list.append(m_next)
        l_new_list.append(l_next)

        if prev_p is not None:
            o_next = flash_attention.flash_attention_pv(
                prev_p,
                v,
                prev_alpha_list,
                acc_val[:, prev_q_slice],
                cfgs=cfgs,
                v_scale=v_scale,
            )
            acc_scratch_ref[:, prev_q_slice] = o_next[-1]
            acc_new_list.append(o_next)

        prev_p = p
        prev_alpha_list = alpha_list
        prev_q_slice = q_slice

    assert prev_p is not None
    o_next = flash_attention.flash_attention_pv(
        prev_p,
        v,
        prev_alpha_list,
        acc_val[:, prev_q_slice],
        cfgs=cfgs,
        v_scale=v_scale,
    )
    acc_scratch_ref[:, prev_q_slice] = o_next[-1]
    acc_new_list.append(o_next)
    if cfgs.serve.return_lse:
        m_next = jnp.concatenate(m_new_list, axis=2)
    l_next = jnp.concatenate(l_new_list, axis=2)
    acc_next = jnp.concatenate(acc_new_list, axis=2)

    # Step 4: Write back outputs.
    calculate_and_store_out(
        step,
        schedule_ref,
        acc_next,
        l_next,
        m_next,
        o_vref,
        lse_o_vref,
        cfgs=cfgs,
    )

    if ring is not None:
        ring.finish()


# Define main kernel.


def create_allocs(
    kv_cache_hbm_ref: jax.Ref,
    o_hbm_ref: jax.Ref,
    lse_hbm_ref: jax.Ref | None,
    cfgs: configs.RpaConfigs,
) -> tuple[
        bref_override.BatchingQRef,
        bref_override.KVBufferedRefSeqAlongLane
        | bref_override.KVBufferedRefHeadAlongSublane,
        bref_override.BatchingORef,
        bref_override.BatchingLSERef | None,
]:
    kv_cache_spec = pl.BlockSpec(
        block_shape=cfgs.kv_vmem_shape,
        memory_space=pltpu.VMEM,
        index_map=lambda i: (i, ),
        pipeline_mode=pl.Buffered(buffer_count=cfgs.n_buffer,
                                  use_lookahead=True),
    )
    q_spec = pl.BlockSpec(
        block_shape=cfgs.q_vmem_shape,
        memory_space=pltpu.VMEM,
        index_map=lambda i: (i, ),
        pipeline_mode=pl.Buffered(buffer_count=cfgs.n_buffer,
                                  use_lookahead=True),
    )
    o_spec = pl.BlockSpec(
        block_shape=cfgs.q_vmem_shape,
        memory_space=pltpu.VMEM,
        index_map=lambda i: (i, ),
        pipeline_mode=pl.Buffered(buffer_count=2, use_lookahead=False),
    )

    if cfgs.serve.kv_layout == configs.KVLayout.SEQ_ALONG_LANE:
        kv_cache_alloc_cls = bref_override.KVBufferedRefSeqAlongLane
    else:
        kv_cache_alloc_cls = bref_override.KVBufferedRefHeadAlongSublane

    kv_cache_alloc = kv_cache_alloc_cls.input_output(
        spec=kv_cache_spec,
        dtype_or_type=kv_cache_hbm_ref,
        buffer_count=cfgs.n_buffer,
        use_lookahead=True,
        cfgs=cfgs,
    )
    q_alloc = bref_override.BatchingQRef.input(
        spec=q_spec,
        dtype_or_type=o_hbm_ref,
        buffer_count=cfgs.n_buffer,
        use_lookahead=True,
        cfgs=cfgs,
    )
    o_alloc = bref_override.BatchingORef.output(
        spec=o_spec,
        dtype_or_type=o_hbm_ref,
        buffer_count=2,
        use_lookahead=False,
        cfgs=cfgs,
    )

    lse_alloc = None
    if cfgs.serve.return_lse:
        lse_spec = pl.BlockSpec(
            block_shape=cfgs.lse_vmem_shape,
            memory_space=pltpu.VMEM,
            index_map=lambda i: (i, ),
            pipeline_mode=pl.Buffered(buffer_count=2, use_lookahead=False),
        )
        lse_alloc = bref_override.BatchingLSERef.output(
            spec=lse_spec,
            dtype_or_type=lse_hbm_ref,
            buffer_count=2,
            use_lookahead=False,
            cfgs=cfgs,
        )

    return q_alloc, kv_cache_alloc, o_alloc, lse_alloc


def get_kernel_name(cfgs: configs.RpaConfigs) -> str:
    serve = cfgs.serve
    name = f"RPA{cfgs.mode.symbol}-{serve.kv_layout.symbol}-p{serve.page_size}"
    name += f"-b{cfgs.batch_size}-q{cfgs.bq_sz}-k{cfgs.bkv_sz}"
    if cfgs.model.sliding_window:
        name += f"-sw{cfgs.model.sliding_window}"
    return name


def get_kernel_metadata(
    cfgs: configs.RpaConfigs, ) -> dict[str, str | int | float]:
    cfgs_dict = dataclasses.asdict(cfgs)
    ret = {}
    for path, val in jax.tree_util.tree_leaves_with_path(cfgs_dict):
        key = jax.tree_util.keystr(path, simple=True, separator=".")
        if not isinstance(val, str | int | float):
            val = str(val)
        ret[key] = val
    return ret


def rpa_kernel(
    cu_q_lens: jax.Array,
    q_positions: jax.Array,
    global_kv_cache_lens: jax.Array,
    kv_new_lens: jax.Array,
    cp_rank: jax.Array,
    page_indices: jax.Array,
    new_kv_page_indices_refs: Sequence[jax.Array],
    schedule_hbm: schedule.RpaSchedule,
    q_hbm: jax.Array,
    new_kv_hbm: jax.Array,
    kv_cache_hbm: jax.Array,
    lse_hbm: jax.Array | None,
    *,
    cfgs: configs.RpaConfigs,
    computer_cls: type[
        schedule.BaseMetadataComputer] = schedule.BaseMetadataComputer,
) -> tuple[jax.Array, jax.Array, jax.Array | None]:
    """Perform batched ragged paged attention with scheduler data.

  Args:
    cu_q_lens: [max_num_seqs + 1]. Cumulative sum of each sequence's query
      length. queries[a:b], keys[a:b], and values[a:b] where a=cu_q_lens[i] and
      b=cu_q_lens[i+1] represents q/k/v of sequence i.
    q_positions: [max_num_seqs]. Position of each sequence's first query token
      in the sequence's KV index space.
    global_kv_cache_lens: [max_num_seqs]. Cache length of each sequence before
      CP sharding; `RpaConfigs.local_kv_cache_len` narrows it to this call's
      share, and the ring also uses it to bound each rotated block by its
      source rank's share.
    kv_new_lens: [max_num_seqs]. New kv length of each sequence.
    cp_rank: [1]. This device's rank in the CP group; 0 without CP.
    new_kv_page_indices_refs: empty, or one [max_num_seqs * pages_per_seq] page
      table for the new KV, resolved in `copy_in` the way `page_indices` is. A
      tuple rather than an optional array so the kernel keeps one signature
      either way, the same trick `extra_scalars` uses in the scheduler.
    page_indices: [max_num_seqs * pages_per_seqs]. kv cache page table of each
      sequence.
    schedule_hbm: Output of scheduler kernel. It informs which: 1. seqs 2. q
      block 3. kv block that should be processed at a given step.
    q_hbm: [max_num_tokens, num_q_heads_per_kv_heads, cdiv(num_kv_heads,
      q_packing), q_packing, head_dim]. Output of q projection that has been
      pre-processed to align with existing kv cache data layout.
    new_kv_hbm: [max_num_tokens, cdiv(num_kv_heads * 2, kv_packing), kv_packing,
      head_dim]. Output of k & v projection that has been pre-processed to align
      with existing kv cache data layout.
    kv_cache_hbm: [num_pages, page_size, cdiv(num_kv_heads * 2, kv_packing),
      kv_packing, head_dim]. Stores existing kv cache data where k & vs are
      concatenated along num kv heads dim.
    lse_hbm: pre-allocated buffer for LSE output. None when return_lse=False.
    cfgs: Configuration of the kernel.

  Returns:
    out: [max_num_tokens, num_q_heads, head_dim]. Output of self attention.
    new_kv_cache: [num_pages, page_size, num_kv_heads // kv_packing, kv_packing,
      head_dim]. Result of new kv cache.
    lse_out: [max_num_tokens, num_q_heads] LSE values, or None.
  """
    return_lse = cfgs.serve.return_lse

    def ragged_paged_attention_pipeline(
        # Scalar prefetch.
        cu_q_lens_ref: jax.Ref,
        q_positions_ref: jax.Ref,
        global_kv_cache_lens_ref: jax.Ref,
        kv_new_lens_ref: jax.Ref,
        cp_rank_ref: jax.Ref,
        page_indices_ref: jax.Ref,
        new_kv_page_indices_refs: Sequence[jax.Ref],
        # Inputs.
        schedule_hbm_ref: schedule.RpaSchedule,
        q_hbm_ref: jax.Ref,
        new_kv_hbm_ref: jax.Ref,
        kv_cache_hbm_ref: jax.Ref,
        lse_hbm_ref: jax.Ref | None,
        # Outputs.
        o_hbm_ref: jax.Ref,
        o_kv_cache_hbm_ref: jax.Ref,
        o_lse_hbm_ref: jax.Ref | None = None,
    ):

        del o_kv_cache_hbm_ref
        if o_lse_hbm_ref is not None:
            del o_lse_hbm_ref

        q_alloc, kv_cache_alloc, o_alloc, lse_alloc = create_allocs(
            kv_cache_hbm_ref, q_hbm_ref, lse_hbm_ref, cfgs)

        actual_steps = schedule_hbm_ref.actual_steps[0]
        num_safe_step_iterations = pl.cdiv(actual_steps, cfgs.max_steps_ub)

        ring_enabled = cfgs.ring_enabled

        @pl.with_scoped(
            final_allocs=(q_alloc, kv_cache_alloc, o_alloc, lse_alloc),
            schedule_ref=computer_cls.get_rpa_schedule(cfgs).scratch_shapes(),
            dma_sem=pltpu.SemaphoreType.DMA((1, )),
            ring_sems=RingSems(
                dma_sems=pltpu.SemaphoreType.DMA((2, )),
                sync_sem=pltpu.SemaphoreType.REGULAR,
            ),
            scratches=(
                pltpu.VMEM(
                    cfgs.lm_scratch_shape,
                    dtype=cfgs.serve.dtype_out,
                ),  # m
                pltpu.VMEM(
                    cfgs.lm_scratch_shape,
                    dtype=cfgs.serve.dtype_out,
                ),  # l
                pltpu.VMEM(
                    cfgs.acc_scratch_shape,
                    dtype=cfgs.serve.dtype_out,
                ),  # acc
            ),
        )
        def _run(final_allocs, schedule_ref, dma_sem, ring_sems, scratches):
            ring = None
            if ring_enabled:
                ring = RingAttention(cfgs,
                                     ring_sems,
                                     kv_window_ref=final_allocs[1].window_ref)
                pass  # HANDSHAKE DISABLED (experiment)
            # Initialize Q to zeros to prevent NaN pollution.
            #
            # When a query block is partially filled, tail slots in uninitialized VMEM
            # may contain residual NaNs. In our implementation, invalid Q rows bypass
            # invalid row masking and produce NaNs in p_rowsum and later pollute
            # l_scratch and acc_scratch. Once l_scratch / acc_scratch becomes NaN,
            # subsequent iterations cannot recover (0 * NaN = NaN).

            # Initialize KV cache to zeros.
            # When perfomring p * v, we perform causal masking on lhs (p) by zeroing
            # out columns that should not be processed for a given row. Even if we
            # don't perform masking on rows of rhs (v), the output is still correct
            # since reuslt of multiplication will be zero thanks zero on lhs. However,
            # this assumption does not hold if a row of rhs has NaNs. To avoid this,
            # we initiallize scratch memory with non-zero values. Even if the scratch
            # memory is storing kv cache from previous step, as long as the data is
            # not NaNs, there will be no numeric concerns.

            scratches[0][...] = jnp.full_like(scratches[0], -jnp.inf)

            num_lanes = pltpu.get_tpu_info().num_lanes
            q_ref_flat = final_allocs[0].window_ref.bitcast(
                jnp.uint32).reshape(-1, num_lanes)
            kv_ref_flat = final_allocs[1].window_ref.bitcast(
                jnp.uint32).reshape(-1, num_lanes)

            # Explicitly zero out scratches and Q/KV buffers.
            for ref in (scratches[1], scratches[2], q_ref_flat, kv_ref_flat):
                ref[...] = jnp.zeros_like(ref)

            def execute_schedule_chunk(start_step, num_steps):
                # All reads are aligned to 128 and some extra steps are copied in the
                # process.
                aligned_start_step = (start_step // 128) * 128
                prefix_steps = start_step - aligned_start_step

                flat_hbm = jax.tree_util.tree_leaves(schedule_hbm_ref)
                flat_smem = jax.tree_util.tree_leaves(schedule_ref)
                dma_list = []
                for h, s in zip(flat_hbm, flat_smem):
                    if jax.typeof(h).memory_space == pltpu.HBM:
                        element_size = s.shape[0] // cfgs.max_steps_ub
                        read_size = element_size * (num_steps + prefix_steps)
                        read_size = utils.align_to(read_size, 1024)
                        read_size = jnp.minimum(read_size, s.shape[0])

                        src_off = element_size * aligned_start_step
                        src_off = pl.multiple_of(src_off, 128)

                        copy = pltpu.make_async_copy(
                            h.at[pl.ds(src_off, read_size)],
                            s.at[pl.ds(0, read_size)],
                            dma_sem.at[0],
                        )
                        copy.start()
                        dma_list.append(copy)
                jax.tree.map(lambda x: x.wait(), dma_list)
                pipeline_func = pltpu.emit_pipeline(
                    body=functools.partial(
                        rpa_body,
                        cfgs=cfgs,
                        cu_q_lens_ref=cu_q_lens_ref,
                        q_positions_ref=q_positions_ref,
                        global_kv_cache_lens_ref=global_kv_cache_lens_ref,
                        kv_new_lens_ref=kv_new_lens_ref,
                        cp_rank_ref=cp_rank_ref,
                        ring=ring,
                    ),
                    grid=(num_steps + prefix_steps, ),
                    in_specs=(q_alloc.spec, kv_cache_alloc.spec),
                    out_specs=(o_alloc.spec,
                               lse_alloc.spec if return_lse else None),
                )
                pipeline_func(
                    (q_hbm_ref, schedule_ref),
                    (kv_cache_hbm_ref, new_kv_hbm_ref, schedule_ref,
                     page_indices_ref,
                     new_kv_page_indices_refs[0]
                     if new_kv_page_indices_refs else None),
                    (o_hbm_ref, schedule_ref),
                    (lse_hbm_ref, schedule_ref) if return_lse else None,
                    scratches=(schedule_ref, ) + scratches,
                    allocations=final_allocs,
                )

            @pl.loop(0, num_safe_step_iterations)
            def loop_body(step_idx):
                start = step_idx * cfgs.max_steps_ub
                rem = actual_steps % cfgs.max_steps_ub
                last_step_size = jnp.where(rem == 0, cfgs.max_steps_ub, rem)
                is_last_step = step_idx == num_safe_step_iterations - 1
                size = jnp.where(is_last_step, last_step_size,
                                 cfgs.max_steps_ub)

                execute_schedule_chunk(start, size)

            if ring is not None and False:
                ring.drain_credits()

        _run()

    scalar_prefetches = (
        cu_q_lens,
        q_positions,
        global_kv_cache_lens,
        kv_new_lens,
        cp_rank,
        page_indices,
        new_kv_page_indices_refs,
    )
    num_scalar_prefetch = len(scalar_prefetches)
    num_active_scalers = len(jax.tree_util.tree_leaves(scalar_prefetches))

    out_shape = [q_hbm, kv_cache_hbm, lse_hbm if return_lse else None]

    schedule_leaves = len(jax.tree_util.tree_leaves(schedule_hbm))
    input_output_aliases = {
        num_active_scalers + schedule_leaves: 0,
        num_active_scalers + schedule_leaves + 2: 1,
    }
    if return_lse:
        input_output_aliases[num_active_scalers + schedule_leaves + 3] = (
            2  # lse_hbm -> out[2]
        )

    return pl.pallas_call(
        ragged_paged_attention_pipeline,
        out_shape=out_shape,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=num_scalar_prefetch,
            in_specs=[
                schedule_hbm.in_specs(),
                pl.BlockSpec(memory_space=pltpu.HBM),  # q_hbm_ref
                pl.BlockSpec(memory_space=pltpu.HBM),  # new_kv_hbm_ref
                pl.BlockSpec(memory_space=pltpu.HBM),  # kv_cache_hbm_ref
                pl.BlockSpec(memory_space=pltpu.HBM) if return_lse else None,
            ],
            out_specs=[
                pl.BlockSpec(memory_space=pltpu.HBM),  # aliased_o_hbm_ref
                pl.BlockSpec(
                    memory_space=pltpu.HBM),  # aliased_kv_cache_hbm_ref
                pl.BlockSpec(memory_space=pltpu.HBM) if return_lse else None,
            ],
        ),
        compiler_params=pltpu.CompilerParams(
            vmem_limit_bytes=cfgs.vmem_limit_bytes,
            disable_bounds_checks=True,
        ),
        input_output_aliases=input_output_aliases,
        name=get_kernel_name(cfgs),
        metadata=get_kernel_metadata(cfgs),
    )(
        cu_q_lens,
        q_positions,
        global_kv_cache_lens,
        kv_new_lens,
        cp_rank,
        page_indices,
        new_kv_page_indices_refs,
        schedule_hbm,
        q_hbm,
        new_kv_hbm,
        kv_cache_hbm,
        lse_hbm if return_lse else None,
    )
