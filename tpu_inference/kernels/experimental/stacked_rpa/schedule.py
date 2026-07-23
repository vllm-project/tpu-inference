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
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from tpu_inference.kernels.experimental.stacked_rpa import configs, utils


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class SmemWrapper:
    """Maps physical 1-D data into logical N-D representation."""

    data: Any
    shape: tuple[int, ...] = dataclasses.field(metadata=dict(static=True))

    @classmethod
    def create_shape_dtype(cls, shape):
        # Pad the flat buffer to a multiple of 1024 (the 1-D int32 DMA tile) so a
        # per-window HBM<->SMEM copy of a whole leaf is tile-aligned regardless of
        # batch_size / per-step stride. Logical indexing uses `shape`, so the
        # padding tail is never addressed.
        n = int(np.prod(shape))
        n_pad = -(-n // 1024) * 1024
        return cls(data=jax.ShapeDtypeStruct((n_pad, ), jnp.int32),
                   shape=shape)

    def _get_pos(self, indices):
        strides = pl.strides_from_shape(self.shape)
        assert len(strides) == len(indices)

        pos = 0
        for stride, idx in zip(strides, indices):
            pos += stride * idx
        return pos

    def __getitem__(self, indices):
        return self.data[self._get_pos(indices)]

    def __setitem__(self, indices, value):
        self.data[self._get_pos(indices)] = value


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class RpaSchedule:
    """Container for metadata arrays with integrated shape/spec logic."""

    s_idx: SmemWrapper  # [steps, batch]
    q_idx: SmemWrapper  # [steps, batch]
    k_idx: SmemWrapper  # [steps, batch]
    is_last_k: SmemWrapper  # [steps, batch]
    do_writeback: SmemWrapper  # [steps, batch]
    skip_mask: SmemWrapper  # [steps, batch]
    combine_span: SmemWrapper  # [steps, batch] (root cell: #cells to combine; else 0)
    is_final: SmemWrapper  # [steps, batch] (root cell: 1 if request completes here)
    dma_q: SmemWrapper  # [steps, batch, 2]
    dma_kv_cache: SmemWrapper  # [steps, batch, bkv_p_cache, 3]
    dma_kv_new: SmemWrapper  # [steps, batch, bkv_p_new, 4]
    actual_steps: Any  # [1]

    cfgs: configs.RpaConfigs = dataclasses.field(metadata=dict(static=True))

    @classmethod
    def create_shape_dtype(cls,
                           cfgs: configs.RpaConfigs,
                           steps: int | None = None):
        # `steps` sizes the leading (step) dimension. The HBM-resident schedule
        # is sized to cfgs.total_steps_ub (the full streamed schedule); the SMEM
        # scratch / consume window is sized to cfgs.max_steps_ub (one window).
        if steps is None:
            steps = cfgs.max_steps_ub

        idx_wrapper = SmemWrapper.create_shape_dtype((steps, cfgs.batch_size))

        return cls(
            s_idx=idx_wrapper,
            q_idx=idx_wrapper,
            k_idx=idx_wrapper,
            is_last_k=idx_wrapper,
            do_writeback=idx_wrapper,
            skip_mask=idx_wrapper,
            combine_span=idx_wrapper,
            is_final=idx_wrapper,
            dma_q=SmemWrapper.create_shape_dtype((steps, cfgs.batch_size, 2)),
            dma_kv_cache=SmemWrapper.create_shape_dtype(
                (steps, cfgs.batch_size, cfgs.bkv_p_cache, 3)),
            dma_kv_new=SmemWrapper.create_shape_dtype((
                steps,
                cfgs.batch_size,
                cfgs.bkv_p_new,
                cfgs.dma_kv_new_size,
            ), ),
            actual_steps=jax.ShapeDtypeStruct((1, ), jnp.int32),
            cfgs=cfgs,
        )

    @classmethod
    def create_hbm_shape_dtype(cls, cfgs: configs.RpaConfigs):
        """HBM schedule sized as num_windows copies of one padded SMEM window.

        Each SMEM window leaf is padded to a 1024 multiple (see SmemWrapper), and
        the per-window HBM<->SMEM DMA writes/reads window ``w`` at offset
        ``w * window_flat``. Sizing the HBM leaf as ``num_windows * window_flat``
        (rather than padding ``total_steps_ub * stride`` once) keeps every window
        offset 1024-tile-aligned even when ``max_steps_ub * stride`` is not.
        """
        win = cls.create_shape_dtype(cfgs, steps=cfgs.sched_window)
        num_windows = cfgs.total_steps_ub // cfgs.sched_window

        def grow(x):
            if isinstance(x, SmemWrapper):
                return SmemWrapper(
                    data=jax.ShapeDtypeStruct(
                        (x.data.shape[0] * num_windows, ), x.data.dtype),
                    shape=x.shape,
                )
            return x

        return cls(
            s_idx=grow(win.s_idx),
            q_idx=grow(win.q_idx),
            k_idx=grow(win.k_idx),
            is_last_k=grow(win.is_last_k),
            do_writeback=grow(win.do_writeback),
            skip_mask=grow(win.skip_mask),
            combine_span=grow(win.combine_span),
            is_final=grow(win.is_final),
            dma_q=grow(win.dma_q),
            dma_kv_cache=grow(win.dma_kv_cache),
            dma_kv_new=grow(win.dma_kv_new),
            actual_steps=win.actual_steps,
            cfgs=cfgs,
        )

    def get_dma_kv_cache(
        self,
        step: jax.typing.ArrayLike,
        batch_idx: jax.typing.ArrayLike,
        page_idx: jax.typing.ArrayLike,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        # 0: src_hbm, 1: dst_vmem, 2: size
        src_off = self.dma_kv_cache[step, batch_idx, page_idx, 0]
        dst_off = self.dma_kv_cache[step, batch_idx, page_idx, 1]
        sz = self.dma_kv_cache[step, batch_idx, page_idx, 2]
        return src_off, dst_off, sz

    def get_dma_fetch_kv_new(
        self,
        step: jax.typing.ArrayLike,
        batch_idx: jax.typing.ArrayLike,
        page_idx: jax.typing.ArrayLike,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        # 0: fetch_hbm, 1: fetch_vmem, 4: packed_dma_valid (bit 0)
        src_hbm = self.dma_kv_new[step, batch_idx, page_idx, 0]
        dst_vmem = self.dma_kv_new[step, batch_idx, page_idx, 1]
        sz = self.dma_kv_new[step, batch_idx, page_idx, 4] & 1
        return src_hbm, dst_vmem, sz

    def get_dma_update_kv_new(
        self,
        step: jax.typing.ArrayLike,
        batch_idx: jax.typing.ArrayLike,
        page_idx: jax.typing.ArrayLike,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        # 2: wb_hbm, 3: wb_vmem, 4: packed_dma_valid (bit 1)
        dst_hbm = self.dma_kv_new[step, batch_idx, page_idx, 2]
        src_vmem = self.dma_kv_new[step, batch_idx, page_idx, 3]
        sz = (self.dma_kv_new[step, batch_idx, page_idx, 4] >> 1) & 1
        return dst_hbm, src_vmem, sz

    def get_dma_q(
            self, step: jax.typing.ArrayLike,
            batch_idx: jax.typing.ArrayLike) -> tuple[jax.Array, jax.Array]:
        # 0: src_hbm, 1: size
        src_hbm = self.dma_q[step, batch_idx, 0]
        sz = self.dma_q[step, batch_idx, 1]
        return src_hbm, sz

    def scratch_shapes(self):
        """Returns a Pytree of SMEM scratch memory."""

        return jax.tree.map(
            lambda x: pltpu.SMEM(x.shape, x.dtype),
            self,
        )

    def in_specs(self):
        """Returns a Pytree of input BlockSpecs."""

        def wrapper(x):
            if x.size == 1:
                return pl.BlockSpec(memory_space=pltpu.SMEM)
            else:
                # Since we use maximum upper bound when allocating scheduler data,
                # it is not feasible to use scalar prefetch and fetch entire scheduler
                # data into the kernel. Instead, we stored it to HBM first and perform
                # dynamic sized DMA inside the kernel using actual number of steps.
                return pl.BlockSpec(memory_space=pltpu.HBM)

        return jax.tree.map(wrapper, self)

    def out_specs(self):
        """Returns a Pytree of output BlockSpecs."""

        return jax.tree.map(
            lambda x: pl.BlockSpec(memory_space=pltpu.HBM),
            self,
        )


def compute_metadata(
    cu_q_lens_ref: jax.Ref,
    kv_lens_ref: jax.Ref,
    distribution_ref: jax.Ref,
    visibility_bounds_ref: jax.Ref,
    schedule: RpaSchedule,
    lane_lengths_ref: jax.Ref,
    page_indices_hbm_ref: jax.Ref,
    seq_page_table_ref: jax.Ref,
    page_dma_sem: jax.Ref,
    *,
    cfgs: configs.RpaConfigs,
    window_lo: int = 0,
    sorted_seq_idx_ref: jax.Ref | None = None,
):
    """Fill metadata using triple nested loop of seq->q->k loop.

    The schedule SMEM scratch holds one window of ``cfgs.max_steps_ub`` (= W)
    steps. ``window_lo`` is the global step index of the first row of this
    window; only steps in ``[window_lo, window_lo + W)`` are written (at the
    window-local index ``step - window_lo``). Steps outside the window still
    advance the per-lane counter (so the assignment is reproduced identically
    across window re-runs) but their writes are suppressed.
    """
    w_size = cfgs.sched_window

    @jax.named_scope("k_loop")
    def k_loop(
        k_idx,
        step,
        *,
        target_lane,
        s_idx,
        q_idx,
        q_end,
        q_src,
        q_sz_task,
        k_len,
        q_len,
        end_k_idx,
    ):
        in_window = jnp.logical_and(step >= window_lo, step
                                    < window_lo + w_size)
        local = jnp.clip(step - window_lo, 0, w_size - 1)

        kv_len_start = k_idx * cfgs.bkv_sz
        kv_p_start = k_idx * cfgs.bkv_p
        if cfgs.use_window_anchor:
            # Anchor the (single) decode block at the page-aligned window start
            # instead of the global k_idx*bkv_sz grid, so it reads exactly one
            # window. The consume mask (kernel.py) recomputes the same anchor.
            anchor_tok = utils.window_anchor_tok(k_len, q_len,
                                                 cfgs.model.sliding_window,
                                                 cfgs.serve.page_size_log2)
            kv_len_start = anchor_tok + k_idx * cfgs.bkv_sz
            kv_p_start = (
                anchor_tok >> cfgs.serve.page_size_log2) + k_idx * cfgs.bkv_p
        kv_left = k_len - kv_len_start
        if cfgs.update_kv_cache:
            kv_left_frm_cache = jnp.maximum(kv_left - q_len, 0)
        else:
            kv_left_frm_cache = kv_left

        kv_left_frm_new = kv_left - kv_left_frm_cache
        bkv_sz_cache = jnp.minimum(kv_left_frm_cache, cfgs.bkv_sz)
        new_sz = jnp.minimum(cfgs.bkv_sz - bkv_sz_cache, kv_left_frm_new)

        # Writeback logic: each new k block is written back by the first q block
        # that attends to it.
        q_wb = jnp.maximum(0, (kv_len_start - (k_len - q_len))) // cfgs.bq_sz
        do_writeback = jnp.where(
            cfgs.update_kv_cache & (new_sz > 0) & (q_idx == q_wb), 1, 0)

        min_q_pos = k_len - q_len + q_idx * cfgs.bq_sz
        skip_mask = jnp.where(
            (not cfgs.disable_skip_mask)
            & (not cfgs.has_visibility)
            & (cfgs.model.sliding_window is None)
            & ((k_idx + 1) * cfgs.bkv_sz - 1 <= min_q_pos),
            1,
            0,
        )

        def fill_dma_kv_new(i, dst_vmem, dma_sz, slot_start):
            cache_pages = pl.cdiv(bkv_sz_cache, cfgs.serve.page_size)
            hbm_token_idx_base = q_end - kv_left_frm_new
            new_tok_offset = hbm_token_idx_base % cfgs.serve.page_size
            num_pages_to_fetch = jnp.where(
                new_sz > 0,
                (new_tok_offset + new_sz - 1) // cfgs.serve.page_size + 1,
                0,
            )

            fetch_dma_valid = jnp.where(i < num_pages_to_fetch, 1, 0)
            new_page_start = (hbm_token_idx_base -
                              new_tok_offset) + i * cfgs.serve.page_size
            fetch_vmem = (cache_pages + i) * cfgs.serve.page_size

            p_idx = jnp.minimum(
                (kv_len_start + slot_start) >> cfgs.serve.page_size_log2,
                cfgs.serve.pages_per_seq - 1,
            )
            # Physical KV-cache page (page_indices folded in at build time):
            # the consume kernel uses dst_hbm directly, no page_indices lookup.
            dst_hbm = seq_page_table_ref[p_idx]

            packed_dma_valid = fetch_dma_valid | (
                jnp.where(dma_sz > 0, 1, 0) << 1)

            schedule.dma_kv_new[local, target_lane, i, 0] = new_page_start
            schedule.dma_kv_new[local, target_lane, i, 1] = fetch_vmem
            schedule.dma_kv_new[local, target_lane, i, 2] = dst_hbm
            schedule.dma_kv_new[local, target_lane, i, 3] = slot_start
            schedule.dma_kv_new[local, target_lane, i, 4] = packed_dma_valid

        @pl.when(in_window)
        def _emit_writes():
            schedule.s_idx[local, target_lane] = s_idx
            schedule.q_idx[local, target_lane] = q_idx
            schedule.k_idx[local, target_lane] = k_idx

            is_last_k = jnp.where(k_idx == end_k_idx - 1, 1, 0)
            schedule.is_last_k[local, target_lane] = is_last_k

            schedule.dma_q[local, target_lane, 0] = q_src
            schedule.dma_q[local, target_lane, 1] = q_sz_task

            # TODO(perf, large-page-overcompute): with page_size > 128 the last
            # (partial) cache page of a request is DMA'd AND flash-computed in
            # full, even though only (kv_len % page_size) tokens are real. For
            # large pages / short contexts this wastes up to ~page_size of KV DMA
            # plus a full flash block of compute per request (measured: page2048
            # ~0.65x at 1k ctx vs page512). A proper fix must trim BOTH sides:
            #   (1) DMA -- store the real 128-rounded per-page token count here
            #       (instead of the 0/1 valid bit) and have bref_override DMA
            #       exactly that for the boundary page; and
            #   (2) flash compute -- dispatch full vs partial blocks per step and
            #       sub-block the bkv axis, skipping pages fully beyond every
            #       cell's KV length (identity for online softmax).
            # Both were prototyped separately; the flash sub-block skip's
            # control-flow + lost V/softmax overlap outweighed the launch-bound
            # decode-matmul savings, so it was reverted. Resolve holistically.
            # page_size stays configurable (128..2048); this only wastes work, not
            # correctness.
            for i in range(cfgs.bkv_p_cache):
                dst_vmem = i << cfgs.serve.page_size_log2
                dma_sz = kv_left_frm_cache - dst_vmem
                dma_sz = jnp.clip(dma_sz, 0, cfgs.serve.page_size)
                # Physical KV-cache page via the per-seq page table (folded in at
                # build time). within-seq logical page = kv_p_start + i.
                src_hbm = seq_page_table_ref[jnp.minimum(
                    kv_p_start + i, cfgs.serve.pages_per_seq - 1)]
                dma_valid = jnp.where(dma_sz > 0, 1, 0)
                schedule.dma_kv_cache[local, target_lane, i, 0] = src_hbm
                schedule.dma_kv_cache[local, target_lane, i, 1] = dst_vmem
                schedule.dma_kv_cache[local, target_lane, i, 2] = dma_valid

            schedule.do_writeback[local, target_lane] = do_writeback
            schedule.skip_mask[local, target_lane] = skip_mask

            if cfgs.mode == configs.RpaCase.DECODE and cfgs.decode_q_len > 1:
                # Multi-token decode: the new tokens sit at the cache boundary and
                # span <= bkv_p_new pages. Anchor the fill at the boundary page so
                # both the staging fetch and the writeback land in the small
                # bkv_p_new leaf (fill_dma_kv_new keys the fetch on the loop index
                # and the writeback on slot_start). This avoids the SMEM blow-up of
                # the full MIXED loop, whose leaf scales with the (large) bkv.
                base_page = bkv_sz_cache // cfgs.serve.page_size
                for i in range(cfgs.bkv_p_new):
                    slot_start = (base_page + i) * cfgs.serve.page_size
                    slot_end = slot_start + cfgs.serve.page_size
                    dst_vmem = jnp.maximum(slot_start, bkv_sz_cache)
                    end_in_slot = jnp.minimum(slot_end, bkv_sz_cache + new_sz)
                    dma_sz = jnp.maximum(0, end_in_slot - dst_vmem)
                    fill_dma_kv_new(i, dst_vmem, dma_sz, slot_start)
            elif cfgs.bkv_p_new < cfgs.bkv_p:
                # Decode path (single token)
                assert cfgs.bkv_p_new == 1
                slot_start = (bkv_sz_cache //
                              cfgs.serve.page_size) * cfgs.serve.page_size
                fill_dma_kv_new(0, bkv_sz_cache, new_sz, slot_start)
            else:
                iters = max(cfgs.bkv_p, cfgs.bkv_p_new)
                for i in range(iters):
                    slot_start = i * cfgs.serve.page_size
                    slot_end = slot_start + cfgs.serve.page_size
                    dst_vmem = jnp.maximum(slot_start, bkv_sz_cache)
                    end_in_slot = jnp.minimum(slot_end, bkv_sz_cache + new_sz)
                    dma_sz = jnp.maximum(0, end_in_slot - dst_vmem)
                    fill_dma_kv_new(i, dst_vmem, dma_sz, slot_start)

        return step + 1

    @jax.named_scope("q_loop")
    def q_loop(q_idx, _, *, s_idx, q_start, q_end, k_len, q_len, num_k):
        # TODO(perf, load-imbalance): all k-blocks of a (seq, q-block) are pinned
        # to one lane (k_loop runs start_k..end_k on `target_lane`) and a single
        # k-run cannot be split across lanes. With a skewed batch (e.g. one very
        # long-context seq + many short seqs), total_steps is dominated by the
        # longest seq's k-run, so lanes are poorly balanced: in the tail steps
        # only the long seq's lane does real work while the other lanes execute
        # masked no-op steps (wasted cycles, lower MXU utilization). Correctness
        # is unaffected. A future scheduler could split long k-runs across lanes
        # to balance work and reclaim the idle lanes.
        target_lane = 0
        min_len = lane_lengths_ref[0]
        for b in range(1, cfgs.batch_size):
            is_better = lane_lengths_ref[b] < min_len
            target_lane = jnp.where(is_better, b, target_lane)
            min_len = jnp.where(is_better, lane_lengths_ref[b], min_len)

        curr_ptr = lane_lengths_ref[target_lane]
        q_src = q_start + q_idx * cfgs.bq_sz
        q_sz_task = jnp.clip(q_end - q_src, 0, cfgs.bq_sz)

        start_k_idx = 0
        if cfgs.has_visibility:
            min_vis_start = visibility_bounds_ref[s_idx, q_idx, 0]
            max_vis_end = visibility_bounds_ref[s_idx, q_idx, 1]
            start_k_idx = jnp.clip(min_vis_start, 0, k_len) // cfgs.bkv_sz
            end_k_idx = jnp.minimum(
                num_k,
                jnp.maximum(0, max_vis_end) // cfgs.bkv_sz + 1,
            )
        elif (sliding_window := cfgs.model.sliding_window) is not None:
            sw_start_idx = k_len - q_len + q_idx * cfgs.bq_sz - sliding_window + 1
            start_k_idx = jnp.maximum(0, sw_start_idx) // cfgs.bkv_sz
            end_k_idx_causal = (k_len - q_len + q_idx * cfgs.bq_sz +
                                q_sz_task - 1) // cfgs.bkv_sz + 1
            end_k_idx = jnp.minimum(num_k, end_k_idx_causal)
        else:
            end_k_idx_causal = (k_len - q_len + q_idx * cfgs.bq_sz +
                                q_sz_task - 1) // cfgs.bkv_sz + 1
            end_k_idx = jnp.minimum(num_k, end_k_idx_causal)

        k_loop_fn = functools.partial(
            k_loop,
            target_lane=target_lane,
            s_idx=s_idx,
            q_idx=q_idx,
            q_end=q_end,
            q_src=q_src,
            q_sz_task=q_sz_task,
            k_len=k_len,
            q_len=q_len,
            end_k_idx=end_k_idx,
        )
        lane_lengths_ref[target_lane] = jax.lax.fori_loop(
            start_k_idx, end_k_idx, k_loop_fn, curr_ptr)

    @jax.named_scope("seq_loop")
    def seq_loop(s_idx, _):
        q_start = cu_q_lens_ref[s_idx]
        q_end = cu_q_lens_ref[s_idx + 1]
        k_len = kv_lens_ref[s_idx]
        q_len = q_end - q_start

        num_q = pl.cdiv(q_len, cfgs.bq_sz)
        num_k = pl.cdiv(k_len, cfgs.bkv_sz)

        # Stream this sequence's block-table slice (page_indices) HBM -> SMEM so the
        # k_loop can fold the logical->physical page lookup into the schedule. Only
        # one seq's slice (pages_per_seq entries) is resident at a time.
        pps_pad = cfgs.seq_page_table_size
        # page_indices is passed in as a 1-D [num_seqs * pps_pad] padded array; copy
        # this seq's contiguous padded slab. pps_pad is a multiple of 128, so the
        # offset s_idx*pps_pad is 128-aligned -- this avoids the dim-0 tile-8
        # constraint that a [num_seqs, pps_pad] 2-D row-slice imposes (which fails
        # to compile for num_seqs > 8 with a dynamic s_idx).
        seq_off = pl.multiple_of(s_idx * pps_pad, pps_pad)
        page_copy = pltpu.make_async_copy(
            page_indices_hbm_ref.at[pl.ds(seq_off, pps_pad)],
            seq_page_table_ref.at[pl.ds(0, pps_pad)],
            page_dma_sem.at[0],
        )
        page_copy.start()
        page_copy.wait()

        q_loop_fn = functools.partial(
            q_loop,
            s_idx=s_idx,
            q_start=q_start,
            q_end=q_end,
            k_len=k_len,
            q_len=q_len,
            num_k=num_k,
        )

        jax.lax.fori_loop(0, num_q, q_loop_fn, None)

    if cfgs.is_stacked and cfgs.dense_pack:
        # Dense cross-step packing. A single global cursor `g` lays every
        # (seq, q-block) unit's k-blocks contiguously across the n cells,
        # wrapping over step boundaries (~100% cell utilization). A unit may
        # straddle a step boundary, so within each step it spans a contiguous
        # cell range; per step we emit that group's (root, span) plus is_final
        # (1 iff the unit's last block is in this step). The consume side merges
        # a straddling unit across steps via a single cross-step carry slot.
        #
        # DECODE has one q-block per sequence (num_q == 1), reproducing the
        # single-unit-per-seq behavior. MIXED iterates num_q > 1: each
        # (seq, q_idx) is an independent unit with its own causal k-range, its
        # own combine group, and its own output token range (q_src).
        #
        # TODO(perf, adaptive-combine): the consume-side combine
        # (_dense_combine_and_store) currently does a *vectorized* normalize+store
        # over ALL n cells on every completion step. That is optimal when many
        # requests complete per step (short / high-concurrency: outputs ~= n), but
        # wastes ~(n-1)/n work when few complete (long context: one span==n group
        # yields a single output). Measured (SEQ_ALONG_LANE, bs=32, hd64, TPU7x)
        # vs the earlier per-root store: 1k/bkv1024 improved 0.918x -> 0.976x, but
        # 64k/bkv8192 regressed ~0.99x -> 0.977x (~1-2% slower long context).
        # Fix: emit a per-step completion count here (number of cells with
        # is_final==1) so the consume can pick adaptively -- vectorized all-cell
        # store when the count is high, per-root store of only the output cells
        # when it is low -- getting the best of both with no regression.
        assert sorted_seq_idx_ref is not None
        n = cfgs.batch_size
        pps_pad = cfgs.seq_page_table_size

        def _meta_write(step, cell, span, is_final, active):
            cin = jnp.logical_and(step >= window_lo, step < window_lo + w_size)
            clocal = jnp.clip(step - window_lo, 0, w_size - 1)

            @pl.when(jnp.logical_and(cin, active))
            def _w():
                schedule.combine_span[clocal, cell] = span
                schedule.is_final[clocal, cell] = is_final

        @jax.named_scope("dense_seq_loop")
        def dense_seq_loop(order_idx, g):
            s_idx = sorted_seq_idx_ref[order_idx]
            seq_off = pl.multiple_of(s_idx * pps_pad, pps_pad)
            page_copy = pltpu.make_async_copy(
                page_indices_hbm_ref.at[pl.ds(seq_off, pps_pad)],
                seq_page_table_ref.at[pl.ds(0, pps_pad)],
                page_dma_sem.at[0],
            )
            page_copy.start()
            page_copy.wait()

            q_start = cu_q_lens_ref[s_idx]
            q_end = cu_q_lens_ref[s_idx + 1]
            k_len = kv_lens_ref[s_idx]
            q_len = q_end - q_start
            num_k = pl.cdiv(k_len, cfgs.bkv_sz)
            num_q = pl.cdiv(q_len, cfgs.bq_sz)

            def dense_q_loop(q_idx, g):
                # Place one (s_idx, q_idx) unit on the global cursor `g`; return
                # the advanced cursor. k-range mirrors the data-parallel q_loop
                # (causal / sliding-window / visibility), parameterized by q_idx.
                q_src = q_start + q_idx * cfgs.bq_sz
                q_sz_task = jnp.clip(q_end - q_src, 0, cfgs.bq_sz)

                start_k_idx = 0
                if cfgs.has_visibility:
                    min_vis_start = visibility_bounds_ref[s_idx, q_idx, 0]
                    max_vis_end = visibility_bounds_ref[s_idx, q_idx, 1]
                    start_k_idx = jnp.clip(min_vis_start, 0,
                                           k_len) // cfgs.bkv_sz
                    end_k_idx = jnp.minimum(
                        num_k,
                        jnp.maximum(0, max_vis_end) // cfgs.bkv_sz + 1)
                elif (sliding_window := cfgs.model.sliding_window) is not None:
                    if cfgs.use_window_anchor:
                        # Single block anchored at the page-aligned window start.
                        anchor_tok = utils.window_anchor_tok(
                            k_len, q_len, sliding_window,
                            cfgs.serve.page_size_log2)
                        start_k_idx = 0
                        end_k_idx = pl.cdiv(k_len - anchor_tok, cfgs.bkv_sz)
                    else:
                        sw_start_idx = (k_len - q_len + q_idx * cfgs.bq_sz -
                                        sliding_window + 1)
                        start_k_idx = jnp.maximum(0,
                                                  sw_start_idx) // cfgs.bkv_sz
                        end_k_idx = jnp.minimum(
                            num_k,
                            (k_len - q_len + q_idx * cfgs.bq_sz + q_sz_task -
                             1) // cfgs.bkv_sz + 1,
                        )
                else:
                    end_k_idx = jnp.minimum(
                        num_k,
                        (k_len - q_len + q_idx * cfgs.bq_sz + q_sz_task - 1) //
                        cfgs.bkv_sz + 1,
                    )

                num_k_seq = jnp.maximum(end_k_idx - start_k_idx, 0)
                p0 = g
                p1 = g + num_k_seq

                def dk_loop(k_idx, _):
                    gg = p0 + (k_idx - start_k_idx)
                    k_loop(
                        k_idx,
                        gg // n,
                        target_lane=gg % n,
                        s_idx=s_idx,
                        q_idx=q_idx,
                        q_end=q_end,
                        q_src=q_src,
                        q_sz_task=q_sz_task,
                        k_len=k_len,
                        q_len=q_len,
                        end_k_idx=end_k_idx,
                    )
                    return None

                jax.lax.fori_loop(start_k_idx, end_k_idx, dk_loop, None)

                s_first = p0 // n
                s_last = jnp.maximum(p1 - 1, p0) // n

                def step_meta(s, _):
                    lo = jnp.maximum(p0, s * n)
                    hi = jnp.minimum(p1, (s + 1) * n)
                    root = lo - s * n
                    span = hi - lo
                    is_final = jnp.where(s == s_last, 1, 0)
                    _meta_write(s, root, span, is_final, num_k_seq > 0)
                    return None

                @pl.when(num_k_seq > 0)
                def _emit_group_meta():
                    jax.lax.fori_loop(s_first, s_last + 1, step_meta, None)

                return p1

            # Units with an empty k-range (num_k_seq == 0) leave `g` unchanged
            # and emit nothing, so they are skipped without stranding a cell.
            return jax.lax.fori_loop(0, num_q, dense_q_loop, g)

        start_seq_idx, end_seq_idx = cfgs.mode.get_range(distribution_ref)
        final_g = jax.lax.fori_loop(start_seq_idx, end_seq_idx, dense_seq_loop,
                                    jnp.int32(0))
        lane_lengths_ref[0] = pl.cdiv(final_g, n)
        return

    if cfgs.is_stacked:
        # Transposed stacked packing (DECODE only), MULTIPLE requests per step.
        # Requests are processed longest-first with a dense (step, cell) cursor:
        #   - SHORT (num_k < n): placed contiguously in the free cells of the
        #     current step (several short requests share a step); combined at that
        #     step over its num_k cells.
        #   - LONG (num_k >= n): occupies WHOLE steps (starts at a fresh step,
        #     ends step-aligned) so no other request shares its cells; combined at
        #     its final step over all n cells.
        # Unified dense placement: block p of a request goes to global cell
        # (base_global + p) -> step=g//n, cell=g%n. `base_global` is bumped to a
        # fresh step when a short request would overflow the step or a long
        # request needs alignment. This "short-within-step / long-whole-step" rule
        # guarantees no cross-request stranding of a cell's partial.
        assert sorted_seq_idx_ref is not None
        n = cfgs.batch_size
        pps_pad = cfgs.seq_page_table_size

        def write_combine_span(cstep, root, span, active):
            cin = jnp.logical_and(cstep >= window_lo, cstep
                                  < window_lo + w_size)
            clocal = jnp.clip(cstep - window_lo, 0, w_size - 1)

            @pl.when(jnp.logical_and(cin, active))
            def _w():
                schedule.combine_span[clocal, root] = span

        @jax.named_scope("sk_seq_loop")
        def sk_seq_loop(order_idx, carry):
            step_base, cell_cursor = carry
            s_idx = sorted_seq_idx_ref[order_idx]

            seq_off = pl.multiple_of(s_idx * pps_pad, pps_pad)
            page_copy = pltpu.make_async_copy(
                page_indices_hbm_ref.at[pl.ds(seq_off, pps_pad)],
                seq_page_table_ref.at[pl.ds(0, pps_pad)],
                page_dma_sem.at[0],
            )
            page_copy.start()
            page_copy.wait()

            q_start = cu_q_lens_ref[s_idx]
            q_end = cu_q_lens_ref[s_idx + 1]
            k_len = kv_lens_ref[s_idx]
            q_len = q_end - q_start
            num_k = pl.cdiv(k_len, cfgs.bkv_sz)

            q_idx = 0
            q_src = q_start
            q_sz_task = jnp.clip(q_end - q_src, 0, cfgs.bq_sz)

            # k-range selection: mirror q_loop (visibility / sliding-window / causal).
            start_k_idx = 0
            if cfgs.has_visibility:
                min_vis_start = visibility_bounds_ref[s_idx, q_idx, 0]
                max_vis_end = visibility_bounds_ref[s_idx, q_idx, 1]
                start_k_idx = jnp.clip(min_vis_start, 0, k_len) // cfgs.bkv_sz
                end_k_idx = jnp.minimum(
                    num_k,
                    jnp.maximum(0, max_vis_end) // cfgs.bkv_sz + 1)
            elif (sliding_window := cfgs.model.sliding_window) is not None:
                if cfgs.use_window_anchor:
                    # Single block anchored at the page-aligned window start.
                    anchor_tok = utils.window_anchor_tok(
                        k_len, q_len, sliding_window,
                        cfgs.serve.page_size_log2)
                    start_k_idx = 0
                    end_k_idx = pl.cdiv(k_len - anchor_tok, cfgs.bkv_sz)
                else:
                    sw_start_idx = (k_len - q_len + q_idx * cfgs.bq_sz -
                                    sliding_window + 1)
                    start_k_idx = jnp.maximum(0, sw_start_idx) // cfgs.bkv_sz
                    end_k_idx = jnp.minimum(
                        num_k,
                        (k_len - q_len + q_idx * cfgs.bq_sz + q_sz_task - 1) //
                        cfgs.bkv_sz + 1,
                    )
            else:
                end_k_idx = jnp.minimum(
                    num_k,
                    (k_len - q_len + q_idx * cfgs.bq_sz + q_sz_task - 1) //
                    cfgs.bkv_sz + 1,
                )

            num_k_seq = jnp.maximum(end_k_idx - start_k_idx, 0)
            is_long = num_k_seq >= n

            # Bump to a fresh step if a short request overflows the current step,
            # or a long request needs step alignment.
            base_raw = step_base * n + cell_cursor
            need_new = jnp.where(is_long, cell_cursor > 0,
                                 cell_cursor + num_k_seq > n)
            base_global = jnp.where(need_new,
                                    pl.cdiv(base_raw, n) * n, base_raw)

            def sk_k_loop(k_idx, _):
                p = k_idx - start_k_idx
                g = base_global + p
                k_loop(
                    k_idx,
                    g // n,  # step
                    target_lane=g % n,  # cell
                    s_idx=s_idx,
                    q_idx=q_idx,
                    q_end=q_end,
                    q_src=q_src,
                    q_sz_task=q_sz_task,
                    k_len=k_len,
                    q_len=q_len,
                    end_k_idx=end_k_idx,
                )
                return None

            jax.lax.fori_loop(start_k_idx, end_k_idx, sk_k_loop, None)

            # Segmented-combine metadata: root cell + #cells for this request.
            last_global = base_global + jnp.maximum(num_k_seq - 1, 0)
            combine_step = last_global // n
            root_cell = jnp.where(is_long, 0, base_global % n)
            span = jnp.where(is_long, n, num_k_seq)
            write_combine_span(combine_step, root_cell, span, num_k_seq > 0)

            # Advance cursor: short stays dense (mid-step ok); long ends step-aligned.
            next_global = jnp.where(
                is_long,
                pl.cdiv(base_global + num_k_seq, n) * n,
                base_global + num_k_seq,
            )
            return next_global // n, next_global % n

        start_seq_idx, end_seq_idx = cfgs.mode.get_range(distribution_ref)
        step_base, cell_cursor = jax.lax.fori_loop(
            start_seq_idx,
            end_seq_idx,
            sk_seq_loop,
            (jnp.int32(0), jnp.int32(0)),
        )
        total_steps = step_base + jnp.where(cell_cursor > 0, 1, 0)
        lane_lengths_ref[0] = total_steps
        return

    start_seq_idx, end_seq_idx = cfgs.mode.get_range(distribution_ref)
    jax.lax.fori_loop(start_seq_idx, end_seq_idx, seq_loop, None)


def rpa_metadata_schedule_kernel(
    ## Scalar prefetch.
    cu_q_lens_ref: jax.Ref,
    kv_lens_ref: jax.Ref,
    distribution_ref: jax.Ref,
    visibility_bounds_ref: jax.Ref,
    # HBM input (streamed per-seq into SMEM during the build).
    page_indices_hbm_ref: jax.Ref,
    # Outputs.
    schedule_hbm_ref: RpaSchedule,
    # Scratch.
    schedule_ref: RpaSchedule,
    lane_lengths_ref: jax.Ref,
    dma_sem: jax.Ref,
    seq_page_table_ref: jax.Ref,
    page_dma_sem: jax.Ref,
    *,
    cfgs: configs.RpaConfigs,
    sorted_seq_idx_ref: jax.Ref | None = None,
):
    """Generates the HBM-to-VMEM DMA schedule.

    This kernel:
    1. Iterates through each (potentially ragged) sequence
    2. Breaks Queries (Q) and Key-Values (KV) into blocks (bq_sz, bkv_sz).
    3. Assigns tasks to 'lanes' (TPU batch items) based on current lane occupancy
      to ensure balanced execution across the batch dimension.
    4. Encodes DMA offsets:
      - dma_q: HBM start index and size for Query blocks.
      - dma_kv_cache: Paged indices for existing KV tokens.
      - dma_kv_new: offsets for new tokens being added to the cache.
      - do_writeback: boolean flag indicating if a block should be flushed to
        HBM (ie does this block contain new tokens to add to KV cache).

    Args:
      cu_q_lens_ref: [max_num_seqs + 1]. Cumulative sum of each sequence's query
        length. queries[a:b], keys[a:b], and values[a:b] where a=cu_q_lens[i] and
        b=cu_q_lens[i+1] represents q/k/v of sequence i.
      kv_lens_ref: [max_num_seqs]. Existing kv cache length of each sequence.
      distribution_ref: [3]. Cumulative sum of number of decode, prefill, and
        mixed
      schedule_hbm_ref: HBM memory that will store output of the kernel.
      schedule_ref: Scratch memory where schedule results gets written.
      lane_lengths_ref: Scratch memory that keeps track of number of steps for
        each batch lane.
      dma_sem: Semaphore used for writing scheduler output to HBM.
      cfgs: Configuration of the kernel.
    """

    w_size = cfgs.sched_window

    flat_hbm = jax.tree_util.tree_leaves(schedule_hbm_ref)
    flat_smem = jax.tree_util.tree_leaves(schedule_ref)

    # Mask one full SMEM window (window-local steps [0, w_size)); steps not
    # written by compute_metadata for this window stay masked (s_idx == -1).
    @jax.named_scope("mask_window")
    def mask_window(step, _):
        for b_idx in range(cfgs.batch_size):
            schedule_ref.s_idx[step, b_idx] = -1
            schedule_ref.q_idx[step, b_idx] = 0
            schedule_ref.k_idx[step, b_idx] = 0
            schedule_ref.is_last_k[step, b_idx] = 0
            schedule_ref.do_writeback[step, b_idx] = 0
            schedule_ref.skip_mask[step, b_idx] = 0
            schedule_ref.combine_span[step, b_idx] = 0
            schedule_ref.is_final[step, b_idx] = 0
            schedule_ref.dma_q[step, b_idx, 0] = 0
            schedule_ref.dma_q[step, b_idx, 1] = 0
            for i in range(cfgs.bkv_p_cache):
                schedule_ref.dma_kv_cache[step, b_idx, i, 0] = 0
                schedule_ref.dma_kv_cache[step, b_idx, i, 1] = 0
                schedule_ref.dma_kv_cache[step, b_idx, i, 2] = 0
            for i in range(cfgs.bkv_p_new):
                for j in range(cfgs.dma_kv_new_size):
                    schedule_ref.dma_kv_new[step, b_idx, i, j] = 0

    # Build the schedule one window at a time and stream each window to HBM.
    # The (q,k) -> lane assignment is a deterministic function of the scalar
    # prefetch inputs, so resetting lane_lengths and re-running it reproduces the
    # same per-lane `step` values on every window; compute_metadata only writes
    # the rows whose global step falls inside the current window. `w` may be a
    # static int (window 0) or a traced index (the fori_loop over the rest).
    def build_window(w):
        window_lo = w * w_size

        for b_idx in range(cfgs.batch_size):
            lane_lengths_ref[b_idx] = 0

        jax.lax.fori_loop(0, w_size, mask_window, None)

        compute_metadata(
            cu_q_lens_ref,
            kv_lens_ref,
            distribution_ref,
            visibility_bounds_ref,
            schedule_ref,
            lane_lengths_ref,
            page_indices_hbm_ref,
            seq_page_table_ref,
            page_dma_sem,
            cfgs=cfgs,
            window_lo=window_lo,
            sorted_seq_idx_ref=sorted_seq_idx_ref,
        )

        # Stream this window's big (per-step) SMEM leaves to their HBM slot.
        dma_list = []
        for h, s in zip(flat_hbm, flat_smem):
            if h.shape[0] > 1:
                s_len = s.shape[
                    0]  # full SMEM leaf == one window (w_size * stride)
                # w is a traced fori_loop index; hint that the HBM offset is a
                # multiple of s_len (a multiple of the leaf tile at W=num_lanes) so
                # Mosaic can prove tile-alignment inside the dynamic loop.
                copy = pltpu.make_async_copy(
                    s.at[pl.ds(0, s_len)],
                    h.at[pl.ds(pl.multiple_of(w * s_len, s_len), s_len)],
                    dma_sem.at[0],
                )
                copy.start()
                dma_list.append(copy)
        jax.tree.map(lambda x: x.wait(), dma_list)

    # Build window 0 explicitly: its assignment pass also yields actual_steps.
    build_window(0)
    max_steps = 0
    for b_idx in range(cfgs.batch_size):
        max_steps = jnp.maximum(max_steps, lane_lengths_ref[b_idx])
    schedule_ref.actual_steps[0] = max_steps
    for h, s in zip(flat_hbm, flat_smem):
        if h.shape[0] == 1:  # actual_steps scalar: write once.
            copy = pltpu.make_async_copy(s.at[pl.ds(0, 1)], h.at[pl.ds(0, 1)],
                                         dma_sem.at[0])
            copy.start()
            copy.wait()

    # Remaining windows: a DYNAMIC loop over only the windows that actually have
    # steps (ceil(actual_steps / w_size)), capped by the static worst-case count.
    # Using fori_loop (instead of a Python range unroll) keeps the compiled
    # program a single window body, so compile cost is independent of num_windows.
    num_windows_actual = jnp.minimum(pl.cdiv(max_steps, w_size),
                                     cfgs.total_steps_ub // w_size)

    def _rest(w, _):
        build_window(w)
        return None

    jax.lax.fori_loop(1, num_windows_actual, _rest, None)


def rpa_metadata_schedule_kernel_stacked(
    ## Scalar prefetch.
    cu_q_lens_ref: jax.Ref,
    kv_lens_ref: jax.Ref,
    distribution_ref: jax.Ref,
    sorted_seq_idx_ref: jax.Ref,
    visibility_bounds_ref: jax.Ref,
    # HBM input (streamed per-seq into SMEM during the build).
    page_indices_hbm_ref: jax.Ref,
    # Outputs.
    schedule_hbm_ref: RpaSchedule,
    # Scratch.
    schedule_ref: RpaSchedule,
    lane_lengths_ref: jax.Ref,
    dma_sem: jax.Ref,
    seq_page_table_ref: jax.Ref,
    page_dma_sem: jax.Ref,
    *,
    cfgs: configs.RpaConfigs,
):
    """Stacked schedule entry: adds ``sorted_seq_idx`` (longest-first order) as an
    extra scalar-prefetch input, then delegates to the shared build."""
    rpa_metadata_schedule_kernel(
        cu_q_lens_ref,
        kv_lens_ref,
        distribution_ref,
        visibility_bounds_ref,
        page_indices_hbm_ref,
        schedule_hbm_ref,
        schedule_ref,
        lane_lengths_ref,
        dma_sem,
        seq_page_table_ref,
        page_dma_sem,
        cfgs=cfgs,
        sorted_seq_idx_ref=sorted_seq_idx_ref,
    )


def generate_rpa_metadata(
    cu_q_lens: jax.Array,
    kv_lens: jax.Array,
    distribution: jax.Array,
    page_indices: jax.Array,
    cfgs: configs.RpaConfigs,
    *,
    visibility: jax.Array | None = None,
    interpret=False,
    sorted_seq_idx_override: jax.Array | None = None,
) -> RpaSchedule:
    """``sorted_seq_idx_override``: optional [num_seqs] int32 array giving the
    stacked compute order explicitly (``sorted_seq_idx[order_idx]`` is the seq
    processed at position ``order_idx`` within the mode range). When None (default)
    the stacked build uses the internal longest-first order, byte-identical to the
    prior behavior. Callers (e.g. the a2a reshard) pass a chunk-grouped order to
    control which owner-chunk is computed/emitted first."""
    hbm_shaped = RpaSchedule.create_hbm_shape_dtype(cfgs)
    smem_shaped = RpaSchedule.create_shape_dtype(cfgs, steps=cfgs.sched_window)
    if cfgs.has_visibility:
        if visibility is None:
            raise ValueError(
                "visibility is required when cfgs.has_visibility=True.")
        max_q_blocks_per_seq = max(
            pl.cdiv(cfgs.serve.total_q_tokens, cfgs.bq_sz), 1)
        q_offsets = (
            jnp.arange(max_q_blocks_per_seq, dtype=jnp.int32)[:, None] *
            cfgs.bq_sz + jnp.arange(cfgs.bq_sz, dtype=jnp.int32)[None, :])

        def seq_visibility_bounds(s_idx):
            q_start = cu_q_lens[s_idx]
            q_end = cu_q_lens[s_idx + 1]
            k_len = kv_lens[s_idx]
            token_idx = q_start + q_offsets
            active = token_idx < q_end
            safe_idx = jnp.clip(token_idx, 0, visibility.shape[0] - 1)
            vis_start = visibility[safe_idx, 0]
            vis_end = visibility[safe_idx, 1]
            min_start = jnp.min(jnp.where(active, vis_start, k_len), axis=1)
            max_end = jnp.max(jnp.where(active, vis_end, -1), axis=1)
            return jnp.stack([min_start, max_end], axis=1).astype(jnp.int32)

        visibility_bounds = jax.vmap(seq_visibility_bounds)(jnp.arange(
            cfgs.serve.num_seqs, dtype=jnp.int32))
    else:
        visibility_bounds = jnp.zeros((1, 1, 2), dtype=jnp.int32)

    # 1-D [num_seqs * pps_pad] padded block table (each seq's slab padded to a
    # 128-multiple so the per-seq HBM->SMEM slice is 128-aligned in the build).
    page_indices_padded = jnp.pad(
        page_indices.reshape(cfgs.serve.num_seqs, cfgs.serve.pages_per_seq),
        ((0, 0), (0, cfgs.seq_page_table_size - cfgs.serve.pages_per_seq)),
    ).reshape(-1)

    if cfgs.is_stacked:
        # Longest-first order of THIS mode's seqs, placed at the mode's range
        # positions [start, end) so the build reads sorted_seq_idx[order_idx]
        # for order_idx in cfgs.mode.get_range(distribution). Seqs are contiguous
        # by mode: DECODE=[0,d0), PREFILL=[d0,d1), MIXED=[d1,d2). Non-mode seqs
        # get key -1 so they sort out of the mode range (never read).
        seq_idx = jnp.arange(cfgs.serve.num_seqs, dtype=jnp.int32)
        start_o, end_o = cfgs.mode.get_range(distribution)
        in_mode = jnp.logical_and(seq_idx >= start_o, seq_idx < end_o)
        sort_key = jnp.where(in_mode, kv_lens, -1).astype(jnp.int32)
        local_perm = jnp.argsort(-sort_key).astype(jnp.int32)
        # local_perm[j] is the j-th longest mode seq (j < count). Shift by start
        # so that sorted_seq_idx[start + j] == local_perm[j]; positions outside
        # [start, end) are never read. For DECODE (start=0) this is a no-op.
        gather_pos = jnp.clip(seq_idx - start_o, 0, cfgs.serve.num_seqs - 1)
        sorted_seq_idx = local_perm[gather_pos]
        # Optional caller-supplied compute order (e.g. a2a chunk-grouped,
        # farthest-first). Overrides the longest-first order above.
        if sorted_seq_idx_override is not None:
            sorted_seq_idx = sorted_seq_idx_override.astype(jnp.int32)

        return pl.pallas_call(
            functools.partial(rpa_metadata_schedule_kernel_stacked, cfgs=cfgs),
            out_shape=hbm_shaped,
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=4,
                in_specs=[
                    pl.BlockSpec(memory_space=pltpu.VMEM),  # visibility_bounds
                    pl.BlockSpec(memory_space=pltpu.HBM),  # page_indices
                ],
                out_specs=hbm_shaped.out_specs(),
                scratch_shapes=[
                    smem_shaped.scratch_shapes(),
                    pltpu.SMEM((cfgs.batch_size, ), jnp.int32),
                    pltpu.SemaphoreType.DMA((1, )),
                    pltpu.SMEM((cfgs.seq_page_table_size, ), jnp.int32),
                    pltpu.SemaphoreType.DMA((1, )),
                ],
            ),
            interpret=interpret,
            name="rpa_metadata_schedule_stacked",
        )(
            cu_q_lens,
            kv_lens,
            distribution,
            sorted_seq_idx,
            visibility_bounds,
            page_indices_padded,
        )

    return pl.pallas_call(
        functools.partial(rpa_metadata_schedule_kernel, cfgs=cfgs),
        out_shape=hbm_shaped,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=3,
            in_specs=[
                pl.BlockSpec(memory_space=pltpu.VMEM),  # visibility_bounds
                pl.BlockSpec(memory_space=pltpu.HBM),  # page_indices
            ],
            out_specs=hbm_shaped.out_specs(),
            scratch_shapes=[
                smem_shaped.scratch_shapes(),
                pltpu.SMEM((cfgs.batch_size, ), jnp.int32),
                pltpu.SemaphoreType.DMA((1, )),
                pltpu.SMEM((cfgs.seq_page_table_size, ), jnp.int32),
                pltpu.SemaphoreType.DMA((1, )),
            ],
        ),
        interpret=interpret,
        name="rpa_metadata_schedule",
    )(
        cu_q_lens,
        kv_lens,
        distribution,
        visibility_bounds,
        page_indices_padded,
    )
