# Copyright (c) Meta Platforms, Inc. and affiliates.
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
"""Global-attention prefill and mixed-request schedule placement."""

import dataclasses
import functools
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from tpu_inference.kernels.experimental.stacked_rpa import configs, utils
from tpu_inference.kernels.experimental.stacked_rpa.prefill import config
from tpu_inference.kernels.experimental.stacked_rpa.utils import cdiv


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class SmemWrapper:
    """Map a padded physical 1-D buffer to a logical N-D representation."""

    data: Any
    shape: tuple[int, ...] = dataclasses.field(metadata=dict(static=True))

    @classmethod
    def create_shape_dtype(cls, shape):
        # Pad to a 1-D int32 DMA tile so every schedule-window copy is aligned.
        n = int(np.prod(shape))
        n_pad = max(1024, -(-n // 1024) * 1024)
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


@dataclasses.dataclass(frozen=True)
class SchedulePlan:
    """Static capacity and windowing plan for the prefill schedule."""

    max_steps_ub: int
    worst_steps: int
    fits_one_window: bool
    sched_window: int
    num_sched_windows: int
    total_steps_ub: int
    seq_page_table_size: int

    @classmethod
    def create(cls, cfgs: config.PrefillConfig) -> "SchedulePlan":
        """Derive prefill capacity and a static upper bound on scheduled steps."""
        seq_page_table_size = cdiv(cfgs.serve.pages_per_seq, 1024) * 1024
        fixed_words = 0
        fixed_words += cfgs.serve.num_seqs  # kv_lens
        fixed_words += cfgs.serve.num_seqs + 1  # cu_q_lens
        fixed_words += seq_page_table_size
        fixed_words += 3  # distribution
        fixed_words += cfgs.batch_size  # lane_lengths
        fixed_words += 1  # actual_steps
        if cfgs.is_stacked:
            fixed_words += cfgs.serve.num_seqs  # sorted_seq_idx

        fixed_bytes = fixed_words * 4
        smem_limit_bytes = pltpu.get_tpu_info(
        ).smem_capacity_bytes - 256 * 1024
        available_bytes = smem_limit_bytes - fixed_bytes

        # Eight int32 indices, dma_q, cache-page triples, and new-page metadata.
        bytes_per_step = (
            32 + 8 + 12 * cfgs.bkv_p_cache +
            4 * cfgs.dma_kv_new_size * cfgs.bkv_p_new) * cfgs.batch_size
        raw_max_steps = available_bytes // bytes_per_step
        num_lanes = pltpu.get_tpu_info().num_lanes
        max_steps_ub = max(1, raw_max_steps // num_lanes) * num_lanes

        worst_steps = calculate_worst_steps_ub(cfgs)

        fits_one_window = worst_steps <= max_steps_ub
        if fits_one_window and cfgs.model.sliding_window is not None:
            sched_window = max(1, worst_steps)
        elif fits_one_window:
            sched_window = max_steps_ub
        else:
            sched_window = max(1, max_steps_ub // 2)
        num_sched_windows = max(1, -(-worst_steps // sched_window))
        return cls(
            max_steps_ub=max_steps_ub,
            worst_steps=worst_steps,
            fits_one_window=fits_one_window,
            sched_window=sched_window,
            num_sched_windows=num_sched_windows,
            total_steps_ub=num_sched_windows * sched_window,
            seq_page_table_size=seq_page_table_size,
        )


def _max_query_blocks_ub(cfgs: config.PrefillConfig) -> int:
    """Bound Q blocks across any runtime PREFILL or MIXED sequence range."""
    total_q = cfgs.serve.total_q_tokens
    active_seqs = min(cfgs.serve.num_seqs, total_q)
    if active_seqs == 0:
        return 0

    # For positive q_i, ceil(q_i / BQ) = 1 + floor((q_i - 1) / BQ).
    # Concentrating all tokens beyond the mandatory one per active sequence
    # maximizes the sum. A runtime mode range is a subset of these sequences and
    # cannot contain more query blocks than this all-sequence bound.
    return active_seqs + (total_q - active_seqs) // cfgs.bq_sz


def _max_k_blocks_per_query_ub(cfgs: config.PrefillConfig) -> int:
    """Bound the length of one Q block's contiguous K-block run."""
    max_model_len = cfgs.serve.pages_per_seq * cfgs.serve.page_size
    model_k_blocks = cdiv(max_model_len, cfgs.bkv_sz)
    if cfgs.model.sliding_window is None:
        return model_k_blocks

    # A causal Q block covers at most W + BQ - 1 KV tokens. An interval of L
    # tokens can intersect ceil((L + BKV - 1) / BKV) blocks when its endpoints
    # are not BKV-aligned.
    attended_tokens = cfgs.model.sliding_window + cfgs.bq_sz - 1
    aligned_k_blocks = cdiv(attended_tokens + cfgs.bkv_sz - 1, cfgs.bkv_sz)
    return min(model_k_blocks, aligned_k_blocks)


def calculate_worst_steps_ub(cfgs: config.PrefillConfig) -> int:
    """Return a safe static upper bound on the schedule's lane length."""
    max_model_len = cfgs.serve.pages_per_seq * cfgs.serve.page_size
    model_k_blocks = cdiv(max_model_len, cfgs.bkv_sz)

    if cfgs.model.sliding_window is None:
        # Retain the established global-attention bound. Its extra K run and
        # pipeline reserve account for per-sequence Q-block fragmentation.
        num_q = cdiv(cfgs.serve.total_q_tokens, cfgs.bq_sz)
        total_work = max(
            cfgs.serve.num_seqs * model_k_blocks,
            num_q * model_k_blocks,
        )
        worst_steps = cdiv(total_work, cfgs.batch_size)
        if cfgs.mode == configs.RpaCase.MIXED and cfgs.dense_pack:
            worst_steps += cfgs.serve.num_seqs
        return worst_steps + model_k_blocks + cfgs.n_buffer + 1

    q_blocks = _max_query_blocks_ub(cfgs)
    max_k_run = _max_k_blocks_per_query_ub(cfgs)
    total_cells = q_blocks * max_k_run

    if cfgs.is_stacked:
        # Dense MIXED placement maps its global cell cursor directly onto lanes.
        return max(1, cdiv(total_cells, cfgs.batch_size))

    # Normal placement assigns each complete Q-block K run to the least-loaded
    # lane. List scheduling is bounded by ceil(total / lanes) + max_run - 1.
    return max(1, cdiv(total_cells, cfgs.batch_size) + max_k_run - 1)


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class PrefillSchedule:
    """Complete PREFILL/MIXED metadata schema and storage helpers."""

    s_idx: SmemWrapper  # [steps, batch]
    k_idx: SmemWrapper  # [steps, batch]
    is_last_k: SmemWrapper  # [steps, batch]
    do_writeback: SmemWrapper  # [steps, batch]
    skip_mask: SmemWrapper  # [steps, batch]
    combine_span: SmemWrapper  # [steps, batch]
    is_final: SmemWrapper  # [steps, batch]
    dma_q: SmemWrapper  # [steps, batch, 2]
    dma_kv_cache: SmemWrapper  # [steps, batch, bkv_p_cache, 3]
    dma_kv_new: SmemWrapper  # [steps, batch, bkv_p_new, 5]
    actual_steps: Any  # [1]

    cfgs: config.PrefillConfig = dataclasses.field(metadata=dict(static=True))
    plan: SchedulePlan = dataclasses.field(metadata=dict(static=True))
    q_idx: SmemWrapper  # [steps, batch]

    @classmethod
    def create_shape_dtype(
        cls,
        cfgs: config.PrefillConfig,
        steps: int | None = None,
        *,
        plan: SchedulePlan | None = None,
    ) -> "PrefillSchedule":
        if cfgs.mode not in (configs.RpaCase.PREFILL, configs.RpaCase.MIXED):
            raise ValueError(f"PrefillSchedule received {cfgs.mode=}")
        if plan is None:
            plan = SchedulePlan.create(cfgs)
        if steps is None:
            steps = plan.max_steps_ub

        idx_wrapper = SmemWrapper.create_shape_dtype((steps, cfgs.batch_size))
        return cls(
            s_idx=idx_wrapper,
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
            )),
            actual_steps=jax.ShapeDtypeStruct((1, ), jnp.int32),
            cfgs=cfgs,
            plan=plan,
            q_idx=idx_wrapper,
        )

    @classmethod
    def create_hbm_shape_dtype(
        cls,
        cfgs: config.PrefillConfig,
        *,
        plan: SchedulePlan | None = None,
    ) -> "PrefillSchedule":
        """Create HBM leaves as padded, independently aligned SMEM windows."""
        if plan is None:
            plan = SchedulePlan.create(cfgs)
        win = cls.create_shape_dtype(cfgs, steps=plan.sched_window, plan=plan)
        num_windows = plan.num_sched_windows

        def grow(value):
            if not isinstance(value, SmemWrapper):
                return value
            return SmemWrapper(
                data=jax.ShapeDtypeStruct(
                    (value.data.shape[0] * num_windows, ), value.data.dtype),
                shape=value.shape,
            )

        replacements = {
            field.name: grow(getattr(win, field.name))
            for field in dataclasses.fields(win)
            if isinstance(getattr(win, field.name), SmemWrapper)
        }
        return dataclasses.replace(win, **replacements)

    def get_dma_kv_cache(
        self,
        step: jax.typing.ArrayLike,
        batch_idx: jax.typing.ArrayLike,
        page_idx: jax.typing.ArrayLike,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        physical_page = self.dma_kv_cache[step, batch_idx, page_idx, 0]
        dst_vmem = self.dma_kv_cache[step, batch_idx, page_idx, 1]
        size = self.dma_kv_cache[step, batch_idx, page_idx, 2]
        return physical_page, dst_vmem, size

    def get_dma_fetch_kv_new(
        self,
        step: jax.typing.ArrayLike,
        batch_idx: jax.typing.ArrayLike,
        page_idx: jax.typing.ArrayLike,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        src_hbm = self.dma_kv_new[step, batch_idx, page_idx, 0]
        dst_vmem = self.dma_kv_new[step, batch_idx, page_idx, 1]
        valid = self.dma_kv_new[step, batch_idx, page_idx, 4] & 1
        return src_hbm, dst_vmem, valid

    def get_dma_update_kv_new(
        self,
        step: jax.typing.ArrayLike,
        batch_idx: jax.typing.ArrayLike,
        page_idx: jax.typing.ArrayLike,
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
        dst_hbm = self.dma_kv_new[step, batch_idx, page_idx, 2]
        src_vmem = self.dma_kv_new[step, batch_idx, page_idx, 3]
        packed = self.dma_kv_new[step, batch_idx, page_idx, 4]
        valid = (packed >> 1) & 1
        width, mask = self.cfgs.wb_lane_bits
        wb_lane = ((packed >> 2) & mask) * self.cfgs.num_lanes
        wb_size = ((packed >> (2 + width)) & mask) * self.cfgs.num_lanes
        return dst_hbm, src_vmem, valid, wb_lane, wb_size

    def get_dma_q(
            self, step: jax.typing.ArrayLike,
            batch_idx: jax.typing.ArrayLike) -> tuple[jax.Array, jax.Array]:
        src_hbm = self.dma_q[step, batch_idx, 0]
        size = self.dma_q[step, batch_idx, 1]
        return src_hbm, size

    def get_q_idx(self, step: jax.typing.ArrayLike,
                  batch_idx: jax.typing.ArrayLike) -> jax.Array:
        return self.q_idx[step, batch_idx]

    def set_q_idx(
        self,
        step: jax.typing.ArrayLike,
        batch_idx: jax.typing.ArrayLike,
        value: jax.typing.ArrayLike,
    ) -> None:
        self.q_idx[step, batch_idx] = value

    def scratch_shapes(self):
        """Return the schedule PyTree as SMEM scratch-memory descriptors."""
        return jax.tree.map(lambda x: pltpu.SMEM(x.shape, x.dtype), self)

    def in_specs(self):
        """Keep large schedule leaves in HBM and scalar-prefetch actual_steps."""

        def wrapper(x):
            memory_space = pltpu.SMEM if x.size == 1 else pltpu.HBM
            return pl.BlockSpec(memory_space=memory_space)

        return jax.tree.map(wrapper, self)

    def out_specs(self):
        """The schedule-builder kernel materializes every output in HBM."""
        return jax.tree.map(lambda x: pl.BlockSpec(memory_space=pltpu.HBM),
                            self)


def get_kv_block_start(
    k_idx,
    *,
    cfgs: config.PrefillConfig,
    k_len,
    q_len,
):
    """Return prefill's global-grid token and logical-page starts."""
    del k_len, q_len
    return k_idx * cfgs.bkv_sz, k_idx * cfgs.bkv_p


def emit_kv_new_metadata(
    *,
    cfgs: config.PrefillConfig,
    bkv_sz_cache,
    new_sz,
    fill_dma_kv_new,
) -> None:
    """Emit the full set of new-KV DMA records needed by prefill."""
    if cfgs.bkv_p_new < cfgs.bkv_p:
        # update_kv_cache=False keeps a single inert boundary-page record.
        assert cfgs.bkv_p_new == 1
        slot_start = (bkv_sz_cache //
                      cfgs.serve.page_size) * cfgs.serve.page_size
        fill_dma_kv_new(0, bkv_sz_cache, new_sz, slot_start)
        return

    for i in range(max(cfgs.bkv_p, cfgs.bkv_p_new)):
        slot_start = i * cfgs.serve.page_size
        slot_end = slot_start + cfgs.serve.page_size
        dst_vmem = jnp.maximum(slot_start, bkv_sz_cache)
        end_in_slot = jnp.minimum(slot_end, bkv_sz_cache + new_sz)
        dma_sz = jnp.maximum(0, end_in_slot - dst_vmem)
        fill_dma_kv_new(i, dst_vmem, dma_sz, slot_start)


def _place_metadata(
    cu_q_lens_ref: jax.Ref,
    kv_lens_ref: jax.Ref,
    distribution_ref: jax.Ref,
    schedule: PrefillSchedule,
    lane_lengths_ref: jax.Ref,
    page_indices_hbm_ref: jax.Ref,
    seq_page_table_ref: jax.Ref,
    page_dma_sem: jax.Ref,
    *,
    cfgs: config.PrefillConfig,
    window_lo: int,
    window_size: int,
    sorted_seq_idx_ref: jax.Ref | None,
    emit_k_block,
) -> None:
    """Place prefill/MIXED q-blocks into schedule lanes."""
    if cfgs.mode not in (configs.RpaCase.PREFILL, configs.RpaCase.MIXED):
        raise ValueError(f"Prefill scheduler received {cfgs.mode=}")

    pps_pad = schedule.plan.seq_page_table_size

    def copy_page_table(s_idx):
        seq_off = pl.multiple_of(s_idx * pps_pad, pps_pad)
        page_copy = pltpu.make_async_copy(
            page_indices_hbm_ref.at[pl.ds(seq_off, pps_pad)],
            seq_page_table_ref.at[pl.ds(0, pps_pad)],
            page_dma_sem.at[0],
        )
        page_copy.start()
        page_copy.wait()

    if cfgs.is_stacked and cfgs.dense_pack:
        if sorted_seq_idx_ref is None:
            raise ValueError(
                "Dense prefill scheduling requires a sorted sequence order.")

        # Dense MIXED placement uses a global cell cursor. Each (sequence,
        # q-block) pair is an independent combine unit and may straddle steps.
        n = cfgs.batch_size

        def write_group_metadata(step, cell, span, is_final, active):
            in_window = jnp.logical_and(step >= window_lo, step
                                        < window_lo + window_size)
            local = jnp.clip(step - window_lo, 0, window_size - 1)

            @pl.when(jnp.logical_and(in_window, active))
            def _write():
                schedule.combine_span[local, cell] = span
                schedule.is_final[local, cell] = is_final

        @jax.named_scope("dense_seq_loop")
        def dense_seq_loop(order_idx, cursor):
            s_idx = sorted_seq_idx_ref[order_idx]
            copy_page_table(s_idx)

            q_start = cu_q_lens_ref[s_idx]
            q_end = cu_q_lens_ref[s_idx + 1]
            k_len = kv_lens_ref[s_idx]
            q_len = q_end - q_start
            num_k = pl.cdiv(k_len, cfgs.bkv_sz)
            num_q = pl.cdiv(q_len, cfgs.bq_sz)

            def q_loop(q_idx, cursor):
                q_src = q_start + q_idx * cfgs.bq_sz
                q_sz_task = jnp.clip(q_end - q_src, 0, cfgs.bq_sz)

                start_k_idx = 0
                if (sliding_window := cfgs.model.sliding_window) is not None:
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

                num_k_for_q = jnp.maximum(end_k_idx - start_k_idx, 0)
                cursor_end = cursor + num_k_for_q

                def k_loop(k_idx, _):
                    global_cell = cursor + (k_idx - start_k_idx)
                    kv_len_start, kv_p_start = get_kv_block_start(k_idx,
                                                                  cfgs=cfgs,
                                                                  k_len=k_len,
                                                                  q_len=q_len)
                    emit_k_block(
                        k_idx,
                        global_cell // n,
                        target_lane=global_cell % n,
                        s_idx=s_idx,
                        q_idx=q_idx,
                        q_end=q_end,
                        q_src=q_src,
                        q_sz_task=q_sz_task,
                        k_len=k_len,
                        q_len=q_len,
                        end_k_idx=end_k_idx,
                        kv_len_start=kv_len_start,
                        kv_p_start=kv_p_start,
                    )
                    return None

                jax.lax.fori_loop(start_k_idx, end_k_idx, k_loop, None)

                first_step = cursor // n
                last_step = jnp.maximum(cursor_end - 1, cursor) // n

                def step_metadata(step, _):
                    lo = jnp.maximum(cursor, step * n)
                    hi = jnp.minimum(cursor_end, (step + 1) * n)
                    root = lo - step * n
                    span = hi - lo
                    is_final = jnp.where(step == last_step, 1, 0)
                    write_group_metadata(step, root, span, is_final,
                                         num_k_for_q > 0)
                    return None

                @pl.when(num_k_for_q > 0)
                def _emit_group_metadata():
                    jax.lax.fori_loop(first_step, last_step + 1, step_metadata,
                                      None)

                return cursor_end

            return jax.lax.fori_loop(0, num_q, q_loop, cursor)

        start_seq_idx, end_seq_idx = cfgs.mode.get_range(distribution_ref)
        final_cursor = jax.lax.fori_loop(start_seq_idx, end_seq_idx,
                                         dense_seq_loop, jnp.int32(0))
        lane_lengths_ref[0] = pl.cdiv(final_cursor, n)
        return

    # The normal prefill path greedily assigns each complete q-block k-run to the
    # currently shortest lane. Prefill has enough independent q-blocks that this
    # avoids the cross-cell combine cost of dense placement on balanced batches.
    @jax.named_scope("q_loop")
    def q_loop(q_idx, _, *, s_idx, q_start, q_end, k_len, q_len, num_k):
        target_lane = 0
        min_len = lane_lengths_ref[0]
        for lane in range(1, cfgs.batch_size):
            is_better = lane_lengths_ref[lane] < min_len
            target_lane = jnp.where(is_better, lane, target_lane)
            min_len = jnp.where(is_better, lane_lengths_ref[lane], min_len)

        current_step = lane_lengths_ref[target_lane]
        q_src = q_start + q_idx * cfgs.bq_sz
        q_sz_task = jnp.clip(q_end - q_src, 0, cfgs.bq_sz)

        start_k_idx = 0
        if (sliding_window := cfgs.model.sliding_window) is not None:
            sw_start_idx = k_len - q_len + q_idx * cfgs.bq_sz - sliding_window + 1
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

        def k_loop(k_idx, step):
            kv_len_start, kv_p_start = get_kv_block_start(k_idx,
                                                          cfgs=cfgs,
                                                          k_len=k_len,
                                                          q_len=q_len)
            return emit_k_block(
                k_idx,
                step,
                target_lane=target_lane,
                s_idx=s_idx,
                q_idx=q_idx,
                q_end=q_end,
                q_src=q_src,
                q_sz_task=q_sz_task,
                k_len=k_len,
                q_len=q_len,
                end_k_idx=end_k_idx,
                kv_len_start=kv_len_start,
                kv_p_start=kv_p_start,
            )

        lane_lengths_ref[target_lane] = jax.lax.fori_loop(
            start_k_idx, end_k_idx, k_loop, current_step)

    @jax.named_scope("seq_loop")
    def seq_loop(s_idx, _):
        q_start = cu_q_lens_ref[s_idx]
        q_end = cu_q_lens_ref[s_idx + 1]
        k_len = kv_lens_ref[s_idx]
        q_len = q_end - q_start
        num_q = pl.cdiv(q_len, cfgs.bq_sz)
        num_k = pl.cdiv(k_len, cfgs.bkv_sz)

        copy_page_table(s_idx)
        q_loop_for_seq = functools.partial(
            q_loop,
            s_idx=s_idx,
            q_start=q_start,
            q_end=q_end,
            k_len=k_len,
            q_len=q_len,
            num_k=num_k,
        )
        jax.lax.fori_loop(0, num_q, q_loop_for_seq, None)

    start_seq_idx, end_seq_idx = cfgs.mode.get_range(distribution_ref)
    jax.lax.fori_loop(start_seq_idx, end_seq_idx, seq_loop, None)


def compute_metadata(
    cu_q_lens_ref: jax.Ref,
    kv_lens_ref: jax.Ref,
    distribution_ref: jax.Ref,
    schedule: PrefillSchedule,
    lane_lengths_ref: jax.Ref,
    page_indices_hbm_ref: jax.Ref,
    seq_page_table_ref: jax.Ref,
    page_dma_sem: jax.Ref,
    *,
    cfgs: config.PrefillConfig,
    window_lo: int = 0,
    sorted_seq_idx_ref: jax.Ref | None = None,
) -> None:
    """Place prefill cells and write their complete DMA metadata."""
    if cfgs.mode not in (configs.RpaCase.PREFILL, configs.RpaCase.MIXED):
        raise ValueError(f"Prefill scheduler received {cfgs.mode=}")

    window_size = schedule.plan.sched_window

    @jax.named_scope("k_loop")
    def write_cell_metadata(
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
        kv_len_start,
        kv_p_start,
    ):
        in_window = jnp.logical_and(step >= window_lo, step
                                    < window_lo + window_size)
        local = jnp.clip(step - window_lo, 0, window_size - 1)

        kv_left = k_len - kv_len_start
        if cfgs.update_kv_cache:
            kv_left_frm_cache = jnp.maximum(kv_left - q_len, 0)
        else:
            kv_left_frm_cache = kv_left

        kv_left_frm_new = kv_left - kv_left_frm_cache
        bkv_sz_cache = jnp.minimum(kv_left_frm_cache, cfgs.bkv_sz)
        new_sz = jnp.minimum(cfgs.bkv_sz - bkv_sz_cache, kv_left_frm_new)

        # Each new k block is written back by the first q block that attends to it.
        q_wb = jnp.maximum(0, (kv_len_start - (k_len - q_len))) // cfgs.bq_sz
        do_writeback = jnp.where(
            cfgs.update_kv_cache & (new_sz > 0) & (q_idx == q_wb), 1, 0)

        min_q_pos = k_len - q_len + q_idx * cfgs.bq_sz
        skip_mask = jnp.where(
            (not cfgs.disable_skip_mask)
            & (cfgs.model.sliding_window is None)
            & (kv_len_start + cfgs.bkv_sz - 1 <= min_q_pos),
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
            dst_hbm = seq_page_table_ref[p_idx]
            lane_lo = dst_vmem - slot_start
            wb_lane = (lane_lo // cfgs.num_lanes) * cfgs.num_lanes
            wb_end = pl.cdiv(lane_lo + dma_sz, cfgs.num_lanes) * cfgs.num_lanes
            wb_size = jnp.where(dma_sz > 0, wb_end - wb_lane, 0)
            width, _ = cfgs.wb_lane_bits
            packed_dma_valid = (fetch_dma_valid
                                | (jnp.where(dma_sz > 0, 1, 0) << 1)
                                | ((wb_lane // cfgs.num_lanes) << 2)
                                | ((wb_size // cfgs.num_lanes) << (2 + width)))

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
            schedule.is_last_k[local, target_lane] = jnp.where(
                k_idx == end_k_idx - 1, 1, 0)

            schedule.dma_q[local, target_lane, 0] = q_src
            schedule.dma_q[local, target_lane, 1] = q_sz_task

            for i in range(cfgs.bkv_p_cache):
                dst_vmem = i << cfgs.serve.page_size_log2
                dma_sz = kv_left_frm_cache - dst_vmem
                dma_sz = jnp.clip(dma_sz, 0, cfgs.serve.page_size)
                dma_sz = utils.align_to(dma_sz, cfgs.num_lanes)
                src_hbm = seq_page_table_ref[jnp.minimum(
                    kv_p_start + i, cfgs.serve.pages_per_seq - 1)]
                schedule.dma_kv_cache[local, target_lane, i, 0] = src_hbm
                schedule.dma_kv_cache[local, target_lane, i, 1] = dst_vmem
                schedule.dma_kv_cache[local, target_lane, i, 2] = dma_sz

            schedule.do_writeback[local, target_lane] = do_writeback
            schedule.skip_mask[local, target_lane] = skip_mask

            emit_kv_new_metadata(
                cfgs=cfgs,
                bkv_sz_cache=bkv_sz_cache,
                new_sz=new_sz,
                fill_dma_kv_new=fill_dma_kv_new,
            )

        return step + 1

    _place_metadata(
        cu_q_lens_ref,
        kv_lens_ref,
        distribution_ref,
        schedule,
        lane_lengths_ref,
        page_indices_hbm_ref,
        seq_page_table_ref,
        page_dma_sem,
        cfgs=cfgs,
        window_lo=window_lo,
        window_size=window_size,
        sorted_seq_idx_ref=sorted_seq_idx_ref,
        emit_k_block=write_cell_metadata,
    )


def rpa_metadata_schedule_kernel(
    # Scalar prefetch.
    cu_q_lens_ref: jax.Ref,
    kv_lens_ref: jax.Ref,
    distribution_ref: jax.Ref,
    # HBM input streamed per sequence into SMEM during the build.
    page_indices_hbm_ref: jax.Ref,
    # Output and scratch.
    schedule_hbm_ref: PrefillSchedule,
    schedule_ref: PrefillSchedule,
    lane_lengths_ref: jax.Ref,
    dma_sem: jax.Ref,
    seq_page_table_ref: jax.Ref,
    page_dma_sem: jax.Ref,
    *,
    cfgs: config.PrefillConfig,
    sorted_seq_idx_ref: jax.Ref | None = None,
) -> None:
    """Build the PREFILL/MIXED HBM-to-VMEM DMA schedule by windows."""
    plan = schedule_ref.plan
    window_size = plan.sched_window
    flat_hbm = jax.tree_util.tree_leaves(schedule_hbm_ref)
    flat_smem = jax.tree_util.tree_leaves(schedule_ref)

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

    # Re-running deterministic placement for each window avoids storing the full
    # worst-case schedule in SMEM. Only rows in the active window are written.
    def build_window(window_idx):
        window_lo = window_idx * window_size

        for b_idx in range(cfgs.batch_size):
            lane_lengths_ref[b_idx] = 0

        jax.lax.fori_loop(0, window_size, mask_window, None)
        compute_metadata(
            cu_q_lens_ref,
            kv_lens_ref,
            distribution_ref,
            schedule_ref,
            lane_lengths_ref,
            page_indices_hbm_ref,
            seq_page_table_ref,
            page_dma_sem,
            cfgs=cfgs,
            window_lo=window_lo,
            sorted_seq_idx_ref=sorted_seq_idx_ref,
        )

        dma_list = []
        for hbm_leaf, smem_leaf in zip(flat_hbm, flat_smem):
            if hbm_leaf.shape[0] > 1:
                leaf_size = smem_leaf.shape[0]
                copy = pltpu.make_async_copy(
                    smem_leaf.at[pl.ds(0, leaf_size)],
                    hbm_leaf.at[pl.ds(
                        pl.multiple_of(window_idx * leaf_size, leaf_size),
                        leaf_size,
                    )],
                    dma_sem.at[0],
                )
                copy.start()
                dma_list.append(copy)
        jax.tree.map(lambda copy: copy.wait(), dma_list)

    # Window zero also computes the true number of scheduled steps.
    build_window(0)
    max_steps = 0
    for b_idx in range(cfgs.batch_size):
        max_steps = jnp.maximum(max_steps, lane_lengths_ref[b_idx])
    schedule_ref.actual_steps[0] = max_steps

    for hbm_leaf, smem_leaf in zip(flat_hbm, flat_smem):
        if hbm_leaf.shape[0] == 1:
            copy = pltpu.make_async_copy(
                smem_leaf.at[pl.ds(0, 1)],
                hbm_leaf.at[pl.ds(0, 1)],
                dma_sem.at[0],
            )
            copy.start()
            copy.wait()

    num_windows_actual = jnp.minimum(pl.cdiv(max_steps, window_size),
                                     plan.num_sched_windows)

    def build_remaining_window(window_idx, _):
        build_window(window_idx)
        return None

    jax.lax.fori_loop(1, num_windows_actual, build_remaining_window, None)


def rpa_metadata_schedule_kernel_stacked(
    # Scalar prefetch.
    cu_q_lens_ref: jax.Ref,
    kv_lens_ref: jax.Ref,
    distribution_ref: jax.Ref,
    sorted_seq_idx_ref: jax.Ref,
    # HBM input streamed per sequence into SMEM during the build.
    page_indices_hbm_ref: jax.Ref,
    # Output and scratch.
    schedule_hbm_ref: PrefillSchedule,
    schedule_ref: PrefillSchedule,
    lane_lengths_ref: jax.Ref,
    dma_sem: jax.Ref,
    seq_page_table_ref: jax.Ref,
    page_dma_sem: jax.Ref,
    *,
    cfgs: config.PrefillConfig,
) -> None:
    """Build a stacked schedule using the supplied sequence order."""
    rpa_metadata_schedule_kernel(
        cu_q_lens_ref,
        kv_lens_ref,
        distribution_ref,
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
    cfgs: config.PrefillConfig,
    *,
    interpret: bool = False,
    sorted_seq_idx_override: jax.Array | None = None,
) -> PrefillSchedule:
    """Generate a complete PREFILL/MIXED schedule.

    ``sorted_seq_idx_override`` optionally supplies the stacked compute order.
    Without it, stacked scheduling processes sequences longest-first within the
    active mode range.
    """
    plan = SchedulePlan.create(cfgs)
    hbm_shaped = PrefillSchedule.create_hbm_shape_dtype(cfgs, plan=plan)
    smem_shaped = PrefillSchedule.create_shape_dtype(cfgs,
                                                     steps=plan.sched_window,
                                                     plan=plan)

    # Pad each sequence's page-table slab to a 128-element DMA alignment.
    page_indices_padded = jnp.pad(
        page_indices.reshape(cfgs.serve.num_seqs, cfgs.serve.pages_per_seq),
        ((0, 0), (0, plan.seq_page_table_size - cfgs.serve.pages_per_seq)),
    ).reshape(-1)

    if cfgs.is_stacked:
        seq_idx = jnp.arange(cfgs.serve.num_seqs, dtype=jnp.int32)
        start_order, end_order = cfgs.mode.get_range(distribution)
        in_mode = jnp.logical_and(seq_idx >= start_order, seq_idx < end_order)
        sort_key = jnp.where(in_mode, kv_lens, -1).astype(jnp.int32)
        local_perm = jnp.argsort(-sort_key).astype(jnp.int32)
        gather_pos = jnp.clip(seq_idx - start_order, 0,
                              cfgs.serve.num_seqs - 1)
        sorted_seq_idx = local_perm[gather_pos]
        if sorted_seq_idx_override is not None:
            sorted_seq_idx = sorted_seq_idx_override.astype(jnp.int32)

        return pl.pallas_call(
            functools.partial(rpa_metadata_schedule_kernel_stacked, cfgs=cfgs),
            out_shape=hbm_shaped,
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=4,
                in_specs=[
                    pl.BlockSpec(memory_space=pltpu.HBM),
                ],
                out_specs=hbm_shaped.out_specs(),
                scratch_shapes=[
                    smem_shaped.scratch_shapes(),
                    pltpu.SMEM((cfgs.batch_size, ), jnp.int32),
                    pltpu.SemaphoreType.DMA((1, )),
                    pltpu.SMEM((plan.seq_page_table_size, ), jnp.int32),
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
            page_indices_padded,
        )

    return pl.pallas_call(
        functools.partial(rpa_metadata_schedule_kernel, cfgs=cfgs),
        out_shape=hbm_shaped,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=3,
            in_specs=[
                pl.BlockSpec(memory_space=pltpu.HBM),
            ],
            out_specs=hbm_shaped.out_specs(),
            scratch_shapes=[
                smem_shaped.scratch_shapes(),
                pltpu.SMEM((cfgs.batch_size, ), jnp.int32),
                pltpu.SemaphoreType.DMA((1, )),
                pltpu.SMEM((plan.seq_page_table_size, ), jnp.int32),
                pltpu.SemaphoreType.DMA((1, )),
            ],
        ),
        interpret=interpret,
        name="rpa_metadata_schedule",
    )(
        cu_q_lens,
        kv_lens,
        distribution,
        page_indices_padded,
    )
