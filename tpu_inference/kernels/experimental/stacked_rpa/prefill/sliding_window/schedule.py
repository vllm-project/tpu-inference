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
"""Compact one-cell-per-Q-block schedule for sliding-window prefill."""

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
from tpu_inference.kernels.experimental.stacked_rpa.prefill.sliding_window import \
    block_sizes
from tpu_inference.kernels.experimental.stacked_rpa.utils import cdiv

_CELL_Q_SRC = 0
_CELL_Q_SIZE = 1
_CELL_PROCESSED_Q = 2
_CELL_PROCESSED_KV = 3
_CELL_EFFECTIVE_KV = 4
_CELL_CACHE_SIZE = 5
_CELL_NEW_KV_START = 6
_CELL_SIZE = 7

_DMA_NEW_SRC_HBM = 0
_DMA_NEW_DST_VMEM = 1
_DMA_NEW_DST_HBM = 2
_DMA_NEW_SRC_VMEM = 3
_DMA_NEW_FLAGS = 4
_FETCH_VALID = 1
_UPDATE_VALID = 2

# Leave room for compiler padding, semaphores, and small spill allocations.
_SMEM_HEADROOM_BYTES = 256 * 1024


def required_window_bkv_size(cfgs: config.PrefillConfig) -> int:
    """Return the page-aligned tile covering any Q block's local window."""
    if cfgs.model.sliding_window is None:
        raise ValueError(
            "Sliding-window prefill requires model.sliding_window.")
    return block_sizes.required_window_bkv_size(
        cfgs.model,
        cfgs.serve,
        cfgs.bq_sz,
    )


def validate_config(cfgs: config.PrefillConfig) -> None:
    """Validate the one-query-block, one-anchored-KV-block contract."""
    if cfgs.mode not in (configs.RpaCase.PREFILL, configs.RpaCase.MIXED):
        raise ValueError(f"Sliding-window prefill received {cfgs.mode=}.")
    if cfgs.mode == configs.RpaCase.PREFILL and not cfgs.update_kv_cache:
        raise ValueError(
            "Sliding-window PREFILL requires update_kv_cache=True because "
            "PREFILL has no cache DMA allocation.")
    if cfgs.model.sliding_window is None:
        raise ValueError(
            "Sliding-window prefill requires model.sliding_window.")
    if cfgs.serve.pages_per_seq < 1:
        raise ValueError(
            "Sliding-window prefill requires at least one KV page.")
    if cfgs.serve.page_size <= 0 or (cfgs.serve.page_size &
                                     (cfgs.serve.page_size - 1)):
        raise ValueError(
            "Sliding-window prefill requires a power-of-two page size.")
    if cfgs.bq_sz < 1:
        raise ValueError(f"bq_sz must be positive, got {cfgs.bq_sz}.")
    if cfgs.bkv_sz % cfgs.serve.page_size:
        raise ValueError(
            "Sliding-window prefill requires a page-aligned KV block.")
    required_bkv = required_window_bkv_size(cfgs)
    if cfgs.bkv_sz < required_bkv:
        raise ValueError(
            "Sliding-window prefill requires one anchored KV block: "
            f"bkv_sz={cfgs.bkv_sz} is smaller than {required_bkv}.")
    if cfgs.n_buffer != 2:
        raise ValueError(
            "Sliding-window prefill requires double buffering, got "
            f"n_buffer={cfgs.n_buffer}.")


def _max_query_blocks_ub(cfgs: config.PrefillConfig) -> int:
    """Bound Q blocks across any runtime PREFILL or MIXED sequence range."""
    total_q = cfgs.serve.total_q_tokens
    active_seqs = min(cfgs.serve.num_seqs, total_q)
    if active_seqs == 0:
        return 0
    return active_seqs + (total_q - active_seqs) // cfgs.bq_sz


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class SmemWrapper:
    """Map an aligned physical 1-D buffer to a logical N-D representation."""

    data: Any
    shape: tuple[int, ...] = dataclasses.field(metadata=dict(static=True))

    @classmethod
    def create_shape_dtype(cls, shape):
        n = int(np.prod(shape))
        n_pad = max(1024, cdiv(n, 1024) * 1024)
        return cls(data=jax.ShapeDtypeStruct((n_pad, ), jnp.int32),
                   shape=shape)

    def _get_pos(self, indices):
        strides = pl.strides_from_shape(self.shape)
        assert len(strides) == len(indices)
        return sum(stride * idx for stride, idx in zip(strides, indices))

    def __getitem__(self, indices):
        return self.data[self._get_pos(indices)]

    def __setitem__(self, indices, value):
        self.data[self._get_pos(indices)] = value


def _schedule_wrappers(cfgs: config.PrefillConfig,
                       steps: int) -> dict[str, SmemWrapper]:
    return {
        "cell":
        SmemWrapper.create_shape_dtype((steps, cfgs.batch_size, _CELL_SIZE)),
        "dma_kv_cache":
        SmemWrapper.create_shape_dtype(
            (steps, cfgs.batch_size, cfgs.bkv_p_cache, 2)),
        "dma_kv_new":
        SmemWrapper.create_shape_dtype(
            (steps, cfgs.batch_size, cfgs.bkv_p_new, cfgs.dma_kv_new_size)),
    }


def _shape_tree_nbytes(tree) -> int:
    return sum(
        int(np.prod(leaf.shape)) * jnp.dtype(leaf.dtype).itemsize
        for leaf in jax.tree_util.tree_leaves(tree))


def _metadata_fixed_smem_shapes(
    cfgs: config.PrefillConfig,
    seq_page_table_size: int,
) -> dict[str, jax.ShapeDtypeStruct]:
    return {
        "cu_q_lens": jax.ShapeDtypeStruct((cfgs.serve.num_seqs + 1, ),
                                          jnp.int32),
        "kv_lens": jax.ShapeDtypeStruct((cfgs.serve.num_seqs, ), jnp.int32),
        "distribution": jax.ShapeDtypeStruct((len(configs.RpaCase), ),
                                             jnp.int32),
        "seq_page_table": jax.ShapeDtypeStruct((seq_page_table_size, ),
                                               jnp.int32),
    }


def schedule_smem_usage_bytes(
    cfgs: config.PrefillConfig,
    steps: int,
    *,
    num_buffers: int = 1,
    seq_page_table_size: int | None = None,
) -> int:
    """Return the compact schedule builder's SMEM footprint."""
    if steps < 1:
        raise ValueError(f"steps must be >= 1, got {steps}.")
    if num_buffers < 1:
        raise ValueError(f"num_buffers must be >= 1, got {num_buffers}.")
    if seq_page_table_size is None:
        seq_page_table_size = cdiv(cfgs.serve.pages_per_seq, 1024) * 1024

    schedule_shapes = _schedule_wrappers(cfgs, steps)
    schedule_shapes["actual_steps"] = jax.ShapeDtypeStruct((1, ), jnp.int32)
    fixed_bytes = _shape_tree_nbytes(
        _metadata_fixed_smem_shapes(cfgs, seq_page_table_size))
    return fixed_bytes + num_buffers * _shape_tree_nbytes(schedule_shapes)


def calculate_max_steps_ub(
    cfgs: config.PrefillConfig,
    *,
    num_buffers: int = 1,
    seq_page_table_size: int | None = None,
) -> int:
    """Find the largest lane-aligned compact schedule fitting in SMEM."""
    if seq_page_table_size is None:
        seq_page_table_size = cdiv(cfgs.serve.pages_per_seq, 1024) * 1024
    num_lanes = pltpu.get_tpu_info().num_lanes
    smem_budget = pltpu.get_tpu_info(
    ).smem_capacity_bytes - _SMEM_HEADROOM_BYTES

    def fits(num_lane_groups: int) -> bool:
        return (schedule_smem_usage_bytes(
            cfgs,
            num_lane_groups * num_lanes,
            num_buffers=num_buffers,
            seq_page_table_size=seq_page_table_size,
        ) <= smem_budget)

    if not fits(1):
        raise ValueError(
            "Sliding-window prefill schedule cannot fit one lane-aligned step "
            f"group in SMEM: {smem_budget=} {num_lanes=} {num_buffers=}.")

    lower = 1
    upper = 2
    while fits(upper):
        lower = upper
        upper *= 2
    while lower + 1 < upper:
        midpoint = (lower + upper) // 2
        if fits(midpoint):
            lower = midpoint
        else:
            upper = midpoint
    return lower * num_lanes


@dataclasses.dataclass(frozen=True)
class SchedulePlan:
    """Static capacity and windowing plan for sliding-window prefill."""

    max_steps_ub: int
    worst_steps: int
    fits_one_window: bool
    sched_window: int
    num_sched_windows: int
    total_steps_ub: int
    seq_page_table_size: int
    compute_bkv_size: int

    @classmethod
    def create(cls, cfgs: config.PrefillConfig) -> "SchedulePlan":
        validate_config(cfgs)
        seq_page_table_size = cdiv(cfgs.serve.pages_per_seq, 1024) * 1024
        max_steps_ub = calculate_max_steps_ub(
            cfgs,
            num_buffers=1,
            seq_page_table_size=seq_page_table_size,
        )
        worst_steps = max(1, cdiv(_max_query_blocks_ub(cfgs), cfgs.batch_size))
        fits_one_window = worst_steps <= max_steps_ub
        sched_window = (worst_steps
                        if fits_one_window else calculate_max_steps_ub(
                            cfgs,
                            num_buffers=2,
                            seq_page_table_size=seq_page_table_size,
                        ))
        num_sched_windows = max(1, cdiv(worst_steps, sched_window))
        compute_bkv_size = block_sizes.compute_window_bkv_size(
            cfgs.model,
            cfgs.serve,
            cfgs.block,
        )
        return cls(
            max_steps_ub=max_steps_ub,
            worst_steps=worst_steps,
            fits_one_window=fits_one_window,
            sched_window=sched_window,
            num_sched_windows=num_sched_windows,
            total_steps_ub=num_sched_windows * sched_window,
            seq_page_table_size=seq_page_table_size,
            compute_bkv_size=compute_bkv_size,
        )


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class SlidingWindowSchedule:
    """Metadata for one independent Q block and its anchored KV tile."""

    # q_src, q_size, processed_q, processed_kv, effective_kv, cache_size,
    # new_kv_start
    cell: SmemWrapper
    # physical_page, transfer_size; VMEM destination is the record's page offset.
    dma_kv_cache: SmemWrapper
    # src_hbm, dst_vmem, dst_hbm, src_vmem, packed fetch/update validity.
    dma_kv_new: SmemWrapper
    actual_steps: Any

    cfgs: config.PrefillConfig = dataclasses.field(metadata=dict(static=True))
    plan: SchedulePlan = dataclasses.field(metadata=dict(static=True))

    @classmethod
    def create_shape_dtype(
        cls,
        cfgs: config.PrefillConfig,
        steps: int | None = None,
        *,
        plan: SchedulePlan | None = None,
    ) -> "SlidingWindowSchedule":
        validate_config(cfgs)
        if plan is None:
            plan = SchedulePlan.create(cfgs)
        if steps is None:
            steps = plan.max_steps_ub
        wrappers = _schedule_wrappers(cfgs, steps)
        return cls(
            **wrappers,
            actual_steps=jax.ShapeDtypeStruct((1, ), jnp.int32),
            cfgs=cfgs,
            plan=plan,
        )

    @classmethod
    def create_hbm_shape_dtype(
        cls,
        cfgs: config.PrefillConfig,
        *,
        plan: SchedulePlan | None = None,
    ) -> "SlidingWindowSchedule":
        if plan is None:
            plan = SchedulePlan.create(cfgs)
        window = cls.create_shape_dtype(cfgs,
                                        steps=plan.sched_window,
                                        plan=plan)

        def grow(value):
            if not isinstance(value, SmemWrapper):
                return value
            return SmemWrapper(
                data=jax.ShapeDtypeStruct(
                    (value.data.shape[0] * plan.num_sched_windows, ),
                    value.data.dtype,
                ),
                shape=value.shape,
            )

        replacements = {
            field.name: grow(getattr(window, field.name))
            for field in dataclasses.fields(window)
            if isinstance(getattr(window, field.name), SmemWrapper)
        }
        return dataclasses.replace(window, **replacements)

    def get_cell_metadata(self, step, batch_idx):
        return tuple(self.cell[step, batch_idx, i] for i in range(_CELL_SIZE))

    def get_dma_q(self, step, batch_idx):
        return (
            self.cell[step, batch_idx, _CELL_Q_SRC],
            self.cell[step, batch_idx, _CELL_Q_SIZE],
        )

    def get_dma_kv_cache(self, step, batch_idx, page_idx):
        physical_page = self.dma_kv_cache[step, batch_idx, page_idx, 0]
        dst_vmem = page_idx * self.cfgs.serve.page_size
        size = self.dma_kv_cache[step, batch_idx, page_idx, 1]
        return physical_page, dst_vmem, size

    def get_dma_fetch_kv_new(self, step, batch_idx, page_idx):
        src_hbm = self.dma_kv_new[step, batch_idx, page_idx, _DMA_NEW_SRC_HBM]
        dst_vmem = self.dma_kv_new[step, batch_idx, page_idx,
                                   _DMA_NEW_DST_VMEM]
        valid = (self.dma_kv_new[step, batch_idx, page_idx, _DMA_NEW_FLAGS]
                 & _FETCH_VALID)
        return src_hbm, dst_vmem, valid

    def get_dma_update_kv_new(self, step, batch_idx, page_idx):
        dst_hbm = self.dma_kv_new[step, batch_idx, page_idx, _DMA_NEW_DST_HBM]
        src_vmem = self.dma_kv_new[step, batch_idx, page_idx,
                                   _DMA_NEW_SRC_VMEM]
        flags = self.dma_kv_new[step, batch_idx, page_idx, _DMA_NEW_FLAGS]
        valid = (flags & _UPDATE_VALID) >> 1
        width, mask = self.cfgs.wb_lane_bits
        wb_lane = ((flags >> 2) & mask) * self.cfgs.num_lanes
        wb_size = ((flags >> (2 + width)) & mask) * self.cfgs.num_lanes
        return dst_hbm, src_vmem, valid, wb_lane, wb_size

    def scratch_shapes(self):
        return jax.tree.map(lambda x: pltpu.SMEM(x.shape, x.dtype), self)

    def in_specs(self):

        def wrapper(value):
            memory_space = pltpu.SMEM if value.size == 1 else pltpu.HBM
            return pl.BlockSpec(memory_space=memory_space)

        return jax.tree.map(wrapper, self)

    def out_specs(self):
        return jax.tree.map(lambda x: pl.BlockSpec(memory_space=pltpu.HBM),
                            self)


def _copy_page_table(
    s_idx,
    page_indices_hbm_ref,
    seq_page_table_ref,
    page_dma_sem,
    *,
    seq_page_table_size: int,
):
    seq_off = pl.multiple_of(s_idx * seq_page_table_size, seq_page_table_size)
    page_copy = pltpu.make_async_copy(
        page_indices_hbm_ref.at[pl.ds(seq_off, seq_page_table_size)],
        seq_page_table_ref.at[pl.ds(0, seq_page_table_size)],
        page_dma_sem.at[0],
    )
    page_copy.start()
    page_copy.wait()


def compute_metadata(
    cu_q_lens_ref: jax.Ref,
    kv_lens_ref: jax.Ref,
    distribution_ref: jax.Ref,
    page_indices_hbm_ref: jax.Ref,
    schedule: SlidingWindowSchedule,
    seq_page_table_ref: jax.Ref,
    page_dma_sem: jax.Ref,
    *,
    cfgs: config.PrefillConfig,
    window_lo: int = 0,
):
    """Emit one compact schedule cell for every active sequence Q block."""
    validate_config(cfgs)
    n = cfgs.batch_size
    page_size = cfgs.serve.page_size
    page_size_log2 = cfgs.serve.page_size_log2
    window_size = schedule.plan.sched_window
    pps_pad = schedule.plan.seq_page_table_size
    start_seq_idx, end_seq_idx = cfgs.mode.get_range(distribution_ref)

    def emit_sequence(s_idx, cursor):
        _copy_page_table(
            s_idx,
            page_indices_hbm_ref,
            seq_page_table_ref,
            page_dma_sem,
            seq_page_table_size=pps_pad,
        )
        q_start = cu_q_lens_ref[s_idx]
        q_end = cu_q_lens_ref[s_idx + 1]
        q_len = q_end - q_start
        k_len = kv_lens_ref[s_idx]
        prefix_len = k_len - q_len
        num_q = pl.cdiv(q_len, cfgs.bq_sz)

        def emit_q_block(q_idx, cursor):
            step = cursor // n
            target_lane = cursor % n
            local = jnp.clip(step - window_lo, 0, window_size - 1)
            in_window = jnp.logical_and(
                step >= window_lo,
                step < window_lo + window_size,
            )
            q_src = q_start + q_idx * cfgs.bq_sz
            q_size = jnp.clip(q_end - q_src, 0, cfgs.bq_sz)
            processed_q = prefix_len + q_idx * cfgs.bq_sz
            processed_kv = (jnp.maximum(
                processed_q - cfgs.model.sliding_window,
                0,
            ) >> page_size_log2) << page_size_log2

            kv_left = jnp.maximum(k_len - processed_kv, 0)
            if cfgs.update_kv_cache:
                cache_size = jnp.clip(
                    prefix_len - processed_kv,
                    0,
                    cfgs.bkv_sz,
                )
            else:
                cache_size = jnp.minimum(kv_left, cfgs.bkv_sz)
            new_left = jnp.maximum(kv_left - cache_size, 0)
            new_size = jnp.minimum(cfgs.bkv_sz - cache_size, new_left)
            new_kv_start = q_end - new_left

            @pl.when(in_window)
            def _emit():
                schedule.cell[local, target_lane, _CELL_Q_SRC] = q_src
                schedule.cell[local, target_lane, _CELL_Q_SIZE] = q_size
                schedule.cell[local, target_lane,
                              _CELL_PROCESSED_Q] = processed_q
                schedule.cell[local, target_lane,
                              _CELL_PROCESSED_KV] = processed_kv
                schedule.cell[local, target_lane, _CELL_EFFECTIVE_KV] = k_len
                schedule.cell[local, target_lane,
                              _CELL_CACHE_SIZE] = cache_size
                schedule.cell[local, target_lane,
                              _CELL_NEW_KV_START] = new_kv_start

                cache_pages = pl.cdiv(cache_size, page_size)
                kv_page_start = processed_kv >> page_size_log2
                for i in range(cfgs.bkv_p_cache):
                    cache_dma_size = utils.align_to(
                        jnp.clip(cache_size - i * page_size, 0, page_size),
                        cfgs.num_lanes,
                    )
                    valid = cache_dma_size > 0
                    page_idx = jnp.minimum(
                        kv_page_start + i,
                        cfgs.serve.pages_per_seq - 1,
                    )
                    schedule.dma_kv_cache[local, target_lane, i,
                                          0] = jnp.where(
                                              valid,
                                              seq_page_table_ref[page_idx], 0)
                    schedule.dma_kv_cache[local, target_lane, i,
                                          1] = cache_dma_size

                new_token_offset = new_kv_start & (page_size - 1)
                fetch_page_start = new_kv_start - new_token_offset
                num_fetch_pages = jnp.where(
                    new_size > 0,
                    pl.cdiv(new_token_offset + new_size, page_size),
                    0,
                )

                # A cache page is owned by the Q block containing that page's
                # first new token. The first Q block also owns the partial page
                # containing the cache/new boundary. This partitions writeback
                # pages even when adjacent Q blocks execute in parallel lanes.
                first_update_page = jnp.where(
                    q_idx == 0,
                    prefix_len >> page_size_log2,
                    pl.cdiv(processed_q, page_size),
                )
                update_page_end = pl.cdiv(processed_q + q_size, page_size)
                num_update_pages = jnp.maximum(
                    update_page_end - first_update_page,
                    0,
                )

                for i in range(cfgs.bkv_p_new):
                    fetch_valid = jnp.where(i < num_fetch_pages, _FETCH_VALID,
                                            0)
                    fetch_src = fetch_page_start + i * page_size
                    fetch_dst = (cache_pages + i) * page_size

                    update_valid = jnp.where(
                        cfgs.update_kv_cache & (i < num_update_pages),
                        _UPDATE_VALID,
                        0,
                    )
                    update_page = jnp.minimum(
                        first_update_page + i,
                        cfgs.serve.pages_per_seq - 1,
                    )
                    update_src = jnp.clip(
                        (update_page << page_size_log2) - processed_kv,
                        0,
                        cfgs.bkv_sz - page_size,
                    )
                    update_page_start = update_page << page_size_log2
                    dirty_lo = jnp.maximum(update_page_start, new_kv_start)
                    dirty_hi = jnp.minimum(update_page_start + page_size,
                                           new_kv_start + new_size)
                    lane_lo = jnp.clip(dirty_lo - update_page_start, 0,
                                       page_size)
                    lane_hi = jnp.clip(dirty_hi - update_page_start, 0,
                                       page_size)
                    wb_lane = (lane_lo // cfgs.num_lanes) * cfgs.num_lanes
                    wb_end = pl.cdiv(lane_hi, cfgs.num_lanes) * cfgs.num_lanes
                    wb_size = jnp.where(
                        (update_valid != 0) & (dirty_hi > dirty_lo),
                        wb_end - wb_lane,
                        0,
                    )
                    width, _ = cfgs.wb_lane_bits
                    flags = (fetch_valid | update_valid
                             | ((wb_lane // cfgs.num_lanes) << 2)
                             | ((wb_size // cfgs.num_lanes) << (2 + width)))

                    schedule.dma_kv_new[local, target_lane, i,
                                        _DMA_NEW_SRC_HBM] = (jnp.where(
                                            fetch_valid != 0, fetch_src, 0))
                    schedule.dma_kv_new[local, target_lane, i,
                                        _DMA_NEW_DST_VMEM] = (jnp.where(
                                            fetch_valid != 0, fetch_dst, 0))
                    schedule.dma_kv_new[local, target_lane, i,
                                        _DMA_NEW_DST_HBM] = (jnp.where(
                                            update_valid != 0,
                                            seq_page_table_ref[update_page],
                                            0,
                                        ))
                    schedule.dma_kv_new[local, target_lane, i,
                                        _DMA_NEW_SRC_VMEM] = (jnp.where(
                                            update_valid != 0, update_src, 0))
                    schedule.dma_kv_new[local, target_lane, i,
                                        _DMA_NEW_FLAGS] = flags

            return cursor + 1

        return jax.lax.fori_loop(0, num_q, emit_q_block, cursor)

    return jax.lax.fori_loop(
        start_seq_idx,
        end_seq_idx,
        emit_sequence,
        jnp.int32(0),
    )


def rpa_metadata_schedule_kernel(
    cu_q_lens_ref: jax.Ref,
    kv_lens_ref: jax.Ref,
    distribution_ref: jax.Ref,
    page_indices_hbm_ref: jax.Ref,
    schedule_hbm_ref: SlidingWindowSchedule,
    schedule_ref: SlidingWindowSchedule,
    dma_sem: jax.Ref,
    seq_page_table_ref: jax.Ref,
    page_dma_sem: jax.Ref,
    *,
    cfgs: config.PrefillConfig,
) -> None:
    """Build compact sliding-window prefill metadata in SMEM windows."""
    plan = schedule_ref.plan
    window_size = plan.sched_window
    flat_hbm = jax.tree_util.tree_leaves(schedule_hbm_ref)
    flat_smem = jax.tree_util.tree_leaves(schedule_ref)

    def mask_window(step, _):
        for b_idx in range(cfgs.batch_size):
            for i in range(_CELL_SIZE):
                schedule_ref.cell[step, b_idx, i] = 0
            for i in range(cfgs.bkv_p_cache):
                for j in range(2):
                    schedule_ref.dma_kv_cache[step, b_idx, i, j] = 0
            for i in range(cfgs.bkv_p_new):
                for j in range(cfgs.dma_kv_new_size):
                    schedule_ref.dma_kv_new[step, b_idx, i, j] = 0
        return None

    def build_window(window_idx):
        window_lo = window_idx * window_size
        jax.lax.fori_loop(0, window_size, mask_window, None)
        total_cells = compute_metadata(
            cu_q_lens_ref,
            kv_lens_ref,
            distribution_ref,
            page_indices_hbm_ref,
            schedule_ref,
            seq_page_table_ref,
            page_dma_sem,
            cfgs=cfgs,
            window_lo=window_lo,
        )

        copies = []
        for hbm_ref, smem_ref in zip(flat_hbm, flat_smem):
            if hbm_ref.shape[0] > 1:
                smem_len = smem_ref.shape[0]
                copy = pltpu.make_async_copy(
                    smem_ref.at[pl.ds(0, smem_len)],
                    hbm_ref.at[pl.ds(
                        pl.multiple_of(window_idx * smem_len, smem_len),
                        smem_len,
                    )],
                    dma_sem.at[0],
                )
                copy.start()
                copies.append(copy)
        jax.tree.map(lambda copy: copy.wait(), copies)
        return total_cells

    total_cells = build_window(0)
    actual_steps = pl.cdiv(total_cells, cfgs.batch_size)
    schedule_ref.actual_steps[0] = actual_steps

    for hbm_ref, smem_ref in zip(flat_hbm, flat_smem):
        if hbm_ref.shape[0] == 1:
            copy = pltpu.make_async_copy(
                smem_ref.at[pl.ds(0, 1)],
                hbm_ref.at[pl.ds(0, 1)],
                dma_sem.at[0],
            )
            copy.start()
            copy.wait()

    num_windows_actual = jnp.minimum(
        pl.cdiv(actual_steps, window_size),
        plan.num_sched_windows,
    )

    def build_remaining(window_idx, _):
        build_window(window_idx)
        return None

    jax.lax.fori_loop(1, num_windows_actual, build_remaining, None)


def generate_rpa_metadata(
    cu_q_lens: jax.Array,
    kv_lens: jax.Array,
    distribution: jax.Array,
    page_indices: jax.Array,
    cfgs: config.PrefillConfig,
    *,
    interpret: bool = False,
) -> SlidingWindowSchedule:
    """Generate the compact HBM schedule for sliding-window prefill."""
    validate_config(cfgs)
    plan = SchedulePlan.create(cfgs)
    hbm_shaped = SlidingWindowSchedule.create_hbm_shape_dtype(cfgs, plan=plan)
    smem_shaped = SlidingWindowSchedule.create_shape_dtype(
        cfgs,
        steps=plan.sched_window,
        plan=plan,
    )
    page_indices_padded = jnp.pad(
        page_indices.reshape(cfgs.serve.num_seqs, cfgs.serve.pages_per_seq),
        ((0, 0), (0, plan.seq_page_table_size - cfgs.serve.pages_per_seq)),
    ).reshape(-1)

    return pl.pallas_call(
        functools.partial(rpa_metadata_schedule_kernel, cfgs=cfgs),
        out_shape=hbm_shaped,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=3,
            in_specs=[pl.BlockSpec(memory_space=pltpu.HBM)],
            out_specs=hbm_shaped.out_specs(),
            scratch_shapes=[
                smem_shaped.scratch_shapes(),
                pltpu.SemaphoreType.DMA((1, )),
                pltpu.SMEM((plan.seq_page_table_size, ), jnp.int32),
                pltpu.SemaphoreType.DMA((1, )),
            ],
        ),
        interpret=interpret,
        name="rpa_metadata_schedule_prefill_sliding_window",
    )(
        cu_q_lens,
        kv_lens,
        distribution,
        page_indices_padded,
    )
