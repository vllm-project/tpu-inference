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

from tpu_inference.kernels.experimental.batched_rpa import configs, utils


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class SmemWrapper:
    """Maps physical 1-D data into logical N-D representation."""

    data: Any
    shape: tuple[int, ...] = dataclasses.field(metadata=dict(static=True))

    @classmethod
    def create_shape_dtype(cls, shape):
        return cls(data=jax.ShapeDtypeStruct((np.prod(shape), ), jnp.int32),
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
class RPASchedule:
    """Container for metadata arrays with integrated shape/spec logic."""

    s_idx: SmemWrapper  # [steps, batch]
    q_idx: SmemWrapper  # [steps, batch]
    k_idx: SmemWrapper  # [steps, batch]
    is_last_k: SmemWrapper  # [steps, batch]
    do_writeback: SmemWrapper  # [steps, batch]
    dma_q: SmemWrapper  # [steps, batch, 2]
    dma_kv_cache: SmemWrapper  # [steps, batch, bkv_p_cache, 3]
    dma_kv_new: SmemWrapper  # [steps, batch, bkv_p_new, 4]
    actual_steps: Any  # [1]

    cfgs: configs.RPAConfig = dataclasses.field(metadata=dict(static=True))

    @classmethod
    def create_shape_dtype(cls, cfgs: configs.RPAConfig):

        idx_wrapper = SmemWrapper.create_shape_dtype(
            (cfgs.max_steps_ub, cfgs.batch_size))

        return cls(
            s_idx=idx_wrapper,
            q_idx=idx_wrapper,
            k_idx=idx_wrapper,
            is_last_k=idx_wrapper,
            do_writeback=idx_wrapper,
            dma_q=SmemWrapper.create_shape_dtype(
                (cfgs.max_steps_ub, cfgs.batch_size, 2)),
            dma_kv_cache=SmemWrapper.create_shape_dtype(
                (cfgs.max_steps_ub, cfgs.batch_size, cfgs.bkv_p_cache, 3)),
            dma_kv_new=SmemWrapper.create_shape_dtype(
                (cfgs.max_steps_ub, cfgs.batch_size, cfgs.bkv_p_new, 4), ),
            actual_steps=jax.ShapeDtypeStruct((1, ), jnp.int32),
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

    def get_dma_kv_new(
        self,
        step: jax.typing.ArrayLike,
        batch_idx: jax.typing.ArrayLike,
        page_idx: jax.typing.ArrayLike,
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        # 0: dst_hbm, 1: src_hbm, 2: dst_vmem, 3: size
        dst_hbm = self.dma_kv_new[step, batch_idx, page_idx, 0]
        src_hbm = self.dma_kv_new[step, batch_idx, page_idx, 1]
        dst_vmem = self.dma_kv_new[step, batch_idx, page_idx, 2]
        sz = self.dma_kv_new[step, batch_idx, page_idx, 3]
        return dst_hbm, src_hbm, dst_vmem, sz

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
    schedule: RPASchedule,
    lane_lengths_ref: jax.Ref,
    *,
    cfgs: configs.RPAConfig,
):
    """Fill metadata using triple nested loop of seq->q->k loop."""

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

        schedule.s_idx[step, target_lane] = s_idx
        schedule.q_idx[step, target_lane] = q_idx
        schedule.k_idx[step, target_lane] = k_idx

        is_last_k = jnp.where(k_idx == end_k_idx - 1, 1, 0)
        schedule.is_last_k[step, target_lane] = is_last_k

        schedule.dma_q[step, target_lane, 0] = q_src
        schedule.dma_q[step, target_lane, 1] = q_sz_task

        kv_len_start = k_idx * cfgs.bkv_sz
        kv_p_start = k_idx * cfgs.bkv_p
        kv_left = k_len - kv_len_start
        kv_left_frm_cache = jnp.maximum(kv_left - q_len, 0)
        p_offset = s_idx * cfgs.serve.pages_per_seq + kv_p_start

        for i in range(cfgs.bkv_p_cache):
            dst_vmem = i << cfgs.serve.page_size_log2
            dma_sz = kv_left_frm_cache - dst_vmem
            dma_sz = jnp.clip(dma_sz, 0, cfgs.serve.page_size)

            src_hbm = jnp.minimum(p_offset + i,
                                  cfgs.serve.num_page_indices - 1)

            schedule.dma_kv_cache[step, target_lane, i, 0] = src_hbm
            schedule.dma_kv_cache[step, target_lane, i, 1] = dst_vmem
            schedule.dma_kv_cache[step, target_lane, i, 2] = dma_sz

        kv_left_frm_new = kv_left - kv_left_frm_cache
        bkv_sz_cache = jnp.minimum(kv_left_frm_cache, cfgs.bkv_sz)
        new_sz = jnp.minimum(cfgs.bkv_sz - bkv_sz_cache, kv_left_frm_new)

        # Writeback logic: each new k block is written back by the first q block
        # that attends to it.
        q_wb = jnp.maximum(0, (kv_len_start - (k_len - q_len))) // cfgs.bq_sz

        do_writeback = jnp.where((new_sz > 0) & (q_idx == q_wb), 1, 0)
        schedule.do_writeback[step, target_lane] = do_writeback
        src_hbm = q_end - kv_left_frm_new

        if cfgs.bkv_p_new < cfgs.bkv_p:
            # Special case where we only need to write back one page.
            assert cfgs.bkv_p_new == 1
            dst_vmem = bkv_sz_cache
            dma_sz = new_sz

            p_idx = (kv_len_start + dst_vmem) >> cfgs.serve.page_size_log2
            p_idx = jnp.minimum(p_idx, cfgs.serve.pages_per_seq - 1)
            p_off = (kv_len_start + dst_vmem) & cfgs.serve.page_size_mask
            global_p_idx = s_idx * cfgs.serve.pages_per_seq + p_idx

            dst_hbm = (global_p_idx << cfgs.serve.page_size_log2) | p_off
            schedule.dma_kv_new[step, target_lane, 0, 0] = dst_hbm
            schedule.dma_kv_new[step, target_lane, 0, 1] = src_hbm
            schedule.dma_kv_new[step, target_lane, 0, 2] = dst_vmem
            schedule.dma_kv_new[step, target_lane, 0, 3] = dma_sz
        else:
            for i in range(cfgs.bkv_p):
                slot_start = i << cfgs.serve.page_size_log2
                slot_end = (i + 1) << cfgs.serve.page_size_log2

                dst_vmem = jnp.maximum(slot_start, bkv_sz_cache)
                end_in_slot = jnp.minimum(slot_end, bkv_sz_cache + new_sz)
                dma_sz = jnp.maximum(0, end_in_slot - dst_vmem)

                p_idx = (kv_len_start + dst_vmem) >> cfgs.serve.page_size_log2
                p_idx = jnp.minimum(p_idx, cfgs.serve.pages_per_seq - 1)
                p_off = (kv_len_start + dst_vmem) & cfgs.serve.page_size_mask
                global_p_idx = s_idx * cfgs.serve.pages_per_seq + p_idx

                dst_hbm = (global_p_idx << cfgs.serve.page_size_log2) | p_off
                schedule.dma_kv_new[step, target_lane, i, 0] = dst_hbm
                schedule.dma_kv_new[step, target_lane, i, 1] = src_hbm
                schedule.dma_kv_new[step, target_lane, i, 2] = dst_vmem
                schedule.dma_kv_new[step, target_lane, i, 3] = dma_sz

        return step + 1

    @jax.named_scope("q_loop")
    def q_loop(q_idx, _, *, s_idx, q_start, q_end, k_len, q_len, num_k):
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
        if (sliding_window := cfgs.model.sliding_window) is not None:
            sw_start_idx = k_len - q_len + q_idx * cfgs.bq_sz - sliding_window + 1
            start_k_idx = jnp.maximum(0, sw_start_idx) // cfgs.bkv_sz

        end_k_idx_causal = (k_len - q_len + q_idx * cfgs.bq_sz + q_sz_task -
                            1) // cfgs.bkv_sz + 1
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

    start_seq_idx, end_seq_idx = cfgs.mode.get_range(distribution_ref)
    jax.lax.fori_loop(start_seq_idx, end_seq_idx, seq_loop, None)


def rpa_metadata_schedule_kernel(
    ## Scalar prefetch.
    cu_q_lens_ref: jax.Ref,
    kv_lens_ref: jax.Ref,
    distribution_ref: jax.Ref,
    # Outputs.
    schedule_hbm_ref: RPASchedule,
    # Scratch.
    schedule_ref: RPASchedule,
    lane_lengths_ref: jax.Ref,
    dma_sem: jax.Ref,
    *,
    cfgs: configs.RPAConfig,
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

    for b_idx in range(cfgs.batch_size):
        lane_lengths_ref[b_idx] = 0

    # Step 1: Compute and fill scheduler metadata.
    compute_metadata(
        cu_q_lens_ref,
        kv_lens_ref,
        distribution_ref,
        schedule_ref,
        lane_lengths_ref,
        cfgs=cfgs,
    )

    # Step 2: Compute actual number of steps.
    max_steps = 0
    for b_idx in range(cfgs.batch_size):
        max_steps = jnp.maximum(max_steps, lane_lengths_ref[b_idx])
    schedule_ref.actual_steps[0] = max_steps

    safe_max_steps = jnp.minimum(max_steps + cfgs.n_buffer + 1,
                                 cfgs.max_steps_ub)

    # Step 3: Mask out unvisited steps.
    @jax.named_scope("mask_out_steps")
    def mask_out_steps(step, _, *, b_idx):
        schedule_ref.s_idx[step, b_idx] = -1
        schedule_ref.q_idx[step, b_idx] = 0
        schedule_ref.k_idx[step, b_idx] = 0
        schedule_ref.is_last_k[step, b_idx] = 0
        schedule_ref.do_writeback[step, b_idx] = 0

        schedule_ref.dma_q[step, b_idx, 0] = 0
        schedule_ref.dma_q[step, b_idx, 1] = 0

        for i in range(cfgs.bkv_p_cache):
            schedule_ref.dma_kv_cache[step, b_idx, i, 0] = 0
            schedule_ref.dma_kv_cache[step, b_idx, i, 1] = 0
            schedule_ref.dma_kv_cache[step, b_idx, i, 2] = 0

        for i in range(cfgs.bkv_p_new):
            schedule_ref.dma_kv_new[step, b_idx, i, 0] = 0
            schedule_ref.dma_kv_new[step, b_idx, i, 1] = 0
            schedule_ref.dma_kv_new[step, b_idx, i, 2] = 0
            schedule_ref.dma_kv_new[step, b_idx, i, 3] = 0

    for b_idx in range(cfgs.batch_size):
        start_step = lane_lengths_ref[b_idx]
        mask_step_fn = functools.partial(mask_out_steps, b_idx=b_idx)
        jax.lax.fori_loop(start_step, safe_max_steps, mask_step_fn, None)

    # Ste 4: Write back results to HBM.
    flat_hbm = jax.tree_util.tree_leaves(schedule_hbm_ref)
    flat_smem = jax.tree_util.tree_leaves(schedule_ref)
    dma_list = []
    for h, s in zip(flat_hbm, flat_smem):
        write_size = h.shape[0]
        if write_size > 1:
            write_size = (write_size // cfgs.max_steps_ub) * safe_max_steps
            write_size = utils.align_to(write_size, 1024)

        copy = pltpu.make_async_copy(
            s.at[pl.ds(0, write_size)],
            h.at[pl.ds(0, write_size)],
            dma_sem.at[0],
        )
        dma_list.append(copy)

    jax.tree.map(lambda x: x.start(), dma_list)
    jax.tree.map(lambda x: x.wait(), dma_list)


def generate_rpa_metadata(
    cu_q_lens: jax.Array,
    kv_lens: jax.Array,
    distribution: jax.Array,
    cfgs: configs.RPAConfig,
    *,
    interpret=False,
) -> RPASchedule:
    schedule_shaped_dtype = RPASchedule.create_shape_dtype(cfgs)

    return pl.pallas_call(
        functools.partial(rpa_metadata_schedule_kernel, cfgs=cfgs),
        out_shape=schedule_shaped_dtype,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=3,
            in_specs=[],
            out_specs=schedule_shaped_dtype.out_specs(),
            scratch_shapes=[
                schedule_shaped_dtype.scratch_shapes(),
                pltpu.SMEM((cfgs.batch_size, ), jnp.int32),
                pltpu.SemaphoreType.DMA((1, )),
            ],
        ),
        interpret=interpret,
        name="rpa_metadata_schedule",
    )(cu_q_lens, kv_lens, distribution)
