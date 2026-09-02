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
"""Context Parallel (CP) DMA schedule computation for Batched RPA."""

import dataclasses
from collections.abc import Sequence

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl

from tpu_inference.kernels.experimental.batched_rpa import configs, schedule


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class HeadAlongSublaneDmaNewCP(schedule.DmaNew):
    """Like HeadAlongSublaneDmaNew but with separate fetch/wb flags for CP.

    wb_val encodes per-page ownership (dma_sz if this rank owns the page,
    else 0), so bref_override copy_out needs no CP-specific logic.
    """

    wb_hbm = schedule.FieldOffset(0)
    fetch_hbm = schedule.FieldOffset(1)
    fetch_vmem = schedule.FieldOffset(2)
    wb_vmem = schedule.FieldOffset(2)
    _fetch_flags = schedule.FieldOffset(3)
    _wb_flags = schedule.FieldOffset(4)

    @staticmethod
    def num_fields() -> int:
        return 5

    @property
    def fetch_val(self):
        return self._fetch_flags[...]

    @property
    def wb_val(self):
        return self._wb_flags[...]

    def set_flags(self, fetch_val, wb_val):
        self._fetch_flags[...] = fetch_val
        self._wb_flags[...] = wb_val


class CPMetadataComputer(schedule.BaseMetadataComputer):
    """Context Parallel (CP) metadata scheduler: overrides only k_loop and schema."""

    @classmethod
    def get_rpa_schedule(cls,
                         cfgs: configs.RpaConfigs,
                         multiplier: int = 1) -> schedule.RpaSchedule:
        """Returns the RpaSchedule shape/dtype struct with CP-specific DMA struct."""
        dma_kv_new_struct_cls = (HeadAlongSublaneDmaNewCP
                                 if cfgs.serve.kv_layout
                                 == configs.KVLayout.HEAD_ALONG_SUBLANE else
                                 schedule.SeqAlongLaneDmaNew)
        return schedule.RpaSchedule.create_shape_dtype(
            cfgs,
            dma_kv_new_struct_cls=dma_kv_new_struct_cls,
            multiplier=multiplier,
        )

    def __init__(
            self,
            schedule: schedule.RpaSchedule,
            schedule_hbm_ref: schedule.RpaSchedule,
            dma_sem: jax.Ref,
            *,
            cfgs: configs.RpaConfigs,
            extra_refs: Sequence[jax.Ref] = (),
            **kwargs,
    ):
        super().__init__(
            schedule=schedule,
            schedule_hbm_ref=schedule_hbm_ref,
            dma_sem=dma_sem,
            cfgs=cfgs,
            extra_refs=extra_refs,
            **kwargs,
        )

    @property
    def rank(self):
        return self.extra_refs[0][0] if self.extra_refs else 0

    @jax.named_scope("k_loop_cp")
    def k_loop(
        self,
        k_idx,
        carry: schedule.LoopCarry,
        *,
        s_idx,
        q_idx,
        q_end,
        q_src,
        q_sz_task,
        kv_cache_len,
        kv_new_len,
        end_k_idx,
    ) -> schedule.LoopCarry:
        cfgs = self.cfgs
        sched = self.schedule
        count = carry.count
        step, target_lane = divmod(count, cfgs.batch_size)

        sched.s_idx[step, target_lane] = s_idx
        sched.q_idx[step, target_lane] = q_idx
        sched.k_idx[step, target_lane] = k_idx

        is_last_k = jnp.where(k_idx == end_k_idx - 1, 1, 0)
        sched.is_last_k[step, target_lane] = is_last_k

        sched.dma_q[step, target_lane, 0] = q_src
        sched.dma_q[step, target_lane, 1] = q_sz_task

        kv_len_start = k_idx * cfgs.bkv_sz
        kv_p_start = k_idx * cfgs.bkv_p
        k_len = kv_cache_len + kv_new_len
        kv_left = k_len - kv_len_start
        kv_left_frm_cache = jnp.maximum(kv_cache_len - kv_len_start, 0)
        p_offset = s_idx * cfgs.serve.pages_per_seq + kv_p_start

        cp_group_size = cfgs.serve.cp_group_size
        assert cp_group_size is not None

        for i in range(cfgs.bkv_p_cache):
            dst_vmem = i << cfgs.serve.page_size_log2
            dma_sz = kv_left_frm_cache - dst_vmem
            dma_sz = jnp.clip(dma_sz, 0, cfgs.serve.page_size)

            if cfgs.serve.kv_layout == configs.KVLayout.SEQ_ALONG_LANE:
                if cfgs.serve.attention_scope == configs.AttentionScope.NEW_TOKENS_ONLY:
                    p_idx = kv_p_start + i
                    local_page_i = jnp.minimum(
                        p_idx // cp_group_size,
                        cfgs.serve.pages_per_seq - 1,
                    )
                    src_hbm_cp = s_idx * cfgs.serve.pages_per_seq + local_page_i
                    dma_valid = jnp.where(
                        (dma_sz > 0) & (p_idx % cp_group_size == self.rank),
                        1,
                        0,
                    )
                    sched.dma_kv_cache[step, target_lane, i, 0] = src_hbm_cp
                else:  # CACHE_ONLY
                    src_hbm = jnp.minimum(p_offset + i,
                                          cfgs.serve.num_page_indices - 1)
                    dma_valid = jnp.where(dma_sz > 0, 1, 0)
                    sched.dma_kv_cache[step, target_lane, i, 0] = src_hbm
                sched.dma_kv_cache[step, target_lane, i, 1] = dst_vmem
                sched.dma_kv_cache[step, target_lane, i, 2] = dma_valid
            else:
                src_hbm = jnp.minimum(p_offset + i,
                                      cfgs.serve.num_page_indices - 1)
                sched.dma_kv_cache[step, target_lane, i, 0] = src_hbm
                sched.dma_kv_cache[step, target_lane, i, 1] = dst_vmem
                if cfgs.serve.attention_scope == configs.AttentionScope.NEW_TOKENS_ONLY:
                    sched.dma_kv_cache[step, target_lane, i, 2] = 0
                else:
                    sched.dma_kv_cache[step, target_lane, i, 2] = dma_sz

        kv_left_frm_new = kv_left - kv_left_frm_cache
        bkv_sz_cache = jnp.minimum(kv_left_frm_cache, cfgs.bkv_sz)
        new_sz = jnp.minimum(cfgs.bkv_sz - bkv_sz_cache, kv_left_frm_new)

        # Writeback logic: each new k block is written back by the first q block
        # that attends to it.
        q_wb = jnp.maximum(0, (kv_len_start - kv_cache_len)) // cfgs.bq_sz

        do_writeback = jnp.where((new_sz > 0) & (q_idx == q_wb), 1, 0)
        sched.do_writeback[step, target_lane] = do_writeback
        src_hbm = q_end - kv_left_frm_new

        def fill_dma_kv_new(i, dst_vmem, dma_sz, slot_start):
            dma_entry = sched.dma_kv_new[step, target_lane, i]
            if cfgs.serve.kv_layout == configs.KVLayout.SEQ_ALONG_LANE:
                cache_pages = pl.cdiv(bkv_sz_cache, cfgs.serve.page_size)
                hbm_token_idx_base = q_end - kv_left_frm_new
                new_tok_offset = hbm_token_idx_base % cfgs.serve.page_size
                num_pages_to_fetch = jnp.where(
                    new_sz > 0,
                    (new_tok_offset + new_sz - 1) // cfgs.serve.page_size + 1,
                    0,
                )
                fetch_val = jnp.where(i < num_pages_to_fetch, 1, 0)
                new_page_start = (hbm_token_idx_base -
                                  new_tok_offset) + i * cfgs.serve.page_size
                fetch_vmem = (cache_pages + i) * cfgs.serve.page_size
                p_idx = jnp.minimum(
                    (kv_len_start + slot_start) >> cfgs.serve.page_size_log2,
                    cfgs.serve.pages_per_seq * cp_group_size - 1,
                )
                local_slot = p_idx // cp_group_size
                dst_hbm = s_idx * cfgs.serve.pages_per_seq + local_slot
                wb_val = jnp.where(
                    (dma_sz > 0) & (p_idx % cp_group_size == self.rank), 1, 0)

                dma_entry.fetch_hbm[...] = new_page_start
                dma_entry.fetch_vmem[...] = fetch_vmem
                dma_entry.wb_hbm[...] = dst_hbm
                dma_entry.wb_vmem[...] = slot_start
                dma_entry.set_flags(fetch_val, wb_val)
            else:
                tok_idx = kv_len_start + dst_vmem
                p_idx = jnp.minimum(
                    tok_idx >> cfgs.serve.page_size_log2,
                    cfgs.serve.pages_per_seq * cp_group_size - 1,
                )
                p_off = tok_idx & cfgs.serve.page_size_mask
                local_slot = p_idx // cp_group_size
                dst_hbm = ((s_idx * cfgs.serve.pages_per_seq + local_slot) <<
                           cfgs.serve.page_size_log2) | p_off
                wb_val = jnp.where(p_idx % cp_group_size == self.rank, dma_sz,
                                   jnp.int32(0))

                dma_entry.fetch_hbm[...] = src_hbm
                dma_entry.fetch_vmem[...] = dst_vmem
                dma_entry.wb_hbm[...] = dst_hbm
                dma_entry.set_flags(dma_sz, wb_val)

        if cfgs.block.bq_sz == 1:
            # Decode path
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

        return self.advance_carry(carry)
