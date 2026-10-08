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

from collections.abc import Sequence
import dataclasses
import functools

import jax
from jax import lax
from jax.experimental import pallas as pl
import jax.numpy as jnp

from tpu_inference.kernels.experimental.batched_rpa_long_ctx import configs
from tpu_inference.kernels.experimental.batched_rpa_long_ctx import schedule
from tpu_inference.kernels.experimental.batched_rpa_long_ctx import utils


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
  def get_rpa_schedule(
      cls, cfgs: configs.RpaConfigs, multiplier: int = 1
  ) -> schedule.RpaSchedule:
    """Returns the RpaSchedule shape/dtype struct with CP-specific DMA struct."""
    dma_kv_new_struct_cls = (
        HeadAlongSublaneDmaNewCP
        if cfgs.serve.kv_layout == configs.KVLayout.HEAD_ALONG_SUBLANE
        else schedule.SeqAlongLaneDmaNew
    )
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

  @property
  def schedule_rank(self):
    """Whose cache stripe this schedule walks.

    Normally this rank's own. The ring makes every rank walk rank 0's -- the
    longest -- so they all run the same number of KV blocks and rotate in
    lockstep, and a short rank's tail gets masked in the kernel.
    """
    return 0 if self.cfgs.ring_enabled else self.rank

  @jax.named_scope("q_loop_cp")
  def q_loop(
      self,
      q_idx,
      carry: schedule.LoopCarry,
      *,
      s_idx,
      q_start,
      q_end,
      q_position,
      global_kv_cache_len,
      kv_new_len,
      num_k,
      num_q,
      update_kv_cache,
  ) -> schedule.LoopCarry:
    """The cache is striped across the group, so this call walks only part of

    it; a sequence's new KV can reach past what its own Q attends to; and
    the ring visits every block once per hop instead of once at all.
    """
    cfgs = self.cfgs
    if cfgs.serve.attention_scope == configs.AttentionScope.NEW_TOKENS_ONLY:
      kv_cache_len = global_kv_cache_len
    else:
      kv_cache_len = cfgs.local_kv_cache_len(
          global_kv_cache_len, self.schedule_rank
      )
    kv_new_start = q_start
    num_k = pl.cdiv(kv_cache_len + kv_new_len, cfgs.bkv_sz)

    q_src = q_start + q_idx * cfgs.bq_sz
    q_sz_task = jnp.clip(q_end - q_src, 0, cfgs.bq_sz)

    start_k_idx = 0
    if (sliding_window := cfgs.model.sliding_window) is not None:
      sw_start_idx = q_position + q_idx * cfgs.bq_sz - sliding_window + 1
      start_k_idx = jnp.maximum(0, sw_start_idx) // cfgs.bkv_sz

    end_k_idx_causal = (
        q_position + q_idx * cfgs.bq_sz + q_sz_task - 1
    ) // cfgs.bkv_sz + 1
    end_k_idx = jnp.minimum(num_k, end_k_idx_causal)

    if cfgs.serve.attention_scope == configs.AttentionScope.NEW_TOKENS_ONLY:
      start_k_idx = jnp.maximum(start_k_idx, kv_cache_len // cfgs.bkv_sz)

    if cfgs.serve.write_last_q_block:
      # visit every new-KV block. The extra blocks are fully masked
      # out by `generate_mask`, so only their cache write runs.
      end_k_idx = jnp.where(
          (q_idx == num_q - 1) & update_kv_cache, num_k, end_k_idx
      )

    if cfgs.ring_enabled:
      start_k_idx = 0
      num_k_groups = pl.cdiv(num_k, cfgs.batch_size)
      end_k_idx = num_k_groups * cfgs.serve.cp_group_size * cfgs.batch_size

    k_loop_fn = functools.partial(
        self.k_loop,
        s_idx=s_idx,
        q_idx=q_idx,
        q_end=q_end,
        q_src=q_src,
        q_sz_task=q_sz_task,
        kv_cache_len=kv_cache_len,
        kv_new_len=kv_new_len,
        kv_new_start=kv_new_start,
        end_k_idx=end_k_idx,
        num_k=num_k,
        num_q=num_q,
        update_kv_cache=update_kv_cache,
    )

    return jax.lax.fori_loop(start_k_idx, end_k_idx, k_loop_fn, carry)

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
      update_kv_cache,
      kv_new_start=0,
      num_k=0,
      num_q=0,
      **kwargs,
  ) -> schedule.LoopCarry:
    """Like the base k_loop, but the cache is striped over the CP group.

    Page `p` of a sequence lives on rank `p % cp_group_size`, in that
    rank's local slot `p // cp_group_size`.

    CACHE_ONLY: `kv_cache_len` is this rank's own share of the cache
    (`utils.cp_local_cache_len`).

    NEW_TOKENS_ONLY: `kv_cache_len` is the global cache length (no cache
    KV is fetched, it only positions the new tokens).
    """
    cfgs = self.cfgs
    sched = self.schedule
    count = carry.count
    step, target_lane = divmod(count, cfgs.batch_size)

    ring = cfgs.ring_enabled
    loop_k_idx = k_idx
    is_round_0 = False
    is_valid_task = True
    if ring:
      cp_group_size = cfgs.serve.cp_group_size
      assert cp_group_size is not None
      step_within_q = loop_k_idx // cfgs.batch_size
      k_group = step_within_q // cp_group_size
      hop = lax.rem(step_within_q, cp_group_size)
      is_round_0 = hop == 0
      local_k_idx = k_group * cfgs.batch_size + target_lane
      is_valid_task = local_k_idx < num_k
      k_idx = local_k_idx
      loop_k_idx = local_k_idx * cp_group_size + hop
      is_last_k = jnp.where(
          (hop == cp_group_size - 1) & (local_k_idx == num_k - 1), 1, 0
      )
    else:
      is_last_k = jnp.where(loop_k_idx == end_k_idx - 1, 1, 0)

    sched.s_idx[step, target_lane] = jnp.where(is_valid_task, s_idx, -1)
    sched.q_idx[step, target_lane] = jnp.where(is_valid_task, q_idx, 0)
    sched.k_idx[step, target_lane] = jnp.where(is_valid_task, loop_k_idx, 0)
    sched.is_last_k[step, target_lane] = is_last_k

    sched.dma_q[step, target_lane, 0] = jnp.where(is_valid_task, q_src, 0)
    sched.dma_q[step, target_lane, 1] = jnp.where(is_valid_task, q_sz_task, 0)

    kv_len_start = k_idx * cfgs.bkv_sz
    kv_p_start = k_idx * cfgs.bkv_p
    k_len = kv_cache_len + kv_new_len
    kv_left = k_len - kv_len_start
    kv_left_frm_cache = jnp.maximum(kv_cache_len - kv_len_start, 0)
    kv_left = jnp.where(is_valid_task, kv_left, 0)
    kv_left_frm_cache = jnp.where(is_valid_task, kv_left_frm_cache, 0)
    p_offset = s_idx * cfgs.serve.pages_per_seq + kv_p_start

    cp_group_size = cfgs.serve.cp_group_size
    assert cp_group_size is not None
    new_tokens_only = (
        cfgs.serve.attention_scope == configs.AttentionScope.NEW_TOKENS_ONLY
    )

    for i in range(cfgs.bkv_p_cache):
      dst_vmem = i << cfgs.serve.page_size_log2
      dma_sz = kv_left_frm_cache - dst_vmem
      dma_sz = jnp.clip(dma_sz, 0, cfgs.serve.page_size)

      src_hbm = jnp.minimum(p_offset + i, cfgs.serve.num_page_indices - 1)
      sched.dma_kv_cache[step, target_lane, i, 1] = dst_vmem
      if cfgs.serve.kv_layout == configs.KVLayout.SEQ_ALONG_LANE:
        if ring:
          fetch_sz = dma_sz
          if new_tokens_only:
            fetch_sz = 0
          else:
            fetch_sz = jnp.where(is_round_0, fetch_sz, 0)
          sched.dma_kv_cache[step, target_lane, i, 0] = src_hbm
          sched.dma_kv_cache[step, target_lane, i, 2] = utils.align_to(
              fetch_sz, cfgs.kv_token_align
          )
        elif new_tokens_only:
          p_idx = kv_p_start + i
          local_page_i = jnp.minimum(
              p_idx // cp_group_size,
              cfgs.serve.pages_per_seq - 1,
          )
          src_hbm_cp = s_idx * cfgs.serve.pages_per_seq + local_page_i
          dma_sz_rank = jnp.where(p_idx % cp_group_size == self.rank, dma_sz, 0)
          sched.dma_kv_cache[step, target_lane, i, 0] = src_hbm_cp
          sched.dma_kv_cache[step, target_lane, i, 2] = utils.align_to(
              dma_sz_rank, cfgs.kv_token_align
          )
        else:  # CACHE_ONLY
          sched.dma_kv_cache[step, target_lane, i, 0] = src_hbm
          sched.dma_kv_cache[step, target_lane, i, 2] = utils.align_to(
              dma_sz, cfgs.kv_token_align
          )
      else:
        sched.dma_kv_cache[step, target_lane, i, 0] = src_hbm
        fetch_sz = dma_sz
        if new_tokens_only:
          fetch_sz = 0
        elif ring:
          fetch_sz = jnp.where(is_round_0, fetch_sz, 0)
        sched.dma_kv_cache[step, target_lane, i, 2] = fetch_sz

    # CACHE_ONLY still falls through.
    kv_left_frm_new = kv_left - kv_left_frm_cache
    bkv_sz_cache = jnp.minimum(kv_left_frm_cache, cfgs.bkv_sz)
    new_sz = jnp.minimum(cfgs.bkv_sz - bkv_sz_cache, kv_left_frm_new)

    if cfgs.serve.write_last_q_block:
      # the last q block that attends the KV block writes
      q_wb = num_q - 1
    else:
      # the first q block that attends the KV block writes
      q_wb = jnp.maximum(0, kv_len_start - kv_cache_len) // cfgs.bq_sz
    do_writeback = jnp.where(
        (new_sz > 0) & (q_idx == q_wb) & update_kv_cache, 1, 0
    )
    sched.do_writeback[step, target_lane] = do_writeback
    # Index of this block's new KV within the sequence's current KV, in
    # global token order.
    new_kv_pos = kv_new_len - kv_left_frm_new
    src_hbm = kv_new_start + new_kv_pos

    def fill_dma_kv_new(i, dst_vmem, dma_sz):
      dma_entry = sched.dma_kv_new[step, target_lane, i]
      if cfgs.serve.kv_layout == configs.KVLayout.SEQ_ALONG_LANE:
        if cfgs.serve.paged_new_kv:
          cache_pages = pl.cdiv(bkv_sz_cache, cfgs.serve.page_size)
          hbm_token_idx_base = new_kv_pos
          new_tok_offset = hbm_token_idx_base % cfgs.serve.page_size
          num_pages_to_fetch = jnp.where(
              new_sz > 0,
              (new_tok_offset + new_sz - 1) // cfgs.serve.page_size + 1,
              0,
          )
          fetch_val = jnp.where(i < num_pages_to_fetch, cfgs.serve.page_size, 0)
          new_page_start = (
              hbm_token_idx_base - new_tok_offset
          ) + i * cfgs.serve.page_size
          fetch_vmem = (cache_pages + i) * cfgs.serve.page_size

          tok_idx = kv_len_start + dst_vmem
          p_idx = jnp.minimum(
              tok_idx >> cfgs.serve.page_size_log2,
              cfgs.serve.pages_per_seq * cp_group_size - 1,
          )
          p_off = tok_idx & cfgs.serve.page_size_mask
          dst, lead, wb_val = utils.align_span(
              p_off, dma_sz, cfgs.kv_token_align
          )
          dma_entry.fetch_hbm[...] = new_page_start
          dma_entry.fetch_vmem[...] = fetch_vmem
          dma_entry.wb_hbm[...] = (
              (s_idx * cfgs.serve.pages_per_seq + p_idx // cp_group_size)
              << cfgs.serve.page_size_log2
          ) | dst
          dma_entry.wb_vmem[...] = dst_vmem - lead
          dma_entry.set_flags(
              fetch_val,
              jnp.where(
                  (dma_sz > 0) & (p_idx % cp_group_size == self.rank), wb_val, 0
              ),
          )
        else:
          align = cfgs.kv_token_align
          # Fetch: the new tokens are contiguous, so one DMA moves all of them
          # even though this loop runs once per page. Entry 0 carries that DMA
          # and the other entries stay zero.
          if i == 0:
            src, _, fetch_val = utils.align_span(src_hbm, new_sz, align)
            dma_entry.fetch_hbm[...] = src
            dma_entry.fetch_vmem[...] = utils.align_to(bkv_sz_cache, align)
            fetch_val = jnp.where(new_sz > 0, fetch_val, 0)
          else:
            fetch_val = 0

          # Writeback: one DMA per page, since page_indices scatters them, and
          # only for the pages this rank owns.
          tok_idx = kv_len_start + dst_vmem
          p_idx = jnp.minimum(
              tok_idx >> cfgs.serve.page_size_log2,
              cfgs.serve.pages_per_seq * cp_group_size - 1,
          )
          p_off = tok_idx & cfgs.serve.page_size_mask
          dst, lead, wb_val = utils.align_span(p_off, dma_sz, align)
          dma_entry.wb_hbm[...] = (
              (s_idx * cfgs.serve.pages_per_seq + p_idx // cp_group_size)
              << cfgs.serve.page_size_log2
          ) | dst
          dma_entry.wb_vmem[...] = dst_vmem - lead
          dma_entry.set_flags(
              fetch_val,
              jnp.where(
                  (dma_sz > 0) & (p_idx % cp_group_size == self.rank), wb_val, 0
              ),
          )
      else:
        tok_idx = kv_len_start + dst_vmem
        p_idx = jnp.minimum(
            tok_idx >> cfgs.serve.page_size_log2,
            cfgs.serve.pages_per_seq * cp_group_size - 1,
        )
        p_off = tok_idx & cfgs.serve.page_size_mask
        local_slot = p_idx // cp_group_size
        dst_hbm = (
            (s_idx * cfgs.serve.pages_per_seq + local_slot)
            << cfgs.serve.page_size_log2
        ) | p_off
        wb_val = jnp.where(
            p_idx % cp_group_size == self.rank, dma_sz, jnp.int32(0)
        )

        fetch_hbm = src_hbm
        if cfgs.serve.paged_new_kv:
          # With a page table the new KV is not contiguous, so
          # store the offset within the sequence's own new KV and let
          # copy_in resolve it.
          offset = dst_vmem - bkv_sz_cache
          fetch_hbm = new_kv_pos + offset
        dma_entry.fetch_hbm[...] = fetch_hbm
        dma_entry.fetch_vmem[...] = dst_vmem
        dma_entry.wb_hbm[...] = dst_hbm
        dma_entry.wb_vmem[...] = dst_vmem
        dma_entry.set_flags(dma_sz, wb_val)

    if cfgs.bkv_p_new < cfgs.bkv_p:
      # General decode path for any bkv_p_new
      curr_vmem = bkv_sz_cache
      curr_rem = new_sz

      for i in range(cfgs.bkv_p_new):
        slot_start = (curr_vmem // cfgs.serve.page_size) * cfgs.serve.page_size
        slot_end = slot_start + cfgs.serve.page_size

        dma_sz = jnp.minimum(curr_rem, slot_end - curr_vmem)
        dma_sz = jnp.where(curr_rem > 0, dma_sz, 0)

        fill_dma_kv_new(i, curr_vmem, dma_sz)

        curr_vmem += dma_sz
        curr_rem = jnp.maximum(0, curr_rem - dma_sz)
    else:
      iters = max(cfgs.bkv_p, cfgs.bkv_p_new)
      for i in range(iters):
        slot_start = i * cfgs.serve.page_size
        slot_end = slot_start + cfgs.serve.page_size

        dst_vmem = jnp.maximum(slot_start, bkv_sz_cache)
        end_in_slot = jnp.minimum(slot_end, bkv_sz_cache + new_sz)
        dma_sz = jnp.maximum(0, end_in_slot - dst_vmem)

        fill_dma_kv_new(i, dst_vmem, dma_sz)

    return self.advance_carry(carry)
