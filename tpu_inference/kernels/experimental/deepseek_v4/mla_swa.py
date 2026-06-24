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
"""TPU-Friendly MLA Ragged Paged Attention kernel."""

import functools
from enum import Enum

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

DEFAULT_MASK_VALUE = -0.7 * float(jnp.finfo(jnp.dtype("float32")).max)

DEFAULT_VMEM_LIMIT_BYTES = 100 * 1024 * 1024


def cdiv(a, b):
    assert b != 0
    return (a + b - 1) // b


def align_to(x, a):
    return cdiv(x, a) * a


def get_dtype_bitwidth(dtype):
    return jax.dtypes.itemsize_bits(dtype)


def get_dtype_packing(dtype):
    bits = get_dtype_bitwidth(dtype)
    return 32 // bits


class MlaCase(Enum):
    """Represents the different cases for MLA.

  - DECODE: Sequences are in decode-only mode (q_len = 1).
  - PREFILL: Sequences are in prefill-only mode (q_len > 1, static).
  - MIXED: Sequences can be a mix of prefill and decode (q_len > 1, dynamic).
  """

    DECODE = 0
    PREFILL = 1
    MIXED = 2

    @property
    def symbol(self):
        return {
            MlaCase.DECODE: "d",
            MlaCase.PREFILL: "p",
            MlaCase.MIXED: "m",
        }[self]


def _mla_sliding_window_ragged_paged_attention_kernel(
    # Prefetch
    kv_lens_ref,  # [max_num_seqs]
    page_indices_ref,  # [max_num_seqs * pages_per_seq]
    cu_q_lens_ref,  # [max_num_seqs + 1]
    start_end_seq_idx_ref,  # [2] (start_seq_idx, end_seq_idx)
    sem_ids_ref,  # [3] (bq_sem_idx, bkv_sem_idx, bo_sem_idx)
    bo_ids_ref,  # [4] (bo_sem_0_seq_idx, bo_sem_1_seq_idx, bo_sem_0_bo_idx, bo_sem_1_bo_idx)
    bkv_update_ids_ref,  # [6] (bkv_sem_0_seq_idx, bkv_sem_1_seq_idx, bkv_sem_0_offset, bkv_sem_1_offset, bkv_sem_0_sz, bkv_sem_1_sz)
    # Input
    attention_sinks_ref,  # float32[num_q_heads]
    q_hbm_ref,  # [max_num_tokens, num_q_heads, head_dim]
    new_kv_hbm_ref,  # [max_num_tokens_per_kv_packing, kv_packing, lkv_dim]
    cache_kv_hbm_ref,  # [total_num_pages, physical_page_size_per_kv_packing, kv_packing, lkv_dim]
    in_output_hbm_ref,  # [max_num_tokens, num_q_heads, head_dim]
    in_l_hbm_ref,  # [max_num_tokens, num_l_heads]
    in_m_hbm_ref,  # [max_num_tokens, num_l_heads]
    # Output
    o_hbm_ref,  # [max_num_tokens, num_q_heads, head_dim]
    updated_cache_kv_hbm_ref,  # [total_num_pages, physical_page_size_per_kv_packing, kv_packing, lkv_dim]
    l_hbm_ref,  # [max_num_tokens, num_l_heads]
    m_hbm_ref,  # [max_num_tokens, num_l_heads]
    # Scratch
    bkv_x2_ref,  # [2, bkv_buf_sz_per_kv_packing, kv_packing, lkv_dim]
    bq_x2_ref,  # [2, bq_sz, num_q_heads, head_dim]
    bo_x2_ref,  # [2, bq_sz, num_q_heads, head_dim]
    bl_x2_ref,  # [2, bq_sz, num_l_heads]
    bm_x2_ref,  # [2, bq_sz, num_l_heads]
    sems,  # [6, 2]
    l_ref,  # [bq_sz * num_q_heads, 128],
    m_ref,  # [bq_sz * num_q_heads, 128],
    acc_ref,  # [bq_sz * num_q_heads, head_dim],
    *,
    static_q_len: int,
    sm_scale: float,
    sliding_window: int,
    logical_page_size: int,
    unnormalized_output: bool,
    mask_value: float = DEFAULT_MASK_VALUE,
    bkv_p,
    bq_sz,
):
    assert q_hbm_ref.shape == o_hbm_ref.shape
    assert sliding_window > 0

    _, num_q_heads, head_dim = q_hbm_ref.shape
    lkv_dim = cache_kv_hbm_ref.shape[-1]
    q_packing = get_dtype_packing(q_hbm_ref.dtype)
    assert num_q_heads % q_packing == 0
    num_q_heads_per_q_packing = num_q_heads // q_packing

    total_num_pages, physical_page_size_per_kv_packing, kv_packing, _ = (
        cache_kv_hbm_ref.shape)
    assert logical_page_size % kv_packing == 0
    page_size_per_kv_packing = logical_page_size // kv_packing
    max_num_seqs = kv_lens_ref.shape[0]
    num_page_indices = page_indices_ref.shape[0]

    assert num_page_indices % max_num_seqs == 0
    pages_per_seq = num_page_indices // max_num_seqs
    q_dtype = q_hbm_ref.dtype
    # Validate against the KV dtype.
    kv_dtype = cache_kv_hbm_ref.dtype
    assert o_hbm_ref.dtype == q_dtype
    assert get_dtype_packing(kv_dtype) == kv_packing
    assert head_dim % 128 == 0
    bkv_sz_per_kv_packing = bkv_p * page_size_per_kv_packing
    bkv_sz = bkv_sz_per_kv_packing * kv_packing
    page_size = logical_page_size

    start_seq_idx = start_end_seq_idx_ref[0]
    end_seq_idx = start_end_seq_idx_ref[1]
    seq_idx = pl.program_id(0) + start_seq_idx
    q_start = cu_q_lens_ref[seq_idx]
    q_end = cu_q_lens_ref[seq_idx + 1]
    q_len = q_end - q_start
    kv_len = kv_lens_ref[seq_idx]

    def flash_attention(
        q,  # [bq_sz * num_q_heads, head_dim]
        kv,  # [bkv_sz, head_dim] <- Correspond to data from bkv_x2_ref
        *,
        bq_idx,
        bkv_idx,
        is_first_bkv,
    ):
        assert len(q.shape) == 2
        assert len(kv.shape) == 2
        assert q.shape[0] % num_q_heads == 0
        assert q.shape[1] == head_dim
        assert kv.shape == (bkv_sz, head_dim)
        head_l_ref = l_ref.at[:q.shape[0]]
        head_m_ref = m_ref.at[:q.shape[0]]
        head_acc_ref = acc_ref.at[:q.shape[0]]

        def load_with_init(ref, init_val):
            return jnp.where(is_first_bkv, jnp.full_like(ref, init_val),
                             ref[...])

        # Follow FlashAttention-2 forward pass.
        s = jnp.einsum("nd,md->nm", q, kv, preferred_element_type=jnp.float32)
        s *= sm_scale

        q_span = (kv_len - q_len + bq_idx * bq_sz +
                  lax.broadcasted_iota(jnp.int32, s.shape, 0) // num_q_heads)
        k_span = bkv_idx * bkv_sz + lax.broadcasted_iota(jnp.int32, s.shape, 1)
        mask = q_span < k_span
        mask = jnp.logical_or(mask, q_span - sliding_window >= k_span)

        s = jnp.where(mask, mask_value, s)
        s_rowmax = jnp.max(s, axis=1, keepdims=True)
        m_prev = load_with_init(head_m_ref, -jnp.inf)
        m_curr = jnp.maximum(m_prev, s_rowmax)
        head_m_ref[...] = m_curr
        p = jnp.exp(s - broadcast_minor(m_curr, s.shape))

        pv = jnp.einsum("nm,md->nd", p, kv, preferred_element_type=jnp.float32)

        p_rowsum = jnp.sum(p, axis=1, keepdims=True)
        exp_m_diff = jnp.exp(m_prev - m_curr)
        l_prev = load_with_init(head_l_ref, 0.0)
        l_curr = exp_m_diff * l_prev + p_rowsum
        head_l_ref[...] = l_curr
        o_prev = load_with_init(head_acc_ref, 0.0)
        o_curr = broadcast_minor(exp_m_diff, o_prev.shape) * o_prev + pv
        head_acc_ref[...] = o_curr

    def _async_copy(src, dst, sem, wait):
        cp = pltpu.make_async_copy(src, dst, sem)
        if wait:
            cp.wait()
        else:
            cp.start()

    def _get_kv_len(seq_idx):
        return jnp.where(seq_idx < end_seq_idx, kv_lens_ref[seq_idx], 0)

    def _get_q_len(seq_idx):
        return jnp.where(
            seq_idx < end_seq_idx,
            cu_q_lens_ref[seq_idx + 1] - cu_q_lens_ref[seq_idx],
            0,
        )

    def _fetch_bkv(seq_idx, bkv_idx, bkv_sem_idx, *, wait=False):
        sem = sems.at[0, bkv_sem_idx]
        # bkv_x2_ref shape: [2, bkv_sz_per_kv_packing + 2, kv_packing, lkv_dim]
        bkv_vmem_ref = bkv_x2_ref.at[bkv_sem_idx]

        # [total_num_pages, physical_page_size_per_kv_packing, kv_packing, lkv_dim]
        # [total_num_pages * physical_page_size_per_kv_packing, kv_packing, lkv_dim]
        reshaped_cache_hbm_ref = cache_kv_hbm_ref.reshape(
            total_num_pages * physical_page_size_per_kv_packing,
            *cache_kv_hbm_ref.shape[2:],
        )

        kv_len = kv_lens_ref[seq_idx]
        kv_len_start = bkv_idx * bkv_sz
        kv_p_start = bkv_idx * bkv_p

        q_start = cu_q_lens_ref[seq_idx]
        q_end = cu_q_lens_ref[seq_idx + 1]
        q_len = q_end - q_start

        kv_left = kv_len - kv_len_start
        kv_left_frm_cache = jnp.maximum(kv_left - q_len, 0)
        kv_left_frm_cache_per_kv_packing = cdiv(kv_left_frm_cache, kv_packing)
        kv_left_frm_new = kv_left - kv_left_frm_cache

        bkv_sz_frm_cache = jnp.minimum(kv_left_frm_cache, bkv_sz)
        bkv_sz_frm_new = jnp.minimum(bkv_sz - bkv_sz_frm_cache,
                                     kv_left_frm_new)
        bkv_sz_frm_cache_per_kv_packing = cdiv(bkv_sz_frm_cache, kv_packing)
        page_indices_offset = seq_idx * pages_per_seq + kv_p_start

        new_kv_len_start = q_end - kv_left_frm_new
        new_kv_len_start_per_kv_packing = new_kv_len_start // kv_packing
        bkv_sz_frm_new_kv_packing_to_fetch = jnp.where(
            bkv_sz_frm_new > 0,
            cdiv(new_kv_len_start + bkv_sz_frm_new, kv_packing) -
            new_kv_len_start_per_kv_packing,
            0,
        )
        dma_bkv_sz = (bkv_sz_frm_cache_per_kv_packing +
                      bkv_sz_frm_new_kv_packing_to_fetch)

        if not wait:
            # Make sure the current bkv buffer is safe to overwrite.
            wait_update_kv_cache(bkv_sem_idx)

            # Fetch effective kv from kv cache. To pipeline multiple DMA calls, we
            # utilize static for loop instead of dynamic for loop.
            # Loop through all pages in a block
            for i in range(bkv_p):
                # Ensure only effective kvs are copied and we don't go negative.
                sz_per_kv_packing = jnp.clip(
                    kv_left_frm_cache_per_kv_packing -
                    i * page_size_per_kv_packing,
                    0,
                    page_size_per_kv_packing,
                )
                # If the page index is out of bound, we set page_idx to the last page.
                # And there will be no copy since sz will be 0.
                page_idx = jnp.minimum(page_indices_offset + i,
                                       num_page_indices - 1)
                _async_copy(
                    reshaped_cache_hbm_ref.at[
                        pl.ds(
                            page_indices_ref[page_idx] *
                            physical_page_size_per_kv_packing,
                            sz_per_kv_packing,
                        ),
                    ],
                    # [bkv_sz_per_kv_packing + 2, kv_packing, lkv_dim].
                    bkv_vmem_ref.at[pl.ds(i * page_size_per_kv_packing,
                                          sz_per_kv_packing)],
                    sem,
                    wait,
                )

            # Fetch new KVs by appending to the existing vmem buffers.
            # Fetch either up to the end of the buffer or kv_left_frm_new, whichever
            # is smaller. Since DMAs are word-aligned based on kv_packing, and the
            # boundary between the old cache and the new KV tokens might not be
            # word-aligned, we append the new KV words right after the last word
            # containing old cache data. This can create "holes" (misalignments
            # within the words), which we will shift and pack correctly later.
            _async_copy(
                new_kv_hbm_ref.at[pl.ds(
                    new_kv_len_start_per_kv_packing,
                    bkv_sz_frm_new_kv_packing_to_fetch,
                )],
                bkv_vmem_ref.at[pl.ds(
                    bkv_sz_frm_cache_per_kv_packing,
                    bkv_sz_frm_new_kv_packing_to_fetch,
                )],
                sem,
                wait,
            )

        else:
            # When we wait, we can use a dummy copy to wait for DMAs to complete where
            # src == dst. However, the dma size must be correct.
            dst_kv = bkv_vmem_ref.at[pl.ds(0, dma_bkv_sz)]
            _async_copy(
                src=dst_kv,
                dst=dst_kv,
                sem=sem,
                wait=True,
            )

        # This returns the (offset, size) in units of tokens:
        #   offset: starting token index where the new KV should be stored
        #   size: number of tokens of the new KV, which is 1 in decode.
        return kv_len_start + bkv_sz_frm_cache, bkv_sz_frm_new

    def _pack_new_kv(bkv_sem_idx, offset, update_sz):
        """Packs newly computed KVs into the correct sub-word alignment in VMEM.

    When new KV tokens are DMA'd from HBM into VMEM, they are copied at the
    granularity of packed words (e.g., 4 tokens per word for fp8) by head
    dimension (mapped to lanes). The starting token `offset` in the KV cache,
    however, might not fall exactly on a word boundary. This means the elements
    within the packed words might be misaligned relative to their final
    destination in the cache.

    This function corrects this alignment by:
    1. Computing the bit-shift amount needed based on the difference between the
       destination token offset (`kv_packing_offset`) and the source token
       offset (`new_kv_packing_offset`).
    2. Looping over the affected words and using bitwise shifts and logical ORs
       to realign the elements across word boundaries.
    3. Merging the correctly aligned new KV elements into the VMEM buffer using
       a mask, leaving existing (older) KV elements intact.

    Args:
      bkv_sem_idx: The semaphore index for the current KV block.
      offset: The starting token offset in the KV cache where the new KVs begin.
      update_sz: The number of new tokens to be packed.
    """
        # shape: [bkv_sz_per_kv_packing + 2, kv_packing, lkv_dim]
        bkv_vmem_ref = bkv_x2_ref.at[bkv_sem_idx]

        update_kv_packing_iters = cdiv((offset % kv_packing) + update_sz,
                                       kv_packing)
        kv_packing_offset = offset % kv_packing
        new_kv_len_start = q_end - kv_len + offset
        new_kv_packing_offset = new_kv_len_start % kv_packing

        token_offset_in_bkv = offset % bkv_sz
        kv_packing_idx = token_offset_in_bkv // kv_packing

        # Compute the shift amount for each word in bits
        shift_amount = kv_packing_offset - new_kv_packing_offset
        bits_per_element = get_dtype_bitwidth(bkv_vmem_ref.dtype)
        shift_bits = bits_per_element * (shift_amount % kv_packing)
        shift_bits = shift_bits.astype(jnp.uint32)

        # Calculate the starting index in the KV buffer corresponding to the new KV
        # to fetch the data from. This index accounts for the potential offset
        # caused by the shift_amount.
        # (-shift_amount) // kv_packing will be:
        #   0 if new_kv_packing_offset <= kv_packing_offset
        #  -1 if new_kv_packing_offset > kv_packing_offset.
        kv_packing_idx_new = (cdiv(token_offset_in_bkv, kv_packing) +
                              (-shift_amount) // kv_packing)
        curr_kv_reg = bkv_vmem_ref[kv_packing_idx_new, :, :]
        next_kv_reg = bkv_vmem_ref[kv_packing_idx_new + 1, :, :]

        def merge_loop_body(i, vals):
            (
                kv_packing_idx,
                kv_packing_idx_new,
                curr_kv_reg,
                next_kv_reg,
            ) = vals
            curr_kv_reg_u32 = pltpu.bitcast(curr_kv_reg, jnp.uint32)
            next_kv_reg_u32 = pltpu.bitcast(next_kv_reg, jnp.uint32)

            shifted_kv_u32 = lax.bitwise_or(
                lax.shift_right_logical(curr_kv_reg_u32, 32 - shift_bits),
                lax.shift_left(next_kv_reg_u32, shift_bits),
            )

            # If shift_bits is 0, we should use the current word. Otherwise,
            # shifting by 32 bits would result in shifted_*_u32 becoming
            # next_*_reg_u32, which is incorrect.
            rotated_kv_u32 = lax.select(shift_bits == 0, curr_kv_reg_u32,
                                        shifted_kv_u32)

            next_kv_reg_shifted = pltpu.bitcast(rotated_kv_u32,
                                                next_kv_reg.dtype)

            offset_in_word = i * kv_packing + lax.broadcasted_iota(
                dtype=jnp.int32, shape=[kv_packing, lkv_dim], dimension=0)
            kv_mask = jnp.logical_and(
                offset_in_word >= kv_packing_offset,
                offset_in_word < kv_packing_offset + update_sz,
            )
            updated_kv_reg = lax.select(
                kv_mask,
                next_kv_reg_shifted,
                bkv_vmem_ref[kv_packing_idx, :, :],
            )

            # Store back the merged word
            bkv_vmem_ref[kv_packing_idx, :, :] = updated_kv_reg

            # Move to the next word.
            kv_packing_idx += 1
            kv_packing_idx_new += 1
            curr_kv_reg = next_kv_reg
            next_kv_reg = bkv_vmem_ref[kv_packing_idx_new + 1, :, :]
            return (
                kv_packing_idx,
                kv_packing_idx_new,
                curr_kv_reg,
                next_kv_reg,
            )

        lax.fori_loop(
            0,
            update_kv_packing_iters,
            merge_loop_body,
            (
                kv_packing_idx,
                kv_packing_idx_new,
                curr_kv_reg,
                next_kv_reg,
            ),
        )

    def _update_kv_cache(
        seq_idx,
        bkv_sem_idx,
        offset,  # In units of tokens.
        update_sz,  # In units of tokens.
        *,
        wait=False,
    ):
        sem = sems.at[3, bkv_sem_idx]
        # shape: [bkv_sz_per_kv_packing + 2, kv_packing, lkv_dim]
        bkv_vmem_ref = bkv_x2_ref.at[bkv_sem_idx]

        update_kv_packing_iters = cdiv((offset % kv_packing) + update_sz,
                                       kv_packing)

        cache_kv_hbm_shape = updated_cache_kv_hbm_ref.shape
        reshaped_cache_kv_hbm_ref = updated_cache_kv_hbm_ref.reshape(
            cache_kv_hbm_shape[0] * cache_kv_hbm_shape[1],
            *cache_kv_hbm_shape[2:],
        )

        if not wait:
            # Issue DMA copy for the updated parts, page by page.
            kv_p_start = offset // page_size
            kv_p_end = cdiv(offset + update_sz, page_size)
            start_word_in_page = (offset % page_size) // kv_packing
            start_word_in_vmem = (offset % bkv_sz) // kv_packing
            words_to_transfer = update_kv_packing_iters
            page_indices_offset = seq_idx * pages_per_seq + kv_p_start

            def loop_body(i, states):
                curr_word_in_page, words_to_transfer, curr_word_in_vmem = states
                sz_words = jnp.minimum(
                    page_size_per_kv_packing - curr_word_in_page,
                    words_to_transfer)
                page_idx = page_indices_ref[page_indices_offset + i]

                _async_copy(
                    # bkv_vmem_ref shape:
                    # [bkv_sz_per_kv_packing+2, kv_packing, lkv_dim]
                    bkv_vmem_ref.at[pl.ds(curr_word_in_vmem, sz_words)],
                    reshaped_cache_kv_hbm_ref.at[
                        pl.ds(
                            page_idx * physical_page_size_per_kv_packing +
                            curr_word_in_page,
                            sz_words,
                        ),
                    ],
                    sem,
                    wait=False,
                )
                return 0, words_to_transfer - sz_words, curr_word_in_vmem + sz_words

            lax.fori_loop(
                0,
                kv_p_end - kv_p_start,
                loop_body,
                (
                    start_word_in_page,
                    words_to_transfer,
                    start_word_in_vmem,
                ),  # initial states
                unroll=False,
            )
        else:  # Wait
            dma_sz_words = update_kv_packing_iters
            # bkv_vmem_ref shape: [bkv_sz_per_kv_packing + 2, kv_packing, lkv_dim]
            dst_kv = bkv_vmem_ref.at[pl.ds(0, dma_sz_words)]
            _async_copy(
                src=dst_kv,
                dst=dst_kv,
                sem=sem,
                wait=True,
            )

    def _fetch_bq(seq_idx, bq_idx, bq_sem_idx, *, wait=False):
        sem = sems.at[1, bq_sem_idx]
        bq_vmem_ref = bq_x2_ref.at[bq_sem_idx]

        q_len_start = cu_q_lens_ref[seq_idx] + bq_idx * bq_sz
        q_end = cu_q_lens_ref[seq_idx + 1]
        sz = jnp.minimum(bq_sz, q_end - q_len_start)

        _async_copy(
            q_hbm_ref.at[pl.ds(q_len_start, sz)],
            bq_vmem_ref.at[pl.ds(0, sz)],
            sem,
            wait,
        )

    def _send_bo(seq_idx, bo_idx, bo_sem_idx, *, wait=False):
        sem = sems.at[2, bo_sem_idx]
        vmem_ref = bo_x2_ref.at[bo_sem_idx]
        q_len_start = cu_q_lens_ref[seq_idx] + bo_idx * bq_sz
        q_end = cu_q_lens_ref[seq_idx + 1]
        sz = jnp.minimum(bq_sz, q_end - q_len_start)

        _async_copy(
            vmem_ref.at[pl.ds(0, sz)],
            o_hbm_ref.at[pl.ds(q_len_start, sz)],
            sem,
            wait,
        )

    def _send_l(seq_idx, bo_idx, bo_sem_idx, *, wait=False):
        sem = sems.at[4, bo_sem_idx]
        vmem_ref = bl_x2_ref.at[bo_sem_idx]
        q_len_start = cu_q_lens_ref[seq_idx] + bo_idx * bq_sz
        q_end = cu_q_lens_ref[seq_idx + 1]
        sz = jnp.minimum(bq_sz, q_end - q_len_start)

        _async_copy(
            vmem_ref.at[pl.ds(0, sz)],
            l_hbm_ref.at[pl.ds(q_len_start, sz)],
            sem,
            wait,
        )

    def _send_m(seq_idx, bo_idx, bo_sem_idx, *, wait=False):
        sem = sems.at[5, bo_sem_idx]
        vmem_ref = bm_x2_ref.at[bo_sem_idx]
        q_len_start = cu_q_lens_ref[seq_idx] + bo_idx * bq_sz
        q_end = cu_q_lens_ref[seq_idx + 1]
        sz = jnp.minimum(bq_sz, q_end - q_len_start)

        _async_copy(
            vmem_ref.at[pl.ds(0, sz)],
            m_hbm_ref.at[pl.ds(q_len_start, sz)],
            sem,
            wait,
        )

    def start_fetch_bkv(seq_idx, bkv_idx, bkv_sem_idx):
        return _fetch_bkv(seq_idx, bkv_idx, bkv_sem_idx)

    def wait_fetch_bkv(seq_idx, bkv_idx, bkv_sem_idx):
        return _fetch_bkv(seq_idx, bkv_idx, bkv_sem_idx, wait=True)

    def start_fetch_bq(seq_idx, bq_idx, bq_sem_idx):
        return _fetch_bq(seq_idx, bq_idx, bq_sem_idx)

    def wait_fetch_bq(seq_idx, bq_idx, bq_sem_idx):
        return _fetch_bq(seq_idx, bq_idx, bq_sem_idx, wait=True)

    def start_send_bo(seq_idx, bo_idx, bo_sem_idx):
        bo_ids_ref[bo_sem_idx] = seq_idx
        bo_ids_ref[bo_sem_idx + 2] = bo_idx
        _send_bo(seq_idx, bo_idx, bo_sem_idx)
        _send_l(seq_idx, bo_idx, bo_sem_idx)
        _send_m(seq_idx, bo_idx, bo_sem_idx)

    def wait_send_bo(bo_sem_idx):
        old_seq_idx = bo_ids_ref[bo_sem_idx]
        old_bo_idx = bo_ids_ref[bo_sem_idx + 2]

        @pl.when(jnp.logical_and(0 <= old_seq_idx, old_seq_idx <= seq_idx))
        def _():
            _send_bo(old_seq_idx, old_bo_idx, bo_sem_idx, wait=True)
            _send_l(old_seq_idx, old_bo_idx, bo_sem_idx, wait=True)
            _send_m(old_seq_idx, old_bo_idx, bo_sem_idx, wait=True)

    def start_update_kv_cache(seq_idx, bkv_sem_idx, offset, update_sz):
        bkv_update_ids_ref[bkv_sem_idx] = seq_idx
        bkv_update_ids_ref[bkv_sem_idx + 2] = offset
        bkv_update_ids_ref[bkv_sem_idx + 4] = update_sz
        _update_kv_cache(seq_idx, bkv_sem_idx, offset, update_sz)

    def wait_update_kv_cache(bkv_sem_idx):
        update_sz = bkv_update_ids_ref[bkv_sem_idx + 4]

        @pl.when(update_sz > 0)
        def _():
            seq_idx = bkv_update_ids_ref[bkv_sem_idx]
            offset = bkv_update_ids_ref[bkv_sem_idx + 2]
            bkv_update_ids_ref[bkv_sem_idx + 4] = 0
            _update_kv_cache(seq_idx,
                             bkv_sem_idx,
                             offset,
                             update_sz,
                             wait=True)

    def load_bq(bq_sem_idx):
        q_ref = (bq_x2_ref.bitcast(jnp.uint32).at[bq_sem_idx].reshape(
            bq_sz * num_q_heads_per_q_packing, head_dim))
        q = pltpu.bitcast(
            q_ref[:bq_sz * num_q_heads_per_q_packing],
            q_dtype,
        ).reshape(bq_sz * num_q_heads, head_dim)
        return q

    def load_bkv(bkv_sem_idx, bkv_idx):
        bkv_ref = (bkv_x2_ref.bitcast(
            jnp.uint32).at[bkv_sem_idx, :bkv_sz_per_kv_packing].reshape(
                bkv_sz_per_kv_packing, lkv_dim))
        bkv = pltpu.bitcast(bkv_ref[...], kv_dtype).reshape(bkv_sz, lkv_dim)

        # In vLLM, multiple caches may overlay on the same KV Tensor. For example,
        # compressor state cache write data in bfloat16 / float32 format, certain
        # byte pattern are interpreted as NaN in FP8, e.g. float8_e8m0fnu byte 0xFF
        # decodes to NaN.
        # We need to mask out the data by the actual kv_len to avoid NaN propagting
        # to the downstream computation.
        k_span = bkv_idx * bkv_sz + lax.broadcasted_iota(
            jnp.int32, bkv.shape, 0)
        bkv = jnp.where(k_span < kv_len, bkv, 0)

        # Dequantize DSV4 FP8 format to BF16.
        # 448 fp8, 64 bf16, 7 fp8 scales, 7 e8m0 scale for 448 fp8 (block size 64)
        nope_fp8 = pltpu.bitcast(bkv[:, :448],
                                 jnp.float8_e4m3fn).astype(jnp.bfloat16)
        # libtpu 0.0.41 not yet support the f8E8M0FNU element type, so decode the
        # E8M0 scale bytes manually. E8M0 stores value = 2**(byte - 127).
        nope_scales = pltpu.bitcast(bkv[:, 576:583], jnp.uint8)
        nope_scales = jnp.exp2(nope_scales.astype(jnp.float32) - 127.0).astype(
            jnp.bfloat16)
        nope_fp8 = nope_fp8.reshape(bkv_sz, 7, 64)
        nope_scales = nope_scales.reshape(bkv_sz, 7, 1)
        dequant_nope = (nope_fp8 * nope_scales).reshape(bkv_sz, 448)
        rope = pltpu.bitcast(bkv[:, 448:576].T, jnp.bfloat16).T
        bkv = jnp.concatenate([dequant_nope, rope], axis=-1)
        return bkv

    def broadcast_minor(src, shape):
        if src.shape == shape:
            return src
        assert src.shape[:-1] == shape[:-1]
        assert src.shape[-1] % 128 == 0
        target_minor = align_to(shape[-1], src.shape[-1])
        # no-op concatenation.
        return jnp.concatenate(
            [src for _ in range(target_minor // src.shape[-1])],
            axis=-1)[..., :shape[-1]]

    def process():
        # Only when bq_idx == 0, we do kv cache update, need to go all the way to
        # the kv_len

        # Force at least one bkv block and one bq block per sequence: the
        # double-buffered DMA pipeline hands the bkv and bq semaphore across
        # sequence boundaries and assumes every sequence runs >=1 bkv and bq
        # iteration.
        num_bkv = jnp.maximum(1, cdiv(kv_len, bkv_sz))
        if static_q_len is None:
            num_bq = jnp.maximum(1, cdiv(q_len, bq_sz))
        else:
            num_bq = jnp.maximum(1, cdiv(static_q_len, bq_sz))

        def get_next_bq_ids(seq_idx, bq_idx, bq_sem_idx):
            next_bq_idx = bq_idx + 1
            is_last_bq = next_bq_idx == num_bq
            next_bq_idx = lax.select(is_last_bq, 0, next_bq_idx)
            next_seq_idx = lax.select(is_last_bq, seq_idx + 1, seq_idx)
            next_bq_sem_idx = lax.select(bq_sem_idx == 0, 1, 0)
            return next_seq_idx, next_bq_idx, next_bq_sem_idx

        def get_next_bkv_ids(seq_idx, bq_idx, bkv_idx, bkv_sem_idx):
            next_bkv_idx = bkv_idx + 1
            is_last_bkv = next_bkv_idx == num_bkv
            next_bq_idx = lax.select(is_last_bkv, bq_idx + 1, bq_idx)
            is_last_bq = next_bq_idx == num_bq
            next_bq_idx = lax.select(is_last_bq, 0, next_bq_idx)
            next_seq_idx = lax.select(is_last_bq, seq_idx + 1, seq_idx)
            next_bkv_idx = lax.select(
                is_last_bkv,
                jnp.maximum(
                    _get_kv_len(next_seq_idx) - _get_q_len(next_seq_idx) -
                    sliding_window,
                    0,
                ) // bkv_sz,
                next_bkv_idx,
            )
            next_bkv_sem_idx = lax.select(bkv_sem_idx == 0, 1, 0)
            return next_seq_idx, next_bq_idx, next_bkv_idx, next_bkv_sem_idx

        def compute_with_bq(bq_idx, _):
            bq_sem_idx = sem_ids_ref[0]
            next_seq_idx, next_bq_idx, next_bq_sem_idx = get_next_bq_ids(
                seq_idx, bq_idx, bq_sem_idx)

            # Prefetch next bq
            @pl.when(next_seq_idx < end_seq_idx)
            def prefetch_next_bq():
                sem_ids_ref[0] = next_bq_sem_idx
                start_fetch_bq(next_seq_idx, next_bq_idx, next_bq_sem_idx)

            def compute_with_bkv(bkv_idx, carry):
                is_first_bkv = carry[0] == 1

                # Get next bkv ids.
                bkv_sem_idx = sem_ids_ref[1]
                next_seq_idx, _, next_bkv_idx, next_bkv_sem_idx = get_next_bkv_ids(
                    seq_idx, bq_idx, bkv_idx, bkv_sem_idx)

                # Prefetch next bkv
                @pl.when(next_seq_idx < end_seq_idx)
                def prefetch_next_bkv():
                    sem_ids_ref[1] = next_bkv_sem_idx
                    start_fetch_bkv(next_seq_idx, next_bkv_idx,
                                    next_bkv_sem_idx)

                # Wait for cur bkv
                offset, update_sz = wait_fetch_bkv(seq_idx, bkv_idx,
                                                   bkv_sem_idx)

                # Pack and align new KVs in VMEM if the block has new KVs.
                # We may have to do this for each block of KV in VMEM.
                @pl.when(update_sz > 0)
                def pack_new_kv():
                    _pack_new_kv(bkv_sem_idx, offset, update_sz)

                # Start updating bkv to kv cache if applicable.
                # Only needed in first bq loop.
                @pl.when(jnp.logical_and(update_sz > 0, bq_idx == 0))
                def update_cur_bkv_to_cache():
                    start_update_kv_cache(seq_idx, bkv_sem_idx, offset,
                                          update_sz)

                # Load bkv into vreg. There is no need to mask out invalid k/v entries,
                # because the score of invalid Q.K^T pairs are masked (to be zero) in
                # flash attention, so that the invalid kv entries
                # (as long as they are not NaN or inf) won't affect to the output.
                bkv = load_bkv(
                    bkv_sem_idx,
                    bkv_idx,
                )

                bq = load_bq(bq_sem_idx)

                flash_attention(
                    bq,
                    bkv,
                    bq_idx=bq_idx,
                    bkv_idx=bkv_idx,
                    is_first_bkv=is_first_bkv,
                )

                # Return whether this is the first bkv.
                return (jnp.int32(0), )

            # Wait for cur bq if not ready yet
            wait_fetch_bq(seq_idx, bq_idx, bq_sem_idx)
            # Skip bkvs that are outside of the sliding window.
            bkv_start_idx = jnp.maximum(kv_len - q_len - sliding_window,
                                        0) // bkv_sz
            lax.fori_loop(
                bkv_start_idx,
                num_bkv,
                compute_with_bkv,
                (jnp.int32(1), ),
                unroll=False,
            )

            # Load acc and calculate final output.
            acc = acc_ref[...]

            if unnormalized_output:
                out = acc.astype(q_dtype)
            else:
                attention_sinks = jnp.concat(
                    [attention_sinks_ref[...] for _ in range(bq_sz)])[...,
                                                                      None]
                exp_attention_sinks = jnp.exp(attention_sinks - m_ref[...])
                L = l_ref[...] + exp_attention_sinks
                L = broadcast_minor(L, acc.shape)
                out = (lax.div(acc, L) if q_dtype == jnp.float32 else
                       (acc * pl.reciprocal(L, approx=True)).astype(q_dtype))

            # Wait for previous bo to be fully sent before storing new bo.
            bo_sem_idx = sem_ids_ref[2]
            sem_ids_ref[2] = lax.select(bo_sem_idx == 0, 1, 0)
            wait_send_bo(bo_sem_idx)

            # Store output from acc to bo.
            bo_x2_ref.at[bo_sem_idx].bitcast(jnp.int32).reshape(
                bq_sz * num_q_heads_per_q_packing,
                head_dim,
            )[...] = pltpu.bitcast(out, jnp.int32)
            bl_x2_ref.at[bo_sem_idx][:bq_sz, :num_q_heads] = l_ref[
                ..., 0].reshape(bq_sz, num_q_heads)
            bm_x2_ref.at[bo_sem_idx][:bq_sz, :num_q_heads] = m_ref[
                ..., 0].reshape(bq_sz, num_q_heads)

            # Send cur bo
            start_send_bo(seq_idx, bq_idx, bo_sem_idx)

        lax.fori_loop(0, num_bq, compute_with_bq, None, unroll=False)

    ### ------- Kernel start ------- ###

    @pl.when(seq_idx == start_seq_idx)
    def prologue():
        start_fetch_bq(start_seq_idx, 0, 0)
        start_fetch_bkv(
            start_seq_idx,
            jnp.maximum(
                _get_kv_len(start_seq_idx) - _get_q_len(start_seq_idx) -
                sliding_window,
                0,
            ) // bkv_sz,
            0,
        )

    process()

    @pl.when(seq_idx == end_seq_idx - 1)
    def epilogue():
        for i in range(2):
            wait_send_bo(i)
            wait_update_kv_cache(i)

    ### ------- Kernel end ------- ###


def prepare_q_inputs(
        q: jax.Array,  # [max_num_tokens, actual_num_q_heads, actual_head_dim],
):
    max_num_tokens, actual_num_q_heads, actual_head_dim = q.shape
    q_packing = get_dtype_packing(q.dtype)
    num_q_heads = align_to(actual_num_q_heads, q_packing)
    head_dim = align_to(actual_head_dim, 128)
    q = jnp.pad(
        q,
        (
            (0, 0),
            (0, num_q_heads - actual_num_q_heads),
            (0, head_dim - actual_head_dim),
        ),
        constant_values=0,
    )
    return q


def prepare_kv_inputs(kv: jax.Array):
    max_num_tokens, actual_head_dim = kv.shape
    kv_packing = get_dtype_packing(kv.dtype)
    # Pad to packing
    if max_num_tokens % kv_packing != 0:
        pad = kv_packing - (max_num_tokens % kv_packing)
        kv = jnp.pad(kv, ((0, pad), (0, 0)), constant_values=0)

    head_dim = align_to(actual_head_dim, 128)
    kv = kv.reshape(-1, kv_packing, actual_head_dim)
    kv = jnp.pad(kv, ((0, 0), (0, 0), (0, head_dim - actual_head_dim)),
                 constant_values=0)
    return kv


# Convert the bf16 KV inputs to the DSv4 FP8 format.
def quantize_kv_inputs(kv: jax.Array):
    actual_head_dim = kv.shape[-1]
    assert actual_head_dim == 512
    nope = kv[..., :448]
    rope = kv[..., 448:512]

    orig_shape = kv.shape
    batch_dims = orig_shape[:-1]

    nope_blocked = nope.reshape(*batch_dims, 7, 64)
    fp8_max = float(jnp.finfo(jnp.float8_e4m3fn).max)
    x_amax = jnp.max(jnp.abs(nope_blocked), axis=-1, keepdims=True)
    x_amax = jnp.clip(x_amax, 1e-4, None)
    sf = jnp.power(2.0, jnp.ceil(jnp.log2(x_amax / fp8_max)))

    fp8_quant = (nope_blocked * (1.0 / sf)).astype(jnp.float8_e4m3fn)
    fp8_quant_flat = fp8_quant.reshape(*batch_dims, 448)
    scales_quant = sf.reshape(*batch_dims, 7).astype(jnp.float8_e8m0fnu)

    fp8_uint8 = jax.lax.bitcast_convert_type(fp8_quant_flat,
                                             jnp.uint8).reshape(
                                                 *batch_dims, 448)
    bf16_uint8 = jax.lax.bitcast_convert_type(rope, jnp.uint8).reshape(
        *batch_dims, 128)
    scales_uint8 = jax.lax.bitcast_convert_type(scales_quant,
                                                jnp.uint8).reshape(
                                                    *batch_dims, 7)
    pad_uint8 = jnp.zeros((*batch_dims, 57), dtype=jnp.uint8)
    quantized = jnp.concatenate(
        [fp8_uint8, bf16_uint8, scales_uint8, pad_uint8], axis=-1)
    return quantized


def prepare_outputs(
    out,  # [max_num_tokens, num_q_heads, head_dim]
    actual_num_q_heads: int,
    actual_head_dim: int,
):
    return out[:, :actual_num_q_heads, :actual_head_dim]


# TODO: support batching decode q tokens as performance optimization.
@functools.partial(
    jax.jit,
    static_argnames=(
        "sm_scale",
        "sliding_window",
        "mask_value",
        "chunk_prefill_size",
        "num_kv_pages_per_block",
        "num_queries_per_block",
        "vmem_limit_bytes",
        "logical_page_size",
        "unnormalized_output",
    ),
    donate_argnames=("cache_kv", ),
)
def mla_sliding_window_ragged_paged_attention(
    q: jax.Array,  # [max_num_tokens, actual_num_q_heads, actual_head_dim]
    new_kv: jax.Array,  # [max_num_tokens, actual_head_dim]
    cache_kv: jax.
    Array,  # [total_num_pages, physical_page_size_per_kv_packing, kv_packing, lkv_dim]
    kv_lens: jax.Array,  # i32[max_num_seqs]
    page_indices: jax.Array,  # i32[max_num_seqs * pages_per_seq]
    cu_q_lens: jax.Array,  # i32[max_num_seqs + 1]
    distribution: jax.Array,  # i32[3]
    attention_sinks: jax.Array,  # float32[actual_num_q_heads]
    *,
    sm_scale: float = 1.0,
    sliding_window: int,
    logical_page_size: int,
    mask_value: float | None = DEFAULT_MASK_VALUE,
    # Kernel optimization params.
    chunk_prefill_size: int | None = None,
    # Kernel tuning params for decode, prefill, and mixed cases.
    # If passsed in as int, all cases are the same.
    num_kv_pages_per_block: tuple[int, int, int] | int | None = None,
    num_queries_per_block: tuple[int, int, int] | int | None = None,
    vmem_limit_bytes: int = DEFAULT_VMEM_LIMIT_BYTES,
    unnormalized_output: bool = False,
) -> tuple[
        jax.Array,  # [max_num_tokens, actual_num_q_heads, actual_head_dim]
        jax.
        Array,  # [total_num_pages, physical_page_size_per_kv_packing, kv_packing, lkv_dim]
        jax.Array,  # [max_num_tokens, actual_num_q_heads]
        jax.Array,  # [max_num_tokens, actual_num_q_heads]
]:
    """MLA Ragged paged attention that supports mixed prefill and decode.

  Args:
    q: concatenated all sequences' queries.
    new_kv: concatenated all sequences' kv values
    cache_kv: the current kv cache.
    kv_lens: the length of each sequence in the kv cache.
    page_indices: flattened page indices look-up table by (seq_id, page_id).
    cu_q_lens: the cumulative sum of the effective query lengths. Similar to
      kv_lens, only the first num_seqs+1 values are valid.
    distribution: (i, j, k) represents that sequences[0:i] are decode-only,
      sequences[i:j] are chunked-prefill-only, and sequences[j:k] are mixed. The
      k is also the total number of sequences.
    sm_scale: the softmax scale which will be applied to the Q@K^T.
    sliding_window: the sliding window size for the attention.
    logical_page_size: the logical page size. The second and third dimensions of
      the kv cache are related to the physical page size, which is guaranteed to
      be >= logical_page_size. For model with hybrid attentions, such as
      DeepSeek V4, vLLM may use a single physical page pool shared across
      different attention types. Therefore, multiple attention types' pages will
      be padded up to the same physical page size. For the perspective of this
      kernel, it only read & write data within the logical page size.
    mask_value: mask value for causal mask.
    num_kv_pages_per_block: number of kv pages to be processed in one flash
      attention block in the pallas kernel. This is a tuple of (decode, prefill,
      mixed) cases.
    num_queries_per_block: number of queries to be processed in one flash
      attention block in the pallas kernel. This is a tuple of (decode, prefill,
      mixed) cases.
    vmem_limit_bytes: the vmem limit for the pallas kernel.

  Returns:
    The output of attention and the updated kv cache.
  """
    if num_kv_pages_per_block is None or num_queries_per_block is None:
        raise ValueError(
            "num_kv_pages_per_block and num_queries_per_block must be specified."
        )
    if isinstance(num_kv_pages_per_block, int):
        num_kv_pages_per_blocks = [num_kv_pages_per_block for _ in range(3)]
    else:
        num_kv_pages_per_blocks = num_kv_pages_per_block

    if isinstance(num_queries_per_block, int):
        num_queries_per_blocks = [num_queries_per_block for _ in range(3)]
    else:
        num_queries_per_blocks = num_queries_per_block

    _, actual_num_q_heads, actual_head_dim = q.shape

    q = prepare_q_inputs(q)  # [max_num_tokens, num_q_heads, head_dim]
    attention_sinks = jnp.pad(
        attention_sinks,
        (0, q.shape[1] - actual_num_q_heads),
        constant_values=-jnp.inf,
    )
    assert new_kv.dtype == jnp.bfloat16
    new_kv = quantize_kv_inputs(new_kv)
    assert new_kv.dtype == jnp.uint8
    # KV cache is in DSv4 FP8 format.
    assert cache_kv.dtype == jnp.uint8
    new_kv = prepare_kv_inputs(
        new_kv)  # [max_num_tokens_per_kv_packing, kv_packing, head_dim]
    lkv_dim = new_kv.shape[-1]
    head_dim = q.shape[-1]

    _, physical_page_size_per_kv_packing, kv_packing, _ = cache_kv.shape
    physical_page_size = physical_page_size_per_kv_packing * kv_packing
    assert logical_page_size <= physical_page_size
    assert logical_page_size % kv_packing == 0
    page_size_per_kv_packing = logical_page_size // kv_packing

    _, num_q_heads, _ = q.shape
    max_num_seqs = kv_lens.shape[0]
    num_page_indices = page_indices.shape[0]
    assert num_page_indices % max_num_seqs == 0

    def run_mla_kernel(
        q: jax.Array,  # [max_num_tokens, actual_num_q_heads, head_dim]
        new_kv: jax.Array,  # [max_num_tokens, lkv_dim]
        cache_kv: jax.
        Array,  # [total_num_pages, physical_page_size_per_kv_packing, kv_packing, lkv_dim]
        kv_lens: jax.Array,  # i32[max_num_seqs]
        page_indices: jax.Array,  # i32[max_num_seqs * pages_per_seq]
        cu_q_lens: jax.Array,  # i32[max_num_seqs + 1]
        start_seq_idx: jax.Array,  # i32
        end_seq_idx: jax.Array,  # i32
        in_output: jax.Array,  # [max_num_tokens, actual_num_q_heads, head_dim]
        in_l: jax.Array,  # [max_num_tokens, num_l_heads]
        in_m: jax.Array,  # [max_num_tokens, num_l_heads]
        attention_sinks: jax.Array,  # float32[num_q_heads]
        static_q_len: int | None,
        unnormalized_output: bool,
        num_kv_pages_per_block: int,
        num_queries_per_block: int,
        case: MlaCase = MlaCase.MIXED,
    ):

        bkv_p = num_kv_pages_per_block
        if static_q_len is not None:
            bq_sz = min(num_queries_per_block, static_q_len)
        else:
            bq_sz = num_queries_per_block
        bkv_sz_per_kv_packing = bkv_p * page_size_per_kv_packing
        # Add 2 additional words of buffering to accommodate misaligned new KV.
        # We need two additional words because the beginning and end of the new KV may
        # both not be aligned to kv_packing boundaries.
        # Example:
        #
        # T0 T4     K2  K6
        # T1        K3
        # T2    K0  K4
        # T3    K1  K5
        #
        # - Ti is existing KV tokens and Ki is the new KV.
        # - Each column is a 32-bit word.
        # - KV packing is 4
        #
        # We have 12 total tokens, so normally we would only allocate 12/4=3 words
        # But due to misalignment, we need to allocate 5 words.
        bkv_buf_sz_per_kv_packing = bkv_sz_per_kv_packing + 2
        grid = (end_seq_idx - start_seq_idx, )

        in_specs = [
            pl.BlockSpec(memory_space=pltpu.VMEM),  # attention_sinks
            pl.BlockSpec(memory_space=pltpu.HBM),  # q
            pl.BlockSpec(memory_space=pltpu.HBM),  # new_kv
            pl.BlockSpec(memory_space=pltpu.HBM),  # cache_kv
            pl.BlockSpec(memory_space=pltpu.HBM),  # in_output
            pl.BlockSpec(memory_space=pltpu.HBM),  # in_l
            pl.BlockSpec(memory_space=pltpu.HBM),  # in_m
        ]

        out_specs = [
            pl.BlockSpec(memory_space=pltpu.HBM),  # o
            pl.BlockSpec(memory_space=pltpu.HBM),  # updated_cache_kv
            pl.BlockSpec(memory_space=pltpu.HBM),  # l
            pl.BlockSpec(memory_space=pltpu.HBM),  # m
        ]

        bkv_double_buf = pltpu.VMEM(
            (2, bkv_buf_sz_per_kv_packing, kv_packing, lkv_dim),
            cache_kv.dtype,
        )

        bq_double_bufq = pltpu.VMEM(
            (2, bq_sz, num_q_heads, head_dim),
            q.dtype,
        )

        bo_double_buf = bq_double_bufq

        num_l_heads = align_to(num_q_heads, 128)
        bl_double_buf = pltpu.VMEM(
            (2, bq_sz, num_l_heads),
            jnp.float32,
        )
        bm_double_buf = bl_double_buf

        l_scratch = pltpu.VMEM(
            (bq_sz * num_q_heads, 128),
            jnp.float32,
        )
        m_scratch = l_scratch

        acc_scratch = pltpu.VMEM(
            (bq_sz * num_q_heads, head_dim),
            jnp.float32,
        )

        scratch_shapes = [
            bkv_double_buf,
            bq_double_bufq,
            bo_double_buf,  # Double buffering for output block.
            bl_double_buf,  # Double buffering for l output.
            bm_double_buf,  # Double buffering for m output.
            # Semaphores for double buffering of bkv, bq, bo, bkv_update, l, m.
            pltpu.SemaphoreType.DMA((6, 2)),
            # Intermediate buffers per kv head for flash attention.
            l_scratch,
            m_scratch,
            acc_scratch,
        ]

        scalar_prefetches = (
            kv_lens,
            page_indices,
            cu_q_lens,
            jnp.array([start_seq_idx, end_seq_idx], jnp.int32),
            # (bq_sem_idx, bkv_sem_idx, bo_sem_idx)
            jnp.zeros((3, ), jnp.int32),
            # (bo_sem_0_seq_idx, bo_sem_1_seq_idx, bo_sem_0_bo_idx, bo_sem_1_bo_idx)
            jnp.full((4, ), -1, jnp.int32),
            # (bkv_sem_0_seq_idx, bkv_sem_1_seq_idx, bkv_sem_0_offset, bkv_sem_1_offset, bkv_sem_0_sz, bkv_sem_1_sz)
            jnp.full((6, ), -1, jnp.int32),
        )

        scope_name = f"MLA-{case.symbol}-bq_{bq_sz}-bkvp_{bkv_p}"
        kernel = jax.named_scope(scope_name)(
            pl.pallas_call(
                functools.partial(
                    _mla_sliding_window_ragged_paged_attention_kernel,
                    sm_scale=sm_scale,
                    sliding_window=sliding_window,
                    mask_value=mask_value,
                    static_q_len=static_q_len,
                    bq_sz=bq_sz,
                    bkv_p=bkv_p,
                    logical_page_size=logical_page_size,
                    unnormalized_output=unnormalized_output,
                ),
                grid_spec=pltpu.PrefetchScalarGridSpec(
                    num_scalar_prefetch=len(scalar_prefetches),
                    in_specs=in_specs,
                    out_specs=out_specs,
                    grid=grid,
                    scratch_shapes=scratch_shapes,
                ),
                compiler_params=pltpu.CompilerParams(
                    dimension_semantics=("arbitrary", ),
                    vmem_limit_bytes=vmem_limit_bytes,
                    disable_bounds_checks=True,
                ),
                out_shape=[
                    jax.ShapeDtypeStruct(shape=q.shape, dtype=q.dtype),
                    jax.ShapeDtypeStruct(shape=cache_kv.shape,
                                         dtype=cache_kv.dtype),
                    jax.ShapeDtypeStruct(shape=(q.shape[0], num_l_heads),
                                         dtype=jnp.float32),
                    jax.ShapeDtypeStruct(shape=(q.shape[0], num_l_heads),
                                         dtype=jnp.float32),
                ],
                input_output_aliases={
                    11: 0,  # Alias output activation with in_output
                    10: 1,  # Aliasing cache_kv with updated_cache_kv
                    12: 2,  # Alias l with in_l
                    13: 3,  # Alias m with in_m
                },
                name=scope_name,
            ))
        return kernel(
            *scalar_prefetches,
            attention_sinks,
            q,
            new_kv,
            cache_kv,
            in_output,
            in_l,
            in_m,
        )

    # Decode-only
    num_l_heads = align_to(num_q_heads, 128)
    L = jnp.zeros((q.shape[0], num_l_heads), dtype=jnp.float32)
    m = jnp.zeros((q.shape[0], num_l_heads), dtype=jnp.float32)
    output, updated_kv, out_l, out_m = run_mla_kernel(
        q,
        new_kv,
        cache_kv,
        kv_lens,
        page_indices,
        cu_q_lens,
        num_kv_pages_per_block=num_kv_pages_per_blocks[0],
        num_queries_per_block=num_queries_per_blocks[0],
        start_seq_idx=jnp.array(0),
        end_seq_idx=distribution[0],
        in_output=jnp.zeros_like(q),
        in_l=L,
        in_m=m,
        attention_sinks=attention_sinks,
        static_q_len=1,
        unnormalized_output=unnormalized_output,
        case=MlaCase.DECODE,
    )

    if chunk_prefill_size is not None:
        # Handle prefill where the query length is fixed per sequence.
        output, updated_kv, out_l, out_m = run_mla_kernel(
            q,
            new_kv,
            updated_kv,
            kv_lens,
            page_indices,
            cu_q_lens,
            num_kv_pages_per_block=num_kv_pages_per_blocks[1],
            num_queries_per_block=num_queries_per_blocks[1],
            start_seq_idx=distribution[0],
            end_seq_idx=distribution[1],
            in_output=output,
            in_l=out_l,
            in_m=out_m,
            attention_sinks=attention_sinks,
            static_q_len=chunk_prefill_size,
            unnormalized_output=unnormalized_output,
            case=MlaCase.PREFILL,
        )

    # Handle mixed case where the query length per sequence is variable.
    output, updated_kv, out_l, out_m = run_mla_kernel(
        q,
        new_kv,
        updated_kv,
        kv_lens,
        page_indices,
        cu_q_lens,
        num_kv_pages_per_block=num_kv_pages_per_blocks[2],
        num_queries_per_block=num_queries_per_blocks[2],
        start_seq_idx=distribution[1],
        end_seq_idx=distribution[2],
        in_output=output,
        in_l=out_l,
        in_m=out_m,
        attention_sinks=attention_sinks,
        static_q_len=None,
        unnormalized_output=unnormalized_output,
        case=MlaCase.MIXED,
    )
    output = prepare_outputs(
        output, actual_num_q_heads, actual_head_dim
    )  # [max_num_tokens, actual_num_q_heads, actual_head_dim]
    out_l = out_l[:, :actual_num_q_heads]
    out_m = out_m[:, :actual_num_q_heads]

    return output, updated_kv, out_l, out_m
