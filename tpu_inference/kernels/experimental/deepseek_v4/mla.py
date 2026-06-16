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


def get_kv_cache_shape(
    total_num_pages,
    page_size,
    kv_dim,
    kv_dtype,
):
    kv_packing = get_dtype_packing(kv_dtype)
    return (
        total_num_pages,
        align_to(page_size, kv_packing) // kv_packing,
        kv_packing,
        align_to(kv_dim, 128),
    )


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


def _mla_ragged_paged_attention_kernel(
    # Prefetch
    kv_lens_ref,  # [max_num_seqs]
    kv_lens_to_attend_ref,  # [max_num_tokens]
    page_indices_ref,  # [max_num_seqs * pages_per_seq]
    cu_q_lens_ref,  # [max_num_seqs + 1]
    start_end_seq_idx_ref,  # [2] (start_seq_idx, end_seq_idx)
    sem_ids_ref,  # [3] (bq_sem_idx, bkv_sem_idx, bo_sem_idx)
    bo_ids_ref,  # [4] (bo_sem_0_seq_idx, bo_sem_1_seq_idx, bo_sem_0_bo_idx, bo_sem_1_bo_idx)
    # Input
    attention_sinks_ref,  # float32[num_q_heads]
    q_hbm_ref,  # [max_num_tokens, num_q_heads, head_dim]
    cache_kv_hbm_ref,  # [total_num_pages, page_size_per_kv_packing, kv_packing, lkv_dim]
    swa_accumution_hbm_ref,  # [max_num_tokens, num_q_heads, head_dim]
    swa_l_hbm_ref,  # [max_num_tokens, num_l_heads]
    swa_m_hbm_ref,  # [max_num_tokens, num_l_heads]
    topk_indices_ref,  # [max_num_tokens, topk]
    # Output
    o_hbm_ref,  # [max_num_tokens, num_q_heads, head_dim]
    # Scratch
    bkv_x2_ref,  # [2, bkv_buf_sz_per_kv_packing, kv_packing, lkv_dim]
    bq_x2_ref,  # [2, bq_sz, num_q_heads, head_dim]
    bo_x2_ref,  # [2, bq_sz, num_q_heads, head_dim]
    bl_x2_ref,  # [2, bq_sz, num_l_heads]
    bm_x2_ref,  # [2, bq_sz, num_l_heads]
    swa_acc_x2_ref,  # [2, bq_sz, num_q_heads, head_dim]
    topk_indices_x2_ref,  # [2, bq_sz, csa_topk]
    sems,  # [7, 2]
    l_ref,  # [bq_sz * num_q_heads, 128],
    m_ref,  # [bq_sz * num_q_heads, 128],
    acc_ref,  # [bq_sz * num_q_heads, head_dim],
    *,
    static_q_len: int,
    sm_scale: float,
    mask_value: float = DEFAULT_MASK_VALUE,
    bkv_p,
    bq_sz,
):
    assert q_hbm_ref.shape == o_hbm_ref.shape

    _, num_q_heads, head_dim = q_hbm_ref.shape
    total_num_pages, page_size_per_kv_packing, kv_packing, lkv_dim = (
        cache_kv_hbm_ref.shape)
    max_num_seqs = kv_lens_ref.shape[0]
    num_page_indices = page_indices_ref.shape[0]

    assert num_page_indices % max_num_seqs == 0
    pages_per_seq = num_page_indices // max_num_seqs
    q_dtype = q_hbm_ref.dtype
    q_packing = get_dtype_packing(q_dtype)
    # Validate against the KV dtype.
    kv_dtype = cache_kv_hbm_ref.dtype
    assert o_hbm_ref.dtype == q_dtype
    assert get_dtype_packing(kv_dtype) == kv_packing
    assert lkv_dim % 128 == 0
    assert head_dim % 128 == 0
    bkv_sz_per_kv_packing = bkv_p * page_size_per_kv_packing
    bkv_sz = bkv_sz_per_kv_packing * kv_packing
    assert num_q_heads % q_packing == 0
    num_q_heads_per_q_packing = num_q_heads // q_packing

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
        kv_lens_to_attend_segment,
        bq_topk_indices,
    ):
        assert len(q.shape) == 2
        assert len(kv.shape) == 2
        assert q.shape[0] % num_q_heads == 0
        assert q.shape[1] == head_dim
        assert kv.shape == (bkv_sz, head_dim)
        head_l_ref = l_ref.at[:q.shape[0]]
        head_m_ref = m_ref.at[:q.shape[0]]
        head_acc_ref = acc_ref.at[:q.shape[0]]

        # Follow FlashAttention-2 forward pass.
        s = jnp.einsum("nd,md->nm", q, kv, preferred_element_type=jnp.float32)
        s *= sm_scale

        if kv_lens_to_attend_segment is not None:
            assert bq_topk_indices is None
            k_span = bkv_idx * bkv_sz + lax.broadcasted_iota(
                jnp.int32, s.shape, 1)
            mask = kv_lens_to_attend_segment.reshape(s.shape) <= k_span
        else:
            assert bq_topk_indices is not None
            k_span = bkv_idx * bkv_sz + jnp.arange(bkv_sz, dtype=jnp.int32)
            valid_mask = jnp.any(bq_topk_indices[:, None, :] == k_span[None, :,
                                                                       None],
                                 axis=-1)  # [bq_sz, bkv_sz]
            valid_mask = jnp.broadcast_to(
                valid_mask[:, None, :],
                (bq_sz, num_q_heads, bkv_sz)).reshape(s.shape)
            mask = jnp.logical_not(valid_mask)

        s = jnp.where(mask, mask_value, s)
        s_rowmax = jnp.max(s, axis=1, keepdims=True)
        m_prev = head_m_ref[...]
        m_curr = jnp.maximum(m_prev, s_rowmax)
        head_m_ref[...] = m_curr
        p = jnp.exp(s - broadcast_minor(m_curr, s.shape))

        pv = jnp.einsum("nm,md->nd", p, kv, preferred_element_type=jnp.float32)

        p_rowsum = jnp.sum(p, axis=1, keepdims=True)
        exp_m_diff = jnp.exp(m_prev - m_curr)
        l_prev = head_l_ref[...]
        l_curr = exp_m_diff * l_prev + p_rowsum
        head_l_ref[...] = l_curr
        o_prev = head_acc_ref[...]
        o_curr = broadcast_minor(exp_m_diff, o_prev.shape) * o_prev + pv
        head_acc_ref[...] = o_curr

    def _async_copy(src, dst, sem, wait):
        cp = pltpu.make_async_copy(src, dst, sem)
        if wait:
            cp.wait()
        else:
            cp.start()

    def _fetch_bkv(seq_idx, bkv_idx, bkv_sem_idx, *, wait=False):
        sem = sems.at[0, bkv_sem_idx]
        # bkv_x2_ref shape: [2, bkv_sz_per_kv_packing, kv_packing, lkv_dim]
        bkv_vmem_ref = bkv_x2_ref.at[bkv_sem_idx]

        # [total_num_pages, page_size_per_kv_packing, kv_packing, lkv_dim]
        # [total_num_pages * page_size_per_kv_packing, kv_packing, lkv_dim]
        reshaped_cache_hbm_ref = cache_kv_hbm_ref.reshape(
            total_num_pages * page_size_per_kv_packing,
            *cache_kv_hbm_ref.shape[2:],
        )

        kv_len = kv_lens_ref[seq_idx]
        kv_len_start = bkv_idx * bkv_sz
        kv_p_start = bkv_idx * bkv_p

        kv_left = kv_len - kv_len_start
        kv_left_per_kv_packing = cdiv(kv_left, kv_packing)
        dma_bkv_sz = jnp.minimum(kv_left_per_kv_packing, bkv_sz_per_kv_packing)
        page_indices_offset = seq_idx * pages_per_seq + kv_p_start

        if not wait:
            # Fetch effective kv from kv cache. To pipeline multiple DMA calls, we
            # utilize static for loop instead of dynamic for loop.
            # Loop through all pages in a block
            for i in range(bkv_p):
                # Ensure only effective kvs are copied and we don't go negative.
                sz_per_kv_packing = jnp.clip(
                    kv_left_per_kv_packing - i * page_size_per_kv_packing,
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
                            page_size_per_kv_packing,
                            sz_per_kv_packing,
                        ),
                    ],
                    # [bkv_sz_per_kv_packing, kv_packing, lkv_dim].
                    bkv_vmem_ref.at[pl.ds(i * page_size_per_kv_packing,
                                          sz_per_kv_packing)],
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

    def _fetch_swa(seq_idx, bq_idx, bq_sem_idx, *, wait=False):
        sem_acc = sems.at[3, bq_sem_idx]
        sem_l = sems.at[4, bq_sem_idx]
        sem_m = sems.at[5, bq_sem_idx]

        q_len_start = cu_q_lens_ref[seq_idx] + bq_idx * bq_sz
        q_end = cu_q_lens_ref[seq_idx + 1]
        sz = jnp.minimum(bq_sz, q_end - q_len_start)

        if not wait:
            _async_copy(
                swa_accumution_hbm_ref.at[pl.ds(q_len_start, sz)],
                swa_acc_x2_ref.at[bq_sem_idx, pl.ds(0, sz)],
                sem_acc,
                wait=False,
            )
            _async_copy(
                swa_l_hbm_ref.at[pl.ds(q_len_start, sz)],
                bl_x2_ref.at[bq_sem_idx, pl.ds(0, sz)],
                sem_l,
                wait=False,
            )
            _async_copy(
                swa_m_hbm_ref.at[pl.ds(q_len_start, sz)],
                bm_x2_ref.at[bq_sem_idx, pl.ds(0, sz)],
                sem_m,
                wait=False,
            )

        else:
            dst_acc = swa_acc_x2_ref.at[bq_sem_idx, pl.ds(0, sz)]
            _async_copy(src=dst_acc, dst=dst_acc, sem=sem_acc, wait=True)

            dst_l = bl_x2_ref.at[bq_sem_idx, pl.ds(0, sz)]
            _async_copy(src=dst_l, dst=dst_l, sem=sem_l, wait=True)

            dst_m = bm_x2_ref.at[bq_sem_idx, pl.ds(0, sz)]
            _async_copy(src=dst_m, dst=dst_m, sem=sem_m, wait=True)

            acc_ref[...] = (swa_acc_x2_ref[bq_sem_idx,
                                           ...].astype(jnp.float32).reshape(
                                               bq_sz * num_q_heads, head_dim))
            bl = jnp.concat([
                bl_x2_ref[bq_sem_idx, i, :num_q_heads] for i in range(bq_sz)
            ])[..., None]
            l_ref[...] = jnp.concat([bl for _ in range(128)], axis=-1)

            bm = jnp.concat([
                bm_x2_ref[bq_sem_idx, i, :num_q_heads] for i in range(bq_sz)
            ])[..., None]
            m_ref[...] = jnp.concat([bm for _ in range(128)], axis=-1)

    def _fetch_topk_indices(seq_idx, bq_idx, bq_sem_idx, *, wait=False):
        if topk_indices_ref is not None:
            sem = sems.at[6, bq_sem_idx]
            vmem_ref = topk_indices_x2_ref.at[bq_sem_idx]

            q_len_start = cu_q_lens_ref[seq_idx] + bq_idx * bq_sz
            q_end = cu_q_lens_ref[seq_idx + 1]
            sz = jnp.minimum(bq_sz, q_end - q_len_start)

            if not wait:
                _async_copy(
                    topk_indices_ref.at[pl.ds(q_len_start, sz)],
                    vmem_ref.at[pl.ds(0, sz)],
                    sem,
                    wait=False,
                )
            else:
                dst = vmem_ref.at[pl.ds(0, sz)]
                _async_copy(src=dst, dst=dst, sem=sem, wait=True)

    def start_fetch_bkv(seq_idx, bkv_idx, bkv_sem_idx):
        return _fetch_bkv(seq_idx, bkv_idx, bkv_sem_idx)

    def wait_fetch_bkv(seq_idx, bkv_idx, bkv_sem_idx):
        return _fetch_bkv(seq_idx, bkv_idx, bkv_sem_idx, wait=True)

    def start_fetch_bq(seq_idx, bq_idx, bq_sem_idx):
        return _fetch_bq(seq_idx, bq_idx, bq_sem_idx)

    def wait_fetch_bq(seq_idx, bq_idx, bq_sem_idx):
        return _fetch_bq(seq_idx, bq_idx, bq_sem_idx, wait=True)

    def start_fetch_swa(seq_idx, bq_idx, bq_sem_idx):
        return _fetch_swa(seq_idx, bq_idx, bq_sem_idx)

    def wait_fetch_swa(seq_idx, bq_idx, bq_sem_idx):
        return _fetch_swa(seq_idx, bq_idx, bq_sem_idx, wait=True)

    def start_fetch_topk_indices(seq_idx, bq_idx, bq_sem_idx):
        return _fetch_topk_indices(seq_idx, bq_idx, bq_sem_idx)

    def wait_fetch_topk_indices(seq_idx, bq_idx, bq_sem_idx):
        return _fetch_topk_indices(seq_idx, bq_idx, bq_sem_idx, wait=True)

    def load_topk_indices(bq_sem_idx):
        return topk_indices_x2_ref[bq_sem_idx, ...]

    def start_send_bo(seq_idx, bo_idx, bo_sem_idx):
        bo_ids_ref[bo_sem_idx] = seq_idx
        bo_ids_ref[bo_sem_idx + 2] = bo_idx
        _send_bo(seq_idx, bo_idx, bo_sem_idx)

    def wait_send_bo(bo_sem_idx):
        old_seq_idx = bo_ids_ref[bo_sem_idx]
        old_bo_idx = bo_ids_ref[bo_sem_idx + 2]

        @pl.when(jnp.logical_and(0 <= old_seq_idx, old_seq_idx <= seq_idx))
        def _():
            _send_bo(old_seq_idx, old_bo_idx, bo_sem_idx, wait=True)

    def load_bq(bq_sem_idx):
        q_ref = (bq_x2_ref.bitcast(jnp.uint32).at[bq_sem_idx].reshape(
            bq_sz * num_q_heads_per_q_packing, head_dim))
        q = pltpu.bitcast(
            q_ref[:bq_sz * num_q_heads_per_q_packing],
            q_dtype,
        ).reshape(bq_sz * num_q_heads, head_dim)
        return q

    def load_bkv(bkv_sem_idx):
        bkv_ref = (bkv_x2_ref.bitcast(
            jnp.uint32).at[bkv_sem_idx, :bkv_sz_per_kv_packing].reshape(
                bkv_sz_per_kv_packing, lkv_dim))
        bkv = pltpu.bitcast(bkv_ref[...], kv_dtype).reshape(bkv_sz, lkv_dim)

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
        num_bkv = jnp.maximum(1, cdiv(kv_len, bkv_sz))
        if static_q_len is None:
            num_bq = cdiv(q_len, bq_sz)
        else:
            num_bq = cdiv(static_q_len, bq_sz)

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
            next_bkv_idx = lax.select(is_last_bkv, 0, next_bkv_idx)
            next_bq_idx = lax.select(is_last_bkv, bq_idx + 1, bq_idx)
            is_last_bq = next_bq_idx == num_bq
            next_bq_idx = lax.select(is_last_bq, 0, next_bq_idx)
            next_seq_idx = lax.select(is_last_bq, seq_idx + 1, seq_idx)
            next_bkv_sem_idx = lax.select(bkv_sem_idx == 0, 1, 0)
            return next_seq_idx, next_bq_idx, next_bkv_idx, next_bkv_sem_idx

        def compute_with_bq(bq_idx, _):
            bq_sem_idx = sem_ids_ref[0]
            next_seq_idx, next_bq_idx, next_bq_sem_idx = get_next_bq_ids(
                seq_idx, bq_idx, bq_sem_idx)

            if kv_lens_to_attend_ref is not None:
                kv_lens_to_attend_segment = jnp.broadcast_to(
                    jnp.stack([
                        kv_lens_to_attend_ref[q_start + bq_idx * bq_sz + i]
                        for i in range(bq_sz)
                    ])[:, None, None],
                    (bq_sz, num_q_heads, bkv_sz),
                )
            else:
                kv_lens_to_attend_segment = None

            # Prefetch next bq
            @pl.when(next_seq_idx < end_seq_idx)
            def prefetch_next_bq():
                sem_ids_ref[0] = next_bq_sem_idx
                start_fetch_bq(next_seq_idx, next_bq_idx, next_bq_sem_idx)
                start_fetch_swa(next_seq_idx, next_bq_idx, next_bq_sem_idx)
                start_fetch_topk_indices(next_seq_idx, next_bq_idx,
                                         next_bq_sem_idx)

            def compute_with_bkv(bkv_idx, carry):
                kv_lens_to_attend_segment = carry[0]

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
                wait_fetch_bkv(seq_idx, bkv_idx, bkv_sem_idx)

                # Load bkv into vreg. There is no need to mask out invalid k/v entries,
                # because the score of invalid Q.K^T pairs are masked (to be zero) in
                # flash attention, so that the invalid kv entries
                # (as long as they are not NaN or inf) won't affect to the output.
                bkv = load_bkv(bkv_sem_idx, )

                bq = load_bq(bq_sem_idx)

                if topk_indices_ref is not None:
                    bq_topk_indices = load_topk_indices(
                        bq_sem_idx)  # [bq_sz, topk]
                else:
                    bq_topk_indices = None

                flash_attention(
                    bq,
                    bkv,
                    bq_idx=bq_idx,
                    bkv_idx=bkv_idx,
                    kv_lens_to_attend_segment=kv_lens_to_attend_segment,
                    bq_topk_indices=bq_topk_indices,
                )
                return (kv_lens_to_attend_segment, )

            # Wait for cur bq if not ready yet
            wait_fetch_bq(seq_idx, bq_idx, bq_sem_idx)
            wait_fetch_swa(seq_idx, bq_idx, bq_sem_idx)
            wait_fetch_topk_indices(seq_idx, bq_idx, bq_sem_idx)
            jax.lax.fori_loop(
                0,
                num_bkv,
                compute_with_bkv,
                (kv_lens_to_attend_segment, ),
                unroll=False,
            )

            # Load acc and calculate final output.
            acc = acc_ref[...]
            attention_sinks = jnp.concat(
                [attention_sinks_ref[...] for _ in range(bq_sz)])[..., None]
            exp_attention_sinks = jnp.exp(attention_sinks - m_ref[...])
            lse = l_ref[...] + exp_attention_sinks
            lse = broadcast_minor(lse, acc.shape)
            out = (lax.div(acc, lse) if q_dtype == jnp.float32 else
                   (acc * pl.reciprocal(lse, approx=True)).astype(q_dtype))

            # Wait for previous bo to be fully sent before storing new bo.
            bo_sem_idx = sem_ids_ref[2]
            sem_ids_ref[2] = lax.select(bo_sem_idx == 0, 1, 0)
            wait_send_bo(bo_sem_idx)

            # Store output from acc to bo.
            bo_x2_ref.at[bo_sem_idx].bitcast(jnp.int32).reshape(
                bq_sz * num_q_heads_per_q_packing,
                head_dim,
            )[...] = pltpu.bitcast(out, jnp.int32)

            # Send cur bo
            start_send_bo(seq_idx, bq_idx, bo_sem_idx)

        lax.fori_loop(0, num_bq, compute_with_bq, None, unroll=False)

    ### ------- Kernel start ------- ###

    @pl.when(seq_idx == start_seq_idx)
    def prologue():
        start_fetch_bq(start_seq_idx, 0, 0)
        start_fetch_swa(start_seq_idx, 0, 0)
        start_fetch_topk_indices(start_seq_idx, 0, 0)

        # Initialize bkv_x2_ref to zeros to avoid NaN issues from accessing
        # uninitialized memory. Bitcast into int32 to avoid tiling issues.
        bkv_x2_int32_ref = bkv_x2_ref.bitcast(jnp.int32).reshape(
            (2, -1, lkv_dim))
        bkv_zeros = jnp.zeros(bkv_x2_int32_ref.shape[1:], jnp.int32)

        # To pipeline VST and DMA, we divide the initialization into two steps.
        bkv_x2_int32_ref[0] = bkv_zeros
        start_fetch_bkv(start_seq_idx, 0, 0)
        bkv_x2_int32_ref[1] = bkv_zeros

    process()

    @pl.when(seq_idx == end_seq_idx - 1)
    def epilogue():
        for i in range(2):
            wait_send_bo(i)

    ### ------- Kernel end ------- ###


def prepare_q_inputs(
        q: jax.Array,  # [max_num_tokens, actual_num_q_heads, actual_head_dim],
):
    _, actual_num_q_heads, actual_head_dim = q.shape
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


def prepare_swa_inputs(
        swa_accumution: jax.Array,  # [max_num_tokens, num_q_heads, head_dim]
        swa_l: jax.Array,  # [max_num_tokens, num_q_heads]
        swa_m: jax.Array,  # [max_num_tokens, num_q_heads]
):
    _, actual_num_q_heads, actual_head_dim = swa_accumution.shape
    swa_packing = get_dtype_packing(swa_accumution.dtype)
    num_q_heads = align_to(actual_num_q_heads, swa_packing)
    head_dim = align_to(actual_head_dim, 128)
    swa_accumution = jnp.pad(
        swa_accumution,
        (
            (0, 0),
            (0, num_q_heads - actual_num_q_heads),
            (0, head_dim - actual_head_dim),
        ),
        constant_values=0,
    )
    num_l_heads = align_to(num_q_heads, 128)
    swa_l = jnp.pad(
        swa_l,
        (
            (0, 0),
            (0, num_l_heads - actual_num_q_heads),
        ),
        constant_values=0,
    )
    swa_m = jnp.pad(
        swa_m,
        (
            (0, 0),
            (0, num_l_heads - actual_num_q_heads),
        ),
        constant_values=0,
    )
    return swa_accumution, swa_l, swa_m


def prepare_outputs(
    out,  # [max_num_tokens, num_q_heads, head_dim]
    actual_num_q_heads: int,
    actual_head_dim: int,
):
    return out[:, :actual_num_q_heads, :actual_head_dim]


# TODO: support batching decode q tokens as performance optimization.


# Main Attention kernel for DeepSeek V4 CSA and HCA.
# Note that the compressed kv tokens of current batch (current forward pass)
# have been written to the `cache_kv` by the compressor module before calling
# this function, `kv_lens` reflects the length after compressed kv cache write.
@functools.partial(
    jax.jit,
    static_argnames=(
        "sm_scale",
        "mask_value",
        "chunk_prefill_size",
        "num_kv_pages_per_block",
        "num_queries_per_block",
        "vmem_limit_bytes",
    ),
)
def mla_ragged_paged_attention(
    q: jax.Array,  # [max_num_tokens, actual_num_q_heads, head_dim]
    cache_kv: jax.
    Array,  # [total_num_pages, page_size_per_kv_packing, kv_packing, lkv_dim]
    kv_lens: jax.Array,  # i32[max_num_seqs]
    kv_lens_to_attend: jax.Array | None,  # i32[max_num_tokens]
    topk_indices: jax.Array | None,  # i32[max_num_tokens, csa_topk]
    page_indices: jax.Array,  # i32[max_num_seqs * pages_per_seq]
    cu_q_lens: jax.Array,  # i32[max_num_seqs + 1]
    distribution: jax.Array,  # i32[3]
    attention_sinks: jax.Array,  # float32[actual_num_q_heads]
    swa_accumution: jax.
    Array,  # float32[max_num_tokens, num_q_heads, head_dim]
    swa_l: jax.Array,  # float32[max_num_tokens, num_q_heads]
    swa_m: jax.Array,  # float32[max_num_tokens, num_q_heads]
    *,
    sm_scale: float = 1.0,
    mask_value: float | None = DEFAULT_MASK_VALUE,
    # Kernel optimization params.
    chunk_prefill_size: int | None = None,
    # Kernel tuning params for decode, prefill, and mixed cases.
    # If passsed in as int, all cases are the same.
    num_kv_pages_per_block: tuple[int, int, int] | int | None = None,
    num_queries_per_block: tuple[int, int, int] | int | None = None,
    vmem_limit_bytes: int = DEFAULT_VMEM_LIMIT_BYTES,
) -> jax.Array:
    """MLA Ragged paged attention that supports mixed prefill and decode.

  Args:
    q: concatenated all sequences' queries.
    cache_kv: the current kv cache.
    kv_lens: the length of each sequence in the kv cache.
    kv_lens_to_attend: for each query token, the length of kv sequence to attend
      to. The attend to length is <= kv_lens[seq_id] for that query token. Only
      used for HCA, should be None for CSA.
    topk_indices: for each query token, the indices of the top k key tokens to
      attend to. Only used for CSA, should be None for HCA.
    page_indices: flattened page indices look-up table by (seq_id, page_id).
    cu_q_lens: the cumulative sum of the effective query lengths. Similar to
      kv_lens, only the first num_seqs+1 values are valid.
    distribution: (i, j, k) represents that sequences[0:i] are decode-only,
      sequences[i:j] are chunked-prefill-only, and sequences[j:k] are mixed. The
      k is also the total number of sequences.
    sm_scale: the softmax scale which will be applied to the Q@K^T.
    mask_value: mask value for causal mask.
    num_kv_pages_per_block: number of kv pages to be processed in one flash
      attention block in the pallas kernel. This is a tuple of (decode, prefill,
      mixed) cases.
    num_queries_per_block: number of queries to be processed in one flash
      attention block in the pallas kernel. This is a tuple of (decode, prefill,
      mixed) cases.
    vmem_limit_bytes: the vmem limit for the pallas kernel.

  Returns:
    The output of attention.
  """
    # The cache is DSV4 FP8 format.
    # 448 fp8, 64 bf16, 7 fp8 scales, 7 e8m0 scale for 448 fp8 (block size 64)
    assert cache_kv.shape[-1] >= (448 + 64 * 2 + 7)
    assert cache_kv.dtype == jnp.uint8

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
    head_dim = q.shape[-1]
    attention_sinks = jnp.pad(
        attention_sinks,
        (0, align_to(actual_num_q_heads, 128) - actual_num_q_heads),
        constant_values=-jnp.inf,
    )
    assert swa_accumution.dtype == q.dtype
    swa_accumution, swa_l, swa_m = prepare_swa_inputs(swa_accumution, swa_l,
                                                      swa_m)

    _, page_size_per_kv_packing, kv_packing, lkv_dim = cache_kv.shape
    page_size = page_size_per_kv_packing * kv_packing
    _, num_q_heads, _ = q.shape
    max_num_seqs = cu_q_lens.shape[0] - 1
    num_page_indices = page_indices.shape[0]
    assert num_page_indices % max_num_seqs == 0

    is_csa = topk_indices is not None
    if is_csa:
        assert kv_lens_to_attend is None
    else:
        # HCA
        assert kv_lens_to_attend is not None

    def run_mla_kernel(
        q: jax.Array,  # [max_num_tokens, num_q_heads, head_dim]
        cache_kv: jax.
        Array,  # [total_num_pages, page_size_per_kv_packing, kv_packing, lkv_dim]
        kv_lens: jax.Array,  # i32[max_num_seqs]
        kv_lens_to_attend: jax.Array | None,  # i32[max_num_tokens]
        topk_indices: jax.Array | None,  # i32[max_num_tokens, csa_topk]
        page_indices: jax.Array,  # i32[max_num_seqs * pages_per_seq]
        cu_q_lens: jax.Array,  # i32[max_num_seqs + 1]
        attention_sinks: jax.Array,  # float32[num_q_heads]
        swa_accumution: jax.
        Array,  # float32[max_num_tokens, num_q_heads, head_dim]
        swa_l: jax.Array,  # float32[max_num_tokens, num_l_heads]
        swa_m: jax.Array,  # float32[max_num_tokens, num_l_heads]
        start_seq_idx: jax.Array,  # i32
        end_seq_idx: jax.Array,  # i32
        static_q_len: int | None,
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
        bkv_buf_sz_per_kv_packing = bkv_sz_per_kv_packing
        grid = (end_seq_idx - start_seq_idx, )

        in_specs = [
            pl.BlockSpec(memory_space=pltpu.VMEM),  # attention_sinks
            pl.BlockSpec(memory_space=pltpu.HBM),  # q
            pl.BlockSpec(memory_space=pltpu.HBM),  # cache_kv
            pl.BlockSpec(memory_space=pltpu.HBM),  # swa_accumution
            pl.BlockSpec(memory_space=pltpu.HBM),  # swa_l
            pl.BlockSpec(memory_space=pltpu.HBM),  # swa_m
            pl.BlockSpec(
                memory_space=pltpu.HBM) if is_csa else None,  # topk_indices
        ]

        out_specs = pl.BlockSpec(memory_space=pltpu.HBM)  # o

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

        swa_acc_double_buf = pltpu.VMEM(
            (2, bq_sz, num_q_heads, head_dim),
            q.dtype,
        )

        if topk_indices is not None:
            csa_topk = topk_indices.shape[1]
            topk_indices_double_buf = pltpu.VMEM(
                (2, bq_sz, csa_topk),
                jnp.int32,
            )
        else:
            topk_indices_double_buf = None

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
            swa_acc_double_buf,  # Buffer for swa_accumution.
            topk_indices_double_buf,  # Buffer for topk_indices.
            # Semaphores for double buffering of bkv, bq, bo, swa_acc, swa_l, swa_m, topk.
            pltpu.SemaphoreType.DMA((7, 2)),
            # Intermediate buffers per kv head for flash attention.
            l_scratch,
            m_scratch,
            acc_scratch,
        ]

        scalar_prefetches = (
            kv_lens,
            kv_lens_to_attend,
            page_indices,
            cu_q_lens,
            jnp.array([start_seq_idx, end_seq_idx], jnp.int32),
            # (bq_sem_idx, bkv_sem_idx, bo_sem_idx)
            jnp.zeros((3, ), jnp.int32),
            # (bo_sem_0_seq_idx, bo_sem_1_seq_idx, bo_sem_0_bo_idx, bo_sem_1_bo_idx)
            jnp.full((4, ), -1, jnp.int32),
        )

        scope_name = f"MLA-{case.symbol}-bq_{bq_sz}-bkvp_{bkv_p}-p_{page_size}"
        kernel = jax.named_scope(scope_name)(
            pl.pallas_call(
                functools.partial(
                    _mla_ragged_paged_attention_kernel,
                    sm_scale=sm_scale,
                    mask_value=mask_value,
                    static_q_len=static_q_len,
                    bq_sz=bq_sz,
                    bkv_p=bkv_p,
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
                out_shape=jax.ShapeDtypeStruct(shape=q.shape, dtype=q.dtype),
                input_output_aliases={
                    7 if kv_lens_to_attend is None else 8:
                    0,  # Alias output activation with q
                },
                name=scope_name,
            ))
        return kernel(
            *scalar_prefetches,
            attention_sinks,
            q,
            cache_kv,
            swa_accumution,
            swa_l,
            swa_m,
            topk_indices,
        )

    # Decode-only
    q = run_mla_kernel(
        q,
        cache_kv,
        kv_lens,
        kv_lens_to_attend,
        topk_indices,
        page_indices,
        cu_q_lens,
        attention_sinks,
        swa_accumution,
        swa_l,
        swa_m,
        num_kv_pages_per_block=num_kv_pages_per_blocks[0],
        num_queries_per_block=num_queries_per_blocks[0],
        start_seq_idx=jnp.array(0),
        end_seq_idx=distribution[0],
        static_q_len=1,
        case=MlaCase.DECODE,
    )

    if chunk_prefill_size is not None:
        # Handle prefill where the query length is fixed per sequence.
        q = run_mla_kernel(
            q,
            cache_kv,
            kv_lens,
            kv_lens_to_attend,
            topk_indices,
            page_indices,
            cu_q_lens,
            attention_sinks,
            swa_accumution,
            swa_l,
            swa_m,
            num_kv_pages_per_block=num_kv_pages_per_blocks[1],
            num_queries_per_block=num_queries_per_blocks[1],
            start_seq_idx=distribution[0],
            end_seq_idx=distribution[1],
            static_q_len=chunk_prefill_size,
            case=MlaCase.PREFILL,
        )

    # Handle mixed case where the query length per sequence is variable.
    q = run_mla_kernel(
        q,
        cache_kv,
        kv_lens,
        kv_lens_to_attend,
        topk_indices,
        page_indices,
        cu_q_lens,
        attention_sinks,
        swa_accumution,
        swa_l,
        swa_m,
        num_kv_pages_per_block=num_kv_pages_per_blocks[2],
        num_queries_per_block=num_queries_per_blocks[2],
        start_seq_idx=distribution[1],
        end_seq_idx=distribution[2],
        static_q_len=None,
        case=MlaCase.MIXED,
    )
    output = prepare_outputs(
        q, actual_num_q_heads, actual_head_dim
    )  # [max_num_tokens, actual_num_q_heads, actual_head_dim]

    return output
