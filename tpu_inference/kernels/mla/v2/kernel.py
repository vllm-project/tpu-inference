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
import math
from enum import Enum

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from tpu_inference.kernels.mla.v2.transpose import xpose_pipeline
from tpu_inference.logger import init_logger

logger = init_logger(__name__)


def cdiv_on_kv_packing(a, kv_packing):
    assert kv_packing == 1 or kv_packing == 2 or kv_packing == 4
    exponent = int(math.log2(kv_packing))
    # Use bit shift instead of division for efficiency.
    return (a + kv_packing - 1) >> exponent


def floor_div_on_kv_packing(a, kv_packing):
    assert kv_packing == 1 or kv_packing == 2 or kv_packing == 4
    exponent = int(math.log2(kv_packing))
    # Use bit shift instead of division for efficiency.
    return a >> exponent


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
    BATCHED_DECODE = 3

    @property
    def symbol(self):
        return {
            MlaCase.DECODE: "d",
            MlaCase.PREFILL: "p",
            MlaCase.MIXED: "m",
            MlaCase.BATCHED_DECODE: "bd",
        }[self]


# Expect to run this validation during compile time.
def static_validate_inputs(
    ql_nope: jax.Array,  # [actual_num_q_heads, max_num_tokens, actual_lkv_dim]
    q_pe: jax.Array,  # [max_num_tokens, actual_num_q_heads, actual_r_dim]
    new_kv_c: jax.Array,  # [max_num_tokens, actual_lkv_dim]
    new_k_pe: jax.Array,  # [max_num_tokens, actual_r_dim]
    cache_kv: jax.
    Array,  # [total_num_pages, page_size_per_kv_packing, kv_packing, lkv_dim]
    kv_lens: jax.Array,  # i32[max_num_seqs]
    page_indices: jax.Array,  # i32[max_num_seqs * pages_per_seq]
    cu_q_lens: jax.Array,  # i32[max_num_seqs + 1]
    distribution: jax.Array,  # i32[3]
    *,
    sm_scale: float = 1.0,
    sliding_window: int | None = None,
    soft_cap: float | None = None,
    mask_value: float | None = None,
    q_scale: float | None = None,
    k_scale: float | None = None,
    v_scale: float | None = None,
    # Kernel optimization params.
    chunk_prefill_size: int | None = None,
    # Kernel tuning params.
    num_kv_pages_per_blocks: tuple[int, int, int] | None = None,
    num_queries_per_blocks: tuple[int, int, int] | None = None,
    vmem_limit_bytes: int | None = None,
    decode_batch_size: int = 1,
    # Debug params.
    debug_mode: bool = False,
):
    """Validate inputs to the MLA RPA kernel statically."""
    if len(ql_nope.shape) != 3:
        raise ValueError(f"Expected 3D array for {ql_nope.shape=}")
    if len(q_pe.shape) != 3:
        raise ValueError(f"Expected 3D array for {q_pe.shape=}")
    if len(new_kv_c.shape) != 2:
        raise ValueError(f"Expected 2D array for {new_kv_c.shape=}")
    if len(new_k_pe.shape) != 2:
        raise ValueError(f"Expected 2D array for {new_k_pe.shape=}")

    if ql_nope.shape[0] != q_pe.shape[1]:
        raise ValueError(
            f"Expected ql_nope num_heads {ql_nope.shape[0]=} to equal q_pe num_heads {q_pe.shape[1]=}"
        )
    if ql_nope.shape[1] != q_pe.shape[0]:
        raise ValueError(
            f"Expected ql_nope num_tokens {ql_nope.shape[1]=} to equal q_pe num_tokens {q_pe.shape[0]=}"
        )
    if ql_nope.shape[1] != new_kv_c.shape[0]:
        raise ValueError(
            f"Expected {ql_nope.shape[1]=} to be equal to {new_kv_c.shape[0]=}"
        )
    if new_kv_c.shape[0] != new_k_pe.shape[0]:
        raise ValueError(
            f"Expected {new_kv_c.shape[0]=} to be equal to {new_k_pe.shape[0]=}"
        )
    if ql_nope.shape[2] != new_kv_c.shape[1]:
        raise ValueError(
            f"Expected {ql_nope.shape[2]=} to be equal to {new_kv_c.shape[1]=}"
        )
    if q_pe.shape[2] != new_k_pe.shape[1]:
        raise ValueError(
            f"Expected {q_pe.shape[2]=} to be equal to {new_k_pe.shape[1]=}")

    actual_lkv_dim = ql_nope.shape[-1]
    actual_r_dim = q_pe.shape[-1]
    lkv_dim = align_to(actual_lkv_dim, 128)
    r_dim = align_to(actual_r_dim, 128)

    (
        _,
        page_size_per_kv_packing,
        kv_packing,
        kv_dim,
    ) = cache_kv.shape

    if lkv_dim + r_dim != kv_dim:
        raise ValueError(
            f"Expected {lkv_dim=} + {r_dim=} to be equal to {kv_dim=}")

    if not (cache_kv.dtype == new_kv_c.dtype):
        raise ValueError(
            f"Expected {cache_kv.dtype=} to be equal to {new_kv_c.dtype=}.")
    if not (cache_kv.dtype == new_k_pe.dtype):
        raise ValueError(
            f"Expected {cache_kv.dtype=} to be equal to {new_k_pe.dtype=}.")

    # Integer kv quantization is currently not supported.
    if not jnp.issubdtype(cache_kv.dtype, jnp.floating):
        raise ValueError(f"Expected {cache_kv.dtype=} to be a floating point.")

    if kv_packing != get_dtype_packing(cache_kv.dtype):
        raise ValueError(
            f"{kv_packing=} does not match with {cache_kv.dtype=}")

    if not (jnp.int32 == kv_lens.dtype == page_indices.dtype == cu_q_lens.dtype
            == distribution.dtype):
        raise ValueError(
            f"Expected int32 dtype for {kv_lens.dtype=}, {page_indices.dtype=},"
            f" {cu_q_lens.dtype=}, {distribution.dtype=}")

    if not (len(kv_lens.shape) == len(page_indices.shape) == len(
            cu_q_lens.shape) == 1):
        raise ValueError(
            f"Expected 1D array for {kv_lens.shape=}, {page_indices.shape=},"
            f" {cu_q_lens.shape=}")

    max_num_seqs = kv_lens.shape[0]
    num_page_indices = page_indices.shape[0]
    if num_page_indices % max_num_seqs != 0:
        raise ValueError(
            f"Expected {num_page_indices=} to be divisible by {max_num_seqs=}."
        )
    if cu_q_lens.shape != (max_num_seqs + 1, ):
        raise ValueError(
            f"Expected {cu_q_lens.shape=} to be ({max_num_seqs + 1},).")
    if distribution.shape != (3, ):
        raise ValueError(f"Expected {distribution.shape=} to be (3,).")

    page_size = page_size_per_kv_packing * kv_packing
    if page_size % kv_packing != 0:
        raise ValueError(f"{page_size=} must be divisible by {kv_packing=}.")
    if sliding_window is not None and sliding_window <= 0:
        raise ValueError(f"{sliding_window=} must be positive.")
    if soft_cap is not None and soft_cap == 0.0:
        raise ValueError(f"{soft_cap=} must not be 0.0.")
    if chunk_prefill_size is not None and chunk_prefill_size <= 0:
        raise ValueError(f"{chunk_prefill_size=} must be positive.")
    if num_kv_pages_per_blocks is not None:
        for num_kv_pages_per_block in num_kv_pages_per_blocks:
            if num_kv_pages_per_block <= 0:
                raise ValueError(
                    f"{num_kv_pages_per_block=} must be positive.")
    if num_queries_per_blocks is not None:
        for num_queries_per_block in num_queries_per_blocks:
            if num_queries_per_block <= 0:
                raise ValueError(f"{num_queries_per_block=} must be positive.")
    if vmem_limit_bytes is not None and vmem_limit_bytes <= 0:
        raise ValueError(f"{vmem_limit_bytes=} must be positive.")

    # No constraints for the following inputs.
    del sm_scale
    del mask_value
    del q_scale
    del k_scale
    del v_scale
    del decode_batch_size
    del debug_mode


def _mla_ragged_paged_attention_kernel(
    # Prefetch
    kv_lens_ref,  # [max_num_seqs]
    page_indices_ref,  # [max_num_seqs * pages_per_seq]
    cu_q_lens_ref,  # [max_num_seqs + 1]
    start_end_seq_idx_ref,  # [2] (start_seq_idx, end_seq_idx)
    sem_ids_ref,  # [3] (bq_sem_idx, bkv_sem_idx, bo_sem_idx)
    bo_ids_ref,  # [4] (bo_sem_0_seq_idx, bo_sem_1_seq_idx, bo_sem_0_bo_idx, bo_sem_1_bo_idx)
    bkv_update_ids_ref,  # [batch_size, 6] (bkv_sem_0_seq_idx, bkv_sem_1_seq_idx, bkv_sem_0_offset, bkv_sem_1_offset, bkv_sem_0_sz, bkv_sem_1_sz) * batch_size
    # Input
    ql_nope_hbm_ref,  # [max_num_tokens, num_q_heads, lkv_dim]
    q_pe_hbm_ref,  # [max_num_tokens, num_q_heads, r_dim]
    new_kv_c_hbm_ref,  # [max_num_tokens_per_kv_packing, kv_packing, lkv_dim]
    new_k_pe_hbm_ref,  # [max_num_tokens_per_kv_packing, kv_packing, r_dim]
    cache_kv_hbm_ref,  # [total_num_pages, page_size_per_kv_packing, kv_packing, align_to(lkv_dim + r_dim, 128)]
    # Output
    o_hbm_ref,  # [max_num_tokens, num_q_heads, lkv_dim]
    updated_cache_kv_hbm_ref,  # [total_num_pages, page_size_per_kv_packing, kv_packing, align_to(lkv_dim + r_dim, 128)]
    # Scratch
    bkvc_x2_ref,  # [2, batch_size, bkv_buf_sz_per_kv_packing, kv_packing, lkv_dim]
    bkpe_x2_ref,  # [2, batch_size, bkv_buf_sz_per_kv_packing, kv_packing, r_dim]
    bq_nope_x2_ref,  # [2, batch_size, bq_sz, num_q_heads, lkv_dim]
    bq_rope_x2_ref,  # [2, batch_size, bq_sz, num_q_heads, r_dim]
    bo_x2_ref,  # [2, batch_size, bq_sz, num_q_heads, lkv_dim]
    sems,  # [4, batch_size, 2]
    l_ref,  # [batch_size, bq_sz * num_q_heads, 128],
    m_ref,  # [batch_size, bq_sz * num_q_heads, 128],
    acc_ref,  # [batch_size, bq_sz * num_q_heads, lkv_dim],
    *,
    static_q_len: int,
    sm_scale: float,
    mask_value: float,
    s_dtype: jnp.dtype,
    p_same_dtype_as_v: bool,
    sliding_window: int | None = None,
    soft_cap: float | None = None,
    q_scale: float | None = None,
    k_scale: float | None = None,
    v_scale: float | None = None,
    bkv_p,
    bq_sz,
    batch_size: int = 1,
    debug_mode: bool = False,
):
    assert ql_nope_hbm_ref.shape == o_hbm_ref.shape
    # Validation checks on the dimensions
    nope_dim = ql_nope_hbm_ref.shape[-1]
    pe_dim = q_pe_hbm_ref.shape[-1]
    assert nope_dim + pe_dim == cache_kv_hbm_ref.shape[-1]

    _, num_q_heads, lkv_dim = ql_nope_hbm_ref.shape
    r_dim = q_pe_hbm_ref.shape[-1]
    q_packing = get_dtype_packing(ql_nope_hbm_ref.dtype)
    num_q_heads_per_q_packing = num_q_heads // q_packing
    total_num_pages, page_size_per_kv_packing, kv_packing, _ = (
        cache_kv_hbm_ref.shape)
    max_num_seqs = kv_lens_ref.shape[0]
    num_page_indices = page_indices_ref.shape[0]

    assert num_page_indices % max_num_seqs == 0
    pages_per_seq = num_page_indices // max_num_seqs
    q_dtype = ql_nope_hbm_ref.dtype
    # Validate against the KV dtype.
    kv_dtype = cache_kv_hbm_ref.dtype
    assert q_pe_hbm_ref.dtype == q_dtype
    assert o_hbm_ref.dtype == q_dtype
    assert get_dtype_packing(q_dtype) == q_packing
    assert get_dtype_packing(kv_dtype) == kv_packing
    assert lkv_dim % 128 == 0
    assert r_dim % 128 == 0
    bkv_sz_per_kv_packing = bkv_p * page_size_per_kv_packing
    bkv_sz = bkv_sz_per_kv_packing * kv_packing
    page_size = page_size_per_kv_packing * kv_packing

    start_seq_idx = start_end_seq_idx_ref[0]
    end_seq_idx = start_end_seq_idx_ref[1]
    batch_start_seq_idx = start_seq_idx + pl.program_id(0) * batch_size
    batch_end_seq_idx = batch_start_seq_idx + batch_size - 1

    def debug_print(msg, *args):
        if debug_mode:
            pl.debug_print(msg, *args)

    debug_print("[RPA debug] ======= In loop batch_start_seq_idx={}",
                batch_start_seq_idx)
    debug_print("[RPA debug] start_seq_idx={}", start_seq_idx)
    debug_print("[RPA debug] end_seq_idx={}", end_seq_idx)
    debug_print("[RPA debug] bkv_p={}", bkv_p)
    debug_print("[RPA debug] page_size={}", page_size)
    debug_print("[RPA debug] pages_per_seq={}", pages_per_seq)
    debug_print("[RPA debug] bkv_sz_per_kv_packing={}", bkv_sz_per_kv_packing)
    debug_print("[RPA debug] bq_sz={}", bq_sz)
    debug_print("[RPA debug] batch_size={}", batch_size)

    def flash_attention_step1_qk_softmax(
        ql_nope,  # [actual_bq_sz * num_q_heads, lkv_dim]
        q_pe,  # [actual_bq_sz * num_q_heads, r_dim]
        kv_c,  # [bkv_sz, lkv_dim] <- Correspond to data from bkvc_x2_ref
        k_pe,  # [bkv_sz, r_dim] <- Correspond to data from bpe_x2_ref
        *,
        kv_len,
        q_len,
        bq_idx,
        bkv_idx,
        head_l_ref,
        head_m_ref,
    ):
        assert len(ql_nope.shape) == 2
        assert len(q_pe.shape) == 2
        assert len(kv_c.shape) == 2
        assert len(k_pe.shape) == 2
        assert ql_nope.shape[0] % num_q_heads == 0
        assert ql_nope.shape[0] == q_pe.shape[0]
        assert q_pe.shape[0] % bq_sz == 0
        assert ql_nope.shape[1] == lkv_dim
        assert q_pe.shape[1] == r_dim
        assert kv_c.shape == (bkv_sz, lkv_dim)
        assert k_pe.shape == (bkv_sz, r_dim)

        def load_with_init(ref, init_val):
            return jnp.where(bkv_idx == 0, jnp.full_like(ref, init_val),
                             ref[...])

        # Follow FlashAttention-2 forward pass.
        q = jnp.concatenate([ql_nope, q_pe], axis=-1)
        k = jnp.concatenate([kv_c, k_pe], axis=-1)
        s = jnp.einsum("nd,md->nm", q, k, preferred_element_type=jnp.float32)
        s *= sm_scale
        if k_scale is not None:
            s *= k_scale
        if q_scale is not None:
            s *= q_scale

        q_span = (kv_len - q_len + bq_idx * bq_sz +
                  lax.broadcasted_iota(jnp.int32, s.shape, 0) // num_q_heads)
        k_span = bkv_idx * bkv_sz + lax.broadcasted_iota(jnp.int32, s.shape, 1)
        mask = q_span < k_span

        if sliding_window is not None:
            mask = jnp.logical_or(mask, q_span - sliding_window >= k_span)

        if soft_cap is not None:
            s = soft_cap * jnp.tanh(s / soft_cap)
        s = s.astype(s_dtype)

        s = jnp.where(mask, mask_value, s)
        s_rowmax = jnp.max(s, axis=1, keepdims=True)
        m_prev = load_with_init(head_m_ref, -jnp.inf)
        m_curr = jnp.maximum(m_prev, s_rowmax)
        head_m_ref[...] = m_curr
        p = jnp.exp(s - broadcast_minor(m_curr, s.shape))

        exp_m_diff = jnp.exp(m_prev - m_curr)
        l_prev = load_with_init(head_l_ref, 0.0)
        p_rowsum = jnp.sum(p, axis=1, keepdims=True)
        l_curr = exp_m_diff * l_prev + p_rowsum
        head_l_ref[...] = l_curr

        return p, exp_m_diff

    def flash_attention_step2_pv(
        p,
        v,
        exp_m_diff,
        bkv_idx,
        head_acc_ref,
    ):

        def load_with_init(ref, init_val):
            return jnp.where(bkv_idx == 0, jnp.full_like(ref, init_val),
                             ref[...])

        if p_same_dtype_as_v:
            p = p.astype(v.dtype)
        pv = jnp.einsum("nm,md->nd", p, v, preferred_element_type=jnp.float32)

        if v_scale is not None:
            pv *= v_scale

        o_prev = load_with_init(head_acc_ref, 0.0)
        o_curr = broadcast_minor(exp_m_diff, o_prev.shape) * o_prev + pv
        head_acc_ref[...] = o_curr

    def _async_copy(src, dst, sem, wait):
        if debug_mode:
            # Skip DMA if debug mode is enabled.
            return
        cp = pltpu.make_async_copy(src, dst, sem)
        if wait:
            cp.wait()
        else:
            cp.start()

    def _fetch_bkv(batch_start_seq_idx, bkv_idx, bkv_sem_idx, *, wait=False):
        if not wait:
            # Make sure the current bkv buffer is safe to overwrite.
            wait_update_kv_cache(bkv_sem_idx)

        offsets = []
        update_szs = []
        for b in range(batch_size):
            sem = sems.at[0, b, bkv_sem_idx]
            # bkvc_x2_ref shape: [2, batch_size, bkv_sz_per_kv_packing + 2, kv_packing, lkv_dim]
            bkvc_vmem_ref = bkvc_x2_ref.at[bkv_sem_idx, b]
            bkvpe_vmem_ref = bkpe_x2_ref.at[bkv_sem_idx, b]

            # [total_num_pages, page_size_per_kv_packing, kv_packing, align_to(lkv_dim + r_dim, 128)]
            # [total_num_pages * page_size_per_kv_packing, kv_packing, align_to(lkv_dim + r_dim, 128)]
            reshaped_cache_hbm_ref = cache_kv_hbm_ref.reshape(
                total_num_pages * page_size_per_kv_packing,
                *cache_kv_hbm_ref.shape[2:],
            )

            seq_idx = batch_start_seq_idx + b
            kv_len = kv_lens_ref[seq_idx]
            kv_len_start = bkv_idx * bkv_sz
            kv_p_start = bkv_idx * bkv_p

            q_start = cu_q_lens_ref[seq_idx]
            q_end = cu_q_lens_ref[seq_idx + 1]
            q_len = q_end - q_start

            kv_left = jnp.maximum(kv_len - kv_len_start, 0)
            kv_left_frm_cache = jnp.maximum(kv_left - q_len, 0)
            kv_left_frm_cache_per_kv_packing = cdiv_on_kv_packing(
                kv_left_frm_cache, kv_packing)
            kv_left_frm_new = jnp.maximum(kv_left - kv_left_frm_cache, 0)

            bkv_sz_frm_cache = jnp.minimum(kv_left_frm_cache, bkv_sz)
            bkv_sz_frm_new = jnp.minimum(bkv_sz - bkv_sz_frm_cache,
                                         kv_left_frm_new)
            bkv_sz_frm_cache_per_kv_packing = cdiv_on_kv_packing(
                bkv_sz_frm_cache, kv_packing)
            bkv_sz_frm_new_per_kv_packing = cdiv_on_kv_packing(
                bkv_sz_frm_new, kv_packing)
            page_indices_offset = seq_idx * pages_per_seq + kv_p_start

            new_kv_len_start = q_end - kv_left_frm_new
            new_kv_len_start_per_kv_packing = floor_div_on_kv_packing(
                new_kv_len_start, kv_packing)
            bkv_sz_frm_new_kv_packing_to_fetch = jnp.where(
                bkv_sz_frm_new > 0,
                cdiv_on_kv_packing(new_kv_len_start + bkv_sz_frm_new,
                                   kv_packing) -
                new_kv_len_start_per_kv_packing,
                0,
            )
            dma_bkv_sz = (bkv_sz_frm_cache_per_kv_packing +
                          bkv_sz_frm_new_kv_packing_to_fetch)

            debug_print(
                "[RPA debug]"
                f" -----------{'wait' if wait else 'start'}_fetch_bkv-----------"
            )
            debug_print("[RPA debug] seq_idx={}", seq_idx)
            debug_print("[RPA debug] bkv_idx={}", bkv_idx)
            debug_print("[RPA debug] bkv_sem_idx={}", bkv_sem_idx)
            debug_print("[RPA debug] kv_len_start={}", kv_len_start)
            debug_print("[RPA debug] kv_p_start={}", kv_p_start)
            debug_print("[RPA debug] kv_left={}", kv_left)
            debug_print("[RPA debug] kv_left_frm_cache={}", kv_left_frm_cache)
            debug_print("[RPA debug] kv_left_frm_new={}", kv_left_frm_new)
            debug_print("[RPA debug] bkv_sz_frm_cache={}", bkv_sz_frm_cache)
            debug_print(
                "[RPA debug] bkv_sz_frm_cache_per_kv_packing={}",
                bkv_sz_frm_cache_per_kv_packing,
            )
            debug_print(
                "[RPA debug] bkv_sz_frm_new_per_kv_packing={}",
                bkv_sz_frm_new_per_kv_packing,
            )
            debug_print("[RPA debug] page_indices_offset={}",
                        page_indices_offset)
            debug_print(
                f"[RPA debug] bkvc_vmem_ref.shape: {bkvc_vmem_ref.shape}")
            debug_print(
                f"[RPA debug] bkvpe_vmem_ref.shape: {bkvpe_vmem_ref.shape}")

            if not wait:

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
                                page_size_per_kv_packing,
                                sz_per_kv_packing,
                            ),
                            ...,
                            :nope_dim,
                        ],
                        # [bkv_sz_per_kv_packing + 2, kv_packing, lkv_dim].
                        bkvc_vmem_ref.at[pl.ds(i * page_size_per_kv_packing,
                                               sz_per_kv_packing)],
                        sem,
                        wait,
                    )
                    _async_copy(
                        reshaped_cache_hbm_ref.at[
                            pl.ds(
                                page_indices_ref[page_idx] *
                                page_size_per_kv_packing,
                                sz_per_kv_packing,
                            ),
                            ...,
                            nope_dim:,
                        ],
                        # [bkv_sz_per_kv_packing + 2, kv_packing, r_dim].
                        bkvpe_vmem_ref.at[pl.ds(i * page_size_per_kv_packing,
                                                sz_per_kv_packing)],
                        sem,
                        wait,
                    )
                    debug_print(
                        "[RPA debug] loop_body bkv_p={}, i={}, page_size_per_kv_packing={},"
                        " sz_per_kv_packing={}, page_idx={}, page_indices_ref[page_idx]={}",
                        bkv_p,
                        i,
                        page_size_per_kv_packing,
                        sz_per_kv_packing,
                        page_idx,
                        page_indices_ref[page_idx],
                    )

                # Fetch new KVs by appending to the existing vmem buffers.
                # Fetch either up to the end of the buffer or kv_left_frm_new, whichever
                # is smaller. Since DMAs are word-aligned based on kv_packing, and the
                # boundary between the old cache and the new KV tokens might not be
                # word-aligned, we append the new KV words right after the last word
                # containing old cache data. This can create "holes" (misalignments
                # within the words), which we will shift and pack correctly later.
                debug_print("[RPA debug] new_kv_len_start={}",
                            new_kv_len_start)
                debug_print(
                    "[RPA debug] new_kv_len_start_per_kv_packing={}",
                    new_kv_len_start_per_kv_packing,
                )
                debug_print(
                    f"new_kv_c_hbm_ref.shape: {new_kv_c_hbm_ref.shape}")
                debug_print(
                    f"new_k_pe_hbm_ref.shape: {new_k_pe_hbm_ref.shape}")
                _async_copy(
                    new_kv_c_hbm_ref.at[pl.ds(
                        new_kv_len_start_per_kv_packing,
                        bkv_sz_frm_new_kv_packing_to_fetch,
                    )],
                    bkvc_vmem_ref.at[pl.ds(
                        bkv_sz_frm_cache_per_kv_packing,
                        bkv_sz_frm_new_kv_packing_to_fetch,
                    )],
                    sem,
                    wait,
                )
                _async_copy(
                    new_k_pe_hbm_ref.at[pl.ds(
                        new_kv_len_start_per_kv_packing,
                        bkv_sz_frm_new_kv_packing_to_fetch,
                    )],
                    bkvpe_vmem_ref.at[pl.ds(
                        bkv_sz_frm_cache_per_kv_packing,
                        bkv_sz_frm_new_kv_packing_to_fetch,
                    )],
                    sem,
                    wait,
                )

            else:
                # When we wait, we can use a dummy copy to wait for DMAs to complete where
                # src == dst. However, the dma size must be correct.
                dst_kvc = bkvc_vmem_ref.at[pl.ds(0, dma_bkv_sz)]
                _async_copy(
                    src=dst_kvc,
                    dst=dst_kvc,
                    sem=sem,
                    wait=True,
                )
                dst_kvpe = bkvpe_vmem_ref.at[pl.ds(0, dma_bkv_sz)]
                _async_copy(
                    src=dst_kvpe,
                    dst=dst_kvpe,
                    sem=sem,
                    wait=True,
                )
            # This returns the (offset, size) in units of tokens:
            #   offset: starting token index where the new KV should be stored
            #   size: number of tokens of the new KV, which is 1 in decode.
            offsets.append(kv_len_start + bkv_sz_frm_cache)
            update_szs.append(bkv_sz_frm_new)

        return offsets, update_szs

    def _pack_new_kv(bkv_sem_idx, offsets, update_szs):
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
      offsets: A list of the starting token offset in the KV cache where the
        new KVs begin for a batch.
      update_szs: A list of the number of new tokens to be packed for a batch.
    """
        for b in range(batch_size):
            offset = offsets[b]
            update_sz = update_szs[b]

            @pl.when(update_sz > 0)
            def _update(b=b):
                # shape: [bkv_sz_per_kv_packing + 2, kv_packing, lkv_dim]
                bkvc_vmem_ref = bkvc_x2_ref.at[bkv_sem_idx, b]
                # shape: [bkv_sz_per_kv_packing + 2, kv_packing, r_dim]
                bkvpe_vmem_ref = bkpe_x2_ref.at[bkv_sem_idx, b]

                seq_idx = batch_start_seq_idx + b
                q_end = cu_q_lens_ref[seq_idx + 1]
                kv_len = kv_lens_ref[seq_idx]

                update_kv_packing_iters = cdiv_on_kv_packing(
                    (offset % kv_packing) + update_sz, kv_packing)
                kv_packing_offset = offset % kv_packing
                new_kv_len_start = q_end - kv_len + offset
                new_kv_packing_offset = new_kv_len_start % kv_packing

                token_offset_in_bkv = offset % bkv_sz
                kv_packing_idx = floor_div_on_kv_packing(
                    token_offset_in_bkv, kv_packing)

                # Compute the shift amount for each word in bits
                shift_amount = kv_packing_offset - new_kv_packing_offset
                bits_per_element = get_dtype_bitwidth(bkvc_vmem_ref.dtype)
                shift_bits = bits_per_element * (shift_amount % kv_packing)
                shift_bits = shift_bits.astype(jnp.uint32)

                # Calculate the starting index in the KV buffer corresponding to the new KV
                # to fetch the data from. This index accounts for the potential offset
                # caused by the shift_amount.
                # (-shift_amount) // kv_packing will be:
                #   0 if new_kv_packing_offset <= kv_packing_offset
                #  -1 if new_kv_packing_offset > kv_packing_offset.
                kv_packing_idx_new = cdiv_on_kv_packing(
                    token_offset_in_bkv, kv_packing) + floor_div_on_kv_packing(
                        -shift_amount, kv_packing)
                curr_kvc_reg = bkvc_vmem_ref[kv_packing_idx_new, :, :]
                curr_kpe_reg = bkvpe_vmem_ref[kv_packing_idx_new, :, :]
                next_kvc_reg = bkvc_vmem_ref[kv_packing_idx_new + 1, :, :]
                next_kpe_reg = bkvpe_vmem_ref[kv_packing_idx_new + 1, :, :]

                def merge_loop_body(i, vals):
                    (
                        kv_packing_idx,
                        kv_packing_idx_new,
                        curr_kvc_reg,
                        curr_kpe_reg,
                        next_kvc_reg,
                        next_kpe_reg,
                    ) = vals
                    curr_kvc_reg_u32 = pltpu.bitcast(curr_kvc_reg, jnp.uint32)
                    curr_kpe_reg_u32 = pltpu.bitcast(curr_kpe_reg, jnp.uint32)
                    next_kvc_reg_u32 = pltpu.bitcast(next_kvc_reg, jnp.uint32)
                    next_kpe_reg_u32 = pltpu.bitcast(next_kpe_reg, jnp.uint32)

                    shifted_kvc_u32 = lax.bitwise_or(
                        lax.shift_right_logical(curr_kvc_reg_u32,
                                                32 - shift_bits),
                        lax.shift_left(next_kvc_reg_u32, shift_bits),
                    )
                    shifted_kpe_u32 = lax.bitwise_or(
                        lax.shift_right_logical(curr_kpe_reg_u32,
                                                32 - shift_bits),
                        lax.shift_left(next_kpe_reg_u32, shift_bits),
                    )

                    # If shift_bits is 0, we should use the current word. Otherwise,
                    # shifting by 32 bits would result in shifted_*_u32 becoming
                    # next_*_reg_u32, which is incorrect.
                    rotated_kvc_u32 = lax.select(shift_bits == 0,
                                                 curr_kvc_reg_u32,
                                                 shifted_kvc_u32)
                    rotated_kpe_u32 = lax.select(shift_bits == 0,
                                                 curr_kpe_reg_u32,
                                                 shifted_kpe_u32)

                    next_kvc_reg_shifted = pltpu.bitcast(
                        rotated_kvc_u32, next_kvc_reg.dtype)
                    next_kpe_reg_shifted = pltpu.bitcast(
                        rotated_kpe_u32, next_kpe_reg.dtype)

                    offset_in_word = i * kv_packing + lax.broadcasted_iota(
                        dtype=jnp.int32,
                        shape=[kv_packing, lkv_dim],
                        dimension=0)
                    kvc_mask = jnp.logical_and(
                        offset_in_word >= kv_packing_offset,
                        offset_in_word < kv_packing_offset + update_sz,
                    )
                    updated_kvc_reg = lax.select(
                        kvc_mask,
                        next_kvc_reg_shifted,
                        bkvc_vmem_ref[kv_packing_idx, :, :],
                    )
                    offset_in_word_pe = i * kv_packing + lax.broadcasted_iota(
                        dtype=jnp.int32,
                        shape=[kv_packing, r_dim],
                        dimension=0)
                    kpe_mask = jnp.logical_and(
                        offset_in_word_pe >= kv_packing_offset,
                        offset_in_word_pe < kv_packing_offset + update_sz,
                    )
                    updated_kpe_reg = lax.select(
                        kpe_mask,
                        next_kpe_reg_shifted,
                        bkvpe_vmem_ref[kv_packing_idx, :, :],
                    )

                    # Store back the merged word
                    bkvc_vmem_ref[kv_packing_idx, :, :] = updated_kvc_reg
                    bkvpe_vmem_ref[kv_packing_idx, :, :] = updated_kpe_reg

                    # Move to the next word.
                    kv_packing_idx += 1
                    kv_packing_idx_new += 1
                    curr_kvc_reg = next_kvc_reg
                    curr_kpe_reg = next_kpe_reg
                    next_kvc_reg = bkvc_vmem_ref[kv_packing_idx_new + 1, :, :]
                    next_kpe_reg = bkvpe_vmem_ref[kv_packing_idx_new + 1, :, :]
                    return (
                        kv_packing_idx,
                        kv_packing_idx_new,
                        curr_kvc_reg,
                        curr_kpe_reg,
                        next_kvc_reg,
                        next_kpe_reg,
                    )

                lax.fori_loop(
                    0,
                    update_kv_packing_iters,
                    merge_loop_body,
                    (
                        kv_packing_idx,
                        kv_packing_idx_new,
                        curr_kvc_reg,
                        curr_kpe_reg,
                        next_kvc_reg,
                        next_kpe_reg,
                    ),
                )

    def _update_kv_cache(
        batch_start_seq_idx,
        b,
        bkv_sem_idx,
        offset,  # In units of tokens.
        update_sz,  # In units of tokens.
        *,
        wait=False,
    ):
        seq_idx = batch_start_seq_idx + b
        sem = sems.at[3, b, bkv_sem_idx]
        # shape: [bkv_sz_per_kv_packing + 2, kv_packing, lkv_dim]
        bkvc_vmem_ref = bkvc_x2_ref.at[bkv_sem_idx, b]
        # shape: [bkv_sz_per_kv_packing + 2, kv_packing, r_dim]
        bkvpe_vmem_ref = bkpe_x2_ref.at[bkv_sem_idx, b]

        update_kv_packing_iters = cdiv_on_kv_packing(
            (offset % kv_packing) + update_sz, kv_packing)

        # Expected shape:
        # [total_num_pages, page_size_per_kv_packing, kv_packing,
        # align_to(lkv_dim + r_dim, 128)]
        cache_kv_hbm_shape = updated_cache_kv_hbm_ref.shape
        reshaped_cache_kv_hbm_ref = updated_cache_kv_hbm_ref.reshape(
            cache_kv_hbm_shape[0] * cache_kv_hbm_shape[1],
            *cache_kv_hbm_shape[2:],
        )

        if not wait:
            # Issue DMA copy for the updated parts, page by page.
            kv_p_start = offset // page_size
            kv_p_end = cdiv(offset + update_sz, page_size)
            start_word_in_page = floor_div_on_kv_packing(
                offset % page_size, kv_packing)
            start_word_in_vmem = floor_div_on_kv_packing(
                offset % bkv_sz, kv_packing)
            words_to_transfer = update_kv_packing_iters
            page_indices_offset = seq_idx * pages_per_seq + kv_p_start

            def loop_body(i, states):
                curr_word_in_page, words_to_transfer, curr_word_in_vmem = states
                sz_words = jnp.minimum(
                    page_size_per_kv_packing - curr_word_in_page,
                    words_to_transfer)
                page_idx = page_indices_ref[page_indices_offset + i]

                _async_copy(
                    # bkvc_vmem_ref shape:
                    # [bkv_sz_per_kv_packing+2, kv_packing, lkv_dim]
                    bkvc_vmem_ref.at[pl.ds(curr_word_in_vmem, sz_words)],
                    reshaped_cache_kv_hbm_ref.at[
                        pl.ds(
                            page_idx * page_size_per_kv_packing +
                            curr_word_in_page,
                            sz_words,
                        ),
                        ...,
                        :nope_dim,
                    ],
                    sem,
                    wait=False,
                )
                _async_copy(
                    # bkvpe_vmem_ref shape: [bkv_sz_per_kv_packing+2, kv_packing, r_dim]
                    bkvpe_vmem_ref.at[pl.ds(curr_word_in_vmem, sz_words)],
                    reshaped_cache_kv_hbm_ref.at[
                        pl.ds(
                            page_idx * page_size_per_kv_packing +
                            curr_word_in_page,
                            sz_words,
                        ),
                        ...,
                        nope_dim:,
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
            # bkvc_vmem_ref shape: [bkv_sz_per_kv_packing + 2, kv_packing, lkv_dim]
            dst_kv = bkvc_vmem_ref.at[pl.ds(0, dma_sz_words)]
            _async_copy(
                src=dst_kv,
                dst=dst_kv,
                sem=sem,
                wait=True,
            )
            dst_kv = bkvpe_vmem_ref.at[pl.ds(0, dma_sz_words)]
            _async_copy(
                src=dst_kv,
                dst=dst_kv,
                sem=sem,
                wait=True,
            )

    def _fetch_bq(batch_start_seq_idx, bq_idx, bq_sem_idx, *, wait=False):
        for b in range(batch_size):
            sem = sems.at[1, b, bq_sem_idx]
            bq_nope_vmem_ref = bq_nope_x2_ref.at[bq_sem_idx, b]
            bq_rope_vmem_ref = bq_rope_x2_ref.at[bq_sem_idx, b]

            seq_idx = batch_start_seq_idx + b
            q_len_start = cu_q_lens_ref[seq_idx] + bq_idx * bq_sz
            q_end = cu_q_lens_ref[seq_idx + 1]
            sz = jnp.minimum(bq_sz, q_end - q_len_start)

            debug_print(
                "[RPA debug]"
                f" -----------{'wait' if wait else 'start'}_fetch_bq-----------"
            )
            debug_print("[RPA debug] seq_idx={}", seq_idx)
            debug_print("[RPA debug] bq_idx={}", bq_idx)
            debug_print("[RPA debug] bq_sem_idx={}", bq_sem_idx)
            debug_print("[RPA debug] q_len_start={}", q_len_start)
            debug_print("[RPA debug] q_end={}", q_end)
            debug_print("[RPA debug] sz={}", sz)

            @pl.when(sz > 0)
            def _copy(q_len_start=q_len_start,
                      sz=sz,
                      bq_nope_vmem_ref=bq_nope_vmem_ref,
                      bq_rope_vmem_ref=bq_rope_vmem_ref,
                      sem=sem):
                _async_copy(
                    ql_nope_hbm_ref.at[pl.ds(q_len_start, sz)],
                    bq_nope_vmem_ref.at[pl.ds(0, sz)],
                    sem,
                    wait,
                )

                _async_copy(
                    q_pe_hbm_ref.at[pl.ds(q_len_start, sz)],
                    bq_rope_vmem_ref.at[pl.ds(0, sz)],
                    sem,
                    wait,
                )

    def _send_bo(batch_start_seq_idx, bo_idx, bo_sem_idx, *, wait=False):
        for b in range(batch_size):
            sem = sems.at[2, b, bo_sem_idx]
            vmem_ref = bo_x2_ref.at[bo_sem_idx, b]

            seq_idx = batch_start_seq_idx + b
            q_len_start = cu_q_lens_ref[seq_idx] + bo_idx * bq_sz
            q_end = cu_q_lens_ref[seq_idx + 1]
            sz = jnp.minimum(bq_sz, q_end - q_len_start)

            debug_print(
                "[RPA debug]"
                f" -----------{'wait' if wait else 'start'}_send_bo-----------"
            )
            debug_print("[RPA debug] seq_idx={}", seq_idx)
            debug_print("[RPA debug] bo_idx={}", bo_idx)
            debug_print("[RPA debug] bo_sem_idx={}", bo_sem_idx)
            debug_print("[RPA debug] q_len_start={}", q_len_start)
            debug_print("[RPA debug] q_end={}", q_end)
            debug_print("[RPA debug] sz={}", sz)

            @pl.when(sz > 0)
            def _copy(vmem_ref=vmem_ref,
                      sz=sz,
                      q_len_start=q_len_start,
                      sem=sem):
                _async_copy(
                    vmem_ref.at[pl.ds(0, sz)],
                    o_hbm_ref.at[pl.ds(q_len_start, sz)],
                    sem,
                    wait,
                )

    def start_fetch_bkv(batch_start_seq_idx, bkv_idx, bkv_sem_idx):
        return _fetch_bkv(batch_start_seq_idx, bkv_idx, bkv_sem_idx)

    def wait_fetch_bkv(batch_start_seq_idx, bkv_idx, bkv_sem_idx):
        return _fetch_bkv(batch_start_seq_idx, bkv_idx, bkv_sem_idx, wait=True)

    def start_fetch_bq(batch_start_seq_idx, bq_idx, bq_sem_idx):
        return _fetch_bq(batch_start_seq_idx, bq_idx, bq_sem_idx)

    def wait_fetch_bq(batch_start_seq_idx, bq_idx, bq_sem_idx):
        return _fetch_bq(batch_start_seq_idx, bq_idx, bq_sem_idx, wait=True)

    def start_send_bo(batch_start_seq_idx, bo_idx, bo_sem_idx):
        bo_ids_ref[bo_sem_idx] = batch_start_seq_idx
        bo_ids_ref[bo_sem_idx + 2] = bo_idx
        _send_bo(batch_start_seq_idx, bo_idx, bo_sem_idx)

    def wait_send_bo(bo_sem_idx):
        old_batch_start_seq_idx = bo_ids_ref[bo_sem_idx]
        old_bo_idx = bo_ids_ref[bo_sem_idx + 2]

        @pl.when(
            jnp.logical_and(0 <= old_batch_start_seq_idx,
                            old_batch_start_seq_idx <= batch_start_seq_idx))
        def _():
            _send_bo(old_batch_start_seq_idx,
                     old_bo_idx,
                     bo_sem_idx,
                     wait=True)

    def start_update_kv_cache(start_seq_idx, bkv_sem_idx, offsets, update_szs):
        for b in range(batch_size):
            offset = offsets[b]
            update_sz = update_szs[b]
            bkv_update_ids_ref[b, bkv_sem_idx + 4] = update_sz

            @pl.when(update_sz > 0)
            def _(b=b,
                  update_sz=update_sz,
                  start_seq_idx=start_seq_idx,
                  offset=offset):
                bkv_update_ids_ref[b, bkv_sem_idx] = start_seq_idx
                bkv_update_ids_ref[b, bkv_sem_idx + 2] = offset
                _update_kv_cache(start_seq_idx, b, bkv_sem_idx, offset,
                                 update_sz)

    def wait_update_kv_cache(bkv_sem_idx):
        for b in range(batch_size):
            update_sz = bkv_update_ids_ref[b, bkv_sem_idx + 4]

            @pl.when(update_sz > 0)
            def _(b=b, update_sz=update_sz):
                start_seq_idx = bkv_update_ids_ref[b, bkv_sem_idx]
                offset = bkv_update_ids_ref[b, bkv_sem_idx + 2]
                bkv_update_ids_ref[b, bkv_sem_idx + 4] = 0
                _update_kv_cache(start_seq_idx,
                                 b,
                                 bkv_sem_idx,
                                 offset,
                                 update_sz,
                                 wait=True)

    def load_bq(batch_item_idx, bq_sem_idx, *, actual_bq_sz=bq_sz):
        q_nope_ref = (bq_nope_x2_ref.bitcast(
            jnp.uint32).at[bq_sem_idx, batch_item_idx].reshape(
                bq_sz * num_q_heads_per_q_packing, lkv_dim))
        q_nope_vec = pltpu.bitcast(
            q_nope_ref[:actual_bq_sz * num_q_heads_per_q_packing],
            q_dtype,
        ).reshape(actual_bq_sz * num_q_heads, lkv_dim)
        q_rope_ref = (bq_rope_x2_ref.bitcast(
            jnp.uint32).at[bq_sem_idx, batch_item_idx].reshape(
                bq_sz * num_q_heads_per_q_packing, r_dim))
        q_rope_vec = pltpu.bitcast(
            q_rope_ref[:actual_bq_sz * num_q_heads_per_q_packing],
            q_dtype,
        ).reshape(actual_bq_sz * num_q_heads, r_dim)
        return q_nope_vec, q_rope_vec

    def load_bkv(batch_item_idx, bkv_sem_idx):

        bkvc_ref = (bkvc_x2_ref.bitcast(
            jnp.uint32).at[bkv_sem_idx,
                           batch_item_idx, :bkv_sz_per_kv_packing].reshape(
                               bkv_sz_per_kv_packing, lkv_dim))
        bkvc_vec = pltpu.bitcast(bkvc_ref[...],
                                 kv_dtype).reshape(bkv_sz, lkv_dim)
        bkpe_ref = (bkpe_x2_ref.bitcast(
            jnp.uint32).at[bkv_sem_idx,
                           batch_item_idx, :bkv_sz_per_kv_packing].reshape(
                               bkv_sz_per_kv_packing, r_dim))
        bkpe_vec = pltpu.bitcast(bkpe_ref[...],
                                 kv_dtype).reshape(bkv_sz, r_dim)
        return bkvc_vec, bkpe_vec

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

    kv_len_max = kv_lens_ref[batch_start_seq_idx]
    for b in range(1, batch_size):
        kv_len_max = jnp.where(
            kv_lens_ref[batch_start_seq_idx + b] > kv_len_max,
            kv_lens_ref[batch_start_seq_idx + b], kv_len_max)

    q_len_max = (cu_q_lens_ref[batch_start_seq_idx + 1] -
                 cu_q_lens_ref[batch_start_seq_idx])
    for b in range(1, batch_size):
        q_len_b = (cu_q_lens_ref[batch_start_seq_idx + b + 1] -
                   cu_q_lens_ref[batch_start_seq_idx + b])
        q_len_max = jnp.where(q_len_b > q_len_max, q_len_b, q_len_max)

    def process():
        num_bkv = cdiv(kv_len_max, bkv_sz)
        if static_q_len is None:
            actual_bq_sz = bq_sz
            num_bq = cdiv(q_len_max, actual_bq_sz)
        else:
            actual_bq_sz = min(bq_sz, static_q_len)
            num_bq = cdiv(static_q_len, actual_bq_sz)

        debug_print("[RPA debug] process")
        debug_print("[RPA debug] num_bkv={}", num_bkv)  # num_bkv=3, bkv_sz=512
        debug_print("[RPA debug] bkv_sz={}", bkv_sz)
        debug_print("[RPA debug] num_bq={}", num_bq)
        debug_print("[RPA debug] kv_len_max={}", kv_len_max)
        debug_print("[RPA debug] q_len_max={}", q_len_max)

        def get_next_bq_ids(seq_idx, bq_idx, bq_sem_idx):
            next_bq_idx = bq_idx + 1
            is_last_bq = next_bq_idx == num_bq
            next_bq_idx = lax.select(is_last_bq, 0, next_bq_idx)
            next_seq_idx = lax.select(is_last_bq, seq_idx + batch_size,
                                      seq_idx)
            next_bq_sem_idx = lax.select(bq_sem_idx == 0, 1, 0)
            return next_seq_idx, next_bq_idx, next_bq_sem_idx

        def get_next_bkv_ids(seq_idx, bq_idx, bkv_idx, bkv_sem_idx):
            next_bkv_idx = bkv_idx + 1
            is_last_bkv = next_bkv_idx == num_bkv
            next_bkv_idx = lax.select(is_last_bkv, 0, next_bkv_idx)
            next_bq_idx = lax.select(is_last_bkv, bq_idx + 1, bq_idx)
            is_last_bq = next_bq_idx == num_bq
            next_bq_idx = lax.select(is_last_bq, 0, next_bq_idx)
            next_seq_idx = lax.select(is_last_bq, seq_idx + batch_size,
                                      seq_idx)
            next_bkv_sem_idx = lax.select(bkv_sem_idx == 0, 1, 0)
            return next_seq_idx, next_bq_idx, next_bkv_idx, next_bkv_sem_idx

        def compute_with_bq(bq_idx, _):
            bq_sem_idx = sem_ids_ref[0]
            next_seq_idx, next_bq_idx, next_bq_sem_idx = get_next_bq_ids(
                batch_start_seq_idx, bq_idx, bq_sem_idx)

            # Prefetch next bq
            @pl.when(next_seq_idx < end_seq_idx)
            def prefetch_next_bq():
                sem_ids_ref[0] = next_bq_sem_idx
                start_fetch_bq(next_seq_idx, next_bq_idx, next_bq_sem_idx)

            def compute_with_bkv(bkv_idx, _):

                # Get next bkv ids.
                bkv_sem_idx = sem_ids_ref[1]
                next_seq_idx, _, next_bkv_idx, next_bkv_sem_idx = get_next_bkv_ids(
                    batch_start_seq_idx, bq_idx, bkv_idx, bkv_sem_idx)

                # Prefetch next bkv
                @pl.when(next_seq_idx < end_seq_idx)
                def prefetch_next_bkv():
                    sem_ids_ref[1] = next_bkv_sem_idx
                    start_fetch_bkv(next_seq_idx, next_bkv_idx,
                                    next_bkv_sem_idx)

                # Wait for cur bq if not ready yet
                @pl.when(bkv_idx == 0)
                def wait_cur_bq():
                    wait_fetch_bq(batch_start_seq_idx, bq_idx, bq_sem_idx)

                # Wait for cur bkv
                offsets, update_szs = wait_fetch_bkv(batch_start_seq_idx,
                                                     bkv_idx, bkv_sem_idx)

                # Pack and align new KVs in VMEM if the block has new KVs.
                # We may have to do this for each block of KV in VMEM.
                _pack_new_kv(bkv_sem_idx, offsets, update_szs)

                # Start updating bkv to kv cache if applicable.
                # Only needed in first bq loop.
                @pl.when(bq_idx == 0)
                def update_cur_bkv_to_cache():
                    start_update_kv_cache(batch_start_seq_idx, bkv_sem_idx,
                                          offsets, update_szs)

                # Load bkv into vreg. There is no need to mask out invalid k/v entries,
                # because the score of invalid Q.K^T pairs are masked (to be zero) in
                # flash attention, so that the invalid kv entries
                # (as long as they are not NaN or inf) won't affect to the output.
                prev_p = None
                prev_v = None
                prev_exp_m_diff = None
                for b in range(batch_size):
                    bkvc, bkpe = load_bkv(
                        b,
                        bkv_sem_idx,
                    )
                    bq_nope_vec, bq_pe_vec = load_bq(b,
                                                     bq_sem_idx,
                                                     actual_bq_sz=actual_bq_sz)

                    cur_p, cur_exp_m_diff = flash_attention_step1_qk_softmax(
                        bq_nope_vec,
                        bq_pe_vec,
                        bkvc,
                        bkpe,
                        kv_len=kv_lens_ref[batch_start_seq_idx + b],
                        q_len=(cu_q_lens_ref[batch_start_seq_idx + b + 1] -
                               cu_q_lens_ref[batch_start_seq_idx + b]),
                        bq_idx=bq_idx,
                        bkv_idx=bkv_idx,
                        head_l_ref=l_ref.at[b, :bq_nope_vec.shape[0]],
                        head_m_ref=m_ref.at[b, :bq_nope_vec.shape[0]],
                    )

                    if prev_p is not None:
                        assert prev_v is not None
                        assert prev_exp_m_diff is not None
                        flash_attention_step2_pv(
                            prev_p,
                            prev_v,
                            prev_exp_m_diff,
                            bkv_idx,
                            acc_ref.at[b - 1, :prev_p.shape[0]],
                        )

                    prev_p = cur_p
                    prev_v = bkvc
                    prev_exp_m_diff = cur_exp_m_diff

                # last flash attention 2nd step
                assert prev_p is not None
                assert prev_v is not None
                assert prev_exp_m_diff is not None
                flash_attention_step2_pv(
                    prev_p,
                    prev_v,
                    prev_exp_m_diff,
                    bkv_idx,
                    acc_ref.at[batch_size - 1, :prev_p.shape[0]],
                )

            lax.fori_loop(0, num_bkv, compute_with_bkv, None, unroll=False)

            # Load acc and calculate final output.
            acc = acc_ref[...]
            l = broadcast_minor(l_ref[...], acc.shape)  # noqa
            out = (lax.div(acc, l) if q_dtype == jnp.float32 else
                   (acc * pl.reciprocal(l, approx=True)).astype(q_dtype))

            # Wait for previous bo to be fully sent before storing new bo.
            bo_sem_idx = sem_ids_ref[2]
            sem_ids_ref[2] = lax.select(bo_sem_idx == 0, 1, 0)
            wait_send_bo(bo_sem_idx)

            # Store output from acc to bo.
            bo_x2_ref.at[bo_sem_idx].bitcast(jnp.int32).reshape(
                batch_size,
                bq_sz * num_q_heads_per_q_packing,
                lkv_dim,
            )[...] = pltpu.bitcast(out, jnp.int32)

            # Send cur bo
            start_send_bo(batch_start_seq_idx, bq_idx, bo_sem_idx)

        lax.fori_loop(0, num_bq, compute_with_bq, None, unroll=False)

    ### ------- Kernel start ------- ###

    @pl.when(batch_start_seq_idx == start_seq_idx)
    def prologue():
        start_fetch_bq(start_seq_idx, 0, 0)

        # Initialize bkvc_x2_ref and bkpe_x2_ref to zeros to avoid NaN issues from accessing
        # uninitialized memory. Bitcast into int32 to avoid tiling issues.
        bkvc_x2_int32_ref = bkvc_x2_ref.bitcast(jnp.int32).reshape(
            (2, -1, lkv_dim))
        bkvc_zeros = jnp.zeros(bkvc_x2_int32_ref.shape[1:], jnp.int32)
        bkpe_x2_int32_ref = bkpe_x2_ref.bitcast(jnp.int32).reshape(
            (2, -1, r_dim))
        bkpe_zeros = jnp.zeros(bkpe_x2_int32_ref.shape[1:], jnp.int32)

        # To pipeline VST and DMA, we divide the initialization into two steps.
        bkvc_x2_int32_ref[0] = bkvc_zeros
        bkpe_x2_int32_ref[0] = bkpe_zeros
        start_fetch_bkv(start_seq_idx, 0, 0)
        bkvc_x2_int32_ref[1] = bkvc_zeros
        bkpe_x2_int32_ref[1] = bkpe_zeros

    process()

    @pl.when(batch_end_seq_idx == end_seq_idx - 1)
    def epilogue():
        for i in range(2):
            wait_send_bo(i)
            wait_update_kv_cache(i)

    ### ------- Kernel end ------- ###


def prepare_q_inputs(
        q: jax.Array,  # [max_num_tokens, actual_num_q_heads, actual_head_dim],
):
    _, actual_num_q_heads, actual_head_dim = q.shape
    num_q_heads = align_to(actual_num_q_heads, get_dtype_packing(q.dtype))
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
    return q  # (max_num_tokens, num_q_heads, head_dim)


def prepare_q_nope_inputs(
        q: jax.Array,  # [actual_num_q_heads, max_num_tokens, actual_head_dim]
):
    """Packs and physically transposes q_nope to the layout expected by the MLA kernel.

    The `q_nope` einsum emits in head-major layout (N, T, L). It needs to be transposed
    into (T, N, L) which uses the custom call kernel to absorb the copy latency.

    Returns: [max_num_tokens, num_q_heads, head_dim]
    """
    actual_num_q_heads, _, actual_head_dim = q.shape
    num_q_heads = align_to(actual_num_q_heads, get_dtype_packing(q.dtype))
    head_dim = align_to(actual_head_dim, 128)
    q = jnp.pad(
        q,
        (
            (0, num_q_heads - actual_num_q_heads),
            (0, 0),
            (0, head_dim - actual_head_dim),
        ),
        constant_values=0,
    )
    # Physical transpose: (N, T, D) -> (T, N, D), pipelined over T.
    try:
        q = xpose_pipeline(q, transpose_axes=(1, 0, 2), n_tile=128,
                           m_tile=32)[0]
    except ValueError as e:
        sublane_multiple = get_dtype_packing(q.dtype) * 8
        logger.warning(
            f"xpose_pipeline failed for shape={q.shape} dtype={q.dtype} "
            f"(sublane_multiple={sublane_multiple}): {e}. "
            f"Falling back to jnp.transpose — this may be slower.")
        q = jnp.transpose(q, (1, 0, 2))
    return q  # (max_num_tokens, num_q_heads, head_dim)


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


def prepare_outputs(
    out,  # [max_num_tokens, num_q_heads, head_dim]
    actual_num_q_heads: int,
    actual_head_dim: int,
):
    # Physical transpose: (T, N, D) -> (N, T, D), pipelined over T.
    try:
        # Tile to maximum of 160 (multi host bsz)
        # or nearest clean divisor of the number of tokens.
        out = xpose_pipeline(out,
                             transpose_axes=(1, 0, 2),
                             n_tile=160,
                             m_tile=64)[0]
    except ValueError as e:
        sublane_multiple = get_dtype_packing(out.dtype) * 8
        logger.warning(
            f"xpose_pipeline failed for shape={out.shape} dtype={out.dtype} "
            f"(sublane_multiple={sublane_multiple}): {e}. "
            f"Falling back to jnp.transpose — this may be slower.")
        out = jnp.transpose(out, (1, 0, 2))
    return out[:actual_num_q_heads, :, :actual_head_dim]


@functools.partial(
    jax.jit,
    static_argnames=(
        "sm_scale",
        "sliding_window",
        "soft_cap",
        "mask_value",
        "q_scale",
        "k_scale",
        "v_scale",
        "chunk_prefill_size",
        "num_kv_pages_per_block",
        "num_queries_per_block",
        "vmem_limit_bytes",
        "decode_batch_size",
        "s_dtype",
        "p_same_dtype_as_v",
        "debug_mode",
    ),
    donate_argnames=("cache_kv", ),
)
def mla_ragged_paged_attention(
    ql_nope: jax.Array,  # [max_num_tokens, actual_num_q_heads, actual_lkv_dim]
    q_pe: jax.Array,  # [max_num_tokens, actual_num_q_heads, actual_r_dim]
    new_kv_c: jax.Array,  # [max_num_tokens, actual_lkv_dim]
    new_k_pe: jax.Array,  # [max_num_tokens, actual_r_dim]
    cache_kv: jax.
    Array,  # [total_num_pages, page_size_per_kv_packing, kv_packing, align_to(lkv_dim, 128)]
    kv_lens: jax.Array,  # i32[max_num_seqs]
    page_indices: jax.Array,  # i32[max_num_seqs * pages_per_seq]
    cu_q_lens: jax.Array,  # i32[max_num_seqs + 1]
    distribution: jax.Array,  # i32[3]
    *,
    sm_scale: float = 1.0,
    sliding_window: int | None = None,
    soft_cap: float | None = None,
    mask_value: float | None = None,
    q_scale: float | None = None,
    k_scale: float | None = None,
    v_scale: float | None = None,
    # Kernel optimization params.
    chunk_prefill_size: int | None = None,
    # Kernel tuning params for decode, prefill, and mixed cases.
    # If passsed in as int, all cases are the same.
    num_kv_pages_per_block: tuple[int, int, int] | int | None = None,
    num_queries_per_block: tuple[int, int, int] | int | None = None,
    vmem_limit_bytes: int | None = None,
    decode_batch_size: int = 1,
    s_dtype: jnp.dtype = jnp.bfloat16,
    p_same_dtype_as_v: bool = True,
    # Debug params.
    debug_mode: bool = False,
) -> tuple[
        jax.Array,  # [max_num_tokens, actual_num_q_heads, actual_lkv_dim]
        jax.
        Array,  # [total_num_pages, page_size_per_kv_packing, kv_packing, align_to(lkv_dim, 128) + align_to(r_dim, 128)]
]:
    """MLA Ragged paged attention that supports mixed prefill and decode.

  Args:
    ql_nope: concatenated all sequences' queries.
    q_pe: concatenated all sequences' rope.
    new_kv_c: concatenated all sequences' kv_c values
    new_k_pe: concatenated all sequences' k_pe values
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
    soft_cap: the logit soft cap for the attention.
    mask_value: mask value for causal mask.
    q_scale: the scale for the query.
    k_scale: the scale for the key cache.
    v_scale: the scale for the value cache.
    num_kv_pages_per_block: number of kv pages to be processed in one flash
      attention block in the pallas kernel. This is a tuple of (decode, prefill,
      mixed) cases.
    num_queries_per_block: number of queries to be processed in one flash
      attention block in the pallas kernel. This is a tuple of (decode, prefill,
      mixed) cases.
    vmem_limit_bytes: the vmem limit for the pallas kernel.
    decode_batch_size: the batch size for the decode case.
    s_dtype: the dtype for q.k dot product.
    p_same_dtype_as_v: if true, cast p to the same dtype as v before p @ v.
    debug_mode: if true, RPA does not issue any DMAs or run flash attention but
      print debug info. Need to compile with `--xla_tpu_enable_log_recorder`.

  Returns:
    The output of attention and the updated kv cache.
  """
    if mask_value is None:
        mask_value = jnp.finfo(s_dtype).min

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

    if vmem_limit_bytes is None:
        vmem_limit_bytes = pltpu.get_tpu_info().vmem_capacity_bytes

    static_validate_inputs(
        ql_nope,
        q_pe,
        new_kv_c,
        new_k_pe,
        cache_kv,
        kv_lens,
        page_indices,
        cu_q_lens,
        distribution,
        sm_scale=sm_scale,
        sliding_window=sliding_window,
        soft_cap=soft_cap,
        mask_value=mask_value,
        q_scale=q_scale,
        k_scale=k_scale,
        v_scale=v_scale,
        chunk_prefill_size=chunk_prefill_size,
        num_kv_pages_per_blocks=num_kv_pages_per_blocks,
        num_queries_per_blocks=num_queries_per_blocks,
        vmem_limit_bytes=vmem_limit_bytes,
        decode_batch_size=decode_batch_size,
        debug_mode=debug_mode,
    )
    actual_num_q_heads, _, actual_lkv_dim = ql_nope.shape

    ql_nope = prepare_q_nope_inputs(
        ql_nope)  # [max_num_tokens, num_q_heads, lkv_dim]
    q_pe = prepare_q_inputs(q_pe)  # [max_num_tokens, num_q_heads, r_dim]
    new_kv_c = prepare_kv_inputs(
        new_kv_c)  # [max_num_tokens_per_kv_packing, kv_packing, lkv_dim]
    new_k_pe = prepare_kv_inputs(
        new_k_pe)  # [max_num_tokens_per_kv_packing, kv_packing, r_dim]
    lkv_dim = new_kv_c.shape[-1]
    r_dim = new_k_pe.shape[-1]

    _, page_size_per_kv_packing, kv_packing, _ = cache_kv.shape
    page_size = page_size_per_kv_packing * kv_packing
    _, num_q_heads, _ = ql_nope.shape
    max_num_seqs = kv_lens.shape[0]
    num_page_indices = page_indices.shape[0]
    assert num_page_indices % max_num_seqs == 0

    def run_mla_kernel(
        ql_nope: jax.
        Array,  # [max_num_tokens, actual_num_q_heads, actual_lkv_dim]
        q_pe: jax.Array,  # [max_num_tokens, actual_num_q_heads, actual_r_dim]
        new_kv_c: jax.Array,  # [max_num_tokens, actual_lkv_dim]
        new_k_pe: jax.Array,  # [max_num_tokens, actual_r_dim]
        cache_kv: jax.
        Array,  # [total_num_pages, page_size_per_kv_packing, kv_packing, align_to(lkv_dim, 128)]
        kv_lens: jax.Array,  # i32[max_num_seqs]
        page_indices: jax.Array,  # i32[max_num_seqs * pages_per_seq]
        cu_q_lens: jax.Array,  # i32[max_num_seqs + 1]
        start_seq_idx: jax.Array,  # i32
        end_seq_idx: jax.Array,  # i32
        static_q_len: int | None,
        num_kv_pages_per_block: int,
        num_queries_per_block: int,
        s_dtype: jnp.dtype,
        p_same_dtype_as_v: bool,
        batch_size: int = 1,
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
        grid = ((end_seq_idx - start_seq_idx) // batch_size, )

        in_specs = [
            pl.BlockSpec(memory_space=pltpu.HBM),  # ql_nope
            pl.BlockSpec(memory_space=pltpu.HBM),  # q_pe
            pl.BlockSpec(memory_space=pltpu.HBM),  # new_kv_c
            pl.BlockSpec(memory_space=pltpu.HBM),  # new_k_pe
            pl.BlockSpec(memory_space=pltpu.HBM),  # cache_kv
        ]

        out_specs = [
            pl.BlockSpec(memory_space=pltpu.HBM),  # o
            pl.BlockSpec(memory_space=pltpu.HBM),  # updated_cache_kv
        ]

        bkvc_double_buf = pltpu.VMEM(
            (2, batch_size, bkv_buf_sz_per_kv_packing, kv_packing, lkv_dim),
            cache_kv.dtype,
        )

        bkpe_double_buf = pltpu.VMEM(
            (2, batch_size, bkv_buf_sz_per_kv_packing, kv_packing, r_dim),
            cache_kv.dtype,
        )
        bq_nope_double_buf = pltpu.VMEM(
            (2, batch_size, bq_sz, num_q_heads, lkv_dim),
            ql_nope.dtype,
        )

        bq_rope_double_buf = pltpu.VMEM(
            (2, batch_size, bq_sz, num_q_heads, r_dim),
            q_pe.dtype,
        )

        bo_double_buf = bq_nope_double_buf

        l_scratch = pltpu.VMEM(
            (batch_size, bq_sz * num_q_heads, 128),
            jnp.float32,
        )
        m_scratch = l_scratch

        acc_scratch = pltpu.VMEM(
            (batch_size, bq_sz * num_q_heads, lkv_dim),
            jnp.float32,
        )

        scratch_shapes = [
            bkvc_double_buf,
            bkpe_double_buf,
            bq_nope_double_buf,
            bq_rope_double_buf,
            bo_double_buf,  # Double buffering for output block.
            # Semaphores for double buffering of bkv, bq, bo and bkv_update.
            pltpu.SemaphoreType.DMA((4, batch_size, 2)),
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
            # (bkv_sem_0_seq_idx, bkv_sem_1_seq_idx, bkv_sem_0_offset, bkv_sem_1_offset, bkv_sem_0_sz, bkv_sem_1_sz) * batch_size
            jnp.full((batch_size, 6), -1, jnp.int32),
        )

        scope_name = f"MLA-{case.symbol}-bq_{bq_sz}-bkvp_{bkv_p}-p_{page_size}-bsz_{batch_size}"
        kernel = jax.named_scope(scope_name)(
            pl.pallas_call(
                functools.partial(
                    _mla_ragged_paged_attention_kernel,
                    sm_scale=sm_scale,
                    sliding_window=sliding_window,
                    soft_cap=soft_cap,
                    mask_value=mask_value,
                    q_scale=q_scale,
                    k_scale=k_scale,
                    v_scale=v_scale,
                    static_q_len=static_q_len,
                    bq_sz=bq_sz,
                    bkv_p=bkv_p,
                    batch_size=batch_size,
                    s_dtype=s_dtype,
                    p_same_dtype_as_v=p_same_dtype_as_v,
                    debug_mode=debug_mode,
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
                    jax.ShapeDtypeStruct(shape=ql_nope.shape,
                                         dtype=ql_nope.dtype),
                    jax.ShapeDtypeStruct(shape=cache_kv.shape,
                                         dtype=cache_kv.dtype),
                ],
                input_output_aliases={
                    7: 0,  # Alias output activation with ql_nope
                    11: 1,  # Aliasing cache_kv with updated_cache_kv
                },
                name=scope_name,
            ))
        return kernel(
            *scalar_prefetches,
            ql_nope,
            q_pe,
            new_kv_c,
            new_k_pe,
            cache_kv,
        )

    batch_distribution = (distribution[0] //
                          decode_batch_size) * decode_batch_size
    # Batched decode
    ql_nope, updated_kv = run_mla_kernel(
        ql_nope,
        q_pe,
        new_kv_c,
        new_k_pe,
        cache_kv,
        kv_lens,
        page_indices,
        cu_q_lens,
        num_kv_pages_per_block=num_kv_pages_per_blocks[0],
        num_queries_per_block=num_queries_per_blocks[0],
        start_seq_idx=jnp.array(0),
        end_seq_idx=batch_distribution,
        static_q_len=1,
        batch_size=decode_batch_size,
        s_dtype=s_dtype,
        p_same_dtype_as_v=p_same_dtype_as_v,
        case=MlaCase.BATCHED_DECODE,
    )
    # Decode-only
    ql_nope, updated_kv = run_mla_kernel(
        ql_nope,
        q_pe,
        new_kv_c,
        new_k_pe,
        updated_kv,
        kv_lens,
        page_indices,
        cu_q_lens,
        num_kv_pages_per_block=num_kv_pages_per_blocks[0],
        num_queries_per_block=num_queries_per_blocks[0],
        start_seq_idx=batch_distribution,
        end_seq_idx=distribution[0],
        static_q_len=1,
        batch_size=1,
        s_dtype=s_dtype,
        p_same_dtype_as_v=p_same_dtype_as_v,
        case=MlaCase.DECODE,
    )

    if chunk_prefill_size is not None:
        # Handle prefill where the query length is fixed per sequence.
        ql_nope, updated_kv = run_mla_kernel(
            ql_nope,
            q_pe,
            new_kv_c,
            new_k_pe,
            updated_kv,
            kv_lens,
            page_indices,
            cu_q_lens,
            num_kv_pages_per_block=num_kv_pages_per_blocks[1],
            num_queries_per_block=num_queries_per_blocks[1],
            start_seq_idx=distribution[0],
            end_seq_idx=distribution[1],
            static_q_len=chunk_prefill_size,
            batch_size=1,
            s_dtype=s_dtype,
            p_same_dtype_as_v=p_same_dtype_as_v,
            case=MlaCase.PREFILL,
        )

    # Handle mixed case where the query length per sequence is variable.
    ql_nope, updated_kv = run_mla_kernel(
        ql_nope,
        q_pe,
        new_kv_c,
        new_k_pe,
        updated_kv,
        kv_lens,
        page_indices,
        cu_q_lens,
        num_kv_pages_per_block=num_kv_pages_per_blocks[2],
        num_queries_per_block=num_queries_per_blocks[2],
        start_seq_idx=distribution[1],
        end_seq_idx=distribution[2],
        static_q_len=None,
        batch_size=1,
        s_dtype=s_dtype,
        p_same_dtype_as_v=p_same_dtype_as_v,
        case=MlaCase.MIXED,
    )
    output = prepare_outputs(
        ql_nope, actual_num_q_heads,
        actual_lkv_dim)  # [max_num_tokens, actual_num_q_heads, actual_lkv_dim]

    return output, updated_kv
