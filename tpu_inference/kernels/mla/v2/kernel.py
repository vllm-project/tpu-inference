# Copyright 2025 Google LLC
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


# Expect to run this validation during compile time.
def static_validate_inputs(
    ql_nope: jax.Array,  # [max_num_tokens, actual_num_q_heads, actual_lkv_dim]
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
    mask_value: float | None = DEFAULT_MASK_VALUE,
    q_scale: float | None = None,
    k_scale: float | None = None,
    v_scale: float | None = None,
    # Kernel optimization params.
    chunk_prefill_size: int | None = None,
    # Kernel tuning params.
    num_kv_pages_per_block: int | None = None,
    num_queries_per_block: int | None = None,
    vmem_limit_bytes: int | None = None,
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

    if ql_nope.shape[:2] != q_pe.shape[:2]:
        raise ValueError(
            f"Expected {ql_nope.shape[:2]=} to be equal to {q_pe.shape[:2]=}")
    if ql_nope.shape[0] != new_kv_c.shape[0]:
        raise ValueError(
            f"Expected {ql_nope.shape[0]=} to be equal to {new_kv_c.shape[0]=}"
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

    actual_lkv_dim = ql_nope.shape[2]
    actual_r_dim = q_pe.shape[2]
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
    if num_kv_pages_per_block is not None:
        if num_kv_pages_per_block <= 0:
            raise ValueError(f"{num_kv_pages_per_block=} must be positive.")
    if num_queries_per_block is not None:
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
    del debug_mode


def _mla_ragged_paged_attention_kernel(
    # Prefetch
    kv_lens_ref,  # [max_num_seqs]
    page_indices_ref,  # [max_num_seqs * pages_per_seq]
    cu_q_lens_ref,  # [max_num_seqs + 1]
    distribution_ref,  # [3] (decode_end, prefill_end, mixed_end)
    sem_ids_ref,  # [3] (bq_sem_idx, bkv_sem_idx, bo_sem_idx)
    bo_ids_ref,  # [4] (bo_sem_0_seq_idx, bo_sem_1_seq_idx, bo_sem_0_bo_idx, bo_sem_1_bo_idx)
    bkv_update_ids_ref,  # [6] (bkv_sem_0_seq_idx, bkv_sem_1_seq_idx, bkv_sem_0_offset, bkv_sem_1_offset, bkv_sem_0_sz, bkv_sem_1_sz)
    # Input
    ql_nope_hbm_ref,  # [max_num_tokens, num_q_heads_per_q_packing, q_packing, lkv_dim]
    q_pe_hbm_ref,  # [max_num_tokens, num_q_heads_per_q_packing, q_packing, r_dim]
    new_kv_c_hbm_ref,  # [max_num_tokens_per_kv_packing, kv_packing, lkv_dim]
    new_k_pe_hbm_ref,  # [max_num_tokens_per_kv_packing, kv_packing, r_dim]
    cache_kv_hbm_ref,  # [total_num_pages, page_size_per_kv_packing, kv_packing, align_to(lkv_dim + r_dim, 128)]
    # Output
    o_hbm_ref,  # [max_num_tokens, num_q_heads_per_q_packing, q_packing, lkv_dim]
    updated_cache_kv_hbm_ref,  # [total_num_pages, page_size_per_kv_packing, kv_packing, align_to(lkv_dim + r_dim, 128)]
    # Scratch
    bkvc_x2_ref,  # [2, bkv_buf_sz_per_kv_packing, kv_packing, lkv_dim]
    bkpe_x2_ref,  # [2, bkv_buf_sz_per_kv_packing, kv_packing, r_dim]
    bq_nope_x2_ref,  # [2, bq_sz, num_q_heads_per_q_packing, q_packing, lkv_dim]
    bq_rope_x2_ref,  # [2, bq_sz, num_q_heads_per_q_packing, q_packing, r_dim]
    bo_x2_ref,  # [2, bq_sz, num_q_heads_per_q_packing, q_packing, lkv_dim]
    sems,  # [4, 2]
    l_ref,  # [bq_sz * num_q_heads, 128],
    m_ref,  # [bq_sz * num_q_heads, 128],
    acc_ref,  # [bq_sz * num_q_heads, lkv_dim],
    *,
    sm_scale: float,
    sliding_window: int | None = None,
    soft_cap: float | None = None,
    mask_value: float = DEFAULT_MASK_VALUE,
    q_scale: float | None = None,
    k_scale: float | None = None,
    v_scale: float | None = None,
    chunk_prefill_size: int | None = None,
    bkv_p,
    bq_sz,
    debug_mode: bool = False,
):
    assert ql_nope_hbm_ref.shape == o_hbm_ref.shape
    # Validation checks on the dimensions
    nope_dim = ql_nope_hbm_ref.shape[-1]
    pe_dim = q_pe_hbm_ref.shape[-1]
    assert nope_dim + pe_dim == cache_kv_hbm_ref.shape[-1]

    _, num_q_heads_per_q_packing, q_packing, lkv_dim = ql_nope_hbm_ref.shape
    r_dim = q_pe_hbm_ref.shape[-1]
    num_q_heads = num_q_heads_per_q_packing * q_packing
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
    seq_idx = pl.program_id(0)
    num_seqs = pl.num_programs(0)
    decode_end = distribution_ref[0]
    prefill_end = distribution_ref[1]
    mixed_end = distribution_ref[2]

    q_start = cu_q_lens_ref[seq_idx]
    q_end = cu_q_lens_ref[seq_idx + 1]
    q_len = q_end - q_start
    kv_len = kv_lens_ref[seq_idx]

    def debug_print(msg, *args):
        if debug_mode:
            pl.debug_print(msg, *args)

    debug_print("[RPA debug] ======= In loop seq_idx={}", seq_idx)
    debug_print("[RPA debug] num_seqs={}", num_seqs)
    debug_print("[RPA debug] decode_end={}", decode_end)
    debug_print("[RPA debug] prefill_end={}", prefill_end)
    debug_print("[RPA debug] mixed_end={}", mixed_end)
    debug_print("[RPA debug] bkv_p={}", bkv_p)
    debug_print("[RPA debug] page_size={}", page_size)
    debug_print("[RPA debug] pages_per_seq={}", pages_per_seq)
    debug_print("[RPA debug] bkv_sz_per_kv_packing={}", bkv_sz_per_kv_packing)
    debug_print("[RPA debug] bq_sz={}", bq_sz)
    debug_print("[RPA debug] q_start={}", q_start)
    debug_print("[RPA debug] q_end={}", q_end)
    debug_print("[RPA debug] q_len={}", q_len)
    debug_print("[RPA debug] kv_len={}", kv_len)

    def flash_attention(
        ql_nope,  # [actual_bq_sz * num_q_heads, lkv_dim]
        q_pe,  # [actual_bq_sz * num_q_heads, r_dim]
        kv_c,  # [bkv_sz, lkv_dim] <- Correspond to data from bkvc_x2_ref
        k_pe,  # [bkv_sz, r_dim] <- Correspond to data from bpe_x2_ref
        *,
        bq_idx,
        bkv_idx,
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
        head_l_ref = l_ref.at[:ql_nope.shape[0]]
        head_m_ref = m_ref.at[:ql_nope.shape[0]]
        head_acc_ref = acc_ref.at[:ql_nope.shape[0]]

        def load_with_init(ref, init_val):
            return jnp.where(bkv_idx == 0, jnp.full_like(ref, init_val),
                             ref[...])

        # Follow FlashAttention-2 forward pass.
        s_nope = jnp.einsum("nd,md->nm",
                            ql_nope,
                            kv_c,
                            preferred_element_type=jnp.float32)
        s_pe = jnp.einsum("nd,md->nm",
                          q_pe,
                          k_pe,
                          preferred_element_type=jnp.float32)
        s = s_nope + s_pe
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
        s = jnp.where(mask, mask_value, s)
        s_rowmax = jnp.max(s, axis=1, keepdims=True)
        m_prev = load_with_init(head_m_ref, -jnp.inf)
        m_curr = jnp.maximum(m_prev, s_rowmax)
        head_m_ref[...] = m_curr
        p = jnp.exp(s - broadcast_minor(m_curr, s.shape))

        pv = jnp.einsum("nm,md->nd",
                        p,
                        kv_c,
                        preferred_element_type=jnp.float32)
        if v_scale is not None:
            pv *= v_scale

        p_rowsum = jnp.sum(p, axis=1, keepdims=True)
        exp_m_diff = jnp.exp(m_prev - m_curr)
        l_prev = load_with_init(head_l_ref, 0.0)
        l_curr = exp_m_diff * l_prev + p_rowsum
        head_l_ref[...] = l_curr
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

    def _fetch_bkv(seq_idx, bkv_idx, bkv_sem_idx, *, wait=False):
        sem = sems.at[0, bkv_sem_idx]
        # bkvc_x2_ref shape: [2, bkv_sz_per_kv_packing + 2, kv_packing, lkv_dim]
        bkvc_vmem_ref = bkvc_x2_ref.at[bkv_sem_idx]
        bkvpe_vmem_ref = bkpe_x2_ref.at[bkv_sem_idx]

        # [total_num_pages, page_size_per_kv_packing, kv_packing, align_to(lkv_dim + r_dim, 128)]
        # [total_num_pages * page_size_per_kv_packing, kv_packing, align_to(lkv_dim + r_dim, 128)]
        reshaped_cache_hbm_ref = cache_kv_hbm_ref.reshape(
            total_num_pages * page_size_per_kv_packing,
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
        bkv_sz_frm_new_per_kv_packing = cdiv(bkv_sz_frm_new, kv_packing)
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

        debug_print(
            "[RPA debug]"
            f" -----------{'wait' if wait else 'start'}_fetch_bkv-----------")
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
        debug_print("[RPA debug] page_indices_offset={}", page_indices_offset)
        debug_print(f"[RPA debug] bkvc_vmem_ref.shape: {bkvc_vmem_ref.shape}")
        debug_print(
            f"[RPA debug] bkvpe_vmem_ref.shape: {bkvpe_vmem_ref.shape}")

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
            debug_print("[RPA debug] new_kv_len_start={}", new_kv_len_start)
            debug_print(
                "[RPA debug] new_kv_len_start_per_kv_packing={}",
                new_kv_len_start_per_kv_packing,
            )
            debug_print(f"new_kv_c_hbm_ref.shape: {new_kv_c_hbm_ref.shape}")
            debug_print(f"new_k_pe_hbm_ref.shape: {new_k_pe_hbm_ref.shape}")
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
        bkvc_vmem_ref = bkvc_x2_ref.at[bkv_sem_idx]
        # shape: [bkv_sz_per_kv_packing + 2, kv_packing, r_dim]
        bkvpe_vmem_ref = bkpe_x2_ref.at[bkv_sem_idx]

        update_kv_packing_iters = cdiv((offset % kv_packing) + update_sz,
                                       kv_packing)
        kv_packing_offset = offset % kv_packing
        new_kv_len_start = q_end - kv_len + offset
        new_kv_packing_offset = new_kv_len_start % kv_packing

        token_offset_in_bkv = offset % bkv_sz
        kv_packing_idx = token_offset_in_bkv // kv_packing

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
        kv_packing_idx_new = (cdiv(token_offset_in_bkv, kv_packing) +
                              (-shift_amount) // kv_packing)
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
                lax.shift_right_logical(curr_kvc_reg_u32, 32 - shift_bits),
                lax.shift_left(next_kvc_reg_u32, shift_bits),
            )
            shifted_kpe_u32 = lax.bitwise_or(
                lax.shift_right_logical(curr_kpe_reg_u32, 32 - shift_bits),
                lax.shift_left(next_kpe_reg_u32, shift_bits),
            )

            # If shift_bits is 0, we should use the current word. Otherwise,
            # shifting by 32 bits would result in shifted_*_u32 becoming
            # next_*_reg_u32, which is incorrect.
            rotated_kvc_u32 = lax.select(shift_bits == 0, curr_kvc_reg_u32,
                                         shifted_kvc_u32)
            rotated_kpe_u32 = lax.select(shift_bits == 0, curr_kpe_reg_u32,
                                         shifted_kpe_u32)

            next_kvc_reg_shifted = pltpu.bitcast(rotated_kvc_u32,
                                                 next_kvc_reg.dtype)
            next_kpe_reg_shifted = pltpu.bitcast(rotated_kpe_u32,
                                                 next_kpe_reg.dtype)

            offset_in_word = i * kv_packing + lax.broadcasted_iota(
                dtype=jnp.int32, shape=[kv_packing, lkv_dim], dimension=0)
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
                dtype=jnp.int32, shape=[kv_packing, r_dim], dimension=0)
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
        seq_idx,
        bkv_sem_idx,
        offset,  # In units of tokens.
        update_sz,  # In units of tokens.
        *,
        wait=False,
    ):
        sem = sems.at[3, bkv_sem_idx]
        # shape: [bkv_sz_per_kv_packing + 2, kv_packing, lkv_dim]
        bkvc_vmem_ref = bkvc_x2_ref.at[bkv_sem_idx]
        # shape: [bkv_sz_per_kv_packing + 2, kv_packing, r_dim]
        bkvpe_vmem_ref = bkpe_x2_ref.at[bkv_sem_idx]

        update_kv_packing_iters = cdiv((offset % kv_packing) + update_sz,
                                       kv_packing)

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

    def _fetch_bq(seq_idx, bq_idx, bq_sem_idx, *, wait=False):
        sem = sems.at[1, bq_sem_idx]
        bq_nope_vmem_ref = bq_nope_x2_ref.at[bq_sem_idx]
        bq_rope_vmem_ref = bq_rope_x2_ref.at[bq_sem_idx]

        q_len_start = cu_q_lens_ref[seq_idx] + bq_idx * bq_sz
        q_end = cu_q_lens_ref[seq_idx + 1]
        sz = jnp.minimum(bq_sz, q_end - q_len_start)

        debug_print(
            "[RPA debug]"
            f" -----------{'wait' if wait else 'start'}_fetch_bq-----------")
        debug_print("[RPA debug] seq_idx={}", seq_idx)
        debug_print("[RPA debug] bq_idx={}", bq_idx)
        debug_print("[RPA debug] bq_sem_idx={}", bq_sem_idx)
        debug_print("[RPA debug] q_len_start={}", q_len_start)
        debug_print("[RPA debug] q_end={}", q_end)
        debug_print("[RPA debug] sz={}", sz)

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

    def _send_bo(seq_idx, bo_idx, bo_sem_idx, *, wait=False):
        sem = sems.at[2, bo_sem_idx]
        vmem_ref = bo_x2_ref.at[bo_sem_idx]
        q_len_start = cu_q_lens_ref[seq_idx] + bo_idx * bq_sz
        q_end = cu_q_lens_ref[seq_idx + 1]
        sz = jnp.minimum(bq_sz, q_end - q_len_start)

        debug_print(
            "[RPA debug]"
            f" -----------{'wait' if wait else 'start'}_send_bo-----------")
        debug_print("[RPA debug] seq_idx={}", seq_idx)
        debug_print("[RPA debug] bo_idx={}", bo_idx)
        debug_print("[RPA debug] bo_sem_idx={}", bo_sem_idx)
        debug_print("[RPA debug] q_len_start={}", q_len_start)
        debug_print("[RPA debug] q_end={}", q_end)
        debug_print("[RPA debug] sz={}", sz)

        _async_copy(
            vmem_ref.at[pl.ds(0, sz)],
            o_hbm_ref.at[pl.ds(q_len_start, sz)],
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

    def wait_send_bo(bo_sem_idx):
        old_seq_idx = bo_ids_ref[bo_sem_idx]
        old_bo_idx = bo_ids_ref[bo_sem_idx + 2]

        @pl.when(jnp.logical_and(0 <= old_seq_idx, old_seq_idx <= seq_idx))
        def _():
            _send_bo(old_seq_idx, old_bo_idx, bo_sem_idx, wait=True)

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

    def load_bq(bq_sem_idx, *, actual_bq_sz=bq_sz):
        q_nope_ref = (bq_nope_x2_ref.bitcast(
            jnp.uint32).at[bq_sem_idx].reshape(
                bq_sz * num_q_heads_per_q_packing, lkv_dim))
        q_nope_vec = pltpu.bitcast(
            q_nope_ref[:actual_bq_sz * num_q_heads_per_q_packing],
            q_dtype,
        ).reshape(actual_bq_sz * num_q_heads, lkv_dim)
        q_rope_ref = (bq_rope_x2_ref.bitcast(
            jnp.uint32).at[bq_sem_idx].reshape(
                bq_sz * num_q_heads_per_q_packing, r_dim))
        q_rope_vec = pltpu.bitcast(
            q_rope_ref[:actual_bq_sz * num_q_heads_per_q_packing],
            q_dtype,
        ).reshape(actual_bq_sz * num_q_heads, r_dim)
        return q_nope_vec, q_rope_vec

    def load_bkv(bkv_sem_idx, *, bkvc_mask, bkpe_mask):
        bkvc_ref = (bkvc_x2_ref.bitcast(
            jnp.uint32).at[bkv_sem_idx, :bkv_sz_per_kv_packing].reshape(
                bkv_sz_per_kv_packing, lkv_dim))
        bkvc_vec = pltpu.bitcast(bkvc_ref[...],
                                 kv_dtype).reshape(bkv_sz, lkv_dim)
        bkvc_vec = lax.select(bkvc_mask, bkvc_vec, jnp.zeros_like(bkvc_vec))

        bkpe_ref = (bkpe_x2_ref.bitcast(
            jnp.uint32).at[bkv_sem_idx, :bkv_sz_per_kv_packing].reshape(
                bkv_sz_per_kv_packing, r_dim))
        bkpe_vec = pltpu.bitcast(bkpe_ref[...],
                                 kv_dtype).reshape(bkv_sz, r_dim)
        bkpe_vec = lax.select(bkpe_mask, bkpe_vec, jnp.zeros_like(bkpe_vec))

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

    def process(static_q_len=None):
        num_bkv = cdiv(kv_len, bkv_sz)
        if static_q_len is None:
            actual_bq_sz = bq_sz
            num_bq = cdiv(q_len, actual_bq_sz)
        else:
            actual_bq_sz = min(bq_sz, static_q_len)
            num_bq = cdiv(static_q_len, actual_bq_sz)

        debug_print("[RPA debug] process")
        debug_print("[RPA debug] num_bkv={}", num_bkv)  # num_bkv=3, bkv_sz=512
        debug_print("[RPA debug] bkv_sz={}", bkv_sz)
        debug_print("[RPA debug] num_bq={}", num_bq)
        debug_print("[RPA debug] kv_len={}", kv_len)

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

            # Prefetch next bq
            @pl.when(next_seq_idx < num_seqs)
            def prefetch_next_bq():
                sem_ids_ref[0] = next_bq_sem_idx
                start_fetch_bq(next_seq_idx, next_bq_idx, next_bq_sem_idx)

            def compute_with_bkv(bkv_idx, _):
                # Create bitmask for KV.
                assert bkv_sz % kv_packing == 0
                actual_bkv_sz = jnp.minimum(bkv_sz, kv_len - bkv_idx * bkv_sz)
                bkvc_shape = (bkv_sz, lkv_dim)
                bkvc_mask = (lax.broadcasted_iota(jnp.int32, bkvc_shape, 0)
                             < actual_bkv_sz)
                bkpe_shape = (bkv_sz, r_dim)
                bkpe_mask = (lax.broadcasted_iota(jnp.int32, bkpe_shape, 0)
                             < actual_bkv_sz)

                # Get next bkv ids.
                bkv_sem_idx = sem_ids_ref[1]
                next_seq_idx, _, next_bkv_idx, next_bkv_sem_idx = get_next_bkv_ids(
                    seq_idx, bq_idx, bkv_idx, bkv_sem_idx)

                # Prefetch next bkv
                @pl.when(next_seq_idx < num_seqs)
                def prefetch_next_bkv():
                    sem_ids_ref[1] = next_bkv_sem_idx
                    start_fetch_bkv(next_seq_idx, next_bkv_idx,
                                    next_bkv_sem_idx)

                # Wait for cur bq if not ready yet
                @pl.when(bkv_idx == 0)
                def wait_cur_bq():
                    wait_fetch_bq(seq_idx, bq_idx, bq_sem_idx)

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

                # Load bkv into vreg
                bkvc, bkpe = load_bkv(bkv_sem_idx,
                                      bkvc_mask=bkvc_mask,
                                      bkpe_mask=bkpe_mask)

                bq_nope_vec, bq_pe_vec = load_bq(bq_sem_idx,
                                                 actual_bq_sz=actual_bq_sz)

                debug_print("[RPA debug] flash attention")
                debug_print(
                    "[RPA debug] bq_nope_vec.shape={}, {}",
                    bq_nope_vec.shape[0],
                    bq_nope_vec.shape[1],
                )  # num_bkv=3, bkv_sz=512
                debug_print(
                    "[RPA debug] bq_pe_vec.shape={}, {}",
                    bq_pe_vec.shape[0],
                    bq_pe_vec.shape[1],
                )
                debug_print("[RPA debug] bkvc.shape={}, {}", bkvc.shape[0],
                            bkvc.shape[1])
                debug_print("[RPA debug] bkpe.shape={}, {}", bkpe.shape[0],
                            bkpe.shape[1])

                if debug_mode:
                    return

                flash_attention(
                    bq_nope_vec,
                    bq_pe_vec,
                    bkvc,
                    bkpe,
                    bq_idx=bq_idx,
                    bkv_idx=bkv_idx,
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
                bq_sz * num_q_heads_per_q_packing,
                lkv_dim,
            )[...] = pltpu.bitcast(out, jnp.int32)

            # Send cur bo
            start_send_bo(seq_idx, bq_idx, bo_sem_idx)

        lax.fori_loop(0, num_bq, compute_with_bq, None, unroll=False)

    ### ------- Kernel start ------- ###

    @pl.when(seq_idx == 0)
    def prologue():
        start_fetch_bq(0, 0, 0)
        start_fetch_bkv(0, 0, 0)

    @pl.when(seq_idx < decode_end)
    def process_decode():
        process(static_q_len=1)

    @pl.when(jnp.logical_and(decode_end <= seq_idx, seq_idx < prefill_end))
    def process_prefill():
        process(static_q_len=chunk_prefill_size)

    @pl.when(jnp.logical_and(prefill_end <= seq_idx, seq_idx < mixed_end))
    def process_mixed():
        process()

    @pl.when(seq_idx == num_seqs - 1)
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
        q.reshape(
            max_num_tokens,
            actual_num_q_heads,
            actual_head_dim,
        ),
        (
            (0, 0),
            (0, num_q_heads - actual_num_q_heads),
            (0, head_dim - actual_head_dim),
        ),
        constant_values=0,
    ).reshape(
        max_num_tokens,
        num_q_heads // q_packing,
        q_packing,
        head_dim,
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


def prepare_outputs(
    out,  # [max_num_tokens, num_q_heads // q_packing, q_packing, head_dim]
    actual_num_q_heads: int,
    actual_head_dim: int,
):
    (
        max_num_tokens,
        num_q_heads_per_q_packing,
        q_packing,
        head_dim,
    ) = out.shape
    return out.reshape(
        max_num_tokens,
        num_q_heads_per_q_packing * q_packing,
        head_dim,
    )[:, :actual_num_q_heads, :actual_head_dim]


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
    mask_value: float | None = DEFAULT_MASK_VALUE,
    q_scale: float | None = None,
    k_scale: float | None = None,
    v_scale: float | None = None,
    # Kernel optimization params.
    chunk_prefill_size: int | None = None,
    # Kernel tuning params.
    num_kv_pages_per_block: int | None = None,
    num_queries_per_block: int | None = None,
    vmem_limit_bytes: int | None = None,
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
      attention block in the pallas kernel.
    num_queries_per_block: number of kv pages to be processed in one flash
      attention block in the pallas kernel.
    vmem_limit_bytes: the vmem limit for the pallas kernel.
    debug_mode: if true, RPA does not issue any DMAs or run flash attention but
      print debug info. Need to compile with `--xla_tpu_enable_log_recorder`.

  Returns:
    The output of attention and the updated kv cache.
  """
    if num_kv_pages_per_block is None or num_queries_per_block is None:
        raise ValueError(
            "num_kv_pages_per_block and num_queries_per_block must be specified."
        )
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
        num_kv_pages_per_block=num_kv_pages_per_block,
        num_queries_per_block=num_queries_per_block,
        vmem_limit_bytes=vmem_limit_bytes,
        debug_mode=debug_mode,
    )

    _, actual_num_q_heads, actual_lkv_dim = ql_nope.shape

    ql_nope = prepare_q_inputs(
        ql_nope
    )  # [max_num_tokens, num_q_heads_per_q_packing, q_packing, lkv_dim]
    q_pe = prepare_q_inputs(
        q_pe)  # [max_num_tokens, num_q_heads_per_q_packing, q_packing, r_dim]
    new_kv_c = prepare_kv_inputs(
        new_kv_c)  # [max_num_tokens_per_kv_packing, kv_packing, lkv_dim]
    new_k_pe = prepare_kv_inputs(
        new_k_pe)  # [max_num_tokens_per_kv_packing, kv_packing, r_dim]
    lkv_dim = new_kv_c.shape[-1]
    r_dim = new_k_pe.shape[-1]

    _, page_size_per_kv_packing, kv_packing, _ = cache_kv.shape
    page_size = page_size_per_kv_packing * kv_packing
    _, num_q_heads_per_q_packing, q_packing, _ = ql_nope.shape
    max_num_seqs = kv_lens.shape[0]
    num_page_indices = page_indices.shape[0]
    assert num_page_indices % max_num_seqs == 0
    num_q_heads = num_q_heads_per_q_packing * q_packing

    bkv_p = num_kv_pages_per_block
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
    grid = (distribution[2], )

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
        (2, bkv_buf_sz_per_kv_packing, kv_packing, lkv_dim),
        cache_kv.dtype,
    )

    bkpe_double_buf = pltpu.VMEM(
        (2, bkv_buf_sz_per_kv_packing, kv_packing, r_dim),
        cache_kv.dtype,
    )
    bq_nope_double_buf = pltpu.VMEM(
        (2, bq_sz, num_q_heads_per_q_packing, q_packing, lkv_dim),
        ql_nope.dtype,
    )

    bq_rope_double_buf = pltpu.VMEM(
        (2, bq_sz, num_q_heads_per_q_packing, q_packing, r_dim),
        q_pe.dtype,
    )

    bo_double_buf = bq_nope_double_buf

    l_scratch = pltpu.VMEM(
        (bq_sz * num_q_heads, 128),
        jnp.float32,
    )
    m_scratch = l_scratch

    acc_scratch = pltpu.VMEM(
        (bq_sz * num_q_heads, lkv_dim),
        jnp.float32,
    )

    scratch_shapes = [
        bkvc_double_buf,
        bkpe_double_buf,
        bq_nope_double_buf,
        bq_rope_double_buf,
        bo_double_buf,  # Double buffering for output block.
        # Semaphores for double buffering of bkv, bq, bo and bkv_update.
        pltpu.SemaphoreType.DMA((4, 2)),
        # Intermediate buffers per kv head for flash attention.
        l_scratch,
        m_scratch,
        acc_scratch,
    ]

    scalar_prefetches = (
        kv_lens,
        page_indices,
        cu_q_lens,
        distribution,
        # (bq_sem_idx, bkv_sem_idx, bo_sem_idx)
        jnp.zeros((3, ), jnp.int32),
        # (bo_sem_0_seq_idx, bo_sem_1_seq_idx, bo_sem_0_bo_idx, bo_sem_1_bo_idx)
        jnp.full((4, ), -1, jnp.int32),
        # (bkv_sem_0_seq_idx, bkv_sem_1_seq_idx, bkv_sem_0_offset, bkv_sem_1_offset, bkv_sem_0_sz, bkv_sem_1_sz)
        jnp.full((6, ), -1, jnp.int32),
    )

    scope_name = f"MLA-RPA-bq_{bq_sz}-bkvp_{bkv_p}-p_{page_size}"
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
                chunk_prefill_size=chunk_prefill_size,
                bq_sz=bq_sz,
                bkv_p=bkv_p,
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
            ),
            out_shape=[
                jax.ShapeDtypeStruct(shape=ql_nope.shape, dtype=ql_nope.dtype),
                jax.ShapeDtypeStruct(shape=cache_kv.shape,
                                     dtype=cache_kv.dtype),
            ],
            input_output_aliases={
                7: 0,  # Alias output activation with ql_nope
                11: 1,  # Aliasing cache_kv with updated_cache_kv
            },
            name=scope_name,
        ))

    output, updated_kv = kernel(
        *scalar_prefetches,
        ql_nope,
        q_pe,
        new_kv_c,
        new_k_pe,
        cache_kv,
    )
    output = prepare_outputs(
        output, actual_num_q_heads,
        actual_lkv_dim)  # [max_num_tokens, actual_num_q_heads, actual_lkv_dim]

    return output, updated_kv
