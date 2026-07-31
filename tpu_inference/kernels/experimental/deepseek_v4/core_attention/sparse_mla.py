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

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from tpu_inference.kernels.experimental.deepseek_v4.core_attention import \
    csa_gather

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


_GATHER_PAGE_CHUNK = 128


def _gather_page_ids_kernel(windows_ref, logical_ref, out_ref, *, num_chunks):
    logical = logical_ref[...]  # i32[block_tokens, topk]
    out = jnp.zeros_like(logical)
    for c in range(num_chunks):
        window_chunk = windows_ref[:, c * _GATHER_PAGE_CHUNK:(
            c + 1) * _GATHER_PAGE_CHUNK]  # i32[block_tokens, 128]
        local = logical - c * _GATHER_PAGE_CHUNK
        gathered = jnp.take_along_axis(window_chunk,
                                       jnp.clip(local, 0,
                                                _GATHER_PAGE_CHUNK - 1),
                                       axis=1)
        out = jnp.where((local >= 0) & (local < _GATHER_PAGE_CHUNK), gathered,
                        out)
    out_ref[...] = out


def gather_page_ids(
    page_indices: jax.Array,  # i32[max_num_seqs * pages_per_seq]
    seq_page_ids: jax.
    Array,  # i32[num_tokens, topk]  (logical page within seq)
    seq_ids_segment: jax.Array,  # i32[num_tokens]  (token -> seq id)
    max_num_seqs: int,
    *,
    block_tokens: int = 8,
) -> jax.Array:
    """Gathers physical page ids for the CSA top-k tokens."""
    num_tokens, topk = seq_page_ids.shape
    pages_per_seq = page_indices.shape[0] // max_num_seqs
    num_chunks = cdiv(pages_per_seq, _GATHER_PAGE_CHUNK)
    padded_pps = num_chunks * _GATHER_PAGE_CHUNK

    page_table = page_indices.reshape(max_num_seqs, pages_per_seq)
    if padded_pps != pages_per_seq:
        page_table = jnp.pad(page_table,
                             ((0, 0), (0, padded_pps - pages_per_seq)))
    # Per-token page-table window. This is a whole-row gather.
    windows = page_table[seq_ids_segment]  # i32[num_tokens, padded_pps]
    logical = jnp.clip(seq_page_ids, 0, pages_per_seq - 1)

    padded_tokens = align_to(num_tokens, block_tokens)
    if padded_tokens != num_tokens:
        pad = padded_tokens - num_tokens
        windows = jnp.pad(windows, ((0, pad), (0, 0)))
        logical = jnp.pad(logical, ((0, pad), (0, 0)))

    out = pl.pallas_call(
        functools.partial(_gather_page_ids_kernel, num_chunks=num_chunks),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            in_specs=[
                pl.BlockSpec((block_tokens, padded_pps), lambda t: (t, 0)),
                pl.BlockSpec((block_tokens, topk), lambda t: (t, 0)),
            ],
            out_specs=pl.BlockSpec((block_tokens, topk), lambda t: (t, 0)),
            grid=(padded_tokens // block_tokens, ),
        ),
        out_shape=jax.ShapeDtypeStruct((padded_tokens, topk), jnp.int32),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("arbitrary", ),
            disable_bounds_checks=True,
        ),
        name="gather_page_ids",
    )(windows, logical)
    return out[:num_tokens]


# DSV4 FP8 nope layout: 448 e4m3 values plus 7 e8m0 scales, one scale per
# 64-wide block of the 448.
def _make_dsv4_fp8_scale_expand_matrix():
    """One-hot [7, 448] matrix mapping each e8m0 scale onto its 64-lane block."""
    scale_id = lax.broadcasted_iota(jnp.int32, (7, 448), 0)
    block_id = lax.broadcasted_iota(jnp.int32, (7, 448), 1) // 64
    return (scale_id == block_id).astype(jnp.bfloat16)


def _dequant_dsv4_fp8(bkv_nope: jax.Array,
                      dsv4_fp8_scale_expand_matrix: jax.Array):
    """Dequantize FP8 values to BF16."""
    nope_fp8 = pltpu.bitcast(bkv_nope[:, :448],
                             jnp.float8_e4m3fn).astype(jnp.bfloat16)
    nope_scales = pltpu.bitcast(
        bkv_nope[:, 448:448 + 7],
        jnp.float8_e8m0fnu,
    ).astype(jnp.bfloat16)
    # Using on-hot matrix to broadcast each scale across its 64-lane block is
    # more efficient than using
    # `nope_scales = jnp.repeat(nope_scales.T, 64, axis=0).T`
    nope_scales = jnp.dot(
        nope_scales,
        dsv4_fp8_scale_expand_matrix,
        preferred_element_type=jnp.float32,
    ).astype(jnp.bfloat16)
    nope = (nope_fp8 * nope_scales).astype(jnp.bfloat16)
    return nope


def _attention_kernel(
    # Prefetch
    kv_lens_ref,  # [max_num_seqs]
    start_end_seq_idx_ref,  # [2] (start_seq_idx, end_seq_idx)
    sem_ids_ref,  # [2] (bi_sem_idx, bo_sem_idx)
    # Input
    attention_sinks_ref,  # float32[num_q_heads]
    q_hbm_ref,  # [max_num_tokens, num_q_heads, head_dim]
    cache_kv_nope_hbm_ref,  # [total_num_pages, page_size, nope_dim]
    cache_kv_rope_hbm_ref,  # [total_num_pages, page_size, rope_dim]
    swa_accumution_hbm_ref,  # [max_num_tokens, num_q_heads, head_dim]
    swa_l_hbm_ref,  # [max_num_tokens, num_l_heads]
    swa_m_hbm_ref,  # [max_num_tokens, num_l_heads]
    # Output
    o_hbm_ref,  # [max_num_tokens, num_q_heads, head_dim]
    # Scratch
    bkv_nope_x2_ref,  # [2, batch_size, page_size, nope_dim]
    bkv_rope_x2_ref,  # [2, batch_size, page_size, rope_dim]
    bq_x2_ref,  # [2, batch_size, num_q_heads, head_dim]
    bo_x2_ref,  # [2, batch_size, num_q_heads, head_dim]
    bl_x2_ref,  # [2, batch_size, num_l_heads]
    bm_x2_ref,  # [2, batch_size, num_l_heads]
    swa_acc_x2_ref,  # [2, batch_size, num_q_heads, head_dim]
    sems,  # [7, 2]
    *,
    sm_scale: float,
    batch_size: int = 1,
):
    assert q_hbm_ref.shape == o_hbm_ref.shape

    num_tokens, num_q_heads, head_dim = q_hbm_ref.shape
    _, page_size, _ = cache_kv_nope_hbm_ref.shape
    assert kv_lens_ref.shape[0] == num_tokens
    bkv_sz = page_size

    q_dtype = q_hbm_ref.dtype
    q_packing = get_dtype_packing(q_dtype)
    # Validate against the KV dtype.
    assert o_hbm_ref.dtype == q_dtype

    assert head_dim % 128 == 0
    assert num_q_heads % q_packing == 0

    start_seq_idx = start_end_seq_idx_ref[0]
    end_seq_idx = start_end_seq_idx_ref[1]

    batch_start_seq_idx = start_seq_idx + pl.program_id(0) * batch_size
    batch_end_seq_idx = batch_start_seq_idx + batch_size - 1

    def flash_attention_step1_qk_softmax(
            q,  # [bq_sz * num_q_heads, head_dim]
            kv,  # [bkv_sz, head_dim] <- Correspond to data from bkv_*_x2_ref
            kv_len,  # real top-k entries for this query token
            swa_m,  # [bq_sz * num_q_heads],
            swa_l,  # [bq_sz * num_q_heads],
            attention_sinks,  # [num_q_heads]
    ):
        assert len(q.shape) == 2
        assert len(kv.shape) == 2
        assert q.shape[0] % num_q_heads == 0
        assert q.shape[1] == head_dim
        assert kv.shape == (bkv_sz, head_dim)

        # Follow FlashAttention-2 forward pass.
        s = jnp.einsum("nd,md->nm", q, kv, preferred_element_type=jnp.float32)
        s *= sm_scale
        k_span = lax.broadcasted_iota(jnp.int32, s.shape, 1)
        s = jnp.where(k_span < kv_len, s, jnp.finfo(s.dtype).min)

        s_rowmax = jnp.max(s, axis=1, keepdims=True)
        m_prev = swa_m
        m_curr = jnp.maximum(m_prev, s_rowmax)
        p = jnp.exp(s - m_curr)
        exp_m_diff = jnp.exp(m_prev - m_curr)
        p_rowsum = jnp.sum(p, axis=1, keepdims=True)
        l_prev = swa_l
        l_curr = exp_m_diff * l_prev + p_rowsum
        exp_attention_sinks = jnp.exp(attention_sinks - m_curr)
        l_sum = l_curr + exp_attention_sinks

        return p, exp_m_diff, l_sum

    def flash_attention_step2_pv(
        p,
        kv,
        exp_m_diff,
        swa_acc,
        l_sum,
    ):
        pv = jnp.einsum("nm,md->nd", p, kv, preferred_element_type=jnp.float32)

        o_prev = swa_acc
        acc = exp_m_diff * o_prev + pv
        out = (lax.div(acc, l_sum) if q_dtype == jnp.float32 else
               (acc * pl.reciprocal(l_sum, approx=True)).astype(q_dtype))
        return out

    def _async_copy(src, dst, sem, wait):
        cp = pltpu.make_async_copy(src, dst, sem)
        if wait:
            cp.wait()
        else:
            cp.start()

    def _fetch_bkv_batch(seq_idx_start, bkv_sem_idx, *, wait=False):
        sem_nope = sems.at[0, bkv_sem_idx]
        sem_rope = sems.at[6, bkv_sem_idx]

        bkv_nope_vmem_ref = bkv_nope_x2_ref.at[bkv_sem_idx, :]
        bkv_rope_vmem_ref = bkv_rope_x2_ref.at[bkv_sem_idx, :]

        # The index into cache_kv_hbm_ref should be relative to the current
        # chunk.
        page_idx_start = seq_idx_start - start_seq_idx
        if not wait:
            _async_copy(
                cache_kv_nope_hbm_ref.at[pl.ds(page_idx_start, batch_size)],
                bkv_nope_vmem_ref,
                sem_nope,
                wait,
            )
            _async_copy(
                cache_kv_rope_hbm_ref.at[pl.ds(page_idx_start, batch_size)],
                bkv_rope_vmem_ref,
                sem_rope,
                wait,
            )
        else:
            dst_nope = bkv_nope_vmem_ref
            _async_copy(src=dst_nope, dst=dst_nope, sem=sem_nope, wait=True)
            dst_rope = bkv_rope_vmem_ref
            _async_copy(src=dst_rope, dst=dst_rope, sem=sem_rope, wait=True)

    def _fetch_bq_batch(seq_idx_start, bq_sem_idx, *, wait=False):
        sem = sems.at[1, bq_sem_idx]
        bq_vmem_ref = bq_x2_ref.at[bq_sem_idx, :]

        if not wait:
            _async_copy(
                q_hbm_ref.at[pl.ds(seq_idx_start, batch_size)],
                bq_vmem_ref,
                sem,
                wait,
            )
        else:
            _async_copy(src=bq_vmem_ref, dst=bq_vmem_ref, sem=sem, wait=True)

    def _send_bo_batch(seq_idx_start, bo_sem_idx, *, wait=False):
        sem = sems.at[2, bo_sem_idx]
        vmem_ref = bo_x2_ref.at[bo_sem_idx, :]

        if not wait:
            _async_copy(
                vmem_ref,
                o_hbm_ref.at[pl.ds(seq_idx_start, batch_size)],
                sem,
                wait,
            )
        else:
            _async_copy(src=vmem_ref, dst=vmem_ref, sem=sem, wait=True)

    def _fetch_swa_batch(seq_idx_start, bq_sem_idx, *, wait=False):
        sem_acc = sems.at[3, bq_sem_idx]
        sem_l = sems.at[4, bq_sem_idx]
        sem_m = sems.at[5, bq_sem_idx]

        if not wait:
            _async_copy(
                swa_accumution_hbm_ref.at[pl.ds(seq_idx_start, batch_size)],
                swa_acc_x2_ref.at[bq_sem_idx, :],
                sem_acc,
                wait=False,
            )
            _async_copy(
                swa_l_hbm_ref.at[pl.ds(seq_idx_start, batch_size)],
                bl_x2_ref.at[bq_sem_idx, :],
                sem_l,
                wait=False,
            )
            _async_copy(
                swa_m_hbm_ref.at[pl.ds(seq_idx_start, batch_size)],
                bm_x2_ref.at[bq_sem_idx, :],
                sem_m,
                wait=False,
            )

        else:
            dst_acc = swa_acc_x2_ref.at[bq_sem_idx, :]
            _async_copy(src=dst_acc, dst=dst_acc, sem=sem_acc, wait=True)

            dst_l = bl_x2_ref.at[bq_sem_idx, :]
            _async_copy(src=dst_l, dst=dst_l, sem=sem_l, wait=True)

            dst_m = bm_x2_ref.at[bq_sem_idx, :]
            _async_copy(src=dst_m, dst=dst_m, sem=sem_m, wait=True)

    def start_fetch_bkv_batch(seq_idx_start, bkv_sem_idx):
        return _fetch_bkv_batch(seq_idx_start, bkv_sem_idx)

    def wait_fetch_bkv_batch(seq_idx_start, bkv_sem_idx):
        return _fetch_bkv_batch(seq_idx_start, bkv_sem_idx, wait=True)

    def start_fetch_bq_batch(seq_idx_start, bq_sem_idx):
        return _fetch_bq_batch(seq_idx_start, bq_sem_idx)

    def wait_fetch_bq_batch(seq_idx_start, bq_sem_idx):
        return _fetch_bq_batch(seq_idx_start, bq_sem_idx, wait=True)

    def start_fetch_swa_batch(seq_idx_start, bq_sem_idx):
        return _fetch_swa_batch(seq_idx_start, bq_sem_idx)

    def wait_fetch_swa_batch(seq_idx_start, bq_sem_idx):
        return _fetch_swa_batch(seq_idx_start, bq_sem_idx, wait=True)

    def start_send_bo_batch(seq_idx_start, bo_sem_idx):
        return _send_bo_batch(seq_idx_start, bo_sem_idx)

    def wait_send_bo_batch(seq_idx_start, bo_sem_idx):
        return _send_bo_batch(seq_idx_start, bo_sem_idx, wait=True)

    def load_bq(bq_sem_idx, batch_idx):
        q = bq_x2_ref.at[bq_sem_idx, batch_idx][...]
        return q

    def load_bkv(bkv_sem_idx, batch_idx, dsv4_fp8_scale_expand_matrix):
        bkv_nope = bkv_nope_x2_ref.at[bkv_sem_idx, batch_idx][...]
        bkv_nope = _dequant_dsv4_fp8(bkv_nope, dsv4_fp8_scale_expand_matrix)

        bkv_rope = bkv_rope_x2_ref.at[bkv_sem_idx, batch_idx][...]
        bkv = jnp.concatenate([bkv_nope, bkv_rope], axis=-1)

        # In vLLM, multiple caches may overlay on the same KV Tensor. For example,
        # compressor state cache write data in bfloat16 / float32 format, certain
        # byte pattern are interpreted as NaN in FP8, e.g. float8_e8m0fnu byte 0xFF
        # decodes to NaN.
        # We need to mask out the data by the actual kv_len to avoid NaN propagting
        # to the downstream computation.
        kv_len = kv_lens_ref[batch_start_seq_idx + batch_idx]
        k_span = lax.broadcasted_iota(jnp.int32, bkv.shape, 0)
        bkv = jnp.where(k_span < kv_len, bkv, 0)
        return bkv

    def load_swa_output(bq_sem_idx, batch_idx):
        swa_acc = swa_acc_x2_ref[bq_sem_idx, batch_idx, ...]
        swa_l = bl_x2_ref[bq_sem_idx, batch_idx, :num_q_heads][..., None]
        swa_m = bm_x2_ref[bq_sem_idx, batch_idx, :num_q_heads][..., None]
        return swa_acc, swa_l, swa_m

    def process():

        def get_next_seq_ids(seq_idx, bi_sem_idx):
            next_seq_idx = seq_idx + batch_size
            next_bi_sem_idx = lax.select(bi_sem_idx == 0, 1, 0)
            return next_seq_idx, next_bi_sem_idx

        bi_sem_idx = sem_ids_ref[0]
        next_seq_idx, next_bi_sem_idx = get_next_seq_ids(
            batch_start_seq_idx, bi_sem_idx)

        # Prefetch next seq
        @pl.when(next_seq_idx < end_seq_idx)
        def prefetch_next_seq():
            sem_ids_ref[0] = next_bi_sem_idx
            start_fetch_bq_batch(next_seq_idx, next_bi_sem_idx)
            start_fetch_swa_batch(next_seq_idx, next_bi_sem_idx)
            start_fetch_bkv_batch(next_seq_idx, next_bi_sem_idx)

        bo_sem_idx = sem_ids_ref[1]
        sem_ids_ref[1] = lax.select(bo_sem_idx == 0, 1, 0)
        attention_sinks = attention_sinks_ref[...][..., None]
        dsv4_fp8_scale_expand_matrix = _make_dsv4_fp8_scale_expand_matrix()

        prev_p = None
        prev_bkv = None
        prev_exp_m_diff = None
        prev_l = None
        prev_swa_acc = None
        prev_out = None

        # Wait for cur blocks if not ready yet
        wait_fetch_bq_batch(batch_start_seq_idx, bi_sem_idx)
        wait_fetch_swa_batch(batch_start_seq_idx, bi_sem_idx)
        wait_fetch_bkv_batch(batch_start_seq_idx, bi_sem_idx)

        @pl.when(pl.program_id(0) >= 2)
        def _wait_send():
            wait_send_bo_batch(batch_start_seq_idx, bo_sem_idx)

        for batch_idx in range(batch_size):
            bkv = load_bkv(bi_sem_idx, batch_idx, dsv4_fp8_scale_expand_matrix)
            bq = load_bq(bi_sem_idx, batch_idx)

            if prev_out is not None:
                # Artificial dependency to force MXU/VPU interleaving by limiting LLO's QK runahead
                # We use jnp.where to prevent XLA from optimizing the dependency away.
                # prev_out won't be inf in practice.
                bq = jnp.where(prev_out == jnp.inf, prev_out, bq)

            swa_acc, swa_l, swa_m = load_swa_output(bi_sem_idx, batch_idx)

            p, exp_m_diff, l_sum = flash_attention_step1_qk_softmax(
                bq,
                bkv,
                kv_lens_ref[batch_start_seq_idx + batch_idx],
                swa_m,
                swa_l,
                attention_sinks,
            )

            if prev_p is not None:
                assert prev_bkv is not None
                assert prev_exp_m_diff is not None
                assert prev_l is not None
                out = flash_attention_step2_pv(
                    prev_p,
                    prev_bkv,
                    prev_exp_m_diff,
                    prev_swa_acc,
                    prev_l,
                )

                # Store output from acc to bo.
                bo_x2_ref.at[bo_sem_idx, batch_idx - 1][...] = out
                prev_out = out

            prev_p = p
            prev_bkv = bkv
            prev_exp_m_diff = exp_m_diff
            prev_l = l_sum
            prev_swa_acc = swa_acc

        # end of pipelining loop
        out = flash_attention_step2_pv(prev_p, prev_bkv, prev_exp_m_diff,
                                       prev_swa_acc, prev_l)
        bo_x2_ref.at[bo_sem_idx, batch_size - 1][...] = out
        start_send_bo_batch(batch_start_seq_idx, bo_sem_idx)

    ### ------- Kernel start ------- ###

    @pl.when(batch_start_seq_idx == start_seq_idx)
    def prologue():
        start_fetch_bq_batch(batch_start_seq_idx, 0)
        start_fetch_swa_batch(batch_start_seq_idx, 0)
        start_fetch_bkv_batch(batch_start_seq_idx, 0)

    process()

    @pl.when(batch_end_seq_idx == end_seq_idx - 1)
    def epilogue():
        # The first argument "0" for seq_idx_start does not matter here.
        wait_send_bo_batch(0, 0)

        @pl.when(pl.num_programs(0) >= 2)
        def _wait_1():
            # The first argument "0" for seq_idx_start does not matter here.
            wait_send_bo_batch(0, 1)

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


# Main Attention kernel for DeepSeek V4 CSA (gather and attention)
# Note that the compressed kv tokens of current batch (current forward pass)
# have been written to the `cache_kv` by the compressor module before calling
# this function, `kv_lens` reflects the length after compressed kv cache write.
@functools.partial(
    jax.jit,
    static_argnames=(
        "sm_scale",
        "attention_kernel_batch_size",
        "gather_and_attention_chunk_size",
        "vmem_limit_bytes",
    ),
)
def sparse_ragged_paged_attention(
    q: jax.Array,  # [max_num_tokens, actual_num_q_heads, head_dim]
    cache_kv_nope: jax.Array,  # [total_num_pages, page_size, 4, 128]
    cache_kv_rope: jax.Array,  # [total_num_pages, page_size // 4, 4, 128]
    topk_indices: jax.Array,  # i32[max_num_tokens, csa_topk]
    page_indices: jax.Array,  # i32[max_num_seqs * pages_per_seq]
    cu_q_lens: jax.Array,  # i32[max_num_seqs + 1]
    distribution: jax.Array,  # i32[3]
    attention_sinks: jax.Array,  # float32[actual_num_q_heads]
    swa_accumution: jax.Array,  # bf16[max_num_tokens, num_q_heads, head_dim]
    swa_l: jax.Array,  # float32[max_num_tokens, num_q_heads]
    swa_m: jax.Array,  # float32[max_num_tokens, num_q_heads]
    *,
    sm_scale: float = 1.0,
    # Kernel optimization params.
    gather_and_attention_chunk_size: int | None = None,
    attention_kernel_batch_size: int = 16,
    vmem_limit_bytes: int = DEFAULT_VMEM_LIMIT_BYTES,
) -> jax.Array:
    """MLA Ragged paged attention that supports mixed prefill and decode.

  Args:
    q: concatenated all sequences' queries.
    cache_kv_nope: the current kv cache for nope.
    cache_kv_rope: the current kv cache for rope.
    topk_indices: for each query token, the indices of the top k key tokens to
      attend to.
    page_indices: flattened page indices look-up table by (seq_id, page_id).
    cu_q_lens: the cumulative sum of the effective query lengths. Similar to
      kv_lens, only the first num_seqs+1 values are valid.
    distribution: (i, j, k) represents that sequences[0:i] are decode-only,
      sequences[i:j] are chunked-prefill-only, and sequences[j:k] are mixed. The
      k is also the total number of sequences.
    sm_scale: the softmax scale which will be applied to the Q@K^T.
    vmem_limit_bytes: the vmem limit for the pallas kernel.

  Returns:
    The output of attention.
  """
    # The cache is DSV4 FP8 format.
    # nope_cache contains 448 fp8 + 7 fp8 scales,
    # rope_cache contains 64 bf16
    assert cache_kv_nope.dtype == jnp.uint8
    assert cache_kv_rope.dtype == jnp.uint8
    if gather_and_attention_chunk_size is None:
        gather_and_attention_chunk_size = q.shape[0]

    _, actual_num_q_heads, actual_head_dim = q.shape

    q = prepare_q_inputs(q)  # [max_num_tokens, num_q_heads, head_dim]
    head_dim = q.shape[-1]
    attention_sinks = jnp.pad(
        attention_sinks,
        (0, q.shape[1] - actual_num_q_heads),
        constant_values=jnp.finfo(attention_sinks.dtype).min,
    )
    assert swa_accumution.dtype == q.dtype
    swa_accumution, swa_l, swa_m = prepare_swa_inputs(swa_accumution, swa_l,
                                                      swa_m)

    _, page_size, _, _ = cache_kv_nope.shape

    _, num_q_heads, _ = q.shape
    max_num_seqs = cu_q_lens.shape[0] - 1
    num_page_indices = page_indices.shape[0]
    assert num_page_indices % max_num_seqs == 0

    def run_mla_kernel(
        q: jax.Array,  # [max_num_tokens, num_q_heads, head_dim]
        cache_kv_nope: jax.Array,  # [total_num_pages, page_size, nope_dim]
        cache_kv_rope: jax.Array,  # [total_num_pages, page_size, rope_dim]
        kv_lens: jax.Array,  # i32[max_num_seqs]
        attention_sinks: jax.Array,  # float32[num_q_heads]
        swa_accumution: jax.
        Array,  # bf16[max_num_tokens, num_q_heads, head_dim]
        swa_l: jax.Array,  # float32[max_num_tokens, num_l_heads]
        swa_m: jax.Array,  # float32[max_num_tokens, num_l_heads]
        start_seq_idx: jax.Array,  # i32
        end_seq_idx: jax.Array,  # i32
        kernel_batch_size: int,
    ):
        batch_size = kernel_batch_size
        end_seq_idx = jnp.maximum(start_seq_idx, end_seq_idx)
        grid = (cdiv(end_seq_idx - start_seq_idx, batch_size), )
        in_specs = [
            pl.BlockSpec(memory_space=pltpu.VMEM),  # attention_sinks
            pl.BlockSpec(memory_space=pltpu.HBM),  # q
            pl.BlockSpec(memory_space=pltpu.HBM),  # cache_kv_nope
            pl.BlockSpec(memory_space=pltpu.HBM),  # cache_kv_rope
            pl.BlockSpec(memory_space=pltpu.HBM),  # swa_accumution
            pl.BlockSpec(memory_space=pltpu.HBM),  # swa_l
            pl.BlockSpec(memory_space=pltpu.HBM),  # swa_m
        ]

        out_specs = pl.BlockSpec(memory_space=pltpu.HBM)  # o

        page_size = cache_kv_nope.shape[1]
        bkv_nope_double_buf = pltpu.VMEM(
            (2, batch_size, page_size, *cache_kv_nope.shape[2:]),
            cache_kv_nope.dtype,
        )
        bkv_rope_double_buf = pltpu.VMEM(
            (2, batch_size, page_size, *cache_kv_rope.shape[2:]),
            cache_kv_rope.dtype,
        )

        bq_double_bufq = pltpu.VMEM(
            (2, batch_size, num_q_heads, head_dim),
            q.dtype,
        )

        bo_double_buf = bq_double_bufq

        num_l_heads = align_to(num_q_heads, 128)
        bl_double_buf = pltpu.VMEM(
            (2, batch_size, num_l_heads),
            jnp.float32,
        )
        bm_double_buf = bl_double_buf

        swa_acc_double_buf = pltpu.VMEM(
            (2, batch_size, num_q_heads, head_dim),
            q.dtype,
        )

        scratch_shapes = [
            bkv_nope_double_buf,
            bkv_rope_double_buf,
            bq_double_bufq,
            bo_double_buf,  # Double buffering for output block.
            bl_double_buf,  # Double buffering for l output.
            bm_double_buf,  # Double buffering for m output.
            swa_acc_double_buf,  # Buffer for swa_accumution.
            # Semaphores for double buffering of bkv_nope, bq, bo, swa_acc, swa_l, swa_m, bkv_rope
            pltpu.SemaphoreType.DMA((7, 2)),
        ]

        scalar_prefetches = (
            kv_lens,
            jnp.array([start_seq_idx, end_seq_idx], jnp.int32),
            # (bi_sem_idx, bo_sem_idx)
            jnp.zeros((2, ), jnp.int32),
        )

        scope_name = f"SparseMLA-p_{page_size}-bz_{batch_size}-gcz_{gather_and_attention_chunk_size}"
        kernel = jax.named_scope(scope_name)(
            pl.pallas_call(
                functools.partial(
                    _attention_kernel,
                    sm_scale=sm_scale,
                    batch_size=batch_size,
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
                    4: 0,  # Alias output activation with q
                },
                name=scope_name,
            ))
        return kernel(
            *scalar_prefetches,
            attention_sinks,
            q,
            cache_kv_nope,
            cache_kv_rope,
            swa_accumution,
            swa_l,
            swa_m,
        )

    tokens_per_seq = cu_q_lens[1:] - cu_q_lens[:-1]
    seq_ids_segment = jnp.repeat(jnp.arange(max_num_seqs),
                                 tokens_per_seq,
                                 total_repeat_length=q.shape[0])
    assert topk_indices is not None
    # TODO: skip gather for padding tokens in topk_indices.
    kv_lens = jnp.sum(topk_indices != -1, axis=-1)

    seq_page_ids = topk_indices // page_size
    token_offset = topk_indices % page_size
    topk = topk_indices.shape[-1]
    page_ids = gather_page_ids(page_indices, seq_page_ids, seq_ids_segment,
                               max_num_seqs)

    # For the "-1" padding elements in topk_indices, we scatter the corresponding
    # page_ids and token_offset to avoid gather memory access hotspotting.
    is_padding = topk_indices == -1
    total_num_pages = cache_kv_nope.shape[0]
    flat_element_index = jnp.arange(q.shape[0] * topk,
                                    dtype=jnp.int32).reshape(q.shape[0], topk)
    # 104729 and 15485863 are randomly chosen large prime numbers.
    scattered_page_ids = (flat_element_index * 104729) % total_num_pages
    scattered_token_offset = (flat_element_index * 15485863) % page_size
    page_ids = jnp.where(is_padding, scattered_page_ids, page_ids)
    token_offset = jnp.where(
        is_padding,
        scattered_token_offset,
        token_offset,
    )

    assert page_ids.shape == (q.shape[0], topk)

    # TODO: handle the case where q.shape[0] is not divisible by
    # gather_and_attention_chunk_size.
    assert q.shape[0] % gather_and_attention_chunk_size == 0
    num_chunks = q.shape[0] // gather_and_attention_chunk_size

    for i in range(num_chunks):
        start_pos = i * gather_and_attention_chunk_size
        end_pos = start_pos + gather_and_attention_chunk_size
        indices = (page_ids[start_pos:end_pos, ...] * page_size +
                   token_offset[start_pos:end_pos, ...]).reshape(-1)

        # For prefilling of short sequences (or early in the sequence), there are
        # very few number of KVs in the sequence, so different qs' selected topk
        # would have large overlap. This causes gather read hotspotting. We've seen
        # 30%+ performance degradation compared to the no-duplicate-indices case.
        #
        # TODO: we could consider let the caller (tpu-runner) to sort the sequences
        # based on their lengths. For the sequences-segment below certain length,
        # we use a different kernel (dense attention and mask), for the rest of
        # sequences, we use this gather-and-attention kernel.
        gathered_nope_buffer, gathered_rope_buffer = csa_gather.csa_gather(
            cache_kv_nope,
            cache_kv_rope,
            indices,
        )
        gathered_nope_buffer = gathered_nope_buffer.reshape(
            gather_and_attention_chunk_size, topk, -1)
        gathered_rope_buffer = gathered_rope_buffer.reshape(
            gather_and_attention_chunk_size, topk, -1)
        # We treat each query token as a one independent sequence, attend to their
        # respective gathered kv tokens in the `gathered_kv_buffer`.
        # -1 in topk_indices is padded elements at the end of each row.
        # Batching
        assert gather_and_attention_chunk_size % attention_kernel_batch_size == 0
        batch_end = (cdiv(
            jnp.minimum(
                cu_q_lens[distribution[2]],
                start_pos + gather_and_attention_chunk_size,
            ),
            attention_kernel_batch_size,
        ) * attention_kernel_batch_size)
        q = run_mla_kernel(
            q,
            gathered_nope_buffer,
            gathered_rope_buffer,
            kv_lens,
            attention_sinks,
            swa_accumution,
            swa_l,
            swa_m,
            start_seq_idx=start_pos,
            end_seq_idx=batch_end,
            kernel_batch_size=attention_kernel_batch_size,
        )
    return prepare_outputs(
        q, actual_num_q_heads, actual_head_dim
    )  # [max_num_tokens, actual_num_q_heads, actual_head_dim]
