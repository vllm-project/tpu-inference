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
"""TPU-Friendly StreamIndex Top-K kernel."""

import enum
import functools

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

Enum = enum.Enum
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


def _streamindex_topk_kernel(
    # Prefetch
    seq_lens_ref,  # [max_num_seqs]
    page_indices_ref,  # [max_num_seqs * pages_per_seq]
    cu_q_lens_ref,  # [max_num_seqs + 1]
    start_end_seq_idx_ref,  # [2] (start_seq_idx, end_seq_idx)
    sem_ids_ref,  # [3] (bq_sem_idx, bkv_sem_idx, bo_sem_idx)
    bo_ids_ref,  # [4] (bo_sem_0_seq_idx, bo_sem_1_seq_idx, bo_sem_0_bo_idx, bo_sem_1_bo_idx)
    # Input
    q_hbm_ref,  # [max_num_tokens, num_q_heads, head_dim]
    indexer_weights_hbm_ref,  # [max_num_tokens, num_q_heads]
    cache_kv_hbm_ref,  # [total_num_pages, page_size_per_kv_packing, kv_packing, lkv_dim]
    # Output
    topk_idxs_hbm_ref,  # [max_num_tokens, k]
    # Scratch
    bkv_x2_ref,  # [2, bkv_buf_sz_per_kv_packing, kv_packing, lkv_dim]
    bq_x2_ref,  # [2, bq_sz, num_q_heads, q_packing, head_dim]
    bq_weights_x2_ref,  # [2, bq_sz, num_q_heads]
    bo_idxs_x2_ref,  # [2, bq_sz, k]
    sems,  # [4, 2]
    topk_vals_scratch,  # [bq_sz, k]
    topk_idxs_scratch,  # [bq_sz, k]
    *,
    k: int,
    compression_ratio: int,
    static_q_len: int,
    bkv_p: int,
    bq_sz: int,
    actual_num_q_heads: int,
    actual_head_dim: int,
    kv_packing: int,
):
    _, num_q_heads, head_dim = q_hbm_ref.shape

    total_num_pages, page_size_per_kv_packing, _, _ = cache_kv_hbm_ref.shape

    max_num_seqs = seq_lens_ref.shape[0]
    num_page_indices = page_indices_ref.shape[0]

    assert num_page_indices % max_num_seqs == 0
    pages_per_seq = num_page_indices // max_num_seqs

    # Validate against the KV dtype.
    kv_dtype = cache_kv_hbm_ref.dtype
    assert get_dtype_packing(kv_dtype) == kv_packing
    assert head_dim % 128 == 0

    bkv_sz_per_kv_packing = bkv_p * page_size_per_kv_packing
    bkv_sz = bkv_sz_per_kv_packing * kv_packing

    start_seq_idx = start_end_seq_idx_ref[0]
    end_seq_idx = start_end_seq_idx_ref[1]
    seq_idx = pl.program_id(0) + start_seq_idx

    q_start = cu_q_lens_ref[seq_idx]
    q_end = cu_q_lens_ref[seq_idx + 1]
    q_len = q_end - q_start

    seq_len = seq_lens_ref[seq_idx]
    kv_len = seq_len // compression_ratio

    def compute_topk(
        q,
        kv,
        scale_val,
        *,
        bkv_idx,
        bq_pos_compressed,
        bq_weights,
    ):
        head_vals_ref = topk_vals_scratch.at[:bq_sz]
        head_idxs_ref = topk_idxs_scratch.at[:bq_sz]

        q_flat = q.reshape(
            bq_sz * actual_num_q_heads,
            head_dim,
        )
        s = jnp.einsum(
            "nd,md->nm",
            q_flat,
            kv,
            preferred_element_type=jnp.float32,
        )
        s = s.reshape(bq_sz, actual_num_q_heads, s.shape[-1])
        s = s * scale_val.reshape(1, 1, -1)
        s = jnp.maximum(s, 0.0)
        s = s * bq_weights.astype(jnp.float32)[:, :, None]
        s_summed = s.sum(axis=1)

        k_span = bkv_idx * bkv_sz + lax.broadcasted_iota(
            jnp.int32, s_summed.shape, 1)

        valid_mask = k_span < kv_len
        causal_mask = k_span <= bq_pos_compressed[:, None]
        mask = jnp.logical_and(valid_mask, causal_mask)

        s_summed = jnp.where(mask, s_summed, -jnp.inf)

        prev_vals = head_vals_ref[...]
        prev_idxs = head_idxs_ref[...]

        k_span_bcast = jnp.broadcast_to(k_span, s_summed.shape)
        k_span_bcast = jnp.where(mask, k_span_bcast, -1)

        concat_s = jnp.concatenate([prev_vals, s_summed], axis=1)
        concat_i = jnp.concatenate([prev_idxs, k_span_bcast], axis=1)

        col_indices = lax.broadcasted_iota(jnp.int32, concat_s.shape, 1)
        s_work = concat_s

        new_vals_list = []
        new_idxs_list = []

        for _ in range(k):
            max_val = jnp.max(s_work, axis=1, keepdims=True)
            is_max = s_work == max_val
            idx = jnp.min(
                jnp.where(is_max, col_indices,
                          jnp.iinfo(jnp.int32).max),
                axis=1,
                keepdims=True,
            )

            is_chosen = col_indices == idx
            orig_idx = jnp.max(jnp.where(is_chosen, concat_i, -1),
                               axis=1,
                               keepdims=True)

            orig_idx = jnp.where(max_val <= -jnp.inf, -1, orig_idx)
            max_val = jnp.where(max_val <= -jnp.inf, -jnp.inf, max_val)

            new_vals_list.append(max_val)
            new_idxs_list.append(orig_idx)

            s_work = jnp.where(is_chosen, -jnp.inf, s_work)

        new_vals = jnp.concatenate(new_vals_list, axis=1)
        new_idxs = jnp.concatenate(new_idxs_list, axis=1)

        head_vals_ref[...] = new_vals
        head_idxs_ref[...] = new_idxs

    def _async_copy(src, dst, sem, wait):
        cp = pltpu.make_async_copy(src, dst, sem)
        if wait:
            cp.wait()
        else:
            cp.start()

    def _fetch_bkv(seq_idx, bkv_idx, bkv_sem_idx, *, wait=False):
        sem = sems.at[0, bkv_sem_idx]
        bkv_vmem_ref = bkv_x2_ref.at[bkv_sem_idx]

        reshaped_cache_hbm_ref = cache_kv_hbm_ref.reshape(
            total_num_pages * page_size_per_kv_packing,
            kv_packing,
            actual_head_dim,
        )

        kv_p_start = bkv_idx * bkv_p
        page_indices_offset = seq_idx * pages_per_seq + kv_p_start

        if not wait:
            for i in range(bkv_p):
                sz_per_kv_packing = page_size_per_kv_packing
                page_idx = jnp.minimum(page_indices_offset + i,
                                       num_page_indices - 1)

                max_hbm_pages = reshaped_cache_hbm_ref.shape[0]
                safe_page_offset = jnp.minimum(
                    page_indices_ref[page_idx] * page_size_per_kv_packing,
                    jnp.maximum(0, max_hbm_pages - page_size_per_kv_packing),
                )

                _async_copy(
                    reshaped_cache_hbm_ref.at[pl.ds(safe_page_offset,
                                                    sz_per_kv_packing)],
                    bkv_vmem_ref.at[pl.ds(i * page_size_per_kv_packing,
                                          sz_per_kv_packing)],
                    sem,
                    wait=False,
                )
        else:
            dma_bkv_sz = bkv_p * page_size_per_kv_packing
            dst_kv = bkv_vmem_ref.at[pl.ds(0, dma_bkv_sz)]
            _async_copy(src=dst_kv, dst=dst_kv, sem=sem, wait=True)

    def _fetch_bq(seq_idx, bq_idx, bq_sem_idx, *, wait=False):
        sem = sems.at[1, bq_sem_idx]
        weights_sem = sems.at[3, bq_sem_idx]
        bq_vmem_ref = bq_x2_ref.at[bq_sem_idx]
        bq_weights_vmem_ref = bq_weights_x2_ref.at[bq_sem_idx]

        q_len_start = cu_q_lens_ref[seq_idx] + bq_idx * bq_sz
        curr_q_end = cu_q_lens_ref[seq_idx + 1]

        sz = jnp.maximum(0, jnp.minimum(bq_sz, curr_q_end - q_len_start))
        safe_q_start = jnp.minimum(
            q_len_start, jnp.maximum(0,
                                     q_hbm_ref.shape[0] - jnp.maximum(1, sz)))

        if not wait:
            _async_copy(
                q_hbm_ref.at[pl.ds(safe_q_start, sz)],
                bq_vmem_ref.at[pl.ds(0, sz)],
                sem,
                wait=False,
            )
            _async_copy(
                indexer_weights_hbm_ref.at[pl.ds(safe_q_start, sz)],
                bq_weights_vmem_ref.at[pl.ds(0, sz)],
                weights_sem,
                wait=False,
            )
        else:
            dst = bq_vmem_ref.at[pl.ds(0, sz)]
            _async_copy(dst, dst, sem, wait=True)
            dst_w = bq_weights_vmem_ref.at[pl.ds(0, sz)]
            _async_copy(dst_w, dst_w, weights_sem, wait=True)

    def _send_bo(seq_idx, bo_idx, bo_sem_idx, *, wait=False):
        sem_idxs = sems.at[2, bo_sem_idx]

        q_len_start = cu_q_lens_ref[seq_idx] + bo_idx * bq_sz
        curr_q_end = cu_q_lens_ref[seq_idx + 1]

        sz = jnp.maximum(0, jnp.minimum(bq_sz, curr_q_end - q_len_start))
        safe_q_start = jnp.minimum(
            q_len_start,
            jnp.maximum(0, topk_idxs_hbm_ref.shape[0] - jnp.maximum(1, sz)),
        )

        if not wait:
            _async_copy(
                bo_idxs_x2_ref.at[bo_sem_idx, pl.ds(0, sz)],
                topk_idxs_hbm_ref.at[pl.ds(safe_q_start, sz)],
                sem_idxs,
                wait=False,
            )
        else:
            dst_i = bo_idxs_x2_ref.at[bo_sem_idx, pl.ds(0, sz)]
            _async_copy(dst_i, dst_i, sem_idxs, wait=True)

    def start_fetch_bkv(seq_idx, bkv_idx, bkv_sem_idx):
        _fetch_bkv(seq_idx, bkv_idx, bkv_sem_idx)

    def wait_fetch_bkv(seq_idx, bkv_idx, bkv_sem_idx):
        _fetch_bkv(seq_idx, bkv_idx, bkv_sem_idx, wait=True)

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

    def load_bq(bq_sem_idx):
        q = bq_x2_ref.at[bq_sem_idx, :bq_sz][...]
        q = q.reshape(bq_sz, num_q_heads, head_dim)
        return q[:, :actual_num_q_heads, :actual_head_dim]

    def load_bq_weights(bq_sem_idx):
        w = bq_weights_x2_ref.at[bq_sem_idx, :bq_sz][...]
        return w[:, :actual_num_q_heads]

    def load_bkv(bkv_sem_idx):
        bkv = bkv_x2_ref.at[bkv_sem_idx, :bkv_sz_per_kv_packing][...]
        # Quantized FP8 index cache path: unpack keys and scale factors locally.
        flat_bkv = bkv.reshape(-1, bkv.shape[-1])
        fp8_val = flat_bkv[:, :head_dim]
        fp8_val = jax.lax.bitcast_convert_type(fp8_val, jnp.float8_e4m3fn)

        scale_val = flat_bkv[:, head_dim:head_dim + 1]
        scale_val = jax.lax.bitcast_convert_type(
            scale_val, jnp.float8_e8m0fnu).astype(jnp.float32)

        # NOTE: Do NOT multiply the scales here. Return them separately.
        return fp8_val.reshape(bkv_sz, head_dim), scale_val.reshape(bkv_sz, 1)

    # --- KERNEL PROCESS LOOP ---

    def process():
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
            @pl.when(next_seq_idx < end_seq_idx)
            def prefetch_next_bq():
                sem_ids_ref[0] = next_bq_sem_idx
                start_fetch_bq(next_seq_idx, next_bq_idx, next_bq_sem_idx)

            # Initialize scratch space for top-K values and indices.
            # TODO(hwanginho): Use b16 or fp8 format for top-K values scratch if
            # possible.
            topk_vals_scratch[...] = jnp.full((bq_sz, k),
                                              -jnp.inf,
                                              dtype=jnp.float32)
            topk_idxs_scratch[...] = jnp.full((bq_sz, k), -1, dtype=jnp.int32)

            q_pos = (seq_len - q_len + bq_idx * bq_sz +
                     jnp.arange(bq_sz, dtype=jnp.int32))
            bq_pos_compressed = q_pos // compression_ratio

            def compute_with_bkv(bkv_idx, _):
                bkv_sem_idx = sem_ids_ref[1]
                next_seq_idx, _, next_bkv_idx, next_bkv_sem_idx = get_next_bkv_ids(
                    seq_idx, bq_idx, bkv_idx, bkv_sem_idx)

                # Prefetch next bkv
                @pl.when(next_seq_idx < end_seq_idx)
                def prefetch_next_bkv():
                    sem_ids_ref[1] = next_bkv_sem_idx
                    start_fetch_bkv(next_seq_idx, next_bkv_idx,
                                    next_bkv_sem_idx)

                # Wait for cur bq if not ready yet
                @pl.when(bkv_idx == 0)
                def wait_cur_bq():
                    wait_fetch_bq(seq_idx, bq_idx, bq_sem_idx)

                # Wait for cur bkv
                wait_fetch_bkv(seq_idx, bkv_idx, bkv_sem_idx)

                bkv, scale_val = load_bkv(bkv_sem_idx)
                bq = load_bq(bq_sem_idx)
                bq_weights = load_bq_weights(bq_sem_idx)

                compute_topk(
                    bq,
                    bkv,
                    scale_val,
                    bkv_idx=bkv_idx,
                    bq_pos_compressed=bq_pos_compressed,
                    bq_weights=bq_weights,
                )

            lax.fori_loop(0, num_bkv, compute_with_bkv, None, unroll=False)

            out_idxs = topk_idxs_scratch[...]

            bo_sem_idx = sem_ids_ref[2]
            sem_ids_ref[2] = lax.select(bo_sem_idx == 0, jnp.int32(1),
                                        jnp.int32(0))

            wait_send_bo(bo_sem_idx)

            bo_idxs_x2_ref[bo_sem_idx, ...] = out_idxs.reshape(
                bo_idxs_x2_ref[bo_sem_idx].shape)

            start_send_bo(seq_idx, bq_idx, bo_sem_idx)

        lax.fori_loop(0, num_bq, compute_with_bq, None, unroll=False)

    ### ------- Kernel start ------- ###

    @pl.when(seq_idx == start_seq_idx)
    def prologue():
        start_fetch_bq(start_seq_idx, 0, 0)

        # Initialize bkv_x2_ref to zeros to avoid NaN issues from accessing
        # uninitialized memory. Bitcast into int32 to avoid tiling issues.
        bkv_x2_int32_ref = bkv_x2_ref.bitcast(jnp.int32).reshape(
            (2, -1, actual_head_dim))
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


def prepare_index_weights(
    index_weights: jax.Array,  # [max_num_tokens, actual_num_q_heads],
    q_dtype,
):
    _, actual_num_q_heads = index_weights.shape
    index_weights = index_weights.astype(jnp.float32)
    num_q_heads = align_to(actual_num_q_heads, get_dtype_packing(q_dtype))
    index_weights = jnp.pad(
        index_weights,
        (
            (0, 0),
            (0, num_q_heads - actual_num_q_heads),
        ),
        constant_values=0,
    )
    return index_weights


def prepare_outputs(out):
    if out.ndim == 3:
        out = out.reshape(out.shape[0], -1)
    return out


@functools.partial(
    jax.jit,
    static_argnames=(
        "k",
        "compression_ratio",
        "num_kv_pages_per_block",
        "num_queries_per_block",
        "vmem_limit_bytes",
    ),
)
def streamindex_topk(
    q: jax.Array,  # [max_num_tokens, actual_num_q_heads, actual_head_dim]
    indexer_weights: jax.Array,  # [max_num_tokens, actual_num_q_heads]
    cache_kv: jax.
    Array,  # [total_num_pages, page_size_per_kv_packing, kv_packing, head_dim]
    seq_lens: jax.Array,  # i32[max_num_seqs]
    page_indices: jax.Array,  # i32[max_num_seqs * pages_per_seq]
    cu_q_lens: jax.Array,  # i32[max_num_seqs + 1]
    distribution: jax.Array,  # i32[3]
    *,
    k: int,
    compression_ratio: int,
    num_kv_pages_per_block: tuple[int, int, int] | int | None = None,
    num_queries_per_block: tuple[int, int, int] | int | None = None,
    vmem_limit_bytes: int = DEFAULT_VMEM_LIMIT_BYTES,
) -> jax.Array:
    """StreamIndex Top-K retrieval.

  Args:
    q: concatenated all sequences' queries.
    indexer_weights: concatenated all sequences' indexer weights.
    cache_kv: the current kv cache.
    seq_lens: the length of each sequence in the kv cache (uncompressed).
    page_indices: flattened page indices look-up table by (seq_id, page_id).
    cu_q_lens: the cumulative sum of the effective query lengths. Similar to
      kv_lens, only the first num_seqs+1 values are valid.
    distribution: (i, j, k) represents that sequences[0:i] are decode-only,
      sequences[i:j] are chunked-prefill-only, and sequences[j:k] are mixed. The
      k is also the total number of sequences.
    k: Number of top-K elements to retrieve.
    compression_ratio: KV cache compression ratio.
    num_kv_pages_per_block: number of kv pages to be processed in one block in
      the pallas kernel. This is a tuple of (decode, prefill, mixed) cases.
    num_queries_per_block: number of queries to be processed in one block in the
      pallas kernel. This is a tuple of (decode, prefill, mixed) cases.
    vmem_limit_bytes: the vmem limit for the pallas kernel.

  Returns:
    Top-K indices (in compressed space).
  """
    # Scale factors for the FP8 index cache format are packed directly inside
    # `cache_kv` along the width dimension, keeping HBM transactions fused.

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

    max_num_seqs = seq_lens.shape[0]

    original_dtype = q.dtype

    prepared_indexer_weights = prepare_index_weights(
        indexer_weights.astype(original_dtype), original_dtype)

    actual_num_q_heads = q.shape[1]
    q = prepare_q_inputs(q)
    actual_head_dim = cache_kv.shape[-1]

    def run_topk_kernel(
        q,
        prepared_indexer_weights,
        cache_kv,
        seq_lens,
        page_indices,
        cu_q_lens,
        start_seq_idx,
        end_seq_idx,
        static_q_len,
        num_kv_pages_per_block,
        num_queries_per_block,
        case=MlaCase.MIXED,
    ):
        # Dynamically extract parameters for the DMA mapping from the RAW tensor
        _, page_size_per_kv_packing, kv_packing, _ = cache_kv.shape

        _, num_q_heads, head_dim = q.shape

        bkv_p = num_kv_pages_per_block
        if static_q_len is not None:
            bq_sz = min(num_queries_per_block, static_q_len)
        else:
            bq_sz = num_queries_per_block
        bkv_sz_per_kv_packing = bkv_p * page_size_per_kv_packing
        bkv_buf_sz_per_kv_packing = bkv_sz_per_kv_packing

        grid = (end_seq_idx - start_seq_idx, )

        in_specs = [
            pl.BlockSpec(memory_space=pltpu.HBM),  # q
            pl.BlockSpec(memory_space=pltpu.HBM),  # prepared_indexer_weights
            pl.BlockSpec(memory_space=pltpu.HBM),  # cache_kv
        ]
        out_specs = pl.BlockSpec(memory_space=pltpu.HBM)  # out_idxs
        assert k % 128 == 0
        topk_shape = (k // 128, 128)

        # Group VMEM layout securely by keeping physical shapes separated!
        bkv_double_buf = pltpu.VMEM(
            (
                2,
                bkv_buf_sz_per_kv_packing,
                kv_packing,
                actual_head_dim,
            ),
            cache_kv.dtype,
        )
        bq_double_bufq = pltpu.VMEM(
            (
                2,
                bq_sz,
                num_q_heads,
                head_dim,
            ),
            q.dtype,
        )
        bq_weights_double_buf = pltpu.VMEM(
            (
                2,
                bq_sz,
                num_q_heads,
            ),
            prepared_indexer_weights.dtype,
        )
        bo_idxs_double_buf = pltpu.VMEM(
            (
                2,
                bq_sz,
                *topk_shape,
            ),
            jnp.int32,
        )

        scratch_shapes = [
            bkv_double_buf,
            bq_double_bufq,
            bq_weights_double_buf,
            bo_idxs_double_buf,
            pltpu.SemaphoreType.DMA((4, 2)),
            pltpu.VMEM((bq_sz, k), jnp.float32),
            pltpu.VMEM((bq_sz, k), jnp.int32),
        ]

        scalar_prefetches = (
            seq_lens,
            page_indices,
            cu_q_lens,
            jnp.array([start_seq_idx, end_seq_idx], jnp.int32),
            jnp.zeros((3, ), jnp.int32),
            jnp.full((4, ), -1, jnp.int32),
        )

        scope_name = f"StreamIdxTopK-{case.symbol}-bq_{bq_sz}-bkvp_{bkv_p}"

        kernel = jax.named_scope(scope_name)(pl.pallas_call(
            functools.partial(
                _streamindex_topk_kernel,
                k=k,
                compression_ratio=compression_ratio,
                static_q_len=static_q_len,
                bq_sz=bq_sz,
                bkv_p=bkv_p,
                actual_num_q_heads=actual_num_q_heads,
                actual_head_dim=actual_head_dim,
                kv_packing=kv_packing,
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
            out_shape=jax.ShapeDtypeStruct(shape=(q.shape[0], *topk_shape),
                                           dtype=jnp.int32),
            name=scope_name,
        ))
        return kernel(
            *scalar_prefetches,
            q,
            prepared_indexer_weights,
            cache_kv,
        )

    decode_end = jnp.minimum(max_num_seqs, jnp.maximum(0, distribution[0]))

    q_idxs_decode = run_topk_kernel(
        q,
        prepared_indexer_weights,
        cache_kv,
        seq_lens,
        page_indices,
        cu_q_lens,
        num_kv_pages_per_block=num_kv_pages_per_blocks[0],
        num_queries_per_block=num_queries_per_blocks[0],
        start_seq_idx=jnp.array(0),
        end_seq_idx=distribution[0],
        static_q_len=1,
        case=MlaCase.DECODE,
    )

    q_idxs_mixed = run_topk_kernel(
        q,
        prepared_indexer_weights,
        cache_kv,
        seq_lens,
        page_indices,
        cu_q_lens,
        num_kv_pages_per_block=num_kv_pages_per_blocks[2],
        num_queries_per_block=num_queries_per_blocks[2],
        start_seq_idx=distribution[1],
        end_seq_idx=distribution[2],
        static_q_len=None,
        case=MlaCase.MIXED,
    )

    decode_tokens_end = cu_q_lens[decode_end]

    topk_idxs_decode = prepare_outputs(q_idxs_decode)
    topk_idxs_mixed = prepare_outputs(q_idxs_mixed)

    token_indices = jnp.arange(topk_idxs_decode.shape[0])[:, None]
    mask = token_indices < decode_tokens_end

    topk_idxs = jnp.where(mask, topk_idxs_decode, topk_idxs_mixed)

    return topk_idxs
