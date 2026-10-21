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


def _scores_kernel(
    # Prefetch
    seq_lens_ref,  # [max_num_seqs]
    page_indices_ref,  # [max_num_seqs * pages_per_seq]
    cu_q_lens_ref,  # [max_num_seqs + 1]
    start_end_seq_idx_ref,  # [2] (start_seq_idx, end_seq_idx)
    sem_ids_ref,  # [3] (bq_sem_idx, bkv_sem_idx, bo_sem_idx)
    bo_sz_ref,  # [2] row count of each output buffer's in-flight DMA (-1 = none)
    # Input
    q_hbm_ref,  # [max_num_tokens, num_q_heads, head_dim]
    indexer_weights_hbm_ref,  # [max_num_tokens, num_q_heads]
    cache_kv_hbm_ref,  # [total_num_pages, page_size_per_kv_packing, kv_packing, lkv_dim]
    scores_in_hbm_ref,  # aliased to the output
    # Output
    scores_hbm_ref,  # [max_num_tokens, num_sublanes_total, 128]
    # Scratch
    bkv_x2_ref,  # [2, bkv_buf_sz_per_kv_packing, kv_packing, lkv_dim]
    bq_x2_ref,  # [2, bq_sz, num_q_heads, head_dim]
    bq_weights_x2_ref,  # [2, bq_sz, num_q_heads]
    scores_block_x2_ref,  # [2, bq_sz, num_sublanes_bkv, 128]
    sems,  # [4, 2]
    *,
    compression_ratio: int,
    static_q_len: int,
    bkv_p: int,
    bq_sz: int,
    seq_batch_size: int,
):
    _, num_q_heads, head_dim = q_hbm_ref.shape
    lkv_dim = cache_kv_hbm_ref.shape[-1]

    total_num_pages, page_size_per_kv_packing, kv_packing, _ = (
        cache_kv_hbm_ref.shape)

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
    num_sublanes_bkv = bkv_sz // 128

    start_seq_idx = start_end_seq_idx_ref[0]
    end_seq_idx = start_end_seq_idx_ref[1]
    batch_start_seq_idx = start_seq_idx + pl.program_id(0) * seq_batch_size
    batch_end_seq_idx = batch_start_seq_idx + seq_batch_size - 1

    q_lens = []
    kv_lens = []
    seq_lens = []
    for batch_idx in range(seq_batch_size):
        q_start = cu_q_lens_ref[batch_start_seq_idx + batch_idx]
        q_end = cu_q_lens_ref[batch_start_seq_idx + batch_idx + 1]
        q_len = q_end - q_start
        q_lens.append(q_len)
        seq_len = seq_lens_ref[batch_start_seq_idx + batch_idx]
        seq_lens.append(seq_len)
        kv_len = seq_len // compression_ratio
        kv_lens.append(kv_len)

    def wait_send_scores(bo_sem_idx):
        # Wait for the output buffer's previous DMA (if any) before reusing it.
        old_sz = bo_sz_ref[bo_sem_idx]

        @pl.when(old_sz >= 0)
        def _():
            dst = scores_block_x2_ref.at[bo_sem_idx,
                                         pl.ds(0, seq_batch_size * old_sz)]
            _async_copy(dst, dst, sems.at[2, bo_sem_idx, 0], wait=True)

    def start_send_scores(bo_sem_idx, sz, token_start, bkv_idx):
        # All sequences in the batch have the same sz. Issue a single DMA for the
        # entire batch.
        bo_sz_ref[bo_sem_idx] = sz
        sublane_start = bkv_idx * num_sublanes_bkv
        _async_copy(
            scores_block_x2_ref.at[bo_sem_idx,
                                   pl.ds(0, seq_batch_size * sz)],
            scores_hbm_ref.at[
                pl.ds(token_start, seq_batch_size * sz),
                pl.ds(sublane_start, num_sublanes_bkv),
            ],
            sems.at[2, bo_sem_idx, 0],
            wait=False,
        )

    def _async_copy(src, dst, sem, wait):
        cp = pltpu.make_async_copy(src, dst, sem)
        if wait:
            cp.wait()
        else:
            cp.start()

    def _fetch_bkv(seq_idx, bkv_idx, bkv_sem_idx, *, wait=False):
        reshaped_cache_hbm_ref = cache_kv_hbm_ref.reshape(
            total_num_pages * page_size_per_kv_packing,
            kv_packing,
            lkv_dim,
        )
        max_hbm_pages = reshaped_cache_hbm_ref.shape[0]

        for batch_idx in range(seq_batch_size):
            sem = sems.at[0, bkv_sem_idx, batch_idx]
            bkv_vmem_ref = bkv_x2_ref.at[bkv_sem_idx, batch_idx]

            kv_p_start = bkv_idx * bkv_p
            page_indices_offset = (seq_idx +
                                   batch_idx) * pages_per_seq + kv_p_start

            if not wait:
                for i in range(bkv_p):
                    sz_per_kv_packing = page_size_per_kv_packing
                    page_idx = jnp.minimum(page_indices_offset + i,
                                           num_page_indices - 1)
                    safe_page_offset = jnp.minimum(
                        page_indices_ref[page_idx] * page_size_per_kv_packing,
                        jnp.maximum(0,
                                    max_hbm_pages - page_size_per_kv_packing),
                    )

                    _async_copy(
                        reshaped_cache_hbm_ref.at[pl.ds(
                            safe_page_offset, sz_per_kv_packing)],
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
        for batch_idx in range(seq_batch_size):
            sem = sems.at[1, bq_sem_idx, batch_idx]
            weights_sem = sems.at[3, bq_sem_idx, batch_idx]
            bq_vmem_ref = bq_x2_ref.at[bq_sem_idx, batch_idx]
            bq_weights_vmem_ref = bq_weights_x2_ref.at[bq_sem_idx, batch_idx]

            q_len_start = cu_q_lens_ref[seq_idx + batch_idx] + bq_idx * bq_sz
            curr_q_end = cu_q_lens_ref[seq_idx + batch_idx + 1]
            sz = jnp.maximum(0, jnp.minimum(bq_sz, curr_q_end - q_len_start))

            if not wait:
                _async_copy(
                    q_hbm_ref.at[pl.ds(q_len_start, sz)],
                    bq_vmem_ref.at[pl.ds(0, sz)],
                    sem,
                    wait=False,
                )
                _async_copy(
                    indexer_weights_hbm_ref.at[pl.ds(q_len_start, sz)],
                    bq_weights_vmem_ref.at[pl.ds(0, sz)],
                    weights_sem,
                    wait=False,
                )
            else:
                dst = bq_vmem_ref.at[pl.ds(0, sz)]
                _async_copy(dst, dst, sem, wait=True)
                dst_w = bq_weights_vmem_ref.at[pl.ds(0, sz)]
                _async_copy(dst_w, dst_w, weights_sem, wait=True)

    def start_fetch_bkv(seq_idx, bkv_idx, bkv_sem_idx):
        _fetch_bkv(seq_idx, bkv_idx, bkv_sem_idx)

    def wait_fetch_bkv(seq_idx, bkv_idx, bkv_sem_idx):
        _fetch_bkv(seq_idx, bkv_idx, bkv_sem_idx, wait=True)

    def start_fetch_bq(seq_idx, bq_idx, bq_sem_idx):
        return _fetch_bq(seq_idx, bq_idx, bq_sem_idx)

    def wait_fetch_bq(seq_idx, bq_idx, bq_sem_idx):
        return _fetch_bq(seq_idx, bq_idx, bq_sem_idx, wait=True)

    def load_bq(bq_sem_idx):
        data = bq_x2_ref.at[bq_sem_idx, :, :bq_sz][...].reshape(
            seq_batch_size, bq_sz * num_q_heads, head_dim)
        bqs = []
        for batch_idx in range(seq_batch_size):
            bqs.append(data[batch_idx])
        return bqs

    def load_bq_weights(bq_sem_idx):
        data = bq_weights_x2_ref.at[bq_sem_idx, :, :bq_sz][...]
        bq_weights = []
        for batch_idx in range(seq_batch_size):
            bq_weights.append(data[batch_idx])
        return bq_weights

    def load_bkv(bkv_sem_idx):
        bkvs = []
        bkv_scales = []
        for batch_idx in range(seq_batch_size):
            bkv = bkv_x2_ref.at[bkv_sem_idx,
                                batch_idx, :bkv_sz_per_kv_packing][...]

            # Unpack quantized values and scales from the DSv4 FP8 cache format.
            flat_bkv = bkv.reshape(-1, bkv.shape[-1])
            fp8_val = flat_bkv[:, :head_dim]
            fp8_val = pltpu.bitcast(fp8_val, jnp.float8_e4m3fn)
            scale_val = pltpu.bitcast(flat_bkv[:, head_dim:head_dim + 1].T,
                                      jnp.float8_e8m0fnu).astype(jnp.bfloat16)

            # NOTE: Do NOT multiply the scales here. Return them separately.
            bkvs.append(fp8_val.reshape(bkv_sz, head_dim))
            bkv_scales.append(scale_val)
        return bkvs, bkv_scales

    def process():
        # num_bkv is determined by the longest sequence length in the batch.
        kv_len_max = jnp.max(jnp.array(kv_lens))
        num_bkv = jnp.maximum(1, cdiv(kv_len_max, bkv_sz))
        if static_q_len is None:
            assert seq_batch_size == 1
            num_bq = jnp.maximum(1, cdiv(q_lens[0], bq_sz))
        else:
            num_bq = jnp.maximum(1, cdiv(static_q_len, bq_sz))

        def get_next_bq_ids(seq_idx, bq_idx, bq_sem_idx):
            next_bq_idx = bq_idx + 1
            is_last_bq = next_bq_idx == num_bq
            next_bq_idx = lax.select(is_last_bq, 0, next_bq_idx)
            next_seq_idx = lax.select(is_last_bq, seq_idx + seq_batch_size,
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
            next_seq_idx = lax.select(is_last_bq, seq_idx + seq_batch_size,
                                      seq_idx)
            next_bkv_sem_idx = lax.select(bkv_sem_idx == 0, 1, 0)
            return next_seq_idx, next_bq_idx, next_bkv_idx, next_bkv_sem_idx

        def compute_scores(
            bq_vec,
            bkv_vec,
            scale_val_vec,
            bq_weights_vec,
            bq_pos_compressed_vec,
            bkv_idx,
        ):
            assert len(bq_vec) == seq_batch_size
            assert len(bkv_vec) == seq_batch_size
            assert len(scale_val_vec) == seq_batch_size
            assert len(bq_weights_vec) == seq_batch_size
            assert len(bq_pos_compressed_vec) == seq_batch_size
            ret = []

            for batch_idx in range(seq_batch_size):
                bq = bq_vec[batch_idx].reshape(-1, head_dim)
                bkv = bkv_vec[batch_idx]
                scale_val = scale_val_vec[batch_idx]
                bq_weights = bq_weights_vec[batch_idx]
                bq_pos_compressed = bq_pos_compressed_vec[batch_idx]

                s = jnp.einsum(
                    "nd,md->nm",
                    bq,
                    bkv,
                    preferred_element_type=jnp.float32,
                )
                s = s.reshape(-1, num_q_heads, s.shape[-1])
                s = jnp.maximum(s, 0.0)
                s = s * bq_weights.astype(jnp.float32)[:, :, None]
                s_summed = s.sum(axis=1)
                s_summed = s_summed * scale_val
                k_span = bkv_idx * bkv_sz + lax.broadcasted_iota(
                    jnp.int32, s_summed.shape, 1)
                valid_mask = k_span < kv_lens[batch_idx]
                causal_mask = k_span <= bq_pos_compressed[:, None]
                mask = jnp.logical_and(valid_mask, causal_mask)
                s_summed = jnp.where(mask, s_summed, -jnp.inf)
                ret.append(s_summed.reshape(-1, num_sublanes_bkv, 128))
            return jnp.concatenate(ret, axis=0)

        def compute_with_bq(bq_idx, _):

            bq_sem_idx = sem_ids_ref[0]
            next_seq_idx, next_bq_idx, next_bq_sem_idx = get_next_bq_ids(
                batch_start_seq_idx, bq_idx, bq_sem_idx)

            # Prefetch next bq
            @pl.when(next_seq_idx < end_seq_idx)
            def prefetch_next_bq():
                sem_ids_ref[0] = next_bq_sem_idx
                start_fetch_bq(next_seq_idx, next_bq_idx, next_bq_sem_idx)

            bq_pos_compressed_vec = []
            for batch_idx in range(seq_batch_size):
                q_pos = (seq_lens[batch_idx] - q_lens[batch_idx] +
                         bq_idx * bq_sz + jnp.arange(bq_sz, dtype=jnp.int32))
                bq_pos_compressed_vec.append(q_pos // compression_ratio)

            # Wait for cur bq if not ready yet
            wait_fetch_bq(batch_start_seq_idx, bq_idx, bq_sem_idx)
            bq_vec = load_bq(bq_sem_idx)
            bq_weights_vec = load_bq_weights(bq_sem_idx)

            # If seq_batch_size > 1, static_q_len is always 1, therefore sz is always
            # 1 for all sequences within the batch.
            token_start = cu_q_lens_ref[batch_start_seq_idx] + bq_idx * bq_sz
            curr_q_end = cu_q_lens_ref[batch_start_seq_idx + 1]
            sz = jnp.maximum(0, jnp.minimum(bq_sz, curr_q_end - token_start))

            def compute_with_bkv(bkv_idx, _):
                bkv_sem_idx = sem_ids_ref[1]
                next_seq_idx, _, next_bkv_idx, next_bkv_sem_idx = get_next_bkv_ids(
                    batch_start_seq_idx, bq_idx, bkv_idx, bkv_sem_idx)

                # Prefetch next bkv
                @pl.when(next_seq_idx < end_seq_idx)
                def prefetch_next_bkv():
                    sem_ids_ref[1] = next_bkv_sem_idx
                    start_fetch_bkv(next_seq_idx, next_bkv_idx,
                                    next_bkv_sem_idx)

                # Wait for cur bkv
                wait_fetch_bkv(batch_start_seq_idx, bkv_idx, bkv_sem_idx)
                bkv_vec, scale_val_vec = load_bkv(bkv_sem_idx)

                scores = compute_scores(
                    bq_vec,
                    bkv_vec,
                    scale_val_vec,
                    bq_weights_vec,
                    bq_pos_compressed_vec,
                    bkv_idx,
                )

                # Double-buffered vreg -> VMEM -> HBM: reuse a buffer only after its
                # previous DMA has drained, so the HBM write overlaps the next compute.
                bo_sem_idx = sem_ids_ref[2]
                wait_send_scores(bo_sem_idx)
                scores_block_x2_ref[bo_sem_idx, ...] = scores
                start_send_scores(bo_sem_idx, sz, token_start, bkv_idx)
                sem_ids_ref[2] = lax.select(bo_sem_idx == 0, 1, 0)

            lax.fori_loop(0, num_bkv, compute_with_bkv, None, unroll=False)

        lax.fori_loop(0, num_bq, compute_with_bq, None, unroll=False)

    ### ------- Kernel start ------- ###

    @pl.when(batch_start_seq_idx == start_seq_idx)
    def prologue():
        start_fetch_bq(start_seq_idx, 0, 0)
        start_fetch_bkv(start_seq_idx, 0, 0)

    process()

    @pl.when(batch_end_seq_idx == end_seq_idx - 1)
    def epilogue():
        for i in range(2):
            wait_send_scores(i)

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
        "decode_req_batch_size",
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
    decode_req_batch_size: int = 4,
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

    prepared_indexer_weights = prepare_index_weights(indexer_weights,
                                                     original_dtype)
    q = prepare_q_inputs(q)
    lkv_dim = cache_kv.shape[-1]
    _, page_size_per_kv_packing, kv_packing, _ = cache_kv.shape
    page_size = page_size_per_kv_packing * kv_packing
    pages_per_seq = page_indices.shape[0] // max_num_seqs

    # Validate bkv_sz alignment due to TPU DMA constraints
    for bkv_p in num_kv_pages_per_blocks:
        bkv_sz = page_size * bkv_p
        if bkv_sz % 128 != 0:
            raise ValueError(
                f"bkv_sz ({page_size} * {bkv_p} = {bkv_sz}) must be a multiple"
                " of 128.")
    num_sublanes_total = max(
        align_to(pages_per_seq, bkv_p) * page_size // 128
        for bkv_p in num_kv_pages_per_blocks)

    def run_topk_kernel(
        q,
        prepared_indexer_weights,
        cache_kv,
        scores,
        seq_lens,
        page_indices,
        cu_q_lens,
        start_seq_idx,
        end_seq_idx,
        static_q_len,
        num_kv_pages_per_block,
        num_queries_per_block,
        seq_batch_size,
        case=MlaCase.MIXED,
    ):
        _, num_q_heads, head_dim = q.shape
        # Only support batching for decode sequences.
        # TODO: support batching for decode sequences with speculative decoding
        # enabled, e.g. static_q_len = gamma + 1.
        if seq_batch_size > 1:
            assert static_q_len == 1

        bkv_p = num_kv_pages_per_block
        if static_q_len is not None:
            bq_sz = min(num_queries_per_block, static_q_len)
        else:
            bq_sz = num_queries_per_block
        bkv_sz_per_kv_packing = bkv_p * page_size_per_kv_packing
        bkv_buf_sz_per_kv_packing = bkv_sz_per_kv_packing
        num_sublanes_bkv = bkv_sz_per_kv_packing * kv_packing // 128

        # If seq_batch_size > 1, caller already guaranteed that
        # end_seq_idx - start_seq_idx % seq_batch_size == 0.
        grid = ((end_seq_idx - start_seq_idx) // seq_batch_size, )

        in_specs = [
            pl.BlockSpec(memory_space=pltpu.HBM),  # q
            pl.BlockSpec(memory_space=pltpu.HBM),  # prepared_indexer_weights
            pl.BlockSpec(memory_space=pltpu.HBM),  # cache_kv
            pl.BlockSpec(
                memory_space=pltpu.HBM),  # scores_init (aliased to out)
        ]
        out_specs = pl.BlockSpec(memory_space=pltpu.HBM)  # scores

        bkv_double_buf = pltpu.VMEM(
            (
                2,
                seq_batch_size,
                bkv_buf_sz_per_kv_packing,
                kv_packing,
                lkv_dim,
            ),
            cache_kv.dtype,
        )
        bq_double_bufq = pltpu.VMEM(
            (
                2,
                seq_batch_size,
                bq_sz,
                num_q_heads,
                head_dim,
            ),
            q.dtype,
        )
        bq_weights_double_buf = pltpu.VMEM(
            (
                2,
                seq_batch_size,
                bq_sz,
                num_q_heads,
            ),
            prepared_indexer_weights.dtype,
        )
        bo_scores_double_buf = pltpu.VMEM(
            (2, seq_batch_size * bq_sz, num_sublanes_bkv, 128), jnp.float32)

        scratch_shapes = [
            bkv_double_buf,
            bq_double_bufq,
            bq_weights_double_buf,
            bo_scores_double_buf,
            pltpu.SemaphoreType.DMA((4, 2, seq_batch_size)),
        ]

        scalar_prefetches = (
            seq_lens,
            page_indices,
            cu_q_lens,
            jnp.array([start_seq_idx, end_seq_idx], jnp.int32),
            jnp.zeros((3, ), jnp.int32),  # (bq, bkv, bo) sem indices
            jnp.full((2, ), -1, jnp.int32),  # in-flight out DMA row counts
        )

        scope_name = f"StreamIdxTC-{case.symbol}-bq_{bq_sz}-bkvp_{bkv_p}"
        kernel = jax.named_scope(scope_name)(pl.pallas_call(
            functools.partial(
                _scores_kernel,
                compression_ratio=compression_ratio,
                static_q_len=static_q_len,
                bq_sz=bq_sz,
                bkv_p=bkv_p,
                seq_batch_size=seq_batch_size,
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
            out_shape=jax.ShapeDtypeStruct(
                shape=(q.shape[0], num_sublanes_total, 128),
                dtype=jnp.float32,
            ),
            input_output_aliases={len(scalar_prefetches) + 3: 0},
            name=scope_name,
        ))
        return kernel(
            *scalar_prefetches,
            q,
            prepared_indexer_weights,
            cache_kv,
            scores,
        )

    # Pre-fill the output with -inf and alias it, so masked / unwritten
    # columns are already -inf.
    scores_init = jnp.full(
        (q.shape[0], num_sublanes_total, 128),
        -jnp.inf,
        dtype=jnp.float32,
    )

    # TODO: we shall sort the sequences by length, so that multiple decode
    # sequences in one batch have similar lengths to reduce waste of compute.
    # With the same batch size, the longest sequence will determine number of
    # blocks to run computation for.
    decode_batch_end = (distribution[0] // decode_req_batch_size *
                        decode_req_batch_size)
    scores = run_topk_kernel(
        q,
        prepared_indexer_weights,
        cache_kv,
        scores_init,
        seq_lens,
        page_indices,
        cu_q_lens,
        num_kv_pages_per_block=num_kv_pages_per_blocks[0],
        num_queries_per_block=num_queries_per_blocks[0],
        start_seq_idx=jnp.array(0),
        end_seq_idx=decode_batch_end,
        static_q_len=1,
        seq_batch_size=decode_req_batch_size,
        case=MlaCase.DECODE,
    )
    # Handle num_decode_seqs % decode_req_batch_size != 0 case.
    scores = run_topk_kernel(
        q,
        prepared_indexer_weights,
        cache_kv,
        scores,
        seq_lens,
        page_indices,
        cu_q_lens,
        num_kv_pages_per_block=num_kv_pages_per_blocks[0],
        num_queries_per_block=num_queries_per_blocks[0],
        start_seq_idx=decode_batch_end,
        end_seq_idx=distribution[1],
        static_q_len=1,
        seq_batch_size=1,
        case=MlaCase.DECODE,
    )

    scores = run_topk_kernel(
        q,
        prepared_indexer_weights,
        cache_kv,
        scores,
        seq_lens,
        page_indices,
        cu_q_lens,
        num_kv_pages_per_block=num_kv_pages_per_blocks[2],
        num_queries_per_block=num_queries_per_blocks[2],
        start_seq_idx=distribution[1],
        end_seq_idx=distribution[2],
        static_q_len=None,
        seq_batch_size=1,
        case=MlaCase.MIXED,
    )

    scores = scores.reshape(q.shape[0], -1)
    if scores.shape[1] < k:
        scores = jnp.pad(
            scores,
            ((0, 0), (0, k - scores.shape[1])),
            constant_values=-jnp.inf,
        )

    # TODO: Re-evaluate replacing this with the sparsecore_topk kernel
    # once SparseCore supports direct VMEM access (e.g., on TPU v8).
    # Currently, jax.lax.approx_max_k wins due to the HBM read/write tax, but
    # direct VMEM streaming will allow SC to beat TensorCore performance.

    # jax.lax.approx_max_k(recall_target=1.0) is equivalent to jax.lax.top_k
    # but faster.
    top_vals, top_idxs = jax.lax.approx_max_k(scores,
                                              k,
                                              reduction_dimension=-1,
                                              recall_target=1.0)
    topk_idxs = jnp.where(top_vals == -jnp.inf, -1, top_idxs)
    return topk_idxs[:q.shape[0], :k]
