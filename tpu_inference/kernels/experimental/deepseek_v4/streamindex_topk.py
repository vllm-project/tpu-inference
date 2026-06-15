"""TPU-Friendly MLA Ragged Paged Attention kernel."""

from enum import Enum
import functools
import jax
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp

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


def _indexer_attend_kernel(
    # Prefetch
    seq_lens_ref,  # [max_num_seqs]
    page_indices_ref,  # [max_num_seqs * pages_per_seq]
    cu_q_lens_ref,  # [max_num_seqs + 1]
    start_end_seq_idx_ref,  # [2] (start_seq_idx, end_seq_idx)
    sem_ids_ref,  # [3] (bq_sem_idx, bkv_sem_idx, bo_sem_idx)
    bo_ids_ref,  # [4] (bo_sem_0_seq_idx, bo_sem_1_seq_idx, bo_sem_0_bo_idx, bo_sem_1_bo_idx)
    # Input
    q_hbm_ref,  # [max_num_tokens, num_q_heads, head_dim]
    cache_kv_hbm_ref,  # [total_num_pages, page_size_per_kv_packing, kv_packing, lkv_dim]
    index_weights_hbm_ref,  # [max_num_tokens, num_q_heads]
    # Output
    o_hbm_ref,  # [max_num_tokens, topk]
    # Scratch
    bkv_x2_ref,  # [2, bkv_buf_sz_per_kv_packing, kv_packing, lkv_dim]
    bq_x2_ref,  # [2, bq_sz, num_q_heads, head_dim]
    b_index_weights_x2_ref,  # [2, bq_sz, num_q_heads]
    bo_x2_ref,  # [2, bq_sz, topk]
    sems,  # [4, 2]
    topk_vals_ref,  # [max_num_tokens, csa_topk]
    topk_indices_ref,  # [max_num_tokens, csa_topk]
    *,
    static_q_len: int,
    compression_ratio: int,
    bkv_p,
    bq_sz,
):
  assert q_hbm_ref.shape[0] == o_hbm_ref.shape[0]

  _, num_q_heads, head_dim = q_hbm_ref.shape
  lkv_dim = cache_kv_hbm_ref.shape[-1]
  total_num_pages, page_size_per_kv_packing, kv_packing, _ = (
      cache_kv_hbm_ref.shape
  )
  max_num_seqs = seq_lens_ref.shape[0]
  num_page_indices = page_indices_ref.shape[0]

  assert num_page_indices % max_num_seqs == 0
  pages_per_seq = num_page_indices // max_num_seqs
  q_dtype = q_hbm_ref.dtype
  q_packing = get_dtype_packing(q_dtype)
  iw_dtype = index_weights_hbm_ref.dtype
  iw_packing = get_dtype_packing(iw_dtype)
  # Validate against the KV dtype.
  kv_dtype = cache_kv_hbm_ref.dtype
  assert o_hbm_ref.dtype == jnp.int32
  assert get_dtype_packing(kv_dtype) == kv_packing
  assert num_q_heads % q_packing == 0
  num_q_heads_per_q_packing = num_q_heads // q_packing
  assert num_q_heads % iw_packing == 0
  num_q_heads_per_iw_packing = num_q_heads // iw_packing
  assert lkv_dim % 128 == 0
  bkv_sz_per_kv_packing = bkv_p * page_size_per_kv_packing
  bkv_sz = bkv_sz_per_kv_packing * kv_packing
  page_size = page_size_per_kv_packing * kv_packing

  start_seq_idx = start_end_seq_idx_ref[0]
  end_seq_idx = start_end_seq_idx_ref[1]
  seq_idx = pl.program_id(0) + start_seq_idx
  q_start = cu_q_lens_ref[seq_idx]
  q_end = cu_q_lens_ref[seq_idx + 1]
  q_len = q_end - q_start
  seq_len = seq_lens_ref[seq_idx]

  def streaming_topk(
      q,  # [bq_sz * num_q_heads, head_dim]
      kv,  # [bkv_sz, head_dim] <- Correspond to data from bkv_x2_ref
      kv_scales,  # [bkv_sz] <- Correspond to data from bkv_x2_ref
      b_index_weights,  # [bq_sz, num_q_heads]
      *,
      bq_idx,
      bkv_idx,
  ):
    assert len(q.shape) == 2
    assert len(kv.shape) == 2
    assert q.shape[0] % num_q_heads == 0
    assert q.shape[1] == head_dim
    assert kv.shape == (bkv_sz, head_dim)
    assert b_index_weights.shape == (bq_sz, num_q_heads)

    def load_with_init(ref, init_val):
      return jnp.where(bkv_idx == 0, jnp.full_like(ref, init_val), ref[...])

    s = jnp.einsum("nd,md->nm", q, kv, preferred_element_type=jnp.float32)
    s = s * kv_scales[None, :]
    s = jnp.maximum(0, s)
    s = s.reshape(bq_sz, num_q_heads, bkv_sz)
    s = s * b_index_weights[:, :, None].astype(s.dtype)
    s = jnp.sum(s, axis=1)  # [bq_sz, bkv_sz]

    q_span = (
        seq_len
        - q_len
        + bq_idx * bq_sz
        + lax.broadcasted_iota(jnp.int32, (bq_sz, bkv_sz), 0)
    )
    kv_chunk_base_indices = bkv_idx * bkv_sz + lax.broadcasted_iota(
        jnp.int32, (bq_sz, bkv_sz), 1
    )
    k_span = kv_chunk_base_indices * compression_ratio
    mask = q_span < k_span
    # Also drop compressed entries beyond the written cache length. Each entry
    # covers `compression_ratio` tokens and is only written once its group
    # completes, so the written count is `seq_len // compression_ratio`. Rows
    # past that hold stale recycled-block bytes (values / e8m0 scales) that can
    # dequantize to NaN/Inf; without this, the partial final group (when seq_len
    # is not a multiple of compression_ratio) escapes the causal `q_span < k_span`
    # check and a non-finite score corrupts the top-k argmax. Masking the score
    # to -inf keeps those rows finite and unselected. `kv_chunk_base_indices` is
    # already the 2D row index, so no reshape is needed.
    mask = jnp.logical_or(mask,
                          kv_chunk_base_indices >= (seq_len // compression_ratio))
    s = jnp.where(mask, -jnp.inf, s)

    topk = topk_vals_ref.shape[-1]
    topk_vals = load_with_init(topk_vals_ref, -jnp.inf)
    topk_indices = load_with_init(topk_indices_ref, -1)
    s_indices = kv_chunk_base_indices

    values = jnp.concatenate([topk_vals, s], axis=-1)
    indices = jnp.concatenate([topk_indices, s_indices], axis=-1)

    batch_size, n_items = values.shape
    neg_inf_values = -jnp.inf

    final_top_k_values = jnp.full(
        (batch_size, topk), -jnp.inf, dtype=values.dtype
    )
    final_top_k_indices = jnp.full((batch_size, topk), -1, dtype=indices.dtype)

    # TODO: if topk > bkv_sz, we may loop over bkv_sz instead.
    for i in range(topk):
      argmax_indices = jnp.argmax(values, axis=1)
      current_top_values = jnp.max(values, axis=1)

      one_hot_mask = jax.nn.one_hot(
          argmax_indices, num_classes=n_items, dtype=values.dtype
      )

      current_top_indices = jnp.sum(
          indices * one_hot_mask.astype(indices.dtype),
          axis=1,
          dtype=indices.dtype,
      )

      column_mask = jnp.arange(topk) == i

      final_top_k_values = jnp.where(
          column_mask[None, :], current_top_values[:, None], final_top_k_values
      )
      final_top_k_indices = jnp.where(
          column_mask[None, :],
          current_top_indices[:, None],
          final_top_k_indices,
      )

      values = jnp.where(one_hot_mask.astype(bool), neg_inf_values, values)
      indices = jnp.where(one_hot_mask.astype(bool), -1, indices)

    topk_vals_ref[...] = final_top_k_values
    topk_indices_ref[...] = final_top_k_indices

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
    kv_len = seq_lens_ref[seq_idx] // compression_ratio
    kv_len_start = bkv_idx * bkv_sz
    kv_p_start = bkv_idx * bkv_p

    kv_left = kv_len - kv_len_start
    kv_left_per_kv_packing = cdiv(kv_left, kv_packing)
    bkv_sz_frm_cache = jnp.minimum(kv_left, bkv_sz)
    bkv_sz_frm_cache_per_kv_packing = cdiv(bkv_sz_frm_cache, kv_packing)
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
        page_idx = jnp.minimum(page_indices_offset + i, num_page_indices - 1)
        _async_copy(
            reshaped_cache_hbm_ref.at[
                pl.ds(
                    page_indices_ref[page_idx] * page_size_per_kv_packing,
                    sz_per_kv_packing,
                ),
            ],
            # [bkv_sz_per_kv_packing, kv_packing, lkv_dim].
            bkv_vmem_ref.at[
                pl.ds(i * page_size_per_kv_packing, sz_per_kv_packing)
            ],
            sem,
            wait,
        )

    else:
      # When we wait, we can use a dummy copy to wait for DMAs to complete where
      # src == dst. However, the dma size must be correct.
      dst_kv = bkv_vmem_ref.at[pl.ds(0, bkv_sz_frm_cache_per_kv_packing)]
      _async_copy(
          src=dst_kv,
          dst=dst_kv,
          sem=sem,
          wait=True,
      )

  def _fetch_bq(seq_idx, bq_idx, bq_sem_idx, *, wait=False):
    sem_q = sems.at[1, bq_sem_idx]
    sem_iw = sems.at[3, bq_sem_idx]
    bq_vmem_ref = bq_x2_ref.at[bq_sem_idx]
    b_index_weights_vmem_ref = b_index_weights_x2_ref.at[bq_sem_idx]

    q_len_start = cu_q_lens_ref[seq_idx] + bq_idx * bq_sz
    q_end = cu_q_lens_ref[seq_idx + 1]
    sz = jnp.minimum(bq_sz, q_end - q_len_start)

    _async_copy(
        q_hbm_ref.at[pl.ds(q_len_start, sz)],
        bq_vmem_ref.at[pl.ds(0, sz)],
        sem_q,
        wait,
    )
    _async_copy(
        index_weights_hbm_ref.at[pl.ds(q_len_start, sz)],
        b_index_weights_vmem_ref.at[pl.ds(0, sz)],
        sem_iw,
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

  def load_bq(
      bq_sem_idx,
  ):
    q_ref = (
        bq_x2_ref.bitcast(jnp.uint32)
        .at[bq_sem_idx]
        .reshape(bq_sz * num_q_heads_per_q_packing, head_dim)
    )
    q = pltpu.bitcast(
        q_ref[: bq_sz * num_q_heads_per_q_packing],
        q_dtype,
    ).reshape(bq_sz * num_q_heads, head_dim)
    return q

  def load_b_index_weights(bq_sem_idx):
    return b_index_weights_x2_ref[bq_sem_idx]

  def load_bkv(bkv_sem_idx):
    bkv_ref = (
        bkv_x2_ref.bitcast(jnp.uint32)
        .at[bkv_sem_idx, :bkv_sz_per_kv_packing]
        .reshape(bkv_sz_per_kv_packing, lkv_dim)
    )
    bkv = pltpu.bitcast(bkv_ref[...], kv_dtype).reshape(bkv_sz, lkv_dim)

    # Dequantize DSV4 FP8 format to BF16.
    # 128 fp8, 1 e8m0 scale
    bkv_q = pltpu.bitcast(bkv[:, :128], jnp.float8_e4m3fn)
    # libtpu 0.0.41 not yet support the f8E8M0FNU element type, so decode the
    # E8M0 scale bytes manually. E8M0 stores value = 2**(byte - 127).
    bkv_scales = pltpu.bitcast(bkv[:, 128:129], jnp.uint8)
    bkv_scales = jnp.exp2(bkv_scales.astype(jnp.float32) - 127.0).astype(
        jnp.bfloat16).reshape(bkv_sz)
    
    return bkv_q, bkv_scales

  def process():
    # Force at least one bkv block per sequence. The double-buffered DMA
    # pipeline hands the bkv semaphore across sequence boundaries and assumes
    # every sequence runs >=1 bkv iteration (where the wait and the next-block
    # prefetch live). A sequence with seq_len < compression_ratio yields
    # seq_len // compression_ratio == 0, so without this guard it would run 0
    # iterations, leaving a prefetched DMA unwaited and the start/wait sizes
    # mismatched on the next sequence -> deadlock. The single forced block is
    # fully masked to -inf (every kv_chunk_base_indices >= seq_len //
    # compression_ratio), so top-k correctly returns empty (-1) indices.
    num_bkv = jnp.maximum(1, cdiv(seq_len // compression_ratio, bkv_sz))
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
          seq_idx, bq_idx, bq_sem_idx
      )

      # Prefetch next bq
      @pl.when(next_seq_idx < end_seq_idx)
      def prefetch_next_bq():
        sem_ids_ref[0] = next_bq_sem_idx
        start_fetch_bq(next_seq_idx, next_bq_idx, next_bq_sem_idx)

      def compute_with_bkv(bkv_idx, _):

        # Get next bkv ids.
        bkv_sem_idx = sem_ids_ref[1]
        next_seq_idx, _, next_bkv_idx, next_bkv_sem_idx = get_next_bkv_ids(
            seq_idx, bq_idx, bkv_idx, bkv_sem_idx
        )

        # Prefetch next bkv
        @pl.when(next_seq_idx < end_seq_idx)
        def prefetch_next_bkv():
          sem_ids_ref[1] = next_bkv_sem_idx
          start_fetch_bkv(next_seq_idx, next_bkv_idx, next_bkv_sem_idx)

        # Wait for cur bq if not ready yet
        @pl.when(bkv_idx == 0)
        def wait_cur_bq():
          wait_fetch_bq(seq_idx, bq_idx, bq_sem_idx)

        # Wait for cur bkv
        wait_fetch_bkv(seq_idx, bkv_idx, bkv_sem_idx)

        # Load bkv into vreg. There is no need to mask out invalid k/v entries,
        # because the score of invalid Q.K^T pairs are masked (to be zero) in
        # flash attention, so that the invalid kv entries
        # (as long as they are not NaN or inf) won't affect to the output.
        bkv_q, bkv_scales = load_bkv(
            bkv_sem_idx,
        )

        bq = load_bq(bq_sem_idx)
        b_index_weights = load_b_index_weights(bq_sem_idx)

        streaming_topk(
            bq,
            bkv_q,
            bkv_scales,
            b_index_weights,
            bq_idx=bq_idx,
            bkv_idx=bkv_idx,
        )

      lax.fori_loop(0, num_bkv, compute_with_bkv, None, unroll=False)

      # Wait for previous bo to be fully sent before storing new bo.
      bo_sem_idx = sem_ids_ref[2]
      sem_ids_ref[2] = lax.select(bo_sem_idx == 0, 1, 0)
      wait_send_bo(bo_sem_idx)

      # Store output from topk_indices to bo.
      bo_x2_ref[bo_sem_idx] = topk_indices_ref[...].reshape(
          bo_x2_ref[bo_sem_idx].shape
      )

      # Send cur bo
      start_send_bo(seq_idx, bq_idx, bo_sem_idx)

    lax.fori_loop(0, num_bq, compute_with_bq, None, unroll=False)

  ### ------- Kernel start ------- ###

  @pl.when(seq_idx == start_seq_idx)
  def prologue():
    start_fetch_bq(start_seq_idx, 0, 0)

    # Initialize bkv_x2_ref to zeros to avoid NaN issues from accessing
    # uninitialized memory. Bitcast into int32 to avoid tiling issues.
    bkv_x2_int32_ref = bkv_x2_ref.bitcast(jnp.int32).reshape((2, -1, lkv_dim))
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
  # Align to 2 to match 16-bit query head padding requirements
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


# TODO: handle DSv4 FP8 kv cache format
@functools.partial(
    jax.jit,
    static_argnames=(
        "topk",
        "chunk_prefill_size",
        "num_kv_pages_per_block",
        "num_queries_per_block",
        "compression_ratio",
        "vmem_limit_bytes",
    ),
)
def indexer_attend(
    q: jax.Array,  # [max_num_tokens, actual_num_q_heads, head_dim]
    cache_kv: jax.Array,  # [total_num_pages, page_size_per_kv_packing, kv_packing, lkv_dim]
    seq_lens: jax.Array,  # i32[max_num_seqs]
    page_indices: jax.Array,  # i32[max_num_seqs * pages_per_seq]
    cu_q_lens: jax.Array,  # i32[max_num_seqs + 1]
    distribution: jax.Array,  # i32[3]
    index_weights: jax.Array,  # [max_num_tokens, actual_num_q_heads]
    topk: int,
    compression_ratio: int,
    *,
    # Kernel optimization params.
    chunk_prefill_size: int | None = None,
    # Kernel tuning params for decode, prefill, and mixed cases.
    # If passsed in as int, all cases are the same.
    num_kv_pages_per_block: tuple[int, int, int] | int | None = None,
    num_queries_per_block: tuple[int, int, int] | int | None = None,
    vmem_limit_bytes: int = DEFAULT_VMEM_LIMIT_BYTES,
) -> jax.Array:  # [max_num_tokens, topk]
  """MLA Ragged paged attention that supports mixed prefill and decode.

  Args:
    q: concatenated all sequences' queries.
    cache_kv: the current kv cache.
    seq_lens: the length of each sequence.
    page_indices: flattened page indices look-up table by (seq_id, page_id).
    cu_q_lens: the cumulative sum of the effective query lengths.
    distribution: (i, j, k) represents that sequences[0:i] are decode-only,
      sequences[i:j] are chunked-prefill-only, and sequences[j:k] are mixed. The
      k is also the total number of sequences.
    index_weights: weights for each query token and head.
    topk: the number of topk.
    chunk_prefill_size: chunk prefill size.
    num_kv_pages_per_block: number of kv pages to be processed in one flash
      attention block in the pallas kernel. This is a tuple of (decode, prefill,
      mixed) cases.
    num_queries_per_block: number of queries to be processed in one flash
      attention block in the pallas kernel. This is a tuple of (decode, prefill,
      mixed) cases.
    vmem_limit_bytes: the vmem limit for the pallas kernel.

  Returns:
    For each query token, the topk most important tokens in the kv cache.
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

  _, actual_num_q_heads, head_dim = q.shape

  q = prepare_q_inputs(q)  # [max_num_tokens, num_q_heads, head_dim]
  index_weights = prepare_index_weights(
      index_weights, q.dtype
  )  # [max_num_tokens, num_q_heads]
  lkv_dim = cache_kv.shape[-1]

  _, page_size_per_kv_packing, kv_packing, _ = cache_kv.shape
  page_size = page_size_per_kv_packing * kv_packing
  _, num_q_heads, _ = q.shape
  max_num_seqs = seq_lens.shape[0]
  num_page_indices = page_indices.shape[0]
  assert num_page_indices % max_num_seqs == 0

  def run_mla_kernel(
      q: jax.Array,  # [max_num_tokens, num_q_heads, head_dim]
      cache_kv: jax.Array,  # [total_num_pages, page_size_per_kv_packing, kv_packing, lkv_dim]
      index_weights: jax.Array,  # [max_num_tokens, num_q_heads]
      seq_lens: jax.Array,  # i32[max_num_seqs]
      page_indices: jax.Array,  # i32[max_num_seqs * pages_per_seq]
      cu_q_lens: jax.Array,  # i32[max_num_seqs + 1]
      start_seq_idx: jax.Array,  # i32
      end_seq_idx: jax.Array,  # i32
      static_q_len: int | None,
      num_kv_pages_per_block: int,
      num_queries_per_block: int,
      compression_ratio: int,
      case: MlaCase = MlaCase.MIXED,
  ):

    bkv_p = num_kv_pages_per_block
    if static_q_len is not None:
      bq_sz = min(num_queries_per_block, static_q_len)
    else:
      bq_sz = num_queries_per_block
    bkv_sz_per_kv_packing = bkv_p * page_size_per_kv_packing
    bkv_buf_sz_per_kv_packing = bkv_sz_per_kv_packing
    if topk > 128:
      assert topk % 128 == 0
    topk_shape = (topk // 128, 128) if topk >= 128 else (topk,)
    grid = (end_seq_idx - start_seq_idx,)

    in_specs = [
        pl.BlockSpec(memory_space=pltpu.HBM),  # q
        pl.BlockSpec(memory_space=pltpu.HBM),  # cache_kv
        pl.BlockSpec(memory_space=pltpu.HBM),  # index_weights
    ]

    out_specs = pl.BlockSpec(memory_space=pltpu.HBM)

    bkv_double_buf = pltpu.VMEM(
        (2, bkv_buf_sz_per_kv_packing, kv_packing, lkv_dim),
        cache_kv.dtype,
    )

    bq_double_bufq = pltpu.VMEM(
        (2, bq_sz, num_q_heads, head_dim),
        q.dtype,
    )

    b_index_weights_double_buf = pltpu.VMEM(
        (2, bq_sz, num_q_heads),
        index_weights.dtype,
    )

    bo_double_buf = pltpu.VMEM(
        (2, bq_sz, *topk_shape),
        jnp.int32,
    )

    topk_vals_scratch = pltpu.VMEM(
        (bq_sz, topk),
        jnp.float32,
    )
    topk_indices_scratch = pltpu.VMEM(
        (bq_sz, topk),
        jnp.int32,
    )

    scratch_shapes = [
        bkv_double_buf,
        bq_double_bufq,
        b_index_weights_double_buf,
        bo_double_buf,  # Double buffering for output block.
        # Semaphores for double buffering of bkv, bq, bo, biw.
        pltpu.SemaphoreType.DMA((4, 2)),
        # Intermediate buffers per kv head for flash attention.
        topk_vals_scratch,
        topk_indices_scratch,
    ]

    scalar_prefetches = (
        seq_lens,
        page_indices,
        cu_q_lens,
        jnp.array([start_seq_idx, end_seq_idx], jnp.int32),
        # (bq_sem_idx, bkv_sem_idx, bo_sem_idx, biw_sem_idx)
        jnp.zeros((4,), jnp.int32),
        # (bo_sem_0_seq_idx, bo_sem_1_seq_idx, bo_sem_0_bo_idx, bo_sem_1_bo_idx)
        jnp.full((4,), -1, jnp.int32),
    )

    scope_name = f"MLA-{case.symbol}-bq_{bq_sz}-bkvp_{bkv_p}-p_{page_size}"
    kernel = jax.named_scope(scope_name)(
        pl.pallas_call(
            functools.partial(
                _indexer_attend_kernel,
                static_q_len=static_q_len,
                compression_ratio=compression_ratio,
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
                dimension_semantics=("arbitrary",),
                vmem_limit_bytes=vmem_limit_bytes,
                disable_bounds_checks=True,
            ),
            out_shape=jax.ShapeDtypeStruct(
                shape=(q.shape[0], *topk_shape), dtype=jnp.int32
            ),
            name=scope_name,
        )
    )
    return kernel(
        *scalar_prefetches,
        q,
        cache_kv,
        index_weights,
    )

  # Decode-only
  topk_indices_decode = run_mla_kernel(
      q,
      cache_kv,
      index_weights,
      seq_lens,
      page_indices,
      cu_q_lens,
      num_kv_pages_per_block=num_kv_pages_per_blocks[0],
      num_queries_per_block=num_queries_per_blocks[0],
      start_seq_idx=jnp.array(0),
      end_seq_idx=distribution[0],
      static_q_len=1,
      compression_ratio=compression_ratio,
      case=MlaCase.DECODE,
  )
  # TODO: evaluate if chunk-prefill-only branch is needed

  # Mixed
  topk_indices_mixed = run_mla_kernel(
      q,
      cache_kv,
      index_weights,
      seq_lens,
      page_indices,
      cu_q_lens,
      num_kv_pages_per_block=num_kv_pages_per_blocks[2],
      num_queries_per_block=num_queries_per_blocks[2],
      start_seq_idx=distribution[1],
      end_seq_idx=distribution[2],
      static_q_len=None,
      compression_ratio=compression_ratio,
      case=MlaCase.MIXED,
  )

  assert topk_indices_decode.shape == topk_indices_mixed.shape
  if len(topk_indices_decode.shape) == 3:
    topk_indices_decode = topk_indices_decode.reshape(
        (topk_indices_decode.shape[0], -1)
    )
    topk_indices_mixed = topk_indices_mixed.reshape(
        (topk_indices_mixed.shape[0], -1)
    )

  return jnp.where(
      (jnp.arange(q.shape[0]) < distribution[0])[:, None],
      topk_indices_decode,
      topk_indices_mixed,
  )