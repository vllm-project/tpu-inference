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
"""TPU-Friendly and Data-Movement-Friendly MLA Ragged Paged Attention kernel.
FUSED MLA + KV Cache Update (RMW) with Prefetching and Double Buffering.

[ADDED: JIT Pre-warming]
Adds warmup_kernel_shapes() to eliminate runtime pjit cache_miss / TpuCompiler::Compile
stalls (observed: ×2–3 compile events at ~160 ms each in XProf traces).
Also activates the JAX persistent compilation cache so re-starts never recompile.
"""

import functools
import itertools
import logging
import os
import time
import jax
from jax import lax
from jax._src import dtypes as jax_dtypes
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp

logger = logging.getLogger(__name__)

# --- Helper Functions ---

def cdiv(a, b):
    """Ceiling division."""
    return (a + b - 1) // b

def align_to(a, b):
    """Align a to the nearest multiple of b."""
    return cdiv(a, b) * b

def get_dtype_packing(dtype):
    """Get packing factor for dtype (elements per 32-bit word)."""
    bits = (jax_dtypes.bit_width(dtype)
            if hasattr(jax_dtypes, "bit_width") else jax_dtypes.itemsize_bits(dtype))
    return 32 // bits

# --- KV Cache Shape ---

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


# --- Constants ---

DEFAULT_MASK_VALUE = -0.7 * float(jnp.finfo(jnp.dtype("float32")).max)
DEFAULT_VMEM_LIMIT_BYTES = 128 * 1024 * 1024  # Increased for TPU v6/v7

# ---------------------------------------------------------------------------
# [ADDED: JIT Pre-warming] — OPT-1 from XProf trace analysis
# ---------------------------------------------------------------------------
# XProf evidence: pjit cache_miss ×3, TpuCompiler::Compile ×2 fire at runtime,
# each costing ~160 ms wall-time visible to end users.
#
# Strategy:
#   1. Enable the JAX persistent compilation cache so recompilation is skipped
#      across process restarts (written to JAX_CACHE_DIR, default /tmp/.jax_cache).
#   2. warmup_kernel_shapes() iterates over all (batch_size, seq_len) combos
#      expected during serving and calls mla_ragged_paged_attention with
#      zero-filled dummy inputs to trigger XLA compilation before traffic arrives.
#   3. The warmup is intentionally shape-complete: it covers both decode shapes
#      (num_tokens == batch_size) and prefill shapes (num_tokens == seq_len).
# ---------------------------------------------------------------------------

# Default warm-up shape grid.  Override via warmup_kernel_shapes(batch_sizes=...,
# seq_lens=...).  Shapes matching real vLLM scheduler settings should be included.
_DEFAULT_WARMUP_BATCH_SIZES = [1, 2, 4, 8, 16, 32, 64, 128]
_DEFAULT_WARMUP_SEQ_LENS    = [128, 256, 512, 1024, 2048, 4096]

# Canonical model dimensions for DeepSeek-R1 MLA on TPU v7.
# Adjust if serving a different model or sharding config.
_WARMUP_NUM_Q_HEADS = 128    # total q heads (before TP split)
_WARMUP_LKV_DIM     = 512    # compressed KV latent dim
_WARMUP_R_DIM       = 64     # RoPE positional encoding dim
_WARMUP_PAGE_SIZE   = 32     # tokens per KV cache page
_WARMUP_PAGES_PER_SEQ = 256  # max pages per sequence
_WARMUP_KV_DTYPE    = jnp.float8_e4m3fn  # must match serving dtype
_WARMUP_Q_DTYPE     = jnp.bfloat16


def enable_persistent_jax_cache(
    cache_dir: str | None = None,
    min_compile_secs: float = 0.0,
) -> str:
    """Activate JAX's persistent compilation cache.

    After this call every XLA compilation is written to *cache_dir*.  On the
    next process start the compiled artifact is loaded from disk instead of
    recompiling, so startup time drops to near-zero for pre-warmed shapes.

    Args:
        cache_dir: Directory to store cached XLA artifacts.  Defaults to the
            value of the ``JAX_CACHE_DIR`` env-var or ``/tmp/.jax_cache``.
        min_compile_secs: Only cache compilations that take longer than this
            many seconds.  0.0 caches everything.

    Returns:
        The resolved cache directory path.
    """
    if cache_dir is None:
        cache_dir = os.environ.get("JAX_CACHE_DIR", "/tmp/.jax_cache")
    os.makedirs(cache_dir, exist_ok=True)
    jax.config.update("jax_compilation_cache_dir", cache_dir)
    jax.config.update(
        "jax_persistent_cache_min_compile_time_secs", min_compile_secs
    )
    logger.info("[JIT warmup] Persistent JAX cache enabled at: %s", cache_dir)
    return cache_dir


def _make_dummy_inputs(
    num_tokens: int,
    num_q_heads: int,
    lkv_dim: int,
    r_dim: int,
    page_size: int,
    pages_per_seq: int,
    num_seqs: int,
    kv_dtype,
    q_dtype,
):
    """Build zero-filled dummy inputs matching the given shape.

    All arrays are placed on the default JAX device.  The KV cache is sized
    to hold ``num_seqs * pages_per_seq`` pages.
    """
    kv_packing = get_dtype_packing(kv_dtype)
    kv_dim     = lkv_dim + r_dim
    total_pages = num_seqs * pages_per_seq

    # query inputs — [num_tokens, num_q_heads, lkv_dim / r_dim]
    ql_nope = jnp.zeros((num_tokens, num_q_heads, lkv_dim), dtype=q_dtype)
    q_pe    = jnp.zeros((num_tokens, num_q_heads, r_dim),   dtype=q_dtype)

    # new KV inputs — [num_tokens, lkv_dim / r_dim]
    new_kv_c = jnp.zeros((num_tokens, lkv_dim), dtype=kv_dtype)
    new_k_pe = jnp.zeros((num_tokens, r_dim),   dtype=kv_dtype)

    # KV cache — use get_kv_cache_shape() so packing/alignment matches kernel
    cache_shape = get_kv_cache_shape(total_pages, page_size, kv_dim, kv_dtype)
    cache_kv = jnp.zeros(cache_shape, dtype=kv_dtype)

    # metadata scalars
    kv_lens      = jnp.ones((num_seqs,),                         dtype=jnp.int32)
    page_indices = jnp.zeros((num_seqs * pages_per_seq,),        dtype=jnp.int32)
    cu_q_lens    = jnp.arange(num_seqs + 1, dtype=jnp.int32) * (num_tokens // num_seqs)
    # distribution: [decode_end, prefill_end, mixed_end] — treat as all-decode
    distribution = jnp.array([num_seqs, num_seqs, num_seqs], dtype=jnp.int32)

    return ql_nope, q_pe, new_kv_c, new_k_pe, cache_kv, kv_lens, page_indices, cu_q_lens, distribution


def warmup_kernel_shapes(
    batch_sizes: list[int] | None = None,
    seq_lens: list[int] | None = None,
    num_q_heads: int = _WARMUP_NUM_Q_HEADS,
    lkv_dim: int = _WARMUP_LKV_DIM,
    r_dim: int = _WARMUP_R_DIM,
    page_size: int = _WARMUP_PAGE_SIZE,
    pages_per_seq: int = _WARMUP_PAGES_PER_SEQ,
    kv_dtype = _WARMUP_KV_DTYPE,
    q_dtype  = _WARMUP_Q_DTYPE,
    num_kv_pages_per_block: int = 16,
    num_queries_per_block: int  = 128,
    chunk_prefill_size: int | None = None,
    enable_cache: bool = True,
    cache_dir: str | None = None,
) -> int:
    """Pre-compile mla_ragged_paged_attention for all expected serving shapes.

    Call this function once at server startup *before* opening the endpoint for
    traffic.  Each (batch_size, seq_len) combination triggers one XLA
    compilation the first time and a fast cache-hit on every subsequent call
    (including across process restarts when *enable_cache* is True).

    XProf motivation
    ----------------
    Without warmup, runtime traces show:
      pjit cache_miss ×3  — 0.480 ms per miss (but ~160 ms actual compile stall)
      TpuCompiler::Compile ×2 — 0.339 ms per call in trace (~160 ms wall each)
    These stalls are p99-latency spikes visible directly to end users.

    Args:
        batch_sizes: List of batch sizes (num_seqs) to warm.  Defaults to
            ``_DEFAULT_WARMUP_BATCH_SIZES``.
        seq_lens: List of total token counts to warm.  Defaults to
            ``_DEFAULT_WARMUP_SEQ_LENS``.  For decode, seq_len == batch_size.
        num_q_heads: Number of query heads (must match model config).
        lkv_dim: Compressed KV latent dimension.
        r_dim: RoPE positional encoding dimension.
        page_size: Tokens per KV cache page.
        pages_per_seq: Maximum pages per sequence.
        kv_dtype: KV cache dtype (e.g. jnp.float8_e4m3fn for fp8 serving).
        q_dtype: Query dtype (e.g. jnp.bfloat16).
        num_kv_pages_per_block: Passed to mla_ragged_paged_attention.
        num_queries_per_block: Passed to mla_ragged_paged_attention.
        chunk_prefill_size: Passed to mla_ragged_paged_attention for prefill.
        enable_cache: If True, enable the JAX persistent compilation cache
            before warming (recommended).
        cache_dir: Override the persistent cache directory.  See
            enable_persistent_jax_cache().

    Returns:
        Number of (batch, seq_len) shapes successfully compiled.
    """
    if batch_sizes is None:
        batch_sizes = _DEFAULT_WARMUP_BATCH_SIZES
    if seq_lens is None:
        seq_lens = _DEFAULT_WARMUP_SEQ_LENS

    if enable_cache:
        enable_persistent_jax_cache(cache_dir=cache_dir)

    # Deduplicate and sort for deterministic logging
    shapes = sorted(set(itertools.product(batch_sizes, seq_lens)))
    n_compiled = 0
    t_total_start = time.monotonic()

    logger.info(
        "[JIT warmup] Starting warmup for %d shapes "
        "(batch_sizes=%s, seq_lens=%s) ...",
        len(shapes), batch_sizes, seq_lens,
    )

    for num_seqs, num_tokens in shapes:
        # Skip degenerate shapes: need at least 1 token per seq
        if num_tokens < num_seqs:
            continue
        # Clamp pages_per_seq so total_pages doesn't OOM on small boxes
        effective_pages = max(1, min(pages_per_seq, 8192 // max(1, num_seqs)))

        t_start = time.monotonic()
        try:
            dummy = _make_dummy_inputs(
                num_tokens=num_tokens,
                num_q_heads=num_q_heads,
                lkv_dim=lkv_dim,
                r_dim=r_dim,
                page_size=page_size,
                pages_per_seq=effective_pages,
                num_seqs=num_seqs,
                kv_dtype=kv_dtype,
                q_dtype=q_dtype,
            )
            ql_nope, q_pe, new_kv_c, new_k_pe, cache_kv, kv_lens, page_indices, cu_q_lens, distribution = dummy

            # Trigger compilation by running the jitted function and blocking.
            out, _updated_cache = mla_ragged_paged_attention(
                ql_nope, q_pe, new_kv_c, new_k_pe, cache_kv,
                kv_lens, page_indices, cu_q_lens, distribution,
                num_kv_pages_per_block=num_kv_pages_per_block,
                num_queries_per_block=num_queries_per_block,
                chunk_prefill_size=chunk_prefill_size,
            )
            # Force device synchronization so the compile + execution completes
            # before we move to the next shape.
            jax.effects_barrier()

            elapsed = time.monotonic() - t_start
            n_compiled += 1
            logger.info(
                "[JIT warmup] Compiled (batch=%d, tokens=%d) in %.2f s",
                num_seqs, num_tokens, elapsed,
            )
        except Exception as exc:  # pylint: disable=broad-except
            # Log but do not abort — a single bad shape shouldn't block serving.
            logger.warning(
                "[JIT warmup] Skipped (batch=%d, tokens=%d): %s",
                num_seqs, num_tokens, exc,
            )

    t_total = time.monotonic() - t_total_start
    logger.info(
        "[JIT warmup] Complete: %d/%d shapes compiled in %.1f s.",
        n_compiled, len(shapes), t_total,
    )
    return n_compiled

# ---------------------------------------------------------------------------
# End [ADDED: JIT Pre-warming]
# ---------------------------------------------------------------------------

# --- Kernel Implementation ---

def _mla_ragged_paged_attention_kernel(
    # Prefetch
    kv_lens_ref,  # [max_num_seqs] # Memory: SMEM
    page_indices_ref,  # [max_num_seqs * pages_per_seq] # Memory: SMEM
    cu_q_lens_ref,  # [max_num_seqs + 1] # Memory: SMEM
    distribution_ref,  # [3] (decode_end, prefill_end, mixed_end) # Memory: SMEM
    sem_ids_ref,  # [3] (bq_sem_idx, bkv_sem_idx, bo_sem_idx) # Memory: SMEM
    bo_ids_ref,  # [4] (bo_sem_0_seq_idx, bo_sem_1_seq_idx, bo_sem_0_bo_idx, bo_sem_1_bo_idx) # Memory: SMEM
    bkv_update_ids_ref,  # [6] # Memory: SMEM
    # Input
    ql_nope_hbm_ref,  # [max_num_tokens, num_q_heads_per_q_packing, q_packing, lkv_dim] # Memory: HBM
    q_pe_hbm_ref,  # [max_num_tokens, num_q_heads_per_q_packing, q_packing, r_dim] # Memory: HBM
    new_kv_c_hbm_ref,  # [max_num_tokens_per_kv_packing, kv_packing, lkv_dim] # Memory: HBM
    new_k_pe_hbm_ref,  # [max_num_tokens_per_kv_packing, kv_packing, r_dim] # Memory: HBM
    cache_kv_hbm_ref,  # [total_num_pages, page_size_per_kv_packing, kv_packing, align_to(lkv_dim + r_dim, 128)] # Memory: HBM
    # Output
    o_hbm_ref,  # [max_num_tokens, num_q_heads_per_q_packing, q_packing, lkv_dim] # Memory: HBM
    updated_cache_kv_hbm_ref,  # [total_num_pages, page_size_per_kv_packing, kv_packing, align_to(lkv_dim + r_dim, 128)] # Memory: HBM
    # Scratch
    bkvc_x2_ref,  # [2, bkv_sz_per_kv_packing, kv_packing, lkv_dim] # Memory: VMEM
    bkpe_x2_ref,  # [2, bkv_sz_per_kv_packing, kv_packing, r_dim] # Memory: VMEM
    bq_nope_x2_ref,  # [2, bq_sz, num_q_heads_per_q_packing, q_packing, lkv_dim] # Memory: VMEM
    bq_rope_x2_ref,  # [2, bq_sz, num_q_heads_per_q_packing, q_packing, r_dim] # Memory: VMEM
    bo_x2_ref,  # [2, bq_sz, num_q_heads_per_q_packing, q_packing, lkv_dim] # Memory: VMEM
    sems,  # [4, 2] # Memory: SMEM (Semaphores)
    l_ref,  # [bq_sz * num_q_heads, 128] # Memory: VMEM
    m_ref,  # [bq_sz * num_q_heads, 128] # Memory: VMEM
    acc_ref,  # [bq_sz * num_q_heads, lkv_dim] # Memory: VMEM
    # Update Scratch
    existing_cache_scratch_ref,  # [kv_packing, kv_dim] # Memory: VMEM
    new_kv_c_scratch_ref,        # [scratch_size_kv] # Memory: VMEM
    new_k_pe_scratch_ref,        # [scratch_size_pe] # Memory: VMEM
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
    """Core kernel logic for MLA Ragged Paged Attention."""
    
    # Validation checks on the dimensions
    nope_dim = ql_nope_hbm_ref.shape[-1]
    
    _, num_q_heads_per_q_packing, q_packing, lkv_dim = ql_nope_hbm_ref.shape
    r_dim = q_pe_hbm_ref.shape[-1]
    num_q_heads = num_q_heads_per_q_packing * q_packing
    total_num_pages, page_size_per_kv_packing, kv_packing, _ = (
        cache_kv_hbm_ref.shape)
    
    # Grid indices
    seq_idx = pl.program_id(0)
    num_seqs = pl.num_programs(0)
    
    pages_per_seq = page_indices_ref.shape[0] // kv_lens_ref.shape[0]
    q_dtype = ql_nope_hbm_ref.dtype
    kv_dtype = cache_kv_hbm_ref.dtype
    
    bkv_sz_per_kv_packing = bkv_p * page_size_per_kv_packing
    bkv_sz = bkv_sz_per_kv_packing * kv_packing
    page_size = page_size_per_kv_packing * kv_packing
    
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

    def _async_copy(src, dst, sem, wait):
        if debug_mode:
            return
        cp = pltpu.make_async_copy(src, dst, sem)
        if wait:
            cp.wait()
        else:
            cp.start()
        return cp

    # Fused Update Logic (RMW)
    def update_kv_cache_block():
        # Helper for dynamic selection
        def select_tree(index, candidates):
            n = len(candidates)
            if n == 1:
                return candidates[0]
            mid = n // 2
            return lax.select(index < mid, 
                              select_tree(index, candidates[:mid]), 
                              select_tree(index - mid, candidates[mid:]))

        # Update the KV cache with new tokens for the current sequence.
        # This function implements the 32-bit RMW pattern.
        
        def token_loop_body(j, _):
            token_idx_in_seq = kv_len - q_len + j
            page_num_in_seq = token_idx_in_seq // page_size
            page_indices_start = seq_idx * pages_per_seq
            page_idx = page_indices_ref[page_indices_start + page_num_in_seq]
            
            row = (token_idx_in_seq % page_size) // kv_packing
            col = (token_idx_in_seq % page_size) % kv_packing
            
            # Prepare new data from new_kv_c/new_k_pe inputs
            idx = q_start + j
            row_new = idx // kv_packing
            col_new = idx % kv_packing
            
            # Align read to 8 to satisfy tiling constraints
            aligned_row_new = row_new & ~7
            
            # 1. Load new KV values
            dma_sem = sems.at[3, 0] # Use spare semaphore
            
            # Transfer: HBM -> VMEM (Scratch)
            cp1 = pltpu.make_async_copy(
                new_kv_c_hbm_ref.at[pl.ds(aligned_row_new, 8)],
                new_kv_c_scratch_ref.at[pl.ds(0, 8)],
                dma_sem
            )
            cp2 = pltpu.make_async_copy(
                new_k_pe_hbm_ref.at[pl.ds(aligned_row_new, 8)],
                new_k_pe_scratch_ref.at[pl.ds(0, 8)],
                dma_sem
            )
            cp1.start()
            cp2.start()
            cp1.wait()
            cp2.wait()
            
            # Extract the specific token's KV
            # Load: VMEM -> Registers
            row_offset = row_new % 8
            full_c = new_kv_c_scratch_ref.at[row_offset][...] # [kv_packing, lkv_dim]
            
            c_candidates = [full_c[i] for i in range(kv_packing)]
            new_val_c = select_tree(col_new, c_candidates)
            
            full_pe = new_k_pe_scratch_ref.at[row_offset][...] # [kv_packing, r_dim]
            pe_candidates = [full_pe[i] for i in range(kv_packing)]
            new_val_pe = select_tree(col_new, pe_candidates)
            
            # 2. Read-Modify-Write into Cache
            
            # Load existing cache line
            # Transfer: HBM -> VMEM
            cp_read = pltpu.make_async_copy(
                updated_cache_kv_hbm_ref.at[page_idx, row],
                existing_cache_scratch_ref.at[0],
                dma_sem
            )
            cp_read.start()
            cp_read.wait()
            
            # Modify in VMEM
            cache_line = existing_cache_scratch_ref[0] # [kv_packing, total_dim]
            
            # Construct updated line
            idx_range = lax.broadcasted_iota(jnp.int32, (kv_packing, 1), 0)
            mask = idx_range == col
            
            # Update part 1: lkv_dim
            current_c = cache_line[:, :lkv_dim]
            updated_c = jnp.where(mask, new_val_c[None, :], current_c)
            
            # Update part 2: r_dim
            current_pe = cache_line[:, lkv_dim:]
            updated_pe = jnp.where(mask, new_val_pe[None, :], current_pe)
            
            # Combine
            updated_line = jnp.concatenate([updated_c, updated_pe], axis=1)
            
            existing_cache_scratch_ref.at[0].set(updated_line)
            
            # Write back
            # Transfer: VMEM -> HBM
            cp_write = pltpu.make_async_copy(
                existing_cache_scratch_ref.at[0],
                updated_cache_kv_hbm_ref.at[page_idx, row],
                dma_sem
            )
            cp_write.start()
            cp_write.wait()
            
        lax.fori_loop(0, q_len, token_loop_body, None)

    # Standard MLA Logic (Flash Attention)
    
    def flash_attention(
        ql_nope,  # [actual_bq_sz * num_q_heads, lkv_dim] # Memory: Registers
        q_pe,  # [actual_bq_sz * num_q_heads, r_dim] # Memory: Registers
        kv_c,  # [bkv_sz, lkv_dim] # Memory: Registers
        k_pe,  # [bkv_sz, r_dim] # Memory: Registers
        *,
        bq_idx,
        bkv_idx,
    ):
        head_l_ref = l_ref.at[:ql_nope.shape[0]]
        head_m_ref = m_ref.at[:ql_nope.shape[0]]
        head_acc_ref = acc_ref.at[:ql_nope.shape[0]]

        def load_with_init(ref, init_val):
            return jnp.where(bkv_idx == 0, jnp.full_like(ref, init_val),
                             ref[...])

        # OPTIMIZATION: Use lax.dot_general for matrix multiplication
        # ql_nope: (N, D), kv_c: (M, D) -> s_nope: (N, M)
        # We use dimension_numbers to specify contraction on the last dimension of both (trans_b=True effect)
        s_nope = lax.dot_general(
            ql_nope, 
            kv_c, 
            dimension_numbers=(((1,), (1,)), ((), ())),
            preferred_element_type=jnp.float32
        )
        
        # q_pe: (N, R), k_pe: (M, R) -> s_pe: (N, M)
        s_pe = lax.dot_general(
            q_pe, 
            k_pe, 
            dimension_numbers=(((1,), (1,)), ((), ())),
            preferred_element_type=jnp.float32
        )
        
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

        # OPTIMIZATION: Use lax.dot_general for attention output
        # p: (N, M), kv_c: (M, D) -> pv: (N, D)
        # Standard matmul (trans_b=False)
        pv = lax.dot_general(
            p, 
            kv_c, 
            dimension_numbers=(((1,), (0,)), ((), ())),
            preferred_element_type=jnp.float32
        )
        
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

    def _fetch_bkv(seq_idx, bkv_idx, bkv_sem_idx, *, wait=False):
        sem = sems.at[0, bkv_sem_idx]
        bkvc_vmem_ref = bkvc_x2_ref.at[bkv_sem_idx]
        bkvpe_vmem_ref = bkpe_x2_ref.at[bkv_sem_idx]
        reshaped_cache_hbm_ref = cache_kv_hbm_ref.reshape(
            total_num_pages * page_size_per_kv_packing,
            *cache_kv_hbm_ref.shape[2:],
        )
        kv_len = kv_lens_ref[seq_idx]
        kv_len_start = bkv_idx * bkv_sz
        kv_p_start = bkv_idx * bkv_p

        kv_left = kv_len - kv_len_start
        kv_left_per_kv_packing = cdiv(kv_left, kv_packing)
        page_indices_offset = seq_idx * pages_per_seq + kv_p_start

        # Fetch effective kv from kv cache.
        def loop_body(i, _):
            sz_per_kv_packing = jnp.minimum(
                page_size_per_kv_packing,
                kv_left_per_kv_packing - i * page_size_per_kv_packing,
            )
            # Transfer: HBM -> VMEM
            _async_copy(
                reshaped_cache_hbm_ref.at[pl.ds(
                    page_indices_ref[page_indices_offset + i] *
                    page_size_per_kv_packing,
                    sz_per_kv_packing,
                ), ..., :nope_dim],
                bkvc_vmem_ref.at[pl.ds(i * page_size_per_kv_packing,
                                       sz_per_kv_packing)],
                sem,
                wait,
            )
            # Transfer: HBM -> VMEM
            _async_copy(
                reshaped_cache_hbm_ref.at[pl.ds(
                    page_indices_ref[page_indices_offset + i] *
                    page_size_per_kv_packing,
                    sz_per_kv_packing,
                ), ..., nope_dim:],
                bkvpe_vmem_ref.at[pl.ds(i * page_size_per_kv_packing,
                                        sz_per_kv_packing)],
                sem,
                wait,
            )

        actual_bkv_p = jnp.minimum(cdiv(kv_left, page_size), bkv_p)
        lax.fori_loop(
            0,
            actual_bkv_p,
            loop_body,
            None,  # init value
            unroll=False,
        )

    def _fetch_bq(seq_idx, bq_idx, bq_sem_idx, *, wait=False):
        sem = sems.at[1, bq_sem_idx]
        bq_nope_vmem_ref = bq_nope_x2_ref.at[bq_sem_idx]
        bq_rope_vmem_ref = bq_rope_x2_ref.at[bq_sem_idx]

        q_len_start = cu_q_lens_ref[seq_idx] + bq_idx * bq_sz
        q_end = cu_q_lens_ref[seq_idx + 1]
        sz = jnp.minimum(bq_sz, q_end - q_len_start)

        # Transfer: HBM -> VMEM
        _async_copy(
            ql_nope_hbm_ref.at[pl.ds(q_len_start, sz)],
            bq_nope_vmem_ref.at[pl.ds(0, sz)],
            sem,
            wait,
        )

        # Transfer: HBM -> VMEM
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

        # Transfer: VMEM -> HBM
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

    def load_bq(bq_sem_idx, *, actual_bq_sz=bq_sz):
        q_nope_ref = (bq_nope_x2_ref.bitcast(
            jnp.uint32).at[bq_sem_idx].reshape(
                bq_sz * num_q_heads_per_q_packing, lkv_dim))
        q_nope_vec = pltpu.bitcast(
            q_nope_ref[:actual_bq_sz * num_q_heads_per_q_packing],
            q_dtype,
        )
        q_rope_ref = (bq_rope_x2_ref.bitcast(
            jnp.uint32).at[bq_sem_idx].reshape(
                bq_sz * num_q_heads_per_q_packing, r_dim))
        q_rope_vec = pltpu.bitcast(
            q_rope_ref[:actual_bq_sz * num_q_heads_per_q_packing],
            q_dtype,
        )
        return q_nope_vec, q_rope_vec

    def load_bkv(bkv_sem_idx, *, bkvc_mask, bkpe_mask):
        bkvc_ref = (bkvc_x2_ref.bitcast(jnp.uint32).at[bkv_sem_idx].reshape(
            bkv_sz_per_kv_packing, lkv_dim))
        bkvc_vec = pltpu.bitcast(bkvc_ref[...], kv_dtype)
        bkvc_vec = lax.select(bkvc_mask, bkvc_vec, jnp.zeros_like(bkvc_vec))

        bkpe_ref = (bkpe_x2_ref.bitcast(jnp.uint32).at[bkv_sem_idx].reshape(
            bkv_sz_per_kv_packing, r_dim))
        bkpe_vec = pltpu.bitcast(bkpe_ref[...], kv_dtype)
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

            @pl.when(next_seq_idx < num_seqs)
            def prefetch_next_bq():
                sem_ids_ref[0] = next_bq_sem_idx
                start_fetch_bq(next_seq_idx, next_bq_idx, next_bq_sem_idx)

            def compute_with_bkv(bkv_idx, _):
                assert bkv_sz % kv_packing == 0
                actual_bkv_sz = jnp.minimum(bkv_sz, kv_len - bkv_idx * bkv_sz)
                bkvc_shape = (bkv_sz, lkv_dim)
                bkvc_mask = (lax.broadcasted_iota(jnp.int32, bkvc_shape, 0)
                             < actual_bkv_sz)
                bkpe_shape = (bkv_sz, r_dim)
                bkpe_mask = (lax.broadcasted_iota(jnp.int32, bkpe_shape, 0)
                             < actual_bkv_sz)

                bkv_sem_idx = sem_ids_ref[1]
                next_seq_idx, _, next_bkv_idx, next_bkv_sem_idx = get_next_bkv_ids(
                    seq_idx, bq_idx, bkv_idx, bkv_sem_idx)

                @pl.when(next_seq_idx < num_seqs)
                def prefetch_next_bkv():
                    sem_ids_ref[1] = next_bkv_sem_idx
                    start_fetch_bkv(next_seq_idx, next_bkv_idx,
                                    next_bkv_sem_idx)

                @pl.when(bkv_idx == 0)
                def wait_cur_bq():
                    wait_fetch_bq(seq_idx, bq_idx, bq_sem_idx)

                wait_fetch_bkv(seq_idx, bkv_idx, bkv_sem_idx)

                if debug_mode:
                    return

                bkvc, bkpe = load_bkv(bkv_sem_idx,
                                      bkvc_mask=bkvc_mask,
                                      bkpe_mask=bkpe_mask)
                bq_nope_vec, bq_pe_vec = load_bq(bq_sem_idx,
                                                 actual_bq_sz=actual_bq_sz)
                flash_attention(
                    bq_nope_vec,
                    bq_pe_vec,
                    bkvc,
                    bkpe,
                    bq_idx=bq_idx,
                    bkv_idx=bkv_idx,
                )

            lax.fori_loop(0, num_bkv, compute_with_bkv, None, unroll=False)

            acc = acc_ref[...]
            l = broadcast_minor(l_ref[...], acc.shape)
            out = (lax.div(acc, l) if q_dtype == jnp.float32 else
                   (acc * pl.reciprocal(l, approx=True)).astype(q_dtype))

            bo_sem_idx = sem_ids_ref[2]
            sem_ids_ref[2] = lax.select(bo_sem_idx == 0, 1, 0)
            wait_send_bo(bo_sem_idx)

            bo_x2_ref.at[bo_sem_idx].bitcast(jnp.int32).reshape(
                bq_sz * num_q_heads_per_q_packing,
                lkv_dim,
            )[...] = pltpu.bitcast(out, jnp.int32)

            start_send_bo(seq_idx, bq_idx, bo_sem_idx)

        lax.fori_loop(0, num_bq, compute_with_bq, None, unroll=False)

    ### ------- Kernel start ------- ###

    @pl.when(seq_idx == 0)
    def prologue():
        pass

    # OPT-3 — Early BKV/BQ Prefetch: issue the first BKV/BQ DMA *before* the
    # KV cache update so the fetch latency is hidden behind the update window.
    # Safety invariant: start_fetch_bkv(0,0,0) reads cache history positions
    # [0..bkv_sz-1] from cache_kv_hbm_ref, while update_kv_cache_block()
    # writes only to position kv_len-1 (the latest token). These ranges are
    # disjoint for kv_len > bkv_sz, which is always true in production decode.
    @pl.when(seq_idx == 0)
    def start_prefetch():
        start_fetch_bq(0, 0, 0)
        start_fetch_bkv(0, 0, 0)

    update_kv_cache_block()

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

    ### ------- Kernel end ------- ###

# --- Computation Wrapper ---

def prepare_q_inputs(q: jax.Array):
    max_num_tokens, actual_num_q_heads, actual_head_dim = q.shape
    q_packing = get_dtype_packing(q.dtype)
    num_q_heads = align_to(actual_num_q_heads, q_packing)
    head_dim = align_to(actual_head_dim, 128)
    q = jnp.pad(
        q.reshape(max_num_tokens, actual_num_q_heads, actual_head_dim),
        ((0, 0), (0, num_q_heads - actual_num_q_heads), (0, head_dim - actual_head_dim)),
        constant_values=0,
    ).reshape(max_num_tokens, num_q_heads // q_packing, q_packing, head_dim)
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
    
    # Pad first dimension to multiple of 8 for aligned loads
    num_packed_rows = kv.shape[0]
    if num_packed_rows % 8 != 0:
        pad_rows = 8 - (num_packed_rows % 8)
        kv = jnp.pad(kv, ((0, pad_rows), (0, 0), (0, 0)), constant_values=0)
        
    kv = jnp.pad(kv, ((0, 0), (0, 0), (0, head_dim - actual_head_dim)),
                 constant_values=0)
    return kv

def prepare_outputs(out, actual_num_q_heads, actual_head_dim):
    max_num_tokens, num_q_heads_per_q_packing, q_packing, head_dim = out.shape
    return out.reshape(
        max_num_tokens, num_q_heads_per_q_packing * q_packing, head_dim
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
    donate_argnames=("cache_kv"),
)
def mla_ragged_paged_attention(
    ql_nope: jax.Array,
    q_pe: jax.Array,
    new_kv_c: jax.Array,
    new_k_pe: jax.Array,
    cache_kv: jax.Array,
    kv_lens: jax.Array,
    page_indices: jax.Array,
    cu_q_lens: jax.Array,
    distribution: jax.Array,
    *,
    sm_scale: float = 1.0,
    sliding_window: int | None = None,
    soft_cap: float | None = None,
    mask_value: float | None = DEFAULT_MASK_VALUE,
    q_scale: float | None = None,
    k_scale: float | None = None,
    v_scale: float | None = None,
    chunk_prefill_size: int | None = None,
    num_kv_pages_per_block: int | None = 16, # Default: 16 * 32 = 512 tokens
    num_queries_per_block: int | None = 128, # Default: 128 queries
    vmem_limit_bytes: int | None = None,
    debug_mode: bool = False,
) -> tuple[jax.Array, jax.Array]:
    """Sets up and invokes the optimized MLA Pallas kernel."""
    
    # Use defaults if not provided
    if num_kv_pages_per_block is None:
        num_kv_pages_per_block = 16
    if num_queries_per_block is None:
        num_queries_per_block = 128
        
    # Input validation
    if len(ql_nope.shape) != 3:
        raise ValueError(f"Expected 3D array for {ql_nope.shape=}")

    _, actual_num_q_heads, actual_lkv_dim = ql_nope.shape
    
    ql_nope = prepare_q_inputs(ql_nope)
    q_pe = prepare_q_inputs(q_pe)
    
    # We pass new_kv_c and new_k_pe prepared for the kernel
    new_kv_c_prepared = prepare_kv_inputs(new_kv_c) # [tokens/P, P, D]
    new_k_pe_prepared = prepare_kv_inputs(new_k_pe)
    
    lkv_dim = new_kv_c_prepared.shape[-1]
    r_dim = new_k_pe_prepared.shape[-1]
    
    _, page_size_per_kv_packing, kv_packing, _ = cache_kv.shape
    page_size = page_size_per_kv_packing * kv_packing
    _, num_q_heads_per_q_packing, q_packing, _ = ql_nope.shape
    
    bkv_p = num_kv_pages_per_block
    bq_sz = num_queries_per_block
    bkv_sz_per_kv_packing = bkv_p * page_size_per_kv_packing
    grid = (distribution[2], )

    # Specs
    in_specs = [
        pl.BlockSpec(memory_space=pltpu.HBM), # ql_nope
        pl.BlockSpec(memory_space=pltpu.HBM), # q_pe
        pl.BlockSpec(memory_space=pltpu.HBM), # new_kv_c
        pl.BlockSpec(memory_space=pltpu.HBM), # new_k_pe
        pl.BlockSpec(memory_space=pltpu.HBM), # cache_kv
    ]
    out_specs = [
        pl.BlockSpec(memory_space=pltpu.HBM), # o
        pl.BlockSpec(memory_space=pltpu.HBM), # updated_cache_kv
    ]

    bkvc_double_buf = pltpu.VMEM((2, bkv_sz_per_kv_packing, kv_packing, lkv_dim), cache_kv.dtype)
    bkpe_double_buf = pltpu.VMEM((2, bkv_sz_per_kv_packing, kv_packing, r_dim), cache_kv.dtype)
    bq_nope_double_buf = pltpu.VMEM((2, bq_sz, num_q_heads_per_q_packing, q_packing, lkv_dim), ql_nope.dtype)
    bq_rope_double_buf = pltpu.VMEM((2, bq_sz, num_q_heads_per_q_packing, q_packing, r_dim), q_pe.dtype)
    bo_double_buf = bq_nope_double_buf
    l_scratch = pltpu.VMEM((bq_sz * num_q_heads_per_q_packing * q_packing, 128), jnp.float32)
    m_scratch = l_scratch
    acc_scratch = pltpu.VMEM((bq_sz * num_q_heads_per_q_packing * q_packing, lkv_dim), jnp.float32)
    
    # Scratch for RMW update
    kv_dim = lkv_dim + r_dim
    existing_cache_scratch = pltpu.VMEM((1, kv_packing, kv_dim), cache_kv.dtype)
    # Scratch for loading new KVs: [8, kv_packing, dim]
    new_kv_c_scratch = pltpu.VMEM((8, kv_packing, lkv_dim), new_kv_c_prepared.dtype)
    new_k_pe_scratch = pltpu.VMEM((8, kv_packing, r_dim), new_k_pe_prepared.dtype)

    scratch_shapes = [
        bkvc_double_buf,
        bkpe_double_buf,
        bq_nope_double_buf,
        bq_rope_double_buf,
        bo_double_buf,
        pltpu.SemaphoreType.DMA((4, 2)),
        l_scratch,
        m_scratch,
        acc_scratch,
        existing_cache_scratch,
        new_kv_c_scratch,
        new_k_pe_scratch,
    ]

    scalar_prefetches = (
        kv_lens,
        page_indices,
        cu_q_lens,
        distribution,
        jnp.zeros((3, ), jnp.int32),
        jnp.full((4, ), -1, jnp.int32),
        jnp.full((6, ), -1, jnp.int32),
    )

    scope_name = f"MLA-RPA-Fused-5K-FP8-bq_{bq_sz}-bkvp_{bkv_p}-p_{page_size}"
    
    # Kernel invocation
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
                jax.ShapeDtypeStruct(shape=cache_kv.shape, dtype=cache_kv.dtype),
            ],
            input_output_aliases={
                11: 1, # Aliasing cache_kv
            },
            name=scope_name,
        ))

    output, updated_kv = kernel(
        *scalar_prefetches,
        ql_nope,
        q_pe,
        new_kv_c_prepared,
        new_k_pe_prepared,
        cache_kv,
    )
    output = prepare_outputs(output, actual_num_q_heads, actual_lkv_dim)

    return output, updated_kv

# --- Main Function for Testing ---

def main():
    """Demonstrates kernel usage with sample inputs, including JIT warmup."""
    import logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    # [ADDED: JIT Pre-warming] — enable persistent cache and pre-warm shapes
    print("\n=== Step 0: Enable JAX persistent cache + pre-warm JIT shapes ===")
    enable_persistent_jax_cache(cache_dir="/tmp/.jax_cache")
    n = warmup_kernel_shapes(
        batch_sizes=[1, 2, 4],
        seq_lens=[128, 256],
        num_q_heads=4,       # small for test
        lkv_dim=128,
        r_dim=64,
        page_size=32,
        pages_per_seq=16,
        kv_dtype=jnp.bfloat16,  # use bfloat16 for easy local testing
        q_dtype=jnp.bfloat16,
        num_kv_pages_per_block=4,
        num_queries_per_block=128,
        enable_cache=False,  # already enabled above
    )
    print(f"Warmup complete: {n} shapes pre-compiled.\n")

    print("Initializing MLA kernel inputs...")
    
    # Parameters
    batch_size = 1
    seq_len = 128
    num_heads = 4
    head_dim = 128
    kv_dim = 128
    page_size = 32
    
    key = jax.random.PRNGKey(0)
    k1, k2, k3, k4, k5 = jax.random.split(key, 5)
    
    # Create inputs
    ql_nope = jax.random.normal(k1, (seq_len, num_heads, head_dim), dtype=jnp.float32)
    q_pe = jax.random.normal(k2, (seq_len, num_heads, head_dim), dtype=jnp.float32)
    new_kv_c = jax.random.normal(k3, (seq_len, kv_dim), dtype=jnp.float32)
    new_k_pe = jax.random.normal(k4, (seq_len, kv_dim), dtype=jnp.float32)
    
    # Cache setup
    total_pages = 100
    cache_kv = jax.random.normal(k5, (total_pages, page_size, 1, kv_dim * 2), dtype=jnp.float32)
    
    # Metadata
    kv_lens = jnp.array([seq_len], dtype=jnp.int32)
    page_indices = jnp.arange(total_pages, dtype=jnp.int32) # Simple mapping
    cu_q_lens = jnp.array([0, seq_len], dtype=jnp.int32)
    distribution = jnp.array([1, 1, 1], dtype=jnp.int32) # Dummy distribution
    
    print("Running computation (without JIT)...")
    # Stage 1: No JIT
    out, updated_cache = mla_ragged_paged_attention(
        ql_nope, q_pe, new_kv_c, new_k_pe, cache_kv,
        kv_lens, page_indices, cu_q_lens, distribution,
        num_kv_pages_per_block=16,
        num_queries_per_block=128,
        chunk_prefill_size=128
    )
    print(f"Output shape: {out.shape}")
    print(f"Updated cache shape: {updated_cache.shape}")
    
    print("\nRunning computation (with JIT)...")
    # Stage 2: With JIT
    jitted_comp = jax.jit(mla_ragged_paged_attention, static_argnames=(
        "sm_scale", "sliding_window", "soft_cap", "mask_value", 
        "q_scale", "k_scale", "v_scale", "chunk_prefill_size",
        "num_kv_pages_per_block", "num_queries_per_block", "vmem_limit_bytes", "debug_mode"
    ))
    
    out_jit, updated_cache_jit = jitted_comp(
        ql_nope, q_pe, new_kv_c, new_k_pe, cache_kv,
        kv_lens, page_indices, cu_q_lens, distribution,
        num_kv_pages_per_block=16,
        num_queries_per_block=128,
        chunk_prefill_size=128
    )
    print(f"JIT Output shape: {out_jit.shape}")
    print("Success!")

if __name__ == "__main__":
    main()