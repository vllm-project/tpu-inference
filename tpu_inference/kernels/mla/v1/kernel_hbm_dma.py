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
"""TPU-Friendly and Data-Movement-Friendly MLA Ragged Paged Attention kernel. FUSED MLA + KV Cache Update (RMW) with Prefetching and Double Buffering."""

import functools
import jax
from jax import lax
from jax._src import dtypes as jax_dtypes
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp

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

    # Fused Update Logic (RMW) — HBM-Bandwidth-Optimised
    def update_kv_cache_block():
        # Helper for dynamic selection (unchanged).
        def select_tree(index, candidates):
            n = len(candidates)
            if n == 1:
                return candidates[0]
            mid = n // 2
            return lax.select(index < mid,
                              select_tree(index, candidates[:mid]),
                              select_tree(index - mid, candidates[mid:]))

        # HBM BANDWIDTH FIX: Process one cache row per iteration instead of one
        # token per iteration.
        #
        # Root cause: with kv_packing=4 (float8), up to kv_packing consecutive
        # new tokens all map to the same 128-byte packed cache row.  The old
        # per-token loop issued a DMA_Read + DMA_Write for every individual token,
        # so a q_len=4 prefill touching one cache row generated 4 reads and 4
        # writes of the *same* row — kv_packing× redundant HBM traffic plus
        # serialised DMA latency.  XProf shows this as a dense burst of tiny
        # DMA_Read/DMA_Write events at kernel start with idle Vector units.
        #
        # Fix: the outer loop iterates over unique cache rows
        # (ceil((first_col + q_len) / kv_packing) iterations).  Each iteration:
        #   1. Issues one DMA for the new-KV aligned window (same as before).
        #   2. DMA_Reads  the target cache row ONCE.
        #   3. Scatters all valid slot updates with a VPU-only unrolled loop
        #      (kv_packing iterations, fully unrolled by XLA — no dynamic gather).
        #   4. DMA_Writes the updated cache row ONCE.
        #
        # HBM round-trip reduction: up to kv_packing× for q_len >= kv_packing.
        # Decode (q_len=1): iteration count is identical to before (1 cache row),
        # so there is zero regression in that regime.
        #
        # Correctness property: cache slot s always equals cache col s.
        # Proof: col = (first_token_idx + j_row_start + s) % kv_packing
        #             = (first_token_idx - first_col + s)  % kv_packing
        #               [r_iter * kv_packing cancels mod kv_packing]
        #             = ((first_token_idx // kv_packing) * kv_packing + s) % kv_packing
        #             = s.   QED.
        # This means we can update slot s directly without a col→slot indirection.

        dma_sem         = sems.at[3, 0]
        first_token_idx = kv_len - q_len          # sequence pos of first new token
        first_col       = first_token_idx % kv_packing  # its col within its cache row
        num_cache_rows  = cdiv(first_col + q_len, kv_packing)

        def cache_row_body(r_iter, _):
            # j_row_start: first j (0-indexed token offset) for the r_iter-th cache
            # row.  For r_iter=0 with first_col>0 this is negative; those leading
            # slots contain old cache data and are skipped via the valid mask below.
            j_row_start = r_iter * kv_packing - first_col

            # Anchor on the first *valid* token in this row to locate it in HBM.
            j_start    = jnp.maximum(j_row_start, 0)
            anchor_seq = first_token_idx + j_start
            page_num   = anchor_seq // page_size
            page_idx   = page_indices_ref[seq_idx * pages_per_seq + page_num]
            cache_row  = (anchor_seq % page_size) // kv_packing

            # --- Steps 1+2: Issue all three read DMAs concurrently ---
            # cp1 (new_kv_c), cp2 (new_k_pe), and cp_read (existing cache row)
            # all read from independent HBM addresses into independent VMEM
            # scratch locations.  Issuing them together lets the DMA engine
            # service all three in parallel, reducing per-row DMA latency from
            # 2 serial waits (cp1/cp2, then cp_read) down to 1 parallel wait.
            # For prefill with num_cache_rows=32 this saves ~1 DMA latency ×
            # 32 iterations ≈ 128 µs per kernel invocation.
            first_new_idx   = q_start + j_start
            aligned_new_row = (first_new_idx // kv_packing) & ~7

            cp1 = pltpu.make_async_copy(
                new_kv_c_hbm_ref.at[pl.ds(aligned_new_row, 8)],
                new_kv_c_scratch_ref.at[pl.ds(0, 8)],
                dma_sem,
            )
            cp2 = pltpu.make_async_copy(
                new_k_pe_hbm_ref.at[pl.ds(aligned_new_row, 8)],
                new_k_pe_scratch_ref.at[pl.ds(0, 8)],
                dma_sem,
            )
            # Define cp_read BEFORE any start() so all three can be issued
            # together in the next line.
            cp_read = pltpu.make_async_copy(
                updated_cache_kv_hbm_ref.at[page_idx, cache_row],
                existing_cache_scratch_ref.at[0],
                dma_sem,
            )
            cp1.start(); cp2.start(); cp_read.start()
            cp1.wait(); cp2.wait(); cp_read.wait()

            cache_line = existing_cache_scratch_ref[0]   # [kv_packing, total_dim]
            updated_c  = cache_line[:, :lkv_dim]         # [kv_packing, lkv_dim]
            updated_pe = cache_line[:, lkv_dim:]          # [kv_packing, r_dim]

            # --- Step 3: Update the cache line (VPU only, select-based) ---
            # Unrolled over the compile-time constant kv_packing.  Each iteration
            # handles one (slot, token) pair.
            #
            # Implementation note: .at[s].set() inside a lax.fori_loop lowers to
            # XLA scatter, which is not supported by the Pallas TPU backend.
            # Instead we use jnp.where with a per-slot boolean mask (slot_iota==s),
            # which lowers to XLA select — fully supported on TPU.
            slot_iota = lax.broadcasted_iota(jnp.int32, (kv_packing, 1), 0)
            for s in range(kv_packing):
                j_s   = j_row_start + s        # dynamic j value for this slot
                valid = jnp.logical_and(j_s >= 0, j_s < q_len)

                # For invalid slots (leading/trailing partial row), fall back to
                # first_new_idx so the scratch access is always in-bounds.  The
                # extracted value is discarded by jnp.where when valid=False.
                new_idx = jnp.where(valid, q_start + j_s, first_new_idx)
                row_off = jnp.clip(new_idx // kv_packing - aligned_new_row, 0, 7)
                col_s   = new_idx % kv_packing

                scratch_c_row  = new_kv_c_scratch_ref.at[row_off][...]  # [kv_packing, lkv_dim]
                scratch_pe_row = new_k_pe_scratch_ref.at[row_off][...]  # [kv_packing, r_dim]

                c_candidates  = [scratch_c_row[i]  for i in range(kv_packing)]
                pe_candidates = [scratch_pe_row[i] for i in range(kv_packing)]
                new_c_val  = select_tree(col_s, c_candidates)   # [lkv_dim]
                new_pe_val = select_tree(col_s, pe_candidates)  # [r_dim]

                # slot_mask is True only for row s (shape [kv_packing, 1]).
                # Combining with the per-token valid flag gives a mask that is
                # True at exactly one position — the slot we want to update.
                slot_mask = (slot_iota == s)                       # [kv_packing, 1]
                apply     = jnp.logical_and(valid, slot_mask)      # [kv_packing, 1]
                updated_c  = jnp.where(apply, new_c_val[None, :],  updated_c)
                updated_pe = jnp.where(apply, new_pe_val[None, :], updated_pe)

            updated_line = jnp.concatenate([updated_c, updated_pe], axis=1)
            existing_cache_scratch_ref.at[0].set(updated_line)

            # --- Step 4: DMA_Write the updated cache row ONCE ---
            cp_write = pltpu.make_async_copy(
                existing_cache_scratch_ref.at[0],
                updated_cache_kv_hbm_ref.at[page_idx, cache_row],
                dma_sem,
            )
            cp_write.start(); cp_write.wait()

        lax.fori_loop(0, num_cache_rows, cache_row_body, None)

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

    update_kv_cache_block()
    
    @pl.when(seq_idx == 0)
    def start_prefetch():
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
    """Demonstrates kernel usage with sample inputs."""
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