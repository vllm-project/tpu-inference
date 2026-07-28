# Copyright 2026 Google LLC
"""C=N fused MoE Pallas kernels — per-token expert loop with dedup.

Cross-expert double buffering: while computing on buffer A, prefetch next
expert's weights into buffer B.  Buffer index (`cur_w1_buf` / `cur_w2_buf`)
is a traced JAX value that swaps only on genuine expert transitions.
"""

import functools
import os as _os

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

M_PAD = 1
TOP_K = 10

_GEMV_NBUF = max(2, int(_os.getenv("MOE_GEMV_NBUF", "2")))
_FUSE_NBUF = max(2, int(_os.getenv("MOE_FUSE_NBUF", str(_GEMV_NBUF))))
_CN_K_TILE = int(_os.getenv("MOE_CN_K_TILE", "4096"))
_CN_I_TILE = int(_os.getenv("MOE_CN_I_TILE", "2048"))

# ── Semaphore layout ────────────────────────────────────────────────
#   [0          .. NBUF_W1)           w1 weight DMAs
#   [NBUF_W1   .. 2*NBUF_W1)         w1 scale  DMAs
#   [2*NBUF_W1 .. 2*NBUF_W1+NBUF_W2) w2 weight DMAs
#   [2*NBUF_W1+NBUF_W2 .. 2*NBUF_W1+2*NBUF_W2) w2 scale DMAs
#   [2*NBUF_W1+2*NBUF_W2]            lhs DMA
#   [2*NBUF_W1+2*NBUF_W2+1]          output DMA
# ────────────────────────────────────────────────────────────────────


# =====================================================================
#  DMA helpers — thin wrappers so start/wait sites stay readable
# =====================================================================
def _start_w1_dma(gj, buf, k, w1_ref, w1_scale_ref,
                  w1_bufs_ref, w1_s_bufs_ref, sem_ref, *,
                  K_TILE, KB_TILE, I, QB, NBUF_W1_):
    """Start async DMA for w1 tile `k` of expert `gj` into buffer `buf`."""
    k_block_start = (k * K_TILE) // QB
    pltpu.make_async_copy(
        w1_ref.at[pl.ds(gj, 1), pl.ds(k * K_TILE, K_TILE), pl.ds(0, 2 * I)],
        w1_bufs_ref.at[pl.ds(buf, 1), pl.ds(0, K_TILE), pl.ds(0, 2 * I)],
        sem_ref.at[0 * NBUF_W1_ + buf]
    ).start()
    pltpu.make_async_copy(
        w1_scale_ref.at[pl.ds(gj, 1), pl.ds(k_block_start, KB_TILE),
                        pl.ds(0, 1), pl.ds(0, 2 * I)],
        w1_s_bufs_ref.at[pl.ds(buf, 1), pl.ds(0, KB_TILE),
                         pl.ds(0, 1), pl.ds(0, 2 * I)],
        sem_ref.at[1 * NBUF_W1_ + buf]
    ).start()


def _wait_w1_dma(gj, buf, k, w1_ref, w1_scale_ref,
                 w1_bufs_ref, w1_s_bufs_ref, sem_ref, *,
                 K_TILE, KB_TILE, I, QB, NBUF_W1_):
    """Wait for w1 tile `k` of expert `gj` in buffer `buf`."""
    k_block_start = (k * K_TILE) // QB
    pltpu.make_async_copy(
        w1_ref.at[pl.ds(gj, 1), pl.ds(k * K_TILE, K_TILE), pl.ds(0, 2 * I)],
        w1_bufs_ref.at[pl.ds(buf, 1), pl.ds(0, K_TILE), pl.ds(0, 2 * I)],
        sem_ref.at[0 * NBUF_W1_ + buf]
    ).wait()
    pltpu.make_async_copy(
        w1_scale_ref.at[pl.ds(gj, 1), pl.ds(k_block_start, KB_TILE),
                        pl.ds(0, 1), pl.ds(0, 2 * I)],
        w1_s_bufs_ref.at[pl.ds(buf, 1), pl.ds(0, KB_TILE),
                         pl.ds(0, 1), pl.ds(0, 2 * I)],
        sem_ref.at[1 * NBUF_W1_ + buf]
    ).wait()


def _start_w2_dma(gj, buf, m, w2_ref, w2_scale_ref,
                  w2_bufs_ref, w2_s_bufs_ref, sem_ref, *,
                  I_TILE, IB_TILE, H, IB, NBUF_W1_, NBUF_W2_):
    """Start async DMA for w2 tile `m` of expert `gj` into buffer `buf`."""
    i_block_start = (m * I_TILE) // IB
    pltpu.make_async_copy(
        w2_ref.at[pl.ds(gj, 1), pl.ds(m * I_TILE, I_TILE), pl.ds(0, H)],
        w2_bufs_ref.at[pl.ds(buf, 1), pl.ds(0, I_TILE), pl.ds(0, H)],
        sem_ref.at[2 * NBUF_W1_ + buf]
    ).start()
    pltpu.make_async_copy(
        w2_scale_ref.at[pl.ds(gj, 1), pl.ds(i_block_start, IB_TILE),
                        pl.ds(0, 1), pl.ds(0, H)],
        w2_s_bufs_ref.at[pl.ds(buf, 1), pl.ds(0, IB_TILE),
                         pl.ds(0, 1), pl.ds(0, H)],
        sem_ref.at[2 * NBUF_W1_ + NBUF_W2_ + buf]
    ).start()


def _wait_w2_dma(gj, buf, m, w2_ref, w2_scale_ref,
                 w2_bufs_ref, w2_s_bufs_ref, sem_ref, *,
                 I_TILE, IB_TILE, H, IB, NBUF_W1_, NBUF_W2_):
    """Wait for w2 tile `m` of expert `gj` in buffer `buf`."""
    i_block_start = (m * I_TILE) // IB
    pltpu.make_async_copy(
        w2_ref.at[pl.ds(gj, 1), pl.ds(m * I_TILE, I_TILE), pl.ds(0, H)],
        w2_bufs_ref.at[pl.ds(buf, 1), pl.ds(0, I_TILE), pl.ds(0, H)],
        sem_ref.at[2 * NBUF_W1_ + buf]
    ).wait()
    pltpu.make_async_copy(
        w2_scale_ref.at[pl.ds(gj, 1), pl.ds(i_block_start, IB_TILE),
                        pl.ds(0, 1), pl.ds(0, H)],
        w2_s_bufs_ref.at[pl.ds(buf, 1), pl.ds(0, IB_TILE),
                         pl.ds(0, 1), pl.ds(0, H)],
        sem_ref.at[2 * NBUF_W1_ + NBUF_W2_ + buf]
    ).wait()


# =====================================================================
#  Compute-only expert body — no DMA; caller handles all transfers
# =====================================================================
def _expert_body(
    gj, weight, tok, lhs_scratch_ref,
    w1_ref, w1_scale_ref, w2_ref, w2_scale_ref,
    w1_bufs_ref, w1_s_bufs_ref, w2_bufs_ref, w2_s_bufs_ref,
    acc_scratch_ref, sem_ref,
    *, K, I, H, K_BLOCKS, QB, I_BLOCKS, IB,
    NBUF_W1_, NBUF_W2_,
    K_TILE, NUM_K, QB_eff, KB_TILE, I_TILE, NUM_I, IB_TILE,
    DTYPE_LHS, DTYPE_OUT, DEQUANT_W1_AFTER_, DEQUANT_W2_AFTER_,
    is_new_expert, cur_w1_buf, cur_w2_buf,
):
    """Process one slot: wait→w1 matmul→SwiGLU→wait→w2 matmul→accumulate.

    All prefetching is done by the caller (kernel loop).  This function
    only waits for the current expert's buffers and computes.
    """
    dma_kw_w1 = dict(w1_ref=w1_ref, w1_scale_ref=w1_scale_ref,
                     w1_bufs_ref=w1_bufs_ref, w1_s_bufs_ref=w1_s_bufs_ref,
                     sem_ref=sem_ref, K_TILE=K_TILE, KB_TILE=KB_TILE,
                     I=I, QB=QB, NBUF_W1_=NBUF_W1_)
    dma_kw_w2 = dict(w2_ref=w2_ref, w2_scale_ref=w2_scale_ref,
                     w2_bufs_ref=w2_bufs_ref, w2_s_bufs_ref=w2_s_bufs_ref,
                     sem_ref=sem_ref, I_TILE=I_TILE, IB_TILE=IB_TILE,
                     H=H, IB=IB, NBUF_W1_=NBUF_W1_, NBUF_W2_=NBUF_W2_)

    # ---- Phase 1: K-tiled gate+up matmul ----
    gate_up_acc = jnp.zeros((M_PAD, 2 * I), dtype=jnp.float32)

    for k in range(NUM_K):
        # Wait for current w1 tile (gated by is_new_expert)
        @pl.when(is_new_expert)
        def _():
            _wait_w1_dma(gj, cur_w1_buf, k, **dma_kw_w1)

        # ---- UNCONDITIONAL compute (buffer valid either way) ----
        # pl.ds keeps size-1 buffer axis; reshape it away.
        w1_fp8 = w1_bufs_ref[pl.ds(cur_w1_buf, 1), pl.ds(0, K_TILE),
                             pl.ds(0, 2 * I)].reshape(K_TILE, 2 * I)
        s1 = w1_s_bufs_ref[pl.ds(cur_w1_buf, 1), pl.ds(0, KB_TILE),
                           pl.ds(0, 1), pl.ds(0, 2 * I)].reshape(KB_TILE, 1, 2 * I)
        lhs_tile = lhs_scratch_ref[pl.ds(0, M_PAD), pl.ds(k * K_TILE, K_TILE)]

        if DEQUANT_W1_AFTER_:
            w1_cast = w1_fp8.astype(DTYPE_LHS)
            block_acc = jnp.matmul(lhs_tile, w1_cast,
                                   preferred_element_type=jnp.float32)
            s1_flat = s1.reshape(KB_TILE, 1, 2 * I)
            block_acc = block_acc * s1_flat.reshape(1, 2 * I).astype(jnp.float32)
            gate_up_acc = gate_up_acc + block_acc
        else:
            w1_fp32 = w1_fp8.astype(jnp.float32).reshape(KB_TILE, QB_eff, 2 * I)
            w1_dequant = (w1_fp32 * s1).reshape(K_TILE, 2 * I).astype(DTYPE_LHS)
            gate_up_acc = gate_up_acc + jnp.matmul(
                lhs_tile, w1_dequant, preferred_element_type=jnp.float32)

    # ---- SwiGLU ----
    gate = gate_up_acc[:, :I]
    up = gate_up_acc[:, I:]
    silu_gate = gate * jax.nn.sigmoid(gate)
    intermediate = (silu_gate * up).astype(DTYPE_LHS)

    # ---- Phase 2: I-tiled down matmul ----
    down_acc = jnp.zeros((M_PAD, H), dtype=jnp.float32)

    for m in range(NUM_I):
        # Wait for current w2 tile (gated by is_new_expert)
        @pl.when(is_new_expert)
        def _():
            _wait_w2_dma(gj, cur_w2_buf, m, **dma_kw_w2)

        # ---- UNCONDITIONAL w2 compute ----
        w2_fp8 = w2_bufs_ref[pl.ds(cur_w2_buf, 1), pl.ds(0, I_TILE),
                             pl.ds(0, H)].reshape(I_TILE, H)
        s2 = w2_s_bufs_ref[pl.ds(cur_w2_buf, 1), pl.ds(0, IB_TILE),
                           pl.ds(0, 1), pl.ds(0, H)].reshape(IB_TILE, 1, H)
        inter_tile = intermediate[:, m * I_TILE : (m + 1) * I_TILE]

        if DEQUANT_W2_AFTER_:
            w2_cast = w2_fp8.astype(DTYPE_LHS)
            block_acc = jnp.matmul(inter_tile, w2_cast,
                                   preferred_element_type=jnp.float32)
            s2_flat = s2.reshape(IB_TILE, 1, H)
            block_acc = block_acc * s2_flat.reshape(1, H).astype(jnp.float32)
            down_acc = down_acc + block_acc
        else:
            w2_fp32 = w2_fp8.astype(jnp.float32).reshape(
                IB_TILE, min(I_TILE, IB), H)
            w2_dequant = (w2_fp32 * s2).reshape(I_TILE, H).astype(DTYPE_LHS)
            down_acc = down_acc + jnp.matmul(
                inter_tile, w2_dequant, preferred_element_type=jnp.float32)

    # ---- Accumulate weighted expert contribution to this token's row ----
    result = (down_acc * weight).reshape(1, 1, H)
    current = acc_scratch_ref[pl.ds(tok, 1), pl.ds(0, 1), pl.ds(0, H)]
    acc_scratch_ref[pl.ds(tok, 1), pl.ds(0, 1), pl.ds(0, H)] = current + result


# =====================================================================
#  Main Pallas kernel
# =====================================================================
def _cn_w1w2_fused_token_kernel_fp8(
    # ---- Scalar prefetch (SMEM) ----
    ids_ref, toks_ref, topk_weights_ref,
    # ---- HBM inputs ----
    lhs_ref, w1_ref, w1_scale_ref, w2_ref, w2_scale_ref,
    # ---- HBM output ----
    o_ref,
    # ---- VMEM scratch ----
    full_lhs_scratch_ref,  # [C_PAD, 1, K]
    lhs_scratch_ref,       # [M_PAD, K]
    w1_bufs_ref,           # [NBUF_W1, K_TILE, 2I]
    w1_s_bufs_ref,         # [NBUF_W1, KB_TILE, 1, 2I] fp32
    w2_bufs_ref,           # [NBUF_W2, I_TILE, H]
    w2_s_bufs_ref,         # [NBUF_W2, IB_TILE, 1, H] fp32
    acc_scratch_ref,       # [N_TOKENS_, 1, H] fp32 (per-token accumulator)
    full_out_scratch_ref,  # [C_PAD, 1, H] bf16
    sem_ref,               # semaphores
    *,
    K, I, H, K_BLOCKS, QB, I_BLOCKS, IB, TOP_K_,
    NBUF_W1_, NBUF_W2_,
    N_TOKENS_, DEQUANT_W1_AFTER_, DEQUANT_W2_AFTER_,
    SKIP_ZERO_WEIGHT_, DTYPE_LHS, DTYPE_OUT,
):
    C_PAD = full_lhs_scratch_ref.shape[0]
    N_SLOTS = N_TOKENS_ * TOP_K_

    K_TILE = w1_bufs_ref.shape[1]
    NUM_K = K // K_TILE
    QB_eff = min(K_TILE, QB)
    KB_TILE = K_TILE // QB_eff
    I_TILE = w2_bufs_ref.shape[1]
    NUM_I = I // I_TILE
    IB_TILE = I_TILE // min(I_TILE, IB)

    # DMA helper kwargs
    dma_kw_w1 = dict(w1_ref=w1_ref, w1_scale_ref=w1_scale_ref,
                     w1_bufs_ref=w1_bufs_ref, w1_s_bufs_ref=w1_s_bufs_ref,
                     sem_ref=sem_ref, K_TILE=K_TILE, KB_TILE=KB_TILE,
                     I=I, QB=QB, NBUF_W1_=NBUF_W1_)
    dma_kw_w2 = dict(w2_ref=w2_ref, w2_scale_ref=w2_scale_ref,
                     w2_bufs_ref=w2_bufs_ref, w2_s_bufs_ref=w2_s_bufs_ref,
                     sem_ref=sem_ref, I_TILE=I_TILE, IB_TILE=IB_TILE,
                     H=H, IB=IB, NBUF_W1_=NBUF_W1_, NBUF_W2_=NBUF_W2_)

    # Compute-body kwargs (everything except DMA-related, buffer indices,
    # and per-slot values that change each iteration)
    body_kw = dict(
        lhs_scratch_ref=lhs_scratch_ref,
        w1_ref=w1_ref, w1_scale_ref=w1_scale_ref,
        w2_ref=w2_ref, w2_scale_ref=w2_scale_ref,
        w1_bufs_ref=w1_bufs_ref, w1_s_bufs_ref=w1_s_bufs_ref,
        w2_bufs_ref=w2_bufs_ref, w2_s_bufs_ref=w2_s_bufs_ref,
        acc_scratch_ref=acc_scratch_ref, sem_ref=sem_ref,
        K=K, I=I, H=H, K_BLOCKS=K_BLOCKS, QB=QB,
        I_BLOCKS=I_BLOCKS, IB=IB,
        NBUF_W1_=NBUF_W1_, NBUF_W2_=NBUF_W2_,
        K_TILE=K_TILE, NUM_K=NUM_K, QB_eff=QB_eff, KB_TILE=KB_TILE,
        I_TILE=I_TILE, NUM_I=NUM_I, IB_TILE=IB_TILE,
        DTYPE_LHS=DTYPE_LHS, DTYPE_OUT=DTYPE_OUT,
        DEQUANT_W1_AFTER_=DEQUANT_W1_AFTER_,
        DEQUANT_W2_AFTER_=DEQUANT_W2_AFTER_,
    )

    # ---- DMA ALL tokens from HBM -> VMEM (once) ----
    full_lhs_copy = pltpu.make_async_copy(
        lhs_ref.at[pl.ds(0, C_PAD), pl.ds(0, 1), pl.ds(0, K)],
        full_lhs_scratch_ref,
        sem_ref.at[2 * NBUF_W1_ + 2 * NBUF_W2_]
    )
    full_lhs_copy.start()
    full_lhs_copy.wait()

    # Initialize output scratch and per-token accumulator to zeros
    full_out_scratch_ref[...] = jnp.zeros((C_PAD, 1, H), dtype=DTYPE_OUT)
    acc_scratch_ref[...] = jnp.zeros((N_TOKENS_, 1, H), dtype=jnp.float32)

    # ---- Flat slot loop (Python-unrolled, sorted by expert) ----
    if SKIP_ZERO_WEIGHT_:
        # EP mode: no dedup, no cross-expert prefetch.
        # Each slot bootstraps its own DMA independently.
        for slot in range(N_SLOTS):
            gj = ids_ref[slot]
            gj = pl.multiple_of(gj, 1)
            tok = toks_ref[slot]
            weight = topk_weights_ref[slot]

            token_row = full_lhs_scratch_ref[pl.ds(tok, 1), pl.ds(0, 1),
                                             pl.ds(0, K)]
            lhs_scratch_ref[...] = token_row.reshape(1, K)

            @pl.when(weight != 0.0)
            def _():
                # Bootstrap: start DMA for this slot's expert into buf 0
                for k in range(NUM_K):
                    _start_w1_dma(gj, jnp.int32(0), k, **dma_kw_w1)
                for m in range(NUM_I):
                    _start_w2_dma(gj, jnp.int32(0), m, **dma_kw_w2)

                _expert_body(gj, weight, tok,
                             is_new_expert=jnp.bool_(True),
                             cur_w1_buf=jnp.int32(0),
                             cur_w2_buf=jnp.int32(0),
                             **body_kw)
    else:
        # TP mode: dedup + cross-expert double buffering.
        # Traced buffer indices — swap only on genuine expert transitions.
        cur_w1_buf = jnp.int32(0)
        cur_w2_buf = jnp.int32(0)

        for slot in range(N_SLOTS):
            gj = ids_ref[slot]
            gj = pl.multiple_of(gj, 1)
            tok = toks_ref[slot]
            weight = topk_weights_ref[slot]

            # ---- Dedup: is this a new expert? ----
            if slot == 0:
                is_new_expert = jnp.bool_(True)
            else:
                prev_gj = ids_ref[slot - 1]
                is_new_expert = (gj != prev_gj)

            # ---- On genuine transition, swap to the prefetched buffer ----
            if slot > 0:
                cur_w1_buf = jnp.where(is_new_expert,
                                       1 - cur_w1_buf, cur_w1_buf)
                cur_w2_buf = jnp.where(is_new_expert,
                                       1 - cur_w2_buf, cur_w2_buf)

            # ---- EARLY PREFETCH: next expert into alternate buffer ----
            # Fires at the TOP of each slot, giving the DMA the entire
            # slot's phase-1 + SwiGLU + phase-2 time to complete.
            if slot + 1 < N_SLOTS:
                _next_gj = pl.multiple_of(ids_ref[slot + 1], 1)
                _is_next_new = (_next_gj != gj)
                nxt_w1_buf = 1 - cur_w1_buf
                nxt_w2_buf = 1 - cur_w2_buf
                @pl.when(_is_next_new)
                def _():
                    for k in range(NUM_K):
                        _start_w1_dma(_next_gj, nxt_w1_buf, k, **dma_kw_w1)
                    for m in range(NUM_I):
                        _start_w2_dma(_next_gj, nxt_w2_buf, m, **dma_kw_w2)

            # ---- Slot 0 bootstrap: start DMA for first expert ----
            if slot == 0:
                for k in range(NUM_K):
                    _start_w1_dma(gj, cur_w1_buf, k, **dma_kw_w1)
                for m in range(NUM_I):
                    _start_w2_dma(gj, cur_w2_buf, m, **dma_kw_w2)

            # ---- Load this token's lhs ----
            token_row = full_lhs_scratch_ref[pl.ds(tok, 1), pl.ds(0, 1),
                                             pl.ds(0, K)]
            lhs_scratch_ref[...] = token_row.reshape(1, K)

            # ---- Compute: wait + matmul + SwiGLU + matmul + accumulate ----
            _expert_body(gj, weight, tok,
                         is_new_expert=is_new_expert,
                         cur_w1_buf=cur_w1_buf,
                         cur_w2_buf=cur_w2_buf,
                         **body_kw)

    # ---- Final: copy each token's accumulated result to output ----
    for t in range(N_TOKENS_):
        row = acc_scratch_ref[pl.ds(t, 1), pl.ds(0, 1), pl.ds(0, H)]
        full_out_scratch_ref[pl.ds(t, 1), pl.ds(0, 1), pl.ds(0, H)] = \
            row.astype(DTYPE_OUT)

    # ---- DMA full output back to HBM ----
    out_copy = pltpu.make_async_copy(
        full_out_scratch_ref,
        o_ref.at[pl.ds(0, C_PAD), pl.ds(0, 1), pl.ds(0, H)],
        sem_ref.at[2 * NBUF_W1_ + 2 * NBUF_W2_ + 1]
    )
    out_copy.start()
    out_copy.wait()


# =====================================================================
#  Public entry point — sets up grid, scratch, and launches Pallas
# =====================================================================
def cn_gemv_w1w2_fused_mb_fp8(lhs, w1, w1_scale, w2, w2_scale,
                              sorted_ids, sorted_toks, sorted_weights, *,
                              n_tokens, use_ep=False, interpret=False):
    """Fused gate+up+SwiGLU+down for the C=N MoE block with expert dedup.

    Inputs are pre-sorted by expert_id so duplicate experts are adjacent.
    sorted_ids [N*TOP_K]: expert indices (sorted).
    sorted_toks [N*TOP_K]: token index each slot belongs to.
    sorted_weights [N*TOP_K]: routing weights (sorted to match).
    """
    G, K, N1 = w1.shape
    assert N1 % 2 == 0, f"w1 last dim must be 2*I; got {N1}"
    I = N1 // 2
    G2, I2, H = w2.shape
    assert I == I2 and G == G2
    K_BLOCKS = w1_scale.shape[1]
    I_BLOCKS = w2_scale.shape[1]
    QB = K // K_BLOCKS
    IB = I // I_BLOCKS
    TOPK_TOTAL = n_tokens * TOP_K
    NBUF = min(_FUSE_NBUF, TOPK_TOTAL)
    sorted_weights = sorted_weights.astype(jnp.float32)

    any_spec = pl.BlockSpec(memory_space=pl.ANY)
    K_TILE = min(_CN_K_TILE, K)
    while K % K_TILE != 0:
        K_TILE -= 1
    QB_eff = min(K_TILE, QB)
    KB_TILE = K_TILE // QB_eff
    # Find largest tile <= _CN_I_TILE that divides I
    I_TILE = min(_CN_I_TILE, I)
    while I % I_TILE != 0:
        I_TILE -= 1
    IB_eff = min(I_TILE, IB)
    IB_TILE = I_TILE // IB_eff

    NUM_K_setup = K // K_TILE
    NUM_I_setup = I // I_TILE
    # Cross-expert double buffering: always ≥ 2 buffers.
    NBUF_W1 = max(2, NUM_K_setup)
    NBUF_W2 = max(2, NUM_I_setup)

    C_PAD = lhs.shape[0]
    grid_spec = pltpu.PrefetchScalarGridSpec(
        num_scalar_prefetch=3,            # ids, toks, weights
        in_specs=[any_spec] * 5,
        out_specs=any_spec,
        grid=(1,),
        scratch_shapes=[
            pltpu.VMEM((C_PAD, 1, K), lhs.dtype),                   # full_lhs_scratch
            pltpu.VMEM((M_PAD, K), lhs.dtype),                      # lhs_scratch
            pltpu.VMEM((NBUF_W1, K_TILE, N1), w1.dtype),            # w1_bufs
            pltpu.VMEM((NBUF_W1, KB_TILE, 1, N1), w1_scale.dtype),  # w1_s_bufs
            pltpu.VMEM((NBUF_W2, I_TILE, H), w2.dtype),             # w2_bufs
            pltpu.VMEM((NBUF_W2, IB_TILE, 1, H), w2_scale.dtype),   # w2_s_bufs
            pltpu.VMEM((n_tokens, 1, H), jnp.float32),              # acc_scratch
            pltpu.VMEM((C_PAD, 1, H), jnp.bfloat16),                # full_out_scratch
            pltpu.SemaphoreType.DMA((2 * NBUF_W1 + 2 * NBUF_W2 + 2,)),
        ])
    compiler_params = None if interpret else pltpu.CompilerParams(
        vmem_limit_bytes=int(pltpu.get_tpu_info().vmem_capacity_bytes * 0.9))

    return pl.pallas_call(
        functools.partial(_cn_w1w2_fused_token_kernel_fp8,
                          K=K, I=I, H=H,
                          K_BLOCKS=K_BLOCKS, QB=QB,
                          I_BLOCKS=I_BLOCKS, IB=IB,
                          TOP_K_=TOP_K,
                          NBUF_W1_=NBUF_W1, NBUF_W2_=NBUF_W2,
                          N_TOKENS_=n_tokens,
                          DEQUANT_W1_AFTER_=(KB_TILE == 1),
                          DEQUANT_W2_AFTER_=(IB_TILE == 1),
                          SKIP_ZERO_WEIGHT_=use_ep,
                          DTYPE_LHS=lhs.dtype, DTYPE_OUT=jnp.bfloat16),
        out_shape=jax.ShapeDtypeStruct((C_PAD, 1, H), jnp.bfloat16),
        grid_spec=grid_spec, compiler_params=compiler_params,
        interpret=interpret, name="cn_gemv_w1w2_fused_token_fp8",
    )(sorted_ids, sorted_toks, sorted_weights,   # scalar prefetch (3)
      lhs, w1, w1_scale, w2, w2_scale)           # inputs (5)


def cn_moe_full(hidden_state, w1, w1_scale, w2, w2_scale,
                active_ids, topk_weights, *, use_ep=False, interpret=False):
    """hidden_state [C, K]; active_ids/topk_weights [C, TOP_K].
    Returns [C, K] -- token t's MoE output at row t.

    Pre-sorts slots by expert_id to enable weight reuse for duplicate
    experts across tokens (dedup via pl.when inside the kernel).
    """
    C = hidden_state.shape[0]
    K = hidden_state.shape[1]

    lhs = hidden_state.reshape(C, 1, K)  # 3D for 1x128 tiling

    # ---- Pre-sort by expert_id to group duplicates ----
    flat_ids = active_ids.reshape(C * TOP_K)
    flat_weights = topk_weights.reshape(C * TOP_K)
    sort_order = jnp.argsort(flat_ids, stable=True)
    sorted_ids = flat_ids[sort_order]
    sorted_toks = (sort_order // TOP_K).astype(jnp.int32)
    sorted_weights = flat_weights[sort_order]

    fused_out = cn_gemv_w1w2_fused_mb_fp8(
        lhs, w1, w1_scale, w2, w2_scale,
        sorted_ids, sorted_toks, sorted_weights,
        n_tokens=C, use_ep=use_ep, interpret=interpret)

    return fused_out[:C, 0, :].astype(jnp.bfloat16)
