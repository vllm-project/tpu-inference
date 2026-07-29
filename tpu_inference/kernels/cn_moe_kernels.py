# Copyright 2026 Google LLC
"""C=N fused MoE Pallas kernels — per-token expert loop with dedup.

N-way cross-expert buffering: buffer `cur` holds the active expert,
buffers `cur+1 .. cur+NBUF-2` hold pre-fetched upcoming experts.  On
each genuine expert transition `cur` rotates by +1 and a SINGLE new
prefetch fires into the freed slot `(cur+NBUF-1) % NBUF`.

Lookahead is precomputed on the host via group_id/group_expert_table
and passed as scalar-prefetch arrays — the kernel does O(1) SMEM reads
instead of O(N_SLOTS) traced scans.

Set MOE_CN_NBUF=2 for double, =3 for triple (default), etc.
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
_CN_NBUF = max(2, int(_os.getenv("MOE_CN_NBUF", "3")))

# ── Semaphore layout ────────────────────────────────────────────────
#   [0                            .. NBUF_W1)           w1 weight DMAs
#   [NBUF_W1                     .. 2*NBUF_W1)         w1 scale  DMAs
#   [2*NBUF_W1                   .. 2*NBUF_W1+NBUF_W2) w2 weight DMAs
#   [2*NBUF_W1+NBUF_W2           .. 2*NBUF_W1+2*NBUF_W2) w2 scale DMAs
#   [2*NBUF_W1+2*NBUF_W2]        lhs DMA
#   [2*NBUF_W1+2*NBUF_W2+1]      output DMA
# ────────────────────────────────────────────────────────────────────


# =====================================================================
#  Buffer selection
# =====================================================================
def _select_buf(ref, cur_buf, nbuf):
    """Read buffer slot `cur_buf` from ref[nbuf, ...].

    Uses a jnp.where chain — each ref[i] is a static-int index that
    Mosaic handles correctly.  Faster than lax.switch in practice.
    """
    result = ref[0]
    for i in range(1, nbuf):
        result = jnp.where(cur_buf == i, ref[i], result)
    return result


# =====================================================================
#  DMA helpers
# =====================================================================
def _start_w1_dma(gj, buf, k, w1_ref, w1_scale_ref,
                  w1_bufs_ref, w1_s_bufs_ref, sem_ref, *,
                  K_TILE, KB_TILE, I, QB, NBUF_W1_):
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


def _start_all_dma(gj, buf_w1, buf_w2, *, NUM_K, NUM_I,
                   dma_kw_w1, dma_kw_w2):
    """Start DMA for all tiles of expert `gj` into the given buffers."""
    for k in range(NUM_K):
        _start_w1_dma(gj, buf_w1, k, **dma_kw_w1)
    for m in range(NUM_I):
        _start_w2_dma(gj, buf_w2, m, **dma_kw_w2)


# =====================================================================
#  Compute-only expert body
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
    """Process one slot: wait→w1 matmul→SwiGLU→wait→w2 matmul→accumulate."""
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
        @pl.when(is_new_expert)
        def _():
            _wait_w1_dma(gj, cur_w1_buf, k, **dma_kw_w1)

        w1_fp8 = _select_buf(w1_bufs_ref, cur_w1_buf, NBUF_W1_)
        s1 = _select_buf(w1_s_bufs_ref, cur_w1_buf, NBUF_W1_)
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
        @pl.when(is_new_expert)
        def _():
            _wait_w2_dma(gj, cur_w2_buf, m, **dma_kw_w2)

        w2_fp8 = _select_buf(w2_bufs_ref, cur_w2_buf, NBUF_W2_)
        s2 = _select_buf(w2_s_bufs_ref, cur_w2_buf, NBUF_W2_)
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
    seed_experts_ref,       # [NBUF]: expert id for seed buffers 0..NBUF-1
    lookahead_ids_ref,      # [N_SLOTS]: expert MAX_DEPTH groups ahead (or -1)
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
    acc_scratch_ref,       # [N_TOKENS_, 1, H] fp32
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
    all_dma_kw = dict(NUM_K=NUM_K, NUM_I=NUM_I,
                      dma_kw_w1=dma_kw_w1, dma_kw_w2=dma_kw_w2)

    # Compute-body kwargs
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

    NBUF = max(NBUF_W1_, NBUF_W2_)  # same in practice
    MAX_DEPTH = NBUF - 1             # how far ahead we look

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
                _start_all_dma(gj, jnp.int32(0), jnp.int32(0), **all_dma_kw)
                _expert_body(gj, weight, tok,
                             is_new_expert=jnp.bool_(True),
                             cur_w1_buf=jnp.int32(0),
                             cur_w2_buf=jnp.int32(0),
                             **body_kw)
    else:
        # TP mode: dedup + N-way cross-expert buffering.
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

            # ============================================================
            #  SLOT 0 — SEED PHASE: fill all NBUF buffers from
            #  precomputed seed_experts_ref (O(1) SMEM reads).
            # ============================================================
            if slot == 0:
                # Buffer 0: current expert
                _start_all_dma(gj, jnp.int32(0), jnp.int32(0), **all_dma_kw)

                # Buffers 1..NBUF-1: precomputed seed experts
                for d in range(1, NBUF):
                    seed_gj = seed_experts_ref[d]
                    buf_d_w1 = jnp.int32(d % NBUF_W1_)
                    buf_d_w2 = jnp.int32(d % NBUF_W2_)
                    @pl.when(seed_gj >= 0)
                    def _(sgj=seed_gj, bw1=buf_d_w1, bw2=buf_d_w2):
                        _start_all_dma(
                            pl.multiple_of(sgj, 1), bw1, bw2, **all_dma_kw)

            # ============================================================
            #  SLOT > 0 — STEADY STATE: rotate + one prefetch from
            #  precomputed lookahead_ids_ref (O(1) SMEM read).
            # ============================================================
            if slot > 0:
                # Rotate buffer index on genuine transition
                cur_w1_buf = jnp.where(is_new_expert,
                                       (cur_w1_buf + 1) % NBUF_W1_,
                                       cur_w1_buf)
                cur_w2_buf = jnp.where(is_new_expert,
                                       (cur_w2_buf + 1) % NBUF_W2_,
                                       cur_w2_buf)

                # ONE prefetch into the freed buffer.
                # lookahead_ids_ref[slot] holds the expert MAX_DEPTH
                # transitions ahead, or -1 if none exists.
                target_w1 = (cur_w1_buf + MAX_DEPTH) % NBUF_W1_
                target_w2 = (cur_w2_buf + MAX_DEPTH) % NBUF_W2_
                ahead_gj = lookahead_ids_ref[slot]
                should_prefetch = is_new_expert & (ahead_gj >= 0)
                @pl.when(should_prefetch)
                def _():
                    _start_all_dma(
                        pl.multiple_of(ahead_gj, 1),
                        target_w1, target_w2, **all_dma_kw)

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
#  Host-side lookahead precomputation
# =====================================================================
def _build_lookahead_tables(sorted_ids, n_slots, nbuf):
    """Build seed_experts and lookahead_ids from sorted expert IDs.

    sorted_ids [N_SLOTS]: expert indices, sorted so duplicates are adjacent.

    Returns:
        seed_experts [NBUF]:   expert id for seed buffers 0..NBUF-1.
                               seed_experts[0] = sorted_ids[0] (current),
                               seed_experts[d] = d-th unique expert (or -1).
        lookahead_ids [N_SLOTS]: for each slot, the expert MAX_DEPTH groups
                                 ahead (or -1 if none).
    """
    max_depth = nbuf - 1

    # ---- Group structure ----
    # is_transition[0] = True (first slot always starts a group)
    is_transition = jnp.concatenate([
        jnp.array([True]),
        sorted_ids[1:] != sorted_ids[:-1]
    ])
    # group_id[s] = which group slot s belongs to (0-indexed)
    group_id = jnp.cumsum(is_transition.astype(jnp.int32)) - 1
    num_groups = group_id[-1] + 1  # traced; max possible = n_slots

    # ---- Group → expert table ----
    # group_expert[g] = expert id for group g.  Padded with -1.
    table_size = n_slots + nbuf  # pad for safe indexing
    group_expert = jnp.full(table_size, -1, dtype=jnp.int32)
    group_expert = group_expert.at[group_id].set(sorted_ids)

    # ---- Seed experts: groups 0..NBUF-1 ----
    seed_experts = jnp.array(
        [group_expert[d] for d in range(nbuf)], dtype=jnp.int32)

    # ---- Lookahead: for each slot, expert MAX_DEPTH groups ahead ----
    target_groups = group_id + max_depth
    # Clamp to table_size-1 so the gather is always in bounds;
    # out-of-range entries will read the -1 padding.
    clamped = jnp.minimum(target_groups, table_size - 1)
    lookahead_ids = group_expert[clamped]

    return seed_experts, lookahead_ids


# =====================================================================
#  Public entry point
# =====================================================================
def cn_gemv_w1w2_fused_mb_fp8(lhs, w1, w1_scale, w2, w2_scale,
                              sorted_ids, sorted_toks, sorted_weights, *,
                              n_tokens, use_ep=False, interpret=False):
    """Fused gate+up+SwiGLU+down for the C=N MoE block with expert dedup."""
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
    I_TILE = min(_CN_I_TILE, I)
    while I % I_TILE != 0:
        I_TILE -= 1
    IB_eff = min(I_TILE, IB)
    IB_TILE = I_TILE // IB_eff

    NUM_K_setup = K // K_TILE
    NUM_I_setup = I // I_TILE
    NBUF_W1 = max(_CN_NBUF, NUM_K_setup)
    NBUF_W2 = max(_CN_NBUF, NUM_I_setup)
    NBUF_EFF = max(NBUF_W1, NBUF_W2)

    # ---- Precompute lookahead tables (host-side, O(N_SLOTS)) ----
    seed_experts, lookahead_ids = _build_lookahead_tables(
        sorted_ids, TOPK_TOTAL, NBUF_EFF)

    C_PAD = lhs.shape[0]
    grid_spec = pltpu.PrefetchScalarGridSpec(
        num_scalar_prefetch=5,        # ids, toks, weights, seed, lookahead
        in_specs=[any_spec] * 5,
        out_specs=any_spec,
        grid=(1,),
        scratch_shapes=[
            pltpu.VMEM((C_PAD, 1, K), lhs.dtype),
            pltpu.VMEM((M_PAD, K), lhs.dtype),
            pltpu.VMEM((NBUF_W1, K_TILE, N1), w1.dtype),
            pltpu.VMEM((NBUF_W1, KB_TILE, 1, N1), w1_scale.dtype),
            pltpu.VMEM((NBUF_W2, I_TILE, H), w2.dtype),
            pltpu.VMEM((NBUF_W2, IB_TILE, 1, H), w2_scale.dtype),
            pltpu.VMEM((n_tokens, 1, H), jnp.float32),
            pltpu.VMEM((C_PAD, 1, H), jnp.bfloat16),
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
    )(sorted_ids, sorted_toks, sorted_weights,  # scalar prefetch (5)
      seed_experts, lookahead_ids,
      lhs, w1, w1_scale, w2, w2_scale)          # inputs (5)


def cn_moe_full(hidden_state, w1, w1_scale, w2, w2_scale,
                active_ids, topk_weights, *, use_ep=False, interpret=False):
    """hidden_state [C, K]; active_ids/topk_weights [C, TOP_K].
    Returns [C, K] -- token t's MoE output at row t.
    """
    C = hidden_state.shape[0]
    K = hidden_state.shape[1]
    lhs = hidden_state.reshape(C, 1, K)

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
