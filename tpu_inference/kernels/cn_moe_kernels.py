# Copyright 2026 Google LLC
"""C=N fused MoE Pallas kernels — per-token expert loop with dedup."""

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
_CN_K_TILE = int(_os.getenv("MOE_CN_K_TILE", "2048"))
_CN_I_TILE = int(_os.getenv("MOE_CN_I_TILE", "2048"))


def _expert_body(
    gj, weight, tok, lhs_scratch_ref,
    w1_ref, w1_scale_ref, w2_ref, w2_scale_ref,
    w1_bufs_ref, w1_s_bufs_ref, w2_bufs_ref, w2_s_bufs_ref,
    acc_scratch_ref, sem_ref,
    *, K, I, H, K_BLOCKS, QB, I_BLOCKS, IB, NBUF_,
    K_TILE, NUM_K, QB_eff, KB_TILE, I_TILE, NUM_I, IB_TILE,
    DTYPE_LHS, DTYPE_OUT, DEQUANT_W1_AFTER_, DEQUANT_W2_AFTER_,
    is_new_expert, is_first_slot, next_expert_gj, is_next_new,
):
    """Core expert computation with dedup gating.

    When is_new_expert is False (duplicate expert), all weight DMA is skipped
    and the computation reuses weights already in VMEM buffers from the
    previous slot. The matmul + accumulate always runs (different token).
    """

    # ---- Self-prefetch w1 tile 0 (only first slot; others use cross-expert prefetch) ----
    if is_first_slot:
        pltpu.make_async_copy(
            w1_ref.at[pl.ds(gj, 1), pl.ds(0, K_TILE), pl.ds(0, 2 * I)],
            w1_bufs_ref.at[pl.ds(0, 1), pl.ds(0, K_TILE), pl.ds(0, 2 * I)],
            sem_ref.at[0 * NBUF_ + 0]
        ).start()
        pltpu.make_async_copy(
            w1_scale_ref.at[pl.ds(gj, 1), pl.ds(0, KB_TILE), pl.ds(0, 1), pl.ds(0, 2 * I)],
            w1_s_bufs_ref.at[pl.ds(0, 1), pl.ds(0, KB_TILE), pl.ds(0, 1), pl.ds(0, 2 * I)],
            sem_ref.at[1 * NBUF_ + 0]
        ).start()

    # ---- Phase 1: K-tiled gate+up matmul ----
    gate_up_acc = jnp.zeros((M_PAD, 2 * I), dtype=jnp.float32)

    for k in range(NUM_K):
        buf = k % NBUF_
        k_block_start = (k * K_TILE) // QB

        @pl.when(is_new_expert)
        def _():
            # Start next tile DMA (if exists)
            nxt_k = k + 1
            if nxt_k < NUM_K:
                nxt_buf = nxt_k % NBUF_
                pltpu.make_async_copy(
                    w1_ref.at[pl.ds(gj, 1), pl.ds(nxt_k * K_TILE, K_TILE), pl.ds(0, 2 * I)],
                    w1_bufs_ref.at[pl.ds(nxt_buf, 1), pl.ds(0, K_TILE), pl.ds(0, 2 * I)],
                    sem_ref.at[0 * NBUF_ + nxt_buf]
                ).start()
                nxt_k_block_start = (nxt_k * K_TILE) // QB
                pltpu.make_async_copy(
                    w1_scale_ref.at[pl.ds(gj, 1), pl.ds(nxt_k_block_start, KB_TILE), pl.ds(0, 1), pl.ds(0, 2 * I)],
                    w1_s_bufs_ref.at[pl.ds(nxt_buf, 1), pl.ds(0, KB_TILE), pl.ds(0, 1), pl.ds(0, 2 * I)],
                    sem_ref.at[1 * NBUF_ + nxt_buf]
                ).start()

            # Wait for current tile
            pltpu.make_async_copy(
                w1_ref.at[pl.ds(gj, 1), pl.ds(k * K_TILE, K_TILE), pl.ds(0, 2 * I)],
                w1_bufs_ref.at[pl.ds(buf, 1), pl.ds(0, K_TILE), pl.ds(0, 2 * I)],
                sem_ref.at[0 * NBUF_ + buf]
            ).wait()
            pltpu.make_async_copy(
                w1_scale_ref.at[pl.ds(gj, 1), pl.ds(k_block_start, KB_TILE), pl.ds(0, 1), pl.ds(0, 2 * I)],
                w1_s_bufs_ref.at[pl.ds(buf, 1), pl.ds(0, KB_TILE), pl.ds(0, 1), pl.ds(0, 2 * I)],
                sem_ref.at[1 * NBUF_ + buf]
            ).wait()

        # ---- UNCONDITIONAL compute (buffer valid either way) ----
        w1_fp8 = w1_bufs_ref[buf]
        s1 = w1_s_bufs_ref[buf]
        lhs_tile = lhs_scratch_ref[pl.ds(0, M_PAD), pl.ds(k * K_TILE, K_TILE)]

        if DEQUANT_W1_AFTER_:
            w1_cast = w1_fp8.astype(DTYPE_LHS)
            block_acc = jnp.matmul(lhs_tile, w1_cast, preferred_element_type=jnp.float32)
            s1_flat = s1.reshape(KB_TILE, 1, 2 * I)
            block_acc = block_acc * s1_flat.reshape(1, 2 * I).astype(jnp.float32)
            gate_up_acc = gate_up_acc + block_acc
        else:
            w1_fp32 = w1_fp8.astype(jnp.float32).reshape(KB_TILE, QB_eff, 2 * I)
            w1_dequant = (w1_fp32 * s1).reshape(K_TILE, 2 * I).astype(DTYPE_LHS)
            gate_up_acc = gate_up_acc + jnp.matmul(lhs_tile, w1_dequant, preferred_element_type=jnp.float32)

    # ---- Prefetch current expert's w2 tile 0 (only first slot) ----
    if is_first_slot:
        pltpu.make_async_copy(
            w2_ref.at[pl.ds(gj, 1), pl.ds(0, I_TILE), pl.ds(0, H)],
            w2_bufs_ref.at[pl.ds(0, 1), pl.ds(0, I_TILE), pl.ds(0, H)],
            sem_ref.at[2 * NBUF_ + 0]
        ).start()
        pltpu.make_async_copy(
            w2_scale_ref.at[pl.ds(gj, 1), pl.ds(0, IB_TILE), pl.ds(0, 1), pl.ds(0, H)],
            w2_s_bufs_ref.at[pl.ds(0, 1), pl.ds(0, IB_TILE), pl.ds(0, 1), pl.ds(0, H)],
            sem_ref.at[3 * NBUF_ + 0]
        ).start()

    # ---- Prefetch NEXT expert's w1 tile 0 (gated by is_next_new) ----
    if next_expert_gj is not None:
        @pl.when(is_next_new)
        def _():
            pltpu.make_async_copy(
                w1_ref.at[pl.ds(next_expert_gj, 1), pl.ds(0, K_TILE), pl.ds(0, 2 * I)],
                w1_bufs_ref.at[pl.ds(0, 1), pl.ds(0, K_TILE), pl.ds(0, 2 * I)],
                sem_ref.at[0 * NBUF_ + 0]
            ).start()
            pltpu.make_async_copy(
                w1_scale_ref.at[pl.ds(next_expert_gj, 1), pl.ds(0, KB_TILE), pl.ds(0, 1), pl.ds(0, 2 * I)],
                w1_s_bufs_ref.at[pl.ds(0, 1), pl.ds(0, KB_TILE), pl.ds(0, 1), pl.ds(0, 2 * I)],
                sem_ref.at[1 * NBUF_ + 0]
            ).start()

    # ---- SwiGLU (DMA runs in parallel) ----
    gate = gate_up_acc[:, :I]
    up = gate_up_acc[:, I:]
    silu_gate = gate * jax.nn.sigmoid(gate)
    intermediate = (silu_gate * up).astype(DTYPE_LHS)

    # ---- Phase 2: I-tiled down matmul ----
    down_acc = jnp.zeros((M_PAD, H), dtype=jnp.float32)

    for m in range(NUM_I):
        buf = m % NBUF_
        i_block_start = (m * I_TILE) // IB

        @pl.when(is_new_expert)
        def _():
            # Start next w2 tile DMA (if exists)
            nxt_m = m + 1
            if nxt_m < NUM_I:
                nxt_buf = nxt_m % NBUF_
                nxt_i_block_start = (nxt_m * I_TILE) // IB
                pltpu.make_async_copy(
                    w2_ref.at[pl.ds(gj, 1), pl.ds(nxt_m * I_TILE, I_TILE), pl.ds(0, H)],
                    w2_bufs_ref.at[pl.ds(nxt_buf, 1), pl.ds(0, I_TILE), pl.ds(0, H)],
                    sem_ref.at[2 * NBUF_ + nxt_buf]
                ).start()
                pltpu.make_async_copy(
                    w2_scale_ref.at[pl.ds(gj, 1), pl.ds(nxt_i_block_start, IB_TILE), pl.ds(0, 1), pl.ds(0, H)],
                    w2_s_bufs_ref.at[pl.ds(nxt_buf, 1), pl.ds(0, IB_TILE), pl.ds(0, 1), pl.ds(0, H)],
                    sem_ref.at[3 * NBUF_ + nxt_buf]
                ).start()

            # Wait for current w2 tile
            pltpu.make_async_copy(
                w2_ref.at[pl.ds(gj, 1), pl.ds(m * I_TILE, I_TILE), pl.ds(0, H)],
                w2_bufs_ref.at[pl.ds(buf, 1), pl.ds(0, I_TILE), pl.ds(0, H)],
                sem_ref.at[2 * NBUF_ + buf]
            ).wait()
            pltpu.make_async_copy(
                w2_scale_ref.at[pl.ds(gj, 1), pl.ds(i_block_start, IB_TILE), pl.ds(0, 1), pl.ds(0, H)],
                w2_s_bufs_ref.at[pl.ds(buf, 1), pl.ds(0, IB_TILE), pl.ds(0, 1), pl.ds(0, H)],
                sem_ref.at[3 * NBUF_ + buf]
            ).wait()

        # ---- UNCONDITIONAL w2 compute ----
        w2_fp8 = w2_bufs_ref[buf]
        s2 = w2_s_bufs_ref[buf]
        inter_tile = intermediate[:, m * I_TILE : (m + 1) * I_TILE]

        if DEQUANT_W2_AFTER_:
            w2_cast = w2_fp8.astype(DTYPE_LHS)
            block_acc = jnp.matmul(inter_tile, w2_cast, preferred_element_type=jnp.float32)
            s2_flat = s2.reshape(IB_TILE, 1, H)
            block_acc = block_acc * s2_flat.reshape(1, H).astype(jnp.float32)
            down_acc = down_acc + block_acc
        else:
            w2_fp32 = w2_fp8.astype(jnp.float32).reshape(IB_TILE, min(I_TILE, IB), H)
            w2_dequant = (w2_fp32 * s2).reshape(I_TILE, H).astype(DTYPE_LHS)
            down_acc = down_acc + jnp.matmul(inter_tile, w2_dequant, preferred_element_type=jnp.float32)

        # After m=0 consumed: prefetch NEXT expert's w2 tile 0 (gated)
        if m == 0 and next_expert_gj is not None:
            @pl.when(is_next_new)
            def _():
                pltpu.make_async_copy(
                    w2_ref.at[pl.ds(next_expert_gj, 1), pl.ds(0, I_TILE), pl.ds(0, H)],
                    w2_bufs_ref.at[pl.ds(0, 1), pl.ds(0, I_TILE), pl.ds(0, H)],
                    sem_ref.at[2 * NBUF_ + 0]
                ).start()
                pltpu.make_async_copy(
                    w2_scale_ref.at[pl.ds(next_expert_gj, 1), pl.ds(0, IB_TILE), pl.ds(0, 1), pl.ds(0, H)],
                    w2_s_bufs_ref.at[pl.ds(0, 1), pl.ds(0, IB_TILE), pl.ds(0, 1), pl.ds(0, H)],
                    sem_ref.at[3 * NBUF_ + 0]
                ).start()

    # ---- Accumulate weighted expert contribution to this token's row ----
    result = (down_acc * weight).reshape(1, 1, H)
    current = acc_scratch_ref[pl.ds(tok, 1), pl.ds(0, 1), pl.ds(0, H)]
    acc_scratch_ref[pl.ds(tok, 1), pl.ds(0, 1), pl.ds(0, H)] = current + result


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
    w1_bufs_ref,           # [NBUF, K_TILE, 2I]
    w1_s_bufs_ref,         # [NBUF, KB_TILE, 1, 2I] fp32
    w2_bufs_ref,           # [NBUF, I_TILE, H]
    w2_s_bufs_ref,         # [NBUF, IB_TILE, 1, H] fp32
    acc_scratch_ref,       # [N_TOKENS_, 1, H] fp32 (per-token accumulator)
    full_out_scratch_ref,  # [C_PAD, 1, H] bf16
    sem_ref,               # semaphores
    *,
    K, I, H, K_BLOCKS, QB, I_BLOCKS, IB, TOP_K_, NBUF_,
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

    # Shared kwargs for _expert_body
    body_kw = dict(
        lhs_scratch_ref=lhs_scratch_ref,
        w1_ref=w1_ref, w1_scale_ref=w1_scale_ref,
        w2_ref=w2_ref, w2_scale_ref=w2_scale_ref,
        w1_bufs_ref=w1_bufs_ref, w1_s_bufs_ref=w1_s_bufs_ref,
        w2_bufs_ref=w2_bufs_ref, w2_s_bufs_ref=w2_s_bufs_ref,
        acc_scratch_ref=acc_scratch_ref, sem_ref=sem_ref,
        K=K, I=I, H=H, K_BLOCKS=K_BLOCKS, QB=QB,
        I_BLOCKS=I_BLOCKS, IB=IB, NBUF_=NBUF_,
        K_TILE=K_TILE, NUM_K=NUM_K, QB_eff=QB_eff, KB_TILE=KB_TILE,
        I_TILE=I_TILE, NUM_I=NUM_I, IB_TILE=IB_TILE,
        DTYPE_LHS=DTYPE_LHS, DTYPE_OUT=DTYPE_OUT, DEQUANT_W1_AFTER_=DEQUANT_W1_AFTER_, DEQUANT_W2_AFTER_=DEQUANT_W2_AFTER_,
    )

    # ---- DMA ALL tokens from HBM -> VMEM (once) ----
    full_lhs_copy = pltpu.make_async_copy(
        lhs_ref.at[pl.ds(0, C_PAD), pl.ds(0, 1), pl.ds(0, K)],
        full_lhs_scratch_ref,
        sem_ref.at[4 * NBUF_]
    )
    full_lhs_copy.start()
    full_lhs_copy.wait()

    # Initialize output scratch and per-token accumulator to zeros
    full_out_scratch_ref[...] = jnp.zeros((C_PAD, 1, H), dtype=DTYPE_OUT)
    acc_scratch_ref[...] = jnp.zeros((N_TOKENS_, 1, H), dtype=jnp.float32)

    # ---- Flat slot loop (Python-unrolled, sorted by expert) ----
    if SKIP_ZERO_WEIGHT_:
        # EP mode: no dedup, no cross-expert prefetch
        for slot in range(N_SLOTS):
            gj = ids_ref[slot]
            gj = pl.multiple_of(gj, 1)
            tok = toks_ref[slot]
            weight = topk_weights_ref[slot]

            # Load this token's lhs
            token_row = full_lhs_scratch_ref[pl.ds(tok, 1), pl.ds(0, 1), pl.ds(0, K)]
            lhs_scratch_ref[...] = token_row.reshape(1, K)

            @pl.when(weight != 0.0)
            def _():
                _expert_body(gj, weight, tok,
                             is_new_expert=jnp.bool_(True),
                             is_first_slot=(slot == 0),
                             next_expert_gj=None, is_next_new=None,
                             **body_kw)
    else:
        # TP mode: dedup + cross-expert prefetch
        for slot in range(N_SLOTS):
            gj = ids_ref[slot]
            gj = pl.multiple_of(gj, 1)
            tok = toks_ref[slot]
            weight = topk_weights_ref[slot]

            # Dedup: is this a new expert compared to previous slot?
            if slot == 0:
                is_new_expert = jnp.bool_(True)
            else:
                prev_gj = ids_ref[slot - 1]
                is_new_expert = (gj != prev_gj)

            # Load this token's lhs (always, even for duplicate expert)
            token_row = full_lhs_scratch_ref[pl.ds(tok, 1), pl.ds(0, 1), pl.ds(0, K)]
            lhs_scratch_ref[...] = token_row.reshape(1, K)

            # Cross-expert look-ahead
            _next_gj = None
            _is_next_new = None
            if slot + 1 < N_SLOTS:
                _next_gj = pl.multiple_of(ids_ref[slot + 1], 1)
                _is_next_new = (_next_gj != gj)

            _expert_body(gj, weight, tok,
                         is_new_expert=is_new_expert,
                         is_first_slot=(slot == 0),
                         next_expert_gj=_next_gj, is_next_new=_is_next_new,
                         **body_kw)

    # ---- Final: copy each token's accumulated result to output ----
    for t in range(N_TOKENS_):
        row = acc_scratch_ref[pl.ds(t, 1), pl.ds(0, 1), pl.ds(0, H)]
        full_out_scratch_ref[pl.ds(t, 1), pl.ds(0, 1), pl.ds(0, H)] = row.astype(DTYPE_OUT)

    # ---- DMA full output back to HBM ----
    out_copy = pltpu.make_async_copy(
        full_out_scratch_ref,
        o_ref.at[pl.ds(0, C_PAD), pl.ds(0, 1), pl.ds(0, H)],
        sem_ref.at[4 * NBUF_ + 1]
    )
    out_copy.start()
    out_copy.wait()


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
    assert I // I_TILE <= 2, (
        f"Cross-expert w2 prefetch requires NUM_I <= 2, got {I // I_TILE} "
        f"(I={I}, I_TILE={I_TILE}). Increase MOE_CN_I_TILE or reduce I.")

    C_PAD = lhs.shape[0]
    grid_spec = pltpu.PrefetchScalarGridSpec(
        num_scalar_prefetch=3,            # ids, toks, weights
        in_specs=[any_spec] * 5,
        out_specs=any_spec,
        grid=(1,),
        scratch_shapes=[
            pltpu.VMEM((C_PAD, 1, K), lhs.dtype),                  # full_lhs_scratch
            pltpu.VMEM((M_PAD, K), lhs.dtype),                    # lhs_scratch
            pltpu.VMEM((NBUF, K_TILE, N1), w1.dtype),             # w1_bufs
            pltpu.VMEM((NBUF, KB_TILE, 1, N1), w1_scale.dtype),   # w1_s_bufs
            pltpu.VMEM((NBUF, I_TILE, H), w2.dtype),              # w2_bufs
            pltpu.VMEM((NBUF, IB_TILE, 1, H), w2_scale.dtype),    # w2_s_bufs
            pltpu.VMEM((n_tokens, 1, H), jnp.float32),            # acc_scratch (per-token, 3D)
            pltpu.VMEM((C_PAD, 1, H), jnp.bfloat16),              # full_out_scratch
            pltpu.SemaphoreType.DMA((4 * NBUF + 2,)),             # semaphores
        ])
    compiler_params = None if interpret else pltpu.CompilerParams(
        vmem_limit_bytes=int(pltpu.get_tpu_info().vmem_capacity_bytes * 0.9))

    return pl.pallas_call(
        functools.partial(_cn_w1w2_fused_token_kernel_fp8,
                          K=K, I=I, H=H,
                          K_BLOCKS=K_BLOCKS, QB=QB,
                          I_BLOCKS=I_BLOCKS, IB=IB,
                          TOP_K_=TOP_K, NBUF_=NBUF,
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
