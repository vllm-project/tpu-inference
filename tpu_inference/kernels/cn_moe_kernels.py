# Copyright 2026 Google LLC
"""Generalized C=N MoE gemv kernels (N = num real tokens, e.g. 3 or 4).
"""
from __future__ import annotations
import functools
import os as _os

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

M_PAD = 8
TOP_K = 10

_GEMV_NBUF = max(2, int(_os.getenv("MOE_GEMV_NBUF", "2")))
_FUSE_NBUF = max(2, int(_os.getenv("MOE_FUSE_NBUF", str(_GEMV_NBUF))))


def _expert_body(
    gj, weight, lhs_scratch_ref,
    w1_ref, w1_scale_ref, w2_ref, w2_scale_ref,
    w1_bufs_ref, w1_s_bufs_ref, w2_bufs_ref, w2_s_bufs_ref,
    acc_scratch_ref, sem_ref,
    *, K, I, H, K_BLOCKS, QB, I_BLOCKS, IB, NBUF_,
    K_TILE, NUM_K, QB_eff, KB_TILE, I_TILE, NUM_I, IB_TILE,
    DTYPE_LHS, DTYPE_OUT, prefetch_first_w1,
):
    """Core expert computation: gate+up matmul, SwiGLU, down matmul, accumulate.

    If prefetch_first_w1 is True, this function starts the first w1 tile DMA.
    If False, it assumes the first w1 tile was already prefetched by the caller
    (e.g. from the previous expert's cross-expert pipelining).
    """
    if prefetch_first_w1:
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

        # Start next tile DMA first (overlaps with wait + compute)
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

        w1_fp8 = w1_bufs_ref[buf]
        s1 = w1_s_bufs_ref[buf]
        w1_fp32 = w1_fp8.astype(jnp.float32).reshape(KB_TILE, QB_eff, 2 * I)
        w1_dequant = (w1_fp32 * s1).reshape(K_TILE, 2 * I).astype(DTYPE_LHS)

        lhs_tile = lhs_scratch_ref[pl.ds(0, M_PAD), pl.ds(k * K_TILE, K_TILE)]
        gate_up_acc = gate_up_acc + jnp.matmul(lhs_tile, w1_dequant, preferred_element_type=jnp.float32)

    # ---- Prefetch first w2 tile early (overlaps with SwiGLU compute) ----
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

    # ---- SwiGLU (w2 DMA runs in parallel) ----
    gate_up_bf16 = gate_up_acc.astype(DTYPE_OUT)
    gate = gate_up_bf16[:, :I].astype(jnp.float32)
    up = gate_up_bf16[:, I:].astype(jnp.float32)
    silu_gate = gate * jax.nn.sigmoid(gate)
    intermediate = (silu_gate * up).astype(DTYPE_LHS)

    # ---- Phase 2: I-tiled down matmul (tile 0 already in flight) ----
    down_acc = jnp.zeros((M_PAD, H), dtype=jnp.float32)

    for m in range(NUM_I):
        buf = m % NBUF_

        # Start next tile DMA first
        nxt_m = m + 1
        if nxt_m < NUM_I:
            nxt_buf = nxt_m % NBUF_
            pltpu.make_async_copy(
                w2_ref.at[pl.ds(gj, 1), pl.ds(nxt_m * I_TILE, I_TILE), pl.ds(0, H)],
                w2_bufs_ref.at[pl.ds(nxt_buf, 1), pl.ds(0, I_TILE), pl.ds(0, H)],
                sem_ref.at[2 * NBUF_ + nxt_buf]
            ).start()
            pltpu.make_async_copy(
                w2_scale_ref.at[pl.ds(gj, 1), pl.ds(nxt_m * IB_TILE, IB_TILE), pl.ds(0, 1), pl.ds(0, H)],
                w2_s_bufs_ref.at[pl.ds(nxt_buf, 1), pl.ds(0, IB_TILE), pl.ds(0, 1), pl.ds(0, H)],
                sem_ref.at[3 * NBUF_ + nxt_buf]
            ).start()

        # Wait for current tile
        pltpu.make_async_copy(
            w2_ref.at[pl.ds(gj, 1), pl.ds(m * I_TILE, I_TILE), pl.ds(0, H)],
            w2_bufs_ref.at[pl.ds(buf, 1), pl.ds(0, I_TILE), pl.ds(0, H)],
            sem_ref.at[2 * NBUF_ + buf]
        ).wait()
        pltpu.make_async_copy(
            w2_scale_ref.at[pl.ds(gj, 1), pl.ds(m * IB_TILE, IB_TILE), pl.ds(0, 1), pl.ds(0, H)],
            w2_s_bufs_ref.at[pl.ds(buf, 1), pl.ds(0, IB_TILE), pl.ds(0, 1), pl.ds(0, H)],
            sem_ref.at[3 * NBUF_ + buf]
        ).wait()

        w2_fp8 = w2_bufs_ref[buf]
        s2 = w2_s_bufs_ref[buf]
        w2_fp32 = w2_fp8.astype(jnp.float32).reshape(IB_TILE, IB, H)
        w2_dequant = (w2_fp32 * s2).reshape(I_TILE, H).astype(DTYPE_LHS)

        inter_tile = intermediate[:, m * I_TILE : (m + 1) * I_TILE]
        down_acc = down_acc + jnp.matmul(inter_tile, w2_dequant, preferred_element_type=jnp.float32)

    # Accumulate weighted expert contribution
    acc_scratch_ref[...] = acc_scratch_ref[...] + down_acc * weight


def _cn_w1w2_fused_token_kernel_fp8(
    # ---- Scalar prefetch (SMEM) — 2 arrays ----
    ids_ref,              # [C*TOP_K] int32 — expert IDs
    topk_weights_ref,     # [C*TOP_K] fp32  — routing weights
    # ---- Inputs (HBM) ----
    lhs_ref,              # [C*M_PAD, K] bf16
    w1_ref,               # [G, K, 2*I] fp8
    w1_scale_ref,         # [G, K_BLOCKS, 1, 2*I] fp32
    w2_ref,               # [G, I, H] fp8
    w2_scale_ref,         # [G, I_BLOCKS, 1, H] fp32
    # ---- Output (HBM) ----
    o_ref,                # [C*M_PAD, H] bf16
    # ---- Scratch (VMEM) ----
    lhs_scratch_ref,      # [M_PAD, K] bf16
    w1_bufs_ref,          # [NBUF, K_TILE, 2*I] fp8
    w1_s_bufs_ref,        # [NBUF, KB_TILE, 1, 2*I] fp32
    w2_bufs_ref,          # [NBUF, I_TILE, H] fp8
    w2_s_bufs_ref,        # [NBUF, IB_TILE, 1, H] fp32
    acc_scratch_ref,      # [M_PAD, H] fp32 — cross-expert accumulator
    o_scratch_ref,        # [M_PAD, H] bf16 — output staging for DMA
    sem_ref,              # DMA semaphores
    *,
    K, I, H, K_BLOCKS, QB, I_BLOCKS, IB, TOP_K_, NBUF_,
    SKIP_ZERO_WEIGHT_, DTYPE_LHS, DTYPE_OUT,
):
    token = pl.program_id(0)
    token_offset = pl.multiple_of(token * M_PAD, M_PAD)

    K_TILE = w1_bufs_ref.shape[1]
    NUM_K = K // K_TILE
    QB_eff = min(K_TILE, QB)
    KB_TILE = K_TILE // QB_eff
    I_TILE = w2_bufs_ref.shape[1]
    NUM_I = I // I_TILE
    IB_TILE = I_BLOCKS // NUM_I

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
        DTYPE_LHS=DTYPE_LHS, DTYPE_OUT=DTYPE_OUT,
    )

    # ---- DMA token LHS row: HBM -> VMEM (once for all experts) ----
    lhs_copy = pltpu.make_async_copy(
        lhs_ref.at[pl.ds(token_offset, M_PAD), pl.ds(0, K)],
        lhs_scratch_ref,
        sem_ref.at[4 * NBUF_]
    )
    lhs_copy.start()
    lhs_copy.wait()

    # Accumulate across all TOP_K experts in fp32
    acc_scratch_ref[...] = jnp.zeros((M_PAD, H), dtype=jnp.float32)

    if SKIP_ZERO_WEIGHT_:
        # ---- EP mode: use pl.when to skip zero-weight slots ----
        # No cross-expert pipelining (semaphore state cannot be conditional).
        for e in range(TOP_K_):
            idx = token * TOP_K_ + e
            gj = ids_ref[idx]
            gj = pl.multiple_of(gj, 1)
            weight = topk_weights_ref[idx]

            @pl.when(weight != 0.0)
            def _():
                _expert_body(gj, weight, prefetch_first_w1=True, **body_kw)
    else:
        # ---- Non-EP mode: cross-expert w1 pipelining ----
        # All weights are nonzero, so we can safely pipeline the first w1
        # tile of expert e+1 at the end of expert e.
        for e in range(TOP_K_):
            idx = token * TOP_K_ + e
            gj = ids_ref[idx]
            gj = pl.multiple_of(gj, 1)
            weight = topk_weights_ref[idx]

            # Expert 0 prefetches its own first w1 tile; subsequent experts
            # have their first tile already in flight from the previous expert.
            _expert_body(gj, weight, prefetch_first_w1=(e == 0), **body_kw)

            # Start NEXT expert's first w1 tile (overlaps with loop overhead
            # and next iteration's LHS setup).
            if e + 1 < TOP_K_:
                next_idx = token * TOP_K_ + (e + 1)
                next_gj = ids_ref[next_idx]
                next_gj = pl.multiple_of(next_gj, 1)
                pltpu.make_async_copy(
                    w1_ref.at[pl.ds(next_gj, 1), pl.ds(0, K_TILE), pl.ds(0, 2 * I)],
                    w1_bufs_ref.at[pl.ds(0, 1), pl.ds(0, K_TILE), pl.ds(0, 2 * I)],
                    sem_ref.at[0 * NBUF_ + 0]
                ).start()
                pltpu.make_async_copy(
                    w1_scale_ref.at[pl.ds(next_gj, 1), pl.ds(0, KB_TILE), pl.ds(0, 1), pl.ds(0, 2 * I)],
                    w1_s_bufs_ref.at[pl.ds(0, 1), pl.ds(0, KB_TILE), pl.ds(0, 1), pl.ds(0, 2 * I)],
                    sem_ref.at[1 * NBUF_ + 0]
                ).start()

    # ---- Write accumulated output for this token ----
    o_scratch_ref[...] = acc_scratch_ref[...].astype(DTYPE_OUT)
    out_copy = pltpu.make_async_copy(
        o_scratch_ref,
        o_ref.at[pl.ds(token_offset, M_PAD), pl.ds(0, H)],
        sem_ref.at[4 * NBUF_ + 1]
    )
    out_copy.start()
    out_copy.wait()


def cn_gemv_w1w2_fused_mb_fp8(lhs, w1, w1_scale, w2, w2_scale,
                              active_ids, topk_weights, *,
                              n_tokens, use_ep=False, interpret=False):
    """Fused gate+up+SwiGLU+down for the C=N MoE block, multi-buffered at
    depth ``_FUSE_NBUF``. lhs [N*M_PAD, K]; active_ids/topk_weights
    [N*TOP_K] flattened. Returns bf16[N*M_PAD, H] with token t's
    accumulated MoE output at row t*M_PAD."""
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
    topk_weights = topk_weights.astype(jnp.float32)

    any_spec = pl.BlockSpec(memory_space=pl.ANY)
    K_TILE = min(256, K)
    assert K % K_TILE == 0, f"K={K} not divisible by K_TILE={K_TILE}"
    QB_eff = min(K_TILE, QB)
    KB_TILE = K_TILE // QB_eff
    I_TILE = min(1792, I)
    assert I % I_TILE == 0, f"I={I} not divisible by I_TILE={I_TILE}"
    IB_TILE = I_BLOCKS // (I // I_TILE)

    grid_spec = pltpu.PrefetchScalarGridSpec(
        num_scalar_prefetch=2,          # ids, weights -> SMEM
        in_specs=[any_spec] * 5,        # lhs, w1, w1_scale, w2, w2_scale
        out_specs=any_spec,
        grid=(n_tokens,),               # one iteration per TOKEN
        scratch_shapes=[
            pltpu.VMEM((M_PAD, K), lhs.dtype),                    # lhs_scratch
            pltpu.VMEM((NBUF, K_TILE, N1), w1.dtype),             # w1_bufs
            pltpu.VMEM((NBUF, KB_TILE, 1, N1), w1_scale.dtype),   # w1_s_bufs
            pltpu.VMEM((NBUF, I_TILE, H), w2.dtype),              # w2_bufs
            pltpu.VMEM((NBUF, IB_TILE, 1, H), w2_scale.dtype),    # w2_s_bufs
            pltpu.VMEM((M_PAD, H), jnp.float32),                  # acc_scratch
            pltpu.VMEM((M_PAD, H), jnp.bfloat16),                 # o_scratch
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
                          SKIP_ZERO_WEIGHT_=use_ep,
                          DTYPE_LHS=lhs.dtype, DTYPE_OUT=jnp.bfloat16),
        out_shape=jax.ShapeDtypeStruct((n_tokens * M_PAD, H), jnp.bfloat16),
        grid_spec=grid_spec, compiler_params=compiler_params,
        interpret=interpret, name="cn_gemv_w1w2_fused_token_fp8",
    )(active_ids, topk_weights,                   # scalar prefetch (2)
      lhs, w1, w1_scale, w2, w2_scale)            # inputs (5)


def cn_moe_full(hidden_state, w1, w1_scale, w2, w2_scale,
                active_ids, topk_weights, *, use_ep=False, interpret=False):
    """hidden_state [C, K]; active_ids/topk_weights [C, TOP_K].
    Returns [C, K] — token t's MoE output at row t (bit-identical to running
    each token's experts independently).

    Expert weights are gathered by direct HBM indexing on active_ids with no
    bounds clamping, so callers must pass a valid expert id for every slot,
    using id 0 + weight 0.0 for any padded/inactive (token, expert) slot.
    """
    C = hidden_state.shape[0]
    K = hidden_state.shape[1]
    lhs = jnp.zeros((C * M_PAD, K), dtype=hidden_state.dtype)
    for t in range(C):
        lhs = lhs.at[t * M_PAD].set(hidden_state[t])
    ids_flat = active_ids.reshape(C * TOP_K)
    weights_flat = topk_weights.reshape(C * TOP_K)

    fused_out = cn_gemv_w1w2_fused_mb_fp8(
        lhs, w1, w1_scale, w2, w2_scale, ids_flat, weights_flat,
        n_tokens=C, use_ep=use_ep, interpret=interpret)

    # Token t's accumulated output sits at row t*M_PAD.
    return fused_out[::M_PAD][:C].astype(jnp.bfloat16)
