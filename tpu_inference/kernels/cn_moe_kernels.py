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


def _cn_w1w2_fused_grid_kernel_fp8(
    # ---- Scalar prefetch (SMEM) — 3 arrays ----
    ids_ref,              # [TOPK_TOTAL] int32 — expert IDs
    token_offsets_ref,    # [TOPK_TOTAL] int32 — pre-computed t * M_PAD
    topk_weights_ref,     # [TOPK_TOTAL] fp32  — routing weights
    # ---- Inputs (HBM) ----
    lhs_ref,              # [C*M_PAD, K] bf16
    w1_ref,               # [G, K, 2*I] fp8
    w1_scale_ref,         # [G, K_BLOCKS, 1, 2*I] fp32
    w2_ref,               # [G, I, H] fp8
    w2_scale_ref,         # [G, I_BLOCKS, 1, H] fp32
    # ---- Output (HBM) ----
    o_ref,                # [TOPK_TOTAL * M_PAD, H] bf16
    # ---- Scratch (VMEM) ----
    lhs_scratch_ref,      # [M_PAD, K] bf16
    w1_bufs_ref,          # [NBUF, K_TILE, 2*I] fp8
    w1_s_bufs_ref,        # [NBUF, KB_TILE, 1, 2*I] fp32
    w2_bufs_ref,          # [NBUF, I_TILE, H] fp8
    w2_s_bufs_ref,        # [NBUF, IB_TILE, 1, H] fp32
    o_scratch_ref,        # [M_PAD, H] bf16
    sem_ref,              # DMA semaphores
    *,
    K, I, H, K_BLOCKS, QB, I_BLOCKS, IB, TOP_K_, NBUF_,
    DTYPE_LHS, DTYPE_OUT,
):
    K_TILE = w1_bufs_ref.shape[1]
    KB_TILE = w1_s_bufs_ref.shape[1]
    NUM_K = K // K_TILE
    QB_eff = min(K_TILE, QB)
    I_TILE = w2_bufs_ref.shape[1]
    IB_TILE = w2_s_bufs_ref.shape[1]
    NUM_I = I // I_TILE

    slot = pl.program_id(0)
    gj = ids_ref[slot]                        # SMEM — expert id
    t_offset = token_offsets_ref[slot]         # SMEM — pre-computed t * M_PAD
    t_offset = pl.multiple_of(t_offset, M_PAD)  # compiler hint

    # ---- DMA token's LHS row: HBM → VMEM scratch ----
    lhs_copy = pltpu.make_async_copy(
        lhs_ref.at[pl.ds(t_offset, M_PAD), pl.ds(0, K)],
        lhs_scratch_ref,
        sem_ref.at[4 * NBUF_]
    )
    lhs_copy.start()

    # ---- Phase 1: K-tiled gate+up matmul ----
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

    gate_up_acc = jnp.zeros((M_PAD, 2 * I), dtype=jnp.float32)

    lhs_copy.wait()

    for k in range(NUM_K):
        buf = k % NBUF_
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

        pltpu.make_async_copy(
            w1_ref.at[pl.ds(gj, 1), pl.ds(k * K_TILE, K_TILE), pl.ds(0, 2 * I)],
            w1_bufs_ref.at[pl.ds(buf, 1), pl.ds(0, K_TILE), pl.ds(0, 2 * I)],
            sem_ref.at[0 * NBUF_ + buf]
        ).wait()
        k_block_start = (k * K_TILE) // QB
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

    # ---- SwiGLU ----
    gate_up_bf16 = gate_up_acc.astype(DTYPE_OUT)
    gate = gate_up_bf16[:, :I].astype(jnp.float32)
    up = gate_up_bf16[:, I:].astype(jnp.float32)
    silu_gate = gate * jax.nn.sigmoid(gate)
    intermediate = (silu_gate * up).astype(DTYPE_LHS)

    # ---- Phase 2: I-tiled down matmul ----
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

    down_acc = jnp.zeros((M_PAD, H), dtype=jnp.float32)

    for m in range(NUM_I):
        buf = m % NBUF_
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


        # Start next tile DMA first (overlaps with wait + compute)
        w2_fp8 = w2_bufs_ref[buf]
        s2 = w2_s_bufs_ref[buf]
        w2_fp32 = w2_fp8.astype(jnp.float32).reshape(IB_TILE, IB, H)
        w2_dequant = (w2_fp32 * s2).reshape(I_TILE, H).astype(DTYPE_LHS)

        inter_tile = intermediate[:, m * I_TILE : (m + 1) * I_TILE]
        down_acc = down_acc + jnp.matmul(inter_tile, w2_dequant, preferred_element_type=jnp.float32)

    # ---- Output: weighted, per-slot ----
    slot_offset = pl.multiple_of(slot * M_PAD, M_PAD)
    o_scratch_ref[...] = (down_acc * topk_weights_ref[slot]).astype(DTYPE_OUT)
    out_copy = pltpu.make_async_copy(
        o_scratch_ref,
        o_ref.at[pl.ds(slot_offset, M_PAD), pl.ds(0, H)],
        sem_ref.at[4 * NBUF_ + 1]
    )
    out_copy.start()
    out_copy.wait()


def cn_gemv_w1w2_fused_mb_fp8(lhs, w1, w1_scale, w2, w2_scale,
                              active_ids, topk_weights, *,
                              n_tokens, interpret=False):
    """Fused gate+up+SwiGLU+down for the C=N MoE block, multi-buffered at
    depth ``_FUSE_NBUF``. lhs [N*M_PAD, K]; active_ids/topk_weights
    [N*TOP_K] flattened. Returns bf16[TOPK_TOTAL*M_PAD, H] with each
    slot's weighted output at row slot*M_PAD."""
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

    # Pre-compute token offsets (Option B)
    token_offsets = (jnp.arange(TOPK_TOTAL, dtype=jnp.int32) // TOP_K) * M_PAD

    any_spec = pl.BlockSpec(memory_space=pl.ANY)
    K_TILE = min(256, K)
    assert K % K_TILE == 0, f"K={K} not divisible by K_TILE={K_TILE}"
    QB_eff = min(K_TILE, QB)
    KB_TILE = K_TILE // QB_eff
    I_TILE = min(1792, I)
    assert I % I_TILE == 0, f"I={I} not divisible by I_TILE={I_TILE}"
    IB_TILE = I_BLOCKS // (I // I_TILE)

    grid_spec = pltpu.PrefetchScalarGridSpec(
        num_scalar_prefetch=3,          # ids, token_offsets, weights → SMEM
        in_specs=[any_spec] * 5,        # lhs, w1, w1_scale, w2, w2_scale
        out_specs=any_spec,
        grid=(TOPK_TOTAL,),
        scratch_shapes=[
            pltpu.VMEM((M_PAD, K), lhs.dtype),                    # lhs_scratch
            pltpu.VMEM((NBUF, K_TILE, N1), w1.dtype),             # w1_bufs
            pltpu.VMEM((NBUF, KB_TILE, 1, N1), w1_scale.dtype),   # w1_s_bufs
            pltpu.VMEM((NBUF, I_TILE, H), w2.dtype),              # w2_bufs
            pltpu.VMEM((NBUF, IB_TILE, 1, H), w2_scale.dtype),    # w2_s_bufs
            pltpu.VMEM((M_PAD, H), jnp.bfloat16),                 # o_scratch
            pltpu.SemaphoreType.DMA((4 * NBUF + 2,)),             # semaphores
        ])
    compiler_params = None if interpret else pltpu.CompilerParams(
        vmem_limit_bytes=int(pltpu.get_tpu_info().vmem_capacity_bytes * 0.9))

    return pl.pallas_call(
        functools.partial(_cn_w1w2_fused_grid_kernel_fp8,
                          K=K, I=I, H=H,
                          K_BLOCKS=K_BLOCKS, QB=QB,
                          I_BLOCKS=I_BLOCKS, IB=IB,
                          TOP_K_=TOP_K, NBUF_=NBUF,
                          DTYPE_LHS=lhs.dtype, DTYPE_OUT=jnp.bfloat16),
        out_shape=jax.ShapeDtypeStruct((TOPK_TOTAL * M_PAD, H), jnp.bfloat16),
        grid_spec=grid_spec, compiler_params=compiler_params,
        interpret=interpret, name="cn_gemv_w1w2_fused_grid_fp8",
    )(active_ids, token_offsets, topk_weights,  # scalar prefetch (3)
      lhs, w1, w1_scale, w2, w2_scale)          # inputs (5)


def cn_moe_full(hidden_state, w1, w1_scale, w2, w2_scale,
                active_ids, topk_weights, *, interpret=False):
    """hidden_state [C, K]; active_ids/topk_weights [C, TOP_K].
    Returns [C, H] — token t's MoE output at row t.
    """
    C = hidden_state.shape[0]
    K = hidden_state.shape[1]
    H = w2.shape[2]
    lhs = jnp.zeros((C * M_PAD, K), dtype=hidden_state.dtype)
    for t in range(C):
        lhs = lhs.at[t * M_PAD].set(hidden_state[t])
    ids_flat = active_ids.reshape(C * TOP_K)
    weights_flat = topk_weights.reshape(C * TOP_K)

    fused_out = cn_gemv_w1w2_fused_mb_fp8(
        lhs, w1, w1_scale, w2, w2_scale, ids_flat, weights_flat,
        n_tokens=C, interpret=interpret)

    # Sum across TOP_K experts per token (weighting applied in kernel)
    slot_outputs = fused_out[::M_PAD][:C * TOP_K]
    token_outputs = slot_outputs.reshape(C, TOP_K, H).sum(axis=1)
    return token_outputs.astype(jnp.bfloat16)
