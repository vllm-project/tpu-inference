"""
Implementation that uses emit_pipeline instead of hand-rolled version.
"""
import functools
import os

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

TOP_K = 10
M_PAD = 1
_CN_NBUF = max(2, int(os.getenv("MOE_CN_NBUF", "4")))

_FP8 = jnp.float8_e4m3fn
_FP8_MAX = float(jnp.finfo(_FP8).max)
# 0 falls back to bf16xbf16: slower, but ~10x more accurate.
_USE_FP8_LHS = os.getenv("MOE_CN_FP8_LHS", "1") == "1"


def _blockwise_matmul(lhs, w, scales, n_blocks, block, out_cols):
    acc = jnp.zeros((1, out_cols), dtype=jnp.float32)
    for b in range(n_blocks):
        s = b * block
        cur = lhs[:, s:s + block]
        if _USE_FP8_LHS:
            cur32 = cur.astype(jnp.float32)
            lsc = jnp.max(jnp.abs(cur32), axis=1, keepdims=True) / _FP8_MAX
            cur_q = (cur32 *
                     jnp.where(lsc == 0, 0.0, 1.0 / lsc)).astype(_FP8)
            blk = jnp.matmul(cur_q,
                             w[s:s + block, :],
                             preferred_element_type=jnp.float32) * lsc
        else:
            blk = jnp.matmul(cur,
                             w[s:s + block, :],
                             preferred_element_type=jnp.float32)
        acc = acc + blk * scales[b].reshape(1, out_cols)
    return acc


def _forward_fill_ids(sorted_ids, sorted_weights):
    valid = (sorted_weights.astype(jnp.float32) != 0.0)

    def _fill_op(a, b):
        a_id, a_v = a
        b_id, b_v = b
        out_id = jnp.where(b_v, b_id, a_id)
        out_v = a_v | b_v
        return (out_id, out_v)

    filled_ids, _ = jax.lax.associative_scan(_fill_op, (sorted_ids, valid))
    return filled_ids


def _inner_kernel(
    # Leading expert dim kept at 1 (not squeezed) for Mosaic layout.
    w1_tile_ref,  # [1, K, 2I]
    w1_scale_tile_ref,  # [1, K_BLOCKS, 1, 2I]
    w2_tile_ref,  # [1, I, H]
    w2_scale_tile_ref,  # [1, I_BLOCKS, 1, H]
    *,
    ids_ref,
    toks_ref,
    weights_ref,
    lhs_full_ref,  # [C_PAD, 1, K] -- 3D so `tok` indexes a leading dim
    acc_ref,  # [N_TOKENS_, 1, H] fp32 -- likewise
    I,
    DTYPE_LHS,
):
    slot = pl.program_id(0)
    tok = toks_ref[slot]
    weight = weights_ref[slot]

    @pl.when(weight.astype(jnp.float32) != 0.0)
    def _():
        K_lhs = lhs_full_ref.shape[2]
        lhs_row = lhs_full_ref[pl.ds(tok, 1),
                               pl.ds(0, 1),
                               pl.ds(0, K_lhs)].reshape(1, K_lhs)

        _, K, TWO_I = w1_tile_ref.shape
        K_BLOCKS = w1_scale_tile_ref.shape[1]
        QB = K // K_BLOCKS
        # MXU has no bf16xfp8 mode; one side has to move.
        w1 = w1_tile_ref[...].reshape(K, TWO_I)
        if not _USE_FP8_LHS:
            w1 = w1.astype(DTYPE_LHS)
        s1 = w1_scale_tile_ref[...].reshape(K_BLOCKS, 1, TWO_I)
        gate_up = _blockwise_matmul(lhs_row, w1, s1, K_BLOCKS, QB, TWO_I)

        gate = gate_up[:, :I]
        up = gate_up[:, I:]
        act = gate * jax.nn.sigmoid(gate) * up
        intermediate = act if _USE_FP8_LHS else act.astype(DTYPE_LHS)

        _, I_, H = w2_tile_ref.shape
        I_BLOCKS = w2_scale_tile_ref.shape[1]
        IB = I_ // I_BLOCKS
        w2 = w2_tile_ref[...].reshape(I_, H)
        if not _USE_FP8_LHS:
            w2 = w2.astype(DTYPE_LHS)
        s2 = w2_scale_tile_ref[...].reshape(I_BLOCKS, 1, H)
        down = _blockwise_matmul(intermediate, w2, s2, I_BLOCKS, IB, H)

        H_acc = acc_ref.shape[2]
        current = acc_ref[pl.ds(tok, 1), pl.ds(0, 1), pl.ds(0, H_acc)]
        acc_ref[pl.ds(tok, 1), pl.ds(0, 1),
                pl.ds(0, H_acc)] = current + (down * weight).reshape(
                    1, 1, H_acc)


def _outer_kernel(
    ids_ref,
    toks_ref,
    weights_ref,
    lhs_ref,
    w1_ref,
    w1_scale_ref,
    w2_ref,
    w2_scale_ref,
    o_ref,
    full_lhs_2d_ref,
    lhs_full_ref,
    acc_ref,
    out_vmem_ref,
    sem_ref,
    *,
    K,
    I,
    H,
    K_BLOCKS,
    I_BLOCKS,
    N_TOKENS_,
    N_SLOTS,
    NBUF,
    DTYPE_LHS,
    DTYPE_OUT,
):
    C_PAD = full_lhs_2d_ref.shape[0]

    # ---- Stage LHS into a 2D pad, then copy 2D -> 3D on chip ----
    full_lhs_copy = pltpu.make_async_copy(lhs_ref.at[pl.ds(0, C_PAD)],
                                          full_lhs_2d_ref, sem_ref.at[0])
    full_lhs_copy.start()
    full_lhs_copy.wait()

    for t in range(C_PAD):
        row = full_lhs_2d_ref[pl.ds(t, 1), pl.ds(0, K)]
        lhs_full_ref[pl.ds(t, 1), pl.ds(0, 1), pl.ds(0, K)] = \
            row.reshape(1, 1, K)

    acc_ref[...] = jnp.zeros((N_TOKENS_, 1, H), dtype=jnp.float32)

    def w1_index_map(slot):
        return (ids_ref[slot], 0, 0)

    def w1_scale_index_map(slot):
        return (ids_ref[slot], 0, 0, 0)

    def w2_index_map(slot):
        return (ids_ref[slot], 0, 0)

    def w2_scale_index_map(slot):
        return (ids_ref[slot], 0, 0, 0)

    pipeline_fn = pltpu.emit_pipeline(
        functools.partial(
            _inner_kernel,
            ids_ref=ids_ref,
            toks_ref=toks_ref,
            weights_ref=weights_ref,
            lhs_full_ref=lhs_full_ref,
            acc_ref=acc_ref,
            I=I,
            DTYPE_LHS=DTYPE_LHS,
        ),
        grid=(N_SLOTS, ),
        in_specs=(
            pl.BlockSpec((1, K, 2 * I),
                        w1_index_map,
                        pipeline_mode=pl.Buffered(buffer_count=NBUF)),
            pl.BlockSpec((1, K_BLOCKS, 1, 2 * I),
                        w1_scale_index_map,
                        pipeline_mode=pl.Buffered(buffer_count=NBUF)),
            pl.BlockSpec((1, I, H),
                        w2_index_map,
                        pipeline_mode=pl.Buffered(buffer_count=NBUF)),
            pl.BlockSpec((1, I_BLOCKS, 1, H),
                        w2_scale_index_map,
                        pipeline_mode=pl.Buffered(buffer_count=NBUF)),
        ),
        out_specs=(),
    )
    pipeline_fn(w1_ref, w1_scale_ref, w2_ref, w2_scale_ref)

    for t in range(N_TOKENS_):
        r = acc_ref[pl.ds(t, 1), pl.ds(0, 1), pl.ds(0, H)]
        out_vmem_ref[pl.ds(t, 1),
                     pl.ds(0, H)] = r.reshape(1, H).astype(DTYPE_OUT)
    out_copy = pltpu.make_async_copy(out_vmem_ref, o_ref.at[pl.ds(0, N_TOKENS_)],
                                     sem_ref.at[0])
    out_copy.start()
    out_copy.wait()


def cn_gemv_w1w2_fused_pipelined(
    lhs,
    w1,
    w1_scale,
    w2,
    w2_scale,
    sorted_ids,
    sorted_toks,
    sorted_weights,
    *,
    n_tokens,
    use_ep=False,
    interpret=False,
):
    """Pipelined-weight-fetch variant of cn_gemv_w1w2_fused_mb_fp8. Only
    supports NUM_K = NUM_I = 1 (the real production shape)."""
    G, K, N1 = w1.shape
    I = N1 // 2
    _, I2, H = w2.shape
    assert I == I2
    K_BLOCKS = w1_scale.shape[1]
    I_BLOCKS = w2_scale.shape[1]

    TOPK_TOTAL = n_tokens * TOP_K
    N_SLOTS = TOPK_TOTAL
    NBUF = min(_CN_NBUF, G)

    effective_ids = (_forward_fill_ids(sorted_ids, sorted_weights)
                     if use_ep else sorted_ids)

    C_PAD = lhs.shape[0]
    any_spec = pl.BlockSpec(memory_space=pl.ANY)
    grid_spec = pltpu.PrefetchScalarGridSpec(
        num_scalar_prefetch=3,  # ids, toks, weights
        in_specs=[any_spec] * 5,
        out_specs=any_spec,
        grid=(1, ),
        scratch_shapes=[
            pltpu.VMEM((C_PAD, K), lhs.dtype),  # full_lhs_2d (DMA pad)
            pltpu.VMEM((C_PAD, 1, K), lhs.dtype),  # full_lhs (3D)
            pltpu.VMEM((n_tokens, 1, H), jnp.float32),  # acc (3D)
            pltpu.VMEM((n_tokens, H), jnp.bfloat16),  # out_2d (DMA pad)
            pltpu.SemaphoreType.DMA((1, )),
        ],
    )

    compiler_params = None if interpret else pltpu.CompilerParams(
        vmem_limit_bytes=int(pltpu.get_tpu_info().vmem_capacity_bytes * 0.9))

    return pl.pallas_call(
        functools.partial(
            _outer_kernel,
            K=K,
            I=I,
            H=H,
            K_BLOCKS=K_BLOCKS,
            I_BLOCKS=I_BLOCKS,
            N_TOKENS_=n_tokens,
            N_SLOTS=N_SLOTS,
            NBUF=NBUF,
            DTYPE_LHS=lhs.dtype,
            DTYPE_OUT=jnp.bfloat16,
        ),
        out_shape=jax.ShapeDtypeStruct((n_tokens, H), jnp.bfloat16),
        grid_spec=grid_spec,
        compiler_params=compiler_params,
        interpret=interpret,
        name="cn_gemv_w1w2_fused_pipelined",
    )(effective_ids, sorted_toks, sorted_weights, lhs, w1, w1_scale, w2,
     w2_scale)


def cn_moe_full_pipelined(hidden_state,
                          w1,
                          w1_scale,
                          w2,
                          w2_scale,
                          active_ids,
                          topk_weights,
                          *,
                          use_ep=False,
                          interpret=False):
    C = hidden_state.shape[0]
    flat_ids = active_ids.reshape(C * TOP_K)
    flat_weights = topk_weights.reshape(C * TOP_K)
    # Skip the argsort permutation — slots stay in natural token-major order.
    # Clamp invalid (zero-weight) expert IDs to 0 so weight DMA stays valid;
    # the @pl.when(weight != 0) guard in _inner_kernel skips the computation.
    sorted_ids = jnp.where(flat_weights.astype(jnp.float32) != 0.0, flat_ids, 0)
    sorted_toks = jnp.arange(C * TOP_K, dtype=jnp.int32) // TOP_K
    sorted_weights = flat_weights

    fused_out = cn_gemv_w1w2_fused_pipelined(hidden_state,
                                             w1,
                                             w1_scale,
                                             w2,
                                             w2_scale,
                                             sorted_ids,
                                             sorted_toks,
                                             sorted_weights,
                                             n_tokens=C,
                                             use_ep=use_ep,
                                             interpret=interpret)
    return fused_out[:C, :].astype(jnp.bfloat16)
