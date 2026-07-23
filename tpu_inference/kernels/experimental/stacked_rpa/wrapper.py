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
"""Wrapper for RPA kernel to match expected interface."""

import dataclasses

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from tpu_inference.kernels.experimental.stacked_rpa import (configs, kernel,
                                                            schedule,
                                                            tuned_block_sizes,
                                                            utils)


def _require_seq_along_lane(kv_layout: configs.KVLayout) -> None:
    """stacked_rpa supports only the 4D SEQ_ALONG_LANE kv-cache layout."""
    if kv_layout != configs.KVLayout.SEQ_ALONG_LANE:
        raise ValueError(
            "stacked_rpa only supports the 4D SEQ_ALONG_LANE kv-cache layout, "
            f"but got kv_layout={kv_layout!r}.")


def prepare_queries(
    q: jax.Array,
    num_kv_heads: int,
    q_dtype: jnp.dtype,
) -> jax.Array:
    total_q_tokens, actual_num_q_heads, actual_head_dim = q.shape
    num_q_heads_per_kv_head = actual_num_q_heads // num_kv_heads
    q_packing = utils.get_dtype_packing(q_dtype)
    aligned_num_q_heads_per_kv_head = utils.align_to(num_q_heads_per_kv_head,
                                                     q_packing)
    num_lanes = pltpu.get_tpu_info().num_lanes
    aligned_q_head_dim = utils.align_to(actual_head_dim, num_lanes)

    # queries: (T, H, D) -> (T, H_kv, G, D)
    return (jnp.pad(
        q.reshape(
            total_q_tokens,
            num_kv_heads,
            num_q_heads_per_kv_head,
            actual_head_dim,
        ),
        (
            (0, 0),
            (0, 0),
            (0, aligned_num_q_heads_per_kv_head - num_q_heads_per_kv_head),
            (0, aligned_q_head_dim - actual_head_dim),
        ),
        constant_values=0,
    ).reshape(
        total_q_tokens,
        num_kv_heads,
        aligned_num_q_heads_per_kv_head // q_packing,
        q_packing,
        aligned_q_head_dim,
    ).swapaxes(0, 1))


def prepare_inputs(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    q_dtype: jnp.dtype,
    kv_dtype: jnp.dtype,
    kv_layout: configs.KVLayout = configs.KVLayout.SEQ_ALONG_LANE,
    prepacked_new_kv_hbm: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array]:
    _require_seq_along_lane(kv_layout)

    total_q_tokens, actual_num_q_heads, actual_head_dim = q.shape
    _, actual_num_kv_heads, _ = k.shape
    kv_packing = utils.get_dtype_packing(kv_dtype)

    num_lanes = pltpu.get_tpu_info().num_lanes
    num_sublanes = pltpu.get_tpu_info().num_sublanes
    aligned_kv_head_dim = utils.align_to(actual_head_dim,
                                         num_sublanes * kv_packing)

    o_hbm_alias_q_hbm = prepare_queries(q, actual_num_kv_heads, q_dtype)

    actual_num_kv_heads_x2 = actual_num_kv_heads * 2

    if prepacked_new_kv_hbm is not None:
        new_kv_hbm = prepacked_new_kv_hbm
        expected_shape = (
            actual_num_kv_heads_x2,
            aligned_kv_head_dim,
            utils.align_to(total_q_tokens, num_lanes),
        )
        if new_kv_hbm.shape != expected_shape:
            raise ValueError(
                f"prepacked_new_kv_hbm has shape {new_kv_hbm.shape}, "
                f"expected {expected_shape}")
    else:
        new_kv_hbm = prepare_seq_along_lane_new_kv_hbm(k, v, kv_dtype=kv_dtype)
    return o_hbm_alias_q_hbm, new_kv_hbm


def prepare_seq_along_lane_new_kv_hbm(
    k: jax.Array,
    v: jax.Array,
    *,
    kv_dtype: jnp.dtype | None = None,
) -> jax.Array:
    """Pack post-RoPE K/V directly into batched-RPA SEQ_ALONG_LANE layout.

    Inputs are the normal projection layout ``[tokens, kv_heads, head_dim]``.
    Output is ``[2 * kv_heads, hd_aligned, padded_tokens]`` (stacked keeps
    head_dim contiguous, unlike the batched 4D packed layout), with sequence on
    the lane/minor axis.
    """
    total_q_tokens, actual_num_kv_heads, actual_head_dim = k.shape
    if v.shape != k.shape:
        raise ValueError(f"k/v shapes differ: {k.shape=} {v.shape=}")
    if kv_dtype is None:
        kv_dtype = k.dtype
    kv_packing = utils.get_dtype_packing(kv_dtype)
    num_lanes = pltpu.get_tpu_info().num_lanes
    num_sublanes = pltpu.get_tpu_info().num_sublanes
    aligned_kv_head_dim = utils.align_to(actual_head_dim,
                                         num_sublanes * kv_packing)
    padded_total_tokens = utils.align_to(total_q_tokens, num_lanes)
    actual_num_kv_heads_x2 = actual_num_kv_heads * 2
    return (jnp.pad(
        jnp.concatenate([k, v], axis=-1).reshape(total_q_tokens,
                                                 actual_num_kv_heads_x2,
                                                 actual_head_dim),
        (
            (0, padded_total_tokens - total_q_tokens),
            (0, 0),
            (0, aligned_kv_head_dim - actual_head_dim),
        ),
        constant_values=0,
    ).reshape(
        padded_total_tokens,
        actual_num_kv_heads_x2,
        aligned_kv_head_dim,
    ).transpose(1, 2, 0))


def prepare_outputs(out: jax.Array) -> jax.Array:
    kv_heads, max_tokens, q_per_kv_packed, q_packing, d = out.shape
    return out.reshape(kv_heads, max_tokens, q_per_kv_packed * q_packing, d)


def get_kv_cache_shape(
    total_num_pages,
    page_size,
    actual_num_kv_heads,
    actual_head_dim,
    kv_dtype,
    kv_layout: configs.KVLayout = configs.KVLayout.SEQ_ALONG_LANE,
):
    # page_size is configurable (any multiple of 128, e.g. 128/256/512/1024/2048)
    # and is fixed at cache-allocation time. Guidance from the bs=32 fp8 sweeps:
    #   - page_size=256 is the best general starting point, especially for SHORT
    #     context: no regression vs 128 at 1k (~0.99x) and up to ~1.24x at 8k,
    #     because it halves KV DMA descriptors while its partial last page wastes
    #     <=256 tokens.
    #   - page_size=2048 gives the BEST perf on very LONG context (~1.17x vs 512
    #     from 32k up, flat through 1M) and reaches ~60% HBM BW util at 512k, plus
    #     the most schedule-SMEM headroom for >1M support. But it regresses short
    #     context (~0.65x at 1k) due to last-page over-compute
    #     (see schedule.py TODO(perf, large-page-overcompute)).
    # Larger pages trade short-context efficiency for long-context throughput +
    # headroom; pick per the deployment's context regime.
    _require_seq_along_lane(kv_layout)
    num_sublanes = pltpu.get_tpu_info().num_sublanes
    kv_packing = utils.get_dtype_packing(kv_dtype)
    return (
        total_num_pages,
        actual_num_kv_heads * 2,
        utils.align_to(actual_head_dim, num_sublanes * kv_packing),
        page_size,
    )


def _calculate_vmem_usage(
    model_cfgs: configs.ModelConfigs,
    serve_cfgs: configs.ServingConfigs,
    batch_size: int,
    n_buffer: int,
    bq_sz: int,
    bkv_sz: int,
) -> int:
    """Estimate VMEM bytes used by the kernel for a given tile.

    Shared by ``calculate_block_sizes`` (the block-size search) and the
    spec-decode floor guard in ``_resolve_attn_static`` so both agree on the
    budget. See ``calculate_block_sizes`` for the derivation of each term.
    """
    tpu_info = pltpu.get_tpu_info()
    num_lanes = tpu_info.num_lanes

    aligned_head_dim = utils.align_to(model_cfgs.head_dim, num_lanes)
    aligned_num_q_heads_per_kv_head = utils.align_to(
        model_cfgs.num_q_heads_per_kv_head, serve_cfgs.packing_q)
    aligned_num_q_heads = aligned_num_q_heads_per_kv_head * model_cfgs.num_kv_heads

    bkv_stride = pl.cdiv(model_cfgs.num_kv_heads * 2, serve_cfgs.packing_kv)
    if utils.has_bank_conflicts(bkv_stride):
        bkv_stride += 1
    aligned_num_kv_heads_x2 = bkv_stride * serve_cfgs.packing_kv

    q_bytes = jnp.dtype(serve_cfgs.dtype_q).itemsize
    kv_bytes = jnp.dtype(serve_cfgs.dtype_kv).itemsize
    out_bytes = jnp.dtype(serve_cfgs.dtype_out).itemsize
    accum_bytes = jnp.dtype(configs.accum_dtype(serve_cfgs.dtype_out)).itemsize

    # Buffer memory (double/triple buffered). SEQ_ALONG_LANE staging carries a
    # +2*page_size halo for the boundary stitch.
    bq_array_size = bq_sz * aligned_num_q_heads * aligned_head_dim
    bkv_array_size = ((bkv_sz + 2 * serve_cfgs.page_size) *
                      aligned_num_kv_heads_x2 * aligned_head_dim)
    bo_array_size = bq_array_size

    buffer_bytes = (bq_array_size * q_bytes * n_buffer +
                    bkv_array_size * kv_bytes * n_buffer +
                    bo_array_size * out_bytes * 2)

    # Worst-case compute memory.
    loaded_bq_bytes = bq_sz * model_cfgs.num_q_heads * aligned_head_dim * q_bytes
    loaded_bkv_bytes = bkv_sz * model_cfgs.num_kv_heads * aligned_head_dim * kv_bytes
    qk_bytes = bq_sz * bkv_sz * model_cfgs.num_q_heads * accum_bytes
    compute_bytes = loaded_bq_bytes + loaded_bkv_bytes + qk_bytes

    return (buffer_bytes + compute_bytes) * batch_size


def calculate_block_sizes(
    model_cfgs: configs.ModelConfigs,
    serve_cfgs: configs.ServingConfigs,
    vmem_limit_bytes: int,
    decode_q_len: int = 1,
) -> tuple[configs.BlockSizes, configs.BlockSizes]:
    """Calculate optimal block size for decode and prefill."""

    tpu_info = pltpu.get_tpu_info()
    mxu_column_size = tpu_info.mxu_column_size

    def calculate_vmem_usage(batch_size: int, n_buffer: int, bq_sz: int,
                             bkv_sz: int) -> int:
        """Given tile size, calculate VMEM usage of the kernel."""
        return _calculate_vmem_usage(model_cfgs, serve_cfgs, batch_size,
                                     n_buffer, bq_sz, bkv_sz)

    def calculate_compute_buffer_time(batch_size: int, bq_c_sz: int,
                                      bkv_sz: int) -> int:
        """Calculate computational complexity of a single compute block."""

        num_k_rows = pl.cdiv(bkv_sz, mxu_column_size)
        num_k_cols = pl.cdiv(model_cfgs.head_dim, mxu_column_size)
        num_k = num_k_rows * num_k_cols
        num_muls = bq_c_sz * num_k * model_cfgs.num_q_heads

        return batch_size * num_muls

    def find_best_block_sizes(
            max_batch_size: int,
            max_n_buffer: int,
            fixed_bq_sz: int | None = None) -> configs.BlockSizes:
        """Loop through different block sizes to find the most optimal one."""

        # Even if we loose some potential performance, we want to avoid OOM at all
        # costs. Therefore, we conservatively only use 80% of the VMEM budget.
        capped_vmem_limit_bytes = vmem_limit_bytes * 0.8

        bkv_sz = bkv_stride = mxu_column_size
        if fixed_bq_sz is None:
            bq_sz = bq_stride = bkv_sz
        else:
            bq_sz = fixed_bq_sz
            bq_stride = 0
        batch_size = max_batch_size
        n_buffer = max_n_buffer

        # Step 1: Lower batch_size and/or n_buffer if even the smallest bq and bkv
        # size can trigger OOM.

        # If current batch size triggers OOM, decrease batch size until the kernel
        # fits within VMEM limit.
        while (batch_size > 1
               and calculate_vmem_usage(batch_size, n_buffer, bq_sz,
                                        bkv_sz) > capped_vmem_limit_bytes):
            batch_size -= 1

        # As a last resort, attempt to decrease number of buffers to avoid OOM.
        while (calculate_vmem_usage(batch_size, n_buffer, bq_sz, bkv_sz)
               > capped_vmem_limit_bytes):
            n_buffer -= 1

        # Indicates OOM was triggered even when batch_size=1 or n_buffer=1.
        # NOTE: If the function does not exit at this point even when either values
        # are zero, it will trigger infinite loop at the next while loop.
        if batch_size == 0 or n_buffer == 0:
            raise ValueError(
                "Cannot find batch size that fits within VMEM limit.")

        # Step 2: Increase block sizes until the kernel is unable to fit into VMEM.
        while (calculate_vmem_usage(batch_size, n_buffer, bq_sz, bkv_sz)
               < capped_vmem_limit_bytes):
            # Unless bq is a fixed value, we want to ensure bq size is the same as bkv
            # size. When using causal masking, if bq size is larger than bkv size,
            # entire kv tile can be masked out for some query tokens. Similarly, if
            # bkv size is larger than bq size, entire query tile can be masked out for
            # some kv tokens.
            bkv_sz += bkv_stride
            bq_sz += bq_stride

        # Rollback one step since the last attempted value triggered OOM.
        bkv_sz -= bkv_stride
        bq_sz -= bq_stride

        # Indicates OOM was triggered from the starting bkv size.
        if bkv_sz == 0:
            raise ValueError(
                "Cannot find block sizes that fit within VMEM limit.")

        # Step 3: Given current tile size, calculate compute tile size.

        threshold = 1500

        num_bq_c = 1
        last_valid_bq_c_sz = bq_c_sz = bq_sz
        bq_c_rem = 0

        while (calculate_compute_buffer_time(batch_size, bq_c_sz, bkv_sz)
               > threshold or bq_c_rem != 0) and num_bq_c < bq_sz:
            if bq_c_rem == 0:
                last_valid_bq_c_sz = bq_c_sz
            num_bq_c += 1
            bq_c_sz, bq_c_rem = divmod(bq_sz, num_bq_c)

        return configs.BlockSizes(
            bq_sz=bq_sz,
            bq_c_sz=last_valid_bq_c_sz,
            bkv_sz=bkv_sz,
            batch_size=batch_size,
            n_buffer=n_buffer,
        )

    # Default to triple buffer as its almost always beneficial.
    n_buffer = 3
    # Fixed value based on experimental results.
    decode_batch_size = 8
    prefill_batch_size = 2

    # Single-token decode pins bq_sz=1. Spec decode (decode_q_len>1) processes
    # decode_q_len query tokens per seq in one q-block, so it only needs
    # bq_sz >= decode_q_len -- use decode_q_len exactly, no sublane(8) rounding.
    # bq_sz is not itself a tiled dimension (the flash row dim is
    # bq_sz * num_q_heads_per_kv_head), and single-token decode already runs with
    # bq_sz=1, so a non-8-aligned bq_sz is valid. The flash compute iterates the
    # full static bq_sz, so any bq_sz above decode_q_len is wasted attention FLOPs
    # (e.g. rounding q_len=4 up to 8 doubles the per-seq QK/PV work).
    decode_fixed_bq_sz = max(decode_q_len, 1)
    decode_block_sizes = find_best_block_sizes(decode_batch_size, n_buffer,
                                               decode_fixed_bq_sz)
    # NOTE: the batched-RPA `long_ctx_decode` override (force bkv_sz=16384/
    # batch_size=4 when max_model_len>=32768) was REMOVED for stacked. It was
    # tuned on batched RPA (batched_rpa.benchmark --tune-blocks), keyed on the
    # static max_model_len rather than the live kv_len, and stacked's flash does
    # not clip compute to kv_len -> short/medium-context decode over-computed a
    # 16384-token block (~11x waste). Stacked decode now uses find_best_block_sizes;
    # a stacked-specific block-size sweep should set any long-context override.

    # DECODE runs the whole draft in ONE flash compute chunk. (1) Ensure
    # bq_sz >= decode_q_len -- the long-context override above pins bq_sz=1, which
    # is invalid for spec decode (decode_q_len>1). (2) Pin bq_c_sz == bq_sz:
    # decode's bq is tiny (1..decode_q_len), so one chunk is the efficient tiling;
    # a smaller bq_c_sz splits it into bq_sz fragmented QK+softmax+PV matmuls
    # (M = qpk each), which measured ~35% slower for spec decode at mid context.
    # PREFILL/MIXED are untouched: their large bq keeps the threshold-derived
    # smaller bq_c_sz (a full-bq compute buffer would blow VMEM there).
    decode_block_sizes = decode_block_sizes.floor_bq_to_decode_q_len(
        decode_q_len)
    decode_block_sizes = dataclasses.replace(decode_block_sizes,
                                             bq_c_sz=decode_block_sizes.bq_sz)
    prefill_block_sizes = find_best_block_sizes(prefill_batch_size, n_buffer)
    # MIXED mode uses the prefill block sizes. Cap bq_sz at the actual query
    # count so small-query MIXED workloads do not run a prefill-sized tile.
    # This is a no-op for prefill (total_q >= bq) and DECODE (bq=1).
    prefill_block_sizes = prefill_block_sizes.cap_bq_to_total_q(
        serve_cfgs.total_q_tokens)

    return decode_block_sizes, prefill_block_sizes


def _resolve_attn_static(
    queries: jax.Array,
    keys: jax.Array | None,
    kv_cache: jax.Array,
    kv_lens: jax.Array,
    page_indices: jax.Array,
    *,
    sm_scale: float,
    sliding_window: int | None,
    soft_cap: float | None,
    mask_value: float | None,
    q_scale: float | None,
    k_scale: float | None,
    v_scale: float | None,
    vmem_limit_bytes: int | None,
    out_dtype: jnp.dtype | None,
    kv_layout: configs.KVLayout,
    decode_block_sizes: configs.BlockSizes | None,
    prefill_block_sizes: configs.BlockSizes | None,
    decode_q_len: int = 1,
    num_kv_heads: int | None = None,
):
    """Resolve the static attention configs + effective block sizes.

    Shared by ``ragged_paged_attention`` and ``build_schedules`` so the
    ``RpaConfigs`` used to precompute a schedule are byte-identical to the ones
    used by the attention kernel (the schedule embeds ``cfgs`` statically and
    the kernel indexes into it, so any drift would be a correctness bug).
    """
    if out_dtype is None:
        out_dtype = queries.dtype
    if mask_value is None:
        mask_value = jnp.finfo(out_dtype).min
    if vmem_limit_bytes is None:
        vmem_limit_bytes = pltpu.get_tpu_info().vmem_capacity_bytes

    max_num_seqs = kv_lens.shape[0]
    _require_seq_along_lane(kv_layout)
    page_size = kv_cache.shape[3]

    num_q_heads = queries.shape[1]
    head_dim = queries.shape[2]
    if num_kv_heads is None:
        if keys is None:
            raise ValueError("num_kv_heads must be provided when keys is None")
        num_kv_heads = keys.shape[1]
    num_page_indices = page_indices.shape[0]

    model_cfgs = configs.ModelConfigs(
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        sliding_window=sliding_window,
        sm_scale=sm_scale,
        soft_cap=soft_cap,
        mask_value=mask_value,
    )
    serve_cfgs = configs.ServingConfigs(
        num_seqs=max_num_seqs,
        num_page_indices=num_page_indices,
        total_q_tokens=queries.shape[0],
        dtype_q=queries.dtype,
        dtype_kv=kv_cache.dtype,
        dtype_out=out_dtype,
        page_size=page_size,
        scale_q=q_scale,
        scale_k=k_scale,
        scale_v=v_scale,
        kv_layout=kv_layout,
    )

    default_decode, default_prefill = calculate_block_sizes(
        model_cfgs, serve_cfgs, vmem_limit_bytes, decode_q_len)
    max_model_len = serve_cfgs.pages_per_seq * page_size
    tuned_mixed = tuned_block_sizes.get_tuned_block_sizes(
        model_cfgs, serve_cfgs, max_model_len, configs.RpaCase.MIXED)
    # Decode block size. Priority: explicit decode_block_sizes arg > env bkv
    # override (STACKED_RPA_BKV) > max_model_len-keyed tuned table > VMEM
    # heuristic. The env override lets the user drive bkv_sz directly and SKIP the
    # max_model_len-keyed lookup entirely (page_size is likewise user-controlled
    # via STACKED_RPA_PAGE_SIZE). Non-page-multiple values are fixed by
    # _clamp_bkv_to_page below. When no override is set, the tuned table (keyed on
    # sliding_window + a bucket of the 128-padded total_q_tokens) still applies.
    from tpu_inference.kernels.experimental.stacked_rpa.envvars import \
        stacked_rpa_bkv

    user_bkv = stacked_rpa_bkv()
    if decode_block_sizes is not None:
        effective_decode = decode_block_sizes
    elif user_bkv > 0:
        effective_decode = dataclasses.replace(default_decode, bkv_sz=user_bkv)
    else:
        tuned_decode = tuned_block_sizes.get_tuned_block_sizes(
            model_cfgs, serve_cfgs, max_model_len, configs.RpaCase.DECODE)
        effective_decode = tuned_decode or default_decode
    effective_prefill = prefill_block_sizes or tuned_mixed or default_prefill

    # Window-anchored sliding-window DECODE (STACKED_RPA_SW_BOUND, default on): size the
    # decode block to cover exactly one page-aligned window in a single iteration.
    # The schedule anchors the block at the window start (not the global grid), so
    # the block only needs to span the window + spec q-len + page misalignment:
    #   pages = cdiv(sliding_window + decode_q_len + page - 1, page)
    # (e.g. page 512, sw 2048, q 4 -> 6 pages = 3072). This overrides the tuned /
    # heuristic / user bkv for the anchored path because the anchor's correctness
    # requires the single block to cover the whole window. Explicit
    # decode_block_sizes still wins (caller override). No effect without a window.
    from tpu_inference.kernels.experimental.stacked_rpa.envvars import \
        stacked_rpa_sw_bound

    sw_bound = stacked_rpa_sw_bound()
    if sliding_window is not None and sw_bound and decode_block_sizes is None:
        page = serve_cfgs.page_size
        win_pages = -(-(sliding_window + decode_q_len + page - 1) // page)
        win_bkv = win_pages * page
        effective_decode = dataclasses.replace(effective_decode,
                                               bkv_sz=win_bkv)

    # Spec decode (decode_q_len>1) runs through the DECODE region, which needs
    # bq_sz >= decode_q_len. Tuned/user/default decode entries pin bq_sz for
    # single-token decode, so floor it up here (no-op at decode_q_len==1).
    if decode_q_len > 1:
        floored = effective_decode.floor_bq_to_decode_q_len(decode_q_len)
        # Flooring grows the bq-scaled buffers. The tuned/user path never runs the
        # VMEM estimator, so a large tuned bkv floored to bq_sz>=decode_q_len could
        # OOM. default_decode is already VMEM-checked with fixed_bq_sz=
        # align(decode_q_len), so fall back to it when the floored tile would
        # exceed the 80% budget (never silently OOM).
        fits = (_calculate_vmem_usage(
            model_cfgs,
            serve_cfgs,
            floored.batch_size,
            floored.n_buffer,
            floored.bq_sz,
            floored.bkv_sz,
        ) <= vmem_limit_bytes * 0.8)
        effective_decode = floored if fits else default_decode

    # SEQ_ALONG_LANE is page-granular: every k-block must cover >= 1 page, so
    # bkv_sz must be a positive multiple of page_size. A block with bkv_sz <
    # page_size gives bkv_p = bkv_sz // page_size = 0, which produces an empty
    # dma_kv_cache schedule leaf (s32[0]) and fails Mosaic compilation. This
    # bites large page sizes (e.g. page_size=1024 with a 512-token mixed block).
    # Round bkv_sz up to a page multiple (>= one page). No-op when bkv_sz is
    # already a page multiple (all existing page<=512 configs).
    def _clamp_bkv_to_page(blk):
        page = serve_cfgs.page_size
        aligned = max(page, -(-blk.bkv_sz // page) * page)
        if aligned == blk.bkv_sz:
            return blk
        return dataclasses.replace(blk, bkv_sz=aligned)

    effective_decode = _clamp_bkv_to_page(effective_decode)
    effective_prefill = _clamp_bkv_to_page(effective_prefill)

    return (
        model_cfgs,
        serve_cfgs,
        effective_decode,
        effective_prefill,
        vmem_limit_bytes,
        out_dtype,
        mask_value,
    )


def _make_cfgs(
    mode,
    model_cfgs,
    serve_cfgs,
    effective_decode,
    effective_prefill,
    vmem_limit_bytes,
    *,
    has_visibility: bool = False,
    update_kv_cache: bool = True,
    decode_q_len: int = 1,
):
    effective_blocks = (effective_decode if mode == configs.RpaCase.DECODE else
                        effective_prefill)
    # The DECODE region always uses the transposed stacked schedule (keyed on
    # mode via RpaConfigs.is_stacked). Dense cross-step packing is controlled by:
    #   - DECODE via STACKED_RPA_DENSE_PACK (default on)
    #   - MIXED via STACKED_RPA_MIXED (also flips MIXED onto the stacked
    #     schedule through is_stacked); default off keeps MIXED data-parallel.
    #
    # TODO(perf, mixed-overhead): STACKED_RPA_MIXED adds measured +5..18%
    # kernel-time overhead on *balanced* prefill vs the data-parallel path
    # (higher for larger q_len; e.g. L=8192 -> ~+18%), because the dense schedule
    # splits every (seq, q-block) unit across cells and pays a cross-cell softmax
    # combine (span>1, so the span==1 fast path never fires) plus q replication --
    # with no upside, since prefill's many q-blocks already fill the cells. Flag
    # OFF is bit-identical to batched (~0% overhead), so it stays the default.
    # To make MIXED-stacking free on balanced work, keep each unit on a single
    # cell (span==1, combine-free) and split only on genuine imbalance
    # (hybrid packing), or detect balance and fall back to the data-parallel
    # path (adaptive dispatch). Until then, only enable for imbalanced/long
    # mixed batches.
    import os

    dense_pack = (os.environ.get("STACKED_RPA_DENSE_PACK", "1") == "1"
                  and mode == configs.RpaCase.DECODE) or (
                      mode == configs.RpaCase.MIXED
                      and os.environ.get("STACKED_RPA_MIXED", "0") == "1")
    from tpu_inference.kernels.experimental.stacked_rpa.envvars import \
        stacked_rpa_sw_bound

    return configs.RpaConfigs(
        block=effective_blocks,
        model=model_cfgs,
        serve=serve_cfgs,
        vmem_limit_bytes=vmem_limit_bytes,
        mode=mode,
        has_visibility=has_visibility,
        update_kv_cache=update_kv_cache,
        # decode_q_len is only meaningful for the DECODE region; MIXED/PREFILL
        # always process dynamic/static q via bq_sz and keep the default 1.
        decode_q_len=decode_q_len if mode == configs.RpaCase.DECODE else 1,
        dense_pack=dense_pack,
        sw_bound=stacked_rpa_sw_bound(),
    )


def _make_visibility_hbm(
    queries: jax.Array,
    effective_decode: configs.BlockSizes,
    effective_prefill: configs.BlockSizes,
    visibility: jax.Array | None,
) -> jax.Array:
    visibility_pad = max(effective_decode.bq_sz, effective_prefill.bq_sz)
    if visibility is None:
        return jnp.zeros((queries.shape[0] + visibility_pad, 128),
                         dtype=jnp.int32)
    return jnp.pad(visibility, ((0, visibility_pad), (0, 126)))


def _validate_prepacked_inputs(
    cfgs: configs.RpaConfigs,
    q: jax.Array,
    prepacked_new_kv_hbm: jax.Array,
    kv_cache: jax.Array,
    kv_lens: jax.Array,
    page_indices: jax.Array,
    cu_q_lens: jax.Array,
    distribution: jax.Array,
    visibility: jax.Array | None = None,
) -> None:
    """Validate the prepacked-KV entry point without requiring normal K/V arrays."""
    if q.ndim != 3:
        raise ValueError(f"Expected 3D array for {q.shape=}")
    if cfgs.serve.kv_layout != configs.KVLayout.SEQ_ALONG_LANE:
        raise ValueError("prepacked KV is only supported for SEQ_ALONG_LANE")

    expected_kv_shape = (
        cfgs.model.num_kv_heads * 2,
        cfgs.aligned_kv_head_dim,
        utils.align_to(q.shape[0],
                       pltpu.get_tpu_info().num_lanes),
    )
    if prepacked_new_kv_hbm.shape != expected_kv_shape:
        raise ValueError(
            f"Expected {prepacked_new_kv_hbm.shape=} to be equal to"
            f" {expected_kv_shape=}")
    if prepacked_new_kv_hbm.dtype != kv_cache.dtype:
        raise ValueError(
            "Expected prepacked KV dtype and KV cache dtype to match, but got"
            f" {prepacked_new_kv_hbm.dtype=} and {kv_cache.dtype=}")

    expected_kv_cache_shape = (
        kv_cache.shape[0],
        cfgs.model.num_kv_heads * 2,
        cfgs.aligned_kv_head_dim,
        cfgs.serve.page_size,
    )
    if kv_cache.shape != expected_kv_cache_shape:
        raise ValueError(
            f"Expected {kv_cache.shape=} to be equal to {expected_kv_cache_shape=}"
        )
    if not jnp.issubdtype(kv_cache.dtype, jnp.floating):
        raise ValueError(f"Expected {kv_cache.dtype=} to be a floating point.")

    if not (jnp.int32 == kv_lens.dtype == page_indices.dtype == cu_q_lens.dtype
            == distribution.dtype):
        raise ValueError(
            f"Expected int32 dtype for {kv_lens.dtype=}, {page_indices.dtype=},"
            f" {cu_q_lens.dtype=}, {distribution.dtype=}")
    if not (kv_lens.ndim == page_indices.ndim == cu_q_lens.ndim == 1):
        raise ValueError(
            f"Expected 1D array for {kv_lens.shape=}, {page_indices.shape=},"
            f" {cu_q_lens.shape=}")
    max_num_seqs = kv_lens.shape[0]
    num_page_indices = page_indices.shape[0]
    if num_page_indices % max_num_seqs != 0:
        raise ValueError(
            f"Expected {num_page_indices=} to be divisible by {max_num_seqs=}."
        )
    if cu_q_lens.shape != (max_num_seqs + 1, ):
        raise ValueError(
            f"Expected {cu_q_lens.shape=} to be ({max_num_seqs + 1},).")
    if distribution.shape != (3, ):
        raise ValueError(f"Expected {distribution.shape=} to be (3,).")
    if visibility is not None:
        if visibility.shape != (q.shape[0], 2):
            raise ValueError(
                f"Expected {visibility.shape=} to be ({q.shape[0]}, 2).")
        if visibility.dtype != jnp.int32:
            raise ValueError(
                f"Expected {visibility.dtype=} to be {jnp.int32=}.")
        if cfgs.model.sliding_window is not None:
            raise ValueError(
                "visibility and sliding_window are mutually exclusive.")


def build_schedules(
    queries: jax.Array,
    keys: jax.Array,
    values: jax.Array,
    kv_cache: jax.Array,
    kv_lens: jax.Array,
    page_indices: jax.Array,
    cu_q_lens: jax.Array,
    distribution: jax.Array,
    visibility: jax.Array | None = None,
    *,
    sm_scale: float = 1.0,
    sliding_window: int | None = None,
    soft_cap: float | None = None,
    mask_value: float | None = None,
    q_scale: float | None = None,
    k_scale: float | None = None,
    v_scale: float | None = None,
    decode_q_len: int = 1,
    vmem_limit_bytes: int | None = None,
    out_dtype: jnp.dtype | None = None,
    kv_layout: configs.KVLayout = configs.KVLayout.SEQ_ALONG_LANE,
    decode_block_sizes: configs.BlockSizes | None = None,
    prefill_block_sizes: configs.BlockSizes | None = None,
    update_kv_cache: bool = True,
):
    """Precompute the (DECODE, MIXED) RPA schedules once for a forward step.

    The schedule depends on ``cu_q_lens, kv_lens, page_indices, distribution`` +
    static cfgs. ``page_indices`` is folded into the emitted DMA offsets, so a
    precomputed schedule may only be reused by layers sharing the same metadata
    stream. ``values`` is accepted for a signature symmetric with
    ``ragged_paged_attention`` (only shapes/dtypes of q/k/kv_cache are needed
    for cfgs).
    """
    del values
    if visibility is not None and sliding_window is not None:
        raise ValueError(
            "visibility and sliding_window are mutually exclusive.")
    if decode_q_len < 1:
        raise ValueError(f"decode_q_len must be >= 1, got {decode_q_len}")
    (
        model_cfgs,
        serve_cfgs,
        effective_decode,
        effective_prefill,
        vmem_limit_bytes,
        _out_dtype,
        _mask_value,
    ) = _resolve_attn_static(
        queries,
        keys,
        kv_cache,
        kv_lens,
        page_indices,
        sm_scale=sm_scale,
        sliding_window=sliding_window,
        soft_cap=soft_cap,
        mask_value=mask_value,
        q_scale=q_scale,
        k_scale=k_scale,
        v_scale=v_scale,
        vmem_limit_bytes=vmem_limit_bytes,
        out_dtype=out_dtype,
        kv_layout=kv_layout,
        decode_block_sizes=decode_block_sizes,
        prefill_block_sizes=prefill_block_sizes,
        decode_q_len=decode_q_len,
    )
    decode_cfgs = _make_cfgs(
        configs.RpaCase.DECODE,
        model_cfgs,
        serve_cfgs,
        effective_decode,
        effective_prefill,
        vmem_limit_bytes,
        has_visibility=visibility is not None,
        update_kv_cache=update_kv_cache,
        decode_q_len=decode_q_len,
    )
    mixed_cfgs = _make_cfgs(
        configs.RpaCase.MIXED,
        model_cfgs,
        serve_cfgs,
        effective_decode,
        effective_prefill,
        vmem_limit_bytes,
        has_visibility=visibility is not None,
        update_kv_cache=update_kv_cache,
    )
    visibility_hbm = _make_visibility_hbm(queries, effective_decode,
                                          effective_prefill, visibility)
    decode_schedule = schedule.generate_rpa_metadata(
        cu_q_lens,
        kv_lens,
        distribution,
        page_indices,
        cfgs=decode_cfgs,
        visibility=visibility_hbm if visibility is not None else None,
    )
    mixed_schedule = schedule.generate_rpa_metadata(
        cu_q_lens,
        kv_lens,
        distribution,
        page_indices,
        cfgs=mixed_cfgs,
        visibility=visibility_hbm if visibility is not None else None,
    )
    return decode_schedule, mixed_schedule


def build_schedules_prepacked_kv(
    queries: jax.Array,
    prepacked_new_kv_hbm: jax.Array,
    kv_cache: jax.Array,
    kv_lens: jax.Array,
    page_indices: jax.Array,
    cu_q_lens: jax.Array,
    distribution: jax.Array,
    visibility: jax.Array | None = None,
    *,
    num_kv_heads: int,
    sm_scale: float = 1.0,
    sliding_window: int | None = None,
    soft_cap: float | None = None,
    mask_value: float | None = None,
    q_scale: float | None = None,
    k_scale: float | None = None,
    v_scale: float | None = None,
    decode_q_len: int = 1,
    vmem_limit_bytes: int | None = None,
    out_dtype: jnp.dtype | None = None,
    decode_block_sizes: configs.BlockSizes | None = None,
    prefill_block_sizes: configs.BlockSizes | None = None,
    update_kv_cache: bool = True,
):
    """Precompute schedules for the prepacked-KV SEQ_ALONG_LANE entry point."""
    if visibility is not None and sliding_window is not None:
        raise ValueError(
            "visibility and sliding_window are mutually exclusive.")
    if decode_q_len < 1:
        raise ValueError(f"decode_q_len must be >= 1, got {decode_q_len}")
    (
        model_cfgs,
        serve_cfgs,
        effective_decode,
        effective_prefill,
        vmem_limit_bytes,
        _out_dtype,
        _mask_value,
    ) = _resolve_attn_static(
        queries,
        None,
        kv_cache,
        kv_lens,
        page_indices,
        sm_scale=sm_scale,
        sliding_window=sliding_window,
        soft_cap=soft_cap,
        mask_value=mask_value,
        q_scale=q_scale,
        k_scale=k_scale,
        v_scale=v_scale,
        vmem_limit_bytes=vmem_limit_bytes,
        out_dtype=out_dtype,
        kv_layout=configs.KVLayout.SEQ_ALONG_LANE,
        decode_block_sizes=decode_block_sizes,
        prefill_block_sizes=prefill_block_sizes,
        num_kv_heads=num_kv_heads,
        decode_q_len=decode_q_len,
    )
    decode_cfgs = _make_cfgs(
        configs.RpaCase.DECODE,
        model_cfgs,
        serve_cfgs,
        effective_decode,
        effective_prefill,
        vmem_limit_bytes,
        has_visibility=visibility is not None,
        update_kv_cache=update_kv_cache,
        decode_q_len=decode_q_len,
    )
    mixed_cfgs = _make_cfgs(
        configs.RpaCase.MIXED,
        model_cfgs,
        serve_cfgs,
        effective_decode,
        effective_prefill,
        vmem_limit_bytes,
        has_visibility=visibility is not None,
        update_kv_cache=update_kv_cache,
    )
    _validate_prepacked_inputs(
        decode_cfgs,
        queries,
        prepacked_new_kv_hbm,
        kv_cache,
        kv_lens,
        page_indices,
        cu_q_lens,
        distribution,
        visibility=visibility,
    )
    visibility_hbm = _make_visibility_hbm(queries, effective_decode,
                                          effective_prefill, visibility)
    # Stacked RPA is SEQ_ALONG_LANE-only and runs native multi-token (spec)
    # decode on the stacked DECODE kernel, so spec-decode sequences stay in the
    # caller's decode bucket (no force-to-MIXED for decode_q_len > 1, unlike the
    # batched HEAD_ALONG_SUBLANE path).
    decode_schedule = schedule.generate_rpa_metadata(
        cu_q_lens,
        kv_lens,
        distribution,
        page_indices,
        cfgs=decode_cfgs,
        visibility=visibility_hbm if visibility is not None else None,
    )
    mixed_schedule = schedule.generate_rpa_metadata(
        cu_q_lens,
        kv_lens,
        distribution,
        page_indices,
        cfgs=mixed_cfgs,
        visibility=visibility_hbm if visibility is not None else None,
    )
    return decode_schedule, mixed_schedule


@jax.jit(
    static_argnames=(
        "num_kv_heads",
        "sm_scale",
        "sliding_window",
        "soft_cap",
        "mask_value",
        "q_scale",
        "k_scale",
        "v_scale",
        "decode_q_len",
        "decode_block_sizes",
        "prefill_block_sizes",
        "vmem_limit_bytes",
        "debug_mode",
        "out_dtype",
        "use_causal_mask",
        "update_kv_cache",
    ),
    donate_argnames=("queries", "kv_cache"),
)
def ragged_paged_attention_prepacked_kv(
    queries: jax.Array,
    prepacked_new_kv_hbm: jax.Array,
    kv_cache: jax.Array,
    kv_lens: jax.Array,
    page_indices: jax.Array,
    cu_q_lens: jax.Array,
    distribution: jax.Array,
    attention_sink: jax.Array | None = None,
    visibility: jax.Array | None = None,
    *,
    num_kv_heads: int,
    sm_scale: float = 1.0,
    sliding_window: int | None = None,
    soft_cap: float | None = None,
    mask_value: float | None = None,
    q_scale: float | None = None,
    k_scale: float | None = None,
    v_scale: float | None = None,
    decode_q_len: int = 1,
    decode_block_sizes: configs.BlockSizes | None = None,
    prefill_block_sizes: configs.BlockSizes | None = None,
    vmem_limit_bytes: int | None = None,
    debug_mode: bool = False,
    out_dtype: jnp.dtype | None = None,
    use_causal_mask: bool = True,
    update_kv_cache: bool = True,
    precomputed_schedules: tuple | None = None,
) -> tuple[jax.Array, jax.Array]:
    """Perform batched RPA with K/V already in SEQ_ALONG_LANE ``new_kv_hbm``."""

    if not use_causal_mask:
        raise ValueError("Only causal attention is supported.")
    if attention_sink is not None:
        raise ValueError("attention_sink is not supported by batched RPA.")
    if visibility is not None and sliding_window is not None:
        raise ValueError(
            "visibility and sliding_window are mutually exclusive.")
    if debug_mode:
        raise ValueError("Debug mode is not supported.")
    if decode_q_len < 1:
        raise ValueError(f"decode_q_len must be >= 1, got {decode_q_len}")

    (
        model_cfgs,
        serve_cfgs,
        effective_decode,
        effective_prefill,
        vmem_limit_bytes,
        out_dtype,
        mask_value,
    ) = _resolve_attn_static(
        queries,
        None,
        kv_cache,
        kv_lens,
        page_indices,
        sm_scale=sm_scale,
        sliding_window=sliding_window,
        soft_cap=soft_cap,
        mask_value=mask_value,
        q_scale=q_scale,
        k_scale=k_scale,
        v_scale=v_scale,
        vmem_limit_bytes=vmem_limit_bytes,
        out_dtype=out_dtype,
        kv_layout=configs.KVLayout.SEQ_ALONG_LANE,
        decode_block_sizes=decode_block_sizes,
        prefill_block_sizes=prefill_block_sizes,
        num_kv_heads=num_kv_heads,
        decode_q_len=decode_q_len,
    )
    num_q_heads = queries.shape[1]
    head_dim = queries.shape[2]

    q_hbm = prepare_queries(queries, num_kv_heads, queries.dtype)
    visibility_hbm = _make_visibility_hbm(queries, effective_decode,
                                          effective_prefill, visibility)

    # Stacked RPA is SEQ_ALONG_LANE-only: keep spec decode on the stacked DECODE
    # kernel (no force-to-MIXED for decode_q_len > 1).
    def run_rpa_kernel(
        mode: configs.RpaCase,
        o_hbm_alias_q_hbm: jax.Array,
        kv_cache: jax.Array,
    ):
        cfgs = _make_cfgs(
            mode,
            model_cfgs,
            serve_cfgs,
            effective_decode,
            effective_prefill,
            vmem_limit_bytes,
            has_visibility=visibility is not None,
            update_kv_cache=update_kv_cache,
            decode_q_len=decode_q_len,
        )
        _validate_prepacked_inputs(
            cfgs,
            queries,
            prepacked_new_kv_hbm,
            kv_cache,
            kv_lens,
            page_indices,
            cu_q_lens,
            distribution,
            visibility=visibility,
        )

        if precomputed_schedules is not None:
            schedule_hbm = (precomputed_schedules[0]
                            if mode == configs.RpaCase.DECODE else
                            precomputed_schedules[1])
        else:
            schedule_hbm = schedule.generate_rpa_metadata(
                cu_q_lens,
                kv_lens,
                distribution,
                page_indices,
                cfgs=cfgs,
                visibility=visibility_hbm if visibility is not None else None,
            )
        return kernel.rpa_kernel(
            cu_q_lens,
            kv_lens,
            schedule_hbm,
            o_hbm_alias_q_hbm,
            prepacked_new_kv_hbm,
            kv_cache,
            visibility_hbm,
            cfgs=cfgs,
        )

    def maybe_run_rpa_kernel(
        mode: configs.RpaCase,
        o_hbm_alias_q_hbm: jax.Array,
        kv_cache: jax.Array,
    ):
        start, end = mode.get_range(distribution)
        return jax.lax.cond(
            end > start,
            lambda args: run_rpa_kernel(mode, *args),
            lambda args: args,
            (o_hbm_alias_q_hbm, kv_cache),
        )

    o_hbm_alias_q_hbm, kv_cache = maybe_run_rpa_kernel(configs.RpaCase.DECODE,
                                                       q_hbm, kv_cache)
    o_hbm_alias_q_hbm, kv_cache = maybe_run_rpa_kernel(configs.RpaCase.MIXED,
                                                       o_hbm_alias_q_hbm,
                                                       kv_cache)

    o_hbm = prepare_outputs(o_hbm_alias_q_hbm)
    num_q_heads_per_kv_head = num_q_heads // num_kv_heads
    o_hbm = o_hbm[:, :, :num_q_heads_per_kv_head, :head_dim]
    o_hbm = o_hbm.swapaxes(1, 0).reshape(queries.shape)

    return o_hbm, kv_cache


@jax.jit(
    static_argnames=(
        "sm_scale",
        "sliding_window",
        "soft_cap",
        "mask_value",
        "q_scale",
        "k_scale",
        "v_scale",
        "chunk_prefill_size",
        "decode_q_len",
        "decode_block_sizes",
        "prefill_block_sizes",
        "vmem_limit_bytes",
        "debug_mode",
        "out_dtype",
        "use_causal_mask",
        "update_kv_cache",
        "kv_layout",
    ),
    donate_argnames=("queries", "keys", "values", "kv_cache"),
)
def ragged_paged_attention(
    queries: jax.Array,
    keys: jax.Array,
    values: jax.Array,
    kv_cache: jax.Array,
    kv_lens: jax.Array,
    page_indices: jax.Array,
    cu_q_lens: jax.Array,
    distribution: jax.Array,
    attention_sink: jax.Array | None = None,
    visibility: jax.Array | None = None,
    *,
    sm_scale: float = 1.0,
    sliding_window: int | None = None,
    soft_cap: float | None = None,
    mask_value: float | None = None,
    q_scale: float | None = None,
    k_scale: float | None = None,
    v_scale: float | None = None,
    chunk_prefill_size: int | None = None,
    decode_q_len: int = 1,
    decode_block_sizes: configs.BlockSizes | None = None,
    prefill_block_sizes: configs.BlockSizes | None = None,
    vmem_limit_bytes: int | None = None,
    debug_mode: bool = False,
    out_dtype: jnp.dtype | None = None,
    use_causal_mask: bool = True,
    update_kv_cache: bool = True,
    kv_layout: configs.KVLayout = configs.KVLayout.SEQ_ALONG_LANE,
    precomputed_schedules: tuple | None = None,
    prepacked_new_kv_hbm: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array]:
    """Perform batched ragged paged attention.

    ``precomputed_schedules``: optional ``(decode_schedule, mixed_schedule)``
    tuple of ``RpaSchedule`` from ``build_schedules(...)``. When provided, the
    per-call ``generate_rpa_metadata`` is skipped and these are used instead
    (lets a caller hoist the schedule build to once-per-step). It is a runtime
    (traced) value, so it is NOT a static argument.

    ``prepacked_new_kv_hbm``: optional post-RoPE K/V already packed in the
    SEQ_ALONG_LANE ``new_kv_hbm`` layout. When provided, ``prepare_inputs`` skips
    the K/V concat+pad+transpose and only prepares Q.

    Args:
        queries: [max_num_tokens, num_q_heads, head_dim]. Output of q projection.
        keys: [max_num_tokens, num_kv_heads, head_dim]. Output of k projection.
        values: [max_num_tokens, num_kv_heads, head_dim]. Output of v projection.
        kv_cache: [num_pages, page_size, cdiv(num_kv_heads * 2, kv_packing),
            kv_packing, head_dim]. Stores existing kv cache data where k & vs are
            concatenated along num kv heads dim.
        kv_lens: [max_num_seqs]. Existing kv cache length of each sequence.
            page_indices: [max_num_seqs * pages_per_seqs]. kv cache page table of each
            sequence.
        cu_q_lens: [max_num_seqs + 1]. Cumulative sum of each sequence's query
            length. queries[a:b], keys[a:b], and values[a:b] where a=cu_q_lens[i] and
            b=cu_q_lens[i+1] represents q/k/v of sequence i.
        distribution: [3]. Cumulative sum of number of decode, prefill, and mixed
            sequences. distribution[2] represents total number of sequences.
        attention_sink: Not supported by batched RPA.
        visibility: Optional per-token visibility ranges. Shape [max_num_tokens, 2]
            of i32, where visibility[t] = (start, end) means token t can attend to KV
            positions in [start, end]. Replaces causal and sliding-window masking.
        sm_scale: Softmax scale value.
        sliding_window: Size of sliding window (also known as local attention). kvs
            outside of the window is not fetched from hbm and masked out during
            computation.
        soft_cap: Cap values of softmax inputs.
        mask_value: Value to use for causal masking. Defaults to smallest
            representable value of the activation dtype.
        q_scale: Quantization scale value of queries.
        k_scale: Quantization scale value of keys.
        v_scale: Quantization scale value of values.
        chunk_prefill_size: Not used.
        decode_block_sizes: Kernel block size to use during decode.
        prefill_block_sizes: Kernel block size to use during prefill.
        vmem_limit_bytes: VMEM size limit of the kernel. Defaults to maximum VMEM
            size of the hardware.
        debug_mode: Not used.
        out_dtype: Dtype of output. Defaults to dtype of queries.
        use_causal_mask: Not used.

    Returns:
        out: [max_num_tokens, num_q_heads, head_dim]. Output of self attention.
        new_kv_cache: [num_pages, page_size, cdiv(num_kv_heads * 2, kv_packing),
            kv_packing, head_dim]. Result of new kv cache where k & vs are
            concatenated along num kv heads dim.
    """

    if not use_causal_mask:
        raise ValueError("Only causal attention is supported.")
    if attention_sink is not None:
        raise ValueError("attention_sink is not supported by batched RPA.")
    if visibility is not None and sliding_window is not None:
        raise ValueError(
            "visibility and sliding_window are mutually exclusive.")
    if chunk_prefill_size is not None:
        raise ValueError("Specifying chunk prefill size is not supported.")
    if debug_mode:
        raise ValueError("Debug mode is not supported.")
    if decode_q_len < 1:
        raise ValueError(f"decode_q_len must be >= 1, got {decode_q_len}")

    (
        model_cfgs,
        serve_cfgs,
        effective_decode,
        effective_prefill,
        vmem_limit_bytes,
        out_dtype,
        mask_value,
    ) = _resolve_attn_static(
        queries,
        keys,
        kv_cache,
        kv_lens,
        page_indices,
        sm_scale=sm_scale,
        sliding_window=sliding_window,
        soft_cap=soft_cap,
        mask_value=mask_value,
        q_scale=q_scale,
        k_scale=k_scale,
        v_scale=v_scale,
        vmem_limit_bytes=vmem_limit_bytes,
        out_dtype=out_dtype,
        kv_layout=kv_layout,
        decode_block_sizes=decode_block_sizes,
        prefill_block_sizes=prefill_block_sizes,
        decode_q_len=decode_q_len,
    )
    num_q_heads = queries.shape[1]
    head_dim = queries.shape[2]
    num_kv_heads = keys.shape[1]

    q_hbm, new_kv_hbm = prepare_inputs(
        queries,
        keys,
        values,
        queries.dtype,
        kv_cache.dtype,
        kv_layout=kv_layout,
        prepacked_new_kv_hbm=prepacked_new_kv_hbm,
    )
    visibility_hbm = _make_visibility_hbm(queries, effective_decode,
                                          effective_prefill, visibility)

    def run_rpa_kernel(
        mode: configs.RpaCase,
        o_hbm_alias_q_hbm: jax.Array,
        kv_cache: jax.Array,
    ):
        cfgs = _make_cfgs(
            mode,
            model_cfgs,
            serve_cfgs,
            effective_decode,
            effective_prefill,
            vmem_limit_bytes,
            has_visibility=visibility is not None,
            update_kv_cache=update_kv_cache,
            decode_q_len=decode_q_len,
        )
        cfgs.validate_inputs(
            q=queries,
            k=keys,
            v=values,
            kv_cache=kv_cache,
            kv_lens=kv_lens,
            page_indices=page_indices,
            cu_q_lens=cu_q_lens,
            distribution=distribution,
            visibility=visibility,
        )

        if precomputed_schedules is not None:
            schedule_hbm = (precomputed_schedules[0]
                            if mode == configs.RpaCase.DECODE else
                            precomputed_schedules[1])
        else:
            schedule_hbm = schedule.generate_rpa_metadata(
                cu_q_lens,
                kv_lens,
                distribution,
                page_indices,
                cfgs=cfgs,
                visibility=visibility_hbm if visibility is not None else None,
            )
        return kernel.rpa_kernel(
            cu_q_lens,
            kv_lens,
            schedule_hbm,
            o_hbm_alias_q_hbm,
            new_kv_hbm,
            kv_cache,
            visibility_hbm,
            cfgs=cfgs,
        )

    def maybe_run_rpa_kernel(
        mode: configs.RpaCase,
        o_hbm_alias_q_hbm: jax.Array,
        kv_cache: jax.Array,
    ):
        start, end = mode.get_range(distribution)
        return jax.lax.cond(
            end > start,
            lambda args: run_rpa_kernel(mode, *args),
            lambda args: args,
            (o_hbm_alias_q_hbm, kv_cache),
        )

    o_hbm_alias_q_hbm, kv_cache = maybe_run_rpa_kernel(configs.RpaCase.DECODE,
                                                       q_hbm, kv_cache)
    o_hbm_alias_q_hbm, kv_cache = maybe_run_rpa_kernel(configs.RpaCase.MIXED,
                                                       o_hbm_alias_q_hbm,
                                                       kv_cache)

    # before: [kv_heads, max_tokens, q_per_kv // q_packing, q_packing, d]
    o_hbm = prepare_outputs(o_hbm_alias_q_hbm)
    # after: [kv_heads, max_tokens, q_per_kv, d]

    # slice back to original shape if padded
    num_q_heads_per_kv_head = num_q_heads // num_kv_heads
    o_hbm = o_hbm[:, :, :num_q_heads_per_kv_head, :head_dim]
    o_hbm = o_hbm.swapaxes(1, 0).reshape(queries.shape)

    return o_hbm, kv_cache
