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

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from tpu_inference.kernels.experimental.batched_rpa import configs, utils


def _stitch_decode_lane(
    vmem_u32_ref: jax.Array,
    bkv_sz_cache: jax.Array,
    cache_end: jax.Array,
    new_tok_offset: jax.Array,
    bkv_sz_new: jax.Array,
    v_len: int,
    *,
    cfgs: configs.RpaConfigs,
):
    """Stitches new kv tokens into the existing cache in VMEM.

    Args:
        vmem_u32_ref: VMEM slice for a single batch item bitcast to uint32, with
            arbitrary outer dimensions and innermost dimension `v_len` (e.g. `[...,
            v_len]`). Reshaped internally to `(-1, num_lanes)` (i.e. `[outer_dim *
            lanes_per_col, num_lanes]`). Contains existing cached KV tokens up to
            `bkv_sz_cache` and newly DMA'd tokens starting at `cache_end +
            new_tok_offset`.
        bkv_sz_cache: Scalar integer (`shape=()`, `int32`), the number of KV tokens
            already present in the current block's cache for this batch item. Defines
            the destination stitch insertion point: column `bkv_sz_cache // num_lanes`
            and lane offset `bkv_sz_cache % num_lanes`.
        cache_end: Scalar integer (`shape=()`, `int32`), the end of the cached token
            region rounded up to `cfgs.kv_token_align` (`align_to(bkv_sz_cache,
            kv_token_align)`). DMA transfers for new KV tokens land immediately after
            this boundary.
        new_tok_offset: Scalar integer (`shape=()`, `int32`), the intra-alignment
            offset (`new_kv_len_start % cfgs.kv_token_align`) of the first new token
            within the DMA transfer region. Together with `cache_end`, `src_tok_idx =
            cache_end + new_tok_offset` gives the exact VMEM token index where newly
            fetched tokens begin.
        bkv_sz_new: Scalar integer (`shape=()`, `int32`), the number of valid new KV
            tokens to be merged into the current block for this sequence (`<= bq_sz`).
            Used in `rel < bkv_sz_new` to mask out padding or uninitialized memory
            beyond the valid new tokens.
        v_len: Total VMEM lane allocation length (`cfgs.kv_vmem_lanes`), a multiple
            of `num_lanes` defining the total capacity of the VMEM buffer per lane.
        cfgs: RPA kernel configuration dataclass.
    """
    num_lanes = pltpu.get_tpu_info().num_lanes
    lanes_per_col = v_len // num_lanes
    strided_vmem_ref = vmem_u32_ref.reshape(-1, num_lanes)
    outer_dim = strided_vmem_ref.shape[0] // lanes_per_col

    # Destination: VREG row (`dst_chunk_idx`) and lane offset (`dst_rel`) for new token insertion.
    dst_chunk_idx = bkv_sz_cache // num_lanes
    dst_rel = bkv_sz_cache % num_lanes

    # Source: first fetched new token in VMEM.
    src_tok_idx = cache_end + new_tok_offset
    src_chunk_idx = src_tok_idx // num_lanes
    src_rel = src_tok_idx % num_lanes

    lane_idx = lax.broadcasted_iota(jnp.int32, (outer_dim, num_lanes), 1)

    # Roll source columns by the lane shift between destination and source.
    shift = lax.rem(
        lax.rem(dst_rel - src_rel, num_lanes) + num_lanes, num_lanes)

    # v_len = bkv_sz + 2 * num_lanes (lanes_per_col = bkv_sz // num_lanes + 2),
    # and bkv_sz_cache <= bkv_sz, so src_chunk_idx <= bkv_sz // num_lanes and
    # dst_chunk_idx <= bkv_sz // num_lanes. Thus src_chunk_idx + 1 and
    # dst_chunk_idx + k (for k in {0, 1}) are always < lanes_per_col.
    src_vreg = strided_vmem_ref[pl.ds(src_chunk_idx, outer_dim, lanes_per_col)]

    # Load and merge the second source column unconditionally when bq_sz > 1 to
    # keep stitch_batch() a single straight-line basic block across all batch
    # slots without per-slot scalar branch barriers.
    if cfgs.block.bq_sz > 1:
        src_next_vreg = strided_vmem_ref[pl.ds(src_chunk_idx + 1, outer_dim,
                                               lanes_per_col)]
        src_vreg = jnp.where(lane_idx >= src_rel, src_vreg, src_next_vreg)

    src_vreg = pltpu.roll(src_vreg, shift, axis=1)

    n_cols = pl.cdiv(num_lanes - 1 + cfgs.block.bq_sz, num_lanes)
    cols = []
    merged = []
    for k in range(n_cols):
        col = dst_chunk_idx + k
        dst_vreg = strided_vmem_ref[pl.ds(col, outer_dim, lanes_per_col)]
        rel = k * num_lanes + lane_idx - dst_rel
        cols.append(col)
        merged.append(
            jnp.where(rel < 0, dst_vreg,
                      jnp.where(rel < bkv_sz_new, src_vreg, 0)))

    return cols, outer_dim, lanes_per_col, merged


def _stitch_prefill_lane(
    vmem_u32_ref: jax.Array,
    bkv_sz_cache: jax.Array,
    cache_end: jax.Array,
    new_tok_offset: jax.Array,
    v_len: int,
    *,
    cfgs: configs.RpaConfigs,
):
    """O(N) Prefill Path: Roll the entire new tokens buffer into place."""
    total_head_words = (cfgs.model.num_kv_heads * 2 *
                        cfgs.aligned_kv_head_dim // cfgs.serve.packing_kv)
    num_sublanes = pltpu.get_tpu_info().num_sublanes
    words_per_sublane = total_head_words // num_sublanes
    vmem_u32_reshaped = vmem_u32_ref.reshape(words_per_sublane, num_sublanes,
                                             v_len)

    roll_shift = (bkv_sz_cache - (cache_end + new_tok_offset)) % v_len
    rolled_u32 = pltpu.roll(vmem_u32_reshaped[...], roll_shift, axis=2)

    lane_idx = jax.lax.broadcasted_iota(jnp.int32,
                                        rolled_u32[..., :cfgs.bkv_sz].shape, 2)
    merged_cache_u32 = jax.lax.select(
        lane_idx >= bkv_sz_cache,
        rolled_u32[..., :cfgs.bkv_sz],
        vmem_u32_reshaped[..., :cfgs.bkv_sz],
    )

    return merged_cache_u32


def store_new_kv_lane(
    vmem_ref: jax.Ref,
    b_idx: int,
    stitch_result,
    *,
    cfgs: configs.RpaConfigs,
):
    """Stores the result of stitch_new_kv_lane back into memory."""
    v_len = cfgs.kv_vmem_lanes
    vmem_u32_ref = vmem_ref.at[b_idx].bitcast(jnp.uint32)

    if (cfgs.mode == configs.RpaCase.DECODE
            and cfgs.block.bq_sz <= pltpu.get_tpu_info().num_lanes):
        cols, outer_dim, lanes_per_col, merged = stitch_result
        num_lanes = pltpu.get_tpu_info().num_lanes
        strided_vmem_ref = vmem_u32_ref.reshape(-1, num_lanes)

        for col, merged_vreg in zip(cols, merged):
            strided_vmem_ref[pl.ds(col, outer_dim,
                                   lanes_per_col)] = merged_vreg

    else:
        merged_cache_u32 = stitch_result
        total_head_words = (cfgs.model.num_kv_heads * 2 *
                            cfgs.aligned_kv_head_dim // cfgs.serve.packing_kv)
        num_sublanes = pltpu.get_tpu_info().num_sublanes
        words_per_sublane = total_head_words // num_sublanes
        vmem_u32_reshaped = vmem_u32_ref.reshape(words_per_sublane,
                                                 num_sublanes, v_len)

        # Store the fully stitched sequence back.
        vmem_u32_reshaped[..., :cfgs.bkv_sz] = merged_cache_u32


def stitch_new_kv_lane(
    vmem_ref: jax.Ref,
    b_idx: int,
    bkv_sz_frm_cache: jax.Array,
    new_kv_len_start: jax.Array,
    bkv_sz_frm_new: jax.Array,
    *,
    cfgs: configs.RpaConfigs,
):
    """Fetches and computes stitched KV tokens (separated to avoid RAW hazards).

    Expects vmem_ref shape: [batch, 2*kv, head_dim / packing, packing,
    cfgs.kv_vmem_lanes]
    """
    new_tok_offset = new_kv_len_start % cfgs.kv_token_align
    cache_end = utils.align_to(bkv_sz_frm_cache, cfgs.kv_token_align)

    vmem_u32_ref = vmem_ref.at[b_idx].bitcast(jnp.uint32)

    # Up to num_lanes new kv tokens in DECODE span at most two 128-lane registers,
    # so they can be rolled in place instead of rolling the entire bkv_sz.
    if (cfgs.mode == configs.RpaCase.DECODE
            and cfgs.block.bq_sz <= pltpu.get_tpu_info().num_lanes):
        return _stitch_decode_lane(
            vmem_u32_ref,
            bkv_sz_frm_cache,
            cache_end,
            new_tok_offset,
            bkv_sz_frm_new,
            cfgs.kv_vmem_lanes,
            cfgs=cfgs,
        )
    else:
        return _stitch_prefill_lane(
            vmem_u32_ref,
            bkv_sz_frm_cache,
            cache_end,
            new_tok_offset,
            cfgs.kv_vmem_lanes,
            cfgs=cfgs,
        )
