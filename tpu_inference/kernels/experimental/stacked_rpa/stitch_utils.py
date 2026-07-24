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
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from tpu_inference.kernels.experimental.stacked_rpa import configs


def _stitch_decode_lane(
    vmem_ref: jax.Ref,
    b_idx: int,
    bkv_sz_cache: jax.Array,
    cache_pages: jax.Array,
    new_tok_offset: jax.Array,
    page_size: int,
):
    """O(1) Decode Path: rewrite only the boundary 128-lane chunk.

    Lane-slices the v_len axis at a 128-aligned offset (tile-provable on the
    collapsed [2*kv, head_dim, v_len] layout, unlike a sublane-tiled strided
    index). The new-KV lands at VMEM token offset cache_pages*page_size (see
    fill_dma_kv_new) with the new token at new_tok_offset within its page; index
    the boundary and new-token chunks in 128-lane units and roll the new token
    onto the boundary lane. bitcast(u32) packs head_dim (sublane), v_len stays
    on the lane.
    """
    num_lanes = pltpu.get_tpu_info().num_lanes
    vmem_u32 = vmem_ref.at[b_idx].bitcast(
        jnp.uint32)  # [2*kv, hd_words, v_len]
    dst_off = (bkv_sz_cache // num_lanes) * num_lanes
    src_off = cache_pages * page_size + (new_tok_offset //
                                         num_lanes) * num_lanes
    dst_rel = bkv_sz_cache % num_lanes
    src_lane = new_tok_offset % num_lanes

    dst = vmem_u32[:, :, pl.ds(dst_off, num_lanes)]
    src = vmem_u32[:, :, pl.ds(src_off, num_lanes)]
    rolled = pltpu.roll(src, dst_rel - src_lane, axis=2)
    lane_idx = jax.lax.broadcasted_iota(jnp.int32, dst.shape, 2)
    merged = jax.lax.select(lane_idx >= dst_rel, rolled, dst)
    return dst_off, merged


def _stitch_prefill_lane(
    vmem_u32_ref: jax.Array,
    bkv_sz_cache: jax.Array,
    cache_pages: jax.Array,
    new_tok_offset: jax.Array,
    v_len: int,
    *,
    cfgs: configs.RpaConfigs,
):
    """O(N) Prefill Path: roll the entire new-tokens buffer into place.

    Native 4D ``[2*kv, head_dim, v_len]`` u32 view (matching _stitch_decode_lane):
    roll along the v_len (lane) axis and select at the cache boundary, with no
    reshape/relayout of the head axes (the old
    ``reshape(words_per_sublane, num_sublanes, v_len)`` forced a relayout).
    """
    roll_shift = (
        bkv_sz_cache -
        (cache_pages * cfgs.serve.page_size + new_tok_offset)) % v_len
    rolled_u32 = pltpu.roll(vmem_u32_ref[...], roll_shift, axis=2)

    lane_idx = jax.lax.broadcasted_iota(jnp.int32,
                                        rolled_u32[..., :cfgs.bkv_sz].shape, 2)
    merged_cache_u32 = jax.lax.select(
        lane_idx >= bkv_sz_cache,
        rolled_u32[..., :cfgs.bkv_sz],
        vmem_u32_ref[..., :cfgs.bkv_sz],
    )

    return merged_cache_u32


# Define inner kernel.
def store_new_kv_lane(
    vmem_ref: jax.Ref,
    b_idx: int,
    stitch_result,
    *,
    cfgs: configs.RpaConfigs,
):
    """Stores the result of stitch_new_kv_lane back into memory."""
    vmem_u32_ref = vmem_ref.at[b_idx].bitcast(jnp.uint32)

    # Must mirror stitch_new_kv_lane's dispatch exactly (they are a matched pair:
    # the store consumes the return shape of the stitch): only single-token
    # decode uses the O(1) path; multi-token decode uses the prefill store.
    if cfgs.mode == configs.RpaCase.DECODE and cfgs.decode_q_len == 1:
        dst_off, merged = stitch_result
        num_lanes = pltpu.get_tpu_info().num_lanes
        # 4D [2*kv, head_dim, v_len] u32 view: store the merged boundary chunk.
        vmem_u32_ref[:, :, pl.ds(dst_off, num_lanes)] = merged

    else:
        merged_cache_u32 = stitch_result
        # 4D [2*kv, head_dim, v_len] u32 view: store the stitched sequence back
        # in the native layout (no reshape/relayout), mirroring the load path.
        vmem_u32_ref[:, :, :cfgs.bkv_sz] = merged_cache_u32


def stitch_new_kv_lane(
    vmem_ref: jax.Ref,
    b_idx: int,
    bkv_sz_frm_cache: jax.Array,
    new_kv_len_start: jax.Array,
    *,
    cfgs: configs.RpaConfigs,
):
    """Fetches and computes stitched KV tokens (separated to avoid RAW hazards).

    Expects vmem_ref shape: [batch, 2*kv, head_dim, bkv_sz + 2 * page_size]
    (native 4D SEQ_ALONG_LANE layout: head_dim on the sublane axis, tokens on
    the lane axis).
    """
    bkv_sz_cache = bkv_sz_frm_cache.astype(jnp.int32)
    new_tok_offset = new_kv_len_start.astype(jnp.int32) % cfgs.serve.page_size
    cache_pages = pl.cdiv(bkv_sz_cache, cfgs.serve.page_size)

    v_len = cfgs.bkv_sz + 2 * cfgs.serve.page_size
    vmem_u32_ref = vmem_ref.at[b_idx].bitcast(jnp.uint32)

    # Single-token decode uses the O(1) boundary-VREG roll. Multi-token decode
    # (spec decode, decode_q_len>1) writes decode_q_len new tokens that can
    # straddle a 128-lane/page boundary, which the O(1) path would drop; route it
    # through the O(N) prefill roll (which handles any new-token count/straddle).
    if cfgs.mode == configs.RpaCase.DECODE and cfgs.decode_q_len == 1:
        return _stitch_decode_lane(
            vmem_ref,
            b_idx,
            bkv_sz_cache,
            cache_pages,
            new_tok_offset,
            cfgs.serve.page_size,
        )
    else:
        return _stitch_prefill_lane(
            vmem_u32_ref,
            bkv_sz_cache,
            cache_pages,
            new_tok_offset,
            v_len,
            cfgs=cfgs,
        )
