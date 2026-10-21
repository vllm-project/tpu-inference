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

from tpu_inference.kernels.experimental.batched_rpa import configs


def _stitch_decode_lane(
    vmem_u32_ref: jax.Array,
    bkv_sz_cache: jax.Array,
    cache_pages: jax.Array,
    new_tok_offset: jax.Array,
    v_len: int,
):
    """O(1) Decode Path: Target exactly the VREG containing the stitch boundary."""
    num_lanes = pltpu.get_tpu_info().num_lanes
    lanes_per_col = v_len // num_lanes
    strided_vmem_ref = vmem_u32_ref.reshape(-1, num_lanes)
    outer_dim = strided_vmem_ref.shape[0] // lanes_per_col

    dst_chunk_idx = bkv_sz_cache // num_lanes

    # Load just the destination and source chunks.
    dst_vreg = strided_vmem_ref[pl.ds(dst_chunk_idx, outer_dim, lanes_per_col)]
    src_vreg = strided_vmem_ref[pl.ds(cache_pages, outer_dim, lanes_per_col)]

    dst_rel = bkv_sz_cache % num_lanes
    rolled_src_vreg = pltpu.roll(src_vreg, dst_rel - new_tok_offset, axis=1)

    lane_idx = jax.lax.broadcasted_iota(jnp.int32, dst_vreg.shape, 1)
    merged_dst_vreg = jax.lax.select(lane_idx >= dst_rel, rolled_src_vreg,
                                     dst_vreg)

    return dst_chunk_idx, outer_dim, lanes_per_col, merged_dst_vreg


def _stitch_prefill_lane(
    vmem_u32_ref: jax.Array,
    bkv_sz_cache: jax.Array,
    cache_pages: jax.Array,
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

    roll_shift = (
        bkv_sz_cache -
        (cache_pages * cfgs.serve.page_size + new_tok_offset)) % v_len
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
    v_len = cfgs.bkv_sz + 2 * cfgs.serve.page_size
    vmem_u32_ref = vmem_ref.at[b_idx].bitcast(jnp.uint32)

    if cfgs.block.bq_sz == 1:
        dst_chunk_idx, outer_dim, lanes_per_col, merged_dst_vreg = stitch_result
        num_lanes = pltpu.get_tpu_info().num_lanes
        strided_vmem_ref = vmem_u32_ref.reshape(-1, num_lanes)

        # Store the merged chunk directly back into memory.
        strided_vmem_ref[pl.ds(dst_chunk_idx, outer_dim,
                               lanes_per_col)] = (merged_dst_vreg)

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
    *,
    cfgs: configs.RpaConfigs,
):
    """Fetches and computes stitched KV tokens (separated to avoid RAW hazards).

    Expects vmem_ref shape: [batch, 2*kv, head_dim / packing, packing, bkv_sz + 2
    * page_size]
    """
    bkv_sz_cache = bkv_sz_frm_cache.astype(jnp.int32)
    new_tok_offset = (new_kv_len_start.astype(jnp.int32) %
                      cfgs.serve.page_size)
    cache_pages = pl.cdiv(bkv_sz_cache, cfgs.serve.page_size)

    v_len = cfgs.bkv_sz + 2 * cfgs.serve.page_size
    vmem_u32_ref = vmem_ref.at[b_idx].bitcast(jnp.uint32)

    # If bq_sz == 1, there is only 1 kv token from new, so we only need one 128
    # sized register to be rolled (compared to rolling the entire bkv_sz).
    if cfgs.block.bq_sz == 1:
        return _stitch_decode_lane(vmem_u32_ref, bkv_sz_cache, cache_pages,
                                   new_tok_offset, v_len)
    else:
        return _stitch_prefill_lane(
            vmem_u32_ref,
            bkv_sz_cache,
            cache_pages,
            new_tok_offset,
            v_len,
            cfgs=cfgs,
        )
