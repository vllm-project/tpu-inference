# Copyright (c) Meta Platforms, Inc. and affiliates.
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
"""Fixed-Q KV stitching used by both decode implementations."""

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from tpu_inference.kernels.experimental.stacked_rpa import utils
from tpu_inference.kernels.experimental.stacked_rpa.decode import config


def _stitch_bounded(
    vmem_ref: jax.Ref,
    b_idx: int,
    cache_size: jax.Array,
    cache_pages: jax.Array,
    new_token_offset: jax.Array,
    page_size: int,
    decode_q_len: int,
    new_kv_vref: jax.Ref | None = None,
):
    num_lanes = pltpu.get_tpu_info().num_lanes
    stitch_span = utils.align_to(num_lanes - 1 + decode_q_len, num_lanes)
    vmem_u32 = vmem_ref.at[b_idx].bitcast(jnp.uint32)
    dst_offset = pl.multiple_of(
        (cache_size // num_lanes) * num_lanes,
        num_lanes,
    )
    src_offset = pl.multiple_of(
        cache_pages * page_size + (new_token_offset // num_lanes) * num_lanes,
        num_lanes,
    )
    dst_lane = cache_size % num_lanes
    src_lane = new_token_offset % num_lanes

    destination = vmem_u32[:, :, pl.ds(dst_offset, stitch_span)]
    if new_kv_vref is None:
        source = vmem_u32[:, :, pl.ds(src_offset, stitch_span)]
    else:
        source_u32 = new_kv_vref.bitcast(jnp.uint32)
        resident_offset = pl.multiple_of(
            (new_token_offset // num_lanes) * num_lanes,
            num_lanes,
        )
        source = source_u32[:, :, pl.ds(resident_offset, stitch_span)]
    rolled = pltpu.roll(source, dst_lane - src_lane, axis=2)
    lane = jax.lax.broadcasted_iota(jnp.int32, destination.shape, 2)
    use_new = lane >= dst_lane
    if decode_q_len > 1:
        use_new = jnp.logical_and(use_new, lane < dst_lane + decode_q_len)
    return dst_offset, jax.lax.select(use_new, rolled, destination)


def _stitch_large_q(
    vmem_ref: jax.Ref,
    b_idx: int,
    cache_size: jax.Array,
    cache_pages: jax.Array,
    new_token_offset: jax.Array,
    *,
    cfgs: config.DecodeConfig,
):
    vmem_u32 = vmem_ref.at[b_idx].bitcast(jnp.uint32)
    vmem_size = cfgs.bkv_sz + 2 * cfgs.serve.page_size
    roll_shift = (
        cache_size -
        (cache_pages * cfgs.serve.page_size + new_token_offset)) % vmem_size
    rolled = pltpu.roll(vmem_u32[...], roll_shift, axis=2)
    lane = jax.lax.broadcasted_iota(
        jnp.int32,
        rolled[..., :cfgs.bkv_sz].shape,
        2,
    )
    return jax.lax.select(
        lane >= cache_size,
        rolled[..., :cfgs.bkv_sz],
        vmem_u32[..., :cfgs.bkv_sz],
    )


def stitch_kv_lane(
    vmem_ref: jax.Ref,
    b_idx: int,
    cache_size: jax.Array,
    new_kv_start: jax.Array,
    *,
    cfgs: config.DecodeConfig,
    new_kv_vref: jax.Ref | None = None,
):
    """Merge a sequence's fixed-size new-KV suffix into its cache tile."""
    cache_size = cache_size.astype(jnp.int32)
    new_token_offset = new_kv_start.astype(jnp.int32) % cfgs.serve.page_size
    cache_pages = pl.cdiv(cache_size, cfgs.serve.page_size)
    if cfgs.decode_q_len > cfgs.serve.page_size:
        return _stitch_large_q(
            vmem_ref,
            b_idx,
            cache_size,
            cache_pages,
            new_token_offset,
            cfgs=cfgs,
        )
    return _stitch_bounded(
        vmem_ref,
        b_idx,
        cache_size,
        cache_pages,
        new_token_offset,
        cfgs.serve.page_size,
        cfgs.decode_q_len,
        new_kv_vref if cfgs.new_kv_resident else None,
    )


def store_kv_lane(
    vmem_ref: jax.Ref,
    b_idx: int,
    stitch_result,
    *,
    cfgs: config.DecodeConfig,
) -> None:
    """Store the bounded stitch, or the full-tile large-Q fallback."""
    vmem_u32 = vmem_ref.at[b_idx].bitcast(jnp.uint32)
    if cfgs.decode_q_len > cfgs.serve.page_size:
        vmem_u32[:, :, :cfgs.bkv_sz] = stitch_result
        return

    dst_offset, merged = stitch_result
    num_lanes = pltpu.get_tpu_info().num_lanes
    stitch_span = utils.align_to(num_lanes - 1 + cfgs.decode_q_len, num_lanes)
    vmem_u32[:, :, pl.ds(dst_offset, stitch_span)] = merged
