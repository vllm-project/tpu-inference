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
"""KV stitching used by the prefill and mixed kernel."""

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from tpu_inference.kernels.experimental.stacked_rpa.prefill import config


def _stitch_lane(
    vmem_u32_ref: jax.Array,
    bkv_sz_cache: jax.Array,
    cache_pages: jax.Array,
    new_tok_offset: jax.Array,
    v_len: int,
    *,
    cfgs: config.PrefillConfig,
):
    """Roll the full new-token buffer into place after cached tokens."""
    roll_shift = (
        bkv_sz_cache -
        (cache_pages * cfgs.serve.page_size + new_tok_offset)) % v_len
    rolled_u32 = pltpu.roll(vmem_u32_ref[...], roll_shift, axis=2)
    lane_idx = jax.lax.broadcasted_iota(jnp.int32,
                                        rolled_u32[..., :cfgs.bkv_sz].shape, 2)
    return jax.lax.select(
        lane_idx >= bkv_sz_cache,
        rolled_u32[..., :cfgs.bkv_sz],
        vmem_u32_ref[..., :cfgs.bkv_sz],
    )


def stitch_kv_lane(
    vmem_ref: jax.Ref,
    b_idx: int,
    bkv_sz_frm_cache: jax.Array,
    new_kv_len_start: jax.Array,
    *,
    cfgs: config.PrefillConfig,
):
    """Compute the full-roll stitch for one prefill or mixed lane."""
    bkv_sz_cache = bkv_sz_frm_cache.astype(jnp.int32)
    new_tok_offset = new_kv_len_start.astype(jnp.int32) % cfgs.serve.page_size
    cache_pages = pl.cdiv(bkv_sz_cache, cfgs.serve.page_size)
    v_len = cfgs.bkv_sz + 2 * cfgs.serve.page_size
    vmem_u32_ref = vmem_ref.at[b_idx].bitcast(jnp.uint32)
    return _stitch_lane(
        vmem_u32_ref,
        bkv_sz_cache,
        cache_pages,
        new_tok_offset,
        v_len,
        cfgs=cfgs,
    )


def store_kv_lane(
    vmem_ref: jax.Ref,
    b_idx: int,
    stitch_result,
    *,
    cfgs: config.PrefillConfig,
):
    """Store one full-roll prefill or mixed stitch."""
    vmem_u32_ref = vmem_ref.at[b_idx].bitcast(jnp.uint32)
    vmem_u32_ref[:, :, :cfgs.bkv_sz] = stitch_result
