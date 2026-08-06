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
"""Top-k expert selection for the fused expert-parallel MoE router:
pallas_select is an in-VMEM max-and-mask pass replacing XLA's lax.top_k.

An all-NaN score row selects the lowest expert on every slot, with NEG as
the weight of each: NaN is flushed to the sentinel before the first max, so
every row has a maximum some column equals and every index returned is a
real expert. What that row's weights MEAN is not decided here. The kernel
layer masks such a row explicitly, on jnp.any(jnp.isfinite(scores)), rather
than relying on the sentinels to sum to something the renormalization turns
into zero -- that outcome was a property of the accumulation dtype and the
association order rather than of the design.
"""
import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl

NEG = -3.0e38  # below any real softmax score (>=0); avoids inf


def _select_kernel(x_ref, w_ref, i_ref, *, topk, n):
    """One block of rows: take the top `topk` by repeated max-and-mask."""
    # NaN is flushed to the sentinel so every row has a maximum some
    # column equals; an all-NaN row selects expert 0 with weight NEG.
    x = jnp.where(jnp.isnan(x_ref[...]), NEG, x_ref[...])  # [R, n] f32
    iota = jax.lax.broadcasted_iota(jnp.int32, x.shape, 1)
    for j in range(topk):
        m = jnp.max(x, axis=1, keepdims=True)  # [R,1] rowmax
        ismax = x == m
        # Lowest matching column; non-matching columns fill with the last
        # expert so every index returned is a real expert.
        idx = jnp.min(jnp.where(ismax, iota, n - 1), axis=1)
        w_ref[:, j] = m[:, 0]
        i_ref[:, j] = idx.astype(jnp.int32)
        x = jnp.where(iota == idx[:, None], NEG, x)


def pallas_select(scores, topk=10, block_rows=256):
    """Top-k over [R, N] scores -> (weights f32, indices i32) [R, topk]."""
    R, N = scores.shape
    assert R % block_rows == 0, (R, block_rows)
    grid = (R // block_rows, )
    out_shapes = (
        jax.ShapeDtypeStruct((R, topk), jnp.float32),
        jax.ShapeDtypeStruct((R, topk), jnp.int32),
    )
    bs = (pl.BlockSpec((block_rows, N), lambda i: (i, 0)), )
    os = (
        pl.BlockSpec((block_rows, topk), lambda i: (i, 0)),
        pl.BlockSpec((block_rows, topk), lambda i: (i, 0)),
    )
    return pl.pallas_call(
        functools.partial(_select_kernel, topk=topk, n=N),
        grid=grid,
        in_specs=list(bs),
        out_specs=list(os),
        out_shape=list(out_shapes),
    )(scores)
