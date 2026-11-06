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
"""Fused MoE router scoring and group-limited top-k selection.

Provides `rowmax_topk` (pure-JAX fallback / CPU reference) and `router_topk`
(Pallas kernel with CPU fallback).
"""

import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl

from tpu_inference.logger import init_logger

logger = init_logger(__name__)

# Clamp floor. Below any real routing score (softmax, sigmoid and sqrtsoftplus
# are all >= 0) and strictly above the dtype minimum used to mask a selected
# column, so a row's maximum is always above the mask.
NEG = -3.0e38

# Rows per grid step, measured fastest at 512 experts. The grid rounds up, so
# this does not have to divide the row count.
MAX_BLOCK_ROWS = 512


def _select_kernel(x_ref, w_ref, i_ref, *, topk: int, n: int):
    """One block of rows: top ``topk`` by repeated max-and-mask.

    Scores at or below ``NEG`` are clamped up to it first, so every row has a
    maximum some column equals and every index written is a real expert. Ties
    go to the lowest id.

    The clamp tests ``x > NEG`` rather than ``isnan(x)`` to catch all three
    spellings of the hazard -- NaN, ``-inf``, ``-FLT_MAX`` -- any of which a
    whole row can be made of. Without it such a row would stay tied with the
    ``finfo.min`` mask and every pass would return column 0.
    """
    x0 = jnp.where(x_ref[...] > NEG, x_ref[...], NEG)
    masked = jnp.finfo(x0.dtype).min
    iota = jax.lax.broadcasted_iota(jnp.int32, x0.shape, 1)
    x = x0
    if x.shape[0] <= 64:
        # One index reduction per pass on the serial chain: argmax returns the
        # lowest column attaining the max. The pass's value is the max of the
        # same scores, a reduction off that chain.
        for j in range(topk):
            idx = jnp.argmax(x, axis=1).astype(jnp.int32)
            w_ref[:, j] = jnp.max(x, axis=1)
            i_ref[:, j] = idx
            x = jnp.where(iota == idx[:, None], masked, x)
    else:
        # At >=128 rows, Mosaic's variadic argmax reduction spills VREGs, so
        # two scalar reductions (max then min-index) are 2x faster.
        for j in range(topk):
            m = jnp.max(x, axis=1, keepdims=True)
            idx = jnp.min(jnp.where(x == m, iota, n - 1), axis=1)
            w_ref[:, j] = m[:, 0]
            i_ref[:, j] = idx.astype(jnp.int32)
            x = jnp.where(iota == idx[:, None], masked, x)
    # No score above the sentinel means no real maximum: keep the row
    # poisonous rather than letting renormalization make it plausible.
    w_ref[...] = jnp.where(w_ref[:, :1] <= NEG, jnp.nan, w_ref[...])


def select(scores: jax.Array,
           topk: int,
           interpret: bool = False) -> tuple[jax.Array, jax.Array]:
    """Top-k over ``[rows, experts]`` f32 scores, values descending.

    Returns ``(weights f32, indices int32)`` of shape ``[rows, topk]``, values
    descending. ``interpret`` runs the kernel on the host, for tests without a
    TPU.

    The last grid step reads a partial block when ``rows`` is not a multiple
    of the block; Mosaic discards the stores past ``rows``, and every
    reduction is along the expert axis, so surplus rows cannot perturb a real
    one.
    """
    if scores.ndim != 2:
        raise ValueError(
            f"router top-k kernel expects [rows, experts], got {scores.shape}")
    rows, n = scores.shape
    block = min(MAX_BLOCK_ROWS, rows)
    return pl.pallas_call(
        functools.partial(_select_kernel, topk=topk, n=n),
        grid=(pl.cdiv(rows, block), ),
        in_specs=[pl.BlockSpec((block, n), lambda i: (i, 0))],
        out_specs=[
            pl.BlockSpec((block, topk), lambda i: (i, 0)),
            pl.BlockSpec((block, topk), lambda i: (i, 0)),
        ],
        out_shape=[
            jax.ShapeDtypeStruct((rows, topk), jnp.float32),
            jax.ShapeDtypeStruct((rows, topk), jnp.int32),
        ],
        interpret=interpret,
    )(scores)


def rowmax_topk(scores: jax.Array, k: int) -> tuple[jax.Array, jax.Array]:
    """Sort-free top-k for MoE routing in pure JAX.

    The same algorithm as the Pallas kernel in ``select``: ``k`` passes of
    (row max -> lowest matching column -> mask that column out). Used off-TPU,
    and as the numerics oracle the kernel tests compare against.

    Returns ``(values, indices)``, values descending and indices int32. Ties
    resolve to the lowest expert id.
    """
    if scores.ndim != 2:
        raise ValueError(
            f"rowmax_topk expects [rows, experts], got {scores.shape}")
    n = scores.shape[-1]
    if k > n:
        raise ValueError(f"topk k={k} exceeds the expert count {n}")
    iota = jax.lax.broadcasted_iota(jnp.int32, scores.shape, 1)
    masked = jnp.finfo(scores.dtype).min

    cur = jnp.where(scores > NEG, scores, NEG)
    values, indices = [], []
    for _ in range(k):
        m = jnp.max(cur, axis=-1, keepdims=True)
        idx = jnp.min(jnp.where(cur == m, iota, n - 1), axis=-1, keepdims=True)
        values.append(m)
        indices.append(idx)
        cur = jnp.where(iota == idx, masked, cur)

    weights = jnp.concatenate(values, axis=-1)
    ids = jnp.concatenate(indices, axis=-1).astype(jnp.int32)
    # No score above the sentinel means no real maximum: keep the row
    # poisonous so the caller's renormalization propagates it.
    weights = jnp.where(weights[..., :1] <= NEG, jnp.nan, weights)
    return weights.astype(scores.dtype), ids


def router_topk(scores: jax.Array, k: int) -> tuple[jax.Array, jax.Array]:
    """Sort-free router top-k: k row-max passes instead of a sort.

    Pure JAX (`rowmax_topk`), not the Pallas `select` kernel: select_experts
    calls this on the gating scores outside any shard_map, under auto-SPMD
    partitioning, where a Mosaic kernel cannot be partitioned. `select`
    remains for callers that run it on a per-shard array.
    """
    return rowmax_topk(scores, k)
