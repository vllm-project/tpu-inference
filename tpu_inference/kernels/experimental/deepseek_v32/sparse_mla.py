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
"""Decode-only sparse MLA over selected DeepSeek-V3.2/GLM latent rows."""

import functools

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

_MASK_VALUE = -0.7 * float(jnp.finfo(jnp.float32).max)


def _online_update(m_prev, l_prev, acc_prev, scores, values, precision):
    m_curr = jnp.max(scores, axis=1, keepdims=True)
    m_new = jnp.maximum(m_prev, m_curr)
    correction = jnp.exp(m_prev - m_new)
    probabilities = jnp.exp(scores - m_new)
    l_new = l_prev * correction + jnp.sum(probabilities, axis=1, keepdims=True)
    pv = lax.dot_general(
        probabilities.astype(values.dtype),
        values,
        (((1, ), (0, )), ((), ())),
        preferred_element_type=jnp.float32,
        precision=precision,
    )
    return m_new, l_new, acc_prev * correction + pv


def _finalize(m, denominator, acc):
    # A fully masked row never raises m above _MASK_VALUE. Select it to zero
    # explicitly; otherwise masked garbage would be averaged into the result.
    normalized = acc / (denominator + jnp.exp(_MASK_VALUE - m))
    return jnp.where(m > _MASK_VALUE, normalized, 0.0)


def _sparse_decode_kernel(
    selected_lens_ref,
    query_ref,
    selected_kv_ref,
    output_ref,
    m_scratch,
    l_scratch,
    acc_scratch,
    *,
    sm_scale,
    num_blocks,
    block_size,
    latent_dim,
    precision,
):
    block = pl.program_id(1)

    @pl.when(block == 0)
    def initialize():
        m_scratch[...] = jnp.full(m_scratch.shape, _MASK_VALUE, jnp.float32)
        l_scratch[...] = jnp.zeros(l_scratch.shape, jnp.float32)
        acc_scratch[...] = jnp.zeros(acc_scratch.shape, jnp.float32)

    query = query_ref[0]
    kv = selected_kv_ref[0, 0]
    selected_len = selected_lens_ref[pl.program_id(0)]
    offsets = block * block_size + lax.broadcasted_iota(
        jnp.int32, (1, block_size), 1)
    scores = lax.dot_general(
        query,
        kv,
        (((1, ), (1, )), ((), ())),
        preferred_element_type=jnp.float32,
        precision=precision,
    ) * sm_scale
    scores = jnp.where(offsets < selected_len, scores, _MASK_VALUE)
    m, denominator, acc = _online_update(
        m_scratch[...],
        l_scratch[...],
        acc_scratch[...],
        scores,
        kv[:, :latent_dim],
        precision,
    )
    m_scratch[...] = m
    l_scratch[...] = denominator
    acc_scratch[...] = acc

    @pl.when(block == num_blocks - 1)
    def store():
        output_ref[0] = _finalize(m, denominator, acc).astype(output_ref.dtype)


@functools.partial(
    jax.jit,
    static_argnames=("sm_scale", "block_size", "vmem_limit_bytes",
                     "interpret"),
)
def sparse_mla_decode(
    q_nope: jax.Array,
    q_pe: jax.Array,
    selected_kv: jax.Array,
    selected_lens: jax.Array,
    *,
    sm_scale: float,
    block_size: int = 512,
    vmem_limit_bytes: int | None = None,
    interpret: bool = False,
) -> jax.Array:
    """Attend to a fixed selected segment and return its normalized latent.

    ``q_nope`` is ``[batch, heads, latent_dim]`` and ``q_pe`` is the roped
    positional query. ``selected_kv`` stores ``[latent | rope | padding]``.
    ``selected_lens`` masks the tail, including the ``-1`` top-k sentinel
    entries gathered by :func:`gather_paged_kv`.
    """
    if q_nope.ndim != 3 or q_pe.ndim != 3 or selected_kv.ndim != 3:
        raise ValueError(
            "q_nope, q_pe, and selected_kv must all be rank-3 arrays")
    batch, num_heads, latent_dim = q_nope.shape
    rope_dim = q_pe.shape[-1]
    selected_len, kv_width = selected_kv.shape[1:]
    if q_pe.shape != (batch, num_heads, rope_dim):
        raise ValueError(f"q_pe has incompatible shape {q_pe.shape}")
    if selected_kv.shape != (batch, selected_len, kv_width):
        raise ValueError(
            f"selected_kv has incompatible shape {selected_kv.shape}")
    if selected_lens.shape != (batch, ):
        raise ValueError(
            f"selected_lens has incompatible shape {selected_lens.shape}")
    if latent_dim % 128:
        raise ValueError(
            f"latent_dim must be a multiple of 128, got {latent_dim}")
    if latent_dim + rope_dim > kv_width:
        raise ValueError((latent_dim, rope_dim, kv_width))
    if q_nope.dtype != q_pe.dtype or q_nope.dtype != selected_kv.dtype:
        raise ValueError((q_nope.dtype, q_pe.dtype, selected_kv.dtype))
    if block_size <= 0 or block_size % 128:
        raise ValueError(
            f"block_size must be a positive multiple of 128, got {block_size}")
    if selected_len == 0:
        raise ValueError("selected_kv must contain at least one row")

    padding = kv_width - latent_dim - rope_dim
    query = jnp.concatenate(
        (
            q_nope,
            q_pe,
            jnp.zeros((batch, num_heads, padding), q_nope.dtype),
        ),
        axis=-1,
    )
    effective_block = min(block_size, selected_len)
    if not interpret and effective_block % 128:
        raise ValueError(
            f"effective block size {effective_block} must be a multiple of 128"
        )
    num_blocks = (selected_len + effective_block - 1) // effective_block
    tail = num_blocks * effective_block - selected_len
    if tail:
        selected_kv = jnp.pad(selected_kv, ((0, 0), (0, tail), (0, 0)))
    selected_kv = selected_kv.reshape(batch, num_blocks, effective_block,
                                      kv_width)
    precision = (lax.Precision.HIGHEST
                 if q_nope.dtype == jnp.float32 else lax.Precision.DEFAULT)

    def query_map(row, block, selected_lens):
        del block, selected_lens
        return (row, 0, 0)

    def kv_map(row, block, selected_lens):
        del selected_lens
        return (row, block, 0, 0)

    kernel = functools.partial(
        _sparse_decode_kernel,
        sm_scale=sm_scale,
        num_blocks=num_blocks,
        block_size=effective_block,
        latent_dim=latent_dim,
        precision=precision,
    )
    call = pl.pallas_call(
        kernel,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=1,
            grid=(batch, num_blocks),
            in_specs=(
                pl.BlockSpec((1, num_heads, kv_width), query_map),
                pl.BlockSpec((1, 1, effective_block, kv_width), kv_map),
            ),
            out_specs=pl.BlockSpec((1, num_heads, latent_dim), query_map),
            scratch_shapes=(
                pltpu.VMEM((num_heads, 1), jnp.float32),
                pltpu.VMEM((num_heads, 1), jnp.float32),
                pltpu.VMEM((num_heads, latent_dim), jnp.float32),
            ),
        ),
        out_shape=jax.ShapeDtypeStruct((batch, num_heads, latent_dim),
                                       q_nope.dtype),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("arbitrary", "arbitrary"),
            vmem_limit_bytes=vmem_limit_bytes,
        ),
        interpret=interpret,
    )
    return call(selected_lens.astype(jnp.int32), query, selected_kv)


def sparse_mla_reference(
    q_nope: jax.Array,
    q_pe: jax.Array,
    selected_kv: jax.Array,
    selected_lens: jax.Array,
    *,
    sm_scale: float,
) -> jax.Array:
    """Pure-JAX reference for :func:`sparse_mla_decode`."""
    latent_dim = q_nope.shape[-1]
    rope_dim = q_pe.shape[-1]
    precision = (lax.Precision.HIGHEST
                 if q_nope.dtype == jnp.float32 else lax.Precision.DEFAULT)
    scores = jnp.einsum(
        "bhd,bnd->bhn",
        q_nope,
        selected_kv[..., :latent_dim],
        preferred_element_type=jnp.float32,
        precision=precision,
    )
    scores += jnp.einsum(
        "bhd,bnd->bhn",
        q_pe,
        selected_kv[..., latent_dim:latent_dim + rope_dim],
        preferred_element_type=jnp.float32,
        precision=precision,
    )
    scores *= sm_scale
    offsets = jnp.arange(selected_kv.shape[1], dtype=jnp.int32)
    valid = offsets[None, None, :] < selected_lens[:, None, None]
    scores = jnp.where(valid, scores, _MASK_VALUE)
    maximum = jnp.max(scores, axis=-1, keepdims=True)
    probabilities = jnp.exp(scores - maximum)
    denominator = jnp.sum(probabilities, axis=-1, keepdims=True)
    output = jnp.einsum(
        "bhn,bnd->bhd",
        probabilities.astype(selected_kv.dtype),
        selected_kv[..., :latent_dim],
        preferred_element_type=jnp.float32,
        precision=precision,
    ) / denominator
    output = jnp.where(maximum > _MASK_VALUE, output, 0.0)
    return output.astype(q_nope.dtype)


def gather_paged_kv(
    cache: jax.Array,
    block_tables: jax.Array,
    topk_indices: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Gather absolute top-k positions from a token-major paged latent cache."""
    if cache.ndim not in (3, 4):
        raise ValueError(f"cache must be rank 3 or 4, got shape {cache.shape}")
    if block_tables.ndim != 2 or topk_indices.ndim != 2:
        raise ValueError((block_tables.shape, topk_indices.shape))
    num_pages = cache.shape[0]
    kv_width = cache.shape[-1]
    page_size = 1
    for size in cache.shape[1:-1]:
        page_size *= size
    batch, topk = topk_indices.shape
    if block_tables.shape[0] != batch:
        raise ValueError((block_tables.shape, topk_indices.shape))
    indices = topk_indices.astype(jnp.int32)
    valid = indices >= 0
    selected_lens = jnp.sum(valid, axis=-1).astype(jnp.int32)
    safe = jnp.where(valid, indices, 0)
    pages = jnp.take_along_axis(block_tables.astype(jnp.int32),
                                safe // page_size,
                                axis=1)
    rows = pages * page_size + safe % page_size
    flat = cache.reshape(num_pages * page_size, kv_width)
    gathered = jnp.take(flat, rows.reshape(-1), axis=0)
    return gathered.reshape(batch, topk, kv_width), selected_lens
