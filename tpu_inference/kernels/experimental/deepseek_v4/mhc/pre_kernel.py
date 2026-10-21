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

import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from tpu_inference.kernels.experimental.deepseek_v4.mhc import utils

# bf16 tiles are (16, 128); keep token blocks sublane-aligned.
_SUBLANE = 16

# Explicit scoped-VMEM budget, repo convention (mla/v1, deepseek_v4 and
# fused_moe use the same constant).
DEFAULT_VMEM_LIMIT_BYTES = 100 * 1024 * 1024


def _round_up(x: int, multiple: int) -> int:
    return (x + multiple - 1) // multiple * multiple


def _split_fn3(fn: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
    """3-chunk bf16 split of fn (8 mantissa bits each = f32's 24).

    Chunks MUST be built with reduce_precision, not dtype round-trips:
    XLA's excess-precision simplification folds f32->bf16->f32 into the
    identity, which silently zeroes the mid/lo chunks.
    """
    fn_hi = jax.lax.reduce_precision(fn, 8, 7)
    rem = fn - fn_hi
    fn_mid = jax.lax.reduce_precision(rem, 8, 7)
    fn_lo = jax.lax.reduce_precision(rem - fn_mid, 8, 7)
    return tuple(t.astype(jnp.bfloat16) for t in (fn_hi, fn_mid, fn_lo))


def _dot3(x, fn_hi_ref, fn_mid_ref, fn_lo_ref):
    """3-pass exact GEMM: x is bf16 (its low chunks are zero), so three
    bf16 passes against the resident fn chunks reproduce HIGHEST. The
    dimension numbers contract x dim 1 with fn dim 1 (i.e. x @ fn.T) so
    fn needs no transpose on the host side."""
    dn = (((1, ), (1, )), ((), ()))
    acc = jax.lax.dot_general(x,
                              fn_hi_ref[...],
                              dn,
                              preferred_element_type=jnp.float32)
    acc = acc + jax.lax.dot_general(
        x, fn_mid_ref[...], dn, preferred_element_type=jnp.float32)
    acc = acc + jax.lax.dot_general(
        x, fn_lo_ref[...], dn, preferred_element_type=jnp.float32)
    return acc


def _mixes_kernel(x_ref, fn_hi_ref, fn_mid_ref, fn_lo_ref, mixes_ref,
                  sqrsum_ref):
    """One token block: 3-pass exact GEMM against resident fn chunks."""
    x = x_ref[...]
    mixes_ref[...] = _dot3(x, fn_hi_ref, fn_mid_ref, fn_lo_ref)
    xf = x.astype(jnp.float32)  # f32 needed only for the squared sum
    sqrsum_ref[...] = jnp.sum(xf * xf, axis=-1, keepdims=True)


def _mixes_collapse_kernel(x_ref, fn_hi_ref, fn_mid_ref, fn_lo_ref, sc_ref,
                           hb_ref, mixes_ref, sqrsum_ref, layer_ref, *,
                           hc_mult, hidden_size, rms_eps, hc_pre_eps):
    """Mixes kernel plus the in-block collapse.

    The collapse's weights are ``pre_mix`` — the sigmoid read gates,
    which need no Sinkhorn and are per-token, so they are computable
    entirely in-block. The Sinkhorn/post/comb gates stay in
    XLA on the raw ``mixes`` output.
    """
    m, h = hc_mult, hidden_size
    x = x_ref[...]
    acc = _dot3(x, fn_hi_ref, fn_mid_ref, fn_lo_ref)
    mixes_ref[...] = acc
    xf = x.astype(jnp.float32)
    sqr = jnp.sum(xf * xf, axis=-1, keepdims=True)
    sqrsum_ref[...] = sqr
    # Same op order as utils.mhc_pre_gates for bit-level agreement.
    scaled = acc[:, :m] * jax.lax.rsqrt(sqr / (m * h) + rms_eps)
    pre_mix = jax.nn.sigmoid(scaled * sc_ref[0, 0] + hb_ref[:, :m]) + \
        hc_pre_eps
    lay = pre_mix[:, 0:1] * xf[:, :h]
    for i in range(1, m):
        lay = lay + pre_mix[:, i:i + 1] * xf[:, i * h:(i + 1) * h]
    layer_ref[...] = lay.astype(jnp.bfloat16)


@functools.partial(jax.jit,
                   static_argnames=("token_block_size", "vmem_limit_bytes"))
def mhc_pre_mixes(
    x2d: jax.Array,
    fn: jax.Array,
    *,
    token_block_size: int = 64,
    vmem_limit_bytes: int = DEFAULT_VMEM_LIMIT_BYTES,
) -> tuple[jax.Array, jax.Array]:
    """Pallas replacement for the mix GEMM + squared sum of mhc_pre.

    Args:
        x2d: (num_tokens, hc_mult * hidden_size), bfloat16.
        fn: (hc_mult3, hc_mult * hidden_size), float32.
        token_block_size: tokens per grid step. The default (64) keeps the
            worst-case VMEM footprint (bf16 block + f32 cast + resident fn)
            around 14 MB at DeepSeek-V4 shapes.

    Returns:
        mixes: (num_tokens, hc_mult3), float32.
        sqrsum: (num_tokens, 1), float32.
    """
    assert x2d.dtype == jnp.bfloat16, x2d.dtype
    assert fn.dtype == jnp.float32, fn.dtype
    num_tokens, hc_hidden = x2d.shape
    hc_mult3 = fn.shape[0]
    assert fn.shape == (hc_mult3, hc_hidden), (fn.shape, x2d.shape)

    def _vmem_need(tb: int) -> int:
        # Double-buffered bf16 x block, transient f32 copy for the squared
        # sum, resident bf16 fn chunks; x2 headroom for spills/scratch
        # (quantized_matmul-style safety factor).
        return 2 * (tb * hc_hidden *
                    (2 * 2 + 4) + 3 * hc_mult3 * hc_hidden * 2)

    tb = min(token_block_size, _round_up(num_tokens, _SUBLANE))
    while tb > _SUBLANE and _vmem_need(tb) > vmem_limit_bytes:
        # Degrade to a smaller block instead of a compile-time VMEM OOM.
        tb //= 2
    padded_tokens = _round_up(num_tokens, tb)
    if padded_tokens != num_tokens:
        # Zero rows are safe: they produce zero mixes/sqrsum and are
        # sliced off below.
        x2d = jnp.pad(x2d, ((0, padded_tokens - num_tokens), (0, 0)))

    # Split in XLA outside the kernel; the chunks stay VMEM-resident.
    fn_hi, fn_mid, fn_lo = _split_fn3(fn)

    compiler_params = pltpu.CompilerParams(dimension_semantics=("parallel", ),
                                           vmem_limit_bytes=vmem_limit_bytes)

    resident = pl.BlockSpec((hc_mult3, hc_hidden), lambda i: (0, 0))
    mixes, sqrsum = pl.pallas_call(
        _mixes_kernel,
        grid=(padded_tokens // tb, ),
        in_specs=[
            pl.BlockSpec((tb, hc_hidden), lambda i: (i, 0)),
            resident,
            resident,
            resident,
        ],
        out_specs=[
            pl.BlockSpec((tb, hc_mult3), lambda i: (i, 0)),
            pl.BlockSpec((tb, 1), lambda i: (i, 0)),
        ],
        out_shape=[
            jax.ShapeDtypeStruct((padded_tokens, hc_mult3), jnp.float32),
            jax.ShapeDtypeStruct((padded_tokens, 1), jnp.float32),
        ],
        compiler_params=compiler_params,
    )(x2d, fn_hi, fn_mid, fn_lo)

    if padded_tokens != num_tokens:
        mixes = mixes[:num_tokens]
        sqrsum = sqrsum[:num_tokens]
    return mixes, sqrsum


@functools.partial(jax.jit,
                   static_argnames=("rms_eps", "hc_pre_eps",
                                    "token_block_size", "vmem_limit_bytes"))
def mhc_pre_mixes_collapse(
    x2d: jax.Array,
    fn: jax.Array,
    hc_scale: jax.Array,
    hc_base: jax.Array,
    rms_eps: float,
    hc_pre_eps: float,
    *,
    token_block_size: int = 64,
    vmem_limit_bytes: int = DEFAULT_VMEM_LIMIT_BYTES,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Mixes + in-block collapse: the full memory-bound half of ``mhc_pre``.

    Returns (mixes (T, hc_mult3) f32 — raw, unscaled, for the XLA gates;
    sqrsum (T, 1) f32; layer_input2d (T, hidden_size) bf16).
    """
    assert x2d.dtype == jnp.bfloat16, x2d.dtype
    assert fn.dtype == jnp.float32, fn.dtype
    num_tokens, hc_hidden = x2d.shape
    hc_mult3 = fn.shape[0]
    assert fn.shape == (hc_mult3, hc_hidden), (fn.shape, x2d.shape)
    # hc_mult3 = hc_mult * (hc_mult + 2) -> unique positive hc_mult.
    hc_mult = next(m for m in range(1, 64) if m * (m + 2) == hc_mult3)
    hidden_size = hc_hidden // hc_mult

    fn_hi, fn_mid, fn_lo = _split_fn3(fn)
    sc2d = hc_scale.reshape(1, 3).astype(jnp.float32)
    hb2d = hc_base.reshape(1, hc_mult3).astype(jnp.float32)

    def _vmem_need(tb: int) -> int:
        # Double-buffered bf16 x block + layer output, transient f32
        # copy, resident bf16 fn chunks; x2 spill/scratch headroom.
        return 2 * (tb * hc_hidden * (2 * 2 + 4) + tb * hidden_size * 2 * 2 +
                    3 * hc_mult3 * hc_hidden * 2)

    tb = min(token_block_size, _round_up(num_tokens, _SUBLANE))
    while tb > _SUBLANE and _vmem_need(tb) > vmem_limit_bytes:
        # Degrade to a smaller block instead of a compile-time VMEM OOM.
        tb //= 2
    padded_tokens = _round_up(num_tokens, tb)
    if padded_tokens != num_tokens:
        # Zero rows are safe: they produce zero mixes/sqrsum/layer rows
        # and are sliced off below.
        x2d = jnp.pad(x2d, ((0, padded_tokens - num_tokens), (0, 0)))

    compiler_params = pltpu.CompilerParams(dimension_semantics=("parallel", ),
                                           vmem_limit_bytes=vmem_limit_bytes)

    resident = pl.BlockSpec((hc_mult3, hc_hidden), lambda i: (0, 0))
    mixes, sqrsum, layer2d = pl.pallas_call(
        functools.partial(_mixes_collapse_kernel,
                          hc_mult=hc_mult,
                          hidden_size=hidden_size,
                          rms_eps=rms_eps,
                          hc_pre_eps=hc_pre_eps),
        grid=(padded_tokens // tb, ),
        in_specs=[
            pl.BlockSpec((tb, hc_hidden), lambda i: (i, 0)),
            resident,
            resident,
            resident,
            pl.BlockSpec((1, 3), lambda i: (0, 0)),
            pl.BlockSpec((1, hc_mult3), lambda i: (0, 0)),
        ],
        out_specs=[
            pl.BlockSpec((tb, hc_mult3), lambda i: (i, 0)),
            pl.BlockSpec((tb, 1), lambda i: (i, 0)),
            pl.BlockSpec((tb, hidden_size), lambda i: (i, 0)),
        ],
        out_shape=[
            jax.ShapeDtypeStruct((padded_tokens, hc_mult3), jnp.float32),
            jax.ShapeDtypeStruct((padded_tokens, 1), jnp.float32),
            jax.ShapeDtypeStruct((padded_tokens, hidden_size), jnp.bfloat16),
        ],
        compiler_params=compiler_params,
    )(x2d, fn_hi, fn_mid, fn_lo, sc2d, hb2d)

    if padded_tokens != num_tokens:
        mixes = mixes[:num_tokens]
        sqrsum = sqrsum[:num_tokens]
        layer2d = layer2d[:num_tokens]
    return mixes, sqrsum, layer2d


def mhc_pre(
    residual: jax.Array,
    fn: jax.Array,
    hc_scale: jax.Array,
    hc_base: jax.Array,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
    *,
    token_block_size: int = 64,
    vmem_limit_bytes: int = DEFAULT_VMEM_LIMIT_BYTES,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """``mhc_pre`` with the mix GEMM AND the collapse in Pallas; only the
    tiny gates/Sinkhorn stay in XLA.

    Same contract as vLLM's ``mhc_pre_torch``:
    residual (..., hc_mult, hidden_size) bf16 in; post_mix (..., hc_mult, 1)
    f32, comb_mix (..., hc_mult, hc_mult) f32, layer_input (..., hidden_size)
    bf16 out. The gates recompute ``pre_mix`` on the raw mixes; its cost is
    a few tiny VPU ops and keeping ``utils.mhc_pre_gates`` shared and
    unsliced is worth more than removing them.
    """
    assert residual.dtype == jnp.bfloat16
    assert fn.dtype == jnp.float32

    outer_shape = residual.shape[:-2]
    hc_mult, hidden_size = residual.shape[-2:]
    x2d = residual.reshape(-1, hc_mult * hidden_size)

    mixes, sqrsum, layer2d = mhc_pre_mixes_collapse(
        x2d,
        fn,
        hc_scale,
        hc_base,
        rms_eps,
        hc_pre_eps,
        token_block_size=token_block_size,
        vmem_limit_bytes=vmem_limit_bytes,
    )
    _, post_mix, comb_mix = utils.mhc_pre_gates(
        mixes, sqrsum, hc_mult, hidden_size, hc_scale, hc_base, rms_eps,
        hc_pre_eps, hc_sinkhorn_eps, hc_post_mult_value, sinkhorn_repeat)

    return (post_mix.reshape(*outer_shape, hc_mult, 1),
            comb_mix.reshape(*outer_shape, hc_mult, hc_mult),
            layer2d.reshape(*outer_shape, hidden_size))
