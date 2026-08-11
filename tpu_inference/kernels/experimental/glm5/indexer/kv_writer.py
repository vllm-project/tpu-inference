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
"""Pallas TPU kernel: GLM-5.2 lightning-indexer per-token Q/K writer.

Fuses, per raw token, the tail of ``Indexer.forward``'s Q/K preparation
pipeline into a single VMEM-resident Pallas kernel:

  - K: RMSNorm (``eps=1e-6``, unweighted -- see the docstring of
    ``index_qk_rope_quant`` for a caveat vs. production's affine LayerNorm)
    -> interleaved (GPT-J-style) RoPE on the *leading* ``rope_dim`` channels
    -> FP8 (``e4m3fn``/``ue8m0``) quantization, one scale per token
    (``quant_block_size == head_dim``).
  - Q: interleaved RoPE on the leading ``rope_dim`` channels (no norm, per
    ``Indexer.forward`` -- only K is normed) -> FP8 quantization, one scale
    per (token, head).

This is the "raw per-token index K-cache writer" from the bring-up plan's
Phase 2 item 1, engineered as a lightweight adaptation of
``tpu_inference/kernels/experimental/deepseek_v4/rope.py``'s
``qnorm_rope``/``rope_quant`` kernels: same fused-VMEM-block idea (grid over
token tiles, automatic Mosaic double-buffering via ``pl.BlockSpec``), but
(a) the RoPE is applied to the *leading* slice of each head (GLM-5.2's
``Indexer.forward`` splits ``q_pe, q_nope = split(q, [rope_dim, head_dim -
rope_dim])``, RoPE-ing the leading half), not DSv4's trailing slice, and
(b) quantization is fused directly into the norm+rope kernel rather than a
separate pass. The per-token ``cos``/``sin`` RoPE table is gathered *outside*
the kernel (a cheap ``O(T x rope_dim)`` trig table lookup, not the
bandwidth-critical part) and fed in as a normal blocked VMEM input, trading
DSv4's fancier per-row async-copy gather (``CosSinRef``) for simplicity --
see the module docstring of ``tests/kernels/glm5/kv_writer_test.py`` for the
numerical verification against the Phase 1 JAX reference.
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

LANE = 128  # TPU native lane count; GLM-5.2's indexer head_dim == LANE.
DEFAULT_VMEM_LIMIT_BYTES = 100 * 1024 * 1024
_FP8_DTYPE = jnp.float8_e4m3fn


def _pick_tile_n(num_tokens: int, cap: int = 128) -> int:
    """Picks a token-axis tile size for the grid: the largest divisor of
    ``num_tokens`` that is both <= ``cap`` and a multiple of 8 (Mosaic's
    sublane-alignment requirement for a non-full-array block), or
    ``num_tokens`` itself if no such divisor exists (a single whole-array
    tile is always allowed, regardless of alignment, since a block shape
    equal to the full array shape is exempt from the divisibility rule)."""
    for candidate in range(min(num_tokens, cap), 0, -1):
        if num_tokens % candidate == 0 and candidate % 8 == 0:
            return candidate
    return num_tokens


def _lane_gather(operand: jax.Array, indices: jax.Array) -> jax.Array:
    """``out[b..., i] = operand[b..., indices[i]]`` over the last axis.

    ``indices`` is a constant 1-D array; this lowers to a lane shuffle rather
    than a general dynamic gather. See
    ``tpu_inference/kernels/experimental/deepseek_v4/rope.py``/
    ``tests/kernels/deepseek_v4/rope_test.py`` for the identical primitive.
    """
    rank = operand.ndim
    lanes = indices.shape[0]
    coords = jnp.broadcast_to(indices, (*operand.shape[:-1], lanes))[..., None]
    batching = tuple(range(rank - 1))
    dimension_numbers = jax.lax.GatherDimensionNumbers(
        offset_dims=(),
        collapsed_slice_dims=(rank - 1, ),
        start_index_map=(rank - 1, ),
        operand_batching_dims=batching,
        start_indices_batching_dims=batching,
    )
    return jax.lax.gather(
        operand,
        coords,
        dimension_numbers=dimension_numbers,
        slice_sizes=(1, ) * rank,
        unique_indices=False,
        mode=jax.lax.GatherScatterMode.PROMISE_IN_BOUNDS,
    )


def _cos_sin_lanes_leading(cos_sin: jax.Array, *,
                           rotary_dim: int) -> tuple[jax.Array, jax.Array]:
    """Expands a packed ``[cos | sin]`` row (``rotary_dim`` wide, each half
    ``rotary_dim // 2``) into full ``LANE``-wide ``(cos, sin)`` arrays, with
    the roped channels on the *leading* ``rotary_dim`` lanes (GLM-5.2's
    convention) rather than DSv4's trailing-slice convention. NoPE lanes get
    the identity rotation (``cos=1, sin=0``).
    """
    half = rotary_dim // 2
    channel = jnp.arange(LANE)
    is_rot = channel < rotary_dim
    # Channels 2k, 2k+1 both read frequency k; clamp so the gather index
    # stays in bounds on the NoPE lanes (replaced by `jnp.where` below).
    freq = jnp.minimum(channel, rotary_dim - 1) // 2

    cos_sin = cos_sin.astype(jnp.float32)
    cos = _lane_gather(cos_sin, freq)  # cos half, repeated per pair.
    sin = _lane_gather(cos_sin, half + freq)  # sin half, repeated per pair.
    return jnp.where(is_rot, cos, 1.0), jnp.where(is_rot, sin, 0.0)


def _rotate_gptj(x: jax.Array) -> jax.Array:
    """``[-x1, x0, -x3, x2, ...]`` along the lanes -- a constant-index lane
    shuffle and sign flip, position-agnostic (works regardless of whether the
    roped slice is leading or trailing)."""
    lane = jnp.arange(x.shape[-1])
    sign = jnp.where(lane % 2 == 0, -1.0, 1.0).astype(jnp.float32)
    return _lane_gather(x, lane ^ 1) * sign


def _quantize_fp8_row(x: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Per-row (whole trailing axis) FP8 ``ue8m0``-style quantization:
    ``x`` is ``[..., LANE]``; returns ``(q, scale)`` with ``scale`` shape
    ``[..., 1]``, ``q.astype(f32) * scale ~= x``."""
    fp8_max = float(jnp.finfo(_FP8_DTYPE).max)
    amax = jnp.clip(jnp.max(jnp.abs(x), axis=-1, keepdims=True), 1e-4, None)
    scale = jnp.exp2(jnp.ceil(jnp.log2(amax / fp8_max)))
    q = (x * (1.0 / scale)).astype(_FP8_DTYPE)
    return q, scale


def _kv_writer_kernel_body(
    q_ref,  # [tile_n, H, LANE] f32/bf16
    k_ref,  # [tile_n, LANE] f32/bf16
    cos_sin_ref,  # [tile_n, rope_dim] f32, packed [cos | sin]
    q_fp8_ref,  # [tile_n, H, LANE] f8
    q_scale_ref,  # [tile_n, H] f32
    k_fp8_ref,  # [tile_n, LANE] f8
    k_scale_ref,  # [tile_n] f32
    *,
    rope_dim: int,
    rmsnorm_eps: float,
    do_k_norm: bool,
):
    q = q_ref[...].astype(jnp.float32)
    k = k_ref[...].astype(jnp.float32)
    cos, sin = _cos_sin_lanes_leading(cos_sin_ref[...], rotary_dim=rope_dim)

    if do_k_norm:
        rms = jax.lax.rsqrt(
            jnp.mean(k * k, axis=-1, keepdims=True) + rmsnorm_eps)
        k = k * rms

    q_roped = q * cos[:, None, :] + _rotate_gptj(q) * sin[:, None, :]
    k_roped = k * cos + _rotate_gptj(k) * sin

    q_fp8, q_scale = _quantize_fp8_row(q_roped)
    k_fp8, k_scale = _quantize_fp8_row(k_roped)

    q_fp8_ref[...] = q_fp8
    q_scale_ref[...] = q_scale[..., 0]
    k_fp8_ref[...] = k_fp8
    k_scale_ref[...] = k_scale[..., 0]


@functools.partial(
    jax.jit,
    static_argnames=("rope_dim", "rope_base", "rmsnorm_eps", "do_k_norm",
                     "tile_n", "vmem_limit_bytes", "name"),
)
def index_qk_rope_quant(
    q_raw: jax.Array,  # [T, H, head_dim], post wq_b projection, pre-RoPE
    k_raw: jax.Array,  # [T, head_dim], post wk_weights_proj, pre-norm/RoPE
    positions: jax.Array,  # [T] int32
    *,
    rope_dim: int = 64,
    rope_base: float = 10000.0,
    rmsnorm_eps: float = 1e-6,
    do_k_norm: bool = True,
    tile_n: int | None = None,
    vmem_limit_bytes: int = DEFAULT_VMEM_LIMIT_BYTES,
    name: str = "glm5_index_qk_rope_quant",
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Fused per-token RMSNorm(K only)+interleaved-RoPE+FP8-quant Pallas
    kernel for the GLM-5.2 lightning indexer's Q/K.

    NOTE (production caveat, out of scope to fix here): upstream vLLM's
    ``Indexer.k_norm`` is an *affine* ``nn.LayerNorm`` (mean-centered, with
    learned weight/bias), not the unweighted RMSNorm implemented here (which
    matches this bring-up plan's Phase 2 spec and DSv4's ``qnorm_rope``
    convention). Model wiring (Phase 3) must either extend this kernel with
    affine-LayerNorm support or apply the affine LayerNorm separately before
    calling this kernel with ``do_k_norm=False``.

    Args:
      q_raw: ``[T, num_heads, head_dim]``, ``head_dim`` must be ``128``
        (one lane block -- matches GLM-5.2's ``index_head_dim``).
      k_raw: ``[T, head_dim]``, single shared (MQA) K vector per token.
      positions: ``[T]`` int32 absolute token positions (RoPE + causal bookkeeping
        upstream).
      rope_dim: width of the leading roped slice (``64`` for GLM-5.2).
      rope_base: RoPE theta.
      rmsnorm_eps: RMSNorm epsilon (K only).
      do_k_norm: apply RMSNorm to K before RoPE (see the production caveat
        above).
      tile_n: grid tile size along the token axis; auto-selected (a divisor
        of ``T``, capped at 128) if ``None``.

    Returns:
      ``(q_fp8, q_scale, k_fp8, k_scale)``:
        - ``q_fp8``: ``[T, num_heads, head_dim]`` ``float8_e4m3fn``.
        - ``q_scale``: ``[T, num_heads]`` float32 (one scale per token/head,
          ``quant_block_size == head_dim``).
        - ``k_fp8``: ``[T, head_dim]`` ``float8_e4m3fn``.
        - ``k_scale``: ``[T]`` float32.
    """
    num_tokens, num_heads, head_dim = q_raw.shape
    if head_dim != LANE:
        raise ValueError(f"head_dim must be {LANE} (one lane block), got "
                         f"{head_dim}")
    if k_raw.shape != (num_tokens, head_dim):
        raise ValueError(f"k_raw shape {k_raw.shape} != "
                         f"{(num_tokens, head_dim)}")
    if rope_dim % 2 != 0 or rope_dim > LANE:
        raise ValueError(f"rope_dim ({rope_dim}) must be even and <= {LANE}")

    # Cheap O(T x rope_dim) trig table lookup outside the kernel -- not the
    # bandwidth-critical part; see the module docstring.
    half = rope_dim // 2
    inv_freq = 1.0 / (rope_base**(
        jnp.arange(0, rope_dim, 2, dtype=jnp.float32) / rope_dim))
    freqs = positions.astype(jnp.float32)[:, None] * inv_freq[None, :]
    cos_sin = jnp.concatenate([jnp.cos(freqs), jnp.sin(freqs)],
                              axis=-1)  # [T, rope_dim]
    del half

    if tile_n is None:
        tile_n = _pick_tile_n(num_tokens, cap=128)
    if num_tokens % tile_n != 0:
        raise ValueError(f"num_tokens ({num_tokens}) must be a multiple of "
                         f"tile_n ({tile_n})")
    num_tiles = num_tokens // tile_n

    kernel = functools.partial(
        _kv_writer_kernel_body,
        rope_dim=rope_dim,
        rmsnorm_eps=rmsnorm_eps,
        do_k_norm=do_k_norm,
    )

    out_shape = (
        jax.ShapeDtypeStruct((num_tokens, num_heads, head_dim), _FP8_DTYPE),
        jax.ShapeDtypeStruct((num_tokens, num_heads), jnp.float32),
        jax.ShapeDtypeStruct((num_tokens, head_dim), _FP8_DTYPE),
        jax.ShapeDtypeStruct((num_tokens, ), jnp.float32),
    )

    q_fp8, q_scale, k_fp8, k_scale = pl.pallas_call(
        kernel,
        grid=(num_tiles, ),
        in_specs=[
            pl.BlockSpec((tile_n, num_heads, head_dim), lambda i: (i, 0, 0)),
            pl.BlockSpec((tile_n, head_dim), lambda i: (i, 0)),
            pl.BlockSpec((tile_n, rope_dim), lambda i: (i, 0)),
        ],
        out_specs=(
            pl.BlockSpec((tile_n, num_heads, head_dim), lambda i: (i, 0, 0)),
            pl.BlockSpec((tile_n, num_heads), lambda i: (i, 0)),
            pl.BlockSpec((tile_n, head_dim), lambda i: (i, 0)),
            pl.BlockSpec((tile_n, ), lambda i: (i, )),
        ),
        out_shape=out_shape,
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("arbitrary", ),
            vmem_limit_bytes=vmem_limit_bytes,
        ),
        name=name,
    )(q_raw, k_raw, cos_sin)

    return q_fp8, q_scale, k_fp8, k_scale
