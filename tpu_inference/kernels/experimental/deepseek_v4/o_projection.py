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

from tpu_inference.kernels.experimental.deepseek_v4.rope import (
    LANE, CosSinRef, _cos_sin_lanes, _rotate_gptj)

DEFAULT_VMEM_LIMIT_BYTES = 100 * 1024 * 1024

SUBLANE = 8

FP8_E4M3_MAX = float(jnp.finfo(jnp.float8_e4m3fn).max)


def _largest_divisor(x: int, cap: int) -> int:
    """Largest divisor of ``x`` that is <= ``cap``."""
    for candidate in range(min(x, cap), 0, -1):
        if x % candidate == 0:
            return candidate
    return 1


def _cos_sin_body(cos_sin_ref, out_ref, *, rotary_dim, inverse, out_dtype):
    """Expand a ``(tile_n, rotary_dim)`` block to ``(tile_n, 2 * LANE)``."""
    cos, sin = _cos_sin_lanes(cos_sin_ref[...],
                              rotary_dim=rotary_dim,
                              inverse=inverse)
    out_ref[...] = jnp.concatenate([cos, sin], axis=-1).astype(out_dtype)


def _cos_sin_kernel(
    # scalar prefetch
    positions_ref,
    # HBM input
    cos_sin_cache_hbm_ref,
    # HBM output
    out_hbm_ref,
    *,
    num_tiles: int,
    tile_n: int,
    rotary_dim: int,
    inverse: bool,
    cos_sin_dtype: jnp.dtype,
    out_dtype: jnp.dtype,
):
    cos_sin_spec = pl.BlockSpec(
        block_shape=(tile_n, rotary_dim),
        memory_space=pltpu.VMEM,
        index_map=lambda i: (i, 0),
    )
    out_spec = pl.BlockSpec(
        block_shape=(tile_n, 2 * LANE),
        memory_space=pltpu.VMEM,
        index_map=lambda i: (i, 0),
    )
    # The per-token row gather; one buffer of lookahead hides its DMAs.
    cos_sin_alloc = CosSinRef.create(spec=cos_sin_spec,
                                     dtype=cos_sin_dtype,
                                     buffer_count=2,
                                     tile_n=tile_n)

    pipeline = pltpu.emit_pipeline(
        functools.partial(
            _cos_sin_body,
            rotary_dim=rotary_dim,
            inverse=inverse,
            out_dtype=out_dtype,
        ),
        grid=(num_tiles, ),
        in_specs=[cos_sin_spec],
        out_specs=[out_spec],
    )
    allocations = (
        cos_sin_alloc,
        pltpu.BufferedRef.output(out_spec, out_dtype, buffer_count=2),
    )

    @pl.with_scoped(allocations=allocations)
    def _run(allocations):
        pipeline(
            (cos_sin_cache_hbm_ref, positions_ref),
            out_hbm_ref,
            allocations=allocations,
        )

    _run()


@functools.partial(
    jax.jit,
    static_argnames=("inverse", "out_dtype", "name"),
)
def gather_cos_sin(
    positions: jax.Array,  # [T] int
    cos_sin_cache: jax.Array,  # [max_position, rotary_dim] float32
    *,
    inverse: bool = False,
    out_dtype: jnp.dtype = jnp.float32,
    name: str = "gather_cos_sin",
) -> jax.Array:
    """Builds ``wo_a_projection``'s ``[T, 2 * LANE]`` cos/sin table.

  Gathers ``cos_sin_cache[positions]`` based on the token positions, expands
  each row to ``[2 * LANE]``
  """
    assert positions.ndim == 1
    assert cos_sin_cache.ndim == 2

    num_tokens = positions.shape[0]
    rotary_dim = cos_sin_cache.shape[1]
    assert rotary_dim % 2 == 0 and rotary_dim <= LANE

    tile_n = _largest_divisor(num_tokens, cap=128)

    return pl.pallas_call(
        functools.partial(
            _cos_sin_kernel,
            num_tiles=num_tokens // tile_n,
            tile_n=tile_n,
            rotary_dim=rotary_dim,
            inverse=inverse,
            cos_sin_dtype=cos_sin_cache.dtype,
            out_dtype=out_dtype,
        ),
        out_shape=jax.ShapeDtypeStruct((num_tokens, 2 * LANE), out_dtype),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=1,
            in_specs=(pl.BlockSpec(memory_space=pltpu.HBM), ),
            out_specs=pl.BlockSpec(memory_space=pltpu.HBM),
        ),
        compiler_params=pltpu.CompilerParams(
            vmem_limit_bytes=DEFAULT_VMEM_LIMIT_BYTES,
            disable_bounds_checks=True,
        ),
        name=name,
    )(positions, cos_sin_cache)


def _rope_heads(x, cos_sin, *, head_dim, heads_per_group):
    """apply RoPE to the last rope dim of head dim of ``x``.

  ``x`` is the ``(rows, heads_per_group * head_dim)`` *flat* view, i.e. the
  heads have already been folded into the lanes, so head ``h`` owns lanes
  ``[h * head_dim, (h + 1) * head_dim)`` and the channels RoPE touches are the
  last ``LANE`` of each of those spans.

  Taking the flat view first and rotating afterwards is numerically identical
  to rotating the ``(rows, heads_per_group, head_dim)`` value and reshaping
  after, but it is cheaper: every slice boundary here (``head_dim`` and
  ``head_dim - LANE``) is a multiple of the 128-lane register tile, so the
  concatenate is a lane-tile relabel. Rotating first instead forces the
  compiler to rebuild a 3-D value and then relayout it, which it lowers to a
  long ``vrot.slane`` / ``vcombine`` chain.
  """
    cos = cos_sin[:, :LANE]
    sin = cos_sin[:, LANE:]
    lo = head_dim - LANE
    pieces = []
    for h in range(heads_per_group):
        base = h * head_dim
        pieces.append(x[:, base:base + lo])
        tail = x[:, base + lo:base + head_dim].astype(jnp.float32)
        roped = tail * cos + _rotate_gptj(tail) * sin
        pieces.append(roped.astype(x.dtype))
    return jnp.concatenate(pieces, axis=-1)


def _kernel(
    # inputs
    x_ref,  # VMEM (tile_t, H, head_dim) bf16, or (1, tile_t, D) if group_major
    w_ref,  # VMEM (D, tile_r) fp8
    scale_ref,  # VMEM (1, tile_r) f32
    cos_sin_ref,  # VMEM (tile_t, 2 * LANE) f32
    # output
    out_ref,  # VMEM (tile_t, tile_r) out_dtype
    *,
    tile_t: int,
    num_sub_t: int,
    quantize_activations: bool,
    head_dim: int,
    heads_per_group: int,
    group_major: bool,
):
    rhs = w_ref[...]
    scale = scale_ref[...]
    cos_sin = cos_sin_ref[...]
    sub_t = tile_t // num_sub_t
    out_dtype = x_ref.dtype
    for s in range(num_sub_t):
        # Subchunk the tile to allow more Instruction level parallelism
        # opportunities to be exploited by the LLO compiler.
        rows = slice(s * sub_t, (s + 1) * sub_t)
        if group_major:
            # The producer already laid the group's heads out along the lanes,
            # so the (t, h, d) -> (t, h * head_dim + d) fold is a no-op here.
            flat = x_ref[0, rows]
        else:
            # Fold the heads into the lanes. ``H`` is the *sublane* dimension
            # of ``[T, G * H, head_dim]``, so this is a sublane -> lane
            # relayout and it is by far the most expensive thing the kernel
            # does; see ``wo_a_projection`` for how to avoid it entirely.
            flat = x_ref[rows].reshape(sub_t, -1)
        # Rotate only after the fold; see ``_rope_heads``.
        x = _rope_heads(flat,
                        cos_sin[rows],
                        head_dim=head_dim,
                        heads_per_group=heads_per_group)

        inv = None
        if quantize_activations:
            amax = jnp.max(jnp.abs(x), axis=1, keepdims=True)
            inv = (FP8_E4M3_MAX /
                   jnp.maximum(amax, jnp.bfloat16(1e-30))).astype(jnp.bfloat16)
            lhs = (x * inv).astype(jnp.float8_e4m3fn)
        else:
            lhs = x

        partial = jax.lax.dot_general(
            lhs,
            rhs,
            (((1, ), (0, )), ((), ())),
            preferred_element_type=jnp.float32,
        )
        if quantize_activations:
            assert inv is not None
            partial = partial * (1.0 / inv.astype(jnp.float32))
        out_ref[rows] = (partial * scale).astype(out_dtype)


@functools.partial(
    jax.jit,
    static_argnames=(
        "tile_t",
        "tile_r",
        "sub_t",
        "quantize_activations",
        "name",
    ),
)
def wo_a_projection(
    x: jax.Array,  # [T, G * H, head_dim] bf16, or [G, T, D] bf16 (group-major)
    wo_a: jax.Array,  # [D, G * R] fp8
    wo_a_scale: jax.Array,  # [G * R] float32
    cos_sin: jax.Array,  # [T, 2 * LANE] float32
    *,
    tile_t: int | None = None,
    tile_r: int | None = None,
    sub_t: int | None = None,
    quantize_activations: bool = True,
    name: str = "wo_a_projection",
) -> jax.Array:
    """Do DSv4 wo_a projection.

  T: num_tokens
  G: num_groups
  H: num_heads_per_group
  D: H * head_dim
  R: o_lora_dim

  Equivalent to
  x = rope(x)
  return ``einsum("tgd,dgr->tgr", x.view(t, g, d), wo_a.view(d, g, r),
  preferred_element_type=f32) * wo_a_scale.view(g, r).to(bf16)``

  ``quantize_activations`` quantizes each activation block's rows to fp8 in
  the kernel so both MXU operands are fp8.

  ``x`` may be given in either of two layouts:

  * ``[T, G * H, head_dim]`` (token-major) -- what the attention kernels emit
    today. The last two dims of this array are the tiled ones, so ``H`` lands
    on the *sublanes*, and folding the group's heads into the lanes that the
    MXU contracts over is a sublane -> lane relayout. Per-phase LLO attribution
    at the ``[1024, 128, 512]`` shape puts that single fold at 28% of all
    issued VLIW instructions with zero MXU work, and it resists every local
    fix (it is not a register-padding artifact -- redoing the shuffle in
    padding-free f32 is 1.24x *slower* -- and DMA cannot help because a strided
    single-head read is not tile-aligned).
  * ``[G, T, D]`` (group-major) -- the group's ``H * head_dim`` channels are
    already contiguous lanes in exactly ``wo_a``'s row order, so the fold
    disappears. This is 1.39x faster end to end and bit-identical; ``wo_a``
    and ``wo_a_scale`` are unchanged. Producing it costs nothing *if the
    producer writes this layout directly* -- relaying out after the fact in
    XLA does not pay for itself (measured at 226us for this shape).
  """
    assert x.ndim == 3
    assert x.dtype == jnp.bfloat16
    assert wo_a.ndim == 2 and wo_a_scale.ndim == 1

    reduction, out_features = wo_a.shape
    assert out_features == wo_a_scale.shape[0]

    # D == H * head_dim is 4096 while head_dim is 512, so the trailing dim
    # distinguishes the two layouts unambiguously.
    group_major = x.shape[2] == reduction
    if group_major:
        num_groups, num_tokens, _ = x.shape
        heads_per_group = SUBLANE
        assert reduction % heads_per_group == 0
        head_dim = reduction // heads_per_group
    else:
        num_tokens, num_heads, head_dim = x.shape
        assert reduction % head_dim == 0
        heads_per_group = reduction // head_dim
        assert num_heads % heads_per_group == 0
        num_groups = num_heads // heads_per_group

    # DSv4's heads_per_group is 8, exactly equal to SUBLANE
    # The kernel is based on this assumption, which makes it
    # simpler.
    assert heads_per_group == SUBLANE
    assert out_features % num_groups == 0
    lora_rank = out_features // num_groups

    assert cos_sin.ndim == 2 and cos_sin.shape == (num_tokens, 2 * LANE)
    assert head_dim % LANE == 0

    if tile_r is None:
        tile_r = lora_rank
    if tile_t is None:
        tile_t = _largest_divisor(num_tokens, cap=1024)
    if sub_t is None:
        # 128 is chosen based on large batch size (t) microbenchmarks
        sub_t = _largest_divisor(tile_t, cap=128)
    assert lora_rank % tile_r == 0
    assert num_tokens % tile_t == 0
    assert tile_t % sub_t == 0
    num_t_tiles = num_tokens // tile_t
    num_r_tiles = lora_rank // tile_r

    if group_major:
        # x: one group's token block, [g, t, 0] over [G, T, D].
        x_spec = pl.BlockSpec((1, tile_t, reduction), lambda g, t, r:
                              (g, t, 0))
    else:
        # x: one group's heads, [t, g, 0] over [T, G * H, head_dim].
        x_spec = pl.BlockSpec((tile_t, heads_per_group, head_dim),
                              lambda g, t, r: (t, g, 0))

    return pl.pallas_call(
        functools.partial(
            _kernel,
            tile_t=tile_t,
            num_sub_t=tile_t // sub_t,
            quantize_activations=quantize_activations,
            head_dim=head_dim,
            heads_per_group=heads_per_group,
            group_major=group_major,
        ),
        out_shape=jax.ShapeDtypeStruct((num_tokens, out_features), x.dtype),
        grid=(num_groups, num_t_tiles, num_r_tiles),
        in_specs=[
            x_spec,
            # wo_a: the whole reduction, column block g * R + r * tile_r.
            pl.BlockSpec((reduction, tile_r), lambda g, t, r:
                         (0, g * num_r_tiles + r)),
            # wo_a_scale
            pl.BlockSpec((1, tile_r), lambda g, t, r:
                         (0, g * num_r_tiles + r)),
            # cos_sin: one token block
            pl.BlockSpec((tile_t, 2 * LANE), lambda g, t, r: (t, 0)),
        ],
        out_specs=pl.BlockSpec((tile_t, tile_r), lambda g, t, r:
                               (t, g * num_r_tiles + r)),
        compiler_params=pltpu.CompilerParams(
            vmem_limit_bytes=DEFAULT_VMEM_LIMIT_BYTES,
            disable_bounds_checks=True,
        ),
        name=name,
    )(x, wo_a, wo_a_scale.reshape(1, out_features), cos_sin)


@functools.partial(
    jax.jit,
    static_argnames=(
        "head_dim",
        "inverse",
        "tile_t",
        "tile_r",
        "sub_t",
        "quantize_activations",
        "name",
    ),
)
def fused_reverse_rope_wo_a_projection(
    x: jax.Array,  # [T, G * H, head_dim] bf16
    positions: jax.Array,  # [T] int
    cos_sin_cache: jax.Array,  # [max_position, rotary_dim] float32
    wo_a: jax.Array,  # [D, G * R] fp8
    wo_a_scale: jax.Array,  # [G * R] float32
    *,
    head_dim: int,
    inverse: bool = True,
    tile_t: int | None = None,
    tile_r: int | None = None,
    sub_t: int | None = None,
    quantize_activations: bool = True,
    name: str = "wo_a_projection",
) -> jax.Array:
    """``The RoPE-fused ``wo_a_projection``."""
    assert x.ndim == 3
    # Either token-major [T, G * H, head_dim] or group-major [G, T, H*head_dim];
    # ``wo_a_projection`` accepts both.
    assert x.shape[2] == head_dim or x.shape[2] % head_dim == 0


    cos_sin = gather_cos_sin(
        positions,
        cos_sin_cache,
        inverse=inverse,
        name=f"{name}_gather_cos_sin",
    )
    return wo_a_projection(
        x,
        wo_a,
        wo_a_scale,
        cos_sin,
        tile_t=tile_t,
        tile_r=tile_r,
        sub_t=sub_t,
        quantize_activations=quantize_activations,
        name=name,
    )
