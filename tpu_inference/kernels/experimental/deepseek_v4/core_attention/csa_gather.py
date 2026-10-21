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
from typing import Any

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.experimental.pallas import tpu_sc as plsc


def main_kernel(
    nope_in_hbm_ref: Any,
    rope_in_hbm_ref: Any,
    indices_hbm_ref: Any,
    nope_out_hbm_ref: Any,
    rope_out_hbm_ref: Any,
    *,
    core_axis_name: str,
    subcore_axis_name: str,
    num_row_subchunks: int,
    num_streams: int,
):
    tpu_info = pltpu.get_tpu_info()
    sc_info = tpu_info.sparse_core
    assert sc_info is not None
    num_simd_lanes = sc_info.num_lanes
    num_cores = jax.lax.axis_size((core_axis_name, subcore_axis_name))
    row_subchunk_size = num_simd_lanes
    row_chunk_size = row_subchunk_size * num_row_subchunks
    block_size = row_chunk_size * num_cores
    num_blocks = pl.cdiv(indices_hbm_ref.shape[0], block_size)

    # Inputs are 8-bit;
    # nope output stays uint8 (4/int32), rope is unpacked to bf16 (2/int32).
    in_bits = jax.dtypes.itemsize_bits(nope_in_hbm_ref.dtype)
    in_packing = 32 // in_bits
    in_mask = (1 << in_bits) - 1  # 0xFF for 8-bit.
    nope_out_packing = 32 // jax.dtypes.itemsize_bits(nope_out_hbm_ref.dtype)
    rope_out_bits = jax.dtypes.itemsize_bits(rope_out_hbm_ref.dtype)
    rope_out_packing = 32 // rope_out_bits
    core_index = lax.axis_index((core_axis_name, subcore_axis_name))

    # SparseCore gather 32-bit words
    nope_in_i32 = nope_in_hbm_ref.bitcast(jnp.int32)
    rope_in_i32 = rope_in_hbm_ref.bitcast(jnp.int32)
    nope_out_i32 = nope_out_hbm_ref.bitcast(jnp.int32)
    rope_out_i32 = rope_out_hbm_ref.bitcast(jnp.int32)

    nope_in_cols = nope_in_i32.shape[1]
    rope_in_cols = rope_in_i32.shape[1]
    nope_out_cols = nope_out_hbm_ref.shape[1]
    rope_out_cols = rope_out_hbm_ref.shape[1]

    # def process_nope_reference(gather_ref, out_ref):
    #   col_slice = pl.ds(0, 128)
    #   mask = (1 << in_bits) - 1  # 0xFF for 8-bit.
    #   for i in range(num_simd_lanes // nope_out_packing):
    #     packed_row0 = jnp.zeros((1, 128), dtype=jnp.int32)
    #     packed_row1 = jnp.zeros((1, 128), dtype=jnp.int32)
    #     packed_row2 = jnp.zeros((1, 128), dtype=jnp.int32)
    #     packed_row3 = jnp.zeros((1, 128), dtype=jnp.int32)
    #     for j in range(nope_out_packing):
    #       k = i * nope_out_packing + j
    #       data = gather_ref[pl.ds(k, 1), col_slice]
    #       val0 = jnp.bitwise_right_shift(data, 24) & mask
    #       val1 = jnp.bitwise_right_shift(data, 16) & mask
    #       val2 = jnp.bitwise_right_shift(data, 8) & mask
    #       val3 = data & mask
    #       pack_shift = j * in_bits
    #       packed_row0 = jnp.bitwise_or(
    #           packed_row0, jnp.left_shift(val3, pack_shift)
    #       )
    #       packed_row1 = jnp.bitwise_or(
    #           packed_row1, jnp.left_shift(val2, pack_shift)
    #       )
    #       packed_row2 = jnp.bitwise_or(
    #           packed_row2, jnp.left_shift(val1, pack_shift)
    #       )
    #       packed_row3 = jnp.bitwise_or(
    #           packed_row3, jnp.left_shift(val0, pack_shift)
    #       )
    #     out_ref[pl.ds(i, 1), pl.ds(0, 128)] = packed_row0
    #     out_ref[pl.ds(i, 1), pl.ds(128, 128)] = packed_row1
    #     out_ref[pl.ds(i, 1), pl.ds(256, 128)] = packed_row2
    #     out_ref[pl.ds(i, 1), pl.ds(384, 128)] = packed_row3

    def _delta_swap(x, y, shift, swap_mask):
        """Exchanges selected bits between two words.

    Swaps x's bits at positions ``swap_mask << shift`` with y's bits at
    positions ``swap_mask``; all other bits are left untouched. Worked
    example with ``shift=8, swap_mask=0x00FF00FF`` on byte-quads
    ``x = [x0 x1 x2 x3]`` and ``y = [y0 y1 y2 y3]`` (byte 0 = least
    significant)::

        swap_mask << 8 = 0xFF00FF00  -> x's bytes 1 and 3
        swap_mask      = 0x00FF00FF  -> y's bytes 0 and 2
        result:  x = [x0 y0 x2 y2],  y = [x1 y1 x3 y3]

    i.e. x's odd bytes trade places with y's even bytes. The identity
    ``t = ((x >> s) ^ y) & mask;  x ^= t << s;  y ^= t`` does this in 6 ops
    with only `t` as scratch. The arithmetic right shift is safe: its
    sign-extended high bits are dropped by the `& swap_mask`.
    """
        t = jnp.bitwise_and(
            jnp.bitwise_xor(jnp.bitwise_right_shift(x, shift), y), swap_mask)
        x = jnp.bitwise_xor(x, jnp.left_shift(t, shift))
        y = jnp.bitwise_xor(y, t)
        return x, y

    def process_nope(gather_ref, out_ref, out_row_base=0):
        # A more efficient implementation of `process_nope_reference`, with fewer
        # ALU ops. We've seen that SparseCore's Integer ALU FLOPs being the
        # bottleneck of the kernel, this is the preferred implementation.
        col_slice = pl.ds(0, 128)
        low_16 = jnp.int32(0x0000FFFF)  # one 16-bit half of each word
        low_byte_of_each_half = jnp.int32(0x00FF00FF)  # bytes 0 and 2
        for i in range(num_simd_lanes // nope_out_packing):
            d = [
                gather_ref[pl.ds(i * nope_out_packing + j, 1), col_slice]
                for j in range(nope_out_packing)
            ]
            # Round 1: transpose the four 2x2 byte blocks -- exchange the high
            # 16 bits of each word with the low 16 bits of its distance-2 peer.
            d[0], d[2] = _delta_swap(d[0], d[2], 16, low_16)
            d[1], d[3] = _delta_swap(d[1], d[3], 16, low_16)
            # Round 2: transpose within each 2x2 block -- exchange the odd bytes
            # of each word with the even bytes of its adjacent peer.
            d[0], d[1] = _delta_swap(d[0], d[1], 8, low_byte_of_each_half)
            d[2], d[3] = _delta_swap(d[2], d[3], 8, low_byte_of_each_half)
            # d[m] now holds byte lane m of all 4 inputs -> output sub-row m.s
            for m in range(nope_out_packing):
                out_ref[pl.ds(out_row_base + i, 1), pl.ds(m * 128, 128)] = d[m]

    def process_rope(gather_ref, out_ref, idx_sub, out_row_base=0):
        # one (1, 128) uint8 is one token's rope data, which encodes 64 bf16
        # values. (0, 64) are the high bits for bf16 data, (64, 128) are the low.
        half = rope_out_cols
        col_hi = pl.ds(0, half)
        col_lo = pl.ds(half, half)

        def bf16_bits(k):
            sub = lax.rem(idx_sub[k], in_packing)
            hi = jnp.bitwise_and(
                jnp.bitwise_right_shift(gather_ref[pl.ds(k, 1), col_hi],
                                        in_bits * sub),
                in_mask,
            )
            lo = jnp.bitwise_and(
                jnp.bitwise_right_shift(gather_ref[pl.ds(k, 1), col_lo],
                                        in_bits * sub),
                in_mask,
            )
            return jnp.bitwise_or(jnp.left_shift(hi, in_bits), lo)

        for t in range(num_simd_lanes // rope_out_packing):
            packed = jnp.zeros((1, half), dtype=jnp.int32)
            for pk in range(rope_out_packing):
                k = t * rope_out_packing + pk
                packed = jnp.bitwise_or(
                    packed, jnp.left_shift(bf16_bits(k), pk * rope_out_bits))
            out_ref[pl.ds(out_row_base + t, 1), pl.ds(0, half)] = packed

    def outer_pipeline(idx_ref):
        b = pl.program_id(0)
        out_row_base = (b * num_cores + core_index) * num_row_subchunks

        # Subchunk handled by stream `s` at inner step `r`. `num_streams`
        # independent `pl.Indirect` gathers run concurrently per step, keeping
        # several gather DMAs in flight to raise effective read bandwidth.
        def subchunk(r, s):
            return r * num_streams + s

        def idx_window(r, s):
            return idx_ref[pl.ds(
                subchunk(r, s) * row_subchunk_size, row_subchunk_size)]

        # Rows produced per stream in each output (nope packs 4/int32, rope 2).
        nope_rows_per_stream = row_subchunk_size // nope_out_packing
        rope_rows_per_stream = row_subchunk_size // rope_out_packing

        def _body(*refs):
            r = pl.program_id(0)
            nope_g = refs[0 * num_streams:1 * num_streams]
            rope_g = refs[1 * num_streams:2 * num_streams]
            nope_o = refs[2 * num_streams]
            rope_o = refs[2 * num_streams + 1]
            for s in range(num_streams):
                process_nope(
                    gather_ref=nope_g[s],
                    out_ref=nope_o,
                    out_row_base=s * nope_rows_per_stream,
                )
                process_rope(
                    gather_ref=rope_g[s],
                    out_ref=rope_o,
                    idx_sub=idx_window(r, s),
                    out_row_base=s * rope_rows_per_stream,
                )

        # Have multiple parallel `pl.Indirect` to hide random access read latency.
        # Output contiguous memory access, multiple output parallel DMAs not help
        # with performance.

        # nope: gather int32 row == index (1 int32 row per entry).
        nope_in_specs = tuple(
            pl.BlockSpec(
                (pl.Indirect(row_subchunk_size), nope_in_cols),
                lambda r, s=s: (idx_window(r, s), 0),
            ) for s in range(num_streams))
        # rope: gather int32 row == index // in_packing (in_packing entries/row).
        rope_in_specs = tuple(
            pl.BlockSpec(
                (pl.Indirect(row_subchunk_size), rope_in_cols),
                lambda r, s=s: (lax.div(idx_window(r, s), in_packing), 0),
            ) for s in range(num_streams))
        # One merged output block per cache, covering all `num_streams`
        # subchunks.
        nope_out_spec = pl.BlockSpec(
            (num_streams * nope_rows_per_stream, nope_out_cols),
            lambda r: (out_row_base // num_streams + r, 0),
        )
        rope_out_spec = pl.BlockSpec(
            (num_streams * rope_rows_per_stream, rope_out_cols),
            lambda r: (out_row_base // num_streams + r, 0),
        )
        pltpu.emit_pipeline(
            _body,
            grid=(num_row_subchunks // num_streams, ),
            in_specs=nope_in_specs + rope_in_specs,
            out_specs=(nope_out_spec, rope_out_spec),
        )(
            *([nope_in_i32] * num_streams),
            *([rope_in_i32] * num_streams),
            nope_out_i32,
            rope_out_i32,
        )

    pltpu.emit_pipeline(
        outer_pipeline,
        grid=(num_blocks, ),
        in_specs=pl.BlockSpec(
            (row_chunk_size, ),
            lambda b: (b * num_cores + core_index, ),
        ),
    )(indices_hbm_ref)


@functools.partial(jax.jit)
def csa_gather(
    nope_cache: jax.Array,
    rope_cache: jax.Array,
    indices: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Fused SparseCore gather of the nope and rope caches.

  Args:
    nope_cache: (total_pages, page_size, 4, 128) uint8. Each (4, 128) uint8 is
      token's nope + nope scales. It encodes 448 fp8 + 7 e8m0 scales + padding.
    rope_cache: (total_pages, page_size // 4, 4, 128) uint8. Each (1, 128) uint8
      is token's rope. It encodes 64 bf16.
    indices: (N,) int32. Token indices into the caches.

  Returns:
    nope_out: (N, 512) uint8.
      Each (512) uint8 is token's nope.
    rope_out: (N, 64) bf16.
      Each (64) bf16 is token's rope.
  """
    assert indices.ndim == 1, "Indices must be 1D."
    assert nope_cache.dtype == rope_cache.dtype, "Caches must share a dtype."
    assert nope_cache.dtype == jnp.uint8, "Caches must be uint8."

    # Flatten both caches to 128-wide rows and view as raw bytes.
    nope_cache = nope_cache.reshape(-1, nope_cache.shape[3])
    rope_cache = rope_cache.reshape(-1, rope_cache.shape[3])
    sc_info = pltpu.get_tpu_info().sparse_core
    assert sc_info is not None, "SparseCore info is missing."
    out_size = indices.size
    nope_out_cols = 512
    # rope: each 128-byte entry encodes 64 bf16 (high bytes [0:64], low [64:128]).
    rope_out_cols = 64
    num_simd_lanes = sc_info.num_lanes
    num_cores = sc_info.num_cores * sc_info.num_subcores

    # `num_streams` independent `pl.Indirect` gathers are issued per
    # pipeline step to keep multiple gather DMAs in flight.
    # See `outer_pipeline` for details.
    num_streams = 2
    num_row_subchunks = 32
    assert (
        num_row_subchunks %
        num_streams == 0), f"{num_streams=} must divide {num_row_subchunks=}."
    row_subchunk_size = num_simd_lanes
    row_chunk_size = row_subchunk_size * num_row_subchunks
    block_size = row_chunk_size * num_cores
    out_pad_size = (
        (out_size + block_size - 1) // block_size) * block_size - out_size
    indices = jnp.pad(indices, ((0, out_pad_size)))
    vector_mesh = plsc.VectorSubcoreMesh(
        num_cores=sc_info.num_cores,
        num_subcores=sc_info.num_subcores,
        core_axis_name="core",
        subcore_axis_name="subcore",
    )
    nope_out, rope_out = pl.kernel(
        functools.partial(
            main_kernel,
            core_axis_name=vector_mesh.core_axis_name,
            subcore_axis_name=vector_mesh.subcore_axis_name,
            num_row_subchunks=num_row_subchunks,
            num_streams=num_streams,
        ),
        out_type=(
            jax.ShapeDtypeStruct((out_size + out_pad_size, nope_out_cols),
                                 jnp.uint8),
            jax.ShapeDtypeStruct((out_size + out_pad_size, rope_out_cols),
                                 jnp.bfloat16),
        ),
        compiler_params=pltpu.CompilerParams(
            use_tc_tiling_on_sc=True,
            needs_layout_passes=True,
            disable_bounds_checks=True,
        ),
        mesh=vector_mesh,
        name="sc_csa_gather",
    )(nope_cache, rope_cache, indices)
    return (
        nope_out[:out_size],
        rope_out[:out_size],
    )
