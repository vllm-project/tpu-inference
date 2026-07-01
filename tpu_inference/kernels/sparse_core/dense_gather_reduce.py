# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""SparseCore gather-reduce kernel implementation using Pallas.

This module contains a Pallas kernel implementation for performing a
gather-reduce operation on TPU SparseCore. It groups rows of an operand
based on provided indices, sums them up, and scatters the results.
"""

import functools

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.experimental.pallas import tpu_sc as plsc


def align_to(x, a):
    """Ceiling-align x to a multiple of a (mirrors gmm_v2.align_to)."""
    return ((x + a - 1) // a) * a


def is_compatible(
    op: jax.Array,
    idx: jax.Array,
    reduce_group_size: int,
    row_chunk_size: int = 512,
    single_sc: bool = False,
) -> bool:
    """Checks if the inputs are compatible with the SparseCore Pallas kernel."""
    if op.dtype != jnp.bfloat16 and op.dtype != jnp.float32:
        return False
    if op.shape[0] % reduce_group_size != 0:
        return False

    sc_info = pltpu.get_tpu_info().sparse_core
    if sc_info is None:
        return False

    if sc_info.num_lanes % reduce_group_size != 0:
        return False

    # The output block has (num_lanes // reduce_group_size) // packing rows;
    # fall back to JAX when that is 0 (the kernel can't emit a zero-row block).
    packing = 32 // jax.dtypes.itemsize_bits(op.dtype)
    if (sc_info.num_lanes // reduce_group_size) // packing < 1:
        return False

    num_cores = 1 if single_sc else sc_info.num_cores
    num_subcores = sc_info.num_subcores
    row_wave_size = row_chunk_size * num_cores * num_subcores
    if idx.size % row_wave_size != 0:
        return False

    return True


def _sc_gather_reduce(
    op: jax.Array,
    idx: jax.Array,
    topk_weights: jax.Array | None = None,
    *,
    reduce_group_size: int,
    single_sc: bool = False,
    col_chunk_size: int = int(3.5 * 1024),
    row_chunk_size: int = 512,
    topk_wgt_zero_nan: bool = False,
) -> jax.Array:
    """Performs a gather-reduce operation on SparseCore.

  This kernel groups rows of the operand ``op`` based on ``idx``, sums them
  up, and scatters the results. The gather and add operations are performed
  in fp32, and the results are written back in bf16.

  Equivalent JAX code::

    gathered = op[idx, :]
    if topk_weights is not None:
      flat_weights = topk_weights.flatten()
      gathered = gathered * flat_weights[:, None].astype(jnp.float32)
    gathered = jnp.reshape(gathered, (-1, reduce_group_size, op.shape[1]))
    output = jnp.sum(gathered.astype(jnp.float32), axis=1).astype(jnp.bfloat16)

  Args:
    op: The operand matrix [B, K] in f32 or bf16 to gather from and reduce.
    idx: The indices [M,] in int32 guiding the gather.
    topk_weights: Optional weights [M // 128, 128] in bf16 to apply to the
      gathered rows before reduction.
    reduce_group_size: The number of gathered rows to sum per output row.
    single_sc: Whether to use a single SparseCore.
    col_chunk_size: The size of column chunks to process.
    row_chunk_size: The size of row chunks for internal processing. Must be ``2
      * reduce_group_size``.
    topk_wgt_zero_nan: If True, treat zero ``topk_weights`` as indicators of NaN
      during multiplication, resulting in zero output.

  Returns:
    The reduced result as a bf16 matrix [M / reduce_group_size, K].
  """

    sc_info = pltpu.get_tpu_info().sparse_core
    if sc_info is None:
        raise RuntimeError("SparseCore is not available on this TPU version.")

    [M] = idx.shape
    _, K = op.shape
    M_out = M // reduce_group_size

    if topk_weights is not None:
        topk_weights = topk_weights.flatten()

    @jax.jit
    @pl.kernel(
        out_type=jax.ShapeDtypeStruct((M_out, K), op.dtype),
        mesh=plsc.VectorSubcoreMesh(
            core_axis_name="core",
            subcore_axis_name="subcore",
            num_cores=1 if single_sc else sc_info.num_cores,
        ),
        compiler_params=pltpu.CompilerParams(
            use_tc_tiling_on_sc=True,
            needs_layout_passes=True,
        ),
    )
    def kernel(in_hbm_ref, idx_hbm_ref, weights_hbm_ref, out_hbm_ref):
        row_wave_size = row_chunk_size * lax.axis_size(("core", "subcore"))
        if M % row_wave_size:
            raise NotImplementedError(
                f"{M=} must be divisible by {row_chunk_size=} *"
                f" num_cores={lax.axis_size('core')} *"
                f" num_vector_subcores={lax.axis_size('subcore')} = {row_wave_size}"
            )
        num_row_chunks = M // row_wave_size
        num_col_chunks = K // col_chunk_size
        packing = 32 // jax.dtypes.itemsize_bits(op.dtype)

        subcore_first_row_chunk = (lax.axis_index(
            ("core", "subcore")) * num_row_chunks)

        in_spec = pl.BlockSpec((row_chunk_size, ), lambda i:
                               (subcore_first_row_chunk + i, ))
        in_specs = (in_spec, ) * (1 + (weights_hbm_ref is not None))

        @functools.partial(pltpu.emit_pipeline,
                           grid=(num_row_chunks, ),
                           in_specs=in_specs)
        def idx_pipeline(idx_ref, weights_ref=None):
            row_chunk_idx = subcore_first_row_chunk + pl.program_id(0)

            row_subchunk_size = sc_info.num_lanes
            out_rows_per_step = row_subchunk_size // reduce_group_size
            assert reduce_group_size * out_rows_per_step == sc_info.num_lanes
            num_row_subchunks = row_chunk_size // row_subchunk_size
            if row_chunk_size % row_subchunk_size:
                raise ValueError(
                    f"row_chunk_size needs to be a multiple of {row_subchunk_size}, but"
                    f" got {row_chunk_size}")

            @functools.partial(
                pltpu.emit_pipeline,
                grid=(num_row_subchunks, num_col_chunks),
                in_specs=pl.BlockSpec(
                    (pl.Indirect(row_subchunk_size), col_chunk_size),
                    lambda r, c: (
                        lax.div(
                            idx_ref[pl.ds(r * row_subchunk_size,
                                          row_subchunk_size)],
                            packing,
                        ),
                        c,
                    ),
                ),
                out_specs=pl.BlockSpec(
                    (out_rows_per_step // packing, col_chunk_size),
                    lambda r, c: (row_chunk_idx * num_row_subchunks + r, c),
                ),
            )
            def data_pipeline(gather_ref, out_ref):
                gather_ref = gather_ref.bitcast(op.dtype)
                out_ref = out_ref.bitcast(op.dtype)

                row_slice = pl.ds(
                    pl.program_id(0) * row_subchunk_size, row_subchunk_size)
                subchunk_idxs = idx_ref[row_slice]
                weights = (None if weights_ref is None else
                           weights_ref[row_slice].astype(jnp.float32))

                unpack_col_chunk = 32  # 32 seems to works best when tuning.

                @plsc.parallel_loop(0, col_chunk_size, step=unpack_col_chunk)
                def _(col_base):
                    accs = []
                    for reduce_group in range(out_rows_per_step):
                        acc = jnp.zeros((unpack_col_chunk, ),
                                        dtype=jnp.float32)
                        for row_in_group in range(reduce_group_size):
                            row = reduce_group * reduce_group_size + row_in_group
                            row_data = gather_ref[
                                pl.ds(row * packing, packing),
                                pl.ds(col_base, unpack_col_chunk),
                            ].astype(jnp.float32)
                            if packing == 1:
                                row_data = row_data[0]
                            else:
                                assert packing == 2
                                # For dtypes narrower than 32-bit, we end up gathering multiple
                                # rows (since we had to bitcast to int32 before the gather).
                                # This uses the remainder of the packing to choose the only row
                                # we actually care about.
                                row_data = jnp.where(
                                    lax.rem(subchunk_idxs[row], 2) == 0,
                                    row_data[0],
                                    row_data[1],
                                )
                            if weights is not None:
                                row_data *= weights[row]
                                if topk_wgt_zero_nan:
                                    row_data = jnp.where(
                                        weights[row] == 0.0,
                                        jnp.zeros_like(row_data), row_data)
                            acc += row_data
                        accs.append(acc)
                    out = jnp.stack(accs, axis=0).astype(op.dtype)
                    out_ref[:, pl.ds(col_base, unpack_col_chunk)] = out

            data_pipeline(in_hbm_ref.bitcast(jnp.int32),
                          out_hbm_ref.bitcast(jnp.int32))

        idx_pipeline(
            idx_hbm_ref,
            *([weights_hbm_ref] if weights_hbm_ref is not None else []))

    return kernel(op, idx, topk_weights)  # pylint: disable=no-value-for-parameter


def _jax_fallback(x,
                  indices,
                  topk_weights,
                  reduce_group_size,
                  topk_wgt_zero_nan=False):
    token_hidden_full = x[indices]
    cur_sorted = token_hidden_full.reshape(
        (-1, reduce_group_size, x.shape[-1]))
    # topk_weights is already 2D [tokens, reduce_group_size]
    cur_topk_weights = jnp.expand_dims(topk_weights, axis=-1)
    # Accumulate in float32 to match reference precision and Pallas kernel
    # behavior.
    if topk_wgt_zero_nan:
        cur_weighted = jnp.where(
            cur_topk_weights == 0.0,
            0.0,
            cur_sorted.astype(jnp.float32) *
            cur_topk_weights.astype(jnp.float32),
        )
    else:
        cur_weighted = cur_sorted.astype(
            jnp.float32) * cur_topk_weights.astype(jnp.float32)
    out = cur_weighted.sum(axis=-2)
    return out.astype(x.dtype)


def _choose_col_chunk(hidden_size: int) -> tuple[int, int]:
    """Pick the SC column-chunk width and padded hidden size for the kernel.

    The kernel slices the operand along the hidden (column) dimension, which
    carries a 128-wide lane tile in the HBM layout (#tpu.tiled<(4, 128)>).
    Mosaic requires every column chunk to be a multiple of that 128 tile, else
    it rejects the tpu.memref_slice at compile time with "Slice sizes along
    tiled dimensions must be aligned to tiles".

    Returns (col_chunk, hidden_pad):
    - hidden_size % 128 == 0: pick the largest 128-multiple <= min(2048,
      hidden_size) that divides hidden_size exactly. No padding
      (hidden_pad == hidden_size). This reproduces the original selection for
      every 128-aligned model, so existing models see no behavior change.
    - hidden_size % 128 != 0 (e.g. gpt-oss-120b-BF16 hidden_size=2880): no
      128-multiple divides it, so pad up to hidden_pad = num_chunks * col_chunk
      (col_chunk a 128-multiple). The caller runs the kernel over hidden_pad
      columns then slices the output back to [:, :hidden_size]; padded columns
      are zero and, since columns are independent in the gather-reduce, do not
      affect the real columns.
    """
    if hidden_size % 128 == 0:
        col_chunk = (min(2048, hidden_size) // 128) * 128
        while hidden_size % col_chunk != 0:
            col_chunk -= 128
        return col_chunk, hidden_size
    num_chunks = align_to(hidden_size, 2048) // 2048  # cdiv(hidden_size, 2048)
    col_chunk = align_to((hidden_size + num_chunks - 1) // num_chunks, 128)
    return col_chunk, num_chunks * col_chunk


@jax.jit(static_argnames=("reduce_group_size", "topk_wgt_zero_nan"))
def dense_gather_reduce(
    x: jax.Array,
    indices: jax.Array,
    topk_weights: jax.Array,
    reduce_group_size: int,
    topk_wgt_zero_nan: bool = False,
) -> jax.Array:
    """Wrapper that redirects to Pallas dense gather reduce kernel if constraints are met.

  Otherwise, it falls back to the JAX baseline.

  Args:
    x: Input array [out_size, hidden_size].
    indices: Gather indices [out_size].
    topk_weights: 2D weights [tokens, reduce_group_size], where tokens *
      reduce_group_size = out_size.
    reduce_group_size: Group size for reduction (topk).
    topk_wgt_zero_nan: If True, treat zero weights as indicators of NaN during
      multiplication, resulting in zero output.
  """
    if is_compatible(x, indices, reduce_group_size):
        K = x.shape[-1]
        # Pick a 128-tile-aligned column chunk; pad the hidden dim up to K_pad
        # only when K itself is not 128-aligned (see _choose_col_chunk).
        col_chunk, K_pad = _choose_col_chunk(K)
        x_in = jnp.pad(x, ((0, 0), (0, K_pad - K))) if K_pad > K else x
        out = _sc_gather_reduce(
            x_in,
            indices,
            topk_weights.reshape(-1),
            reduce_group_size=reduce_group_size,
            col_chunk_size=col_chunk,
            topk_wgt_zero_nan=topk_wgt_zero_nan,
        )
        return out[:, :K]
    # Fallback to JAX baseline
    return _jax_fallback(x, indices, topk_weights, reduce_group_size,
                         topk_wgt_zero_nan)
