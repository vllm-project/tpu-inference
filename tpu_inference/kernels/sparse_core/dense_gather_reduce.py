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


def _get_kernel_out_dtype(
    dtype: jnp.dtype,
    num_lanes: int,
    reduce_group_size: int,
) -> jnp.dtype:
    """Returns the dtype of the kernel's *output* buffer.

  The kernel's output block has ``(num_lanes // reduce_group_size) //
  out_packing`` rows, where ``out_packing = 32 // bits(out_dtype)``. When a
  SIMD step produces fewer rows than the packing factor (e.g. 8 lanes on v6e
  with topk=8 produces 1 row, while bf16 has packing=2), integer division
  floors that to 0 rows, which Mosaic cannot lower. Writing the output buffer
  as FP32 gives ``out_packing = 1`` and a valid 1-row block.

  Only the output is widened. The operand keeps its native dtype: the gather
  side already addresses HBM in 32-bit words (a bf16 gather fetches the whole
  word and selects a half by index parity), so upcasting the operand would not
  save any gather bytes while materialising a full FP32 copy of an array that
  is ``reduce_group_size`` times larger than the output.

  ``is_compatible`` and ``dense_gather_reduce`` must agree on this decision:
  the former predicts the output packing the kernel will use, the latter
  requests the matching output dtype. Keep them driven by this single helper.

  Args:
    dtype: dtype of the operand as supplied by the caller.
    num_lanes: Number of SparseCore SIMD lanes.
    reduce_group_size: Number of gathered rows summed per output row.

  Returns:
    The dtype for the kernel output buffer (jnp.float32 when the native dtype
    would floor the output block to zero rows, otherwise dtype).
  """
    native_packing = 32 // jax.dtypes.itemsize_bits(dtype)
    simd_rows = num_lanes // reduce_group_size
    if (simd_rows // native_packing < 1) and simd_rows >= 1:
        return jnp.float32
    return dtype


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

    out_dtype = _get_kernel_out_dtype(op.dtype, sc_info.num_lanes,
                                      reduce_group_size)
    out_packing = 32 // jax.dtypes.itemsize_bits(out_dtype)
    # Unreachable for the bf16/f32 operands allowed above, but kept as a guard
    # for narrower dtypes (e.g. fp8 -> packing=4) or future lane geometries.
    if (sc_info.num_lanes // reduce_group_size) // out_packing < 1:
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
    out_dtype: jnp.dtype | None = None,
    single_sc: bool = False,
    col_chunk_size: int = int(3.5 * 1024),
    row_chunk_size: int = 512,
    topk_wgt_zero_nan: bool = False,
) -> jax.Array:
    """Performs a gather-reduce operation on SparseCore.

  This kernel groups rows of the operand ``op`` based on ``idx``, sums them
  up, and scatters the results. The gather and add operations are performed
  in fp32, and the results are written back in ``out_dtype``.

  Equivalent JAX code::

    gathered = op[idx, :]
    if topk_weights is not None:
      flat_weights = topk_weights.flatten()
      gathered = gathered * flat_weights[:, None].astype(jnp.float32)
    gathered = jnp.reshape(gathered, (-1, reduce_group_size, op.shape[1]))
    output = jnp.sum(gathered.astype(jnp.float32), axis=1).astype(out_dtype)

  Args:
    op: The operand matrix [B, K] in f32 or bf16 to gather from and reduce.
    idx: The indices [M,] in int32 guiding the gather.
    topk_weights: Optional weights [M // 128, 128] in bf16 to apply to the
      gathered rows before reduction.
    reduce_group_size: The number of gathered rows to sum per output row.
    out_dtype: dtype of the output buffer. Defaults to ``op.dtype``. This is
      independent of the operand dtype: the operand dtype sets the gather
      packing while ``out_dtype`` sets the output block packing. Widening only
      the output (see ``_get_kernel_out_dtype``) is what makes 8-lane
      SparseCore viable at ``reduce_group_size == num_lanes``.
    single_sc: Whether to use a single SparseCore.
    col_chunk_size: The size of column chunks to process.
    row_chunk_size: The size of row chunks for internal processing. Must be ``2
      * reduce_group_size``.
    topk_wgt_zero_nan: If True, treat zero ``topk_weights`` as indicators of NaN
      during multiplication, resulting in zero output.

  Returns:
    The reduced result as an ``out_dtype`` matrix [M / reduce_group_size, K].
  """
    out_dtype = op.dtype if out_dtype is None else jnp.dtype(out_dtype)

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
        out_type=jax.ShapeDtypeStruct((M_out, K), out_dtype),
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
        # The gather side and the output side pack independently. ``in_packing``
        # is how many operand rows share one 32-bit word in HBM (it drives the
        # indirect gather addressing); ``out_packing`` is how many output rows
        # share one 32-bit word (it drives the output block row count). Only the
        # latter can floor to a zero-row block, so only the latter is widened.
        in_packing = 32 // jax.dtypes.itemsize_bits(op.dtype)
        out_packing = 32 // jax.dtypes.itemsize_bits(out_dtype)

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
                            in_packing,
                        ),
                        c,
                    ),
                ),
                out_specs=pl.BlockSpec(
                    (out_rows_per_step // out_packing, col_chunk_size),
                    lambda r, c: (row_chunk_idx * num_row_subchunks + r, c),
                ),
            )
            def data_pipeline(gather_ref, out_ref):
                gather_ref = gather_ref.bitcast(op.dtype)
                out_ref = out_ref.bitcast(out_dtype)

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
                        row_datas = []
                        for row_in_group in range(reduce_group_size):
                            row = reduce_group * reduce_group_size + row_in_group
                            row_data = gather_ref[
                                pl.ds(row * in_packing, in_packing),
                                pl.ds(col_base, unpack_col_chunk),
                            ].astype(jnp.float32)
                            if in_packing == 1:
                                row_data = row_data[0]
                            else:
                                assert in_packing == 2
                                row_data = jnp.where(
                                    lax.bitwise_and(subchunk_idxs[row],
                                                    1) == 0,
                                    row_data[0],
                                    row_data[1],
                                )
                            if weights is not None:
                                row_data *= weights[row]
                                if topk_wgt_zero_nan:
                                    row_data = jnp.where(
                                        weights[row] == 0.0,
                                        jnp.zeros_like(row_data), row_data)
                            row_datas.append(row_data)

                        # Tree reduction to reduce critical path and stalls
                        while len(row_datas) > 1:
                            next_level = []
                            for i in range(0, len(row_datas), 2):
                                if i + 1 < len(row_datas):
                                    next_level.append(row_datas[i] +
                                                      row_datas[i + 1])
                                else:
                                    next_level.append(row_datas[i])
                            row_datas = next_level
                        accs.append(row_datas[0])
                    out = jnp.stack(accs, axis=0).astype(out_dtype)
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
        sc_info = pltpu.get_tpu_info().sparse_core
        K = x.shape[-1]
        # The kernel slices the operand along the hidden (column) dimension,
        # which carries a 128-wide lane tile in the HBM layout
        # (#tpu.tiled<(4, 128)>). A column chunk that is not a multiple of 128
        # produces a tpu.memref_slice whose size along the tiled dimension is
        # not tile-aligned, which Mosaic rejects at compile time with
        # "Slice sizes along tiled dimensions must be aligned to tiles" (e.g.
        # hidden_size=2880 -> chunk 1440, and 1440 % 128 = 32). Require the
        # chunk to be a multiple of the 128 lane tile; when 128 does not divide
        # hidden_size (as for gpt-oss's 2880) no valid chunk exists and we fall
        # back to the JAX implementation below.
        col_chunk_size = (min(2048, K) // 128) * 128
        while col_chunk_size > 0:
            if K % col_chunk_size == 0:
                break
            col_chunk_size -= 128
        if col_chunk_size > 0:
            # Same decision is_compatible() used to pick the output packing;
            # the two must match or the kernel emits a zero-row output block.
            # Note this widens the *output buffer only*. The operand is passed
            # through untouched: the gather addresses HBM in 32-bit words, so a
            # bf16 gather already fetches the whole word and upcasting x would
            # cost a full fp32 copy of an array reduce_group_size times larger
            # than the output while fetching exactly the same bytes.
            out_dtype = _get_kernel_out_dtype(x.dtype, sc_info.num_lanes,
                                              reduce_group_size)
            # Pallas kernel expects 1D weights
            res = _sc_gather_reduce(
                x,
                indices,
                topk_weights.reshape(-1),
                reduce_group_size=reduce_group_size,
                out_dtype=out_dtype,
                col_chunk_size=col_chunk_size,
                topk_wgt_zero_nan=topk_wgt_zero_nan,
            )
            # Cheap elementwise convert on the small output; XLA generally
            # fuses it into the consumer. No-op when out_dtype == x.dtype.
            return res.astype(x.dtype) if res.dtype != x.dtype else res
    # Fallback to JAX baseline
    return _jax_fallback(x, indices, topk_weights, reduce_group_size,
                         topk_wgt_zero_nan)
