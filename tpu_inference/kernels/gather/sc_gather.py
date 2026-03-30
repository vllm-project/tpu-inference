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

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.experimental.pallas import tpu_sc as plsc


def main_kernel(
    # Inputs.
    start_ref: jax.Ref,
    end_ref: jax.Ref,
    in_hbm_ref: jax.Ref,
    indices_hbm_ref: jax.Ref,
    # Outputs.
    out_hbm_ref: jax.Ref,
    # Scratch.
    start_vmem_ref: jax.Ref,
    end_vmem_ref: jax.Ref,
    out_vmem_ref: jax.Ref,
    indices_vmem_ref: jax.Ref,
    sem_ref: jax.Ref,
    *,
    core_axis_name: str = "core",
    subcore_axis_name: str = "subcore",
):
    tpu_info = pltpu.get_tpu_info()
    sc_info = tpu_info.sparse_core
    assert sc_info is not None
    num_simd_lanes = sc_info.num_lanes
    num_lanes = tpu_info.num_lanes
    hidden_size = in_hbm_ref.shape[-1]
    col_size = out_vmem_ref.shape[-1]
    num_cores = jax.lax.axis_size((core_axis_name, subcore_axis_name))
    block_size = num_simd_lanes * num_cores

    # Read start and end tensor values.
    dma_list = []
    dma = pltpu.make_async_copy(start_ref.at[:1], start_vmem_ref.at[:1],
                                sem_ref.at[0])
    dma_list.append(dma)
    dma = pltpu.make_async_copy(end_ref.at[:1], end_vmem_ref.at[:1],
                                sem_ref.at[0])
    dma_list.append(dma)

    jax.tree.map(lambda x: x.start(), dma_list)
    jax.tree.map(lambda x: x.wait(), dma_list)

    # Calculate number of tiles to visit using start and end arrays.
    start = start_vmem_ref[...][0]
    end = end_vmem_ref[...][0]

    block_start = start // block_size
    block_end = pl.cdiv(end, block_size)
    num_blocks = block_end - block_start
    num_blocks = jnp.where(end == start, 0, num_blocks)
    aligned_start = block_start * block_size

    num_cols = pl.cdiv(hidden_size, col_size)

    def inner_kernel():
        block_id = pl.program_id(0)
        core_id = pl.program_id(1)
        col_id = pl.program_id(2)

        row_tile_start = (aligned_start + block_id * block_size +
                          core_id * num_simd_lanes)
        col_tile_start = col_id * col_size

        @pl.when(col_id == 0)
        def _():
            pltpu.sync_copy(
                indices_hbm_ref.at[pl.ds(row_tile_start, num_simd_lanes)],
                indices_vmem_ref,
            )

        # HBM to VMEM transfer.
        indices = indices_vmem_ref[...]

        dtype = out_hbm_ref.dtype
        packing = 4 // out_hbm_ref.dtype.itemsize
        dtype_bits = jax.dtypes.itemsize_bits(dtype)

        # To fetch only one sublane at a time, we need to use (packing, 128) layout.
        # But, the inputs are in (8, 128) layout and thus we need to perform
        # relayout. For 32-bits, this can be done with a simple reinterpretation,
        # but for other bitwidths, this is not possible. Therefore, we bitcast data
        # into 32-bits first to fetch packing number of rows per dma and later
        # perform bitwise unpacking / packing to obtain desired results.
        in_src_hbm_ref = in_hbm_ref.bitcast(jnp.uint32)
        out_dst_hbm_ref = out_hbm_ref.bitcast(jnp.uint32)

        for col_vmem_start in range(0, col_size, num_lanes):
            col_hbm_start = col_tile_start + col_vmem_start
            for row_vmem in range(num_simd_lanes):
                row_hbm = indices[row_vmem] // packing
                pltpu.make_async_copy(
                    in_src_hbm_ref.at[row_hbm,
                                      pl.ds(col_hbm_start, num_lanes)],
                    out_vmem_ref.at[row_vmem,
                                    pl.ds(col_vmem_start, num_lanes)],
                    sem_ref.at[0],
                ).start()

        # VMEM to HBM transfer.
        # Use dynamic loop to minimize register spills.
        @pl.loop(0, col_size, step=num_lanes)
        @jax.named_scope("dma_write_loop")
        def dma_write_loop(col_vmem_start):
            col_hbm_start = col_tile_start + col_vmem_start

            # Wait for data to be received.
            for row_vmem in range(num_simd_lanes):
                row_hbm = indices[row_vmem] // packing
                pltpu.make_async_copy(
                    in_src_hbm_ref.at[row_hbm,
                                      pl.ds(col_hbm_start, num_lanes)],
                    out_vmem_ref.at[row_vmem,
                                    pl.ds(col_vmem_start, num_lanes)],
                    sem_ref.at[0],
                ).wait()

            # If multiple elements are packed in single 32-bits, extract the desired
            # elements and reorder them.
            if packing > 1:
                # TODO(kyuyeunk): Add support for smaller bitwidths.
                assert packing == 2
                for col_compute_offset in range(0, num_lanes, num_simd_lanes):
                    col_slice = pl.ds(col_vmem_start + col_compute_offset,
                                      num_simd_lanes)

                    out = None
                    for row_src in range(num_simd_lanes):
                        row_pack = row_src % packing
                        offset = indices[row_src] % packing
                        bitshift = row_pack * dtype_bits

                        # Load data from vmem.
                        data = out_vmem_ref[row_src, col_slice]

                        # Split data into upper and lower 16 bits.
                        upper = jnp.bitwise_right_shift(data, dtype_bits)
                        lower = data

                        # Choose correct data using index offset.
                        data = jnp.where(offset == 0, lower, upper)
                        # Zero out upper 16-bits.
                        data = jnp.bitwise_and(data, (2**dtype_bits) - 1)
                        # Move bits to the offset position.
                        data = jnp.bitwise_left_shift(data, bitshift)

                        if row_pack == 0:
                            out = data
                        elif row_pack == 1:
                            out = jnp.bitwise_or(out, data)
                            # Store packed data into correct position.
                            row_dst = row_src // packing
                            out_vmem_ref[row_dst, col_slice] = out
                        else:
                            raise ValueError("Packing factor must be 2.")

            # Start dma write.
            for row_vmem in range(num_simd_lanes // packing):
                row_hbm = row_tile_start // packing + row_vmem
                pltpu.make_async_copy(
                    out_vmem_ref.at[row_vmem,
                                    pl.ds(col_vmem_start, num_lanes)],
                    out_dst_hbm_ref.at[row_hbm,
                                       pl.ds(col_hbm_start, num_lanes)],
                    sem_ref.at[0],
                ).start()

        # Wait for dma write to finish.
        for col_vmem_start in range(0, col_size, num_lanes):
            col_hbm_start = col_tile_start + col_vmem_start
            for src in range(num_simd_lanes // packing):
                dst = row_tile_start // packing + src
                pltpu.make_async_copy(
                    out_vmem_ref.at[src, pl.ds(col_vmem_start, num_lanes)],
                    out_dst_hbm_ref.at[dst,
                                       pl.ds(col_hbm_start, num_lanes)],
                    sem_ref.at[0],
                ).wait()

    pltpu.emit_pipeline(
        inner_kernel,
        grid=(num_blocks, num_cores, num_cols),
        core_axis_name=(core_axis_name, subcore_axis_name),
        dimension_semantics=(pltpu.ARBITRARY, pltpu.PARALLEL, pltpu.ARBITRARY),
    )()


@jax.jit
def ragged_gather(x: jax.Array, indices: jax.Array, start: jax.Array,
                  end: jax.Array) -> jax.Array:
    """Perform gather on indices within dynamic array start and end."""

    assert x.dtype in (jnp.float32, jnp.bfloat16)
    assert x.ndim == 2
    assert indices.ndim == 1

    if jnp.isscalar(start):
        start = start[None]
    if jnp.isscalar(end):
        end = end[None]

    dtype = x.dtype

    sc_info = pltpu.get_tpu_info().sparse_core
    assert sc_info is not None

    hidden_size = x.shape[-1]
    out_size = indices.size

    num_simd_lanes = sc_info.num_lanes
    num_cores = sc_info.num_cores * sc_info.num_subcores
    if out_size < (num_simd_lanes * num_cores):
        return x[indices]

    assert out_size % (num_simd_lanes * num_cores) == 0
    col_size = hidden_size

    vector_mesh = plsc.VectorSubcoreMesh(
        num_cores=sc_info.num_cores,
        num_subcores=sc_info.num_subcores,
        core_axis_name="core",
        subcore_axis_name="subcore",
    )
    return pl.kernel(
        main_kernel,
        out_shape=jax.ShapeDtypeStruct((out_size, hidden_size), dtype),
        compiler_params=pltpu.CompilerParams(
            use_tc_tiling_on_sc=True,
            disable_bounds_checks=True,
        ),
        scratch_shapes=[
            pltpu.VMEM((16, ), jnp.int32),
            pltpu.VMEM((16, ), jnp.int32),
            pltpu.VMEM((num_simd_lanes, col_size), jnp.uint32),
            pltpu.VMEM((num_simd_lanes, ), jnp.int32),
            pltpu.SemaphoreType.DMA((1, )),
        ],
        mesh=vector_mesh,
        name="sc_ragged_gather",
    )(start, end, x, indices)
