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

import dataclasses
import functools
from typing import Any, Callable, Tuple

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

# Define data classes.


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class MetadataRef:
    gm_id_to_group_id: jax.Array
    gm_id_to_m_offset: jax.Array


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class WeightsRef:
    weight: Any
    scale: Any | None
    bias: Any | None


@dataclasses.dataclass(frozen=True)
class TileSizes:
    tile_m: int
    tile_k: int
    tile_n: int


@dataclasses.dataclass(frozen=True)
class Dimensions:
    size_m: int
    size_k: int
    size_n: int
    size_group: int
    size_lhs_group: int
    size_lhs_sublane: int
    quant_block_size: int = 0
    num_quant_blocks: int = 0
    rhs_packing: int = 1
    rhs_dtype: jnp.dtype | None = None


@dataclasses.dataclass(frozen=True)
class InputConfigs:
    quant_dtype: jnp.dtype | None
    quant_block_size: int | None
    has_bias: bool = False
    has_scale: bool = False


@dataclasses.dataclass(frozen=True)
class GmmConfigs:
    tiles: TileSizes
    dims: Dimensions
    lhs_cfgs: InputConfigs
    rhs_cfgs: InputConfigs
    out_dtype: jnp.dtype
    acc_dtype: jnp.dtype
    zero_init: bool


TileFn = Callable[[jnp.dtype, jnp.dtype, Dimensions, int], TileSizes]


class IndexMaps:
    """Index maps for GMM kernel."""

    def __init__(self, metadata_ref: MetadataRef, cfgs: GmmConfigs):
        self.metadata_ref = metadata_ref
        self.cfgs = cfgs

    def lhs_index_map(self, _: jax.Array, gm_id: jax.Array, k_id: jax.Array):
        m_start = self.metadata_ref.gm_id_to_m_offset[gm_id]
        m_end = self.metadata_ref.gm_id_to_m_offset[gm_id + 1]

        row_start = m_start // self.cfgs.dims.size_lhs_sublane
        row_end = pl.cdiv(m_end, self.cfgs.dims.size_lhs_sublane)
        row_size = row_end - row_start

        return (pl.ds(row_start, row_size), 0, k_id)

    def rhs_weight_index_map(self, n_id: jax.Array, gm_id: jax.Array,
                             k_id: jax.Array):
        group_id = self.metadata_ref.gm_id_to_group_id[gm_id]
        return (group_id, k_id, n_id)

    def rhs_bias_index_map(self, n_id: jax.Array, gm_id: jax.Array,
                           _: jax.Array):
        group_id = self.metadata_ref.gm_id_to_group_id[gm_id]
        return (group_id, 0, n_id)

    def rhs_scale_index_map(self, n_id: jax.Array, gm_id: jax.Array,
                            k_id: jax.Array):
        group_id = self.metadata_ref.gm_id_to_group_id[gm_id]
        num_quant_blocks_per_tile_k = pl.cdiv(self.cfgs.tiles.tile_k,
                                              self.cfgs.dims.quant_block_size)
        b_i = (k_id * self.cfgs.tiles.tile_k) // self.cfgs.dims.quant_block_size
        b_tile_i = b_i // num_quant_blocks_per_tile_k
        return (group_id, b_tile_i, 0, n_id)

    def out_index_map(self, n_id: jax.Array, gm_id: jax.Array, _: jax.Array):
        is_last_gm = gm_id == (pl.num_programs(1) - 1)
        m_start = self.metadata_ref.gm_id_to_m_offset[gm_id]
        m_end = self.metadata_ref.gm_id_to_m_offset[gm_id + 1]

        row_start = m_start // self.cfgs.dims.size_lhs_sublane
        capped_row_end = m_end // self.cfgs.dims.size_lhs_sublane
        last_row_end = pl.cdiv(m_end, self.cfgs.dims.size_lhs_sublane)
        row_end = jnp.where(is_last_gm, last_row_end, capped_row_end)
        row_size = row_end - row_start

        return (pl.ds(row_start, row_size), 0, n_id)


def generate_block_specs(
        metadata_ref: MetadataRef, cfgs: GmmConfigs
) -> Tuple[Tuple[pl.BlockSpec, WeightsRef], pl.BlockSpec]:
    """Generates block specs for the given lhs, rhs, and out refs."""

    index_map = IndexMaps(metadata_ref, cfgs)
    bounded_slice_gm = pl.BoundedSlice(cfgs.tiles.tile_m //
                                       cfgs.dims.size_lhs_sublane)

    lhs_block_spec = pl.BlockSpec(
        (bounded_slice_gm, cfgs.dims.size_lhs_sublane, cfgs.tiles.tile_k),
        index_map.lhs_index_map,
    )

    rhs_weight_spec = pl.BlockSpec(
        (None, cfgs.tiles.tile_k // cfgs.dims.rhs_packing, cfgs.tiles.tile_n),
        index_map.rhs_weight_index_map,
        pipeline_mode=pl.Buffered(buffer_count=3),
    )
    rhs_scale_block_spec = rhs_bias_block_spec = None
    if cfgs.rhs_cfgs.has_bias:
        rhs_bias_block_spec = pl.BlockSpec(
            (None, 1, cfgs.tiles.tile_n),
            index_map.rhs_bias_index_map,
        )
    if cfgs.rhs_cfgs.has_scale:
        num_quant_blocks_per_tile_k = pl.cdiv(cfgs.tiles.tile_k,
                                              cfgs.dims.quant_block_size)
        rhs_scale_block_spec = pl.BlockSpec(
            (None, num_quant_blocks_per_tile_k, 1, cfgs.tiles.tile_n),
            index_map.rhs_scale_index_map,
        )

    rhs_block_spec = WeightsRef(
        weight=rhs_weight_spec,
        scale=rhs_scale_block_spec,
        bias=rhs_bias_block_spec,
    )

    out_block_spec = pl.BlockSpec(
        (bounded_slice_gm, cfgs.dims.size_lhs_sublane, cfgs.tiles.tile_n),
        index_map.out_index_map,
    )

    return (lhs_block_spec, rhs_block_spec), out_block_spec


# Define kernels.


def inner_kernel(
    # In
    tiled_lhs_ref: jax.Array,
    # [tile_m // size_lhs_sublane, size_lhs_sublane, tile_k]
    tiled_rhs_ref: WeightsRef,  # [tile_k, tile_n]
    # Out
    tiled_out_ref: jax.Array,
    # [tile_m // size_lhs_sublane, size_lhs_sublane, tile_n]
    # Scratch
    partial_out_ref: jax.Array,  # [size_lhs_sublane, tile_n]
    acc_ref: jax.Array,  # [tile_m, tile_n]
    metadata_ref: MetadataRef,
    *,
    cfgs: GmmConfigs,
):
    """Inner kernel invoked by emit_pipeline to perform matmul.

    tiled_lhs_ref and tiled_out_ref points to rows [m_start:m_end] of lhs and out.
    Additionally, m_start and m_end does not have to align with tile boundaries
    [m_offset:m_offset+tile_m]. Therefore, rows [m_offset:m_start] and
    [m_end:m_offset+tile_m] of tiled_lhs_ref and tiled_out_ref will contain
    invalid data and needs to be masked out.

    Args:
        tiled_lhs_ref: Contains value lhs[m_start:m_end, k_start:k_end]
        tiled_rhs_ref: Contains value rhs[g_id, k_start:k_end, n_start:n_end].
            where g_id is the group associated with lhs[m_start:m_end, :]
        tiled_out_ref: Contains value out[m_start:m_end, n_start:n_end]
        partial_out_ref: Contains last size_lhs_sublane rows of the previous
            output.  If this is the first tile for grid[n_id, :, :], it will be
            initialized to zeros.
        acc_ref: Reference to the accumulator.
        metadata_ref: Reference to the metadata.
        cfgs: GmmConfigs.
    """

    def _matmul(is_first_k_step: bool, is_last_k_step: bool):
        tiled_lhs = tiled_lhs_ref.reshape(-1, cfgs.tiles.tile_k)[...]
        num_quant_blocks_per_tile_k = pl.cdiv(cfgs.tiles.tile_k,
                                              cfgs.dims.quant_block_size)

        # When rhs is packed, unpack it back to the original dtype using
        # pltpu.bitcast which operates on K axis.
        # This doubles the K dimension back to tile_k.
        weight = tiled_rhs_ref.weight[...]
        if cfgs.dims.rhs_packing > 1:
            weight = pltpu.bitcast(weight, cfgs.dims.rhs_dtype)

        if cfgs.lhs_cfgs.quant_dtype is None:
            # Unquantized matmul path.
            acc = jnp.zeros((cfgs.tiles.tile_m, cfgs.tiles.tile_n),
                            dtype=acc_ref.dtype)
            for b_i in range(num_quant_blocks_per_tile_k):
                k_start = b_i * cfgs.dims.quant_block_size
                k_end = (b_i + 1) * cfgs.dims.quant_block_size
                partial_result = jnp.matmul(
                    tiled_lhs[:, k_start:k_end],
                    weight[k_start:k_end, :],
                    preferred_element_type=jnp.float32,
                )
                if tiled_rhs_ref.scale is not None:
                    partial_result *= jnp.broadcast_to(
                        tiled_rhs_ref.scale[..., b_i:b_i + 1, :, :].reshape(
                            1, cfgs.tiles.tile_n),
                        partial_result.shape,
                    )
                acc = acc + partial_result
            acc = acc.astype(acc_ref.dtype)
        else:
            # Quantized matmul path.
            lhs_q_dtype = cfgs.lhs_cfgs.quant_dtype
            q_block_size = cfgs.lhs_cfgs.quant_block_size

            if jnp.issubdtype(lhs_q_dtype, jnp.floating):
                dtype_max = float(jnp.finfo(lhs_q_dtype).max)
                preferred_element_type = jnp.float32
            else:
                dtype_max = float(jnp.iinfo(lhs_q_dtype).max)
                preferred_element_type = jnp.int32

            # Without n outer loop, result of quantized matmul becomes available only
            # at the last iteration of the loop. This means [tile_m, tile_n] value
            # needs to be stored until the last iteration. By adding n outer loop,
            # result of [tile_m, mxu_size] becomes available at the end of every k
            # inner loop which can be used to pipeline subsequent VPU or VST ops with
            # MXU ops for the next [tile_m, mxu_size].
            acc_list = []
            mxu_size = pltpu.get_tpu_info().mxu_column_size
            for start_n in range(0, cfgs.tiles.tile_n, mxu_size):
                end_n = min(cfgs.tiles.tile_n, start_n + mxu_size)
                col_size = end_n - start_n

                acc_n = jnp.zeros((cfgs.tiles.tile_m, col_size),
                                  dtype=acc_ref.dtype)
                rhs_qbs = cfgs.dims.quant_block_size
                for start_k in range(0, cfgs.tiles.tile_k, q_block_size):
                    end_k = min(cfgs.tiles.tile_k, start_k + q_block_size)

                    block_lhs = tiled_lhs[:, start_k:end_k]
                    block_rhs = weight[start_k:end_k, start_n:end_n]

                    # Perform lhs quantization. Note that for every block_lhs,
                    # same computation will be performed tiles_n//mxu_size times.
                    # But we can let compiler perform CSE and avoid recomputation.
                    block_abs_max = jnp.max(jnp.abs(block_lhs),
                                            axis=1,
                                            keepdims=True)

                    # If block_abs_max=0, it will cause division by zero and return NaNs
                    # during quantization. To avoid this, we add smallest representable
                    # value to avoid block_abs_max being zero.
                    block_abs_max += jnp.finfo(
                        block_abs_max.dtype).smallest_subnormal
                    block_scale = block_abs_max / dtype_max
                    # Convert lhs into quantized dtype.
                    block_lhs_q = (block_lhs / block_scale).astype(lhs_q_dtype)

                    block_acc = jnp.matmul(
                        block_lhs_q,
                        block_rhs,
                        preferred_element_type=preferred_element_type,
                    ).astype(acc_ref.dtype)

                    block_acc = block_acc * block_scale.astype(acc_ref.dtype)

                    # Apply rhs subchannel scale per quant block.
                    if tiled_rhs_ref.scale is not None:
                        b_i = start_k // rhs_qbs
                        rhs_scale_block = tiled_rhs_ref.scale[
                            ..., b_i:b_i + 1, :, :].reshape(
                                1, cfgs.tiles.tile_n)
                        block_acc *= rhs_scale_block[:, start_n:end_n].astype(
                            acc_ref.dtype)

                    acc_n += block_acc

                acc_list.append(acc_n)
            acc = jnp.concatenate(acc_list, axis=1)

        if not is_first_k_step:
            acc += acc_ref[...]

        if is_last_k_step:
            if cfgs.rhs_cfgs.has_bias:
                acc += tiled_rhs_ref.bias[...].astype(acc.dtype)

            gm_id = pl.program_id(1)

            # Mask out rows that does not belong to the current group.
            m_start = metadata_ref.gm_id_to_m_offset[gm_id]
            m_end = metadata_ref.gm_id_to_m_offset[gm_id + 1]

            m_offset = m_start - m_start % cfgs.dims.size_lhs_sublane

            m_start_local = m_start - m_offset
            m_end_local = m_end - m_offset

            iota = lax.broadcasted_iota(jnp.int32, acc.shape, 0)
            mask = jnp.logical_and(m_start_local <= iota, iota < m_end_local)
            acc_masked = jnp.where(mask, acc, 0).reshape(tiled_out_ref.shape)

            # Write the final output to the output ref.
            tiled_out_ref[...] = acc_masked.astype(tiled_out_ref.dtype)

            # If this is the first tile for grid[n_id, :, :], we initialize the
            # partial out to zeros. Otherwise, partial out from last tile of
            # grid[n_id-1, :, :] can be used and cause numeric issues.
            partial_out_zeros = jnp.zeros_like(partial_out_ref)

            # Accumulate the partial output from the previous step.
            tiled_out_ref[0] += jnp.where(gm_id == 0, partial_out_zeros,
                                          partial_out_ref[...])

            # Consider following case where size_lhs_sublane = 4, number denotes group
            # id and | denotes boundaries between sublanes:
            # | 0 0 1 2 | 2 2 2 2 | 3 3 4 4 |
            #
            # Assuming group id of current step is 1, current step will not completely
            # fill size_lhs_sublane rows and will be revisited at the next step. By
            # storing the partial rows into the partial_out_ref, the next step can
            # read them and accumulate to them.  Additionally, for group id of 2,
            # since it completely fills the size_lhs_sublane rows, we need to
            # initialize the partial_out_ref to zeros.
            last_row = m_end_local // cfgs.dims.size_lhs_sublane
            partial_out_ref[...] = jnp.where(
                m_end_local % cfgs.dims.size_lhs_sublane == 0,
                partial_out_zeros,
                tiled_out_ref[last_row],
            )
        else:
            acc_ref[...] = acc

    # Define matmul wrapper functions.
    @jax.named_scope("matmul_first_last")
    def matmul_first_last():
        _matmul(is_first_k_step=True, is_last_k_step=True)

    @jax.named_scope("matmul_first")
    def matmul_first():
        _matmul(is_first_k_step=True, is_last_k_step=False)

    @jax.named_scope("matmul")
    def matmul():
        _matmul(is_first_k_step=False, is_last_k_step=False)

    @jax.named_scope("matmul_last")
    def matmul_last():
        _matmul(is_first_k_step=False, is_last_k_step=True)

    # Select and execute matmul function based on the current step.
    num_k = pl.num_programs(2)
    k_id = pl.program_id(2)

    is_first_k_step = k_id == 0
    is_last_k_step = k_id == (num_k - 1)

    lax.cond(
        is_first_k_step,
        lambda: lax.cond(
            is_last_k_step,
            matmul_first_last,
            matmul_first,
        ),
        lambda: lax.cond(
            is_last_k_step,
            matmul_last,
            matmul,
        ),
    )


def fill_metadata(
    lhs_group_sizes_ref: jax.Array,  # int32[size_lhs_group]
    group_offset_ref: jax.Array,  # int32[1]
    metadata_ref: MetadataRef,
    *,
    cfgs: GmmConfigs,
) -> jax.Array:
    """Fills the metadata for the given lhs group sizes and group offset.

    Iterates over the lhs group sizes and if the group id is valid, determines
    the number of gm tiles that are needed to process the current group. Then,
    it fills starting and ending offset (gm_id_to_m_offset), and the group id
    (gm_id_to_group_id) for each gm tile.

    Args:
        lhs_group_sizes_ref: The group sizes of lhs.
        group_offset_ref: Offset of the first group to process.
        metadata_ref: Metadata that is used to determine the group id and m
            offsets for each gmm tile.
        cfgs: GmmConfigs.

    Returns:
        The number of gm tiles to process lhs with given group offset.
    """

    group_offset = group_offset_ref[0]
    max_num_group = group_offset + cfgs.dims.size_group
    metadata_ref.gm_id_to_m_offset[0] = 0

    @jax.named_scope("inner_tm_loop")
    def inner_tm_loop(tm_id, curr_m_offset, *, end_m_offset, group_id, num_gm):
        local_offset = curr_m_offset % cfgs.dims.size_lhs_sublane
        tm_size = jnp.minimum(cfgs.tiles.tile_m - local_offset,
                              end_m_offset - curr_m_offset)

        metadata_ref.gm_id_to_group_id[num_gm + tm_id] = group_id

        next_m_offset = curr_m_offset + tm_size
        metadata_ref.gm_id_to_m_offset[num_gm + tm_id] = curr_m_offset
        metadata_ref.gm_id_to_m_offset[num_gm + tm_id + 1] = next_m_offset

        return next_m_offset

    @jax.named_scope("outer_group_loop")
    def outer_group_loop(lhs_group_id, carry):
        num_gm, start_m_offset = carry

        group_id = lhs_group_id - group_offset
        group_size = lhs_group_sizes_ref[lhs_group_id]
        end_m_offset = start_m_offset + group_size

        # Assume following arguments:
        # - size_lhs_sublane & tile_m = 4
        # - group_size = 3
        # - start_m_offset = 7
        #
        # If we visualize it, it will look like this where:
        # - |: denotes boundaries between sublanes
        # - 0: denotes values for other groups
        # - 1: denotes values for the current group
        # | 0 0 0 0 | 0 0 0 1 | 1 1 0 0 |
        #
        # In this example, we see that we require processing 2 m tiles.
        # But, performing a naive cdiv(group_size, tile_m) will return 1.
        # Instead, adding local_offset will give us the correct value.
        local_offset = start_m_offset % cfgs.dims.size_lhs_sublane
        aligned_group_size = group_size + local_offset
        curr_num_gm = pl.cdiv(aligned_group_size, cfgs.tiles.tile_m)

        # We need to handle cases where we should not process the group.
        # 1. Even if group_size is 0, if local_offset is not 0, cdiv will return 1.
        # 2. If group comes before the group_offset, we should not process it.
        should_process = jnp.logical_and(group_size > 0, group_id >= 0)
        curr_num_gm = jnp.where(should_process, curr_num_gm, 0)

        tm_loop_fn = functools.partial(
            inner_tm_loop,
            end_m_offset=end_m_offset,
            group_id=group_id,
            num_gm=num_gm,
        )
        lax.fori_loop(0, curr_num_gm, tm_loop_fn, start_m_offset)

        return num_gm + curr_num_gm, end_m_offset

    num_gm, _ = lax.fori_loop(0, max_num_group, outer_group_loop, (0, 0))
    return num_gm


def zero_out_start(
    out_ref: jax.Array,  # [size_m, size_n]
    zero_ref: jax.Array,  # [tile_zero_m, num_lanes]
    semaphore_ref: jax.Array,  # [1]
    metadata_ref: MetadataRef,
    num_gm: jax.Array,
    *,
    dims: Dimensions,
):
    """Zero out output rows that are not used in the computation."""

    num_lanes = pltpu.get_tpu_info().num_lanes
    assert num_lanes == zero_ref.shape[-1]
    zero_ref[...] = jnp.zeros_like(zero_ref)

    zero_dma = zero_ref.reshape(-1, dims.size_lhs_sublane, num_lanes)
    out_dma = out_ref.reshape(-1, dims.size_lhs_sublane, dims.size_n)
    row_size = zero_dma.shape[0]

    compute_start = metadata_ref.gm_id_to_m_offset[0]
    compute_end = metadata_ref.gm_id_to_m_offset[num_gm]

    left_zero_start = 0
    left_zero_end = compute_start // dims.size_lhs_sublane
    left_zero_size = left_zero_end - left_zero_start
    left_num_loops = pl.cdiv(left_zero_size, row_size)

    right_zero_start = pl.cdiv(compute_end, dims.size_lhs_sublane)
    right_zero_end = out_dma.shape[0]
    right_zero_size = right_zero_end - right_zero_start
    right_num_loops = pl.cdiv(right_zero_size, row_size)

    def fill_zero(i, zero_size, *, start, end):
        dma_start = start + i * row_size
        dma_end = jnp.minimum(dma_start + row_size, end)
        dma_size = dma_end - dma_start

        # Static loop. Will be unrolled during compile time.
        for n_start in range(0, dims.size_n, num_lanes):
            n_end = min(n_start + num_lanes, dims.size_n)
            pltpu.make_async_copy(
                src_ref=zero_dma.at[pl.ds(0, dma_size), :, :n_end - n_start],
                dst_ref=out_dma.at[pl.ds(dma_start, dma_size), :,
                                   n_start:n_end],
                sem=semaphore_ref.at[0],
            ).start(priority=1)

        return zero_size + dma_size

    @jax.named_scope("left_fill_zero")
    def left_fill_zero(i, zero_size):
        return fill_zero(i,
                         zero_size,
                         start=left_zero_start,
                         end=left_zero_end)

    @jax.named_scope("right_fill_zero")
    def right_fill_zero(i, zero_size):
        return fill_zero(i,
                         zero_size,
                         start=right_zero_start,
                         end=right_zero_end)

    zero_size = lax.fori_loop(0, left_num_loops, left_fill_zero, 0)
    zero_size = lax.fori_loop(0, right_num_loops, right_fill_zero, zero_size)
    return zero_size


def zero_out_end(
    out_ref: jax.Array,  # [size_m, size_n]
    semaphore_ref: jax.Array,  # [1]
    zero_size: jax.Array,
    *,
    dims: Dimensions,
):
    out_dma = out_ref.reshape(-1, dims.size_lhs_sublane, dims.size_n)
    pltpu.make_async_copy(
        src_ref=out_dma.at[pl.ds(0, zero_size)],
        dst_ref=out_dma.at[pl.ds(0, zero_size)],
        sem=semaphore_ref.at[0],
    ).wait()


def kernel_main(
    # Scalar prefetch
    lhs_group_sizes_ref: jax.Array,  # int32[size_lhs_group]
    group_offset_ref: jax.Array,  # int32[1]
    # In
    lhs_ref: jax.Array,  # [size_m, size_k]
    rhs_ref: WeightsRef,  # [size_group, size_k, size_n]
    # Out
    out_ref: jax.Array,  # [size_m, size_n]
    # Scratch memory
    partial_out_ref: jax.Array,  # [size_lhs_sublane, tile_n]
    acc_ref: jax.Array,  # [tile_m, tile_n]
    metadata_ref: MetadataRef,
    zero_ref: jax.Array | None,  # [tile_zero_m, num_lanes]
    semaphore_ref: jax.Array | None,  # [1]
    *,
    cfgs: GmmConfigs,
):
    """Entry point for GMM kernel.

    Computes metadata to determine which rows of lhs needs processing and how
    they will be tiled. And then, invoke inner kernel using metadata.

    Uses the following notation:
    - g: rhs group dimension
    - m: Batch dimension
    - gm: Batch tiling dimension. Aligned to size_lhs_sublane and has tile size
        of tile_m. Skips over empty groups and accounts for revisited tiles.
    - k: in dimension
    - n: out dimension

    Args:
        lhs_group_sizes_ref: Reference to the group sizes of lhs.
        group_offset_ref: Reference to the group offset.
        lhs_ref: Reference to the lhs.
        rhs_ref: Reference to the rhs.
        out_ref: Reference to the out.
        partial_out_ref: Reference to the partial output.
        acc_ref: Reference to the accumulator.
        metadata_ref: Reference to the metadata.
        zero_ref: Scratch memory for storing zero values used in initialization.
        semaphore_ref: Semaphore for zero initialization DMAs.
        cfgs: GmmConfigs.
    """

    num_k = cfgs.dims.size_k // cfgs.tiles.tile_k
    num_n = cfgs.dims.size_n // cfgs.tiles.tile_n

    # Pack along K (2nd minor dim) so that pltpu.bitcast can unpack inside the
    # kernel.
    # [G, K, N] f4 -> [G, K//2, N] u8
    if cfgs.dims.rhs_packing > 1:
        rhs_weight = rhs_ref.weight
        assert isinstance(rhs_weight, jax.Array)
        rhs_weight = rhs_weight.bitcast(jnp.uint8)
        assert rhs_weight.shape == (cfgs.dims.size_group,
                                    cfgs.dims.size_k // cfgs.dims.rhs_packing,
                                    cfgs.dims.size_n), (
            f"Expected shape {(cfgs.dims.size_group, cfgs.dims.size_k // cfgs.dims.rhs_packing, cfgs.dims.size_n)}, "
            f"got {rhs_weight.shape}"
        )
        rhs_ref = dataclasses.replace(rhs_ref, weight=rhs_weight)

    # Fill metadata buffer and return number of group & m interations.
    num_gm = fill_metadata(
        lhs_group_sizes_ref,
        group_offset_ref,
        metadata_ref,
        cfgs=cfgs,
    )

    in_specs, out_specs = generate_block_specs(metadata_ref, cfgs)

    if cfgs.zero_init:
        zero_size = zero_out_start(
            out_ref,
            zero_ref,
            semaphore_ref,
            metadata_ref,
            num_gm,
            dims=cfgs.dims,
        )

    # Execute the inner kernel.
    pipeline_fn = pltpu.emit_pipeline(
        functools.partial(inner_kernel, cfgs=cfgs),
        grid=(num_n, num_gm, num_k),
        in_specs=in_specs,
        out_specs=out_specs,
    )

    # Bounded slice requires second last dim to be aligned to the sublane size.
    # rhs_ref uses static tiling thus reshape is not needed.
    lhs_in = lhs_ref.reshape(-1, cfgs.dims.size_lhs_sublane, cfgs.dims.size_k)
    out_in = out_ref.reshape(-1, cfgs.dims.size_lhs_sublane, cfgs.dims.size_n)
    scratches = [partial_out_ref, acc_ref, metadata_ref]
    pipeline_fn(lhs_in, rhs_ref, out_in, scratches=scratches)

    if cfgs.zero_init:
        zero_out_end(out_ref, semaphore_ref, zero_size, dims=cfgs.dims)


def calculate_tiling(
    lhs_dtype: jnp.dtype,
    rhs_dtype: jnp.dtype,
    dims: Dimensions,
    vmem_limit_bytes: int,
) -> TileSizes:
    """Calculate optimal tile sizes for GMM kernel."""

    lhs_bits = jax.dtypes.itemsize_bits(lhs_dtype)
    rhs_bits = jax.dtypes.itemsize_bits(rhs_dtype)

    # Otherwise we run into a VMEM OOM
    # TODO (jacobplatin/wenxindongwork): remove this hotfix once FP4 RHS bug is fixed
    if rhs_bits == 4:
        rhs_bits = 8

    # When using bf16 for lhs and rhs, 128 is the largest tile_m value that is
    # safe to use for most scenarios. But if are using lower bitwidth, we need
    # to tweak tile_m to account for using faster hardware unit.
    # TODO(kyuyeunk): Account for different TPU hardware specs.
    bf16_bf16_tile_m = 128
    lhs_mod = min(pl.cdiv(16, lhs_bits), 2)
    rhs_mod = min(pl.cdiv(16, rhs_bits), 2)
    tile_m = bf16_bf16_tile_m * lhs_mod // rhs_mod
    tile_m = min(tile_m, dims.size_m)

    # Calculate vmem limit for a single rhs buffer when using triple buffers.
    num_rhs_buffers = 3
    rhs_vmem_target = vmem_limit_bytes // num_rhs_buffers
    rhs_base_size_bytes = rhs_size_bytes = (dims.size_k * dims.size_n *
                                            rhs_bits // 8)

    num_lanes = pltpu.get_tpu_info().num_lanes

    # To avoid stalling MXU, we add some buffer room where tile_n cannot go
    # smaller than 2x of mxu_column_size.
    tile_n_limit = pltpu.get_tpu_info().mxu_column_size * 2
    tile_n_limit = min(tile_n_limit, dims.size_n)

    def _is_tile_k_valid(tk: int) -> bool:
        """Check all tile_k constraints."""
        if tk % num_lanes != 0:
            return False
        if tk % dims.rhs_packing != 0:
            return False
        if (tk % dims.quant_block_size != 0
                and dims.quant_block_size % tk != 0):
            return False
        return True

    # Initialize tile_k and tile_n to their maximum valid values.
    num_k_tiles = 1
    tile_k = dims.size_k
    k_rem = 0

    # last_valid_n_tiles stores num_n_tiles that can evenly divide size_n but may
    # or may not be sufficient to fit rhs into vmem target without changing
    # tile_k.
    last_valid_n_tiles = num_n_tiles = 1
    tile_n = dims.size_n
    n_rem = 0

    # Multiple k tiles will introduce accumulation overhead. Thus, we first try
    # to fit rhs into vmem by only adjusting tile_n.

    # Decrease tile_n until rhs fits in vmem target.
    while rhs_size_bytes > rhs_vmem_target or n_rem or tile_n % num_lanes:
        if n_rem == 0 and tile_n % num_lanes == 0:
            last_valid_n_tiles = num_n_tiles
        num_n_tiles += 1
        tile_n = dims.size_n // num_n_tiles
        n_rem = dims.size_n % num_n_tiles
        rhs_size_bytes = rhs_base_size_bytes // num_n_tiles

    # If decreasing tile_n is no longer possible, we decrease tile_k instead.
    if tile_n % num_lanes or n_rem or tile_n < tile_n_limit:
        num_n_tiles = last_valid_n_tiles
        tile_n = dims.size_n // num_n_tiles
        rhs_size_bytes = rhs_base_size_bytes // num_n_tiles

        # Decrease tile_k until rhs fits in vmem target.
        while (rhs_size_bytes > rhs_vmem_target or k_rem
               or not _is_tile_k_valid(tile_k)):
            num_k_tiles += 1
            tile_k = dims.size_k // num_k_tiles
            k_rem = dims.size_k % num_k_tiles
            rhs_size_bytes = rhs_base_size_bytes // (num_k_tiles * num_n_tiles)

    is_tile_n_invalid = tile_n % num_lanes or n_rem or tile_n < tile_n_limit
    is_tile_k_invalid = k_rem or not _is_tile_k_valid(tile_k)

    if is_tile_n_invalid or is_tile_k_invalid:
        raise ValueError(
            f"Could not find valid tile sizes for {dims=} and"
            f" {rhs_vmem_target=}. Last tried tiles: {tile_m=} {tile_k=} {tile_n=}"
        )
    print(f"Chosen tile sizes: tile_m={tile_m}, tile_k={tile_k}, tile_n={tile_n}")
    return TileSizes(tile_m=tile_m, tile_k=tile_k, tile_n=tile_n)


def validate_inputs(
    lhs: jax.Array,
    rhs: jax.Array,
    rhs_scale: jax.Array | None,
    rhs_bias: jax.Array | None,
    group_sizes: jax.Array,
    group_offset: jax.Array,
):
    """Validates the inputs for the GMM kernel."""

    size_m = lhs.shape[0]
    size_group, size_k, size_n = rhs.shape
    size_lhs_group = group_sizes.shape[0]

    assert size_group <= size_lhs_group
    assert lhs.shape == (size_m, size_k)
    assert rhs.shape == (size_group, size_k, size_n)
    if rhs_bias is not None:
        assert rhs_bias.shape == (size_group, 1, size_n)
    if rhs_scale is not None:
        num_quant_blocks = rhs_scale.shape[1]
        assert rhs_scale.shape == (size_group, num_quant_blocks, 1, size_n)
    else:
        num_quant_blocks = 1
    quant_block_size = size_k // num_quant_blocks

    assert group_offset.shape == (1, )

    # TODO(kyuyeunk): Add support for implicit padding along lane dimensions.
    num_lanes = pltpu.get_tpu_info().num_lanes
    if size_k % num_lanes != 0 or size_n % num_lanes != 0:
        raise NotImplementedError(
            "Implicit padding along lane dimensions is not supported.")

    bitwidth = jax.dtypes.itemsize_bits(lhs.dtype)
    packing = 32 // bitwidth
    size_lhs_sublane = pltpu.get_tpu_info().num_sublanes * packing

    # TODO(wenxindong): Work around a compiler bug in scoped memory allocation
    # for fp4. When rhs is fp4, we pack two fp4
    # elements into one uint8 element along the K axis (rhs_packing=2) in
    # kernel_main via bitcast. The inner kernel then unpacks back to fp4 using
    # pltpu.bitcast. Remove this once we upgrade to libtpu >0.0.36.
    if jax.dtypes.itemsize_bits(rhs.dtype) == 4:
        rhs_packing = 2
    else:
        rhs_packing = 1

    return Dimensions(
        size_m=size_m,
        size_k=size_k,
        size_n=size_n,
        size_group=size_group,
        size_lhs_group=size_lhs_group,
        size_lhs_sublane=size_lhs_sublane,
        quant_block_size=quant_block_size,
        num_quant_blocks=num_quant_blocks,
        rhs_packing=rhs_packing,
        rhs_dtype=rhs.dtype,
    )


def validate_tiles(tiles: TileSizes, dims: Dimensions):
    """Validates the tile sizes for the GMM kernel."""

    def _validate(x: int, tx: int, name: str):
        if x % tx != 0:
            raise ValueError(
                f"size_{name}={x} is not divisible by tile_{name}={tx}.")
        if tx > x:
            raise ValueError(
                f"tile_{name}={tx} is larger than size_{name}={x}.")

    _validate(dims.size_m, pltpu.get_tpu_info().num_sublanes, "m")
    _validate(dims.size_k, tiles.tile_k, "k")
    _validate(dims.size_n, tiles.tile_n, "n")

    if (tiles.tile_k % dims.quant_block_size != 0
            and dims.quant_block_size % tiles.tile_k != 0):
        raise ValueError(f"tile_k={tiles.tile_k} and"
                         f" quant_block_size={dims.quant_block_size} must be"
                         " divisible by one another.")


def get_cost_estimate(
    lhs: jax.Array,
    rhs: WeightsRef,
    out_dtype: jnp.dtype,
    dims: Dimensions,
):
    """Returns the cost estimate for the GMM kernel."""
    assert isinstance(rhs.weight, jax.Array)

    flops = 2 * dims.size_m * dims.size_k * dims.size_n

    lhs_bytes = dims.size_m * dims.size_k * lhs.dtype.itemsize

    # TODO(kyuyeunk): Handle 4-bit quantization case.
    rhs_bytes = (dims.size_group * dims.size_k * dims.size_n *
                 rhs.weight.dtype.itemsize)
    if rhs.scale is not None:
        rhs_bytes += (dims.size_group * dims.num_quant_blocks * dims.size_n *
                      jnp.dtype(jnp.float32).itemsize)
    if rhs.bias is not None:
        rhs_bytes += dims.size_group * dims.size_n * jnp.dtype(
            jnp.float32).itemsize

    out_bytes = dims.size_m * dims.size_n * jnp.dtype(out_dtype).itemsize

    total_bytes = lhs_bytes + rhs_bytes + out_bytes

    return pl.CostEstimate(
        flops=flops,
        bytes_accessed=total_bytes,
        transcendentals=0,
    )


def get_scope_name(dims: Dimensions, tiles: TileSizes) -> str:
    return (
        f"gmm_v2-g_{dims.size_group}-m_{dims.size_m}-k_{dims.size_k}"
        f"-n_{dims.size_n}-tm_{tiles.tile_m}-tk_{tiles.tile_k}-tn_{tiles.tile_n}"
    )


def make_gmm_configs(
    lhs: jax.Array,
    rhs: jax.Array,
    rhs_scale: jax.Array | None,
    rhs_bias: jax.Array | None,
    group_sizes: jax.Array,
    group_offset: jax.Array,
    *,
    tile_info: TileSizes | TileFn,
    vmem_limit_bytes: int | None,
    out_dtype: jnp.dtype | None,
    acc_dtype: jnp.dtype | None,
    maybe_quantize_lhs: bool,
    zero_initialize: bool,
):
    """Fills the GMM config for the GMM kernel."""

    dims = validate_inputs(lhs, rhs, rhs_scale, rhs_bias, group_sizes,
                           group_offset)

    if rhs_scale is not None:
        has_scale = True
        rhs_quant_dtype = rhs.dtype
        num_blocks = rhs_scale.shape[1]
        block_size = dims.size_k // num_blocks
    else:
        has_scale = False
        rhs_quant_dtype = None
        block_size = None

    rhs_cfgs = InputConfigs(
        quant_dtype=rhs_quant_dtype,
        quant_block_size=block_size,
        has_bias=rhs_bias is not None,
        has_scale=has_scale,
    )

    lhs_q_dtype = None
    if maybe_quantize_lhs and rhs_quant_dtype is not None:
        # Choose lhs quantization dtype based on TPU hardware support.
        is_rhs_float = jnp.issubdtype(rhs_quant_dtype, jnp.floating)
        tpu_info = pltpu.get_tpu_info()
        # Check if there is hardware compute support for rhs dtype group.
        if is_rhs_float:
            if tpu_info.fp8_ops_per_second > 0:
                lhs_q_dtype = jnp.float8_e4m3fn.dtype
        else:
            if tpu_info.int8_ops_per_second > 0:
                lhs_q_dtype = jnp.int8.dtype

    lhs_cfgs = InputConfigs(
        quant_dtype=lhs_q_dtype,
        # Input quantization involves reading all elements in a block to compute
        # scale value. Since this operation is very memory intensive, we use a
        # block size that is small enough to minimize memory overhead but large
        # enough to minimize compute overhead of quantization.
        quant_block_size=512,
    )

    # Validate that rhs quant block size >= lhs quant block size and is
    # divisible by it. This ensures the lhs quantization loop (which steps by
    # lhs_quant_block_size) correctly maps to rhs scale blocks.
    if (lhs_cfgs.quant_dtype is not None and rhs_cfgs.has_scale
            and rhs_cfgs.quant_block_size is not None):
        lhs_qbs = lhs_cfgs.quant_block_size
        rhs_qbs = rhs_cfgs.quant_block_size
        if rhs_qbs < lhs_qbs:
            raise ValueError(
                f"rhs quant_block_size={rhs_qbs} must be >= lhs"
                f" quant_block_size={lhs_qbs}.")
        if rhs_qbs % lhs_qbs != 0:
            raise ValueError(
                f"rhs quant_block_size={rhs_qbs} must be divisible by lhs"
                f" quant_block_size={lhs_qbs}.")

    if out_dtype is None:
        out_dtype = lhs.dtype

    if acc_dtype is None:
        if lhs_cfgs.quant_dtype is None:
            acc_dtype = jnp.float32.dtype
        else:
            # Input quantization requires elementwise ops which can put pressure on
            # VPUs. Using faster bf16 hardware during accumulation can help offset the
            # pressure.
            acc_dtype = jnp.bfloat16.dtype

    if isinstance(tile_info, TileSizes):
        tiles = tile_info
    else:
        lhs_q_dtype = lhs_q_dtype if lhs_q_dtype is not None else lhs.dtype
        tiles = tile_info(lhs_q_dtype, rhs.dtype, dims, vmem_limit_bytes)

    validate_tiles(tiles, dims)

    return GmmConfigs(
        dims=dims,
        tiles=tiles,
        lhs_cfgs=lhs_cfgs,
        rhs_cfgs=rhs_cfgs,
        out_dtype=out_dtype,
        acc_dtype=acc_dtype,
        zero_init=zero_initialize,
    )


def get_metadata(cfgs: GmmConfigs):
    cfgs_dict = dataclasses.asdict(cfgs)
    ret = {}
    for path, val in jax.tree_util.tree_leaves_with_path(cfgs_dict):
        key = jax.tree_util.keystr(path, simple=True, separator=".")
        if isinstance(val, jnp.dtype):
            val = val.name
        ret[key] = val
    return ret


@jax.jit(static_argnames=[
    "tile_info",
    "vmem_limit_bytes",
    "precision",
    "preferred_element_type",
    "acc_dtype",
    "maybe_quantize_lhs",
    "zero_initialize",
])
def gmm_v2(
    lhs: jax.Array,  # [size_m, size_k]
    rhs: jax.Array,  # [size_group, size_k, size_n]
    group_sizes: jax.Array,  # int32[size_lhs_group]
    rhs_scale: jax.Array | None = None,  # [size_group, num_blocks, 1, out_size]
    rhs_bias: jax.Array | None = None,  # [size_group, 1, out_size]
    group_offset: jax.Array | None = None,  # int32[1]
    *,
    tile_info: TileSizes | TileFn = calculate_tiling,
    vmem_limit_bytes: int | None = None,
    precision: jax.lax.Precision = jax.lax.Precision.DEFAULT,
    preferred_element_type: jnp.dtype | None = None,
    acc_dtype: jnp.dtype | None = None,
    maybe_quantize_lhs: bool = True,
    zero_initialize: bool = True,
) -> jax.Array:
    """GMM kernel implemented with emit_pipeline.

    Dynamically calculate offset lhs/out tiles to reduce redundant computations.
    Additionally, adjusting dma size based on number of valid rows and utilize
    triple buffering on weights to better utilize memory.

    Args:
        lhs: lhs with shape [size_m, size_k].
        rhs: rhs with shape [size_group, size_k, size_n].
        group_sizes: The group sizes of lhs rows of shape [size_lhs_group,].
        rhs_scale: The rhs scale of shape [size_group, num_blocks, 1, out_size].
        rhs_bias: The rhs bias of shape [size_group, 1, out_size].
        group_offset: Optional. The group offset of shape [1,].
        tile_info: The tile sizes or tile function to use.
        vmem_limit_bytes: Optional vmem limit in bytes.
        precision: Unused. Exists for compatibility reasons.
        preferred_element_type: Optional jnp.dtype for the output matrix.
        acc_dtype: Optional jnp.dtype for the accumulator.
        maybe_quantize_lhs: Quantize lhs if set to True and rhs is quantized.
        zero_initialize: Whether to initialize unvisited output elements to zero.

    Returns:
        Output of shape [size_m, size_n].
    """

    del precision

    if group_offset is None:
        group_offset = jnp.array([0], dtype=jnp.int32)
    else:
        if jnp.isscalar(group_offset):
            group_offset = group_offset[None]

    if vmem_limit_bytes is None:
        vmem_limit_bytes = int(pltpu.get_tpu_info().vmem_capacity_bytes * 0.9)

    cfgs = make_gmm_configs(
        lhs,
        rhs,
        rhs_scale,
        rhs_bias,
        group_sizes,
        group_offset,
        tile_info=tile_info,
        vmem_limit_bytes=vmem_limit_bytes,
        out_dtype=preferred_element_type,
        acc_dtype=acc_dtype,
        maybe_quantize_lhs=maybe_quantize_lhs,
        zero_initialize=zero_initialize,
    )
    dims = cfgs.dims
    tiles = cfgs.tiles

    # Prepare block specs.
    rhs_scale_spec = rhs_bias_spec = None
    if rhs_scale is not None:
        rhs_scale = rhs_scale.astype(jnp.float32)
        rhs_scale_spec = pl.BlockSpec(memory_space=pltpu.HBM)
    if rhs_bias is not None:
        rhs_bias = rhs_bias.astype(jnp.float32)
        rhs_bias_spec = pl.BlockSpec(memory_space=pltpu.HBM)

    # Initialize scratch shapes.
    max_num_gm = dims.size_group + dims.size_m // tiles.tile_m - 1

    scratch_shapes = [
        # partial_out_ref
        pltpu.VMEM((dims.size_lhs_sublane, tiles.tile_n), cfgs.out_dtype),
        # acc_ref
        pltpu.VMEM((tiles.tile_m, tiles.tile_n), cfgs.acc_dtype),
        # metadata_ref
        MetadataRef(
            gm_id_to_group_id=pltpu.SMEM((max_num_gm, ), jnp.int32),
            gm_id_to_m_offset=pltpu.SMEM((max_num_gm + 1, ), jnp.int32),
        ),
    ]

    if cfgs.zero_init:
        # TODO(kyuyeunk): Create better heuristics for determining this value.
        target_zero_ref_bytes = 2 * 1024 * 1024

        # Zero initialization is done by tiling size_m dim where each tile invokes
        # zero initializing DMA for up-to tile_zero_m rows. This means larger
        # tile_zero_m will result in fewer number of tiles and lead to smaller
        # overhead. However, in order to invoke DMA call up-to tile_zero_m rows, we
        # need to store equivalent sized memory in VMEM buffer for the duration of
        # DMA. Storing [tile_zero_m, size_n] in buffer will trigger OOM if
        # tile_zero_m is too large. Instead, if we set column size as num_lanes
        # (which is smallest allowed column size for DMA) and reuse the buffer by
        # size_n//num_lanes times in a single tile, we can significantly increase
        # tile_zero_m without triggering OOM.
        num_lanes = pltpu.get_tpu_info().num_lanes
        out_bytes = jnp.dtype(cfgs.out_dtype).itemsize
        tile_zero_m = target_zero_ref_bytes // num_lanes // out_bytes
        tile_zero_m = min(tile_zero_m, dims.size_m)

        scratch_shapes += [
            pltpu.VMEM((tile_zero_m, num_lanes), cfgs.out_dtype),
            pltpu.SemaphoreType.DMA((1, )),
        ]
    else:
        scratch_shapes += [None, None]

    out_init = jax.ShapeDtypeStruct((dims.size_m, dims.size_n), cfgs.out_dtype)
    rhs_weights = WeightsRef(weight=rhs, scale=rhs_scale, bias=rhs_bias)

    return pl.pallas_call(
        functools.partial(kernel_main, cfgs=cfgs),
        out_shape=out_init,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=2,
            in_specs=[
                pl.BlockSpec(memory_space=pltpu.HBM),
                WeightsRef(
                    weight=pl.BlockSpec(memory_space=pltpu.HBM),
                    scale=rhs_scale_spec,
                    bias=rhs_bias_spec,
                ),
            ],
            out_specs=pl.BlockSpec(memory_space=pltpu.HBM),
            scratch_shapes=scratch_shapes,
        ),
        compiler_params=pltpu.CompilerParams(
            vmem_limit_bytes=vmem_limit_bytes,
            disable_bounds_checks=True,
        ),
        name=get_scope_name(dims, tiles),
        cost_estimate=get_cost_estimate(lhs, rhs_weights, out_init.dtype,
                                        dims),
        metadata=get_metadata(cfgs),
    )(group_sizes, group_offset, lhs, rhs_weights)


def is_supported_by_gmm_v2(lhs: jax.Array, rhs: jax.Array,
                           rhs_scale: jax.Array | None) -> bool:
    """Return false if gmm_v2 does not support the inputs yet."""

    # gmm_v2 does not support implicit padding along lane dimension.
    num_lanes = pltpu.get_tpu_info().num_lanes
    if lhs.shape[-1] % num_lanes != 0 or rhs.shape[-1] % num_lanes != 0:
        return False
    # gmm_v2 does not support when lhs is not multiple of sublane size.
    lhs_bytes = lhs.dtype.itemsize
    if lhs.shape[0] % (pltpu.get_tpu_info().num_sublanes * lhs_bytes) != 0:
        return False
    # Handle weird edge cases where inputs are already quantized.
    if lhs.dtype not in [jnp.bfloat16, jnp.float32]:
        return False
    return True