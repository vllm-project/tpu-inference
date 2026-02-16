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
from typing import Callable, Tuple

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
    weight: pl.BlockSpec | jax.Array
    scale: pl.BlockSpec | jax.Array | None
    bias: pl.BlockSpec | jax.Array | None


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


TileFn = Callable[[jnp.dtype, jnp.dtype, Dimensions, int], TileSizes]


class IndexMaps:
    """Index maps for GMM kernel."""

    def __init__(
        self,
        metadata_ref: MetadataRef,
        tiles: TileSizes,
        dims: Dimensions,
    ):
        self.metadata_ref = metadata_ref
        self.tiles = tiles
        self.dims = dims

    def _get_sublane_start_and_size(self, gm_id: jax.Array):
        m_start = self.metadata_ref.gm_id_to_m_offset[gm_id]
        m_end = self.metadata_ref.gm_id_to_m_offset[gm_id + 1]

        sublane_start = m_start // self.dims.size_lhs_sublane
        sublane_end = pl.cdiv(m_end, self.dims.size_lhs_sublane)
        sublane_size = sublane_end - sublane_start

        return sublane_start, sublane_size

    def lhs_index_map(self, _: jax.Array, gm_id: jax.Array, k_id: jax.Array):
        sublane_start, sublane_size = self._get_sublane_start_and_size(gm_id)
        return (pl.ds(sublane_start, sublane_size), 0, k_id)

    def rhs_weight_index_map(self, n_id: jax.Array, gm_id: jax.Array,
                             k_id: jax.Array):
        group_id = self.metadata_ref.gm_id_to_group_id[gm_id]
        return (group_id, k_id, n_id)

    def rhs_bias_index_map(self, n_id: jax.Array, gm_id: jax.Array,
                           _: jax.Array):
        group_id = self.metadata_ref.gm_id_to_group_id[gm_id]
        return (group_id, 0, n_id)

    def rhs_scale_index_map(self, n_id: jax.Array, gm_id: jax.Array,
                            _: jax.Array):
        group_id = self.metadata_ref.gm_id_to_group_id[gm_id]
        return (group_id, 0, 0, n_id)

    def out_index_map(self, n_id: jax.Array, gm_id: jax.Array, _: jax.Array):
        sublane_start, sublane_size = self._get_sublane_start_and_size(gm_id)
        return (pl.ds(sublane_start, sublane_size), 0, n_id)


def generate_block_specs(
    lhs_ref: jax.Array,  # [size_m, size_k]
    rhs_ref: WeightsRef,  # [size_group, size_k, size_n]
    out_ref: jax.Array,  # [size_m, size_n]
    metadata_ref: MetadataRef,
    *,
    tiles: TileSizes,
    dims: Dimensions,
) -> Tuple[Tuple[pl.BlockSpec, WeightsRef], pl.BlockSpec]:
    """Generates block specs for the given lhs, rhs, and out refs."""

    # lhs and out do not have optional fields and therefore not used here.
    # But we keep them as arguments for future extensions.
    del lhs_ref, out_ref

    index_map = IndexMaps(metadata_ref, tiles, dims)
    bounded_slice_gm = pl.BoundedSlice(tiles.tile_m // dims.size_lhs_sublane)

    lhs_block_spec = pl.BlockSpec(
        (bounded_slice_gm, dims.size_lhs_sublane, tiles.tile_k),
        index_map.lhs_index_map,
    )

    rhs_weight_spec = pl.BlockSpec(
        (None, tiles.tile_k, tiles.tile_n),
        index_map.rhs_weight_index_map,
        pipeline_mode=pl.Buffered(buffer_count=3),
    )
    rhs_scale_block_spec = rhs_bias_block_spec = None
    if rhs_ref.bias is not None:
        rhs_bias_block_spec = pl.BlockSpec(
            (1, 1, tiles.tile_n),
            index_map.rhs_bias_index_map,
        )
    if rhs_ref.scale is not None:
        rhs_scale_block_spec = pl.BlockSpec(
            (None, 1, 1, tiles.tile_n),
            index_map.rhs_scale_index_map,
        )

    rhs_block_spec = WeightsRef(
        weight=rhs_weight_spec,
        scale=rhs_scale_block_spec,
        bias=rhs_bias_block_spec,
    )

    out_block_spec = pl.BlockSpec(
        (bounded_slice_gm, dims.size_lhs_sublane, tiles.tile_n),
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
    acc_ref: jax.Array,
    # [tile_m // size_lhs_sublane, size_lhs_sublane, tile_n]
    metadata_ref: MetadataRef,
    *,
    tiles: TileSizes,
    dims: Dimensions,
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
        tiles: Tile sizes.
        dims: Dimensions.
    """

    gm_id = pl.program_id(1)
    k_id = pl.program_id(2)

    # If this is the first tile for grid[n_id, :, :], we initialize the partial
    # out to zeros. Otherwise, partial out from last tile of grid[n_id-1, :, :]
    # can be used and cause numeric issues.
    @pl.when(gm_id + k_id == 0)
    def _():
        partial_out_ref[...] = jnp.zeros_like(partial_out_ref)

    m_start = metadata_ref.gm_id_to_m_offset[gm_id]
    m_end = metadata_ref.gm_id_to_m_offset[gm_id + 1]

    m_offset = (m_start // dims.size_lhs_sublane) * dims.size_lhs_sublane

    m_start_local = m_start - m_offset
    m_end_local = m_end - m_offset

    def _matmul(is_first_k_step: bool, is_last_k_step: bool):
        acc = jnp.matmul(
            tiled_lhs_ref[...],
            tiled_rhs_ref.weight[...],
            preferred_element_type=jnp.float32,
        )

        if not is_first_k_step:
            acc += acc_ref[...]

        if is_last_k_step:
            if tiled_rhs_ref.scale is not None:
                acc *= tiled_rhs_ref.scale[...]
            if tiled_rhs_ref.bias is not None:
                acc += tiled_rhs_ref.bias[...]

            # Mask out values that does not belong to the current group.
            iota_mask = lax.broadcasted_iota(jnp.int32,
                                             (tiles.tile_m, tiles.tile_n), 0)
            mask = jnp.logical_and(m_start_local <= iota_mask, iota_mask
                                   < m_end_local).reshape(acc.shape)
            acc_masked = jnp.where(mask, acc, 0)

            # Write the final output to the output ref.
            tiled_out_ref[...] = acc_masked.astype(tiled_out_ref.dtype)

            # Accumulate the partial output from the previous step.
            tiled_out_ref[0] += partial_out_ref[...]

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
            last_row = m_end_local // dims.size_lhs_sublane
            partial_out_ref[...] = jnp.where(
                m_end_local % dims.size_lhs_sublane == 0,
                jnp.zeros_like(partial_out_ref),
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
    is_first_k_step = k_id == 0
    is_last_k_step = k_id == num_k - 1

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


def wait_zero_out(
    hbm_buf: jax.Array,
    start_chunk_idx: jax.Array,
    end_chunk_idx: jax.Array,
    sem_ref: jax.Array,
    *,
    namescope: str = "zero_out_wait",
    unroll: bool = True,
):
    """Waits for the given hbm buffer, with chunking on 2nd minor dimension."""
    with jax.named_scope(namescope):
        lax.fori_loop(
            start_chunk_idx,
            end_chunk_idx,
            lambda i, _: pltpu.make_async_copy(hbm_buf, hbm_buf.at[i, :, :],
                                               sem_ref.at[0]).wait(),
            None,
            unroll=unroll,
        )


def zero_out(
    hbm_buf: jax.Array,
    zero_buf: jax.Array,
    start_chunk_idx: jax.Array,
    end_chunk_idx: jax.Array,
    sem_ref: jax.Array,
    *,
    namescope: str = "zero_out_start",
    dma_priority: int = 1,
    unroll: bool = True,
):
    """Zeroes out the given hbm buffer, with chunking on 2nd minor dimension."""
    with jax.named_scope(namescope):
        lax.fori_loop(
            start_chunk_idx,
            end_chunk_idx,
            lambda i, _: pltpu.make_async_copy(
                zero_buf,
                hbm_buf.at[i, :, :],
                sem_ref.at[0],
            ).start(dma_priority),
            None,
            unroll=unroll,
        )


def zero_out_vregs(
    hbm_buf: jax.Array,
    zero_buf: jax.Array,
    start_vreg_idx: jax.Array,
    end_vreg_idx: jax.Array,
    sem_ref: jax.Array,
    *,
    namescope: str = "zero_out_vregs",
    dma_priority: int = 1,
):
    """Zeroes out the given hbm buffer, with vreg alignment."""
    with jax.named_scope(namescope):
        num_vregs = end_vreg_idx - start_vreg_idx

        @pl.when(num_vregs > 0)
        def _():
            src_slice = zero_buf.at[pl.ds(0, num_vregs), :, :]
            dst_slice = hbm_buf.at[pl.ds(start_vreg_idx, num_vregs), :, :]
            pltpu.make_async_copy(
                src_slice,
                dst_slice,
                sem_ref.at[0],
            ).start(dma_priority)


def wait_zero_out_vregs(
    hbm_buf: jax.Array,
    start_vreg_idx: jax.Array,
    end_vreg_idx: jax.Array,
    sem_ref: jax.Array,
    *,
    namescope: str = "zero_out_wait_vregs",
):
    """Waits for the given hbm buffer, with vreg alignment."""
    with jax.named_scope(namescope):
        num_vregs = end_vreg_idx - start_vreg_idx

        @pl.when(num_vregs > 0)
        def _():
            pltpu.make_async_copy(
                hbm_buf.at[pl.ds(start_vreg_idx, num_vregs), :, :],
                hbm_buf.at[pl.ds(start_vreg_idx, num_vregs), :, :],
                sem_ref.at[0],
            ).wait()


def fill_metadata(
    lhs_group_sizes_ref: jax.Array,  # int32[size_lhs_group]
    group_offset_ref: jax.Array,  # int32[1]
    metadata_ref: MetadataRef,
    out_init_hbm_ref: jax.Array,
    local_dma_sem_ref: jax.Array,
    mini_zero_buf: jax.Array,
    *,
    dims: Dimensions,
    tiles: TileSizes,
    chunk_m: int,
) -> jax.Array:
    """Fills the metadata for the given lhs group sizes and group offset.

    Iterates over the lhs group sizes and if the group id is valid, determines
    the number of gm tiles that are needed to process the current group. Then,
    it fills starting and ending offset (gm_id_to_m_offset), and the group id
    (gm_id_to_group_id) for each gm tile.

    Args:
      lhs_group_sizes_ref: The group sizes of lhs.
      group_offset_ref: Offset of the first group to process.
      metadata_ref: Metadata that is used to determine the group id and m offsets
        for each gmm tile.
      out_init_hbm_ref: Reference to out_ref alias.
      local_dma_sem_ref: Reference to the local dma semaphore.
      mini_zero_buf: A small buffer used for zeroing out.
      dims: Dimensions of arguments.
      tiles: Tile sizes for this kernel.
      chunk_m: Number of m tiles in a zero out dma chunk.

    Returns:
      The number of gm tiles to process lhs with given group offset.
    """

    group_offset = group_offset_ref[0]
    max_num_group = group_offset + dims.size_group

    @jax.named_scope("inner_tm_loop")
    def inner_tm_loop(tm_id, curr_m_offset, *, end_m_offset, group_id, num_gm):
        local_offset = curr_m_offset % dims.size_lhs_sublane
        tm_size = jnp.minimum(tiles.tile_m - local_offset,
                              end_m_offset - curr_m_offset)

        metadata_ref.gm_id_to_group_id[num_gm + tm_id] = group_id

        next_m_offset = curr_m_offset + tm_size
        metadata_ref.gm_id_to_m_offset[num_gm + tm_id] = curr_m_offset
        metadata_ref.gm_id_to_m_offset[num_gm + tm_id + 1] = next_m_offset

        return next_m_offset

    @jax.named_scope("outer_group_loop")
    def outer_group_loop(lhs_group_id, carry):
        num_gm, start_m_offset, zeroed_end_chunk_idx, remainder_m = carry

        group_id = lhs_group_id - group_offset
        group_size = lhs_group_sizes_ref[lhs_group_id]
        end_m_offset = start_m_offset + group_size

        ## Head zeroing out logic.
        ## Given facts:
        # - chunk_m is multiple of lhs_sublane_size
        ## Global constraints:
        # - zero vregs from 0 to (start_valid_m_offset//lhs_sublane_size)
        #   exclusive
        #   - last vreg if shared is handled by inner loop
        #   - this ensures that head zeroing's start chunk_m and end chunk_m are
        #     vreg aligned
        #
        ## Per group(iteration of outer group loop) constraints:
        # State:
        # - zeroed_end_chunk_idx (chunk index of last zeroed chunk)
        # - remainder_m leftover rows to be zeroed out from previous group
        # Per iteration
        # - Check cond to zero group_id < 0 -> group's row should be zeroed out,
        #   change carry
        #     - group_id >= 0 -> no zeroing no carry change
        # - head_zero_acc += groupsize
        # - curent chunks to zero out = cumsum // chunk_m
        # - start_cur_zero_chunk_idx = zeroed_end_chunk_idx
        # - end_cur_zero_chunk_idx =
        #   start_cur_zero_chunk_idx + current chunks to zero out
        # - zeroed_end_chunk_idx = end_cur_zero_chunk_idx
        # - zero_out(out_init_hbm_ref, mini_zero_buf, start_cur_zero_chunk_idx,
        #            end_cur_zero_chunk_idx, local_dma_sem_ref)

        def zero_out_group():
            acc = remainder_m + group_size
            num_chunks_to_zero_out = acc // chunk_m
            start_cur_zero_chunk_idx = zeroed_end_chunk_idx
            end_cur_zero_chunk_idx = (start_cur_zero_chunk_idx +
                                      num_chunks_to_zero_out)
            zero_out(
                out_init_hbm_ref,
                mini_zero_buf,
                start_cur_zero_chunk_idx,
                end_cur_zero_chunk_idx,
                local_dma_sem_ref,
            )
            return end_cur_zero_chunk_idx, acc % chunk_m

        zeroed_end_chunk_idx, remainder_m = lax.cond(
            group_id < 0,
            zero_out_group,
            lambda: (zeroed_end_chunk_idx, remainder_m),
        )

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
        local_offset = start_m_offset % dims.size_lhs_sublane
        aligned_group_size = group_size + local_offset
        curr_num_gm = pl.cdiv(aligned_group_size, tiles.tile_m)

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

        return (
            num_gm + curr_num_gm,
            end_m_offset,
            zeroed_end_chunk_idx,
            remainder_m,
        )

    num_gm, end_m_offset, zeroed_end_chunk_idx, remainder_m = lax.fori_loop(
        0, max_num_group, outer_group_loop, (0, 0, 0, 0))
    remainder_chunk_vregs = remainder_m // dims.size_lhs_sublane

    return num_gm, end_m_offset, zeroed_end_chunk_idx, remainder_chunk_vregs


def kernel_main(
    # Scalar prefetch
    lhs_group_sizes_ref: jax.Array,  # int32[size_lhs_group]
    group_offset_ref: jax.Array,  # int32[1]
    # In
    lhs_ref: jax.Array,  # [size_m, size_k]
    rhs_ref: WeightsRef,  # [size_group, size_k, size_n]
    out_init_hbm_ref: jax.Array,  # [size_m, size_n]
    # Out
    out_ref: jax.Array,  # [size_m, size_n]
    # Scratch memory
    partial_out_ref: jax.Array,  # [size_lhs_sublane, tile_n]
    acc_ref: jax.
    Array,  # [tile_m // size_lhs_sublane, size_lhs_sublane, tile_n]
    metadata_ref: MetadataRef,
    local_dma_sem_ref: jax.Array,  # [1]
    mini_zero_buf: jax.Array,  # [size_lhs_sublane, size_n]
    *,
    tiles: TileSizes,
    dims: Dimensions,
    chunk_m: int = 32,
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
        _: Reference to out_ref alias.
        out_ref: Reference to the out.
        metadata_ref: Reference to the metadata.
        partial_out_ref: Reference to the partial output.
        acc_ref: Reference to the accumulator.
        tiles: Tile sizes.
        dims: Dimensions.
  """

    num_k = dims.size_k // tiles.tile_k
    num_n = dims.size_n // tiles.tile_n
    mini_zero_buf[...] = jnp.zeros_like(mini_zero_buf)
    out_init_hbm_ref = out_init_hbm_ref.reshape(
        -1, chunk_m, dims.size_n)  # shape (num_chunks, chunk_m, size_n)

    # Fill metadata buffer and return number of group & m interations.
    num_gm, end_m_offset, zeroed_end_chunk_idx, remainder_chunk_vregs = fill_metadata(
        lhs_group_sizes_ref,
        group_offset_ref,
        metadata_ref,
        out_init_hbm_ref,
        local_dma_sem_ref,
        mini_zero_buf,
        dims=dims,
        tiles=tiles,
        chunk_m=chunk_m,
    )

    # Zero out Head remainder vregs, excluding the last vreg if it is shared.
    out_vreg_view = out_init_hbm_ref.reshape(-1, dims.size_lhs_sublane,
                                             dims.size_n)
    mini_zero_vreg_view = mini_zero_buf.reshape(-1, dims.size_lhs_sublane,
                                                dims.size_n)
    head_rem_start_vreg_idx = (zeroed_end_chunk_idx * chunk_m //
                               dims.size_lhs_sublane)
    head_end_vreg_idx = head_rem_start_vreg_idx + remainder_chunk_vregs
    zero_out_vregs(
        out_vreg_view,
        mini_zero_vreg_view,
        head_rem_start_vreg_idx,
        head_end_vreg_idx,
        local_dma_sem_ref,
    )

    # Zero out Tail m rows after end_m_offset.
    out_init_hbm_ref = out_init_hbm_ref.reshape(
        -1, chunk_m, dims.size_n)  # shape (num_chunks, chunk_m, size_n)
    mini_zero_buf = mini_zero_buf.reshape(chunk_m, dims.size_n)
    # zero out chunks after end_m_offset, starting from first chunk after
    # end_m_offset
    tail_start = (pl.cdiv(end_m_offset, dims.size_lhs_sublane) *
                  dims.size_lhs_sublane)
    tail_chunk_start = pl.cdiv(tail_start, chunk_m)
    tail_chunk_end = jnp.array(dims.size_m // chunk_m)
    zero_out(
        out_init_hbm_ref,
        mini_zero_buf,
        tail_chunk_start,
        tail_chunk_end,
        local_dma_sem_ref,
    )
    # Zero out Tail remainder vregs starting from first vreg after end_m_offset
    # till first chunk.
    chunk_m_vregs = chunk_m // dims.size_lhs_sublane
    tail_rem_start_vreg_idx = tail_start // dims.size_lhs_sublane
    tail_rem_end_vreg_idx = tail_chunk_start * chunk_m_vregs
    zero_out_vregs(
        out_vreg_view,
        mini_zero_vreg_view,
        tail_rem_start_vreg_idx,
        tail_rem_end_vreg_idx,
        local_dma_sem_ref,
    )

    in_block_specs, out_block_specs = generate_block_specs(lhs_ref,
                                                           rhs_ref,
                                                           out_ref,
                                                           metadata_ref,
                                                           tiles=tiles,
                                                           dims=dims)

    # Execute the inner kernel.
    pipeline_fn = pltpu.emit_pipeline(
        functools.partial(inner_kernel, tiles=tiles, dims=dims),
        grid=(num_n, num_gm, num_k),
        in_specs=in_block_specs,
        out_specs=out_block_specs,
    )

    # Bounded slice requires second last dim to be aligned to the sublane size.
    # rhs_ref uses static tiling thus reshape is not needed.
    lhs_in = lhs_ref.reshape(-1, dims.size_lhs_sublane, dims.size_k)
    out_in = out_ref.reshape(-1, dims.size_lhs_sublane, dims.size_n)
    scratches = [partial_out_ref, acc_ref, metadata_ref]
    pipeline_fn(lhs_in, rhs_ref, out_in, scratches=scratches)

    # Wait for Head zeroing out DMA to complete.
    out_init_hbm_ref = out_init_hbm_ref.reshape(-1, chunk_m, dims.size_n)
    zero_index = jnp.array(0, dtype=jnp.int32)
    wait_zero_out(out_init_hbm_ref, zero_index, zeroed_end_chunk_idx,
                  local_dma_sem_ref)
    wait_zero_out_vregs(
        out_vreg_view,
        head_rem_start_vreg_idx,
        head_end_vreg_idx,
        local_dma_sem_ref,
    )

    # Wait for Tail zeroing out DMA to complete.
    wait_zero_out(
        out_init_hbm_ref,
        tail_chunk_start,
        tail_chunk_end,
        local_dma_sem_ref,
    )
    wait_zero_out_vregs(
        out_vreg_view,
        tail_rem_start_vreg_idx,
        tail_rem_end_vreg_idx,
        local_dma_sem_ref,
    )


def calculate_tiling(
    lhs_dtype: jnp.dtype,
    rhs_dtype: jnp.dtype,
    dims: Dimensions,
    vmem_limit_bytes: int,
) -> TileSizes:
    """Calculate optimal tile sizes for GMM kernel."""

    lhs_bits = jax.dtypes.itemsize_bits(lhs_dtype)
    rhs_bits = jax.dtypes.itemsize_bits(rhs_dtype)

    # When using bf16 for lhs and rhs, 128 is the largest tile_m value that is
    # safe to use for most scenarios. But if are using lower bitwidth, we need
    # to tweak tile_m to account for using faster hardware unit.
    # TODO(kyuyeunk): Account for different TPU hardware specs.
    bf16_bf16_tile_m = 128
    lhs_mod = min(pl.cdiv(16, lhs_bits), 2)
    rhs_mod = min(pl.cdiv(16, rhs_bits), 2)
    tile_m = bf16_bf16_tile_m * lhs_mod // rhs_mod

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
        while rhs_size_bytes > rhs_vmem_target or k_rem or tile_k % num_lanes:
            num_k_tiles += 1
            tile_k = dims.size_k // num_k_tiles
            k_rem = dims.size_k % num_k_tiles
            rhs_size_bytes = rhs_base_size_bytes // (num_k_tiles * num_n_tiles)

    is_tile_n_invalid = tile_n % num_lanes or n_rem or tile_n < tile_n_limit
    is_tile_k_invalid = tile_k % num_lanes or k_rem

    if is_tile_n_invalid or is_tile_k_invalid:
        raise ValueError(
            f"Could not find valid tile sizes for {dims=} and"
            f" {rhs_vmem_target=}. Last tried tiles: {tile_m=} {tile_k=} {tile_n=}"
        )

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
        # TODO(kyuyeunk, wenxindong): Add support for subchannel quantization.
        if rhs_scale.shape[1] != 1:
            raise NotImplementedError(
                "Only per-channel quantization is supported.")
        assert rhs_scale.shape == (size_group, 1, 1, size_n)

    assert group_offset.shape == (1, )

    # TODO(kyuyeunk): Add support for implicit padding along lane dimensions.
    num_lanes = pltpu.get_tpu_info().num_lanes
    if size_k % num_lanes != 0 or size_n % num_lanes != 0:
        raise NotImplementedError(
            "Implicit padding along lane dimensions is not supported.")

    bitwidth = jax.dtypes.itemsize_bits(lhs.dtype)
    packing = 32 // bitwidth
    size_lhs_sublane = pltpu.get_tpu_info().num_sublanes * packing

    return Dimensions(
        size_m=size_m,
        size_k=size_k,
        size_n=size_n,
        size_group=size_group,
        size_lhs_group=size_lhs_group,
        size_lhs_sublane=size_lhs_sublane,
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
        rhs_bytes += dims.size_n * jnp.dtype(jnp.float32).itemsize
    if rhs.bias is not None:
        rhs_bytes += dims.size_n * jnp.dtype(jnp.float32).itemsize

    out_bytes = dims.size_m * dims.size_n * jnp.dtype(out_dtype).itemsize

    total_bytes = lhs_bytes + rhs_bytes + out_bytes

    return pl.CostEstimate(
        flops=flops,
        bytes_accessed=total_bytes,
        transcendentals=0,
    )


def get_scope_name(dims: Dimensions, tiles: TileSizes) -> str:
    return (
        f"gmm-g_{dims.size_group}-m_{dims.size_m}-k_{dims.size_k}"
        f"-n_{dims.size_n}-tm_{tiles.tile_m}-tk_{tiles.tile_k}-tn_{tiles.tile_n}"
    )


def calculated_dma_chunk_m_size(size_lhs_sublane: int, size_m: int,
                                size_n: int, dtype: jnp.dtype):
    """Calculates the DMA chunk m size for the GMM kernel.
  Conditions
    - dma_size >= 4194304 bytes (4 MiB)
    - dma_size % size_lhs_sublane == 0
  Args:
    size_lhs_sublane: The size of the LHS sublane dimension.
    size_n: The size of the output dimension.
    dtype: The data type of the elements.

  Returns:
    The DMA chunk m size.
  """
    optimal_dma_bytes = 4194304
    dtype_bytes = jnp.dtype(dtype).itemsize
    min_chunk_m = pl.cdiv(optimal_dma_bytes, size_n * dtype_bytes)
    chunk_m = max(min_chunk_m, size_lhs_sublane)
    chunk_m = 1 << (chunk_m - 1).bit_length()
    max_pow2_factor = size_m & -size_m
    if chunk_m > max_pow2_factor:
        return size_lhs_sublane
    return chunk_m


@jax.jit(static_argnames=[
    "tile_info",
    "vmem_limit_bytes",
    "precision",
    "preferred_element_type",
])
def gmm_v2(
    lhs: jax.Array,  # [size_m, size_k]
    rhs: jax.Array,  # [size_group, size_k, size_n]
    group_sizes: jax.Array,  # int32[size_lhs_group]
    rhs_scale: jax.Array | None = None,  # [size_group, 1, 1, out_size]
    rhs_bias: jax.Array | None = None,  # [size_group, 1, out_size]
    group_offset: jax.Array | None = None,  # int32[1]
    *,
    tile_info: TileSizes | TileFn = calculate_tiling,
    vmem_limit_bytes: int | None = None,
    precision: jax.lax.Precision = jax.lax.Precision.DEFAULT,
    preferred_element_type: jnp.dtype | None = None,
) -> jax.Array:
    """GMM kernel implemented with emit_pipeline.

    Dynamically calculate offset lhs/out tiles to reduce redundant computations.
    Additionally, adjusting dma size based on number of valid rows and utilize
    triple buffering on weights to better utilize memory.

    Args:
        lhs: lhs with shape [size_m, size_k].
        rhs: rhs with shape [size_group, size_k, size_n].
        group_sizes: The group sizes of lhs rows of shape [size_lhs_group,].
        rhs_scale: The rhs scale of shape [size_group, 1, 1, out_size].
        rhs_bias: The rhs bias of shape [size_group, 1, out_size].
        group_offset: Optional. The group offset of shape [1,].
        tile_info: The tile sizes or tile function to use.
        vmem_limit_bytes: Optional vmem limit in bytes.
        precision: Unused. Exists for compatibility reasons.
        preferred_element_type: Optional jnp.dtype for the output matrix.

  Returns:
        Output of shape [size_m, size_n].
  """

    del precision

    if group_offset is None:
        group_offset = jnp.array([0], dtype=jnp.int32)
    else:
        if jnp.isscalar(group_offset):
            group_offset = group_offset[None]

    if preferred_element_type is None:
        preferred_element_type = lhs.dtype

    if vmem_limit_bytes is None:
        vmem_limit_bytes = int(pltpu.get_tpu_info().vmem_capacity_bytes * 0.9)

    dims = validate_inputs(lhs, rhs, rhs_scale, rhs_bias, group_sizes,
                           group_offset)

    if isinstance(tile_info, TileSizes):
        tiles = tile_info
    else:
        tiles = tile_info(lhs.dtype, rhs.dtype, dims, vmem_limit_bytes)
    validate_tiles(tiles, dims)

    chunk_m = calculated_dma_chunk_m_size(dims.size_lhs_sublane, dims.size_m,
                                          dims.size_n, preferred_element_type)

    # Prepare block specs and input aliases.
    input_aliases = 4
    rhs_scale_spec = rhs_bias_spec = None
    if rhs_scale is not None:
        input_aliases += 1
        rhs_scale = rhs_scale.astype(jnp.float32)
        rhs_scale_spec = pl.BlockSpec(memory_space=pltpu.HBM)
    if rhs_bias is not None:
        input_aliases += 1
        rhs_bias = rhs_bias.astype(jnp.float32)
        rhs_bias_spec = pl.BlockSpec(memory_space=pltpu.HBM)

    # Initialize scratch shapes.
    max_num_gm = dims.size_group + dims.size_m // tiles.tile_m - 1
    m_num_sublane_units = tiles.tile_m // dims.size_lhs_sublane

    scratch_shapes = [
        # partial_out_ref
        pltpu.VMEM((dims.size_lhs_sublane, tiles.tile_n),
                   preferred_element_type),
        # acc_ref
        pltpu.VMEM(
            (m_num_sublane_units, dims.size_lhs_sublane, tiles.tile_n),
            jnp.float32,
        ),
        # metadata_ref
        MetadataRef(
            gm_id_to_group_id=pltpu.SMEM((max_num_gm, ), jnp.int32),
            gm_id_to_m_offset=pltpu.SMEM((max_num_gm + 1, ), jnp.int32),
        ),
        # semaphore_ref
        pltpu.SemaphoreType.DMA((1, )),
        # mini_zero_buf
        pltpu.VMEM((chunk_m, dims.size_n), preferred_element_type),
    ]

    # Prepare inputs.
    # TODO(kyuyeunk, kunjanp): Add support for fusing zero initialization.
    # out_init = jnp.zeros((dims.size_m, dims.size_n),
    #                      dtype=preferred_element_type)
    out_init = pl.empty((dims.size_m, dims.size_n),
                        dtype=preferred_element_type)
    rhs_weights = WeightsRef(weight=rhs, scale=rhs_scale, bias=rhs_bias)

    return pl.pallas_call(
        functools.partial(kernel_main, tiles=tiles, dims=dims,
                          chunk_m=chunk_m),
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
                pl.BlockSpec(memory_space=pltpu.HBM),
            ],
            out_specs=pl.BlockSpec(memory_space=pltpu.HBM),
            scratch_shapes=scratch_shapes,
        ),
        input_output_aliases={input_aliases: 0},
        compiler_params=pltpu.CompilerParams(
            vmem_limit_bytes=vmem_limit_bytes),
        name=get_scope_name(dims, tiles),
        cost_estimate=get_cost_estimate(lhs, rhs_weights, out_init.dtype,
                                        dims),
    )(group_sizes, group_offset, lhs, rhs_weights, out_init)


def is_supported_by_gmm_v2(lhs: jax.Array, rhs: jax.Array,
                           rhs_scale: jax.Array | None) -> bool:
    if rhs_scale is not None and rhs_scale.shape[1] != 1:
        # gmm_v2 does not support subchannel quantization.
        return False
    # gmm_v2 does not support implicit padding along lane dimension.
    num_lanes = pltpu.get_tpu_info().num_lanes
    if lhs.shape[-1] % num_lanes != 0 or rhs.shape[-1] % num_lanes != 0:
        return False
    # gmm_v2 does not support when lhs is not multiple of sublane size.
    if lhs.shape[0] % pltpu.get_tpu_info().num_sublanes:
        return False
    # Handle weird edge cases where inputs are already quantized.
    if lhs.dtype not in [jnp.bfloat16, jnp.float32]:
        return False
    return True
