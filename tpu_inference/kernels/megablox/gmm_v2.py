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
from typing import Tuple

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

    def lhs_index_map(self, _: jax.Array, gm_id: jax.Array, k_id: jax.Array):
        m_start = self.metadata_ref.gm_id_to_m_offset[gm_id]
        m_end = self.metadata_ref.gm_id_to_m_offset[gm_id + 1]

        sublane_start = m_start // self.dims.size_lhs_sublane
        sublane_end = pl.cdiv(m_end, self.dims.size_lhs_sublane)
        sublane_size = sublane_end - sublane_start
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
        m_start = self.metadata_ref.gm_id_to_m_offset[gm_id]
        m_end = self.metadata_ref.gm_id_to_m_offset[gm_id + 1]

        sublane_start = m_start // self.dims.size_lhs_sublane
        sublane_end = pl.cdiv(m_end, self.dims.size_lhs_sublane)
        sublane_size = sublane_end - sublane_start
        return (pl.ds(sublane_start, sublane_size), 0, n_id)


def generate_block_specs(
    lhs_ref: jax.Array,
    rhs_ref: WeightsRef,
    out_ref: jax.Array,
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
    tiled_lhs_ref: jax.Array,  # [tile_m // num_sublanes, num_sublanes, tile_k]
    tiled_rhs_ref: WeightsRef,  # [tile_k, tile_n]
    # Out
    tiled_out_ref: jax.Array,  # [tile_m // num_sublanes, num_sublanes, tile_n]
    # Scratch
    partial_out_ref: jax.Array,  # [num_sublanes, time_n]
    acc_ref: jax.Array,  # [tile_m // num_sublanes, num_sublanes, tile_n]
    *,
    metadata_ref: MetadataRef,
    tiles: TileSizes,
    dims: Dimensions,
):
    """Inner kernel invoked by emit_pipeline to perform matmul.

  tiled_lhs_ref and tiled_out_ref points to rows [m_start:m_end] of lhs and out.
  Additionally, m_start and m_end does not have to align with tile boundaries
  [m_offset:m_offset+tile_m]. Therefore, rows [m_offset:m_start] and
  [m_end]:m_offset+tile_m] of tiled_lhs_ref and tiled_out_ref will contain
  invalid data and needs to be masked out.

  Args:
    tiled_lhs_ref: Contains value lhs[m_start:m_end, k_start:k_end]
    tiled_rhs_ref: Contains value rhs[g_id, k_start:k_end, n_start:n_end]. where
      g_id is the group associated with lhs[m_start:m_end, :]
    tiled_out_ref: Contains value out[m_start:m_end, n_start:n_end]
    partial_out_ref: Contains last size_lhs_sublane rows of the previous output.
      If this is the first tile for grid[n_id, :, :], it will be initialized to
      zeros.
    acc_ref: Reference to the accumulator.
    metadata_ref: Reference to the metadata.
    tiles: Tile sizes.
    dims: Dimensions.
  """

    gm_id = pl.program_id(1)
    k_id = pl.program_id(2)

    # If this is the first tile for grid[n_id, :, :], we initialize the partial
    # out to zeros. Otherwise, partial out from last tile of grid[n_id-1, :, :]
    # will be used and cause numeric issues.
    @pl.when(gm_id + k_id == 0)
    def _():
        partial_out_ref[...] = jnp.zeros_like(partial_out_ref)

    m_start = metadata_ref.gm_id_to_m_offset[gm_id]
    m_end = metadata_ref.gm_id_to_m_offset[gm_id + 1]

    m_offset = (m_start // dims.size_lhs_sublane) * dims.size_lhs_sublane

    m_start_local = m_start - m_offset
    m_end_local = m_end - m_offset

    def matmul(is_first_k_step: bool, is_last_k_step: bool):
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

            # If the current step does not completely fill size_lhs_sublane rows,
            # these rows will be revisited by the next block. By storing the partial
            # rows into the partial_out_ref, the next step can read them and
            # accumulate to them.  Otherwise, initialize it to zeros.
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
    def matmul_t_t():
        matmul(is_first_k_step=True, is_last_k_step=True)

    @jax.named_scope("matmul_first")
    def matmul_t_f():
        matmul(is_first_k_step=True, is_last_k_step=False)

    @jax.named_scope("matmul")
    def matmul_f_f():
        matmul(is_first_k_step=False, is_last_k_step=False)

    @jax.named_scope("matmul_last")
    def matmul_f_t():
        matmul(is_first_k_step=False, is_last_k_step=True)

    # Select and execute matmul function based on the current step.
    num_k = pl.num_programs(2)
    is_first_k_step = k_id == 0
    is_last_k_step = k_id == num_k - 1

    lax.cond(
        is_first_k_step,
        lambda: lax.cond(
            is_last_k_step,
            matmul_t_t,
            matmul_t_f,
        ),
        lambda: lax.cond(
            is_last_k_step,
            matmul_f_t,
            matmul_f_f,
        ),
    )


def fill_metadata(
    lhs_group_sizes_ref: jax.Array,
    group_offset_ref: jax.Array,
    metadata_ref: MetadataRef,
    *,
    dims: Dimensions,
    tiles: TileSizes,
) -> jax.Array:
    """Fills the metadata for the given lhs group sizes and group offset.

  Iterates over the lhs group sizes and if the group id is valid, determines
  the number of gm tiles that are needed to process the current group. Then,
  it fills starting and ending offset for each gm tile (gm_id_to_m_offset),
  and the group id for each gm tile (gm_id_to_group_id).

  Args:
    lhs_group_sizes_ref: The group sizes of lhs.
    group_offset_ref: Offset of the first group to process.
    metadata_ref: Metadata that is used to determine the group id and m offsets
      for each gmm tile.
    dims: Dimensions of arguments.
    tiles: Tile sizes for this kernel.

  Returns:
    The number of gm tiles to process lhs with given group offset.
  """

    group_offset = group_offset_ref[0]
    max_num_group = group_offset + dims.size_group

    @jax.named_scope("inner_tm_loop")
    def inner_tm_loop(tm_id, start_m_offset, *, end_m_offset, group_id,
                      num_gm):
        local_offset = start_m_offset % dims.size_lhs_sublane
        tm_size = jnp.minimum(tiles.tile_m - local_offset,
                              end_m_offset - start_m_offset)

        metadata_ref.gm_id_to_group_id[num_gm + tm_id] = group_id

        next_m_offset = start_m_offset + tm_size
        metadata_ref.gm_id_to_m_offset[num_gm + tm_id] = start_m_offset
        metadata_ref.gm_id_to_m_offset[num_gm + tm_id + 1] = next_m_offset

        return next_m_offset

    @jax.named_scope("outer_group_loop")
    def outer_group_loop(lhs_group_id, carry):
        num_gm, start_m_offset = carry

        group_id = lhs_group_id - group_offset
        group_size = lhs_group_sizes_ref[lhs_group_id]
        end_m_offset = start_m_offset + group_size

        # We need to make sure that end_m_offset does not exceed size_m.
        capped_end_m_offset = jnp.minimum(end_m_offset, dims.size_m)
        capped_group_size = capped_end_m_offset - start_m_offset
        # Override group size to 0 if the current group does not need to be
        # processed as it comes before the group_offset.
        capped_group_size = jnp.where(0 <= group_id, capped_group_size, 0)

        # Assume following arguments:
        # - size_lhs_sublane = 4
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
        aligned_group_size = capped_group_size + local_offset
        curr_num_gm = pl.cdiv(aligned_group_size, tiles.tile_m)
        # Even if group_size is 0, if local_offset is not 0, cdiv will return 1.
        # Thus, we need to override it to 0.
        curr_num_gm = jnp.where(capped_group_size == 0, 0, curr_num_gm)

        tm_loop_fn = functools.partial(
            inner_tm_loop,
            end_m_offset=capped_end_m_offset,
            group_id=group_id,
            num_gm=num_gm,
        )
        lax.fori_loop(0, curr_num_gm, tm_loop_fn, start_m_offset)

        return num_gm + curr_num_gm, capped_end_m_offset

    num_gm, _ = lax.fori_loop(0, max_num_group, outer_group_loop, (0, 0))
    return num_gm


def kernel_main(
    # Scalar prefetch
    lhs_group_sizes_ref: jax.Array,  # [size_lhs_group]
    group_offset_ref: jax.Array,  # [1]
    # In
    lhs_ref: jax.Array,  # [size_m, size_k]
    rhs_ref: WeightsRef,  # [size_group, size_k, size_n]
    _: jax.Array,  # [size_m, size_k]
    # Out
    out_ref: jax.Array,  # [size_m, size_n]
    # Scratch memory
    metadata_ref: MetadataRef,
    partial_out_ref: jax.Array,  # [num_sublanes, tile_n]
    acc_ref: jax.Array,  # [tile_m // num_sublane, num_sublane, tile_n]
    *,
    tiles: TileSizes,
    dims: Dimensions,
):
    """Entry point for GMM kernel.

  Computes metadata to determine which rows of lhs needs processing and how
  they will be tiled. And then, invoke inner kernel using metadata.

  Uses the following notation:
  - g (group_size): rhs group dimension
  - m: Batch dimension
  - gm: Batch tiling dimension. Aligned to size_lhs_sublane and has tile size
    of tile_m. Skips over empty groups and account for revisited tiles.
  - k: in dimension
  - N: out dimension

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

    # Fill metadata buffer and return number of group & m interations.
    num_gm = fill_metadata(
        lhs_group_sizes_ref,
        group_offset_ref,
        metadata_ref,
        dims=dims,
        tiles=tiles,
    )

    in_block_specs, out_block_specs = generate_block_specs(
        lhs_ref,
        rhs_ref,
        out_ref,
        metadata_ref,
        tiles=tiles,
        dims=dims,
    )

    # Execute the inner kernel.
    pipeline_fn = pltpu.emit_pipeline(
        functools.partial(inner_kernel,
                          metadata_ref=metadata_ref,
                          tiles=tiles,
                          dims=dims),
        grid=(num_n, num_gm, num_k),
        in_specs=in_block_specs,
        out_specs=out_block_specs,
    )

    # Bounded slice requires second last dim to be aligned to the sublane size.
    # rhs_ref uses static tiling thus reshape is not needed.
    lhs_in = lhs_ref.reshape(-1, dims.size_lhs_sublane, dims.size_k)
    out_in = out_ref.reshape(-1, dims.size_lhs_sublane, dims.size_n)
    pipeline_fn(lhs_in, rhs_ref, out_in, scratches=[partial_out_ref, acc_ref])


def calculate_tiling(
    lhs_dtype: jnp.dtype,
    rhs_dtype: jnp.dtype,
    dims: Dimensions,
    vmem_limit_bytes: int,
) -> TileSizes:
    """Calculate optimal tile sizes for GMM kernel."""

    # TODO(kyuyeunk): Add logic for calculating tile_m.
    del lhs_dtype
    tile_m = 64

    rhs_bits = jax.dtypes.itemsize_bits(rhs_dtype)

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
    # num_k_tiles.
    last_valid_n_tiles = num_n_tiles = 1
    tile_n = dims.size_n
    n_rem = 0

    # Multiple k tiles will introduce accumulation overhead. Thus, we first try
    # reducing tile_n to fit rhs into vmem.

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


def get_scope_name(dims: Dimensions, tiles: TileSizes) -> str:
    return (
        f"gmm-g_{dims.size_group}-m_{dims.size_m}-k_{dims.size_k}"
        f"-n_{dims.size_n}-tm_{tiles.tile_m}-tk_{tiles.tile_k}-tn_{tiles.tile_n}"
    )


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


@jax.jit
def gmm_v2(
    lhs: jax.Array,  # [size_m, size_k]
    rhs: jax.Array,  # [size_group, size_k, size_n]
    group_sizes: jax.Array,  # [size_lhs_group]
    rhs_scale: jax.Array | None = None,  # [size_group, 1, 1, out_size]
    rhs_bias: jax.Array | None = None,  # [size_group, 1, out_size]
    group_offset: jax.Array | None = None,  # [1]
    *,
    tiles: TileSizes | None = None,
    vmem_limit_bytes: int | None = None,
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
    tiles: Optional. The tile sizes of the kernel.
    vmem_limit_bytes: Optional. The vmem limit in bytes.

  Returns:
    Output of shape [size_m, size_n].
  """

    if group_offset is None:
        group_offset = jnp.array([0], dtype=jnp.int32)
    else:
        if jnp.isscalar(group_offset):
            group_offset = group_offset[None]

    dims = validate_inputs(lhs, rhs, rhs_scale, rhs_bias, group_sizes,
                           group_offset)

    if vmem_limit_bytes is None:
        vmem_limit_bytes = int(pltpu.get_tpu_info().vmem_capacity_bytes * 0.8)

    if tiles is None:
        tiles = calculate_tiling(lhs.dtype, rhs.dtype, dims, vmem_limit_bytes)

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
    max_num_gm = dims.size_group + dims.size_m // tiles.tile_m + 1
    m_rows_per_sublane = tiles.tile_m // dims.size_lhs_sublane

    scratch_shapes = [
        # metadata_ref
        MetadataRef(
            gm_id_to_group_id=pltpu.SMEM((max_num_gm, ), jnp.int32),
            gm_id_to_m_offset=pltpu.SMEM((max_num_gm + 1, ), jnp.int32),
        ),
        # partial_out_ref
        pltpu.VMEM((dims.size_lhs_sublane, tiles.tile_n), lhs.dtype),
        # acc_ref
        pltpu.VMEM((m_rows_per_sublane, dims.size_lhs_sublane, tiles.tile_n),
                   jnp.float32),
    ]

    # Prepare inputs.
    # TODO(kyuyeunk, kunjanp): Add support for fusing zero initialization.
    out_init = jnp.zeros((dims.size_m, dims.size_n), dtype=lhs.dtype)
    rhs_weights = WeightsRef(weight=rhs, scale=rhs_scale, bias=rhs_bias)

    return pl.pallas_call(
        functools.partial(kernel_main, tiles=tiles, dims=dims),
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
    return True
