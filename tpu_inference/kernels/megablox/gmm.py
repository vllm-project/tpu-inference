# Copyright 2025 Google LLC
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
"""Grouped matrix multiplication kernels for TPU written in Pallas."""

import functools
from collections.abc import Callable
from typing import Any, Optional

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from tpu_inference.kernels.megablox import common

partial = functools.partial


def _validate_args(
    *,
    lhs: jnp.ndarray,
    rhs: jnp.ndarray,
    group_sizes: jnp.ndarray,
    rhs_scale: jnp.ndarray | None = None,
    rhs_bias: jnp.ndarray | None = None,
):
    """Validates the arguments for the gmm function."""
    # Validate 'lhs'.
    if lhs.ndim != 2:
        raise ValueError(f"Expected 2-tensor for 'lhs' but got {lhs.ndim=}.")
    common.assert_is_supported_dtype(lhs.dtype)

    # Validate 'rhs'.
    if rhs.ndim != 3:
        raise ValueError(f"Expected 3-tensor for 'rhs' but got {rhs.ndim=}.")
    common.assert_is_supported_dtype(rhs.dtype)

    if lhs.shape[1] != rhs.shape[2]:
        raise ValueError(
            "Expected 'lhs' and 'rhs' to have the same number of input features."
            f" But instead got {lhs.shape[1]=} and {rhs.shape[2]=}")

    # Validate 'group_sizes'.
    if group_sizes.dtype != jnp.int32:
        raise ValueError(
            f"Expected 32-bit integer 'group_sizes' but got {group_sizes.dtype=}."
        )

    num_groups, out_size, in_size = rhs.shape

    if rhs_scale is not None:
        # Validate 'rhs_scale'.
        if rhs_scale.ndim != 4:
            raise ValueError(
                f"Expected 4-tensor for 'rhs_scale' but got {rhs_scale.ndim=}."
            )
        expected_rhs_scale_shape = (num_groups, rhs_scale.shape[1], 1,
                                    out_size)
        if rhs_scale.shape != expected_rhs_scale_shape:
            raise ValueError(
                "Expected 'rhs_scale' to have the shape of"
                f" {expected_rhs_scale_shape} but got {rhs_scale.shape=}.")

    if rhs_bias is not None:
        # Validate 'rhs_bias'.
        if rhs_bias.ndim != 3:
            raise ValueError(
                f"Expected 3-tensor for 'rhs_bias' but got {rhs_bias.ndim=}.")
        expected_rhs_bias_shape = (num_groups, 1, out_size)
        if rhs_bias.shape != expected_rhs_bias_shape:
            raise ValueError(
                "Expected 'rhs_bias' to have the shape of"
                f" {expected_rhs_bias_shape} but got {rhs_bias.shape=}.")


def _calculate_num_tiles(x: int, tx: int) -> int:
    tiles, rem = divmod(x, tx)
    if rem:
        raise ValueError(
            f"{x} must be divisible by x-dimension tile size ({tx}).")
    return tiles


def _calculate_irregular_num_tiles(x: int, tx: int) -> tuple[int, int]:
    tiles, rem = divmod(x, tx)
    if rem:
        tiles += 1
    return tiles, rem


GroupMetadata = Any  # TODO(enriqueps): Clean this up and use a namedtuple


def make_group_metadata(
    *,
    group_sizes: jnp.ndarray,
    m: int,
    tm: int,
    start_group: jnp.ndarray,
    num_nonzero_groups: int,
    visit_empty_groups: bool = True,
) -> GroupMetadata:
    """Create the metadata needed for grouped matmul computation.

  Args:
    group_sizes: A 1d, jnp.ndarray with shape [num_groups] and jnp.int32 dtype.
    m: The number of rows in lhs.
    tm: The m-dimension tile size being used.
    start_group: The group in group sizes to start computing from. This is
      particularly useful for when rhs num_groups is sharded.
    num_nonzero_groups: Number of groups in group sizes to compute on. Useful in
      combination with group_offset.
    visit_empty_groups: If True, do not squeeze tiles for empty groups out of
      the metadata. This is necessary for tgmm, where we at least need to zero
      the output for each group.

  Returns:
    tuple of:
      group_offsets: A 1d, jnp.ndarray with shape [num_groups+1] and jnp.int32
        dtype. group_offsets[i] indicates the row at which group [i] starts in
        the lhs matrix and group_offsets[i-1] = m.
      group_ids: A 1d, jnp.ndarray with shape [m_tiles + num_groups] and
        jnp.int32 dtype. group_ids[i] indicates which group grid index 'i' will
        work on.
      m_tile_ids: A 1d, jnp.ndarray with shape [m_tiles + num_groups] and
        jnp.int32. m_tile_ids[i] indicates which m-dimension tile grid index 'i'
        will work on.
    num_tiles: The number of m-dimension tiles to execute.
  """
    num_groups = group_sizes.shape[0]
    end_group = start_group + num_nonzero_groups - 1

    # Calculate the offset of each group, starting at zero. This metadata is
    # similar to row offsets in a CSR matrix. The following properties hold:
    #
    # group_offsets.shape = [num_groups + 1]
    # group_offsets[0] = 0
    # group_offsets[num_groups] = m
    #
    # The row at which group 'i' starts is group_offsets[i].
    group_ends = jnp.cumsum(group_sizes)
    group_offsets = jnp.concatenate(
        [jnp.zeros(1, dtype=jnp.int32), group_ends])

    # Assign a group id to each grid index.
    #
    # If a group starts somewhere other than the start of a tile or ends somewhere
    # other than the end of a tile we need to compute that full tile. Calculate
    # the number of tiles for each group by rounding their end up to the nearest
    # 'tm' and their start down to the nearest 'tm'.

    # (1) Round the group_ends up to the nearest multiple of 'tm'.
    #
    # NOTE: This does not change group_offsets[num_groups], which is m
    # (because we enforce m is divisible by tm).
    rounded_group_ends = ((group_ends + tm - 1) // tm * tm).astype(jnp.int32)

    # (2) Round the group_starts down to the nearest multiple of 'tm'.
    group_starts = jnp.concatenate(
        [jnp.zeros(1, dtype=jnp.int32), group_ends[:-1]])
    rounded_group_starts = group_starts // tm * tm

    # (3) Calculate the number of rows in each group.
    #
    # NOTE: Handle zero-sized groups as a special case. If the start for a
    # zero-sized group is not divisible by 'tm' its start will be rounded down and
    # its end will be rounded up such that its size will become 1 tile here.
    rounded_group_sizes = rounded_group_ends - rounded_group_starts
    rounded_group_sizes = jnp.where(group_sizes == 0, 0, rounded_group_sizes)

    # (4) Convert the group sizes from units of rows to unit of 'tm' sized tiles.
    #
    # An m-dimension tile is 'owned' by group 'i' if the first row of the tile
    # belongs to group 'i'. In addition to owned tiles, each group can have 0 or 1
    # initial partial tiles if it's first row does not occur in the first row of a
    # tile. The '0-th' group never has a partial tile because it always starts at
    # the 0-th row.
    #
    # If no group has a partial tile, the total number of tiles is equal to
    # 'm // tm'. If every group has a partial except the 0-th group, the total
    # number of tiles is equal to 'm // tm + num_groups - 1'. Thus we know that
    #
    # tiles_m <= group_tiles.sum() <= tiles_m + num_groups - 1
    #
    # Where tiles_m = m // tm.
    #
    # NOTE: All group sizes are divisible by 'tm' because of the rounding in steps
    # (1) and (2) so this division is exact.
    group_tiles = rounded_group_sizes // tm

    if visit_empty_groups:
        # Insert one tile for empty groups.
        group_tiles = jnp.where(group_sizes == 0, 1, group_tiles)

    # Create the group ids for each grid index based on the tile counts for each
    # group.
    #
    # NOTE: This repeat(...) will pad group_ids with the final group id if
    # group_tiles.sum() < tiles_m + num_groups - 1. The kernel grid will be sized
    # such that we only execute the necessary number of tiles.
    tiles_m = _calculate_num_tiles(m, tm)
    group_ids = jnp.repeat(
        jnp.arange(num_groups, dtype=jnp.int32),
        group_tiles,
        total_repeat_length=tiles_m + num_groups - 1,
    )

    # Assign an m-dimension tile id to each grid index.
    #
    # NOTE: Output tiles can only be re-visited consecutively. The following
    # procedure guarantees that m-dimension tile indices respect this.

    # (1) Calculate how many times each m-dimension tile will be visited.
    #
    # Each tile is guaranteed to be visited once by the group that owns the tile.
    # The remaining possible visits occur when a group starts inside of a tile at
    # a position other than the first row. We can calculate which m-dimension tile
    # each group starts in by floor-dividing its offset with `tm` and then count
    # tile visits with a histogram.
    #
    # To avoid double counting tile visits from the group that owns the tile,
    # filter these out by assigning their tile id to `tile_m` (one beyond the max)
    # such that they're ignored by the subsequent histogram. Also filter out any
    # group which is empty.
    #
    # TODO(tgale): Invert the 'partial_tile_mask' predicates to be more clear.
    partial_tile_mask = jnp.logical_or((group_offsets[:-1] % tm) == 0,
                                       group_sizes == 0)

    # Explicitly enable tiles for zero sized groups, if specified. This covers
    # zero sized groups that start on a tile-aligned row and those that do not.
    if visit_empty_groups:
        partial_tile_mask = jnp.where(group_sizes == 0, 0, partial_tile_mask)

    partial_tile_ids = jnp.where(partial_tile_mask, tiles_m,
                                 group_offsets[:-1] // tm)

    tile_visits = (jnp.histogram(
        partial_tile_ids, bins=tiles_m, range=(0, tiles_m - 1))[0] + 1)

    # Create the m-dimension tile ids for each grid index based on the visit
    # counts for each tile.
    m_tile_ids = jnp.repeat(
        jnp.arange(tiles_m, dtype=jnp.int32),
        tile_visits.astype(jnp.int32),
        total_repeat_length=tiles_m + num_groups - 1,
    )

    # Account for sharding.
    #
    # Find the start of the groups owned by our shard and shift the group_ids and
    # m_tile_ids s.t. the metadata for our tiles are at the front of the arrays.
    #
    # TODO(tgale): Move this offset into the kernel to avoid these rolls.
    first_tile_in_shard = (group_ids < start_group).sum()
    group_ids = jnp.roll(group_ids, shift=-first_tile_in_shard, axis=0)
    m_tile_ids = jnp.roll(m_tile_ids, shift=-first_tile_in_shard, axis=0)

    # Calculate the number of tiles we need to compute for our shard.
    #
    # Remove tile visits that belong to a group not in our shard.
    iota = jnp.arange(num_groups, dtype=jnp.int32)
    active_group_mask = jnp.logical_and(iota <= end_group, iota >= start_group)
    group_tiles = jnp.where(active_group_mask, group_tiles, 0)
    num_tiles = group_tiles.sum()
    return (group_offsets, group_ids, m_tile_ids), num_tiles


def _get_store_mask(
    *,
    grid_id: jnp.ndarray,
    group_metadata: GroupMetadata,
    tm: int,
    tn: int,
) -> jnp.ndarray:
    """Mask for rows that belong to the current group in the current tile."""
    group_offsets, group_ids, m_tile_ids = group_metadata[:3]
    group_id = group_ids[grid_id]
    group_start = group_offsets[group_id]
    group_end = group_offsets[group_id + 1]
    m_id = m_tile_ids[grid_id] * tm
    iota = jax.lax.broadcasted_iota(jnp.int32, (tm, tn), 0) + m_id
    return jnp.logical_and(iota >= group_start, iota < group_end)


def _zero_uninitialized_memory(
    out: jnp.ndarray,
    *,
    start_group: jnp.ndarray,
    num_nonzero_groups: int,
    group_metadata: GroupMetadata,
) -> jnp.ndarray:
    """Zero out uninitialized memory from output."""
    group_offsets = group_metadata[0]
    group_start = group_offsets[start_group]
    group_end = group_offsets[start_group + num_nonzero_groups]
    valid_mask = jax.lax.broadcasted_iota(jnp.int32, (out.shape[0], ), 0)
    valid_mask = (valid_mask >= group_start) & (valid_mask < group_end)
    return jnp.where(valid_mask[:, None], out, 0)


LutFn = Callable[[int, int, int], Optional[tuple[int, int, int]]]


@functools.partial(
    jax.jit,
    static_argnames=[
        "preferred_element_type",
        "tiling",
        "transpose_rhs",
        "interpret",
    ],
)
def gmm(
    lhs: jnp.ndarray,
    rhs: jnp.ndarray,
    group_sizes: jnp.ndarray,
    preferred_element_type: jnp.dtype = jnp.float32,
    rhs_scale: jnp.ndarray | None = None,
    rhs_bias: jnp.ndarray | None = None,
    tiling: tuple[int, int, int] | LutFn | None = (128, 128, 128),
    group_offset: jnp.ndarray | None = None,
    existing_out: jnp.ndarray | None = None,
    transpose_rhs: bool = False,
    interpret: bool = False,
) -> jnp.ndarray:
    """Compute lhs[sizes[i-1]:sizes[i], :] @ rhs for each group 'i'.

  Args:
    lhs: A 2d, jnp.ndarray with shape [m, k].
    rhs: A 3d, jnp.ndarray with shape [num_groups, n, k].
    group_sizes: A 1d, jnp.ndarray with shape [num_groups] and jnp.int32 dtype.
    preferred_element_type: jnp.dtype, the element type for the output matrix.
    rhs_scale: A 4d, jnp.ndarray with shape [num_groups, num_blocks, 1, n].
    rhs_bias: A 3d, jnp.ndarray with shape [num_groups, 1, n].
    tiling: 3-tuple of ints. The m, k and n-dimension tile sizes.
    group_offset: The group in group sizes to start computing from. This is
      particularly useful for when rhs num_groups is sharded.
    existing_out: Existing output to write to.
    transpose_rhs: True if the rhs needs to be transposed.
    interpret: Whether or not to run the kernel in interpret mode, helpful for
      testing and debugging.

  Returns:
    A 2d, jnp.ndarray with shape [m, n].
  """

    # TODO(kyuyeunk): Instead of transpose_rhs==True, modify logic to only
    # transpose_rhs==False instead as it simplifies the logic in kernel.
    assert transpose_rhs

    if existing_out is not None:
        assert isinstance(existing_out, jax.Array)
        expected_dtype = existing_out.dtype
        if expected_dtype != preferred_element_type:
            raise ValueError(
                "Existing output dtype must match preferred_element_type.")
    if group_offset is None:
        group_offset = jnp.array([0], dtype=jnp.int32)
    else:
        if group_offset.shape:
            raise ValueError(
                f"group_offset must be a ()-shaped array. Got: {group_offset.shape}."
            )
        group_offset = group_offset[None]
    num_current_groups = rhs.shape[0]
    num_total_groups = group_sizes.shape[0]
    _validate_args(
        lhs=lhs,
        rhs=rhs,
        group_sizes=group_sizes,
        rhs_scale=rhs_scale,
        rhs_bias=rhs_bias,
    )

    # Gather shape information.
    m, k, n = (lhs.shape[0], lhs.shape[1], rhs.shape[1])

    # If tiling is callable, look up the problem dimensions in the LUT. If no
    # tuned tile dimensions are available throw an error.
    if callable(tiling):
        tiling = tiling(m, k, n)

    if tiling is None:
        raise ValueError(
            f"No tuned tiling found for (m, k, n) = ({m}, {k}, {n})")

    tm, tk, tn = tiling

    if rhs_scale is not None:
        assert isinstance(rhs_scale, jax.Array)
        assert rhs_scale.shape[0] == num_current_groups
        num_quant_blocks = rhs_scale.shape[1]
    else:
        num_quant_blocks = 1
    block_size = k // num_quant_blocks

    if tk > block_size or block_size % tk != 0:
        tk = block_size

    tiles_k, k_rem = _calculate_irregular_num_tiles(k, tk)
    tiles_n, n_rem = _calculate_irregular_num_tiles(n, tn)
    del n_rem

    tiles_k //= num_quant_blocks

    # Create the metadata we need for computation.
    group_metadata, num_active_tiles = make_group_metadata(  # pylint: disable=unbalanced-tuple-unpacking
        group_sizes=group_sizes,
        m=m,
        tm=tm,
        start_group=group_offset[0],
        num_nonzero_groups=rhs.shape[0],
        visit_empty_groups=False,
    )

    def kernel(
        group_metadata,
        group_offset,
        lhs,
        rhs,
        rhs_scale,
        rhs_bias,
        existing_out,
        out,
        acc_scratch,
    ):
        group_offsets, group_ids, m_tile_ids = group_metadata
        del group_offsets, group_ids, group_offset

        grid_id = pl.program_id(1)
        b_i = pl.program_id(2)
        k_i = pl.program_id(3)

        @pl.when(k_i == 0)
        def _zero_acc():
            acc_scratch[...] = jnp.zeros_like(acc_scratch)

            if existing_out is not None:
                prev_grid_id = jnp.where(grid_id > 0, grid_id - 1, 0)
                is_first_processed_group = grid_id == 0
                m_tile_changed = m_tile_ids[grid_id] != m_tile_ids[
                    prev_grid_id]
                first_time_seeing_out = jnp.logical_or(
                    is_first_processed_group, m_tile_changed)

                @pl.when(first_time_seeing_out)
                def _init_out():
                    out[...] = existing_out[...]

        def mask_k_rem(x, *, dim):
            if k_rem == 0:
                return x

            orig_dtype = x.dtype
            iota = lax.broadcasted_iota(jnp.int32, x.shape, dim)
            x = x.astype(jnp.float32)
            return jnp.where(iota < k_rem, x, 0).astype(orig_dtype)

        def _accum(is_last_k_tile, is_first_b_tile):
            if is_last_k_tile:
                mask_k_rem_lhs = partial(mask_k_rem, dim=1)
                mask_k_rem_rhs = partial(mask_k_rem, dim=1)
            else:

                def _wrapper(x):
                    return x

                mask_k_rem_lhs = _wrapper
                mask_k_rem_rhs = _wrapper

            loaded_lhs = lhs[...]
            loaded_rhs = rhs[...]

            acc = acc_scratch[...] + jax.lax.dot_general(
                mask_k_rem_lhs(loaded_lhs),
                mask_k_rem_rhs(loaded_rhs),
                preferred_element_type=jnp.float32,
                dimension_numbers=(((1, ), (1, )), ((), ())),
            )

            if is_last_k_tile:
                if rhs_scale is not None:
                    acc *= jnp.broadcast_to(rhs_scale[...], acc.shape)

                loaded_out = out[...].astype(jnp.float32)
                if not is_first_b_tile:
                    acc += loaded_out
                elif rhs_bias is not None:
                    acc += rhs_bias[...].astype(jnp.float32)

                mask = _get_store_mask(
                    grid_id=grid_id,
                    group_metadata=group_metadata,
                    tm=tm,
                    tn=tn,
                )
                out[...] = jax.lax.select(
                    mask[...], acc, loaded_out).astype(preferred_element_type)
            else:
                acc_scratch[...] = acc

        is_last_k_tile = k_i == (tiles_k - 1)
        is_first_b_tile = b_i == 0

        lax.cond(
            is_last_k_tile,
            lambda: lax.cond(
                is_first_b_tile,
                partial(_accum, True, True),
                partial(_accum, True, False),
            ),
            partial(_accum, False, False),
        )

    def lhs_transform_indices(n_i, grid_id, b_i, k_i, group_metadata,
                              group_offset):
        # lhs is (m, k). Load the [tm, tk] matrix for this m-tile.
        group_offsets, group_ids, m_tile_ids = group_metadata
        del n_i, group_offsets, group_ids, group_offset
        return m_tile_ids[grid_id], b_i * tiles_k + k_i

    def rhs_transform_indices(n_i, grid_id, b_i, k_i, group_metadata,
                              group_offset):
        # rhs is (num_groups, k, n). Load the [tk, tn] matrix based on the group id
        # for this m-tile.
        group_offsets, group_ids, m_tile_ids = group_metadata
        del group_offsets, m_tile_ids

        # NOTE: If we're working on only a shard of the rhs we need to adjust the
        # group index we load from to account for this. The group_ids are in the
        # "unsharded" domain.
        return group_ids[grid_id] - group_offset[0], n_i, b_i * tiles_k + k_i

    def rhs_scale_transform_indices(n_i, grid_id, b_i, k_i, group_metadata,
                                    group_offset):
        group_offsets, group_ids, m_tile_ids = group_metadata
        del group_offsets, m_tile_ids, k_i
        return group_ids[grid_id] - group_offset[0], b_i, 0, n_i

    def rhs_bias_transform_indices(n_i, grid_id, b_i, k_i, group_metadata,
                                   group_offset):
        group_offsets, group_ids, m_tile_ids = group_metadata
        del group_offsets, m_tile_ids, k_i, b_i
        return group_ids[grid_id] - group_offset[0], 0, n_i

    def out_transform_indices(n_i, grid_id, b_i, k_i, group_metadata,
                              group_offset):
        # out is (m, n). Load the [tm, tn] matrix for this m-tile.
        group_offsets, group_ids, m_tile_ids = group_metadata
        del k_i, group_offsets, group_ids, group_offset, b_i
        return m_tile_ids[grid_id], n_i

    out_block_spec = pl.BlockSpec((tm, tn), out_transform_indices)
    if existing_out is None:
        in_out_block_spec: Any = None
        input_output_aliases = {}
    else:
        in_out_block_spec = out_block_spec
        input_output_aliases = {7: 0}

    lhs_block_spec = pl.BlockSpec((tm, tk), lhs_transform_indices)
    rhs_block_spec = pl.BlockSpec((None, tn, tk), rhs_transform_indices)

    if rhs_scale is None:
        rhs_scale_block_spec = None
    else:
        rhs_scale_block_spec = pl.BlockSpec((None, None, 1, tn),
                                            rhs_scale_transform_indices)

    if rhs_bias is None:
        rhs_bias_block_spec = None
    else:
        rhs_bias_block_spec = pl.BlockSpec((None, 1, tn),
                                           rhs_bias_transform_indices)

    lhs_bytes = lhs.size * lhs.itemsize
    rhs_bytes = (k * n) * rhs.itemsize  # We don't read all of rhs
    if rhs_scale is not None:
        rhs_bytes += (num_quant_blocks * n) * rhs_scale.itemsize
    if rhs_bias is not None:
        rhs_bytes += n * rhs_bias.itemsize
    out_bytes = (m * n) * jnp.dtype(preferred_element_type).itemsize
    max_active_tiles = group_metadata[1].size
    bytes_accessed = ((lhs_bytes * tiles_n) + (rhs_bytes * max_active_tiles) +
                      out_bytes)
    flops = 2 * m * k * n
    cost_estimate = pl.CostEstimate(flops=flops,
                                    bytes_accessed=bytes_accessed,
                                    transcendentals=0)
    call_gmm = pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((m, n), preferred_element_type),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=2,
            in_specs=[
                lhs_block_spec,
                rhs_block_spec,
                rhs_scale_block_spec,
                rhs_bias_block_spec,
                in_out_block_spec,
            ],
            out_specs=out_block_spec,
            grid=(tiles_n, num_active_tiles, num_quant_blocks, tiles_k),
            scratch_shapes=[pltpu.VMEM((tm, tn), jnp.float32)],
        ),
        input_output_aliases=input_output_aliases,
        compiler_params=pltpu.CompilerParams(dimension_semantics=(
            "parallel",
            "arbitrary",
            "arbitrary",
            "arbitrary",
        )),
        interpret=interpret,
        cost_estimate=cost_estimate,
    )

    out = call_gmm(
        group_metadata,
        group_offset,
        lhs,
        rhs,
        rhs_scale,
        rhs_bias,
        existing_out,
    )
    if existing_out is None and num_current_groups < num_total_groups:
        out = _zero_uninitialized_memory(
            out,
            start_group=group_offset[0],
            num_nonzero_groups=rhs.shape[0],
            group_metadata=group_metadata,
        )
    return out
