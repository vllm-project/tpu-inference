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
"""Grouped matrix multiplication kernels for TPU written in Pallas."""

import functools
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Optional

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from tpu_inference.kernels.megablox import common
from tpu_inference.kernels.megablox.tuned_block_sizes import \
    get_tuned_block_sizes

partial = functools.partial

# ---------------------------------------------------------------------------
# AWQ tile-level unpacking inside Pallas
# ---------------------------------------------------------------------------
# AWQ packs 8 × uint4 values into one uint32 in the following logical order:
#   index 0 → bits  0: 3    index 1 → bits 16:19
#   index 2 → bits  4: 7    index 3 → bits 20:23
#   index 4 → bits  8:11    index 5 → bits 24:27
#   index 6 → bits 12:15    index 7 → bits 28:31
# The tuple below maps physical bit-positions so that the unpacked elements
# come out in ascending logical order (0, 1, 2, 3, 4, 5, 6, 7).
_AWQ_SHIFTS = (0, 16, 4, 20, 8, 24, 12, 28)
_AWQ_PACK_FACTOR = 8  # 8 × uint4 per uint32


def _awq_unpack_tile(packed_tile: jnp.ndarray) -> jnp.ndarray:
    """Unpack a (..., N_packed) uint32 tile → (..., N_packed*8) int8 tile.

    This implementation uses only bit-shift and mask ops that are legal inside
    Pallas TPU kernels (no ``bitcast_convert_type``).
    """
    mask = jnp.uint32(0xF)
    parts = []
    for s in _AWQ_SHIFTS:
        # Use int32 instead of int8 to avoid illegal minor-dim insertion on TPU
        parts.append(((packed_tile >> jnp.uint32(s)) & mask).astype(jnp.int32))
    stacked = jnp.stack(parts,
                        axis=-1)  # (..., N_packed, 8) — int32, legal on TPU
    return stacked.reshape(stacked.shape[:-2] + (-1, )).astype(jnp.int8)


# ---------------------------------------------------------------------------


def _validate_args(
    *,
    lhs: jnp.ndarray,
    rhs: jnp.ndarray,
    group_sizes: jnp.ndarray,
    rhs_scale: jnp.ndarray | None = None,
    rhs_bias: jnp.ndarray | None = None,
    rhs_zeros: jnp.ndarray | None = None,
    awq_pack_factor: int = 0,
):
    """Validates the arguments for the gmm function."""

    # Validate 'lhs'.
    if lhs.ndim != 2:
        raise ValueError(f"Expected 2-tensor for 'lhs' but got {lhs.ndim=}.")
    common.assert_is_supported_dtype(lhs.dtype)

    # Validate 'rhs'.
    if rhs.ndim != 3:
        raise ValueError(f"Expected 3-tensor for 'rhs' but got {rhs.ndim=}.")

    if awq_pack_factor > 0:
        # When AWQ packed, rhs is uint32 and output dim is packed.
        if rhs.dtype != jnp.uint32:
            raise ValueError(f"AWQ packed rhs must be uint32, got {rhs.dtype}")
        if rhs_zeros is None:
            raise ValueError("rhs_zeros is required when awq_pack_factor > 0")
        if rhs_scale is None:
            raise ValueError("rhs_scale is required when awq_pack_factor > 0")
        # k dimension is NOT packed; only the n (output) dimension is packed.
        if lhs.shape[1] != rhs.shape[1]:
            raise ValueError(
                "Expected 'lhs' and 'rhs' to have the same contracting dim. "
                f"But got {lhs.shape[1]=} and {rhs.shape[1]=}")
    else:
        common.assert_is_supported_dtype(rhs.dtype)
        if lhs.shape[1] != rhs.shape[1]:
            raise ValueError(
                "Expected 'lhs' and 'rhs' to have the same number of input "
                f"features. But got {lhs.shape[1]=} and {rhs.shape[1]=}")

    # Validate 'group_sizes'.
    if group_sizes.dtype != jnp.int32:
        raise ValueError(
            f"Expected 32-bit integer 'group_sizes' but got {group_sizes.dtype=}."
        )

    num_groups = rhs.shape[0]
    if awq_pack_factor > 0:
        out_size = rhs.shape[2] * awq_pack_factor
    else:
        out_size = rhs.shape[2]

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

    if rhs_zeros is not None:
        # Accept either 3D (E, blocks, n_packed) or 4D (E, blocks, 1, n_packed).
        if rhs_zeros.ndim not in (3, 4):
            raise ValueError(
                f"Expected 3- or 4-tensor for 'rhs_zeros' but got "
                f"{rhs_zeros.ndim=}.")
        if rhs_zeros.dtype != jnp.uint32:
            raise ValueError(
                f"Expected uint32 'rhs_zeros' but got {rhs_zeros.dtype=}.")


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


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class GroupMetadata:
    group_offsets: jnp.ndarray
    group_ids: jnp.ndarray
    m_tile_ids: jnp.ndarray


def make_group_metadata(
    *,
    group_sizes: jnp.ndarray,
    m: int,
    tm: int,
    start_group: jnp.ndarray,
    num_nonzero_groups: int,
    visit_empty_groups: bool = True,
) -> tuple[GroupMetadata, jnp.ndarray]:
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

    group_ends = jnp.cumsum(group_sizes)
    group_offsets = jnp.concatenate(
        [jnp.zeros(1, dtype=jnp.int32), group_ends])

    rounded_group_ends = ((group_ends + tm - 1) // tm * tm).astype(jnp.int32)

    group_starts = jnp.concatenate(
        [jnp.zeros(1, dtype=jnp.int32), group_ends[:-1]])
    rounded_group_starts = group_starts // tm * tm

    rounded_group_sizes = rounded_group_ends - rounded_group_starts
    rounded_group_sizes = jnp.where(group_sizes == 0, 0, rounded_group_sizes)

    group_tiles = rounded_group_sizes // tm

    if visit_empty_groups:
        group_tiles = jnp.where(group_sizes == 0, 1, group_tiles)

    tiles_m = _calculate_num_tiles(m, tm)
    group_ids = jnp.repeat(
        jnp.arange(num_groups, dtype=jnp.int32),
        group_tiles,
        total_repeat_length=tiles_m + num_groups - 1,
    )

    partial_tile_mask = jnp.logical_or((group_offsets[:-1] % tm) == 0,
                                       group_sizes == 0)
    if visit_empty_groups:
        partial_tile_mask = jnp.where(group_sizes == 0, 0, partial_tile_mask)

    partial_tile_ids = jnp.where(partial_tile_mask, tiles_m,
                                 group_offsets[:-1] // tm)

    tile_visits = (jnp.histogram(
        partial_tile_ids, bins=tiles_m, range=(0, tiles_m - 1))[0] + 1)

    m_tile_ids = jnp.repeat(
        jnp.arange(tiles_m, dtype=jnp.int32),
        tile_visits.astype(jnp.int32),
        total_repeat_length=tiles_m + num_groups - 1,
    )

    first_tile_in_shard = (group_ids < start_group).sum()
    group_ids = jnp.roll(group_ids, shift=-first_tile_in_shard, axis=0)
    m_tile_ids = jnp.roll(m_tile_ids, shift=-first_tile_in_shard, axis=0)

    iota = jnp.arange(num_groups, dtype=jnp.int32)
    active_group_mask = jnp.logical_and(iota <= end_group, iota >= start_group)
    group_tiles = jnp.where(active_group_mask, group_tiles, 0)
    num_tiles = group_tiles.sum()

    return GroupMetadata(group_offsets, group_ids, m_tile_ids), num_tiles


def _get_store_mask(
    *,
    grid_id: jnp.ndarray,
    group_metadata: GroupMetadata,
    tm: int,
    tn: int,
) -> jnp.ndarray:
    """Mask for rows that belong to the current group in the current tile."""
    group_id = group_metadata.group_ids[grid_id]
    group_start = group_metadata.group_offsets[group_id]
    group_end = group_metadata.group_offsets[group_id + 1]
    m_id = group_metadata.m_tile_ids[grid_id] * tm

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
    group_start = group_metadata.group_offsets[start_group]
    group_end = group_metadata.group_offsets[start_group + num_nonzero_groups]
    valid_mask = jax.lax.broadcasted_iota(jnp.int32, (out.shape[0], ), 0)
    valid_mask = (valid_mask >= group_start) & (valid_mask < group_end)
    return jnp.where(valid_mask[:, None], out, 0)


LutFn = Callable[[int, int, int], Optional[tuple[int, int, int]]]


@functools.partial(
    jax.jit,
    static_argnames=[
        "preferred_element_type",
        "tiling",
        "interpret",
        "awq_pack_factor",
    ],
)
def gmm(
    lhs: jnp.ndarray,
    rhs: jnp.ndarray,
    group_sizes: jnp.ndarray,
    preferred_element_type: jnp.dtype = jnp.float32,
    rhs_scale: jnp.ndarray | None = None,
    rhs_bias: jnp.ndarray | None = None,
    tiling: tuple[int, int, int] | LutFn | None = None,
    group_offset: jnp.ndarray | None = None,
    existing_out: jnp.ndarray | None = None,
    interpret: bool = False,
    rhs_zeros: jnp.ndarray | None = None,
    awq_pack_factor: int = 0,
) -> jnp.ndarray:
    """Compute lhs[sizes[i-1]:sizes[i], :] @ rhs for each group 'i'.

  Args:
    lhs: A 2d, jnp.ndarray with shape [m, k].
    rhs: A 3d, jnp.ndarray with shape [num_groups, k, n].
        When awq_pack_factor > 0: [num_groups, k, n // awq_pack_factor] uint32.
    group_sizes: A 1d, jnp.ndarray with shape [num_groups] and jnp.int32 dtype.
    preferred_element_type: jnp.dtype, the element type for the output matrix.
    rhs_scale: A 4d, jnp.ndarray with shape [num_groups, num_blocks, 1, n].
    rhs_bias: A 3d, jnp.ndarray with shape [num_groups, 1, n].
    tiling: 3-tuple of ints. The m, k and n-dimension tile sizes.
    group_offset: The group in group sizes to start computing from. This is
      particularly useful for when rhs num_groups is sharded.
    existing_out: Existing output to write to.
    interpret: Whether or not to run the kernel in interpret mode, helpful for
      testing and debugging.
    rhs_zeros: AWQ packed zero-points. Either 3D
        [num_groups, num_blocks, n // awq_pack_factor] or 4D
        [num_groups, num_blocks, 1, n // awq_pack_factor] uint32.
        Required when awq_pack_factor > 0.
    awq_pack_factor: Number of uint4 values packed per uint32 element (8 for
        AWQ 4-bit). 0 means AWQ unpacking is disabled.

  Returns:
    A 2d, jnp.ndarray with shape [m, n].
  """
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
        rhs_zeros=rhs_zeros,
        awq_pack_factor=awq_pack_factor,
    )

    # ---- Canonicalize rhs_zeros to 4D ----
    # Pallas TPU lowering requires the last two block-shape dimensions to be
    # divisible by 8 and 128 respectively (or equal to the array dimension).
    # A 3D rhs_zeros (E, num_blocks, n_packed) with block dim
    # num_quant_blocks_per_tk (often 1) in the second-to-last position
    # violates the "divisible by 8" rule.  Reshaping to 4D
    # (E, num_blocks, 1, n_packed) moves the problematic dimension to the
    # third position and makes the last-two dims (1, n_packed) which satisfy
    # the rule because 1 == array_dim and tn_packed % 128 == 0.
    if rhs_zeros is not None and rhs_zeros.ndim == 3:
        rhs_zeros = jnp.expand_dims(rhs_zeros, 2)

    # Gather shape information.
    m, k = lhs.shape[0], lhs.shape[1]
    if awq_pack_factor > 0:
        n = rhs.shape[-1] * awq_pack_factor
        n_packed = rhs.shape[-1]
    else:
        n = rhs.shape[-1]
        n_packed = n

    # If tiling is callable, look up the problem dimensions in the LUT. If no
    # tuned tile dimensions are available throw an error.
    if callable(tiling):
        tiling = tiling(m, k, n)
    elif tiling is None:
        tiling = get_tuned_block_sizes(
            m=m,
            k=k,
            n=n,
            num_total_groups=num_total_groups,
            num_current_groups=rhs.shape[0],
            lhs_dtype=str(lhs.dtype),
            rhs_dtype=str(rhs.dtype),
            quant_block_size=k,
        )

    if tiling is None:
        raise ValueError(
            f"No tuned tiling found for (m, k, n) = ({m}, {k}, {n})")

    tm, tk, tn = tiling

    # Validate tn alignment for AWQ packed path.
    # The Pallas TPU lowering requires the last dimension of every block shape
    # to be divisible by 128 (or equal to the full array dimension).  Because
    # packed rhs / rhs_zeros tiles have last-dim size  tn_packed = tn / pack,
    # we must ensure  tn >= 128 * awq_pack_factor  so that tn_packed >= 128.
    if awq_pack_factor > 0:
        min_tn = 128 * awq_pack_factor  # 1024 for 4-bit AWQ
        if tn < min_tn:
            tn = min_tn
        assert tn % awq_pack_factor == 0, (
            f"tn={tn} must be divisible by awq_pack_factor={awq_pack_factor}")
        tn_packed = tn // awq_pack_factor
    else:
        tn_packed = tn

    if rhs_scale is not None:
        assert isinstance(rhs_scale, jax.Array)
        assert rhs_scale.shape[0] == num_current_groups
        num_quant_blocks = rhs_scale.shape[1]
    else:
        num_quant_blocks = 1

    quant_block_size = k // num_quant_blocks

    if tk % quant_block_size != 0 and quant_block_size % tk != 0:
        tk = quant_block_size

    tiles_k, k_rem = _calculate_irregular_num_tiles(k, tk)
    tiles_n, n_rem = _calculate_irregular_num_tiles(n, tn)
    del n_rem

    num_quant_blocks_per_tk = pl.cdiv(tk, quant_block_size)

    # Create the metadata we need for computation.
    group_metadata, num_active_tiles = make_group_metadata(
        group_sizes=group_sizes,
        m=m,
        tm=tm,
        start_group=group_offset[0],
        num_nonzero_groups=rhs.shape[0],
        visit_empty_groups=False,
    )

    # ----- build the Pallas kernel closure -----
    # Capture awq_pack_factor as a Python-level constant so that the if/else
    # branches are resolved at trace time (no runtime overhead when disabled).
    _is_awq = awq_pack_factor > 0

    def kernel(
        group_metadata,
        group_offset,
        lhs,
        rhs,
        rhs_scale,
        rhs_bias,
        rhs_zeros,
        existing_out,
        out,
        acc_scratch,
    ):
        m_tile_ids = group_metadata.m_tile_ids
        del group_offset

        grid_id = pl.program_id(1)
        k_i = pl.program_id(2)

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

        def _accum(is_last_k_tile):
            if is_last_k_tile:
                mask_k_rem_lhs = partial(mask_k_rem, dim=1)
                mask_k_rem_rhs = partial(mask_k_rem, dim=0)
            else:

                def _wrapper(x):
                    return x

                mask_k_rem_lhs = _wrapper
                mask_k_rem_rhs = _wrapper

            loaded_lhs = mask_k_rem_lhs(lhs[...])
            loaded_rhs_raw = mask_k_rem_rhs(rhs[...])

            # ---- AWQ on-the-fly unpack ----
            if _is_awq:
                # loaded_rhs_raw: (tk, tn_packed) uint32
                # Unpack to (tk, tn) int8
                loaded_rhs = _awq_unpack_tile(loaded_rhs_raw)

                # rhs_zeros is 4D: loaded tile is
                # (num_quant_blocks_per_tk, 1, tn_packed) uint32.
                # Squeeze out the unit dim, then unpack.
                loaded_zeros_4d = rhs_zeros[...]
                loaded_zeros_raw = loaded_zeros_4d.reshape(
                    num_quant_blocks_per_tk, tn_packed)
                loaded_zeros = _awq_unpack_tile(
                    loaded_zeros_raw)  # (num_qb_per_tk, tn) int8
            else:
                loaded_rhs = loaded_rhs_raw

            acc = acc_scratch[...]

            for b_i in range(num_quant_blocks_per_tk):
                rhs_slice = loaded_rhs[b_i * quant_block_size:(b_i + 1) *
                                       quant_block_size, ...]
                lhs_slice = loaded_lhs[..., b_i * quant_block_size:(b_i + 1) *
                                       quant_block_size]

                # Subtract AWQ zero-point before the dot product.
                if _is_awq:
                    zeros_row = loaded_zeros[b_i:b_i + 1, :]  # (1, tn) int8
                    rhs_slice = (rhs_slice - zeros_row).astype(jnp.bfloat16)

                partial_result = jnp.dot(
                    lhs_slice,
                    rhs_slice,
                    preferred_element_type=jnp.float32,
                )

                if rhs_scale is not None:
                    partial_result *= jnp.broadcast_to(rhs_scale[b_i],
                                                       partial_result.shape)
                acc = acc + partial_result

            if is_last_k_tile:
                loaded_out = out[...].astype(jnp.float32)

                if rhs_bias is not None:
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
        lax.cond(
            is_last_k_tile,
            partial(_accum, True),
            partial(_accum, False),
        )

    # ----- index transforms -----

    def lhs_transform_indices(n_i, grid_id, k_i, group_metadata, group_offset):
        m_tile_ids = group_metadata.m_tile_ids
        del n_i, group_offset
        return m_tile_ids[grid_id], k_i

    def rhs_transform_indices(n_i, grid_id, k_i, group_metadata, group_offset):
        group_ids = group_metadata.group_ids
        return group_ids[grid_id] - group_offset[0], k_i, n_i

    def rhs_scale_transform_indices(n_i, grid_id, k_i, group_metadata,
                                    group_offset):
        group_ids = group_metadata.group_ids
        b_i = (k_i * tk) // quant_block_size
        b_tile_i = b_i // num_quant_blocks_per_tk
        return group_ids[grid_id] - group_offset[0], b_tile_i, 0, n_i

    def rhs_bias_transform_indices(n_i, grid_id, k_i, group_metadata,
                                   group_offset):
        group_ids = group_metadata.group_ids
        del k_i
        return group_ids[grid_id] - group_offset[0], 0, n_i

    def rhs_zeros_transform_indices(n_i, grid_id, k_i, group_metadata,
                                    group_offset):
        """Index into 4D rhs_zeros: (E, num_blocks, 1, n_packed)."""
        group_ids = group_metadata.group_ids
        b_i = (k_i * tk) // quant_block_size
        b_tile_i = b_i // num_quant_blocks_per_tk
        return group_ids[grid_id] - group_offset[0], b_tile_i, 0, n_i

    def out_transform_indices(n_i, grid_id, k_i, group_metadata, group_offset):
        m_tile_ids = group_metadata.m_tile_ids
        del k_i, group_offset
        return m_tile_ids[grid_id], n_i

    out_block_spec = pl.BlockSpec((tm, tn), out_transform_indices)

    if existing_out is None:
        in_out_block_spec: Any = None
        input_output_aliases = {}
    else:
        in_out_block_spec = out_block_spec
        input_output_aliases = {8: 0}  # existing_out → out

    lhs_block_spec = pl.BlockSpec((tm, tk), lhs_transform_indices)

    # rhs block spec uses tn_packed for AWQ path.
    rhs_block_spec = pl.BlockSpec((None, tk, tn_packed), rhs_transform_indices)

    if rhs_scale is None:
        rhs_scale_block_spec = None
    else:
        rhs_scale_block_spec = pl.BlockSpec(
            (None, num_quant_blocks_per_tk, 1, tn),
            rhs_scale_transform_indices)

    if rhs_bias is None:
        rhs_bias_block_spec = None
    else:
        rhs_bias_block_spec = pl.BlockSpec((None, 1, tn),
                                           rhs_bias_transform_indices)

    # rhs_zeros is 4D: (E, num_blocks, 1, n_packed).
    # Block spec mirrors rhs_scale layout so last-two dims are (1, tn_packed)
    # which satisfies Pallas TPU requirement: 1 == array_dim and
    # tn_packed % 128 == 0.
    if rhs_zeros is None:
        rhs_zeros_block_spec = None
    else:
        rhs_zeros_block_spec = pl.BlockSpec(
            (None, num_quant_blocks_per_tk, 1, tn_packed),
            rhs_zeros_transform_indices)

    # ----- cost estimate -----

    lhs_bytes = lhs.size * lhs.itemsize
    rhs_bytes = (k * n_packed) * rhs.itemsize
    if rhs_scale is not None:
        rhs_bytes += (num_quant_blocks * n) * rhs_scale.itemsize
    if rhs_bias is not None:
        rhs_bytes += n * rhs_bias.itemsize
    if rhs_zeros is not None:
        rhs_bytes += (num_quant_blocks * n_packed) * rhs_zeros.itemsize
    out_bytes = (m * n) * jnp.dtype(preferred_element_type).itemsize

    max_active_tiles = group_metadata.group_ids.size
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
                rhs_zeros_block_spec,
                in_out_block_spec,
            ],
            out_specs=out_block_spec,
            grid=(tiles_n, num_active_tiles, tiles_k),
            scratch_shapes=[pltpu.VMEM((tm, tn), jnp.float32)],
        ),
        input_output_aliases=input_output_aliases,
        compiler_params=pltpu.CompilerParams(dimension_semantics=(
            "parallel",
            "arbitrary",
            "arbitrary",
        )),
        interpret=interpret,
        cost_estimate=cost_estimate,
        name=f"gmm-m_{m}-k_{k}-n_{n}-tm_{tm}-tk_{tk}-tn_{tn}",
    )

    out = call_gmm(
        group_metadata,
        group_offset,
        lhs,
        rhs,
        rhs_scale,
        rhs_bias,
        rhs_zeros,
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
