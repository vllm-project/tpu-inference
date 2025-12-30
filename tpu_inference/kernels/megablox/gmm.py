"""Grouped matrix multiplication kernels for TPU written in Pallas.
Updated: Optimized Quantized Path (Flattened Loop).
"""

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

    if lhs.shape[1] != rhs.shape[1]:
        raise ValueError(
            "Expected 'lhs' columns and 'rhs' rows (in_size) to match."
            f" But instead got {lhs.shape[1]=} and {rhs.shape[1]=}")

    # Validate 'group_sizes'.
    if group_sizes.dtype != jnp.int32:
        raise ValueError(
            f"Expected 32-bit integer 'group_sizes' but got {group_sizes.dtype=}."
        )

    num_groups, in_size, out_size = rhs.shape

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
    """Create the metadata needed for grouped matmul computation."""
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
    """Compute lhs[sizes[i-1]:sizes[i], :] @ rhs for each group 'i'."""

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

    m = lhs.shape[0]
    k = rhs.shape[1] 
    n = rhs.shape[2] 

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

    # Pallas requires the k-dimension of the lhs block (tk) to be a multiple of 128
    # (or equal to k).
    # We also need to align with quantization block_size.
    
    if block_size < 128:
        # Quantization block is smaller than physical tile.
        # We must load multiple quantization blocks per physical tile.
        if 128 % block_size != 0:
             raise ValueError(f"Unsupported: block_size {block_size} < 128 but does not divide 128.")
        tk = 128
        scales_per_tile = 128 // block_size
    else:
        # Quantization block is large. We can tile it.
        # Ensure tk divides block_size and is multiple of 128.
        if tk > block_size or block_size % tk != 0 or (tk % 128 != 0 and tk != k):
            # Find largest valid tk <= requested tk
            candidate_tk = (block_size // 128) * 128
            # If requested tk is smaller and valid, prefer that? 
            # Original logic preferred finding a valid tk close to block_size? 
            # Actually original logic checked if tk was valid, if not, reset.
            # Let's just find largest multiple of 128 that divides block_size.
            # But we should respect user's 'tk' if it is valid.
            
            if tk <= block_size and block_size % tk == 0 and (tk % 128 == 0 or tk == k):
                 pass # tk is fine
            else:
                 # Search for best tk
                 candidate = 128
                 best_tk = 128 # Default fallback
                 while candidate <= block_size:
                     if block_size % candidate == 0:
                         best_tk = candidate
                     candidate += 128
                 tk = best_tk
        scales_per_tile = 1

    # tiles_k is now the TOTAL number of K tiles
    tiles_k, k_rem = _calculate_irregular_num_tiles(k, tk)
    
    tiles_n, n_rem = _calculate_irregular_num_tiles(n, tn)
    del n_rem

    group_metadata, num_active_tiles = make_group_metadata(
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
        k_i = pl.program_id(2)

        @pl.when(k_i == 0)
        def _zero_acc():
            acc_scratch[...] = jnp.zeros_like(acc_scratch)

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
                def _wrapper(x): return x
                mask_k_rem_lhs = _wrapper
                mask_k_rem_rhs = _wrapper

            loaded_lhs = mask_k_rem_lhs(lhs[...])
            loaded_rhs = mask_k_rem_rhs(rhs[...])
            
            # Accumulated partial sum for this tile
            tile_acc = jnp.zeros(acc_scratch.shape, dtype=jnp.float32)
            
            sub_k = tk // scales_per_tile
            
            for j in range(scales_per_tile):
                # Slice the tile
                lhs_slice = loaded_lhs[:, j*sub_k : (j+1)*sub_k]
                rhs_slice = loaded_rhs[j*sub_k : (j+1)*sub_k, :]
                
                term = jax.lax.dot_general(
                    lhs_slice,
                    rhs_slice,
                    preferred_element_type=jnp.float32,
                    dimension_numbers=(((1, ), (0, )), ((), ())),
                )
                
                if rhs_scale is not None:
                    # rhs_scale loaded slice: (scales_per_tile, 1, tn)
                    # We select j-th scale: (1, tn)
                    scale_slice = rhs_scale[j, 0, :]
                    term *= jnp.broadcast_to(scale_slice, term.shape)
                
                tile_acc += term

            acc = acc_scratch[...] + tile_acc

            if is_last_k_tile:
                loaded_out = out[...].astype(jnp.float32)
                
                # Check if we need to add existing_out (only read once at the end)
                if existing_out is not None:
                    prev_grid_id = jnp.where(grid_id > 0, grid_id - 1, 0)
                    is_first_processed_group = grid_id == 0
                    m_tile_changed = m_tile_ids[grid_id] != m_tile_ids[prev_grid_id]
                    first_time_seeing_out = jnp.logical_or(is_first_processed_group, m_tile_changed)
                    
                    acc = jax.lax.select(first_time_seeing_out, 
                                         acc + existing_out[...], 
                                         acc)

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
        
        # Simple loop structure (pipeline-able)
        lax.cond(
            is_last_k_tile,
            partial(_accum, True),
            partial(_accum, False),
        )

    def lhs_transform_indices(n_i, grid_id, k_i, group_metadata, group_offset):
        # LHS: (m, k). Map k_i directly.
        group_offsets, group_ids, m_tile_ids = group_metadata
        del n_i, group_offsets, group_ids, group_offset
        return m_tile_ids[grid_id], k_i

    def rhs_transform_indices(n_i, grid_id, k_i, group_metadata, group_offset):
        # RHS: (group, k, n). Map k_i directly.
        group_offsets, group_ids, m_tile_ids = group_metadata
        del group_offsets, m_tile_ids
        return group_ids[grid_id] - group_offset[0], k_i, n_i

    def rhs_scale_transform_indices(n_i, grid_id, k_i, group_metadata, group_offset):
        # RHS Scale: (group, num_blocks, 1, n).
        group_offsets, group_ids, m_tile_ids = group_metadata
        del group_offsets, m_tile_ids
        
        # Calculate start block index
        # Each physical k-tile covers 'scales_per_tile' blocks.
        block_idx = k_i * scales_per_tile
        
        return group_ids[grid_id] - group_offset[0], block_idx, 0, n_i

    def rhs_bias_transform_indices(n_i, grid_id, k_i, group_metadata, group_offset):
        # RHS Bias: (group, 1, n).
        group_offsets, group_ids, m_tile_ids = group_metadata
        del group_offsets, m_tile_ids, k_i
        return group_ids[grid_id] - group_offset[0], 0, n_i

    def out_transform_indices(n_i, grid_id, k_i, group_metadata, group_offset):
        # Out: (m, n).
        group_offsets, group_ids, m_tile_ids = group_metadata
        del k_i, group_offsets, group_ids, group_offset
        return m_tile_ids[grid_id], n_i

    out_block_spec = pl.BlockSpec((tm, tn), out_transform_indices)
    
    if existing_out is None:
        in_out_block_spec: Any = None
        input_output_aliases = {}
    else:
        in_out_block_spec = out_block_spec
        input_output_aliases = {} 
    
    lhs_block_spec = pl.BlockSpec((tm, tk), lhs_transform_indices)
    rhs_block_spec = pl.BlockSpec((None, tk, tn), rhs_transform_indices)

    if rhs_scale is None:
        rhs_scale_block_spec = None
    else:
        # Scale tile size: (scales_per_tile, 1, tn)
        rhs_scale_block_spec = pl.BlockSpec((None, scales_per_tile, 1, tn),
                                            rhs_scale_transform_indices)

    if rhs_bias is None:
        rhs_bias_block_spec = None
    else:
        rhs_bias_block_spec = pl.BlockSpec((None, 1, tn),
                                           rhs_bias_transform_indices)
    
    # Cost Estimate Logic
    lhs_bytes = lhs.size * lhs.itemsize
    # RHS is read fully (K * N) across the K loop
    rhs_bytes = (k * n) * rhs.itemsize 
    if rhs_scale is not None:
        # Scales are read multiple times? 
        # Pallas caches, but for estimate: we read N * num_blocks
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

    # Grid Spec: Flattened
    # (tiles_n, num_active_tiles, tiles_k)
    
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