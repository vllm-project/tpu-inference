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
"""Grouped matrix multiplication kernels for TPU written in Pallas.
Based on the code in tpu_inference/kernels/megablox/gmm.py but modifies the
following:

1. Flattens the K-dimension loop: The nested grid structure (num_quant_blocks, tiles_per_block)
   is replaced by a single flattened loop over all K tiles. Block indices for quantization
   are now derived mathematically (integer division) inside the kernel.
2. Quantization Logic: `rhs_scale` is now applied to the partial product of each tile
   immediately before accumulation, rather than adjusting the accumulator at the end of a block.
"""

import functools
from typing import Any

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

# yapf: disable
from tpu_inference.kernels.megablox.gmm import (LutFn,
                                                _calculate_irregular_num_tiles,
                                                _get_store_mask,
                                                _validate_args,
                                                _zero_uninitialized_memory,
                                                make_group_metadata)

# yapf: enable
partial = functools.partial


@partial(
    jax.jit,
    static_argnames=[
        "preferred_element_type",
        "tiling",
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
        interpret: Whether or not to run the kernel in interpret mode, helpful for
        testing and debugging.

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

    if tk > block_size or block_size % tk != 0:
        tk = block_size

    # tiles_k is now the TOTAL number of K tiles
    tiles_k, k_rem = _calculate_irregular_num_tiles(k, tk)
    # Calculate tiles per block for indexing logic
    tiles_per_block = tiles_k // num_quant_blocks

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

                def _wrapper(x):
                    return x

                mask_k_rem_lhs = _wrapper
                mask_k_rem_rhs = _wrapper

            loaded_lhs = lhs[...]
            loaded_rhs = rhs[...]

            # Calculate partial product for this tile
            partial_acc = jax.lax.dot_general(
                mask_k_rem_lhs(loaded_lhs),
                mask_k_rem_rhs(loaded_rhs),
                preferred_element_type=jnp.float32,
                dimension_numbers=(((1, ), (0, )), ((), ())),
            )

            if rhs_scale is not None:
                partial_acc *= jnp.broadcast_to(rhs_scale[...],
                                                partial_acc.shape)

            acc = acc_scratch[...] + partial_acc

            if is_last_k_tile:
                loaded_out = out[...].astype(jnp.float32)
                # Check if we need to add existing_out (only read once at the end)
                if existing_out is not None:
                    prev_grid_id = jnp.where(grid_id > 0, grid_id - 1, 0)
                    is_first_processed_group = grid_id == 0
                    m_tile_changed = m_tile_ids[grid_id] != m_tile_ids[
                        prev_grid_id]
                    first_time_seeing_out = jnp.logical_or(
                        is_first_processed_group, m_tile_changed)

                    acc = jax.lax.select(first_time_seeing_out,
                                         acc + existing_out[...], acc)

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

    def rhs_scale_transform_indices(n_i, grid_id, k_i, group_metadata,
                                    group_offset):
        # RHS Scale: (group, num_blocks, 1, n).
        # We must map the global tile index k_i to the block index.
        group_offsets, group_ids, m_tile_ids = group_metadata
        del group_offsets, m_tile_ids

        # Calculate block index
        block_idx = k_i // tiles_per_block

        return group_ids[grid_id] - group_offset[0], block_idx, 0, n_i

    def rhs_bias_transform_indices(n_i, grid_id, k_i, group_metadata,
                                   group_offset):
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
        # Scale tile size: (1, tn)
        rhs_scale_block_spec = pl.BlockSpec((None, None, 1, tn),
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
