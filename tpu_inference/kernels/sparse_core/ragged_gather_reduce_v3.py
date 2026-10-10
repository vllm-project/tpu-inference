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
"""Experimental destination-major SparseCore gather-reduce.

The v2 kernel streams valid routes and scatters FP32 partial rows.  This
prototype instead assigns each core a contiguous block of destination tokens.
Valid routes are compacted across each block, gathered exactly once, scattered
into FP32 VMEM with SparseCore vector scatter-add, and emitted as full 16-row
BF16 tiles.
"""

import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.experimental.pallas import tpu_sc as plsc

from tpu_inference.kernels.sparse_core import core_map_helper

_TOKEN_SUBCHUNK = 16
_NUM_TOKEN_SUBCHUNKS = 4
_TOKEN_BLOCK = _TOKEN_SUBCHUNK * _NUM_TOKEN_SUBCHUNKS
_ROUTE_INDEX_BITS = 20
_ROUTE_INDEX_MASK = (1 << _ROUTE_INDEX_BITS) - 1


def _align_to(value: int, alignment: int) -> int:
    return pl.cdiv(value, alignment) * alignment


def _fallback_implementation(
    x: jax.Array,
    indices: jax.Array,
    topk_weights: jax.Array,
    valid_rows_mask: jax.Array,
    reduce_group_size: int,
) -> jax.Array:
    out = x[indices] * topk_weights[:, None].astype(jnp.float32)
    out = jnp.where(valid_rows_mask[:, None], out, 0)
    out = out.reshape(-1, reduce_group_size, out.shape[-1])
    return jnp.sum(out, axis=1).astype(x.dtype)


def _calculate_num_column_partitions(
    hidden_size: int, num_cores: int, num_lanes: int
) -> int:
    preferred_num_stages = 4
    num_column_partitions = 1
    while (
        num_cores % (num_column_partitions * 2) == 0
        and hidden_size % (num_lanes * num_column_partitions * 2) == 0
        and hidden_size // (num_column_partitions * 2 * num_lanes)
        >= preferred_num_stages
    ):
        num_column_partitions *= 2
    # Destination-major accumulation keeps a dense FP32 token block in VMEM.
    # For columns up to 1024, using one fewer column partition amortizes the
    # per-block metadata, clear, and output-DMA overhead while still leaving
    # enough room for the double-buffered indirect gather.  Wider columns stay
    # with the conservative partitioning selected above.
    if num_column_partitions >= 2 and hidden_size <= 1024 * (
        num_column_partitions // 2
    ):
        num_column_partitions //= 2
    return num_column_partitions


def partition_counts(hidden_size: int, tpu_info) -> tuple[int, int]:
    """Column and row partition counts of the SparseCore subcores for a width."""
    sc_info = tpu_info.sparse_core
    num_cores = sc_info.num_cores * sc_info.num_subcores
    num_column_partitions = _calculate_num_column_partitions(
        hidden_size, num_cores, tpu_info.num_lanes
    )
    return num_column_partitions, num_cores // num_column_partitions


def token_block_alignment(hidden_size: int, tpu_info) -> int:
    """Destination-token granularity: one 64-token block per row partition."""
    return partition_counts(hidden_size, tpu_info)[1] * _TOKEN_BLOCK


def _main_kernel(
    x_hbm_ref: jax.Ref,
    route_metadata_hbm_ref: jax.Ref,
    route_weights_hbm_ref: jax.Ref,
    route_counts_hbm_ref: jax.Ref,
    out_hbm_ref: jax.Ref,
    accum_vmem_ref: jax.Ref,
    out_tile_vmem_ref: jax.Ref,
    sem_ref: jax.Ref,
    *,
    core_axis_name: str,
    subcore_axis_name: str,
    num_row_partitions: int,
    num_column_partitions: int,
    blocks_per_row_partition: int,
    topk: int,
):
    sc_info = pltpu.get_tpu_info().sparse_core
    assert sc_info is not None
    num_simd_lanes = sc_info.num_lanes
    assert num_simd_lanes == _TOKEN_SUBCHUNK

    core_id = jax.lax.axis_index((core_axis_name, subcore_axis_name))
    row_partition_id = core_id // num_column_partitions
    col_partition_id = core_id % num_column_partitions
    col_size = accum_vmem_ref.shape[-1]
    col_start = col_partition_id * col_size
    metadata_block_size = topk * _TOKEN_BLOCK
    x_32b_hbm_ref = x_hbm_ref.bitcast(jnp.int32)
    out_32b_hbm_ref = out_hbm_ref.bitcast(jnp.uint32)

    def metadata_block_index(block_id):
        return (row_partition_id * blocks_per_row_partition + block_id,)

    @functools.partial(
        pltpu.emit_pipeline,
        grid=(blocks_per_row_partition,),
        in_specs=(
            pl.BlockSpec((metadata_block_size,), metadata_block_index),
            pl.BlockSpec((metadata_block_size,), metadata_block_index),
            pl.BlockSpec((_TOKEN_SUBCHUNK,), metadata_block_index),
        ),
        out_specs=(),
    )
    def token_block_pipeline(
        route_metadata_ref, route_weights_ref, route_counts_ref, output_sem_ref
    ):
        block_id = pl.program_id(0)
        global_block_id = row_partition_id * blocks_per_row_partition + block_id
        valid_route_count = route_counts_ref[...][0]
        fixed_two_routes = valid_route_count == 2 * _TOKEN_BLOCK
        num_route_chunks = pl.cdiv(valid_route_count, _TOKEN_SUBCHUNK)

        # Scatter-add below reads the previous accumulator value, so initialize
        # the complete dense destination block first.  This also supplies zeros
        # for tokens with no local route.
        def zero_column(col_offset):
            col_slice = pl.ds(col_offset, _TOKEN_SUBCHUNK)
            zero = jnp.zeros((_TOKEN_SUBCHUNK,), jnp.float32)
            for token_row in range(_TOKEN_BLOCK):
                accum_vmem_ref[token_row, col_slice] = zero

        plsc.parallel_loop(0, col_size, step=_TOKEN_SUBCHUNK)(zero_column)

        def route_indices_slice(route_chunk):
            start = route_chunk * _TOKEN_SUBCHUNK
            metadata = route_metadata_ref[pl.ds(start, _TOKEN_SUBCHUNK)]
            return jnp.bitwise_and(metadata, _ROUTE_INDEX_MASK)

        @functools.partial(
            pltpu.emit_pipeline,
            grid=(num_route_chunks,),
            in_specs=pl.BlockSpec(
                (pl.Indirect(_TOKEN_SUBCHUNK), col_size),
                lambda route_chunk: (
                    jnp.bitwise_right_shift(
                        route_indices_slice(route_chunk),
                        1,
                    ),
                    col_partition_id,
                ),
            ),
            out_specs=(),
        )
        def route_pipeline(gather_ref):
            route_chunk = pl.program_id(0)
            metadata_start = route_chunk * _TOKEN_SUBCHUNK
            metadata_slice = pl.ds(metadata_start, _TOKEN_SUBCHUNK)
            route_metadata = route_metadata_ref[metadata_slice]
            source_indices = jnp.bitwise_and(route_metadata, _ROUTE_INDEX_MASK)
            route_destinations = jnp.bitwise_right_shift(
                route_metadata, _ROUTE_INDEX_BITS
            )
            route_weights = route_weights_ref[metadata_slice]

            def fixed_two_route_column_loop(col_offset):
                col_slice = pl.ds(col_offset, _TOKEN_SUBCHUNK)
                feature_indices = col_offset + jnp.arange(
                    _TOKEN_SUBCHUNK, dtype=jnp.int32
                )
                route_batch_size = 16
                for route_batch_start in range(0, _TOKEN_SUBCHUNK, route_batch_size):
                    weighted_values = []
                    for route_lane in range(
                        route_batch_start, route_batch_start + route_batch_size
                    ):
                        value_i32 = gather_ref[route_lane, col_slice]
                        shift = jnp.where(
                            jnp.bitwise_and(source_indices[route_lane], 1) == 0,
                            16,
                            0,
                        )
                        shifted = jnp.bitwise_and(
                            jnp.left_shift(value_i32, shift),
                            jnp.int32(-65536),
                        )
                        value_f32 = plsc.bitcast(shifted, jnp.float32)
                        value_f32 *= route_weights[route_lane]
                        weighted_values.append(value_f32)

                    for pair_start in range(0, route_batch_size, 2):
                        route_lane = route_batch_start + pair_start
                        destination_indices = jnp.full_like(
                            feature_indices, route_destinations[route_lane]
                        )
                        plsc.addupdate_scatter(
                            accum_vmem_ref,
                            (destination_indices, feature_indices),
                            weighted_values[pair_start]
                            + weighted_values[pair_start + 1],
                        )

            def generic_column_loop(col_offset):
                col_slice = pl.ds(col_offset, _TOKEN_SUBCHUNK)
                feature_indices = col_offset + jnp.arange(
                    _TOKEN_SUBCHUNK, dtype=jnp.int32
                )
                route_batch_size = 4
                for route_batch_start in range(0, _TOKEN_SUBCHUNK, route_batch_size):
                    weighted_values = []
                    for route_lane in range(
                        route_batch_start, route_batch_start + route_batch_size
                    ):
                        value_i32 = gather_ref[route_lane, col_slice]
                        shift = jnp.where(
                            jnp.bitwise_and(source_indices[route_lane], 1) == 0,
                            16,
                            0,
                        )
                        shifted = jnp.bitwise_and(
                            jnp.left_shift(value_i32, shift),
                            jnp.int32(-65536),
                        )
                        value_f32 = plsc.bitcast(shifted, jnp.float32)
                        value_f32 *= route_weights[route_lane]
                        weighted_values.append(value_f32)

                    for route_batch_lane in range(route_batch_size):
                        route_lane = route_batch_start + route_batch_lane
                        destination_indices = jnp.full_like(
                            feature_indices, route_destinations[route_lane]
                        )
                        route_valid = metadata_start + route_lane < valid_route_count
                        scatter_mask = jnp.broadcast_to(route_valid, (_TOKEN_SUBCHUNK,))
                        plsc.addupdate_scatter(
                            accum_vmem_ref,
                            (destination_indices, feature_indices),
                            weighted_values[route_batch_lane],
                            mask=scatter_mask,
                        )

            @pl.when(fixed_two_routes)
            def run_fixed_two_route_path():
                plsc.parallel_loop(0, col_size, step=_TOKEN_SUBCHUNK)(
                    fixed_two_route_column_loop
                )

            @pl.when(jnp.logical_not(fixed_two_routes))
            def run_generic_path():
                plsc.parallel_loop(0, col_size, step=_TOKEN_SUBCHUNK)(
                    generic_column_loop
                )

        route_pipeline(x_32b_hbm_ref)

        # Convert pairs of FP32 destination rows to row-packed BF16 and write
        # four complete, tile-aligned 16-row output tiles.
        for token_subchunk_pair in range(_NUM_TOKEN_SUBCHUNKS // 2):
            copies = []
            for output_buffer in range(2):
                token_subchunk = token_subchunk_pair * 2 + output_buffer
                accum_row_start = token_subchunk * _TOKEN_SUBCHUNK

                def cast_column(col_offset):
                    col_slice = pl.ds(col_offset, _TOKEN_SUBCHUNK)
                    for token_lane in range(0, _TOKEN_SUBCHUNK, 2):
                        accum_row = accum_row_start + token_lane
                        packed_bf16 = plsc.pack(
                            accum_vmem_ref[accum_row, col_slice],
                            accum_vmem_ref[accum_row + 1, col_slice],
                            format=plsc.PackFormat.INTERLEAVED,
                        )
                        out_tile_vmem_ref[output_buffer, token_lane // 2, col_slice] = (
                            plsc.bitcast(packed_bf16, jnp.uint32)
                        )

                plsc.parallel_loop(0, col_size, step=_TOKEN_SUBCHUNK)(cast_column)

                output_row = (
                    global_block_id * _TOKEN_BLOCK + token_subchunk * _TOKEN_SUBCHUNK
                )
                output_row_packed = pl.multiple_of(
                    output_row // 2, _TOKEN_SUBCHUNK // 2
                )
                copy = pltpu.make_async_copy(
                    out_tile_vmem_ref.at[output_buffer],
                    out_32b_hbm_ref.at[
                        pl.ds(output_row_packed, _TOKEN_SUBCHUNK // 2),
                        pl.ds(col_start, col_size),
                    ],
                    output_sem_ref.at[output_buffer],
                )
                copy.start()
                copies.append(copy)
            for copy in copies:
                copy.wait()

    token_block_pipeline(
        route_metadata_hbm_ref,
        route_weights_hbm_ref,
        route_counts_hbm_ref,
        scratches=(sem_ref,),
    )


@functools.partial(jax.jit, static_argnames=("reduce_group_size",))
def ragged_gather_reduce(
    x: jax.Array,
    indices: jax.Array,
    topk_weights: jax.Array,
    valid_rows_mask: jax.Array,
    reduce_group_size: int,
) -> jax.Array:
    """Destination-major prototype for MoE output combine."""
    sc_info = pltpu.get_tpu_info().sparse_core
    if (
        sc_info is None
        or x.dtype != jnp.bfloat16
        or x.shape[-1] % pltpu.get_tpu_info().num_lanes != 0
    ):
        return _fallback_implementation(
            x,
            indices,
            topk_weights,
            valid_rows_mask,
            reduce_group_size,
        )
    dtype_bytes = jax.dtypes.itemsize_bits(x.dtype) // 8
    if jnp.size(x) * dtype_bytes * 2 < pltpu.get_tpu_info().vmem_capacity_bytes * 0.6:
        return _fallback_implementation(
            x,
            indices,
            topk_weights,
            valid_rows_mask,
            reduce_group_size,
        )

    input_size = indices.size
    if input_size % reduce_group_size != 0:
        raise ValueError(f"{input_size=} must be divisible by {reduce_group_size=}")
    # SparseCore row-packing views BF16 as pairs of rows.
    if x.shape[0] % 2:
        x = jnp.pad(x, ((0, 1), (0, 0)))
    num_tokens = input_size // reduce_group_size
    hidden_size = x.shape[-1]
    if x.shape[0] > 1 << _ROUTE_INDEX_BITS:
        raise ValueError(
            f"destination-major prototype supports at most "
            f"{1 << _ROUTE_INDEX_BITS} source rows, got {x.shape[0]}"
        )
    num_column_partitions, num_row_partitions = partition_counts(
        hidden_size, pltpu.get_tpu_info()
    )

    token_alignment = num_row_partitions * _TOKEN_BLOCK
    padded_num_tokens = _align_to(num_tokens, token_alignment)
    token_padding = padded_num_tokens - num_tokens

    indices_2d = indices.reshape(num_tokens, reduce_group_size)
    weights_2d = topk_weights.reshape(num_tokens, reduce_group_size)
    valid_2d = valid_rows_mask.reshape(num_tokens, reduce_group_size)
    if token_padding:
        indices_2d = jnp.pad(indices_2d, ((0, token_padding), (0, 0)))
        weights_2d = jnp.pad(weights_2d, ((0, token_padding), (0, 0)))
        valid_2d = jnp.pad(valid_2d, ((0, token_padding), (0, 0)))

    num_blocks = padded_num_tokens // _TOKEN_BLOCK
    metadata_block_size = _TOKEN_BLOCK * reduce_group_size
    indices_blocks = indices_2d.reshape(num_blocks, metadata_block_size)
    valid_blocks = valid_2d.reshape(num_blocks, metadata_block_size)
    destination_template = jnp.repeat(
        jnp.arange(_TOKEN_BLOCK, dtype=jnp.int32), reduce_group_size
    )
    destination_blocks = jnp.broadcast_to(destination_template, indices_blocks.shape)

    # A stable boolean partition keeps routes for each destination adjacent
    # while placing the exact valid prefix first in every destination block.
    route_order = jnp.argsort(~valid_blocks, axis=1, stable=True)
    compact_valid = jnp.take_along_axis(valid_blocks, route_order, axis=1)
    route_indices = jnp.take_along_axis(indices_blocks, route_order, axis=1)
    route_destinations = jnp.take_along_axis(destination_blocks, route_order, axis=1)
    weights_blocks = weights_2d.reshape(num_blocks, metadata_block_size)
    route_weights = jnp.take_along_axis(weights_blocks, route_order, axis=1)
    route_indices = jnp.where(compact_valid, route_indices, 0).astype(jnp.int32)
    route_destinations = jnp.where(compact_valid, route_destinations, 0).astype(
        jnp.int32
    )
    route_metadata = jnp.bitwise_or(
        route_indices,
        jnp.left_shift(route_destinations, _ROUTE_INDEX_BITS),
    ).reshape(-1)
    route_weights = jnp.where(compact_valid, route_weights, 0).reshape(-1)
    route_weights = route_weights.astype(jnp.float32)
    route_counts = jnp.sum(valid_blocks, axis=1, dtype=jnp.int32)
    paired_destinations = route_destinations[:, : 2 * _TOKEN_BLOCK].reshape(
        num_blocks, _TOKEN_BLOCK, 2
    )
    fixed_two_route_blocks = jnp.logical_and(
        route_counts == 2 * _TOKEN_BLOCK,
        jnp.all(paired_destinations[:, :, 0] == paired_destinations[:, :, 1], axis=1),
    )
    # Count 128 selects the adjacent-pair fast path in the SC kernel. If a full
    # two-route block does not actually have adjacent equal destinations,
    # expose one already-zero compacted slot as a harmless sentinel so the block
    # takes the generic path instead.
    needs_zero_weight_sentinel = jnp.logical_and(
        route_counts == 2 * _TOKEN_BLOCK,
        jnp.logical_not(fixed_two_route_blocks),
    )
    route_counts += needs_zero_weight_sentinel.astype(jnp.int32)
    route_counts = jnp.broadcast_to(
        route_counts[:, None], (num_blocks, _TOKEN_SUBCHUNK)
    ).reshape(-1)

    col_size = hidden_size // num_column_partitions
    blocks_per_row_partition = num_blocks // num_row_partitions
    vector_mesh = plsc.VectorSubcoreMesh(
        num_cores=sc_info.num_cores,
        num_subcores=sc_info.num_subcores,
        core_axis_name="core",
        subcore_axis_name="subcore",
    )
    out = core_map_helper.kernel(
        functools.partial(
            _main_kernel,
            core_axis_name=vector_mesh.core_axis_name,
            subcore_axis_name=vector_mesh.subcore_axis_name,
            num_row_partitions=num_row_partitions,
            num_column_partitions=num_column_partitions,
            blocks_per_row_partition=blocks_per_row_partition,
            topk=reduce_group_size,
        ),
        out_type=jax.ShapeDtypeStruct((padded_num_tokens, hidden_size), jnp.bfloat16),
        compiler_params=pltpu.CompilerParams(
            use_tc_tiling_on_sc=True,
            disable_bounds_checks=True,
            needs_layout_passes=False,
        ),
        scratch_types=(
            pltpu.VMEM((_TOKEN_BLOCK, col_size), jnp.float32),
            pltpu.VMEM((2, _TOKEN_SUBCHUNK // 2, col_size), jnp.uint32),
            pltpu.SemaphoreType.DMA((2,)),
        ),
        mesh=vector_mesh,
        name="sc_ragged_gather_reduce_v3",
    )(x, route_metadata, route_weights, route_counts)
    return out[:num_tokens]
