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
"""Shared utilities for short Conv1D kernels."""

from __future__ import annotations

import dataclasses
import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class GDNChunkIndices:
    num_blocks: jax.Array
    block_id_to_seq_idx: jax.Array
    block_id_to_t_offset: jax.Array


@jax.named_scope("calculate_chunk_indices")
def calculate_chunk_indices(
    cu_seqlens: jax.Array,
    distribution: jax.Array,
    max_num_blocks: int,
    bt: int,
    region_start_idx: int = 0,
) -> GDNChunkIndices:
    """Pre-computes per-block sequence and token offsets."""

    def _kernel(
        cu_seqlens_ref,
        distribution_ref,
        meta_out,
        *,
        bt: int,
        region_start_idx: int,
    ):
        seq_start = distribution_ref[region_start_idx]
        seq_end = distribution_ref[region_start_idx + 1]
        n_seqs = seq_end - seq_start

        @jax.named_scope("inner_block_loop")
        def inner_block_loop(blk_rel, carry, *, seq_idx, eos):
            num_blocks, t_cursor = carry
            block_id = num_blocks + blk_rel

            t_start = t_cursor + blk_rel * bt
            t_end = jnp.minimum(t_start + bt, eos)

            meta_out.block_id_to_seq_idx[block_id] = seq_idx
            meta_out.block_id_to_t_offset[block_id] = t_start
            meta_out.block_id_to_t_offset[block_id + 1] = t_end

            return num_blocks, t_cursor

        @jax.named_scope("outer_seq_loop")
        def outer_seq_loop(seq_rel, carry):
            num_blocks, t_cursor = carry
            seq_idx = seq_start + seq_rel
            eos = cu_seqlens_ref[seq_idx + 1]

            seq_len_from_cursor = eos - t_cursor
            n_seq_blocks = pl.cdiv(seq_len_from_cursor, bt)

            loop_fn = functools.partial(
                inner_block_loop,
                seq_idx=seq_idx,
                eos=eos,
            )
            jax.lax.fori_loop(0, n_seq_blocks, loop_fn, (num_blocks, t_cursor))

            return num_blocks + n_seq_blocks, eos

        first_token = cu_seqlens_ref[seq_start]
        num_blocks, _ = jax.lax.fori_loop(
            0,
            n_seqs,
            outer_seq_loop,
            (jnp.int32(0), first_token),
        )
        meta_out.block_id_to_seq_idx[num_blocks] = jnp.int32(-1)
        meta_out.num_blocks[0] = num_blocks

    smem_spec = pl.BlockSpec(memory_space=pltpu.SMEM)
    return pl.pallas_call(
        functools.partial(_kernel, bt=bt, region_start_idx=region_start_idx),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=2,
            out_specs=GDNChunkIndices(
                num_blocks=smem_spec,
                block_id_to_seq_idx=smem_spec,
                block_id_to_t_offset=smem_spec,
            ),
            grid=(1, ),
        ),
        out_shape=GDNChunkIndices(
            num_blocks=jax.ShapeDtypeStruct((1, ), jnp.int32),
            block_id_to_seq_idx=jax.ShapeDtypeStruct((max_num_blocks + 1, ),
                                                     jnp.int32),
            block_id_to_t_offset=jax.ShapeDtypeStruct((max_num_blocks + 1, ),
                                                      jnp.int32),
        ),
        compiler_params=pltpu.CompilerParams(disable_bounds_checks=True),
    )(cu_seqlens, distribution)
