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
"""Pallas kernel for writing decode tokens into a paged KV cache.

Phase A of the DCP decode forward pass: each DCP device writes the new decode
token for every sequence it owns into the correct page slot of the local KV cache shard.
"""

import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

GROUP_SIZE = 16

def _write_decode_kv_kernel_optimized(
    # Scalar prefetch
    kv_lens_ref,          # i32[max_num_seqs]
    page_indices_ref,     # i32[max_num_seqs * pages_per_seq]
    cu_q_lens_ref,        # i32[max_num_seqs + 1]
    distribution_ref,     # i32[3]
    cp_rank_ref,          # i32[1]
    # HBM
    merged_kv_hbm_ref,    # bf16[max_num_tokens, nh_x2//kp, kp, head_dim]
    kv_cache_hbm_ref,     # bf16[local_pages, local_page_size, nh_x2//kp, kp, hd]
    updated_kv_cache_hbm_ref,  # same shape, aliased with kv_cache (in-place)
    # Scratch
    kv_slot_ref,          # VMEM bf16[GROUP_SIZE, 1, nh_x2//kp, kp, head_dim]
    sem,                  # DMA semaphore [GROUP_SIZE] 
    *,
    # ── Static params ───────────────────────────────────────────────────────
    cp_group_size: int,
    pages_per_seq: int,
    cp_kv_cache_interleaved_size: int,
):
    batch_idx = pl.program_id(0)
    base_seq_idx = batch_idx * GROUP_SIZE
    
    local_page_size = kv_cache_hbm_ref.shape[1]
    cp_rank = cp_rank_ref[0]

    cs = updated_kv_cache_hbm_ref.shape
    cache_flat = updated_kv_cache_hbm_ref.reshape(cs[0] * cs[1], *cs[2:])

    # PHASE 1: Start a group of HBM -> VMEM read (Non-blocking)
    for i in range(GROUP_SIZE):
        seq_idx = base_seq_idx + i

        safe_seq_idx = jnp.minimum(seq_idx, distribution_ref[0] - 1)
        write_pos = kv_lens_ref[safe_seq_idx] - 1
        
        # Check if this token is belong to current cp_rank
        write_pos_rank = (write_pos // cp_kv_cache_interleaved_size) % cp_group_size
        cond = (seq_idx < distribution_ref[0]) & (write_pos_rank == cp_rank)
        
        @pl.when(cond)
        def _start_read():
            token_i = cu_q_lens_ref[safe_seq_idx]
            pltpu.make_async_copy(
                merged_kv_hbm_ref.at[pl.ds(token_i, 1)],
                kv_slot_ref.at[pl.ds(i, 1)],
                sem.at[i],
            ).start()

    # PHASE 2: Wait every HBM -> VMEM read to complete
    for i in range(GROUP_SIZE):
        seq_idx = base_seq_idx + i
        safe_seq_idx = jnp.minimum(seq_idx, distribution_ref[0] - 1)
        write_pos = kv_lens_ref[safe_seq_idx] - 1
        
        write_pos_rank = (write_pos // cp_kv_cache_interleaved_size) % cp_group_size
        cond = (seq_idx < distribution_ref[0]) & (write_pos_rank == cp_rank)
        
        @pl.when(cond)
        def _wait_read():
            token_i = cu_q_lens_ref[safe_seq_idx]
            pltpu.make_async_copy(
                merged_kv_hbm_ref.at[pl.ds(token_i, 1)],
                kv_slot_ref.at[pl.ds(i, 1)],
                sem.at[i],
            ).wait()

    # PHASE 3: Start a group of VMEM -> HBM write (Non-blocking)
    for i in range(GROUP_SIZE):
        seq_idx = base_seq_idx + i
        safe_seq_idx = jnp.minimum(seq_idx, distribution_ref[0] - 1)
        write_pos = kv_lens_ref[safe_seq_idx] - 1
        
        write_pos_rank = (write_pos // cp_kv_cache_interleaved_size) % cp_group_size
        cond = (seq_idx < distribution_ref[0]) & (write_pos_rank == cp_rank)
        
        @pl.when(cond)
        def _start_write():
            local_pos = ((write_pos // cp_kv_cache_interleaved_size) // cp_group_size) * cp_kv_cache_interleaved_size + (write_pos % cp_kv_cache_interleaved_size)
            kv_p = local_pos // local_page_size
            kv_off = local_pos % local_page_size
            phys_page = page_indices_ref[safe_seq_idx * pages_per_seq + kv_p]
            flat_idx = phys_page * local_page_size + kv_off

            pltpu.make_async_copy(
                kv_slot_ref.at[pl.ds(i, 1)],  
                cache_flat.at[pl.ds(flat_idx, 1)],
                sem.at[i],
            ).start()

    # PHASE 4: Wait every every token to write back safely.
    for i in range(GROUP_SIZE):
        seq_idx = base_seq_idx + i
        safe_seq_idx = jnp.minimum(seq_idx, distribution_ref[0] - 1)
        write_pos = kv_lens_ref[safe_seq_idx] - 1
        
        write_pos_rank = (write_pos // cp_kv_cache_interleaved_size) % cp_group_size
        cond = (seq_idx < distribution_ref[0]) & (write_pos_rank == cp_rank)
        
        @pl.when(cond)
        def _wait_write():
            local_pos = ((write_pos // cp_kv_cache_interleaved_size) // cp_group_size) * cp_kv_cache_interleaved_size + (write_pos % cp_kv_cache_interleaved_size)
            kv_p = local_pos // local_page_size
            kv_off = local_pos % local_page_size
            phys_page = page_indices_ref[safe_seq_idx * pages_per_seq + kv_p]
            flat_idx = phys_page * local_page_size + kv_off
            
            pltpu.make_async_copy(
                kv_slot_ref.at[pl.ds(i, 1)],  
                cache_flat.at[pl.ds(flat_idx, 1)],
                sem.at[i],
            ).wait()


@jax.jit(
    static_argnames=("cp_group_size", "cp_kv_cache_interleaved_size"), donate_argnames="kv_cache",
)
def write_decode_kv(
    merged_kv: jax.Array,      # [max_num_tokens, nh_x2//kp, kp, head_dim]
    kv_cache: jax.Array,       # [local_pages, local_page_size, nh_x2//kp, kp, hd]
    kv_lens: jax.Array,        # i32[max_num_seqs]
    page_indices: jax.Array,   # i32[max_num_seqs * pages_per_seq]
    cu_q_lens: jax.Array,      # i32[max_num_seqs + 1]
    distribution: jax.Array,   # i32[3]
    cp_rank: jax.Array,        # i32[1]
    *,
    cp_group_size: int,
    cp_kv_cache_interleaved_size: int = 1,
) -> jax.Array:
    """Write one decode token per sequence into the local KV cache shard.

    Designed to run inside ``jax.shard_map``; all inputs are already local
    (per-device) shards.  Uses Pallas DMA for truly in-place writes — no
    HBM-wide data-formatting op is emitted.

    Args:
        merged_kv:    Pre-merged K+V tensor (output of ``merge_kv``).
        kv_cache:     Local KV cache shard (CONTEXT-sharded along page_size).
        kv_lens:      Per-sequence total token lengths *including* the new token.
        page_indices: Physical page mapping for each sequence.
        cu_q_lens:    Cumulative query start positions (maps seq → token index).
        distribution: [decode_end, prefill_end, total].
        cp_rank:      This device's DCP rank, shape i32[1].
        cp_group_size: DCP group size (static).
        cp_kv_cache_interleaved_size: Interleaved block size for block-wise CP sharded caches.

    Returns:
        Updated kv_cache with the new decode tokens written in-place.
    """
    max_num_seqs  = kv_lens.shape[0]
    pages_per_seq = page_indices.shape[0] // max_num_seqs

    _, _, num_kv_heads_x2_per_kp, kv_packing, head_dim = kv_cache.shape

    kv_slot_scratch = pltpu.VMEM(
        (GROUP_SIZE, num_kv_heads_x2_per_kp, kv_packing, head_dim),
        kv_cache.dtype,
    )
    sem_scratch = pltpu.SemaphoreType.DMA((GROUP_SIZE,))

    scalar_prefetches = (kv_lens, page_indices, cu_q_lens, distribution, cp_rank)
    num_scalar_prefetch = len(scalar_prefetches)

    out_shape = jax.ShapeDtypeStruct(shape=kv_cache.shape, dtype=kv_cache.dtype)

    kernel = pl.pallas_call(
        functools.partial(
            _write_decode_kv_kernel_optimized,
            cp_group_size=cp_group_size,
            pages_per_seq=pages_per_seq,
            cp_kv_cache_interleaved_size=cp_kv_cache_interleaved_size,
        ),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=num_scalar_prefetch,
            in_specs=[
                pl.BlockSpec(memory_space=pltpu.HBM),  # merged_kv
                pl.BlockSpec(memory_space=pltpu.HBM),  # kv_cache
            ],
            out_specs=pl.BlockSpec(memory_space=pltpu.HBM),  # updated_kv_cache
            grid=(max_num_seqs // GROUP_SIZE,),
            scratch_shapes=[kv_slot_scratch, sem_scratch],
        ),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("arbitrary",),
            disable_bounds_checks=True,
        ),
        out_shape=out_shape,
        # kv_cache (input index num_scalar_prefetch + 1) aliased to output 0.
        input_output_aliases={num_scalar_prefetch + 1: 0},
        name="write_decode_kv",
    )

    return kernel(*scalar_prefetches, merged_kv, kv_cache)
