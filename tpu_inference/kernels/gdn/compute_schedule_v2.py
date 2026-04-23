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

import jax
import jax.numpy as jnp


def compute_schedule_table_v2(
    query_start_loc: jax.Array,
    decode_tokens: int | jax.Array,
    num_tokens: int,
    chunk_size: int,
    BT: int | None = None,
    alignment: int = 8,
) -> tuple[jax.Array, jax.Array]:
    """Compute number of iterations in grid and work each iteration will do

  At high level
    - each iteration of grid is either prefill and or decode
    - grid moves in size of bt decode tokens (sequence) backwards starting from
    boundary
    - and prefill moves in chunk sized tokens forward from boundary to end
  Input characteristics
    - each sequence start and end may not be sublane aligned,
    boundary between decode and prefill maybe in shared sublane
    - sequence may not divide chunk size

  hardware req
    - offset for each block has to be sublane aligned

  So for this we have transition blocks at boundaries between prefill sequences,
  including first one with decode, token by token math is done here instead of
  chunk wise
  """
    if BT is None:
        BT = chunk_size

    num_decode_batches = (decode_tokens + BT - 1) // BT
    num_seqs = query_start_loc.shape[0] - 1

    max_blocks = (num_tokens + chunk_size - 1) // chunk_size
    safe_max_blocks = int(max_blocks + num_seqs * 2)

    # =========================================================================
    # 1. Get each prefill sequence's effective start for chunkwise math
    # =========================================================================
    r_idx = jnp.arange(num_seqs)
    is_last_seq = r_idx == num_seqs - 1
    seq_start = query_start_loc[:-1]
    seq_end = query_start_loc[1:]

    # create vector of sequence ends
    prev_seq_end = jnp.pad(seq_end[:-1], (1, 0), constant_values=0)
    effective_start = jnp.where(
        prev_seq_end % alignment != 0,
        (prev_seq_end // alignment) * alignment + alignment,
        prev_seq_end,
    )

    # if seq_len < sublane size
    is_swallowed = effective_start >= seq_end

    # compute the effective end of the rounded up to nearest sublane
    next_aligned_start = (seq_end // alignment) * alignment
    needs_transition = ((seq_end % alignment != 0) & (~is_last_seq) &
                        (~is_swallowed))

    effective_end = jnp.where(needs_transition, next_aligned_start, seq_end)
    effective_end = jnp.maximum(effective_start, effective_end)

    # Block counts per sequence
    num_regular_blocks = (effective_end - effective_start + chunk_size -
                          1) // chunk_size
    total_blocks_per_seq = num_regular_blocks + needs_transition.astype(
        jnp.int32)
    total_blocks_per_seq = jnp.where(is_swallowed, 0, total_blocks_per_seq)

    # Calculate the last perfectly aligned decode boundary
    aligned_decode_boundary = (decode_tokens // alignment) * alignment

    is_pure_decode = seq_end <= aligned_decode_boundary
    total_blocks_per_seq = jnp.where(is_pure_decode, 0, total_blocks_per_seq)

    # Starting block index for each sequence
    base_idx = jnp.cumsum(total_blocks_per_seq) - total_blocks_per_seq
    total_prefill_blocks = jnp.sum(total_blocks_per_seq)

    # =========================================================================
    # 2. shows up as gathers
    # create block table
    # =========================================================================
    b_idx = jnp.arange(safe_max_blocks)
    prefill_valid_mask = b_idx < total_prefill_blocks

    # map grid index to sequence/request,
    # key for previous metadata arrays constructed to gather by sequence
    r_for_block = jnp.sum(b_idx[:, None] >= base_idx[None, :], axis=-1) - 1
    r_for_block = jnp.minimum(jnp.maximum(r_for_block, 0), num_seqs - 1)

    # index of block within blocks for a sequence
    local_b = b_idx - base_idx[r_for_block]

    is_trans_block = needs_transition[r_for_block] & (
        local_b == num_regular_blocks[r_for_block])

    # Gather regular properties, effective start maps sequence to start index,
    # r_for_block returns sequences sorted by grid index
    reg_offset = effective_start[r_for_block] + local_b * chunk_size
    reg_count = jnp.minimum(chunk_size,
                            effective_end[r_for_block] - reg_offset)
    reg_is_last = reg_offset + reg_count >= seq_end[r_for_block]
    reg_is_first = reg_offset == seq_start[r_for_block]

    # Gather transition properties
    trans_offset = next_aligned_start[r_for_block]

    # Apply predication
    block_offset = jnp.where(is_trans_block, trans_offset, reg_offset)
    block_count = jnp.where(is_trans_block, alignment, reg_count)
    block_is_last = jnp.where(is_trans_block, False, reg_is_last)
    block_is_first = jnp.where(is_trans_block, False, reg_is_first)

    # =========================================================================
    # 3. Metadata for shared sublane tiles
    # =========================================================================
    glob_idxs = block_offset[:, None] + jnp.arange(alignment)[None, :]

    # [safe_max_blocks, sublane size, num_seqs]
    t_reqs = (jnp.sum(glob_idxs[:, :, None] >= query_start_loc[None, None, :],
                      axis=-1) - 1)
    t_reqs = jnp.minimum(jnp.maximum(t_reqs, 0), num_seqs - 1)

    is_first_tok = (glob_idxs == query_start_loc[t_reqs]).astype(jnp.int32)
    is_last_tok = (glob_idxs == query_start_loc[t_reqs + 1] - 1).astype(
        jnp.int32)

    # =========================================================================
    # 4. Decode blocks metadata
    # =========================================================================
    decode_valid_mask = b_idx < num_decode_batches
    decode_batch_idx = jnp.where(decode_valid_mask,
                                 (num_decode_batches - 1) - b_idx, 0)
    decode_offsets = decode_batch_idx * BT
    decode_req_ids = decode_batch_idx * BT
    decode_counts = jnp.where(decode_valid_mask,
                              jnp.minimum(BT, decode_tokens - decode_offsets),
                              0)

    # Mask out invalid prefill
    prefill_valid_ints = prefill_valid_mask.astype(jnp.int32)
    block_offset = jnp.where(prefill_valid_mask, block_offset, 0)
    r_for_block = jnp.where(prefill_valid_mask, r_for_block, 0)
    block_count = jnp.where(prefill_valid_mask, block_count, 0)
    block_is_last = jnp.where(prefill_valid_mask, block_is_last, False)
    block_is_first = jnp.where(prefill_valid_mask, block_is_first, False)
    is_trans_block = jnp.where(prefill_valid_mask, is_trans_block, False)
    t_reqs = jnp.where(prefill_valid_mask[:, None], t_reqs, 0)
    is_first_tok = jnp.where(prefill_valid_mask[:, None], is_first_tok, 0)
    is_last_tok = jnp.where(prefill_valid_mask[:, None], is_last_tok, 0)

    # =========================================================================
    # 5. Merge all
    # =========================================================================
    cols = [
        prefill_valid_ints,  # 0
        block_offset,  # 1
        r_for_block,  # 2
        block_count,  # 3
        decode_valid_mask.astype(jnp.int32),  # 4
        decode_offsets,  # 5
        decode_req_ids,  # 6
        decode_counts,  # 7
        block_is_last.astype(jnp.int32),  # 8
        block_is_first.astype(jnp.int32),  # 9
        jnp.zeros(safe_max_blocks, dtype=jnp.int32),  # 10
        is_trans_block.astype(jnp.int32),  # 11
    ]

    for i in range(alignment):
        cols.append(t_reqs[:, i])  # 12-19
    for i in range(alignment):
        cols.append(is_first_tok[:, i])  # 20-27
    for i in range(alignment):
        cols.append(is_last_tok[:, i])  # 28-35

    final_table = jnp.stack(cols, axis=1)
    total_blocks = jnp.maximum(total_prefill_blocks, num_decode_batches)

    return final_table, total_blocks
