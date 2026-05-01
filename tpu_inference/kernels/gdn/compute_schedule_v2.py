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
    num_valid_seqs: int | jax.Array,
    max_tokens: int,
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

  TODO: optimize table ,
    remove metadata which can be derived from other metadata or loop indices,
    like
        block offset can be derived from block idx and sequence start,
        block count can be derived from block idx and sequence start/end.
        also some metadata is only used for prefill or decode and can be stored
        in separate tables or encoded in same table with fewer bits.
        dtype of some metadata can be reduced to save space, for example
        block_is_first and block_is_last can be stored in 2 bits together,
        Sublane token by token metadata can be optimized by only storing
        boundaries
  """
    if BT is None:
        BT = chunk_size

    num_decode_batches = (decode_tokens + BT - 1) // BT
    num_seqs = query_start_loc.shape[0] - 1

    max_blocks = (max_tokens + chunk_size - 1) // chunk_size
    safe_max_blocks = int(max_blocks + num_seqs * 2)

    # =========================================================================
    # 1. Get each prefill sequence's effective start for chunkwise math
    # =========================================================================
    r_idx = jnp.arange(num_seqs)
    is_last_seq = r_idx == num_seqs - 1
    seq_start = query_start_loc[:-1]
    seq_end = query_start_loc[1:]
    num_tokens = query_start_loc[num_valid_seqs]

    # create vector of sequence ends
    prev_seq_end = jnp.pad(seq_end[:-1], (1, 0), constant_values=0)
    effective_start = jnp.where(
        prev_seq_end % alignment != 0,
        (prev_seq_end // alignment) * alignment + alignment,
        prev_seq_end,
    )

    # if seq_len < sublane size
    is_decode_boundary = prev_seq_end == decode_tokens
    is_swallowed = (effective_start >= seq_end) & (~is_decode_boundary)

    # compute the effective end of the rounded up to nearest sublane
    next_aligned_start = (seq_end // alignment) * alignment
    needs_transition = ((seq_end % alignment != 0) & (~is_last_seq) &
                        (~is_swallowed))

    is_decode_boundary = prev_seq_end == decode_tokens

    needs_start_transition = ((prev_seq_end % alignment != 0) & (~is_swallowed)
                              & is_decode_boundary)

    effective_end = jnp.where(needs_transition, next_aligned_start, seq_end)
    effective_end = jnp.maximum(effective_start, effective_end)

    # Block counts per sequence
    num_regular_blocks = (effective_end - effective_start + chunk_size -
                          1) // chunk_size
    total_blocks_per_seq = (num_regular_blocks +
                            needs_transition.astype(jnp.int32) +
                            needs_start_transition.astype(jnp.int32))
    total_blocks_per_seq = jnp.where(is_swallowed, 0, total_blocks_per_seq)

    # Calculate the last perfectly aligned decode boundary
    is_pure_decode = seq_end <= decode_tokens
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

    start_trans_offset = (seq_start[r_for_block] // alignment) * alignment

    is_start_trans = needs_start_transition[r_for_block] & (local_b == 0)

    # Adjust local_b for regular blocks if there was a start transition
    adj_local_b = jnp.where(needs_start_transition[r_for_block], local_b - 1,
                            local_b)

    is_end_trans = needs_transition[r_for_block] & (
        adj_local_b == num_regular_blocks[r_for_block])

    reg_offset = effective_start[r_for_block] + adj_local_b * chunk_size
    reg_count = jnp.minimum(chunk_size,
                            effective_end[r_for_block] - reg_offset)
    #   reg_is_last = reg_offset + reg_count >= seq_end[r_for_block]
    #   reg_is_first = reg_offset == seq_start[r_for_block]

    trans_offset = next_aligned_start[r_for_block]

    # Apply predication
    block_offset = jnp.where(
        is_start_trans,
        start_trans_offset,
        jnp.where(is_end_trans, trans_offset, reg_offset),
    )

    block_count = jnp.where(
        is_start_trans,
        effective_start[r_for_block] - seq_start[r_for_block],
        jnp.where(is_end_trans, alignment, reg_count),
    )

    is_trans_block = is_start_trans | is_end_trans

    # =========================================================================
    # 3. Metadata for shared sublane tiles
    # =========================================================================
    last_valid_loc = query_start_loc[num_valid_seqs]
    valid_loc_mask = jnp.arange(query_start_loc.shape[0]) <= num_valid_seqs
    fixed_query_start_loc = jnp.where(valid_loc_mask, query_start_loc,
                                      last_valid_loc)
    glob_idxs = block_offset[:, None] + jnp.arange(alignment)[None, :]

    # [safe_max_blocks, sublane size, num_seqs]
    valid_mask = glob_idxs < num_tokens
    t_reqs = (
        jnp.sum(glob_idxs[:, :, None] >= fixed_query_start_loc[None, None, :],
                axis=-1) - 1)
    # there could be padding in query_start_loc
    last_valid_seq = jnp.max(
        jnp.where(total_blocks_per_seq > 0, jnp.arange(num_seqs), -1))
    t_reqs = jnp.where(valid_mask, t_reqs, last_valid_seq)
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
    block_is_first = block_offset <= seq_start[r_for_block]
    block_is_last = (block_offset + block_count) >= seq_end[r_for_block]
    block_is_first = jnp.where(prefill_valid_mask, block_is_first, False)
    block_is_last = jnp.where(prefill_valid_mask, block_is_last, False)
    is_trans_block = jnp.where(prefill_valid_mask, is_trans_block, False)
    t_reqs = jnp.where(prefill_valid_mask[:, None], t_reqs, 0)
    is_first_tok = jnp.where(prefill_valid_mask[:, None], is_first_tok, 0)
    is_last_tok = jnp.where(prefill_valid_mask[:, None], is_last_tok, 0)

    # =========================================================================
    # 5. Merge all
    # =========================================================================
    # Columns mapping:
    # 0: prefill_valid_ints - 1 if this grid block has valid prefill work,
    # .                  0 otherwise
    # 1: block_offset - start index of prefill start in tile, usually 0
    #                     but in shared sublane case its not
    # 2: r_for_block - request ID (sequence index) this prefill block belongs to
    # 3: block_count - number of valid tokens in this prefill block
    # 4: decode_valid_mask - 1 if this step has valid decode work, 0 otherwise
    # 5: decode_offsets - start index for the decode batch
    # 6: decode_req_ids - starting request ID in decode batch
    # 7: decode_counts - number of valid decode requests in this batch
    # 8: block_is_last - 1 if this is the last block for the request, 0 otherwise
    # 9: block_is_first - 1 if first block for request, 0 otherwise
    # 10: is_trans_block - 1 if this is a transition block, 0 otherwise
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
        is_trans_block.astype(jnp.int32),  # 10
    ]

    # 11 to 11 + alignment - 1: Request ID for each token in the sublane tile
    for i in range(alignment):
        cols.append(t_reqs[:, i])  # e.g., 11-18 if alignment=8
    # 11 + alignment to 11 + 2*alignment - 1: 1 if token is first in request
    for i in range(alignment):
        cols.append(is_first_tok[:, i])  # e.g., 19-26
    # 11 + 2*alignment to 11 + 3*alignment - 1: 1 if token is last in request
    for i in range(alignment):
        cols.append(is_last_tok[:, i])  # e.g., 27-34

    final_table = jnp.stack(cols, axis=1)
    total_blocks = jnp.maximum(total_prefill_blocks, num_decode_batches)

    return final_table, total_blocks
