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
import numpy as np
from jax.sharding import PartitionSpec

from tpu_inference.layers.common.sharding import ShardingAxisName
from tpu_inference.layers.jax.sample.sampling import _apply_sampling_transforms
from tpu_inference.layers.jax.sample.sampling_metadata import \
    TPUSupportedSamplingMetadata
from tpu_inference.runner.utils import SpecDecodeMetadata

PLACEHOLDER_TOKEN_ID = -1


@jax.jit(static_argnames=[
    "num_speculative_tokens", "max_num_reqs_per_dp_rank", "vocab_size", "mesh"
])
def extract_last_sampled_tokens(
        spec_decode_metadata: SpecDecodeMetadata,
        sampled_token_ids: jnp.ndarray, num_speculative_tokens: int,
        vocab_size: int, max_num_reqs_per_dp_rank: int,
        mesh: jax.sharding.Mesh) -> tuple[jnp.ndarray, jnp.ndarray]:

    def _body(draft_lengths, sampled_token_ids):
        return _extract_last_sampled_tokens(draft_lengths, sampled_token_ids,
                                            num_speculative_tokens, vocab_size,
                                            max_num_reqs_per_dp_rank)

    data_spec = PartitionSpec(ShardingAxisName.ATTN_DATA)
    return jax.shard_map(
        _body,
        mesh=mesh,
        in_specs=(data_spec, data_spec),
        out_specs=(data_spec, data_spec),
    )(spec_decode_metadata.draft_lengths, sampled_token_ids)


@jax.jit(
    static_argnames=["num_speculative_tokens", "max_num_seq", "vocab_size"])
def _extract_last_sampled_tokens(
        draft_lengths: jnp.ndarray, sampled_token_ids: jnp.ndarray,
        num_speculative_tokens: int, vocab_size: int,
        max_num_seq: int) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Extract the last sampled token and number rejected tokens per seq. """
    # `sampled_token_ids` is the output of the rejection sampler.
    num_draft_tokens = draft_lengths
    batch_size = num_draft_tokens.shape[0]
    index_range = jax.lax.broadcasted_iota(
        jnp.int32, (batch_size, num_speculative_tokens), 1)
    valid_mask = index_range < num_draft_tokens[:, None]
    # `sampled_token_ids` has the flat layout
    # [main_tokens (sum(num_draft_tokens)), bonus_tokens (batch_size)].
    # `segment_starts[i]` is the offset of seq i's main tokens in that flat
    # array, so main_tokens_indices[i, j] = segment_starts[i] + j.
    segment_starts = jnp.pad(jnp.cumsum(num_draft_tokens)[:-1], (1, 0),
                             constant_values=0)
    main_tokens_indices = segment_starts[:, None] + index_range
    main_tokens_indices = jnp.where(valid_mask, main_tokens_indices, 0)
    main_tokens = jnp.where(valid_mask, sampled_token_ids[main_tokens_indices],
                            PLACEHOLDER_TOKEN_ID)
    main_tokens = jnp.where(main_tokens < vocab_size, main_tokens,
                            PLACEHOLDER_TOKEN_ID)
    bonus_tokens = sampled_token_ids[-batch_size:]
    bonus_tokens = jnp.where(bonus_tokens < vocab_size, bonus_tokens,
                             PLACEHOLDER_TOKEN_ID)
    num_valid_main = jnp.sum(main_tokens != PLACEHOLDER_TOKEN_ID, axis=1)
    last_main_idx = jnp.maximum(num_valid_main - 1, 0)
    last_main = main_tokens[jnp.arange(batch_size), last_main_idx]
    last_main = jnp.where(num_valid_main > 0, last_main, PLACEHOLDER_TOKEN_ID)
    has_bonus = bonus_tokens != PLACEHOLDER_TOKEN_ID
    last_sampled_per_seq = jnp.where(has_bonus, bonus_tokens, last_main)
    last_sampled_tokens = jnp.pad(last_sampled_per_seq,
                                  (0, max_num_seq - batch_size),
                                  constant_values=PLACEHOLDER_TOKEN_ID)
    num_rejected_per_seq = jnp.where(
        num_draft_tokens > 0,
        num_draft_tokens + 1 - num_valid_main - has_bonus.astype(jnp.int32),
        jnp.zeros_like(num_draft_tokens))
    num_rejected_tokens = jnp.pad(num_rejected_per_seq,
                                  (0, max_num_seq - batch_size),
                                  constant_values=0)
    return last_sampled_tokens, num_rejected_tokens


@jax.jit
def concat_last_sampled_tokens_and_draft_tokens(last_sampled_tokens,
                                                draft_tokens):
    return jnp.concat([last_sampled_tokens[:, None], draft_tokens],
                      axis=1).reshape(-1)


def filter_speculative_logprobs(
    log_token_ids: np.ndarray,
    logprobs_arr: np.ndarray,
    selected_token_ranks: np.ndarray,
    spec_decode_metadata: SpecDecodeMetadata,
    vocab_size: int,
    dp_size: int,
    num_reqs: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[int]]:
    """Filters and reorganizes logprobs for speculative decoding.

    This function extracts logprobs for accepted draft tokens and the bonus token
    for each request, and returns them in a flattened format along with cumulative
    counts.

    The input arrays (log_token_ids, logprobs_arr, selected_token_ranks) have a flat
    layout structured per DP rank. For each rank, the layout is:
    [draft_tokens (padded_tokens_length), bonus_tokens (padded_num_seqs_per_rank)]
    """
    INVALID_TOKEN_ID = -1

    # final_logits_indices contains indices for all draft tokens across all ranks.
    # We split it by dp_size to get the draft block length per rank.
    padded_tokens_length = spec_decode_metadata.final_logits_indices.shape[0]
    assert padded_tokens_length % dp_size == 0
    padded_tokens_length = padded_tokens_length // dp_size

    # Total output size per rank (draft block + bonus block)
    per_rank_output_size = log_token_ids.shape[0] // dp_size

    # Lists to collect filtered outputs per request (indexed by original request ID)
    req_logprob_token_ids = [[] for _ in range(num_reqs)]
    req_logprobs = [[] for _ in range(num_reqs)]
    req_sampled_token_ranks = [[] for _ in range(num_reqs)]

    # Process outputs rank by rank
    for rank in range(dp_size):
        rank_offset = rank * per_rank_output_size

        # Slice the draft (main) tokens portion for this rank
        main_token_ids = log_token_ids[rank_offset:rank_offset +
                                       padded_tokens_length]
        main_logprobs = logprobs_arr[rank_offset:rank_offset +
                                     padded_tokens_length]
        main_ranks = selected_token_ranks[rank_offset:rank_offset +
                                          padded_tokens_length]

        # Slice the bonus tokens portion for this rank (comes after draft tokens)
        bonus_token_ids = log_token_ids[rank_offset +
                                        padded_tokens_length:rank_offset +
                                        per_rank_output_size]
        bonus_logprobs = logprobs_arr[rank_offset +
                                      padded_tokens_length:rank_offset +
                                      per_rank_output_size]
        bonus_ranks = selected_token_ranks[rank_offset +
                                           padded_tokens_length:rank_offset +
                                           per_rank_output_size]

        # Number of sequences (requests) allocated for this rank
        padded_num_seqs_per_rank = bonus_token_ids.shape[0]
        # Actual draft lengths for requests in this rank
        cur_rank_num_draft_tokens = spec_decode_metadata.draft_lengths_cpu[
            rank * padded_num_seqs_per_rank:(rank + 1) *
            padded_num_seqs_per_rank]

        start_idx = 0
        # Map from rank-local sequence index to global request index
        req_indices = spec_decode_metadata.req_indices_dp[rank]
        for i, req_idx in enumerate(req_indices):
            # Skip padding requests
            if req_idx >= num_reqs:
                continue

            seq_length = int(cur_rank_num_draft_tokens[i])
            end_idx = start_idx + seq_length

            # Get draft logprobs for this specific sequence
            seq_main_token_ids = main_token_ids[start_idx:end_idx]
            seq_main_logprobs = main_logprobs[start_idx:end_idx]
            seq_main_ranks = main_ranks[start_idx:end_idx]

            # Filter out invalid/padding tokens.
            # During speculative execution, some draft positions might be filled with
            # placeholders if they are not used or if they are rejected.
            valid_mask = (seq_main_token_ids[:, 0] != INVALID_TOKEN_ID) & (
                seq_main_token_ids[:, 0] < vocab_size)

            valid_token_ids = seq_main_token_ids[valid_mask]
            valid_logprobs = seq_main_logprobs[valid_mask]
            valid_ranks = seq_main_ranks[valid_mask]

            # Check if the bonus token is valid (not a placeholder)
            bonus_token = bonus_token_ids[i, 0]
            if bonus_token != INVALID_TOKEN_ID and bonus_token < vocab_size:
                # Append bonus token logprobs to the valid draft logprobs
                valid_token_ids = np.concatenate(
                    [valid_token_ids, [bonus_token_ids[i]]], axis=0)
                valid_logprobs = np.concatenate(
                    [valid_logprobs, [bonus_logprobs[i]]], axis=0)
                valid_ranks = np.concatenate([valid_ranks, [bonus_ranks[i]]],
                                             axis=0)

            # Store the filtered logprobs for this request
            req_logprob_token_ids[req_idx] = valid_token_ids
            req_logprobs[req_idx] = valid_logprobs
            req_sampled_token_ranks[req_idx] = valid_ranks
            start_idx = end_idx

    # Flatten the collected lists back to 2D/1D arrays to match the expected output format.
    # We also compute cumulative sum of generated tokens (cu_num_generated_tokens).
    flat_token_ids = []
    flat_logprobs = []
    flat_ranks = []

    cu_num_generated_tokens = [0]
    current_cu = 0

    for r in range(num_reqs):
        num_gen = len(req_logprob_token_ids[r])
        current_cu += num_gen
        cu_num_generated_tokens.append(current_cu)

        if num_gen > 0:
            flat_token_ids.append(req_logprob_token_ids[r])
            flat_logprobs.append(req_logprobs[r])
            flat_ranks.append(req_sampled_token_ranks[r])

    if flat_token_ids:
        filtered_token_ids = np.concatenate(flat_token_ids, axis=0)
        filtered_logprobs = np.concatenate(flat_logprobs, axis=0)
        filtered_ranks = np.concatenate(flat_ranks, axis=0)
    else:
        # Return empty arrays with correct shape/dtype if no tokens were generated
        filtered_token_ids = np.empty((0, log_token_ids.shape[1]),
                                      dtype=log_token_ids.dtype)
        filtered_logprobs = np.empty((0, logprobs_arr.shape[1]),
                                     dtype=logprobs_arr.dtype)
        filtered_ranks = np.empty((0, ), dtype=selected_token_ranks.dtype)

    return (filtered_token_ids, filtered_logprobs, filtered_ranks,
            cu_num_generated_tokens)


@jax.jit(static_argnames=["mesh"])
def extend_logits_simple(
    target_logits: jax.Array,
    bonus_logits: jax.Array,
    mesh: jax.sharding.Mesh,
) -> jax.Array:
    """Concatenates target and bonus logits along the first axis."""

    def concat_fn(x, y):
        return jnp.concatenate([x, y], axis=0)

    logits_spec = PartitionSpec(ShardingAxisName.ATTN_DATA, None)
    return jax.shard_map(
        concat_fn,
        mesh=mesh,
        in_specs=(logits_spec, logits_spec),
        out_specs=logits_spec,
    )(target_logits.astype(jnp.float32), bonus_logits.astype(jnp.float32))


@jax.jit(static_argnames=["mesh"])
def process_and_extend_logits(
    mesh: jax.sharding.Mesh,
    target_logits: jax.Array,
    processed_bonus_logits: jax.Array,
    spec_decode_metadata: SpecDecodeMetadata,
    tpu_sampling_metadata: TPUSupportedSamplingMetadata,
) -> jax.Array:
    """Processes target logits and concatenates them with processed bonus logits."""
    # target_logits: [L, vocab]
    # processed_bonus_logits: [R, vocab]
    # draft_lengths: [B]
    target_logits = target_logits.astype(jnp.float32)

    def local_fn(local_target, local_bonus, local_draft_lengths, local_temp,
                 local_top_k, local_top_p):
        segment_ids = jnp.repeat(
            jnp.arange(local_draft_lengths.shape[0]),
            local_draft_lengths,
            total_repeat_length=local_target.shape[0],
        )
        temp = local_temp[segment_ids]
        top_k = local_top_k[segment_ids]
        top_p = local_top_p[segment_ids]

        local_meta = TPUSupportedSamplingMetadata(
            temperature=temp,
            top_k=top_k,
            top_p=top_p,
            do_sampling=tpu_sampling_metadata.do_sampling,
            logprobs=tpu_sampling_metadata.logprobs,
        )
        processed_target = _apply_sampling_transforms(local_target, local_meta)
        return jnp.concatenate([processed_target, local_bonus], axis=0)

    return jax.shard_map(
        local_fn,
        mesh=mesh,
        in_specs=(
            PartitionSpec(ShardingAxisName.ATTN_DATA),  # target_logits
            PartitionSpec(
                ShardingAxisName.ATTN_DATA),  # processed_bonus_logits
            PartitionSpec(ShardingAxisName.ATTN_DATA),  # draft_lengths
            PartitionSpec(ShardingAxisName.ATTN_DATA),  # temperature
            PartitionSpec(ShardingAxisName.ATTN_DATA),  # top_k
            PartitionSpec(ShardingAxisName.ATTN_DATA),  # top_p
        ),
        out_specs=PartitionSpec(ShardingAxisName.ATTN_DATA),
    )(
        target_logits,
        processed_bonus_logits,
        spec_decode_metadata.draft_lengths,
        tpu_sampling_metadata.temperature,
        tpu_sampling_metadata.top_k,
        tpu_sampling_metadata.top_p,
    )
