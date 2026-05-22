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

from typing import Optional

import jax
import jax.numpy as jnp

from tpu_inference.runner.utils import SpecDecodeMetadata

PLACEHOLDER_TOKEN_ID = -1


@jax.jit(static_argnames=[
    "num_speculative_tokens", "max_num_reqs_per_dp_rank", "vocab_size",
    "sharding"
])
def extract_last_sampled_tokens(
        spec_decode_metadata: Optional[SpecDecodeMetadata],
        sampled_token_ids: jnp.ndarray, num_speculative_tokens: int,
        vocab_size: int, max_num_reqs_per_dp_rank: int,
        sharding: jax.sharding.NamedSharding
) -> tuple[jnp.ndarray, jnp.ndarray]:

    def _body(draft_lengths, sampled_token_ids):
        return _extract_last_sampled_tokens(draft_lengths, sampled_token_ids,
                                            num_speculative_tokens, vocab_size,
                                            max_num_reqs_per_dp_rank)

    data_spec = sharding.spec
    return jax.shard_map(
        _body,
        mesh=sharding.mesh,
        in_specs=(data_spec if spec_decode_metadata is not None else None,
                  data_spec),
        out_specs=(data_spec, data_spec),
    )(spec_decode_metadata.draft_lengths
      if spec_decode_metadata is not None else None, sampled_token_ids)


@jax.jit(
    static_argnames=["num_speculative_tokens", "max_num_seq", "vocab_size"])
def _extract_last_sampled_tokens(
        draft_lengths: Optional[jnp.ndarray], sampled_token_ids: jnp.ndarray,
        num_speculative_tokens: int, vocab_size: int,
        max_num_seq: int) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Extract the last sampled token and number rejected tokens per seq. """
    if draft_lengths is None:
        # No speculative decoding in the previous step.
        batch_size = sampled_token_ids.shape[0]
        last_sampled_tokens = jnp.pad(sampled_token_ids,
                                      (0, max_num_seq - batch_size),
                                      constant_values=PLACEHOLDER_TOKEN_ID)
        num_rejected_tokens = jnp.zeros(max_num_seq, dtype=jnp.int32)
        return last_sampled_tokens, num_rejected_tokens

    # If `draft_lengths` is not None, `sampled_token_ids`
    # is the output of the rejection sampler.
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
