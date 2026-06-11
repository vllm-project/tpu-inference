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
import pytest

from tpu_inference.layers.jax.sample.rejection_sampler import RejectionSampler
from tpu_inference.runner.utils import SpecDecodeMetadata
from tpu_inference.spec_decode.jax.utils import (PLACEHOLDER_TOKEN_ID,
                                                 extract_last_sampled_tokens,
                                                 filter_speculative_logprobs)

VOCAB_SIZE = 100


def _make_mesh():
    devices = np.array(jax.devices()).reshape((1, len(jax.devices())))
    return jax.sharding.Mesh(devices, axis_names=('data', 'model'))


def _build_inputs(seqs, num_speculative_tokens):
    """Build (sampled_token_ids, metadata, num_draft_tokens_cpu).

    `seqs` is a list of (main_tokens, bonus_token). `main_tokens` contains the
    per-position outputs of the rejection sampler for that sequence (with
    PLACEHOLDER_TOKEN_ID at rejected positions); its length is the number of
    draft tokens for that sequence and must be <= num_speculative_tokens.
    """
    main_flat = []
    bonus_flat = []
    num_draft = []
    for main_tokens, bonus_token in seqs:
        assert len(main_tokens) <= num_speculative_tokens
        main_flat.extend(main_tokens)
        bonus_flat.append(bonus_token)
        num_draft.append(len(main_tokens))

    sampled = jnp.asarray(main_flat + bonus_flat, dtype=jnp.int32)
    num_draft_cpu = np.asarray(num_draft, dtype=np.int32)

    metadata = SpecDecodeMetadata(
        draft_lengths=jnp.asarray(num_draft_cpu),
        target_logits_indices=jnp.zeros(int(num_draft_cpu.sum()),
                                        dtype=jnp.int32),
        bonus_logits_indices=jnp.zeros(len(num_draft_cpu), dtype=jnp.int32),
        final_logits_indices=jnp.zeros(len(num_draft_cpu), dtype=jnp.int32),
    )
    metadata.draft_lengths_cpu = num_draft_cpu
    return sampled, metadata, num_draft_cpu


def _reference(sampled, num_draft_cpu, vocab_size, max_num_seq):
    """Compute the reference (last_sampled, num_rejected) using parse_output."""
    batch_size = len(num_draft_cpu)
    padded_tokens_length = int(num_draft_cpu.sum())
    per_seq_tokens = RejectionSampler.parse_output(
        sampled,
        vocab_size,
        num_draft_cpu,
        batch_size,
        padded_tokens_length,
        dp_size=1,
        req_indices_dp={0: list(range(batch_size))},
    )

    last_sampled = np.full((max_num_seq, ),
                           PLACEHOLDER_TOKEN_ID,
                           dtype=np.int32)
    num_rejected = np.zeros((max_num_seq, ), dtype=np.int32)
    for i, toks in enumerate(per_seq_tokens):
        if toks:
            last_sampled[i] = toks[-1]
        if num_draft_cpu[i] > 0:
            num_rejected[i] = int(num_draft_cpu[i]) + 1 - len(toks)
    return last_sampled, num_rejected


@pytest.mark.parametrize(
    "case_name, seqs, num_speculative",
    [
        (
            "all_accepted_full_drafts",
            [([10, 20, 30], 40), ([5, 15, 25], 35)],
            3,
        ),
        (
            "all_accepted_partial_drafts",
            [([10, 20, 30], 40), ([5, 15], 25)],
            3,
        ),
        (
            "with_rejection_no_bonus",
            [
                ([10, 20, PLACEHOLDER_TOKEN_ID], PLACEHOLDER_TOKEN_ID),
                ([5, PLACEHOLDER_TOKEN_ID, PLACEHOLDER_TOKEN_ID
                  ], PLACEHOLDER_TOKEN_ID),
            ],
            3,
        ),
        (
            "some_seq_no_draft_tokens",
            [
                ([10, 20], 30),
                ([5, PLACEHOLDER_TOKEN_ID], PLACEHOLDER_TOKEN_ID),
                ([], 99),
            ],
            2,
        ),
        (
            "bonus_vocab_overflow_falls_back_to_last_main",
            [
                ([10, 20, 30], VOCAB_SIZE + 1),
                ([5, 15], VOCAB_SIZE + 7),
            ],
            3,
        ),
    ],
)
def test_extract_last_sampled_tokens_matches_parse_output(
        case_name, seqs, num_speculative):
    max_num_seq = 8
    sampled, metadata, num_draft_cpu = _build_inputs(seqs, num_speculative)

    mesh = _make_mesh()
    with jax.set_mesh(mesh):
        last_sampled, num_rejected = extract_last_sampled_tokens(
            metadata, sampled, num_speculative, VOCAB_SIZE, max_num_seq, mesh)

    ref_last, ref_num_rej = _reference(sampled, num_draft_cpu, VOCAB_SIZE,
                                       max_num_seq)

    np.testing.assert_array_equal(np.asarray(last_sampled), ref_last)
    np.testing.assert_array_equal(np.asarray(num_rejected), ref_num_rej)


def test_filter_speculative_logprobs():
    # Setup inputs matching the manual trace
    dp_size = 2
    num_reqs = 3
    vocab_size = 100
    padded_tokens_length = 5

    # log_token_ids: shape [14, 1]
    log_token_ids = np.array(
        [
            # Rank 0 (draft)
            [10],
            [11],
            [12],
            [20],
            [-1],
            # Rank 0 (bonus)
            [13],
            [-1],
            # Rank 1 (draft)
            [30],
            [31],
            [-1],
            [-1],
            [-1],
            # Rank 1 (bonus)
            [-1],
            [-1]
        ],
        dtype=np.int32)

    # logprobs_arr: shape [14, 1]
    logprobs_arr = np.array(
        [[110.0], [111.0], [112.0], [120.0], [0.0], [113.0], [0.0], [130.0],
         [131.0], [0.0], [0.0], [0.0], [0.0], [0.0]],
        dtype=np.float32)

    # selected_token_ranks: shape [14]
    selected_token_ranks = np.array([1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0],
                                    dtype=np.int32)

    metadata = SpecDecodeMetadata(
        draft_lengths=jnp.zeros(num_reqs),  # Not used by the function
        target_logits_indices=jnp.zeros(1),  # Not used
        bonus_logits_indices=jnp.zeros(1),  # Not used
        final_logits_indices=jnp.zeros(padded_tokens_length *
                                       dp_size),  # Shape is used: [10]
    )
    metadata.draft_lengths_cpu = np.array([3, 2, 4, 0], dtype=np.int32)
    metadata.req_indices_dp = {
        0: [0, 1],
        1: [2, 3]  # 3 is padding req (>= num_reqs)
    }

    # Call the function
    (filtered_token_ids, filtered_logprobs, filtered_ranks,
     cu_num_generated_tokens) = filter_speculative_logprobs(
         log_token_ids=log_token_ids,
         logprobs_arr=logprobs_arr,
         selected_token_ranks=selected_token_ranks,
         spec_decode_metadata=metadata,
         vocab_size=vocab_size,
         dp_size=dp_size,
         num_reqs=num_reqs,
     )

    # Expected outputs
    expected_token_ids = np.array([[10], [11], [12], [13], [20], [30], [31]],
                                  dtype=np.int32)

    expected_logprobs = np.array(
        [[110.0], [111.0], [112.0], [113.0], [120.0], [130.0], [131.0]],
        dtype=np.float32)

    expected_ranks = np.array([1, 1, 1, 1, 1, 1, 1], dtype=np.int32)

    expected_cu = [0, 4, 5, 7]

    # Assertions
    np.testing.assert_array_equal(filtered_token_ids, expected_token_ids)
    np.testing.assert_array_equal(filtered_logprobs, expected_logprobs)
    np.testing.assert_array_equal(filtered_ranks, expected_ranks)
    assert cu_num_generated_tokens == expected_cu
