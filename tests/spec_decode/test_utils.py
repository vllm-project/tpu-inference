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

import jax.numpy as jnp
import numpy as np
import pytest

from tpu_inference.layers.jax.sample.rejection_sampler import RejectionSampler
from tpu_inference.runner.utils import SpecDecodeMetadata
from tpu_inference.spec_decode.jax.utils import (PLACEHOLDER_TOKEN_ID,
                                                 extract_last_sampled_tokens)

VOCAB_SIZE = 100


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
        draft_token_ids=jnp.zeros(int(num_draft_cpu.sum()), dtype=jnp.int32),
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

    last_sampled, num_rejected = extract_last_sampled_tokens(
        metadata, sampled, num_speculative, VOCAB_SIZE, max_num_seq)

    ref_last, ref_num_rej = _reference(sampled, num_draft_cpu, VOCAB_SIZE,
                                       max_num_seq)

    np.testing.assert_array_equal(np.asarray(last_sampled), ref_last)
    np.testing.assert_array_equal(np.asarray(num_rejected), ref_num_rej)


def test_extract_last_sampled_tokens_no_spec_decode():
    max_num_seq = 8
    sampled = jnp.asarray([7, 11, 42], dtype=jnp.int32)

    last_sampled, num_rejected = extract_last_sampled_tokens(
        None,
        sampled,
        num_speculative_tokens=3,
        vocab_size=VOCAB_SIZE,
        max_num_seq=max_num_seq,
    )

    expected_last = np.array([7, 11, 42] + [PLACEHOLDER_TOKEN_ID] * 5,
                             dtype=np.int32)
    expected_num_rejected = np.zeros(max_num_seq, dtype=np.int32)
    np.testing.assert_array_equal(np.asarray(last_sampled), expected_last)
    np.testing.assert_array_equal(np.asarray(num_rejected),
                                  expected_num_rejected)
