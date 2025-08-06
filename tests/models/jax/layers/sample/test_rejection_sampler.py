"""
Tests for the JAX-based rejection sampler for speculative decoding on TPU.
This test suite is structured to mirror the GPU rejection sampler tests.
"""
from typing import List

import jax.numpy as jnp
import pytest

from tpu_commons.models.jax.layers.sample.rejection_sampler import (
    PLACEHOLDER_TOKEN_ID, RejectionSampler)
from tpu_commons.models.jax.layers.sample.sampling_metadata import \
    TPUSupportedSamplingMetadata


@pytest.fixture
def rejection_sampler():
    """Fixture for the RejectionSampler."""
    return RejectionSampler()


def create_batched_target_probs(output_token_ids: List[List[int]],
                                vocab_size: int = 100) -> jnp.ndarray:
    """
    Helper function to create batched target probabilities that will produce
    desired token ids on argmax.
    """
    # Remove bonus tokens to get the target tokens for each step
    token_ids = [tokens[:-1] for tokens in output_token_ids]
    batch_size = len(token_ids)
    max_spec_len = max(len(tokens) for tokens in token_ids) if token_ids else 1

    # Create batched target probs with low values
    target_probs = jnp.full((batch_size, max_spec_len, vocab_size),
                            -100.0,
                            dtype=jnp.float32)

    # Set high values at desired token positions to make them the argmax
    for i, tokens in enumerate(token_ids):
        for j, token_id in enumerate(tokens):
            target_probs = target_probs.at[i, j, token_id].set(100.0)

    return target_probs


def create_sampling_metadata(
        all_greedy: bool = True) -> TPUSupportedSamplingMetadata:
    """Create TPU sampling metadata object."""
    return TPUSupportedSamplingMetadata(
        do_sampling=not all_greedy,  # do_sampling=False means greedy
        logprobs=False,
    )


def _run_rejection_sampler_test(rejection_sampler: RejectionSampler,
                                spec_tokens: List[List[int]],
                                output_tokens: List[List[int]],
                                expected_output: List[List[int]]):
    """Helper function to run a single rejection sampler test case."""
    metadata = create_sampling_metadata(all_greedy=True)
    target_probs = create_batched_target_probs(output_tokens)

    # Bonus tokens are the last token of each output sequence
    bonus_token_ids = jnp.array([[tokens[-1]] for tokens in output_tokens],
                                dtype=jnp.int32)

    # Create batched draft token IDs
    batch_size = len(spec_tokens)
    max_spec_len = max(len(tokens)
                       for tokens in spec_tokens) if spec_tokens else 1
    draft_token_ids = jnp.full((batch_size, max_spec_len),
                               PLACEHOLDER_TOKEN_ID,
                               dtype=jnp.int32)

    # Fill in the draft tokens
    for i, tokens in enumerate(spec_tokens):
        if tokens:
            draft_token_ids = draft_token_ids.at[i, :len(tokens)].set(
                jnp.array(tokens))

    # Create num_draft_tokens array
    num_draft_tokens = jnp.array([len(tokens) for tokens in spec_tokens],
                                 dtype=jnp.int32)

    # Call the rejection sampler with batched inputs
    output = rejection_sampler(
        draft_token_ids=draft_token_ids,
        num_draft_tokens=num_draft_tokens,
        max_spec_len=max_spec_len,
        draft_probs=None,
        target_probs=target_probs,
        bonus_token_ids=bonus_token_ids,
        sampling_metadata=metadata,
    )

    # Pad expected output with placeholders to match the output shape
    max_len = output.shape[1]
    padded_expected = [
        row + [PLACEHOLDER_TOKEN_ID] * (max_len - len(row))
        for row in expected_output
    ]
    expected = jnp.array(padded_expected, dtype=jnp.int32)

    assert jnp.array_equal(output,
                           expected), f"Expected {expected}, got {output}"


def test_perfect_match(rejection_sampler):
    """Test when draft tokens perfectly match target argmax (all accepted)."""
    spec_tokens = [[1, 2, 3]]
    output_tokens = [[1, 2, 3, 4]]  # Target argmax + bonus token 4
    expected_output = [[1, 2, 3, 4]]
    _run_rejection_sampler_test(rejection_sampler, spec_tokens, output_tokens,
                                expected_output)


def test_early_mismatch(rejection_sampler):
    """Test when there's an early mismatch in tokens."""
    spec_tokens = [[1, 2, 3]]  # Draft tokens
    output_tokens = [[1, 5, 3, 4]]  # Target argmax has mismatch at pos 1
    # Expected: accept 1, take target's 5, reject rest
    expected_output = [[1, 5]]
    _run_rejection_sampler_test(rejection_sampler, spec_tokens, output_tokens,
                                expected_output)


def test_multiple_sequences(rejection_sampler):
    """Test handling multiple sequences with mixed results."""
    spec_tokens = [[1, 2], [3, 4]]
    output_tokens = [[1, 2, 5], [3, 7, 6]]  # Req 1 matches, Req 2 mismatches
    # Expected: Req 1 all accepted + bonus, Req 2 mismatch at pos 1
    expected_output = [[1, 2, 5], [3, 7]]
    _run_rejection_sampler_test(rejection_sampler, spec_tokens, output_tokens,
                                expected_output)


def test_single_token_sequence(rejection_sampler):
    """Test handling sequences with a single token."""
    spec_tokens = [[1]]
    output_tokens = [[1, 2]]  # Match, so bonus token is added
    expected_output = [[1, 2]]
    _run_rejection_sampler_test(rejection_sampler, spec_tokens, output_tokens,
                                expected_output)


def test_empty_sequence(rejection_sampler):
    """Test handling an empty sequence of draft tokens."""
    spec_tokens = [[]]
    output_tokens = [[5]]  # No draft tokens, so just the bonus token
    expected_output = [[5]]
    _run_rejection_sampler_test(rejection_sampler, spec_tokens, output_tokens,
                                expected_output)


def test_multiple_mismatches(rejection_sampler):
    """Test handling multiple sequences where both have mismatches."""
    spec_tokens = [[1, 2, 3], [4, 5, 6]]
    output_tokens = [[1, 2, 7, 8], [4, 8, 6, 9]]
    # Expected: Req 1 mismatch at pos 2, Req 2 mismatch at pos 1
    expected_output = [[1, 2, 7], [4, 8]]
    _run_rejection_sampler_test(rejection_sampler, spec_tokens, output_tokens,
                                expected_output)


@pytest.mark.parametrize(
    "spec_tokens, output_tokens, expected_output",
    [
        # Perfect match with bonus
        ([[1, 2]], [[1, 2, 3]], [[1, 2, 3]]),
        # First token mismatch
        ([[1]], [[2, 3]], [[2]]),
        # Mixed matches in a batch
        ([[1, 2], [3, 4]], [[1, 5, 6], [3, 4, 7]], [[1, 5], [3, 4, 7]]),
    ])
def test_parametrized_cases(rejection_sampler, spec_tokens, output_tokens,
                            expected_output):
    """Run various parametrized test scenarios."""
    _run_rejection_sampler_test(rejection_sampler, spec_tokens, output_tokens,
                                expected_output)


def test_parse_output(rejection_sampler):
    """Test the parse_output method."""
    vocab_size = 100
    output_token_ids = jnp.array(
        [[10, 20, 30, PLACEHOLDER_TOKEN_ID],
         [40, 50, PLACEHOLDER_TOKEN_ID, PLACEHOLDER_TOKEN_ID]],
        dtype=jnp.int32)

    parsed_output = rejection_sampler.parse_output(output_token_ids,
                                                   vocab_size)

    expected = [[10, 20, 30], [40, 50]]
    assert parsed_output == expected, f"Expected {expected}, got {parsed_output}"


def test_parse_output_edge_cases(rejection_sampler):
    """Test the parse_output method with edge cases."""
    vocab_size = 100

    # All placeholders
    output_token_ids = jnp.full((2, 4), PLACEHOLDER_TOKEN_ID, dtype=jnp.int32)
    parsed_output = rejection_sampler.parse_output(output_token_ids,
                                                   vocab_size)
    assert parsed_output == [[], []]

    # Tokens outside vocab size
    output_token_ids = jnp.array(
        [[10, vocab_size + 1, 20], [vocab_size + 2, 30, PLACEHOLDER_TOKEN_ID]],
        dtype=jnp.int32)
    parsed_output = rejection_sampler.parse_output(output_token_ids,
                                                   vocab_size)
    assert parsed_output == [[10, 20], [30]]
