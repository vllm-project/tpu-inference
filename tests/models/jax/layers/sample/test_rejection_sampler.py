"""
Tests for the JAX-based rejection sampler for speculative decoding on TPU.
This test suite is structured to mirror the GPU rejection sampler tests.
"""
from typing import List, Optional

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


def create_flattened_target_probs(output_token_ids: List[List[int]],
                                  vocab_size: int = 100) -> jnp.ndarray:
    """
    Helper function to create flattened target probabilities that will produce
    desired token ids on argmax.
    """
    # Remove bonus tokens to get the target tokens for each step
    token_ids = [tokens[:-1] for tokens in output_token_ids]

    # Flatten the token IDs
    flattened_tokens = []
    for tokens in token_ids:
        flattened_tokens.extend(tokens)

    num_tokens = len(flattened_tokens)
    if num_tokens == 0:
        return jnp.empty((0, vocab_size), dtype=jnp.float32)

    # Create flattened target probs with low values
    target_probs = jnp.full((num_tokens, vocab_size),
                            -100.0,
                            dtype=jnp.float32)

    # Set high values at desired token positions to make them the argmax
    for i, token_id in enumerate(flattened_tokens):
        target_probs = target_probs.at[i, token_id].set(100.0)

    return target_probs


def convert_batched_to_flattened(
    spec_tokens: List[List[int]],
    output_tokens: List[List[int]],
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Helper function to convert batched test inputs to flattened format.

    Returns:
        draft_token_ids: Flattened draft tokens [num_tokens]
        target_probs: Flattened target probabilities [num_tokens, vocab_size]
        num_draft_tokens: Number of draft tokens per sequence [batch_size]
        bonus_token_ids: Bonus tokens [batch_size]
    """
    # Flatten draft tokens
    flattened_draft_tokens = []
    for tokens in spec_tokens:
        flattened_draft_tokens.extend(tokens)

    draft_token_ids = jnp.array(flattened_draft_tokens, dtype=jnp.int32)

    # Create flattened target probabilities
    target_probs = create_flattened_target_probs(output_tokens)

    # Number of draft tokens per sequence
    num_draft_tokens = jnp.array([len(tokens) for tokens in spec_tokens],
                                 dtype=jnp.int32)

    # Bonus tokens are the last token of each output sequence
    bonus_token_ids = jnp.array([tokens[-1] for tokens in output_tokens],
                                dtype=jnp.int32)

    return draft_token_ids, target_probs, num_draft_tokens, bonus_token_ids


def create_sampling_metadata(
        all_greedy: bool = True) -> TPUSupportedSamplingMetadata:
    """Create TPU sampling metadata object."""
    return TPUSupportedSamplingMetadata(
        do_sampling=not all_greedy,  # do_sampling=False means greedy
        logprobs=False,
    )


def _run_rejection_sampler_test(
    rejection_sampler: RejectionSampler,
    spec_tokens: List[List[int]],
    output_tokens: List[List[int]],
    expected_output: List[List[int]],
    num_draft_tokens_override: Optional[List[int]] = None,
):
    """Helper function to run a single rejection sampler test case."""
    metadata = create_sampling_metadata(all_greedy=True)

    # Convert to flattened format
    draft_token_ids, target_probs, num_draft_tokens, bonus_token_ids = convert_batched_to_flattened(
        spec_tokens, output_tokens)

    # Override num_draft_tokens if specified (for padding tests)
    if num_draft_tokens_override:
        num_draft_tokens = jnp.array(num_draft_tokens_override,
                                     dtype=jnp.int32)
        # Need to adjust flattened inputs based on the override
        total_tokens = sum(num_draft_tokens_override)
        draft_token_ids = draft_token_ids[:total_tokens]
        target_probs = target_probs[:total_tokens]

    max_spec_len = int(
        jnp.max(num_draft_tokens)) if len(num_draft_tokens) > 0 else 1

    # Call the rejection sampler with flattened inputs
    output = rejection_sampler(
        draft_token_ids=draft_token_ids,
        num_draft_tokens=num_draft_tokens,
        max_spec_len=max_spec_len,
        draft_probs=None,
        target_probs=target_probs,
        bonus_token_ids=bonus_token_ids,
        sampling_metadata=metadata,
    )

    # Parse the output using the new format
    parsed_output = rejection_sampler.parse_output(
        output, vocab_size=100, num_draft_tokens=num_draft_tokens)

    assert parsed_output == expected_output, f"Expected {expected_output}, got {parsed_output}"


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


def test_draft_tokens_with_explicit_padding(rejection_sampler):
    """
    Tests the case where the number of draft tokens is explicitly set to be
    less than the length of the provided draft token list.
    """
    # We provide a draft token list of length 4, but specify that only
    # the first 2 are actual draft tokens. The rest are padding.
    spec_tokens = [[1, 2, 98, 99]]  # 98 and 99 are just padding
    num_draft_tokens = [2]

    # Target argmax: [1, 5, ...], which mismatches at position 1
    output_tokens = [[1, 5, 8, 9, 10]]

    # Expected:
    # - Pos 0: draft 1 == target 1. Accept 1.
    # - Pos 1: draft 2 != target 5. Reject. Take target's 5.
    # - num_draft_tokens is 2, so we stop.
    # - Final output is [1, 5]. No bonus token because of rejection.
    expected_output = [[1, 5]]

    _run_rejection_sampler_test(
        rejection_sampler,
        spec_tokens,
        output_tokens,
        expected_output,
        num_draft_tokens_override=num_draft_tokens,
    )


def test_parse_output(rejection_sampler):
    """Test the parse_output method with the new flattened format."""
    vocab_size = 100

    # Create flattened output: [main_tokens..., bonus_tokens...]
    # Sequence 1: [10, 20, 30] + bonus 40
    # Sequence 2: [50, 60] + bonus 70
    main_tokens = jnp.array([10, 20, 30, 50, 60], dtype=jnp.int32)
    bonus_tokens = jnp.array([40, 70], dtype=jnp.int32)
    output_token_ids = jnp.concatenate([main_tokens, bonus_tokens])

    num_draft_tokens = jnp.array([3, 2], dtype=jnp.int32)

    parsed_output = rejection_sampler.parse_output(output_token_ids,
                                                   vocab_size,
                                                   num_draft_tokens)

    expected = [[10, 20, 30, 40], [50, 60, 70]]
    assert parsed_output == expected, f"Expected {expected}, got {parsed_output}"


def test_parse_output_edge_cases(rejection_sampler):
    """Test the parse_output method with edge cases."""
    vocab_size = 100

    # Test with rejected tokens (placeholders)
    # Sequence 1: [10, -1, -1] + bonus -1 (rejected)
    # Sequence 2: [20, 30] + bonus 40 (accepted)
    main_tokens = jnp.array(
        [10, PLACEHOLDER_TOKEN_ID, PLACEHOLDER_TOKEN_ID, 20, 30],
        dtype=jnp.int32)
    bonus_tokens = jnp.array([PLACEHOLDER_TOKEN_ID, 40], dtype=jnp.int32)
    output_token_ids = jnp.concatenate([main_tokens, bonus_tokens])

    num_draft_tokens = jnp.array([3, 2], dtype=jnp.int32)

    parsed_output = rejection_sampler.parse_output(output_token_ids,
                                                   vocab_size,
                                                   num_draft_tokens)
    expected = [[10], [20, 30, 40]]
    assert parsed_output == expected, f"Expected {expected}, got {parsed_output}"

    # Test with tokens outside vocab size
    # Sequence 1: [10, vocab_size+1, 20] + bonus vocab_size+2 (invalid tokens)
    main_tokens = jnp.array([10, vocab_size + 1, 20], dtype=jnp.int32)
    bonus_tokens = jnp.array([vocab_size + 2], dtype=jnp.int32)
    output_token_ids = jnp.concatenate([main_tokens, bonus_tokens])

    num_draft_tokens = jnp.array([3], dtype=jnp.int32)

    parsed_output = rejection_sampler.parse_output(output_token_ids,
                                                   vocab_size,
                                                   num_draft_tokens)
    expected = [[10, 20]]  # Invalid tokens filtered out
    assert parsed_output == expected, f"Expected {expected}, got {parsed_output}"

    # Test with empty sequences
    main_tokens = jnp.array([], dtype=jnp.int32)
    bonus_tokens = jnp.array([50, 60], dtype=jnp.int32)
    output_token_ids = jnp.concatenate([main_tokens, bonus_tokens])

    num_draft_tokens = jnp.array([0, 0], dtype=jnp.int32)

    parsed_output = rejection_sampler.parse_output(output_token_ids,
                                                   vocab_size,
                                                   num_draft_tokens)
    expected = [[50], [60]]  # Only bonus tokens
    assert parsed_output == expected, f"Expected {expected}, got {parsed_output}"
