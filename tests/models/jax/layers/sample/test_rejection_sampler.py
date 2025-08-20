"""
Tests for the JAX-based rejection sampler for speculative decoding on TPU.
This test suite is structured to mirror the GPU rejection sampler tests.
"""
from dataclasses import dataclass
from typing import List, Tuple

import jax.numpy as jnp
import numpy as np
import pytest

from tpu_commons.models.jax.layers.sample.rejection_sampler import (
    PLACEHOLDER_TOKEN_ID, RejectionSampler)
from tpu_commons.models.jax.layers.sample.sampling_metadata import \
    TPUSupportedSamplingMetadata

# ======================== CONSTANTS ========================

PAD_TOKEN_ID = -999  # Padding token for draft_token_ids
VOCAB_SIZE = 128  # Default vocabulary size for tests
DEFAULT_PADDING_FACTOR = 1.5  # Default padding factor for padded tests

# ======================== DATA STRUCTURES ========================


@dataclass
class RejectionSamplerTestCase:
    """Test case data structure for rejection sampler scenarios."""
    name: str
    draft_tokens: List[int]
    target_tokens: List[int]
    num_draft_per_seq: List[int]  # number of draft tokens per sequence
    bonus_tokens: List[int]
    expected: List[List[int]]
    description: str = ""
    use_padding: bool = False  # Whether to add padding to draft tokens


# ======================== TEST DATA FACTORY ========================


class TestDataFactory:
    """Factory class for generating test cases."""

    @staticmethod
    def create_test_case(
            name: str,
            draft_tokens: List[int],
            target_tokens: List[int],
            num_draft_per_seq: List[int],
            bonus_tokens: List[int],
            expected: List[List[int]],
            description: str = "",
            use_padding: bool = False) -> RejectionSamplerTestCase:
        """Create a single test case."""
        return RejectionSamplerTestCase(name=name,
                                        draft_tokens=draft_tokens,
                                        target_tokens=target_tokens,
                                        num_draft_per_seq=num_draft_per_seq,
                                        bonus_tokens=bonus_tokens,
                                        expected=expected,
                                        description=description
                                        or name.replace("_", " ").title(),
                                        use_padding=use_padding)

    @classmethod
    def create_with_padding_variant(
            cls,
            name: str,
            draft_tokens: List[int],
            target_tokens: List[int],
            num_draft_per_seq: List[int],
            bonus_tokens: List[int],
            expected: List[List[int]],
            description: str = "") -> List[RejectionSamplerTestCase]:
        """Create both normal and padded versions of a test case."""
        test_cases = []

        # Create normal version
        test_cases.append(
            cls.create_test_case(name=name,
                                 draft_tokens=draft_tokens,
                                 target_tokens=target_tokens,
                                 num_draft_per_seq=num_draft_per_seq,
                                 bonus_tokens=bonus_tokens,
                                 expected=expected,
                                 description=description))

        # Create padded version if there are tokens
        if draft_tokens:
            test_cases.append(
                cls.create_test_case(
                    name=f"{name}_padded",
                    draft_tokens=draft_tokens,
                    target_tokens=target_tokens,
                    num_draft_per_seq=num_draft_per_seq,
                    bonus_tokens=bonus_tokens,
                    expected=expected,
                    description=f"{description} (with padding)",
                    use_padding=True))

        return test_cases

    @classmethod
    def get_basic_test_cases(cls) -> List[RejectionSamplerTestCase]:
        """Generate basic functionality test cases."""
        test_cases = []

        # Perfect match
        test_cases.extend(
            cls.create_with_padding_variant(
                name="perfect_match",
                draft_tokens=[1, 2, 3],
                target_tokens=[1, 2, 3],
                num_draft_per_seq=[3],
                bonus_tokens=[4],
                expected=[[1, 2, 3, 4]],
                description="Draft tokens perfectly match target argmax"))

        # Early mismatch
        test_cases.extend(
            cls.create_with_padding_variant(
                name="early_mismatch",
                draft_tokens=[1, 2, 3],
                target_tokens=[1, 5, 3],
                num_draft_per_seq=[3],
                bonus_tokens=[4],
                expected=[[1, 5]],
                description="Mismatch at position 1"))

        # Multiple sequences
        test_cases.extend(
            cls.create_with_padding_variant(
                name="multiple_sequences",
                draft_tokens=[1, 2, 3, 4],
                target_tokens=[1, 2, 3, 7],
                num_draft_per_seq=[2, 2],
                bonus_tokens=[5, 6],
                expected=[[1, 2, 5], [3, 7]],
                description="Multiple sequences with mixed results"))

        # Single token sequence
        test_cases.extend(
            cls.create_with_padding_variant(
                name="single_token_sequence",
                draft_tokens=[1],
                target_tokens=[1],
                num_draft_per_seq=[1],
                bonus_tokens=[2],
                expected=[[1, 2]],
                description="Single token sequence with perfect match"))

        # Empty sequence (no padding variant)
        test_cases.append(
            cls.create_test_case(
                name="empty_sequence",
                draft_tokens=[],
                target_tokens=[],
                num_draft_per_seq=[0],
                bonus_tokens=[5],
                expected=[[5]],
                description="Empty sequence gets bonus token"))

        return test_cases

    @classmethod
    def get_variable_length_test_cases(cls) -> List[RejectionSamplerTestCase]:
        """Generate variable length test cases."""
        test_cases = []

        # Variable length sequences
        test_cases.extend(
            cls.create_with_padding_variant(
                name="variable_length_sequences",
                draft_tokens=[1, 2, 3],
                target_tokens=[1, 5, 3],
                num_draft_per_seq=[2, 1],
                bonus_tokens=[6, 7],
                expected=[[1, 5], [3, 7]],
                description="Sequences with different lengths"))

        # All different lengths
        test_cases.extend(
            cls.create_with_padding_variant(
                name="all_different_lengths",
                draft_tokens=[1, 2, 3, 4, 5, 6],
                target_tokens=[1, 2, 3, 4, 5, 6],
                num_draft_per_seq=[1, 2, 3],
                bonus_tokens=[7, 9, 10],
                expected=[[1, 7], [2, 3, 9], [4, 5, 6, 10]],
                description="All sequences have different lengths"))

        # Mixed sequence lengths
        test_cases.extend(
            cls.create_with_padding_variant(
                name="mixed_sequence_lengths",
                draft_tokens=[1, 2, 3, 4, 5],
                target_tokens=[1, 2, 3, 7, 5],
                num_draft_per_seq=[2, 3],
                bonus_tokens=[6, 8],
                expected=[[1, 2, 6], [3, 7]],
                description="Mixed lengths with different outcomes"))

        return test_cases

    @classmethod
    def get_edge_case_test_cases(cls) -> List[RejectionSamplerTestCase]:
        """Generate edge case test cases."""
        test_cases = []

        # Zero length mixed
        test_cases.extend(
            cls.create_with_padding_variant(
                name="zero_length_mixed",
                draft_tokens=[1, 2],
                target_tokens=[1, 2],
                num_draft_per_seq=[0, 2],
                bonus_tokens=[5, 6],
                expected=[[5], [1, 2, 6]],
                description="Zero-length sequence mixed with normal"))

        # All zero length (no padding variant)
        test_cases.append(
            cls.create_test_case(name="all_zero_length",
                                 draft_tokens=[],
                                 target_tokens=[],
                                 num_draft_per_seq=[0, 0],
                                 bonus_tokens=[5, 6],
                                 expected=[[5], [6]],
                                 description="All sequences are zero-length"))

        # Immediate rejection
        test_cases.extend(
            cls.create_with_padding_variant(
                name="immediate_rejection",
                draft_tokens=[1, 2, 3, 4, 5, 6],
                target_tokens=[9, 2, 3, 4, 5, 6],
                num_draft_per_seq=[3, 2, 1],
                bonus_tokens=[10, 11, 12],
                expected=[[9], [4, 5, 11], [6, 12]],
                description="Mixed immediate rejection and perfect matches"))

        # First token mismatch
        test_cases.extend(
            cls.create_with_padding_variant(
                name="first_token_mismatch",
                draft_tokens=[1],
                target_tokens=[2],
                num_draft_per_seq=[1],
                bonus_tokens=[3],
                expected=[[2]],
                description="Single token mismatch"))

        return test_cases

    @classmethod
    def get_all_test_cases(cls) -> List[RejectionSamplerTestCase]:
        """Get all test cases including basic, variable length, and edge cases."""
        all_cases = []
        all_cases.extend(cls.get_basic_test_cases())
        all_cases.extend(cls.get_variable_length_test_cases())
        all_cases.extend(cls.get_edge_case_test_cases())
        return all_cases


# ======================== TEST HELPERS ========================


class RejectionSamplerTestHelper:
    """Helper class for rejection sampler tests."""

    @staticmethod
    def create_target_probs_from_tokens(
            target_token_ids: List[int],
            vocab_size: int = VOCAB_SIZE) -> jnp.ndarray:
        """
        Create target probabilities that will produce desired token ids on argmax.

        Args:
            target_token_ids: List of target token IDs
            vocab_size: Size of the vocabulary

        Returns:
            JAX array of target probabilities
        """
        num_tokens = len(target_token_ids)
        if num_tokens == 0:
            return jnp.empty((0, vocab_size), dtype=jnp.float32)

        # Create target probs with low values
        target_probs = jnp.full((num_tokens, vocab_size),
                                -100.0,
                                dtype=jnp.float32)

        # Set high values at desired token positions to make them the argmax
        for i, token_id in enumerate(target_token_ids):
            target_probs = target_probs.at[i, token_id].set(100.0)

        return target_probs

    @staticmethod
    def create_sampling_metadata(
            all_greedy: bool = True) -> TPUSupportedSamplingMetadata:
        """
        Create TPU sampling metadata object.

        Args:
            all_greedy: Whether to use greedy sampling (True) or not

        Returns:
            TPUSupportedSamplingMetadata object
        """
        return TPUSupportedSamplingMetadata(
            do_sampling=not all_greedy,  # do_sampling=False means greedy
            logprobs=False,
        )

    @staticmethod
    def create_padded_draft_tokens(
            draft_tokens: List[int],
            padding_factor: float = DEFAULT_PADDING_FACTOR) -> jnp.ndarray:
        """
        Create padded draft tokens array.

        Args:
            draft_tokens: List of draft tokens
            padding_factor: Factor to determine padding length

        Returns:
            JAX array of padded tokens
        """
        if not draft_tokens:
            return jnp.array([], dtype=jnp.int32)

        # Calculate padded length (at least 50% more than actual tokens)
        actual_length = len(draft_tokens)
        padded_length = max(actual_length + 2,
                            int(actual_length * padding_factor))

        # Create padded array
        padded_tokens = [PAD_TOKEN_ID] * padded_length

        # Copy actual tokens to the beginning
        for i, token in enumerate(draft_tokens):
            padded_tokens[i] = token

        return jnp.array(padded_tokens, dtype=jnp.int32)

    @staticmethod
    def prepare_test_inputs(
        test_case: RejectionSamplerTestCase,
        vocab_size: int = VOCAB_SIZE
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, int]:
        """
        Prepare inputs for rejection sampler test.

        Args:
            test_case: Test case with input data
            vocab_size: Vocabulary size

        Returns:
            Tuple of (draft_token_ids, target_probs, num_draft_tokens,
                     bonus_token_ids, max_spec_len)
        """
        helper = RejectionSamplerTestHelper()

        # Prepare draft tokens (with or without padding)
        if test_case.use_padding and test_case.draft_tokens:
            # For padded inputs, simulate how a real system would handle padding
            padded_draft_tokens = helper.create_padded_draft_tokens(
                test_case.draft_tokens)

            # Extract only the actual tokens
            num_draft_tokens = jnp.array(test_case.num_draft_per_seq,
                                         dtype=jnp.int32)
            total_actual_tokens = int(jnp.sum(num_draft_tokens))

            # Extract only the first total_actual_tokens from the padded array
            draft_token_ids = padded_draft_tokens[:total_actual_tokens]
            target_probs = helper.create_target_probs_from_tokens(
                test_case.target_tokens, vocab_size)
        else:
            draft_token_ids = jnp.array(test_case.draft_tokens,
                                        dtype=jnp.int32)
            target_probs = helper.create_target_probs_from_tokens(
                test_case.target_tokens, vocab_size)
            num_draft_tokens = jnp.array(test_case.num_draft_per_seq,
                                         dtype=jnp.int32)

        bonus_token_ids = jnp.array(test_case.bonus_tokens, dtype=jnp.int32)
        max_spec_len = int(
            jnp.max(num_draft_tokens)) if len(num_draft_tokens) > 0 else 1

        return (draft_token_ids, target_probs, num_draft_tokens,
                bonus_token_ids, max_spec_len)

    @staticmethod
    def run_rejection_sampler_test(
        rejection_sampler: RejectionSampler,
        test_case: RejectionSamplerTestCase,
        vocab_size: int = VOCAB_SIZE,
    ) -> None:
        """
        Run a rejection sampler test from test case data.

        Args:
            rejection_sampler: RejectionSampler instance
            test_case: Test case to run
            vocab_size: Vocabulary size
        """
        helper = RejectionSamplerTestHelper()
        metadata = helper.create_sampling_metadata(all_greedy=True)

        # Prepare inputs
        (draft_token_ids, target_probs, num_draft_tokens, bonus_token_ids,
         max_spec_len) = helper.prepare_test_inputs(test_case, vocab_size)

        # Call the rejection sampler
        output = rejection_sampler(
            draft_token_ids=draft_token_ids,
            num_draft_tokens=num_draft_tokens,
            max_spec_len=max_spec_len,
            draft_probs=None,
            target_probs=target_probs,
            bonus_token_ids=bonus_token_ids,
            sampling_metadata=metadata,
        )

        # Parse the output
        parsed_output = rejection_sampler.parse_output(
            output,
            vocab_size=vocab_size,
            num_draft_tokens_cpu=np.asarray(num_draft_tokens),
            batch_size=len(num_draft_tokens),
            padded_tokens_length=int(sum(num_draft_tokens)))

        assert parsed_output == test_case.expected, \
            f"Test '{test_case.name}': Expected {test_case.expected}, got {parsed_output}"


# ======================== FIXTURES ========================


@pytest.fixture
def rejection_sampler():
    """Fixture for the RejectionSampler."""
    return RejectionSampler()


@pytest.fixture
def test_helper():
    """Fixture for the test helper."""
    return RejectionSamplerTestHelper()


@pytest.fixture
def test_factory():
    """Fixture for the test data factory."""
    return TestDataFactory()


# ======================== TEST CLASSES ========================


class TestRejectionSampler:
    """Comprehensive test suite for rejection sampler."""

    # =============== Basic Functionality Tests ===============

    @pytest.mark.parametrize("test_case",
                             TestDataFactory.get_all_test_cases(),
                             ids=lambda tc: tc.name)
    def test_rejection_sampler_scenarios(self, rejection_sampler, test_case):
        """Test all rejection sampler scenarios including padded versions."""
        RejectionSamplerTestHelper.run_rejection_sampler_test(
            rejection_sampler, test_case)

    def test_multiple_mismatches(self, rejection_sampler, test_factory):
        """Test handling multiple sequences where both have mismatches."""
        test_cases = test_factory.create_with_padding_variant(
            name="multiple_mismatches",
            draft_tokens=[1, 2, 3, 4, 5, 6],
            target_tokens=[1, 2, 7, 4, 8, 6],
            num_draft_per_seq=[3, 3],
            bonus_tokens=[8, 9],
            expected=[[1, 2, 7], [4, 8]],
            description="Both sequences have mismatches")

        for test_case in test_cases:
            RejectionSamplerTestHelper.run_rejection_sampler_test(
                rejection_sampler, test_case)

    # =============== Parse Output Tests ===============

    def test_parse_output_basic(self, rejection_sampler):
        """Test the parse_output method with basic flattened format."""
        vocab_size = VOCAB_SIZE

        # Create flattened output: [main_tokens..., bonus_tokens...]
        main_tokens = jnp.array([10, 20, 30, 50, 60], dtype=jnp.int32)
        bonus_tokens = jnp.array([40, 70], dtype=jnp.int32)
        output_token_ids = jnp.concatenate([main_tokens, bonus_tokens])

        num_draft_tokens = jnp.array([3, 2], dtype=jnp.int32)

        parsed_output = rejection_sampler.parse_output(
            output_token_ids,
            vocab_size,
            num_draft_tokens_cpu=np.asarray(num_draft_tokens),
            batch_size=len(num_draft_tokens),
            padded_tokens_length=int(sum(num_draft_tokens)))

        expected = [[10, 20, 30, 40], [50, 60, 70]]
        assert parsed_output == expected, f"Expected {expected}, got {parsed_output}"

    def test_parse_output_with_placeholders(self, rejection_sampler):
        """Test parse_output with rejected tokens (placeholders)."""
        vocab_size = VOCAB_SIZE

        # Test with rejected tokens (placeholders)
        main_tokens = jnp.array(
            [10, PLACEHOLDER_TOKEN_ID, PLACEHOLDER_TOKEN_ID, 20, 30],
            dtype=jnp.int32)
        bonus_tokens = jnp.array([PLACEHOLDER_TOKEN_ID, 40], dtype=jnp.int32)
        output_token_ids = jnp.concatenate([main_tokens, bonus_tokens])

        num_draft_tokens = jnp.array([3, 2], dtype=jnp.int32)

        parsed_output = rejection_sampler.parse_output(
            output_token_ids,
            vocab_size,
            num_draft_tokens_cpu=np.asarray(num_draft_tokens),
            batch_size=len(num_draft_tokens),
            padded_tokens_length=int(sum(num_draft_tokens)))

        expected = [[10], [20, 30, 40]]
        assert parsed_output == expected, f"Expected {expected}, got {parsed_output}"

    def test_parse_output_invalid_tokens(self, rejection_sampler):
        """Test parse_output with tokens outside vocab size."""
        vocab_size = VOCAB_SIZE

        # Test with tokens outside vocab size
        main_tokens = jnp.array([10, vocab_size + 1, 20], dtype=jnp.int32)
        bonus_tokens = jnp.array([vocab_size + 2], dtype=jnp.int32)
        output_token_ids = jnp.concatenate([main_tokens, bonus_tokens])

        num_draft_tokens = jnp.array([3], dtype=jnp.int32)

        parsed_output = rejection_sampler.parse_output(
            output_token_ids,
            vocab_size,
            num_draft_tokens_cpu=np.asarray(num_draft_tokens),
            batch_size=len(num_draft_tokens),
            padded_tokens_length=int(sum(num_draft_tokens)))

        expected = [[10, 20]]  # Invalid tokens filtered out
        assert parsed_output == expected, f"Expected {expected}, got {parsed_output}"

    def test_parse_output_empty_sequences(self, rejection_sampler):
        """Test parse_output with empty sequences."""
        vocab_size = VOCAB_SIZE

        # Test with empty sequences
        main_tokens = jnp.array([], dtype=jnp.int32)
        bonus_tokens = jnp.array([50, 60], dtype=jnp.int32)
        output_token_ids = jnp.concatenate([main_tokens, bonus_tokens])

        num_draft_tokens = jnp.array([0, 0], dtype=jnp.int32)

        parsed_output = rejection_sampler.parse_output(
            output_token_ids,
            vocab_size,
            num_draft_tokens_cpu=np.asarray(num_draft_tokens),
            batch_size=len(num_draft_tokens),
            padded_tokens_length=int(sum(num_draft_tokens)))

        expected = [[50], [60]]  # Only bonus tokens
        assert parsed_output == expected, f"Expected {expected}, got {parsed_output}"

    # =============== Padding-Specific Tests ===============

    def test_padding_ignored_correctly(self, rejection_sampler, test_factory):
        """Test that padding tokens are completely ignored."""
        # Both versions should produce identical results
        test_cases = test_factory.create_with_padding_variant(
            name="padding_test",
            draft_tokens=[1, 2],
            target_tokens=[1, 5],
            num_draft_per_seq=[2],
            bonus_tokens=[3],
            expected=[[1, 5]],
            description="Test padding is ignored")

        for test_case in test_cases:
            RejectionSamplerTestHelper.run_rejection_sampler_test(
                rejection_sampler, test_case)

    def test_extreme_padding(self, rejection_sampler, test_helper):
        """Test with extreme padding (much longer than actual tokens)."""
        metadata = test_helper.create_sampling_metadata(all_greedy=True)

        # Create heavily padded input: [1, 2] + 20 padding tokens
        draft_tokens_with_extreme_padding = [1, 2] + [PAD_TOKEN_ID] * 20
        padded_draft_tokens = jnp.array(draft_tokens_with_extreme_padding,
                                        dtype=jnp.int32)

        # Extract only the actual tokens (first 2)
        num_draft_tokens = jnp.array([2], dtype=jnp.int32)
        total_actual_tokens = int(jnp.sum(num_draft_tokens))
        draft_token_ids = padded_draft_tokens[:total_actual_tokens]

        target_probs = test_helper.create_target_probs_from_tokens([1, 5],
                                                                   VOCAB_SIZE)
        bonus_token_ids = jnp.array([3], dtype=jnp.int32)
        max_spec_len = 2

        output = rejection_sampler(
            draft_token_ids=draft_token_ids,
            num_draft_tokens=num_draft_tokens,
            max_spec_len=max_spec_len,
            draft_probs=None,
            target_probs=target_probs,
            bonus_token_ids=bonus_token_ids,
            sampling_metadata=metadata,
        )

        parsed_output = rejection_sampler.parse_output(
            output,
            VOCAB_SIZE,
            num_draft_tokens_cpu=np.asarray(num_draft_tokens),
            batch_size=len(num_draft_tokens),
            padded_tokens_length=int(sum(num_draft_tokens)))

        expected = [[1, 5]]  # Should ignore all padding
        assert parsed_output == expected, f"Expected {expected}, got {parsed_output}"

    def test_realistic_flattened_with_padding(self, rejection_sampler,
                                              test_factory):
        """Test with realistic flattened input including padding."""
        test_case = test_factory.create_test_case(
            name="realistic_flattened_with_padding",
            draft_tokens=[1, 2, 3],
            target_tokens=[1, 5, 3],
            num_draft_per_seq=[2, 1],
            bonus_tokens=[6, 7],
            expected=[[1, 5], [3, 7]],
            description="Realistic flattened input with padding",
            use_padding=True)
        RejectionSamplerTestHelper.run_rejection_sampler_test(
            rejection_sampler, test_case)

    # =============== Segment Operation Edge Case Tests ===============

    def test_all_sequences_immediate_mismatch(self, rejection_sampler,
                                              test_factory):
        """Test where all sequences have immediate mismatches (first token rejected)."""
        test_cases = test_factory.create_with_padding_variant(
            name="all_immediate_mismatch",
            draft_tokens=[1, 2, 3, 4, 5, 6, 7, 8, 9],
            target_tokens=[10, 2, 3, 11, 5, 6, 12, 8,
                           9],  # All first tokens mismatch
            num_draft_per_seq=[3, 3, 3],
            bonus_tokens=[20, 21, 22],
            expected=[[10], [11], [12]],  # Only correction tokens, no bonus
            description="All sequences have immediate first token mismatch")

        for test_case in test_cases:
            RejectionSamplerTestHelper.run_rejection_sampler_test(
                rejection_sampler, test_case)

    def test_all_sequences_perfect_match(self, rejection_sampler,
                                         test_factory):
        """Test where all sequences have perfect matches (all tokens accepted)."""
        test_cases = test_factory.create_with_padding_variant(
            name="all_perfect_match",
            draft_tokens=[1, 2, 3, 4, 5, 6, 7, 8, 9],
            target_tokens=[1, 2, 3, 4, 5, 6, 7, 8,
                           9],  # All tokens match perfectly
            num_draft_per_seq=[3, 3, 3],
            bonus_tokens=[10, 11, 12],
            expected=[[1, 2, 3, 10], [4, 5, 6, 11],
                      [7, 8, 9, 12]],  # All accepted + bonus
            description="All sequences have perfect token matches")

        for test_case in test_cases:
            RejectionSamplerTestHelper.run_rejection_sampler_test(
                rejection_sampler, test_case)

    def test_extreme_length_imbalance(self, rejection_sampler, test_factory):
        """Test with extreme length imbalance between sequences."""
        # One very long sequence (15 tokens) with others being short (1-2 tokens)
        test_case = test_factory.create_test_case(
            name="extreme_length_imbalance",
            draft_tokens=[
                1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18
            ],
            target_tokens=[
                1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 20, 18
            ],
            num_draft_per_seq=[15, 1, 2],  # Very imbalanced lengths
            bonus_tokens=[100, 101, 102],
            expected=[
                [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
                 100],  # All 15 accepted + bonus
                [16, 101],  # Single token accepted + bonus
                [20]
            ],  # First token mismatch, no bonus
            description="Extreme length imbalance between sequences")
        RejectionSamplerTestHelper.run_rejection_sampler_test(
            rejection_sampler, test_case)

    def test_mixed_accept_reject_patterns(self, rejection_sampler,
                                          test_factory):
        """Test mixed scenarios with perfect matches and immediate rejections."""
        test_cases = test_factory.create_with_padding_variant(
            name="mixed_accept_reject",
            draft_tokens=[1, 2, 3, 4, 5, 6, 7, 8, 9],
            target_tokens=[
                1, 2, 3, 10, 5, 6, 7, 8, 9
            ],  # First: perfect, Second: immediate reject, Third: perfect
            num_draft_per_seq=[3, 3, 3],
            bonus_tokens=[20, 21, 22],
            expected=[[1, 2, 3, 20], [10], [7, 8, 9, 22]],  # Mixed results
            description="Mix of perfect matches and immediate rejections")

        for test_case in test_cases:
            RejectionSamplerTestHelper.run_rejection_sampler_test(
                rejection_sampler, test_case)

    def test_mismatches_at_same_position(self, rejection_sampler,
                                         test_factory):
        """Test where mismatches occur at exactly the same position across sequences."""
        test_cases = test_factory.create_with_padding_variant(
            name="same_position_mismatch",
            draft_tokens=[1, 2, 3, 4, 5, 6, 7, 8, 9],
            target_tokens=[1, 10, 3, 4, 11, 6, 7, 12,
                           9],  # All mismatch at position 1 (middle token)
            num_draft_per_seq=[3, 3, 3],
            bonus_tokens=[20, 21, 22],
            expected=[[1, 10], [4, 11], [7,
                                         12]],  # All reject at same position
            description="Mismatches at same position in all sequences")

        for test_case in test_cases:
            RejectionSamplerTestHelper.run_rejection_sampler_test(
                rejection_sampler, test_case)

    def test_single_long_sequence(self, rejection_sampler, test_helper):
        """Test a single very long sequence (approaching MAX_SPEC_LEN)."""
        metadata = test_helper.create_sampling_metadata(all_greedy=True)

        # Create a sequence with 30 draft tokens (close to MAX_SPEC_LEN=32)
        draft_tokens = list(range(1, 31))
        target_tokens = list(range(1, 28)) + [99, 29, 30
                                              ]  # Mismatch at position 27

        draft_token_ids = jnp.array(draft_tokens, dtype=jnp.int32)
        target_probs = test_helper.create_target_probs_from_tokens(
            target_tokens, VOCAB_SIZE)
        num_draft_tokens = jnp.array([30], dtype=jnp.int32)
        bonus_token_ids = jnp.array([100], dtype=jnp.int32)
        max_spec_len = 30

        output = rejection_sampler(
            draft_token_ids=draft_token_ids,
            num_draft_tokens=num_draft_tokens,
            max_spec_len=max_spec_len,
            draft_probs=None,
            target_probs=target_probs,
            bonus_token_ids=bonus_token_ids,
            sampling_metadata=metadata,
        )

        parsed_output = rejection_sampler.parse_output(
            output,
            VOCAB_SIZE,
            num_draft_tokens_cpu=np.asarray(num_draft_tokens),
            batch_size=len(num_draft_tokens),
            padded_tokens_length=int(sum(num_draft_tokens)))

        expected = [list(range(1, 28)) + [99]]  # Tokens up to mismatch point
        assert parsed_output == expected, f"Expected {expected}, got {parsed_output}"
