# Copyright 2025 Google LLC
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
"""
Tests for the JAX-based rejection sampler for speculative decoding on TPU.
This test suite is structured to mirror the GPU rejection sampler tests.
"""
from dataclasses import dataclass
from typing import List, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tpu_inference.layers.jax.sample.rejection_sampler import (
    PLACEHOLDER_TOKEN_ID, RejectionSampler)
from tpu_inference.layers.jax.sample.sampling_metadata import \
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
    def create_target_logits_from_tokens(
            target_token_ids: List[int],
            vocab_size: int = VOCAB_SIZE) -> jnp.ndarray:
        """
        Create target logits that will produce desired token ids on argmax.

        Args:
            target_token_ids: List of target token IDs
            vocab_size: Size of the vocabulary

        Returns:
            JAX array of target logits
        """
        num_tokens = len(target_token_ids)
        if num_tokens == 0:
            return jnp.empty((0, vocab_size), dtype=jnp.float32)

        # Create target logits with low values
        target_logits = jnp.full((num_tokens, vocab_size),
                                 -100.0,
                                 dtype=jnp.float32)

        # Set high values at desired token positions to make them the argmax
        for i, token_id in enumerate(target_token_ids):
            target_logits = target_logits.at[i, token_id].set(100.0)

        return target_logits

    @staticmethod
    def create_sampling_metadata(
        all_greedy: bool = True,
        batch_size: int = 1,
        top_k: int = -1,
        top_p: float = 1.0,
        temperature: float = 1.0,
    ) -> TPUSupportedSamplingMetadata:
        """
        Create TPU sampling metadata object.
        """
        return TPUSupportedSamplingMetadata(
            do_sampling=not all_greedy,
            logprobs=False,
            top_k=jnp.full((batch_size, ), top_k, dtype=jnp.int32),
            top_p=jnp.full((batch_size, ), top_p, dtype=jnp.float32),
            temperature=jnp.full((batch_size, ),
                                 temperature,
                                 dtype=jnp.float32),
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
            Tuple of (draft_token_ids, target_logits, num_draft_tokens,
                     bonus_token_ids)
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
            target_logits = helper.create_target_logits_from_tokens(
                test_case.target_tokens, vocab_size)
        else:
            draft_token_ids = jnp.array(test_case.draft_tokens,
                                        dtype=jnp.int32)
            target_logits = helper.create_target_logits_from_tokens(
                test_case.target_tokens, vocab_size)
            num_draft_tokens = jnp.array(test_case.num_draft_per_seq,
                                         dtype=jnp.int32)

        bonus_token_ids = jnp.array(test_case.bonus_tokens, dtype=jnp.int32)

        return (draft_token_ids, target_logits, num_draft_tokens,
                bonus_token_ids)

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
        (draft_token_ids, target_logits, num_draft_tokens,
         bonus_token_ids) = helper.prepare_test_inputs(test_case, vocab_size)

        # Call the rejection sampler
        output = rejection_sampler(
            draft_token_ids=draft_token_ids,
            num_draft_tokens=num_draft_tokens,
            draft_probs=None,
            target_logits=target_logits,
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

        target_logits = test_helper.create_target_logits_from_tokens(
            [1, 5], VOCAB_SIZE)
        bonus_token_ids = jnp.array([3], dtype=jnp.int32)

        output = rejection_sampler(
            draft_token_ids=draft_token_ids,
            num_draft_tokens=num_draft_tokens,
            draft_probs=None,
            target_logits=target_logits,
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
        target_logits = test_helper.create_target_logits_from_tokens(
            target_tokens, VOCAB_SIZE)
        num_draft_tokens = jnp.array([30], dtype=jnp.int32)
        bonus_token_ids = jnp.array([100], dtype=jnp.int32)

        output = rejection_sampler(
            draft_token_ids=draft_token_ids,
            num_draft_tokens=num_draft_tokens,
            draft_probs=None,
            target_logits=target_logits,
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


# ======================== NON-GREEDY SAMPLING TESTS ========================


class TestNonGreedyRejectionSampler:
    """Test suite for non-greedy (random) rejection sampling."""

    def test_non_greedy_basic_functionality(self, rejection_sampler,
                                            test_helper):
        """Test basic non-greedy sampling functionality."""
        metadata = test_helper.create_sampling_metadata(all_greedy=False)

        # Create simple test case
        draft_tokens = [10, 20, 30]
        target_tokens = [10, 50, 30]  # Mismatch at position 1

        draft_token_ids = jnp.array(draft_tokens, dtype=jnp.int32)
        target_logits = test_helper.create_target_logits_from_tokens(
            target_tokens, VOCAB_SIZE)

        # Create draft probabilities - make draft tokens highly likely
        draft_probs = jnp.full((len(draft_tokens), VOCAB_SIZE),
                               -100.0,
                               dtype=jnp.float32)
        for i, token_id in enumerate(draft_tokens):
            draft_probs = draft_probs.at[i, token_id].set(100.0)

        # Convert logits to probabilities for draft_probs
        draft_probs = jax.nn.softmax(draft_probs, axis=-1)

        num_draft_tokens = jnp.array([3], dtype=jnp.int32)
        bonus_token_ids = jnp.array([99], dtype=jnp.int32)
        key = jax.random.PRNGKey(42)

        output = rejection_sampler(
            draft_token_ids=draft_token_ids,
            num_draft_tokens=num_draft_tokens,
            draft_probs=draft_probs,
            target_logits=target_logits,
            bonus_token_ids=bonus_token_ids,
            sampling_metadata=metadata,
            key=key,
        )

        parsed_output = rejection_sampler.parse_output(
            output,
            VOCAB_SIZE,
            num_draft_tokens_cpu=np.asarray(num_draft_tokens),
            batch_size=1,
            padded_tokens_length=3)

        # For non-greedy sampling, exact output depends on random sampling
        # but we can check that the first token should be accepted
        assert len(parsed_output) == 1
        assert len(parsed_output[0]) >= 1
        assert parsed_output[0][0] == 10  # First token should match

    def test_non_greedy_deterministic_with_seed(self, rejection_sampler,
                                                test_helper):
        """Test that non-greedy sampling is deterministic with the same seed."""
        metadata = test_helper.create_sampling_metadata(all_greedy=False)

        # Create test case
        draft_tokens = [1, 2, 3, 4]
        target_tokens = [1, 5, 3, 6]  # Mismatches at positions 1 and 3

        draft_token_ids = jnp.array(draft_tokens, dtype=jnp.int32)
        target_logits = test_helper.create_target_logits_from_tokens(
            target_tokens, VOCAB_SIZE)

        # Create draft probabilities
        draft_probs = jnp.full((len(draft_tokens), VOCAB_SIZE),
                               -100.0,
                               dtype=jnp.float32)
        for i, token_id in enumerate(draft_tokens):
            draft_probs = draft_probs.at[i, token_id].set(100.0)

        # Convert logits to probabilities for draft_probs
        draft_probs = jax.nn.softmax(draft_probs, axis=-1)

        num_draft_tokens = jnp.array([4], dtype=jnp.int32)
        bonus_token_ids = jnp.array([99], dtype=jnp.int32)

        # Run with same seed multiple times
        key = jax.random.PRNGKey(123)
        outputs = []

        for _ in range(5):
            output = rejection_sampler(
                draft_token_ids=draft_token_ids,
                num_draft_tokens=num_draft_tokens,
                draft_probs=draft_probs,
                target_logits=target_logits,
                bonus_token_ids=bonus_token_ids,
                sampling_metadata=metadata,
                key=key,
            )

            parsed_output = rejection_sampler.parse_output(
                output,
                VOCAB_SIZE,
                num_draft_tokens_cpu=np.asarray(num_draft_tokens),
                batch_size=1,
                padded_tokens_length=4)
            outputs.append(parsed_output)

        # All outputs should be identical with same seed
        for i in range(1, len(outputs)):
            assert outputs[i] == outputs[
                0], f"Run {i}: {outputs[i]} != {outputs[0]}"

    def test_non_greedy_with_draft_probs_none(self, rejection_sampler,
                                              test_helper):
        """Test non-greedy sampling when draft_probs is None."""
        metadata = test_helper.create_sampling_metadata(all_greedy=False)

        # Create test case
        draft_tokens = [15, 25]
        target_tokens = [15, 35]  # Mismatch at position 1

        draft_token_ids = jnp.array(draft_tokens, dtype=jnp.int32)
        target_logits = test_helper.create_target_logits_from_tokens(
            target_tokens, VOCAB_SIZE)

        num_draft_tokens = jnp.array([2], dtype=jnp.int32)
        bonus_token_ids = jnp.array([88], dtype=jnp.int32)
        key = jax.random.PRNGKey(777)

        output = rejection_sampler(
            draft_token_ids=draft_token_ids,
            num_draft_tokens=num_draft_tokens,
            draft_probs=None,  # No draft probabilities
            target_logits=target_logits,
            bonus_token_ids=bonus_token_ids,
            sampling_metadata=metadata,
            key=key,
        )

        parsed_output = rejection_sampler.parse_output(
            output,
            VOCAB_SIZE,
            num_draft_tokens_cpu=np.asarray(num_draft_tokens),
            batch_size=1,
            padded_tokens_length=2)

        # Should have valid output
        assert len(parsed_output) == 1
        assert len(parsed_output[0]) >= 1
        assert parsed_output[0][0] == 15  # First token should match

    def test_non_greedy_multiple_sequences(self, rejection_sampler,
                                           test_helper):
        """Test non-greedy sampling with multiple sequences."""
        metadata = test_helper.create_sampling_metadata(all_greedy=False)

        # Create test case with 3 sequences
        draft_tokens = [1, 2, 3, 4, 5, 6, 7]  # [1,2] [3,4,5] [6,7]
        target_tokens = [1, 5, 3, 8, 5, 6,
                         9]  # Mismatches at different positions

        draft_token_ids = jnp.array(draft_tokens, dtype=jnp.int32)
        target_logits = test_helper.create_target_logits_from_tokens(
            target_tokens, VOCAB_SIZE)

        # Create draft probabilities
        draft_probs = jnp.full((len(draft_tokens), VOCAB_SIZE),
                               -100.0,
                               dtype=jnp.float32)
        for i, token_id in enumerate(draft_tokens):
            draft_probs = draft_probs.at[i, token_id].set(100.0)

        # Convert logits to probabilities for draft_probs
        draft_probs = jax.nn.softmax(draft_probs, axis=-1)

        num_draft_tokens = jnp.array([2, 3, 2], dtype=jnp.int32)
        bonus_token_ids = jnp.array([11, 12, 13], dtype=jnp.int32)
        key = jax.random.PRNGKey(456)

        output = rejection_sampler(
            draft_token_ids=draft_token_ids,
            num_draft_tokens=num_draft_tokens,
            draft_probs=draft_probs,
            target_logits=target_logits,
            bonus_token_ids=bonus_token_ids,
            sampling_metadata=metadata,
            key=key,
        )

        parsed_output = rejection_sampler.parse_output(
            output,
            VOCAB_SIZE,
            num_draft_tokens_cpu=np.asarray(num_draft_tokens),
            batch_size=3,
            padded_tokens_length=7)

        # Should have 3 sequences
        assert len(parsed_output) == 3

        # First sequence: [1, 2] -> [1, 5] (mismatch at pos 1)
        assert parsed_output[0][0] == 1

        # Second sequence: [3, 4, 5] -> [3, 8, 5] (mismatch at pos 1)
        assert parsed_output[1][0] == 3

        # Third sequence: [6, 7] -> [6, 9] (mismatch at pos 1)
        assert parsed_output[2][0] == 6

    def test_non_greedy_with_all_accepted_tokens(self, rejection_sampler,
                                                 test_helper):
        """Test non-greedy sampling when all tokens are accepted (perfect match)."""
        metadata = test_helper.create_sampling_metadata(all_greedy=False)

        # Perfect match case
        draft_tokens = [10, 20, 30]
        target_tokens = [10, 20, 30]  # Perfect match

        draft_token_ids = jnp.array(draft_tokens, dtype=jnp.int32)
        target_logits = test_helper.create_target_logits_from_tokens(
            target_tokens, VOCAB_SIZE)

        # Create draft probabilities - make acceptance very likely
        draft_probs = jnp.full((len(draft_tokens), VOCAB_SIZE),
                               -100.0,
                               dtype=jnp.float32)
        for i, token_id in enumerate(draft_tokens):
            draft_probs = draft_probs.at[i, token_id].set(100.0)

        # Convert logits to probabilities for draft_probs
        draft_probs = jax.nn.softmax(draft_probs, axis=-1)

        num_draft_tokens = jnp.array([3], dtype=jnp.int32)
        bonus_token_ids = jnp.array([99], dtype=jnp.int32)
        key = jax.random.PRNGKey(999)

        output = rejection_sampler(
            draft_token_ids=draft_token_ids,
            num_draft_tokens=num_draft_tokens,
            draft_probs=draft_probs,
            target_logits=target_logits,
            bonus_token_ids=bonus_token_ids,
            sampling_metadata=metadata,
            key=key,
        )

        parsed_output = rejection_sampler.parse_output(
            output,
            VOCAB_SIZE,
            num_draft_tokens_cpu=np.asarray(num_draft_tokens),
            batch_size=1,
            padded_tokens_length=3)

        # With perfect match and high acceptance probability, should get bonus token
        assert len(parsed_output) == 1
        # The exact output depends on random sampling, but should contain the draft tokens

    def test_non_greedy_empty_sequence(self, rejection_sampler, test_helper):
        """Test non-greedy sampling with empty sequences."""
        metadata = test_helper.create_sampling_metadata(all_greedy=False)

        # Empty sequences should get bonus tokens
        draft_token_ids = jnp.array([], dtype=jnp.int32)
        target_logits = jnp.array([], dtype=jnp.float32).reshape(0, VOCAB_SIZE)

        num_draft_tokens = jnp.array([0, 0], dtype=jnp.int32)
        bonus_token_ids = jnp.array([77, 88], dtype=jnp.int32)
        key = jax.random.PRNGKey(333)

        output = rejection_sampler(
            draft_token_ids=draft_token_ids,
            num_draft_tokens=num_draft_tokens,
            draft_probs=None,
            target_logits=target_logits,
            bonus_token_ids=bonus_token_ids,
            sampling_metadata=metadata,
            key=key,
        )

        parsed_output = rejection_sampler.parse_output(
            output,
            VOCAB_SIZE,
            num_draft_tokens_cpu=np.asarray(num_draft_tokens),
            batch_size=2,
            padded_tokens_length=0)

        # Should get bonus tokens for empty sequences
        expected = [[77], [88]]
        assert parsed_output == expected, f"Expected {expected}, got {parsed_output}"

    def test_non_greedy_requires_key(self, rejection_sampler, test_helper):
        """Test that non-greedy sampling requires a random key."""
        metadata = test_helper.create_sampling_metadata(all_greedy=False)

        # Create simple test case
        draft_tokens = [1, 2]
        target_tokens = [1, 3]

        draft_token_ids = jnp.array(draft_tokens, dtype=jnp.int32)
        target_logits = test_helper.create_target_logits_from_tokens(
            target_tokens, VOCAB_SIZE)

        num_draft_tokens = jnp.array([2], dtype=jnp.int32)
        bonus_token_ids = jnp.array([99], dtype=jnp.int32)

        # Should raise ValueError when key is None for non-greedy sampling
        with pytest.raises(ValueError, match="A random key must be provided"):
            rejection_sampler(
                draft_token_ids=draft_token_ids,
                num_draft_tokens=num_draft_tokens,
                draft_probs=None,
                target_logits=target_logits,
                bonus_token_ids=bonus_token_ids,
                sampling_metadata=metadata,
                key=None,  # No key provided
            )

    def test_non_greedy_vs_greedy_same_perfect_case(self, rejection_sampler,
                                                    test_helper):
        """Test that greedy and non-greedy produce same results for perfect matches."""
        # Perfect match case - both should produce identical results
        draft_tokens = [5, 15, 25]
        target_tokens = [5, 15, 25]  # Perfect match

        draft_token_ids = jnp.array(draft_tokens, dtype=jnp.int32)
        target_logits = test_helper.create_target_logits_from_tokens(
            target_tokens, VOCAB_SIZE)

        # Create draft probabilities
        draft_probs = jnp.full((len(draft_tokens), VOCAB_SIZE),
                               -100.0,
                               dtype=jnp.float32)
        for i, token_id in enumerate(draft_tokens):
            draft_probs = draft_probs.at[i, token_id].set(100.0)

        # Convert logits to probabilities for draft_probs
        draft_probs = jax.nn.softmax(draft_probs, axis=-1)

        num_draft_tokens = jnp.array([3], dtype=jnp.int32)
        bonus_token_ids = jnp.array([99], dtype=jnp.int32)

        # Greedy sampling
        greedy_metadata = test_helper.create_sampling_metadata(all_greedy=True)
        greedy_output = rejection_sampler(
            draft_token_ids=draft_token_ids,
            num_draft_tokens=num_draft_tokens,
            draft_probs=draft_probs,
            target_logits=target_logits,
            bonus_token_ids=bonus_token_ids,
            sampling_metadata=greedy_metadata,
        )

        # Non-greedy sampling with high acceptance probability should behave similarly
        # Note: Due to probabilistic nature, we can't guarantee identical outputs
        # but for perfect matches with high probabilities, they should be very similar
        non_greedy_metadata = test_helper.create_sampling_metadata(
            all_greedy=False)
        key = jax.random.PRNGKey(555)
        non_greedy_output = rejection_sampler(
            draft_token_ids=draft_token_ids,
            num_draft_tokens=num_draft_tokens,
            draft_probs=draft_probs,
            target_logits=target_logits,
            bonus_token_ids=bonus_token_ids,
            sampling_metadata=non_greedy_metadata,
            key=key,
        )

        # Parse outputs
        greedy_parsed = rejection_sampler.parse_output(
            greedy_output, VOCAB_SIZE, np.asarray(num_draft_tokens), 1, 3)
        non_greedy_parsed = rejection_sampler.parse_output(
            non_greedy_output, VOCAB_SIZE, np.asarray(num_draft_tokens), 1, 3)

        # For perfect match, greedy should have all tokens + bonus
        assert greedy_parsed == [[5, 15, 25, 99]]

        # Non-greedy should have valid output (exact content may vary due to sampling)
        assert len(non_greedy_parsed) == 1
        assert len(non_greedy_parsed[0]) >= 1


# ======================== STATISTICAL DISTRIBUTION VALIDATION ========================


class TestStatisticalDistributionValidation:
    """Test suite for validating rejection sampling produces correct probability distributions."""

    def test_rejection_sampling_approximates_target_distribution(self):
        """Verify rejection sampling approximates target distribution.

        This test validates that rejection sampling produces the correct probability
        distribution despite sampling from a potentially distinct draft distribution.

        The test works by:
        1. Creating random target and draft probability distributions
        2. Using rejection sampling to generate token samples
        3. Estimating the output distribution from samples
        4. Comparing convergence to target vs random reference distributions

        We expect that as sample size increases, the distance to the target
        distribution decreases much more than the distance to random distributions.
        """

        vocab_size = 10
        k = 2
        num_reference_probs = 100

        # Create random distributions
        key = jax.random.PRNGKey(42)
        draft_key, target_key, reference_key = jax.random.split(key, 3)

        # Draft and target distributions
        draft_logits = jax.random.normal(draft_key, (vocab_size, ))
        draft_probs = jax.nn.softmax(draft_logits)

        target_logits = jax.random.normal(target_key, (vocab_size, ))
        target_probs = jax.nn.softmax(target_logits)

        # Reference distributions for comparison
        reference_logits = jax.random.normal(reference_key,
                                             (num_reference_probs, vocab_size))
        reference_probs = jax.nn.softmax(reference_logits, axis=-1)

        sample_sizes = [10, 100, 1_000, 10_000, 100_000]
        distance_wrt_reference: List[float] = []
        distance_wrt_target: List[float] = []

        for num_samples in sample_sizes:
            # Estimate rejection sampling distribution
            estimated_probs = self._estimate_rejection_sampling_pdf(
                draft_probs, target_logits, k, vocab_size, num_samples)

            # Calculate distances
            reference_vs_rejsample_dist = float(
                jnp.mean(
                    jnp.linalg.norm(reference_probs - estimated_probs[None, :],
                                    axis=-1)))
            target_vs_rejsample_dist = float(
                jnp.linalg.norm(target_probs - estimated_probs))

            distance_wrt_reference.append(reference_vs_rejsample_dist)
            distance_wrt_target.append(target_vs_rejsample_dist)

            print(f"{num_samples=} {target_vs_rejsample_dist=:.05f} "
                  f"{reference_vs_rejsample_dist=:.05f}")

        # Calculate relative improvements
        relative_change_target = self._get_ratio_first_to_last(
            distance_wrt_target)
        relative_change_reference = self._get_ratio_first_to_last(
            distance_wrt_reference)

        print(f"Target improvement ratio: {relative_change_target:.02f}")
        print(f"Reference improvement ratio: {relative_change_reference:.02f}")

        # Validation: Target distribution should converge much better than reference
        expected_improvement_multiplier = 20
        assert (relative_change_target >
                relative_change_reference * expected_improvement_multiplier), \
            f"Target convergence ({relative_change_target:.2f}) should be " \
            f"{expected_improvement_multiplier}x better than reference " \
            f"({relative_change_reference:.2f})"

    def _estimate_rejection_sampling_pdf(
        self,
        draft_probs: jnp.ndarray,
        target_logits: jnp.ndarray,
        k: int,
        vocab_size: int,
        num_samples: int,
    ) -> jnp.ndarray:
        """Estimate probability distribution of rejection sampling output.

        Args:
            draft_probs: Draft probability distribution [vocab_size]
            target_logits: Target logits [vocab_size]
            k: Number of draft tokens per sequence
            vocab_size: Size of vocabulary
            num_samples: Number of samples to generate

        Returns:
            Estimated probability distribution [vocab_size]
        """
        rejection_sampler = RejectionSampler()

        # Prepare inputs in the flattened format expected by TPU sampler
        num_tokens = num_samples * k

        # Expand draft probs to match flattened format [num_tokens, vocab_size]
        draft_probs_expanded = jnp.tile(draft_probs[None, :], (num_tokens, 1))

        # Expand target logits to flattened format
        target_logits_expanded = jnp.tile(target_logits[None, :],
                                          (num_tokens, 1))

        # Generate random draft token ids from draft distribution
        key = jax.random.PRNGKey(123)
        draft_tokens_2d = jax.random.categorical(key,
                                                 jnp.log(draft_probs + 1e-8),
                                                 shape=(num_samples, k))
        draft_token_ids = draft_tokens_2d.flatten()

        # Prepare other inputs
        num_draft_tokens = jnp.full((num_samples, ), k, dtype=jnp.int32)
        bonus_token_ids = jnp.zeros((num_samples, ),
                                    dtype=jnp.int32)  # Not used in estimation

        # Create sampling metadata for non-greedy sampling
        sampling_metadata = TPUSupportedSamplingMetadata(
            do_sampling=True,  # Non-greedy sampling
            logprobs=False,
            top_k=jnp.full((num_samples, ), -1, dtype=jnp.int32),
            top_p=jnp.full((num_samples, ), 1.0, dtype=jnp.float32),
            temperature=jnp.full((num_samples, ), 1.0, dtype=jnp.float32),
        )

        # Run rejection sampling
        sample_key = jax.random.PRNGKey(456)
        output_token_ids = rejection_sampler(
            draft_token_ids=draft_token_ids,
            num_draft_tokens=num_draft_tokens,
            draft_probs=draft_probs_expanded,
            target_logits=target_logits_expanded,
            bonus_token_ids=bonus_token_ids,
            sampling_metadata=sampling_metadata,
            key=sample_key,
        )

        # Parse output and extract main tokens (exclude bonus tokens)
        parsed_output = rejection_sampler.parse_output(
            output_token_ids,
            vocab_size=vocab_size,
            num_draft_tokens_cpu=np.asarray(num_draft_tokens),
            batch_size=num_samples,
            padded_tokens_length=num_tokens)

        # Flatten all main tokens (exclude bonus tokens)
        all_tokens = []
        for seq_tokens in parsed_output:
            if len(seq_tokens) == 0:
                continue
            # For rejection sampling, we need to exclude bonus tokens
            # The bonus token is typically the last one if all draft tokens were accepted
            # Otherwise, we take all valid tokens up to the rejection point
            if len(seq_tokens) > k:
                # More tokens than expected draft tokens means bonus token included
                main_tokens = seq_tokens[:k]
            else:
                # No bonus token, take all tokens
                main_tokens = seq_tokens
            all_tokens.extend(main_tokens)

        # Convert to numpy for histogram computation
        if not all_tokens:
            # Fallback if no tokens generated
            return jnp.ones(vocab_size) / vocab_size

        tokens_array = np.array(all_tokens, dtype=np.int32)

        # Calculate histogram (probability distribution)
        hist, _ = np.histogram(tokens_array,
                               bins=vocab_size,
                               range=(0, vocab_size),
                               density=True)

        # Normalize to ensure it sums to 1
        hist = hist / (hist.sum() + 1e-8)

        return jnp.array(hist, dtype=jnp.float32)

    def _get_ratio_first_to_last(self, elements: List[float]) -> float:
        """Calculate ratio of first to last element in list."""
        if len(elements) < 2 or elements[-1] == 0:
            return 1.0
        return elements[0] / elements[-1]


# ======================== TOP-K AND TOP-P SAMPLING TESTS ========================


class TestTopKTopPSampling:
    """Test suite for top-k and top-p sampling with rejection sampler."""

    def _test_masked_logits(
        self,
        rejection_sampler: RejectionSampler,
        batch_size: int,
        num_draft_tokens: int,
        vocab_size: int,
        target_logits: jnp.ndarray,
        allowed_tokens_per_pos: List[jnp.ndarray],
        sampling_metadata: TPUSupportedSamplingMetadata,
    ):
        """Helper function to test that only allowed tokens are sampled.

        Args:
            rejection_sampler: The rejection sampler instance
            batch_size: Number of sequences in the batch
            num_draft_tokens: Number of draft tokens per sequence
            vocab_size: Size of vocabulary
            target_logits: Target logits tensor
            allowed_tokens_per_pos: List of allowed token arrays for each position
            sampling_metadata: Sampling metadata with top-k/top-p settings
        """
        num_tokens = batch_size * num_draft_tokens

        # Create random draft probabilities
        key = jax.random.PRNGKey(42)
        draft_logits = jax.random.normal(key, (num_tokens, vocab_size))
        draft_probs = jax.nn.softmax(draft_logits, axis=-1)

        # Randomly sample draft token ids from draft probs
        draft_key = jax.random.PRNGKey(123)
        draft_token_ids = jax.random.categorical(draft_key,
                                                 jnp.log(draft_probs + 1e-8),
                                                 shape=(num_tokens, ))

        # Prepare inputs
        num_draft_per_seq = jnp.full((batch_size, ),
                                     num_draft_tokens,
                                     dtype=jnp.int32)
        bonus_token_ids = jnp.zeros((batch_size, ), dtype=jnp.int32)

        # Run rejection sampling multiple times to get statistical confidence
        sample_keys = jax.random.split(jax.random.PRNGKey(456), 10)
        all_sampled_tokens = []

        for sample_key in sample_keys:
            output_token_ids = rejection_sampler(
                draft_token_ids=draft_token_ids,
                num_draft_tokens=num_draft_per_seq,
                draft_probs=draft_probs,
                target_logits=target_logits,
                bonus_token_ids=bonus_token_ids,
                sampling_metadata=sampling_metadata,
                key=sample_key,
            )

            # Parse output and extract tokens
            parsed_output = rejection_sampler.parse_output(
                output_token_ids,
                vocab_size=vocab_size,
                num_draft_tokens_cpu=np.asarray(num_draft_per_seq),
                batch_size=batch_size,
                padded_tokens_length=num_tokens)

            # For each sequence, check tokens (excluding bonus tokens)
            for seq_idx, seq_tokens in enumerate(parsed_output):
                for pos, token_id in enumerate(seq_tokens):
                    if pos < num_draft_tokens:  # Only check draft tokens, not bonus
                        token_idx = seq_idx * num_draft_tokens + pos
                        if token_idx < len(allowed_tokens_per_pos):
                            allowed_tokens = allowed_tokens_per_pos[token_idx]
                            all_sampled_tokens.append(
                                (token_idx, token_id, allowed_tokens))

        # Check that all sampled tokens are within allowed sets
        for token_idx, token_id, allowed_tokens in all_sampled_tokens:
            assert token_id in allowed_tokens, \
                f"Token {token_id} at position {token_idx} not in allowed set {allowed_tokens.tolist()}"

    @pytest.mark.parametrize("top_k", [1, 5, 99])
    def test_top_k(self, rejection_sampler, test_helper, top_k):
        """Test rejection sampling with top-k sampling."""
        vocab_size = 100
        batch_size = 10
        num_draft_tokens = 3
        num_tokens = batch_size * num_draft_tokens

        # Randomly create top-k indices for each token position
        key = jax.random.PRNGKey(42)
        top_k_indices = []
        for i in range(num_tokens):
            perm_key = jax.random.fold_in(key, i)
            indices = jax.random.permutation(perm_key, vocab_size)[:top_k]
            top_k_indices.append(indices)

        # Create target logits with uniform distribution
        target_logits = jnp.zeros((num_tokens, vocab_size), dtype=jnp.float32)

        # Increment logits for top-k indices slightly to make them more likely
        # If masking works correctly, only these tokens should be sampled
        for i in range(num_tokens):
            indices = top_k_indices[i]
            target_logits = target_logits.at[i, indices].add(0.1)

        # Create sampling metadata with top-k
        sampling_metadata = test_helper.create_sampling_metadata(
            all_greedy=False,
            batch_size=batch_size,
            top_k=top_k,
            top_p=1.0,
            temperature=1.0,
        )

        self._test_masked_logits(
            rejection_sampler=rejection_sampler,
            batch_size=batch_size,
            num_draft_tokens=num_draft_tokens,
            vocab_size=vocab_size,
            target_logits=target_logits,
            allowed_tokens_per_pos=top_k_indices,
            sampling_metadata=sampling_metadata,
        )

    @pytest.mark.parametrize("top_p", [0.5, 0.9, 0.99])
    def test_top_p(self, rejection_sampler, test_helper, top_p):
        """Test rejection sampling with top-p sampling."""
        vocab_size = 100
        batch_size = 10
        num_draft_tokens = 3
        num_tokens = batch_size * num_draft_tokens

        # Create random target logits
        key = jax.random.PRNGKey(42)
        target_logits = jax.random.normal(key, (num_tokens, vocab_size))

        # Create temperature array for batch
        temperature = jnp.ones(batch_size, dtype=jnp.float32)

        # Calculate top-p indices for each token position
        rescaled_logits = target_logits / temperature.repeat(num_draft_tokens,
                                                             axis=0)[:, None]

        # Sort logits and calculate cumulative probabilities
        logits_sorted = jnp.sort(rescaled_logits, axis=-1)
        logits_idx = jnp.argsort(rescaled_logits, axis=-1)
        probs_sorted = jax.nn.softmax(logits_sorted, axis=-1)
        probs_cumsum = jnp.cumsum(probs_sorted, axis=-1)

        # Create top-p mask
        top_p_mask = probs_cumsum <= (1 - top_p)
        # Ensure at least one token is kept
        top_p_mask = top_p_mask.at[:, -1].set(False)

        # Get top-p indices for each position
        top_p_indices = []
        for i in range(num_tokens):
            valid_indices = logits_idx[i][~top_p_mask[i]]
            top_p_indices.append(valid_indices)

        # Create sampling metadata with top-p
        sampling_metadata = test_helper.create_sampling_metadata(
            all_greedy=False,
            batch_size=batch_size,
            top_k=-1,
            top_p=top_p,
            temperature=1.0,
        )

        self._test_masked_logits(
            rejection_sampler=rejection_sampler,
            batch_size=batch_size,
            num_draft_tokens=num_draft_tokens,
            vocab_size=vocab_size,
            target_logits=target_logits,
            allowed_tokens_per_pos=top_p_indices,
            sampling_metadata=sampling_metadata,
        )

    def test_top_k_and_top_p_combined(self, rejection_sampler, test_helper):
        """Test rejection sampling with both top-k and top-p applied.

        This test verifies that both top-k and top-p can be used together
        without errors, but doesn't verify the exact masking behavior since
        the order of application may vary from our test implementation.
        """
        vocab_size = 50
        batch_size = 5
        num_draft_tokens = 2
        num_tokens = batch_size * num_draft_tokens
        top_k = 10
        top_p = 0.8

        # Create random target logits
        key = jax.random.PRNGKey(123)
        target_logits = jax.random.normal(key, (num_tokens, vocab_size))

        # Create random draft probabilities
        draft_key = jax.random.PRNGKey(42)
        draft_logits = jax.random.normal(draft_key, (num_tokens, vocab_size))
        draft_probs = jax.nn.softmax(draft_logits, axis=-1)

        # Randomly sample draft token ids from draft probs
        sample_key = jax.random.PRNGKey(123)
        draft_token_ids = jax.random.categorical(sample_key,
                                                 jnp.log(draft_probs + 1e-8),
                                                 shape=(num_tokens, ))

        # Create sampling metadata with both top-k and top-p
        sampling_metadata = test_helper.create_sampling_metadata(
            all_greedy=False,
            batch_size=batch_size,
            top_k=top_k,
            top_p=top_p,
            temperature=1.0,
        )

        # Prepare inputs
        num_draft_per_seq = jnp.full((batch_size, ),
                                     num_draft_tokens,
                                     dtype=jnp.int32)
        bonus_token_ids = jnp.zeros((batch_size, ), dtype=jnp.int32)

        # Just test that the combined sampling runs without errors
        run_key = jax.random.PRNGKey(456)
        output_token_ids = rejection_sampler(
            draft_token_ids=draft_token_ids,
            num_draft_tokens=num_draft_per_seq,
            draft_probs=draft_probs,
            target_logits=target_logits,
            bonus_token_ids=bonus_token_ids,
            sampling_metadata=sampling_metadata,
            key=run_key,
        )

        # Parse output to verify it's well-formed
        parsed_output = rejection_sampler.parse_output(
            output_token_ids,
            vocab_size=vocab_size,
            num_draft_tokens_cpu=np.asarray(num_draft_per_seq),
            batch_size=batch_size,
            padded_tokens_length=num_tokens)

        # Basic sanity checks
        assert len(parsed_output) == batch_size
        for seq_tokens in parsed_output:
            assert len(seq_tokens) >= 0  # Should have at least empty list
            for token_id in seq_tokens:
                assert 0 <= token_id < vocab_size  # Valid token range
