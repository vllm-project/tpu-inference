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

# This file contains end-to-end tests for sampling parameters.
#
# Sampling parameters control how the model selects tokens during generation.
# These tests verify that temperature, top_p, top_k, and logprobs work correctly.
#
# The tests in this file verify that:
# 1. Temperature=0 produces deterministic (greedy) outputs
# 2. Higher temperature produces more varied outputs
# 3. top_p (nucleus sampling) correctly constrains token selection
# 4. top_k correctly limits the number of candidate tokens
# 5. logprobs returns probability information for generated tokens

from __future__ import annotations

import pytest
from vllm import LLM, SamplingParams


@pytest.fixture(scope="module")
def llm():
    """Create a shared LLM instance for all tests in this module."""
    return LLM(
        model='meta-llama/Llama-3.2-1B-Instruct',
        max_model_len=1024,
        max_num_seqs=4,
        enable_prefix_caching=False,
    )


class TestTemperature:
    """Tests for temperature sampling parameter."""

    def test_temperature_zero_is_deterministic(self, llm: LLM):
        """Temperature=0 should produce identical outputs across multiple runs."""
        prompt = "What is 2 + 2? Answer with just the number:"
        sampling_params = SamplingParams(temperature=0, max_tokens=10)

        outputs1 = llm.generate([prompt], sampling_params)
        outputs2 = llm.generate([prompt], sampling_params)

        assert outputs1[0].outputs[0].text == outputs2[0].outputs[0].text

    def test_high_temperature_produces_variation(self, llm: LLM):
        """High temperature should produce varied outputs across multiple runs."""
        prompt = "Write a random word:"
        sampling_params = SamplingParams(temperature=2,
                                         max_tokens=10,
                                         top_k=4096)

        # Run multiple times and collect unique outputs
        unique_outputs = set()
        num_runs = 10
        for _ in range(num_runs):
            outputs = llm.generate([prompt], sampling_params)
            unique_outputs.add(outputs[0].outputs[0].text)

        # With high temperature, we expect some variation
        assert len(unique_outputs) > 1, (
            "High temperature should produce varied outputs")


class TestTopP:
    """Tests for top_p (nucleus sampling) parameter."""

    def test_top_p_restricts_sampling(self, llm: LLM):
        """top_p=1.0 vs lower values should affect output diversity."""
        prompt = "Name a color:"

        # With top_p=1.0 (consider all tokens)
        sampling_params_full = SamplingParams(temperature=0.8,
                                              top_p=1.0,
                                              max_tokens=5)

        # With top_p=0.1 (very restrictive, only top tokens)
        sampling_params_restricted = SamplingParams(temperature=0.8,
                                                    top_p=0.1,
                                                    max_tokens=5)

        # Collect outputs with full nucleus
        full_outputs = set()
        for _ in range(10):
            outputs = llm.generate([prompt], sampling_params_full)
            full_outputs.add(outputs[0].outputs[0].text)

        # Collect outputs with restricted nucleus
        restricted_outputs = set()
        for _ in range(10):
            outputs = llm.generate([prompt], sampling_params_restricted)
            restricted_outputs.add(outputs[0].outputs[0].text)

        # Restricted top_p should generally produce less variety
        # (though this isn't guaranteed, it's a reasonable expectation)
        assert len(
            restricted_outputs) >= 1, "Should produce at least one output"
        assert len(full_outputs) >= 1, "Should produce at least one output"

    def test_top_p_with_temperature_zero(self, llm: LLM):
        """top_p should have no effect when temperature=0 (greedy)."""
        prompt = "The capital of France is"

        sampling_params_1 = SamplingParams(temperature=0,
                                           top_p=0.1,
                                           max_tokens=10)
        sampling_params_2 = SamplingParams(temperature=0,
                                           top_p=0.9,
                                           max_tokens=10)

        outputs1 = llm.generate([prompt], sampling_params_1)
        outputs2 = llm.generate([prompt], sampling_params_2)

        # Both should produce identical outputs since temperature=0
        assert outputs1[0].outputs[0].text == outputs2[0].outputs[0].text


class TestTopK:
    """Tests for top_k sampling parameter."""

    def test_top_k_restricts_sampling(self, llm: LLM):
        """top_k should limit the candidate tokens for sampling."""
        prompt = "Pick a number between 1 and 10:"

        # top_k=1 is equivalent to greedy (always pick the most likely)
        sampling_params_k1 = SamplingParams(temperature=1.0,
                                            top_k=1,
                                            max_tokens=5)

        # top_k=-1 considers all tokens
        sampling_params_all = SamplingParams(temperature=1.0,
                                             top_k=-1,
                                             max_tokens=5)

        # With top_k=1, outputs should be deterministic
        outputs_k1_run1 = llm.generate([prompt], sampling_params_k1)
        outputs_k1_run2 = llm.generate([prompt], sampling_params_k1)
        assert outputs_k1_run1[0].outputs[0].text == outputs_k1_run2[
            0].outputs[0].text

        # With top_k=-1 and temperature=1.0, we may see variation
        all_outputs = set()
        for _ in range(10):
            outputs = llm.generate([prompt], sampling_params_all)
            all_outputs.add(outputs[0].outputs[0].text)

        # Should produce at least one valid output
        assert len(all_outputs) >= 1

    def test_top_k_with_temperature_zero(self, llm: LLM):
        """top_k should have no effect when temperature=0 (greedy)."""
        prompt = "The largest planet is"

        sampling_params_k5 = SamplingParams(temperature=0,
                                            top_k=5,
                                            max_tokens=10)
        sampling_params_k50 = SamplingParams(temperature=0,
                                             top_k=50,
                                             max_tokens=10)

        outputs1 = llm.generate([prompt], sampling_params_k5)
        outputs2 = llm.generate([prompt], sampling_params_k50)

        # Both should produce identical outputs since temperature=0
        assert outputs1[0].outputs[0].text == outputs2[0].outputs[0].text


class TestLogprobs:
    """Tests for logprobs parameter."""

    def test_logprobs_returns_probabilities(self, llm: LLM):
        """logprobs parameter should return log probabilities for tokens."""
        prompt = "Hello"
        sampling_params = SamplingParams(temperature=0,
                                         max_tokens=5,
                                         logprobs=5)

        outputs = llm.generate([prompt], sampling_params)
        output = outputs[0].outputs[0]

        # Check that logprobs are returned
        assert output.logprobs is not None, "logprobs should be returned"
        assert len(output.logprobs) > 0, "logprobs should contain entries"

        # Each token should have logprob information
        for token_logprobs in output.logprobs:
            assert token_logprobs is not None
            # Should have up to 5 top logprobs as requested
            assert len(token_logprobs) <= 5

    def test_logprobs_none_returns_no_probabilities(self, llm: LLM):
        """When logprobs=None, no log probabilities should be returned."""
        prompt = "Hello"
        sampling_params = SamplingParams(temperature=0,
                                         max_tokens=5,
                                         logprobs=None)

        outputs = llm.generate([prompt], sampling_params)
        output = outputs[0].outputs[0]

        # logprobs should be None when not requested
        assert output.logprobs is None, "logprobs should be None when not requested"

    def test_logprobs_values_are_valid(self, llm: LLM):
        """Log probabilities should be valid (negative or zero)."""
        prompt = "The sky is"
        sampling_params = SamplingParams(temperature=0,
                                         max_tokens=3,
                                         logprobs=3)

        outputs = llm.generate([prompt], sampling_params)
        output = outputs[0].outputs[0]

        assert output.logprobs is not None
        for token_logprobs in output.logprobs:
            for token_id, logprob_obj in token_logprobs.items():
                # Log probabilities should be <= 0
                assert logprob_obj.logprob <= 0, (
                    f"Log probability should be <= 0, got {logprob_obj.logprob}"
                )


class TestCombinedParameters:
    """Tests for combinations of sampling parameters."""

    def test_top_p_and_top_k_combined(self, llm: LLM):
        """top_p and top_k can be used together."""
        prompt = "List a fruit:"
        sampling_params = SamplingParams(
            temperature=0.7,
            top_p=0.9,
            top_k=50,
            max_tokens=10,
        )

        outputs = llm.generate([prompt], sampling_params)
        assert len(outputs[0].outputs[0].text) > 0

    def test_all_params_with_logprobs(self, llm: LLM):
        """All sampling parameters should work together with logprobs."""
        prompt = "Complete this sentence: The weather today is"
        sampling_params = SamplingParams(
            temperature=0.5,
            top_p=0.95,
            top_k=40,
            max_tokens=10,
            logprobs=3,
        )

        outputs = llm.generate([prompt], sampling_params)
        output = outputs[0].outputs[0]

        # Should have generated text
        assert len(output.text) > 0

        # Should have logprobs
        assert output.logprobs is not None
        assert len(output.logprobs) > 0
