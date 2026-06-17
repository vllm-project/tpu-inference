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

# Tests for Processed Logprobs.
#
# Correctness tests verify that:
#   1. Processed logprobs (post temperature/top-k/top-p scaling) are returned
#      successfully, and all returned logprobs are valid (non-positive values).
#   2. Returned logprobs correctly reflect and scale with the temperature parameter.

from __future__ import annotations

import os

import pytest
from vllm import LLM, SamplingParams

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"
MAX_MODEL_LEN = 1024
MAX_NUM_SEQS = 32
MAX_TOKENS_DEFAULT = 32

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def llm():
    """Initializes LLM instance with processed_logprobs mode enabled."""
    os.environ.setdefault("SKIP_JAX_PRECOMPILE", "0")
    engine = LLM(
        model=MODEL_NAME,
        max_model_len=MAX_MODEL_LEN,
        max_num_seqs=MAX_NUM_SEQS,
        logprobs_mode="processed_logprobs",
    )
    yield engine
    engine.llm_engine.engine_core.shutdown()


class TestProcessedLogprobs:
    """Verifies that processed_logprobs returns logprobs post temperature/top-k/top-p scaling."""

    def test_processed_logprobs_returned_and_valid(self, llm: LLM):
        """Verifies that valid logprobs are returned and all are <= 0."""
        prompt = "The capital of France is"
        sampling_params = SamplingParams(temperature=0.8,
                                         max_tokens=5,
                                         logprobs=5)

        outputs = llm.generate([prompt], sampling_params)
        output = outputs[0].outputs[0]

        assert output.logprobs is not None, "logprobs should be returned"
        assert len(output.logprobs) > 0

        for token_logprobs in output.logprobs:
            assert len(token_logprobs) <= 5
            for _, logprob_obj in token_logprobs.items():
                assert logprob_obj.logprob <= 0, (
                    f"Logprob must be non-positive, got {logprob_obj.logprob}")

    def test_processed_logprobs_reflect_temperature(self, llm: LLM):
        """Verifies that the returned logprobs are scaled by temperature."""
        prompt = "Explain quantum physics in one short sentence:"

        # High temperature distributes probability more evenly
        sp_high_temp = SamplingParams(temperature=2.0,
                                      max_tokens=1,
                                      logprobs=5)
        out_high = llm.generate([prompt], sp_high_temp)[0].outputs[0]

        # Low temperature pushes the highest probability closer to 1 (logprob closer to 0)
        sp_low_temp = SamplingParams(temperature=0.1, max_tokens=1, logprobs=5)
        out_low = llm.generate([prompt], sp_low_temp)[0].outputs[0]

        high_logprobs = [obj.logprob for obj in out_high.logprobs[0].values()]
        low_logprobs = [obj.logprob for obj in out_low.logprobs[0].values()]

        # The peak token under low temperature should have a much higher probability
        # (logprob closer to 0) than the peak token under high temperature.
        assert max(low_logprobs) > max(high_logprobs), (
            f"Expected peak logprob at low temp ({max(low_logprobs)}) "
            f"to be greater than at high temp ({max(high_logprobs)})")
