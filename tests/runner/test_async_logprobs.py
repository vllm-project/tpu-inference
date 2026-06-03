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

# Tests for Async Logprobs with Lazy Materialization.
#
# Correctness tests verify that:
#   1. Requesting logprobs works seamlessly, exercising the async background copy
#      and lazy materialization pipeline.
#   2. Materialized logprobs are valid (non-positive values) and correctly formatted.

from __future__ import annotations

import os
import time

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
    """Create a shared LLM instance to test async logprobs."""
    os.environ.setdefault("SKIP_JAX_PRECOMPILE", "0")
    engine = LLM(
        model=MODEL_NAME,
        max_model_len=MAX_MODEL_LEN,
        max_num_seqs=MAX_NUM_SEQS,
    )
    yield engine
    del engine
    time.sleep(5)


class TestAsyncLogprobs:
    """Verify that Async Logprobs with Lazy Materialization works correctly."""

    def test_async_logprobs_lazy_materialization(self, llm: LLM):
        """Verify that requesting logprobs works seamlessly, exercising the lazy materialization pipeline."""
        prompt = "The water cycle is"
        sampling_params = SamplingParams(temperature=0.0,
                                         max_tokens=10,
                                         logprobs=5)

        outputs = llm.generate([prompt], sampling_params)
        output = outputs[0].outputs[0]

        # Verify that we actually got the logprobs successfully on host,
        # confirming that async background copy and lazy materialization resolved without issues.
        assert output.logprobs is not None, (
            "Async logprobs should have materialized successfully")
        assert len(output.logprobs) == 10, (
            "Should have 10 generated tokens' logprobs")

        # Check validity of materialized logprobs
        for token_logprobs in output.logprobs:
            assert token_logprobs is not None
            assert len(token_logprobs) <= 5
            for _, logprob_obj in token_logprobs.items():
                assert logprob_obj.logprob <= 0, (
                    f"Logprob must be non-positive, got {logprob_obj.logprob}")
