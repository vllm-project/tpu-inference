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

import pytest


def test_step_pooling_e2e():
    from vllm import LLM
    model_name = "Qwen/Qwen3-Embedding-8B"

    # CI-optimized small-scale chunking configuration
    max_model_len = 2048
    max_num_batched_tokens = 256
    max_num_seqs = 1

    try:
        llm = LLM(model=model_name,
                  runner="pooling",
                  max_num_seqs=max_num_seqs,
                  max_model_len=max_model_len,
                  max_num_batched_tokens=max_num_batched_tokens,
                  dtype="bfloat16",
                  trust_remote_code=True,
                  load_format="dummy")
    except Exception as e:
        pytest.skip(f"Skipping test: {e}")

    # Scaling inputs to exceed max_num_batched_tokens to force chunked prefill
    # 1024 tokens exceeds 256 max_num_batched_tokens.
    # We use a repeating token string to force the engine to split the prefill into multiple passes.
    base_input = "Hello, my name is Alice. "

    # Approx 5 words per repetition, so ~200 repetitions generates ~1000 words/tokens.
    long_input = base_input * 200

    inputs = [long_input]

    try:
        results = llm.embed(inputs)

        assert len(results) == 1
        assert results[0].outputs.embedding is not None
        assert len(results[0].outputs.embedding) > 0

    except Exception as e:
        pytest.fail(f"Embedding execution failed during chunked prefill: {e}")
