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
from vllm import LLM, SamplingParams

# Global constant replacing class attribute
SYSTEM_CONTEXT = (
    "You are a helpful and precise assistant. "
    "The quick brown fox jumps over the lazy dog. "
    "Artificial Intelligence and TPU acceleration make inference incredibly fast."
)


@pytest.fixture(scope="module")
def tpu_llm():
    """Initialize TPU vLLM instance with Prefix Caching enabled."""
    return LLM(
        model='meta-llama/Llama-3.2-1B-Instruct',
        max_model_len=1024,
        max_num_seqs=4,
        enable_prefix_caching=True,
    )


def _apply_template(llm: LLM, system_msg: str, user_msg: str) -> str:
    """Helper to apply proper Chat Template for Llama-3-Instruct models."""
    tokenizer = llm.get_tokenizer()
    messages = [{
        "role": "system",
        "content": system_msg
    }, {
        "role": "user",
        "content": user_msg
    }]
    return tokenizer.apply_chat_template(messages,
                                         tokenize=False,
                                         add_generation_prompt=True)


def test_cache_hit_with_deterministic_sampling(tpu_llm: LLM):
    """Verify that greedy sampling (temp=0) remains deterministic after a cache hit."""
    prompt1 = _apply_template(tpu_llm, SYSTEM_CONTEXT,
                              "What is 5 + 5? Answer with just the number:")
    prompt2 = _apply_template(tpu_llm, SYSTEM_CONTEXT,
                              "What is 5 + 5? Answer with just the number:")

    sampling_params = SamplingParams(temperature=0, max_tokens=5)

    outputs1 = tpu_llm.generate([prompt1], sampling_params)
    outputs2 = tpu_llm.generate([prompt2], sampling_params)

    text1 = outputs1[0].outputs[0].text.strip()
    text2 = outputs2[0].outputs[0].text.strip()

    assert text1 == text2, f"Cache hit altered greedy output: '{text1}' vs '{text2}'"


def test_cache_hit_with_random_sampling(tpu_llm: LLM):
    """Verify that random sampling (temp > 0) maintains diversity on cache hits."""
    prompt = _apply_template(
        tpu_llm, SYSTEM_CONTEXT,
        "Write a completely random and creative single word:")
    sampling_params = SamplingParams(temperature=1.5, top_k=50, max_tokens=5)

    tpu_llm.generate([prompt], sampling_params)

    unique_outputs = set()
    for _ in range(5):
        outputs = tpu_llm.generate([prompt], sampling_params)
        unique_outputs.add(outputs[0].outputs[0].text.strip())

    assert len(
        unique_outputs
    ) > 1, "Random sampling failed to produce varied outputs on cached prompt."


def test_mixed_sampling_params_on_cached_prefix(tpu_llm: LLM):
    """Verify that different sampling states are properly isolated when sharing the same cached prefix."""
    prompt_greedy = _apply_template(
        tpu_llm, SYSTEM_CONTEXT,
        "Task A: Reply with the exact word 'APPLE' and nothing else.")
    prompt_random = _apply_template(
        tpu_llm, SYSTEM_CONTEXT,
        "Task B: Write a highly creative and random poem line.")

    params_greedy = SamplingParams(temperature=0, max_tokens=15)
    params_random = SamplingParams(temperature=1.2, top_p=0.9, max_tokens=15)

    # Execute mixed tasks using the same prefix cache
    outputs_random = tpu_llm.generate([prompt_random], params_random)
    outputs_greedy = tpu_llm.generate([prompt_greedy], params_greedy)

    greedy_text = outputs_greedy[0].outputs[0].text.upper()
    random_text = outputs_random[0].outputs[0].text.upper()
    assert "APPLE" in greedy_text, f"Greedy task failed. Output was: '{outputs_greedy[0].outputs[0].text}'"
    assert greedy_text != random_text, f"Random task failed. Output was: '{outputs_random[0].outputs[0].text}'"


def test_logprobs_with_prefix_caching(tpu_llm: LLM):
    """Verify that logprobs are correctly calculated and returned during a cache hit."""
    prompt = _apply_template(tpu_llm, SYSTEM_CONTEXT,
                             "Explain AI in three words:")
    sampling_params = SamplingParams(temperature=0, max_tokens=5, logprobs=3)

    tpu_llm.generate([prompt], sampling_params)

    outputs = tpu_llm.generate([prompt], sampling_params)
    output = outputs[0].outputs[0]

    assert output.logprobs is not None, "Logprobs missing on cache hit."
    for token_logprobs in output.logprobs:
        for token_id, logprob_obj in token_logprobs.items():
            assert logprob_obj.logprob <= 0, f"Invalid logprob value: {logprob_obj.logprob}"
