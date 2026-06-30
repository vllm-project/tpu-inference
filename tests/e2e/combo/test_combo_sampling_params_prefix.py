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

# A common system context that is long enough to be cached as a prefix.
# Prefix caching in vLLM splits the prompt into blocks and caches blocks
# that are identical across requests.
SYSTEM_CONTEXT = (
    "You are a helpful and precise assistant. "
    "The quick brown fox jumps over the lazy dog. "
    "Artificial Intelligence and TPU acceleration make inference incredibly fast."
)


@pytest.fixture(scope="module")
def tpu_llm():
    """Initialize TPU vLLM instance with Prefix Caching enabled."""
    # Prefix caching must be explicitly enabled in the LLM engine config.
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
    # Two identical prompts sharing the same system prefix and user message.
    prompt1 = _apply_template(tpu_llm, SYSTEM_CONTEXT,
                              "What is 5 + 5? Answer with just the number:")
    prompt2 = _apply_template(tpu_llm, SYSTEM_CONTEXT,
                              "What is 5 + 5? Answer with just the number:")

    sampling_params = SamplingParams(temperature=0, max_tokens=5)

    # First generation processes the prompt and populates the prefix/prompt cache.
    outputs1 = tpu_llm.generate([prompt1], sampling_params)
    # Second generation hits the cache for the prefix and generated content.
    outputs2 = tpu_llm.generate([prompt2], sampling_params)

    text1 = outputs1[0].outputs[0].text.strip()
    text2 = outputs2[0].outputs[0].text.strip()

    # The output should remain exactly identical (deterministic) even with cache hits.
    assert text1 == text2, f"Cache hit altered greedy output: '{text1}' vs '{text2}'"


def test_cache_hit_with_random_sampling(tpu_llm: LLM):
    """Verify that random sampling (temp > 0) maintains diversity on cache hits."""
    prompt = _apply_template(
        tpu_llm, SYSTEM_CONTEXT,
        "Write a completely random and creative single word:")
    # High temperature triggers random sampling.
    sampling_params = SamplingParams(temperature=1.5, top_k=50, max_tokens=5)

    # Prime the prefix cache by running the prompt once.
    tpu_llm.generate([prompt], sampling_params)

    unique_outputs = set()
    # Perform multiple generations on the cached prompt.
    for _ in range(5):
        outputs = tpu_llm.generate([prompt], sampling_params)
        unique_outputs.add(outputs[0].outputs[0].text.strip())

    # Assert that even though the prefix and prompt are cached,
    # the random sampling is still properly seeding and generating diverse outputs.
    assert len(
        unique_outputs
    ) > 1, "Random sampling failed to produce varied outputs on cached prompt."


def test_mixed_sampling_params_on_cached_prefix(tpu_llm: LLM):
    """Verify that different sampling states are properly isolated when sharing the same cached prefix."""
    # Prompts share the same SYSTEM_CONTEXT (prefix) but have different suffixes (user requests).
    prompt_greedy = _apply_template(
        tpu_llm, SYSTEM_CONTEXT,
        "Task A: Reply with the exact word 'APPLE' and nothing else.")
    prompt_random = _apply_template(
        tpu_llm, SYSTEM_CONTEXT,
        "Task B: Write a highly creative and random poem line.")

    params_greedy = SamplingParams(temperature=0, max_tokens=15)
    params_random = SamplingParams(temperature=1.2, top_p=0.9, max_tokens=15)

    # Execute the random task first. This processes SYSTEM_CONTEXT and caches it.
    outputs_random = tpu_llm.generate([prompt_random], params_random)
    # Execute the greedy task. It should hit the cached prefix (SYSTEM_CONTEXT)
    # but process the unique greedy prompt suffix without being contaminated by the
    # random sampling parameters or states of the previous run.
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

    # Prime the cache.
    tpu_llm.generate([prompt], sampling_params)

    # Request the same prompt again (cache hit).
    outputs = tpu_llm.generate([prompt], sampling_params)
    output = outputs[0].outputs[0]

    # Verify that logprobs metadata is still present and valid when served from cache.
    assert output.logprobs is not None, "Logprobs missing on cache hit."
    for token_logprobs in output.logprobs:
        for token_id, logprob_obj in token_logprobs.items():
            assert logprob_obj.logprob <= 0, f"Invalid logprob value: {logprob_obj.logprob}"
