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

from __future__ import annotations

import json
import math

import pytest
from vllm import LLM, SamplingParams
from vllm.sampling_params import StructuredOutputsParams

CHOICES = ["Positive", "Negative"]
DIGIT_CHOICES = [str(digit) for digit in range(10)]
REVIEW_SCHEMA = {
    "type": "object",
    "properties": {
        "sentiment": {
            "type": "string",
            "enum": CHOICES,
        },
        "score": {
            "type": "integer",
            "minimum": 1,
            "maximum": 5,
        },
    },
    "required": ["sentiment", "score"],
    "additionalProperties": False,
}


@pytest.fixture(scope="module")
def llm():
    """Create one LLM for all structured-sampling combination tests."""
    return LLM(
        model="meta-llama/Llama-3.2-1B-Instruct",
        max_model_len=1024,
        max_num_seqs=4,
        enable_prefix_caching=False,
        logprobs_mode="processed_logprobs",
    )


def _choice_params(**kwargs) -> SamplingParams:
    return SamplingParams(
        structured_outputs=StructuredOutputsParams(choice=CHOICES),
        **kwargs,
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
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def test_structured_choice_with_temperature_zero_is_deterministic(llm: LLM):
    """Greedy sampling must remain deterministic and correct after grammar masking."""
    prompt = _apply_template(
        llm,
        "You are a sentiment classifier. Respond only with Positive or Negative.",
        "Classify the sentiment: TPU inference is wonderful!",
    )
    sampling_params = _choice_params(temperature=0, max_tokens=5)

    outputs1 = llm.generate([prompt], sampling_params)
    outputs2 = llm.generate([prompt], sampling_params)
    text1 = outputs1[0].outputs[0].text
    text2 = outputs2[0].outputs[0].text

    assert text1 in CHOICES
    assert text2 == text1


def test_structured_choice_with_high_temperature_stays_constrained(llm: LLM):
    """Random sampling must never select tokens outside the choice grammar."""
    prompt = _apply_template(
        llm,
        "You are a sentiment classifier. Respond only with Positive or Negative.",
        "Classify the sentiment: The experience was complicated.",
    )
    sampling_params = _choice_params(
        temperature=1.5,
        top_k=-1,
        max_tokens=15,
    )

    outputs = llm.generate([prompt] * 10, sampling_params)
    results = [out.outputs[0].text for out in outputs]

    assert all(text in CHOICES for text in results), results


def test_json_schema_with_top_k_and_top_p(llm: LLM):
    """top_k and top_p can be combined with JSON-schema constraints."""
    prompt = _apply_template(
        llm,
        "You are a reviewer. Review the statement and return its sentiment (Positive/Negative) and a score (1 to 5).",
        "Review this statement and return its sentiment and a score from 1 to 5: TPU inference is fast and reliable.",
    )
    sampling_params = SamplingParams(
        temperature=0.8,
        top_k=50,
        top_p=0.9,
        max_tokens=32,
        structured_outputs=StructuredOutputsParams(json=REVIEW_SCHEMA),
    )

    outputs = llm.generate([prompt] * 5, sampling_params)
    for out in outputs:
        text = out.outputs[0].text
        try:
            result = json.loads(text)
        except json.JSONDecodeError as e:
            pytest.fail(
                f"Failed to parse JSON from output: {text!r}. Error: {e}")

        assert set(result) == {"sentiment", "score"}
        assert result["sentiment"] in CHOICES
        assert isinstance(result["score"], int)
        assert not isinstance(result["score"], bool)
        assert 1 <= result["score"] <= 5


@pytest.mark.parametrize(
    "filter_params",
    [
        pytest.param({
            "top_k": 1,
            "top_p": 1.0
        }, id="top-k"),
        pytest.param({
            "top_k": 0,
            "top_p": 1e-6
        }, id="top-p"),
    ],
)
def test_structured_choice_sampling_filters_are_effective(
        llm: LLM, filter_params: dict[str, float | int]):
    """Restrictive top-k/top-p filters must collapse the candidate set."""
    prompt = _apply_template(
        llm,
        "Choose one decimal digit and respond with only that digit.",
        "Choose a digit from 0 through 9.",
    )
    common_params = {
        "temperature": 1.0,
        "max_tokens": 3,
        "logprobs": len(DIGIT_CHOICES),
        "structured_outputs": StructuredOutputsParams(choice=DIGIT_CHOICES),
    }

    unrestricted = llm.generate(
        [prompt],
        SamplingParams(top_k=0, top_p=1.0, **common_params),
    )[0].outputs[0]
    restricted = llm.generate(
        [prompt],
        SamplingParams(**filter_params, **common_params),
    )[0].outputs[0]

    assert unrestricted.text in DIGIT_CHOICES
    assert restricted.text in DIGIT_CHOICES
    assert unrestricted.logprobs is not None
    assert restricted.logprobs is not None

    unrestricted_first = unrestricted.logprobs[0]
    restricted_first = restricted.logprobs[0]
    unrestricted_finite = [
        entry.logprob for entry in unrestricted_first.values()
        if math.isfinite(entry.logprob) and entry.logprob > -1e6
    ]

    assert len(unrestricted_finite) > 1, (
        "The unrestricted control must expose multiple grammar-valid candidates"
    )
    restricted_token_id = restricted.token_ids[0]
    assert restricted_token_id in restricted_first
    restricted_probs = [
        math.exp(entry.logprob) for entry in restricted_first.values()
        if math.isfinite(entry.logprob) and entry.logprob > -1e6
    ]
    assert sum(restricted_probs) == pytest.approx(1.0, abs=1e-5)


def test_structured_output_logprobs_include_selected_tokens(llm: LLM):
    """Every grammar-approved sampled token must have a finite logprob."""
    prompt = _apply_template(
        llm,
        "You are a sentiment classifier. Respond only with Positive or Negative.",
        "Classify the sentiment: TPU inference is wonderful!",
    )
    sampling_params = _choice_params(
        temperature=0.7,
        top_p=0.9,
        max_tokens=5,
        logprobs=3,
    )

    output = llm.generate([prompt], sampling_params)[0].outputs[0]

    assert output.text in CHOICES
    assert output.logprobs is not None
    assert len(output.logprobs) == len(output.token_ids)

    for token_id, token_logprobs in zip(output.token_ids, output.logprobs):
        assert token_id in token_logprobs, (
            f"Selected token {token_id} is missing from returned logprobs")
        selected_logprob = token_logprobs[token_id].logprob
        assert math.isfinite(selected_logprob)
        assert selected_logprob <= 0
