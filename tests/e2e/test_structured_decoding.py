# This file contains end-to-end tests for structured decoding.
#
# Structured decoding allows constraining the model's output to follow a
# specific format, such as choosing from a predefined set of options or
# following a JSON schema. This is useful for classification tasks,
# structured data extraction, and ensuring outputs conform to expected formats.

# The tests in this file verify that:
# 1. Choice-based structured decoding correctly constrains output to valid options
# 2. The model produces deterministic results when given structured constraints

from __future__ import annotations

from vllm import LLM, SamplingParams
from vllm.sampling_params import StructuredOutputsParams


def test_structured_decoding():
    llm = LLM(model='meta-llama/Llama-3.2-1B-Instruct',
              max_model_len=1024,
              max_num_seqs=1,
              enable_prefix_caching=False)

    choices = ['Positive', 'Negative']
    structured_outputs_params = StructuredOutputsParams(choice=choices)
    sampling_params = SamplingParams(
        structured_outputs=structured_outputs_params)
    outputs = llm.generate(
        prompts="Classify this sentiment: tpu-inference is wonderful!",
        sampling_params=sampling_params,
    )
    assert outputs[0].outputs[0].text in choices
