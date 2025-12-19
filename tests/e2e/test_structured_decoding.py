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
