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

import os
import time

import pytest
import vllm
from vllm.lora.request import LoRARequest

TP = [2] if os.environ.get("USE_V6E8_QUEUE", False) else [1]


@pytest.mark.parametrize("tp", TP)
def test_lora_performance(tp):
    prompt = "What is 1+1? \n"
    llm_without_lora = vllm.LLM(
        model="Qwen/Qwen2.5-3B-Instruct",
        max_model_len=256,
        max_num_batched_tokens=64,
        max_num_seqs=8,
        tensor_parallel_size=tp,
    )
    start_time = time.time()
    llm_without_lora.generate(
        prompt,
        sampling_params=vllm.SamplingParams(max_tokens=16, temperature=0),
    )[0].outputs[0].text
    base_time = time.time() - start_time

    del llm_without_lora
    # Waiting for TPUs to be released
    time.sleep(10)

    llm_with_lora = vllm.LLM(model="Qwen/Qwen2.5-3B-Instruct",
                             max_model_len=256,
                             max_num_batched_tokens=64,
                             max_num_seqs=8,
                             tensor_parallel_size=tp,
                             enable_lora=True,
                             max_loras=1,
                             max_lora_rank=8)
    lora_request = LoRARequest(
        "lora_adapter_2", 2,
        "Username6568/Qwen2.5-3B-Instruct-1_plus_1_equals_2_adapter")
    start_time = time.time()
    llm_with_lora.generate(prompt,
                           sampling_params=vllm.SamplingParams(max_tokens=16,
                                                               temperature=0),
                           lora_request=lora_request)[0].outputs[0].text
    lora_time = time.time() - start_time
    print(f"Base time: {base_time}, LoRA time: {lora_time}")
    assert (base_time /
            lora_time) < 8, f"Base time: {base_time}, LoRA time: {lora_time}"

    del llm_with_lora
