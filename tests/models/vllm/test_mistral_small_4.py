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

import os
import pytest
from vllm import LLM, SamplingParams
from tpu_inference import tpu_info

MODEL_ID = "mistralai/Mistral-Small-4-119B-2603"

def set_env_vars():
    """Sets the environment variables required for running Mistral Small 4 on TPU."""
    os.environ['MODEL_IMPL_TYPE'] = 'vllm'
    os.environ['SKIP_JAX_PRECOMPILE'] = '1'  # Set to '1' to skip expensive multi-shape precompilation in unit tests
    os.environ['VLLM_TPU_PATCH_MM_EMBEDDINGS'] = '1'
    os.environ['VLLM_DISABLE_SHARED_EXPERTS_STREAM'] = '0'
    os.environ['MOE_REQUANTIZE_BLOCK_SIZE'] = '512'
    os.environ['NEW_MODEL_DESIGN'] = '1'
    os.environ['TPU_BACKEND_TYPE'] = 'jax'
    os.environ['VLLM_USE_V1'] = '0'  # Force V0 engine for diagnostics


def get_num_tpu_cores() -> int:
    """Safely get the number of TPU cores without initializing JAX."""
    try:
        return tpu_info.get_num_chips() * tpu_info.get_num_cores_per_chip()
    except Exception:
        return 0

num_tpu_cores = get_num_tpu_cores()
run_correctness = os.environ.get("RUN_MISTRAL_SMALL_4_CORRECTNESS_TEST") == "1"

@pytest.mark.skipif(not run_correctness or num_tpu_cores < 8,
                    reason=f"Correctness test is skipped unless RUN_MISTRAL_SMALL_4_CORRECTNESS_TEST=1 is set. It also requires at least 8 TPU cores (TP=8) but only {num_tpu_cores} available.")
def test_mistral_small_4_correctness():
    """Correctness test using real weights on TPU, matching user's production config."""
    set_env_vars()
    
    # Match user's config exactly
    tensor_parallel_size = 8 
    max_model_len = 3072
    max_num_batched_tokens = 2048
    max_num_seqs = 16
    kv_cache_dtype = "fp8"
    gpu_memory_utilization = 0.90
    
    prompts = [
        "Hello, my name is",
        "The capital of France is",
        "Write a short poem about artificial intelligence on TPUs.",
    ]
    
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=64,
    )
    
    additional_config = {
        "sharding": {
            "sharding_strategy": {
                "enable_dp_attention": True
            }
        }
    }
    
    print(f"Initializing LLM with model {MODEL_ID}...")
    llm = LLM(
        model=MODEL_ID,
        tensor_parallel_size=tensor_parallel_size,
        max_model_len=max_model_len,
        max_num_batched_tokens=max_num_batched_tokens,
        max_num_seqs=max_num_seqs,
        kv_cache_dtype=kv_cache_dtype,
        gpu_memory_utilization=gpu_memory_utilization,
        additional_config=additional_config,
        enable_expert_parallel=True,  # Match user's --enable-expert-parallel
        trust_remote_code=True,
    )
    
    print("Generating outputs...")
    outputs = llm.generate(prompts, sampling_params)
    
    assert len(outputs) == len(prompts)
    
    for i, output in enumerate(outputs):
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"\n--- Prompt: {prompt} ---")
        print(f"Generated: {generated_text}")
        assert len(generated_text) > 0, f"Generated text for prompt '{prompt}' is empty"
        
        # Simple sanity check for correctness
        if i == 1:
            assert "Paris" in generated_text, f"Expected 'Paris' in response to France capital, got: {generated_text}"


if __name__ == "__main__":
    import sys
    pytest.main([__file__] + sys.argv[1:])
