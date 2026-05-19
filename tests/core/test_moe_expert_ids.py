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

import json
import os
import tempfile

import pytest
from vllm import LLM, SamplingParams


@pytest.fixture(scope="module")
def llm():
    # Define a tiny Qwen3 MoE model config
    config_data = {
        "architectures": ["Qwen3MoeForCausalLM"],
        "model_type": "qwen3",
        "hidden_size": 64,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 4,
        "intermediate_size": 256,
        "moe_intermediate_size": 128,
        "num_experts": 4,
        "num_experts_per_tok": 2,
        "decoder_sparse_step": 1,
        "vocab_size": 50257,  # Match GPT2 vocab size
        "rms_norm_eps": 1e-6,
        "tie_word_embeddings": False,
        "hidden_act": "silu"
    }

    with tempfile.TemporaryDirectory() as temp_dir:
        config_path = os.path.join(temp_dir, "config.json")
        with open(config_path, "w") as f:
            json.dump(config_data, f)

        # Force JAX backend and flax_nnx implementation.
        # We set it in the environment so it propagates to workers if any.
        os.environ["MODEL_IMPL_TYPE"] = "flax_nnx"

        engine = LLM(
            model=temp_dir,
            tokenizer="gpt2",
            load_format="dummy",
            trust_remote_code=True,
        )
        yield engine


class TestMoEExpertIds:
    """Verify that MoE routed experts are successfully returned when enabled,
    and not returned when disabled.
    """

    def test_moe_expert_ids_returned_when_enabled(self, llm: LLM):
        prompt = "The capital of France is"
        # Enable the flag
        llm.llm_engine.vllm_config.model_config.enable_return_routed_experts = True

        sampling_params = SamplingParams(temperature=0, max_tokens=10)
        outputs = llm.generate([prompt], sampling_params)
        output = outputs[0].outputs[0]

        # Verify that routed_experts is populated and has correct shape
        assert output.routed_experts is not None, (
            "MoE models must populate routed_experts when enabled")
        assert len(output.routed_experts.shape) == 3, (
            f"Expected 3D expert shape, got {output.routed_experts.shape}")

        # Verify that the token dimension has size P + G - 1
        P = len(outputs[0].prompt_token_ids)
        G = len(output.token_ids)
        expected_len = P + G - 1
        actual_len = output.routed_experts.shape[0]
        assert actual_len == expected_len, (
            f"Expected expert 0-th dim to be P + G - 1 ({expected_len}), "
            f"got {actual_len}")

    def test_moe_expert_ids_not_returned_when_disabled(self, llm: LLM):
        prompt = "The capital of France is"
        # Disable the flag
        llm.llm_engine.vllm_config.model_config.enable_return_routed_experts = False

        sampling_params = SamplingParams(temperature=0, max_tokens=10)
        outputs = llm.generate([prompt], sampling_params)
        output = outputs[0].outputs[0]

        # Verify that routed_experts is None
        assert output.routed_experts is None, (
            "MoE models must not populate routed_experts when disabled")
