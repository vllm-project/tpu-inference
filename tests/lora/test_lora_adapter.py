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

import gc
import json
import os
import tempfile
import time

import torch
import vllm
from safetensors.torch import save_file
from vllm.lora.request import LoRARequest


def setup_vllm(num_loras: int, tp: int = 1) -> vllm.LLM:
    return vllm.LLM(
        model="Qwen/Qwen2.5-3B-Instruct",
        max_model_len=256,
        max_num_batched_tokens=64,
        max_num_seqs=8,
        tensor_parallel_size=tp,
        enable_lora=True,
        max_loras=num_loras,
        async_scheduling=0,
        max_lora_rank=128,
    )


def test_dynamic_lora_loading_api():
    """This test verifies we can load, list, pin, and unload adapters dynamically

    using the LLMEngine dynamic adapter-management APIs.
    """
    llm = setup_vllm(4)

    lora_name_template = "Username6568/Qwen2.5-3B-Instruct-1_plus_1_equals_{}_adapter"
    lora_request = LoRARequest("lora_adapter_2", 2,
                               lora_name_template.format(2))

    # 1. Dynamically add the adapter
    success = llm.llm_engine.add_lora(lora_request)
    assert success is True

    # 2. Verify it is listed in the registered adapters
    registered_loras = llm.llm_engine.list_loras()
    assert 2 in registered_loras

    # 3. Pin the adapter to prevent eviction
    assert llm.llm_engine.pin_lora(2) is True

    # 4. Dynamically remove the adapter
    success_remove = llm.llm_engine.remove_lora(2)
    assert success_remove is True

    # 5. Verify it is no longer listed
    assert 2 not in llm.llm_engine.list_loras()

    llm.llm_engine.engine_core.shutdown()
    del llm
    gc.collect()


def test_dynamic_lora_loading_multiple():
    """Loads multiple adapters dynamically and verifies pinning/listing."""
    llm = setup_vllm(4)

    lora_name_template = "Username6568/Qwen2.5-3B-Instruct-1_plus_1_equals_{}_adapter"
    req2 = LoRARequest("lora_adapter_2", 2, lora_name_template.format(2))
    req3 = LoRARequest("lora_adapter_3", 3, lora_name_template.format(3))

    # Load both adapters
    assert llm.llm_engine.add_lora(req2) is True
    assert llm.llm_engine.add_lora(req3) is True

    # Verify listings
    registered = llm.llm_engine.list_loras()
    assert 2 in registered
    assert 3 in registered

    # Pin adapter 3
    assert llm.llm_engine.pin_lora(3) is True

    # Unload both
    assert llm.llm_engine.remove_lora(2) is True
    assert llm.llm_engine.remove_lora(3) is True
    assert len(llm.llm_engine.list_loras()) == 0

    llm.llm_engine.engine_core.shutdown()
    del llm
    gc.collect()


def test_dynamic_lora_lru_eviction():
    """Tests LRU caching behaviour by loading more adapters than max_loras."""
    # Set max_loras to 1
    llm = setup_vllm(1)

    lora_name_template = "Username6568/Qwen2.5-3B-Instruct-1_plus_1_equals_{}_adapter"
    req2 = LoRARequest("lora_adapter_2", 2, lora_name_template.format(2))
    req3 = LoRARequest("lora_adapter_3", 3, lora_name_template.format(3))

    # 1. Load adapter 2
    assert llm.llm_engine.add_lora(req2) is True
    assert 2 in llm.llm_engine.list_loras()

    # 2. Load adapter 3 (since max_loras=1, this should evict adapter 2)
    assert llm.llm_engine.add_lora(req3) is True
    registered = llm.llm_engine.list_loras()
    assert 3 in registered
    assert 2 not in registered  # evicted

    llm.llm_engine.engine_core.shutdown()
    del llm
    gc.collect()


def test_dynamic_lora_with_bundled_base_weights():
    """Ensures that adapters with bundled base weights (using custom prefixes)

    do not crash the server during loading due to exact-match allow-list
    failures.
    """
    llm = setup_vllm(1)

    try:
        with tempfile.TemporaryDirectory() as tmp_dir:
            # 1. Create a minimal dummy adapter config
            # We target "q_proj" with rank 8 so the engine expects a specific shape
            config = {
                "r": 8,
                "lora_alpha": 16,
                "target_modules": ["q_proj"],
                "peft_type": "LORA",
            }
            with open(os.path.join(tmp_dir, "adapter_config.json"), "w") as f:
                json.dump(config, f)

            # 2. Create a dummy safetensors file
            # Qwen2.5-3B-Instruct q_proj dimensions:
            # In_features: 2048, Out_features: 2048
            # LoRA A shape: (r, in_features) -> (8, 2048)
            # LoRA B shape: (out_features, r) -> (2048, 8)
            tensors = {
                # THE CURE: A valid LoRA weight so the adapter isn't totally empty.
                # (Sized exactly for Qwen2.5-3B so PyTorch doesn't throw a shape mismatch)
                "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight":
                torch.zeros((8, 2048)),
                "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight":
                torch.zeros((2048, 8)),

                # THE POISON: This exact prefix crashes unpatched vLLM.
                # Our patch correctly categorizes it as a base weight and skips it.
                "base_model.model.lm_head.weight":
                torch.zeros((1, 1))
            }

            # Save the dummy adapter locally
            safetensors_path = os.path.join(tmp_dir,
                                            "adapter_model.safetensors")
            save_file(tensors, safetensors_path)

            # 3. Attempt to load the poisoned adapter
            lora_id = 1
            req = LoRARequest(f"lora_adapter_{lora_id}", lora_id, tmp_dir)

            # Without our patch, this will throw "ValueError: unsupported LoRA weight"
            # on the `lm_head.weight` tensor.
            # With our patch, it skips the base weight safely, loads the valid q_proj
            # LoRA weights, and succeeds!
            success = llm.llm_engine.add_lora(req)

            assert success
            assert lora_id in llm.llm_engine.list_loras()

    finally:
        llm.llm_engine.engine_core.shutdown()
        del llm
        gc.collect()


def test_dynamic_lora_e2e_generation():
    """Verifies end-to-end generation with dynamically loaded and hot-swapped LoRA adapters."""
    llm = setup_vllm(1)

    try:
        prompt = "What is 1+1?"
        sampling_params = vllm.SamplingParams(max_tokens=16, temperature=0.0)

        lora_name_template = "Username6568/Qwen2.5-3B-Instruct-1_plus_1_equals_{}_adapter"

        # 1. Load and verify lora_id = 1
        # If the TPU backend mistakenly uses the global ID instead of the
        # mapped local ID, passing `1` prevents the silent JAX out-of-bounds
        # clamping bug.
        lora_id = 1
        req = LoRARequest(f"lora_adapter_{lora_id}", lora_id,
                          lora_name_template.format(lora_id))

        success = llm.llm_engine.add_lora(req)
        assert success
        assert lora_id in llm.llm_engine.list_loras()

        # Give the background TPU workers a moment to compile/pin the weights.
        time.sleep(2)

        raw_output = llm.generate(prompt,
                                  sampling_params=sampling_params,
                                  lora_request=req)

        full_text = raw_output[0].outputs[0].text.strip()

        assert str(lora_id) in full_text, (
            f"TPU ENGINE BUG: The LoRA adapter weights were ignored during the forward pass!\n"
            f"Expected output to contain: '{lora_id}'\n"
            f"Got base model output: {full_text!r}")

        # 2. Load and verify lora_id = 3 (this also tests hot-swapping/eviction
        # since max_loras=1)
        lora_id_3 = 3
        req_3 = LoRARequest(f"lora_adapter_{lora_id_3}", lora_id_3,
                            lora_name_template.format(lora_id_3))

        success_3 = llm.llm_engine.add_lora(req_3)
        assert success_3
        assert lora_id_3 in llm.llm_engine.list_loras()

        # Give the background TPU workers a moment to compile/pin the weights.
        time.sleep(2)

        raw_output_3 = llm.generate(prompt,
                                    sampling_params=sampling_params,
                                    lora_request=req_3)

        full_text_3 = raw_output_3[0].outputs[0].text.strip()

        assert str(lora_id_3) in full_text_3, (
            f"TPU ENGINE BUG: The LoRA adapter weights were ignored during the forward pass!\n"
            f"Expected output to contain: '{lora_id_3}'\n"
            f"Got base model output: {full_text_3!r}")

    finally:
        llm.llm_engine.engine_core.shutdown()
        del llm
        gc.collect()
