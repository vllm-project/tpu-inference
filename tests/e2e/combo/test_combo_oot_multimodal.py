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

import difflib
import os
from dataclasses import asdict

import pytest
from vllm import LLM, EngineArgs, SamplingParams
from vllm.assets.image import ImageAsset
from vllm.model_executor.models.qwen2_5_vl import \
    Qwen2_5_VLForConditionalGeneration as VllmOfficialTorchModel
from vllm.model_executor.models.registry import ModelRegistry
from vllm.multimodal.image import convert_image_mode

# Official tpu_inference libraries and registries
from tpu_inference.models.common.model_loader import _MODEL_REGISTRY
from tpu_inference.models.jax.qwen2_5_vl import \
    Qwen2_5_VLForConditionalGeneration as TpuOfficialJaxModel

try:
    from vllm.model_executor.models.interfaces_base import is_vllm_model
    VLLM_INTERFACE_CHECK_AVAILABLE = True
except ImportError:
    VLLM_INTERFACE_CHECK_AVAILABLE = False

# --- MODULE LEVEL REGISTRATION ---
# This executes in every process (Main, Subprocess, Worker) that imports this file.
custom_arch = "My_Inherited_OOT_Multimodal_Model"


# 1. Define the real execution class (Pure JAX)
class OOTMultimodalModel(TpuOfficialJaxModel):
    _inference_verified = False

    def __init__(self, *args, **kwargs):
        # Provenance signature to verify active process execution in logs
        print(f"[Debug] OOT PLUGIN: Instance Initialized (PID {os.getpid()})")
        super().__init__(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        # Use class variable to ensure we only print once per process during inference
        if not OOTMultimodalModel._inference_verified:
            print(f"[Debug] OOT INFERENCE START (PID {os.getpid()})")
            OOTMultimodalModel._inference_verified = True
        return super().__call__(*args, **kwargs)


# 2. Define the Inspection Shadow Class (Pure Torch)
class OOTMultimodalModelShadow(VllmOfficialTorchModel):
    pass


# 3. Perform Double Registration
# A. TPU Local Registry
_MODEL_REGISTRY[custom_arch] = OOTMultimodalModel
# B. vLLM Global Registry (via string to force subprocess sync)
# This forces the vLLM subprocess to import this file and see OOTMultimodalModelShadow.
ModelRegistry.register_model(custom_arch,
                             f"{__name__}:OOTMultimodalModelShadow")

# Standard gold-standard texts for accuracy check
EXPECTED_TEXTS = (
    "The image depicts a tall, cylindrical tower with a lattice-like structure, surrounded by cherry blossom trees in full bloom. The cherry blossoms are in various stages of opening, with pink petals covering the branches. The sky is clear and blue, providing a vibrant backdrop to the scene. The tower appears to be a significant landmark",
    "The image depicts a stunning view of the Tokyo Skytree, a tall broadcasting tower located in the Odaiba district of Tokyo, Japan. The skytree is surrounded by cherry blossom trees in full bloom, creating a picturesque and vibrant scene. The cherry blossoms are in various stages of bloom, with some branches densely covered",
)


# NOTE: Could be extended to more mm models/configs as needed
@pytest.mark.parametrize("enable_dynamic_image_sizes", [False, True])
def test_oot_multimodal_full_stack_verification(monkeypatch,
                                                enable_dynamic_image_sizes):
    """
    Combined E2E Test: OOT Inheritance + Registry Integrity + TPU Inference.
    """
    os.environ['SKIP_JAX_PRECOMPILE'] = '1'  # Skip warmup to save time.
    os.environ[
        'VLLM_XLA_CHECK_RECOMPILATION'] = '0'  # Allow compilation during execution.

    # monkeypatch.setenv("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

    print("Preparing for multi-modal inference...")

    # --- Configuration ---
    model = "Qwen/Qwen2.5-VL-3B-Instruct"
    # tp=2 for tpu7x, tp=1 for tpu6e (matches accuracy test convention)
    tensor_parallel_size = 2 if os.environ.get('TPU_VERSION') == 'tpu7x' else 1
    temperature = 0.0
    max_tokens = 64
    max_model_len = 4096
    gpu_memory_utilization = 0.5
    modality = "image"

    engine_args = EngineArgs(
        model=model,
        # Redirect vLLM to our custom architecture name
        hf_overrides={"architectures": [custom_arch]},
        max_model_len=max_model_len,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
        max_num_seqs=1,
        mm_processor_kwargs={
            "size": {
                "longest_edge": 1003520,
                "shortest_edge": 3136
            },
            "fps": 1,
        },
        limit_mm_per_prompt={modality: 1},
    )

    engine_args = asdict(engine_args)
    if engine_args.get("additional_config") is None:
        engine_args["additional_config"] = {}

    engine_args["additional_config"][
        "enable_dynamic_image_sizes"] = enable_dynamic_image_sizes
    engine_args["compilation_config"]["cudagraph_capture_sizes"] = []

    # asdict() leaves pass_config fields as None; LLM(**engine_args) re-validates
    # the dict and pydantic rejects None for some of them. Drop the None entries
    # so PassConfig uses its own defaults.
    pass_config = engine_args["compilation_config"].get("pass_config") or {}
    pass_config = {k: v for k, v in pass_config.items() if v is not None}
    engine_args["compilation_config"]["pass_config"] = pass_config

    # Initialize Engine.
    llm = LLM(**engine_args)

    print("Registration Verification...")

    # 1. Verify TPU registry has the real JAX class
    assert _MODEL_REGISTRY[custom_arch] is OOTMultimodalModel

    # 2. Verify ModelConfig has successfully adopted our custom architecture name.
    assert llm.llm_engine.model_config.architectures == [custom_arch]

    # 3. Final Runtime Registry Check.
    # Confirm the registry inside the main process points to our custom classes.
    resolved_cls, _ = ModelRegistry.resolve_model_cls(
        architectures=[custom_arch], model_config=llm.llm_engine.model_config)
    assert issubclass(resolved_cls, OOTMultimodalModelShadow)

    # 4. Interface Compatibility Check
    if VLLM_INTERFACE_CHECK_AVAILABLE:
        assert is_vllm_model(resolved_cls)

    # --- Prepare Inputs ---
    image = convert_image_mode(ImageAsset("cherry_blossom").pil_image, "RGB")
    question = "What is the content of this image?"

    # Using Qwen2.5-VL prompt template
    # NOTE: other models may be different
    prompt = ("<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
              f"<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>"
              f"{question}<|im_end|>\n"
              "<|im_start|>assistant\n")

    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=max_tokens,
    )

    inputs = {
        "prompt": prompt,
        "multi_modal_data": {
            "image": image
        },
    }

    # --- Run Inference ---
    print("Running inference...")
    inputs = {"prompt": prompt, "multi_modal_data": {"image": image}}
    outputs = llm.generate(inputs, sampling_params)
    generated_text = outputs[0].outputs[0].text.strip()

    print("-" * 50)
    print("Generated Text:")
    print(generated_text)
    print("-" * 50)

    print("Verified Response from OOT Class")
    # Accuracy similarity check
    similarity_score = max(
        difflib.SequenceMatcher(None, generated_text, expected,
                                autojunk=False).ratio()
        for expected in EXPECTED_TEXTS)
    print(f"Similarity Score: {similarity_score:.4f}")
    assert similarity_score >= 0.85, (
        f"Text similarity too low ({similarity_score:.2f}).\n"
        f"Expected one of: {EXPECTED_TEXTS}\n"
        f"Actual:   {generated_text}")
