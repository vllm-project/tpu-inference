# SPDX-License-Identifier: Apache-2.0
#
# A simplified example to run multi-modal inference and verify the output.
# This script is a self-contained test that runs a single prompt and
# compares the output to a known-good output.

import difflib
import os
from dataclasses import asdict

import pytest
from vllm import LLM, EngineArgs, SamplingParams
from vllm.assets.image import ImageAsset
from vllm.multimodal.image import convert_image_mode

# Expected partial text output from the model. This is based on a previous
# run and is used for verification. The test is considered passed if the
# generated output match with this text.
EXPECTED_TEXT = (
    "The image depicts a tall, cylindrical tower with a lattice-like structure, surrounded by cherry blossom trees in full bloom. The cherry blossoms are in various stages of opening, with pink petals covering the branches. The sky is clear and blue, providing a vibrant backdrop to the scene. The tower appears to be a significant landmark"
)


# NOTE: Could be extended to more mm models/configs as needed
@pytest.mark.parametrize("enable_dynamic_image_sizes", [False, True])
def test_multi_modal_inference(monkeypatch, enable_dynamic_image_sizes):
    """
    Runs multi-modal inference and verifies the output.
    """
    os.environ['SKIP_JAX_PRECOMPILE'] = '1'  # Skip warmup to save time.
    os.environ[
        'VLLM_XLA_CHECK_RECOMPILATION'] = '0'  # Allow compilation during execution.

    monkeypatch.setenv("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

    # --- Configuration ---
    model = "Qwen/Qwen2.5-VL-3B-Instruct"
    tensor_parallel_size = 1
    temperature = 0.0
    max_tokens = 64
    max_model_len = 4096
    gpu_memory_utilization = 0.5
    modality = "image"

    print("Preparing for multi-modal inference...")

    # --- Prepare Inputs ---
    image = convert_image_mode(ImageAsset("cherry_blossom").pil_image, "RGB")
    question = "What is the content of this image?"

    # Using Qwen2.5-VL prompt template
    # NOTE: other models may be different
    prompt = ("<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
              f"<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>"
              f"{question}<|im_end|>\n"
              "<|im_start|>assistant\n")

    # --- Setup vLLM Engine ---
    engine_args = EngineArgs(
        model=model,
        max_model_len=max_model_len,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
        max_num_seqs=1,
        mm_processor_kwargs={
            "min_pixels": 28 * 28,
            "max_pixels": 1280 * 28 * 28,
            "fps": 1,
        },
        limit_mm_per_prompt={modality: 1},
    )
    engine_args = asdict(engine_args)
    if engine_args.get("additional_config") is None:
        engine_args["additional_config"] = {}

    engine_args["additional_config"][
        "enable_dynamic_image_sizes"] = enable_dynamic_image_sizes
    llm = LLM(**engine_args)

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
    outputs = llm.generate(inputs, sampling_params)

    # --- Verification ---
    generated_text = outputs[0].outputs[0].text.strip()

    print("-" * 50)
    print("Generated Text:")
    print(generated_text)
    print("-" * 50)

    # Check output
    similarity_score = difflib.SequenceMatcher(None, generated_text,
                                               EXPECTED_TEXT).ratio()
    print(f"Similarity Score: {similarity_score:.4f}")
    assert similarity_score >= 0.85, (
        f"Text similarity too low ({similarity_score:.2f}).\n"
        f"Expected: {EXPECTED_TEXT}\n"
        f"Actual:   {generated_text}")
