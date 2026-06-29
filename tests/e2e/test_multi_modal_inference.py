# SPDX-License-Identifier: Apache-2.0
#
# A simplified example to run multi-modal inference and verify the output.
# This script is a self-contained test that runs a single prompt and
# compares the output to a known-good output.

import difflib
import gc
import os
import traceback
from dataclasses import asdict

import pytest
from vllm import LLM, EngineArgs, SamplingParams
from vllm.assets.image import ImageAsset
from vllm.multimodal.image import convert_image_mode

# Known-good partial text outputs for the cherry-blossom image. The exact
# caption shifts with upstream / vllm changes and with enable_dynamic_image_
# sizes, so the test passes if the generated output matches *any* of these.
EXPECTED_TEXTS = (
    "The image depicts a tall, cylindrical tower with a lattice-like structure, surrounded by cherry blossom trees in full bloom. The cherry blossoms are in various stages of opening, with pink petals covering the branches. The sky is clear and blue, providing a vibrant backdrop to the scene. The tower appears to be a significant landmark",
    # Prior output (PR #1947, "Tokyo Skytree" caption).
    "The image depicts a stunning view of the Tokyo Skytree, a tall broadcasting tower located in the Odaiba district of Tokyo, Japan. The skytree is surrounded by cherry blossom trees in full bloom, creating a picturesque and vibrant scene. The cherry blossoms are in various stages of bloom, with some branches densely covered",
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
    # tp=2 for tpu7x, tp=1 for tpu6e (matches accuracy test convention)
    tensor_parallel_size = 2 if os.environ.get('TPU_VERSION') == 'tpu7x' else 1
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

    llm = LLM(**engine_args)

    try:
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

        # Check output against the closest known-good caption.
        similarity_score = max(
            difflib.SequenceMatcher(
                None, generated_text, expected, autojunk=False).ratio()
            for expected in EXPECTED_TEXTS)
        print(f"Similarity Score: {similarity_score:.4f}")
        assert similarity_score >= 0.85, (
            f"Text similarity too low ({similarity_score:.2f}).\n"
            f"Expected one of: {EXPECTED_TEXTS}\n"
            f"Actual:   {generated_text}")
    finally:
        # Release the TPU before the next parametrization builds a new engine.
        # This test is parametrized over enable_dynamic_image_sizes=[False, True]
        # and pytest runs both in one process; without an explicit engine
        # shutdown the first engine's worker keeps the TPU and the second
        # LLM(...) fails with "TPU already in use by pid <first>". Mirrors the
        # teardown in tests/e2e/test_continue_decode.py.
        #
        # Guard the shutdown so a teardown failure cannot mask a body
        # AssertionError (the similarity check) -- surface it loudly instead of
        # swallowing it.
        try:
            llm.llm_engine.engine_core.shutdown()
        except Exception:
            print("[multimodal-test] engine_core.shutdown() failed during "
                  "teardown (non-fatal):")
            traceback.print_exc()
        gc.collect()
