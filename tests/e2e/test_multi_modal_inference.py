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

# Known-good partial text outputs for the cherry-blossom image. The exact
# caption shifts with upstream / vllm changes and with enable_dynamic_image_
# sizes, so the test passes if the generated output matches *any* of these.
EXPECTED_TEXTS = (
    "The image depicts a tall, cylindrical tower with a lattice-like structure, surrounded by cherry blossom trees in full bloom. The cherry blossoms are in various stages of opening, with pink petals covering the branches. The sky is clear and blue, providing a vibrant backdrop to the scene. The tower appears to be a significant landmark",
    # Prior output (PR #1947, "Tokyo Skytree" caption).
    "The image depicts a stunning view of the Tokyo Skytree, a tall broadcasting tower located in the Odaiba district of Tokyo, Japan. The skytree is surrounded by cherry blossom trees in full bloom, creating a picturesque and vibrant scene. The cherry blossoms are in various stages of bloom, with some branches densely covered",
    "This image captures a beautiful and iconic scene: the **Tokyo Tower**, viewed through the blossoming branches of a **cherry blossom tree**. Here's a breakdown of the content:*   **Foreground:** The image is framed by the delicate, pink blossoms of a cherry tree."
)


def _cleanup_tpu_zombies():
    """Clears lingering JAX/libtpu process locks under our user to prevent TPU OOM."""
    import subprocess
    try:
        # Kill orphaned EngineCore child workers
        subprocess.run(["pkill", "-9", "-f", "VLLM::EngineCore"],
                       stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL)
        # Kill orphaned vLLM engine subprocesses
        subprocess.run(["pkill", "-9", "-f", "vllm"],
                       stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL)
    except Exception:
        pass


# NOTE: Parameterized across key model variants and execution backends
@pytest.mark.parametrize("enable_dynamic_image_sizes", [False, True])
@pytest.mark.parametrize(
    "model_name, model_impl_type",
    [
        ("Qwen/Qwen2.5-VL-7B-Instruct", "flax_nnx"),
        ("Qwen/Qwen3-VL-8B-Instruct", "vllm"),
        ("Qwen/Qwen3-VL-30B-A3B-Instruct", "vllm"),
    ],
)
def test_multi_modal_inference(monkeypatch, enable_dynamic_image_sizes,
                               model_name, model_impl_type):
    """
    Runs multi-modal inference and verifies the output.
    """
    os.environ['SKIP_JAX_PRECOMPILE'] = '1'  # Skip warmup to save time.
    os.environ[
        'VLLM_XLA_CHECK_RECOMPILATION'] = '0'  # Allow compilation during execution.

    monkeypatch.setenv("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

    # Set the model implementation backend for the test case execution
    monkeypatch.setenv("MODEL_IMPL_TYPE", model_impl_type)

    # --- Configuration ---
    model = model_name

    # Count visible TPU accelerators via filesystem devices instead of JAX.
    # This prevents the master process from locking the TPU backend (/dev/vfio/0)
    # and blocking the spawned vLLM worker child processes.
    import glob
    accel_devices = glob.glob("/dev/accel*")
    num_chips = len(accel_devices) if accel_devices else 1

    # Hardware topology checking to prevent Out Of Memory (OOM) on smaller single-host slices
    if "30B" in model:
        if num_chips < 4:
            pytest.skip(
                f"Skipping 30B model: Requires at least 4 TPU chips (found {num_chips}) to prevent OOM."
            )
        tensor_parallel_size = num_chips
    else:
        # For 3B / 7B models, run on 1 or 2 chips maximum to minimize communication latency
        tensor_parallel_size = min(num_chips, 2)

    # Respect explicit environment variable overrides for TP size if set
    if os.environ.get("TENSOR_PARALLEL_SIZE"):
        tensor_parallel_size = int(os.environ["TENSOR_PARALLEL_SIZE"])

    temperature = 0.0
    max_tokens = 64
    max_model_len = 4096
    gpu_memory_utilization = 0.75
    modality = "image"

    print(
        f"Preparing for multi-modal inference with model {model} (MODEL_IMPL_TYPE={os.environ.get('MODEL_IMPL_TYPE', 'auto')}, TP={tensor_parallel_size})..."
    )

    # --- Prepare Inputs ---
    image = convert_image_mode(ImageAsset("cherry_blossom").pil_image, "RGB")
    question = "What is the content of this image?"

    # Using Qwen models prompt template
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

    # Clean up before initialization to release any stale locks immediately
    _cleanup_tpu_zombies()

    try:
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

        # Check output against the closest known-good caption.
        similarity_score = max(
            difflib.SequenceMatcher(
                None, generated_text, expected, autojunk=False).ratio()
            for expected in EXPECTED_TEXTS)
        print(f"Similarity Score: {similarity_score:.4f}")

        assert similarity_score >= 0.85, (
            f"Text verification failed.\n"
            f"Generated: {generated_text}\n"
            f"Expected similarity >= 0.85 (got {similarity_score:.2f})")
    finally:
        # Re-run cleanup on test teardown to release TPU locking resources immediately
        _cleanup_tpu_zombies()
