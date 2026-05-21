# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Vision smoke test for Gemma-4 multimodal checkpoints on the JAX path.

Loads the requested gemma-4-*-it model, runs a single chat-with-image
request against the vllm `cherry_blossom` asset, and asserts that the
generation is non-empty. Exits non-zero on any failure so the surrounding
CI step fails fast.

Example:
  python examples/multi_modal_inference_gemma.py \\
    --model google/gemma-4-E4B-it \\
    --tensor-parallel-size 1
"""

import argparse

from vllm import LLM, SamplingParams
from vllm.assets.image import ImageAsset
from vllm.multimodal.image import convert_image_mode


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-num-batched-tokens", type=int, default=4096)
    parser.add_argument("--max-num-seqs", type=int, default=4)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--max-tokens", type=int, default=64)
    args = parser.parse_args()

    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_num_seqs=args.max_num_seqs,
        gpu_memory_utilization=args.gpu_memory_utilization,
        limit_mm_per_prompt={"image": 1},
        disable_chunked_mm_input=True,
    )

    img = convert_image_mode(ImageAsset("cherry_blossom").pil_image, "RGB")
    messages = [{
        "role":
        "user",
        "content": [
            {
                "type": "image_pil",
                "image_pil": img
            },
            {
                "type": "text",
                "text": "Describe what is in this image in one sentence."
            },
        ],
    }]

    outputs = llm.chat(
        messages=messages,
        sampling_params=SamplingParams(temperature=0.0,
                                       max_tokens=args.max_tokens),
    )

    text = outputs[0].outputs[0].text
    print(f"GENERATED: {text!r}")
    if not text or len(text.strip()) < 10:
        raise SystemExit(
            f"vision smoke test produced empty/too-short output: {text!r}")


if __name__ == "__main__":
    main()
