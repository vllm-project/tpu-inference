# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
This example shows how to use vLLM for running offline inference with
the correct prompt format on vision language models for text generation.

For most models, the prompt format should follow corresponding examples
on HuggingFace model repository.

Example Command:
python examples/multi_modal_inference.py \
  --model Qwen/Qwen2.5-VL-3B-Instruct \
  --tensor-parallel-size 1 \
  --num-prompts 1
"""

from contextlib import contextmanager
from dataclasses import asdict
from typing import NamedTuple, Optional

from vllm import LLM, EngineArgs, SamplingParams
from vllm.assets.image import ImageAsset
from vllm.multimodal.image import convert_image_mode
from vllm.utils.argparse_utils import FlexibleArgumentParser


class ModelRequestData(NamedTuple):
    engine_args: EngineArgs
    prompts: list[str]
    stop_token_ids: Optional[list[int]] = None


# Currently Qwen2.5-VL is the only supported multi-modal
# Qwen2.5-VL
def run_qwen2_5_vl(questions: list[str], modality: str,
                   args) -> ModelRequestData:
    engine_args = EngineArgs(
        model=args.model,
        max_model_len=args.max_model_len,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_num_seqs=5,
        mm_processor_kwargs={
            "size": {
                "longest_edge": 1003520,
                "shortest_edge": 3136
            },
            "fps": 1,
        },
        limit_mm_per_prompt={modality: 1},
    )

    if modality == "image":
        placeholder = "<|image_pad|>"
    elif modality == "video":
        placeholder = "<|video_pad|>"

    prompts = [
        ("<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
         f"<|im_start|>user\n<|vision_start|>{placeholder}<|vision_end|>"
         f"{question}<|im_end|>\n"
         "<|im_start|>assistant\n") for question in questions
    ]

    return ModelRequestData(
        engine_args=engine_args,
        prompts=prompts,
    )


model_example_map = {
    "qwen2_5_vl": run_qwen2_5_vl,
}


def get_multi_modal_input(args):
    """
    return {
        "data": image or video,
        "question": question,
    }
    """
    if args.modality == "image":
        # Input image and question
        image = convert_image_mode(
            ImageAsset("cherry_blossom").pil_image, "RGB")
        img_questions = [
            "What is the content of this image?",
            "Describe the content of this image in detail.",
            "What's in the image?",
            "Where is this image taken?",
        ]

        return {
            "data": image,
            "questions": img_questions,
        }

    # NOTE: not used for now, saved for future reference
    # if args.modality == "video":
    #     # Input video and question
    #     video = VideoAsset(name="baby_reading",
    #                        num_frames=args.num_frames).np_ndarrays
    #     metadata = VideoAsset(name="baby_reading",
    #                           num_frames=args.num_frames).metadata
    #     vid_questions = ["Why is this video funny?"]

    #     return {
    #         "data": video,
    #         "questions": vid_questions,
    #     }

    msg = f"Modality {args.modality} is not supported."
    raise ValueError(msg)


@contextmanager
def time_counter(enable: bool):
    if enable:
        import time

        start_time = time.time()
        yield
        elapsed_time = time.time() - start_time
        print("-" * 50)
        print("-- generate time = {}".format(elapsed_time))
        print("-" * 50)
    else:
        yield


def parse_args():
    parser = FlexibleArgumentParser(
        description="Demo on using vLLM for offline inference with "
        "vision language models for text generation")
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default="Qwen/Qwen2.5-VL-3B-Instruct",
        help="Huggingface model name.",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        "-tp",
        type=int,
        default=1,
        help="Tensor parallelism degree",
    )

    # TODO: Currently it needs much more space for intermidiate tensors
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.5,
        help="GPU memory utilization",
    )

    parser.add_argument(
        "--max-model-len",
        type=int,
        default=4096,
        help="Maximum sequence length of the model.",
    )
    parser.add_argument("--num-prompts",
                        type=int,
                        default=4,
                        help="Number of prompts to run.")

    # NOTE: Currently, only image modality is supported
    parser.add_argument(
        "--modality",
        type=str,
        default="image",
        choices=["image", "video"],
        help="Modality of the input.",
    )

    # NOTE: For video. Not used for now
    parser.add_argument(
        "--num-frames",
        type=int,
        default=16,
        help="Number of frames to extract from the video.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Set the seed when initializing `vllm.LLM`.",
    )
    parser.add_argument(
        "--disable-mm-preprocessor-cache",
        action="store_true",
        help="If True, disables caching of multi-modal preprocessor/mapper.",
    )

    parser.add_argument(
        "--time-generate",
        action="store_true",
        help="If True, then print the total generate() call time",
    )

    parser.add_argument(
        "--use-same-prompt-per-request",
        action="store_false",
        dest="use_different_prompt_per_request",
        help="If set, use the same prompt for all requests. By default, "
        "different prompts are used for each request.",
    )
    return parser.parse_args()


def main(args):

    modality = args.modality
    mm_input = get_multi_modal_input(args)
    data = mm_input["data"]
    questions = mm_input["questions"]

    # NOTE: Currently, only Qwen2.5-VL is supported. If later we want to support a model with new chat template, we may need to change this
    req_data = model_example_map["qwen2_5_vl"](questions, modality, args)

    # Disable other modalities to save memory
    # Initial all modalities to be 0s and add the specifc modality limit later accordingly
    default_limits = {"image": 0, "video": 0, "audio": 0}
    req_data.engine_args.limit_mm_per_prompt = default_limits | dict(
        req_data.engine_args.limit_mm_per_prompt or {})

    engine_args = asdict(req_data.engine_args) | {
        "seed": args.seed,
        "disable_mm_preprocessor_cache": args.disable_mm_preprocessor_cache,
    }
    llm = LLM(**engine_args)

    # Don't want to check the flag multiple times, so just hijack `prompts`.
    prompts = (req_data.prompts if args.use_different_prompt_per_request else
               [req_data.prompts[0]])

    # We set temperature to 0.2 so that outputs can be different
    # even when all prompts are identical when running batch inference.
    sampling_params = SamplingParams(temperature=0.2,
                                     max_tokens=64,
                                     stop_token_ids=req_data.stop_token_ids)

    assert args.num_prompts > 0
    if args.num_prompts == 1:
        # Single inference
        inputs = {
            "prompt": prompts[0],
            "multi_modal_data": {
                modality: data
            },
        }
    else:
        # Batch inference
        # Use the same image for all prompts
        inputs = [{
            "prompt": prompts[i % len(prompts)],
            "multi_modal_data": {
                modality: data
            },
        } for i in range(args.num_prompts)]

    with time_counter(args.time_generate):
        outputs = llm.generate(
            inputs,
            sampling_params=sampling_params,
        )

    print("-" * 50)
    for o in outputs:
        generated_text = o.outputs[0].text
        print(generated_text)
        print("-" * 50)


if __name__ == "__main__":
    args = parse_args()
    main(args)
