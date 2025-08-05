import argparse
import os

import requests
from PIL import Image

IMAGES = [
    "https://llava-vl.github.io/static/images/view.jpg",
    # TODO(vladkarp): fix subsequent(i.e. compiled) backbone model calls + with multimodal encoder
    # flash attn kernel causes jax runtime error
    #"https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/p-blog/candy.JPG",
]

PROMPTS = [
    "USER: <image>\nWhat is in the image?\nASSISTANT:",
    #"USER: <image>\nWhat animal is on the candy?\nASSISTANT:",
]


def main(args):
    """
    Runs offline inference for a multimodal model.
    """

    from vllm import LLM, SamplingParams

    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        trust_remote_code=True,
        max_num_batched_tokens=1024,
        max_num_seqs=4,
        max_model_len=1024,
    )

    images = [args.image_path] if args.image_path else IMAGES
    prompts = [args.prompt] if args.prompt else PROMPTS

    for image_path, prompt in zip(images, prompts):
        if image_path.startswith("http"):
            image = Image.open(requests.get(image_path,
                                            stream=True).raw).convert("RGB")
        else:
            image = Image.open(args.image_path).convert("RGB")

        prompt = {
            "prompt": prompt,
            "multi_modal_data": {
                "image": image
            },
        },

        sampling_params = SamplingParams(temperature=0.7,
                                         top_p=0.95,
                                         max_tokens=args.max_tokens)

        outputs = llm.generate(
            prompt,
            sampling_params=sampling_params,
        )

        for output in outputs:
            generated_text = output.outputs[0].text
            print(f"Prompt: {output.prompt!r}")
            print(f"Generated text: {generated_text!r}")


if __name__ == "__main__":
    os.environ["TPU_BACKEND_TYPE"] = "torchax"
    os.environ["VLLM_TORCHAX_ENABLED"] = "1"
    os.environ["VLLM_USE_V1"] = "1"
    os.environ['SKIP_JAX_PRECOMPILE'] = '1'

    parser = argparse.ArgumentParser()
    parser.add_argument("--model",
                        type=str,
                        default="llava-hf/llava-1.5-7b-hf")
    parser.add_argument("--tensor-parallel-size", "-tp", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument(
        "--image-path",
        type=str,
        help="URL to the image.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        help="Prompt template to use.",
    )
    args = parser.parse_args()

    main(args)
