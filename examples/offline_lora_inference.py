# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import time

import vllm.envs as vllm_envs
from vllm import LLM, EngineArgs
from vllm.lora.request import LoRARequest
from vllm.utils.argparse_utils import FlexibleArgumentParser


def create_parser():
    parser = FlexibleArgumentParser()
    # Add engine args
    EngineArgs.add_cli_args(parser)
    parser.set_defaults(model="Qwen/Qwen2.5-3B-Instruct")
    parser.set_defaults(max_model_len=256)
    parser.set_defaults(max_num_seqs=8)
    parser.set_defaults(enable_lora=True)
    parser.set_defaults(max_lora_rank=8)

    # Skip long warmup for local simple test.
    parser.set_defaults(enforce_eager=True)

    # Add sampling params
    sampling_group = parser.add_argument_group("Sampling parameters")
    sampling_group.add_argument("--max-tokens", type=int, default=16)
    sampling_group.add_argument("--temperature", type=float, default=0)
    sampling_group.add_argument("--top-p", type=float)
    sampling_group.add_argument("--top-k", type=int)
    return parser


def main(args: dict):
    # Pop arguments not used by LLM
    max_tokens = args.pop("max_tokens")
    temperature = args.pop("temperature")
    top_p = args.pop("top_p")
    top_k = args.pop("top_k")

    # Create an LLM
    llm = LLM(**args)

    # Create a sampling params object
    sampling_params = llm.get_default_sampling_params()
    if max_tokens is not None:
        sampling_params.max_tokens = max_tokens
    if temperature is not None:
        sampling_params.temperature = temperature
    if top_p is not None:
        sampling_params.top_p = top_p
    if top_k is not None:
        sampling_params.top_k = top_k

    prompt = "What is 1+1? \n"
    lora_request = LoRARequest(
        "lora_adapter_3", 3,
        "Username6568/Qwen2.5-3B-Instruct-1_plus_1_equals_3_adapter")

    if vllm_envs.VLLM_TORCH_PROFILER_DIR is not None:
        llm.start_profile()
    start = time.perf_counter()
    outputs = llm.generate(prompt,
                           sampling_params=sampling_params,
                           lora_request=lora_request)
    if vllm_envs.VLLM_TORCH_PROFILER_DIR is not None:
        llm.stop_profile()

    # Print the outputs.
    print("-" * 50)
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}\nGenerated text: {generated_text!r}")
        print("-" * 50)
    end = time.perf_counter()
    print(f'total time: {end - start} [secs].')


if __name__ == "__main__":
    parser = create_parser()
    args: dict = vars(parser.parse_args())

    main(args)
