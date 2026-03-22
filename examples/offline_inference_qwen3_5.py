# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# CMD: MODEL_IMPL_TYPE=vllm python examples/offline_inference.py --model=Qwen/Qwen3.5-35B-A3B-FP8 --tensor-parallel-size=1  --limit-mm-per-prompt '{"image": 0, "video": 0}'   --hf-overrides '{"text_config": {"rope_parameters": null, "rope_theta": 10000000, "partial_rotary_factor": 0.25}}' --block-size=256 --gpu-memory-utilization=0.4 --kv-cache-dtype=fp8 --temperature=0 --top_p=1.0 --top_k=-1

import os

from vllm import LLM, EngineArgs
from vllm.utils.argparse_utils import FlexibleArgumentParser


def create_parser():
    parser = FlexibleArgumentParser()
    # Add engine args
    EngineArgs.add_cli_args(parser)
    parser.set_defaults(model="meta-llama/Llama-3.2-1B-Instruct")
    parser.set_defaults(max_model_len=1024)

    # Add sampling params
    sampling_group = parser.add_argument_group("Sampling parameters")
    sampling_group.add_argument("--max-tokens", type=int)
    sampling_group.add_argument("--temperature", type=float)
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

    prompts = [
        "Hello, my name is",
        "The capital of France is",
        "The colors of the rainbow are",
        "The future of AI is",
        "The president of the United States is",
        "How many players are on a standard soccer team on the field at one time?",
        "In Greek mythology, who is the god of the sea?",
        "In what year did the Titanic sink?",
        "In which museum is the Mona Lisa displayed?",
        "Mount Everest is located in which mountain range?",
    ]

    # CHANGE 1: Convert raw text prompts into conversation objects
    conversations = [[{
        "role": "user",
        "content": prompt
    }] for prompt in prompts]

    profiler_config = llm.llm_engine.vllm_config.profiler_config
    if profiler_config.profiler == "torch":
        llm.start_profile()

    # CHANGE 2: Use llm.chat() instead of llm.generate()
    # and pass the chat_template_kwargs to disable thinking!

    outputs = llm.chat(messages=conversations,
                       sampling_params=sampling_params,
                       chat_template_kwargs={"enable_thinking": False})

    if profiler_config.profiler == "torch":
        llm.stop_profile()

    # Print the outputs.
    print("-" * 50)
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}\nGenerated text: {generated_text!r}")
        print("-" * 50)


if __name__ == "__main__":
    # Skip long warmup for local simple test.
    os.environ.setdefault('SKIP_JAX_PRECOMPILE', '1')

    parser = create_parser()
    args: dict = vars(parser.parse_args())

    main(args)
