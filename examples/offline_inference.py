# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os

import vllm.envs as envs
from vllm import LLM, EngineArgs
from vllm.utils import FlexibleArgumentParser

from tpu_commons.core import disagg_utils


def create_parser():
    parser = FlexibleArgumentParser()
    # Add engine args
    EngineArgs.add_cli_args(parser)
    parser.set_defaults(model="meta-llama/Llama-3.2-1B-Instruct")
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

    # Generate texts from the prompts. The output is a list of RequestOutput
    # objects that contain the prompt, generated text, and other information.
    prompts = [
        "Hello, my name is",
        "The capital of France is",
        "The colors of the rainbow are"
        "The future of AI is",
        "The president of the United States is",
    ]
    if envs.VLLM_TORCH_PROFILER_DIR is not None:
        llm.start_profile()
    outputs = llm.generate(prompts, sampling_params)
    if envs.VLLM_TORCH_PROFILER_DIR is not None:
        llm.stop_profile()

    # Print the outputs.
    print("-" * 50)
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}\nGenerated text: {generated_text!r}")
        print("-" * 50)


if __name__ == "__main__":
    parser = create_parser()
    args: dict = vars(parser.parse_args())

    if not disagg_utils.is_disagg_enabled():
        main(args)
    else:
        from unittest.mock import patch

        from tpu_commons.core.core_tpu import EngineCore as TPUEngineCore
        from tpu_commons.core.core_tpu import \
            EngineCoreProc as TPUEngineCoreProc
        with patch('vllm.v1.engine.core.EngineCore', TPUEngineCore):
            with patch('vllm.v1.engine.core.EngineCoreProc',
                       TPUEngineCoreProc):
                main(args)
