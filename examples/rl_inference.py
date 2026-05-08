# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os

from vllm import LLM, EngineArgs
from vllm.utils.argparse_utils import FlexibleArgumentParser

from tpu_inference.core import disagg_utils
from tpu_inference.logger import init_logger

logger = init_logger(__name__)


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
    sampling_group.add_argument("--log-probs", type=int)

    return parser


def main(args: dict):
    # Pop arguments not used by LLM
    max_tokens = args.pop("max_tokens")
    temperature = args.pop("temperature")
    top_p = args.pop("top_p")
    top_k = args.pop("top_k")
    log_probs = args.pop("log_probs")

    # Create an LLM
    args["enable_return_routed_experts"] = True
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
    if log_probs is not None:
        sampling_params.logprobs = log_probs

    # Generate texts from the prompts. The output is a list of RequestOutput
    # objects that contain the prompt, generated text, and other information.
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
        "What ancient empire was ruled by Julius Caesar?",
        "What are the four fundamental forces of nature?",
        'What does "CPU" stand for?',
        'What does "HTML" stand for?',
        "What is the capital of Australia?",
        "What is the chemical symbol for gold?",
        "What is the currency of Switzerland?",
        "What is the distance from the Earth to the Sun called?",
        "What is the freezing point of water in Celsius?",
        "What is the hardest known natural substance on Earth?",
        "What is the largest planet in our solar system?",
        "What is the longest river in the world?",
        "What is the main function of the kidneys in the human body?",
        "What is the main ingredient in guacamole?",
        "What is the most spoken language in the world by number of native speakers?",
        "What is the process by which plants use sunlight to create food?",
        "Which country is known as the Land of the Rising Sun?",
        "Who developed the theory of general relativity?",
        'Who directed the original "Star Wars" trilogy?',
        "Who is credited with inventing the telephone?",
        "Who painted the ceiling of the Sistine Chapel?",
        "Who was the first female Prime Minister of the United Kingdom?",
        "Who was the first person to walk on the moon?",
        "Who wrote the American Declaration of Independence?",
        'Who wrote the novel "Pride and Prejudice"?',
    ]

    profiler_config = llm.llm_engine.vllm_config.profiler_config
    if profiler_config.profiler == "torch":
        llm.start_profile()

    outputs = llm.generate(prompts, sampling_params)

    if profiler_config.profiler == "torch":
        llm.stop_profile()

    # Print the outputs.
    print("-" * 50)
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}\nGenerated text: {generated_text!r}")

        P = len(output.prompt_token_ids)
        print(f"Prompt token count (P): {P}")

        prompt_experts_len = 0
        if hasattr(output, 'prompt_routed_experts'
                   ) and output.prompt_routed_experts is not None:
            print(
                f"Prompt routed experts shape: {output.prompt_routed_experts.shape}"
            )
            prompt_experts_len = output.prompt_routed_experts.shape[0]

        for completion in output.outputs:
            gen_experts_len = 0
            if completion.routed_experts is not None:
                # Shape will be [len(completion.token_ids) - 1, num_layers, top_k] in vLLM V1
                print(
                    f"Gen routed experts shape: {completion.routed_experts.shape}"
                )
                gen_experts_len = completion.routed_experts.shape[0]

            G = len(completion.token_ids)
            print(f"Generated token count (G): {G}")

            total_experts_len = prompt_experts_len + gen_experts_len
            if total_experts_len > 0:
                expected_len = P + G - 1
                print(
                    f"Expected total routed experts (P + G - 1): {expected_len}"
                )
                print(f"Actual total routed experts: {total_experts_len}")
                assert expected_len == total_experts_len, "Actual total routed experts does not match expected (P + G - 1)"

            if completion.logprobs is not None:
                print(f"Logprobs for first token: {completion.logprobs[0]}")

        print("-" * 50)


if __name__ == "__main__":
    # Skip long warmup for local simple test.
    os.environ.setdefault('SKIP_JAX_PRECOMPILE', '1')

    parser = create_parser()
    args: dict = vars(parser.parse_args())

    if not disagg_utils.is_disagg_enabled():
        main(args)
    else:
        from unittest.mock import patch

        from tpu_inference.core.core_tpu import (DisaggEngineCore,
                                                 DisaggEngineCoreProc)

        with patch("vllm.v1.engine.core.EngineCore", DisaggEngineCore), patch(
                "vllm.v1.engine.core.EngineCoreProc", DisaggEngineCoreProc):
            main(args)
