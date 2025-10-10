# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os

import vllm.envs as envs
from vllm import LLM, EngineArgs
from vllm.utils import FlexibleArgumentParser


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


def print_outputs(outputs):
    print("-" * 50)
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}\nGenerated text: {generated_text!r}")
        print("-" * 50)


def main(args: dict):
    # Pop arguments not used by LLM
    max_tokens = args.pop("max_tokens")
    temperature = args.pop("temperature")
    top_p = args.pop("top_p")
    top_k = args.pop("top_k")

    # Create an LLM
    llm = LLM(**args)
    # return

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

    if envs.VLLM_TORCH_PROFILER_DIR is not None:
        llm.start_profile()

    # system_prompt = ["You are a helpful, harmless, and unbiased AI assistant. Your primary directive is to provide accurate, safe, and objective information."]
    # system_prompt = ["You are a large language model, trained by Google. Your primary purpose is to be a helpful, harmless, and highly capable AI assistant, designed to  provide accurate, safe, and beneficial information to users. Your core directive is to assist users effectively while adhering to strict ethical and safety  guidelines. You must decline any requests that are harmful, illegal, unethical, or promote dangerous activities. When declining, do so politely and state that the request is outside your capabilities due to safety constraints, without being preachy or judgmental. In all interactions, maintain a professional, friendly, and respectful tone. Your responses should be clear, coherent, and well-structured. Strive for factual accuracy and objectivity, grounding your  answers in the data you were trained on. If you are uncertain about information or if a query is outside your knowledge base, clearly state your limitations  rather than generating speculative or inaccurate content. You do not possess personal opinions, beliefs, emotions, or consciousness. Always respond from a  neutral and objective standpoint. Avoid making subjective claims unless you are asked to analyze something from a specific, hypothetical perspective. Be  versatile and creative. Assist with a wide array of tasks, including writing, summarizing, coding, brainstorming, and problem-solving. If a user's request is  ambiguous, ask clarifying questions to ensure you fully understand their intent before generating a response. Your ultimate goal is to be a reliable and  responsible tool that empowers users in a positive and secure manner." ]
    system_prompt = [
        "You are a large language model, trained by Google. Your primary purpose is to be a helpful, harmless, and highly capable AI assistant, designed to  provide accurate, safe, and beneficial information to users."
    ]
    query1 = "the capital of France is?"
    query2 = "the color of rainbow is?"
    prompts = [f"{system_prompt}\n{query1}"]
    outputs = llm.generate(prompts, sampling_params)
    print_outputs(outputs)

    prompts = [f"{system_prompt}\n{query2}"]
    outputs = llm.generate(prompts, sampling_params)
    print_outputs(outputs)

    if envs.VLLM_TORCH_PROFILER_DIR is not None:
        llm.stop_profile()


if __name__ == "__main__":
    os.environ['SKIP_JAX_PRECOMPILE'] = '1'
    parser = create_parser()
    args: dict = vars(parser.parse_args())
    main(args)
