# SPDX-License-Identifier: Apache-2.0
"""
This script performs an automated correctness verification for the TPUOffloadConnector.

The verification works by performing a two-stage experiment for multiple prompts:
1.  Baseline Run: For each prompt, it first runs a text generation using a
    standard vLLM engine configuration without any KV cache connector. The
    output from this run is considered the "source of truth".

2.  Test Run: It then runs the exact same text generation, but this time
    with the TPUOffloadConnector enabled via the `--kv-transfer-config` argument.
    It runs the generation twice to verify prefix caching.

3.  Comparison: The script compares the output from each test run against the
    output from the baseline run for that prompt.

The script succeeds (exits with code 0) only if the generated text is
bit-for-bit identical in all runs for all prompts. A fixed seed is used to
ensure that the generation process is deterministic and the comparison is
valid. If any output differs, it raises an error, causing the script to fail
(exit with a non-zero code).
"""

import copy
import os
import time
from typing import List, Tuple

import vllm.envs as envs
from vllm import LLM, EngineArgs, SamplingParams
from vllm.utils.argparse_utils import FlexibleArgumentParser


def create_parser():
    parser = FlexibleArgumentParser()
    # Add engine args, which includes the --seed parameter
    EngineArgs.add_cli_args(parser)
    parser.set_defaults(model="meta-llama/Llama-3.1-8B")
    parser.set_defaults(max_model_len=1024)

    # Add sampling params
    sampling_group = parser.add_argument_group("Sampling parameters")
    sampling_group.add_argument("--max-tokens", type=int)
    sampling_group.add_argument("--top-p", type=float)
    sampling_group.add_argument("--top-k", type=int)
    return parser


def setup_llm(llm_args: dict) -> Tuple[LLM, SamplingParams]:
    """
    Initializes a vLLM engine and sampling parameters from the given args.
    """
    args_copy = copy.deepcopy(llm_args)
    # Pop arguments not used by LLM
    max_tokens = args_copy.pop("max_tokens")
    top_p = args_copy.pop("top_p")
    top_k = args_copy.pop("top_k")

    # Create an LLM. The --seed argument is passed in via **args.
    llm = LLM(**args_copy)

    # Create a sampling params object
    sampling_params = llm.get_default_sampling_params()
    sampling_params.temperature = 0
    sampling_params.ignore_eos = True
    if max_tokens is not None:
        sampling_params.max_tokens = max_tokens
    if top_p is not None:
        sampling_params.top_p = top_p
    if top_k is not None:
        sampling_params.top_k = top_k

    return llm, sampling_params


def run_invocations(llm: LLM, sampling_params: SamplingParams,
                    prompts: List[str], num_invocations: int) -> List[str]:
    """
    Runs generation on the given LLM object for a specified number of
    invocations and returns the output texts.
    """
    if envs.VLLM_TORCH_PROFILER_DIR is not None:
        llm.start_profile()

    all_outputs = []
    for i in range(num_invocations):
        print(f"--- Invocation {i + 1}/{num_invocations} ---")
        outputs = llm.generate(prompts, sampling_params)
        all_outputs.append(outputs[0].outputs[0].text)
        # reset prefix cache
        llm.llm_engine.engine_core.reset_prefix_cache()
        time.sleep(5)

    if envs.VLLM_TORCH_PROFILER_DIR is not None:
        llm.stop_profile()

    return all_outputs


def main(args: dict):
    # prompt lesser than the kv cache block size
    short_input_prompt = "Google is a "

    system_prompt = "You are a large language model, trained by Google. Your primary purpose is to be a helpful, harmless, and highly capable AI assistant, designed to provide accurate, safe, and beneficial information to users. Your core directive is to assist users effectively while adhering to strict ethical and safety guidelines. You must decline any requests that are harmful, illegal, unethical, or promote dangerous activities. "
    query = "the color of rainbow is?"
    input_prompt = f"{system_prompt}\n{query}"

    prompts_to_test = [
        ("Short Prompt", [short_input_prompt]),
        ("Prompt", [input_prompt]),
    ]

    all_tests_passed = True
    for prompt_name, prompts in prompts_to_test:
        print(f"\n\n===== Running verification for: {prompt_name} =====")
        print(f"Prompt: {prompts[0]}")

        # 1. Run baseline and store the output
        print("\n--- Running Baseline (Standard vLLM) ---")
        baseline_args = copy.deepcopy(args)
        baseline_args.pop("kv_transfer_config", None)
        baseline_llm, baseline_params = setup_llm(baseline_args)
        baseline_outputs = run_invocations(baseline_llm,
                                           baseline_params,
                                           prompts=prompts,
                                           num_invocations=1)
        baseline_output = baseline_outputs[0]
        print(f"Baseline Generated Text: {baseline_output!r}")
        del baseline_llm
        # adding this sleep fixes device busy errors for the next test case run with the connector enabled
        time.sleep(10)

        # 2. Run the test with the local tpu kv connector enabled
        print("\n--- Running Test (with TPUOffloadConnector) ---")
        # With the connector, we run generation twice to test the prefix cache
        test_llm, test_params = setup_llm(args)
        test_outputs = run_invocations(test_llm,
                                       test_params,
                                       prompts=prompts,
                                       num_invocations=2)
        del test_llm

        # 3. Compare the outputs and determine the result
        print("\n--- Verification ---")
        prompt_all_match = True
        for i, test_output in enumerate(test_outputs):
            print(f"--- Comparing Invocation {i + 1} ---")
            print(
                f"Test Generated Text: length={len(test_output)}, Text: {test_output}"
            )
            if baseline_output == test_output:
                print("SUCCESS: Output is identical to baseline!")
            else:
                print("FAILURE: Output does not match baseline!")
                prompt_all_match = False

        if not prompt_all_match:
            all_tests_passed = False
            print(f"===== Verification FAILED for: {prompt_name} =====")
        else:
            print(f"===== Verification SUCCEEDED for: {prompt_name} =====")

        time.sleep(10)

    if not all_tests_passed:
        raise ValueError(
            "Verification failed: One or more test outputs differ from the baseline."
        )
    else:
        print("\n\n===== All verification runs passed successfully! =====")


if __name__ == "__main__":
    os.environ['SKIP_JAX_PRECOMPILE'] = '1'
    parser = create_parser()
    args: dict = vars(parser.parse_args())
    main(args)
