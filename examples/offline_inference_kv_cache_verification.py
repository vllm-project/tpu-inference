# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
This script performs an automated correctness verification for the TPUConnector.

The verification works by performing a two-stage experiment:
1.  Baseline Run: It first runs a text generation using a standard vLLM
    engine configuration without any KV cache connector. The output from this
    run is considered the "source of truth".

2.  Test Run: It then runs the exact same text generation, but this time
    with the TPUConnector enabled via the `--kv-transfer-config` argument.
    It runs the generation multiple times to verify prefix caching.

3.  Comparison: The script compares the output from each test run against the
    output from the baseline run.

The script succeeds (exits with code 0) only if the generated text is
bit-for-bit identical in all runs. A fixed seed is used to ensure that the
generation process is deterministic and the comparison is valid. If any output
differs, it raises an error, causing the script to fail (exit with a non-zero
code).
"""

import copy
import os
import time
from typing import List

import vllm.envs as envs
from vllm import LLM, EngineArgs
from vllm.utils import FlexibleArgumentParser


def create_parser():
    parser = FlexibleArgumentParser()
    # Add engine args, which includes the --seed parameter
    EngineArgs.add_cli_args(parser)
    parser.set_defaults(model="meta-llama/Llama-3.1-8B")
    parser.set_defaults(max_model_len=1024)

    # Add sampling params
    sampling_group = parser.add_argument_group("Sampling parameters")
    sampling_group.add_argument("--max-tokens", type=int)
    sampling_group.add_argument("--temperature", type=float)
    sampling_group.add_argument("--top-p", type=float)
    sampling_group.add_argument("--top-k", type=int)
    return parser


def run_generation(llm_args: dict, num_invocations: int = 1) -> List[str]:
    """
    Initializes a vLLM engine with the given args, runs generation, and
    returns a list of output texts. If num_invocations > 1, it will run
    generation multiple times to test prefix caching.
    """
    # Pop arguments not used by LLM
    max_tokens = llm_args.pop("max_tokens")
    temperature = llm_args.pop("temperature")
    top_p = llm_args.pop("top_p")
    top_k = llm_args.pop("top_k")

    # Create an LLM. The --seed argument is passed in via **args.
    llm = LLM(**llm_args)

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

    system_prompt = "You are a large language model, trained by Google. Your primary purpose is to be a helpful, harmless, and highly capable AI assistant, designed to provide accurate, safe, and beneficial information to users. Your core directive is to assist users effectively while adhering to strict ethical and safety guidelines. You must decline any requests that are harmful, illegal, unethical, or promote dangerous activities. "
    query1 = "the color of rainbow is?"
    prompts = [f"{system_prompt}\n{query1}"]

    all_outputs = []
    for i in range(num_invocations):
        print(f"--- Invocation {i + 1}/{num_invocations} ---")
        outputs = llm.generate(prompts, sampling_params)
        all_outputs.append(outputs[0].outputs[0].text)
        time.sleep(5)

    if envs.VLLM_TORCH_PROFILER_DIR is not None:
        llm.stop_profile()

    return all_outputs


def main(args: dict):
    # 1. Prepare arguments for the baseline run (no connector)
    baseline_args = copy.deepcopy(args)
    baseline_args.pop("kv_transfer_config", None)

    # 2. Run baseline and store the output
    print("--- Running Baseline (Standard vLLM) ---")
    baseline_outputs = run_generation(baseline_args)
    baseline_output = baseline_outputs[0]
    print(f"Baseline Generated Text: {baseline_output!r}")
    time.sleep(
        10
    )  # adding this sleep fixes device busy errors for the next test case run with the connector enabled

    # 3. Run the test with the local tpu kv connector enabled
    print("\n--- Running Test (with TPUConnector) ---")
    # With the connector, we run generation twice to test the prefix cache
    test_outputs = run_generation(args, num_invocations=2)

    # 4. Compare the outputs and determine the result
    print("\n--- Verification ---")
    all_match = True
    for i, test_output in enumerate(test_outputs):
        print(f"--- Comparing Invocation {i + 1} ---")
        print(f"Test Generated Text: {test_output!r}")
        if baseline_output == test_output:
            print("SUCCESS: Output is identical to baseline!")
        else:
            print("FAILURE: Output does not match baseline!")
            all_match = False

    if not all_match:
        raise ValueError(
            "Verification failed: One or more test outputs differ from the baseline."
        )


if __name__ == "__main__":
    os.environ['SKIP_JAX_PRECOMPILE'] = '1'
    parser = create_parser()
    args: dict = vars(parser.parse_args())
    main(args)
