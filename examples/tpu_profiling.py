# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Implements profiling for vLLM on TPU VMs using the JAX profiler.
# NOTE: you will need the tensorboard-plugin-profile python package to
# visualize the results in TensorBoard.
# Please see docs/profiler.md for more details.
# Usage example for prefilling 1 request of 1024 tokens:
# python3 examples/tpu_profiling.py --input-len 1024 --output-len 1   --batch-size 1
# Usage example for decoding 256 requests of 1 token each:
# python3 examples/tpu_profiling.py --input-len 1 --output-len 1 --batch-size=256

import argparse
import dataclasses
import os
import time

import numpy as np
from tqdm import tqdm
from vllm import LLM, SamplingParams
from vllm.engine.arg_utils import EngineArgs
from vllm.inputs import PromptType
from vllm.utils.argparse_utils import FlexibleArgumentParser

DURATION_MS = int(os.getenv("VLLM_TPU_PROFILE_DURATION_MS", 3000))
DELAY_MS = int(os.getenv("VLLM_TPU_PROFILE_DELAY_MS", 0))


def main(args: argparse.Namespace):
    print(args)

    # Profile
    profile_dir = args.profile_result_dir
    print(f"Profiling (results will be saved to '{profile_dir}')...")
    args.setdefault("profiler_config.profiler", "torch")
    args.setdefault("profiler_config.torch_profiler_dir", profile_dir)

    engine_args = EngineArgs.from_cli_args(args)
    llm = LLM(**dataclasses.asdict(engine_args))

    sampling_params = SamplingParams(
        temperature=0.0,
        ignore_eos=True,
        max_tokens=args.output_len,
    )
    print(sampling_params)
    dummy_prompt_token_ids = np.random.randint(10000,
                                               size=(args.batch_size,
                                                     args.input_len))
    dummy_prompts: list[PromptType] = [{
        "prompt_token_ids": batch
    } for batch in dummy_prompt_token_ids.tolist()]

    def run_to_completion():
        start_time = time.perf_counter()
        llm.generate(dummy_prompts,
                     sampling_params=sampling_params,
                     use_tqdm=False)
        end_time = time.perf_counter()
        latency = end_time - start_time
        return latency

    # Warmup
    print("Warming up...")
    warmup_latencies = []
    for _ in tqdm(range(args.num_iters_warmup), desc="Warmup iterations"):
        warmup_latencies.append(run_to_completion())
    print(f"Average warmup latency: {np.mean(warmup_latencies):.4f}s")

    # Enable tracing on server
    llm.start_profile()
    if DELAY_MS == 0:
        time.sleep(1.0)
    profile_latencies = []
    for _ in tqdm(range(args.num_iters), desc="Profile iterations"):
        profile_latencies.append(run_to_completion())
    llm.stop_profile()
    print(f"Average profile latency: {np.mean(profile_latencies):.4f}s")

    return


def parse_args():
    parser = FlexibleArgumentParser(
        description="Benchmark the latency of processing a single batch of "
        "requests till completion.")
    parser.add_argument("--input-len", type=int, default=32)
    parser.add_argument("--output-len", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--num-iters-warmup",
        type=int,
        default=5,
        help="Number of iterations to run for warmup.",
    )
    parser.add_argument(
        "--num-iters",
        type=int,
        default=1,
        help="Number of iterations to run for profiling.",
    )
    parser.add_argument(
        "--profile-result-dir",
        type=str,
        default="profiles",
        help=("path to save the JAX profiler output. Can be visualized "
              "with ui.perfetto.dev, Tensorboard, or XProf"),
    )

    parser = EngineArgs.add_cli_args(parser)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
