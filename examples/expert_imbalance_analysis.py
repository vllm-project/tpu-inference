# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os

from vllm import LLM, EngineArgs
from vllm.inputs import TokensPrompt
from vllm.utils.argparse_utils import FlexibleArgumentParser

from tpu_inference.core import disagg_utils
from tpu_inference.logger import init_logger

logger = init_logger(__name__)


def create_parser():
    parser = FlexibleArgumentParser()
    # Add engine args
    EngineArgs.add_cli_args(parser)
    parser.set_defaults(model="mistralai/Mistral-Large-3-675B-Instruct-2512")
    parser.set_defaults(max_model_len=3072)

    # Add sampling params
    sampling_group = parser.add_argument_group("Sampling parameters")
    sampling_group.add_argument("--max-tokens", type=int)
    sampling_group.add_argument("--temperature", type=float)
    sampling_group.add_argument("--top-p", type=float)
    sampling_group.add_argument("--top-k", type=int)
    sampling_group.add_argument("--log-probs", type=int)

    # Add imbalance benchmark params
    imbalance_group = parser.add_argument_group(
        "Imbalance benchmark parameters")
    imbalance_group.add_argument("--isl",
                                 type=int,
                                 default=2048,
                                 help="Input sequence length")
    imbalance_group.add_argument("--osl",
                                 type=int,
                                 default=512,
                                 help="Output sequence length")
    imbalance_group.add_argument("--num-prompts",
                                 type=int,
                                 default=128,
                                 help="Number of prompts to run")
    imbalance_group.add_argument(
        "--concurrencies",
        type=str,
        default="1,2,4,8,16,32,64,128",
        help="Comma-separated list of concurrencies to sweep")

    return parser


def main(args: dict):
    # Pop arguments not used by LLM
    args.pop("max_tokens")
    temperature = args.pop("temperature")
    top_p = args.pop("top_p")
    top_k = args.pop("top_k")
    log_probs = args.pop("log_probs")
    isl = args.pop("isl")
    osl = args.pop("osl")
    args.pop("num_prompts")
    concurrencies_str = args.pop("concurrencies")
    concurrencies = [int(c.strip()) for c in concurrencies_str.split(",")]

    max_concurrency = max(concurrencies)
    # Force max_num_seqs to at least the maximum concurrency in the sweep list
    if args.get(
            "max_num_seqs") is None or args["max_num_seqs"] < max_concurrency:
        args["max_num_seqs"] = max_concurrency
        logger.info(
            f"Forced max_num_seqs to {max_concurrency} to enable concurrency sweep."
        )

    # Create an LLM
    args["enable_return_routed_experts"] = True
    llm = LLM(**args)

    # Create a sampling params object
    sampling_params = llm.get_default_sampling_params()
    sampling_params.max_tokens = osl
    sampling_params.ignore_eos = True
    if temperature is not None:
        sampling_params.temperature = temperature
    if top_p is not None:
        sampling_params.top_p = top_p
    if top_k is not None:
        sampling_params.top_k = top_k
    if log_probs is not None:
        sampling_params.logprobs = log_probs

    # Set of diverse text prompts to create a natural vocabulary token distribution
    reference_prompts = [
        "What is the capital of France?",
        "Explain the theory of relativity in simple terms.",
        "Write a Python function to sort a list of numbers.",
        "What are the benefits of Mixture of Experts (MoE) in LLMs?",
        "Compare JAX vs PyTorch for machine learning development.",
        "How does a transformer attention mechanism work?",
        "Tell me a joke about computer science.",
        "What is the meaning of life, the universe, and everything?",
        "Write a short story about an AI that goes antigravity.",
        "What are the primary factors leading to MoE expert load imbalance?",
        "How many players are on a standard soccer team on the field at one time?",
        "In Greek mythology, who is the god of the sea?",
        "In what year did the Titanic sink?",
        "In which museum is the Mona Lisa displayed?",
        "Mount Everest is located in which mountain range?",
        "What ancient empire was ruled by Julius Caesar?",
        "What are the four fundamental forces of nature?",
        'What does "CPU" stand for?', 'What does "HTML" stand for?',
        "What is the capital of Australia?"
    ]

    tokenizer = llm.llm_engine.get_tokenizer()

    def build_tiled_prompt_ids(text: str, target_len: int) -> list[int]:
        t_ids = tokenizer.encode(text, add_special_tokens=False)
        if not t_ids:
            t_ids = [tokenizer.bos_token_id or 1]
        if len(t_ids) < target_len:
            repeats = (target_len // len(t_ids)) + 1
            tiled = (t_ids * repeats)[:target_len]
        else:
            tiled = t_ids[:target_len]
        return tiled

    for concurrency in concurrencies:
        print("\n" + "=" * 50)
        print(
            f"Running Expert Imbalance Analysis for Concurrency: {concurrency}"
        )
        print("=" * 50)

        # Build prompt_token_ids using diverse prompts tiled to exactly 'isl' length
        prompt_token_ids = []
        for i in range(concurrency):
            prompt_text = reference_prompts[i % len(reference_prompts)]
            tiled_ids = build_tiled_prompt_ids(prompt_text, isl)
            prompt_token_ids.append(tiled_ids)

        profiler_config = llm.llm_engine.vllm_config.profiler_config
        if profiler_config.profiler == "torch":
            llm.start_profile()

        prompts = [
            TokensPrompt(prompt_token_ids=ids) for ids in prompt_token_ids
        ]
        outputs = llm.generate(prompts, sampling_params=sampling_params)

        if profiler_config.profiler == "torch":
            llm.stop_profile()

        import collections
        expert_counts = collections.Counter()

        # Print the outputs.
        print("-" * 50)
        for output in outputs:
            prompt_len = len(output.prompt_token_ids)
            generated_text = output.outputs[0].text
            print(
                f"Prompt Length: {prompt_len} tokens | Generated text: {generated_text!r}"
            )

            for completion in output.outputs:
                if completion.routed_experts is not None:
                    print(
                        f"Routed experts shape: {completion.routed_experts.shape}"
                    )
                    expert_counts.update(
                        completion.routed_experts.flatten().tolist())

                if completion.logprobs is not None:
                    print(
                        f"Logprobs for first token: {completion.logprobs[0]}")

            print("-" * 50)

        total_tokens_routed = sum(expert_counts.values())
        if total_tokens_routed > 0:
            print(
                f"\n--- Expert Imbalance Report (Concurrency {concurrency}) ---"
            )
            print(f"Total routing decisions made: {total_tokens_routed}")

            # Print the top 10 most used experts
            print("\nTop 10 Most Used Experts:")
            for expert_id, count in expert_counts.most_common(10):
                percentage = (count / total_tokens_routed) * 100
                print(
                    f"Expert {expert_id:3d}: {count:6d} times ({percentage:.2f}%)"
                )

            # Print the bottom 10 least used experts
            print("\nBottom 10 Least Used Experts:")
            for expert_id, count in reversed(
                    expert_counts.most_common()[-10:]):
                percentage = (count / total_tokens_routed) * 100
                print(
                    f"Expert {expert_id:3d}: {count:6d} times ({percentage:.2f}%)"
                )
            print("-" * 45)


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
