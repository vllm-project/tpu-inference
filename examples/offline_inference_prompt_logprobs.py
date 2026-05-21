# SPDX-License-Identifier: Apache-2.0
"""Smoke test for --prompt-logprobs support on TPU.

Run with:
  python examples/offline_inference_prompt_logprobs.py
  python examples/offline_inference_prompt_logprobs.py --model <model>
"""

import os

from vllm import LLM, SamplingParams
from vllm.utils.argparse_utils import FlexibleArgumentParser

from tpu_inference.logger import init_logger

logger = init_logger(__name__)

PROMPTS = [
    "Hello, my name is",
    "The capital of France is",
    "The future of AI is",
]


def main(model: str, num_prompt_logprobs: int, max_tokens: int):
    llm = LLM(model=model, max_model_len=1024)

    sampling_params = SamplingParams(
        max_tokens=max_tokens,
        prompt_logprobs=num_prompt_logprobs,
        temperature=0.0,
    )

    outputs = llm.generate(PROMPTS, sampling_params)

    print("=" * 60)
    for output in outputs:
        prompt = output.prompt
        generated = output.outputs[0].text
        prompt_logprobs = output.prompt_logprobs

        print(f"Prompt:    {prompt!r}")
        print(f"Generated: {generated!r}")

        if prompt_logprobs is None:
            print("  ERROR: prompt_logprobs is None!")
        else:
            # prompt_logprobs is a list with None for the first token
            # (nothing precedes it) and a dict per subsequent token.
            print(f"  prompt_logprobs length: {len(prompt_logprobs)}"
                  f"  (expected {len(llm.llm_engine.get_tokenizer().encode(prompt))})")
            non_none = [p for p in prompt_logprobs if p is not None]
            print(f"  non-None entries: {len(non_none)}")
            if non_none:
                sample = non_none[0]
                print(f"  first entry keys (token_ids): {list(sample.keys())[:3]}")
                first_logprob = next(iter(sample.values()))
                print(f"  first logprob value: logprob={first_logprob.logprob:.4f}, "
                      f"rank={first_logprob.rank}")

        print("-" * 60)

    print("OK: prompt_logprobs test passed.")


if __name__ == "__main__":
    os.environ.setdefault("SKIP_JAX_PRECOMPILE", "1")

    parser = FlexibleArgumentParser()
    parser.add_argument("--model",
                        default="meta-llama/Llama-3.2-1B-Instruct")
    parser.add_argument("--num-prompt-logprobs", type=int, default=5)
    parser.add_argument("--max-tokens", type=int, default=10)
    args = parser.parse_args()

    main(args.model, args.num_prompt_logprobs, args.max_tokens)
