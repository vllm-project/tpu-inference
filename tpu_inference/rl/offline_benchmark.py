# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the tpu-inference project

"""Offline benchmarking script for RL VllmSampler in tpu-inference."""

import argparse
import asyncio
import logging
import time
from types import SimpleNamespace
from typing import Any

import numpy as np

from tpu_inference.rl.vllm_sampler import (
  VllmSampler,
  VllmSamplerConfig,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def generate_synthetic_prompts(
    num_groups: int, group_size: int, prompt_len_words: int = 150
) -> list[Any]:
  """Constructs GRPO-style shared prefix prompt requests using standard duck-typed objects."""
  base_math_problems = [
      "Let f(x) = x^3 - 3x + 2. Find all critical points and evaluate local extrema using the second derivative test.",
      "Consider a Markov chain with transition matrix P. Compute the stationary distribution pi such that pi P = pi.",
      "Prove by mathematical induction that sum_{k=1}^n k^2 = n(n+1)(2n+1)/6 for all positive integers n.",
      "Evaluate the definite integral integral_0^infinity (sin(x)/x) dx using contour integration.",
  ]

  requests = []
  total_prompts = num_groups * group_size

  for i in range(total_prompts):
    group_idx = i // group_size
    problem_template = base_math_problems[group_idx % len(base_math_problems)]
    padding = " " + " ".join([f"token_{w}" for w in range(prompt_len_words)])
    full_prompt = f"Group #{group_idx} Problem: {problem_template}{padding}\n\nSolution Step-by-Step:"

    req = SimpleNamespace(
        prompt=full_prompt,
        request_id=f"grpo_grp{group_idx}_idx{i}",
        route_key=f"group_prefix_{group_idx}",
        sampling_params=SimpleNamespace(
            max_tokens=256,
            temperature=0.7,
            top_p=0.95,
            return_logprobs=True,
        ),
    )
    requests.append(req)

  return requests


async def run_offline_benchmark(args: argparse.Namespace) -> None:
  """Runs full offline rollout sampling benchmark."""
  num_groups = max(1, args.num_prompts // args.group_size)
  total_prompts = num_groups * args.group_size

  logger.info("=" * 70)
  logger.info("Starting Offline VllmSampler Rollout Benchmark")

  logger.info("Model:                 %s", args.model_path)
  logger.info("Tensor Parallel Size:  %d", args.tensor_parallel_size)
  logger.info("Total Prompts:         %d (%d groups x %d rollout streams)", total_prompts, num_groups, args.group_size)
  logger.info("Max Target Tokens:     %d", args.max_tokens)
  logger.info("Prefix Caching:        %s", args.enable_prefix_caching)
  logger.info("=" * 70)

  config = VllmSamplerConfig(
      model_path=args.model_path,
      tensor_parallel_size=args.tensor_parallel_size,
      max_num_seqs=args.max_num_seqs,
      max_num_batched_tokens=args.max_num_batched_tokens,
      hbm_utilization=args.hbm_utilization,
      enable_prefix_caching=args.enable_prefix_caching,
      weight_dtype=args.weight_dtype,
  )

  start_init_t = time.perf_counter()
  sampler = VllmSampler(config=config)
  if not getattr(args, "dry_run", False):
    await sampler.start()
    init_elapsed = time.perf_counter() - start_init_t
  else:
    init_elapsed = 0.12
  logger.info("Engine Initialization Complete in %.2f seconds.", init_elapsed)

  requests = generate_synthetic_prompts(
      num_groups=num_groups,
      group_size=args.group_size,
      prompt_len_words=args.prompt_words,
  )
  for req in requests:
    if getattr(req, "sampling_params", None):
      req.sampling_params.max_tokens = args.max_tokens

  if args.warmup_prompts > 0 and not getattr(args, "dry_run", False):
    logger.info("Running warmup on %d prompts...", args.warmup_prompts)
    warmup_reqs = requests[: args.warmup_prompts]
    await sampler.sample(warmup_reqs)
    logger.info("Warmup complete.")

  logger.info("Launching sampling batch of %d requests...", len(requests))
  start_sample_t = time.perf_counter()
  if not getattr(args, "dry_run", False):
    results = await sampler.sample(requests)
  else:
    results = []
    for req in requests:
      n_toks = args.max_tokens
      dummy_toks = np.arange(100, 100 + n_toks, dtype=np.int32)
      dummy_lps = np.full(n_toks, -0.25, dtype=np.float32)
      results.append(
          SimpleNamespace(
              request_id=req.request_id,
              text="Step 1: Simplify expression... Step 2: Critical points at x=-1, x=1. Therefore Q.E.D.",
              token_ids=dummy_toks,
              logprobs=dummy_lps,
              cumulative_logprob=float(-0.25 * n_toks),
              route_key=req.route_key,
              error=None,
          )
      )
    await asyncio.sleep(0.05)
  total_sample_t = time.perf_counter() - start_sample_t

  successful_requests = [r for r in results if getattr(r, "error", None) is None]
  failed_requests = [r for r in results if getattr(r, "error", None) is not None]

  generated_tokens_per_req = [
      len(r.token_ids) if getattr(r, "token_ids", None) is not None else 0 for r in successful_requests
  ]
  total_generated_tokens = sum(generated_tokens_per_req)

  logprob_check_passed = all(
      getattr(r, "logprobs", None) is not None and len(r.logprobs) == len(r.token_ids)
      for r in successful_requests
  )

  req_throughput = len(successful_requests) / total_sample_t if total_sample_t > 0 else 0.0
  tok_throughput = total_generated_tokens / total_sample_t if total_sample_t > 0 else 0.0

  print("\n" + "=" * 65)
  print("         VLLMSAMPLER OFFLINE ROLLOUT BENCHMARK REPORT         ")
  print("=" * 65)
  print(f"Engine Startup Time (s):        {init_elapsed:.2f}")
  print(f"Total Sampling Wall Time (s):   {total_sample_t:.3f}")
  print(f"Total Requests Processed:      {len(results)}")
  print(f"Successful Requests:           {len(successful_requests)}")
  print(f"Failed Requests:               {len(failed_requests)}")
  print(f"Total Output Tokens Generated:  {total_generated_tokens}")
  print(f"Avg Output Tokens / Request:   {np.mean(generated_tokens_per_req):.1f}")
  print(f"Request Throughput (req/s):    {req_throughput:.2f}")
  print(f"Output Token Throughput (tok/s): {tok_throughput:.2f}")
  print(f"Logprobs Validity Verification: {'PASSED' if logprob_check_passed else 'FAILED'}")
  print("=" * 65 + "\n", flush=True)

  if failed_requests:
    logger.error("Sample error output: %s", failed_requests[0].error)

  if not getattr(args, "dry_run", False):
    await sampler.stop()


def main() -> None:
  parser = argparse.ArgumentParser(
      description="Offline Rollout Benchmarking for VllmSampler in tpu-inference."
  )
  parser.add_argument("--model_path", type=str, default="Qwen/Qwen2.5-1.5B", help="Model path or HF ID.")
  parser.add_argument("--tensor_parallel_size", type=int, default=1, help="Tensor parallelism size.")
  parser.add_argument("--num_prompts", type=int, default=64, help="Total benchmark prompts.")
  parser.add_argument("--group_size", type=int, default=8, help="GRPO group size per prompt.")
  parser.add_argument("--prompt_words", type=int, default=150, help="Prompt token word padding.")
  parser.add_argument("--max_tokens", type=int, default=256, help="Max tokens generated per completion.")
  parser.add_argument("--max_num_seqs", type=int, default=256, help="Max batched sequences in vLLM.")
  parser.add_argument("--max_num_batched_tokens", type=int, default=8192, help="Max batched tokens.")
  parser.add_argument("--hbm_utilization", type=float, default=0.80, help="Target HBM utilization fraction.")
  parser.add_argument("--enable_prefix_caching", action="store_true", default=True, help="Enable vLLM prefix cache.")
  parser.add_argument("--weight_dtype", type=str, default="bfloat16", help="Model weight dtype.")
  parser.add_argument("--warmup_prompts", type=int, default=4, help="Warmup prompts count.")
  parser.add_argument("--dry_run", action="store_true", default=False, help="Run dry run benchmark without GPU/TPU engine initialization.")

  args = parser.parse_args()
  asyncio.run(run_offline_benchmark(args))


if __name__ == "__main__":
  main()
