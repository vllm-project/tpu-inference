# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Reproducible Qwen3-0.6B throughput benchmark via vLLM on TPU.

Used as the "real battle" comparison vehicle: run this once before evolving
a kernel to record the baseline, then run again after applying the evolved
diff. Same fixed prompt set + same sampling params + same warmup discipline.

Output is a JSON blob with: model, num_prompts, total_tokens_generated,
wall_time_s, throughput_tokens_per_s, per_prompt_seconds.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Fixed prompt set — chosen to have stable input lengths and exercise
# both prefill (long prompts) and decode (short prompts).
_PROMPTS = [
    "Write a one-paragraph explanation of how a transformer attention "
    "block computes its output, naming the matrices involved.",
    "Summarize the difference between BF16 and FP32 numerical formats "
    "for ML training.",
    "List five common pitfalls when porting a CUDA kernel to a TPU.",
    "Explain why ragged paged attention is preferable to dense attention "
    "for serving LLMs at scale.",
    "Describe how speculative decoding differs from regular autoregressive "
    "decoding.",
    "What are the four main components of a vLLM serving stack?",
    "Compare grouped-query attention and multi-head attention.",
    "Define VMEM in the context of a TPU Pallas kernel.",
    "Why does prefill latency dominate first-token latency in LLM serving?",
    "List three reasons why one might choose evolutionary search over "
    "Bayesian optimization for kernel autotuning.",
]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model", default="Qwen/Qwen3-0.6B")
    p.add_argument("--max-model-len", type=int, default=2048)
    p.add_argument("--max-tokens", type=int, default=128)
    p.add_argument("--tensor-parallel-size", type=int, default=1)
    p.add_argument("--num-warmup-rounds",
                   type=int,
                   default=1,
                   help="Untimed warmup generations before measurement.")
    p.add_argument("--num-measure-rounds",
                   type=int,
                   default=2,
                   help="Timed measurement rounds; result is the mean.")
    p.add_argument("--output",
                   type=Path,
                   default=Path("/tmp/qwen3_bench.json"))
    p.add_argument("--label",
                   default="baseline",
                   help="Label written into the output JSON for comparison.")
    p.add_argument("--enforce-eager",
                   action="store_true",
                   help="Skip JAX compile cache priming — slower start, but "
                   "guarantees a fresh measurement of the kernel under test.")
    args = p.parse_args(argv)

    # Import vLLM lazily so the script can be parsed without vLLM installed.
    os.environ.setdefault("VLLM_USE_V1", "1")
    from vllm import LLM
    from vllm.sampling_params import SamplingParams

    print(
        f"[{args.label}] Loading {args.model} (tp={args.tensor_parallel_size}, "
        f"max_model_len={args.max_model_len})...",
        file=sys.stderr)
    load_t0 = time.time()
    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
        enforce_eager=args.enforce_eager,
    )
    load_secs = time.time() - load_t0
    print(f"[{args.label}] LLM ready in {load_secs:.1f}s.", file=sys.stderr)

    sampling = SamplingParams(
        max_tokens=args.max_tokens,
        temperature=0.0,  # greedy — reproducible
        top_p=1.0,
    )

    def _generate() -> tuple[float, int]:
        t0 = time.time()
        outputs = llm.generate(_PROMPTS, sampling, use_tqdm=False)
        elapsed = time.time() - t0
        total_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
        return elapsed, total_tokens

    print(f"[{args.label}] Warmup ({args.num_warmup_rounds} round/s)...",
          file=sys.stderr)
    for _ in range(args.num_warmup_rounds):
        _generate()

    print(f"[{args.label}] Measure ({args.num_measure_rounds} round/s)...",
          file=sys.stderr)
    times: list[float] = []
    tokens_list: list[int] = []
    for _ in range(args.num_measure_rounds):
        t, tok = _generate()
        times.append(t)
        tokens_list.append(tok)

    mean_time = sum(times) / len(times)
    mean_tokens = sum(tokens_list) // len(tokens_list)
    throughput = mean_tokens / mean_time if mean_time > 0 else 0.0
    per_prompt_secs = mean_time / len(_PROMPTS)

    result = {
        "label": args.label,
        "model": args.model,
        "tensor_parallel_size": args.tensor_parallel_size,
        "max_model_len": args.max_model_len,
        "max_tokens": args.max_tokens,
        "num_prompts": len(_PROMPTS),
        "num_warmup_rounds": args.num_warmup_rounds,
        "num_measure_rounds": args.num_measure_rounds,
        "mean_total_tokens": mean_tokens,
        "mean_wall_time_s": mean_time,
        "per_round_wall_times_s": times,
        "per_round_tokens": tokens_list,
        "throughput_tokens_per_s": throughput,
        "per_prompt_seconds": per_prompt_secs,
        "load_time_s": load_secs,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))

    print()
    print("=" * 78)
    print(f"[{args.label}] Qwen3-0.6B throughput benchmark")
    print(f"  model: {args.model}  tp={args.tensor_parallel_size}  "
          f"max_model_len={args.max_model_len}")
    print(f"  num_prompts: {len(_PROMPTS)}  max_tokens: {args.max_tokens}")
    print(f"  wall_time:       {mean_time:7.3f} s")
    print(f"  tokens_generated:{mean_tokens:7d}")
    print(f"  throughput:      {throughput:7.2f} tok/s")
    print(f"  per_prompt:      {per_prompt_secs:7.3f} s")
    print(f"Result JSON: {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
