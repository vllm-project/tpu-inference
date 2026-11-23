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
"""Measure speculative decoding acceptance and speedup on RL-style rollouts.

Each configuration runs in a fresh process: a non-speculative baseline per
``--max-num-seqs`` value, then one speculative run per
(``--num-speculative-tokens``, ``--max-num-seqs``) pair. Rollouts default to
RL-style sampling (temperature 1.0, no top-k/top-p truncation, several samples
per prompt, long outputs that stop at EOS).

Reported per run:
  * acceptance rate          accepted draft tokens / draft tokens
  * mean acceptance length   1 + accepted / drafts (tokens per target step)
  * per-position acceptance  accepted at draft position i / drafts
  * output tokens/s and wall-clock speedup over the matching baseline

Prefix caching is off by default in every run. On TPU, hybrid
(mamba / linear-attention) models silently lose prefix caching when
speculative decoding is enabled, so leaving it on in the baseline only would
make the comparison unfair. The effective setting is reported per run.

Example (Qwen3.5 MTP on one host):
  python scripts/benchmarking/spec_decode/benchmark_spec_decode.py \\
      --model Qwen/Qwen3.5-4B --method mtp \\
      --num-speculative-tokens 1 3 --max-num-seqs 64 256 \\
      --num-prompts 128 --n 8 --max-tokens 4096 \\
      --output-json mtp_rl.json

Every run pays full engine start-up (weight load and XLA precompilation).
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import queue as queue_lib
import sys
import time
import traceback
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Optional

NUM_DRAFTS = "vllm:spec_decode_num_drafts"
NUM_DRAFT_TOKENS = "vllm:spec_decode_num_draft_tokens"
NUM_ACCEPTED_TOKENS = "vllm:spec_decode_num_accepted_tokens"
ACCEPTED_PER_POS = "vllm:spec_decode_num_accepted_tokens_per_pos"

Conversation = list[dict[str, str]]


@dataclass
class SpecDecodeCounters:
    """Cumulative spec-decode counters, summed across engines."""
    num_drafts: int = 0
    num_draft_tokens: int = 0
    num_accepted_tokens: int = 0
    accepted_per_pos: list[int] = field(default_factory=list)

    def __sub__(self, other: SpecDecodeCounters) -> SpecDecodeCounters:
        width = max(len(self.accepted_per_pos), len(other.accepted_per_pos))
        mine = _pad(self.accepted_per_pos, width)
        theirs = _pad(other.accepted_per_pos, width)
        return SpecDecodeCounters(
            num_drafts=self.num_drafts - other.num_drafts,
            num_draft_tokens=self.num_draft_tokens - other.num_draft_tokens,
            num_accepted_tokens=(self.num_accepted_tokens -
                                 other.num_accepted_tokens),
            accepted_per_pos=[a - b for a, b in zip(mine, theirs)],
        )


@dataclass
class SpecDecodeSummary:
    acceptance_rate: float
    mean_acceptance_length: float
    per_position_acceptance: list[float]


def _pad(values: list[int], width: int) -> list[int]:
    return list(values) + [0] * (width - len(values))


def collect_spec_decode_counters(metrics: Iterable[Any]) -> SpecDecodeCounters:
    """Sums spec-decode counters from ``LLM.get_metrics()`` output.

    With data parallelism each engine reports its own labelled series, so
    counters are summed and per-position vectors are added element-wise.
    """
    counters = SpecDecodeCounters()
    for metric in metrics:
        if metric.name == NUM_DRAFTS:
            counters.num_drafts += int(metric.value)
        elif metric.name == NUM_DRAFT_TOKENS:
            counters.num_draft_tokens += int(metric.value)
        elif metric.name == NUM_ACCEPTED_TOKENS:
            counters.num_accepted_tokens += int(metric.value)
        elif metric.name == ACCEPTED_PER_POS:
            width = max(len(counters.accepted_per_pos), len(metric.values))
            counters.accepted_per_pos = [
                a + int(b)
                for a, b in zip(_pad(counters.accepted_per_pos, width),
                                _pad(metric.values, width))
            ]
    return counters


def summarize_counters(counters: SpecDecodeCounters) -> SpecDecodeSummary:
    if counters.num_drafts <= 0 or counters.num_draft_tokens <= 0:
        raise ValueError(
            "No speculative drafts were recorded. Check that speculative "
            "decoding is enabled and the engine runs with "
            "disable_log_stats=False.")
    return SpecDecodeSummary(
        acceptance_rate=(counters.num_accepted_tokens /
                         counters.num_draft_tokens),
        mean_acceptance_length=(
            1 + counters.num_accepted_tokens / counters.num_drafts),
        per_position_acceptance=[
            accepted / counters.num_drafts
            for accepted in counters.accepted_per_pos
        ],
    )


def compute_speedup(baseline_tok_per_s: float, spec_tok_per_s: float) -> float:
    if baseline_tok_per_s <= 0:
        raise ValueError(
            f"Baseline throughput must be positive, got {baseline_tok_per_s}")
    return spec_tok_per_s / baseline_tok_per_s


def to_conversation(record: Any, prompt_field: str) -> Conversation:
    """Turns a dataset row or JSONL record into a chat conversation.

    Accepts a bare string, a dict holding ``messages`` (an OpenAI-style list of
    {"role", "content"}), or a dict holding a string under ``prompt_field``.
    """
    if isinstance(record, str):
        text = record
    elif isinstance(record, dict) and isinstance(record.get("messages"), list):
        messages = record["messages"]
        if not messages or not all(
                isinstance(m, dict) and "role" in m and "content" in m
                for m in messages):
            raise ValueError(f"Malformed 'messages' in record: {record!r}")
        return [{"role": m["role"], "content": m["content"]} for m in messages]
    elif isinstance(record, dict) and isinstance(record.get(prompt_field),
                                                 str):
        text = record[prompt_field]
    else:
        raise ValueError(f"Record has neither 'messages' nor a string "
                         f"'{prompt_field}' field: {record!r}")
    if not text.strip():
        raise ValueError("Empty prompt text")
    return [{"role": "user", "content": text}]


def load_conversations(args: argparse.Namespace) -> list[Conversation]:
    if args.prompts_jsonl:
        with open(args.prompts_jsonl, encoding="utf-8") as f:
            records = [json.loads(line) for line in f if line.strip()]
    else:
        from datasets import load_dataset
        records = load_dataset(args.dataset,
                               args.dataset_subset,
                               split=args.dataset_split)
    conversations = []
    for record in records:
        conversations.append(to_conversation(record, args.prompt_field))
        if len(conversations) == args.num_prompts:
            break
    if len(conversations) < args.num_prompts:
        raise ValueError(f"Requested {args.num_prompts} prompts but only "
                         f"{len(conversations)} are available")
    return conversations


def build_engine_kwargs(args: argparse.Namespace, max_num_seqs: int,
                        num_speculative_tokens: int) -> dict[str, Any]:
    """Engine kwargs for one run; ``num_speculative_tokens=0`` is baseline."""
    kwargs: dict[str, Any] = {
        "model": args.model,
        "tensor_parallel_size": args.tensor_parallel_size,
        "max_model_len": args.max_model_len,
        "max_num_seqs": max_num_seqs,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "enable_prefix_caching": args.enable_prefix_caching,
        "async_scheduling": args.async_scheduling,
        "seed": args.seed,
        "disable_log_stats": False,  # Spec-decode counters need stats.
    }
    if args.kv_cache_dtype:
        kwargs["kv_cache_dtype"] = args.kv_cache_dtype
    if args.max_num_batched_tokens:
        kwargs["max_num_batched_tokens"] = args.max_num_batched_tokens
    kwargs.update(json.loads(args.extra_engine_args))
    if num_speculative_tokens > 0:
        spec_config: dict[str, Any] = {
            "method": args.method,
            "num_speculative_tokens": num_speculative_tokens,
        }
        if args.draft_model:
            spec_config["model"] = args.draft_model
        kwargs["speculative_config"] = spec_config
    return kwargs


def _run_engine(engine_kwargs: dict[str, Any], conversations: list,
                args: argparse.Namespace) -> dict[str, Any]:
    from vllm import LLM, SamplingParams

    llm = LLM(**engine_kwargs)
    try:
        prefix_caching = bool(
            llm.llm_engine.vllm_config.cache_config.enable_prefix_caching)
        tokenizer = llm.get_tokenizer()
        prompts = [
            tokenizer.apply_chat_template(c,
                                          tokenize=False,
                                          add_generation_prompt=True)
            for c in conversations
        ]
        sampling = SamplingParams(n=args.n,
                                  temperature=args.temperature,
                                  top_p=args.top_p,
                                  max_tokens=args.max_tokens,
                                  ignore_eos=args.ignore_eos)
        if args.num_warmup_prompts > 0:
            warmup = SamplingParams(n=args.n,
                                    temperature=args.temperature,
                                    top_p=args.top_p,
                                    max_tokens=32)
            llm.generate(prompts[:args.num_warmup_prompts],
                         warmup,
                         use_tqdm=False)

        # Counters are cumulative since engine start; diff out the warm-up.
        before = collect_spec_decode_counters(llm.get_metrics())
        start = time.perf_counter()
        outputs = llm.generate(prompts, sampling)
        elapsed = time.perf_counter() - start
        counters = collect_spec_decode_counters(llm.get_metrics()) - before
    finally:
        llm.llm_engine.engine_core.shutdown()

    completions = [c for out in outputs for c in out.outputs]
    num_output_tokens = sum(len(c.token_ids) for c in completions)
    return {
        "num_completions": len(completions),
        "num_output_tokens": num_output_tokens,
        "mean_output_len": num_output_tokens / max(len(completions), 1),
        "elapsed_s": elapsed,
        "output_tok_per_s": num_output_tokens / elapsed,
        "prefix_caching": prefix_caching,
        "counters": asdict(counters),
    }


def _worker(result_queue, engine_kwargs, conversations, args) -> None:
    try:
        result_queue.put(("ok", _run_engine(engine_kwargs, conversations,
                                            args)))
    except BaseException:  # Report every failure, including SystemExit.
        result_queue.put(("error", traceback.format_exc()))


def run_isolated(engine_kwargs: dict[str, Any], conversations: list,
                 args: argparse.Namespace) -> dict[str, Any]:
    """Runs one engine in a spawned process so TPU memory is fully freed."""
    ctx = multiprocessing.get_context("spawn")
    result_queue = ctx.Queue()
    proc = ctx.Process(target=_worker,
                       args=(result_queue, engine_kwargs, conversations, args))
    proc.start()
    while True:
        try:
            status, payload = result_queue.get(timeout=10)
            break
        except queue_lib.Empty:
            if not proc.is_alive():
                raise RuntimeError(
                    f"Benchmark process exited with code {proc.exitcode} "
                    "before reporting a result") from None
    proc.join()
    if status != "ok":
        raise RuntimeError(f"Benchmark run failed:\n{payload}")
    return payload


def _format_table(rows: list[dict[str, Any]]) -> str:
    header = ("max_num_seqs", "method", "k", "accept", "mean_len", "per_pos",
              "out_tok/s", "speedup", "prefix_cache")
    lines = [header]
    for r in rows:
        spec = r.get("spec")
        lines.append((
            str(r["max_num_seqs"]),
            r["method"],
            str(r["num_speculative_tokens"] or "-"),
            f"{spec['acceptance_rate']:.3f}" if spec else "-",
            f"{spec['mean_acceptance_length']:.2f}" if spec else "-",
            ",".join(
                f"{p:.2f}"
                for p in spec["per_position_acceptance"]) if spec else "-",
            f"{r['output_tok_per_s']:.1f}",
            f"{r['speedup']:.2f}x" if r.get("speedup") else "-",
            "on" if r["prefix_caching"] else "off",
        ))
    widths = [max(len(line[i]) for line in lines) for i in range(len(header))]
    return "\n".join("  ".join(cell.ljust(w) for cell, w in zip(line, widths))
                     for line in lines)


def run_benchmark(args: argparse.Namespace) -> list[dict[str, Any]]:
    conversations = load_conversations(args)
    rows: list[dict[str, Any]] = []
    for max_num_seqs in args.max_num_seqs:
        baseline: Optional[dict[str, Any]] = None
        configs = ([0] if not args.skip_baseline else []) + list(
            args.num_speculative_tokens)
        for k in configs:
            label = "baseline" if k == 0 else f"{args.method} k={k}"
            print(f"\n=== max_num_seqs={max_num_seqs} {label} ===", flush=True)
            result = run_isolated(build_engine_kwargs(args, max_num_seqs, k),
                                  conversations, args)
            row = {
                "max_num_seqs": max_num_seqs,
                "method": "baseline" if k == 0 else args.method,
                "num_speculative_tokens": k,
                **result,
            }
            if k == 0:
                baseline = row
            else:
                row["spec"] = asdict(
                    summarize_counters(SpecDecodeCounters(**row["counters"])))
                if baseline is not None:
                    row["speedup"] = compute_speedup(
                        baseline["output_tok_per_s"], row["output_tok_per_s"])
                    if baseline["prefix_caching"] != row["prefix_caching"]:
                        print("WARNING: prefix caching differs between the "
                              "baseline and this run; speedup is not "
                              "like-for-like.")
            rows.append(row)
    return rows


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    arg_group = parser.add_argument_group("Model and engine")
    arg_group.add_argument("--model", default="Qwen/Qwen3.5-4B")
    arg_group.add_argument("--method",
                   default="mtp",
                   help="Speculative method, e.g. mtp, eagle3, ngram, dflash.")
    arg_group.add_argument("--draft-model",
                   help="Draft model path; not needed for mtp or ngram.")
    arg_group.add_argument("--num-speculative-tokens",
                   type=int,
                   nargs="+",
                   default=[3])
    arg_group.add_argument("--max-num-seqs", type=int, nargs="+", default=[64])
    arg_group.add_argument("--tensor-parallel-size", type=int, default=1)
    arg_group.add_argument("--max-model-len", type=int, default=8192)
    arg_group.add_argument("--max-num-batched-tokens", type=int)
    arg_group.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    arg_group.add_argument("--kv-cache-dtype", help="e.g. fp8")
    arg_group.add_argument("--enable-prefix-caching", action="store_true")
    arg_group.add_argument("--async-scheduling", action="store_true")
    arg_group.add_argument("--seed", type=int, default=0)
    arg_group.add_argument("--extra-engine-args",
                   default="{}",
                   help="JSON dict merged into the LLM(...) kwargs.")
    arg_group.add_argument("--skip-baseline", action="store_true")

    dataset_group = parser.add_argument_group("Prompts")
    dataset_group.add_argument("--prompts-jsonl",
                   help="JSONL with 'messages' or a prompt field per line. "
                   "Overrides --dataset.")
    dataset_group.add_argument("--dataset", default="openai/gsm8k")
    dataset_group.add_argument("--dataset-subset", default="main")
    dataset_group.add_argument("--dataset-split", default="test")
    dataset_group.add_argument("--prompt-field", default="question")
    dataset_group.add_argument("--num-prompts", type=int, default=128)
    dataset_group.add_argument("--num-warmup-prompts", type=int, default=4)

    sample_group = parser.add_argument_group("Sampling (RL rollout defaults)")
    sample_group.add_argument("--n", type=int, default=8, help="Samples per prompt.")
    sample_group.add_argument("--temperature", type=float, default=1.0)
    sample_group.add_argument("--top-p", type=float, default=1.0)
    sample_group.add_argument("--max-tokens", type=int, default=4096)
    sample_group.add_argument("--ignore-eos", action="store_true")

    parser.add_argument("--output-json", help="Write args and results here.")
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if any(k <= 0 for k in args.num_speculative_tokens):
        raise ValueError("--num-speculative-tokens values must be positive")
    if any(m <= 0 for m in args.max_num_seqs):
        raise ValueError("--max-num-seqs values must be positive")
    if args.num_prompts <= 0 or args.n <= 0 or args.max_tokens <= 0:
        raise ValueError("--num-prompts, --n and --max-tokens must be "
                         "positive")
    if args.num_warmup_prompts > args.num_prompts:
        raise ValueError("--num-warmup-prompts exceeds --num-prompts")
    extra = json.loads(args.extra_engine_args)
    if not isinstance(extra, dict):
        raise ValueError("--extra-engine-args must be a JSON object")


def main(argv: Optional[list[str]] = None) -> int:
    args = create_parser().parse_args(argv)
    validate_args(args)
    rows = run_benchmark(args)
    print("\n" + _format_table(rows))
    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump({"args": vars(args), "results": rows}, f, indent=2)
        print(f"\nWrote {args.output_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
