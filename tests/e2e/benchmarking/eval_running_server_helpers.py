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
"""Python side of eval_running_server.sh.

Subcommands:
  smoke-payload MODEL              print the smoke test's /v1/completions body
  check-smoke SMOKE_JSON           exit 0 if the completion returned text
  stage-gsm8k-task DIR TASK EFFORT write an lm_eval task: stock gsm8k plus
                                   chat_template_kwargs.reasoning_effort
  write-metrics OUT_DIR MODEL      merge the leg statuses, the lm_eval results
                                   and bench.json into OUT_DIR/metrics.json
"""

import argparse
import glob
import json
import os
import sys

# Fields copied from `vllm bench serve`'s saved result into metrics.json.
BENCH_FIELDS = ("num_prompts", "completed", "failed", "request_throughput",
                "output_throughput", "total_token_throughput",
                "median_ttft_ms", "p99_ttft_ms", "median_tpot_ms",
                "median_itl_ms", "p99_itl_ms")


def smoke_payload(args: argparse.Namespace) -> int:
    print(
        json.dumps({
            "model": args.model,
            "prompt": "San Francisco is a",
            "max_tokens": 16,
            "temperature": 0,
        }))
    return 0


def check_smoke(args: argparse.Namespace) -> int:
    with open(args.smoke_json) as f:
        text = json.load(f)["choices"][0]["text"]
    print(f"[smoke] completion: {text!r}")
    return 0 if text.strip() else 1


def stage_gsm8k_task(args: argparse.Namespace) -> int:
    # Imported here so the other subcommands need nothing beyond the stdlib.
    import lm_eval.tasks
    import yaml

    stock = os.path.join(os.path.dirname(lm_eval.tasks.__file__), "gsm8k",
                         "gsm8k.yaml")
    override = {
        "include": stock,
        "task": args.task,
        # The include carries the stock tag; a tag of its own keeps the
        # override from re-registering the stock `math_word_problems` group.
        "tag": f"{args.task}_tag",
        # An `include:` override replaces generation_kwargs as a whole, so the
        # stock values are restated alongside the addition.
        "generation_kwargs": {
            "until": ["Question:", "</s>", "<|im_end|>"],
            "do_sample": False,
            "temperature": 0.0,
            "chat_template_kwargs": {
                "reasoning_effort": args.effort
            },
        },
    }
    os.makedirs(args.task_dir, exist_ok=True)
    with open(os.path.join(args.task_dir, f"{args.task}.yaml"), "w") as f:
        yaml.safe_dump(override, f, sort_keys=False)
    print(f"[gsm8k] staged {args.task} (reasoning_effort={args.effort})")
    return 0


def write_metrics(args: argparse.Namespace) -> int:
    out_dir = args.out_dir
    metrics = {"model": args.model}
    with open(os.path.join(out_dir, "leg_status.tsv")) as f:
        for line in f:
            leg, rc, seconds = line.rstrip("\n").split("\t")
            metrics[leg] = {
                "status": "ok" if rc == "0" else "failed",
                "exit_code": int(rc),
                "seconds": int(seconds),
            }

    gsm8k = metrics.get("gsm8k")
    results = sorted(
        glob.glob(os.path.join(out_dir, "gsm8k", "**", "results_*.json"),
                  recursive=True))
    if gsm8k is not None and gsm8k["status"] == "ok" and results:
        with open(results[-1]) as f:
            data = json.load(f)
        task, scores = next((t, s) for t, s in data["results"].items()
                            if "exact_match,flexible-extract" in s)
        gsm8k.update({
            "task": task,
            "flexible_extract": scores["exact_match,flexible-extract"],
            "strict_match": scores["exact_match,strict-match"],
            "num_samples": data["n-samples"][task]["effective"],
        })

    bench = metrics.get("bench")
    bench_file = os.path.join(out_dir, "bench.json")
    if (bench is not None and bench["status"] == "ok"
            and os.path.isfile(bench_file)):
        with open(bench_file) as f:
            data = json.load(f)
        for key in BENCH_FIELDS:
            bench[key] = data.get(key)

    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("smoke-payload")
    p.add_argument("model")
    p.set_defaults(func=smoke_payload)

    p = sub.add_parser("check-smoke")
    p.add_argument("smoke_json")
    p.set_defaults(func=check_smoke)

    p = sub.add_parser("stage-gsm8k-task")
    p.add_argument("task_dir")
    p.add_argument("task")
    p.add_argument("effort")
    p.set_defaults(func=stage_gsm8k_task)

    p = sub.add_parser("write-metrics")
    p.add_argument("out_dir")
    p.add_argument("model")
    p.set_defaults(func=write_metrics)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
