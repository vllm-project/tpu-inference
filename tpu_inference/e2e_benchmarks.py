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

import json
import logging
import os
import random
import subprocess
import time
import warnings
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
from google.cloud import storage
from transformers import AutoTokenizer

warnings.filterwarnings("ignore")
logging.disable(logging.WARNING)

# ==============================================================================
# CONFIGURATION
# ==============================================================================
MODEL_PATH = "/mnt/disks/persist/hf/hub/models--mistralai--Mistral-Large-3-675B-Instruct-2512/snapshots/5bb3d32a0a147594527f38cb136a8390eaa82222"
SERVED_MODEL_NAME = "mistralai/Mistral-Large-3-675B-Instruct-2512"
SERVER_URL = "http://localhost:8000"
SHAREGPT_FILE = "/mnt/disks/persist/hf/ShareGPT_V3_unfiltered_cleaned_split.json"

RESULTS_DIR = Path("benchmark_results")
DATASETS_DIR = Path("datasets")
RESULTS_DIR.mkdir(exist_ok=True)
DATASETS_DIR.mkdir(exist_ok=True)

# Only run the requested 2048 input / 512 output tokens profile
PROFILES = [
    {
        "isl": 2048,
        "osl": 512
    },
]
# Reversed order to compile the largest batch size first
CONCURRENCIES = [1, 2, 4, 8, 16, 32, 64, 128]

# Generate a single unique timestamp directory for this entire run session
RUN_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
SESSION_RESULTS_DIR = RESULTS_DIR / RUN_TIMESTAMP
SESSION_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ==============================================================================
# DATASET GENERATION
# ==============================================================================


def generate_fixed_len_dataset(isl, osl, target_prompts):
    """Replicates sharegpt_to_fixed_len.py logic natively."""
    dataset_path = DATASETS_DIR / f"bench_{isl}_{osl}.jsonl"

    if dataset_path.exists():
        print(
            f">>> Dataset already exists at {dataset_path}, skipping generation."
        )
        return dataset_path

    print(f"\n>>> Generating dataset for ISL: {isl}, OSL: {osl}...")

    if not os.path.exists(SHAREGPT_FILE):
        raise FileNotFoundError(
            f"Missing {SHAREGPT_FILE}. Please download it first.")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH,
                                              trust_remote_code=True,
                                              use_fast=True)

    with open(SHAREGPT_FILE, encoding="utf-8") as f:
        data = json.load(f)

    # Filter for human turns
    data = [
        entry for entry in data
        if "conversations" in entry and any(turn["from"] == "human"
                                            for turn in entry["conversations"])
    ]

    rng = random.Random(42)
    rng.shuffle(data)

    sources = []
    for entry in data:
        if len(sources) >= 50:
            break
        human_turns = [
            t["value"] for t in entry["conversations"] if t["from"] == "human"
        ]
        text = human_turns[0].strip()
        if not text:
            continue
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        if len(token_ids) >= 50:
            sources.append(token_ids)

    def build_prompt(t_ids, target_len):
        if len(t_ids) < target_len:
            repeats = (target_len // len(t_ids)) + 1
            tiled = (t_ids * repeats)[:target_len]
        else:
            tiled = t_ids[:target_len]
        return tokenizer.decode(tiled, skip_special_tokens=True)

    unique_prompts = [build_prompt(s, isl) for s in sources]
    rows = [
        unique_prompts[i % len(unique_prompts)] for i in range(target_prompts)
    ]
    rng.shuffle(rows)

    with open(dataset_path, "w", encoding="utf-8") as f:
        for prompt in rows:
            row = {"prompt": prompt, "output_tokens": osl}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f">>> Saved {target_prompts} prompts to {dataset_path}")
    return dataset_path


# ==============================================================================
# EVALUATION & BENCHMARKING
# ==============================================================================


def run_gsm8k_eval():
    out_dir = RESULTS_DIR / "gsm8k"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Check for existing results first
    for f in out_dir.rglob("results_*.json"):
        try:
            with open(f, "r") as file:
                data = json.load(file)
                accuracy = data["results"]["gsm8k"][
                    "exact_match,flexible-extract"]
                print(
                    f">>> Found existing GSM8K Accuracy: {accuracy * 100:.2f}% ({f})"
                )
                return accuracy
        except Exception:
            continue

    print("\n>>> Running GSM8K Evaluation against active server...")
    cmd = [
        "lm_eval",
        "--model",
        "local-completions",
        "--tasks",
        "gsm8k",
        "--model_args",
        f"model={SERVED_MODEL_NAME},tokenizer={MODEL_PATH},base_url={SERVER_URL}/v1/completions,num_concurrent=64,max_retries=3,tokenized_requests=False,trust_remote_code=True",
        "--output_path",
        str(out_dir),
    ]

    try:
        subprocess.run(cmd, check=True)
        for f in out_dir.rglob("results_*.json"):
            with open(f, "r") as file:
                data = json.load(file)
                accuracy = data["results"]["gsm8k"][
                    "exact_match,flexible-extract"]
                print(f">>> GSM8K Accuracy: {accuracy * 100:.2f}%")
                return accuracy
    except Exception as e:
        print(f">>> ERROR: GSM8K Eval failed or could not be parsed: {e}")

    return None


def get_existing_result(isl, osl, concurrency):
    # Cache disabled for session isolation
    return None


def run_benchmark(isl, osl, concurrency, dataset_path):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"bench_isl{isl}_osl{osl}_c{concurrency}_{timestamp}.json"
    num_prompts = concurrency * 10

    cmd = [
        "vllm",
        "bench",
        "serve",
        "--backend",
        "vllm",
        "--model",
        SERVED_MODEL_NAME,
        "--tokenizer",
        MODEL_PATH,
        "--base-url",
        SERVER_URL,
        "--endpoint",
        "/v1/completions",
        "--dataset-name",
        "custom",
        "--dataset-path",
        str(dataset_path),
        "--custom-output-len",
        "-1",
        "--num-prompts",
        str(num_prompts),
        "--max-concurrency",
        str(concurrency),
        "--request-rate",
        "inf",
        "--ignore-eos",
        "--percentile-metrics",
        "ttft,tpot,itl,e2el",
        "--metric-percentiles",
        "10,50,90,99",
        "--save-result",
        "--result-dir",
        str(SESSION_RESULTS_DIR),
        "--result-filename",
        filename,
        "--trust-remote-code",
    ]

    print(f"\n>>> Running: ISL: {isl} | OSL: {osl} | Concurrency:"
          f" {concurrency}")
    try:
        subprocess.run(cmd, check=True)
        return SESSION_RESULTS_DIR / filename
    except subprocess.CalledProcessError as e:
        print(f">>> ERROR: Benchmark failed for c{concurrency}, {e}")
        return None


# ==============================================================================
# PLOTTING LOGIC
# ==============================================================================


def plot_results(all_results, gsm8k_acc):
    plt.figure(figsize=(12, 8))

    acc_str = f" (GSM8K: {gsm8k_acc * 100:.1f}%)" if gsm8k_acc else " (GSM8K: N/A)"
    colors = ["#1f77b4", "#2ca02c", "#ff7f0e", "#d62728"]

    for idx, profile in enumerate(PROFILES):
        isl, osl = profile["isl"], profile["osl"]

        p_data = [
            r for r in all_results if r["isl"] == isl and r["osl"] == osl
        ]
        p_data.sort(key=lambda x: x["concurrency"])

        if not p_data:
            continue

        x = [r["p50_speed_tok_s"] for r in p_data]
        y = [r["throughput_req_min"] for r in p_data]
        label = f"Mistral Large 3 {isl}/{osl}{acc_str}"

        plt.plot(
            x,
            y,
            label=label,
            marker="o",
            linewidth=2,
            markersize=8,
            color=colors[idx % len(colors)],
        )

        for r in p_data:
            plt.annotate(
                f"{r['concurrency']}",
                (r["p50_speed_tok_s"], r["throughput_req_min"]),
                textcoords="offset points",
                xytext=(0, 8),
                ha="center",
                fontsize=9,
                fontweight="bold",
                color="#444444",
            )

    # GPU Target: NVIDIA B200 Baseline
    b200_concurrencies = [1, 2, 4, 8, 16, 32, 64, 128]
    b200_tok_s = [131.1, 231.2, 426.1, 596.3, 1052.3, 1553.7, 2233.6, 3141.3]
    b200_req_min = [tok * (60.0 / 512.0) for tok in b200_tok_s]
    b200_itl_ms = [7.45, 8.38, 9.04, 12.51, 14.23, 19.36, 27.31, 38.94]
    b200_speed_tok_s = [1000.0 / itl for itl in b200_itl_ms]

    plt.plot(
        b200_speed_tok_s,
        b200_req_min,
        label="GPU Target: NVIDIA B200",
        marker="s",
        linestyle="--",
        linewidth=2,
        markersize=8,
        color="#d62728",
    )

    for i, c in enumerate(b200_concurrencies):
        plt.annotate(
            f"{c}",
            (b200_speed_tok_s[i], b200_req_min[i]),
            textcoords="offset points",
            xytext=(0, 8),
            ha="center",
            fontsize=9,
            fontweight="bold",
            color="#d62728",
        )

    plt.xlabel("P50 Speed (tok/s)", fontsize=12)
    plt.ylabel("Request Throughput (req/min)", fontsize=12)
    plt.title("vLLM Benchmark: Request Throughput vs. P50 Speed",
              fontsize=14,
              pad=20)
    plt.grid(True, ls="-", alpha=0.3)
    plt.legend(loc="best")

    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines["right"].set_visible(False)

    plt.tight_layout()
    plot_path = SESSION_RESULTS_DIR / "benchmark_plot.png"
    plt.savefig(plot_path, dpi=300)
    print(f"\n>>> Graph saved locally to {plot_path}")

    # Native Python GCS Upload
    print("\n>>> Uploading graph to GCS bucket natively...")
    try:
        client = storage.Client()
        bucket = client.bucket("mlperf-exp-europe-w")
        blob_path = f"amandaliang/mistral/{RUN_TIMESTAMP}/benchmark_plot.png"
        blob = bucket.blob(blob_path)
        blob.upload_from_filename(str(plot_path))
        print(">>> Successfully uploaded to GCS natively!")
        print(
            f">>> View Graph directly here:"
            f" https://storage.cloud.google.com/mlperf-exp-europe-w/amandaliang/mistral/{RUN_TIMESTAMP}/benchmark_plot.png"
        )
    except Exception as e:
        print(f">>> WARNING: Native GCS upload failed: {e}")


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================


def main():
    print(f"Waiting for vLLM server at {SERVER_URL}...")
    try:
        # 1. Run GSM8K Eval first
        gsm8k_acc = run_gsm8k_eval()

        # 2. Run Throughput Concurrency Sweeps
        all_results = []
        for profile in PROFILES:
            isl, osl = profile["isl"], profile["osl"]

            # Max prompts needed is max_concurrency * 10
            max_prompts_needed = max(CONCURRENCIES) * 10

            # Generate shared dataset
            dataset_path = generate_fixed_len_dataset(isl, osl,
                                                      max_prompts_needed)

            for c in CONCURRENCIES:
                result_file = get_existing_result(isl, osl, c)
                if result_file:
                    print(
                        f">>> Found existing result for ISL: {isl}, OSL: {osl},"
                        f" C: {c}: {result_file}")
                else:
                    result_file = run_benchmark(isl, osl, c, dataset_path)

                if result_file and result_file.exists():
                    with open(result_file, "r") as f:
                        data = json.load(f)
                        req_throughput_s = data.get("request_throughput", 0)
                        p50_itl_ms = data.get("p50_itl_ms")

                        if p50_itl_ms:
                            summary = {
                                "isl": isl,
                                "osl": osl,
                                "concurrency": c,
                                "throughput_req_min": req_throughput_s * 60.0,
                                "p50_speed_tok_s": 1000.0 / p50_itl_ms,
                            }
                            all_results.append(summary)

                if not get_existing_result(isl, osl, c):
                    time.sleep(2)

        # 3. Generate final plot
        if all_results:
            plot_results(all_results, gsm8k_acc)
        else:
            print(">>> No benchmark data collected.")

    except KeyboardInterrupt:
        print("\n>>> Interrupted by user. Exiting...")


if __name__ == "__main__":
    main()
