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

import ast
import json
import re
import sys


def parse_benchmark_log(log_path, result_path):
    # Mapping of what vllm prints vs what Spanner column expects
    METRIC_MAPPING = {
        "Request throughput": "Throughput",
        "Output token throughput": "OutputTokenThroughput",
        "Total Token throughput": "TotalTokenThroughput",
        "Median TTFT": "MedianTTFT",
        "P99 TTFT": "P99TTFT",
        "Median TPOT": "MedianTPOT",
        "P99 TPOT": "P99TPOT",
        "Median ITL": "MedianITL",
        "P99 ITL": "P99ITL",
        "Median E2EL": "MedianETEL",
        "P99 E2EL": "P99ETEL"
    }

    try:
        with open(log_path, "r") as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"Warning: log file {log_path} not found.")
        lines = []

    results = {}
    in_results = False
    for i, line in enumerate(lines):
        line = line.strip()
        if "============ Serving Benchmark Result ============" in line:
            in_results = True
            continue
        if "==================================================" in line and in_results:
            in_results = False

        if in_results and ":" in line:
            key, val = line.split(":", 1)
            val = val.strip()

            # Remove units like (ms) or (tok/s) or (req/s)
            clean_key = re.sub(r"\(.*?\)", "", key).strip()

            if clean_key in METRIC_MAPPING and METRIC_MAPPING[
                    clean_key] not in results:
                if val != "N/A":
                    results[METRIC_MAPPING[clean_key]] = val

        # Handle explicit AccuracyMetrics: json (used by some benchmark wrappers)
        if line.startswith("AccuracyMetrics:"):
            try:
                json_str = line.split("AccuracyMetrics:")[1].strip()
                # Verify it is valid JSON
                json.loads(json_str)
                results["AccuracyMetrics"] = json_str
            except Exception:
                pass

        # Fallback: Parse legacy Accuracy result dict printed by benchmark_serving.py
        if line == "Results":
            for j in range(1, min(6, len(lines) - i)):
                try:
                    acc_dict = ast.literal_eval(lines[i + j].strip())
                    if isinstance(acc_dict, dict) and "accuracy" in acc_dict:
                        results["AccuracyMetrics"] = json.dumps(
                            {"accuracy": acc_dict["accuracy"]})
                        break
                except Exception:
                    pass

    with open(result_path, "w") as out:
        for k, v in results.items():
            out.write(f"{k}={v}\n")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <log_file_path> <result_file_path>")
        sys.exit(1)

    log_file = sys.argv[1]
    result_file = sys.argv[2]
    parse_benchmark_log(log_file, result_file)
