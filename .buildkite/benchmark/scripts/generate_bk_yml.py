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

import argparse
import json
import os
import sys

import yaml


def get_queue_name(tpu_string):
    """
    Maps tpu_type to queue name.
    Example: 'v6e-1' -> 'vllm-tpu-v6e', 'v7x-2' -> 'vllm-tpu-v7x'
    """
    platform = tpu_string.split('-')[0]
    return f"vllm-tpu-{platform}"


def create_step(case_data, global_env, file_path):
    """
    Creates a Buildkite Step with a Matrix for TPU types.
    """
    # 1. Identify Case Name
    model_name = case_data.get("server_command_options",
                               {}).get("args", {}).get("model", "unknown")
    case_name = case_data.get("case_name", model_name)

    # 2. Identify TPU Types for Matrix
    tpu_types = case_data.get("tpu_type", ["v6e-1"])

    # 3. Merge Env Vars (Global + Case Specific)
    combined_env = {**global_env, **case_data.get("env", {})}

    # 4. Define the Step with Matrix
    step = {
        "label": f"Benchmark: {case_name} on {{{{matrix.tpu}}}}",
        "command": f"bash .buildkite/benchmark/scripts/run_job.sh {file_path}",
        "env": {
            **combined_env,
            "TARGET_CASE_NAME": case_name,
            "TPU_TYPE": "{{matrix.tpu}}"  # Injected by matrix at runtime
        },
        "matrix": {
            "setup": {
                "tpu": tpu_types
            }
        },
        "agents": {
            # Map the queue dynamically based on matrix value
            # Note: We use a simplified mapping here assuming queue naming matches prefix
            "queue": "vllm-tpu-{{matrix.tpu}}"
        },
        "retry": {
            "automatic": True,
            "limit": 1
        }
    }
    return step


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",
                        required=True,
                        help="Path to JSON case file")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: File {args.input} not found", file=sys.stderr)
        sys.exit(1)

    with open(args.input, 'r') as f:
        data = json.load(f)

    global_env = data.get("global_env", {})
    steps = []

    # Detect if Multi-case or Single-case
    if "benchmark_cases" in data:
        # Multi-case Logic
        for case in data["benchmark_cases"]:
            steps.append(create_step(case, global_env, args.input))
    else:
        # Single-case Logic
        steps.append(create_step(data, global_env, args.input))

    # Output to stdout for Buildkite Pipeline Upload
    print(
        yaml.dump({"steps": steps}, sort_keys=False, default_flow_style=False))


if __name__ == "__main__":
    main()
