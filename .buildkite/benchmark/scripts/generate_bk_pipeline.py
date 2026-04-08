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


def create_benchmark_group(case_data,
                           global_env,
                           file_path,
                           is_single_case=False):
    """
    Creates a Buildkite Group step.
    """
    # Identify Case Name (fallback to model name)
    model_name = case_data.get("server_command_options",
                               {}).get("args", {}).get("model", "unknown")
    case_name = case_data.get("case_name", model_name)

    # Extract TPU types from the case data
    ci_queues = case_data.get("ci_queue")

    # Merge Environment Variables (Global + Case Specific)
    combined_env = {**global_env, **case_data.get("env", {})}
    safe_key = case_name.replace("/", "-").replace(" ", "-")

    # Construct the Step dictionary
    child_steps = []
    for agent in ci_queues:
        # Build the environment for this specific step
        step_env = {**combined_env, "ci_queue": agent}

        if not is_single_case:
            step_env["TARGET_CASE_NAME"] = case_name

        child_steps.append({
            "label":
            f"{agent} {case_name}",
            "env":
            step_env,
            "agents": {
                "queue": agent
            },
            "command":
            f"bash .buildkite/benchmark/scripts/run_job.sh {file_path}",
        })

    return {"group": case_name, "key": safe_key, "steps": child_steps}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",
                        required=True,
                        help="Path to the multi_case JSON file")
    args = parser.parse_args()

    # Verify input file existence
    if not os.path.exists(args.input):
        print(f"Error: File {args.input} not found", file=sys.stderr)
        sys.exit(1)

    with open(args.input, 'r') as f:
        data = json.load(f)

    global_env = data.get("global_env", {})
    steps = []

    if "benchmark_cases" in data:
        for case in data["benchmark_cases"]:
            # Multi-case
            steps.append(
                create_benchmark_group(case,
                                       global_env,
                                       args.input,
                                       is_single_case=False))
    else:
        # Single case file
        steps.append(
            create_benchmark_group(data,
                                   global_env,
                                   args.input,
                                   is_single_case=True))

    print(
        yaml.dump({"steps": steps}, sort_keys=False, default_flow_style=False))


if __name__ == "__main__":
    main()
