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
import re
import sys
from typing import Any, Dict

import yaml


def clean_key_string(key: str) -> str:
    # r'[^a-zA-Z0-9/-]' matches anything NOT in the specified set
    return re.sub(r'[^a-zA-Z0-9/-_]', '-', key)


def extract_arg_from_command_options(case_data: Dict[str, Any],
                                     target_arg: str) -> Any:
    # Define the option sources to search in order
    target_options = ["server_command_options", "client_command_options"]

    for opt_key in target_options:
        # Safely extract layer by layer (prevents AttributeError caused by None)
        options = case_data.get(opt_key) or {}
        args = options.get("args") or {}
        current_value = args.get(target_arg)

        # Check if the extracted value is valid (excluding None, or empty strings)
        if current_value:
            # Return immediately once a valid value is found (Return-Early pattern)
            return current_value

    # If the loop finishes without returning, no valid value was found. Raise an error.
    raise ValueError(
        f"Extraction failed! Could not find a valid '{target_arg}' in the "
        f"{target_options} structures of case_data.")


def create_benchmark_group(case_data,
                           global_env,
                           file_path,
                           is_single_case=False):
    """
    Creates a Buildkite Group step.
    """
    # Extract filename without extension to be used as the step label
    file_basename = os.path.splitext(os.path.basename(file_path))[0]

    # Identify Case Name
    model_name = extract_arg_from_command_options(case_data, "model")
    case_name = case_data.get("case_name", model_name)

    # Extract TPU types from the case data
    ci_queues = case_data.get("ci_queue", [])

    # Merge Environment Variables (Global + Case Specific)
    combined_env = {**global_env, **case_data.get("env", {})}
    safe_key = clean_key_string(file_basename)

    # Construct the Step dictionary
    child_steps = []
    for agent in ci_queues:
        # Build the environment for this specific step
        step_env = {**combined_env, "ci_queue": agent}

        if is_single_case:
            # Use the filename without extension as the step label
            step_label = f"{agent} {file_basename}"
            case_parameter = f"{file_path}"
        else:
            step_env["TARGET_CASE_NAME"] = case_name
            step_label = f"{agent} {file_basename} {case_name}"
            case_parameter = f"{file_path} {case_name}"

        # Define step key
        step_safe_key = clean_key_string(step_label)

        child_steps.append({
            "label":
            step_label,
            "key":
            step_safe_key,
            "env":
            step_env,
            "agents": {
                "queue": agent
            },
            "command":
            f"bash .buildkite/benchmark/scripts/run_job.sh {case_parameter}",
        })

    return {"group": file_basename, "key": safe_key, "steps": child_steps}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",
                        required=True,
                        help="Path to the benchmark JSON configuration file")
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
            # Multi-case: Requires TARGET_CASE_NAME to distinguish between tests
            steps.append(
                create_benchmark_group(case,
                                       global_env,
                                       args.input,
                                       is_single_case=False))
    else:
        # Single-case
        steps.append(
            create_benchmark_group(data,
                                   global_env,
                                   args.input,
                                   is_single_case=True))

    print(
        yaml.dump({"steps": steps}, sort_keys=False, default_flow_style=False))


if __name__ == "__main__":
    main()
