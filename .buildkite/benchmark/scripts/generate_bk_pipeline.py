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
import hashlib
import json
import os
import re
import sys
from typing import Any, Dict, List, Set

import yaml

# List of authorized command types
ALLOWED_SERVER_COMMAND_TYPES = {"vllm_serve"}
ALLOWED_CLIENT_COMMAND_TYPES = {"vllm_bench_serve", "lm_eval"}


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def clean_key_string(key: str) -> str:
    """
    Sanitizes the string and ensures the length does not exceed 100 characters.
    Buildkite keys may only contain alphanumeric characters, underscores, dashes and colons.
    """
    # Replace invalid characters with '-'
    # Note: We allow a-z, A-Z, 0-9, _, :, and -. Everything else becomes -
    sanitized = re.sub(r"[^a-zA-Z0-9_:\-]", "-", key)

    # Collapse multiple dashes into one and strip leading/trailing dashes
    sanitized = re.sub(r"-+", "-", sanitized).strip("-")

    # If length exceeds 100, truncate and append a hash for uniqueness
    if len(sanitized) > 100:
        # Take the first 90 chars and append the first 8 chars of an MD5 hash.
        # This ensures that even if truncated, the Key remains unique.
        suffix = hashlib.md5(key.encode()).hexdigest()[:8]
        return f"{sanitized[:90]}-{suffix}"

    return sanitized


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


def validate_command_options(case_data: Dict[str, Any], file_path: str,
                             errors: List[str]):
    """Validates command options and their types."""
    # Server options are optional (e.g. for certain client-only benchmarks, like `lm_eval`)
    server_options = case_data.get("server_command_options")
    if server_options:
        cmd_type = server_options.get("command_type")
        if cmd_type and cmd_type not in ALLOWED_SERVER_COMMAND_TYPES:
            errors.append(
                f"Validation Error: Unauthorized server command_type '{cmd_type}' in {file_path}. "
                f"Allowed types are: {sorted(list(ALLOWED_SERVER_COMMAND_TYPES))}"
            )

    # Client options are required for all benchmark cases
    client_options = case_data.get("client_command_options")
    if not client_options:
        errors.append(
            f"Validation Error: 'client_command_options' is missing in {file_path}."
        )
    else:
        cmd_type = client_options.get("command_type")
        if not cmd_type:
            errors.append(
                f"Validation Error: 'command_type' is missing in client_command_options within {file_path}."
            )
        elif cmd_type not in ALLOWED_CLIENT_COMMAND_TYPES:
            errors.append(
                f"Validation Error: Unauthorized client command_type '{cmd_type}' in {file_path}. "
                f"Allowed types are: {sorted(list(ALLOWED_CLIENT_COMMAND_TYPES))}"
            )


def validate_parameter_dependencies(case_data: Dict[str, Any], file_path: str,
                                    errors: List[str]):
    """Validates dependencies required for the benchmark infra (run_bm.sh) to function."""
    server_options = case_data.get("server_command_options") or {}
    server_args = server_options.get("args") or {}

    client_options = case_data.get("client_command_options") or {}
    client_args = client_options.get("args") or {}
    client_cmd_type = client_options.get("command_type")

    # Specific Rules for `lm_eval`: must be a known dataset
    if client_cmd_type == "lm_eval":
        dataset_name = client_args.get("dataset-name")
        allowed_datasets = {"math500", "mlperf", "mmlu"}
        if dataset_name not in allowed_datasets:
            errors.append(
                f"Validation Error: {file_path} has dataset-name '{dataset_name}' "
                f"but for 'lm_eval', it must be one of {sorted(list(allowed_datasets))}."
            )

    # Specific Rules for `vllm_bench_serve`
    if client_cmd_type == "vllm_bench_serve":
        # Ensure percentile-metrics is present and contains 'e2el'
        percentile_metrics = client_args.get("percentile-metrics")
        if not percentile_metrics:
            errors.append(
                f"Validation Error: {file_path} is missing 'percentile-metrics' in client_command_options. "
                f"It must be defined and include 'e2el'.")
        else:
            # Check if 'e2el' is in the comma-separated string
            metrics_list = [m.strip() for m in percentile_metrics.split(",")]
            if "e2el" not in metrics_list:
                errors.append(
                    f"Validation Error: {file_path} has 'percentile-metrics' set to '{percentile_metrics}' "
                    f"but it must include 'e2el' for proper benchmark reporting."
                )

        # Model consistency check to ensure client/server are aligned
        server_model = server_args.get("model")
        client_model = client_args.get("model")
        if not client_model:
            errors.append(
                f"Validation Error: {file_path} is missing 'model' in client_command_options."
            )
        elif client_model != server_model:
            errors.append(
                f"Validation Error: Model mismatch in {file_path}. Server is '{server_model}' "
                f"but client is '{client_model}'.")

        # vllm_bench_serve requires a server to be defined
        if not case_data.get("server_command_options"):
            errors.append(
                f"Validation Error: {file_path} uses 'vllm_bench_serve' but is missing 'server_command_options'. "
                "The infra requires a server to start for this benchmark type."
            )


def _get_mlcompass_select_tests() -> Set[str]:
    selected = os.getenv('MLCOMPASS_SELECT_TESTS')
    if selected:
        return {s.strip() for s in selected.split(',') if s.strip()}
    return set()


def create_benchmark_steps(case_data: Dict[str, Any],
                           global_env: Dict[str, Any],
                           file_path: str,
                           file_basename: str,
                           parent_dir: str,
                           used_keys: Set[str],
                           errors: List[str],
                           no_verify: bool = False) -> List[Dict[str, Any]]:
    """
    Generates a list of Buildkite steps for a case.
    """
    # Identify Case Name
    try:
        model_name = extract_arg_from_command_options(case_data, "model")
    except ValueError as e:
        # Collect the error and continue with a placeholder to allow reporting
        # multiple issues across the entire file in a single run.
        if not no_verify:
            errors.append(str(e))
        model_name = "unknown_model"

    case_name = case_data.get("case_name", model_name)

    # Extract TPU types from the case data
    ci_queues = case_data.get("ci_queue", [])

    # Skip validation if no_verify is set
    if not no_verify:
        validate_command_options(case_data, file_path, errors)
        validate_parameter_dependencies(case_data, file_path, errors)

        # Ensure ci_queue is not empty
        if not ci_queues:
            errors.append(
                f"Validation Error: 'ci_queue' is missing or empty in {file_path} "
                f"for case '{case_name}'.")

    # Merge Environment Variables (Global + Case Specific)
    combined_env = {**global_env, **case_data.get("env", {})}

    # Construct the Step dictionary
    child_steps = []
    mlcompass_select_tests = _get_mlcompass_select_tests()
    for agent in ci_queues:
        # Build the environment for this specific step
        step_env = {
            **combined_env, "ci_queue": agent,
            "USE_PREBUILT_IMAGE": "1"
        }

        step_env["TARGET_CASE_NAME"] = case_name
        # Include parent_dir in label for uniqueness
        step_label = f"[{parent_dir}] {agent} {file_basename} {case_name}"
        case_parameter = f"{file_path} {case_name}"
        step_env["MLCOMPASS_TEST_NAME"] = f"vllm:{agent}:{case_name}"
        if mlcompass_select_tests and step_env[
                "MLCOMPASS_TEST_NAME"] not in mlcompass_select_tests:
            continue

        # Define step key and check for internal collisions
        step_safe_key = clean_key_string(step_label)
        if not no_verify and step_safe_key in used_keys:
            errors.append(
                f"Collision Error: Duplicate key '{step_safe_key}' detected "
                f"within {file_path}. Ensure all of case_name are unique.")
        used_keys.add(step_safe_key)

        step = {
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
        }

        # Add dependency on global case name validation if it was uploaded in bootstrap
        if os.environ.get("BENCHMARK_VALIDATION_UPLOADED") == "true":
            step["depends_on"] = "validate_benchmark_case_name"

        child_steps.append(step)

    return child_steps


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",
                        required=True,
                        help="Path to the benchmark JSON configuration file")
    parser.add_argument("--no-verify",
                        type=str2bool,
                        default=False,
                        help="Skip validation rules")
    parser.add_argument(
        "--dependency-step",
        type=str,
        default=None,
        required=False,
        help="Optional Buildkite step key that all benchmark steps depend on")
    args = parser.parse_args()

    # Verify input file existence
    if not os.path.exists(args.input):
        print(f"Error: File {args.input} not found", file=sys.stderr)
        sys.exit(1)

    with open(args.input, 'r') as f:
        data = json.load(f)

    # Pre-calculate file metadata used for Namespacing
    file_path = args.input
    file_basename = os.path.splitext(os.path.basename(file_path))[0]
    parent_dir = os.path.basename(os.path.dirname(file_path))

    global_env = data.get("global_env", {})
    # Inject UPLOAD_DB environment variable if present in parent environment
    if "UPLOAD_DB" in os.environ:
        global_env["UPLOAD_DB"] = os.environ["UPLOAD_DB"]
    if "EXTRA_ENVS" in os.environ:
        global_env["EXTRA_ENVS"] = os.environ["EXTRA_ENVS"]

    all_steps = []
    used_keys = set()  # Track keys for this file
    errors = []

    # Process cases
    if "benchmark_cases" not in data:
        print(f"Error: 'benchmark_cases' is missing in {file_path}",
              file=sys.stderr)
        sys.exit(1)

    for case in data["benchmark_cases"]:
        # Aggregate all steps from all cases
        all_steps.extend(
            create_benchmark_steps(case,
                                   global_env,
                                   file_path,
                                   file_basename,
                                   parent_dir,
                                   used_keys,
                                   errors,
                                   no_verify=args.no_verify))

    # Final check: Ensure we actually produced steps and no errors occurred
    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        sys.exit(1)

    if not all_steps:
        if _get_mlcompass_select_tests():
            all_steps.append({
                "label":
                "⏭️ Benchmark case skipped — not selected by MLCompass.",
                "command":
                "echo Benchmark case skipped — not selected by MLCompass.",
                "skip": True
            })
        else:
            print(f"Error: No steps were generated for {file_path}",
                  file=sys.stderr)
            sys.exit(1)

    # Wrap everything in a single group
    # Group name and key both use parent_dir for absolute uniqueness
    group_display_name = f"{parent_dir}-{file_basename}"
    group_key = clean_key_string(group_display_name)

    grouped_pipeline = {
        "steps": [{
            "group": group_display_name,
            "key": group_key,
            "depends_on": args.dependency_step or "build_docker",
            "steps": all_steps
        }]
    }

    print(
        yaml.dump(grouped_pipeline,
                  sort_keys=False,
                  default_flow_style=False,
                  width=float('inf')))


if __name__ == "__main__":
    main()
