#!/usr/bin/env python3
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
r"""Buildkite YAML to GKE TPU Helm Values Converter.

This script extracts test and benchmark steps from Buildkite CI model pipeline
definitions (e.g. .buildkite/models/meta-llama_Llama-3_1-8B-Instruct.yml) and
converts them into GKE TPU Helm values YAML files for JobSet execution.

================================================================================
Key Concepts & Things to Know
================================================================================
1. Step Filtering:
   - Only steps invoking `.buildkite/scripts/run_in_docker.sh` (such as UnitTest,
     Accuracy, and Benchmark) are transferred.
   - Non-containerized CI steps (e.g. record_step_result.sh) are automatically
     skipped.

2. Dynamic Accelerator Replacement (${TPU_VERSION:-...}):
   - All occurrences of `${TPU_VERSION:-...}` in step keys, labels, commands,
     and environment variables are dynamically replaced with the value of the
     `--accelerator` flag.
   - The `--accelerator` flag defaults to "tpu7x". If you specify
     `--accelerator tpu6e`, all `${TPU_VERSION:-tpu6e}` will populate with
     "tpu6e" instead.

3. Job Naming & RFC 1123 Parity:
   - The JobSet replicatedJob name is derived from the substring after the
     last '_' of the step key, lowercased, and sanitized to RFC 1123 format
     (e.g. 'benchmark', 'unittest', 'accuracy') with a max length limit.
   - This ensures 100% compatibility with Kubernetes DNS subdomain and
     resource naming rules.

4. Unified `scriptJobs` Architecture:
   - Values are always structured as a `scriptJobs` list under `mode: "script"`.
   - Full pipeline conversion outputs all qualifying steps as independent
     ReplicatedJobs.
   - Single-step extraction (`--step <key>`) outputs a `scriptJobs` list with
     that single step.

5. Step Matching (`--step`):
   - Matches against the resolved or raw Buildkite step key (e.g.
     `tpu7x_meta-llama_Llama-3_1-8B-Instruct_Benchmark`).
   - If an invalid or unmatched key is passed, the script exits with an error
     and prints all available step keys found in the pipeline.

6. Base Values Inheritance:
   - Base settings (image registry, commit hashes, storage, resources, secrets)
     are automatically inherited from `values-transfer-template.yaml` or `values.yaml`
     in the chart directory.

================================================================================
Usage Examples
================================================================================
1. Convert all qualifying steps in a pipeline (outputs values-<model>-ci.yaml):
   $ python3 buildkite_to_helm.py \\
       -b /path/to/.buildkite/models/meta-llama_Llama-3_1-8B-Instruct.yml

2. Extract a single benchmark step by step key:
   $ python3 buildkite_to_helm.py \\
       -b /path/to/.buildkite/models/meta-llama_Llama-3_1-8B-Instruct.yml \\
       --step tpu7x_meta-llama_Llama-3_1-8B-Instruct_Benchmark

3. Target a different TPU accelerator (e.g. tpu6e):
   $ python3 buildkite_to_helm.py \\
       -b /path/to/.buildkite/models/meta-llama_Llama-3_1-8B-Instruct.yml \\
       --accelerator tpu6e

4. Override Tensor Parallel size and specify custom output file:
   $ python3 buildkite_to_helm.py \\
       -b /path/to/.buildkite/models/meta-llama_Llama-3_1-8B-Instruct.yml \\
       --step tpu7x_meta-llama_Llama-3_1-8B-Instruct_Benchmark \\
       --tensor-parallel-size 2 \\
       -o values-llama8b-bench.yaml

5. Preview generated YAML on stdout without writing to file:
   $ python3 buildkite_to_helm.py \\
       -b /path/to/.buildkite/models/meta-llama_Llama-3_1-8B-Instruct.yml \\
       -v

6. Deploy the generated values with Helm on GKE TPU:
   $ helm install llama8b-ci . -f values-meta-llama_Llama-3_1-8B-Instruct-ci.yaml
"""

import argparse
import os
import re
import sys
from typing import Any, Dict, List, Optional

import yaml


# Accelerator environment patterns matching .buildkite/scripts/upload_models_and_features.sh
ACCELERATOR_PROFILES: Dict[str, Dict[str, str]] = {
    "tpu7x": {
        "TPU_VERSION": "tpu7x",
        "TPU_QUEUE_SINGLE": "tpu_v7x_2_queue",
        "TPU_QUEUE_MULTI": "tpu_v7x_8_queue",
        "TENSOR_PARALLEL_SIZE_SINGLE": "2",
        "TENSOR_PARALLEL_SIZE_MULTI": "8",
    },
    "tpu6e": {
        "TPU_VERSION": "tpu6e",
        "TPU_QUEUE_SINGLE": "tpu_v6e_queue",
        "TPU_QUEUE_MULTI": "tpu_v6e_8_queue",
        "TENSOR_PARALLEL_SIZE_SINGLE": "1",
        "TENSOR_PARALLEL_SIZE_MULTI": "8",
    },
}


def resolve_bash_var(val: Any,
                     overrides: Optional[Dict[str, str]] = None) -> str:
    """Resolves bash parameter expansion syntax like ${VAR:-DEFAULT} or $VAR."""
    if not isinstance(val, str):
        return str(val) if val is not None else ""

    val = val.strip()

    def _replace_default(match):
        var_name = match.group(1)
        default_val = match.group(2)
        if overrides and var_name in overrides:
            return overrides[var_name]
        if var_name in os.environ:
            return os.environ[var_name]
        return default_val

    def _replace_simple(match):
        var_name = match.group(2) or match.group(3)
        if overrides and var_name in overrides:
            return overrides[var_name]
        return os.environ.get(var_name, "")

    val = re.sub(r"\$\{([a-zA-Z_][a-zA-Z0-9_]*):-?(.*?)\}", _replace_default,
                 val)
    val = re.sub(
        r"(\$\{([a-zA-Z_][a-zA-Z0-9_]*)\}|\$([a-zA-Z_][a-zA-Z0-9_]*))",
        _replace_simple,
        val,
    )
    return val


def clean_docker_command(raw_command: str) -> str:
    """Strips .buildkite/scripts/run_in_docker.sh and bash line continuations."""
    # Split by lines
    lines = [
        line.strip() for line in raw_command.strip().splitlines()
        if line.strip()
    ]
    if not lines:
        return ""

    joined = " ".join(lines)
    # Remove run_in_docker.sh invocation
    pattern = r"^(?:\.?/?\.?buildkite/scripts/)?run_in_docker\.sh\s*(?:\\\s*)?"
    cleaned = re.sub(pattern, "", joined).strip()
    return cleaned


def extract_stage_from_step(step: Dict[str, Any], index: int) -> str:
    """Infers stage name (unittest, accuracy, benchmark) from step key, label, or env."""
    key = str(step.get("key", ""))
    label = str(step.get("label", ""))

    combined = f"{key} {label}".lower()
    if "unittest" in combined or "unit test" in combined:
        return "unittest"
    if "accuracy" in combined or "correctness" in combined:
        return "accuracy"
    if "benchmark" in combined or "perf" in combined:
        return "benchmark"

    # Fallback: clean key or label
    if key:
        clean_key = re.sub(r"^\$\{[^}]+\}_?", "", key)
        parts = clean_key.split("_")
        if parts:
            return parts[-1].lower()
    return f"step{index}"


def sanitize_k8s_job_name(
    raw_key: str,
    stage: str,
    existing_names: set,
    max_len: int = 20,
) -> str:
    """Derives a valid RFC 1123 ReplicatedJob name.

  Extracts the substring after the last '_', converts to lowercase, replaces
  invalid characters with '-', and trims to max_len (default: 20 chars).
  Ensures uniqueness across jobs in the JobSet.
  """
    candidate = raw_key.strip() if raw_key else ""
    if "_" in candidate:
        candidate = candidate.rsplit("_", 1)[-1]
    elif not candidate:
        candidate = stage

    # Lowercase and convert non-alphanumeric characters to hyphens
    clean = re.sub(r"[^a-z0-9]+", "-", candidate.lower()).strip("-")
    if not clean:
        clean = re.sub(r"[^a-z0-9]+", "-",
                       stage.lower()).strip("-") or "runner"

    # Trim to max_len without trailing hyphen
    clean = clean[:max_len].rstrip("-")

    # Ensure uniqueness across ReplicatedJobs in this JobSet
    final_name = clean
    counter = 1
    while final_name in existing_names:
        suffix = f"-{counter}"
        prefix_len = max(1, max_len - len(suffix))
        final_name = f"{clean[:prefix_len].rstrip('-')}{suffix}"
        counter += 1

    existing_names.add(final_name)
    return final_name


def parse_buildkite_steps(
    buildkite_data: Dict[str, Any],
    tp_size_override: Optional[str] = None,
    accelerator_override: Optional[str] = "tpu7x",
) -> List[Dict[str, Any]]:
    """Extracts and parses all qualifying run_in_docker.sh steps."""
    qualifying_steps = []
    existing_job_names = set()
    steps = buildkite_data.get("steps", [])

    overrides: Dict[str, str] = {}
    if accelerator_override:
        if accelerator_override not in ACCELERATOR_PROFILES:
            supported = ", ".join(f"'{k}'" for k in ACCELERATOR_PROFILES.keys())
            raise ValueError(
                f"Unsupported accelerator '{accelerator_override}'. Supported accelerators: {supported}"
            )
        overrides.update(ACCELERATOR_PROFILES[accelerator_override])

    if tp_size_override:
        overrides["TENSOR_PARALLEL_SIZE_SINGLE"] = str(tp_size_override)
        overrides["TENSOR_PARALLEL_SIZE_MULTI"] = str(tp_size_override)
        overrides["TENSOR_PARALLEL_SIZE"] = str(tp_size_override)

    for idx, step in enumerate(steps):
        if not isinstance(step, dict):
            continue

        raw_commands = step.get("commands", [])
        if isinstance(raw_commands, str):
            raw_commands = [raw_commands]

        # Find commands using run_in_docker.sh
        docker_cmd_str = None
        for cmd in raw_commands:
            if "run_in_docker.sh" in cmd:
                docker_cmd_str = cmd
                break

        if not docker_cmd_str:
            continue

        # Extract pure command
        clean_cmd = clean_docker_command(docker_cmd_str)
        clean_cmd = resolve_bash_var(clean_cmd, overrides=overrides)
        stage = extract_stage_from_step(step, idx)
        raw_key = step.get("key", "")
        resolved_key = resolve_bash_var(raw_key,
                                        overrides=overrides) if raw_key else ""

        # Sanitize job name: substring after last '_', lowercase, max 20 chars
        job_name = sanitize_k8s_job_name(
            raw_key=resolved_key or raw_key,
            stage=stage,
            existing_names=existing_job_names,
            max_len=20,
        )

        # Resolve env vars
        step_env = step.get("env", {})
        parsed_env: Dict[str, str] = {}

        for k, v in step_env.items():
            if k == "TENSOR_PARALLEL_SIZE" and tp_size_override:
                parsed_env[k] = str(tp_size_override)
            else:
                parsed_env[k] = resolve_bash_var(v, overrides=overrides)

        qualifying_steps.append({
            "name":
            job_name,
            "stage":
            stage,
            "key":
            raw_key,
            "resolved_key":
            resolved_key,
            "label":
            resolve_bash_var(step.get("label", ""), overrides=overrides),
            "command":
            clean_cmd,
            "workingDir":
            "/workspace/vllm",
            "backoffLimit":
            1 if not step.get("soft_fail") else 0,
            "env":
            parsed_env,
        })

    return qualifying_steps


def load_base_values(base_path: Optional[str]) -> Dict[str, Any]:
    """Loads base values YAML if present, else returns standard CI defaults."""
    if base_path and os.path.isfile(base_path):
        with open(base_path, "r") as f:
            data = yaml.safe_load(f)
            if isinstance(data, dict):
                return data

    # Default fallback template
    return {
        "mode": "script",
        "image": {
            "registry":
            ("us-central1-docker.pkg.dev/cloud-tpu-shared-capacity/tpu-images/dennis-vllm-tpu"
             ),
            "tpuInferenceCommit":
            "",
            "vllmCommit":
            "",
        },
        "builder": {
            "enabled": False,
            "gitRepo": "https://github.com/vllm-project/tpu-inference.git",
        },
        "tpu": {
            "accelerator": "tpu7x",
            "topology": "2x2x1",
        },
        "resources": {
            "tpu": 4,
            "cpu": "32",
            "memory": "100Gi",
        },
        "storage": {
            "type": "ramdisk",
        },
        "hfTokenSecret": {
            "name": "dennis-hf-token",
            "key": "token",
        },
    }


def generate_helm_values(
    parsed_steps: List[Dict[str, Any]],
    base_values: Dict[str, Any],
    target_step: Optional[str] = None,
    accelerator: Optional[str] = "tpu7x",
    topology: Optional[str] = None,
    tpu_limit: Optional[int] = None,
    registry: Optional[str] = None,
    hf_secret_name: Optional[str] = None,
    hf_secret_key: Optional[str] = None,
) -> Dict[str, Any]:
    """Assembles final Helm values dictionary."""
    values = dict(base_values)
    values["mode"] = "script"

    # Image overrides
    if "image" not in values or not isinstance(values["image"], dict):
        values["image"] = {}
    if registry:
        values["image"]["registry"] = registry

    # TPU overrides
    if "tpu" not in values or not isinstance(values["tpu"], dict):
        values["tpu"] = {}
    if accelerator:
        values["tpu"]["accelerator"] = accelerator
    if topology:
        values["tpu"]["topology"] = topology

    # Resources
    if "resources" not in values or not isinstance(values["resources"], dict):
        values["resources"] = {}
    if tpu_limit is not None:
        values["resources"]["tpu"] = tpu_limit

    # HF Secret
    if "hfTokenSecret" not in values or not isinstance(values["hfTokenSecret"],
                                                       dict):
        values["hfTokenSecret"] = {}
    if hf_secret_name:
        values["hfTokenSecret"]["name"] = hf_secret_name
    if hf_secret_key:
        values["hfTokenSecret"]["key"] = hf_secret_key

    # Filter steps if a specific step key is requested
    steps_to_include = parsed_steps
    if target_step:
        target_clean = target_step.strip()
        target_lower = target_clean.lower()
        matched = []
        # 1. Exact match against step name, resolved key, raw key, or stage
        for step in parsed_steps:
            resolved_key = step.get("resolved_key", "").lower()
            if (step["name"].lower() == target_lower
                    or resolved_key == target_lower
                    or step["key"].lower() == target_lower
                    or step["stage"].lower() == target_lower):
                matched.append(step)

        # 2. Fallback: match if target is a substring of name, resolved key, or key
        if not matched:
            for step in parsed_steps:
                resolved_key = step.get("resolved_key", "").lower()
                if (target_lower in step["name"].lower()
                        or (resolved_key and target_lower in resolved_key)
                        or target_lower in step["key"].lower()
                        or target_lower in step["stage"].lower()):
                    matched.append(step)

        if not matched:
            available = [
                f"'{s['name']}' (key: {s.get('resolved_key') or s['key']})"
                for s in parsed_steps
            ]
            raise ValueError(
                f"Step key '{target_step}' not found in pipeline.\n"
                "Available steps:\n  " + "\n  ".join(available))
        steps_to_include = matched

    # Always generate `scriptJobs` (even for a single step)
    values.pop("script", None)
    values["scriptJobs"] = []
    for step in steps_to_include:
        values["scriptJobs"].append({
            "name": step["name"],
            "command": step["command"],
            "workingDir": step["workingDir"],
            "backoffLimit": step["backoffLimit"],
            "env": step["env"],
        })

    return values


def format_yaml_comment_header(
    source_file: str,
    steps_count: int,
) -> str:
    """Formats top-level header comments for generated values YAML."""
    header = [
        "# Copyright 2026 Google LLC",
        "#",
        "# Licensed under the Apache License, Version 2.0 (the \"License\");",
        "# you may not use this file except in compliance with the License.",
        "# You may obtain a copy of the License at",
        "#",
        "#     http://www.apache.org/licenses/LICENSE-2.0",
        "#",
        "# Unless required by applicable law or agreed to in writing, software",
        "# distributed under the License is distributed on an \"AS IS\" BASIS,",
        "# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.",
        "# See the License for the specific language governing permissions and",
        "# limitations under the License.",
        "",
        "# ==============================================================================",
        "# GKE TPU vLLM CI Configuration (Transferred from Buildkite)",
        f"# Source Buildkite YAML: {source_file}",
        f"# Script Jobs Count:     {steps_count}",
        "# ==============================================================================",
        "",
    ]
    return "\n".join(header)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Transfer Buildkite CI model YAML to GKE TPU Helm values.yaml"),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  1. Convert all qualifying steps in Buildkite pipeline to multi-job Helm values:
     python3 buildkite_to_helm.py -b /path/to/.buildkite/models/meta-llama_Llama-3_1-8B-Instruct.yml -o values-meta-llama_Llama-3_1-8B-Instruct-ci.yaml

  2. Extract a specific step by matching its step key:
     python3 buildkite_to_helm.py -b /path/to/.buildkite/models/meta-llama_Llama-3_1-8B-Instruct.yml --step tpu7x_meta-llama_Llama-3_1-8B-Instruct_Benchmark -o values-llama8b-bench.yaml

  3. Preview generated YAML on stdout:
     python3 buildkite_to_helm.py -b /path/to/.buildkite/models/meta-llama_Llama-3_1-8B-Instruct.yml -v
""",
    )

    parser.add_argument(
        "-b",
        "--buildkite-yml",
        required=True,
        help=("Path to Buildkite model pipeline YAML file (e.g."
              " .buildkite/models/meta-llama_Llama-3_1-8B-Instruct.yml)"),
    )
    parser.add_argument(
        "-o",
        "--output",
        default="",
        help=("Output path for generated Helm values.yaml (default:"
              " values-<buildkite_file_basename>-ci.yaml)"),
    )
    parser.add_argument(
        "--base-values",
        default="",
        help=(
            "Path to base Helm values.yaml to inherit settings from (default:"
            " auto-detect values-transfer-template.yaml or values.yaml)"),
    )
    parser.add_argument(
        "--step",
        default="",
        help=("Extract a specific step only by matching its step key (e.g."
              " 'tpu7x_meta-llama_Llama-3_1-8B-Instruct_Benchmark')."),
    )
    parser.add_argument(
        "-v",
        "--values-only",
        action="store_true",
        help="Print generated values.yaml to stdout without writing to file",
    )

    # TPU & Resource Overrides
    parser.add_argument(
        "--accelerator",
        default="tpu7x",
        choices=list(ACCELERATOR_PROFILES.keys()),
        help=(
            f"TPU accelerator type (choices: {', '.join(ACCELERATOR_PROFILES.keys())})."
            " Defaults to 'tpu7x'."),
    )
    parser.add_argument(
        "--topology",
        default="",
        help=(
            "TPU topology shape (e.g. 2x2x1, 2x2x2). Defaults to base values or"
            " '2x2x1'."),
    )
    parser.add_argument(
        "--tpu-limit",
        type=int,
        default=None,
        help=
        ("TPU resource limit (e.g. 4). Defaults to topology chip count or base"
         " values."),
    )
    parser.add_argument(
        "--tensor-parallel-size",
        default="",
        help=(
            "Override TENSOR_PARALLEL_SIZE in step environment variables (e.g."
            " '2' or '8')."),
    )

    # Image & Secret Overrides
    parser.add_argument(
        "--registry",
        default="",
        help="Container image registry override.",
    )
    parser.add_argument(
        "--hf-secret-name",
        default="",
        help="Kubernetes secret name for HuggingFace token.",
    )
    parser.add_argument(
        "--hf-secret-key",
        default="",
        help="Kubernetes secret key for HuggingFace token.",
    )

    args = parser.parse_args()

    bk_path = os.path.abspath(args.buildkite_yml)
    if not os.path.isfile(bk_path):
        sys.stderr.write(f"Error: Buildkite YAML file not found: {bk_path}\n")
        sys.exit(1)

    with open(bk_path, "r") as f:
        bk_data = yaml.safe_load(f)

    if not isinstance(bk_data, dict) or "steps" not in bk_data:
        sys.stderr.write(
            f"Error: Invalid Buildkite pipeline format in {bk_path}\n")
        sys.exit(1)

    accelerator = args.accelerator or "tpu7x"

    # Parse qualifying steps
    parsed_steps = parse_buildkite_steps(
        bk_data,
        tp_size_override=args.tensor_parallel_size or None,
        accelerator_override=accelerator,
    )
    if not parsed_steps:
        sys.stderr.write(
            "No steps found using .buildkite/scripts/run_in_docker.sh in this"
            " pipeline.\n")
        sys.exit(1)

    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Auto-detect base values if not provided
    base_values_path = args.base_values
    if not base_values_path:
        candidate = os.path.join(script_dir, "values-transfer-template.yaml")
        if os.path.isfile(candidate):
            base_values_path = candidate
        else:
            candidate = os.path.join(script_dir, "values.yaml")
            if os.path.isfile(candidate):
                base_values_path = candidate

    base_values = load_base_values(base_values_path)

    # Generate final values
    try:
        final_values = generate_helm_values(
            parsed_steps=parsed_steps,
            base_values=base_values,
            target_step=args.step or None,
            accelerator=accelerator,
            topology=args.topology or None,
            tpu_limit=args.tpu_limit,
            registry=args.registry or None,
            hf_secret_name=args.hf_secret_name or None,
            hf_secret_key=args.hf_secret_key or None,
        )
    except ValueError as e:
        sys.stderr.write(f"Error: {e}\n")
        sys.exit(1)

    jobs_count = len(final_values.get("scriptJobs", []))
    header_comment = format_yaml_comment_header(
        source_file=bk_path,
        steps_count=jobs_count,
    )

    yaml_str = yaml.dump(
        final_values,
        default_flow_style=False,
        sort_keys=False,
        allow_unicode=True,
    )
    full_output = f"{header_comment}{yaml_str}"

    if args.values_only:
        print(full_output)
        return

    # Determine output file path
    out_path = args.output
    if not out_path:
        base_file = os.path.splitext(os.path.basename(bk_path))[0]
        suffix = f"-{args.step}" if args.step else "-ci"
        out_path = os.path.join(script_dir, f"values-{base_file}{suffix}.yaml")

    out_path = os.path.abspath(out_path)
    with open(out_path, "w") as f:
        f.write(full_output)

    job_names = [j["name"] for j in final_values.get("scriptJobs", [])]
    print(f" Successfully transferred {jobs_count} step(s) to: {out_path}")
    print(f"   Configured scriptJobs: {job_names}")

    print("\nTo deploy this JobSet onto GKE TPU with Helm, run:")
    rel_name = os.path.splitext(os.path.basename(out_path))[0].replace(
        "values-", "")
    print(f"  helm install {rel_name} {script_dir} -f {out_path}")


if __name__ == "__main__":
    main()
