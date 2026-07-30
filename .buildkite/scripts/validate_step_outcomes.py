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

import glob
import os
import re
import sys

# Define valid outcome allowlist in record_step_result.sh
VALID_OUTCOMES = {
    "passed",
    "skipped",
    "unverified",
    "not enough HBM",
}

# Regular expression to match: buildkite-agent meta-data set <key> "<value>"
METADATA_SET_PATTERN = re.compile(
    r'\bbuildkite-agent\s+meta-data\s+set\s+'
    r'(?:[^\s]+|"[^"]*"|\'[^\']*\')\s+'
    r'(?:([\'"])(?P<quoted_val>[^\'"]+)\1|(?P<raw_val>[^\s#]+))')


def validate_step_outcomes(buildkite_dir):
    """
  Validates that all custom outcome strings in buildkite-agent meta-data set
  conform to the valid outcomes in record_step_result.sh.
  """
    if not os.path.exists(buildkite_dir):
        print(f"Error: Directory {buildkite_dir} not found.")
        return False

    has_error = False
    target_dirs = [
        "models",
        "features",
        "parallelism",
        "quantization",
        "rl",
        "kernel_microbenchmarks",
    ]

    print(
        f"--- Validating custom outcomes in buildkite-agent meta-data set in: {buildkite_dir}"
    )

    filepaths = []
    for target in target_dirs:
        dir_path = os.path.join(buildkite_dir, target)
        if not os.path.isdir(dir_path):
            continue

        # Target only .yml and .yaml files recursively
        filepaths.extend(
            glob.glob(os.path.join(dir_path, "**", "*.y*ml"), recursive=True))

    for filepath in filepaths:
        with open(filepath, "r") as f:
            for line_num, line in enumerate(f, 1):
                # Strip Bash comments (ignore everything after '#')
                clean_line = line.split("#")[0].strip()
                if not clean_line:
                    continue

                # Match against Regex
                match = METADATA_SET_PATTERN.search(clean_line)
                if match:
                    # Extract outcome (prefer quoted value, fallback to unquoted)
                    outcome = match.group("quoted_val") or match.group(
                        "raw_val")
                    # Allow shell variables (starting with $) or valid allowlisted outcomes
                    if outcome not in VALID_OUTCOMES and not outcome.startswith(
                            "$"):
                        print(
                            f"❌ [Invalid Outcome] File: {filepath}:{line_num}")
                        print(f'Found invalid outcome string: "{outcome}"')
                        has_error = True
                    else:
                        print(
                            f'✅ [Valid] {filepath}:{line_num} -> "{outcome}"')

    if has_error:
        print(
            "\n--- ❌ Validation failed! Please ensure all meta-data set strings conform to the valid outcomes in record_step_result.sh."
        )
        return False

    print("\n--- ✅ All step outcome strings validated successfully!")
    return True


if __name__ == "__main__":
    # Dynamically find the git repository root
    import subprocess

    try:
        repo_root = subprocess.check_output(
            ["git", "rev-parse", "--show-toplevel"],
            stderr=subprocess.STDOUT).decode("utf-8").strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        # Fallback to structure-based detection if git is not available
        script_dir = os.path.dirname(os.path.abspath(__file__))
        repo_root = os.path.abspath(os.path.join(script_dir, "..", ".."))

    # Target directory for buildkite relative to repo root
    target_dir = os.path.join(repo_root, ".buildkite")

    if not validate_step_outcomes(target_dir):
        sys.exit(1)
