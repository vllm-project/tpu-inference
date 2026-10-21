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
import os
import sys
from collections import defaultdict


def validate_global_case_names(cases_root):
    """
    Validates that all case_name values across all JSON files in the 
    cases_root directory are globally unique.
    """
    case_name_to_files = defaultdict(list)
    files_processed = 0

    print(f"Starting global case_name validation in: {cases_root}")

    if not os.path.exists(cases_root):
        print(f"Error: Directory {cases_root} not found.")
        return False

    has_errors = False
    for root, _, files in os.walk(cases_root):
        for file in files:
            if file.endswith('.json'):
                file_path = os.path.join(root, file)
                files_processed += 1
                try:
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                        if 'benchmark_cases' in data:
                            for case in data['benchmark_cases']:
                                case_name = case.get('case_name')
                                if case_name:
                                    case_name_to_files[case_name].append(
                                        file_path)
                                else:
                                    print(
                                        f"Error: Missing 'case_name' in {file_path}"
                                    )
                                    has_errors = True
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")
                    has_errors = True

    if has_errors:
        print("❌ Validation failed. Please fix the errors and try again.")
        return False

    duplicates = {
        name: files
        for name, files in case_name_to_files.items() if len(files) > 1
    }

    if duplicates:
        print(
            "❌ Validation Error: Duplicate case_names found across benchmark files!"
        )
        print(
            "Each benchmark case must have a unique 'case_name' for database reporting."
        )
        for name, files in duplicates.items():
            print(f"Duplicate Name: '{name}'")
            print("Found in:")
            for f in sorted(files):
                print(f"  - {f}")
        print(f"Total duplicate names found: {len(duplicates)}")
        return False

    print(f"✅ All {len(case_name_to_files)} case_names are globally unique.")
    print(f"Processed {files_processed} files.")
    return True


if __name__ == "__main__":
    # Dynamically find the git repository root
    import subprocess
    try:
        repo_root = subprocess.check_output(
            ['git', 'rev-parse', '--show-toplevel'],
            stderr=subprocess.STDOUT).decode('utf-8').strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        # Fallback to structure-based detection if git is not available
        script_dir = os.path.dirname(os.path.abspath(__file__))
        repo_root = os.path.abspath(os.path.join(script_dir, "..", "..", ".."))

    # Target directory for benchmark cases relative to repo root
    target_dir = os.path.join(repo_root, ".buildkite/benchmark/cases")

    if not validate_global_case_names(target_dir):
        sys.exit(1)
