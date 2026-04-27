#!/bin/bash
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

# Exit on error, exit on unset variable, fail on pipe errors.
set -euo pipefail

RAW_FILES_TO_CHECK="${1:-}"
BUILDKITE_DIR=".buildkite"

# Define directories relative to the repo root for exact matching
EXCLUDED_FOLDERS=(
    "\.buildkite/kubernetes/"
    "\.buildkite/benchmark/lm_eval/"
)

# Convert the array into a pipe-separated string for regex
EXCLUDE_PATTERN=$(printf "|%s" "${EXCLUDED_FOLDERS[@]}")
EXCLUDE_PATTERN=${EXCLUDE_PATTERN:1}

# Filter: Include YAML files in .buildkite/ while skipping excluded patterns
YAML_FILES_TO_CHECK=$(echo "$RAW_FILES_TO_CHECK" | \
    grep -E "^\.buildkite/.*\.ya?ml$" | \
    grep -Ev "^($EXCLUDE_PATTERN)" || true)

# Early exit: If no YAML files were modified, skip validation
if [ -z "$YAML_FILES_TO_CHECK" ]; then
    echo "--- :fast_forward: No applicable YAML changes found (none detected or all excluded). Skipping validation."
    exit 0
fi

# Helper function to enforce that ALL occurrences of a field in a file are not empty.
validate_all_entries() {
    local label="$1"
    local pattern="$2"
    local file="$3"
    local description="$4"

    # Fetch all lines matching the pattern.
    local lines
    lines=$(grep -E "$pattern" "$file" || true)

    # Ensure the field appears at least once in the file.
    if [[ -z "$lines" ]]; then
        echo "+++ ❌ Error: Missing mandatory field '$label' in $file"
        echo "💡 Tip: $description"
        exit 1
    fi

    # Iterate through every match to check for empty values.
    while IFS= read -r line; do
        local val
        # Extract value after colon, remove quotes, and trim whitespace.
        val=$(echo "$line" | sed 's/^[^:]*:[[:space:]]*//' | tr -d '"'\' | xargs)
        if [[ -z "$val" ]]; then
            echo "+++ ❌ Error: Detected empty value for '$label' in $file"
            echo "   Problematic line: $line"
            echo "💡 Tip: $description"
            exit 1
        fi
    done <<< "$lines"
}

# --- GLOBAL UNIQUENESS VALIDATION ---
# Ensure that pipeline-name and CI_TARGET are unique across all spec directories
declare -a SPEC_DIRS=("quantization" "parallelism" "models" "features" "rl")
KERNEL_PARENT_DIR="$BUILDKITE_DIR/kernel_microbenchmarks"

echo "--- 🌍 Scanning all spec folders for global metadata uniqueness..."

if [[ -d "$KERNEL_PARENT_DIR" ]]; then
    while IFS= read -r dir; do
        SPEC_DIRS+=("${dir#"$BUILDKITE_DIR"/}")
    done < <(find "$KERNEL_PARENT_DIR" -maxdepth 1 -mindepth 1 -type d)
fi

declare -A PIPELINE_NAMES
declare -A CI_TARGETS

for folder in "${SPEC_DIRS[@]}"; do
    full_path="$BUILDKITE_DIR/$folder"
    [[ ! -d "$full_path" ]] && continue

    while IFS= read -r -d '' file; do
        # Extract the primary pipeline-name for the file
        P_NAME=$(awk '/^[[:space:]]*#[[:space:]]*pipeline-name:/ {print $0; exit}' "$file" | sed 's/^[^:]*:[[:space:]]*//' | xargs)
        
        # Global uniqueness check for pipeline-name
        if [[ -n "$P_NAME" ]]; then
            if [[ -n "${PIPELINE_NAMES[$P_NAME]:-}" && "${PIPELINE_NAMES[$P_NAME]}" != "$file" ]]; then
                echo "+++ ❌ Error: Global duplicate '# pipeline-name: $P_NAME' detected!"
                echo "Conflict: $file and ${PIPELINE_NAMES[$P_NAME]}"
                echo "💡 Tip: The '# pipeline-name' comment must be unique across all configuration files. It serves as the display title in the support matrices. For models, ensure you use the full model name on Hugging Face (e.g., meta-llama/Llama-3.1-8B) to prevent name collisions."
                exit 1
            fi
            PIPELINE_NAMES["$P_NAME"]="$file"
        fi

        # Global uniqueness check for CI_TARGET
        while IFS= read -r line; do
            t_val=$(echo "$line" | sed 's/^[^:]*:[[:space:]]*//' | tr -d '"'\' | xargs)
            if [[ -n "$t_val" ]]; then
                if [[ -n "${CI_TARGETS[$t_val]:-}" && "${CI_TARGETS[$t_val]}" != "$file" ]]; then
                    echo "+++ ❌ Error: Global duplicate 'CI_TARGET: $t_val' detected!"
                    echo "Conflict: $file and ${CI_TARGETS[$t_val]}"
                    echo "💡 Tip: 'CI_TARGET' is the unique identifier used by the backend for result tracking. It must be distinct for every test configuration and must be identical to the '# pipeline-name' defined at the top of the file."
                    exit 1
                fi
                CI_TARGETS["$t_val"]="$file"
            fi
        done < <(grep -E "^[[:space:]]*CI_TARGET:" "$file" || true)
    done < <(find "$full_path" -maxdepth 1 -type f \( -name "*.yml" -o -name "*.yaml" \) -print0)
done


# --- METADATA COMPLETENESS & CONSISTENCY (Modified files only) ---
echo "--- 📂 Checking metadata completeness and consistency for changed files"

while IFS= read -r file; do
    [ -z "$file" ] || [ ! -f "$file" ] && continue

    # Check presence ONLY for changed files inside spec folders.
    if [[ "$file" =~ ^\.buildkite/(quantization|parallelism|models|features|rl|kernel_microbenchmarks)/ ]]; then
        echo "🔍 Verifying metadata for spec file: $file"

        # Audit ALL entries to ensure no empty fields exist in any step.
        validate_all_entries "pipeline-name"   "^[[:space:]]*#[[:space:]]*pipeline-name:" "$file" "This maps to the name displayed in support matrices. For models, use the full model name on Hugging Face (e.g., meta-llama/Llama-3.1-8B-Instruct). Acts as the canonical reference; every 'CI_TARGET' entry must match this value."
        validate_all_entries "CI_TARGET"       "^[[:space:]]*CI_TARGET:"      "$file" "Identifier for result tracking. Every 'CI_TARGET' entry must be identical to the '# pipeline-name' defined at the top of the file."
        validate_all_entries "CI_TPU_VERSION"  "^[[:space:]]*CI_TPU_VERSION:" "$file" "Specifies the TPU hardware generation (e.g., tpu6e, tpu7x) required for this test."
        validate_all_entries "CI_STAGE"        "^[[:space:]]*CI_STAGE:"       "$file" "Defines the testing phase (e.g., UnitTest, Accuracy/Correctness, Benchmark, CorrectnessTest, PerformanceTest)."
        validate_all_entries "CI_CATEGORY"     "^[[:space:]]*CI_CATEGORY:"    "$file" "Determines which support matrix (e.g., multimodal, text-only, feature support matrix) the results belong to."

        # Consistency: Ensure all 'CI_TARGET' entries match the '# pipeline-name'.
        # This also ensures all CI_TARGET entries in the same file are identical.
        FILE_P_NAME=$(awk '/^[[:space:]]*#[[:space:]]*pipeline-name:/ {print $0; exit}' "$file" | sed 's/^[^:]*:[[:space:]]*//' | xargs)
        while IFS= read -r line; do
            [ -z "$line" ] && continue
            FILE_T_VAL=$(echo "$line" | sed 's/^[^:]*:[[:space:]]*//' | tr -d '"'\' | xargs)
            
            if [[ "$FILE_T_VAL" != "$FILE_P_NAME" ]]; then
                echo "+++ ❌ Error: Metadata mismatch detected in $file"
                echo "   The '# pipeline-name' is: '$FILE_P_NAME'"
                echo "   But a 'CI_TARGET' is set to: '$FILE_T_VAL'"
                echo "💡 Tip: All 'CI_TARGET' entries in a file must be identical and match the '# pipeline-name' comment."
                exit 1
            fi
        done < <(grep -E "^[[:space:]]*CI_TARGET:" "$file" || true)

    else
        echo "⏭️  Skipping metadata check for non-spec file: $file"
    fi

done < <(echo "$YAML_FILES_TO_CHECK")

echo "✅ Metadata verification passed."
exit 0
