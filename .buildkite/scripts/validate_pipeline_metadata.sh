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

# Allowed CI_STAGE values to prevent typos
ALLOWED_STAGES=(
    "UnitTest"
    "Accuracy/Correctness"
    "Benchmark"
    "CorrectnessTest"
    "PerformanceTest"
    "Single-Host CorrectnessTest"
    "Single-Host PerformanceTest"
    "Multi-Host CorrectnessTest"
    "Multi-Host PerformanceTest"
)

ALLOWED_QUEUES=(
    "cpu"
    "cpu_64_core"
    "tpu_v6e_queue"
    "tpu_v7x_2_queue"
    "tpu_v6e_8_queue"
    "tpu_v7x_8_queue"
    "tpu_v7x_16_queue"
)

# Helper function: check if an array contains a value
contains() {
    local n=$#
    local value=${!n}
    for ((i=1;i < $#;i++)); do
        if [ "${!i}" == "${value}" ]; then return 0; fi
    done
    return 1
}

# Define a list of directories within .buildkite/ that should be skipped during validation
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

# Helper to validate a field exists and is non-empty in step envs.
validate_field() {
    local label="$1"
    local file="$2"
    local tip="$3"

    # Ensure the field exists at least once in any step's env
    local exists
    exists=$(yq ".steps[].env.$label | select(. != null)" "$file" | head -1 || true)
    if [[ -z "$exists" ]]; then
        echo "+++ ❌ Error: Missing mandatory field '$label' in $file"
        echo "💡 Tip: $tip"
        echo ""
        ERRORS_FOUND=1
    else
        # Ensure that for all steps where it IS defined, it is not an empty string
        local empty_steps
        empty_steps=$(yq ".steps[] | select(.env.$label == \"\") | .key // .label // \"unknown_step\"" "$file")
        if [[ -n "$empty_steps" ]]; then
            echo "+++ ❌ Error: Detected empty value for '$label' in $file"
            echo "   Problematic steps: $(echo "$empty_steps" | xargs | tr ' ' ',')"
            echo "💡 Tip: $tip"
            echo ""
            ERRORS_FOUND=1
        fi
    fi
}

# Variable to track if any errors were found during validation
ERRORS_FOUND=0

# --- GLOBAL UNIQUENESS VALIDATION ---
# Ensure that CI_TARGET is unique across all spec directories
declare -a SPEC_DIRS=("quantization" "parallelism" "models" "features" "rl")
KERNEL_PARENT_DIR="$BUILDKITE_DIR/kernel_microbenchmarks"

echo "--- 🌍 Scanning all spec folders for global metadata uniqueness..."

if [[ -d "$KERNEL_PARENT_DIR" ]]; then
    while IFS= read -r dir; do
        SPEC_DIRS+=("${dir#"$BUILDKITE_DIR"/}")
    done < <(find "$KERNEL_PARENT_DIR" -maxdepth 1 -mindepth 1 -type d)
fi

declare -A CI_TARGETS

for folder in "${SPEC_DIRS[@]}"; do
    full_path="$BUILDKITE_DIR/$folder"
    [[ ! -d "$full_path" ]] && continue

    while IFS= read -r -d '' file; do
        # Global uniqueness & consistency check for CI_TARGET
        UNIQUE_TARGETS=$(yq '.steps[].env.CI_TARGET | select(. != null)' "$file" | sort -u)
        TARGET_COUNT=$(echo "$UNIQUE_TARGETS" | grep -c ".*" || echo 0)

        # Consistency: Ensure all steps in a file share the same CI_TARGET
        if [[ "$TARGET_COUNT" -gt 1 ]]; then
            echo "+++ ❌ Error: Multiple different 'CI_TARGET' values detected within $file"
            echo "Found: $(echo "$UNIQUE_TARGETS" | xargs | tr ' ' ',')"
            echo "💡 Tip: All steps within a single configuration file must share the same 'CI_TARGET'."
            echo ""
            ERRORS_FOUND=1
        fi

        while IFS= read -r t_val; do
            [[ -z "$t_val" ]] && continue
            
            # Global Uniqueness: CI_TARGETS must not be used by other files
            if [[ -n "${CI_TARGETS[$t_val]:-}" && "${CI_TARGETS[$t_val]}" != "$file" ]]; then
                echo "+++ ❌ Error: Global duplicate 'CI_TARGET: $t_val' detected!"
                echo "Conflict: $file and ${CI_TARGETS[$t_val]}"
                echo "💡 Tip: 'CI_TARGET' is a unique identifier for support matrices and cannot be shared across files."
                echo ""
                ERRORS_FOUND=1
            fi
            CI_TARGETS["$t_val"]="$file"
        done <<< "$UNIQUE_TARGETS"
    done < <(find "$full_path" -maxdepth 1 -type f \( -name "*.yml" -o -name "*.yaml" \) -print0)
done


# Metadata validation (Modified files only)
echo "--- 📂 Checking metadata completeness and consistency for changed files"

while IFS= read -r file; do
    [[ -z "$file" ]] || [[ ! -f "$file" ]] && continue

    # Check spec folders only
    if [[ "$file" =~ ^\.buildkite/(quantization|parallelism|models|features|rl|kernel_microbenchmarks)/ ]]; then
        echo "🔍 Verifying metadata for spec file: $file"

        # Field presence check
        validate_field "CI_TARGET"     "$file" "Identifier for result tracking. This maps to the name displayed in support matrices. For models, use the full model name on Hugging Face (e.g., meta-llama/Llama-3.1-8B-Instruct)."
        validate_field "CI_TPU_VERSION" "$file" "Specifies the TPU hardware generation (e.g., tpu6e, tpu7x) required for this test."
        validate_field "CI_STAGE"      "$file" "Defines the testing phase (e.g., UnitTest, Accuracy/Correctness, Benchmark, CorrectnessTest, PerformanceTest, Single-Host CorrectnessTest, Single-Host PerformanceTest, Multi-Host CorrectnessTest, Multi-Host PerformanceTest."
        validate_field "CI_CATEGORY"   "$file" "Determines which support matrix (e.g., multimodal, text-only, feature support matrix) the results belong to."

        # Extract values for consistency checks
        C_CAT=$(yq '.steps[].env.CI_CATEGORY | select(. != null)' "$file" | head -1 || true)

        # CI_CATEGORY consistency within file
        UNIQUE_CATEGORIES=$(yq '.steps[].env.CI_CATEGORY | select(. != null)' "$file" | sort -u)
        CAT_COUNT=$(echo "$UNIQUE_CATEGORIES" | grep -c ".*" || echo 0)

        if [[ "$CAT_COUNT" -gt 1 ]]; then
            echo "+++ ❌ Error: Multiple 'CI_CATEGORY' values in $file"
            echo "Found: $(echo "$UNIQUE_CATEGORIES" | xargs | tr ' ' ',')"
            echo "💡 Tip: All steps in a file must share the same 'CI_CATEGORY'."
            echo ""
            ERRORS_FOUND=1
        fi

        # CI_CATEGORY directory-specific allowed values
        if [[ "$file" =~ ^\.buildkite/models/ ]]; then
            if [[ ! "$C_CAT" =~ ^(text-only|multimodal|embedding|diffusion)$ ]]; then
                echo "+++ ❌ Error: Invalid CI_CATEGORY '$C_CAT' for models/ in $file"
                echo "💡 Tip: Files in 'models/' should use 'text-only', 'multimodal', 'embedding' or 'diffusion' as their CI_CATEGORY."
                echo ""
                ERRORS_FOUND=1
            fi
        elif [[ "$file" =~ ^\.buildkite/features/ ]]; then
            FEATURE_RE='^(feature|kernel) support matrix$'
            if [[ ! "$C_CAT" =~ $FEATURE_RE ]]; then
                echo "+++ ❌ Error: Invalid CI_CATEGORY '$C_CAT' for features/ in $file"
                echo "💡 Tip: Files in 'features/' should use 'feature support matrix' or 'kernel support matrix' as their CI_CATEGORY."
                echo ""
                ERRORS_FOUND=1
            fi
        elif [[ "$file" =~ ^\.buildkite/parallelism/ ]]; then
            if [[ "$C_CAT" != "parallelism support matrix" ]]; then
                echo "+++ ❌ Error: Invalid CI_CATEGORY '$C_CAT' for parallelism/ in $file"
                echo "💡 Tip: Files in 'parallelism/' must use 'parallelism support matrix' as their CI_CATEGORY."
                echo ""
                ERRORS_FOUND=1
            fi
        elif [[ "$file" =~ ^\.buildkite/quantization/ ]]; then
            if [[ "$C_CAT" != "quantization support matrix" ]]; then
                echo "+++ ❌ Error: Invalid CI_CATEGORY '$C_CAT' for quantization/ in $file"
                echo "💡 Tip: Files in 'quantization/' must use 'quantization support matrix' as their CI_CATEGORY."
                echo ""
                ERRORS_FOUND=1
            fi
        elif [[ "$file" =~ ^\.buildkite/rl/ ]]; then
            if [[ "$C_CAT" != "rl support matrix" ]]; then
                echo "+++ ❌ Error: Invalid CI_CATEGORY '$C_CAT' for rl/ in $file"
                echo "💡 Tip: Files in 'rl/' must use 'rl support matrix' as their CI_CATEGORY."
                echo ""
                ERRORS_FOUND=1
            fi
        elif [[ "$file" =~ ^\.buildkite/kernel_microbenchmarks/ ]]; then
            if [[ "$C_CAT" != "kernel support matrix microbenchmarks" ]]; then
                echo "+++ ❌ Error: Invalid CI_CATEGORY '$C_CAT' for kernel_microbenchmarks/ in $file"
                echo "💡 Tip: Files in 'kernel_microbenchmarks/' must use 'kernel support matrix microbenchmarks' as their CI_CATEGORY."
                echo ""
                ERRORS_FOUND=1
            fi
        fi

        # CI_STAGE whitelist check
        ALL_FILE_STAGES=$(yq '.steps[].env.CI_STAGE | select(. != null)' "$file" | sort -u)
        while read -r stage; do
            [[ -z "$stage" ]] && continue
            if ! contains "${ALLOWED_STAGES[@]}" "$stage"; then
                echo "+++ ❌ Error: Invalid CI_STAGE '$stage' in $file"
                echo "💡 Tip: Use approved stage names: ${ALLOWED_STAGES[*]}."
                echo ""
                ERRORS_FOUND=1
            fi
        done < <(echo "$ALL_FILE_STAGES")

        # Step key and label format (prefix check)
        INVALID_KEYS=$(yq '.steps[] | select(.key != null and (.key | test("^\$\{TPU_VERSION(:-|\})") | not)) | .key' "$file")
        if [[ -n "$INVALID_KEYS" ]]; then
            echo "+++ ❌ Error: Invalid key format in $file"
            echo "Found keys: $INVALID_KEYS"
            echo "💡 Tip: All step keys must start with '\${TPU_VERSION}'."
            echo ""
            ERRORS_FOUND=1
        fi

        INVALID_LABELS=$(yq '.steps[] | select(.label != null and (.label | test("^\$\{TPU_VERSION(:-|\})") | not)) | .label' "$file")
        if [[ -n "$INVALID_LABELS" ]]; then
            echo "+++ ❌ Error: Invalid label format in $file"
            echo "Found labels: $INVALID_LABELS"
            echo "💡 Tip: All step labels must start with '\${TPU_VERSION}'."
            echo ""
            ERRORS_FOUND=1
        fi

        # Recording step consistency (record_step_result.sh)
        RECORD_STEPS_JSON=$(yq '[.steps[] | select(.commands != null) | select(.commands[] | test("record_step_result.sh"))]' -o json "$file" || echo "[]")

        if [[ "$RECORD_STEPS_JSON" != "[]" ]]; then
            echo "$RECORD_STEPS_JSON" | jq -c '.[]' | while read -r step; do
                S_KEY=$(echo "$step" | jq -r '.key')
                S_DEP=$(echo "$step" | jq -r '.depends_on')
                S_ARG=$(echo "$step" | jq -r '.commands[] | select(test("record_step_result.sh"))' | sed -E 's/.*record_step_result.sh[[:space:]]+([^[:space:]]+).*/\1/')
                
                # Ensure recording step depends on and matches the target test key
                if [[ "$S_DEP" == "null" ]]; then
                    echo "+++ ❌ Error: Recording step '$S_KEY' in $file has no 'depends_on' field."
                    echo "💡 Tip: Every recording step must depend on the test step it is recording."
                    echo ""
                    ERRORS_FOUND=1
                elif [[ "$S_ARG" != "$S_DEP" ]]; then
                    echo "+++ ❌ Error: record_step_result.sh argument mismatch in $file"
                    echo "Step '$S_KEY' depends on '$S_DEP' but tries to record '$S_ARG'."
                    echo "💡 Tip: The argument to record_step_result.sh must match the 'depends_on' key."
                    echo ""
                    ERRORS_FOUND=1
                fi

                # Ensure recording step defines mandatory env vars (CI_TPU_VERSION, CI_TARGET, CI_STAGE, CI_CATEGORY)
                for meta_field in "CI_TPU_VERSION" "CI_TARGET" "CI_STAGE" "CI_CATEGORY"; do
                    meta_val=$(echo "$step" | jq -r ".env.$meta_field")
                    if [[ "$meta_val" == "null" ]]; then
                        echo "+++ ❌ Error: Recording step '$S_KEY' in $file missing metadata 'env.$meta_field'."
                        echo "💡 Tip: All recording steps must contain full metadata for result tracking."
                        echo ""
                        ERRORS_FOUND=1
                    fi
                done
            done
        fi

        # Require agents.queue for all steps
        STEPS_MISSING_QUEUE=$(yq '.steps[] | select(.wait == null and . != "wait" and .agents.queue == null) | .key // .label // "unknown_step"' "$file")
        if [[ -n "$STEPS_MISSING_QUEUE" ]]; then
            echo "+++ ❌ Error: Missing agents.queue in $file"
            echo "Problematic steps: $STEPS_MISSING_QUEUE"
            echo "💡 Tip: Every execution step must define an 'agents.queue'."
            echo ""
            ERRORS_FOUND=1
        fi

        # Queue name format check
        # shellcheck disable=SC2016
        INVALID_QUEUES=""
        while IFS=$'\t' read -r step_key step_queue; do
            [[ -z "$step_key" ]] && continue
            if [[ "$step_queue" =~ ^\$\{TPU_QUEUE_SINGLE(:-|\}) ]] || [[ "$step_queue" =~ ^\$\{TPU_QUEUE_MULTI(:-|\}) ]]; then
                continue
            fi
            if ! contains "${ALLOWED_QUEUES[@]}" "$step_queue"; then
                INVALID_QUEUES+="${step_key}: [${step_queue:-EMPTY}]"$'\n'
            fi
        done < <(yq '.steps[] | select(.agents.queue != null) | .key + "\t" + .agents.queue' -r "$file" || true)

        if [[ -n "$INVALID_QUEUES" ]]; then
            echo "+++ ❌ Error: Unsupported agents.queue in $file"
            echo "Found:"
            echo -e "$INVALID_QUEUES"
            echo "💡 Tip: Use 'cpu', '\${TPU_QUEUE_SINGLE}', '\${TPU_QUEUE_MULTI}', or hardware queues like 'tpu_v6e_queue'."
            echo ""
            ERRORS_FOUND=1
        fi

        # TENSOR_PARALLEL_SIZE matching for models/
        if [[ "$file" =~ ^\.buildkite/models/ ]]; then
            TP_QUEUE_ERRORS=""
            while IFS=$'\t' read -r step_key tp_size step_queue; do
                [[ -z "$step_key" ]] && continue
                # if the queue is "cpu", the step might not implement. Skip for now.
                [[ "$step_queue" == "cpu" ]] && continue

                is_multi=-1
                if [[ "$tp_size" =~ ^\$\{TENSOR_PARALLEL_SIZE_MULTI(:-|\}) ]] || [[ "$tp_size" == "8" ]]; then
                    is_multi=1
                elif [[ "$tp_size" =~ ^\$\{TENSOR_PARALLEL_SIZE_SINGLE(:-|\}) ]] || [[ "$tp_size" == "1" ]] || [[ "$tp_size" == "2" ]]; then
                    is_multi=0
                fi
                
                if [[ $is_multi -eq 1 ]]; then
                    if [[ ! "$step_queue" =~ ^\$\{TPU_QUEUE_MULTI(:-|\}) ]] && [[ ! "$step_queue" =~ _8_queue$ ]] && [[ ! "$step_queue" =~ _16_queue$ ]]; then
                        TP_QUEUE_ERRORS+="${step_key}: TP_SIZE=${tp_size} but queue=${step_queue} (Expected multi queue)"$'\n'
                    fi
                elif [[ $is_multi -eq 0 ]]; then
                    if [[ ! "$step_queue" =~ ^\$\{TPU_QUEUE_SINGLE(:-|\}) ]] && [[ "$step_queue" != "tpu_v6e_queue" ]] && [[ "$step_queue" != "tpu_v7x_2_queue" ]]; then
                        TP_QUEUE_ERRORS+="${step_key}: TP_SIZE=${tp_size} but queue=${step_queue} (Expected single queue)"$'\n'
                    fi
                fi

            done < <(yq '.steps[] | select(.env.TENSOR_PARALLEL_SIZE != null) | .key + "\t" + .env.TENSOR_PARALLEL_SIZE + "\t" + (.agents.queue // "missing")' -r "$file" || true)

            if [[ -n "$TP_QUEUE_ERRORS" ]]; then
                echo "+++ ❌ Error: Queue mismatch for TENSOR_PARALLEL_SIZE in $file"
                echo "Problematic steps:"
                echo -e "$TP_QUEUE_ERRORS"
                echo "💡 Tip: 'TENSOR_PARALLEL_SIZE' 1 or 2 requires a small scale queue (e.g. '\${TPU_QUEUE_SINGLE}'). 'TENSOR_PARALLEL_SIZE' 8 requires a large scale queue (e.g. '\${TPU_QUEUE_MULTI}')."
                echo ""
                ERRORS_FOUND=1
            fi
        fi

    else
        echo "⏭️  Skipping metadata check for non-spec file: $file"
    fi

done < <(echo "$YAML_FILES_TO_CHECK")

if [[ "$ERRORS_FOUND" -eq 1 ]]; then
    echo "❌ Metadata verification failed."
    exit 1
else
    echo "✅ Metadata verification passed."
    exit 0
fi
