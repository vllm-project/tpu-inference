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

ALLOWED_SINGLE_QUEUES=(
    "cpu"
    "tpu_v6e_queue"
    "tpu_v7x_2_queue"
    "\${TPU_QUEUE_SINGLE:-tpu_v6e_queue}"
)

ALLOWED_MULTI_QUEUES=(
    "cpu_64_core"
    "tpu_v6e_8_queue"
    "tpu_v7x_8_queue"
    "tpu_v7x_16_queue"
    "\${TPU_QUEUE_MULTI:-tpu_v6e_8_queue}"
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

# Helper to extract value from a comment like '# key: value'
get_comment_value() {
    local key="$1"
    local file="$2"
    grep -E "^[[:space:]]*#[[:space:]]*$key:" "$file" | head -1 | sed 's/^[^:]*:[[:space:]]*//' | xargs || true
}

# Helper to validate a field exists and is non-empty. 
# If is_comment is true, it checks the file's comments.
# Otherwise, it ensures at least one step has the field and no step has an empty value for it.
validate_field() {
    local label="$1"
    local file="$2"
    local tip="$3"
    local is_comment="${4:-false}"

    if [[ "$is_comment" == "true" ]]; then
        local val
        val=$(get_comment_value "$label" "$file")
        if [[ -z "$val" ]]; then
            echo "+++ ❌ Error: Missing or empty mandatory comment '# $label' in $file"
            echo "💡 Tip: $tip"
            ERRORS_FOUND=1
        fi
    else
        # 1. Ensure the field exists at least once in any step's env
        local exists
        exists=$(yq ".steps[].env.$label | select(. != null)" "$file" | head -1 || true)
        if [[ -z "$exists" ]]; then
            echo "+++ ❌ Error: Missing mandatory field '$label' in $file"
            echo "💡 Tip: $tip"
            ERRORS_FOUND=1
        else
            # 2. Ensure that for all steps where it IS defined, it is not an empty string
            local empty_steps
            empty_steps=$(yq ".steps[] | select(.env.$label == \"\") | .key // .label // \"unknown_step\"" "$file")
            if [[ -n "$empty_steps" ]]; then
                echo "+++ ❌ Error: Detected empty value for '$label' in $file"
                echo "   Problematic steps: $(echo "$empty_steps" | xargs | tr ' ' ',')"
                echo "💡 Tip: $tip"
                ERRORS_FOUND=1
            fi
        fi
    fi
}

# Variable to track if any errors were found during validation
ERRORS_FOUND=0

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
        P_NAME=$(get_comment_value "pipeline-name" "$file")

        # Global uniqueness check for pipeline-name
        if [[ -n "$P_NAME" ]]; then
            if [[ -n "${PIPELINE_NAMES[$P_NAME]:-}" && "${PIPELINE_NAMES[$P_NAME]}" != "$file" ]]; then
                echo "+++ ❌ Error: Global duplicate '# pipeline-name: $P_NAME' detected!"
                echo "Conflict: $file and ${PIPELINE_NAMES[$P_NAME]}"
                echo "💡 Tip: The '# pipeline-name' comment must be unique across all configuration files. It serves as the display title in the support matrices. For models, ensure you use the full model name on Hugging Face (e.g., meta-llama/Llama-3.1-8B) to prevent name collisions."
                ERRORS_FOUND=1
            fi
            PIPELINE_NAMES["$P_NAME"]="$file"
        fi

        # Global uniqueness & consistency check for CI_TARGET
        UNIQUE_TARGETS=$(yq '.steps[].env.CI_TARGET | select(. != null)' "$file" | sort -u)
        TARGET_COUNT=$(echo "$UNIQUE_TARGETS" | grep -c ".*" || echo 0)

        if [[ "$TARGET_COUNT" -gt 1 ]]; then
            echo "+++ ❌ Error: Multiple different 'CI_TARGET' values detected within $file"
            echo "Found: $(echo "$UNIQUE_TARGETS" | xargs | tr ' ' ',')"
            echo "💡 Tip: All steps within a single configuration file must share the same 'CI_TARGET'."
            ERRORS_FOUND=1
        fi

        while IFS= read -r t_val; do
            [[ -z "$t_val" ]] && continue
            
            # 1. Consistency: Must match the pipeline-name header
            if [[ -n "$P_NAME" && "$t_val" != "$P_NAME" ]]; then
                echo "+++ ❌ Error: Metadata mismatch in $file"
                echo "   '# pipeline-name' is: '$P_NAME'"
                echo "   But 'CI_TARGET' is set to: '$t_val'"
                echo "💡 Tip: 'CI_TARGET' must be identical to the '# pipeline-name' comment at the top of the file."
                ERRORS_FOUND=1
            fi

            # 2. Global Uniqueness: Must not be used by other files
            if [[ -n "${CI_TARGETS[$t_val]:-}" && "${CI_TARGETS[$t_val]}" != "$file" ]]; then
                echo "+++ ❌ Error: Global duplicate 'CI_TARGET: $t_val' detected!"
                echo "Conflict: $file and ${CI_TARGETS[$t_val]}"
                echo "💡 Tip: 'CI_TARGET' is a unique identifier for support matrices and cannot be shared across files."
                ERRORS_FOUND=1
            fi
            CI_TARGETS["$t_val"]="$file"
        done <<< "$UNIQUE_TARGETS"
    done < <(find "$full_path" -maxdepth 1 -type f \( -name "*.yml" -o -name "*.yaml" \) -print0)
done


# --- METADATA COMPLETENESS & CONSISTENCY (Modified files only) ---
echo "--- 📂 Checking metadata completeness and consistency for changed files"

while IFS= read -r file; do
    [ -z "$file" ] || [ ! -f "$file" ] && continue

    # Check presence ONLY for changed files inside spec folders.
    if [[ "$file" =~ ^\.buildkite/(quantization|parallelism|models|features|rl|kernel_microbenchmarks)/ ]]; then
        echo "🔍 Verifying: $file"

        # Audit ALL entries to ensure no empty fields exist in steps.
        validate_field "pipeline-name" "$file" "This maps to the name displayed in support matrices. For models, use the full model name on Hugging Face (e.g., meta-llama/Llama-3.1-8B-Instruct). Acts as the canonical reference; every 'CI_TARGET' entry must match this value." true
        validate_field "pipeline-type" "$file" "Specifies the matrix type (e.g., text-only, feature support matrix). Every 'CI_CATEGORY' entry in the steps must match this value." true
        validate_field "CI_TARGET"     "$file" "Identifier for result tracking. Every 'CI_TARGET' entry must be identical to the '# pipeline-name' defined at the top of the file."
        validate_field "CI_TPU_VERSION" "$file" "Specifies the TPU hardware generation (e.g., tpu6e, tpu7x) required for this test."
        validate_field "CI_STAGE"      "$file" "Defines the testing phase (e.g., UnitTest, Accuracy/Correctness, Benchmark, CorrectnessTest, PerformanceTest, Single-Host CorrectnessTest, Single-Host PerformanceTest, Multi-Host CorrectnessTest, Multi-Host PerformanceTest."
        validate_field "CI_CATEGORY"   "$file" "Determines which support matrix the results belong to. Every 'CI_CATEGORY' must match the '# pipeline-type' comment at the top of the file."

        # Extract primary metadata values for specific validation checks
        P_NAME=$(get_comment_value "pipeline-name" "$file")
        P_TYPE=$(get_comment_value "pipeline-type" "$file")
        C_CAT=$(yq '.steps[].env.CI_CATEGORY | select(. != null)' "$file" | head -1 || true)

        # B. Metadata Consistency
        # B1. CI_CATEGORY Consistency: All steps must share the same category and match # pipeline-type
        UNIQUE_CATEGORIES=$(yq '.steps[].env.CI_CATEGORY | select(. != null)' "$file" | sort -u)
        CAT_COUNT=$(echo "$UNIQUE_CATEGORIES" | grep -c ".*" || echo 0)

        if [[ "$CAT_COUNT" -gt 1 ]]; then
            echo "+++ ❌ Error: Multiple different 'CI_CATEGORY' values detected within $file"
            echo "Found: $(echo "$UNIQUE_CATEGORIES" | xargs | tr ' ' ',')"
            echo "💡 Tip: All steps within a single configuration file must share the same 'CI_CATEGORY'."
            ERRORS_FOUND=1
        fi

        while IFS= read -r c_val; do
            [[ -z "$c_val" ]] && continue
            if [[ -n "$P_TYPE" && "$c_val" != "$P_TYPE" ]]; then
                echo "+++ ❌ Error: Category mismatch detected in $file"
                echo "   The '# pipeline-type' is: '$P_TYPE'"
                echo "   But a 'CI_CATEGORY' is set to: '$c_val'"
                echo "💡 Tip: All 'CI_CATEGORY' entries in a file must be identical and match the '# pipeline-type' comment."
                ERRORS_FOUND=1
            fi
        done <<< "$UNIQUE_CATEGORIES"

        # B2. CI_CATEGORY directory-specific guardrails
        if [[ "$file" =~ ^\.buildkite/models/ ]]; then
            if [[ ! "$C_CAT" =~ ^(text-only|multimodal)$ ]]; then
                echo "+++ ❌ Error: Invalid CI_CATEGORY '$C_CAT' for models/ in $file"
                echo "💡 Tip: Files in 'models/' should use 'text-only' or 'multimodal' as their CI_CATEGORY."
                ERRORS_FOUND=1
            fi
        elif [[ "$file" =~ ^\.buildkite/features/ ]]; then
            FEATURE_RE='^(feature|kernel) support matrix$'
            if [[ ! "$C_CAT" =~ $FEATURE_RE ]]; then
                echo "+++ ❌ Error: Invalid CI_CATEGORY '$C_CAT' for features/ in $file"
                echo "💡 Tip: Files in 'features/' should use 'feature support matrix' or 'kernel support matrix' as their CI_CATEGORY."
                ERRORS_FOUND=1
            fi
        elif [[ "$file" =~ ^\.buildkite/parallelism/ ]]; then
            if [[ "$C_CAT" != "parallelism support matrix" ]]; then
                echo "+++ ❌ Error: Invalid CI_CATEGORY '$C_CAT' for parallelism/ in $file"
                echo "💡 Tip: Files in 'parallelism/' must use 'parallelism support matrix' as their CI_CATEGORY."
                ERRORS_FOUND=1
            fi
        elif [[ "$file" =~ ^\.buildkite/quantization/ ]]; then
            if [[ "$C_CAT" != "quantization support matrix" ]]; then
                echo "+++ ❌ Error: Invalid CI_CATEGORY '$C_CAT' for quantization/ in $file"
                echo "💡 Tip: Files in 'quantization/' must use 'quantization support matrix' as their CI_CATEGORY."
                ERRORS_FOUND=1
            fi
        elif [[ "$file" =~ ^\.buildkite/rl/ ]]; then
            if [[ "$C_CAT" != "rl support matrix" ]]; then
                echo "+++ ❌ Error: Invalid CI_CATEGORY '$C_CAT' for rl/ in $file"
                echo "💡 Tip: Files in 'rl/' must use 'rl support matrix' as their CI_CATEGORY."
                ERRORS_FOUND=1
            fi
        elif [[ "$file" =~ ^\.buildkite/kernel_microbenchmarks/ ]]; then
            if [[ "$C_CAT" != "kernel support matrix microbenchmarks" ]]; then
                echo "+++ ❌ Error: Invalid CI_CATEGORY '$C_CAT' for kernel_microbenchmarks/ in $file"
                echo "💡 Tip: Files in 'kernel_microbenchmarks/' must use 'kernel support matrix microbenchmarks' as their CI_CATEGORY."
                ERRORS_FOUND=1
            fi
        fi

        # B3. CI_STAGE whitelist check
        ALL_FILE_STAGES=$(yq '.steps[].env.CI_STAGE | select(. != null)' "$file" | sort -u)
        while read -r stage; do
            [ -z "$stage" ] && continue
            if ! contains "${ALLOWED_STAGES[@]}" "$stage"; then
                echo "+++ ❌ Error: Invalid CI_STAGE '$stage' in $file"
                echo "💡 Tip: Use only approved stage names: ${ALLOWED_STAGES[*]}."
                ERRORS_FOUND=1
            fi
        done < <(echo "$ALL_FILE_STAGES")

        # C. Naming Conventions & Key Formatting
        # C1. Keys must start with ${TPU_VERSION:-...}
        INVALID_KEYS=$(yq '.steps[] | select(.key != null and (.key | test("^\$\{TPU_VERSION:-") | not)) | .key' "$file")
        if [[ -n "$INVALID_KEYS" ]]; then
            echo "+++ ❌ Error: Invalid key format in $file"
            echo "Found keys: $INVALID_KEYS"
            echo "💡 Tip: All step keys must start with '\${TPU_VERSION:-...}' to differetiate the TPU version of the step in this build"
            ERRORS_FOUND=1
        fi

        # C2. Labels must start with ${TPU_VERSION:-...}
        INVALID_LABELS=$(yq '.steps[] | select(.label != null and (.label | test("^\$\{TPU_VERSION:-") | not)) | .label' "$file")
        if [[ -n "$INVALID_LABELS" ]]; then
            echo "+++ ❌ Error: Invalid label format in $file"
            echo "Found labels: $INVALID_LABELS"
            echo "💡 Tip: All step labels must start with '\${TPU_VERSION:-...}' to differetiate the TPU version of the step in this build"
            ERRORS_FOUND=1
        fi

        # D. Step Consistency
        # D1. record_step_result.sh consistency
        RECORD_STEPS_JSON=$(yq '[.steps[] | select(.commands != null) | select(.commands[] | test("record_step_result.sh"))]' -o json "$file" || echo "[]")

        if [[ "$RECORD_STEPS_JSON" != "[]" ]]; then
            echo "$RECORD_STEPS_JSON" | jq -c '.[]' | while read -r step; do
                S_KEY=$(echo "$step" | jq -r '.key')
                S_DEP=$(echo "$step" | jq -r '.depends_on')
                S_ARG=$(echo "$step" | jq -r '.commands[] | select(test("record_step_result.sh"))' | sed -E 's/.*record_step_result.sh[[:space:]]+([^[:space:]]+).*/\1/')
                
                # 1. Dependency Integrity
                if [[ "$S_DEP" == "null" ]]; then
                    echo "+++ ❌ Error: Recording step '$S_KEY' in $file has no 'depends_on' field."
                    echo "💡 Tip: Every recording step must depend on the test step it is recording."
                    ERRORS_FOUND=1
                elif [[ "$S_ARG" != "$S_DEP" ]]; then
                    echo "+++ ❌ Error: record_step_result.sh argument mismatch in $file"
                    echo "Step '$S_KEY' depends on '$S_DEP' but tries to record '$S_ARG'."
                    echo "💡 Tip: The argument passed to '.buildkite/scripts/record_step_result.sh' must be EXACTLY the same as the 'depends_on' key."
                    ERRORS_FOUND=1
                fi

                # 2. Mandatory Metadata Integrity
                for meta_field in "CI_TPU_VERSION" "CI_TARGET" "CI_STAGE" "CI_CATEGORY"; do
                    meta_val=$(echo "$step" | jq -r ".env.$meta_field")
                    if [[ "$meta_val" == "null" ]]; then
                        echo "+++ ❌ Error: Recording step '$S_KEY' in $file is missing mandatory metadata 'env.$meta_field'."
                        echo "💡 Tip: All recording steps must contain full metadata to ensure results are correctly tracked in the support matrices."
                        ERRORS_FOUND=1
                    fi
                done
            done
        fi

        # E. Resource & Agent Validation
        # E1. Require agents.queue for all steps
        STEPS_MISSING_QUEUE=$(yq '.steps[] | select(.wait == null and . != "wait" and .agents.queue == null) | .key // .label // "unknown_step"' "$file")
        if [[ -n "$STEPS_MISSING_QUEUE" ]]; then
            echo "+++ ❌ Error: Missing agents.queue in $file"
            echo "Problematic steps:"
            echo "$STEPS_MISSING_QUEUE"
            echo "💡 Tip: Every execution step must explicitly define an 'agents.queue'."
            ERRORS_FOUND=1
        fi

        # E2. Queue name and format validity
        # shellcheck disable=SC2016
        INVALID_QUEUES=""
        while IFS=$'\t' read -r step_key step_queue; do
            [ -z "$step_queue" ] && continue
            
            is_valid=0
            for allowed in "${ALLOWED_SINGLE_QUEUES[@]}" "${ALLOWED_MULTI_QUEUES[@]}"; do
                if [[ "$step_queue" == "$allowed" ]]; then
                    is_valid=1
                    break
                fi
            done
            
            if [[ $is_valid -eq 0 ]]; then
                INVALID_QUEUES+="${step_key}: ${step_queue}"$'\n'
            fi
        done < <(yq '.steps[] | select(.agents.queue != null) | .key + "\t" + .agents.queue' -r "$file" || true)

        if [[ -n "$INVALID_QUEUES" ]]; then
            echo "+++ ❌ Error: Unsupported or incorrectly formatted agents.queue value in $file"
            echo "Found:"
            echo -e "$INVALID_QUEUES"
            echo "💡 Tip: Use standard queue names: 'cpu', EXACTLY '\${TPU_QUEUE_SINGLE:-tpu_v6e_queue}', EXACTLY '\${TPU_QUEUE_MULTI:-tpu_v6e_8_queue}', or a specific hardware queue like 'tpu_v6e_queue'."
            ERRORS_FOUND=1
        fi

        # E3. Multi-host queue matching for parallelism specs
        if [[ "$file" =~ ^\.buildkite/parallelism/ ]]; then
            MULTI_HOST_QUEUE_ERRORS=""
            while IFS=$'\t' read -r step_key step_queue; do
                [ -z "$step_queue" ] && continue
                
                # Allow cpu queue for record steps even if they contain Multi_Host in the key
                if [[ "$step_queue" == "cpu" ]]; then
                    continue
                fi

                is_multi_valid=0
                for allowed in "${ALLOWED_MULTI_QUEUES[@]}"; do
                    if [[ "$step_queue" == "$allowed" ]]; then
                        is_multi_valid=1
                        break
                    fi
                done
                
                if [[ $is_multi_valid -eq 0 ]]; then
                    MULTI_HOST_QUEUE_ERRORS+="${step_key} (Must use multi-host queue, found: ${step_queue})"$'\n'
                fi
            done < <(yq '.steps[] | select(.key | test("Multi_Host")) | select(.agents.queue != null) | .key + "\t" + .agents.queue' -r "$file" || true)

            if [[ -n "$MULTI_HOST_QUEUE_ERRORS" ]]; then
                echo "+++ ❌ Error: Multi-host queue mismatch in $file"
                echo "Problematic keys:"
                echo -e "$MULTI_HOST_QUEUE_ERRORS"
                echo "💡 Tip: Steps involving 'Multi_Host' must use a multi-device queue (e.g., '\${TPU_QUEUE_MULTI:-tpu_v6e_8_queue}' or ending in '_8_queue')."
                ERRORS_FOUND=1
            fi
        fi

    else
        echo "⏭️  Skipping extended check for non-spec file: $file"
    fi

done < <(echo "$YAML_FILES_TO_CHECK")

if [[ "$ERRORS_FOUND" -eq 1 ]]; then
    echo "--- ❌ Comprehensive metadata and logic verification failed."
    exit 1
else
    echo "--- ✅ Comprehensive metadata and logic verification passed."
    exit 0
fi
