#!/bin/bash
# Copyright 2025 Google LLC
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

set -euo pipefail

ANY_FAILED=false

MODEL_LIST_KEY="model-list"
FEATURE_LIST_KEY="feature-list"
DEFAULT_FEATURES_FILE=".buildkite/features/default_features.txt"

# Note: This script assumes the metadata keys contain newline-separated lists.
mapfile -t model_list < <(buildkite-agent meta-data get "${MODEL_LIST_KEY}" --default "")
mapfile -t metadata_feature_list < <(buildkite-agent meta-data get "${FEATURE_LIST_KEY}" --default "")
MODEL_STAGES=("UnitTest" "IntegrationTest" "Benchmark")
FEATURE_STAGES=("CorrectnessTest" "PerformanceTest")
FEATURE_STAGES_QUANTIZATION=("RecommendedTPUGenerations" "CorrectnessTest" "PerformanceTest")

declare -A TPU_GENERATIONS=(
    ["INT8 W8A8"]="\"v5, v6\""
    ["INT4 W4A16"]="\"v5, v6\""
    ["FP8 W8A8"]="v7"
    ["FP8 W8A16"]="v7"
    ["FP4 W4A16"]="v7"
    ["AWQ INT4"]="\"v5, v6\""
)
declare -a model_csv_files=()
declare -a feature_csv_files=()
declare -a default_feature_names=()

# Parse Default Features File & Set Categories
if [[ -f "${DEFAULT_FEATURES_FILE}" ]]; then
    mapfile -t raw_default_lines < <(sed 's/\r$//; /^$/d' "${DEFAULT_FEATURES_FILE}")
    # Regex to capture "Feature Name (Category Name)"
    REGEX='^(.+) \((.+)\)$'

    echo "--- Loading Feature Categories from file ---"
    for line in "${raw_default_lines[@]}"; do
        if [[ $line =~ $REGEX ]]; then
            feature_name="${BASH_REMATCH[1]}"
            category="${BASH_REMATCH[2]}"
            default_feature_names+=("$feature_name")
            # Set metadata so we know which CSV to put it in later
            echo "Setting category for '$feature_name': $category"
            buildkite-agent meta-data set "${feature_name}_category" "$category"
        else
            # Fallback if no category found
            default_feature_names+=("$line")
            echo "Warning: No category found for '$line', defaulting to 'feature support matrix'"
        fi
    done
else
    echo "Warning: Default features file not found at ${DEFAULT_FEATURES_FILE}"
fi

# Process Models (Split by Category)
process_models() {
    for model in "${model_list[@]:-}"; do
        if [[ -z "$model" ]]; then continue; fi
        # Get category (default: text-only)
        local category
        category=$(buildkite-agent meta-data get "${model}_category" --default "text-only")
        # Define the category-specific CSV filename
        local category_filename
        if [ "$category" == "text-only" ]; then
            category_filename="text_only_model"
        elif [ "$category" == "multimodal" ]; then
            category_filename="multimodal_model"
        else
            category_filename=${category// /_}
        fi
        local category_csv="${category_filename}_support_matrix.csv"
        # Initialize CSV if not exists
        if [ ! -f "$category_csv" ]; then
            echo "Model,UnitTest,IntegrationTest,Benchmark" > "$category_csv"
            model_csv_files+=("$category_csv")
        fi
        # Build Row
        local row="\"$model\""
        for stage in "${MODEL_STAGES[@]}"; do
            local result
            result=$(buildkite-agent meta-data get "${model}:${stage}" --default "N/A")
            row="$row,$result"
            if [ "${result}" != "✅" ] && [ "${result}" != "N/A" ] && [ "${result}" != "unverified" ]; then
                ANY_FAILED=true
            fi
        done
        echo "$row" >> "$category_csv"
    done
}

# Process Features (Split by Category)
process_features() {
    local mode="$1"
    shift # Shift $1 so $@ now contains only the feature list

    for feature in "$@"; do
        if [[ -z "$feature" ]]; then continue; fi

        # Get Category (default: feature support matrix)
        local category
        category=$(buildkite-agent meta-data get "${feature}_category" --default "feature support matrix")

        local category_filename=${category// /_}
        local category_csv="${category_filename}.csv"

        # Determine which stages array and header to use
        local stages_to_use=("${FEATURE_STAGES[@]}")
        local header="Feature,CorrectnessTest,PerformanceTest"
        local is_quantization_matrix=false

        if [ "$category" == "quantization support matrix" ]; then
            is_quantization_matrix=true
            stages_to_use=("${FEATURE_STAGES_QUANTIZATION[@]}")
            header="Feature,Recommended TPU Generations,CorrectnessTest,PerformanceTest"
        fi

        if [ ! -f "$category_csv" ]; then
            echo "$header" > "$category_csv"
            feature_csv_files+=("$category_csv")
        fi

        # Build Row
        local row="\"$feature\""
        local stage_index=0
        for stage in "${stages_to_use[@]}"; do
            local result

            if [ "$is_quantization_matrix" = true ] && [ "$stage" == "RecommendedTPUGenerations" ]; then
                # If it's the quantization matrix, hardcode the TPU generation
                result="${TPU_GENERATIONS["$feature"]:-N/A}"

            elif [[ "$mode" == "DEFAULT" ]]; then
                result="✅"
            else
                result=$(buildkite-agent meta-data get "${feature}:${stage}" --default "N/A")
            fi

            row="$row,$result"

            # Check for failure (exclude the hardcoded TPU generation column)
            if [ "$stage" != "RecommendedTPUGenerations" ] && [ "${result}" != "✅" ] && [ "${result}" != "N/A" ] && [ "${result}" != "unverified" ]; then
                ANY_FAILED=true
            fi

            stage_index=$((stage_index + 1))
        done
        echo "$row" >> "$category_csv"
    done
}

if [ ${#model_list[@]} -gt 0 ]; then
    process_models
fi

if [ ${#default_feature_names[@]} -gt 0 ]; then
    process_features "DEFAULT" "${default_feature_names[@]}"
fi

if [ ${#metadata_feature_list[@]} -gt 0 ]; then
    process_features "METADATA" "${metadata_feature_list[@]}"
fi

buildkite-agent meta-data set "CI_TESTS_FAILED" "${ANY_FAILED}"

# Model support matrices
for csv_file in "${model_csv_files[@]}"; do
    if [[ -f "$csv_file" ]]; then
        echo "--- $csv_file ---"
        cat "$csv_file"
        buildkite-agent artifact upload "$csv_file"
    fi
done

# Feature support matrices
for csv_file in "${feature_csv_files[@]}"; do
    if [[ -f "$csv_file" ]]; then
        echo "--- $csv_file ---"
        sorted_content=$(tail -n +2 "$csv_file" | sort -V)
        header=$(head -n 1 "$csv_file")
        echo "$header" > "$csv_file"
        echo "$sorted_content" >> "$csv_file"

        cat "$csv_file"
        buildkite-agent artifact upload "$csv_file"
    fi
done

echo "Reports uploaded successfully."

# Cleanup
rm -f "${model_csv_files[@]}" "${feature_csv_files[@]}"
