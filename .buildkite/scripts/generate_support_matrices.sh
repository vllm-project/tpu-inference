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
LOCAL_TPU_VERSION="${BUILDKITE_TAG:-nightly_$(date +%Y%m%d)}"

# Note: This script assumes the metadata keys contain newline-separated lists.
mapfile -t model_list < <(buildkite-agent meta-data get "${MODEL_LIST_KEY}" --default "")
mapfile -t metadata_feature_list < <(buildkite-agent meta-data get "${FEATURE_LIST_KEY}" --default "")
MODEL_STAGES=("UnitTest" "IntegrationTest" "Benchmark")
FEATURE_STAGES=("CorrectnessTest" "PerformanceTest")
FEATURE_STAGES_QUANTIZATION=("QuantizationMethods" "RecommendedTPUGenerations" "CorrectnessTest" "PerformanceTest")
FEATURE_STAGES_MICROBENCHMARKS=("CorrectnessTest" "PerformanceTest" "TPU Versions")

declare -A TPU_GENERATIONS=(
    ["INT8 W8A8"]="\"v5, v6\""
    ["INT4 W4A16"]="\"v5, v6\""
    ["FP8 W8A8"]="v7"
    ["FP8 W8A16"]="v7"
    ["FP4 W4A16"]="v7"
    ["AWQ INT4"]="\"v5, v6\""
)
declare -A QUANTIZATION_METHODS=(
    ["INT8 W8A8"]="compressed-tensor"
    ["INT4 W4A16"]="awq"
    ["FP8 W8A8"]="compressed-tensor"
    ["FP8 W8A16"]="compressed-tensor"
    ["FP4 W4A16"]="mxfp4"
    ["AWQ INT4"]=""
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
        local is_kernel_microbenchmarks=false

        if [ "$category" == "quantization support matrix" ]; then
            is_quantization_matrix=true
            stages_to_use=("${FEATURE_STAGES_QUANTIZATION[@]}")
            header="Quantization dtype,Quantization methods,Recommended TPU Generations,CorrectnessTest,PerformanceTest"
        elif [ "$category" == "kernel support matrix microbenchmarks" ]; then
            is_kernel_microbenchmarks=true
            stages_to_use=("${FEATURE_STAGES_MICROBENCHMARKS[@]}")
            header="kernels,CorrectnessTest,PerformanceTest,TPU Versions"
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
            elif [ "$is_quantization_matrix" = true ] && [ "$stage" == "QuantizationMethods" ]; then
                # If it's the quantization matrix, hardcode the quantization methods
                result="${QUANTIZATION_METHODS["$feature"]:-N/A}"
            elif [ "$is_kernel_microbenchmarks" = true ] && [ "$stage" == "TPU Versions" ]; then
                # If it's kernel microbenchmarks matrix, hardcode the tpu version
                result="${LOCAL_TPU_VERSION}"
            elif [[ "$mode" == "DEFAULT" ]]; then
                result="✅"
            else
                result=$(buildkite-agent meta-data get "${feature}:${stage}" --default "N/A")
            fi

            row="$row,$result"

            # Check for failure (exclude the hardcoded TPU generation column and Quantization Methods column)
            if [ "$stage" != "TPU Versions" ] && [ "$stage" != "QuantizationMethods" ] && [ "$stage" != "RecommendedTPUGenerations" ] && [ "${result}" != "✅" ] && [ "${result}" != "N/A" ] && [ "${result}" != "unverified" ]; then
                ANY_FAILED=true
            fi

            stage_index=$((stage_index + 1))
        done
        echo "$row" >> "$category_csv"
    done
}

process_kernel_matrix_to_pivot() {
    local input_csv="kernel_support_matrix_microbenchmarks.csv"
    local output_file="kernel_support_matrix-microbenchmarks.csv"

    if [ ! -f "$input_csv" ]; then
        echo "Warning: Input CSV $input_csv not found. Skipping pivot."
        return
    fi

    # Define Headers for Display
    local quant_cols_list="w16a16 w8a8 w8a16 w4a4 w4a8 w4a16"
    local AWK_QUANT_COLS
    AWK_QUANT_COLS=$(IFS=" "; echo "${quant_cols_list[*]}")

    # Line 1: ,w16a16,,,w8a8,,,w8a16,,,w4a4,,,w4a8,,,w4a16,,
    local header_line1=","
    for quant_type in $quant_cols_list; do
        header_line1="${header_line1}${quant_type},,,"
    done
    # Remove the trailing comma from the last group
    header_line1="${header_line1%,}"

    # Line 2: kernels,correctness,performance,tpu versions,correctness,performance,tpu versions,...
    local header_line2="kernels"
    for _ in $quant_cols_list; do
        header_line2="${header_line2},correctness,performance,tpu versions"
    done

    # Write the two-line header structure to the output file
    echo "$header_line1" > "$output_file"
    echo "$header_line2" >> "$output_file"

    # Awk Script for Pivoting (Data Rows)
   awk -v AWK_QUANT_COLS="$AWK_QUANT_COLS" '
        BEGIN { FS=","; OFS="," }
        NR > 1 {
            # Kernel parsing logic remains the same
            gsub(/"/, "", $1);

            if (match($1, /-(w[0-9]+a[0-9]+)$/)) {
                quant_type = substr($1, RSTART + 1, RLENGTH - 1);
                base_kernel_key = substr($1, 1, RSTART - 1);
            } else {
                 base_kernel_key = $1;
                 quant_type = "w16a16";
            }

            matrix[base_kernel_key][quant_type] = $2 OFS $3 OFS $4;

            if (! (base_kernel_key in kernels)) {
                kernels[base_kernel_key] = 1;
                kernel_list[num_kernels++] = base_kernel_key;
            }
        }
        END {
            split(AWK_QUANT_COLS, quant_cols, " ");
            default_val = "N/A" OFS "N/A" OFS "N/A";

            # Iterate through all unique base kernels found in the input (ordered by kernel_list)
            for (i=0; i<num_kernels; i++) {
                local_original_key = kernel_list[i];
                kernel_for_output = local_original_key;

                # Apply desired renames/modifications to the output name
                if (kernel_for_output == "generic ragged paged attention v3") {
                    kernel_for_output = "generic ragged paged attention v3*";
                } else if (kernel_for_output == "mla") {
                    kernel_for_output = "mla*";
                } else if (kernel_for_output == "attention_kernels") {
                    kernel_for_output = "* For attention kernels, W[x]A[y] denotes KV cache as W, A as compute, and x, y as bit precision";
                } else if (kernel_for_output == "ragged paged attention v3 head_dim 64") {
                    kernel_for_output = "ragged paged attention v3 head_dim 64*";
                }

                row = "\"" kernel_for_output "\"";

                # Append data for all columns
                for (j in quant_cols) {
                    quant = quant_cols[j];
                    data = (matrix[local_original_key][quant] == "") ? default_val : matrix[local_original_key][quant];
                    row = row OFS data;
                }
                print row >> "'"$output_file"'";
            }
        }
    ' "$input_csv"

    # Display the pivoted result
    echo "--- $output_file ---"
    cat "$output_file"
    buildkite-agent artifact upload "$output_file"
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
        sorted_content=$(tail -n +2 "$csv_file" | sort -V)
        header=$(head -n 1 "$csv_file")
        echo "$header" > "$csv_file"
        echo "$sorted_content" >> "$csv_file"
        if [[ "$csv_file" != "kernel_support_matrix_microbenchmarks.csv" ]]; then
            echo "--- $csv_file ---"
            cat "$csv_file"
            buildkite-agent artifact upload "$csv_file"
        fi
    fi
done

# Process the Kernel Matrix into the pivoted format
process_kernel_matrix_to_pivot

echo "Reports uploaded successfully."

# Cleanup
rm -f "${model_csv_files[@]}" "${feature_csv_files[@]}"
