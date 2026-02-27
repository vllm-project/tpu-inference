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
MODEL_STAGES=("Type" "UnitTest" "Accuracy/Correctness" "Benchmark")
FEATURE_STAGES=("CorrectnessTest" "PerformanceTest")
FEATURE_STAGES_QUANTIZATION=("QuantizationMethods" "RecommendedTPUGenerations" "CorrectnessTest" "PerformanceTest")
FEATURE_STAGES_MICROBENCHMARKS=("CorrectnessTest" "PerformanceTest")

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

# Determine sub-directory based on TPU_VERSION
if [[ "${TPU_VERSION:-tpu6e}" == "v7"* ]]; then
    TPU_DIR="v7"
    TPU_METADATA_PREFIX="v7"
else
    TPU_DIR="v6"
    TPU_METADATA_PREFIX="v6"
fi

mkdir -p "${TPU_DIR}"
echo "Output directory set to: ${TPU_DIR} (Prefix: '${TPU_METADATA_PREFIX}')"

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
            buildkite-agent meta-data set "${TPU_METADATA_PREFIX}${feature_name}_category" "$category"
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
        category=$(buildkite-agent meta-data get "${TPU_METADATA_PREFIX}${model}_category" --default "text-only")
        # Use the TPU_DIR prefix for the CSV path
        local category_csv="${TPU_DIR}/model_support_matrix.csv"
        # Initialize CSV if not exists
        if [ ! -f "$category_csv" ]; then
            echo "Model,Type,UnitTest,Accuracy/Correctness,Benchmark" > "$category_csv"
            model_csv_files+=("$category_csv")
        fi
        # Build Row
        local row="\"$model\""
        for stage in "${MODEL_STAGES[@]}"; do
            local result
            if [ "$stage" == "Type" ]; then
                if [ "$category" == "multimodal" ]; then
                    result="Multimodal"
                else
                    result="Text"
                fi
            else
                result=$(buildkite-agent meta-data get "${TPU_METADATA_PREFIX}${model}:${stage}" --default "N/A")
            fi
            row="$row,$result"
            if [ "$stage" != "Type" ] && [ "${result}" != "✅" ] && [ "${result}" != "N/A" ] && [ "${result}" != "unverified" ]; then
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
        category=$(buildkite-agent meta-data get "${TPU_METADATA_PREFIX}${feature}_category" --default "feature support matrix")

        local category_filename=${category// /_}
        # Use the TPU_DIR prefix for the CSV path
        local category_csv="${TPU_DIR}/${category_filename}.csv"

        # Determine which stages array and header to use
        local stages_to_use=("${FEATURE_STAGES[@]}")
        local header="Feature,CorrectnessTest,PerformanceTest"
        local is_quantization_matrix=false

        if [ "$category" == "quantization support matrix" ]; then
            is_quantization_matrix=true
            stages_to_use=("${FEATURE_STAGES_QUANTIZATION[@]}")
            header="Quantization dtype,Quantization methods,Recommended TPU Generations,CorrectnessTest,PerformanceTest"
        elif [ "$category" == "kernel support matrix microbenchmarks" ]; then
            stages_to_use=("${FEATURE_STAGES_MICROBENCHMARKS[@]}")
            header="kernels,CorrectnessTest,PerformanceTest"
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
            elif [[ "$mode" == "DEFAULT" ]]; then
                result="✅"
            else
                result=$(buildkite-agent meta-data get "${TPU_METADATA_PREFIX}${feature}:${stage}" --default "N/A")
            fi

            row="$row,$result"

            # Check for failure (exclude the hardcoded TPU generation column and Quantization Methods column)
            if [ "$stage" != "QuantizationMethods" ] && [ "$stage" != "RecommendedTPUGenerations" ] && [ "${result}" != "✅" ] && [ "${result}" != "N/A" ] && [ "${result}" != "unverified" ]; then
                ANY_FAILED=true
            fi

            stage_index=$((stage_index + 1))
        done
        echo "$row" >> "$category_csv"
    done
}

process_kernel_matrix_to_pivot() {
    local input_csv="${TPU_DIR}/kernel_support_matrix_microbenchmarks.csv"
    local output_file="${TPU_DIR}/kernel_support_matrix-microbenchmarks.csv"

    if [ ! -f "$input_csv" ]; then
        echo "Warning: Input CSV $input_csv not found. Skipping pivot."
        return
    fi

    # Define Headers for Display
    local header="Kernel,W16 A16 (Correctness),W16 A16 (Performance),W8 A8 (Correctness),W8 A8 (Performance),W8 A16 (Correctness),W8 A16 (Performance),W4 A4 (Correctness),W4 A4 (Performance),W4 A8 (Correctness),W4 A8 (Performance),W4 A16 (Correctness),W4 A16 (Performance)"
    echo "$header" > "$output_file"

    # Define the quantization order to match the header
    local quant_cols_list="w16a16 w8a8 w8a16 w4a4 w4a8 w4a16"

    # Awk Script for Pivoting (Data Rows)
    awk -v AWK_QUANT_COLS="$quant_cols_list" '
        BEGIN { 
            FS=","; OFS=",";
            split(AWK_QUANT_COLS, q_order, " ");
        }
        
        NR > 1 {
            gsub(/"/, "", $1);
            if (match($1, /-(w[0-9]+a[0-9]+)$/)) {
                quant_type = substr($1, RSTART + 1, RLENGTH - 1);
                base_kernel_key = substr($1, 1, RSTART - 1);
            } else {
                 base_kernel_key = $1;
                 quant_type = "w16a16";
            }

            # Store original Correctness ($2) and Performance ($3)
            matrix[base_kernel_key][quant_type] = $2 OFS $3;

            if (! (base_kernel_key in kernels)) {
                kernels[base_kernel_key] = 1;
                kernel_list[num_kernels++] = base_kernel_key;
            }
        }
        END {
            for (i=0; i<num_kernels; i++) {
                k = kernel_list[i];
                out_name = k;
                if (out_name == "generic ragged paged attention v3") {
                    out_name = "\"generic ragged paged<br>attention v3*\"";
                } else if (out_name == "mla") {
                    out_name = "\"mla*\"";
                } else if (out_name == "ragged paged attention v3 head_dim 64") {
                    out_name = "\"ragged paged attention v3<br>head_dim 64*\"";
                } else {
                    out_name = "\"" out_name "\"";
                }

                row = out_name;
                for (j=1; j<=6; j++) {
                    q = q_order[j];
                    # Use original N/A,N/A if data is missing
                    data = (matrix[k][q] == "") ? "N/A,N/A" : matrix[k][q];
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
for csv_file in "${model_csv_files[@]:-}"; do
    if [[ -n "$csv_file" && -f "$csv_file" ]]; then
        header=$(head -n 1 "$csv_file")
        # Sort data rows based on the 'Type' column
        sorted_content=$(tail -n +2 "$csv_file" | sort -t',' -k2,2)

        # Reconstruct the file with sorted data
        echo "$header" > "$csv_file"
        echo "$sorted_content" >> "$csv_file"

        echo "--- Uploading Model Matrix: $csv_file ---"
        cat "$csv_file"
        buildkite-agent artifact upload "$csv_file"
    fi
done

# Feature support matrices
for csv_file in "${feature_csv_files[@]:-}"; do
    if [[ -n "$csv_file" && -f "$csv_file" ]]; then
        header=$(head -n 1 "$csv_file")
        sorted_content=$(tail -n +2 "$csv_file" | sort -V)
        echo "$header" > "$csv_file"
        echo "$sorted_content" >> "$csv_file"
        
        # Skip raw kernel microbenchmarks; upload only the pivoted version
        if [[ "$csv_file" != *"kernel_support_matrix_microbenchmarks.csv" ]]; then
            echo "--- Uploading Feature Matrix: $csv_file ---"
            cat "$csv_file"
            buildkite-agent artifact upload "$csv_file"
        else
            echo "Skipping direct upload for $csv_file (will be pivoted later)."
        fi
    fi
done

# Process the Kernel Matrix into the pivoted format
process_kernel_matrix_to_pivot

echo "Reports uploaded successfully."

# Cleanup
rm -rf "${TPU_DIR}"
