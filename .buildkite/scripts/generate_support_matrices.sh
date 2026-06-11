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

RUN_V6=$(buildkite-agent meta-data get run_v6_matrix --default 'false')
RUN_V7=$(buildkite-agent meta-data get run_v7_matrix --default 'false')

if [[ "$RUN_V6" != "true" && "$RUN_V7" != "true" ]]; then
    echo "--- Skipping matrix generation (No v6 or v7 test data found)"
    exit 0
fi

MODEL_LIST_KEY="model-list"
FEATURE_LIST_KEY="feature-list"
DEFAULT_FEATURES_FILE=".buildkite/features/default_features.txt"
FRAMEWORKS=("vllm" "flax_nnx")

# Note: This script assumes the metadata keys contain newline-separated lists.
mapfile -t model_list < <(buildkite-agent meta-data get "${MODEL_LIST_KEY}" --default "")
mapfile -t metadata_feature_list < <(buildkite-agent meta-data get "${FEATURE_LIST_KEY}" --default "")
MODEL_STAGES=("Type" "Machine Type" "Framework" "UnitTest" "Accuracy/Correctness" "Benchmark")
FEATURE_STAGES=("Machine Type" "Framework" "CorrectnessTest" "PerformanceTest")
FEATURE_STAGES_QUANTIZATION=("Machine Type" "Framework" "QuantizationMethods" "RecommendedTPUGenerations" "CorrectnessTest" "PerformanceTest")
FEATURE_STAGES_MICROBENCHMARKS=("Machine Type" "Framework" "CorrectnessTest" "PerformanceTest")
PARALLELISM_STAGES=("Machine Type" "Framework" "Single-Host CorrectnessTest" "Single-Host PerformanceTest" "Multi-Host CorrectnessTest" "Multi-Host PerformanceTest")

get_tpu_generation() {
    local key="$1"
    case "$key" in
        "INT8 W8A8") echo "\"v5, v6\"" ;;
        "INT4 W4A16") echo "\"v5, v6\"" ;;
        "FP8 W8A8") echo "v7" ;;
        "FP8 W8A16") echo "v7" ;;
        "FP4 W4A16") echo "v7" ;;
        "NVFP4 W4A16") echo "v7" ;;
        *) echo "N/A" ;;
    esac
}

get_quantization_method() {
    local key="$1"
    case "$key" in
        "INT8 W8A8") echo "compressed-tensor" ;;
        "INT4 W4A16") echo "awq" ;;
        "FP8 W8A8") echo "compressed-tensor" ;;
        "FP8 W8A16") echo "compressed-tensor" ;;
        "FP4 W4A16") echo "mxfp4" ;;
        "NVFP4 W4A16") echo "modelopt_fp4" ;;
        *) echo "N/A" ;;
    esac
}

# Helper function to sanitize strings for metadata keys
sanitize_name() {
    echo "$1" | tr '/:.[[:space:]]' '_' | sed 's/^_//;s/_$//;s/__\+/_/g'
}

declare -a model_csv_files=()
declare -a feature_csv_files=()
declare -a default_feature_names=()
declare -a ACTIVE_TPU_CONFIGS=()

if [[ "$RUN_V6" == "true" ]]; then
    ACTIVE_TPU_CONFIGS+=("tpu6e")
fi

if [[ "$RUN_V7" == "true" ]]; then
    ACTIVE_TPU_CONFIGS+=("tpu7x")
fi

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
            
            # Sanitize the feature name for safe use in CI metadata keys
            safe_feature=$(sanitize_name "$feature_name")
            
            # Register the category against all active hardware targets
            for ci_tpu_version in "${ACTIVE_TPU_CONFIGS[@]}"; do
                category_key="${ci_tpu_version}_${safe_feature}_category"
                echo "Setting category for '$feature_name' ($ci_tpu_version): $category"
                buildkite-agent meta-data set "$category_key" "$category"
            done
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
    local category_csv="model_support_matrix.csv"
    if [ ! -f "$category_csv" ]; then
        echo "Model,Type,Machine Type,Framework,UnitTest,Accuracy/Correctness,Benchmark" > "$category_csv"
        model_csv_files+=("$category_csv")
    fi

    # Return directly if the model list is empty
    if [ ${#model_list[@]} -eq 0 ]; then return; fi

    # Iterate through all permutations of frameworks, TPU configs, and models
    for framework in "${FRAMEWORKS[@]}"; do
        for ci_tpu_version in "${ACTIVE_TPU_CONFIGS[@]}"; do
            for model in "${model_list[@]:-}"; do
                if [[ -z "$model" ]]; then continue; fi

                # Sanitize the model name for Buildkite metadata keys (e.g., replaces / : . with _)
                local safe_model
                safe_model=$(sanitize_name "$model")

                # Get the category (default: text-only)
                local category_key="${ci_tpu_version}_${safe_model}_category"
                local category
                category=$(buildkite-agent meta-data get "${category_key}" --default "text-only")

                local row="\"$model\""
                for stage in "${MODEL_STAGES[@]}"; do
                    local result
                    if [ "$stage" == "Type" ]; then
                        if [ "$category" == "multimodal" ]; then result="Multimodal"
                        elif [ "$category" == "embedding" ]; then result="Embedding"
                        elif [ "$category" == "diffusion" ]; then result="Diffusion"
                        else result="Text"
                        fi
                    elif [ "$stage" == "Machine Type" ]; then
                        if [ "${ci_tpu_version}" == "tpu7x" ]; then
                            result="v7x"
                        else
                            result="v6e"
                        fi
                    elif [ "$stage" == "Framework" ]; then
                        result="${framework}"
                    else
                        local safe_model_name
                        safe_model_name=$(sanitize_name "$model")
                        local safe_stage_name
                        safe_stage_name=$(sanitize_name "$stage")
                        local meta_key="${ci_tpu_version}_${framework}_${safe_model_name}_${safe_stage_name}"
                        result=$(buildkite-agent meta-data get "${meta_key}" --default "❓ Untested")
                    fi
                    
                    row="$row,$result"

                    # Record if there are any failed tests (excluding Type, Machine Type, and Framework fields)
                    if [ "$stage" != "Type" ] && [ "$stage" != "Machine Type" ] && [ "$stage" != "Framework" ] && \
                       [ "${result}" != "✅ Passing" ] && [ "${result}" != "⚪ N/A" ] && \
                       [ "${result}" != "❓ Untested" ] && [ "${result}" != "not enough HBM" ] && [ "${result}" != "transformers version too low" ]; then
                        ANY_FAILED=true
                    fi
                done
                
                # Write the assembled row data into the CSV
                echo "$row" >> "$category_csv"
            done
        done
    done
}

# Process Features (Split by Category)
process_features() {
    local mode="$1"
    shift # Shift $1 so $@ now contains only the feature list

    # Iterate through all permutations of frameworks, TPU environments, and target features
    for framework in "${FRAMEWORKS[@]}"; do
        for ci_tpu_version in "${ACTIVE_TPU_CONFIGS[@]}"; do
            for feature in "$@"; do
                if [[ -z "$feature" ]]; then continue; fi

                # Sanitize the feature name for Buildkite metadata lookups
                local safe_feature
                safe_feature=$(sanitize_name "$feature")
                
                # Get Category (default: feature support matrix)
                local category_key="${ci_tpu_version}_${safe_feature}_category"
                local category
                category=$(buildkite-agent meta-data get "${category_key}" --default "feature support matrix")
                local category_filename=${category// /_}
                local category_csv="${category_filename}.csv"

                # Determine which stages array and header to use
                local stages_to_use=("${FEATURE_STAGES[@]}")
                local header="Feature,Machine Type,Framework,CorrectnessTest,PerformanceTest"
                local is_quantization_matrix=false

                if [ "$category" == "quantization support matrix" ]; then
                    is_quantization_matrix=true
                    stages_to_use=("${FEATURE_STAGES_QUANTIZATION[@]}")
                    header="Quantization dtype,Machine Type,Framework,Quantization methods,Recommended TPU Generations,CorrectnessTest,PerformanceTest"
                elif [ "$category" == "kernel support matrix microbenchmarks" ]; then
                    stages_to_use=("${FEATURE_STAGES_MICROBENCHMARKS[@]}")
                    header="kernels,Machine Type,Framework,CorrectnessTest,PerformanceTest"
                elif [ "$category" == "parallelism support matrix" ]; then
                    stages_to_use=("${PARALLELISM_STAGES[@]}")
                    header="Feature,Machine Type,Framework,Single-Host CorrectnessTest,Single-Host PerformanceTest,Multi-Host CorrectnessTest,Multi-Host PerformanceTest"
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
                        result="$(get_tpu_generation "$feature")"
                    elif [ "$is_quantization_matrix" = true ] && [ "$stage" == "QuantizationMethods" ]; then
                        result="$(get_quantization_method "$feature")"
                    elif [ "$stage" == "Machine Type" ]; then
                        if [ "${ci_tpu_version}" == "tpu7x" ]; then
                            result="v7x"
                        else
                            result="v6e"
                        fi
                    elif [ "$stage" == "Framework" ]; then
                        result="${framework}"
                    elif [[ "$mode" == "DEFAULT" ]]; then
                        result="✅ Passing"
                    else
                        local safe_feature
                        safe_feature=$(sanitize_name "$feature")
                        local safe_stage
                        safe_stage=$(sanitize_name "$stage")
                        local meta_key="${ci_tpu_version}_${framework}_${safe_feature}_${safe_stage}"
                        result=$(buildkite-agent meta-data get "${meta_key}" --default "❓ Untested")

                        # Format any remaining custom strings from upstream configs
                        local result_lower
                        result_lower="$(echo "$result" | tr '[:upper:]' '[:lower:]')"
                        if [[ "$result_lower" == "beta" ]]; then
                            result="⚠️ Beta"
                        elif [[ "$result_lower" == "experimental" ]]; then
                            result="🧪 Experimental"
                        elif [[ "$result_lower" == "planned" ]]; then
                            result="📝 Planned"
                        elif [[ "$result_lower" == "unplanned" ]]; then
                            result="⛔️ Unplanned"
                        fi
                    fi
                    row="$row,$result"

                    # Check for failure (exclude the descriptive structural columns)
                    if [ "$stage" != "QuantizationMethods" ] && \
                       [ "$stage" != "RecommendedTPUGenerations" ] && \
                       [ "$stage" != "Machine Type" ] && \
                       [ "$stage" != "Framework" ] && \
                       [[ "${result}" != "✅ Passing" && "${result}" != "⚪ N/A" && "${result}" != "❓ Untested" && "${result}" != "⚠️ Beta" && "${result}" != "🧪 Experimental" && "${result}" != "📝 Planned" && "${result}" != "⛔️ Unplanned" ]]; then
                        ANY_FAILED=true
                    fi

                    stage_index=$((stage_index + 1))
                done
                echo "$row" >> "$category_csv"
            done
        done
    done
}

# Pivot Logic (Microbenchmarks)
process_kernel_matrix_to_pivot() {
    local input_csv="kernel_support_matrix_microbenchmarks.csv"
    local output_file="kernel_support_matrix-microbenchmarks.csv"

    if [ ! -f "$input_csv" ]; then
        echo "Warning: Input CSV $input_csv not found. Skipping pivot."
        return
    fi

    # Define Headers for Display
    local header="Kernel microbenchmark,Machine Type,Framework,W16 A16 (Corr),W16 A16 (Perf),W8 A8 (Corr),W8 A8 (Perf),W8 A16 (Corr),W8 A16 (Perf),W4 A4 (Corr),W4 A4 (Perf),W4 A8 (Corr),W4 A8 (Perf),W4 A16 (Corr),W4 A16 (Perf)"
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
            # Map the new 5-column structure from process_features():
            # $1: Kernel, $2: Machine Type, $3: Framework, $4: Correctness, $5: Performance
            
            raw_kernel = $1;
            gsub(/"/, "", raw_kernel);
            machine_type = $2;
            framework = $3;
            
            if (match(raw_kernel, /-(w[0-9]+a[0-9]+)$/)) {
                quant_type = substr(raw_kernel, RSTART + 1, RLENGTH - 1);
                base_kernel_key = substr(raw_kernel, 1, RSTART - 1);
            } else {
                 base_kernel_key = raw_kernel;
                 quant_type = "w16a16";
            }

            # Create a composite key to group by Kernel + Machine Type + Framework
            composite_key = base_kernel_key SUBSEP machine_type SUBSEP framework;

            # Store Correctness ($4) and Performance ($5) based on the composite key
            matrix[composite_key, quant_type] = $4 OFS $5;

            # Track unique composite keys to preserve insertion order
            if (! (composite_key in seen_keys)) {
                seen_keys[composite_key] = 1;
                key_list[num_keys++] = composite_key;
            }
        }
        END {
            for (i=0; i<num_keys; i++) {
                composite_key = key_list[i];
                split(composite_key, parts, SUBSEP);
                k = parts[1];
                m_type = parts[2];
                fw = parts[3];
                
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

                # Construct the starting row with the new axes
                row = out_name OFS m_type OFS fw;
                
                for (j=1; j<=6; j++) {
                    q = q_order[j];
                    # Use formatted N/A if data is missing
                    data = (matrix[composite_key, q] == "") ? "❓ Untested,❓ Untested" : matrix[composite_key, q];
                    row = row OFS data;
                }
                print row >> "'"$output_file"'";
            }
        }
    ' "$input_csv"

    # Upload the newly created pivot table
    echo "--- Uploading Pivoted Kernel Matrix: $output_file ---"
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
        # Sort by Framework (k4 asc), Machine Type (k3 desc), Type (k2 asc), then Model Name (k1 asc)
        sorted_content=$(tail -n +2 "$csv_file" | sort -t',' -k4,4 -k3,3r -k2,2 -k1,1V)
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
        # Conditionally apply the correct sort pattern based on the matrix type
        if [[ "$csv_file" == *"quantization"* || "$csv_file" == *"rl_support"* ]]; then
            # Framework (k3 asc) -> Machine Type (k2 desc) -> Feature (k1 asc)
            sorted_content=$(tail -n +2 "$csv_file" | sort -t',' -k3,3 -k2,2r -k1,1V)
        else
            # Machine Type (k2 desc) -> Framework (k3 asc) -> Feature (k1 asc)
            sorted_content=$(tail -n +2 "$csv_file" | sort -t',' -k2,2r -k3,3 -k1,1V)
        fi

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
rm -f "${model_csv_files[@]}" "${feature_csv_files[@]}"
