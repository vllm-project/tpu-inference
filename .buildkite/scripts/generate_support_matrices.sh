#!/bin/bash
set -euo pipefail

ANY_FAILED=false

MODEL_LIST_KEY="model-list"
FEATURE_LIST_KEY="feature-list"
DEFAULT_FEATURES_FILE=".buildkite/features/default_features.txt"

# Note: This script assumes the metadata keys contain newline-separated lists.
# The `mapfile` command reads these lists into arrays, correctly handling spaces.
mapfile -t model_list < <(buildkite-agent meta-data get "${MODEL_LIST_KEY}" --default "")
mapfile -t metadata_feature_list < <(buildkite-agent meta-data get "${FEATURE_LIST_KEY}" --default "")
MODEL_STAGES=("UnitTest" "IntegrationTest" "Benchmark")
FEATURE_STAGES=("CorrectnessTest" "PerformanceTest")

# Output CSV files
model_support_matrix_csv="model_support_matrix.csv"
echo "Model,UnitTest,IntegrationTest,Benchmark" > "$model_support_matrix_csv"

feature_support_matrix_csv="feature_support_matrix.csv"
echo "Feature,CorrectnessTest,PerformanceTest" > "$feature_support_matrix_csv"

# Read the list of default features from the specified file
if [[ -f "${DEFAULT_FEATURES_FILE}" ]]; then
    mapfile -t default_feature_list < <(sed 's/\r$//; /^$/d' "${DEFAULT_FEATURES_FILE}")
else
    default_feature_list=()
    echo "Warning: Default features file not found at ${DEFAULT_FEATURES_FILE}"
fi

process_models() {
    local row
    local result
    for model in "${model_list[@]:-}"; do
        row="\"$model\""
        for stage in "${MODEL_STAGES[@]}"; do
            result=$(buildkite-agent meta-data get "${model}:${stage}" --default "N/A")
            row="$row,$result"
            if [ "${result}" != "✅" ] && [ "${result}" != "N/A" ] ; then
                ANY_FAILED=true
            fi
        done
        echo "$row" >> "$model_support_matrix_csv"
    done
}

process_features() {
    declare -A feature_rows
    local result
    # Process features from the default list
    for feature in "${default_feature_list[@]:-}"; do
        if [[ -z "$feature" ]]; then continue; fi
        local row="\"$feature\""
        for stage in "${FEATURE_STAGES[@]}"; do
            result="✅"
            row="$row,$result"
        done
        feature_rows["$feature"]="$row"
    done
    # Process features from the metadata list
    for feature in "${metadata_feature_list[@]:-}"; do
        if [[ -z "$feature" ]]; then continue; fi
        if [[ -v feature_rows["$feature"] ]]; then
            continue
        fi
        local row="\"$feature\""
        for stage in "${FEATURE_STAGES[@]}"; do
            result=$(buildkite-agent meta-data get "${feature}:${stage}" --default "N/A")
            row="$row,$result"
            if [ "${result}" != "✅" ] && [ "${result}" != "N/A" ] ; then
                ANY_FAILED=true
            fi
        done
        feature_rows["$feature"]="$row"
    done
    # Output all unique rows, sorted, to the CSV file
    for row in "${feature_rows[@]}"; do
        echo "$row"
    done | sort -V >> "$feature_support_matrix_csv"
}

if [ ${#model_list[@]} -gt 0 ]; then
    process_models
fi

if [ ${#metadata_feature_list[@]} -gt 0 ] || [ ${#default_feature_list[@]} -gt 0 ]; then
    process_features
fi

buildkite-agent meta-data set "CI_TESTS_FAILED" "${ANY_FAILED}"

echo "--- Model support matrix ---"
cat "$model_support_matrix_csv"

echo "--- Feature support matrix ---"
cat "$feature_support_matrix_csv"

echo "--- Saving support matrices as Buildkite Artifacts ---"
buildkite-agent artifact upload "$model_support_matrix_csv"
buildkite-agent artifact upload "$feature_support_matrix_csv"
echo "Reports uploaded successfully."

# cleanup
rm "$model_support_matrix_csv" "$feature_support_matrix_csv"
