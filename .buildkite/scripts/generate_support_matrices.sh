#!/bin/bash
set -euo pipefail

ANY_FAILED=false

MODEL_LIST_KEY="model-list"
FEATURE_LIST_KEY="feature-list"

# Note: This script assumes the metadata keys contain newline-separated lists.
# The `mapfile` command reads these lists into arrays, correctly handling spaces.
mapfile -t model_list < <(buildkite-agent meta-data get "${MODEL_LIST_KEY}" --default "")
mapfile -t feature_list < <(buildkite-agent meta-data get "${FEATURE_LIST_KEY}" --default "")
MODEL_STAGES=("UnitTest" "IntegrationTest" "Benchmark")
FEATURE_STAGES=("CorrectnessTest" "PerformanceTest")

# Output CSV files
model_support_matrix_csv="model_support_matrix.csv"
echo "Model,UnitTest,IntegrationTest,Benchmark" > "$model_support_matrix_csv"

feature_support_matrix_csv="feature_support_matrix.csv"
echo "Feature,CorrectnessTest,PerformanceTest" > "$feature_support_matrix_csv"

process_models() {
    for model in "$@"; do
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
    for feature in "$@"; do
        row="\"$feature\""
        for stage in "${FEATURE_STAGES[@]}"; do
            result=$(buildkite-agent meta-data get "${feature}:${stage}" --default "N/A")
            row="$row,$result"
            if [ "${result}" != "✅" ] && [ "${result}" != "N/A" ] ; then
                ANY_FAILED=true
            fi
        done
        echo "$row" >> "$feature_support_matrix_csv"
    done
}

if [ ${#model_list[@]} -gt 0 ]; then
    process_models "${model_list[@]}"
fi

if [ ${#feature_list[@]} -gt 0 ]; then
    process_features "${feature_list[@]}"
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
