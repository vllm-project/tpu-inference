#!/bin/bash
set -euo pipefail

ANY_FAILED=false

MODEL_LIST_KEY="tpu-model-list"
INFORMATIONAL_MODEL_LIST_KEY="vllm-model-list"
POPURLAR_MODEL_LIST_KEY="popular-model-list"

FEATURE_LIST_METADATA_KEY="feature-list"

# tpu_model_list="Qwen/Qwen2.5-0.5B-Instruct Qwen/Qwen2.5-2B-Instruct"
# vllm_model_list="NousResearch/Nous-Hermes-1.4B NousResearch/Nous-Hermes-2.5B"
# popular_model_list="meta-llama/Llama-3.1-8B-Instruct meta-llama/Llama-3.2-8B-Instruct"
tpu_model_list=$(buildkite-agent meta-data get "${MODEL_LIST_KEY}" --default "")
vllm_model_list=$(buildkite-agent meta-data get "${INFORMATIONAL_MODEL_LIST_KEY}" --default "")
popular_model_list=$(buildkite-agent meta-data get "${POPURLAR_MODEL_LIST_KEY}" --default "")

feature_list="f1 f2"
STAGES="UnitTest IntTest Benchmark StressTest"

# Output CSV files
output_model_support_matrix_file="model_support_matrix.csv"
echo "Model,UnitTest,IntTest,Benchmark,StressTest" > "$output_model_support_matrix_file"

output_feature_support_matrix_file="feature_support_matrix.csv"
echo "Feature,UnitTest,IntTest,Benchmark,StressTest" > "$output_feature_support_matrix_file"

# All stages must pass for TPU models
check_tpu_model() {
    local model="$1"
    for stage in $STAGES; do
        result=$(buildkite-agent meta-data get "${model}:${stage}" --default "not_run")
        if [[ "$result" != "passed" ]]; then
            echo "TPU model $model failed at $stage ($result)"
            ANY_FAILED=true
        fi
    done
}

# Only UnitTest and IntTest must pass for VLLM models
check_vllm_model() {
    local model="$1"
    local required="UnitTest IntTest"
    for stage in $required; do
        result=$(buildkite-agent meta-data get "${model}:${stage}" --default "not_run")
        if [[ "$result" != "passed" ]]; then
            echo "VLLM model $model failed at $stage ($result)"
            ANY_FAILED=true
        fi
    done
}

process_models() {
    local model_list="$1"
    local mode="$2"   # tpu | vllm | popular
    for model in $model_list; do
        row="$model"
        for stage in $STAGES; do
            result=$(buildkite-agent meta-data get "${model}:${stage}" --default "${model}:${stage} not_run")
            row="$row,$result"
        done
        echo "$row" >> "$output_model_support_matrix_file"

        # run checks
        case $mode in
            tpu) check_tpu_model "$model" ;;
            vllm) check_vllm_model "$model" ;;
            popular) ;;
        esac
    done
}

process_features() {
    local feature_list="$1"
    for feature in $feature_list; do
        row="$feature"
        for stage in $STAGES; do
            result=$(buildkite-agent meta-data get "${feature}:${stage}" --default "${feature}:${stage} not_run")
            row="$row,$result"
        done
        echo "$row" >> "$output_feature_support_matrix_file"
    done
}

echo "--- Checking TPU models Outcomes and Generating Reports ---"
process_models "$tpu_model_list" tpu

echo "--- Checking VLLM models Outcomes and Generating Reports ---"
process_models "$vllm_model_list" vllm

echo "--- Checking popular models Outcomes and Generating Reports ---"
process_models "$popular_model_list" popular

echo "--- Checking features Outcomes and Generating Reports ---"
process_features "$feature_list"

# Get commit hashes
VLLM_COMMIT_HASH=$(buildkite-agent meta-data get 'VLLM_COMMIT_HASH' --default "not_set")
TPU_COMMONS_COMMIT_HASH=$(buildkite-agent meta-data get 'TPU_COMMONS_COMMIT_HASH' --default "not_set")

if [ "$ANY_FAILED" = true ]; then
    echo "Some checks failed!"
    echo "VLLM_COMMIT_HASH: $VLLM_COMMIT_HASH"
    echo "TPU_COMMONS_COMMIT_HASH: $TPU_COMMONS_COMMIT_HASH"
    exit 1
else
    echo "--- Uploading Commit Hash to Repo ---"
    echo "Will commit to tpu_commons main"
fi

echo "--- Print Model Report Content ---"
cat "$output_model_support_matrix_file"

echo "--- Uploading CSV Reports as Buildkite Artifacts ---"
buildkite-agent artifact upload "$output_model_support_matrix_file"
buildkite-agent artifact upload "$output_feature_support_matrix_file"
echo "Reports uploaded successfully."