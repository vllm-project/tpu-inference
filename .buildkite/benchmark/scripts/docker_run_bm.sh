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

set -euo pipefail

if [ ! -f "$1" ]; then
  echo "Error: The env file '$1' does not exist."
  exit 1  # Exit the script with a non-zero status to indicate an error
fi

ENV_FILE=$1

# For testing on local vm, use `set -a` to export all variables
# shellcheck disable=1091
source /etc/environment
# shellcheck source=/dev/null
source "$ENV_FILE"

echo "--- Cat Env start"
cat "$ENV_FILE"
echo "--- Cat Env end"

echo "Code_Hash: $CODE_HASH"

# shellcheck disable=SC2034
IFS='-' read -r VLLM_HASH TPU_INFERENCE_HASH _ <<< "$CODE_HASH"

LOG_ROOT=$(mktemp -d)
# Temp write to another bucket
# REMOTE_LOG_ROOT="gs://$GCS_BUCKET/job_logs/$RECORD_ID/"
REMOTE_LOG_ROOT="gs://vllm-bm-bk-storage/job_logs/$RECORD_ID/"

# If mktemp fails, set -e will cause the script to exit.
echo "Results will be stored in: $LOG_ROOT"

echo "Run model $MODEL"
echo

EXTRA_DOCKER_ARGS=()
if [ -n "${SKIP_JAX_PRECOMPILE:-}" ]; then
  EXTRA_DOCKER_ARGS+=("-e SKIP_JAX_PRECOMPILE=${SKIP_JAX_PRECOMPILE}")
fi

echo "starting docker...$CONTAINER_NAME"
echo
# shellcheck disable=SC2153
docker run \
 --rm \
 -v "$LOCAL_HF_HOME":"$DOCKER_HF_HOME" \
 --env-file "$ENV_FILE" \
 -e HF_TOKEN="$HF_TOKEN" \
 -e HF_HOME="$DOCKER_HF_HOME" \
 -e TARGET_COMMIT="$TARGET_COMMIT" \
 -e MODEL="$MODEL" \
 -e DATASET="$DATASET" \
 -e WORKSPACE=/workspace \
 "${EXTRA_DOCKER_ARGS[@]}" \
 --name "$CONTAINER_NAME" \
 -d \
 --privileged \
 --network host \
 -v /dev/shm:/dev/shm \
 "$IMAGE_TAG" tail -f /dev/null

# =============== temp solution start ===============

DATASETS=("custom-token" "mmlu" "mlperf" "bench-custom-token" "math500" "bench-custom-mm")
if [[ " ${DATASETS[*]} " == *" $DATASET "* ]]; then
  echo "Temp solution: Syncing dataset for $DATASET"

  DATASET_DOWNLOAD_DIR="./artifacts/dataset"
  mkdir -p "$DATASET_DOWNLOAD_DIR"

  if [ "$DATASET" = "custom-token" ]; then
    # Download flat files for custom-token
    gsutil -m cp gs://"$GCS_BUCKET"/dataset/*.* "$DATASET_DOWNLOAD_DIR/"
  elif [ "$DATASET" = "mmlu" ]; then
    # Download mmlu directory recursively
    gsutil -m cp -r gs://"$GCS_BUCKET"/dataset/mmlu/* "$DATASET_DOWNLOAD_DIR/"
  elif [ "$DATASET" = "mlperf" ]; then
    # Download single jsonl file for MLPerf
    gsutil -m cp gs://vllm-cb-storage2/dataset/mlperf/mlperf_shuffled.jsonl "$DATASET_DOWNLOAD_DIR/mlperf.jsonl"
  elif [[ "$DATASET" == "bench-custom-token" || "$DATASET" == "bench-custom-mm" ]]; then
    # Download flat files for custom-token and custom-mm
    gsutil -m cp -r gs://"$GCS_BUCKET"/bench-dataset/* "$DATASET_DOWNLOAD_DIR/"
  elif [ "$DATASET" = "math500" ]; then
    # Download single jsonl file for math500
    gsutil -m cp -r gs://"$GCS_BUCKET"/dataset/math500/math500.jsonl "$DATASET_DOWNLOAD_DIR/"
  fi

  echo "Copying dataset to container..."
  docker cp "$DATASET_DOWNLOAD_DIR" "$CONTAINER_NAME:/workspace/"

  echo docker cp .buildkite/benchmark/scripts/benchmark_serving.py "$CONTAINER_NAME:/workspace/vllm/benchmarks/benchmark_serving.py"
  docker cp .buildkite/benchmark/scripts/benchmark_serving.py "$CONTAINER_NAME:/workspace/vllm/benchmarks/benchmark_serving.py"

  echo docker cp .buildkite/benchmark/scripts/benchmark_dataset.py "$CONTAINER_NAME:/workspace/vllm/benchmarks/benchmark_dataset.py"
  docker cp .buildkite/benchmark/scripts/benchmark_dataset.py "$CONTAINER_NAME:/workspace/vllm/benchmarks/benchmark_dataset.py"
fi


# =============== temp solution end ===============
# Download deepseek generation configs into the container if the model is DeepSeek-R1
if [[ "$MODEL" == "deepseek-ai/DeepSeek-R1" ]]; then
  echo "Downloading deepseek generation_configs..."
  mkdir -p ./artifacts/generation_configs
  gsutil -m cp -r gs://gpolovets-inference/deepseek/generation_configs/* ./artifacts/generation_configs/
  docker exec "$CONTAINER_NAME" mkdir -p /workspace/generation_configs/
  docker cp ./artifacts/generation_configs/. "$CONTAINER_NAME:/workspace/generation_configs/"
fi

# TODO(patemotter): split these into functions
if [ "$DATASET" = "sharegpt" ]; then
  echo "Copying dataset to container..."
  mkdir -p ./artifacts/dataset/
  gsutil cp gs://"$GCS_BUCKET"/dataset/sharegpt/*.* ./artifacts/dataset/
  docker cp artifacts/dataset "$CONTAINER_NAME:/workspace/"
fi

if [[ "$DATASET" == "math500" || "$DATASET" == "mmlu" || "$DATASET" == "mlperf" ]]; then
  echo "Copying lm_eval directory to container..."
  docker cp .buildkite/benchmark/lm_eval "$CONTAINER_NAME:/workspace/"
fi

echo "Copying bench_serving directory to container..."
docker exec "$CONTAINER_NAME" mkdir -p /workspace/vllm/scripts/bench_serving
docker cp .buildkite/benchmark/scripts/bench_serving/. "$CONTAINER_NAME:/workspace/vllm/scripts/bench_serving/"

echo "copy script run_bm.sh to container..."
docker cp .buildkite/benchmark/scripts/run_bm.sh "$CONTAINER_NAME:/workspace/vllm/run_bm.sh"

echo "grant chmod +x"
echo
docker exec "$CONTAINER_NAME" chmod +x "/workspace/vllm/run_bm.sh"

echo "run script..."
echo
docker exec -w /workspace/vllm "$CONTAINER_NAME" /bin/bash -c "echo always > /sys/kernel/mm/transparent_hugepage/enabled && ./run_bm.sh"

echo "copy results and logs back..."
VLLM_LOG="$LOG_ROOT/$TEST_NAME"_vllm_log.txt
BM_LOG="$LOG_ROOT/$TEST_NAME"_bm_log.txt
PROFILE_FOLDER="$LOG_ROOT/$TEST_NAME"_profile
docker cp "$CONTAINER_NAME:/workspace/vllm_log.txt" "$VLLM_LOG"
docker cp "$CONTAINER_NAME:/workspace/bm_log.txt" "$BM_LOG"
docker cp "$CONTAINER_NAME:/workspace/profile/plugins/profile" "$PROFILE_FOLDER"
docker cp "$CONTAINER_NAME:/workspace/failed_output.json" "$LOG_ROOT/failed_output.json" || true

# Upload vllm and bm log to Buildkite aritfact
ARTIFACT_VLLM="${RECORD_ID}_vllm_log.txt"
ARTIFACT_BM="${RECORD_ID}_bm_log.txt"

echo "Preparing Buildkite artifacts..."
cp "$VLLM_LOG" "$ARTIFACT_VLLM"
cp "$BM_LOG" "$ARTIFACT_BM"

echo "Uploading artifacts to Buildkite..."
buildkite-agent artifact upload "$ARTIFACT_VLLM"
buildkite-agent artifact upload "$ARTIFACT_BM"

echo "Cleaning up temporary artifact files..."
rm -f "$ARTIFACT_VLLM" "$ARTIFACT_BM"

echo "gsutil cp $LOG_ROOT/* $REMOTE_LOG_ROOT"
gsutil cp -r "$LOG_ROOT"/* "$REMOTE_LOG_ROOT"

AccuracyMetricsJSON=$(grep -a "AccuracyMetrics:" "$BM_LOG" | sed 's/AccuracyMetrics: //')
echo "AccuracyMetricsJSON: $AccuracyMetricsJSON"

if [[ "$RUN_TYPE" == *"ACCURACY"* ]]; then
    # Accuracy run logic
    echo "Accuracy run ($RUN_TYPE) detected. Parsing accuracy metrics."
    if [ -n "$AccuracyMetricsJSON" ]; then
        echo "AccuracyMetrics=$AccuracyMetricsJSON" > "artifacts/$RECORD_ID.result"
    else
        echo "Error: Accuracy run but no AccuracyMetrics found."
        exit 1
    fi
else
    # Performance run logic
    throughput=$(grep -i "Request throughput (req/s):" "$BM_LOG" | sed 's/[^0-9.]//g')
    echo "throughput for $TEST_NAME at $VLLM_HASH: $throughput"

    output_token_throughput=$(grep -i "Output token throughput (tok/s):" "$BM_LOG" | sed 's/[^0-9.]//g')
    total_token_throughput=$(grep -i "Total Token throughput (tok/s):" "$BM_LOG" | sed 's/[^0-9.]//g')

    if [[ -z "$throughput" || ! "$throughput" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
        echo "Failed to get the throughput and this is not an accuracy run."
        exit 1
    fi

    if (( $(echo "$throughput < ${EXPECTED_THROUGHPUT:-0}" | bc -l) )); then
        echo "Error: throughput($throughput) is less than expected(${EXPECTED_THROUGHPUT:-0})"
    fi
    echo "Throughput=$throughput" > "artifacts/$RECORD_ID.result"

    extract_value() {
        local section="$1"
        local label="$2"  # Mean, Median, or P99
        grep "$section (ms):" "$BM_LOG" | \
        awk -v label="$label" '$0 ~ label { print $NF }'
    }

    # Median values
    MedianITL=$(extract_value "ITL" "Median")
    MedianTPOT=$(extract_value "TPOT" "Median")
    MedianTTFT=$(extract_value "TTFT" "Median")
    MedianETEL=$(extract_value "E2EL" "Median")

    # P99 values
    P99ITL=$(extract_value "ITL" "P99")
    P99TPOT=$(extract_value "TPOT" "P99")
    P99TTFT=$(extract_value "TTFT" "P99")
    P99ETEL=$(extract_value "E2EL" "P99")

    # Write results to file
    (
        printf '%s=%s\n' \
        "MedianITL" "$MedianITL" \
        "MedianTPOT" "$MedianTPOT" \
        "MedianTTFT" "$MedianTTFT" \
        "MedianETEL" "$MedianETEL" \
        "P99ITL" "$P99ITL" \
        "P99TPOT" "$P99TPOT" \
        "P99TTFT" "$P99TTFT" \
        "P99ETEL" "$P99ETEL" \
        "OutputTokenThroughput" "$output_token_throughput" \
        "TotalTokenThroughput" "$total_token_throughput"
    ) >> "artifacts/$RECORD_ID.result"
fi

