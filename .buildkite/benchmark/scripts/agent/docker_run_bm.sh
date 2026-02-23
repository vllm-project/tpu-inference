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

if [ ! -f "$1" ]; then
  echo "Error: The env file '$1' does not exist."
  exit 1  # Exit the script with a non-zero status to indicate an error
fi

ENV_FILE=$1

# For testing on local vm, use `set -a` to export all variables
source /etc/environment
source $ENV_FILE

echo "--- Cat Env start"
echo $(cat $ENV_FILE)
echo "--- Cat Env end"

remove_docker_container() {
    echo "Removing Docker container: $CONTAINER_NAME"
    docker rm -f tpu-test || true;
    docker rm -f vllm-tpu || true;
    docker rm -f $CONTAINER_NAME || true;
}

cleanup_docker_image() {
    if [[ -n "$IMAGE_TAG" ]]; then
        echo "Removing Docker image: $IMAGE_TAG"
        docker rmi "$IMAGE_TAG" 2>/dev/null || true
    else
        echo "IMAGE_TAG not found"
    fi
}
trap "remove_docker_container; cleanup_docker_image" EXIT

# Remove the container that might not be cleaned up in the previous run.
remove_docker_container

echo "Code_Hash: $CODE_HASH"

IMAGE_TAG="$GCP_REGION-docker.pkg.dev/$GCP_PROJECT_ID/$ARTIFACT_REPO/vllm-tpu:$CODE_HASH"

IFS='-' read -r VLLM_HASH TPU_INFERENCE_HASH TORCHAX_HASH _ <<< "$CODE_HASH"

echo "image tag: $IMAGE_TAG"

gcloud auth configure-docker $GCP_REGION-docker.pkg.dev --quiet
docker pull $IMAGE_TAG

if [ $? -ne 0 ]; then
  echo "Failed to pull the Docker image: $IMAGE_TAG"
  exit 1
fi

LOG_ROOT=$(mktemp -d)
# Temp write to another bucket
# REMOTE_LOG_ROOT="gs://$GCS_BUCKET/job_logs/$RECORD_ID/"
REMOTE_LOG_ROOT="gs://vllm-bm-bk-storage/job_logs/$RECORD_ID/"

# If mktemp fails, set -e will cause the script to exit.
echo "Results will be stored in: $LOG_ROOT"

if [ -z "$HF_TOKEN" ]; then
  echo "Error: HF_TOKEN is not set or is empty."
  exit 1
fi

# Make sure mounted disk or dir exists
if [ ! -d "$DOWNLOAD_DIR" ]; then
    echo "Error: Folder $DOWNLOAD_DIR does not exist. This is useually a mounted drive. If no mounted drive, just create a folder."
    exit 1
fi

if ! mountpoint -q "$DOWNLOAD_DIR"; then
    echo "Error: $DOWNLOAD_DIR exists but is not a mounted directory."
    exit 1
fi


# Check and trim
# Example value
# TARGET_COMMIT="bb81182d3_de509ae8e"
# TARGET_COMMIT="bb81182d3"
TARGET_COMMIT=$VLLM_HASH
if [[ "$TARGET_COMMIT" == *_* ]]; then
  TARGET_COMMIT="${TARGET_COMMIT%%_*}"
fi

echo "Run model $MODEL"
echo

EXTRA_DOCKER_ARGS=""
if [ -n "$SKIP_JAX_PRECOMPILE" ]; then
  EXTRA_DOCKER_ARGS="-e SKIP_JAX_PRECOMPILE=$SKIP_JAX_PRECOMPILE"
fi

echo "starting docker...$CONTAINER_NAME"
echo
docker run \
 -v $DOWNLOAD_DIR:$DOWNLOAD_DIR \
 --env-file $ENV_FILE \
 -e HF_TOKEN="$HF_TOKEN" \
 -e TARGET_COMMIT=$TARGET_COMMIT \
 -e MODEL=$MODEL \
 -e DATASET=$DATASET \
 -e WORKSPACE=/workspace \
 $EXTRA_DOCKER_ARGS \
 --name $CONTAINER_NAME \
 -d \
 --privileged \
 --network host \
 -v /dev/shm:/dev/shm \
 $IMAGE_TAG tail -f /dev/null

# =============== temp solution start ===============

DATASETS=("custom-token" "mmlu" "mlperf" "bench-custom-token" "math500" "bench-custom-mm")
if [[ " ${DATASETS[*]} " == *" $DATASET "* ]]; then
  echo "Temp solution: Syncing dataset for $DATASET"

  DATASET_DOWNLOAD_DIR="./artifacts/dataset"
  mkdir -p "$DATASET_DOWNLOAD_DIR"

  if [ "$DATASET" = "custom-token" ]; then
    # Download flat files for custom-token
    gsutil -m cp gs://$GCS_BUCKET/dataset/*.* "$DATASET_DOWNLOAD_DIR/"
  elif [ "$DATASET" = "mmlu" ]; then
    # Download mmlu directory recursively
    gsutil -m cp -r gs://$GCS_BUCKET/dataset/mmlu/* "$DATASET_DOWNLOAD_DIR/"
  elif [ "$DATASET" = "mlperf" ]; then
    # Download single jsonl file for MLPerf
    gsutil -m cp gs://vllm-cb-storage2/dataset/mlperf/mlperf_shuffled.jsonl "$DATASET_DOWNLOAD_DIR/mlperf.jsonl"
  elif [[ "$DATASET" == "bench-custom-token" || "$DATASET" == "bench-custom-mm" ]]; then
    # Download flat files for custom-token and custom-mm
    gsutil -m cp -r gs://$GCS_BUCKET/bench-dataset/* "$DATASET_DOWNLOAD_DIR/"
  elif [ "$DATASET" = "math500" ]; then
    # Download single jsonl file for math500
    gsutil -m cp -r gs://$GCS_BUCKET/dataset/math500/math500.jsonl "$DATASET_DOWNLOAD_DIR/"
  fi

  echo "Copying dataset to container..."
  docker cp "$DATASET_DOWNLOAD_DIR" "$CONTAINER_NAME:/workspace/"

  echo docker cp .buildkite/benchmark/scripts/agent/benchmark_serving.py "$CONTAINER_NAME:/workspace/vllm/benchmarks/benchmark_serving.py"
  docker cp .buildkite/benchmark/scripts/agent/benchmark_serving.py "$CONTAINER_NAME:/workspace/vllm/benchmarks/benchmark_serving.py"

  echo docker cp s.buildkite/benchmark/cripts/agent/benchmark_dataset.py "$CONTAINER_NAME:/workspace/vllm/benchmarks/benchmark_dataset.py"
  docker cp .buildkite/benchmark/scripts/agent/benchmark_dataset.py "$CONTAINER_NAME:/workspace/vllm/benchmarks/benchmark_dataset.py"
fi


# =============== temp solution end ===============

# TODO(patemotter): split these into functions
if [ "$DATASET" = "sharegpt" ]; then
  echo "Copying dataset to container..."
  mkdir -p ./artifacts/dataset/
  gsutil cp gs://$GCS_BUCKET/dataset/sharegpt/*.* ./artifacts/dataset/
  docker cp artifacts/dataset "$CONTAINER_NAME:/workspace/"
fi

if [[ "$DATASET" == "math500" || "$DATASET" == "mmlu" || "$DATASET" == "mlperf" ]]; then
  echo "Copying lm_eval directory to container..."
  docker cp lm_eval "$CONTAINER_NAME:/workspace/"
fi

# echo "Copying bench_serving directory to container..."
# docker exec "$CONTAINER_NAME" mkdir -p /workspace/vllm/scripts/agent/bench_serving
# docker cp .buildkite/benchmark/scripts/agent/bench_serving/. "$CONTAINER_NAME:/workspace/vllm/scripts/agent/bench_serving/"

echo "copy script run_bm.sh to container..."
docker cp .buildkite/benchmark/scripts/agent/run_bm.sh "$CONTAINER_NAME:/workspace/vllm/run_bm.sh"

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

echo "gsutil cp $LOG_ROOT/* $REMOTE_LOG_ROOT"
gsutil cp -r $LOG_ROOT/* $REMOTE_LOG_ROOT

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
    throughput=$(grep "Request throughput (req/s):" "$BM_LOG" | sed 's/[^0-9.]//g')
    echo "throughput for $TEST_NAME at $VLLM_HASH: $throughput"

    output_token_throughput=$(grep "Output token throughput (tok/s):" "$BM_LOG" | sed 's/[^0-9.]//g')
    total_token_throughput=$(grep "Total Token throughput (tok/s):" "$BM_LOG" | sed 's/[^0-9.]//g')

    if [[ -z "$throughput" || ! "$throughput" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
        echo "Failed to get the throughput and this is not an accuracy run."
        exit 1
    fi

    if (( $(echo "$throughput < ${EXPECTED_THROUGHPUT:-0}" | bc -l) )); then
        echo "Error: throughput($throughput) is less than expected($EXPECTED_THROUGHPUT)"
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

