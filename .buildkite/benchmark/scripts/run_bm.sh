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

EXPECTED_ETEL="${EXPECTED_ETEL:-3600000}"

# Datasets using lm-evaluation-harness `lm_eval`.
LM_EVAL_DATASETS=("math500" "mmlu" "mlperf")

# Ignore the error because in case of using uv, the packages are installed outside this script.
pip install evaluate==0.4.5 || true
pip install rouge-score==0.1.2 || true
# Install lm_eval with dependencies, version is same as https://github.com/vllm-project/vllm/blob/main/.buildkite/scripts/hardware_ci/run-tpu-v1-test.sh#L64
pip install "lm-eval[api,math]>=0.4.9.2" || true

VLLM_LOG="$DOCKER_LOG_FOLDER/vllm_log.txt"
BM_LOG="$DOCKER_LOG_FOLDER/bm_log.txt"
BEST_BM_LOG="$DOCKER_LOG_FOLDER/best_bm_log.txt"
# shellcheck disable=SC2034
PROFILE_FOLDER="$DOCKER_LOG_FOLDER/profile"
DOCKER_ARTIFACT_FOLDER=${DOCKER_ARTIFACT_FOLDER:-"/workspace/artifacts"}
printf "[INFO] %-25s = %s\n" "VLLM_LOG" "$VLLM_LOG"
printf "[INFO] %-25s = %s\n" "BM_LOG" "$BM_LOG"
printf "[INFO] %-25s = %s\n" "DOCKER_ARTIFACT_FOLDER" "$DOCKER_ARTIFACT_FOLDER"

# Helper function to check if a value is in an array
contains_element () {
  local e match="$1"
  shift
  for e; do [[ "$e" == "$match" ]] && return 0; done
  return 1
}

# Run accuracy benchmark via lm_eval (Retained for backward compatibility)
if contains_element "$DATASET" "${LM_EVAL_DATASETS[@]}"; then
  echo "DATASET ($DATASET) is an accuracy benchmark. Running lm_eval path."
  {
    ".buildkite/benchmark/lm_eval/$DATASET/run.sh"
    printf "AccuracyMetrics: "
    tr -d '\n' < "/workspace/${DATASET}_accuracy.json"
    echo ""
  } >> "$BM_LOG"
  echo "Finished running $DATASET benchmark."
  exit 0
fi

if [ "$DATASET" = "sonnet" ]; then
  echo "Create sonnet_4x.txt"
  echo "" > benchmarks/sonnet_4x.txt
  for _ in {1..4}
    do
     cat benchmarks/sonnet.txt >> benchmarks/sonnet_4x.txt
  done
fi

#
# Start vllm service in backend using the full command from YAML
#
echo "Launching vllm..."
echo "Logging to $VLLM_LOG"

if [ -z "$VLLM_SERVE_CMD" ]; then
    echo "[ERROR] VLLM_SERVE_CMD is empty. Please check your YAML configuration and Parser."
    exit 1
fi

# Execute the parsed full command in background
echo "Executing: $VLLM_SERVE_CMD > \"$VLLM_LOG\" 2>&1 &"
eval "$VLLM_SERVE_CMD > \"$VLLM_LOG\" 2>&1 &"

echo "Waiting for vLLM Server Ready (up to 60 minutes).."
echo
for _ in {1..360}; do
    # TODO: detect other type of errors.
    if grep -Fq "raise RuntimeError" "$VLLM_LOG"; then
        echo "Detected RuntimeError, exiting."
        cat "$VLLM_LOG"
        exit 1
    elif grep -Fq "Application startup complete" "$VLLM_LOG"; then
        echo "Application started"
        break
    else
        echo "wait for 10 seconds..."
        sleep 10
    fi
done

# Resolve DATASET_PATH
case "${DATASET}" in
  "sonnet")
    export DATASET_PATH="benchmarks/sonnet_4x.txt"
    ;;
  "mmlu")
    export DATASET_PATH="$DOCKER_ARTIFACT_FOLDER/dataset"
    ;;
  "mlperf")
    export DATASET_PATH="$DOCKER_ARTIFACT_FOLDER/dataset/processed-data.pkl"
    ;;
  "custom-token")
    export DATASET_PATH="$DOCKER_ARTIFACT_FOLDER/dataset/${MODEL##*/}_${INPUT_LEN}_${OUTPUT_LEN}_tp${TENSOR_PARALLEL_SIZE}.json"
    ;;
  "bench-custom-token")
    export DATASET_PATH="$DOCKER_ARTIFACT_FOLDER/dataset/${MODEL##*/}/inlen${INPUT_LEN}_outlen${OUTPUT_LEN}_prefixlen${PREFIX_LEN}.jsonl"
    ;;
  "bench-custom-mm")
    DATA_DIR="$DOCKER_ARTIFACT_FOLDER/dataset/${MODEL##*/}"
    dataset_files=()
    mapfile -d $'\0' dataset_files < <(find "$DATA_DIR" -name "inlen${INPUT_LEN}_outlen${OUTPUT_LEN}_prefixlen${PREFIX_LEN}*.jsonl" -print0)
    if [ ${#dataset_files[@]} -ne 1 ]; then
      echo "Error: Found ${#dataset_files[@]} matching datasets in $DATA_DIR, but expected 1." >&2
      echo "Matching files:" >&2
      printf " - %s\n" "${dataset_files[@]}" >&2
      exit 1
    fi
    export DATASET_PATH="${dataset_files[0]}"
    ;;
  "sharegpt")
    export DATASET_PATH="$DOCKER_ARTIFACT_FOLDER/dataset/ShareGPT_V3_unfiltered_cleaned_split.json"
    ;;
  "hf")
    export DATASET_PATH="lmarena-ai/VisionArena-Chat"
    ;;
  *)
    echo "Error: unsupported dataset '$DATASET'" > "$BM_LOG" 2>&1
    exit 1
    ;;
esac

if [ -z "$BENCHMARK_CMD" ]; then
    echo "[ERROR] BENCHMARK_CMD is empty. Please check your YAML configuration and Parser."
    exit 1
fi

# Check original request rate
ORIGINAL_RATE=$(echo "$BENCHMARK_CMD" | grep -oE '--request-rate +[^ ]+' | awk '{print $2}' || echo "")
if [ -n "$ORIGINAL_RATE" ] && [ "$ORIGINAL_RATE" != "inf" ]; then
    echo "[WARNING] User provided --request-rate $ORIGINAL_RATE in BENCHMARK_CMD. Ignoring and substituting dynamically during benchmark."
fi

run_benchmark(){
  local request_rate="$1"
  echo "Running benchmark with request_rate=$request_rate..."
  echo "Logging to $BM_LOG"

  # Replace --request-rate <value> (e.g., inf) dynamically with the current rate ($request_rate)
  local CURRENT_CMD
  CURRENT_CMD=$(echo "$BENCHMARK_CMD" | sed -E "s/--request-rate +[^ ]+/--request-rate $request_rate/g")

  echo "Executing: $CURRENT_CMD > \"$BM_LOG\" 2>&1"

  # Execute the benchmark command
  eval "$CURRENT_CMD > \"$BM_LOG\" 2>&1"

  local throughput
  local p99_e2el
  throughput=$(grep "Request throughput (req/s):" "$BM_LOG" | sed 's/[^0-9.]//g')
  p99_e2el=$(grep "P99 E2EL (ms):" "$BM_LOG" | awk '{print $NF}')
  echo "throughput: $throughput, P99 E2EL:$p99_e2el" >&2
  echo "$throughput $p99_e2el"
}

printf "[DEBUG] Checking folder structure in container...\n"
printf "[DEBUG] pwd=%s\n\nls $DOCKER_ARTIFACT_FOLDER=\n%s\n" "$(pwd)" "$(ls "$DOCKER_ARTIFACT_FOLDER")" || true
printf "[DEBUG] ls $DOCKER_ARTIFACT_FOLDER/temp_logs=\n%s\n" "$(ls "$DOCKER_ARTIFACT_FOLDER"/temp_logs)" || true

# Initial run with 'inf' request rate
read -r throughput p99_e2el < <(run_benchmark "inf" | tail -n 1)

echo "Initial Throughput: $throughput"
echo "Initial P99 E2EL: $p99_e2el"

# Step 1: check if initial run meets the E2EL requirement
p99_int=$(printf "%.0f" "$p99_e2el")
goal_int=$(printf "%.0f" "$EXPECTED_ETEL")

if (( p99_int <= goal_int )); then
  echo "Initial run: P99 E2EL ($p99_e2el ms) <= EXPECTED_ETEL ($EXPECTED_ETEL ms), good enough. Exiting 0."
  exit 0
fi

echo "Initial run failed: P99 E2EL ($p99_e2el ms) > EXPECTED_ETEL ($EXPECTED_ETEL ms)"
echo "Starting binary search to lower request rate..."

# Step 2: Begin binary search
low=0
high=$(printf "%.0f" "$throughput")
goal=$EXPECTED_ETEL

# Round goal to nearest int
goal_int=$(printf "%.0f" "$goal")

best_rate=0
best_throughput=0
best_e2el=0

while (( high - low > 0 )); do
  mid=$(( (low + high + 1) / 2 ))
  echo "Trying request_rate=$mid"

  read -r throughput p99_e2el < <(run_benchmark "$mid" | tail -n 1)

  # Convert p99_e2el to integer
  p99_int=$(printf "%.0f" "$p99_e2el")

  if (( p99_int <= goal_int )); then
    echo "PASS: p99_e2el=$p99_e2el <= $goal"
    best_rate=$mid
    best_throughput=$throughput
    best_e2el=$p99_e2el
    low=$mid

    # Backup best log
    cp "$BM_LOG" "$BEST_BM_LOG"
  else
    echo "FAIL: p99_e2el=$p99_e2el > $goal"
    high=$((mid - 1))
  fi
done

if (( best_rate == 0 )); then
  echo "Could not find a valid request_rate >= 1 that meets EXPECTED_ETEL=$EXPECTED_ETEL" | tee -a "$BM_LOG"
  exit 1
fi

# Restore the best log to BM_LOG
cp "$BEST_BM_LOG" "$BM_LOG"

echo
echo "======================================"
echo "✓ Final best request_rate: $best_rate"
echo "✓ Throughput: $best_throughput"
echo "✓ P99 E2EL: $best_e2el"
echo "======================================"
