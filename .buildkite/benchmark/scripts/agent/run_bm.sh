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

# Strip quotes from environment variables (important when passed via docker --env-file)
# which doesn't strip quotes like bash 'source' does.
for var in MODEL DATASET NUM_PROMPTS INPUT_LEN OUTPUT_LEN EXPECTED_ETEL TENSOR_PARALLEL_SIZE MAX_NUM_SEQS MAX_NUM_BATCHED_TOKENS MAX_MODEL_LEN PREFIX_LEN ADDITIONAL_CONFIG EXTRA_ARGS; do
  if [ -n "${!var:-}" ]; then
    val="${!var}"
    val="${val#\'}"
    val="${val%\'}"
    val="${val#\"}"
    val="${val%\"}"
    export "$var"="$val"
  fi
done

# Datasets using lm-evaluation-harness `lm_eval`.
LM_EVAL_DATASETS=("math500" "mmlu" "mlperf")

# All other datasets will use the standard `vllm bench serve` command.

# TODO: Move to image building.
# Ingore the error because in case of using uv, the packages are installed outside this script.
pip install pandas || true
pip install datasets || true
pip install evaluate==0.4.5 || true
pip install rouge-score==0.1.2 || true
# Install lm_eval with dependencies, version is same as https://github.com/vllm-project/vllm/blob/main/.buildkite/scripts/hardware_ci/run-tpu-v1-test.sh#L64
pip install "lm-eval[api,math]>=0.4.9.2" || true

VLLM_LOG="$WORKSPACE/vllm_log.txt"
BM_LOG="$WORKSPACE/bm_log.txt"
BEST_BM_LOG="$WORKSPACE/best_bm_log.txt"
PROFILE_FOLDER="$WORKSPACE/profile"


if [ -n "$TARGET_COMMIT" ]; then
  head_hash=$(git rev-parse HEAD)
  resolved_target=$(git rev-parse "$TARGET_COMMIT" 2>/dev/null)

  if [ -z "$resolved_target" ]; then
    echo "Error: target commit '$TARGET_COMMIT' is not a valid Git object" | tee -a $VLLM_LOG
    exit 1
  fi

  if [ "$resolved_target" != "$head_hash" ]; then
    echo "Error: target commit '$TARGET_COMMIT' does not match HEAD: $head_hash" | tee -a $VLLM_LOG
    exit 1
  fi
fi

echo "model: $MODEL"
echo

# Helper function to check if a value is in an array
contains_element () {
  local e match="$1"
  shift
  for e; do [[ "$e" == "$match" ]] && return 0; done
  return 1
}

# Run accuracy benchmark via lm_eval
if contains_element "$DATASET" "${LM_EVAL_DATASETS[@]}"; then
  echo "DATASET ($DATASET) is an accuracy benchmark. Running lm_eval path."
  /workspace/lm_eval/$DATASET/run.sh >> /workspace/bm_log.txt
  printf "AccuracyMetrics: " >> /workspace/bm_log.txt
  cat "/workspace/${DATASET}_accuracy.json" | tr -d '\n' >> /workspace/bm_log.txt
  echo "" >> /workspace/bm_log.txt
  echo "Finished running $DATASET benchmark."
  exit 0
fi


# create a log and profile folder
mkdir -p "$WORKSPACE/log"
mkdir -p "$PROFILE_FOLDER"

if [ "$DATASET" = "sonnet" ]; then
  echo "Create sonnet_4x.txt"
  echo "" > benchmarks/sonnet_4x.txt
  for _ in {1..4}
    do
     cat benchmarks/sonnet.txt >> benchmarks/sonnet_4x.txt
  done
fi

#
# start vllm service in backend
#
echo "lanching vllm..."
echo "logging to $VLLM_LOG"
echo

if [[ -z "${EXTRA_ARGS:-}" ]]; then
  # If it is unset or empty, we initialize it as an empty string.
  # This makes the append operation (+=) safe to use later.
  EXTRA_ARGS=""
fi

if [[ "$MODEL" == "google/gemma-3-27b-it" ]]; then
  echo "google/gemma-3-27b-it"
  EXTRA_ARGS+="--limit-mm-per-prompt {\"image\":0}"
elif [[ "$MODEL" == "Qwen/Qwen2.5-VL-7B-Instruct" || "$MODEL" == "Qwen/Qwen2.5-VL-32B-Instruct" ]]; then
  echo "$MODEL"
  EXTRA_ARGS+="--limit-mm-per-prompt {\"image\":1} --mm-processor-kwargs {\"max_pixels\":1024000}"
elif [[ "$MODEL" == "deepseek-ai/DeepSeek-R1" ]]; then
  echo "deepseek-ai/DeepSeek-R1"
  EXTRA_ARGS+=" --hf-config=deepseek-ai/DeepSeek-R1 --hf_overrides '{\"architectures\": [\"DeepseekV3ForCausalLM\"]}' --gpu-memory-utilization 0.91"
fi

if [[ -n "${ADDITIONAL_CONFIG:-}" ]]; then
  printf -v quoted_config "%q" "$ADDITIONAL_CONFIG"
  echo "Adding --additional_config=${quoted_config} to EXTRA_ARGS for running vllm serve ..."
  EXTRA_ARGS+=" --additional_config=${quoted_config}"
fi

echo "Printing the vllm serve command used to start the server:"
echo "VLLM_USE_V1=1 VLLM_TORCH_PROFILER_DIR=\"$PROFILE_FOLDER\" vllm serve $MODEL \
 --seed 42 \
 --disable-log-requests \
 --max-num-seqs $MAX_NUM_SEQS \
 --max-num-batched-tokens $MAX_NUM_BATCHED_TOKENS \
 --tensor-parallel-size $TENSOR_PARALLEL_SIZE \
 --no-enable-prefix-caching \
 --download_dir $DOWNLOAD_DIR \
 --max-model-len $MAX_MODEL_LEN $EXTRA_ARGS \
 --async-scheduling > \"$VLLM_LOG\" 2>&1 &"

eval "VLLM_USE_V1=1 VLLM_TORCH_PROFILER_DIR=\"$PROFILE_FOLDER\" vllm serve $MODEL \
 --seed 42 \
 --disable-log-requests \
 --max-num-seqs $MAX_NUM_SEQS \
 --max-num-batched-tokens $MAX_NUM_BATCHED_TOKENS \
 --tensor-parallel-size $TENSOR_PARALLEL_SIZE \
 --no-enable-prefix-caching \
 --download_dir $DOWNLOAD_DIR \
 --max-model-len $MAX_MODEL_LEN $EXTRA_ARGS \
 --async-scheduling > \"$VLLM_LOG\" 2>&1 &"


echo "wait for 20 minutes.."
echo
for i in {1..120}; do
    # TODO: detect other type of errors.
    if grep -Fq "raise RuntimeError" "$VLLM_LOG"; then
        echo "Detected RuntimeError, exiting."
        exit 1
    elif grep -Fq "Application startup complete" "$VLLM_LOG"; then
        echo "Application started"
        break
    else
        echo "wait for 10 seconds..."
        sleep 10
    fi
done

EXPECTED_ETEL=${EXPECTED_ETEL:-3600000}
NUM_PROMPTS=${NUM_PROMPTS:-1000}
PREFIX_LEN=${PREFIX_LEN:-0}

PROFILE_FLAG=""
# Check if the PROFILE variable is numerically equal to 1
if [[ "${PROFILE:-0}" -eq 1 ]]; then
  PROFILE_FLAG="--profile"
fi

run_benchmark(){
  echo "running benchmark..."
  echo "logging to $BM_LOG"
  echo

  local request_rate="$1"
  local command_to_run
  local ARGS=()

  if [[ "$MODEL" == "BCCard/Qwen3-Coder-480B-A35B-Instruct-FP8-Dynamic" ]]; then
    command_to_run=("python3" "scripts/agent/bench_serving/benchmark_serving.py")
  else
    command_to_run=("vllm" "bench" "serve")
  fi

  # Common arguments
  ARGS+=(
    --backend vllm
    --model "$MODEL"
    --request-rate "$request_rate"
    --dataset-name "$DATASET"
    --num-prompts "$NUM_PROMPTS"
    --percentile-metrics "ttft,tpot,itl,e2el"
    --ignore-eos
    $PROFILE_FLAG
  )

  # Dataset-specific arguments
  case "$DATASET" in
    sonnet)
      ARGS+=(--dataset-path "benchmarks/sonnet_4x.txt" --sonnet-input-len "$INPUT_LEN" --sonnet-output-len "$OUTPUT_LEN")
      ;;
    random)
      ARGS+=(--random-input-len "$INPUT_LEN" --random-output-len "$OUTPUT_LEN")
      ;;
    mmlu)
      ARGS+=(--dataset-path "/workspace/dataset" --mmlu-num-shots 0 --mmlu-method "HELM")
      ;;
    mlperf)
      ARGS+=(--dataset-path "/workspace/dataset/processed-data.pkl" --mlperf-input-len "$INPUT_LEN" --max-model-len "$MAX_MODEL_LEN")
      ;;
    custom-token)
      local dataset_path="$WORKSPACE/dataset/${MODEL##*/}_${INPUT_LEN}_${OUTPUT_LEN}_tp${TENSOR_PARALLEL_SIZE}.json"
      ARGS+=(--dataset-path "$dataset_path")
      ;;
    bench-custom-token)
      local dataset_path="$WORKSPACE/dataset/${MODEL##*/}/inlen${INPUT_LEN}_outlen${OUTPUT_LEN}_prefixlen${PREFIX_LEN}.jsonl"
      echo "dataset_path: $dataset_path"
      # The original script set dataset-name to 'custom' for this case
      ARGS[7]="custom" # This replaces the --dataset-name value in the array
      ARGS+=(--dataset-path "$dataset_path" --custom-output-len "$OUTPUT_LEN" --skip-chat-template)
      ;;
    bench-custom-mm)
      DATA_DIR="$WORKSPACE/dataset/${MODEL##*/}"
      local dataset_files=($(find "$DATA_DIR" -name "inlen${INPUT_LEN}_outlen${OUTPUT_LEN}_prefixlen${PREFIX_LEN}*.jsonl"))
      if [ ${#dataset_files[@]} -ne 1 ]; then
        echo "Error: Found ${#dataset_files[@]} matching datasets in $DATA_DIR, but expected 1."
        echo "Matching files:"
        printf " - %s\n" "${dataset_files[@]}"
        exit 1
      fi
      local dataset_path="${dataset_files[0]}"
      echo "multimodal dataset_path: $dataset_path"
      ARGS[1]="openai-chat" # Replaces --backend value
      ARGS[7]="custom"      # Replaces --dataset-name value
      ARGS+=(--dataset-path "$dataset_path" --custom-output-len "$OUTPUT_LEN" --custom-skip-chat-template --endpoint /v1/chat/completions)
      ;;
    sharegpt)
      local dataset_path="$WORKSPACE/dataset/ShareGPT_V3_unfiltered_cleaned_split.json"
      if [ "$INPUT_LEN" -gt 0 ]; then
        echo "Please set INPUT_LEN to 0 for sharegpt dataset because it is not used." > "$BM_LOG" 2>&1
        exit 1
      fi
      ARGS+=(--dataset-path "$dataset_path")
      if [ "$OUTPUT_LEN" -ne 0 ]; then
        ARGS+=(--sharegpt-output-len "$OUTPUT_LEN")
      fi
      ;;
    hf)
      # Override backend for this specific case
      ARGS[1]="openai-chat" # Replaces --backend value
      ARGS+=(--dataset-path "lmarena-ai/VisionArena-Chat" --endpoint "/v1/chat/completions")
      ;;
    *)
      echo "Error: unsupported dataset '$DATASET'" > "$BM_LOG" 2>&1
      exit 1
      ;;
  esac

  # Execute the command
  "${command_to_run[@]}" "${ARGS[@]}" > "$BM_LOG" 2>&1

  echo "completed..."
  echo

  throughput=$(grep "Request throughput (req/s):" "$BM_LOG" | sed 's/[^0-9.]//g')
  p99_e2el=$(grep "P99 E2EL (ms):" "$BM_LOG" | awk '{print $NF}')
  echo "throughput: $throughput, P99 E2EL:$p99_e2el"
  echo
  echo "$throughput $p99_e2el"
}

read throughput p99_e2el < <(run_benchmark "inf" | tail -n 1)

echo "throughput:$throughput"
echo "p99_e2el:$p99_e2el"

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

  read throughput p99_e2el < <(run_benchmark "$mid" | tail -n 1)

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
