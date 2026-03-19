#!/bin/bash
# Copyright 2026 Google LLC
#
# A nightly benchmarking cron script to launch vLLM via run_multihost,
# execute a benchmark, extract results to an artifact, and update Spanner.

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
TOP_DIR=$(dirname "$(dirname "$SCRIPT_DIR")")
RUN_MULTIHOST_SCRIPT="${TOP_DIR}/.buildkite/scripts/run_multihost.sh"

# Auto-update the codebase before running the benchmark
echo "--- Updating codebase..."
pushd "$TOP_DIR" > /dev/null
git pull origin main || echo "Warning: Failed to pull latest changes. Continuing with current codebase."
popd > /dev/null

# Ensure essential environment variables are set for Spanner reporting
export GCP_PROJECT_ID="${GCP_PROJECT_ID:-cloud-tpu-inference-test}"
export GCP_INSTANCE_ID="${GCP_INSTANCE_ID:-vllm-bm-inst}"
export GCP_DATABASE_ID="${GCP_DATABASE_ID:-vllm-bm-runs}"
export GCP_REGION="${GCP_REGION:-southamerica-west1}"
export GCS_BUCKET="${GCS_BUCKET:-vllm-cb-storage2}"

# GCP_INSTANCE_NAME defaults to TPU_NAME
export GCP_INSTANCE_NAME="${GCP_INSTANCE_NAME:-${TPU_NAME:-$(curl -s -H "Metadata-Flavor: Google" "http://metadata.google.internal/computeMetadata/v1/instance/description" 2>/dev/null || echo "unknown-tpu")}}"
# Unique record ID for the run
RECORD_ID="$(uuidgen)"
JOB_REFERENCE="$(date +%Y%m%d_%H%M%S)"
mkdir -p "$SCRIPT_DIR/artifacts"

# ---------------------------------------------------------
# Benchmark Configuration Variables & Argument Parsing
# ---------------------------------------------------------
# Default values align with the original Qwen configuration
export RUN_TYPE="DAILY"
export MAX_NUM_SEQS="128"
export MAX_MODEL_LEN="10240"
export MAX_NUM_BATCHED_TOKENS="1024"
export TENSOR_PARALLEL_SIZE="16"
export INPUT_LEN="1024"
export OUTPUT_LEN="8192"
export NUM_PROMPTS="128"
export DATASET_NAME="random"
export TARGET_MODEL_PATH="gs://tpu-commons-ci/qwen/models--Qwen--Qwen3-Coder-480B-A35B-Instruct/snapshots/9d90cf8fca1bf7b7acca42d3fc9ae694a2194069"
export TARGET_TOKENIZER="Qwen/Qwen3-Coder-480B-A35B-Instruct"
export MODEL_NAME="Qwen3-Coder-480B-A35B-Instruct"
export DEVICE="tpu7x-16"
export CODE_HASH="a4047d4-cf732f1-"
export CREATED_BY="bm-scheduler"

# New parameters for advanced/experimental models like DeepSeek
export NEW_MODEL_DESIGN="0"
export GPU_MEMORY_UTILIZATION="0.90"
export ENABLE_EXPERT_PARALLEL=""
export ADDITIONAL_CONFIG=""
export DISABLE_SHARED_EXPERTS_STREAM="1"
export GENERATION_CONFIG=""
export VLLM_MLA_DISABLE_ENV=""
export MOE_REQUANTIZE_BLOCK_SIZE=""
export MOE_REQUANTIZE_WEIGHT_DTYPE=""
export MOE_REQUANTIZE_BLOCK_SIZE_ENV=""
export MOE_REQUANTIZE_WEIGHT_DTYPE_ENV=""
export PHASED_PROFILING_DIR=""
export PHASED_PROFILING_DIR_ENV=""
export SKIP_DB_UPLOAD="false"
export MODEL_IMPL_TYPE_ENV="MODEL_IMPL_TYPE=vllm"
export USE_UNFUSED_MEGABLOCKS_ENV=""
export HF_CONFIG=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --model-path) TARGET_MODEL_PATH="$2"; shift 2 ;;
    --model-name) MODEL_NAME="$2"; shift 2 ;;
    --tokenizer) TARGET_TOKENIZER="$2"; shift 2 ;;
    --input-len) INPUT_LEN="$2"; shift 2 ;;
    --output-len) OUTPUT_LEN="$2"; shift 2 ;;
    --tp-size) TENSOR_PARALLEL_SIZE="$2"; shift 2 ;;
    --max-seqs) MAX_NUM_SEQS="$2"; shift 2 ;;
    --max-model-len) MAX_MODEL_LEN="$2"; shift 2 ;;
    --max-batched-tokens) MAX_NUM_BATCHED_TOKENS="$2"; shift 2 ;;
    --num-prompts) NUM_PROMPTS="$2"; shift 2 ;;
    --dataset-name) DATASET_NAME="$2"; shift 2 ;;
    --run-type) RUN_TYPE="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    --code-hash) CODE_HASH="$2"; shift 2 ;;
    --created-by) CREATED_BY="$2"; shift 2 ;;
    --new-model-design) NEW_MODEL_DESIGN="$2"; shift 2 ;;
    --gpu-memory-utilization) GPU_MEMORY_UTILIZATION="$2"; shift 2 ;;
    --enable-expert-parallel) ENABLE_EXPERT_PARALLEL="--enable-expert-parallel"; shift 1 ;;
    --additional-config) ADDITIONAL_CONFIG="--additional_config='${2}'"; shift 2 ;;
    --disable-shared-experts-stream) DISABLE_SHARED_EXPERTS_STREAM="$2"; shift 2 ;;
    --generation-config) GENERATION_CONFIG="$2"; shift 2 ;;
    --vllm-mla-disable) VLLM_MLA_DISABLE_ENV="VLLM_MLA_DISABLE=${2}"; shift 2 ;;
    --moe-requantize-block-size) export MOE_REQUANTIZE_BLOCK_SIZE="$2"; MOE_REQUANTIZE_BLOCK_SIZE_ENV="MOE_REQUANTIZE_BLOCK_SIZE=$2"; shift 2 ;;
    --moe-requantize-weight-dtype) export MOE_REQUANTIZE_WEIGHT_DTYPE="$2"; MOE_REQUANTIZE_WEIGHT_DTYPE_ENV="MOE_REQUANTIZE_WEIGHT_DTYPE=$2"; shift 2 ;;
    --phased-profiling-dir) export PHASED_PROFILING_DIR="$2"; PHASED_PROFILING_DIR_ENV="PHASED_PROFILING_DIR=$2"; shift 2 ;;
    --skip-db-upload) export SKIP_DB_UPLOAD="true"; shift 1 ;;
    --model-impl-type) export MODEL_IMPL_TYPE_ENV="MODEL_IMPL_TYPE=$2"; shift 2 ;;
    --use-unfused-megablocks) export USE_UNFUSED_MEGABLOCKS_ENV="USE_UNFUSED_MEGABLOCKS=$2"; shift 2 ;;
    --hf-config) export HF_CONFIG="$2"; shift 2 ;;
    *) echo "Unknown parameter passed: $1"; exit 1 ;;
  esac
done

BENCHMARK_LOG="$SCRIPT_DIR/artifacts/${MODEL_NAME}_${INPUT_LEN}_${OUTPUT_LEN}_${RECORD_ID}_benchmark.log"
RESULT_FILE="$SCRIPT_DIR/artifacts/${MODEL_NAME}_${INPUT_LEN}_${OUTPUT_LEN}_${RECORD_ID}.result"

PRE_SERVER_CMD=""
EXTRA_SERVER_ARGS=""

if [[ -n "${GENERATION_CONFIG}" ]]; then
  echo "--- Generation config URL provided, preparing download to workspace..."
  # e.g GENERATION_CONFIG="gs://gpolovets-inference/deepseek/generation_configs/DeepSeek-R1"
  CONFIG_URL="${GENERATION_CONFIG%/*}" # gs://gpolovets-inference/deepseek/generation_configs
  CONFIG_DIR_NAME=$(basename "${CONFIG_URL}") # generation_configs
  CONFIG_FILE_NAME=$(basename "${GENERATION_CONFIG}") # DeepSeek-R1
  
  # Download to /workspace/ so it creates /workspace/generation_configs/ inside the docker container
  PRE_SERVER_CMD="if ! command -v gsutil &> /dev/null; then curl -sSL https://sdk.cloud.google.com | bash -s -- --disable-prompts > /dev/null; export PATH=\"\$PATH:/root/google-cloud-sdk/bin\"; fi && mkdir -p /workspace && gsutil -m cp -r ${CONFIG_URL} /workspace/ && "
  EXTRA_SERVER_ARGS="--generation-config /workspace/${CONFIG_DIR_NAME}/${CONFIG_FILE_NAME}"
fi

if [[ -n "${HF_CONFIG}" ]]; then
  EXTRA_SERVER_ARGS="${EXTRA_SERVER_ARGS} --hf-config=${HF_CONFIG}"
fi

# Define the commands utilizing the unified parameters
SERVER_CMD="${PRE_SERVER_CMD}VLLM_DISABLE_SHARED_EXPERTS_STREAM=${DISABLE_SHARED_EXPERTS_STREAM} \
NEW_MODEL_DESIGN=${NEW_MODEL_DESIGN} \
${VLLM_MLA_DISABLE_ENV} \
${MOE_REQUANTIZE_BLOCK_SIZE_ENV} \
${MOE_REQUANTIZE_WEIGHT_DTYPE_ENV} \
${PHASED_PROFILING_DIR_ENV} \
${USE_UNFUSED_MEGABLOCKS_ENV} \
TPU_BACKEND_TYPE=jax \
${MODEL_IMPL_TYPE_ENV} \
vllm serve \
  --seed 42 \
  --model ${TARGET_MODEL_PATH} \
  ${EXTRA_SERVER_ARGS} \
  --served-model-name ${TARGET_TOKENIZER} \
  --max-model-len=${MAX_MODEL_LEN} \
  --max-num-batched-tokens ${MAX_NUM_BATCHED_TOKENS} \
  --max-num-seqs ${MAX_NUM_SEQS} \
  --no-enable-prefix-caching \
  --tensor-parallel-size ${TENSOR_PARALLEL_SIZE} \
  --kv_cache_dtype=\"fp8\" \
  --async-scheduling \
  --gpu-memory-utilization=${GPU_MEMORY_UTILIZATION} \
  ${ENABLE_EXPERT_PARALLEL} \
  ${ADDITIONAL_CONFIG} \
  --load-format=runai_streamer \
  --trust-remote-code"

BENCHMARK_CMD="vllm bench serve \
  --model ${TARGET_TOKENIZER} \
  --dataset-name ${DATASET_NAME} \
  --random-input-len ${INPUT_LEN} \
  --random-output-len ${OUTPUT_LEN} \
  --num-prompts ${NUM_PROMPTS} \
  --ignore-eos \
  --trust-remote-code"


echo "=== Starting nightly benchmark (Record ID: $RECORD_ID) ==="
echo "Logging output to: $BENCHMARK_LOG"

# 1. Run the benchmark using multihost launcher script
if ! bash "$RUN_MULTIHOST_SCRIPT" "$SERVER_CMD" "$BENCHMARK_CMD" > "$BENCHMARK_LOG" 2>&1; then
  echo "Benchmarking failed. See log: $BENCHMARK_LOG"
  echo "Status=FAILED" > "$RESULT_FILE"
else
  echo "Benchmarking completed. Parsing results..."
  
  # 2. Parse benchmark log and generate key-value .result file
  python3 -c '
import sys, re

# Mapping of what vllm prints vs what Spanner column expects
METRIC_MAPPING = {
    "Request throughput": "Throughput",
    "Output token throughput": "OutputTokenThroughput",
    "Total token throughput": "TotalTokenThroughput",
    "Median TTFT": "MedianTTFT",
    "P99 TTFT": "P99TTFT",
    "Median TPOT": "MedianTPOT",
    "P99 TPOT": "P99TPOT",
    "Median ITL": "MedianITL",
    "P99 ITL": "P99ITL"
}

try:
    with open(sys.argv[1], "r") as f:
        lines = f.readlines()
except FileNotFoundError:
    lines = []

results = {}
in_results = False
for line in lines:
    line = line.strip()
    if line.startswith("============ Serving Benchmark Result ============"):
        in_results = True
        continue
    if line.startswith("==================================================") and in_results:
        break
    if in_results and ":" in line:
        key, val = line.split(":", 1)
        val = val.strip()
        
        # Remove units like (ms) or (tok/s) or (excl. 1st token)
        clean_key = re.sub(r"\(.*?\)", "", key).strip()
        
        if clean_key in METRIC_MAPPING:
            results[METRIC_MAPPING[clean_key]] = val

with open(sys.argv[2], "w") as out:
    for k, v in results.items():
        out.write(f"{k}={v}\n")
  ' "$BENCHMARK_LOG" "$RESULT_FILE"

  # Append static Spanner schema parameters using the bash environment variables
  cat <<EOF >> "$RESULT_FILE"
RunType=${RUN_TYPE}
MaxNumSeqs=${MAX_NUM_SEQS}
MaxNumBatchedTokens=${MAX_NUM_BATCHED_TOKENS}
TensorParallelSize=${TENSOR_PARALLEL_SIZE}
MaxModelLen=${MAX_MODEL_LEN}
Dataset=${DATASET_NAME}
CreatedBy=${CREATED_BY}
InputLen=${INPUT_LEN}
OutputLen=${OUTPUT_LEN}
Device=${DEVICE}
NumPrompts=${NUM_PROMPTS}
CodeHash=${CODE_HASH}
Model=${MODEL_NAME}
JobReference=${JOB_REFERENCE}
EOF

fi

# Upload vllm_serve.log to GCS
LOG_GCS_URI="gs://tpu-commons-ci/logs/${MODEL_NAME}_${INPUT_LEN}_${OUTPUT_LEN}_${JOB_REFERENCE}_vllm_serve.log"
if [ -f "/tmp/vllm_serve.log" ]; then
  echo "Uploading vllm_serve.log to $LOG_GCS_URI"
  gsutil cp /tmp/vllm_serve.log "$LOG_GCS_URI" || echo "Warning: Failed to upload vllm_serve.log"
else
  echo "Warning: /tmp/vllm_serve.log not found, skipping upload."
fi

if [[ "${SKIP_DB_UPLOAD}" == "true" ]]; then
  echo "=== Skipping Spanner DB Upload (--skip-db-upload specified) ==="
  echo "=== Nightly benchmark script completed successfully ==="
  exit 0
fi

# 3. Report results to Spanner (mimicking bm-infra/scripts/agent/report_result.sh but inserting instead of updating)
keys="RecordId, "
vals="'${RECORD_ID}', "
while IFS='=' read -r key value; do
  if [[ -n "$key" && -n "$value" ]]; then
    keys+="${key}, "
    if [[ "$key" == "AccuracyMetrics" ]]; then
      vals+="JSON '${value}', "
    elif [[ "$value" =~ ^[0-9.]+$ ]]; then
      vals+="${value}, "
    else
      vals+="'${value}', "
    fi
  fi
done < "$RESULT_FILE"

if [ "$keys" == "RecordId, " ]; then
  echo "Result file was empty or parsing failed. Marking status as FAILED."
  keys+="Status, RunBy, LastUpdate, CreatedTime"
  vals+="'FAILED', '${GCP_INSTANCE_NAME}', CURRENT_TIMESTAMP(), CURRENT_TIMESTAMP()"
else
  keys+="Status, RunBy, LastUpdate, CreatedTime"
  vals+="'COMPLETED', '${GCP_INSTANCE_NAME}', CURRENT_TIMESTAMP(), CURRENT_TIMESTAMP()"
fi

SQL="INSERT INTO RunRecord (${keys}) VALUES (${vals});"

echo "Executing SQL for Spanner update:"
echo "$SQL"

if ! gcloud spanner databases execute-sql "$GCP_DATABASE_ID" \
  --project="$GCP_PROJECT_ID" \
  --instance="$GCP_INSTANCE_ID" \
  --sql="$SQL"; then
  echo "Failed to update Spanner record!"
  exit 1
fi

echo "=== Nightly benchmark script completed successfully ==="
exit 0
