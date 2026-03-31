#!/bin/bash
# Minimal Bare-Metal TPU Disaggregated Serving Script (1x Host Prefill / 1x Host Decode)
set -e

echo "=== Cleaning up existing Ray/vLLM processes ==="
ray stop -f 2>/dev/null || true
pkill -f "vllm serve" || true
pkill -f "toy_proxy_server" || true

# --- MANUAL CONFIGURATION REQUIRED ---
PREFILL_IPS=("10.128.0.34")
DECODE_IPS=("10.128.0.114")

HOST_IPS=("${PREFILL_IPS[@]}" "${DECODE_IPS[@]}")
HEAD_IP=${PREFILL_IPS[0]}

WORKER_ID=-1
PROFILER_DIR="gs://shenyang-cloud/vllm"

while [[ "$#" -gt 0 ]]; do
  case $1 in
    -p|--prefill)
      WORKER_ID=0
      shift
      ;;
    -d|--decode)
      WORKER_ID=1
      shift
      ;;
    --profiler-dir)
      PROFILER_DIR="$2"
      shift 2
      ;;
    *)
      if [[ -z "$MODEL_ARG" ]]; then
        MODEL_ARG="$1"
      else
        echo "Unknown parameter passed: $1"
        exit 1
      fi
      shift
      ;;
  esac
done

MODEL=${MODEL_ARG:-"Qwen/Qwen3-0.6B"}

if [ "$WORKER_ID" -eq -1 ]; then
  echo "Usage: $0 [-p | --prefill OR -d | --decode] [--profiler-dir DIR] [model_name_or_path]"
  echo "Example: $0 -p --profiler-dir gs://shenyang-cloud/vllm2 Qwen/Qwen2.5-1.5B"
  exit 1
fi

export PYTHONPATH="$HOME/vllm:$HOME/tpu-inference:$PYTHONPATH"
export HF_HOME="/mnt/nfs/huggingface"

export USE_MOE_EP_KERNEL="0"
export MODEL_IMPL_TYPE="vllm"

# Profiler Config
PROFILER_CONFIG_JSON='{"profiler": "torch", "torch_profiler_dir": "'"${PROFILER_DIR}"'"}'

LOG_DIR="$HOME/logs"
mkdir -p "$LOG_DIR"

echo "=== Starting Bare-Metal Disaggregated vLLM on WORKER_ID: $WORKER_ID ==="
echo "=== Loading Model: $MODEL ==="

# Global TPU Variables for independent 4-chip Ray Clusters
export TPU_MULTIHOST_BACKEND="ray"
export SKIP_JAX_PRECOMPILE="0"
export RAY_DEDUP_LOGS="0"

export TPU_PROCESS_BOUNDS="1,1,1"
export TPU_VISIBLE_CHIPS="0,1,2,3,4,5,6,7"
export CLOUD_TPU_TASK_ID="0" 
export TPU_NODE_ID="0"
export TPU_PROCESS_ADDRESSES="${HOST_IPS[$WORKER_ID]}:8476"
export TPU_PROCESS_PORT="8476"

wait_for_server() {
  local port=$1
  local name=$2
  local target_host=${3:-127.0.0.1}
  echo "Waiting for $name on $target_host:$port to become healthy (this can take 5-10 minutes)..."
  while true; do
    if curl -s "${target_host}:${port}/health" > /dev/null; then
      echo "=== $name is healthy on $target_host:$port ==="
      break
    fi
    sleep 10
  done
}

######## PREFILL STAGE (Host 0 Only) ########
if [ "$WORKER_ID" -eq 0 ]; then
    echo "Starting Prefill Ray Head on Host 0 (Chips 0-3)..."
    export TPU_KV_TRANSFER_PORT=8200
    export TPU_SIDE_CHANNEL_PORT=8900
    # Ray starts processes from port 10002, use 10000 for prefill client and 10001 for decode.
    ray start --head --port=8100 --ray-client-server-port=10000 --temp-dir=/tmp/ray_prefill

    sleep 5
    echo "Dispatching vllm serve to Ray..."
    RAY_ADDRESS="127.0.0.1:8100" nohup vllm serve "$MODEL" \
        --port 8400 \
        --gpu-memory-utilization 0.95 \
        --max-model-len 10240 \
        --max-num-seqs 512 \
        --max-num-batched-tokens 8192 \
        --block-size 128 \
        --no-enable-prefix-caching \
        --tensor-parallel-size 8 \
        --no-async-scheduling \
        --profiler-config "$PROFILER_CONFIG_JSON" \
        --kv-transfer-config '{"kv_connector":"TPUConnector","kv_connector_module_path":"tpu_inference.distributed.tpu_connector","kv_role":"kv_producer"}' \
        > "$LOG_DIR/prefill.txt" 2>&1 &

    wait_for_server 8400 "Prefill Server"

    # Wait for Decode server explicitly from Host 0 before routing requests to it via proxy
    wait_for_server 9400 "Decode Server" "${DECODE_IPS[0]}"

    echo "Starting proxy server locally..."
    nohup python3 $HOME/tpu-inference/examples/disagg/toy_proxy_server.py \
      --port 8002 \
      --prefiller-hosts "${PREFILL_IPS[@]}" \
      --prefiller-ports 8400 \
      --decoder-hosts "${DECODE_IPS[@]}" \
      --decoder-ports 9400 \
      > "$LOG_DIR/proxy.txt" 2>&1 &

    wait_for_server 8002 "Proxy Server"
    
    LOG_FILE="$LOG_DIR/benchmark_multi_host.txt"
    echo "--- Running Disagg Benchmark ---" > "$LOG_FILE"
    
    echo "Prefill entrypoint complete! Running benchmarks..."
    set -x
    vllm bench serve \
      --model="$MODEL" \
      --num-warmups=3 \
      --dataset-name=random \
      --random-input-len=4096 \
      --random-output-len=128 \
      --num-prompts=30 \
      --ignore-eos \
      --host=localhost \
      --port 8002 \
      --request-rate 30 \
      >> "$LOG_FILE" 2>&1

    echo -e "\n\n--- Running Non-Disagg Benchmark ---" >> "$LOG_FILE"
    # non-disagg running against decode node directly
    vllm bench serve \
      --model="$MODEL" \
      --num-warmups=3 \
      --dataset-name=random \
      --random-input-len=4096 \
      --random-output-len=128 \
      --num-prompts=30 \
      --ignore-eos \
      --host="${DECODE_IPS[0]}" \
      --port 9400 \
      --request-rate 30 \
      >> "$LOG_FILE" 2>&1
    set +x
    echo "Benchmarks finished. Results available in $LOG_FILE"
fi

######## DECODE STAGE (Host 1 Only) ########
if [ "$WORKER_ID" -eq 1 ]; then
    echo "Starting Decode Ray Head on Host 1 (Chips 0-3)..."
    export TPU_KV_TRANSFER_PORT=9200
    export TPU_SIDE_CHANNEL_PORT=8902
    
    # Expose the Prefill head IP if the cluster natively pulls discovery from the environment
    export PREFILL_HEAD_IP="${PREFILL_IPS[0]}"

    ray start --head --port=9100 --ray-client-server-port=10001 --temp-dir=/tmp/ray_decode

    sleep 5
    echo "Dispatching vllm serve to Ray..."
    RAY_ADDRESS="127.0.0.1:9100" nohup vllm serve "$MODEL" \
        --port 9400 \
        --gpu-memory-utilization 0.95 \
        --max-model-len 10240 \
        --max-num-seqs 512 \
        --max-num-batched-tokens 8192 \
        --block-size 128 \
        --no-enable-prefix-caching \
        --tensor-parallel-size 8 \
        --no-async-scheduling \
        --profiler-config "$PROFILER_CONFIG_JSON" \
        --kv-transfer-config '{"kv_connector":"TPUConnector","kv_connector_module_path":"tpu_inference.distributed.tpu_connector","kv_role":"kv_consumer"}' \
        > "$LOG_DIR/decode.txt" 2>&1 &
        
    wait_for_server 9400 "Decode Server"
    echo "Decode entrypoint launched in background! You can safely exit this shell."
    echo "Monitor outputs via: tail -f $LOG_DIR/decode.txt"
fi

cat <<EOF

>> To completely stop the bare-metal Ray nodes and vLLM processes:
ray stop -f && pkill -f "vllm serve" && pkill -f "toy_proxy"

>> Complete End-to-End Execution Sample (from proxy server host):
curl -s http://127.0.0.1:8002/v1/completions \\
  -H "Content-Type: application/json" \\
  -d '{"model": "${MODEL}", "prompt": "Your pet name is", "max_tokens": 15, "temperature": 0}'

EOF
