#!/bin/bash
set -e

echo "=== Cleaning up existing Ray/vLLM processes ==="
ray stop -f 2>/dev/null || true
pkill -f "vllm serve" || true
pkill -f "toy_proxy_server" || true

# --- MANUAL CONFIGURATION REQUIRED ---
PREFILL_HEAD_IP="10.128.0.34"
PREFILL_WORKER_IP="10.128.0.114"
DECODE_IP="34.29.132.33"

WORKER_ROLE=""
PROFILER_DIR="gs://shenyang-cloud/vllm"

while [[ "$#" -gt 0 ]]; do
  case $1 in
    --prefill-head)
      WORKER_ROLE="PREFILL_HEAD"
      shift
      ;;
    --prefill-worker)
      WORKER_ROLE="PREFILL_WORKER"
      shift
      ;;
    --decode)
      WORKER_ROLE="DECODE"
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

if [ -z "$WORKER_ROLE" ]; then
  echo "Usage: $0 [--prefill-head | --prefill-worker | --decode] [--profiler-dir DIR] [model_name_or_path]"
  exit 1
fi

export PYTHONPATH="$HOME/vllm:$HOME/tpu-inference:$PYTHONPATH"
export HF_HOME="/mnt/nfs/huggingface"
export USE_MOE_EP_KERNEL="0"
export MODEL_IMPL_TYPE="vllm"

PROFILER_CONFIG_JSON='{"profiler": "torch", "torch_profiler_dir": "'"${PROFILER_DIR}"'"}'
LOG_DIR="$HOME/logs"
mkdir -p "$LOG_DIR"

echo "=== Starting Bare-Metal Disaggregated vLLM for Role: $WORKER_ROLE ==="
echo "=== Loading Model: $MODEL ==="

# Global TPU Variables 
export TPU_MULTIHOST_BACKEND="ray"
export SKIP_JAX_PRECOMPILE="1"
export RAY_DEDUP_LOGS="0"

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

######## PREFILL HEAD (Host 0 of v7x-16) ########
if [ "$WORKER_ROLE" == "PREFILL_HEAD" ]; then
    echo "Starting Prefill Ray Head on Host 0..."
    export TPU_KV_TRANSFER_PORT=8200
    export TPU_SIDE_CHANNEL_PORT=8900
    export TPU_VISIBLE_CHIPS="0,1,2,3"
    export CLOUD_TPU_TASK_ID="0"
    export TPU_NODE_ID="0"

    ray start --head --port=8100 --ray-client-server-port=10000 --temp-dir=/tmp/ray_prefill

    echo "Waiting 10s for Prefill Worker to join..."
    sleep 10

    echo "Dispatching Multi-Host vllm serve to Ray..."
    RAY_ADDRESS="127.0.0.1:8100" nohup vllm serve "$MODEL" \
        --port 8400 \
        --gpu-memory-utilization 0.95 \
        --max-model-len 10240 \
        --max-num-seqs 512 \
        --max-num-batched-tokens 8192 \
        --block-size 128 \
        --no-enable-prefix-caching \
        --tensor-parallel-size 8 \
        --async-scheduling \
        --profiler-config "$PROFILER_CONFIG_JSON" \
        --kv-transfer-config '{"kv_connector":"TPUConnector","kv_connector_module_path":"tpu_inference.distributed.tpu_connector","kv_role":"kv_producer"}' \
        > "$LOG_DIR/prefill_head.txt" 2>&1 &

    wait_for_server 8400 "Prefill Server"

    echo "Starting proxy server locally..."
    nohup python3 $HOME/tpu-inference/examples/disagg/toy_proxy_server.py \
      --port 8002 \
      --prefiller-hosts "$PREFILL_HEAD_IP" \
      --prefiller-ports 8400 \
      --decoder-hosts "$DECODE_IP" \
      --decoder-ports 9400 \
      > "$LOG_DIR/proxy.txt" 2>&1 &

    wait_for_server 8002 "Proxy Server"
    echo "Prefill Head complete! Proxy is listening on port 8002."
fi

######## PREFILL WORKER (Host 1 of v7x-16) ########
if [ "$WORKER_ROLE" == "PREFILL_WORKER" ]; then
    echo "Starting Prefill Ray Worker on Host 1..."
    export TPU_KV_TRANSFER_PORT=8200
    export TPU_SIDE_CHANNEL_PORT=8900
    export TPU_VISIBLE_CHIPS="0,1,2,3"
    export CLOUD_TPU_TASK_ID="1"
    export TPU_NODE_ID="1"

    ray start --address="${PREFILL_HEAD_IP}:8100"
    
    echo "Worker successfully joined the Prefill Ray cluster! Tailing logs..."
    tail -f /tmp/ray_prefill/session_*/logs/raylet.out
fi

######## DECODE STAGE (v7x-8 Host) ########
if [ "$WORKER_ROLE" == "DECODE" ]; then
    echo "Starting Decode Ray Head on v7x-8 (Chips 0-3)..."
    export TPU_KV_TRANSFER_PORT=9200
    export TPU_SIDE_CHANNEL_PORT=8902
    export TPU_VISIBLE_CHIPS="0,1,2,3"
    export CLOUD_TPU_TASK_ID="0" 
    export TPU_NODE_ID="0"
    
    export PREFILL_HEAD_IP="${PREFILL_HEAD_IP}"

    ray start --head --port=9100 --ray-client-server-port=10001 --temp-dir=/tmp/ray_decode

    sleep 5
    echo "Dispatching vllm serve to Ray..."
    # Note: Decode only uses 4 chips because v7x-8 has 4 chips!
    RAY_ADDRESS="127.0.0.1:9100" nohup vllm serve "$MODEL" \
        --port 9400 \
        --gpu-memory-utilization 0.95 \
        --max-model-len 10240 \
        --max-num-seqs 512 \
        --max-num-batched-tokens 8192 \
        --block-size 128 \
        --no-enable-prefix-caching \
        --tensor-parallel-size 4 \
        --async-scheduling \
        --profiler-config "$PROFILER_CONFIG_JSON" \
        --kv-transfer-config '{"kv_connector":"TPUConnector","kv_connector_module_path":"tpu_inference.distributed.tpu_connector","kv_role":"kv_consumer"}' \
        > "$LOG_DIR/decode.txt" 2>&1 &
        
    wait_for_server 9400 "Decode Server"
    echo "Decode entrypoint launched in background! Monitor via: tail -f $LOG_DIR/decode.txt"
fi

cat <<EOF

>> To completely stop the bare-metal Ray nodes and vLLM processes:
ray stop -f && pkill -f "vllm serve" && pkill -f "toy_proxy"

EOF
