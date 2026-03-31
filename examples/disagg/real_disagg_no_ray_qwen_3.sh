#!/bin/bash
set -e

echo "=== Cleaning up existing Ray/vLLM processes ==="
pkill -f "vllm" || true
pkill -f "toy_proxy_server" || true

# --- MANUAL CONFIGURATION REQUIRED ---
PREFILL_IPS=("10.128.0.34")
DECODE_IPS=("10.128.0.114")
PROFILER_DIR=gs://shenyang-cloud/vllm
PROFILER_CONFIG_JSON='{"profiler": "torch", "torch_profiler_dir": "'"${PROFILER_DIR}"'"}'
export HF_HOME="/mnt/nfs/huggingface"

# Read target from CLI
TARGET_ROLE=$1
MODEL=${2:-"Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8"}

if [ "$TARGET_ROLE" != "-p" ] && [ "$TARGET_ROLE" != "-d" ]; then
    echo "Usage: $0 [-p (Prefill)] | [-d (Decode)] [MODEL_PATH]"
    exit 1
fi

# Internal host mapping list mimicking Ray discovery
HOST_IPS=("${PREFILL_IPS[@]}" "${DECODE_IPS[@]}")
LOCAL_IP=$(hostname -I | awk '{print $1}')
WORKER_ID=-1

for i in "${!HOST_IPS[@]}"; do
    if [ "${HOST_IPS[$i]}" == "$LOCAL_IP" ]; then
        WORKER_ID=$i
        break
    fi
done

if [ "$WORKER_ID" -eq -1 ]; then
    echo "Error: Local IP $LOCAL_IP not found in node lists!"
    exit 1
fi

LOG_DIR="$HOME/logs"
mkdir -p "$LOG_DIR"
echo "=== Successfully mapped $LOCAL_IP as Worker $WORKER_ID ==="
echo "=== Loading Model: $MODEL ==="

export SKIP_JAX_PRECOMPILE="0"  # DO NOT CHANGE to have sane traces.
export TPU_PROCESS_BOUNDS="1,1,1"
export TPU_VISIBLE_CHIPS="0,1,2,3,4,5,6,7"
export CLOUD_TPU_TASK_ID="0" 
export TPU_NODE_ID="0"
export TPU_PROCESS_ADDRESSES="${HOST_IPS[$WORKER_ID]}:8476"
export TPU_PROCESS_PORT="8476"
# To avoid MapDmaBuffer.
export TPU_PREMAPPED_BUFFER_SIZE=68719476736
export TPU_PREMAPPED_BUFFER_TRANSFER_THRESHOLD_BYTES=68719476736
export JAX_PJRT_CLIENT_CREATE_OPTIONS="pinned_host_allocation_mode:recycle"

check_server_health() {
    local ip=$1
    local port=$2
    local label=$3
    echo "Waiting for ${label} to initialize at ${ip}:${port}..."
    sleep 30
    while ! curl -s http://${ip}:${port}/v1/models > /dev/null; do
        sleep 5
    done
    echo "${label} successfully initialized."
}

# The serving flags are tuned towards Qwen/Qwen3.5-397B-A17B-FP8 (b/493286697).
######## PREFILL STAGE (Host 0 Only) ########
if [ "$WORKER_ID" -eq 0 ]; then
    echo "Starting Prefill Server on Host 0..."
    export TPU_KV_TRANSFER_PORT=8300
    export TPU_SIDE_CHANNEL_PORT=9000
    export VLLM_HOST_IP="${PREFILL_IPS[0]}"
    
    echo "Dispatching standalone vllm serve locally..."
    vllm serve "$MODEL" \
        --host "${PREFILL_IPS[0]}" \
        --port 8400 \
        --gpu-memory-utilization 0.85 \
        --max-model-len 10240 \
        --max-num-seqs 512 \
        --max-num-batched-tokens 16384 \
        --block-size 256 \
        --no-enable-prefix-caching \
        --tensor-parallel-size 8 \
        --async-scheduling \
        --profiler-config "$PROFILER_CONFIG_JSON" \
        --kv-transfer-config '{"kv_connector":"TPUConnector","kv_connector_module_path":"tpu_inference.distributed.tpu_connector","kv_role":"kv_producer"}' \
        > "$LOG_DIR/prefill.txt" 2>&1 &


    # Wait for the server to be completely healthy
    check_server_health "${PREFILL_IPS[0]}" 8400 "Prefill vLLM Server"
    check_server_health "${DECODE_IPS[0]}" 9400 "Decode vLLM Server"
    
    echo "Starting proxy server locally..."
    nohup python3 $HOME/tpu-inference/examples/disagg/toy_proxy_server.py \
      --port 8002 \
      --prefiller-hosts "${PREFILL_IPS[@]}" \
      --prefiller-ports 8400 \
      --decoder-hosts "${DECODE_IPS[@]}" \
      --decoder-ports 9400 \
      > "$LOG_DIR/proxy.txt" 2>&1 &

    check_server_health "localhost" 8002 "Proxy Server"
    
    LOG_FILE="$LOG_DIR/benchmark_single_host.txt"
    echo "--- Running Disagg Benchmark ---" > "$LOG_FILE"
    
    echo "Prefill entrypoint complete! Running benchmarks..."
    set -x
    vllm bench serve \
      --model="$MODEL" \
      --num-warmups=3 \
      --dataset-name=random \
      --random-input-len=8192 \
      --random-output-len=1024 \
      --num-prompts=256 \
      --ignore-eos \
      --host=localhost \
      --port 8002 \
      --request-rate 256 \
      >> "$LOG_FILE" 2>&1
    set +x
fi

######## DECODE STAGE (Host 1 Only) ########
if [ "$WORKER_ID" -eq 1 ]; then
    echo "Starting Decode Server on Host 1..."
    export TPU_KV_TRANSFER_PORT=9300
    export TPU_SIDE_CHANNEL_PORT=9002
    
    # Bypassing Ray discovery and declaring Prefill manually for the custom tpu_connector.
    export TPU_KV_PRODUCER_IP="${PREFILL_IPS[0]}"
    # We explicitly tell Decode to bind to its OWN TRUE internal IP.
    export VLLM_HOST_IP="${DECODE_IPS[0]}"
    
    echo "Dispatching standalone vllm serve locally..."
    # Large max-num-batched-tokens for Qwen 3.5.
    vllm serve "$MODEL" \
        --host "${DECODE_IPS[0]}" \
        --port 9400 \
        --gpu-memory-utilization 0.85 \
        --max-model-len 10240 \
        --max-num-seqs 512 \
        --max-num-batched-tokens 16384 \
        --block-size 256 \
        --no-enable-prefix-caching \
        --tensor-parallel-size 8 \
        --async-scheduling \
        --profiler-config "$PROFILER_CONFIG_JSON" \
        --kv-transfer-config '{"kv_connector":"TPUConnector","kv_connector_module_path":"tpu_inference.distributed.tpu_connector","kv_role":"kv_consumer"}' \
        > "$LOG_DIR/decode.txt" 2>&1 &
        
    # Wait for Decode server explicitly on Host 1
    check_server_health "${DECODE_IPS[0]}" 9400 "Decode Server"
    
    echo "Decode endpoint properly running."
    echo "Monitor outputs via: tail -f $LOG_DIR/decode.txt"
fi

echo "Disaggregated backend successfully launched!"
echo "Holding script open to keep detached background processes alive indefinitely..."
wait
