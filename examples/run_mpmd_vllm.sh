#!/usr/bin/env bash
# Launch 4 independent `vllm serve` instances on a single v7 host
# (4 chips / 8 cores), MPMD-style.
#
# This replaces `--data-parallel-size 4` with 4 manually-launched processes.
# Each instance is its own standalone single-process JAX cluster pinned to
# 1 physical chip (2 cores) and runs `--tensor-parallel-size 2`. There are
# no collectives between instances -- they are 4 independent servers.
#
#   ./examples/run_mpmd_vllm.sh
#
# Each instance serves HTTP on its own port (8000..8003) and writes its log
# to ${LOG_DIR}/instance_<i>.log. Ctrl-C stops all instances.
#
# Overrides:
#   MODEL=...    model to serve   (default google/gemma-4-31B-it)
#   LOG_DIR=...  log directory    (default /tmp/mpmd_vllm)

set -u

MODEL="${MODEL:-google/gemma-4-31B-it}"
LOG_DIR="${LOG_DIR:-/tmp/mpmd_vllm}"
NUM_INSTANCES=4
BASE_HTTP_PORT=8000   # vLLM OpenAI server port, one per instance
BASE_TPU_PORT=8476    # libtpu gRPC port, one per instance

mkdir -p "${LOG_DIR}"
pids=()

for i in $(seq 0 $((NUM_INSTANCES - 1))); do
  http_port=$((BASE_HTTP_PORT + i))
  tpu_port=$((BASE_TPU_PORT + i))
  log="${LOG_DIR}/instance_${i}.log"

  echo "[launcher] instance ${i}: chip ${i}, http :${http_port}, log ${log}"

  # Model env vars + per-instance TPU chip isolation. Each instance is a
  # standalone 1-process JAX cluster owning exactly chip ${i} (2 cores ->
  # 2 JAX devices -> tensor-parallel-size 2).
  USE_BATCHED_RPA_KERNEL=1 \
  MODEL_IMPL_TYPE=flax_nnx \
  TPU_VISIBLE_CHIPS="${i}" \
  TPU_CHIPS_PER_PROCESS_BOUNDS=1,1,1 \
  TPU_PROCESS_BOUNDS=1,1,1 \
  TPU_PROCESS_PORT="${tpu_port}" \
  TPU_PROCESS_ADDRESSES="localhost:${tpu_port}" \
  CLOUD_TPU_TASK_ID=0 \
    vllm serve "${MODEL}" \
      --port="${http_port}" \
      --max-model-len=1524 \
      --max-num-seqs=256 \
      --tensor-parallel-size 2 \
      --max-num-batched-tokens 16384 \
      --no-enable-prefix-caching \
      --additional_config='{"quantization": { "qwix": { "rules": [{ "module_path": ".*", "weight_qtype": "float8_e4m3fn", "act_qtype": "float8_e4m3fn"}]}}}' \
      --kv-cache-dtype=fp8 \
      --gpu-memory-utilization=0.9 \
      --async-scheduling \
      --limit-mm-per-prompt '{"image": 0, "video": 0, "audio": 0}' \
      --block-size=256 \
      >"${log}" 2>&1 &
  pids+=($!)
done

echo "[launcher] started ${NUM_INSTANCES} instances: pids ${pids[*]}"
echo "[launcher] follow logs:  tail -f ${LOG_DIR}/instance_*.log"

# Stop every instance on Ctrl-C / TERM.
trap 'echo "[launcher] stopping all instances..."; kill "${pids[@]}" 2>/dev/null; exit 130' INT TERM

rc=0
for i in "${!pids[@]}"; do
  wait "${pids[$i]}" || rc=$?
  echo "[launcher] instance ${i} (pid ${pids[$i]}) exited"
done
exit "${rc}"
