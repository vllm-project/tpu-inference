#!/bin/bash
# Automated deploy + benchmark script for Pathways vLLM serving.
# Safe for parallel runs: each DP size gets its own JobSet name, GCS tarball,
# local port, and temp file, so multiple instances can run simultaneously.
#
# Usage: ./deploy.sh --data-parallel-size <1|2|4|8|16|32|64|128> [--delete-prev true|false] [--no-benchmark]
#
# Flags:
#   --data-parallel-size  Required. The DP size to deploy.
#   --delete-prev         Optional (default: true). If false, skip packaging/uploading/
#                         deploying and pick up the already-running job for benchmarking.
#   --no-benchmark        Optional. If set, deploy and wait for the server to be ready
#                         but stop before running any benchmarks (port-forward stays up).
#   1. Looks up TPU topology / completions / parallelism for the requested DP size
#   2. Tars the local tpu-inference repo and uploads to GCS
#   3. Renders the YAML template and saves it to benchmark_artifacts/
#   4. Deploys the JobSet and waits for the vLLM API server to be ready
#   5. Port-forwards a per-DP-size local port and runs the benchmark twice:
#        - Run 1: warmup / profiling run (runs to completion)
#        - Run 2: real benchmark (waits for completion, parses results to CSV)
#   6. Saves all artifacts under benchmark_artifacts/dp_<N>/<datetime>/
set -euo pipefail

# ─── DP-size → config mapping ───────────────────────────────────────────────
# Each entry: TOPOLOGY  COMPLETIONS_PARALLELISM
# TOPOLOGY is the GKE slice topology (4m x 4n x 4k).
# COMPLETIONS_PARALLELISM = total_chips / 4 chips_per_pod.
declare -A DP_TOPOLOGY DP_COMPLETIONS
DP_TOPOLOGY[1]="4x4x4"      ; DP_COMPLETIONS[1]=16    # 64 chips
DP_TOPOLOGY[2]="4x4x4"      ; DP_COMPLETIONS[2]=16    # 64 chips
DP_TOPOLOGY[4]="4x4x4"      ; DP_COMPLETIONS[4]=16    # 64 chips
DP_TOPOLOGY[8]="4x4x4"      ; DP_COMPLETIONS[8]=16    # 64 chips
DP_TOPOLOGY[16]="4x4x4"      ; DP_COMPLETIONS[16]=16    # 64 chips
DP_TOPOLOGY[32]="4x4x8"      ; DP_COMPLETIONS[32]=32    # 128 chips
DP_TOPOLOGY[64]="4x8x8"      ; DP_COMPLETIONS[64]=64    # 256 chips
DP_TOPOLOGY[128]="8x8x8"     ; DP_COMPLETIONS[128]=128  # 512 chips
# ─────────────────────────────────────────────────────────────────────────────

usage() {
  echo "Usage: $0 --data-parallel-size <1|2|4|8|16|32|64|128> [--tensor-parallel-size <N>] [--enable-attn-dp] [--delete-prev true|false] [--no-benchmark]"
  exit 1
}

# ─── Parse arguments ────────────────────────────────────────────────────────
DATA_PARALLEL_SIZE=""
TENSOR_PARALLEL_SIZE=8
ENABLE_ATTN_DP="false"
DELETE_PREV="true"
RUN_BENCHMARK="true"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --data-parallel-size|--data_parallel_size)
      DATA_PARALLEL_SIZE="$2"; shift 2 ;;
    --tensor-parallel-size|--tensor_parallel_size)
      TENSOR_PARALLEL_SIZE="$2"; shift 2 ;;
    --enable-attn-dp|--enable_attn_dp)
      ENABLE_ATTN_DP="true"; shift ;;
    --delete-prev|--delete_prev)
      DELETE_PREV="$2"; shift 2 ;;
    --no-benchmark|--no_benchmark)
      RUN_BENCHMARK="false"; shift ;;
    -h|--help) usage ;;
    *) echo "Unknown flag: $1"; usage ;;
  esac
done

if [[ -z "${DATA_PARALLEL_SIZE}" ]]; then
  echo "ERROR: --data-parallel-size is required."
  usage
fi

if [[ -z "${DP_TOPOLOGY[${DATA_PARALLEL_SIZE}]+_}" ]]; then
  echo "ERROR: Unsupported --data-parallel-size=${DATA_PARALLEL_SIZE}. Supported: ${!DP_TOPOLOGY[*]}"
  exit 1
fi

TOPOLOGY="${DP_TOPOLOGY[${DATA_PARALLEL_SIZE}]}"
COMPLETIONS_PARALLELISM="${DP_COMPLETIONS[${DATA_PARALLEL_SIZE}]}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
YAML_TEMPLATE="${SCRIPT_DIR}/pathways_job_new_cluster.yaml"

# Total DP size: DP if not using attn_dp, DP*(TP/4) if using attn_dp
if [[ "${ENABLE_ATTN_DP}" == "true" ]]; then
  TOTAL_DP_SIZE=$(( DATA_PARALLEL_SIZE * TENSOR_PARALLEL_SIZE / 4 ))
else
  TOTAL_DP_SIZE="${DATA_PARALLEL_SIZE}"
fi

NUM_PROMPTS=$(( TOTAL_DP_SIZE * 256 ))

# Build experiment suffix for naming artifacts/jobs
if [[ "${ENABLE_ATTN_DP}" == "true" ]]; then
  EXPERIMENT_SUFFIX="dp${DATA_PARALLEL_SIZE}-tp${TENSOR_PARALLEL_SIZE}-adp"
else
  EXPERIMENT_SUFFIX="dp${DATA_PARALLEL_SIZE}-tp${TENSOR_PARALLEL_SIZE}"
fi

GCS_BUCKET="gs://wenxindong-multipod-dev"
GCS_PATCH_PATH="${GCS_BUCKET}/patches/tpu-inference-${EXPERIMENT_SUFFIX}.tar.gz"

# Keep the JobSet name short to avoid exceeding the 49-character limit
# on Kueue Slice resource names (pattern: default-job-<name>-worker-0-<hash>-main-0).
JOBSET_NAME="w-${EXPERIMENT_SUFFIX}"
KUBE_CONTEXT="gke_cloud-tpu-multipod-dev_us-central1_bodaborg-super-alpha-cluster"

# Per-DP-size local port so parallel runs don't collide
declare -A DP_LOCAL_PORT
DP_LOCAL_PORT[1]=8001
DP_LOCAL_PORT[2]=8002
DP_LOCAL_PORT[4]=8004
DP_LOCAL_PORT[8]=8008
DP_LOCAL_PORT[16]=8016
DP_LOCAL_PORT[32]=8032
DP_LOCAL_PORT[64]=8064
DP_LOCAL_PORT[128]=8128
LOCAL_PORT="${DP_LOCAL_PORT[${DATA_PARALLEL_SIZE}]}"

# ─── Artifact directory ─────────────────────────────────────────────────────
DATETIME="$(date +%Y%m%d_%H%M%S)"
ARTIFACT_DIR="${REPO_DIR}/benchmark_artifacts/${EXPERIMENT_SUFFIX}/${DATETIME}"
mkdir -p "${ARTIFACT_DIR}"

# Convenience symlink to latest run for this experiment
ln -sfn "${DATETIME}" "${REPO_DIR}/benchmark_artifacts/${EXPERIMENT_SUFFIX}/latest"

PROFILING_DIR_SUFFIX="${EXPERIMENT_SUFFIX}/${DATETIME}"

echo "=== Configuration ==="
echo "  DATA_PARALLEL_SIZE      = ${DATA_PARALLEL_SIZE}"
echo "  TENSOR_PARALLEL_SIZE    = ${TENSOR_PARALLEL_SIZE}"
echo "  ENABLE_ATTN_DP          = ${ENABLE_ATTN_DP}"
echo "  TOTAL_DP_SIZE           = ${TOTAL_DP_SIZE}"
echo "  TOPOLOGY                = ${TOPOLOGY}"
echo "  COMPLETIONS_PARALLELISM = ${COMPLETIONS_PARALLELISM}"
echo "  INSTANCE_TYPE           = tpu7x:${TOPOLOGY}"
echo "  PROFILING_DIR_SUFFIX    = ${PROFILING_DIR_SUFFIX}"
echo "  JOBSET_NAME             = ${JOBSET_NAME}"
echo "  LOCAL_PORT              = ${LOCAL_PORT}"
echo "  NUM_PROMPTS             = ${NUM_PROMPTS}"
echo "  ARTIFACT_DIR            = ${ARTIFACT_DIR}"
echo ""

# ─── Build additional vllm args ─────────────────────────────────────────────
ADDITIONAL_ARGS=""
if [[ "${ENABLE_ATTN_DP}" == "true" ]]; then
  ADDITIONAL_ARGS="--additional_config='{\"sharding\":{\"sharding_strategy\": {\"enable_dp_attention\":1}}}'"
fi

# ─── Save benchmarking command ──────────────────────────────────────────────
BENCH_CMD="vllm bench serve \\
  --base-url=http://localhost:${LOCAL_PORT} \\
  --dataset-name=random \\
  --random-input-len=512 \\
  --random-output-len=4096 \\
  --num-prompts=${NUM_PROMPTS} \\
  --max-concurrency=${NUM_PROMPTS} \\
  --ignore-eos \\
  --model=Qwen/Qwen3-235B-A22B"

echo "${BENCH_CMD}" > "${REPO_DIR}/benchmark_artifacts/${EXPERIMENT_SUFFIX}/benchmarking_command.txt"
echo "  Saved benchmarking command to benchmark_artifacts/${EXPERIMENT_SUFFIX}/benchmarking_command.txt"

if [[ "${DELETE_PREV}" == "true" ]]; then
  # ─── Render the YAML template ─────────────────────────────────────────────
  echo "=== Rendering YAML template ==="
  GCS_TARBALL_NAME="tpu-inference-${EXPERIMENT_SUFFIX}.tar.gz"

  # Build the additional_config line for the YAML (or remove the placeholder)
  if [[ "${ENABLE_ATTN_DP}" == "true" ]]; then
    YAML_ADDITIONAL_CONFIG="--additional_config='{\"sharding\":{\"sharding_strategy\": {\"enable_dp_attention\":1}}}'"
  else
    YAML_ADDITIONAL_CONFIG=""
  fi

  sed \
    -e "s/__JOBSET_NAME__/${JOBSET_NAME}/g" \
    -e "s/__GCS_TARBALL_NAME__/${GCS_TARBALL_NAME}/g" \
    -e "s/__DATA_PARALLEL_SIZE__/${DATA_PARALLEL_SIZE}/g" \
    -e "s/__TENSOR_PARALLEL_SIZE__/${TENSOR_PARALLEL_SIZE}/g" \
    -e "s/__TOPOLOGY__/${TOPOLOGY}/g" \
    -e "s/__COMPLETIONS_PARALLELISM__/${COMPLETIONS_PARALLELISM}/g" \
    -e "s|__PROFILING_DIR_SUFFIX__|${PROFILING_DIR_SUFFIX}|g" \
    -e "s|__ADDITIONAL_CONFIG__|${YAML_ADDITIONAL_CONFIG}|g" \
    "${YAML_TEMPLATE}" > "${ARTIFACT_DIR}/pathways_job_rendered.yaml"
  echo "  Saved to ${ARTIFACT_DIR}/pathways_job_rendered.yaml"

  echo "=== Packaging tpu-inference ==="
  cd "${REPO_DIR}"
  tar czf /tmp/tpu-inference-${EXPERIMENT_SUFFIX}.tar.gz \
    --exclude='.git' \
    --exclude='__pycache__' \
    --exclude='*.egg-info' \
    --exclude='.tox' \
    --exclude='.nox' \
    --exclude='dist' \
    --exclude='build' \
    --exclude='benchmark_artifacts' \
    .

  echo "=== Uploading to ${GCS_PATCH_PATH} ==="
  gcloud storage cp /tmp/tpu-inference-${EXPERIMENT_SUFFIX}.tar.gz "${GCS_PATCH_PATH}"

  echo "=== Deleting existing JobSet '${JOBSET_NAME}' (if any) ==="
  kubectl --context="${KUBE_CONTEXT}" delete jobset "${JOBSET_NAME}" --ignore-not-found

  echo "=== Applying rendered YAML ==="
  kubectl --context="${KUBE_CONTEXT}" apply -f "${ARTIFACT_DIR}/pathways_job_rendered.yaml"

  # Kueue manages the lifecycle via the queue-name label on the worker Job.
  # It handles admission, topology assignment, and unsuspension.
  # The head job (CPU-only) has no Kueue label and is managed by the JobSet directly.

  echo "=== Waiting for workload to be admitted by Kueue ==="
  WORKLOAD_NAME=""
  until [[ -n "${WORKLOAD_NAME}" ]]; do
    WORKLOAD_NAME=$(kubectl --context="${KUBE_CONTEXT}" get workloads -o custom-columns=NAME:.metadata.name --no-headers 2>/dev/null | grep "${JOBSET_NAME}" | head -1 || true)
    if [[ -z "${WORKLOAD_NAME}" ]]; then
      echo "  Waiting for workload to appear..."
      sleep 2
    fi
  done
  echo "  Found workload: ${WORKLOAD_NAME}"

  until [[ "$(kubectl --context="${KUBE_CONTEXT}" get workload "${WORKLOAD_NAME}" -o jsonpath='{.status.conditions[?(@.type=="Admitted")].status}' 2>/dev/null)" == "True" ]]; do
    echo "  Waiting for workload '${WORKLOAD_NAME}' to be admitted..."
    sleep 5
  done
  echo "  Workload admitted!"

  echo "=== Waiting for jobs to be unsuspended by Kueue ==="
  WORKER_JOB="${JOBSET_NAME}-worker-0"
  HEAD_JOB="${JOBSET_NAME}-pathways-head-0"
  until [[ "$(kubectl --context="${KUBE_CONTEXT}" get job "${WORKER_JOB}" -o jsonpath='{.spec.suspend}' 2>/dev/null)" == "false" ]] && \
        [[ "$(kubectl --context="${KUBE_CONTEXT}" get job "${HEAD_JOB}" -o jsonpath='{.spec.suspend}' 2>/dev/null)" == "false" ]]; do
    echo "  Waiting for Kueue to unsuspend jobs..."
    sleep 5
  done
  echo "  Jobs unsuspended!"

else
  echo "=== --delete-prev=false: skipping deploy, picking up existing job ==="
fi

# ─── Helper: find the current Running head pod ─────────────────────────────
# Pods may be replaced on restart, so we always re-discover by label.
# We require the pod to actually be Ready (all containers running), not just
# phase=Running (which includes pods with crashing containers).
get_ready_head_pod() {
  # Return the name of a head pod whose Ready condition is True.
  kubectl --context="${KUBE_CONTEXT}" get pods \
    -l "jobset.sigs.k8s.io/jobset-name=${JOBSET_NAME},jobset.sigs.k8s.io/replicatedjob-name=pathways-head" \
    --field-selector=status.phase=Running \
    -o jsonpath='{range .items[*]}{.metadata.name}{"\t"}{range .status.conditions[*]}{.type}={.status}{" "}{end}{"\n"}{end}' 2>/dev/null \
    | grep 'Ready=True' | head -1 | cut -f1 || true
}

wait_for_ready_pod() {
  local pod=""
  local attempts=0
  while [[ -z "${pod}" ]]; do
    pod=$(get_ready_head_pod)
    if [[ -z "${pod}" ]]; then
      attempts=$((attempts + 1))
      echo "  Waiting for a Ready head pod... (attempt ${attempts})"
      sleep 10
    fi
  done
  echo "  Head pod: ${pod}"
  # Double-check the jax-tpu container is ready
  local ready=""
  while [[ "${ready}" != "true" ]]; do
    ready=$(kubectl --context="${KUBE_CONTEXT}" get pod "${pod}" \
      -o jsonpath='{range .status.containerStatuses[*]}{.name}={.ready}{"\n"}{end}' 2>/dev/null \
      | grep '^jax-tpu=' | cut -d= -f2 || true)
    if [[ "${ready}" != "true" ]]; then
      echo "  Waiting for container 'jax-tpu' to become ready in pod ${pod}..."
      sleep 10
      # Re-check that this pod is still the right one (it may have been replaced)
      local current
      current=$(get_ready_head_pod)
      if [[ -n "${current}" && "${current}" != "${pod}" ]]; then
        echo "  Pod changed from ${pod} to ${current}, switching..."
        pod="${current}"
      fi
    fi
  done
  sleep 3
  echo "${pod}"
}

# ─── Wait for head pod + tail logs with automatic retry on pod restart ──────
echo "=== Waiting for head pod to be created ==="
POD_NAME=$(wait_for_ready_pod | tail -1)

echo "=== Tailing server logs, waiting for 'Application startup complete' ==="
SERVER_LOG="${ARTIFACT_DIR}/server_startup.log"
> "${SERVER_LOG}"  # truncate

LOG_RETRIES=0
MAX_LOG_RETRIES=60  # give up after ~30 minutes of retries
while true; do
  # (Re-)attach to logs from the current head pod
  kubectl --context="${KUBE_CONTEXT}" logs -f "pod/${POD_NAME}" -c jax-tpu >> "${SERVER_LOG}" 2>&1 &
  LOG_PID=$!

  # Poll for the success string OR for the log process dying (pod restart)
  while true; do
    if grep -q "Application startup complete" "${SERVER_LOG}" 2>/dev/null; then
      echo ""
      echo "  ✅ API server is ready!"
      kill ${LOG_PID} 2>/dev/null || true
      wait ${LOG_PID} 2>/dev/null || true
      break 2  # break out of both loops
    fi
    if ! kill -0 ${LOG_PID} 2>/dev/null; then
      LOG_RETRIES=$((LOG_RETRIES + 1))
      if [[ ${LOG_RETRIES} -ge ${MAX_LOG_RETRIES} ]]; then
        echo "  ❌ Exceeded ${MAX_LOG_RETRIES} log-stream retries. Giving up."
        exit 1
      fi
      echo "  ⚠️  Log stream died (attempt ${LOG_RETRIES}/${MAX_LOG_RETRIES}). Waiting before re-discovering pod..."
      wait ${LOG_PID} 2>/dev/null || true
      # Back off: wait longer on repeated failures (15s, 15s, 15s, ...)
      sleep 15
      # Re-discover a fully Ready head pod (blocks until one exists)
      POD_NAME=$(wait_for_ready_pod | tail -1)
      > "${SERVER_LOG}"  # reset log for the new pod
      break  # restart the outer loop to re-attach logs
    fi
    sleep 5
  done
done

# ─── Continuously save full head pod logs to artifact dir ────────────────────
echo "=== Saving full pathways head logs to ${ARTIFACT_DIR}/pathways_logs.txt ==="
PATHWAYS_LOG="${ARTIFACT_DIR}/pathways_logs.txt"
kubectl --context="${KUBE_CONTEXT}" logs -f "pod/${POD_NAME}" -c jax-tpu > "${PATHWAYS_LOG}" 2>&1 &
PATHWAYS_LOG_PID=$!

# ─── Port-forward ───────────────────────────────────────────────────────────
echo "=== Setting up port-forward to ${POD_NAME}:8000 (local port ${LOCAL_PORT}) ==="
kubectl --context="${KUBE_CONTEXT}" port-forward "pod/${POD_NAME}" "${LOCAL_PORT}:8000" &
PF_PID=$!
sleep 3

if ! kill -0 ${PF_PID} 2>/dev/null; then
  echo "ERROR: port-forward died unexpectedly"
  exit 1
fi
echo "  Port-forward active (PID=${PF_PID})"

# ─── Helper: run the benchmark command ──────────────────────────────────────
run_benchmark() {
  vllm bench serve \
    --base-url="http://localhost:${LOCAL_PORT}" \
    --dataset-name=random \
    --random-input-len=512 \
    --random-output-len=4096 \
    --num-prompts="${NUM_PROMPTS}" \
    --max-concurrency="${NUM_PROMPTS}" \
    --ignore-eos \
    --model=Qwen/Qwen3-235B-A22B
}

# ─── Benchmark Run 1: Profiling warmup ──────────────────────────────────────
if [[ "${RUN_BENCHMARK}" == "false" ]]; then
  echo ""
  echo "============================================================"
  echo "=== --no-benchmark: skipping benchmarks                  ==="
  echo "=== Server is ready at http://localhost:${LOCAL_PORT}     ==="
  echo "=== Port-forward PID: ${PF_PID}                          ==="
  echo "=== Artifacts dir: ${ARTIFACT_DIR}/                      ==="
  echo "============================================================"
  exit 0
fi

echo ""
echo "============================================================"
echo "=== Benchmark Run 1 (profiling warmup) ====================="
echo "============================================================"

BENCH1_LOG="${ARTIFACT_DIR}/benchmark_run1_profiling.log"
run_benchmark 2>&1 | tee "${BENCH1_LOG}"
echo ""
echo "  ✅ Benchmark run 1 complete!"

# Verify that profiling finished during run 1
echo "  Checking that profiling completed during run 1..."
if grep -q "Profiling for decode_heavy phase finished" "${PATHWAYS_LOG}" 2>/dev/null; then
  echo "  ✅ Profiling completed during benchmark run 1."
else
  echo "  ⏳ Profiling not yet detected, waiting for server-side profiling to finish..."
  PROFILING_LOG="${ARTIFACT_DIR}/server_profiling.log"
  SINCE_TIME="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  kubectl --context="${KUBE_CONTEXT}" logs -f "pod/${POD_NAME}" -c jax-tpu --since-time="${SINCE_TIME}" > "${PROFILING_LOG}" 2>&1 &
  PROFLOG_PID=$!

  while true; do
    if grep -q "Profiling for decode_heavy phase finished" "${PROFILING_LOG}" 2>/dev/null || \
       grep -q "Profiling for decode_heavy phase finished" "${PATHWAYS_LOG}" 2>/dev/null; then
      break
    fi
    sleep 5
  done
  echo "  ✅ Profiling complete!"

  kill ${PROFLOG_PID} 2>/dev/null || true
  wait ${PROFLOG_PID} 2>/dev/null || true
fi

# Short pause to let the server settle
sleep 5

# ─── Re-check port-forward is alive before run 2 ────────────────────────────
if ! kill -0 ${PF_PID} 2>/dev/null; then
  echo "  ⚠️  Port-forward died. Re-discovering head pod and re-establishing..."
  POD_NAME=$(wait_for_ready_pod | tail -1)
  kubectl --context="${KUBE_CONTEXT}" port-forward "pod/${POD_NAME}" "${LOCAL_PORT}:8000" &
  PF_PID=$!
  sleep 3
  if ! kill -0 ${PF_PID} 2>/dev/null; then
    echo "ERROR: port-forward died unexpectedly on retry"
    exit 1
  fi
  echo "  ✅ Port-forward re-established (PID=${PF_PID})"
fi

# Quick connectivity check
if ! curl -sf "http://localhost:${LOCAL_PORT}/health" > /dev/null 2>&1; then
  echo "  ⚠️  Health check failed on localhost:${LOCAL_PORT}. Waiting 10s and retrying..."
  sleep 10
  if ! curl -sf "http://localhost:${LOCAL_PORT}/health" > /dev/null 2>&1; then
    echo "ERROR: Server not reachable at localhost:${LOCAL_PORT}"
    exit 1
  fi
fi
echo "  ✅ Server reachable at localhost:${LOCAL_PORT}"

# ─── Benchmark Run 2: Real benchmark ────────────────────────────────────────
echo ""
echo "============================================================"
echo "=== Benchmark Run 2 (real benchmark) ======================="
echo "============================================================"

BENCH2_LOG="${ARTIFACT_DIR}/benchmark_run2_results.log"
run_benchmark 2>&1 | tee "${BENCH2_LOG}"
echo ""
echo "  ✅ Benchmark run 2 complete!"

# ─── Parse results to CSV ───────────────────────────────────────────────────
echo "=== Parsing benchmark results to CSV ==="
CSV_FILE="${ARTIFACT_DIR}/benchmark_results.csv"

RESULT_BLOCK=$(sed -n '/Serving Benchmark Result/,/^=\+$/p' "${BENCH2_LOG}")

if [[ -z "${RESULT_BLOCK}" ]]; then
  echo "  WARNING: Could not find benchmark result block in output."
  echo "  Raw log saved to ${BENCH2_LOG}"
else
  # Parse "Metric Name:   value" lines from the result block
  # Build header and values
  HEADER="data_parallel_size,topology,num_prompts,datetime"
  VALUES="${DATA_PARALLEL_SIZE},${TOPOLOGY},${NUM_PROMPTS},${DATETIME}"

  while IFS= read -r line; do
    # Match lines with a colon followed by a numeric value
    if [[ "${line}" =~ ^([A-Za-z][^:]+):[[:space:]]+([0-9.]+) ]]; then
      metric="${BASH_REMATCH[1]}"
      value="${BASH_REMATCH[2]}"
      # Clean up metric name: trim trailing spaces
      metric="$(echo "${metric}" | sed 's/[[:space:]]*$//')"
      HEADER="${HEADER},${metric}"
      VALUES="${VALUES},${value}"
    fi
  done <<< "${RESULT_BLOCK}"

  {
    echo "${HEADER}"
    echo "${VALUES}"
  } > "${CSV_FILE}"

  echo ""
  echo "  Saved to ${CSV_FILE}"
  echo ""
  echo "  CSV contents:"
  cat "${CSV_FILE}"
fi

# ─── Cleanup ────────────────────────────────────────────────────────────────
echo ""
echo "=== Cleaning up background processes ==="
kill ${PF_PID} 2>/dev/null || true
wait ${PF_PID} 2>/dev/null || true
kill ${PATHWAYS_LOG_PID} 2>/dev/null || true
wait ${PATHWAYS_LOG_PID} 2>/dev/null || true

echo "=== Deleting JobSet '${JOBSET_NAME}' to release resources ==="
kubectl --context="${KUBE_CONTEXT}" delete jobset "${JOBSET_NAME}" --ignore-not-found
echo "  JobSet deleted."

echo ""
echo "============================================================"
echo "=== All done! Artifacts saved to:                        ==="
echo "    ${ARTIFACT_DIR}/"
echo "============================================================"
echo ""
echo "Files:"
ls -la "${ARTIFACT_DIR}/"
