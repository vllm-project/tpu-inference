#!/bin/bash
#
# Sample usage:
#   bash tpu-inference/examples/disagg/qwen3_5/bench_disagg_single_host.sh \
#       --base_dir=$HOME/mnt/disagg \
#       --prefill_host_ip=10.128.0.36 --decode_host_ip=10.128.0.155 \
#       --iter=1 --input_len=1024 --output_len=8192 \
#       --output_subdir=logs/1k8k

set -euo pipefail

# ──────────────────────── start CONFIG ───────────────────────
# EDIT this section for your environment. The ones most commonly
# needing changes are marked with `# EDIT` below.

# Layout: BASE_DIR is the parent directory that contains your venv
# (VENV_SUBDIR) and the benchmark repo (BENCH_REPO_SUBDIR). Output
# logs/results land at BASE_DIR/OUTPUT_SUBDIR. Expected tree:
#
#   BASE_DIR/
#   ├── VENV_SUBDIR/            # python venv
#   │   └── bin/activate
#   ├── BENCH_REPO_SUBDIR/      # benchmark client
#   │   └── benchmark_serving.py
#   └── OUTPUT_SUBDIR/          # created by this script (logs, results)
#
BASE_DIR="${HOME}"                  # EDIT if your repo is not under $HOME
VENV_SUBDIR="venv"                  # EDIT if your venv lives elsewhere
BENCH_REPO_SUBDIR="bench_serving"   # EDIT to point at your benchmark_serving.py
OUTPUT_SUBDIR="output"              # overridable via --output_subdir=NAME; results land in $BASE_DIR/$OUTPUT_SUBDIR

# Host IPs are required. Set them here before running.
PREFILL_HOST_IP="10.128.0.36"       # EDIT to point to your prefill HOST IP
DECODE_HOST_IP="10.128.0.155"       # EDIT to point to your decode HOST IP
PREFILL_PORT=8400
DECODE_PORT=9400
PROXY_PORT=8001

# Model to serve. Must be reachable from both Prefill and Decode hosts.
MODEL_URI="gs://wyzhang-dev/ckpt/hf/Qwen3.5-397B-A17B-FP8"  # EDIT: your model weight path 
MODEL_NAME="Qwen/Qwen3.5-397B-A17B-FP8"                     # EDIT: served name

# Benchmark defaults.
ITER=1              # overridable via --iter=N (number of benchmark iterations)
INPUT_LEN=1024      # overridable via --input_len=N (prompt token length)
OUTPUT_LEN=8192     # overridable via --output_len=N (decode token length)
NUM_PROMPTS=64      # overridable via --num_prompts=N (total prompts per iteration)
MAX_CONCURRENCY=64  # overridable via --max_concurrency=N (concurrent in-flight requests)
XPROF_ENABLED=false # overridable via --xprof (appends '-xprof' to output subdir)

# TPU topology. Must match your pod.
TPU_PROCESS_BOUNDS_ENV="1,1,1"              # EDIT comment out or change to match your TPU setup
TPU_CHIPS_PER_PROCESS_BOUNDS_ENV="2,2,1"    # EDIT comment out or change to match your TPU setup

# ──────────────────────── end CONFIG ───────────────────────

WAIT_FOR_VLLM_INTERVAL_SEC=60

function parse_arguments() {
    for arg in "$@"; do
        case $arg in
            --base_dir=*)        BASE_DIR="${arg#*=}" ;;
            --prefill_host_ip=*) PREFILL_HOST_IP="${arg#*=}" ;;
            --decode_host_ip=*)  DECODE_HOST_IP="${arg#*=}" ;;
            --iter=*)            ITER="${arg#*=}" ;;
            --input_len=*)       INPUT_LEN="${arg#*=}" ;;
            --output_len=*)      OUTPUT_LEN="${arg#*=}" ;;
            --num_prompts=*)     NUM_PROMPTS="${arg#*=}" ;;
            --max_concurrency=*) MAX_CONCURRENCY="${arg#*=}" ;;
            --output_subdir=*)   OUTPUT_SUBDIR="${arg#*=}" ;;
            --xprof)             XPROF_ENABLED=true ;;
            -h|--help)
                sed -n '1,30p' "$0" | sed 's/^#//;s/^ //'
                exit 0 ;;
            *)
                echo "Unknown arg: $arg" >&2
                exit 1 ;;
        esac
    done

    if [ "$XPROF_ENABLED" != "false" ]; then
        OUTPUT_SUBDIR="${OUTPUT_SUBDIR}-xprof"
    fi

    OUTPUT_DIR="${BASE_DIR}/${OUTPUT_SUBDIR}"
    echo "Output dir: ${OUTPUT_DIR}"
}

function validate() {
    local ok=true
    if [[ -z "${PREFILL_HOST_IP}" || -z "${DECODE_HOST_IP}" ]]; then
        echo "ERROR: PREFILL_HOST_IP and DECODE_HOST_IP must be set in the CONFIG block at the top of $0." >&2
        ok=false
    fi
    if [[ ! -d "${BASE_DIR}/${VENV_SUBDIR}" ]]; then
        echo "ERROR: venv subdir not found: ${BASE_DIR}/${VENV_SUBDIR}" >&2
        ok=false
    fi
    if [[ ! -d "${BASE_DIR}/${BENCH_REPO_SUBDIR}" ]]; then
        echo "ERROR: benchmark repo subdir not found: ${BASE_DIR}/${BENCH_REPO_SUBDIR}" >&2
        ok=false
    fi
    for host in "${PREFILL_HOST_IP}" "${DECODE_HOST_IP}"; do
        if ! ssh -o BatchMode=yes -o ConnectTimeout=5 "${host}" true 2>/dev/null; then
            echo "ERROR: cannot ssh to ${host} without password prompt" >&2
            ok=false
        fi
    done
    $ok || exit 1

    # Always start from a clean output dir.
    rm -rf "${OUTPUT_DIR}"
    mkdir -p "${OUTPUT_DIR}"
}

function start_vllm_server() {
    local host=$1
    local port=$2
    local kv_role=$3

    echo "Starting vLLM server on ${host}:${port} role=${kv_role}..."

    local launch_script
    launch_script="${OUTPUT_DIR}/vllm_launch_${kv_role}.sh"

    cat > "${launch_script}" << CONFIG_EOF
#!/bin/bash
set -euo pipefail

# Baked-in config (expanded at launch time on the driver).
BASE_DIR='${BASE_DIR}'
OUTPUT_DIR='${OUTPUT_DIR}'
VENV_SUBDIR='${VENV_SUBDIR}'
PORT='${port}'
KV_ROLE='${kv_role}'
MODEL_URI='${MODEL_URI}'
MODEL_NAME='${MODEL_NAME}'
TPU_PROCESS_BOUNDS_ENV='${TPU_PROCESS_BOUNDS_ENV}'
TPU_CHIPS_PER_PROCESS_BOUNDS_ENV='${TPU_CHIPS_PER_PROCESS_BOUNDS_ENV}'
CONFIG_EOF

    cat >> "${launch_script}" << 'BODY_EOF'

mkdir -p "${OUTPUT_DIR}"

pkill -9 -u "$USER" -x VLLM::EngineCor || true
pkill -9 -u "$USER" -f 'vllm serve' || true
sleep 15
[[ -f /tmp/libtpu_lockfile && -O /tmp/libtpu_lockfile ]] && rm -f /tmp/libtpu_lockfile
sleep 15

source "${BASE_DIR}/${VENV_SUBDIR}/bin/activate"

kv_transfer_config="{\"kv_connector\":\"TPUConnectorHMA\",\"kv_connector_module_path\":\"tpu_inference.distributed.tpu_connector_hma\",\"kv_role\":\"${KV_ROLE}\"}"

vllm_args=(
    "${MODEL_URI}"
    "--served-model-name" "${MODEL_NAME}"
    "--async-scheduling"
    "--gpu-memory-utilization=0.95"
    "--kv-cache-dtype=fp8"
    "--max-model-len=9216"
    "--max-num-batched-tokens=4096"
    "--max-num-seqs=512"
    "--no-enable-prefix-caching"
    "--no-disable-hybrid-kv-cache-manager"
    "--port=${PORT}"
    "--tensor-parallel-size=8"
    "--block-size=256"
    "--limit-mm-per-prompt" '{"image": 0, "video": 0}'
    "--hf-overrides" '{"text_config": {"rope_parameters": null, "rope_theta": 10000000, "partial_rotary_factor": 0.25}}'
    "--kv-transfer-config" "${kv_transfer_config}"
)

TPU_STDERR_LOG_LEVEL=2 \
USE_MOE_EP_KERNEL="0" \
MODEL_IMPL_TYPE="vllm" \
TPU_PROCESS_BOUNDS="${TPU_PROCESS_BOUNDS_ENV}" \
TPU_CHIPS_PER_PROCESS_BOUNDS="${TPU_CHIPS_PER_PROCESS_BOUNDS_ENV}" \
CLOUD_TPU_TASK_ID="0" \
vllm serve "${vllm_args[@]}" > "${OUTPUT_DIR}/vllm.log" 2>&1
BODY_EOF

    # Ship to the remote host at the same path (so remote OUTPUT_DIR
    # also has the script alongside vllm.log) and launch it.
    ssh "${host}" "mkdir -p '${OUTPUT_DIR}'"
    scp -q "${launch_script}" "${host}:${launch_script}"
    ssh "${host}" "nohup bash '${launch_script}' > /dev/null 2>&1" &
}

function wait_for_vllm_server() {
    local host=$1
    local kv_role=$2
    local timeout=1800 # 30 minutes
    local elapsed=0

    while true; do
        if ssh "${host}" "grep -q 'Application startup complete' '${OUTPUT_DIR}/vllm.log'" &>/dev/null; then
            echo "Server on ${host} (${kv_role}) ready"
            break
        fi
        if [[ "$elapsed" -ge "$timeout" ]]; then
            echo "ERROR: Timed out waiting for ${kv_role} on ${host}" >&2
            exit 1
        fi
        echo "Waiting for ${kv_role} on ${host}..."
        sleep "$WAIT_FOR_VLLM_INTERVAL_SEC"
        elapsed=$((elapsed + WAIT_FOR_VLLM_INTERVAL_SEC))
    done
}

function start_proxy() {
    pkill -u "$USER" -f toy_proxy_server.py || true
    echo "Starting proxy locally on :${PROXY_PORT}..."

    # Write the proxy launcher to OUTPUT_DIR for reference / repro.
    local launch_script="${OUTPUT_DIR}/toy_proxy_launch.sh"
    cat > "${launch_script}" << PROXY_EOF
#!/bin/bash
set -euo pipefail

python3 '${BASE_DIR}/tpu-inference/examples/disagg/toy_proxy_server.py' \\
    --host 0.0.0.0 \\
    --port '${PROXY_PORT}' \\
    --prefiller-hosts '${PREFILL_HOST_IP}' \\
    --prefiller-ports '${PREFILL_PORT}' \\
    --decoder-hosts '${DECODE_HOST_IP}' \\
    --decoder-ports '${DECODE_PORT}'
PROXY_EOF

    nohup bash "${launch_script}" > "${OUTPUT_DIR}/proxy.log" 2>&1 &
}

function run_benchmarks() {
    for i in $(seq 1 "$ITER"); do
        bench_log_dir="${OUTPUT_DIR}/bench_log_${i}"
        bench_res_dir="${OUTPUT_DIR}/bench_results_${i}"
        mkdir -p "$bench_log_dir" "$bench_res_dir"
        bench_log="${bench_log_dir}/bench_${i}.log"
        local launch_script="${OUTPUT_DIR}/run_bench_${i}.sh"

        echo "Running benchmark iteration ${i}/${ITER}..."

        # Write the benchmark invocation to OUTPUT_DIR for reference /
        # repro: bash ${launch_script} replays this exact iteration.
        cat > "${launch_script}" << BENCH_EOF
#!/bin/bash
set -euo pipefail

python3 '${BASE_DIR}/${BENCH_REPO_SUBDIR}/benchmark_serving.py' \\
    --backend=vllm \\
    --base-url=http://127.0.0.1:${PROXY_PORT} \\
    --dataset-name=random \\
    --ignore-eos \\
    --max-concurrency=${MAX_CONCURRENCY} \\
    --model='${MODEL_NAME}' \\
    --num-prompts=${NUM_PROMPTS} \\
    --percentile-metrics=ttft,tpot,itl,itl0,e2el \\
    --random-input-len=${INPUT_LEN} \\
    --random-output-len=${OUTPUT_LEN} \\
    --random-range-ratio=1 \\
    --request-rate=inf \\
    --result-dir='${bench_res_dir}' \\
    --save-result
BENCH_EOF

        bash "${launch_script}" 2>&1 | tee "$bench_log"
    done
}

function main() {
    parse_arguments "$@"
    validate

    start_vllm_server "${PREFILL_HOST_IP}" "${PREFILL_PORT}" "kv_producer"
    start_vllm_server "${DECODE_HOST_IP}"  "${DECODE_PORT}"  "kv_consumer"
    sleep 30

    wait_for_vllm_server "${PREFILL_HOST_IP}" "kv_producer"
    wait_for_vllm_server "${DECODE_HOST_IP}"  "kv_consumer"
    sleep 30    

    start_proxy
    sleep 30

    run_benchmarks

    echo "Benchmarking complete. Results under ${OUTPUT_DIR}"
}

main "$@"
