#!/bin/bash

readonly ROOT_DIR="/home/wyzhang_google_com/mnt/ullm/debug"
readonly BENCH_SERVING_BASEDIR="/home/wyzhang_google_com/mnt/ullm/bench_serving"
readonly SLEEP_INTERVAL_SEC=60

ITER=3 # default iteration count
INPUT_LEN=1024 # default input len
OUTPUT_LEN=8192 # default output len
TARGET_DIR="qwen450b-0310" # default target dir

parse_arguments() {
    for arg in "$@"; do
        case $arg in
            --iter=*)
            ITER="${arg#*=}"
            ;;
            --input_len=*)
            INPUT_LEN="${arg#*=}"
            shift
            ;;
            --output_len=*)
            OUTPUT_LEN="${arg#*=}"
            shift
            ;;
            --target_dir=*)
            TARGET_DIR="${arg#*=}"
            shift
            ;;
        esac
    done

    # Compose the storage path after parsing overrides
    BASE_DIR="${ROOT_DIR}/${TARGET_DIR}"
}

setup_directories() {
    rm -rf "$BASE_DIR"
    mkdir -p "$BASE_DIR"
}

start_vllm_server() {
    pkill vllm

    echo "Starting vllm server"
    vllm_log_file="${BASE_DIR}/vllm.log"
    # TPU_VMODULE=tpu_pjrt_client=0 \
    # PHASED_PROFILING_DIR=$xprof_dir \
    TPU_STDERR_LOG_LEVEL=0 \
    USE_MOE_EP_KERNEL="0" \
    MODEL_IMPL_TYPE="vllm" \
    HF_HOME="/home/wyzhang_google_com/ckpt/hf" \
    vllm serve Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8 \
        --async-scheduling \
        --download-dir=/home/wyzhang_google_com/ckpt/hf \
        --gpu-memory-utilization=0.95 \
        --kv-cache-dtype=fp8 \
        --max-model-len=10240 \
        --max-num-batched-tokens=8192 \
        --max-num-seqs=512 \
        --no-enable-prefix-caching \
        --port=8000 \
        --tensor-parallel-size=8 \
        --safetensors-load-strategy=eager \
        --load-format=runai_streamer \
        --model-loader-extra-config='{"memory_limit":68719476736,"concurrency":4}' \
        &> "$vllm_log_file" &
}

wait_for_vllm() {
    vllm_log_file="${BASE_DIR}/vllm.log"
    local timeout=1800 # 30 minutes in seconds
    local elapsed=0

    while true; do
        if grep -q "Application startup complete" "$vllm_log_file"; then
            echo "vllm ready"
            break
        fi

        if [[ "$elapsed" -ge "$timeout" ]]; then
            echo "ERROR: Timed out waiting for vllm to startup after 30 minutes." >&2
            exit 1
        fi

        echo "Waiting for vllm to be ready..."
        sleep "$SLEEP_INTERVAL_SEC"
        elapsed=$((elapsed + SLEEP_INTERVAL_SEC))
    done
}

run_benchmarks() {
    for i in $(seq 1 "$ITER"); do
        vllm_bench_log_dir="${BASE_DIR}/bench_log_${i}"
        vllm_bench_results_dir="${BASE_DIR}/bench_results_${i}"
        xprof_dir="${BASE_DIR}/xprof/exp-${i}"

        mkdir -p "$vllm_bench_log_dir"
        mkdir -p "$vllm_bench_results_dir"
        mkdir -p "$xprof_dir"

        vllm_bench_log_file="${vllm_bench_log_dir}/bench_${i}.log"

        echo "Running benchmark iteration ${i}/${ITER}..."
        python3 "${BENCH_SERVING_BASEDIR}/benchmark_serving.py" \
            --backend vllm \
            --dataset-name random \
            --ignore-eos \
            --max-concurrency=64 \
            --model Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8 \
            --num-prompts=320 \
            --percentile-metrics='ttft,tpot,itl,e2el' \
            --random-input-len="$INPUT_LEN" \
            --random-output-len="$OUTPUT_LEN" \
            --random-range-ratio=1 \
            --request-rate=inf \
            --result-dir "$vllm_bench_results_dir" \
            --save-result \
            --temperature=0.0 \
            2>&1 | tee "$vllm_bench_log_file"

        sleep "$SLEEP_INTERVAL_SEC"
    done
}

main() {
    parse_arguments "$@"
    setup_directories
    start_vllm_server
    wait_for_vllm
    run_benchmarks
    
    echo "Benchmarking complete."
}

# Execute main function
main "$@"
