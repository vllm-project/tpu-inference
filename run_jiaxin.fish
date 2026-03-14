#!/path/to/fish
set base_dir /home/wyzhang_google_com/mnt/ullm/debug/qwen450b

rm -rf $base_dir
mkdir -p $base_dir

for i in (seq 1 1)
    set vllm_log_dir {$base_dir}/vllm_{$i}
    set vllm_bench_log_dir {$base_dir}/bench_log_{$i}
    set vllm_bench_results_dir {$base_dir}/bench_results_{$i}
    set xprof_dir {$base_dir}/xprof/exp-{$i}

    mkdir -p $vllm_log_dir
    mkdir -p $vllm_bench_log_dir
    mkdir -p $vllm_bench_results_dir
    mkdir -p $xprof_dir

    pkill vllm

    echo "Starting vllm server"
    set vllm_log_file {$vllm_log_dir}/vllm.log
    # TPU_VMODULE=tpu_pjrt_client=0 \
    PHASED_PROFILING_DIR=$xprof_dir \
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
        &> $vllm_log_file &

    # Waiting for vllm to be ready.
    while true
        echo "waiting for vllm to be ready"

        if grep "Application startup complete" $vllm_log_file > /dev/null
            echo "vllm ready"
            break
        end
        sleep 15
    end    

    set vllm_bench_log_file {$vllm_bench_log_dir}/bench_{$i}.log

    python3 /home/wyzhang_google_com/mnt/ullm/bench_serving/benchmark_serving.py \
        --backend vllm \
        --dataset-name random \
        --ignore-eos \
        --max-concurrency=64 \
        --model Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8 \
        --num-prompts=4 \
        --percentile-metrics='ttft,tpot,itl,e2el' \
        --random-input-len=1024 \
        --random-output-len=256 \
        --random-range-ratio=1.0 \
        --request-rate=inf \
        --result-dir $vllm_bench_results_dir \
        --save-result \
        --temperature=0.0 \
        2>&1 | tee $vllm_bench_log_file

    pkill vllm

    sleep 15
end
