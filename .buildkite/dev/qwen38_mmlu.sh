#!/bin/bash
# Dev experiment: Qwen3.8-27B-FP8 on v7x-8 via the torchax path
# (MODEL_IMPL_TYPE=vllm). Serves the model, then drives 200 MMLU prompts
# through scripts/vllm/benchmarking/benchmark_serving.py and reports both
# serving latency percentiles and MMLU accuracy.
#
# Runs inside the CI container. Invoked by .buildkite/dev/qwen38_mmlu.yml via
# .buildkite/scripts/run_in_docker.sh.

set -uo pipefail

ROOT_DIR=/workspace
ARTIFACT_DIR=/workspace/artifacts

# Same repo id on both sides: the benchmark sends "model": <--model> in the
# request payload, and vLLM 404s on a name it did not register (see
# _is_model_supported) unless VLLM_SKIP_MODEL_NAME_VALIDATION is set. The FP8
# repo ships its own tokenizer.json/chat_template.jinja, so the client-side
# tokenizer for --mmlu-use-chat-template resolves from it too.
#
# Weights come straight from HF (~31 GB, public/ungated). run_in_docker.sh
# bind-mounts the agent's persistent /mnt/disks/persist/models as HF_HOME, so
# this is a one-time cost per agent, not per build. There is no pre-staged
# copy in gs://tpu-commons-ci/ -- that bucket only holds the models too large
# to pull from HF (DeepSeek, Kimi, Qwen3-Coder-480B, Qwen3-30B-A3B), which are
# served via `--load-format=runai_streamer` off a gs:// path instead.
MODEL=Qwen/Qwen3.8-27B-FP8

# Keep the live logs off the mounted artifact dir; they are copied in on exit
# so a crashed or timed-out run still uploads them.
LOG_FILE=/tmp/server.log
BENCHMARK_LOG_FILE=/tmp/benchmark.log
export READY_MESSAGE="Application startup complete."
export TIMEOUT_SECONDS=3600

mkdir -p "$ARTIFACT_DIR"
touch "$LOG_FILE" "$BENCHMARK_LOG_FILE"

cd "$ROOT_DIR/tpu_inference" || exit 1

# waitForServerReady only. Deliberately NOT using bench_utils.sh's cleanUp:
# it rm -f's $LOG_FILE and $BENCHMARK_LOG_FILE, which would delete the run's
# only artifacts before the upload step.
# shellcheck disable=SC1091
source tests/e2e/benchmarking/bench_utils.sh

finish() {
    rc=$?
    echo "--- collecting artifacts (rc=$rc) ---"
    cp -f "$LOG_FILE" "$BENCHMARK_LOG_FILE" "$ARTIFACT_DIR/" 2>/dev/null || true
    pkill -f "vllm serve $MODEL" 2>/dev/null || true
    pgrep -f -i vllm | xargs -r kill -9 2>/dev/null || true
    exit "$rc"
}
trap finish EXIT

# --- MMLU dataset ---
# No dataset is staged in GCS; every in-repo caller wgets the Hendrycks tarball
# from Berkeley (run_bm.sh:171, tests/e2e/benchmarking/mmlu.sh:136,
# scripts/multihost/nightly_benchmarking.sh:199).
#
# That host is not dependable. It took down the first run of this experiment:
# every request 302'd to an EECS incident page which then 404'd, ten seconds
# into a job that had queued 89 minutes for a TPU. `wget --tries` is no defence
# -- a 404 is a non-retryable error, so wget gives up on the first attempt.
#
# cais/mmlu on HuggingFace serves the identical tarball (same 166184960 bytes,
# same `data/` layout), and HF is already a hard dependency of this run for the
# weights, so it is tried first. Berkeley stays as the fallback in case the
# mirror is ever withdrawn. The tar is validated before use so a saved error
# page cannot masquerade as a download.
DATASET_ROOT="$ROOT_DIR/mmlu"
MMLU_URLS=(
    https://huggingface.co/datasets/cais/mmlu/resolve/main/data.tar
    https://people.eecs.berkeley.edu/~hendrycks/data.tar
)
if [ ! -d "$DATASET_ROOT/data/test" ]; then
    echo "--- downloading MMLU ---"
    mkdir -p "$DATASET_ROOT"
    for url in "${MMLU_URLS[@]}"; do
        echo "trying $url"
        if wget --tries=3 --timeout=30 --retry-connrefused \
                -O "$DATASET_ROOT/data.tar" "$url" \
           && tar -tf "$DATASET_ROOT/data.tar" >/dev/null 2>&1; then
            echo "got MMLU from $url"
            break
        fi
        echo "failed: $url"
        rm -f "$DATASET_ROOT/data.tar"
    done
    if [ ! -s "$DATASET_ROOT/data.tar" ]; then
        echo "ERROR: could not fetch MMLU from any mirror" >&2
        exit 1
    fi
    tar -xf "$DATASET_ROOT/data.tar" -C "$DATASET_ROOT" || exit 1
fi
DATASET_PATH="$DATASET_ROOT/data/test"
if [ ! -d "$DATASET_PATH" ]; then
    echo "ERROR: $DATASET_PATH missing after extract" >&2
    exit 1
fi

echo "--- run context ---"
echo "tpu_inference: $(git -C "$ROOT_DIR/tpu_inference" rev-parse HEAD 2>/dev/null || echo n/a)"
echo "vllm:          $(git -C "$ROOT_DIR/vllm" rev-parse HEAD 2>/dev/null || echo n/a)"
echo "MODEL_IMPL_TYPE=${MODEL_IMPL_TYPE:-unset}"
echo "VLLM_XLA_CHECK_RECOMPILATION=${VLLM_XLA_CHECK_RECOMPILATION:-unset}"
echo "dataset=$DATASET_PATH"

echo "--- starting vllm serve ---"
(vllm serve "$MODEL" \
    --seed=42 \
    --tensor-parallel-size=8 \
    --max-model-len=2048 \
    --max-num-batched-tokens=8192 \
    --max-num-seqs=128 \
    --gpu-memory-utilization=0.9 \
    --kv-cache-dtype=fp8 \
    --no-enable-prefix-caching \
    --async-scheduling \
    --language-model-only \
    --limit-mm-per-prompt='{"image":0,"video":0}' \
    --host=127.0.0.1 --port=8000 2>&1 | tee -a "$LOG_FILE") &

waitForServerReady

echo "--- starting MMLU benchmark ---"
python scripts/vllm/benchmarking/benchmark_serving.py \
    --backend vllm \
    --model "$MODEL" \
    --host 127.0.0.1 --port 8000 \
    --dataset-name mmlu \
    --dataset-path "$DATASET_PATH" \
    --mmlu-use-chat-template \
    --chat-template-system-prompt "Reasoning effort: high, skip thinking process and must only give short answer in something like A, B, C, D. Answer with only the letter in parentheses, e.g. (A)." \
    --chat-template-kwargs '{"enable_thinking": false}' \
    --mmlu-output-len=16 \
    --num-prompts 200 \
    --max-concurrency 32 \
    --request-rate inf \
    --seed 42 \
    --percentile-metrics 'ttft,tpot,itl,e2el' \
    --metric-percentiles '50,90,99' \
    --run-eval 2>&1 | tee -a "$BENCHMARK_LOG_FILE"
BENCH_RC=${PIPESTATUS[0]}

echo "--- summary ---"
grep -E "'accuracy'|Successful requests|Total token throughput|Mean TTFT|Mean TPOT|Mean ITL|Mean E2EL" \
    "$BENCHMARK_LOG_FILE" || echo "(no metrics parsed)"

exit "$BENCH_RC"
