#!/bin/bash

MODEL="BCCard/Qwen3-Coder-480B-A35B-Instruct-FP8-Dynamic"
PREFILL_LABEL="app=vllm-prefill"
DECODE_LABEL="app=vllm-decode"
PROXY_LABEL="app=vllm-proxy"
INTERVAL=5
TIMEOUT_SECONDS=7200

init_env() {
    # Get credentials to GKE cluster
    echo "gcloud container clusters get-credentials $CLUSTER_NAME --zone $ZONE --project $PROJECT_NAME"
    gcloud container clusters get-credentials $CLUSTER_NAME --zone $ZONE --project $PROJECT_NAME

    # Ensure HF_TOKEN is set
    echo "kubectl create secret generic hf-token-secret --from-literal=token=$HF_TOKEN"
    kubectl create secret generic hf-token-secret --from-literal=token=$HF_TOKEN --dry-run=client -o yaml | kubectl apply -f -

    # Create storage class
    echo "kubectl apply -f ./kubernetes/manifests/storageclass.yaml"
    kubectl apply -f ./kubernetes/manifests/storageclass.yaml
}

deploy_1p1d() {
    echo "kubectl apply -f ./kubernetes/manifests/v7x/single_prefill.yaml"
    kubectl apply -f ./kubernetes/manifests/v7x/single_prefill.yaml

    echo "kubectl apply -f ./kubernetes/manifests/v7x/single_decode.yaml"
    kubectl apply -f ./kubernetes/manifests/v7x/single_decode.yaml

    echo "kubectl apply -f ./kubernetes/manifests/v7x/proxy1p1d.yaml"
    kubectl apply -f ./kubernetes/manifests/v7x/proxy1p1d.yaml
}

cleanup_1p1d() {
    echo "kubectl delete -f ./kubernetes/manifests/v7x/single_prefill.yaml"
    kubectl delete -f ./kubernetes/manifests/v7x/single_prefill.yaml

    echo "kubectl delete -f ./kubernetes/manifests/v7x/single_decode.yaml"
    kubectl delete -f ./kubernetes/manifests/v7x/single_decode.yaml

    echo "kubectl delete -f ./kubernetes/manifests/v7x/proxy1p1d.yaml"
    kubectl delete -f ./kubernetes/manifests/v7x/proxy1p1d.yaml
}

wait_for_vllm() {
    local label=$1
    local name=$2

    echo "Waiting for $name ($label) to be ready..."

    while true; do
        # Check if we have exceeded the 2-hour timeout
        if [ "$SECONDS" -ge "$TIMEOUT_SECONDS" ]; then
            echo "ERROR: Pods failed to start after the maximum timeout."
            exit 1
        fi

        # Get the pod name dynamically
        POD_NAME=$(kubectl get pods -l "$label" -o jsonpath="{.items[0].metadata.name}" 2>/dev/null)

        if [ -n "$POD_NAME" ]; then
            # Check logs for the startup string
            if kubectl logs "$POD_NAME" --tail 50 2>/dev/null | grep -q "Application startup complete"; then
                echo "$name is ready! (Elapsed: $((SECONDS / 60))m)"
                break
            fi
        fi

        sleep "$INTERVAL"
    done
}

run_disagg_benchmark() {
    local proxy=$1
    local model=$2
    local input_len=$3
    local output_len=$4
    local num_prompts=$5
    local filename="${input_len}_${output_len}.json"

    for RATE in 0.5 1.0 2.0 3.0 4.0 5.0
    do
        echo "-------------------------------------------------------"
        echo "Starting Benchmark: Rate=$RATE, Input=$input_len, Output=$output_len"
        echo "-------------------------------------------------------"

        kubectl exec $proxy -- vllm bench serve \
            --model=$model \
            --dataset-name=random \
            --random-input-len=$input_len \
            --random-output-len=$output_len \
            --num-prompts=$num_prompts \
            --ignore-eos \
            --host=localhost \
            --port=10000 \
            --request-rate=$RATE \
            --metric-percentiles 90,99 \
            --append-result \
            --result-file=$filename

        sleep 2
    done

    kubectl cp "${proxy}:${filename}" "./${filename}"
}


if [ -z "${HF_TOKEN:-}" ]; then
  echo "Error: HF_TOKEN is not set."
  exit 1
fi

if [ -z "${PROJECT_NAME:-}" ]; then
  echo "Error: PROJECT_NAME is not set."
  exit 1
fi

if [ -z "${CLUSTER_NAME:-}" ]; then
  echo "Error: CLUSTER_NAME is not set."
  exit 1
fi

if [ -z "${ZONE:-}" ]; then
  echo "Error: ZONE is not set."
  exit 1
fi

# Initialize GKE environment
init_env

# Deploy 1P1D disaggregated serving
deploy_1p1d

START_TIME=$SECONDS

# Wait for Prefill
wait_for_vllm "$PREFILL_LABEL" "prefill"

# Wait for Decode
wait_for_vllm "$DECODE_LABEL" "decode"

echo "------------------------------------------------"
echo "Ready to benchmark. Total wait time: $(( (SECONDS - START_TIME) / 60 )) minutes."
echo "------------------------------------------------"

# Find the pod name for the proxy server
PROXY_POD=$(kubectl get pods -l "$PROXY_LABEL" -o jsonpath="{.items[0].metadata.name}")

# Run input=1024, output=8192
run_disagg_benchmark $PROXY_POD $MODEL 1024 8192 256

# Run input=8192, output=1024
run_disagg_benchmark $PROXY_POD $MODEL 8192 1024 256

# Benchmark results should be saved in local files 1024_8192.json and
# 8192_1024.json.
# Need to extract metrics like:
# - Throughput
# - TTFT (mean, median, P90)
# - TPOT (mean, median, P90)

# Cleanup
cleanup_1p1d