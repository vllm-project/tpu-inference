#!/bin/bash
set -e

CURRENT_USER="${USER:-$(whoami)}"
CLEAN_USER=$(echo "$CURRENT_USER" | tr  '[:upper:] '  '[:lower:] ' | tr -dc  'a-z0-9 ')

export MODEL_NAME="meta-llama/Llama-3.1-8B-Instruct"
export PREFILL_TOPOLOGY="2x2x1"
export DECODE_TOPOLOGY="2x2x2"
export DECODE_REPLICAS="1"
export SERVICE_ACCOUNT="vllm-sa"
export IMAGE="${IMAGE:-docker.io/vllm/vllm-tpu:v0.27.0}"
export KV_CONNECTOR=""
export KV_CONNECTOR_MODULE=""
export GCS_BUCKET=""
export DRY_RUN="false"
export OUTPUT_FILE=""
export JOB_DURATION_MINUTES="90"
export PREFILL_PORT=8400
export DECODE_PORT=9400

# Parse long options
POSITIONAL=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run|--generate-only)
            export DRY_RUN="true"
            shift
            ;;
        --decode-replicas|--d-replicas)
            export DECODE_REPLICAS="$2"
            shift 2
            ;;
        --decode-replicas=*|--d-replicas=*)
            export DECODE_REPLICAS="${1#*=}"
            shift
            ;;
        -i|--image)
            export IMAGE="$2"
            shift 2
            ;;
        --image=*)
            export IMAGE="${1#*=}"
            shift
            ;;
        *)
            POSITIONAL+=("$1")
            shift
            ;;
    esac
done
set -- "${POSITIONAL[@]}"

usage() {
    echo "============================================================"
    echo " TPU v7 Disaggregated Prefill/Decode Deployment Script"
    echo "============================================================"
    echo "Usage: $0 [options]"
    echo ""
    echo "Options:"
    echo "  -m <model>      Model name from Hugging Face or GCS (default: meta-llama/Llama-3.1-8B-Instruct)"
    echo "  -p <topology>   Prefill slice TPU v7 topology (default: 2x2x1)"
    echo "  -d <topology>   Decode slice TPU v7 topology (default: 2x2x2)
  -N <replicas>   Number of Decode replicas to scale horizontally (default: 1)
  --decode-replicas <N>  Same as -N"
    echo "  -i <image>      Docker container image (default: docker.io/vllm/vllm-tpu:v0.27.0)"
    echo "  -c <connector>  KV Connector (default: auto-detected TPUConnector or TPURaidenConnector)"
    echo "  -b <bucket>     GCS bucket path for caching (e.g. gs://<bucket>/hf-cache)"
    echo "  -s <sa>         Kubernetes ServiceAccount name (default: vllm-sa)"
    echo "  -r <rps>        Benchmark request rate in req/s (default: 10.0)"
    echo "  -D <mins>       Job declared duration in minutes for Kueue (default: 90)"
    echo "  -o <file>       Output filepath to save the rendered YAML"
    echo "  -g, --dry-run   Dry-run / Generate YAML only (do not deploy to Kubernetes)"
    echo "  -h, --help      Show this help message"
    echo ""
    echo "Examples:"
    echo "  1. Default Disaggregated Serving (Prefill 2x2x1, Decode 2x2x2):"
    echo "     $0 -m meta-llama/Llama-3.1-8B-Instruct -p 2x2x1 -d 2x2x2"
    echo ""
    echo "  2. High-Load Asymmetric Serving (30 RPS):"
    echo "     $0 -m meta-llama/Llama-3.1-8B-Instruct -p 2x2x1 -d 2x2x2 -r 30.0"
    echo ""
    echo "  3. With GCS Cache Persistence:"
    echo "     $0 -m meta-llama/Llama-3.1-8B-Instruct -p 2x2x1 -d 2x2x2 -b gs://my-bucket/hf-cache"
    echo ""
    echo "  4. Production TorchTPU Image with TPURaidenConnector:"
    echo "     $0 -m Qwen/Qwen3.5-4B -p 2x2x1 -d 2x2x1 \\"
    echo "        -i us-central1-docker.pkg.dev/cloud-ullm-inference-ci-cd/vllm-torchtpu/torchtpu-vllm-prod:latest"
    echo ""
    echo "  5. Dry-run / Generate YAML Only:"
    echo "     $0 -m Qwen/Qwen3.5-4B -p 2x2x1 -d 2x2x2 --dry-run -o disagg-job.yaml"
    echo "============================================================"
    exit 1
}

while getopts "m:p:d:N:i:c:b:s:D:o:r:gh" opt; do
    case ${opt} in
        m ) export MODEL_NAME=$OPTARG ;;
        p ) export PREFILL_TOPOLOGY=$OPTARG ;;
        d ) export DECODE_TOPOLOGY=$OPTARG ;;
        N ) export DECODE_REPLICAS=$OPTARG ;;
        i ) export IMAGE=$OPTARG ;;
        c ) export KV_CONNECTOR=$OPTARG ;;
        b ) export GCS_BUCKET=$OPTARG ;;
        s ) export SERVICE_ACCOUNT=$OPTARG ;;
        D ) export JOB_DURATION_MINUTES=$OPTARG ;;
        o ) export OUTPUT_FILE=$OPTARG ;;
        r ) export REQUEST_RATE=$OPTARG ;;
        g ) export DRY_RUN="true" ;;
        h ) usage ;;
        * ) usage ;;
    esac
done

export REQUEST_RATE="${REQUEST_RATE:-10.0}"

# --- KV CONNECTOR AUTO-DETECTION ---
if [ -z "$KV_CONNECTOR" ]; then
    if [[ "$IMAGE" == *"torchtpu"* ]]; then
        export KV_CONNECTOR="TPURaidenConnector"
        export KV_CONNECTOR_MODULE="vllm_torchtpu.distributed.kv_transfer.tpu_connector"
    else
        export KV_CONNECTOR="TPUConnector"
        export KV_CONNECTOR_MODULE="tpu_inference.distributed.tpu_connector"
    fi
else
    if [ "$KV_CONNECTOR" == "TPURaidenConnector" ]; then
        export KV_CONNECTOR_MODULE="vllm_torchtpu.distributed.kv_transfer.tpu_connector"
    elif [ "$KV_CONNECTOR" == "TPUConnector" ]; then
        export KV_CONNECTOR_MODULE="tpu_inference.distributed.tpu_connector"
    else
        echo "⚠️ Unknown KV Connector  '$KV_CONNECTOR ', defaulting module path to tpu_inference."
        export KV_CONNECTOR_MODULE="tpu_inference.distributed.tpu_connector"
    fi
fi

# --- TOPOLOGY & HARDWARE CALCULATION HELPER ---
calc_limits() {
    local top=$1
    IFS='x' read -r tx ty tz <<< "$top"
    local chips=$((tx * ty * tz))
    local vms=$((chips / 4))
    local tp=$((chips * 2))
    echo "$vms $tp"
}

# Process Prefill slice
read PREFILL_NUM_VMS PREFILL_TP_SIZE <<< $(calc_limits "$PREFILL_TOPOLOGY")
export PREFILL_NUM_VMS PREFILL_TP_SIZE
if [ "$PREFILL_NUM_VMS" -gt 1 ]; then
    echo "Fetching placement policy for Prefill slice ($PREFILL_TOPOLOGY)..."
    PREFILL_POLICY=$(gcloud compute resource-policies list --filter="region: us-central1" --project=cloud-tpu-shared-capacity --format="value(name)" 2>/dev/null | grep "\-${PREFILL_TOPOLOGY}-" | head -n 1 || true)
    if [ -n "$PREFILL_POLICY" ]; then
        export PREFILL_PLACEMENT_LINE="cloud.google.com/placement-policy-name: $PREFILL_POLICY"
    else
        export PREFILL_PLACEMENT_LINE=""
    fi
else
    export PREFILL_PLACEMENT_LINE=""
fi

# Process Decode slice
read DECODE_NUM_VMS DECODE_TP_SIZE <<< $(calc_limits "$DECODE_TOPOLOGY")
export DECODE_NUM_VMS DECODE_TP_SIZE
if [ "$DECODE_NUM_VMS" -gt 1 ]; then
    echo "Fetching placement policy for Decode slice ($DECODE_TOPOLOGY)..."
    DECODE_POLICY=$(gcloud compute resource-policies list --filter="region: us-central1" --project=cloud-tpu-shared-capacity --format="value(name)" 2>/dev/null | grep "\-${DECODE_TOPOLOGY}-" | head -n 1 || true)
    if [ -n "$DECODE_POLICY" ]; then
        export DECODE_PLACEMENT_LINE="cloud.google.com/placement-policy-name: $DECODE_POLICY"
    else
        export DECODE_PLACEMENT_LINE=""
    fi
else
    export DECODE_PLACEMENT_LINE=""
fi

# --- STORAGE & WEIGHT CACHE CONFIGURATION ---
MODEL_LOWER=$(echo "${MODEL_NAME}" | tr  '[:upper:] '  '[:lower:] ')

# Estimate memory limit based on model size
if [[ "${MODEL_LOWER}" == *"70b"* ]] || [[ "${MODEL_LOWER}" == *"coder-480b"* ]]; then
    CONTAINER_MEM_LIMIT="72Gi"
    RAM_CACHE_LIMIT="48Gi"
    GCSFUSE_MEM_LIMIT="72Gi"
elif [[ "${MODEL_LOWER}" == *"32b"* ]] || [[ "${MODEL_LOWER}" == *"35b"* ]] || [[ "${MODEL_LOWER}" == *"397b"* ]]; then
    CONTAINER_MEM_LIMIT="72Gi"
    RAM_CACHE_LIMIT="36Gi"
    GCSFUSE_MEM_LIMIT="64Gi"
else
    CONTAINER_MEM_LIMIT="64Gi"
    RAM_CACHE_LIMIT="24Gi"
    GCSFUSE_MEM_LIMIT="32Gi"
fi
export CONTAINER_MEM_LIMIT RAM_CACHE_LIMIT

if [[ "${MODEL_NAME}" == gs://* ]]; then
    STORAGE_TYPE="Direct GCS Mount"
    GCS_PATH_CLEAN="${MODEL_NAME#gs://}"
    BUCKET_NAME="${GCS_PATH_CLEAN%%/*}"
    SUBPATH="${GCS_PATH_CLEAN#*/}"

    export LOCAL_MODEL_PATH="/data/${SUBPATH}"
    export SERVED_MODEL_NAME="${SUBPATH##*/}"

    export GCS_FUSE_ANNOTATIONS=$(cat <<EOF
            annotations:
              gke-gcsfuse/volumes: "true"
              gke-gcsfuse/memory-limit: "${GCSFUSE_MEM_LIMIT}"
              gke-gcsfuse/cpu-limit: "8"
EOF
)
    export CACHE_VOLUME_MOUNTS=$(cat <<EOF
              - name: gcs-model-dir
                mountPath: /data
                readOnly: true
EOF
)
    export CACHE_VOLUMES=$(cat <<EOF
            - name: gcs-model-dir
              csi:
                driver: gcsfuse.csi.storage.gke.io
                readOnly: true
                volumeAttributes:
                  bucketName: "${BUCKET_NAME}"
                  mountOptions: "implicit-dirs"
EOF
)
elif [ -n "${GCS_BUCKET}" ]; then
    STORAGE_TYPE="GCS FUSE Cache Persistence"
    GCS_PATH_CLEAN="${GCS_BUCKET#gs://}"
    BUCKET_NAME="${GCS_PATH_CLEAN%%/*}"
    SUBPATH="${GCS_PATH_CLEAN#*/}"
    if [ "$SUBPATH" == "$BUCKET_NAME" ]; then SUBPATH=""; fi

    export LOCAL_MODEL_PATH="${MODEL_NAME}"
    export SERVED_MODEL_NAME="${MODEL_NAME}"

    export GCS_FUSE_ANNOTATIONS=$(cat <<EOF
            annotations:
              gke-gcsfuse/volumes: "true"
              gke-gcsfuse/memory-limit: "${GCSFUSE_MEM_LIMIT}"
              gke-gcsfuse/cpu-limit: "8"
EOF
)
    export CACHE_VOLUME_MOUNTS=$(cat <<EOF
              - name: gcs-hf-cache
                mountPath: /root/.cache/huggingface
EOF
)
    if [ -n "$SUBPATH" ]; then
        export CACHE_VOLUMES=$(cat <<EOF
            - name: gcs-hf-cache
              csi:
                driver: gcsfuse.csi.storage.gke.io
                volumeAttributes:
                  bucketName: "${BUCKET_NAME}"
                  mountOptions: "implicit-dirs,file-cache:max-size-mb:204800,only-dir:${SUBPATH}"
EOF
)
    else
        export CACHE_VOLUMES=$(cat <<EOF
            - name: gcs-hf-cache
              csi:
                driver: gcsfuse.csi.storage.gke.io
                volumeAttributes:
                  bucketName: "${BUCKET_NAME}"
                  mountOptions: "implicit-dirs,file-cache:max-size-mb:204800"
EOF
)
    fi
else
    STORAGE_TYPE="Ephemeral RAM Disk (In-Memory emptyDir)"
    export LOCAL_MODEL_PATH="${MODEL_NAME}"
    export SERVED_MODEL_NAME="${MODEL_NAME}"
    export GCS_FUSE_ANNOTATIONS=""
    export CACHE_VOLUME_MOUNTS=$(cat <<EOF
              - name: cache-ramdisk
                mountPath: /root/.cache/huggingface
EOF
)
    export CACHE_VOLUMES=$(cat <<EOF
            - name: cache-ramdisk
              emptyDir:
                medium: Memory
                sizeLimit: "${RAM_CACHE_LIMIT}"
EOF
)
fi

RANDOM_SUFFIX=$(cat /dev/urandom | tr -dc  'a-z0-9 ' | fold -w 5 | head -n 1)
export JOB_NAME="${CLEAN_USER}-test-${RANDOM_SUFFIX}"
export HF_TOKEN_SECRET="${CLEAN_USER}-test-token"

echo "============================================================"
echo " ⚡ TPU v7 Disaggregated Prefill/Decode Architecture"
echo "============================================================"
echo " Job Name        : ${JOB_NAME}"
echo " Container Image : ${IMAGE}"
echo " KV Connector    : ${KV_CONNECTOR} (${KV_CONNECTOR_MODULE})"
echo " Model Target    : ${MODEL_NAME}"
echo " Storage Mode    : ${STORAGE_TYPE}"
echo " Prefill Slice   : ${PREFILL_TOPOLOGY} (${PREFILL_NUM_VMS} VM(s), TP=${PREFILL_TP_SIZE}, Port ${PREFILL_PORT})"
echo " Decode Slice    : ${DECODE_TOPOLOGY} (${DECODE_NUM_VMS} VM(s), TP=${DECODE_TP_SIZE}, Port ${DECODE_PORT}, Replicas: ${DECODE_REPLICAS})"
echo " Request Rate    : ${REQUEST_RATE} RPS"
echo " Service Account : ${SERVICE_ACCOUNT}"
echo " Job Duration    : ${JOB_DURATION_MINUTES} mins"
echo "============================================================"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEMPLATE_FILE="${SCRIPT_DIR}/benchmark_disagg_template.yaml"

RENDERED_YAML=$(envsubst  '${JOB_NAME} ${MODEL_NAME} ${LOCAL_MODEL_PATH} ${SERVED_MODEL_NAME} ${IMAGE} ${PREFILL_TOPOLOGY} ${PREFILL_NUM_VMS} ${PREFILL_TP_SIZE} ${PREFILL_PLACEMENT_LINE} ${PREFILL_PORT} ${DECODE_TOPOLOGY} ${DECODE_NUM_VMS} ${DECODE_TP_SIZE} ${DECODE_PLACEMENT_LINE} ${DECODE_PORT} ${GCS_FUSE_ANNOTATIONS} ${CACHE_VOLUME_MOUNTS} ${CACHE_VOLUMES} ${HF_TOKEN_SECRET} ${SERVICE_ACCOUNT} ${CONTAINER_MEM_LIMIT} ${RAM_CACHE_LIMIT} ${JOB_DURATION_MINUTES} ${KV_CONNECTOR} ${KV_CONNECTOR_MODULE} ${REQUEST_RATE} ${DECODE_REPLICAS}' < "${TEMPLATE_FILE}")

# Save to file if requested
if [ -n "$OUTPUT_FILE" ]; then
    echo "$RENDERED_YAML" > "$OUTPUT_FILE"
    echo "✅ Disaggregated JobSet manifest saved to: $OUTPUT_FILE"
    if [ "$DRY_RUN" == "true" ]; then exit 0; fi
fi

# Print manifest if dry-run
if [ "$DRY_RUN" == "true" ]; then
    echo "$RENDERED_YAML"
    exit 0
fi

# Deploy to Kubernetes
echo "🚀 Applying JobSet manifest to Kubernetes cluster..."
echo "$RENDERED_YAML" | kubectl apply -f -

echo ""
echo "============================================================"
echo " 📋 Useful Commands to Monitor Disaggregated Stack:"
echo "============================================================"
echo " 1. Watch all disagg pods (prefill, decode, proxy, client):"
echo "    kubectl get pods -l jobset.sigs.k8s.io/jobset-name=${JOB_NAME} -w"
echo ""
echo " 2. Stream & Tee all logs into log/*.log<N>:"
echo "    ./gke/bin/tee_logs.sh ${JOB_NAME} [N]"
echo ""
echo " 3. Stream Prefill Server logs (Port ${PREFILL_PORT}):"
echo "    kubectl logs -l jobset.sigs.k8s.io/jobset-name=${JOB_NAME},jobset.sigs.k8s.io/replicatedjob-name=p -c vllm-tpu -f"
echo ""
echo " 4. Stream Decode Server logs (Port ${DECODE_PORT}):"
echo "    kubectl logs -l jobset.sigs.k8s.io/jobset-name=${JOB_NAME},jobset.sigs.k8s.io/replicatedjob-name=d -c vllm-tpu -f"
echo ""
echo " 5. Stream Proxy Router logs (Port 8000):"
echo "    kubectl logs -l jobset.sigs.k8s.io/jobset-name=${JOB_NAME},jobset.sigs.k8s.io/replicatedjob-name=x -f"
echo ""
echo " 6. Stream Benchmark Client progress:"
echo "    kubectl logs -l jobset.sigs.k8s.io/jobset-name=${JOB_NAME},jobset.sigs.k8s.io/replicatedjob-name=client -f"
echo ""
echo " 7. Teardown / Cleanup:"
echo "    ./gke/bin/cleanup.sh ${JOB_NAME}"
echo "    # Or manually via kubectl:"
echo "    kubectl delete jobset ${JOB_NAME} && kubectl delete configmap ${JOB_NAME}-scripts 2>/dev/null || true"
echo "============================================================"
