#!/bin/bash
set -e

# --- AUTO-DETECT USERNAME ---
CURRENT_USER="${USER:-$(whoami)}"
CLEAN_USER=$(echo "$CURRENT_USER" | tr '[:upper:]' '[:lower:]' | tr -dc 'a-z0-9')

# --- DEFAULT CONFIGURATIONS ---
export MODEL_NAME="meta-llama/Llama-3.1-8B-Instruct"
export TOPOLOGY="2x2x1" # Changed default to 2x2x1 for quick single-host testing
IMAGE_TPU_INFERENCE="docker.io/vllm/vllm-tpu:v0.27.0"
IMAGE_TORCHTPU="us-central1-docker.pkg.dev/cloud-ullm-inference-ci-cd/vllm-torchtpu/torchtpu-vllm-prod:latest"
export IMAGE="${IMAGE:-${IMAGE_TPU_INFERENCE}}"
export PLACEMENT_POLICY=""
export SERVICE_ACCOUNT="vllm-sa"
export GCS_BUCKET=""
export DRY_RUN="false"
export OUTPUT_FILE=""
export JOB_DURATION_MINUTES="90"
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
export PREFIX_LEN="${PREFIX_LEN:-0}"
export ENABLE_PREFIX_CACHING="${ENABLE_PREFIX_CACHING:-true}"
export ENABLE_ASYNC="${ENABLE_ASYNC:-true}"
export ENABLE_CHUNKED_PREFILL="${ENABLE_CHUNKED_PREFILL:-true}"
export MAX_BATCHED_TOKENS="${MAX_BATCHED_TOKENS:-2048}"
export REQUEST_RATE="${REQUEST_RATE:-10}"

# Parse long options
POSITIONAL=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run|--generate-only)
            export DRY_RUN="true"
            shift
            ;;
        --torchtpu|--vllm-torchtpu|--prod)
            export IMAGE="${IMAGE_TORCHTPU}"
            shift
            ;;
        --tpu-inference|--vllm-tpu)
            export IMAGE="${IMAGE_TPU_INFERENCE}"
            shift
            ;;
        -i|--image)
            case "$2" in
                torchtpu|vllm-torchtpu|prod) export IMAGE="${IMAGE_TORCHTPU}" ;;
                tpu-inference|vllm-tpu|default) export IMAGE="${IMAGE_TPU_INFERENCE}" ;;
                *) export IMAGE="$2" ;;
            esac
            shift 2
            ;;
        --image=*)
            val="${1#*=}"
            case "$val" in
                torchtpu|vllm-torchtpu|prod) export IMAGE="${IMAGE_TORCHTPU}" ;;
                tpu-inference|vllm-tpu|default) export IMAGE="${IMAGE_TPU_INFERENCE}" ;;
                *) export IMAGE="$val" ;;
            esac
            shift
            ;;
        --prefix-caching)
            export ENABLE_PREFIX_CACHING="true"
            shift
            ;;
        --no-prefix-caching)
            export ENABLE_PREFIX_CACHING="false"
            shift
            ;;
        --prefix-len)
            export PREFIX_LEN="$2"
            shift 2
            ;;
        --prefix-len=*)
            export PREFIX_LEN="${1#*=}"
            shift
            ;;
        --request-rate)
            export REQUEST_RATE="$2"
            shift 2
            ;;
        --request-rate=*)
            export REQUEST_RATE="${1#*=}"
            shift
            ;;
        --max-model-len)
            export MAX_MODEL_LEN="$2"
            shift 2
            ;;
        --max-model-len=*)
            export MAX_MODEL_LEN="${1#*=}"
            shift
            ;;
        --async)
            export ENABLE_ASYNC="true"
            shift
            ;;
        --no-async)
            export ENABLE_ASYNC="false"
            shift
            ;;
        --chunked-prefill)
            export ENABLE_CHUNKED_PREFILL="true"
            shift
            ;;
        --no-chunked-prefill)
            export ENABLE_CHUNKED_PREFILL="false"
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
    echo " TPU v7 vLLM Benchmark Deployment Script"
    echo "============================================================"
    echo "Usage: $0 [-m model_name] [-t topology] [-i image] [-p placement_policy] [-s service_account] [-b gcs_bucket] [-D duration_mins] [-r request_rate] [-d] [-o output.yaml]"
    echo "  -m  Model name from HuggingFace or GCS (default: meta-llama/Llama-3.1-8B-Instruct)"
    echo "  -t  TPU v7 topology shape (default: 2x2x1)"
    echo "  -i  Docker container image or alias (default: docker.io/vllm/vllm-tpu:v0.27.0)"
    echo "      Quick aliases: 'torchtpu', 'tpu-inference'"
    echo "  --torchtpu                  Quick switch to Google TorchTPU image (torchtpu-vllm-prod:latest)"
    echo "  --tpu-inference             Quick switch to upstream tpu-inference image (vllm-tpu:v0.27.0)"
    echo "  -p  Placement policy name (optional: auto-fetched from gcloud if omitted)"
    echo "  -s  Kubernetes ServiceAccount name with Workload Identity (default: vllm-sa)"
    echo "  -b  GCS bucket/path (optional: persist HuggingFace downloads to GCS for fast reuse)"
    echo "  -D  Job declared duration in minutes for Kueue (default: 90)"
    echo "  -r  Benchmark client request rate in req/s (default: 10)"
    echo "  -d  Dry-run / Generate YAML only (do not deploy to Kubernetes)"
    echo "  -o  Output filepath to save the rendered YAML"
    echo ""
    echo "Serving & Prefix Caching Flags:"
    echo "  --prefix-caching / --no-prefix-caching  Enable/disable Automatic Prefix Caching (default: enabled)"
    echo "  --prefix-len <N>                        Shared prefix token length for client benchmarks (default: 0)"
    echo "  --request-rate <RPS>                    Target request rate in req/s (default: 10)"
    echo "  --max-model-len <N>                     Maximum model context length (default: 8192)"
    echo "  --async / --no-async                    Async scheduling engine (default: true)"
    echo "  --chunked-prefill / --no-chunked-prefill Enable chunked prefill (default: true)"
    echo ""
    echo "Execution Modes & Examples:"
    echo ""
    echo "  1. Direct GCS Model URI (-m gs://...):"
    echo "     $0 -t 2x2x4 -m gs://<bucket-name>/Llama-3.1-70B-Instruct -s vllm-sa"
    echo ""
    echo "  2. HuggingFace Model with GCS Cache Persistence (-m ... -b gs://...):"
    echo "     $0 -t 2x2x4 -m meta-llama/Llama-3.1-70B-Instruct -b gs://<bucket-name>/hf-cache -s vllm-sa"
    echo "     $0 -t 2x2x1 -m meta-llama/Llama-3.1-8B-Instruct -b gs://<bucket-name>/hf-cache -s vllm-sa"
    echo ""
    echo "  3. Automatic Prefix Caching (APC) Benchmark Comparison:"
    echo "     # Baseline with APC disabled:"
    echo "     $0 -t 2x2x1 -m meta-llama/Llama-3.1-8B-Instruct --no-prefix-caching --prefix-len 1024"
    echo "     # Evaluation with APC enabled:"
    echo "     $0 -t 2x2x1 -m meta-llama/Llama-3.1-8B-Instruct --prefix-caching --prefix-len 1024"
    echo ""
    echo "  4. Switch Between TPU Images:"
    echo "     # Run with default upstream tpu-inference image:"
    echo "     $0 -t 2x2x1 -m meta-llama/Llama-3.1-8B-Instruct"
    echo "     # Quick switch to Google TorchTPU production image:"
    echo "     $0 -t 2x2x1 -m meta-llama/Llama-3.1-8B-Instruct --torchtpu"
    echo "     # Or using -i alias:"
    echo "     $0 -t 2x2x1 -m meta-llama/Llama-3.1-8B-Instruct -i torchtpu"
    echo ""
    echo "  5. Generate YAML Only (Inspect specification details):"
    echo "     $0 -t 2x2x1 -m Qwen/Qwen3.5-4B -b gs://<bucket-name>/hf-cache -d"
    echo "     $0 -t 2x2x4 -m meta-llama/Llama-3.1-70B-Instruct -i my-image:tag -d -o jobset.yaml"
    echo ""
    echo "Recommended Models (-m):"
    echo "  Llama Family (Gated - Requires HF Token):"
    echo "    meta-llama/Llama-3.1-8B-Instruct   (Fits on 1 VM,   -t 2x2x1)"
    echo "    meta-llama/Llama-3.1-70B-Instruct  (Requires 4+ VMs, -t 2x2x4)"
    echo "    meta-llama/Llama-Guard-4-12B       (Fits on 1 VM,   -t 2x2x1)"
    echo ""
    echo "  Qwen Family (Open):"
    echo "    Qwen/Qwen3.5-4B                    (Fits on 1 VM,   -t 2x2x1)"
    echo "    Qwen/Qwen3-32B                     (Requires 2+ VMs, -t 2x2x2)"
    echo ""
    echo "  Gemma Family (Gated):"
    echo "    google/gemma-2-9b-it               (Fits on 1 VM,   -t 2x2x1)"
    echo ""
    echo "  DeepSeek (Open):"
    echo "    deepseek-ai/DeepSeek-V3            (Requires massive multi-host slice)"
    echo "============================================================"
    exit 1
}

while getopts "m:t:i:p:s:b:D:o:r:dgh" opt; do
    case ${opt} in
        m ) export MODEL_NAME=$OPTARG ;;
        t ) export TOPOLOGY=$OPTARG ;;
        i )
            case "$OPTARG" in
                torchtpu|vllm-torchtpu|prod) export IMAGE="${IMAGE_TORCHTPU}" ;;
                tpu-inference|vllm-tpu|default) export IMAGE="${IMAGE_TPU_INFERENCE}" ;;
                *) export IMAGE="$OPTARG" ;;
            esac
            ;;
        p ) export PLACEMENT_POLICY=$OPTARG ;;
        s ) export SERVICE_ACCOUNT=$OPTARG ;;
        b ) export GCS_BUCKET=$OPTARG ;;
        D ) export JOB_DURATION_MINUTES=$OPTARG ;;
        o ) export OUTPUT_FILE=$OPTARG ;;
        r ) export REQUEST_RATE=$OPTARG ;;
        d | g ) export DRY_RUN="true" ;;
        h ) usage ;;
        * ) usage ;;
    esac
done

# --- FEATURE FLAGS & SERVING OPTIONS ---
FEATURE_FLAGS=""
if [ "$ENABLE_ASYNC" == "true" ]; then
    FEATURE_FLAGS="${FEATURE_FLAGS} --async-scheduling"
else
    FEATURE_FLAGS="${FEATURE_FLAGS} --no-async-scheduling"
fi
if [ "$ENABLE_PREFIX_CACHING" == "true" ]; then
    FEATURE_FLAGS="${FEATURE_FLAGS} --enable-prefix-caching"
else
    FEATURE_FLAGS="${FEATURE_FLAGS} --no-enable-prefix-caching"
fi
if [ "$ENABLE_CHUNKED_PREFILL" == "true" ]; then
    FEATURE_FLAGS="${FEATURE_FLAGS} --enable-chunked-prefill --max-num-batched-tokens=${MAX_BATCHED_TOKENS}"
else
    FEATURE_FLAGS="${FEATURE_FLAGS} --no-enable-chunked-prefill"
fi
export FEATURE_FLAGS

if [ -n "$PREFIX_LEN" ] && [ "$PREFIX_LEN" -gt 0 ] 2>/dev/null; then
    export PREFIX_LEN_FLAG="--random-prefix-len=${PREFIX_LEN}"
else
    export PREFIX_LEN_FLAG=""
fi

# --- AUTO-CALCULATE HARDWARE LIMITS ---
IFS='x' read -r TOPO_X TOPO_Y TOPO_Z <<< "$TOPOLOGY"
TOTAL_CHIPS=$((TOPO_X * TOPO_Y * TOPO_Z))
export NUM_VMS=$((TOTAL_CHIPS / 4))
export TP_SIZE=$((TOTAL_CHIPS * 2))
if [ -z "$TPU_ACCELERATOR_TYPE" ]; then
    export TPU_ACCELERATOR_TYPE="tpu7x-${TP_SIZE}"
fi

# Calculate TPU process and chip bounds for JAX/libtpu multi-host SPMD
CHIP_X=2
CHIP_Y=2
CHIP_Z=1
PROC_X=$(( (TOPO_X + CHIP_X - 1) / CHIP_X ))
PROC_Y=$(( (TOPO_Y + CHIP_Y - 1) / CHIP_Y ))
PROC_Z=$(( (TOPO_Z + CHIP_Z - 1) / CHIP_Z ))

if [ -z "$TPU_CHIPS_PER_PROCESS_BOUNDS" ]; then
    export TPU_CHIPS_PER_PROCESS_BOUNDS="${CHIP_X},${CHIP_Y},${CHIP_Z}"
fi
if [ -z "$TPU_PROCESS_BOUNDS" ]; then
    export TPU_PROCESS_BOUNDS="${PROC_X},${PROC_Y},${PROC_Z}"
fi

# --- DYNAMIC RAM TUNING BASED ON MODEL SIZE ---
LOWER_MODEL=$(echo "$MODEL_NAME" | tr '[:upper:]' '[:lower:]')

# Extract parameter size if present in model name (e.g. 4b, 8b, 9b, 12b, 27b, 32b, 70b, 72b, 235b, 405b)
if [[ "$LOWER_MODEL" =~ ([0-9]+(\.[0-9]+)?)b ]]; then
    PARAM_COUNT="${BASH_REMATCH[1]}"
    # Calculate RAM Cache Size in GiB (~2.4x parameter count for safety headroom, min 32GiB, max 650GiB)
    CALC_CACHE_GB=$(awk -v p="$PARAM_COUNT" 'BEGIN { c = int(p * 2.4); if (c < 32) c = 32; if (c > 650) c = 650; printf "%d", c }')
    # Container Memory Limit: Cache size + 40GB base OS/vLLM/JAX overhead (capped at 750GB)
    CALC_CONT_GB=$(awk -v c="$CALC_CACHE_GB" 'BEGIN { m = c + 40; if (m > 750) m = 750; printf "%d", m }')
    # GCS FUSE Sidecar Memory Limit: Cache size + 20GB buffer
    CALC_GCSFUSE_GB=$(awk -v c="$CALC_CACHE_GB" 'BEGIN { s = c + 20; if (s > 700) s = 700; printf "%d", s }')
else
    # Safe defaults when parameter size is not detectable in the name
    if [ "$NUM_VMS" -eq 1 ]; then
        CALC_CACHE_GB=40
        CALC_CONT_GB=64
        CALC_GCSFUSE_GB=64
    else
        CALC_CACHE_GB=160
        CALC_CONT_GB=200
        CALC_GCSFUSE_GB=180
    fi
fi

export RAM_CACHE_LIMIT="${CALC_CACHE_GB}Gi"
export CONTAINER_MEM_LIMIT="${CALC_CONT_GB}Gi"
export GCSFUSE_MEM_LIMIT="${CALC_GCSFUSE_GB}Gi"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# --- DETERMINE DEPLOYMENT TYPE ---
if [ "$NUM_VMS" -eq 1 ]; then
    echo "Detected Single-Host Topology ($TOPOLOGY)."
    TEMPLATE_FILE="${SCRIPT_DIR}/benchmark_singlehost_template.yaml"
    # Placement policies are not supported/required for single-host topologies
    export PLACEMENT_POLICY="" 
else
    echo "Detected Multi-Host Topology ($TOPOLOGY)."
    TEMPLATE_FILE="${SCRIPT_DIR}/benchmark_multihost_template.yaml"
    
    # Auto-fetch placement policy if not provided
    if [ -z "$PLACEMENT_POLICY" ]; then
        echo "Searching for placement policy for topology ${TOPOLOGY}..."
        export PLACEMENT_POLICY=$(gcloud compute resource-policies list \
            --filter="region: us-central1" \
            --project=cloud-tpu-shared-capacity \
            --format="value(name)" | grep "\-${TOPOLOGY}-" | head -n 1)

        if [ -z "$PLACEMENT_POLICY" ]; then
            echo "Error: Could not find an existing placement policy for topology ${TOPOLOGY}."
            echo "Please verify the topology shape is valid and supported by your cluster."
            exit 1
        fi
    fi
fi

# --- GENERATE KUBERNETES NAMES ---
RANDOM_SUFFIX=$(cat /dev/urandom | tr -dc 'a-z0-9' | fold -w 5 | head -n 1)
export JOB_NAME="${CLEAN_USER}-test-${RANDOM_SUFFIX}"
export HF_TOKEN_SECRET="${CLEAN_USER}-test-token"

# --- CONFIGURE GCS FUSE CSI MOUNT & MODEL NAMES ---
if [[ "$MODEL_NAME" =~ ^gs://([^/]+)/(.*) ]]; then
    # Case 1: Direct GCS Model URI (gs://bucket/model_dir)
    export GCS_BUCKET_NAME="${BASH_REMATCH[1]}"
    export GCS_MODEL_SUBPATH="${BASH_REMATCH[2]}"
    export LOCAL_MODEL_PATH="/gcs-models/${GCS_MODEL_SUBPATH}"

    # Infer standard HuggingFace repo name for tokenizer resolution on benchmark client
    if [[ "$GCS_MODEL_SUBPATH" =~ ^[Ll]lama ]]; then
        export SERVED_MODEL_NAME="meta-llama/${GCS_MODEL_SUBPATH}"
    elif [[ "$GCS_MODEL_SUBPATH" =~ ^[Gg]emma ]]; then
        export SERVED_MODEL_NAME="google/${GCS_MODEL_SUBPATH}"
    elif [[ "$GCS_MODEL_SUBPATH" =~ ^[Qq]wen ]]; then
        export SERVED_MODEL_NAME="Qwen/${GCS_MODEL_SUBPATH}"
    elif [[ "$GCS_MODEL_SUBPATH" =~ ^[Dd]eep[Ss]eek ]]; then
        export SERVED_MODEL_NAME="deepseek-ai/${GCS_MODEL_SUBPATH}"
    else
        export SERVED_MODEL_NAME="${GCS_MODEL_SUBPATH}"
    fi

    export GCS_FUSE_ANNOTATIONS=$(cat <<EOF
            annotations:
              gke-gcsfuse/volumes: "true"
              gke-gcsfuse/memory-limit: "${GCSFUSE_MEM_LIMIT}"
              gke-gcsfuse/cpu-limit: "8"
EOF
)
    export CACHE_VOLUME_MOUNTS=$(cat <<'EOF'
              - name: gcs-fuse-models
                mountPath: /gcs-models
                readOnly: true
EOF
)
    export CACHE_VOLUMES=$(cat <<EOF
            - name: gcs-fuse-models
              csi:
                driver: gcsfuse.csi.storage.gke.io
                volumeAttributes:
                  bucketName: ${GCS_BUCKET_NAME}
                  mountOptions: "implicit-dirs,file-cache:max-size-mb:-1,file-cache:enable-parallel-downloads:true,file-cache:parallel-downloads-per-file:100"
            - name: gke-gcsfuse-cache
              emptyDir:
                medium: Memory
                sizeLimit: ${RAM_CACHE_LIMIT}
EOF
)
elif [ -n "$GCS_BUCKET" ]; then
    # Case 2: HuggingFace Hub Model with GCS Cache Persistence (-b gs://bucket/path or -b bucket/path)
    CLEAN_GCS_BUCKET="${GCS_BUCKET#gs://}"
    CLEAN_GCS_BUCKET="${CLEAN_GCS_BUCKET%/}"

    if [[ "$CLEAN_GCS_BUCKET" =~ ^([^/]+)(/(.*))?$ ]]; then
        export GCS_BUCKET_NAME="${BASH_REMATCH[1]}"
        GCS_CACHE_SUBPATH="${BASH_REMATCH[3]}"
    else
        export GCS_BUCKET_NAME="${CLEAN_GCS_BUCKET}"
        GCS_CACHE_SUBPATH=""
    fi

    BASE_OPTS="implicit-dirs,file-cache:max-size-mb:-1,file-cache:enable-parallel-downloads:true,file-cache:parallel-downloads-per-file:100"
    if [ -n "$GCS_CACHE_SUBPATH" ]; then
        MOUNT_OPTS="${BASE_OPTS},only-dir=${GCS_CACHE_SUBPATH}"
    else
        MOUNT_OPTS="${BASE_OPTS}"
    fi

    export LOCAL_MODEL_PATH="${MODEL_NAME}"
    export SERVED_MODEL_NAME="${MODEL_NAME}"

    export GCS_FUSE_ANNOTATIONS=$(cat <<EOF
            annotations:
              gke-gcsfuse/volumes: "true"
              gke-gcsfuse/memory-limit: "${GCSFUSE_MEM_LIMIT}"
              gke-gcsfuse/cpu-limit: "8"
EOF
)
    export CACHE_VOLUME_MOUNTS=$(cat <<'EOF'
              - name: hf-gcs-cache
                mountPath: /root/.cache/huggingface
EOF
)
    export CACHE_VOLUMES=$(cat <<EOF
            - name: hf-gcs-cache
              csi:
                driver: gcsfuse.csi.storage.gke.io
                volumeAttributes:
                  bucketName: ${GCS_BUCKET_NAME}
                  mountOptions: "${MOUNT_OPTS}"
            - name: gke-gcsfuse-cache
              emptyDir:
                medium: Memory
                sizeLimit: ${RAM_CACHE_LIMIT}
EOF
)
else
    # Case 3: HuggingFace Hub Model with Pure In-Memory RAM Disk (-b omitted)
    export GCS_BUCKET_NAME=""
    export LOCAL_MODEL_PATH="${MODEL_NAME}"
    export SERVED_MODEL_NAME="${MODEL_NAME}"
    export GCS_FUSE_ANNOTATIONS=""
    export CACHE_VOLUME_MOUNTS=$(cat <<'EOF'
              - name: cache-ramdisk
                mountPath: /root/.cache
EOF
)
    export CACHE_VOLUMES=$(cat <<EOF
            - name: cache-ramdisk
              emptyDir:
                medium: Memory
                sizeLimit: ${RAM_CACHE_LIMIT}
EOF
)
fi

echo "============================================================"
echo "Preparing to deploy vLLM JobSet: $JOB_NAME"
echo "Executed by: $CURRENT_USER"
if [[ "$MODEL_NAME" == gs://* ]]; then
    echo "Model Source: Google Cloud Storage (Mounted at $LOCAL_MODEL_PATH)"
    echo "GCS Bucket: $GCS_BUCKET_NAME"
    echo "Served/Tokenizer Name: $SERVED_MODEL_NAME"
    echo "GCS Cache RAM Limit: $RAM_CACHE_LIMIT"
elif [ -n "$GCS_BUCKET" ]; then
    echo "Model Source: HuggingFace Hub ($MODEL_NAME)"
    echo "GCS Cache Persistence: Enabled (gs://${CLEAN_GCS_BUCKET} mounted at /root/.cache/huggingface)"
    echo "Served/Tokenizer Name: $SERVED_MODEL_NAME"
    echo "GCS Cache RAM Limit: $RAM_CACHE_LIMIT"
else
    echo "Model Source: HuggingFace Hub ($MODEL_NAME)"
    echo "Served/Tokenizer Name: $SERVED_MODEL_NAME"
    echo "Cache Storage: In-Memory RAM Disk ($RAM_CACHE_LIMIT at /root/.cache)"
fi
echo "Container Memory Limit: $CONTAINER_MEM_LIMIT"
echo "Container Image: $IMAGE"
echo "Topology: $TOPOLOGY"
echo "Accelerator Type: $TPU_ACCELERATOR_TYPE"
echo "Chip Bounds: $TPU_CHIPS_PER_PROCESS_BOUNDS | Process Bounds: $TPU_PROCESS_BOUNDS"
echo "Service Account: $SERVICE_ACCOUNT"
echo "Max Model Len: $MAX_MODEL_LEN"
echo "Prefix Caching: $ENABLE_PREFIX_CACHING (Shared Prefix: ${PREFIX_LEN:-0} tokens)"
echo "Request Rate: $REQUEST_RATE req/s"
echo "Feature Flags: $FEATURE_FLAGS"
if [ "$NUM_VMS" -gt 1 ]; then echo "Placement Policy: $PLACEMENT_POLICY"; fi
echo "Calculated Requirements: $NUM_VMS Virtual Machines, Tensor Parallel Size: $TP_SIZE"
echo "Job Max Duration: ${JOB_DURATION_MINUTES} minutes"
echo "Using Template: $TEMPLATE_FILE"
echo "============================================================"

# Inject the environment variables into the chosen template
RENDERED_YAML=$(envsubst '${JOB_NAME} ${MODEL_NAME} ${LOCAL_MODEL_PATH} ${SERVED_MODEL_NAME} ${GCS_FUSE_ANNOTATIONS} ${CACHE_VOLUME_MOUNTS} ${CACHE_VOLUMES} ${TOPOLOGY} ${NUM_VMS} ${TP_SIZE} ${HF_TOKEN_SECRET} ${PLACEMENT_POLICY} ${SERVICE_ACCOUNT} ${TPU_ACCELERATOR_TYPE} ${TPU_CHIPS_PER_PROCESS_BOUNDS} ${TPU_PROCESS_BOUNDS} ${CONTAINER_MEM_LIMIT} ${RAM_CACHE_LIMIT} ${JOB_DURATION_MINUTES} ${IMAGE} ${MAX_MODEL_LEN} ${FEATURE_FLAGS} ${PREFIX_LEN_FLAG} ${REQUEST_RATE}' < "$TEMPLATE_FILE")

if [ "$DRY_RUN" == "true" ]; then
    if [ -n "$OUTPUT_FILE" ]; then
        echo "$RENDERED_YAML" > "$OUTPUT_FILE"
        echo ""
        echo "✅ Generated YAML saved to: $OUTPUT_FILE"
    else
        echo ""
        echo "==================== GENERATED YAML ===================="
        echo "$RENDERED_YAML"
        echo "========================================================"
    fi
    exit 0
fi

echo "$RENDERED_YAML" | kubectl apply -f -

echo ""
echo "Deployed successfully!"
echo "------------------------------------------------------------"
echo "To monitor ALL pods spinning up:"
echo "  kubectl get pods -l jobset.sigs.k8s.io/jobset-name=${JOB_NAME} -w"
echo ""
echo "To watch the vLLM Server logs during XLA compilation (after the pod is running):"
echo "  kubectl logs -l jobset.sigs.k8s.io/jobset-name=${JOB_NAME},jobset.sigs.k8s.io/replicatedjob-name=server,batch.kubernetes.io/job-completion-index=0 -f"
echo ""
echo "To watch the Benchmark Client logs (once the server is ready):"
echo "  kubectl logs -l jobset.sigs.k8s.io/jobset-name=${JOB_NAME},jobset.sigs.k8s.io/replicatedjob-name=client -f"
echo ""
echo "Teardown / Cleanup:"
echo "  ./gke/bin/cleanup.sh ${JOB_NAME}"
echo "  # Or manually via kubectl:"
echo "  kubectl delete jobset ${JOB_NAME} && kubectl delete configmap ${JOB_NAME}-scripts 2>/dev/null || true"
echo "============================================================"

