#!/bin/bash
# ==============================================================================
# TPU vLLM Benchmark Runner (Helm-Powered)
# Supports both Monolithic (Aggregated) and Disaggregated (P/D) Architectures
# ==============================================================================
set -e

# --- AUTO-DETECT USER & DEFAULTS ---
CURRENT_USER="${USER:-$(whoami)}"
CLEAN_USER=$(echo "$CURRENT_USER" | tr '[:upper:]' '[:lower:]' | tr -dc 'a-z0-9')
RANDOM_SUFFIX=$(head /dev/urandom | tr -dc a-z0-9 | head -c 5 ; echo '')
DEFAULT_RELEASE="${CLEAN_USER}-test-${RANDOM_SUFFIX}"

MODE="aggregated"
MODEL_NAME="meta-llama/Llama-3.1-8B-Instruct"
TOPOLOGY="2x2x1"
PREFILL_TOPOLOGY="2x2x1"
DECODE_TOPOLOGY="2x2x2"
DECODE_REPLICAS="1"
REQUEST_RATE="10.0"
BENCHMARK_PREFIX_LEN="0"
KV_CONNECTOR=""
IMAGE_TPU_INFERENCE="docker.io/vllm/vllm-tpu:v0.27.0"
IMAGE_TORCHTPU="us-central1-docker.pkg.dev/cloud-ullm-inference-ci-cd/vllm-torchtpu/torchtpu-vllm-prod:latest"
IMAGE="${IMAGE:-${IMAGE_TPU_INFERENCE}}"
SERVICE_ACCOUNT="vllm-sa"
GCS_BUCKET=""
PLACEMENT_POLICY=""
PREFILL_PLACEMENT_POLICY=""
DECODE_PLACEMENT_POLICY=""
DURATION_MINUTES="90"
MAX_MODEL_LEN="8192"
RELEASE_NAME=""
OUTPUT_FILE=""
VALUES_ONLY=false
DRY_RUN=false

# Feature Defaults (Bulk & Low-Latency)
FEATURE_ASYNC_SCHEDULING=true
FEATURE_PREFIX_CACHING=true
FEATURE_CHUNKED_PREFILL=true
FEATURE_MAX_BATCHED_TOKENS=2048
FEATURE_WIDE_EP=false
FEATURE_SPEC_DECODING=false
FEATURE_DRAFT_MODEL=""
FEATURE_SPEC_TOKENS=3
FEATURE_KV_OFFLOAD=false
FEATURE_OFFLOAD_CONNECTOR="RaidenOffloadConnector"
FEATURE_STRUCTURED_OUTPUT=false

usage() {
    echo "=================================================================="
    echo " TPU vLLM Benchmark Runner (Helm-Powered)"
    echo "=================================================================="
    echo "Usage: $0 [options]"
    echo ""
    echo "Common Options:"
    echo "  -m, --model <model>         Model name / HF repo / GCS URI (default: meta-llama/Llama-3.1-8B-Instruct)"
    echo "  -i, --image <image>         Docker container image or alias (default: docker.io/vllm/vllm-tpu:v0.27.0)"
    echo "                              Quick aliases: 'torchtpu', 'tpu-inference'"
    echo "  --torchtpu                  Quick switch to Google TorchTPU image (torchtpu-vllm-prod:latest)"
    echo "  --tpu-inference             Quick switch to upstream tpu-inference image (vllm-tpu:v0.27.0)"
    echo "  -b, --bucket <bucket>       GCS bucket path for caching (e.g. gs://<bucket>/hf-cache)"
    echo "  -r, --release <release>     Helm release name (default: auto-generated '${DEFAULT_RELEASE}')"
    echo "  -s, --sa <sa>               Kubernetes ServiceAccount (default: vllm-sa)"
    echo "  -R, --rate <rps>            Benchmark request rate in req/s (default: 10.0)"
    echo "  -L, --prefix-len <N>        Fixed shared prefix length per request for Prefix Caching testing (default: 0)"
    echo "  -D, --duration <mins>       Job duration in minutes (default: 90)"
    echo "  -o, --output <file>         Save calculated values.yaml to file (skips deployment)"
    echo "  -v, --values-only           Print calculated values.yaml to stdout without deploying"
    echo "  -g, --dry-run               Render the final JobSet manifest (helm template) to stdout"
    echo "  -h, --help                  Show this help message"
    echo ""
    echo "Monolithic Mode Options (default):"
    echo "  -t, --topology <topology>   TPU v7 topology shape (default: 2x2x1)"
    echo "  -P, --policy <policy>       Placement policy name (auto-fetched via gcloud if omitted on multihost)"
    echo ""
    echo "Advanced Serving & Feature Toggles:"
    echo "  --async-scheduling / --no-async-scheduling   Enable/disable async CPU/TPU scheduling (default: enabled)"
    echo "  --prefix-caching / --no-prefix-caching       Enable/disable Automatic Prefix Caching (default: enabled)"
    echo "  --chunked-prefill / --no-chunked-prefill     Enable/disable chunked prefill (default: enabled)"
    echo "  --max-batched-tokens <N>                     Max tokens per chunked prefill step (default: 2048)"
    echo "  --wide-ep / --ep                             Enable Wide Expert Parallelism for MoE (default: disabled)"
    echo "  --spec-decode / --draft-model <repo>         Enable Eagle3 Speculative Decoding (default: disabled)"
    echo "  --spec-tokens <N>                            Candidate tokens proposed per step (default: 3)"
    echo "  --kv-offload / --no-kv-offload               Enable/disable Host DRAM KV Cache Offloading (default: disabled)"
    echo "  --offload-connector <connector>              KV Offload connector (default: RaidenOffloadConnector)"
    echo "  --structured-output                          Enable structured decoding / JSON schema enforcement"
    echo ""
    echo "Disaggregated Mode Options:"
    echo "  --disagg                    Enable Disaggregated Prefill/Decode architecture"
    echo "  -p, --prefill <topology>    Prefill TPU v7 topology (default: 2x2x1)"
    echo "  -d, --decode <topology>     Decode TPU v7 topology (default: 2x2x2)"
    echo "  -N, --decode-replicas <N>   Number of Decode slices to scale horizontally (default: 1)"
    echo "  -c, --connector <connector> KV Connector: TPUConnector or TPURaidenConnector (auto-detected if omitted)"
    echo ""
    echo "Examples:"
    echo "  1. Monolithic Single-host (TPU 2x2x1):"
    echo "     $0 -t 2x2x1 -m meta-llama/Llama-3.1-8B-Instruct -b gs://my-bucket/hf-cache"
    echo ""
    echo "  2. Monolithic Multi-host (TPU 2x2x4 / 4 VMs):"
    echo "     $0 -t 2x2x4 -m meta-llama/Llama-3.1-70B-Instruct -b gs://my-bucket/hf-cache"
    echo ""
    echo "  3. Disaggregated Asymmetric Serving (Prefill 2x2x1, Decode 2x2x2 @ 35 RPS):"
    echo "     $0 --disagg -p 2x2x1 -d 2x2x2 -m meta-llama/Llama-3.1-8B-Instruct -R 35.0 -b gs://my-bucket/hf-cache"
    echo ""
    echo "  4. Disaggregated Symmetric Serving (Prefill 2x2x1, Decode 2x2x1):"
    echo "     $0 --disagg -p 2x2x1 -d 2x2x1 -m meta-llama/Llama-3.1-8B-Instruct"
    echo ""
    echo "  5. Disaggregated Horizontal Decode Scaling (1x Prefill 2x2x1, 2x Decode 2x2x1 @ 35 RPS):"
    echo "     $0 --disagg -p 2x2x1 -d 2x2x1 -N 2 -m meta-llama/Llama-3.1-8B-Instruct -R 35.0 -b gs://my-bucket/hf-cache"
    echo ""
    echo "  6. Dry-run to view full rendered YAML:"
    echo "     $0 --disagg -p 2x2x1 -d 2x2x2 -m meta-llama/Llama-3.1-8B-Instruct -R 35.0 -g"
    echo "=================================================================="
    exit 0
}

# Single unified CLI option parsing
while [[ $# -gt 0 ]]; do
    case "$1" in
        --disagg|--disaggregated)
            MODE="disaggregated"
            shift
            ;;
        -m|--model)
            MODEL_NAME="$2"
            shift 2
            ;;
        -t|--topology)
            TOPOLOGY="$2"
            shift 2
            ;;
        -p|--prefill)
            PREFILL_TOPOLOGY="$2"
            MODE="disaggregated"
            shift 2
            ;;
        -d|--decode)
            DECODE_TOPOLOGY="$2"
            MODE="disaggregated"
            shift 2
            ;;
        -N|--decode-replicas|--d-replicas)
            DECODE_REPLICAS="$2"
            MODE="disaggregated"
            shift 2
            ;;
        -c|--connector|--kv-connector)
            KV_CONNECTOR="$2"
            shift 2
            ;;
        --torchtpu|--vllm-torchtpu|--prod)
            IMAGE="${IMAGE_TORCHTPU}"
            shift
            ;;
        --tpu-inference|--vllm-tpu)
            IMAGE="${IMAGE_TPU_INFERENCE}"
            shift
            ;;
        -i|--image)
            case "$2" in
                torchtpu|vllm-torchtpu|prod)
                    IMAGE="${IMAGE_TORCHTPU}"
                    ;;
                tpu-inference|vllm-tpu|default)
                    IMAGE="${IMAGE_TPU_INFERENCE}"
                    ;;
                *)
                    IMAGE="$2"
                    ;;
            esac
            shift 2
            ;;
        --image=*)
            val="${1#*=}"
            case "$val" in
                torchtpu|vllm-torchtpu|prod)
                    IMAGE="${IMAGE_TORCHTPU}"
                    ;;
                tpu-inference|vllm-tpu|default)
                    IMAGE="${IMAGE_TPU_INFERENCE}"
                    ;;
                *)
                    IMAGE="$val"
                    ;;
            esac
            shift
            ;;
        -b|--bucket)
            GCS_BUCKET="$2"
            shift 2
            ;;
        -r|--release)
            RELEASE_NAME="$2"
            shift 2
            ;;
        -R|--rate|--request-rate)
            REQUEST_RATE="$2"
            shift 2
            ;;
        -L|--prefix-len|--random-prefix-len)
            BENCHMARK_PREFIX_LEN="$2"
            shift 2
            ;;
        -l|--max-model-len)
            MAX_MODEL_LEN="$2"
            shift 2
            ;;
        -s|--sa|--service-account)
            SERVICE_ACCOUNT="$2"
            shift 2
            ;;
        -P|--policy|--placement-policy)
            PLACEMENT_POLICY="$2"
            shift 2
            ;;
        -D|--duration)
            DURATION_MINUTES="$2"
            shift 2
            ;;
        -o|--output)
            OUTPUT_FILE="$2"
            shift 2
            ;;
        -v|--values-only)
            VALUES_ONLY=true
            shift
            ;;
        -g|--dry-run|--template|--generate-only)
            DRY_RUN=true
            shift
            ;;
        --async-scheduling)
            FEATURE_ASYNC_SCHEDULING=true
            shift
            ;;
        --no-async-scheduling)
            FEATURE_ASYNC_SCHEDULING=false
            shift
            ;;
        --prefix-caching)
            FEATURE_PREFIX_CACHING=true
            shift
            ;;
        --no-prefix-caching)
            FEATURE_PREFIX_CACHING=false
            shift
            ;;
        --chunked-prefill)
            FEATURE_CHUNKED_PREFILL=true
            shift
            ;;
        --no-chunked-prefill)
            FEATURE_CHUNKED_PREFILL=false
            shift
            ;;
        --max-num-batched-tokens|--max-batched-tokens)
            FEATURE_MAX_BATCHED_TOKENS="$2"
            shift 2
            ;;
        --wide-ep|--ep|--expert-parallel)
            FEATURE_WIDE_EP=true
            shift
            ;;
        --no-wide-ep|--no-ep)
            FEATURE_WIDE_EP=false
            shift
            ;;
        --spec-decoding|--spec-decode)
            FEATURE_SPEC_DECODING=true
            shift
            ;;
        --draft-model|--spec-model)
            FEATURE_SPEC_DECODING=true
            FEATURE_DRAFT_MODEL="$2"
            shift 2
            ;;
        --spec-tokens|--num-spec-tokens)
            FEATURE_SPEC_TOKENS="$2"
            shift 2
            ;;
        --kv-offload|--offload)
            FEATURE_KV_OFFLOAD=true
            shift
            ;;
        --no-kv-offload)
            FEATURE_KV_OFFLOAD=false
            shift
            ;;
        --offload-connector)
            FEATURE_KV_OFFLOAD=true
            FEATURE_OFFLOAD_CONNECTOR="$2"
            shift 2
            ;;
        --structured-output|--structured-decoding)
            FEATURE_STRUCTURED_OUTPUT=true
            shift
            ;;
        --no-structured-output)
            FEATURE_STRUCTURED_OUTPUT=false
            shift
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo "Error: Unknown option $1"
            usage
            ;;
    esac
done

if [ -z "$RELEASE_NAME" ]; then
    RELEASE_NAME="$DEFAULT_RELEASE"
fi

# Function to calculate VMs and TP
calc_topology() {
    local topo="$1"
    IFS='x' read -r tx ty tz <<< "$topo"
    local chips=$((tx * ty * tz))
    local vms=$((chips / 4))
    local tp=$((chips * 2))
    echo "$vms $tp"
}

# Auto-discover placement policies for multihost topologies
discover_placement_policy() {
    local topo="$1"
    gcloud compute resource-policies list \
        --filter="region: us-central1" \
        --project=cloud-tpu-shared-capacity \
        --format="value(name)" 2>/dev/null | grep "\-${topo}-" | head -n 1 || true
}

if [ "$MODE" = "aggregated" ]; then
    read -r NUM_VMS TP_SIZE <<< "$(calc_topology "$TOPOLOGY")"
    if [ "$NUM_VMS" -gt 1 ] && [ -z "$PLACEMENT_POLICY" ]; then
        PLACEMENT_POLICY=$(discover_placement_policy "$TOPOLOGY")
    fi
else
    read -r PREFILL_NUM_VMS PREFILL_TP_SIZE <<< "$(calc_topology "$PREFILL_TOPOLOGY")"
    read -r DECODE_NUM_VMS DECODE_TP_SIZE <<< "$(calc_topology "$DECODE_TOPOLOGY")"

    if [ "$PREFILL_NUM_VMS" -gt 1 ]; then
        PREFILL_PLACEMENT_POLICY=$(discover_placement_policy "$PREFILL_TOPOLOGY")
    fi
    if [ "$DECODE_NUM_VMS" -gt 1 ]; then
        DECODE_PLACEMENT_POLICY=$(discover_placement_policy "$DECODE_TOPOLOGY")
    fi
fi

# --- DYNAMIC MEMORY & CACHE TUNING BASED ON MODEL SIZE ---
LOWER_MODEL=$(echo "$MODEL_NAME" | tr '[:upper:]' '[:lower:]')

if [[ "$LOWER_MODEL" =~ ([0-9]+(\.[0-9]+)?)b ]]; then
    PARAM_COUNT="${BASH_REMATCH[1]}"
    CALC_CACHE_GB=$(awk -v p="$PARAM_COUNT" 'BEGIN { c = int(p * 2.4); if (c < 32) c = 32; if (c > 650) c = 650; printf "%d", c }')
    CALC_CONT_GB=$(awk -v c="$CALC_CACHE_GB" 'BEGIN { m = c + 160; if (m < 200) m = 200; if (m > 750) m = 750; printf "%d", m }')
    CALC_GCSFUSE_GB=$(awk -v c="$CALC_CACHE_GB" 'BEGIN { s = c + 20; if (s > 700) s = 700; printf "%d", s }')
else
    CALC_CACHE_GB=32
    CALC_CONT_GB=200
    CALC_GCSFUSE_GB=72
fi

RAM_CACHE_LIMIT="${CALC_CACHE_GB}Gi"
CONTAINER_MEM_LIMIT="${CALC_CONT_GB}Gi"
GCSFUSE_MEM_LIMIT="${CALC_GCSFUSE_GB}Gi"

# --- STORAGE TYPE & PATH RESOLUTION ---
if [[ "$MODEL_NAME" =~ ^gs://([^/]+)/+(.*) ]]; then
    STORAGE_TYPE="gcs-direct"
    BUCKET_NAME="${BASH_REMATCH[1]}"
    CACHE_SUBPATH="${BASH_REMATCH[2]}"
elif [ -n "$GCS_BUCKET" ]; then
    STORAGE_TYPE="gcs-cache"
    CLEAN_GCS_BUCKET="${GCS_BUCKET#gs://}"
    CLEAN_GCS_BUCKET="${CLEAN_GCS_BUCKET%/}"
    if [[ "$CLEAN_GCS_BUCKET" =~ ^([^/]+)(/(.*))?$ ]]; then
        BUCKET_NAME="${BASH_REMATCH[1]}"
        CACHE_SUBPATH="${BASH_REMATCH[3]}"
    else
        BUCKET_NAME="${CLEAN_GCS_BUCKET}"
        CACHE_SUBPATH=""
    fi
else
    STORAGE_TYPE="ramdisk"
    BUCKET_NAME=""
    CACHE_SUBPATH=""
fi

HF_TOKEN_SECRET="${CLEAN_USER}-test-token"

# --- GENERATE STRUCTURED YAML CONTENT ---
if [ "$MODE" = "aggregated" ]; then
OUTPUT_BODY=$(cat <<YAML_BLOCK
# Auto-generated by run_benchmark.sh (Helm - Monolithic Mode)
# User: ${CURRENT_USER} | Release: ${RELEASE_NAME} | Date: $(date '+%Y-%m-%d %H:%M:%S')

mode: "aggregated"
image: "${IMAGE:-${IMAGE_TPU_INFERENCE}}"

model:
  name: "${MODEL_NAME}"
  maxModelLen: ${MAX_MODEL_LEN}
  safetensorsLoadStrategy: "prefetch"

features:
  asyncScheduling: ${FEATURE_ASYNC_SCHEDULING}
  kvCacheOffload:
    enabled: ${FEATURE_KV_OFFLOAD}
    connector: "${FEATURE_OFFLOAD_CONNECTOR}"
    numCpuChunks: 1024
  prefixCaching: ${FEATURE_PREFIX_CACHING}
  chunkedPrefill:
    enabled: ${FEATURE_CHUNKED_PREFILL}
    maxNumBatchedTokens: ${FEATURE_MAX_BATCHED_TOKENS}
  wideEP:
    enabled: ${FEATURE_WIDE_EP}
    size: 1
  speculativeDecoding:
    enabled: ${FEATURE_SPEC_DECODING}
    method: "eagle3"
    draftModel: "${FEATURE_DRAFT_MODEL}"
    numSpeculativeTokens: ${FEATURE_SPEC_TOKENS}
  structuredOutput:
    enabled: ${FEATURE_STRUCTURED_OUTPUT}
    backend: "auto"

tpu:
  accelerator: "tpu7x"
  topology: "${TOPOLOGY}"
  placementPolicy: "${PLACEMENT_POLICY}"
  tensorParallelSize: ${TP_SIZE}

storage:
  type: "${STORAGE_TYPE}"
  bucketName: "${BUCKET_NAME}"
  cacheSubpath: "${CACHE_SUBPATH}"
  ramCacheLimit: "${RAM_CACHE_LIMIT}"
  gcsFuse:
    memoryLimit: "${GCSFUSE_MEM_LIMIT}"
    cpuLimit: "8"

serviceAccount: "${SERVICE_ACCOUNT}"
hfTokenSecret:
  name: "${HF_TOKEN_SECRET}"
  key: "HF_TOKEN"

resources:
  cpu: "32"
  memory: "${CONTAINER_MEM_LIMIT}"
  tpu: 4

job:
  declaredDurationMinutes: ${DURATION_MINUTES}
  priorityClass: "medium"

benchmark:
  requestRate: ${REQUEST_RATE}
  randomPrefixLen: ${BENCHMARK_PREFIX_LEN}
  stages:
    - inputLen: 128
      outputLen: 128
      numPrompts: 100
    - inputLen: 512
      outputLen: 256
      numPrompts: 100
    - inputLen: 1024
      outputLen: 512
      numPrompts: 100
    - inputLen: 2048
      outputLen: 512
      numPrompts: 100
    - inputLen: 3072
      outputLen: 512
      numPrompts: 100
  resources:
    cpu: "4"
    memory: "8Gi"
YAML_BLOCK
)
else
# Disaggregated Mode
OUTPUT_BODY=$(cat <<YAML_BLOCK
# Auto-generated by run_benchmark.sh (Helm - Disaggregated Mode)
# User: ${CURRENT_USER} | Release: ${RELEASE_NAME} | Date: $(date '+%Y-%m-%d %H:%M:%S')

mode: "disaggregated"
image: "${IMAGE:-${IMAGE_TPU_INFERENCE}}"

model:
  name: "${MODEL_NAME}"
  maxModelLen: ${MAX_MODEL_LEN}
  safetensorsLoadStrategy: "prefetch"

features:
  asyncScheduling: ${FEATURE_ASYNC_SCHEDULING}
  kvCacheOffload:
    enabled: ${FEATURE_KV_OFFLOAD}
    connector: "${FEATURE_OFFLOAD_CONNECTOR}"
    numCpuChunks: 1024
  prefixCaching: ${FEATURE_PREFIX_CACHING}
  chunkedPrefill:
    enabled: ${FEATURE_CHUNKED_PREFILL}
    maxNumBatchedTokens: ${FEATURE_MAX_BATCHED_TOKENS}
  wideEP:
    enabled: ${FEATURE_WIDE_EP}
    size: 1
  speculativeDecoding:
    enabled: ${FEATURE_SPEC_DECODING}
    method: "eagle3"
    draftModel: "${FEATURE_DRAFT_MODEL}"
    numSpeculativeTokens: ${FEATURE_SPEC_TOKENS}
  structuredOutput:
    enabled: ${FEATURE_STRUCTURED_OUTPUT}
    backend: "auto"

disaggregated:
  kvConnector: "${KV_CONNECTOR}"
  prefill:
    topology: "${PREFILL_TOPOLOGY}"
    port: 8400
    placementPolicy: "${PREFILL_PLACEMENT_POLICY}"
    resources:
      cpu: "32"
      memory: "${CONTAINER_MEM_LIMIT}"
      tpu: 4
  decode:
    replicas: ${DECODE_REPLICAS}
    topology: "${DECODE_TOPOLOGY}"
    port: 9400
    placementPolicy: "${DECODE_PLACEMENT_POLICY}"
    resources:
      cpu: "32"
      memory: "${CONTAINER_MEM_LIMIT}"
      tpu: 4
  proxy:
    port: 8000
    resources:
      cpu: "8"
      memory: "16Gi"

storage:
  type: "${STORAGE_TYPE}"
  bucketName: "${BUCKET_NAME}"
  cacheSubpath: "${CACHE_SUBPATH}"
  ramCacheLimit: "${RAM_CACHE_LIMIT}"
  gcsFuse:
    memoryLimit: "${GCSFUSE_MEM_LIMIT}"
    cpuLimit: "8"

serviceAccount: "${SERVICE_ACCOUNT}"
hfTokenSecret:
  name: "${HF_TOKEN_SECRET}"
  key: "HF_TOKEN"

job:
  declaredDurationMinutes: ${DURATION_MINUTES}
  priorityClass: "medium"

benchmark:
  requestRate: ${REQUEST_RATE}
  randomPrefixLen: ${BENCHMARK_PREFIX_LEN}
  stages:
    - inputLen: 128
      outputLen: 128
      numPrompts: 100
    - inputLen: 512
      outputLen: 256
      numPrompts: 100
    - inputLen: 1024
      outputLen: 512
      numPrompts: 100
    - inputLen: 2048
      outputLen: 512
      numPrompts: 100
    - inputLen: 3072
      outputLen: 512
      numPrompts: 100
  resources:
    cpu: "4"
    memory: "8Gi"
YAML_BLOCK
)
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Mode 1: Save values to file
if [ -n "$OUTPUT_FILE" ]; then
    echo "$OUTPUT_BODY" > "$OUTPUT_FILE"
    echo "✅ Dynamically calculated values saved to: $OUTPUT_FILE"
    exit 0
fi

# Mode 2: Print values to stdout
if [ "$VALUES_ONLY" = true ]; then
    echo "$OUTPUT_BODY"
    exit 0
fi

# Mode 3: Dry-run / render full Kubernetes JobSet manifest
if [ "$DRY_RUN" = true ]; then
    TMP_VALS=$(mktemp /tmp/values-XXXXXX.yaml)
    echo "$OUTPUT_BODY" > "$TMP_VALS"
    helm template "$RELEASE_NAME" "$SCRIPT_DIR" -f "$TMP_VALS"
    rm -f "$TMP_VALS"
    exit 0
fi

# Mode 4: DEFAULT -> Deploy immediately with Helm!
TMP_VALS=$(mktemp /tmp/values-XXXXXX.yaml)
echo "$OUTPUT_BODY" > "$TMP_VALS"

echo "============================================================"
echo " 🚀 Deploying Benchmark via Helm (${MODE} mode)"
echo "============================================================"
echo " Release Name : ${RELEASE_NAME}"
echo " Image        : ${IMAGE}"
echo " Model        : ${MODEL_NAME}"
if [ "$MODE" = "aggregated" ]; then
echo " Topology     : ${TOPOLOGY} (${NUM_VMS} VM(s), TP=${TP_SIZE})"
else
echo " Prefill Slice: ${PREFILL_TOPOLOGY} (${PREFILL_NUM_VMS} VM(s), TP=${PREFILL_TP_SIZE})"
echo " Decode Slice : ${DECODE_TOPOLOGY} (${DECODE_NUM_VMS} VM(s), TP=${DECODE_TP_SIZE})"
fi
echo " Request Rate : ${REQUEST_RATE} RPS"
echo " Prefix Length : ${BENCHMARK_PREFIX_LEN} tokens (0 = pure random, >0 = shared APC prefix)"
echo " Storage Mode : ${STORAGE_TYPE} (${BUCKET_NAME:-none})"
echo " RAM Cache    : ${RAM_CACHE_LIMIT}"
echo " Features     : Async=${FEATURE_ASYNC_SCHEDULING}, PrefixCache=${FEATURE_PREFIX_CACHING}, ChunkedPrefill=${FEATURE_CHUNKED_PREFILL}, WideEP=${FEATURE_WIDE_EP}, SpecDecode=${FEATURE_SPEC_DECODING}, KVOffload=${FEATURE_KV_OFFLOAD}, StructOutput=${FEATURE_STRUCTURED_OUTPUT}"
echo "============================================================"

helm install "$RELEASE_NAME" "$SCRIPT_DIR" -f "$TMP_VALS"
rm -f "$TMP_VALS"

JOBSET_NAME="${RELEASE_NAME}"

echo ""
echo "============================================================"
echo " 📋 Useful Commands to Monitor Benchmark:"
echo "============================================================"
echo " 1. Watch pods:"
echo "    kubectl get pods -l jobset.sigs.k8s.io/jobset-name=${JOBSET_NAME} -w"
echo ""
echo " 6. Stream & Tee all logs into log/*.log<N>:"
echo "    ./gke/bin/tee_logs.sh ${JOBSET_NAME} [N]"
echo ""
if [ "$MODE" = "aggregated" ]; then
echo " 6. Stream server logs (vLLM / XLA compilation):"
echo "    kubectl logs -l jobset.sigs.k8s.io/jobset-name=${JOBSET_NAME},jobset.sigs.k8s.io/replicatedjob-name=server -f"
else
echo " 6. Stream Prefill logs:"
echo "    kubectl logs -l jobset.sigs.k8s.io/jobset-name=${JOBSET_NAME},jobset.sigs.k8s.io/replicatedjob-name=p -c vllm-tpu -f"
echo ""
echo " 6. Stream Decode logs:"
echo "    kubectl logs -l jobset.sigs.k8s.io/jobset-name=${JOBSET_NAME},jobset.sigs.k8s.io/replicatedjob-name=d -c vllm-tpu -f"
echo ""
echo " 6. Stream Proxy logs:"
echo "    kubectl logs -l jobset.sigs.k8s.io/jobset-name=${JOBSET_NAME},jobset.sigs.k8s.io/replicatedjob-name=x -f"
fi
echo ""
echo " 6. Stream client benchmark results:"
echo "    kubectl logs -l jobset.sigs.k8s.io/jobset-name=${JOBSET_NAME},jobset.sigs.k8s.io/replicatedjob-name=client -f"
echo ""
echo " 7. Teardown / Cleanup:"
echo "    ./gke/bin/cleanup.sh ${RELEASE_NAME}"
echo "    # Or via helm directly:"
echo "    helm uninstall ${RELEASE_NAME}"
echo "============================================================"
