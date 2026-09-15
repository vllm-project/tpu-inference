#!/bin/bash
# ==============================================================================
# TPU vLLM Test Case Runner (Helm-Powered)
# Executes functional & integration test cases instead of synthetic benchmarks
# Supports both Monolithic (Aggregated) and Disaggregated (P/D) Architectures
# ==============================================================================
set -e

# --- AUTO-DETECT USER & DEFAULTS ---
CURRENT_USER="${USER:-$(whoami)}"
CLEAN_USER=$(echo "$CURRENT_USER" | tr '[:upper:]' '[:lower:]' | tr -dc 'a-z0-9')
RANDOM_SUFFIX=$(LC_ALL=C tr -dc 'a-z0-9' < /dev/urandom | head -c 5)
DEFAULT_RELEASE="${CLEAN_USER}-tc-${RANDOM_SUFFIX}"

MODE="aggregated"
MODEL_NAME="meta-llama/Llama-3.1-8B-Instruct"
TOPOLOGY="2x2x1"
PREFILL_TOPOLOGY="2x2x1"
DECODE_TOPOLOGY="2x2x2"
DECODE_REPLICAS="1"
KV_CONNECTOR=""
IMAGE_TPU_INFERENCE="docker.io/vllm/vllm-tpu:v0.27.0"
IMAGE_TORCHTPU="us-central1-docker.pkg.dev/cloud-ullm-inference-ci-cd/vllm-torchtpu/torchtpu-vllm-prod:latest"
IMAGE="${IMAGE:-${IMAGE_TPU_INFERENCE}}"
IMAGE_EXPLICIT=false
SERVICE_ACCOUNT="vllm-sa"
GCS_BUCKET=""
PLACEMENT_POLICY=""
PREFILL_PLACEMENT_POLICY=""
DECODE_PLACEMENT_POLICY=""
DURATION_MINUTES="30"
DURATION_EXPLICIT=false
MAX_MODEL_LEN="8192"
RELEASE_NAME=""
OUTPUT_FILE=""
VALUES_ONLY=false
DRY_RUN=false

# --- TEST CASE CONFIGURATIONS ---
TEST_SUITE="full"
CUSTOM_PROMPT=""
EXPECTED_PATTERN=""
TEMPERATURE="0.0"
MAX_TOKENS="128"
TEST_SCRIPT_PATH=""
CUSTOM_SCRIPT_B64=""

# --- OFFLINE E2E PYTEST CONFIGURATIONS ---
# Drives the `e2etest` values block, which replaces the api_server + HTTP client
# jobs with a single TPU pod running pytest in-process (see helm/README.md).
E2E_MODE=false
E2E_PATHS=()          # in-image node-ids, e.g. tests/e2e/test_x.py::test_y
E2E_LOCAL_FILE=""     # local .py injected through the ConfigMap
E2E_SELECTOR=""       # pytest -k
E2E_EXTRA_ARGS=()     # extra pytest flags
E2E_MODELS=()         # HF repos to pre-cache before pytest starts
E2E_ENV=()            # K=V env vars for the test pod (mirror a .buildkite step)
E2E_WORKDIR="/workspace/tpu_inference"

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

# Prints the help text and exits. Pass a non-zero code when called from an
# error path so callers/CI can distinguish "user asked for help" from "bad input".
usage() {
    local exit_code="${1:-0}"
    echo "=================================================================="
    echo " TPU vLLM Test Case Runner (Helm-Powered)"
    echo "=================================================================="
    echo "Usage: $0 [options]"
    echo ""
    echo "Test Case Options (HTTP client against a running api_server):"
    echo "  --suite <name>              Test suite to execute: 'full' (all tests), 'smoke' (health+chat), 'chat' (only chat) (default: full)"
    echo "  --prompt <text>             Custom user prompt to test against the model"
    echo "  --expected <pattern>        Expected regex or substring in the model response for assertion"
    echo "  --test-script <file.py>     Path to a local Python script to execute inside the test container"
    echo "  --temperature <temp>        Generation temperature for test cases (default: 0.0 for reproducibility)"
    echo "  --max-tokens <N>            Max generation tokens for test cases (default: 128)"
    echo ""
    echo "Offline E2E Pytest Options (in-process LLM(), same suites as .buildkite):"
    echo "  --pytest <path|nodeid>      Run a test that already exists IN THE IMAGE, relative to"
    echo "                              /workspace/tpu_inference (repeatable, or space-separated)."
    echo "                              e.g. --pytest 'tests/e2e/test_data_parallel.py::test_dp_correctness'"
    echo "  --pytest-file <file.py>     Run a LOCAL test file (injected via ConfigMap). Use for local"
    echo "                              edits, or images that do not ship tests/."
    echo "  -k, --pytest-k <expr>       pytest -k selector, e.g. 'correctness_DP_torchax'"
    echo "  --pytest-arg <flag>         Extra pytest flag (repeatable), e.g. --pytest-arg --forked"
    echo "  --pytest-model <repo>       Extra HF repo to pre-cache (repeatable; --model is always included)"
    echo "  --pytest-env <K=V>          Env var for the test pod (repeatable). Mirror the 'env:' block"
    echo "                              of the .buildkite step, e.g. --pytest-env NEW_MODEL_DESIGN=1"
    echo "  --pytest-workdir <dir>      pytest rootdir inside the image (default: /workspace/tpu_inference)"
    echo "                              NOTE: these replace the api_server + HTTP client entirely, and"
    echo "                              raise the default duration to 300 minutes."
    echo ""
    echo "Common Infrastructure Options:"
    echo "  -m, --model <model>         Model name / HF repo / GCS URI (default: meta-llama/Llama-3.1-8B-Instruct)"
    echo "  -i, --image <image>         Docker container image or alias (default: docker.io/vllm/vllm-tpu:v0.27.0)"
    echo "                              Quick aliases: 'torchtpu', 'tpu-inference'"
    echo "  --torchtpu                  Quick switch to Google TorchTPU image (torchtpu-vllm-prod:latest)"
    echo "  --tpu-inference             Quick switch to upstream tpu-inference image (vllm-tpu:v0.27.0)"
    echo "  -b, --bucket <bucket>       GCS bucket path for caching (e.g. gs://<bucket>/hf-cache)"
    echo "  -r, --release <release>     Helm release name (default: auto-generated '${DEFAULT_RELEASE}')"
    echo "  -s, --sa <sa>               Kubernetes ServiceAccount (default: vllm-sa)"
    echo "  -D, --duration <mins>       Job duration in minutes (default: 30)"
    echo "  -o, --output <file>         Save calculated values.yaml to file (skips deployment)"
    echo "  -v, --values-only           Print calculated values.yaml to stdout without deploying"
    echo "  -g, --dry-run               Render the final JobSet manifest (helm template) to stdout"
    echo "  -h, --help                  Show this help message"
    echo ""
    echo "Monolithic Mode Options (default):"
    echo "  -t, --topology <topology>   TPU v7 topology shape (default: 2x2x1)"
    echo "  -P, --policy <policy>       Placement policy name (auto-fetched via gcloud if omitted on multihost)"
    echo ""
    echo "Disaggregated Mode Options:"
    echo "  --disagg                    Enable Disaggregated Prefill/Decode architecture"
    echo "  -p, --prefill <topology>    Prefill TPU v7 topology (default: 2x2x1)"
    echo "  -d, --decode <topology>     Decode TPU v7 topology (default: 2x2x2)"
    echo "  -N, --decode-replicas <N>   Number of Decode slices to scale horizontally (default: 1)"
    echo "  -c, --connector <connector> KV Connector: TPUConnector or TPURaidenConnector (auto-detected if omitted)"
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
    echo "Examples:"
    echo "  1. Run full test suite on Monolithic 8B (TPU 2x2x1):"
    echo "     $0 -t 2x2x1 -m meta-llama/Llama-3.1-8B-Instruct -b gs://my-bucket/hf-cache"
    echo ""
    echo "  2. Quick smoke test on Disaggregated serving:"
    echo "     $0 --disagg -p 2x2x1 -d 2x2x2 -m meta-llama/Llama-3.1-8B-Instruct --suite smoke"
    echo ""
    echo "  3. Test custom prompt with expected response assertion:"
    echo "     $0 -t 2x2x1 -m meta-llama/Llama-3.1-8B-Instruct \\"
    echo "        --prompt \"What is the chemical formula for water?\" \\"
    echo "        --expected \"H2O\""
    echo ""
    echo "  4. Execute a local custom python test script inside the test container:"
    echo "     $0 -t 2x2x1 -m meta-llama/Llama-3.1-8B-Instruct --test-script ./my_tests.py"
    echo ""
    echo "  5. Dry-run to preview rendered YAML without deploying:"
    echo "     $0 -t 2x2x1 -m meta-llama/Llama-3.1-8B-Instruct -g"
    echo ""
    echo "  6. Run an e2e suite that ships in the image (equivalent to the .buildkite DP step):"
    echo "     $0 --pytest 'tests/e2e/test_data_parallel.py::test_dp_correctness' \\\\"
    echo "        -b gs://my-bucket/hf-cache"
    echo ""
    echo "  7. Run a LOCAL test file with a -k selector:"
    echo "     $0 --pytest-file tests/e2e/test_continue_decode.py \\\\"
    echo "        -k correctness_DP_torchax --pytest-model Qwen/Qwen1.5-MoE-A2.7B"
    echo "============================================================"
    exit "$exit_code"
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
        --suite|--test-suite)
            TEST_SUITE="$2"
            shift 2
            ;;
        --prompt|--custom-prompt)
            CUSTOM_PROMPT="$2"
            shift 2
            ;;
        --expected|--expected-pattern)
            EXPECTED_PATTERN="$2"
            shift 2
            ;;
        --test-script|--script)
            TEST_SCRIPT_PATH="$2"
            shift 2
            ;;
        --temperature|--temp)
            TEMPERATURE="$2"
            shift 2
            ;;
        --max-tokens)
            MAX_TOKENS="$2"
            shift 2
            ;;
        --pytest|--pytest-path|--e2e)
            # Accept either one node-id per flag or several space-separated ones,
            # so a command copied out of .buildkite/**/*.yml can be pasted as-is.
            # shellcheck disable=SC2206
            E2E_PATHS+=($2)
            E2E_MODE=true
            shift 2
            ;;
        --pytest-file|--e2e-file)
            E2E_LOCAL_FILE="$2"
            E2E_MODE=true
            shift 2
            ;;
        -k|--pytest-k|--select)
            E2E_SELECTOR="$2"
            shift 2
            ;;
        --pytest-arg|--pytest-args)
            E2E_EXTRA_ARGS+=("$2")
            shift 2
            ;;
        --pytest-model|--e2e-model)
            E2E_MODELS+=("$2")
            shift 2
            ;;
        --pytest-env|--e2e-env)
            E2E_ENV+=("$2")
            shift 2
            ;;
        --pytest-workdir)
            E2E_WORKDIR="$2"
            shift 2
            ;;
        --torchtpu|--vllm-torchtpu|--prod)
            IMAGE="${IMAGE_TORCHTPU}"
            IMAGE_EXPLICIT=true
            shift
            ;;
        --tpu-inference|--vllm-tpu)
            IMAGE="${IMAGE_TPU_INFERENCE}"
            IMAGE_EXPLICIT=true
            shift
            ;;
        -i|--image)
            IMAGE_EXPLICIT=true
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
            IMAGE_EXPLICIT=true
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
            DURATION_EXPLICIT=true
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
            usage 0
            ;;
        *)
            echo "Error: Unknown option $1" >&2
            usage 1
            ;;
    esac
done

if [ -z "$RELEASE_NAME" ]; then
    RELEASE_NAME="$DEFAULT_RELEASE"
fi

# Process user-provided custom test script if specified
if [ -n "$TEST_SCRIPT_PATH" ]; then
    if [ ! -f "$TEST_SCRIPT_PATH" ]; then
        echo "❌ Error: Specified test script not found: $TEST_SCRIPT_PATH"
        exit 1
    fi
    CUSTOM_SCRIPT_B64=$(base64 < "$TEST_SCRIPT_PATH" | tr -d '\r\n')
fi

# --- OFFLINE E2E PYTEST VALIDATION & DERIVED DEFAULTS ---
if [ "$E2E_MODE" = true ]; then
    if [ ${#E2E_PATHS[@]} -gt 0 ] && [ -n "$E2E_LOCAL_FILE" ]; then
        echo "❌ Error: --pytest and --pytest-file are mutually exclusive."
        echo "   --pytest runs a path inside the image; --pytest-file injects a local file."
        exit 1
    fi
    if [ -n "$E2E_LOCAL_FILE" ] && [ ! -f "$E2E_LOCAL_FILE" ]; then
        echo "❌ Error: Test file not found: $E2E_LOCAL_FILE"
        exit 1
    fi
    if [ -n "$TEST_SCRIPT_PATH" ]; then
        echo "❌ Error: --test-script runs a plain script in the CPU client pod and cannot be"
        echo "   combined with the pytest options. Use --pytest-file instead."
        exit 1
    fi
    for _kv in "${E2E_ENV[@]}"; do
        if [[ "$_kv" != *=* ]]; then
            echo "❌ Error: --pytest-env expects KEY=VALUE, got: ${_kv}"
            exit 1
        fi
    done
    # The e2e job takes the whole slice; there is no prefill/decode split to make.
    if [ "$MODE" = "disaggregated" ]; then
        echo "⚠️  Warning: disaggregated mode is ignored for pytest runs (the test process"
        echo "   owns the entire TPU slice). Falling back to aggregated."
        MODE="aggregated"
    fi
    # Always pre-cache the model passed via -m, plus any extra repos the suite
    # needs (test_continue_decode.py, for example, uses Llama AND Qwen-MoE).
    E2E_ALL_MODELS=("$MODEL_NAME" "${E2E_MODELS[@]}")
    E2E_MODELS=()
    for _m in "${E2E_ALL_MODELS[@]}"; do
        if [ -z "$_m" ]; then
            continue
        fi
        _dup=false
        for _seen in "${E2E_MODELS[@]}"; do
            if [ "$_seen" = "$_m" ]; then _dup=true; break; fi
        done
        if [ "$_dup" = false ]; then
            E2E_MODELS+=("$_m")
        fi
    done
    # Weight download + repeated engine cold starts blow well past the 30 min default.
    if [ "$DURATION_EXPLICIT" = false ]; then
        DURATION_MINUTES="300"
    fi
    # E2E pytest requires in-image test sources and test utilities from torchtpu.
    # If the user did not explicitly specify an image, default to IMAGE_TORCHTPU.
    if [ "$IMAGE_EXPLICIT" = false ]; then
        IMAGE="${IMAGE_TORCHTPU}"
    fi
fi

# Function to calculate VMs and TP
calc_topology() {
    local topo="$1"
    IFS='x' read -r tx ty tz <<< "$topo"
    local chips=$((tx * ty * tz))
    local vms=$((chips / 4))
    [ "$vms" -lt 1 ] && vms=1
    local tp=$((chips * 2))
    [ "$tp" -lt 1 ] && tp=1
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
    # The e2e job hardcodes parallelism=1: the offline pytest suites all fit in a
    # single host, so it runs no Ray bootstrap and sets none of the multi-host
    # TPU_* env vars. A multi-host topology would still be requested through the
    # nodeSelector, so the extra VMs would sit idle while the lone Pod hangs
    # waiting for slice peers that never join. Reject it up front rather than
    # letting it fail later as an opaque libtpu error.
    if [ "$E2E_MODE" = true ] && [ "$NUM_VMS" -gt 1 ]; then
        echo "❌ Error: pytest mode only supports single-host topologies."
        echo "   ${TOPOLOGY} needs ${NUM_VMS} VMs, but the e2e job always runs exactly 1 Pod,"
        echo "   so the other $((NUM_VMS - 1)) VM(s) would be allocated and left idle."
        echo "   Use a single-host topology instead (e.g. -t 2x2x1)."
        exit 1
    fi
    if [ "$NUM_VMS" -gt 1 ] && [ -z "$PLACEMENT_POLICY" ]; then
        PLACEMENT_POLICY=$(discover_placement_policy "$TOPOLOGY")
    fi
else
    read -r PREFILL_NUM_VMS PREFILL_TP_SIZE <<< "$(calc_topology "$PREFILL_TOPOLOGY")"
    read -r DECODE_NUM_VMS DECODE_TP_SIZE <<< "$(calc_topology "$DECODE_TOPOLOGY")"

    if [ -z "$PREFILL_PLACEMENT_POLICY" ]; then
        if [ -n "$PLACEMENT_POLICY" ]; then
            PREFILL_PLACEMENT_POLICY="$PLACEMENT_POLICY"
        elif [ "$PREFILL_NUM_VMS" -gt 1 ]; then
            PREFILL_PLACEMENT_POLICY=$(discover_placement_policy "$PREFILL_TOPOLOGY")
        fi
    fi
    if [ -z "$DECODE_PLACEMENT_POLICY" ]; then
        if [ -n "$PLACEMENT_POLICY" ]; then
            DECODE_PLACEMENT_POLICY="$PLACEMENT_POLICY"
        elif [ "$DECODE_NUM_VMS" -gt 1 ]; then
            DECODE_PLACEMENT_POLICY=$(discover_placement_policy "$DECODE_TOPOLOGY")
        fi
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

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# In pytest mode the HTTP client suite is replaced by the e2e job entirely.
if [ "$E2E_MODE" = true ]; then
    TESTCASE_ENABLED=false
else
    TESTCASE_ENABLED=true
fi

# Build the `e2etest` values section.
# NOTE: this is deliberately NOT part of the heredoc below. The heredoc is
# unquoted (it interpolates ${VARS}), and a test file containing `${...}`,
# backticks or `$(...)` would otherwise be mangled by the shell.
build_e2e_block() {
    printf '\n'
    printf 'e2etest:\n'
    printf '  enabled: true\n'
    printf '  workdir: "%s"\n' "$E2E_WORKDIR"
    printf '  selector: "%s"\n' "$E2E_SELECTOR"

    if [ ${#E2E_PATHS[@]} -gt 0 ]; then
        printf '  testPaths:\n'
        for p in "${E2E_PATHS[@]}"; do
            printf '    - "%s"\n' "$p"
        done
    else
        printf '  testPaths: []\n'
    fi

    printf '  models:\n'
    for m in "${E2E_MODELS[@]}"; do
        printf '    - "%s"\n' "$m"
    done

    printf '  extraArgs:\n'
    printf '    - "-x"\n'
    printf '    - "--durations=0"\n'
    for a in "${E2E_EXTRA_ARGS[@]}"; do
        printf '    - "%s"\n' "$a"
    done

    if [ ${#E2E_ENV[@]} -gt 0 ]; then
        printf '  env:\n'
        for kv in "${E2E_ENV[@]}"; do
            printf '    %s: "%s"\n' "${kv%%=*}" "${kv#*=}"
        done
    fi

    if [ -n "$E2E_LOCAL_FILE" ]; then
        printf '  testFile: "%s"\n' "$(basename "$E2E_LOCAL_FILE")"
        printf '  fileContent: |\n'
        sed 's/^/    /' "$E2E_LOCAL_FILE"
        # tests/conftest.py registers the custom markers (bvt, disable_jax_cache)
        # and the JAX compilation-cache fixtures; ship it alongside the test.
        local repo_conftest="${SCRIPT_DIR}/../tests/conftest.py"
        if [ -f "$repo_conftest" ]; then
            printf '  conftestContent: |\n'
            sed 's/^/    /' "$repo_conftest"
        fi
    fi
}

# --- GENERATE STRUCTURED YAML CONTENT ---
if [ "$MODE" = "aggregated" ]; then
OUTPUT_BODY=$(cat <<YAML_BLOCK
# Auto-generated by run_testcase.sh (Helm - Monolithic Mode)
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

testcase:
  enabled: ${TESTCASE_ENABLED}
  suite: "${TEST_SUITE}"
  customPrompt: "${CUSTOM_PROMPT}"
  expectedPattern: "${EXPECTED_PATTERN}"
  temperature: ${TEMPERATURE}
  maxTokens: ${MAX_TOKENS}
  customScriptBase64: "${CUSTOM_SCRIPT_B64}"

benchmark:
  resources:
    cpu: "4"
    memory: "8Gi"
YAML_BLOCK
)
else
# Disaggregated Mode
OUTPUT_BODY=$(cat <<YAML_BLOCK
# Auto-generated by run_testcase.sh (Helm - Disaggregated Mode)
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

testcase:
  enabled: ${TESTCASE_ENABLED}
  suite: "${TEST_SUITE}"
  customPrompt: "${CUSTOM_PROMPT}"
  expectedPattern: "${EXPECTED_PATTERN}"
  temperature: ${TEMPERATURE}
  maxTokens: ${MAX_TOKENS}
  customScriptBase64: "${CUSTOM_SCRIPT_B64}"

benchmark:
  resources:
    cpu: "4"
    memory: "8Gi"
YAML_BLOCK
)
fi

# Append the pytest section (kept out of the heredoc: see build_e2e_block).
if [ "$E2E_MODE" = true ]; then
    OUTPUT_BODY="${OUTPUT_BODY}
$(build_e2e_block)"
fi

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
trap 'rm -f "$TMP_VALS"' EXIT
echo "$OUTPUT_BODY" > "$TMP_VALS"

echo "============================================================"
echo " 🚀 Deploying Test Suite via Helm (${MODE} mode)"
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
if [ "$E2E_MODE" = true ]; then
echo " Runner       : offline pytest (no api_server, no HTTP client)"
if [ ${#E2E_PATHS[@]} -gt 0 ]; then
echo " Test Targets : ${E2E_PATHS[*]}"
echo " Workdir      : ${E2E_WORKDIR} (in-image)"
else
echo " Test File    : ${E2E_LOCAL_FILE} (injected via ConfigMap)"
fi
if [ -n "$E2E_SELECTOR" ]; then
echo " Selector     : -k ${E2E_SELECTOR}"
fi
echo " Pre-cache    : ${E2E_MODELS[*]}"
echo " Duration     : ${DURATION_MINUTES} min"
else
echo " Test Suite   : ${TEST_SUITE}"
if [ -n "$CUSTOM_PROMPT" ]; then
echo " Custom Prompt: \"${CUSTOM_PROMPT}\""
echo " Expected     : \"${EXPECTED_PATTERN:-<any valid response>}\""
fi
if [ -n "$TEST_SCRIPT_PATH" ]; then
echo " Custom Script: ${TEST_SCRIPT_PATH}"
fi
fi
echo " Storage Mode : ${STORAGE_TYPE} (${BUCKET_NAME:-none})"
echo " RAM Cache    : ${RAM_CACHE_LIMIT}"
echo " Features     : Async=${FEATURE_ASYNC_SCHEDULING}, PrefixCache=${FEATURE_PREFIX_CACHING}, ChunkedPrefill=${FEATURE_CHUNKED_PREFILL}, WideEP=${FEATURE_WIDE_EP}, SpecDecode=${FEATURE_SPEC_DECODING}, KVOffload=${FEATURE_KV_OFFLOAD}, StructOutput=${FEATURE_STRUCTURED_OUTPUT}"
echo "============================================================"

helm install "$RELEASE_NAME" "$SCRIPT_DIR" -f "$TMP_VALS"
rm -f "$TMP_VALS"

JOBSET_NAME="${RELEASE_NAME}"

echo ""
echo "============================================================"
echo " 📋 Useful Commands to Monitor Test Execution:"
echo "============================================================"
echo " 1. Watch pods lifecycle:"
echo "    kubectl get pods -l jobset.sigs.k8s.io/jobset-name=${JOBSET_NAME} -w"
echo ""
if [ "$E2E_MODE" = true ]; then
echo " 2. Stream pytest logs (live PASS/FAIL):"
echo "    kubectl logs -l jobset.sigs.k8s.io/jobset-name=${JOBSET_NAME},jobset.sigs.k8s.io/replicatedjob-name=e2e -f"
echo "    # Or stream, save and generate test summary via e2e_log.sh:"
echo "    ./helm/bin/e2e_log.sh ${JOBSET_NAME}"
echo ""
echo " 3. Check overall JobSet completion status:"
echo "    kubectl get jobset ${JOBSET_NAME}"
echo ""
echo " 4. Copy the JUnit report out of the pod:"
echo "    kubectl cp \$(kubectl get pod -l jobset.sigs.k8s.io/jobset-name=${JOBSET_NAME} -o name | head -1 | cut -d/ -f2):/tmp/e2e_report.xml ./e2e_report.xml"
else
echo " 2. Stream test client logs & assertions (live PASS/FAIL):"
echo "    kubectl logs -l jobset.sigs.k8s.io/jobset-name=${JOBSET_NAME},jobset.sigs.k8s.io/replicatedjob-name=client -f"
echo ""
echo " 3. Check overall JobSet completion status:"
echo "    kubectl get jobset ${JOBSET_NAME}"
echo ""
if [ "$MODE" = "aggregated" ]; then
echo " 4. Stream server logs (vLLM / XLA compilation):"
echo "    kubectl logs -l jobset.sigs.k8s.io/jobset-name=${JOBSET_NAME},jobset.sigs.k8s.io/replicatedjob-name=server -f"
else
echo " 4. Stream Prefill logs:"
echo "    kubectl logs -l jobset.sigs.k8s.io/jobset-name=${JOBSET_NAME},jobset.sigs.k8s.io/replicatedjob-name=p -c vllm-tpu -f"
echo ""
echo " 5. Stream Decode logs:"
echo "    kubectl logs -l jobset.sigs.k8s.io/jobset-name=${JOBSET_NAME},jobset.sigs.k8s.io/replicatedjob-name=d -c vllm-tpu -f"
echo ""
echo " 6. Stream Proxy logs:"
echo "    kubectl logs -l jobset.sigs.k8s.io/jobset-name=${JOBSET_NAME},jobset.sigs.k8s.io/replicatedjob-name=x -f"
fi
fi
echo ""
echo " 7. Teardown / Cleanup after testing:"
echo "    ./helm/bin/cleanup.sh ${RELEASE_NAME}"
echo "    # Or via helm directly:"
echo "    helm uninstall ${RELEASE_NAME}"
echo "============================================================"
