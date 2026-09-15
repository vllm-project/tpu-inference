#!/bin/bash
# ==============================================================================
# TPU vLLM Log Streaming & Tee Utility
# Streams & tees logs from client, p (prefill), d (decode), and x (proxy)
# into log/<component>.log<number>
# ==============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Resolve log directory (prefers repo root log/ if symlinked, or gke/log)
if [ -d "${REPO_ROOT}/log" ]; then
    LOG_DIR="${REPO_ROOT}/log"
elif [ -d "${SCRIPT_DIR}/log" ]; then
    LOG_DIR="${SCRIPT_DIR}/log"
else
    LOG_DIR="${SCRIPT_DIR}/log"
    mkdir -p "${LOG_DIR}"
fi

# Locate generate_summary.py companion script
SUMMARY_SCRIPT=""
if [ -f "${SCRIPT_DIR}/generate_summary.py" ]; then
    SUMMARY_SCRIPT="${SCRIPT_DIR}/generate_summary.py"
elif [ -f "${SCRIPT_DIR}/bin/generate_summary.py" ]; then
    SUMMARY_SCRIPT="${SCRIPT_DIR}/bin/generate_summary.py"
elif [ -f "${REPO_ROOT}/bin/generate_summary.py" ]; then
    SUMMARY_SCRIPT="${REPO_ROOT}/bin/generate_summary.py"
fi

CURRENT_USER="${USER:-$(whoami)}"
CLEAN_USER=$(echo "$CURRENT_USER" | tr '[:upper:]' '[:lower:]' | tr -dc 'a-z0-9')

JOB_NAME=""
LOG_NUM=""
COMPONENT="all"
MODE="stream"      # "stream" | "dump" | "mux"
FOLLOW=true

usage() {
    echo "=================================================================="
    echo " ⚡ TPU vLLM Benchmark Log Streaming & Tee Utility"
    echo "=================================================================="
    echo "Usage: $0 [options] [JOB_NAME] [LOG_NUMBER]"
    echo ""
    echo "Arguments (can be passed positionally in any intuitive order):"
    echo "  JOB_NAME            JobSet / Release name (e.g. my-benchmark-job)."
    echo "                      If omitted, automatically detects latest active JobSet."
    echo "  LOG_NUMBER          Numeric suffix for log files (e.g. 4 -> client.log4)."
    echo "                      If omitted, auto-increments to next available number."
    echo ""
    echo "Options:"
    echo "  -j, --job <NAME>       Specify JobSet name explicitly"
    echo "  -n, --number <NUM>     Specify log number suffix explicitly"
    echo "  -c, --component <COMP> Component to stream: client, prefill (or p), decode (or d), proxy (or x), all."
    echo "                         Default: all (client in foreground, p/d/x in background)"
    echo "  -m, --mux              Multiplex all streams to stdout with colored prefixes while teeing"
    echo "  -s, --save, --dump     Snapshot/dump current logs from pods immediately without following (-f)
  -S, --summary          Generate and display log_summary.<number> report"
    echo "  -o, --dir <DIR>        Custom output directory (default: ${LOG_DIR})"
    echo "  -h, --help             Show this help message"
    echo ""
    echo "Examples:"
    echo "  1. Stream all 4 components (auto-detect job & next log number):"
    echo "     $0"
    echo ""
    echo "  2. Stream all components into log/*.log4:"
    echo "     $0 4"
    echo ""
    echo "  3. Stream specific job into log/*.log4:"
    echo "     $0 my-benchmark-job 4"
    echo ""
    echo "  4. Stream ONLY client live to screen and tee to log/client.log4:"
    echo "     $0 client 4"
    echo ""
    echo "  5. Stream ONLY decode logs to screen and tee to log/decode.log4:"
    echo "     $0 d 4"
    echo ""
    echo "  6. Multiplex all 4 logs live to screen with [TAGS] while saving to files:"
    echo "     $0 --mux 4"
    echo ""
    echo "  7. Snapshot/dump existing logs from a finished or running job into log/*.log4:"
    echo "     $0 --dump 4"
    echo "=================================================================="
    exit 1
}

# --- PARSE ARGUMENTS ---
POSITIONAL=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        -j|--job)
            JOB_NAME="$2"
            shift 2
            ;;
        -n|--number)
            LOG_NUM="$2"
            shift 2
            ;;
        -c|--component)
            COMPONENT="$2"
            shift 2
            ;;
        -m|--mux|--all-stdout)
            MODE="mux"
            shift
            ;;
        -s|--save|--dump)
            MODE="dump"
            FOLLOW=false
            shift
            ;;
        -S|--summary)
            MODE="summary"
            FOLLOW=false
            shift
            ;;
        -o|--dir)
            LOG_DIR="$2"
            mkdir -p "${LOG_DIR}"
            shift 2
            ;;
        -h|--help)
            usage
            ;;
        *)
            POSITIONAL+=("$1")
            shift
            ;;
    esac
done

# Process positional arguments intuitively
for arg in "${POSITIONAL[@]}"; do
    if [[ "$arg" =~ ^[0-9]+$ ]] && [ -z "$LOG_NUM" ]; then
        LOG_NUM="$arg"
    elif [[ "$arg" =~ ^(client|prefill|p|decode|d|proxy|x|server|all)$ ]] && [ "$COMPONENT" = "all" ]; then
        COMPONENT="$arg"
    elif [ -z "$JOB_NAME" ]; then
        JOB_NAME="$arg"
    fi
done

# Normalize component alias
case "$COMPONENT" in
    p) COMPONENT="prefill" ;;
    d) COMPONENT="decode" ;;
    x) COMPONENT="proxy" ;;
esac

# --- AUTO-DISCOVER ACTIVE JOBSET IF NOT PROVIDED ---
if [ -z "$JOB_NAME" ]; then
    echo "🔍 Detecting latest active JobSet..."
    # Prefer current user's active jobs
    JOB_NAME=$(kubectl get jobset -o jsonpath='{range .items[*]}{.metadata.name}{"\n"}{end}' 2>/dev/null | grep "${CLEAN_USER}-test-" | tail -n 1 || true)
    
    if [ -z "$JOB_NAME" ]; then
        # Fallback to any latest jobset
        JOB_NAME=$(kubectl get jobset -o jsonpath='{.items[-1].metadata.name}' 2>/dev/null || true)
    fi

    if [ -z "$JOB_NAME" ]; then
        echo "❌ Error: No active JobSets found on cluster. Please specify JobSet name explicitly."
        exit 1
    fi
    echo "   Found JobSet: ${JOB_NAME}"
fi

# --- AUTO-DISCOVER NEXT LOG NUMBER IF NOT PROVIDED ---
if [ -z "$LOG_NUM" ]; then
    MAX_NUM=$(ls -1 "${LOG_DIR}"/{client,prefill,decode,proxy}.log* 2>/dev/null | grep -o '[0-9]\+' | sort -n | tail -n 1 || true)
    if [ -z "$MAX_NUM" ]; then
        LOG_NUM=1
    else
        LOG_NUM=$((MAX_NUM + 1))
    fi
fi

# Verify JobSet exists
if ! kubectl get jobset "${JOB_NAME}" >/dev/null 2>&1; then
    echo "⚠️ Warning: JobSet '${JOB_NAME}' not found. Checking if pods exist..."
    if [ -z "$(kubectl get pods -l "jobset.sigs.k8s.io/jobset-name=${JOB_NAME}" -o jsonpath='{.items[*].metadata.name}' 2>/dev/null)" ]; then
        echo "❌ Error: No pods found for '${JOB_NAME}'."
        exit 1
    fi
fi

# --- DETECT ARCHITECTURE (DISAGGREGATED VS MONOLITHIC) ---
IS_DISAGG=false
if kubectl get pods -l "jobset.sigs.k8s.io/jobset-name=${JOB_NAME},jobset.sigs.k8s.io/replicatedjob-name=p" -o jsonpath='{.items[0].metadata.name}' 2>/dev/null | grep -q .; then
    IS_DISAGG=true
elif kubectl get jobset "${JOB_NAME}" -o jsonpath='{.spec.replicatedJobs[*].name}' 2>/dev/null | grep -q "p"; then
    IS_DISAGG=true
fi

# Define target log paths
CLIENT_LOG="${LOG_DIR}/client.log${LOG_NUM}"
PREFILL_LOG="${LOG_DIR}/prefill.log${LOG_NUM}"
DECODE_LOG="${LOG_DIR}/decode.log${LOG_NUM}"
PROXY_LOG="${LOG_DIR}/proxy.log${LOG_NUM}"
SERVER_LOG="${LOG_DIR}/server.log${LOG_NUM}"

echo "============================================================"

if [ "$MODE" = "summary" ]; then
    if [ -n "${SUMMARY_SCRIPT}" ]; then
        python3 "${SUMMARY_SCRIPT}" --job "${JOB_NAME}" --number "${LOG_NUM}" --dir "${LOG_DIR}"
    else
        echo "❌ generate_summary.py not found." >&2
        exit 1
    fi
    exit 0
fi
echo " ⚡ TPU vLLM Log Streamer & Tee Utility"
echo "============================================================"
echo " JobSet Name : ${JOB_NAME}"
echo " Log Suffix  : ${LOG_NUM}"
echo " Mode        : ${MODE} (component: ${COMPONENT})"
echo " Output Dir  : ${LOG_DIR}"
if [ "$IS_DISAGG" = true ]; then
echo " Stack Type  : Disaggregated (p, d, x, client)"
echo " Target Logs :"
echo "   - Client  : ${CLIENT_LOG}"
echo "   - Prefill : ${PREFILL_LOG}"
echo "   - Decode  : ${DECODE_LOG}"
echo "   - Proxy   : ${PROXY_LOG}"
else
echo " Stack Type  : Monolithic (server, client)"
echo " Target Logs :"
echo "   - Client  : ${CLIENT_LOG}"
echo "   - Server  : ${SERVER_LOG}"
fi
echo "============================================================"

# Background process tracking
PIDS=()
CLEANED_UP=false
cleanup() {
    if [ "$CLEANED_UP" = true ]; then
        return
    fi
    CLEANED_UP=true

    echo ""
    echo "Flushing and stopping log streams..."
    for pid in "${PIDS[@]}"; do
        kill "$pid" 2>/dev/null || true
    done
    sleep 1

    # Automatically call generate_summary.py to compile and display the benchmark report
    if [ -n "${SUMMARY_SCRIPT}" ]; then
        echo ""
        python3 "${SUMMARY_SCRIPT}" --job "${JOB_NAME}" --number "${LOG_NUM}" --dir "${LOG_DIR}" || true
    else
        echo ""
        echo "============================================================"
        echo " 📋 Logs Captured in ${LOG_DIR}:"
        echo "============================================================"
        for f in "${PREFILL_LOG}" "${DECODE_LOG}" "${PROXY_LOG}" "${CLIENT_LOG}" "${SERVER_LOG}"; do
            if [ -f "$f" ]; then
                LINES=$(wc -l < "$f")
                SIZE=$(ls -lh "$f" | awk '{print $5}')
                printf " - %-22s : %8s (%d lines)\n" "$(basename "$f")" "$SIZE" "$LINES"
            fi
        done
        echo "============================================================"
    fi
}
trap cleanup EXIT INT TERM

# Wait for at least one pod to be present
echo "⏳ Waiting for pods to appear..."
while [ -z "$(kubectl get pods -l "jobset.sigs.k8s.io/jobset-name=${JOB_NAME}" -o jsonpath='{.items[*].metadata.name}' 2>/dev/null)" ]; do
    sleep 2
done

# --- FUNCTION: STREAM / CAPTURE CONTAINER LOG ---
stream_container() {
    local rep_job="$1"
    local container="$2"
    local outfile="$3"
    local tag="$4"
    local color="$5"
    local tee_mode="$6" # "tee", "bg", "dump", "mux"

    local label="jobset.sigs.k8s.io/jobset-name=${JOB_NAME},jobset.sigs.k8s.io/replicatedjob-name=${rep_job}"
    if [ "$rep_job" = "d" ] || [ "$rep_job" = "server" ]; then
        # For multi-host slices, target Pod 0 (API Server & Ray Head)
        label="${label},batch.kubernetes.io/job-completion-index=0"
    fi

    # Ensure container argument
    local c_arg=""
    if [ -n "$container" ]; then
        c_arg="-c ${container}"
    fi

    if [ "$tee_mode" = "dump" ]; then
        echo "   Dumping ${rep_job} (${container:-default}) -> $(basename "$outfile")..."
        kubectl logs -l "$label" $c_arg --tail=-1 > "$outfile" 2>&1 || true
        return
    fi

    # Stream mode with auto-retry during container initialization
    local follow_flag=""
    if [ "$FOLLOW" = true ]; then
        follow_flag="-f"
    fi

    if [ "$tee_mode" = "tee" ]; then
        echo "   Teeing ${rep_job} (${container:-default}) live to screen -> $(basename "$outfile")..."
        while true; do
            kubectl logs -l "$label" $c_arg --tail=-1 $follow_flag 2>&1 | tee "$outfile" || true
            local phase=$(kubectl get pods -l "$label" -o jsonpath='{.items[0].status.phase}' 2>/dev/null || true)
            if [ "$phase" = "Succeeded" ] || [ "$phase" = "Failed" ] || [ -z "$phase" ]; then
                break
            fi
            sleep 2
        done
    elif [ "$tee_mode" = "mux" ]; then
        echo "   Muxing ${rep_job} live to screen with [${tag}] -> $(basename "$outfile")..."
        (
            while true; do
                kubectl logs -l "$label" $c_arg --tail=-1 $follow_flag 2>&1 | tee "$outfile" | awk -v col="$color" -v tag="$tag" '{print col "[" tag "]\033[0m " $0}' || true
                local phase=$(kubectl get pods -l "$label" -o jsonpath='{.items[0].status.phase}' 2>/dev/null || true)
                if [ "$phase" = "Succeeded" ] || [ "$phase" = "Failed" ] || [ -z "$phase" ]; then
                    break
                fi
                sleep 2
            done
        ) &
        PIDS+=("$!")
    else # "bg"
        (
            while true; do
                kubectl logs -l "$label" $c_arg --tail=-1 $follow_flag >> "$outfile" 2>&1 || true
                local phase=$(kubectl get pods -l "$label" -o jsonpath='{.items[0].status.phase}' 2>/dev/null || true)
                if [ "$phase" = "Succeeded" ] || [ "$phase" = "Failed" ] || [ -z "$phase" ]; then
                    break
                fi
                sleep 2
            done
        ) &
        PIDS+=("$!")
    fi
}

# --- EXECUTION MODES ---

# Colors for multiplexed mode
CLR_P="\033[36m"   # Cyan
CLR_D="\033[32m"   # Green
CLR_X="\033[33m"   # Yellow
CLR_C="\033[35m"   # Magenta
CLR_S="\033[34m"   # Blue

if [ "$MODE" = "dump" ]; then
    echo "📥 Snapshotting current logs to files..."
    if [ "$IS_DISAGG" = true ]; then
        stream_container "p" "vllm-tpu" "$PREFILL_LOG" "PREFILL" "$CLR_P" "dump"
        stream_container "d" "vllm-tpu" "$DECODE_LOG" "DECODE" "$CLR_D" "dump"
        stream_container "x" "proxy-server" "$PROXY_LOG" "PROXY" "$CLR_X" "dump"
        stream_container "client" "benchmark-client" "$CLIENT_LOG" "CLIENT" "$CLR_C" "dump"
    else
        stream_container "server" "vllm-tpu" "$SERVER_LOG" "SERVER" "$CLR_S" "dump"
        stream_container "client" "benchmark-client" "$CLIENT_LOG" "CLIENT" "$CLR_C" "dump"
    fi
    exit 0
fi

if [ "$MODE" = "mux" ]; then
    echo "📺 Multiplexing live logs to terminal (Ctrl+C to stop)..."
    if [ "$IS_DISAGG" = true ]; then
        stream_container "p" "vllm-tpu" "$PREFILL_LOG" "PREFILL" "$CLR_P" "mux"
        stream_container "d" "vllm-tpu" "$DECODE_LOG" "DECODE" "$CLR_D" "mux"
        stream_container "x" "proxy-server" "$PROXY_LOG" "PROXY" "$CLR_X" "mux"
        stream_container "client" "benchmark-client" "$CLIENT_LOG" "CLIENT" "$CLR_C" "mux"
    else
        stream_container "server" "vllm-tpu" "$SERVER_LOG" "SERVER" "$CLR_S" "mux"
        stream_container "client" "benchmark-client" "$CLIENT_LOG" "CLIENT" "$CLR_C" "mux"
    fi
    wait
    exit 0
fi

# Single component direct tee
if [ "$COMPONENT" != "all" ]; then
    case "$COMPONENT" in
        client)
            stream_container "client" "benchmark-client" "$CLIENT_LOG" "CLIENT" "$CLR_C" "tee"
            ;;
        prefill)
            stream_container "p" "vllm-tpu" "$PREFILL_LOG" "PREFILL" "$CLR_P" "tee"
            ;;
        decode)
            stream_container "d" "vllm-tpu" "$DECODE_LOG" "DECODE" "$CLR_D" "tee"
            ;;
        proxy)
            stream_container "x" "proxy-server" "$PROXY_LOG" "PROXY" "$CLR_X" "tee"
            ;;
        server)
            stream_container "server" "vllm-tpu" "$SERVER_LOG" "SERVER" "$CLR_S" "tee"
            ;;
        *)
            echo "Unknown component: $COMPONENT"
            exit 1
            ;;
    esac
    exit 0
fi

# DEFAULT MODE:
# Stream servers/routers in background to their log files, and tee client in foreground to terminal!
if [ "$IS_DISAGG" = true ]; then
    echo "📡 Background streaming p (prefill), d (decode), x (proxy) into log files..."
    stream_container "p" "vllm-tpu" "$PREFILL_LOG" "PREFILL" "$CLR_P" "bg"
    stream_container "d" "vllm-tpu" "$DECODE_LOG" "DECODE" "$CLR_D" "bg"
    stream_container "x" "proxy-server" "$PROXY_LOG" "PROXY" "$CLR_X" "bg"

    echo "📊 Foreground streaming & teeing client benchmark results to terminal..."
    stream_container "client" "benchmark-client" "$CLIENT_LOG" "CLIENT" "$CLR_C" "tee"
else
    echo "📡 Background streaming server into ${SERVER_LOG}..."
    stream_container "server" "vllm-tpu" "$SERVER_LOG" "SERVER" "$CLR_S" "bg"

    echo "📊 Foreground streaming & teeing client benchmark results to terminal..."
    stream_container "client" "benchmark-client" "$CLIENT_LOG" "CLIENT" "$CLR_C" "tee"
fi
