#!/bin/bash
# ==============================================================================
# TPU Testcase Log Streaming & Tee Utility
#
# Slimmed-down companion to tee_logs.sh, dedicated to the Helm chart deployed by
# .buildkite/gke/helm/run_testcase.sh (mode: "script").
#
# That chart renders exactly ONE replicatedJob:
#   replicatedjob-name = runner
#     initContainers : tpu-node-setup, [git-sync]
#     container      : test-runner
#
# Everything related to the benchmark stack (client / p / d / x / server),
# multiplexed output, colored tags and generate_summary.py reporting has been
# removed, since none of it is ever deployed in script mode.
# ==============================================================================
set -eu

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GKE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

REP_JOB="runner"
CONTAINER="test-runner"

# Resolve default log directory (gke/log)
LOG_DIR="${GKE_ROOT}/log"

CURRENT_USER="${USER:-$(whoami)}"
CLEAN_USER="$(printf '%s' "$CURRENT_USER" | tr '[:upper:]' '[:lower:]' | tr -dc 'a-z0-9')"

JOB_NAME=""
LOG_NUM=""
FOLLOW=true
WAIT_TIMEOUT=900   # seconds to wait for the runner pod to appear

usage() {
    cat <<EOF
==================================================================
 TPU Testcase Log Streamer & Tee Utility (helm mode: script)
==================================================================
Usage: $0 [options] [JOB_NAME] [LOG_NUMBER]

Arguments (positional, order independent):
  JOB_NAME          JobSet / Helm release name from run_testcase.sh.
                    If omitted, auto-detects the newest JobSet that owns a
                    '${REP_JOB}' replicatedJob (prefers '${CLEAN_USER}-test-*').
                    The log is always saved to '<JOB_NAME>.log' (or '<JOB_NAME>.log<N>').
  LOG_NUMBER        Optional numeric suffix for the log file (e.g. 4 -> <JOB_NAME>.log4).

Options:
  -j, --job <NAME>     Specify JobSet name explicitly
  -n, --number <NUM>   Specify log number suffix explicitly
  -c, --container <C>  Container to read (default: ${CONTAINER};
                       use 'git-sync' or 'tpu-node-setup' for init containers)
  -s, --dump           Snapshot current logs without following (-f)
  -o, --dir <DIR>      Output directory (default: ${LOG_DIR})
  -t, --timeout <SEC>  Seconds to wait for the runner pod (default: ${WAIT_TIMEOUT})
  -h, --help           Show this help message

Exit code mirrors the testcase result: 0 = Succeeded, 1 = Failed/unknown.

Examples:
  1. Stream the newest testcase run into log/<detected_jobset>.log:
     $0

  2. Stream a specific release into log/<JOB_NAME>.log:
     $0 dennis-test-a1b2c

  3. Stream a specific release with numeric suffix into log/<JOB_NAME>.log4:
     $0 dennis-test-a1b2c 4

  4. Snapshot the logs of a finished run:
     $0 --dump dennis-test-a1b2c

  5. Inspect the git-sync init container instead:
     $0 -c git-sync --dump
==================================================================
EOF
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
        -c|--container)
            CONTAINER="$2"
            shift 2
            ;;
        -s|--save|--dump)
            FOLLOW=false
            shift
            ;;
        -o|--dir)
            LOG_DIR="$2"
            shift 2
            ;;
        -t|--timeout)
            WAIT_TIMEOUT="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        -*)
            echo "❌ Error: unknown option '$1'" >&2
            usage >&2
            exit 2
            ;;
        *)
            POSITIONAL+=("$1")
            shift
            ;;
    esac
done

for arg in ${POSITIONAL[@]+"${POSITIONAL[@]}"}; do
    if [[ "$arg" =~ ^[0-9]+$ ]] && [ -z "$LOG_NUM" ]; then
        LOG_NUM="$arg"
    elif [ -z "$JOB_NAME" ]; then
        JOB_NAME="$arg"
    else
        echo "❌ Error: unexpected argument '$arg'" >&2
        exit 2
    fi
done

if ! command -v kubectl >/dev/null 2>&1; then
    echo "❌ Error: kubectl is required." >&2
    exit 1
fi

mkdir -p "${LOG_DIR}"

# --- AUTO-DISCOVER THE TESTCASE JOBSET ---
# Only JobSets exposing a 'runner' replicatedJob come from run_testcase.sh.
if [ -z "$JOB_NAME" ]; then
    echo "🔍 Detecting latest testcase JobSet (replicatedJob '${REP_JOB}')..."
    CANDIDATES="$(kubectl get jobset \
        -o jsonpath="{range .items[?(@.spec.replicatedJobs[*].name=='${REP_JOB}')]}{.metadata.name}{'\n'}{end}" \
        2>/dev/null || true)"

    if [ -n "$CLEAN_USER" ]; then
        JOB_NAME="$(printf '%s\n' "$CANDIDATES" | grep "^${CLEAN_USER}-" | tail -n 1 || true)"
    fi
    if [ -z "$JOB_NAME" ]; then
        JOB_NAME="$(printf '%s\n' "$CANDIDATES" | grep -v '^$' | tail -n 1 || true)"
    fi

    if [ -z "$JOB_NAME" ]; then
        echo "❌ Error: no testcase JobSet found. Pass the release name explicitly." >&2
        exit 1
    fi
    echo "   Found JobSet: ${JOB_NAME}"
fi

LABEL="jobset.sigs.k8s.io/jobset-name=${JOB_NAME},jobset.sigs.k8s.io/replicatedjob-name=${REP_JOB}"

# Verify the JobSet (or at least its pods) exists
if ! kubectl get jobset "${JOB_NAME}" >/dev/null 2>&1; then
    echo "⚠️  JobSet '${JOB_NAME}' not found, falling back to pod lookup..."
    if [ -z "$(kubectl get pods -l "$LABEL" -o jsonpath='{.items[*].metadata.name}' 2>/dev/null)" ]; then
        echo "❌ Error: no '${REP_JOB}' pods found for '${JOB_NAME}'." >&2
        exit 1
    fi
fi

# --- LOG FILE DETERMINATION (AVOID OVERWRITING) ---
if [ -n "$LOG_NUM" ]; then
    TESTCASE_LOG="${LOG_DIR}/${JOB_NAME}.log${LOG_NUM}"
    if [ -e "$TESTCASE_LOG" ]; then
        SUFFIX=$((LOG_NUM + 1))
        while [ -e "${LOG_DIR}/${JOB_NAME}.log${SUFFIX}" ]; do
            SUFFIX=$((SUFFIX + 1))
        done
        TESTCASE_LOG="${LOG_DIR}/${JOB_NAME}.log${SUFFIX}"
        LOG_NUM="$SUFFIX"
    fi
else
    TESTCASE_LOG="${LOG_DIR}/${JOB_NAME}.log"
    if [ -e "$TESTCASE_LOG" ]; then
        SUFFIX=2
        while [ -e "${LOG_DIR}/${JOB_NAME}.log${SUFFIX}" ]; do
            SUFFIX=$((SUFFIX + 1))
        done
        TESTCASE_LOG="${LOG_DIR}/${JOB_NAME}.log${SUFFIX}"
        LOG_NUM="$SUFFIX"
    fi
fi

echo "============================================================"
echo " ⚡ TPU Testcase Log Streamer & Tee Utility"
echo "============================================================"
echo " JobSet Name : ${JOB_NAME}"
echo " Target      : ${REP_JOB} / ${CONTAINER}"
if [ -n "$LOG_NUM" ]; then
    echo " Log Suffix  : ${LOG_NUM}"
fi
echo " Mode        : $([ "$FOLLOW" = true ] && echo stream || echo dump)"
echo " Target Log  : ${TESTCASE_LOG}"
echo "============================================================"

summary() {
    if [ -f "${TESTCASE_LOG}" ]; then
        LINES="$(wc -l < "${TESTCASE_LOG}" | tr -d ' ')"
        SIZE="$(ls -lh "${TESTCASE_LOG}" | awk '{print $5}')"
        echo ""
        echo "============================================================"
        printf " 📋 Captured %-20s : %8s (%s lines)\n" "$(basename "${TESTCASE_LOG}")" "$SIZE" "$LINES"
        echo "============================================================"
    fi
}

pod_phase() {
    kubectl get pods -l "$LABEL" -o jsonpath='{.items[0].status.phase}' 2>/dev/null || true
}

# --- DUMP MODE: one-shot snapshot, no waiting ---
if [ "$FOLLOW" = false ]; then
    echo "📥 Snapshotting ${REP_JOB} (${CONTAINER}) logs..."
    kubectl logs -l "$LABEL" -c "$CONTAINER" --tail=-1 > "${TESTCASE_LOG}" 2>&1 || true
    summary
    [ "$(pod_phase)" = "Succeeded" ] && exit 0 || exit 1
fi

# --- WAIT FOR THE RUNNER POD ---
echo "⏳ Waiting for the ${REP_JOB} pod (timeout ${WAIT_TIMEOUT}s)..."
WAITED=0
while [ -z "$(kubectl get pods -l "$LABEL" -o jsonpath='{.items[*].metadata.name}' 2>/dev/null)" ]; do
    if [ "$WAITED" -ge "$WAIT_TIMEOUT" ]; then
        echo "❌ Error: timed out waiting for the ${REP_JOB} pod." >&2
        exit 1
    fi
    sleep 2
    WAITED=$((WAITED + 2))
done

trap 'summary' EXIT INT TERM

# --- STREAM & TEE ---
# kubectl logs -f drops out while the pod is still pulling images / running init
# containers (tpu-node-setup, git-sync), so retry until the job reaches a
# terminal phase. Truncate once, then append across retries.
echo "📡 Streaming & teeing ${REP_JOB} (${CONTAINER}) to screen -> $(basename "${TESTCASE_LOG}")..."
: > "${TESTCASE_LOG}"
while true; do
    kubectl logs -l "$LABEL" -c "$CONTAINER" --tail=-1 -f 2>&1 | tee -a "${TESTCASE_LOG}" || true
    PHASE="$(pod_phase)"
    if [ "$PHASE" = "Succeeded" ] || [ "$PHASE" = "Failed" ] || [ -z "$PHASE" ]; then
        break
    fi
    sleep 2
done

echo ""
echo "🏁 Testcase pod phase: ${PHASE:-unknown}"
[ "$PHASE" = "Succeeded" ] && exit 0 || exit 1
