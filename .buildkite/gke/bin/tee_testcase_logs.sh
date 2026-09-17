#!/bin/bash
# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# ==============================================================================
# TPU Testcase Log Streaming & Tee Utility
#
# Companion to tee_logs.sh, dedicated to Helm charts deployed by
# .buildkite/gke/helm/run_testcase.sh (mode: "script").
#
# Supports one or more replicatedJobs under the scriptJobs list (e.g. unittest,
# accuracy, benchmark). Streams each step to a separate log file
# (<JOB_NAME>-<STEP>.log), or a single <JOB_NAME>.log for single-job runs.
# ==============================================================================
set -eu

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GKE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

REP_JOB=""
CONTAINER="test-runner"

# Resolve default log directory (gke/log)
LOG_DIR="${GKE_ROOT}/log"

CURRENT_USER="${USER:-$(whoami)}"
CLEAN_USER="$(printf '%s' "$CURRENT_USER" | tr '[:upper:]' '[:lower:]' | tr -dc 'a-z0-9')"

JOB_NAME=""
LOG_NUM=""
FOLLOW=true
WAIT_TIMEOUT=900   # seconds to wait for each pod to appear

usage() {
    cat <<EOF
==================================================================
 TPU Testcase Log Streamer & Tee Utility (helm mode: script)
==================================================================
Usage: $0 [options] [JOB_NAME] [LOG_NUMBER]

Arguments (positional, order independent):
  JOB_NAME          JobSet / Helm release name from run_testcase.sh.
                    If omitted, auto-detects the newest testcase JobSet.
                    Logs are saved to '<JOB_NAME>-<step>.log' (or '<JOB_NAME>.log').
  LOG_NUMBER        Optional numeric suffix for the log file (e.g. 4 -> *.log4).

Options:
  -j, --job <NAME>             Specify JobSet name explicitly
  -r, --replicated-job <NAME>  Target specific ReplicatedJob (e.g. 'benchmark')
  -n, --number <NUM>           Specify log number suffix explicitly
  -c, --container <C>          Container to read (default: ${CONTAINER};
                               use 'git-sync' or 'tpu-node-setup' for init containers)
  -s, --dump                   Snapshot current logs without following (-f)
  -o, --dir <DIR>              Output directory (default: ${LOG_DIR})
  -t, --timeout <SEC>          Seconds to wait for the pod (default: ${WAIT_TIMEOUT})
  -h, --help                   Show this help message

Exit code mirrors the testcase result: 0 = Succeeded, 1 = Failed/unknown.

Examples:
  1. Stream newest testcase run (all steps into separate log files):
     $0

  2. Stream a specific release:
     $0 dennis-test-a1b2c

  3. Stream only the benchmark step of a release:
     $0 -r benchmark dennis-test-a1b2c

  4. Snapshot the logs of a finished run:
     $0 --dump dennis-test-a1b2c
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
        -r|--replicated-job)
            REP_JOB="$2"
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
if [ -z "$JOB_NAME" ]; then
    echo "🔍 Detecting latest testcase JobSet..."
    CANDIDATES="$(kubectl get jobset \
        -o jsonpath="{range .items[?(@.spec.replicatedJobs[*].template.metadata.labels.role=='test-runner')]}{.metadata.name}{'\n'}{end}" \
        2>/dev/null || true)"

    if [ -z "$CANDIDATES" ]; then
        CANDIDATES="$(kubectl get jobset \
            -o jsonpath="{range .items[?(@.spec.replicatedJobs[*].name=='runner')]}{.metadata.name}{'\n'}{end}" \
            2>/dev/null || true)"
    fi
    if [ -z "$CANDIDATES" ] && [ -n "$CLEAN_USER" ]; then
        CANDIDATES="$(kubectl get jobset -o jsonpath='{range .items[*]}{.metadata.name}{"\n"}{end}' 2>/dev/null | grep "^${CLEAN_USER}-" || true)"
    fi

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

# --- RESOLVE TARGET REPLICATED JOBS ---
TARGET_JOBS=()
if [ -n "$REP_JOB" ]; then
    TARGET_JOBS=("$REP_JOB")
else
    DISCOVERED_JOBS="$(kubectl get jobset "${JOB_NAME}" -o jsonpath='{.spec.replicatedJobs[*].name}' 2>/dev/null || true)"
    if [ -n "$DISCOVERED_JOBS" ]; then
        read -r -a TARGET_JOBS <<< "$DISCOVERED_JOBS"
    else
        TARGET_JOBS=("runner")
    fi
fi

TOTAL_JOBS="${#TARGET_JOBS[@]}"
CURRENT_LOG_FILE=""

summary() {
    if [ -n "${CURRENT_LOG_FILE}" ] && [ -f "${CURRENT_LOG_FILE}" ]; then
        LINES="$(wc -l < "${CURRENT_LOG_FILE}" | tr -d ' ')"
        SIZE="$(ls -lh "${CURRENT_LOG_FILE}" | awk '{print $5}')"
        echo ""
        echo "============================================================"
        printf " 📋 Captured %-20s : %8s (%s lines)\n" "$(basename "${CURRENT_LOG_FILE}")" "$SIZE" "$LINES"
        echo "============================================================"
    fi
}

stream_single_job() {
    local TARGET_REP="$1"
    local STEP_NUM="$2"

    local LABEL="jobset.sigs.k8s.io/jobset-name=${JOB_NAME},jobset.sigs.k8s.io/replicatedjob-name=${TARGET_REP}"

    # Determine log file: separate file per step if multiple jobs exist
    local BASE_NAME="${JOB_NAME}"
    if [ "$TOTAL_JOBS" -gt 1 ]; then
        BASE_NAME="${JOB_NAME}-${TARGET_REP}"
    fi

    local TESTCASE_LOG=""
    if [ -n "$LOG_NUM" ]; then
        TESTCASE_LOG="${LOG_DIR}/${BASE_NAME}.log${LOG_NUM}"
        if [ -e "$TESTCASE_LOG" ]; then
            local SUFFIX=$((LOG_NUM + 1))
            while [ -e "${LOG_DIR}/${BASE_NAME}.log${SUFFIX}" ]; do
                SUFFIX=$((SUFFIX + 1))
            done
            TESTCASE_LOG="${LOG_DIR}/${BASE_NAME}.log${SUFFIX}"
        fi
    else
        TESTCASE_LOG="${LOG_DIR}/${BASE_NAME}.log"
        if [ -e "$TESTCASE_LOG" ]; then
            local SUFFIX=2
            while [ -e "${LOG_DIR}/${BASE_NAME}.log${SUFFIX}" ]; do
                SUFFIX=$((SUFFIX + 1))
            done
            TESTCASE_LOG="${LOG_DIR}/${BASE_NAME}.log${SUFFIX}"
        fi
    fi

    CURRENT_LOG_FILE="${TESTCASE_LOG}"

    echo "============================================================"
    echo " ⚡ TPU Testcase Log Streamer & Tee Utility"
    if [ "$TOTAL_JOBS" -gt 1 ]; then
        echo " Step        : [${STEP_NUM}/${TOTAL_JOBS}] ${TARGET_REP}"
    fi
    echo " JobSet Name : ${JOB_NAME}"
    echo " Target      : ${TARGET_REP} / ${CONTAINER}"
    if [ -n "$LOG_NUM" ]; then
        echo " Log Suffix  : ${LOG_NUM}"
    fi
    echo " Mode        : $([ "$FOLLOW" = true ] && echo stream || echo dump)"
    echo " Target Log  : ${TESTCASE_LOG}"
    echo "============================================================"

    pod_phase() {
        kubectl get pods -l "$LABEL" -o jsonpath='{.items[0].status.phase}' 2>/dev/null || true
    }

    # --- DUMP MODE ---
    if [ "$FOLLOW" = false ]; then
        echo "📥 Snapshotting ${TARGET_REP} (${CONTAINER}) logs..."
        kubectl logs -l "$LABEL" -c "$CONTAINER" --tail=-1 > "${TESTCASE_LOG}" 2>&1 || true
        summary
        local DUMP_PHASE="$(pod_phase)"
        [ "$DUMP_PHASE" = "Succeeded" ] && return 0 || return 1
    fi

    # --- WAIT FOR POD ---
    echo "⏳ Waiting for the ${TARGET_REP} pod (timeout ${WAIT_TIMEOUT}s)..."
    local WAITED=0
    while [ -z "$(kubectl get pods -l "$LABEL" -o jsonpath='{.items[*].metadata.name}' 2>/dev/null)" ]; do
        if [ "$WAITED" -ge "$WAIT_TIMEOUT" ]; then
            echo "❌ Error: timed out waiting for the ${TARGET_REP} pod." >&2
            return 1
        fi
        sleep 2
        WAITED=$((WAITED + 2))
    done

    trap 'summary' EXIT INT TERM

    echo "📡 Streaming & teeing ${TARGET_REP} (${CONTAINER}) to screen -> $(basename "${TESTCASE_LOG}")..."
    : > "${TESTCASE_LOG}"
    local PHASE=""
    while true; do
        kubectl logs -l "$LABEL" -c "$CONTAINER" --tail=-1 -f 2>&1 | tee -a "${TESTCASE_LOG}" || true
        PHASE="$(pod_phase)"
        if [ "$PHASE" = "Succeeded" ] || [ "$PHASE" = "Failed" ] || [ -z "$PHASE" ]; then
            break
        fi
        sleep 2
    done

    echo ""
    echo "🏁 Testcase step '${TARGET_REP}' pod phase: ${PHASE:-unknown}"
    summary
    [ "$PHASE" = "Succeeded" ] && return 0 || return 1
}

# --- EXECUTE STREAMING ACROSS ALL TARGET REPLICATED JOBS ---
STEP_IDX=1
for JOB in "${TARGET_JOBS[@]}"; do
    if ! stream_single_job "$JOB" "$STEP_IDX"; then
        echo "❌ Step '${JOB}' failed! Aborting log streaming." >&2
        exit 1
    fi
    STEP_IDX=$((STEP_IDX + 1))
done

echo ""
echo "🎉 All testcase step(s) in JobSet '${JOB_NAME}' completed successfully."
exit 0
