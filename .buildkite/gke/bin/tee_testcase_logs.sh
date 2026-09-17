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
# Such a chart has TWO dimensions that both need to be followed:
#
#   1. Steps (replicatedJobs) -- one per entry of `scriptJobs` in the values
#      file, e.g. unittest / accuracy / benchmark. The JobSet runs them
#      sequentially (startupPolicyOrder: InOrder) and fails fast, so this
#      script walks them in the same order and aborts as soon as one fails.
#
#   2. Containers inside each step's Pod:
#        initContainers : [image-builder], tpu-node-setup, [git-sync]
#        containers     : test-runner (+ [gke-gcsfuse-sidecar] on GCS storage)
#      The container list is discovered from the live Pod spec, so optional
#      containers are picked up automatically. '-c all' streams every one of
#      them concurrently, each tagged and teed to its own file -- handy when
#      the on-demand image build (image-builder) is the part you care about.
#
# Everything related to the benchmark stack (client / p / d / x / server) and
# generate_summary.py reporting has been removed, since none of it is ever
# deployed in script mode.
# ==============================================================================
set -eu

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GKE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

DEFAULT_MAIN_CONTAINER="test-runner"

# Resolve default log directory (gke/log)
LOG_DIR="${GKE_ROOT}/log"

CURRENT_USER="${USER:-$(whoami)}"
CLEAN_USER="$(printf '%s' "$CURRENT_USER" | tr '[:upper:]' '[:lower:]' | tr -dc 'a-z0-9')"

JOB_NAME=""
REP_JOB=""          # empty => every replicatedJob of the JobSet, in order
LOG_NUM=""
CONTAINER="$DEFAULT_MAIN_CONTAINER"
FOLLOW=true
WAIT_TIMEOUT=900   # seconds to wait for each step's pod to appear

# Colors used to tag each stream in 'all' mode
PALETTE=($'\033[36m' $'\033[32m' $'\033[33m' $'\033[35m' $'\033[34m' $'\033[31m')
CLR_RESET=$'\033[0m'

usage() {
    cat <<EOF
==================================================================
 TPU Testcase Log Streamer & Tee Utility (helm mode: script)
==================================================================
Usage: $0 [options] [JOB_NAME] [LOG_NUMBER]

Follows a testcase JobSet along both dimensions:
  * every step (replicatedJob: unittest / accuracy / benchmark ...),
    sequentially, in the order declared by the JobSet;
  * every container of the step's Pod when '-c all' is given.

Arguments (positional, order independent):
  JOB_NAME          JobSet / Helm release name from run_testcase.sh.
                    If omitted, auto-detects the newest testcase JobSet
                    (prefers '${CLEAN_USER}-test-*').
  LOG_NUMBER        Optional numeric suffix for the log files (e.g. 4 -> *.log4).

Options:
  -j, --job <NAME>             Specify JobSet name explicitly
  -r, --replicated-job <NAME>  Follow only this step (e.g. 'benchmark');
                               default is every step of the JobSet
  -n, --number <NUM>           Specify log number suffix explicitly
  -c, --container <C>          Container to read (default: ${DEFAULT_MAIN_CONTAINER}).
                               Use 'all' (or -a) to stream every container of
                               each step concurrently, tagged and teed to its
                               own file. Individual names: image-builder,
                               tpu-node-setup, git-sync, test-runner
                               (whichever the Pod actually has).
  -a, --all                    Shorthand for '-c all'
  -l, --list                   List the containers of each step's Pod and exit
  -s, --dump                   Snapshot current logs without following (-f)
  -o, --dir <DIR>              Output directory (default: ${LOG_DIR})
  -t, --timeout <SEC>          Seconds to wait for each step's pod (default: ${WAIT_TIMEOUT})
  -h, --help                   Show this help message

Log files (the step name is always part of the file name):
  main container  -> <JOB_NAME>-<step>.log[N]
  other container -> <JOB_NAME>-<step>.<container>.log[N]
  An existing file is never overwritten; the whole run shares one suffix,
  bumped until it is free for every step.

Exit code mirrors the testcase result: the first step whose main container
exits non-zero aborts the run and its exit code is propagated.

Examples:
  1. Stream the main test-runner of every step of the newest run:
     $0

  2. Stream EVERY container side by side (build + setup + test):
     $0 -c all

  3. Follow only the benchmark step of a given release:
     $0 -r benchmark dennis-test-a1b2c

  4. Watch only the on-demand image build of the first step:
     $0 -c image-builder

  5. Snapshot all logs of a finished run:
     $0 -c all --dump dennis-test-a1b2c

  6. See which containers each step's Pod has:
     $0 --list
==================================================================
EOF
}

# --- PARSE ARGUMENTS ---
LIST_ONLY=false
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
        -a|--all)
            CONTAINER="all"
            shift
            ;;
        -l|--list)
            LIST_ONLY=true
            shift
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
# Testcase pods carry `role: test-runner`, whatever the step is called, so that
# label is the reliable marker of a chart deployed by run_testcase.sh.
if [ -z "$JOB_NAME" ]; then
    echo "🔍 Detecting latest testcase JobSet..."
    CANDIDATES="$(kubectl get jobset \
        -o jsonpath="{range .items[?(@.spec.replicatedJobs[*].template.spec.template.metadata.labels.role=='test-runner')]}{.metadata.name}{'\n'}{end}" \
        2>/dev/null || true)"

    if [ -z "$CANDIDATES" ] && [ -n "$CLEAN_USER" ]; then
        CANDIDATES="$(kubectl get jobset -o jsonpath='{range .items[*]}{.metadata.name}{"\n"}{end}' 2>/dev/null \
            | grep "^${CLEAN_USER}-" || true)"
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

# --- RESOLVE THE STEPS (replicatedJobs) TO FOLLOW ---
STEPS=()
if [ -n "$REP_JOB" ]; then
    STEPS=("$REP_JOB")
else
    DISCOVERED="$(kubectl get jobset "${JOB_NAME}" -o jsonpath='{.spec.replicatedJobs[*].name}' 2>/dev/null || true)"
    if [ -z "$DISCOVERED" ]; then
        echo "❌ Error: cannot read the replicatedJobs of JobSet '${JOB_NAME}'." >&2
        echo "   Is the release deployed? Use -r <step> to target one explicitly." >&2
        exit 1
    fi
    read -r -a STEPS <<< "$DISCOVERED"
fi
TOTAL_STEPS="${#STEPS[@]}"

# ==============================================================================
# Pod / container introspection helpers
#
# All of them read $LABEL, which is re-pointed at the current step by
# select_step() before that step is processed.
# ==============================================================================
LABEL=""
STEP=""

select_step() {
    STEP="$1"
    LABEL="jobset.sigs.k8s.io/jobset-name=${JOB_NAME},jobset.sigs.k8s.io/replicatedjob-name=${STEP}"
}

pod_exists() {
    [ -n "$(kubectl get pods -l "$LABEL" -o jsonpath='{.items[*].metadata.name}' 2>/dev/null)" ]
}

pod_phase() {
    kubectl get pods -l "$LABEL" -o jsonpath='{.items[0].status.phase}' 2>/dev/null || true
}

pod_terminal() {
    case "$(pod_phase)" in
        Succeeded|Failed|"") return 0 ;;
        *) return 1 ;;
    esac
}

# Container names straight from the Pod spec: init containers first (in the
# order they run), then the regular containers.
pod_init_containers() {
    kubectl get pods -l "$LABEL" -o jsonpath='{.items[0].spec.initContainers[*].name}' 2>/dev/null || true
}

pod_main_containers() {
    kubectl get pods -l "$LABEL" -o jsonpath='{.items[0].spec.containers[*].name}' 2>/dev/null || true
}

# Read one status field of a container, regardless of whether it is an init
# container or a regular one (only one of the two lookups can ever match).
container_field() {
    local name="$1" field="$2"
    kubectl get pods -l "$LABEL" -o \
        jsonpath="{.items[0].status.initContainerStatuses[?(@.name=='${name}')].${field}}{.items[0].status.containerStatuses[?(@.name=='${name}')].${field}}" \
        2>/dev/null || true
}

container_terminated() {
    [ -n "$(container_field "$1" 'state.terminated.reason')" ]
}

container_started() {
    [ -n "$(container_field "$1" 'state.running.startedAt')" ]
}

container_exit_code() {
    container_field "$1" 'state.terminated.exitCode'
}

# The container whose exit code decides the step's result: the conventional
# test-runner if present, otherwise the first regular container.
pick_main_container() {
    local mains="$1" c
    for c in $mains; do
        if [ "$c" = "$DEFAULT_MAIN_CONTAINER" ]; then
            printf '%s' "$c"
            return 0
        fi
    done
    for c in $mains; do
        printf '%s' "$c"
        return 0
    done
}

# ==============================================================================
# Log file naming
# ==============================================================================
# main container  -> <JOB_NAME>-<step>.log[N]
# other container -> <JOB_NAME>-<step>.<container>.log[N]
#
# MAIN_CONTAINER is the main container of the step currently being processed.
MAIN_CONTAINER=""

log_base_for() {
    if [ "$1" = "$MAIN_CONTAINER" ]; then
        printf '%s/%s-%s' "$LOG_DIR" "$JOB_NAME" "$STEP"
    else
        printf '%s/%s-%s.%s' "$LOG_DIR" "$JOB_NAME" "$STEP" "$1"
    fi
}

log_path_for() {
    printf '%s.log%s' "$(log_base_for "$1")" "$LOG_SUFFIX"
}

# One suffix for the whole run, so every file produced by a single invocation
# shares it. The container names of steps that have not started yet are still
# unknown, hence the glob over '<JOB>-<step>*.log<N>'.
suffix_is_free() {
    local sfx="$1" step f
    for step in "${STEPS[@]}"; do
        for f in "${LOG_DIR}/${JOB_NAME}-${step}.log${sfx}" \
                 "${LOG_DIR}/${JOB_NAME}-${step}."*".log${sfx}"; do
            if [ -e "$f" ]; then
                return 1
            fi
        done
    done
    return 0
}

if [ -n "$LOG_NUM" ]; then
    LOG_SUFFIX="$LOG_NUM"
else
    LOG_SUFFIX=""
fi
if ! suffix_is_free "$LOG_SUFFIX"; then
    NEXT=$(( ${LOG_NUM:-1} + 1 ))
    while ! suffix_is_free "$NEXT"; do
        NEXT=$((NEXT + 1))
    done
    LOG_SUFFIX="$NEXT"
fi

# Every file written so far, for the final summary (steps are sequential, so
# this grows as the run progresses).
WRITTEN_LOGS=()

summary() {
    if [ "${#WRITTEN_LOGS[@]}" -eq 0 ]; then
        return
    fi
    echo ""
    echo "============================================================"
    echo " 📋 Captured logs in ${LOG_DIR}:"
    local f lines size
    for f in "${WRITTEN_LOGS[@]}"; do
        if [ -f "$f" ]; then
            lines="$(wc -l < "$f" | tr -d ' ')"
            size="$(ls -lh "$f" | awk '{print $5}')"
            printf "   - %-42s : %8s (%s lines)\n" "$(basename "$f")" "$size" "$lines"
        fi
    done
    echo "============================================================"
}

# ==============================================================================
# Per-step setup
# ==============================================================================
# Fills MAIN_CONTAINER / INIT_CONTAINERS / MAIN_CONTAINERS / TARGETS for the
# step selected by select_step(). Returns 1 if the pod does not exist.
resolve_step_containers() {
    if ! pod_exists; then
        return 1
    fi
    INIT_CONTAINERS="$(pod_init_containers)"
    MAIN_CONTAINERS="$(pod_main_containers)"
    ALL_CONTAINERS="${INIT_CONTAINERS} ${MAIN_CONTAINERS}"
    MAIN_CONTAINER="$(pick_main_container "$MAIN_CONTAINERS")"

    if [ "$CONTAINER" = "all" ]; then
        TARGETS="$ALL_CONTAINERS"
        return 0
    fi

    local c
    for c in $ALL_CONTAINERS; do
        if [ "$c" = "$CONTAINER" ]; then
            TARGETS="$CONTAINER"
            return 0
        fi
    done
    echo "❌ Error: container '${CONTAINER}' does not exist in the '${STEP}' pod." >&2
    echo "   Available: ${ALL_CONTAINERS}" >&2
    echo "   (use '-c all' to stream all of them, or '--list' to inspect)" >&2
    exit 2
}

# Waits for the step's pod, counting the timeout only from the moment the step
# is reached (earlier steps may legitimately run for hours before this one).
wait_for_step_pod() {
    if pod_exists; then
        return 0
    fi
    echo "⏳ Waiting for the ${STEP} pod (timeout ${WAIT_TIMEOUT}s)..."
    local waited=0
    while ! pod_exists; do
        if [ "$waited" -ge "$WAIT_TIMEOUT" ]; then
            echo "❌ Error: timed out waiting for the ${STEP} pod." >&2
            return 1
        fi
        sleep 2
        waited=$((waited + 2))
    done
    return 0
}

# ==============================================================================
# Streaming
# ==============================================================================
# Streams ONE container until that container terminates.
#
# Two things the previous single-container version got wrong and that matter a
# lot once init containers are involved:
#   1. The retry loop used to exit only when the whole Pod reached a terminal
#      phase. An init container finishes long before that, so `kubectl logs -f`
#      returned immediately and the same log got re-appended every 2s forever.
#      We now poll the *container's own* terminated state.
#   2. `kubectl logs` fails with "is waiting to start: PodInitializing" until
#      the container actually starts, and that error used to be teed into the
#      log file on every retry. We now wait for the container to start first
#      and keep stderr out of the file.
stream_container() {
    local container="$1"
    local outfile="$2"
    local tag="$3"     # empty => no prefix, print raw to stdout
    local color="$4"

    : > "$outfile"

    # Phase 1: wait until the container starts (or is already done).
    while ! container_started "$container" && ! container_terminated "$container"; do
        if pod_terminal; then
            break
        fi
        sleep 2
    done

    # Phase 2: stream until THIS container terminates.
    while true; do
        if [ -n "$tag" ]; then
            kubectl logs -l "$LABEL" -c "$container" --tail=-1 -f 2>/dev/null \
                | tee -a "$outfile" \
                | awk -v col="$color" -v tag="$tag" -v rst="$CLR_RESET" \
                    '{ printf "%s[%-16s]%s %s\n", col, tag, rst, $0; fflush() }' || true
        else
            kubectl logs -l "$LABEL" -c "$container" --tail=-1 -f 2>/dev/null \
                | tee -a "$outfile" || true
        fi

        if container_terminated "$container" || pod_terminal; then
            break
        fi
        sleep 2
    done
    return 0
}

dump_container() {
    local container="$1"
    local outfile="$2"
    kubectl logs -l "$LABEL" -c "$container" --tail=-1 > "$outfile" 2>&1 || true
}

# The container whose result we report: in 'all' mode that is the main
# container, otherwise the single container the user asked for.
exit_target() {
    if [ "$CONTAINER" = "all" ]; then
        printf '%s' "$MAIN_CONTAINER"
    else
        printf '%s' "$CONTAINER"
    fi
}

step_banner() {
    local idx="$1"
    echo ""
    echo "============================================================"
    echo " ⚡ TPU Testcase Log Streamer & Tee Utility"
    echo " Step        : [${idx}/${TOTAL_STEPS}] ${STEP}"
    echo " JobSet Name : ${JOB_NAME}"
    echo " Pod Phase   : $(pod_phase)"
    echo " Mode        : $([ "$FOLLOW" = true ] && echo stream || echo dump) (container: ${CONTAINER})"
    echo " Target Logs :"
    local c
    for c in $TARGETS; do
        printf "   - %-18s : %s\n" "$c" "$(log_path_for "$c")"
    done
    echo "============================================================"
}

# ==============================================================================
# --- LIST MODE: show the containers of every step and exit ---
# ==============================================================================
if [ "$LIST_ONLY" = true ]; then
    for STEP_NAME in "${STEPS[@]}"; do
        select_step "$STEP_NAME"
        echo "Containers of ${JOB_NAME} (step: ${STEP_NAME}):"
        if ! pod_exists; then
            echo "  (pod not created yet)"
            continue
        fi
        INIT_CONTAINERS="$(pod_init_containers)"
        MAIN_CONTAINERS="$(pod_main_containers)"
        MAIN_CONTAINER="$(pick_main_container "$MAIN_CONTAINERS")"
        for c in $INIT_CONTAINERS; do
            echo "  [init] ${c}"
        done
        for c in $MAIN_CONTAINERS; do
            if [ "$c" = "$MAIN_CONTAINER" ]; then
                echo "  [main] ${c}   <- default target"
            else
                echo "  [main] ${c}"
            fi
        done
    done
    exit 0
fi

# ==============================================================================
# --- DUMP MODE: one-shot snapshot of every step that already has a pod ---
# ==============================================================================
if [ "$FOLLOW" = false ]; then
    trap summary EXIT INT TERM
    DUMPED=0
    FIRST_FAILURE=""
    STEP_IDX=0
    for STEP_NAME in "${STEPS[@]}"; do
        STEP_IDX=$((STEP_IDX + 1))
        select_step "$STEP_NAME"
        if ! resolve_step_containers; then
            echo "⏭️  Step '${STEP_NAME}' has no pod yet, skipping."
            continue
        fi
        step_banner "$STEP_IDX"
        for c in $TARGETS; do
            echo "📥 Snapshotting ${c} -> $(basename "$(log_path_for "$c")")..."
            dump_container "$c" "$(log_path_for "$c")"
            WRITTEN_LOGS+=("$(log_path_for "$c")")
        done
        DUMPED=$((DUMPED + 1))
        TARGET="$(exit_target)"
        CODE="$(container_exit_code "$TARGET")"
        echo "🏁 ${STEP_NAME}/${TARGET} exit code: ${CODE:-unknown} (pod phase: $(pod_phase))"
        if [ -z "$FIRST_FAILURE" ] && [ "${CODE:-1}" -ne 0 ]; then
            FIRST_FAILURE="${CODE:-1}"
        fi
    done

    if [ "$DUMPED" -eq 0 ]; then
        echo "❌ Error: none of the steps of '${JOB_NAME}' has a pod to snapshot." >&2
        exit 1
    fi
    exit "${FIRST_FAILURE:-0}"
fi

# ==============================================================================
# --- STREAM MODE: walk the steps in order, abort on the first failure ---
# ==============================================================================
PIDS=()
CLEANED=false
cleanup() {
    if [ "$CLEANED" = true ]; then
        return
    fi
    CLEANED=true
    for p in ${PIDS[@]+"${PIDS[@]}"}; do
        kill "$p" >/dev/null 2>&1 || true
    done
    summary
}
trap cleanup EXIT INT TERM

STEP_IDX=0
for STEP_NAME in "${STEPS[@]}"; do
    STEP_IDX=$((STEP_IDX + 1))
    select_step "$STEP_NAME"

    if ! wait_for_step_pod; then
        exit 1
    fi
    resolve_step_containers
    step_banner "$STEP_IDX"

    for c in $TARGETS; do
        WRITTEN_LOGS+=("$(log_path_for "$c")")
    done

    if [ "$CONTAINER" != "all" ]; then
        # Single container: plain, untagged output in the foreground.
        echo "📡 Streaming & teeing ${STEP_NAME}/${CONTAINER} -> $(basename "$(log_path_for "$CONTAINER")")..."
        stream_container "$CONTAINER" "$(log_path_for "$CONTAINER")" "" ""
    else
        # All containers concurrently, each tagged with its own color and file.
        echo "📺 Streaming ${STEP_NAME}: ${TARGETS} concurrently (Ctrl+C to stop)..."
        PIDS=()
        i=0
        for c in $TARGETS; do
            color="${PALETTE[$(( i % ${#PALETTE[@]} ))]}"
            stream_container "$c" "$(log_path_for "$c")" "$c" "$color" &
            PIDS+=("$!")
            i=$((i + 1))
        done
        wait ${PIDS[@]+"${PIDS[@]}"} 2>/dev/null || true
        PIDS=()
    fi

    TARGET="$(exit_target)"
    CODE="$(container_exit_code "$TARGET")"
    echo ""
    echo "🏁 Step [${STEP_IDX}/${TOTAL_STEPS}] '${STEP_NAME}' -> ${TARGET} exit code: ${CODE:-unknown} (pod phase: $(pod_phase))"
    if [ "${CODE:-1}" -ne 0 ]; then
        echo "❌ Step '${STEP_NAME}' failed! Aborting the remaining steps." >&2
        exit "${CODE:-1}"
    fi
done

echo ""
if [ "$TOTAL_STEPS" -gt 1 ]; then
    echo "🎉 All ${TOTAL_STEPS} step(s) of JobSet '${JOB_NAME}' completed successfully."
else
    echo "🎉 JobSet '${JOB_NAME}' completed successfully."
fi
exit 0
