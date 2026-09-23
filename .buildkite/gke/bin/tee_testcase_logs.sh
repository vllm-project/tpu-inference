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
# Reads Helm releases deployed by .buildkite/gke/helm/run_testcase.sh. JobSet
# auto-detection keys off the `role: test-runner` pod label, and steps are
# walked one after another to match the chart's `startupPolicyOrder: InOrder`
# plus fail-fast policy.
#
# A script-mode chart has TWO dimensions that both need to be followed:
#
#   1. Steps (replicatedJobs) -- one per entry of `scriptJobs` in the values
#      file, e.g. unittest / accuracy / benchmark. The JobSet runs them
#      sequentially (startupPolicyOrder: InOrder) and fails fast, so this
#      script walks them in the same order and aborts as soon as one fails.
#
#   2. Containers inside each step's Pod:
#        test-runner (+ [gke-gcsfuse-sidecar] on GCS storage)
#      The container list is discovered from the live Pod spec, so optional
#      containers are picked up automatically. '-c all' streams every one of
#      them concurrently, each tagged and teed to its own file.
#
#      NOTE: the on-demand image build is NOT here. It runs before the JobSet
#      exists, as a CPU-only Helm pre-install hook Job, and run_testcase.sh
#      tees it to log/<release>.image-builder.log.
#
# Pods are not stable: Kueue can evict and requeue the whole JobSet mid-run
# (preemption, TAS node failures, ...), and a Job can restart a pod on backoff.
# A missing pod therefore does NOT mean the run is over -- only the step's Job
# condition (Complete/Failed) does. When a pod disappears the streamer waits for
# its replacement, re-attaches, and records the switch in the log file.
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
 TPU Testcase Log Streamer & Tee Utility
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
                               own file. Individual names: test-runner,
                               gke-gcsfuse-sidecar (whichever the Pod
                               actually has). The image build is not part of
                               the JobSet; see run_testcase.sh.
  -a, --all                    Shorthand for '-c all'
  -l, --list                   List the containers of each step's Pod and exit
  -s, --dump                   Snapshot current logs without following (-f)
  -o, --dir <DIR>              Output directory (default: ${LOG_DIR})
  -t, --timeout <SEC>          Seconds to wait for each step's pod (default: ${WAIT_TIMEOUT});
                               time spent Suspended in the Kueue queue does not count
  -h, --help                   Show this help message

Log files (the step name is always part of the file name):
  main container  -> <JOB_NAME>-<step>.log[N]
  other container -> <JOB_NAME>-<step>.<container>.log[N]
  An existing file is never overwritten; the whole run shares one suffix,
  bumped until it is free for every step.

Evictions: if a pod is destroyed mid-run (Kueue preemption, TAS node failures,
Job backoff), the streamer waits for the replacement pod, re-attaches and marks
the switch in the log file. Only the step's Job condition ends the wait.

Exit code mirrors the testcase result: the first step whose main container
exits non-zero aborts the run and its exit code is propagated. If the JobSet is
deleted mid-run (helm uninstall, cleanup.sh) the run is reported as cancelled
and the exit code is 130.

Examples:
  1. Stream the main test-runner of every step of the newest run:
     $0

  2. Stream EVERY container side by side (setup + test):
     $0 -c all

  3. Follow only the benchmark step of a given release:
     $0 -r benchmark dennis-test-a1b2c

  4. Snapshot all logs of a finished run:
     $0 -c all --dump dennis-test-a1b2c

  5. See which containers each step's Pod has:
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
# Pod / Job / JobSet introspection helpers
#
# $LABEL is re-pointed at the current step by select_step(); $POD holds the one
# pod we are currently reading. Pinning a pod name matters during an eviction:
# the outgoing pod lingers in Terminating while its replacement is created, and
# a bare `kubectl logs -l <label>` would happily mix the two.
# ==============================================================================
LABEL=""
STEP=""
POD=""

select_step() {
    STEP="$1"
    LABEL="jobset.sigs.k8s.io/jobset-name=${JOB_NAME},jobset.sigs.k8s.io/replicatedjob-name=${STEP}"
    POD=""
}

# Newest pod of the current step that is not already being deleted.
current_pod() {
    kubectl get pods -l "$LABEL" --sort-by=.metadata.creationTimestamp \
        -o go-template='{{range .items}}{{.metadata.name}}{{"\t"}}{{if .metadata.deletionTimestamp}}deleting{{else}}live{{end}}{{"\n"}}{{end}}' \
        2>/dev/null | awk -F'\t' '$2 == "live" { name = $1 } END { if (name != "") print name }' || true
}

# True once the pinned pod is gone or has been marked for deletion, i.e. the
# stream we were following will never produce anything again.
pod_lost() {
    if [ -z "$POD" ]; then
        return 0
    fi
    local state
    state="$(kubectl get pod "$POD" \
        -o go-template='{{if .metadata.deletionTimestamp}}deleting{{else}}live{{end}}' 2>/dev/null || true)"
    [ "$state" != "live" ]
}

pod_phase() {
    if [ -z "$POD" ]; then
        return 0
    fi
    kubectl get pod "$POD" -o jsonpath='{.status.phase}' 2>/dev/null || true
}

pod_terminal() {
    case "$(pod_phase)" in
        Succeeded|Failed) return 0 ;;
        *) return 1 ;;
    esac
}

# Container names straight from the Pod spec: init containers first (in the
# order they run), then the regular containers.
pod_init_containers() {
    kubectl get pod "$POD" -o jsonpath='{.spec.initContainers[*].name}' 2>/dev/null || true
}

pod_main_containers() {
    kubectl get pod "$POD" -o jsonpath='{.spec.containers[*].name}' 2>/dev/null || true
}

# Read one status field of a container, regardless of whether it is an init
# container or a regular one (only one of the two lookups can ever match).
container_field() {
    local name="$1" field="$2"
    if [ -z "$POD" ]; then
        return 0
    fi
    kubectl get pod "$POD" -o \
        jsonpath="{.status.initContainerStatuses[?(@.name=='${name}')].${field}}{.status.containerStatuses[?(@.name=='${name}')].${field}}" \
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

# --- Job / JobSet: the only sources that survive a pod being destroyed ---
#
# A pod can vanish for reasons that have nothing to do with the test finishing
# (Kueue preemption, TAS node failures, Job backoff). The step's Job condition
# is what actually says "this step is over".
job_condition() {
    kubectl get jobs -l "$LABEL" \
        -o jsonpath="{.items[0].status.conditions[?(@.type=='$1')].status}" 2>/dev/null || true
}

job_complete()  { [ "$(job_condition Complete)" = "True" ]; }
job_failed()    { [ "$(job_condition Failed)" = "True" ]; }
job_suspended() { [ "$(job_condition Suspended)" = "True" ]; }
step_finished() { job_complete || job_failed; }

job_state() {
    if job_complete; then
        printf 'Complete'
    elif job_failed; then
        printf 'Failed'
    elif job_suspended; then
        printf 'Suspended'
    else
        printf 'Running'
    fi
}

# Whether the JobSet is still there.
#
# The naive version of this -- `[ -n "$(kubectl get jobset ... 2>/dev/null)" ]`
# -- conflates four situations, because all of them leave stdout empty: the
# JobSet really was deleted, the API server throttled us, the request timed out,
# or credentials needed refreshing. Every caller treats "gone" as a reason to
# stop, so one unlucky poll used to abort a perfectly healthy wait -- and it
# looked exactly like a deliberate `helm uninstall`, which made it hard to spot.
#
# So: only the API positively reporting NotFound counts as proof of deletion.
# Anything else is an inconclusive poll, which we report and tolerate for
# JOBSET_MISS_LIMIT rounds (~10s at the 2s poll interval) before giving up.
JOBSET_MISSES=0
JOBSET_MISS_LIMIT="${JOBSET_MISS_LIMIT:-5}"

jobset_exists() {
    # `|| rc=$?` keeps the assignment out of `set -e`'s reach no matter how this
    # function is called; reading $? after a bare assignment would only be safe
    # from inside an `if`, which is how every current caller happens to invoke
    # it -- but that is not a property worth depending on.
    local out rc=0
    out="$(kubectl get jobset "$JOB_NAME" -o jsonpath='{.metadata.name}' 2>&1)" || rc=$?
    if [ "$rc" -eq 0 ] && [ -n "$out" ]; then
        JOBSET_MISSES=0
        return 0
    fi
    case "$out" in
        *NotFound*|*'not found'*)
            return 1
            ;;
    esac
    JOBSET_MISSES=$((JOBSET_MISSES + 1))
    echo "   ⚠️  Could not determine whether JobSet '${JOB_NAME}' still exists" \
         "(${JOBSET_MISSES}/${JOBSET_MISS_LIMIT}): ${out}" >&2
    [ "$JOBSET_MISSES" -lt "$JOBSET_MISS_LIMIT" ]
}

# JobSet-wide brake: once the JobSet itself is Failed, no replacement pod is
# ever coming for any step.
jobset_failed() {
    [ "$(kubectl get jobset "$JOB_NAME" -o jsonpath='{.status.terminalState}' 2>/dev/null || true)" = "Failed" ]
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
# Fills MAIN_CONTAINER / INIT_CONTAINERS / MAIN_CONTAINERS / TARGETS from the
# pod currently pinned in $POD. Returns 1 when there is no live pod.
resolve_step_containers() {
    if [ -z "$POD" ]; then
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

# Waits for the step's pod and pins it in $POD.
#
# Exit codes:
#   0 - a live pod is available
#   1 - timed out (or the JobSet vanished)
#   2 - the step already finished and its pod has been reclaimed
#
# The timeout is only counted from the moment the step is reached, and it is not
# counted at all while the Job is Suspended: with Kueue a workload can sit in the
# queue for far longer than WAIT_TIMEOUT before it is admitted.
wait_for_step_pod() {
    POD="$(current_pod)"
    if [ -n "$POD" ]; then
        return 0
    fi
    if step_finished; then
        return 2
    fi

    echo "⏳ Waiting for the ${STEP} pod (timeout ${WAIT_TIMEOUT}s; time spent Suspended in the queue is not counted)..."
    local waited=0 announced_suspend=false
    while [ -z "$POD" ]; do
        if job_suspended; then
            if [ "$announced_suspend" = false ]; then
                echo "   ⏸️  Job is Suspended (queued / requeued by Kueue) -- waiting without a deadline."
                announced_suspend=true
            fi
            waited=0
        elif [ "$waited" -ge "$WAIT_TIMEOUT" ]; then
            echo "❌ Error: timed out waiting for the ${STEP} pod." >&2
            return 1
        fi
        if ! jobset_exists; then
            # Neutral wording on purpose: the caller decides whether this is a
            # cancellation (helm uninstall) or a genuine error.
            echo "   ℹ️  JobSet '${JOB_NAME}' no longer exists; stopping the wait." >&2
            return 1
        fi
        sleep 2
        waited=$((waited + 2))
        POD="$(current_pod)"
        if [ -z "$POD" ] && step_finished; then
            return 2
        fi
    done
    return 0
}

# ==============================================================================
# Streaming
# ==============================================================================
# Streams ONE container until that container terminates for good.
#
# Three things matter here:
#   1. An init container finishes long before the Pod reaches a terminal phase,
#      so the retry loop polls the *container's own* terminated state instead of
#      the Pod phase. Otherwise `kubectl logs -f` returns immediately once the
#      init container is done and the same log gets re-appended every 2s.
#   2. `kubectl logs` fails with "is waiting to start: PodInitializing" until the
#      container actually starts, and that error must not be teed into the log
#      file, hence the wait-for-start phase and the discarded stderr.
#   3. The Pod can be destroyed and recreated underneath us -- Kueue preemption,
#      TAS node failures ("Workload eviction triggered due to ... node
#      failures"), or a Job backoff restart. That is NOT the end of the run: the
#      JobSet gets suspended, requeued and resumed minutes later with a brand
#      new Pod. We therefore re-attach to the replacement Pod and keep going,
#      and only stop when the step's Job reports Complete/Failed (or the JobSet
#      as a whole fails / disappears).
stream_once() {
    local container="$1" outfile="$2" tag="$3" color="$4"
    if [ -n "$tag" ]; then
        kubectl logs "$POD" -c "$container" --tail=-1 -f 2>/dev/null \
            | tee -a "$outfile" \
            | awk -v col="$color" -v tag="$tag" -v rst="$CLR_RESET" \
                '{ printf "%s[%-16s]%s %s\n", col, tag, rst, $0; fflush() }' || true
    else
        kubectl logs "$POD" -c "$container" --tail=-1 -f 2>/dev/null \
            | tee -a "$outfile" || true
    fi
}

# Records a pod replacement both on screen and inside the log file, so the file
# never silently mixes the output of two different pods.
note_pod_replacement() {
    local container="$1" outfile="$2" attempt="$3" old_pod="$4"
    local ts marker suffix=""
    ts="$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
    if job_suspended; then
        suffix=", job is Suspended -- evicted/requeued (Kueue)"
    fi
    marker="===== [tee] ${STEP}/${container}: pod ${old_pod} disappeared${suffix}; waiting for its replacement (attempt ${attempt}, ${ts}) ====="
    printf '%s\n' "$marker" >> "$outfile"
    echo "⚠️  $marker" >&2
}

stream_container() {
    local container="$1"
    local outfile="$2"
    local tag="$3"     # empty => no prefix, print raw to stdout
    local color="$4"
    local attempt=1 old_pod=""

    : > "$outfile"

    while true; do
        # --- Attach: make sure we are pinned to a live pod of this step. ---
        POD="$(current_pod)"
        while [ -z "$POD" ]; do
            # No pod: either the step is genuinely over, or it is being
            # rescheduled. Only the Job / JobSet can tell us which.
            if step_finished || jobset_failed || ! jobset_exists; then
                return 0
            fi
            sleep 3
            POD="$(current_pod)"
        done

        # Phase 1: wait until the container starts (or is already done).
        while ! container_started "$container" && ! container_terminated "$container"; do
            if pod_lost || pod_terminal; then
                break
            fi
            sleep 2
        done

        # Phase 2: stream until THIS container terminates, or the pod vanishes.
        while true; do
            stream_once "$container" "$outfile" "$tag" "$color"
            if container_terminated "$container"; then
                return 0
            fi
            if pod_lost; then
                break
            fi
            if pod_terminal; then
                return 0
            fi
            sleep 2
        done

        # --- The pod we were pinned to is gone. ---
        if step_finished || jobset_failed || ! jobset_exists; then
            return 0
        fi
        old_pod="$POD"
        attempt=$((attempt + 1))
        note_pod_replacement "$container" "$outfile" "$attempt" "$old_pod"
        POD=""
    done
}

dump_container() {
    local container="$1"
    local outfile="$2"
    kubectl logs "$POD" -c "$container" --tail=-1 > "$outfile" 2>&1 || true
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

# Result of the current step, most precise source first:
#   1. the container's own terminated.exitCode (needs the pod to still exist);
#   2. the step's Job condition (survives pod deletion / Kueue requeue);
#   3. nothing -- the caller reports "unknown" and treats it as a failure.
step_exit_code() {
    local target="$1" code=""
    if [ -n "$POD" ] && [ -n "$target" ]; then
        code="$(container_exit_code "$target")"
    fi
    if [ -n "$code" ]; then
        printf '%s' "$code"
        return 0
    fi
    if job_complete; then
        printf '0'
    elif job_failed; then
        printf '1'
    fi
}

step_banner() {
    local idx="$1"
    echo ""
    echo "============================================================"
    echo " ⚡ TPU Testcase Log Streamer & Tee Utility"
    echo " Step        : [${idx}/${TOTAL_STEPS}] ${STEP}"
    echo " JobSet Name : ${JOB_NAME}"
    echo " Pod / Phase : ${POD:-none} ($(pod_phase))"
    echo " Job State   : $(job_state)"
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
        POD="$(current_pod)"
        echo "Containers of ${JOB_NAME} (step: ${STEP_NAME}, job: $(job_state)):"
        if [ -z "$POD" ]; then
            echo "  (no live pod)"
            continue
        fi
        echo "  pod: ${POD}"
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
# --- DUMP MODE: one-shot snapshot of every step that still has a live pod ---
# ==============================================================================
if [ "$FOLLOW" = false ]; then
    trap summary EXIT INT TERM
    DUMPED=0
    FIRST_FAILURE=""
    STEP_IDX=0
    for STEP_NAME in "${STEPS[@]}"; do
        STEP_IDX=$((STEP_IDX + 1))
        select_step "$STEP_NAME"
        POD="$(current_pod)"
        if ! resolve_step_containers; then
            echo "⏭️  Step '${STEP_NAME}' has no live pod (job: $(job_state)), skipping."
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
        CODE="$(step_exit_code "$TARGET")"
        echo "🏁 ${STEP_NAME}/${TARGET} exit code: ${CODE:-unknown} (job: $(job_state))"
        if [ -z "$FIRST_FAILURE" ] && [ "${CODE:-1}" -ne 0 ]; then
            FIRST_FAILURE="${CODE:-1}"
        fi
    done

    if [ "$DUMPED" -eq 0 ]; then
        echo "❌ Error: none of the steps of '${JOB_NAME}' has a live pod to snapshot." >&2
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

    WAIT_RC=0
    wait_for_step_pod || WAIT_RC=$?
    if [ "$WAIT_RC" -eq 1 ]; then
        # wait_for_step_pod also returns 1 on a plain timeout, so only claim
        # cancellation when the JobSet really is gone.
        if ! jobset_exists; then
            echo "🛑 JobSet '${JOB_NAME}' was deleted (helm uninstall?) - run cancelled." >&2
            exit 130
        fi
        exit 1
    fi

    TARGET=""
    if [ "$WAIT_RC" -eq 0 ]; then
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
    else
        # WAIT_RC == 2: the step finished before we got here and its pod has
        # already been reclaimed, so there is nothing left to stream.
        echo ""
        echo "⏭️  Step [${STEP_IDX}/${TOTAL_STEPS}] '${STEP_NAME}' already finished and its pod is gone; nothing to stream."
    fi

    # A vanished JobSet means someone tore the run down (helm uninstall,
    # cleanup.sh, kubectl delete) rather than the step actually failing. Report
    # that distinctly, and with the conventional "interrupted" exit code, so it
    # is not mistaken for a red test result.
    if ! jobset_exists; then
        echo ""
        echo "🛑 JobSet '${JOB_NAME}' was deleted (helm uninstall?) - run cancelled." >&2
        exit 130
    fi

    # The pod we streamed may have been replaced or deleted in the meantime, so
    # re-pin before reading the result.
    POD="$(current_pod)"
    CODE="$(step_exit_code "$TARGET")"
    PHASE="$(pod_phase)"
    echo ""
    echo "🏁 Step [${STEP_IDX}/${TOTAL_STEPS}] '${STEP_NAME}' -> ${TARGET:-job status} exit code: ${CODE:-unknown} (pod phase: ${PHASE:-none}, job: $(job_state))"
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
