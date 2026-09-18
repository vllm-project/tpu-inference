#!/bin/bash
# ==============================================================================
# TPU Testcase Log Streaming & Tee Utility
#
# Slimmed-down companion to tee_logs.sh, dedicated to the Helm chart deployed by
# .buildkite/gke/helm/run_testcase.sh (mode: "script").
#
# That chart renders exactly ONE replicatedJob:
#   replicatedjob-name = runner
#     initContainers : [image-builder], tpu-node-setup, [git-sync]
#     container      : test-runner  (+ [gke-gcsfuse-sidecar] when GCS storage)
#
# The container list is discovered from the live Pod spec, so optional
# containers (image-builder, git-sync, gcsfuse) are picked up automatically.
#
# Everything related to the benchmark stack (client / p / d / x / server) and
# generate_summary.py reporting has been removed, since none of it is ever
# deployed in script mode.
# ==============================================================================
set -eu

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GKE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

REP_JOB="runner"
DEFAULT_MAIN_CONTAINER="test-runner"

# Resolve default log directory (gke/log)
LOG_DIR="${GKE_ROOT}/log"

CURRENT_USER="${USER:-$(whoami)}"
CLEAN_USER="$(printf '%s' "$CURRENT_USER" | tr '[:upper:]' '[:lower:]' | tr -dc 'a-z0-9')"

JOB_NAME=""
LOG_NUM=""
CONTAINER="$DEFAULT_MAIN_CONTAINER"
FOLLOW=true
WAIT_TIMEOUT=900   # seconds to wait for the runner pod to appear

# Colors used to tag each stream in 'all' mode
PALETTE=($'\033[36m' $'\033[32m' $'\033[33m' $'\033[35m' $'\033[34m' $'\033[31m')
CLR_RESET=$'\033[0m'

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
  LOG_NUMBER        Optional numeric suffix for the log file (e.g. 4 -> <JOB>.log4).

Options:
  -j, --job <NAME>     Specify JobSet name explicitly
  -n, --number <NUM>   Specify log number suffix explicitly
  -c, --container <C>  Container to read (default: ${DEFAULT_MAIN_CONTAINER}).
                       Use 'all' to stream every container of the Pod
                       concurrently, each tagged and teed to its own file.
                       Individual names: image-builder, tpu-node-setup,
                       git-sync, test-runner (whichever the Pod actually has).
  -l, --list           List the containers of the Pod and exit
  -s, --dump           Snapshot current logs without following (-f)
  -o, --dir <DIR>      Output directory (default: ${LOG_DIR})
  -t, --timeout <SEC>  Seconds to wait for the runner pod (default: ${WAIT_TIMEOUT})
  -h, --help           Show this help message

Log files:
  main container  -> <JOB_NAME>.log[N]
  other container -> <JOB_NAME>.<container>.log[N]
  An existing file is never overwritten; the suffix is bumped instead.

Exit code mirrors the testcase result: the main container's exit code
(0 = Succeeded, non-zero = Failed).

Examples:
  1. Stream the main test-runner of the newest run:
     $0

  2. Stream EVERY container side by side (build + setup + test):
     $0 -c all

  3. Watch only the on-demand image build:
     $0 -c image-builder

  4. Snapshot all logs of a finished run:
     $0 -c all --dump dennis-test-a1b2c

  5. See which containers the Pod has:
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

# ==============================================================================
# Pod / container introspection helpers
# ==============================================================================

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

# --- WAIT FOR THE RUNNER POD (needed to enumerate containers) ---
if ! pod_exists; then
    if [ "$FOLLOW" = false ]; then
        echo "❌ Error: no '${REP_JOB}' pods found for '${JOB_NAME}'." >&2
        exit 1
    fi
    echo "⏳ Waiting for the ${REP_JOB} pod (timeout ${WAIT_TIMEOUT}s)..."
    WAITED=0
    while ! pod_exists; do
        if [ "$WAITED" -ge "$WAIT_TIMEOUT" ]; then
            echo "❌ Error: timed out waiting for the ${REP_JOB} pod." >&2
            exit 1
        fi
        sleep 2
        WAITED=$((WAITED + 2))
    done
fi

INIT_CONTAINERS="$(pod_init_containers)"
MAIN_CONTAINERS="$(pod_main_containers)"
ALL_CONTAINERS="${INIT_CONTAINERS} ${MAIN_CONTAINERS}"

# The container whose exit code decides our own exit code.
MAIN_CONTAINER=""
for c in $MAIN_CONTAINERS; do
    if [ "$c" = "$DEFAULT_MAIN_CONTAINER" ]; then
        MAIN_CONTAINER="$c"
    fi
done
if [ -z "$MAIN_CONTAINER" ]; then
    for c in $MAIN_CONTAINERS; do
        MAIN_CONTAINER="$c"
        break
    done
fi

if [ "$LIST_ONLY" = true ]; then
    echo "Containers of ${JOB_NAME} (${REP_JOB}):"
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
    exit 0
fi

# --- RESOLVE TARGET CONTAINERS ---
if [ "$CONTAINER" = "all" ]; then
    TARGETS="$ALL_CONTAINERS"
else
    FOUND=false
    for c in $ALL_CONTAINERS; do
        if [ "$c" = "$CONTAINER" ]; then
            FOUND=true
        fi
    done
    if [ "$FOUND" = false ]; then
        echo "❌ Error: container '${CONTAINER}' does not exist in this Pod." >&2
        echo "   Available: ${ALL_CONTAINERS}" >&2
        echo "   (use '-c all' to stream all of them, or '--list' to inspect)" >&2
        exit 2
    fi
    TARGETS="$CONTAINER"
fi

# ==============================================================================
# Log file naming
# ==============================================================================
# main container  -> <JOB_NAME>.log[N]
# other container -> <JOB_NAME>.<container>.log[N]
log_base_for() {
    if [ "$1" = "$MAIN_CONTAINER" ]; then
        printf '%s/%s' "$LOG_DIR" "$JOB_NAME"
    else
        printf '%s/%s.%s' "$LOG_DIR" "$JOB_NAME" "$1"
    fi
}

log_path_for() {
    printf '%s.log%s' "$(log_base_for "$1")" "$LOG_SUFFIX"
}

# A suffix is usable only if it is free for *every* target, so that one run's
# files always share the same suffix.
suffix_is_free() {
    local sfx="$1" c
    for c in $TARGETS; do
        if [ -e "$(log_base_for "$c").log${sfx}" ]; then
            return 1
        fi
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

echo "============================================================"
echo " ⚡ TPU Testcase Log Streamer & Tee Utility"
echo "============================================================"
echo " JobSet Name : ${JOB_NAME}"
echo " Pod Phase   : $(pod_phase)"
echo " Mode        : $([ "$FOLLOW" = true ] && echo stream || echo dump) (container: ${CONTAINER})"
echo " Target Logs :"
for c in $TARGETS; do
    printf "   - %-18s : %s\n" "$c" "$(log_path_for "$c")"
done
echo "============================================================"

summary() {
    echo ""
    echo "============================================================"
    echo " 📋 Captured logs in ${LOG_DIR}:"
    local c f lines size
    for c in $TARGETS; do
        f="$(log_path_for "$c")"
        if [ -f "$f" ]; then
            lines="$(wc -l < "$f" | tr -d ' ')"
            size="$(ls -lh "$f" | awk '{print $5}')"
            printf "   - %-34s : %8s (%s lines)\n" "$(basename "$f")" "$size" "$lines"
        fi
    done
    echo "============================================================"
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
EXIT_TARGET="$MAIN_CONTAINER"
if [ "$CONTAINER" != "all" ]; then
    EXIT_TARGET="$CONTAINER"
fi

# --- DUMP MODE: one-shot snapshot ---
if [ "$FOLLOW" = false ]; then
    for c in $TARGETS; do
        echo "📥 Snapshotting ${c} -> $(basename "$(log_path_for "$c")")..."
        dump_container "$c" "$(log_path_for "$c")"
    done
    summary
    CODE="$(container_exit_code "$EXIT_TARGET")"
    echo ""
    echo "🏁 ${EXIT_TARGET} exit code: ${CODE:-unknown} (pod phase: $(pod_phase))"
    exit "${CODE:-1}"
fi

# --- STREAM MODE ---
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

if [ "$CONTAINER" != "all" ]; then
    # Single container: plain, untagged output in the foreground.
    echo "📡 Streaming & teeing ${CONTAINER} -> $(basename "$(log_path_for "$CONTAINER")")..."
    stream_container "$CONTAINER" "$(log_path_for "$CONTAINER")" "" ""
else
    # All containers concurrently, each tagged with its own color and file.
    echo "📺 Streaming ${TARGETS} concurrently (Ctrl+C to stop)..."
    i=0
    for c in $TARGETS; do
        color="${PALETTE[$(( i % ${#PALETTE[@]} ))]}"
        stream_container "$c" "$(log_path_for "$c")" "$c" "$color" &
        PIDS+=("$!")
        i=$((i + 1))
    done
    wait ${PIDS[@]+"${PIDS[@]}"} 2>/dev/null || true
fi

CODE="$(container_exit_code "$EXIT_TARGET")"
echo ""
echo "🏁 ${EXIT_TARGET} exit code: ${CODE:-unknown} (pod phase: $(pod_phase))"
exit "${CODE:-1}"
