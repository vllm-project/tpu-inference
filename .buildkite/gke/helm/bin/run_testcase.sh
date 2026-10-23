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



set -euo pipefail


# --- Usage information ---
usage() {
    echo "Usage: $0 -f <values-*.yaml> [-r <release-name>]"
    echo
    echo "Install the Helm testcase using the specified values file."
    echo
    echo "Required:"
    echo "  -f <file>   Values file name (values.yaml or values-*.yaml) located"
    echo "              next to this script. No default is assumed."
    echo
    echo "Optional:"
    echo "  -r <name>   Helm release name (default: auto-generated)"
    echo "  -h          Show this help message"
    echo
    echo "Environment:"
    echo "  BUILD_TIMEOUT   Overrides builder.timeout from the values file (default: 90m)."
    echo
    echo "Note: when builder.enabled is true and the target image is not yet in the"
    echo "      registry, the image is built by a CPU-only Helm pre-install hook Job and"
    echo "      this script blocks until that build finishes, streaming it to"
    echo "      ../log/<release>.image-builder.log. No TPU capacity is reserved while"
    echo "      building. If the image already exists, the hook is skipped entirely."
    echo
    echo "Example: $0 -f values-transfer-template.yaml -r my-helm-test"
}


# --- Command-line option parsing ---
VALUES_FILE=""
RELEASE_NAME=""


while getopts ":f:r:h" option; do
    case "$option" in
        f)
            VALUES_FILE="$OPTARG"
            ;;
        r)
            RELEASE_NAME="$OPTARG"
            ;;
        h)
            usage
            exit 0
            ;;
        :)
            if [[ "$OPTARG" == "f" ]]; then
                echo "Error: -f requires a values file name." >&2
            else
                echo "Error: -r requires a release name." >&2
            fi
            usage >&2
            exit 2
            ;;
        \?)
            echo "Error: unknown option -$OPTARG." >&2
            usage >&2
            exit 2
            ;;
    esac
done


# --- Values-file validation ---
if [[ -z "$VALUES_FILE" ]]; then
    echo "Error: -f is required. Specify the values file to deploy." >&2
    usage >&2
    exit 2
fi


if [[ "$VALUES_FILE" == */* || ( "$VALUES_FILE" != "values.yaml" && "$VALUES_FILE" != values-*.yaml ) ]]; then
    echo "Error: -f must be the name of values.yaml or a values-*.yaml file." >&2
    exit 2
fi


# --- Resolve the Helm chart and selected values file ---
# This script lives in <chart>/bin/, so the chart root is one level up.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CHART_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
VALUES_PATH="$CHART_DIR/$VALUES_FILE"


if [[ ! -f "$VALUES_PATH" ]]; then
    echo "Error: values file not found: $VALUES_PATH" >&2
    exit 2
fi


# --- Derive the per-user Hugging Face token secret name ---
CURRENT_USER="${USER:-$(whoami)}"
CLEAN_USER="$(printf '%s' "$CURRENT_USER" | tr '[:upper:]' '[:lower:]' | tr -dc 'a-z0-9')"
if [[ -z "$CLEAN_USER" ]]; then
    echo "Error: could not derive a valid Helm release prefix from the current user." >&2
    exit 2
fi


HF_TOKEN_SECRET="${CLEAN_USER}-test-token"


# --- Verify that the required Hugging Face token secret exists ---
if ! command -v kubectl >/dev/null 2>&1; then
    echo "Error: kubectl is required to check the Hugging Face token secret." >&2
    exit 1
fi


HF_SECRET_KEY="$(awk -F: '/^[[:space:]]*key[[:space:]]*:/ { gsub(/[[:space:]"'\'' ]/, "", $2); print $2; exit }' "$VALUES_PATH")"
HF_SECRET_KEY="${HF_SECRET_KEY:-token}"


echo "🔍 Checking Kubernetes secret '${HF_TOKEN_SECRET}' (key: '${HF_SECRET_KEY}') in current namespace..."
SECRET_CHECK_OUTPUT=""
if ! SECRET_CHECK_OUTPUT="$(kubectl get secret "$HF_TOKEN_SECRET" -o jsonpath="{.data.${HF_SECRET_KEY}}" 2>&1)" || [[ -z "$SECRET_CHECK_OUTPUT" ]]; then
    if [[ "$SECRET_CHECK_OUTPUT" == *"not found"* ]] || ! kubectl get secret "$HF_TOKEN_SECRET" >/dev/null 2>&1; then
        echo "❌ Error: Kubernetes secret '${HF_TOKEN_SECRET}' does not exist in the current namespace." >&2
    else
        echo "❌ Error: Key '${HF_SECRET_KEY}' in Kubernetes secret '${HF_TOKEN_SECRET}' does not exist or is empty." >&2
    fi
    echo "" >&2
    echo "Create or update it with:" >&2
    echo "  kubectl create secret generic '${HF_TOKEN_SECRET}' \\" >&2
    echo "    --from-literal=${HF_SECRET_KEY}='<your-hugging-face-token>' \\" >&2
    echo "    --dry-run=client -o yaml | kubectl apply -f -" >&2
    if [[ -n "$SECRET_CHECK_OUTPUT" && "$SECRET_CHECK_OUTPUT" != *"not found"* ]]; then
        echo "" >&2
        echo "Error details:" >&2
        echo "$SECRET_CHECK_OUTPUT" >&2
    fi
    exit 1
fi
echo "✅ Kubernetes secret '${HF_TOKEN_SECRET}' (key: '${HF_SECRET_KEY}') verified successfully."


# --- Generate and validate the Helm release name ---
RANDOM_SUFFIX="$(LC_ALL=C od -An -N8 -tx1 /dev/urandom | tr -d ' \n' | head -c 5)"
DEFAULT_RELEASE_NAME="${CLEAN_USER}-test-${RANDOM_SUFFIX}"
RELEASE_NAME="${RELEASE_NAME:-$DEFAULT_RELEASE_NAME}"


if (( ${#RELEASE_NAME} > 53 )) || [[ ! "$RELEASE_NAME" =~ ^[a-z0-9]([-a-z0-9]*[a-z0-9])?$ ]]; then
    echo "Error: release name must be 1-53 characters of lowercase letters, digits, or hyphens, and cannot start or end with a hyphen." >&2
    exit 2
fi


JOBSET_NAME="$RELEASE_NAME"


cd "$CHART_DIR"


# Read a scalar declared directly under the top-level "builder:" block of the
# selected values file. Only that block is inspected, because keys such as
# "enabled:" also appear under other sections (features, script.git, ...).
# Requiring exactly two leading spaces also skips the nested "cache:" and
# "resources:" sub-blocks.
builder_value() {
    local key="$1" default="$2" found
    found="$(awk -v key="$key" '
        /^[^[:space:]#]/ { in_builder = ($0 ~ /^builder[[:space:]]*:/) }
        in_builder && $0 ~ ("^  " key "[[:space:]]*:") {
            sub(/^[^:]*:/, "")
            sub(/#.*/, "")
            gsub(/[[:space:]"'\'']/, "")
            print
            exit
        }
    ' "$VALUES_PATH")"
    printf '%s' "${found:-$default}"
}

# Normalises a Kubernetes storage quantity (300Gi, 1Ti, 500G, ...) to whole GiB
# so that two spellings of the same size are not mistaken for a mismatch.
# Unparseable input yields 0, which makes the caller stay quiet.
to_gib() {
    awk -v q="$1" 'BEGIN {
        if (match(q, /^[0-9.]+/) == 0) { print 0; exit }
        n = substr(q, 1, RLENGTH) + 0
        u = substr(q, RLENGTH + 1)
        if (u == "Gi") ;                                  # already GiB
        else if (u == "Pi") n *= 1024 * 1024
        else if (u == "Ti") n *= 1024
        else if (u == "Mi") n /= 1024
        else if (u == "Ki") n /= 1024 * 1024
        else if (u == "P") n *= 1000000000000000 / 1073741824
        else if (u == "T") n *= 1000000000000 / 1073741824
        else if (u == "G") n *= 1000000000 / 1073741824
        else if (u == "M") n *= 1000000 / 1073741824
        else if (u == "k") n *= 1000 / 1073741824
        else n /= 1073741824                              # no suffix means bytes
        printf "%d", n
    }'
}


BUILDER_ENABLED="$(builder_value enabled false)"
# Helm blocks on the pre-install hook, so this bounds the whole build.
# Precedence: BUILD_TIMEOUT env > builder.timeout in the values file > 90m.
BUILD_TIMEOUT="${BUILD_TIMEOUT:-$(builder_value timeout 90m)}"

LOG_DIR="${CHART_DIR}/log"
BUILD_LOG="${LOG_DIR}/${RELEASE_NAME}.image-builder.log"
BUILDER_JOB="${RELEASE_NAME}-image-builder"

HELM_ARGS=(--set hfTokenSecret.name="$HF_TOKEN_SECRET")
STREAM_BUILD=""


# --- Image builder pre-flight ---
# The image build runs as a Helm pre-install hook Job on a CPU node, so
# `helm install` blocks until it finishes (see templates/image-builder-job.yaml
# for why it cannot live inside the JobSet). Two things happen here to keep that
# bearable:
#   1. Ask the registry directly whether the target image already exists. On a hit
#      the hook is disabled outright and `helm install` returns immediately,
#      instead of spending ~40-90s scheduling a pod just to hit the in-pod
#      fast-path and exit.
#   2. Make sure the shared build cache volume exists before the hook runs.
if [[ "$BUILDER_ENABLED" == "true" ]]; then
    # Render the hook manifest rather than re-deriving the tag here, so the
    # client-side check can never disagree with what the builder would target.
    BUILDER_MANIFEST="$(helm template "$RELEASE_NAME" . -f "$VALUES_FILE" \
        --set builder.enabled=true \
        --show-only templates/image-builder-job.yaml 2>/dev/null || true)"
    TARGET_IMAGE="$(printf '%s\n' "$BUILDER_MANIFEST" |
        awk -F'"' '/^[[:space:]]*TARGET_IMAGE=/ { print $2; exit }')"
    CACHE_CLAIM="$(printf '%s\n' "$BUILDER_MANIFEST" |
        awk '/^[[:space:]]*claimName:/ { print $2; exit }')"

    IMAGE_PRESENT=""
    if [[ -z "$TARGET_IMAGE" ]]; then
        echo "⚠️  Could not resolve the target image from the chart; skipping the registry pre-check."
    elif ! command -v gcloud >/dev/null 2>&1; then
        # Not fatal: the builder repeats this check inside the pod anyway. Only
        # the "return immediately" optimisation is lost.
        echo "⚠️  gcloud not found; skipping the registry pre-check (the build pod will check instead)."
    else
        echo "🔍 Checking whether ${TARGET_IMAGE} already exists in the registry..."
        if gcloud artifacts docker images describe "$TARGET_IMAGE" >/dev/null 2>&1; then
            IMAGE_PRESENT="yes"
        fi
    fi

    if [[ -n "$IMAGE_PRESENT" ]]; then
        echo "✅ Image already published; skipping the build hook entirely."
        HELM_ARGS+=(--set builder.enabled=false)
    else
        echo "🏗️  Image not found. It will be built by the pre-install hook on a CPU node."
        # An empty CACHE_CLAIM means builder.cache.enabled=false, i.e. the hook
        # uses an emptyDir and there is nothing to provision.
        if [[ -n "$CACHE_CLAIM" ]]; then
            if kubectl get pvc "$CACHE_CLAIM" >/dev/null 2>&1; then
                # The claim is shared by every release, and growing a PVC cannot
                # be undone, so a size change in the manifest is reported rather
                # than applied.
                WANT_SIZE="$(awk '/^[[:space:]]*storage:/ { print $2; exit }' \
                    "${CHART_DIR}/extras/build-cache-pvc.yaml")"
                HAVE_SIZE="$(kubectl get pvc "$CACHE_CLAIM" \
                    -o jsonpath='{.spec.resources.requests.storage}' 2>/dev/null || true)"
                if [[ -n "$WANT_SIZE" && -n "$HAVE_SIZE" ]] &&
                   (( $(to_gib "$HAVE_SIZE") < $(to_gib "$WANT_SIZE") )); then
                    echo "⚠️  Build cache '${CACHE_CLAIM}' is ${HAVE_SIZE}, but extras/build-cache-pvc.yaml now asks for ${WANT_SIZE}."
                    echo "    It was NOT resized automatically. To apply it:"
                    echo "      kubectl patch pvc ${CACHE_CLAIM} -p '{\"spec\":{\"resources\":{\"requests\":{\"storage\":\"${WANT_SIZE}\"}}}}'"
                else
                    echo "✅ Build cache '${CACHE_CLAIM}' already exists (${HAVE_SIZE:-size unknown}, left untouched)."
                fi
            else
                echo "📦 Creating build cache '${CACHE_CLAIM}' from extras/build-cache-pvc.yaml..."
                kubectl create -f "${CHART_DIR}/extras/build-cache-pvc.yaml"
            fi
        fi
        HELM_ARGS+=(--timeout "$BUILD_TIMEOUT")
        STREAM_BUILD="yes"
    fi
fi


# Follow the pre-install hook's pod while `helm install` blocks on it. The pod can
# be replaced (Job retry, node eviction) and can be absent for a while (Kueue
# admission), so this re-attaches rather than assuming a single stable pod.
stream_build_logs() {
    local helm_pid="$1"
    local pod="" phase="" streamed="" waiting=""

    mkdir -p "$LOG_DIR"
    : > "$BUILD_LOG"
    echo "📜 Streaming build log to ${BUILD_LOG}"

    while kill -0 "$helm_pid" 2>/dev/null; do
        # Newest pod that is not being deleted. A jsonpath filter cannot express
        # "field is absent", hence the go-template.
        pod="$(kubectl get pods -l job-name="$BUILDER_JOB" \
            --sort-by=.metadata.creationTimestamp \
            -o go-template='{{range .items}}{{if not .metadata.deletionTimestamp}}{{.metadata.name}}{{"\n"}}{{end}}{{end}}' \
            2>/dev/null | tail -n 1)"

        if [[ -z "$pod" ]]; then
            if [[ -z "$waiting" ]]; then
                echo "⏳ Waiting for the image-builder pod (Kueue admission)..."
                waiting="yes"
            fi
            sleep 3
            continue
        fi

        # Already followed this pod to completion; only a replacement is interesting.
        if [[ "$pod" == "$streamed" ]]; then
            sleep 3
            continue
        fi

        phase="$(kubectl get pod "$pod" -o jsonpath='{.status.phase}' 2>/dev/null || true)"
        if [[ -z "$phase" || "$phase" == "Pending" ]]; then
            sleep 3
            continue
        fi

        if [[ -n "$streamed" ]]; then
            {
                echo ""
                echo "===== build pod replaced: ${streamed} -> ${pod} (retry or eviction) ====="
                echo ""
            } | tee -a "$BUILD_LOG"
        fi
        waiting=""

        kubectl logs -f "$pod" 2>/dev/null | tee -a "$BUILD_LOG" || true
        streamed="$pod"
    done
}


# --- Install the chart ---
echo "🚀 Deploying Helm release '${RELEASE_NAME}' using values file '${VALUES_FILE}' (hfTokenSecret: '${HF_TOKEN_SECRET}')..."

HELM_OUT="$(mktemp)"
trap 'rm -f "$HELM_OUT"' EXIT
HELM_RC=0

if [[ -n "$STREAM_BUILD" ]]; then
    echo "   (helm will block until the image build finishes; timeout ${BUILD_TIMEOUT})"
    helm install "$RELEASE_NAME" . -f "$VALUES_FILE" "${HELM_ARGS[@]}" >"$HELM_OUT" 2>&1 &
    HELM_PID=$!
    trap 'kill "$HELM_PID" 2>/dev/null || true
          echo ""
          echo "⚠️  Interrupted. The build Job may still be running:"
          echo "    kubectl get job ${BUILDER_JOB}"
          echo "    helm uninstall ${RELEASE_NAME}   # to clean up the failed release"
          exit 130' INT
    stream_build_logs "$HELM_PID"
    wait "$HELM_PID" || HELM_RC=$?
    trap 'rm -f "$HELM_OUT"' INT
    cat "$HELM_OUT"
else
    helm install "$RELEASE_NAME" . -f "$VALUES_FILE" "${HELM_ARGS[@]}" 2>&1 | tee "$HELM_OUT" || HELM_RC=$?
fi

if (( HELM_RC != 0 )); then
    echo "" >&2
    echo "❌ helm install failed (exit ${HELM_RC})." >&2
    if grep -q 'failed pre-install' "$HELM_OUT" 2>/dev/null; then
        echo "   The image build failed. The JobSet was never created, so no TPU capacity was used." >&2
        [[ -f "$BUILD_LOG" ]] && echo "   Build log:  ${BUILD_LOG}" >&2
        echo "   Inspect:    kubectl describe job ${BUILDER_JOB}" >&2
        echo "   The failed Job is kept on purpose and is replaced on the next install." >&2
    elif grep -q 'context deadline exceeded' "$HELM_OUT" 2>/dev/null; then
        echo "   Timed out after ${BUILD_TIMEOUT}." >&2
        echo "   ⚠️  Helm does NOT stop the hook Job on timeout: the build is probably still running." >&2
        echo "   Check:      kubectl get job ${BUILDER_JOB}" >&2
        echo "   Raise it:   BUILD_TIMEOUT=3h $0 -f ${VALUES_FILE} -r ${RELEASE_NAME}" >&2
    fi
    echo "" >&2
    echo "   The release is left in 'failed' state; remove it before retrying:" >&2
    echo "     helm uninstall ${RELEASE_NAME}" >&2
    exit "$HELM_RC"
fi


# --- Print post-install monitoring and cleanup commands ---
# Steps are numbered by a counter so the list stays contiguous no matter which
# branches below are taken.
STEP=0
step() {
    local title="$1"
    shift
    STEP=$((STEP + 1))
    if (( STEP > 1 )); then
        echo ""
    fi
    echo " ${STEP}. ${title}"
    local line
    for line in "$@"; do
        echo "    ${line}"
    done
}


echo ""
echo "============================================================"
echo " 📋 Useful Commands to Monitor the Testcase Run:"
echo "============================================================"

step "Check JobSet status:" \
    "kubectl get jobset ${JOBSET_NAME}"

step "Watch pods:" \
    "kubectl get pods -l jobset.sigs.k8s.io/jobset-name=${JOBSET_NAME} -w"

if [[ "$BUILDER_ENABLED" == "true" ]]; then
    # By the time this prints, the build has already finished: it ran as a
    # pre-install hook Job, which Helm waits for before creating the JobSet.
    # The Job is intentionally not deleted, so its logs remain available.
    if [[ -n "$STREAM_BUILD" ]]; then
        step "Review the image build that ran before this JobSet:" \
            "less ${BUILD_LOG}" \
            "# Or straight from the (retained) hook Job:" \
            "kubectl logs job/${BUILDER_JOB}"
    else
        step "Image build was skipped (already published in the registry):" \
            "# Force a rebuild by deleting the tag, or point the values file at a new commit."
    fi
fi

step "Stream & Tee every step into log/${JOBSET_NAME}-<step>.log:" \
    "${SCRIPT_DIR}/tee_testcase_logs.sh ${JOBSET_NAME}" \
    "# Or follow every container (build + setup + test) side by side:" \
    "${SCRIPT_DIR}/tee_testcase_logs.sh -c all ${JOBSET_NAME}" \
    "# Or follow a single step only:" \
    "${SCRIPT_DIR}/tee_testcase_logs.sh -r <step> ${JOBSET_NAME}"

step "Stream testcase runner logs:" \
    "kubectl logs -l jobset.sigs.k8s.io/jobset-name=${JOBSET_NAME},role=test-runner -f"


step "Teardown / Cleanup:" \
    "${SCRIPT_DIR}/cleanup.sh ${RELEASE_NAME}" \
    "# Or via helm directly:" \
    "helm uninstall ${RELEASE_NAME}"

echo "============================================================"
