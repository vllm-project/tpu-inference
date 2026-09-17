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
    echo "Example: $0 -f values-llama8b-ci.yaml -r my-helm-test"
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
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VALUES_PATH="$SCRIPT_DIR/$VALUES_FILE"


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
MODE="$(awk -F: '/^[[:space:]]*mode[[:space:]]*:/ { gsub(/[[:space:]"'\'' ]/, "", $2); print $2; exit }' "$VALUES_PATH")"
MODE="${MODE:-aggregated}"


# Detect the on-demand image builder so the build-progress hint is only printed
# when the image-builder initContainer is actually part of the pod. Only the
# top-level "builder:" block is inspected, since "enabled:" also appears under
# other sections (features, script.git, ...).
BUILDER_ENABLED="$(awk -F: '
    /^[^[:space:]#]/ { in_builder = ($0 ~ /^builder[[:space:]]*:/) }
    in_builder && /^[[:space:]]+enabled[[:space:]]*:/ {
        sub(/#.*/, "", $2)
        gsub(/[[:space:]"'\'' ]/, "", $2)
        print $2
        exit
    }
' "$VALUES_PATH")"


# --- Install the chart ---
echo "🚀 Deploying Helm release '${RELEASE_NAME}' using values file '${VALUES_FILE}' (hfTokenSecret: '${HF_TOKEN_SECRET}')..."
cd "$SCRIPT_DIR"
helm install "$RELEASE_NAME" . -f "$VALUES_FILE" --set hfTokenSecret.name="$HF_TOKEN_SECRET"


# --- Print post-install monitoring and cleanup commands ---
# Steps are numbered by a counter so the list stays contiguous no matter which
# mode-specific branches below are taken.
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
echo " 📋 Useful Commands to Monitor Benchmark:"
echo "============================================================"

step "Check JobSet status:" \
    "kubectl get jobset ${JOBSET_NAME}"

step "Watch pods:" \
    "kubectl get pods -l jobset.sigs.k8s.io/jobset-name=${JOBSET_NAME} -w"

if [[ "$BUILDER_ENABLED" == "true" ]]; then
    step "Monitor on-demand image build progress (runs before the main container):" \
        "kubectl logs -l jobset.sigs.k8s.io/jobset-name=${JOBSET_NAME} -c image-builder --tail=100" \
        "# Append -f to follow the build live:" \
        "kubectl logs -l jobset.sigs.k8s.io/jobset-name=${JOBSET_NAME} -c image-builder -f"
fi

step "Stream & Tee every step into log/${JOBSET_NAME}-<step>.log:" \
    "${SCRIPT_DIR}/../bin/tee_testcase_logs.sh ${JOBSET_NAME}" \
    "# Or follow every container (build + setup + test) side by side:" \
    "${SCRIPT_DIR}/../bin/tee_testcase_logs.sh -c all ${JOBSET_NAME}" \
    "# Or follow a single step only:" \
    "${SCRIPT_DIR}/../bin/tee_testcase_logs.sh -r <step> ${JOBSET_NAME}"

if [[ "$MODE" == "script" ]]; then
    step "Stream testcase runner logs:" \
        "kubectl logs -l jobset.sigs.k8s.io/jobset-name=${JOBSET_NAME},role=test-runner -f"
elif [[ "$MODE" == "aggregated" ]]; then
    step "Stream server logs (vLLM / XLA compilation):" \
        "kubectl logs -l jobset.sigs.k8s.io/jobset-name=${JOBSET_NAME},jobset.sigs.k8s.io/replicatedjob-name=server -f"
else
    step "Stream Prefill logs:" \
        "kubectl logs -l jobset.sigs.k8s.io/jobset-name=${JOBSET_NAME},jobset.sigs.k8s.io/replicatedjob-name=p -c vllm-tpu -f"
    step "Stream Decode logs:" \
        "kubectl logs -l jobset.sigs.k8s.io/jobset-name=${JOBSET_NAME},jobset.sigs.k8s.io/replicatedjob-name=d -c vllm-tpu -f"
    step "Stream Proxy logs:" \
        "kubectl logs -l jobset.sigs.k8s.io/jobset-name=${JOBSET_NAME},jobset.sigs.k8s.io/replicatedjob-name=x -f"
fi


if [[ "$MODE" != "script" ]]; then
    step "Stream client benchmark results:" \
        "kubectl logs -l jobset.sigs.k8s.io/jobset-name=${JOBSET_NAME},jobset.sigs.k8s.io/replicatedjob-name=client -f"
fi


step "Teardown / Cleanup:" \
    "${SCRIPT_DIR}/../bin/cleanup.sh ${RELEASE_NAME}" \
    "# Or via helm directly:" \
    "helm uninstall ${RELEASE_NAME}"

echo "============================================================"
