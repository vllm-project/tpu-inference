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
#
# ------------------------------------------------------
# Publish a release to PyPI with Trusted Publishing (no stored API token).
# Run tpu-inference first, then vllm-tpu (which pins that tpu-inference).
#
# Usage:
#   publish_wheel.sh tpu-inference   # tpu-inference (wheel only)
#   publish_wheel.sh vllm-tpu        # vllm-tpu, the combined wheel (+ sdist)
#
# What it does:
#   1. Download the files built earlier in this build and check their name
#      and version.
#   2. Log in to PyPI: get a Google identity token for this VM's service
#      account and trade it for a PyPI token (valid ~15 min, this project only).
#   3. Sign each file with the same Google identity, so PyPI shows its
#      provenance (PEP 740 attestation).
#   4. Upload the files and their attestations with twine.
#   5. Wait until the release is installable from PyPI.
#
# One-time PyPI setup, per project (tpu-inference, vllm-tpu):
#   Project -> Publishing -> Add a new publisher -> Google Cloud ->
#   email: <service-account>@<project>.iam.gserviceaccount.com
#   (use a "pending publisher" if the project doesn't exist yet).
#
# Environment:
#   PUBLISH_PYPI   PyPI project name; the files must match it.
#   PYTHON         Interpreter for the venv (default: python3).
#
# Publishes to pypi.org only. The index is not configurable: the Google
# identity token minted below is sent to it.
#
# CI only: must run on a GCE VM agent, after the matching build step.
# ------------------------------------------------------

set -euo pipefail
set +x      # Never trace commands, so tokens can't leak into logs.
umask 077   # Temp files are readable by this user only.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WHEEL_UTILS="${SCRIPT_DIR}/wheel_utils.py"
PYTHON="${PYTHON:-python3}"
PUBLISH_PYPI="${PUBLISH_PYPI:-}"
# Never read from the build's env: the identity token goes to INDEX_BASE.
INDEX_BASE="https://pypi.org"
UPLOAD_URL="https://upload.pypi.org/legacy/"
OIDC_AUDIENCE="pypi"   # what pypi.org/_/oidc/audience answers

# shellcheck source=/dev/null
source "${SCRIPT_DIR}/../release/lib.sh"

# --- Pick the release --------------------------------------------------------
# Hardcoded here (not derived from PUBLISH_PYPI) so the name check below
# compares two independent sources.
TARGET="${1:-}"
case "${TARGET}" in
    tpu-inference)
        WHEEL_GLOB="tpu_inference-*.whl"
        SDIST_GLOB=""   # build_tpu_inference_wheel.sh makes no sdist.
        PUBLISHED_KEY="TPU_INFERENCE_PUBLISHED_VERSION"
        ;;
    vllm-tpu)
        WHEEL_GLOB="vllm_tpu-*.whl"
        SDIST_GLOB="vllm_tpu-*.tar.gz"
        PUBLISHED_KEY="VLLM_TPU_PUBLISHED_VERSION"
        ;;
    *)
        die "usage: $0 <tpu-inference|vllm-tpu>"
        ;;
esac

command -v buildkite-agent >/dev/null 2>&1 \
    || die "buildkite-agent not found: this script only runs in Buildkite."
[[ -n "${PUBLISH_PYPI}" ]] || die "PyPI project (PUBLISH_PYPI) is not set."

# --- Download the built files ------------------------------------------------
# Publish exactly what was built and tested, never a rebuild.
echo "--- Fetching the files built in this build"
VERSION="$(release_version)"

WORK_DIR="$(mktemp -d "${TMPDIR:-/tmp}/${TARGET}-publish.XXXXXX")"
trap 'rm -rf "${WORK_DIR}"' EXIT
mkdir -p "${WORK_DIR}/dist"

# Both releases upload to dist/; the globs never match each other's files
# (`tpu_inference-` vs `vllm_tpu-`).
(cd "${WORK_DIR}" && buildkite-agent artifact download "dist/${WHEEL_GLOB}" .)
if [[ -n "${SDIST_GLOB}" ]]; then
    (cd "${WORK_DIR}" && buildkite-agent artifact download "dist/${SDIST_GLOB}" .)
fi

shopt -s nullglob
WHEELS=("${WORK_DIR}"/dist/*.whl)
SDISTS=("${WORK_DIR}"/dist/*.tar.gz)
shopt -u nullglob
[[ "${#WHEELS[@]}" -eq 1 ]] \
    || die "expected exactly one wheel artifact, found ${#WHEELS[@]}."
DISTS=("${WHEELS[0]}")
if [[ -n "${SDIST_GLOB}" ]]; then
    [[ "${#SDISTS[@]}" -eq 1 ]] \
        || die "expected exactly one sdist artifact, found ${#SDISTS[@]}."
    DISTS+=("${SDISTS[0]}")
fi

# --- Private venv ------------------------------------------------------------
echo "--- Setting up isolated virtualenv"
if command -v uv >/dev/null 2>&1; then
    uv venv "${WORK_DIR}/venv" --python "${PYTHON}"
    PY="${WORK_DIR}/venv/bin/python"
    uv pip install --quiet --python "${PY}" id twine packaging pypi-attestations pip
else
    # CI agents lack python3-venv, so create the venv without pip and
    # bootstrap pip with get-pip.py.
    "${PYTHON}" -m venv --without-pip "${WORK_DIR}/venv" \
        || die "could not create a venv with ${PYTHON}."
    PY="${WORK_DIR}/venv/bin/python"
    curl -fsSL --retry 3 -o "${WORK_DIR}/get-pip.py" \
        https://bootstrap.pypa.io/get-pip.py \
        || die "could not download get-pip.py from bootstrap.pypa.io."
    "$PY" "${WORK_DIR}/get-pip.py" --quiet
    "$PY" -m pip install --quiet id twine packaging pypi-attestations
fi

# Check every file has the right project name and the tested version.
for dist in "${DISTS[@]}"; do
    "$PY" "$WHEEL_UTILS" verify_publish_dist "$dist" "$PUBLISH_PYPI" "$VERSION"
done

# --- Log in: trade a Google identity for a PyPI token ------------------------
echo "--- Step 1: Checking the OIDC audience of ${INDEX_BASE}"
AUDIENCE="$("$PY" "$WHEEL_UTILS" fetch_oidc_audience "${INDEX_BASE}")"
[[ "${AUDIENCE}" == "${OIDC_AUDIENCE}" ]] \
    || die "${INDEX_BASE} reports OIDC audience '${AUDIENCE}', expected" \
        "'${OIDC_AUDIENCE}'."
echo "Target PyPI URL  : ${INDEX_BASE}"
echo "Expected Audience: ${OIDC_AUDIENCE}"

echo "--- Step 2: Minting GCP OIDC ID token"
OIDC_TOKEN="$("$PY" -m id "${OIDC_AUDIENCE}" 2>/dev/null || true)"
# `id` may succeed with empty output, so check it looks like a JWT (3 parts).
if [[ "$(printf '%s' "${OIDC_TOKEN}" | tr -cd '.' | wc -c | tr -d ' ')" != "2" ]]; then
    die "could not get an identity token. The publish step must run on a GCE" \
        "VM agent with a service account."
fi

# Fail early if the token isn't from Google or isn't for PyPI. Logs only
# non-identifying claims. Tokens are always passed via stdin, never argv or env.
printf '%s' "${OIDC_TOKEN}" | "$PY" "$WHEEL_UTILS" verify_gcp_oidc_token "${OIDC_AUDIENCE}"

echo "--- Step 3: Exchanging GCP OIDC token for short-lived PyPI API token"
# The PyPI token stays in memory only; it's never written to disk.
API_TOKEN="$(printf '%s' "${OIDC_TOKEN}" \
    | "$PY" "$WHEEL_UTILS" mint_pypi_token "${INDEX_BASE}" "${PUBLISH_PYPI}")"
unset OIDC_TOKEN

# Tell Buildkite to hide the token if anything ever prints it.
printf '%s' "${API_TOKEN}" | buildkite-agent redactor add >/dev/null 2>&1 || true
echo "Successfully minted short-lived PyPI API token (redacted)."

# --- Sign: provenance (PEP 740) ----------------------------------------------
# Writes <file>.publish.attestation, signed with the Google identity Step 2
# checked; PyPI shows it as the file's provenance. (Without that identity,
# pypi_attestations would wait for an interactive login instead of failing.)
echo "--- Step 4: Signing provenance attestations"
"$PY" -m pypi_attestations sign "${DISTS[@]}"

# --- Upload ------------------------------------------------------------------
echo "--- Step 5: Uploading ${DISTS[*]##*/} to ${UPLOAD_URL}"
# --skip-existing makes a retry safe: files already on PyPI are left alone.
# twine attaches only the attestation files it is given, so pass them too.
TWINE_USERNAME="__token__" TWINE_PASSWORD="${API_TOKEN}" \
    "$PY" -m twine upload --non-interactive --skip-existing --attestations \
    --repository-url "${UPLOAD_URL}" \
    "${DISTS[@]}" "${WORK_DIR}"/dist/*.publish.attestation
unset API_TOKEN

# --- Wait until PyPI serves it -----------------------------------------------
# New uploads can take a moment to appear. The next step (or users) needs the
# release to be installable, so poll for up to 5 minutes.
#   --python-version 3.12        look for a cp312 wheel, whatever Python this
#                                agent has.
#   --platform <the wheel's tag> manylinux_X_Y_x86_64 for vllm-tpu, any for
#                                tpu-inference. pip matches a manylinux_X_Y
#                                --platform exactly, not "this or older", so a
#                                fixed tag would miss a wheel tagged otherwise.
#   --only-binary :all:          check the wheel; never build the sdist.
WHEEL_PLATFORM="$(wheel_platform_tag "${WHEELS[0]}")"
echo "--- Waiting for ${PUBLISH_PYPI}==${VERSION} (${WHEEL_PLATFORM}) on PyPI"
for attempt in $(seq 1 20); do
    if "$PY" -m pip download --quiet --no-deps --no-cache-dir \
        --python-version 3.12 --platform "${WHEEL_PLATFORM}" \
        --only-binary :all: \
        --index-url "${INDEX_BASE}/simple/" --dest "${WORK_DIR}/verify" \
        "${PUBLISH_PYPI}==${VERSION}" >/dev/null 2>&1; then
        echo "Available after ${attempt} check(s)."
        buildkite-agent meta-data set "${PUBLISHED_KEY}" "${VERSION}"
        buildkite-agent annotate --style success --context "${TARGET}-publish" \
            "Published **${PUBLISH_PYPI} ${VERSION}** to PyPI."
        exit 0
    fi
    sleep 15
done
die "uploaded, but ${PUBLISH_PYPI}==${VERSION} is not installable from" \
    "${INDEX_BASE}/simple/ after 5 minutes. Check the project page before retrying."
