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
# Build the tpu-inference release wheel.
#
# Version (resolved by release_version in .buildkite/release/lib.sh):
#   - Set once by preflight.sh in Buildkite meta-data (release-version), or
#     resolved on the spot from RELEASE_VERSION / BUILDKITE_BRANCH
#     (RELEASE_BRANCH_REGEX in .buildkite/release/lib.sh).
#   - setup.py reads it from VLLM_VERSION_OVERRIDE (it falls back to 0.0.0),
#     so it is forced there.
#
# Environment:
#   PUBLISH_PYPI   PyPI project checked for an existing release
#                  (default: tpu-inference).
#   PYPI_BASE_URL  Index to check (default: https://pypi.org).
#   PYTHON         Interpreter for the build venv (default: python3).
#
# Output:
#   dist/tpu_inference-<version>-py3-none-any.whl, uploaded as a Buildkite
#   artifact. Build meta-data TPU_INFERENCE_WHEEL_NAME for later steps.
#
# ------------------------------------------------------

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON="${PYTHON:-python3}"
WHEEL_UTILS="${SCRIPT_DIR}/wheel_utils.py"
PUBLISH_PYPI="${PUBLISH_PYPI:-tpu-inference}"
# Provides release_version, check_not_published and die.
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/../release/lib.sh"

command -v buildkite-agent >/dev/null 2>&1 \
    || die "buildkite-agent not found: this script only runs in Buildkite."

# --- Resolve the release version ---------------------------------------------
echo "--- :label: Resolving release version"
VERSION="$(release_version)"
echo "Release version: ${VERSION}"

# Stop before building if this version is already on PyPI (versions can't be
# reused).
check_not_published "${PUBLISH_PYPI}" "${VERSION}" "${PYPI_BASE_URL:-https://pypi.org}"

# --- Private build venv ------------------------------------------------------
WORK_DIR="$(mktemp -d "${TMPDIR:-/tmp}/tpu-inference-build.XXXXXX")"
trap 'rm -rf "${WORK_DIR}"' EXIT

# CI agents lack python3-venv (no ensurepip), so create the venv without pip
# and bootstrap pip with PyPA's get-pip.py.
"${PYTHON}" -m venv --without-pip "${WORK_DIR}/venv" \
    || die "could not create a venv with ${PYTHON}."
PY="${WORK_DIR}/venv/bin/python"
curl -fsSL --retry 3 -o "${WORK_DIR}/get-pip.py" \
    https://bootstrap.pypa.io/get-pip.py \
    || die "could not download get-pip.py from bootstrap.pypa.io."
"$PY" "${WORK_DIR}/get-pip.py" --quiet
# setup.py only needs setuptools (no [build-system] in pyproject.toml), so a
# --no-isolation build with these is enough.
"$PY" -m pip install --quiet build twine packaging "setuptools>=61.0.0" wheel

# --- Build -------------------------------------------------------------------
echo "--- :package: Building wheel"
# setup.py opens README.md and requirements.txt relative to the cwd.
cd "${REPO_ROOT}"
rm -rf dist build ./*.egg-info
VLLM_VERSION_OVERRIDE="${VERSION}" "$PY" -m build --wheel --no-isolation

shopt -s nullglob
WHEELS=(dist/*.whl)
shopt -u nullglob
[[ "${#WHEELS[@]}" -eq 1 ]] \
    || die "expected exactly one wheel in dist/, found ${#WHEELS[@]}."
WHEEL="${WHEELS[0]}"

# The file name must carry exactly the release version; anything else (e.g.
# 0.0.0) means VLLM_VERSION_OVERRIDE did not take effect.
"$PY" "$WHEEL_UTILS" verify_built_wheel "$WHEEL" "$VERSION"

echo "--- :mag: Checking metadata"
"$PY" -m twine check --strict "$WHEEL"

# --- Hand off to later steps -------------------------------------------------
buildkite-agent artifact upload "$WHEEL"
buildkite-agent meta-data set "TPU_INFERENCE_WHEEL_NAME" "$(basename "$WHEEL")"

echo "Built ${WHEEL}"
