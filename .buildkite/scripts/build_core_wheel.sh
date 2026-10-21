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
# Version:
#   - Official release (default): the release branch name without
#     "releases/v" (e.g. releases/vX.Y.Z -> X.Y.Z). Must be exactly
#     releases/vX.Y.Z (RELEASE_BRANCH_REGEX in .buildkite/release/lib.sh).
#   - Pre-release (e.g. rc): set RELEASE_VERSION when starting the build
#     (Buildkite "New Build" -> Environment Variables), e.g.
#     RELEASE_VERSION=v0.29.0rc1. Used as-is; a leading "v" is dropped.
# Either way it must be a valid PEP 440 version that PyPI accepts (no local
# "+..." part), or the script exits 1. The version is forced onto
# setuptools-scm, so a shallow CI checkout without tags still gets it.
#
# Output:
#   dist/tpu_inference-<version>-py3-none-any.whl, uploaded as a Buildkite artifact.
#   Build meta-data CORE_RELEASE_VERSION / CORE_WHEEL_NAME for later steps.
# ------------------------------------------------------

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON="${PYTHON:-python3}"
WHEEL_UTILS="${SCRIPT_DIR}/wheel_utils.py"
# Provides resolve_release_version and die.
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/../release/lib.sh"

# --- Resolve the release version ---------------------------------------------
echo "--- :label: Resolving release version"
VERSION="$(resolve_release_version)"
VERSION_SOURCE="${RELEASE_VERSION:+RELEASE_VERSION=${RELEASE_VERSION}}"
VERSION_SOURCE="${VERSION_SOURCE:-branch ${BUILDKITE_BRANCH:-}}"

# --- Private build venv ------------------------------------------------------
WORK_DIR="$(mktemp -d "${TMPDIR:-/tmp}/core-build.XXXXXX")"
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
"$PY" -m pip install --quiet build twine packaging \
    "setuptools>=61.0.0" wheel "setuptools_scm[toml]>=8.0"

# Exit 1 unless the version is valid PEP 440 and accepted by PyPI (e.g. a
# mistyped branch like releases/v0.29.x or releases/v0.29.0+abc); prints it normalized.
VERSION="$("$PY" "$WHEEL_UTILS" validate_version "$VERSION")" \
    || die "version from ${VERSION_SOURCE} is not publishable to PyPI."
echo "Release version: ${VERSION}"

# Stop before building if this version is already on PyPI (versions can't be
# reused). PUBLISH_PYPI comes from the step's env in .buildkite/release/pipeline.yml.
if [[ -n "${PUBLISH_PYPI:-}" ]]; then
    INDEX_BASE="${PYPI_BASE_URL:-https://pypi.org}"
    "$PY" "$WHEEL_UTILS" check_not_published "${INDEX_BASE%/}" "$PUBLISH_PYPI" "$VERSION"
fi

# --- Build -------------------------------------------------------------------
echo "--- :package: Building wheel"
cd "${REPO_ROOT}"
rm -rf dist
SETUPTOOLS_SCM_PRETEND_VERSION="${VERSION}" "$PY" -m build --wheel --no-isolation

shopt -s nullglob
WHEELS=(dist/tpu_inference-*.whl)
shopt -u nullglob
[[ "${#WHEELS[@]}" -eq 1 ]] \
    || die "expected exactly one tpu_inference wheel in dist/, found ${#WHEELS[@]}."
WHEEL="${WHEELS[0]}"

# The file name must carry exactly the release version; anything else means
# the version override did not take effect.
"$PY" "$WHEEL_UTILS" verify_built_wheel "$WHEEL" "$VERSION"

echo "--- :mag: Checking metadata"
"$PY" -m twine check --strict "$WHEEL"

# --- Hand off to later steps -------------------------------------------------
if [[ -n "${BUILDKITE:-}" ]] && command -v buildkite-agent >/dev/null 2>&1; then
    buildkite-agent artifact upload "$WHEEL"
    buildkite-agent meta-data set "CORE_RELEASE_VERSION" "$VERSION"
    buildkite-agent meta-data set "CORE_WHEEL_NAME" "$(basename "$WHEEL")"
fi

echo "Built ${WHEEL}"
