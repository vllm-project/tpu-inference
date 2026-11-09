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
# Build the vllm-jax-core release wheel (shipping tpu_inference and the
# vllm_jax singleton compatibility alias) plus the transitional empty
# tpu-inference and vllm-tpu meta-wheels.
#
# Version (resolved by release_version in .buildkite/release/lib.sh):
#   - Set once by preflight.sh in Buildkite meta-data (release-version), or
#     resolved on the spot from RELEASE_VERSION / BUILDKITE_BRANCH
#     (RELEASE_BRANCH_REGEX in .buildkite/release/lib.sh).
#   - setup.py reads it from VLLM_VERSION_OVERRIDE (it falls back to 0.0.0),
#     so it is forced there.
#
# Environment:
#   PUBLISH_PYPI   PyPI project checked for an existing release on pypi.org
#                  (default: CORE_PROJECT in .buildkite/release/lib.sh).
#   PYTHON         Interpreter for the build venv (default: python3).
#
# Output:
#   dist/vllm_jax_core-<version>-py3-none-any.whl,
#   dist/tpu_inference-<version>-py3-none-any.whl, and
#   dist/vllm_tpu-<version>-py3-none-any.whl, uploaded as Buildkite artifacts.
#   Build meta-data VLLM_JAX_CORE_WHEEL_NAME for later steps.
#
# ------------------------------------------------------

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON="${PYTHON:-python3}"
WHEEL_UTILS="${SCRIPT_DIR}/wheel_utils.py"
# Provides CORE_PROJECT, release_version, check_not_published and die.
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/../release/lib.sh"
PUBLISH_PYPI="${PUBLISH_PYPI:-${CORE_PROJECT}}"

command -v buildkite-agent >/dev/null 2>&1 \
    || die "buildkite-agent not found: this script only runs in Buildkite."

# --- Resolve the release version ---------------------------------------------
echo "--- :label: Resolving release version"
VERSION="$(release_version)"
echo "Release version: ${VERSION}"

# Stop before building if this version is already on PyPI (versions can't be
# reused).
check_not_published "${PUBLISH_PYPI}" "${VERSION}"
if [[ "${PUBLISH_PYPI}" == "${CORE_PROJECT}" ]]; then
    check_not_published "${LEGACY_CORE_PROJECT}" "${VERSION}"
    check_not_published "${LEGACY_COMBINED_PROJECT}" "${VERSION}"
fi

# --- Private build venv ------------------------------------------------------
WORK_DIR="$(mktemp -d "${TMPDIR:-/tmp}/vllm-jax-core-build.XXXXXX")"
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
echo "--- :package: Building ${CORE_PROJECT} wheel"
# setup.py opens README.md and requirements.txt relative to the cwd.
cd "${REPO_ROOT}"
rm -rf dist build ./*.egg-info packaging/*/*.egg-info packaging/*/build
VLLM_VERSION_OVERRIDE="${VERSION}" "$PY" -m build --wheel --no-isolation

shopt -s nullglob
CORE_WHEELS=(dist/vllm_jax_core-*.whl)
shopt -u nullglob
[[ "${#CORE_WHEELS[@]}" -eq 1 ]] \
    || die "expected exactly one vllm_jax_core wheel in dist/, found ${#CORE_WHEELS[@]}."
WHEEL="${CORE_WHEELS[0]}"

# The file name must carry exactly the release version; anything else (e.g.
# 0.0.0) means VLLM_VERSION_OVERRIDE did not take effect.
"$PY" "$WHEEL_UTILS" verify_built_wheel "$WHEEL" "$VERSION"
# Ensure both the canonical package and compatibility alias are shipped.
"$PY" - "$WHEEL" <<'EOF'
import sys, zipfile
with zipfile.ZipFile(sys.argv[1]) as zf:
    names = set(zf.namelist())
for required in ("tpu_inference/__init__.py", "vllm_jax/__init__.py"):
    if required not in names:
        sys.exit(f"ERROR: {required} missing from {sys.argv[1]}")
EOF

echo "--- :package: Building transitional meta-wheels (${LEGACY_CORE_PROJECT}, ${LEGACY_COMBINED_PROJECT})"
VLLM_VERSION_OVERRIDE="${VERSION}" "$PY" -m build --wheel --no-isolation \
    --outdir "${REPO_ROOT}/dist" "${REPO_ROOT}/packaging/tpu_inference_meta"
VLLM_VERSION_OVERRIDE="${VERSION}" "$PY" -m build --wheel --no-isolation \
    --outdir "${REPO_ROOT}/dist" "${REPO_ROOT}/packaging/vllm_tpu_meta"

shopt -s nullglob
LEGACY_CORE_WHEELS=(dist/tpu_inference-*.whl)
LEGACY_COMBINED_WHEELS=(dist/vllm_tpu-*.whl)
shopt -u nullglob
[[ "${#LEGACY_CORE_WHEELS[@]}" -eq 1 ]] \
    || die "expected one tpu_inference meta-wheel in dist/, found ${#LEGACY_CORE_WHEELS[@]}."
[[ "${#LEGACY_COMBINED_WHEELS[@]}" -eq 1 ]] \
    || die "expected one vllm_tpu meta-wheel in dist/, found ${#LEGACY_COMBINED_WHEELS[@]}."
LEGACY_CORE_WHEEL="${LEGACY_CORE_WHEELS[0]}"
LEGACY_COMBINED_WHEEL="${LEGACY_COMBINED_WHEELS[0]}"

"$PY" "$WHEEL_UTILS" verify_empty_meta_wheel \
    "$LEGACY_CORE_WHEEL" "$VERSION" "${CORE_PROJECT}==${VERSION}"
"$PY" "$WHEEL_UTILS" verify_empty_meta_wheel \
    "$LEGACY_COMBINED_WHEEL" "$VERSION" "${COMBINED_PROJECT}==${VERSION}"

echo "--- :mag: Checking metadata"
"$PY" -m twine check --strict "$WHEEL" "$LEGACY_CORE_WHEEL" "$LEGACY_COMBINED_WHEEL"

# --- Hand off to later steps -------------------------------------------------
buildkite-agent artifact upload "$WHEEL"
buildkite-agent artifact upload "$LEGACY_CORE_WHEEL"
buildkite-agent artifact upload "$LEGACY_COMBINED_WHEEL"
buildkite-agent meta-data set "VLLM_JAX_CORE_WHEEL_NAME" "$(basename "$WHEEL")"

echo "Built ${WHEEL}, ${LEGACY_CORE_WHEEL}, ${LEGACY_COMBINED_WHEEL}"
