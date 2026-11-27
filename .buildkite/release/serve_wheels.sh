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

# Serve the two release wheels on a TPU the way a user would after
# `pip install vllm-tpu==<version>`: install them into a fresh venv (everything
# else from pypi.org), check the install, then serve and verify them with
# serve_and_verify.sh, the same flow the staged image is checked with.
# check_wheels.sh only resolves the wheels; this step imports and runs them,
# the "install, serve and bench" check otherwise done by hand before a release.
#
# Needs a TPU host (one v6e chip is enough for the default model) with curl
# and network access to pypi.org, astral.sh and huggingface.co; uv provides
# Python 3.12 if the host has none. The default model is gated: HF_TOKEN (or a
# `huggingface-cli login`) must give access to it. As a Buildkite step it
# downloads the build's wheel artifacts. By hand, on a TPU VM, point it at the
# wheels:
#
#   WHEELS_DIR=./dist bash .buildkite/release/serve_wheels.sh
#
# Environment:
#   WHEELS_DIR    Directory with tpu_inference-*.whl and vllm_tpu-*.whl; skips
#                 the artifact download. In a release, verify_wheels_on_tpu.sh
#                 sets it to the wheel carrier's /wheels.
#   WHEELS_BUILD  Build to download the wheel artifacts from (default: this
#                 build).
#   TEST_MODEL, MAX_MODEL_LEN, NUM_PROMPTS, SERVER_TIMEOUT_S
#                 Passed on to serve_and_verify.sh.
#   TMPDIR        Where the venv and the scratch go; the release check on kube
#                 sets /dev/shm.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/lib.sh"
# The install checks below run with the Python of the venv the wheels go into.
WHEEL_UTILS="${SCRIPT_DIR}/../scripts/wheel_utils.py"

# Only pypi.org and the two wheels, as for a user: ignore any uv index that
# the agent or the image configures.
unset UV_INDEX UV_INDEX_URL UV_DEFAULT_INDEX UV_EXTRA_INDEX_URL UV_FIND_LINKS \
  UV_NO_INDEX UV_OFFLINE

WORK="$(mktemp -d "${TMPDIR:-/tmp}/serve-wheels.XXXXXX")"
trap 'rm -rf "${WORK}"' EXIT
# Stop on signals instead of continuing; the EXIT trap then cleans up once.
trap 'exit 129' HUP
trap 'exit 130' INT
trap 'exit 143' TERM
# uv's downloads stay in the scratch too, not in ~/.cache on the boot disk.
export UV_CACHE_DIR="${WORK}/uv-cache"

# --- The wheels -------------------------------------------------------------
echo "--- :inbox_tray: The wheels"
mkdir -p "${WORK}/dist"
if [[ -n "${WHEELS_DIR:-}" ]]; then
  cp "${WHEELS_DIR}"/tpu_inference-*.whl "${WHEELS_DIR}"/vllm_tpu-*.whl "${WORK}/dist/" \
    || die "WHEELS_DIR=${WHEELS_DIR} must hold tpu_inference-*.whl and vllm_tpu-*.whl."
else
  command -v buildkite-agent >/dev/null 2>&1 \
    || die "buildkite-agent not found; outside Buildkite, set WHEELS_DIR."
  DOWNLOAD_ARGS=()
  if [[ -n "${WHEELS_BUILD:-}" ]]; then
    DOWNLOAD_ARGS=(--build "${WHEELS_BUILD}")
  fi
  (cd "${WORK}" && buildkite-agent artifact download "dist/*.whl" . "${DOWNLOAD_ARGS[@]}")
fi

shopt -s nullglob
CORE_WHEELS=("${WORK}"/dist/tpu_inference-*.whl)
COMBINED_WHEELS=("${WORK}"/dist/vllm_tpu-*.whl)
shopt -u nullglob
[[ "${#CORE_WHEELS[@]}" -eq 1 && "${#COMBINED_WHEELS[@]}" -eq 1 ]] \
  || die "expected one tpu_inference and one vllm_tpu wheel, found" \
    "${#CORE_WHEELS[@]} and ${#COMBINED_WHEELS[@]}."
CORE_WHEEL="${CORE_WHEELS[0]}"
COMBINED_WHEEL="${COMBINED_WHEELS[0]}"
ls -l "${WORK}/dist"

# The version is the second dash-separated field of a wheel's file name. The
# build scripts already checked it is the release version; both must agree.
VERSION="$(basename "${CORE_WHEEL}" | cut -d- -f2)"
[[ "$(basename "${COMBINED_WHEEL}" | cut -d- -f2)" == "${VERSION}" ]] \
  || die "the wheels disagree on the version: $(basename "${CORE_WHEEL}")," \
    "$(basename "${COMBINED_WHEEL}")."

# --- A fresh venv -----------------------------------------------------------
echo "--- :python: Installing the wheels into a fresh venv"
UV="$(install_uv "${WORK}/uv")"
"${UV}" venv --no-config --quiet --python 3.12 "${WORK}/venv"
VENV_BIN="${WORK}/venv/bin"
# Both wheel files, by path. vllm-tpu pins tpu-inference==<version> but does
# not bundle it, and before the publish no index has that version, so the
# local tpu_inference wheel is what meets the pin; a user's
# `pip install vllm-tpu` gets it from PyPI instead. By path, these exact bytes
# are what gets tested. Everything else comes from pypi.org.
"${UV}" pip install --no-config --python "${VENV_BIN}/python" \
  --index-url https://pypi.org/simple "${COMBINED_WHEEL}" "${CORE_WHEEL}"

# --- What got installed -----------------------------------------------------
echo "--- :mag: Checking the install"
# Run from ${WORK}, not the checkout: the checkout's tpu_inference/ would
# shadow the installed wheel.
cd "${WORK}"
"${VENV_BIN}/python" "${WHEEL_UTILS}" verify_installed_wheels \
  "${VERSION}" "${WORK}/venv"

# The CLI reads its own version from the vllm-tpu metadata, a lookup the build
# renames, so this checks the rename too. vLLM logs to stdout; the version is
# the last line.
CLI_VERSION="$("${VENV_BIN}/vllm" --version 2>/dev/null | tail -n 1)" \
  || die "\`vllm --version\` failed."
[[ "${CLI_VERSION}" == "${VERSION}" ]] \
  || die "\`vllm --version\` says '${CLI_VERSION}', not ${VERSION}."
echo "vllm --version: ${CLI_VERSION}"

# --- Serve and verify -------------------------------------------------------
# Run as a child process rather than exec, so this script's EXIT trap still
# removes the venv afterwards.
bash "${SCRIPT_DIR}/serve_and_verify.sh" "${VENV_BIN}/python" wheels \
  "the built wheels in a fresh venv (vllm-tpu==${VERSION}, tpu-inference==${VERSION}, the rest from pypi.org)"
