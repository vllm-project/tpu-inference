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

# Serve the two wheels this build made on a TPU, as a user would after
# `pip install vllm-tpu==<version>`: install them into a fresh venv (everything
# else from pypi.org), start `vllm serve`, check that it answers and run
# `vllm bench serve` against it. check_wheels.sh only resolves the wheels; this
# is the step that imports and runs them - the "install the wheel, serve it and
# bench it" check otherwise done by hand before a release.
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
#   WHEELS_DIR        Directory with tpu_inference-*.whl and vllm_tpu-*.whl;
#                     skips the artifact download.
#   WHEELS_BUILD      Build to download the wheel artifacts from (default: this
#                     build), e.g. $BUILDKITE_TRIGGERED_FROM_BUILD_ID in a
#                     build that the release build triggers.
#   TEST_MODEL        Model to serve and bench (default:
#                     meta-llama/Llama-3.1-8B-Instruct).
#   MAX_MODEL_LEN     --max-model-len (default: 1024).
#   SERVER_TIMEOUT_S  How long to wait for /health (default: 1800); covers the
#                     model download and the first XLA compile.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/lib.sh"
# The serve checks, run below by the Python of the venv the wheels go into.
WHEEL_UTILS="${SCRIPT_DIR}/../scripts/wheel_utils.py"

TEST_MODEL="${TEST_MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-1024}"
SERVER_TIMEOUT_S="${SERVER_TIMEOUT_S:-1800}"

# pypi.org and the two wheels only, as for a user: no index that the agent or
# the image configures for uv.
unset UV_INDEX UV_INDEX_URL UV_DEFAULT_INDEX UV_EXTRA_INDEX_URL UV_FIND_LINKS \
  UV_NO_INDEX UV_OFFLINE

WORK="$(mktemp -d "${TMPDIR:-/tmp}/serve-wheels.XXXXXX")"
SERVER_PID=""
TAIL_PID=""

# vllm serve runs its engine, which holds the TPU, in child processes: killing
# only the server can leave them behind (what `fuser /dev/vfio/*` and `kill -9`
# clean up by hand). Started with setsid, the server leads its own process
# group, so signalling the group reaches them all. <grace> is how long it gets
# to shut down.
stop_server() {
  local grace="$1"
  [[ -n "${SERVER_PID}" ]] || return 0
  if kill -0 "${SERVER_PID}" 2>/dev/null; then
    echo "--- :octagonal_sign: Stopping the server"
    kill -TERM -- "-${SERVER_PID}" 2>/dev/null || kill -TERM "${SERVER_PID}" 2>/dev/null
    local deadline=$((SECONDS + grace))
    while kill -0 "${SERVER_PID}" 2>/dev/null && (( SECONDS < deadline )); do
      sleep 1
    done
  fi
  kill -KILL -- "-${SERVER_PID}" 2>/dev/null || kill -KILL "${SERVER_PID}" 2>/dev/null
  SERVER_PID=""
}

cleanup() {
  local rc=$?
  set +e
  # Expand the section the step failed in, before stop_server opens another.
  (( rc == 0 )) || echo "^^^ +++"
  # Being in its own process group, the server never sees the agent's signal
  # when the job is cancelled or times out, and the agent kills this script
  # after its grace period (10s by default), so stop it fast in that case.
  if (( rc > 128 )); then stop_server 5; else stop_server 60; fi
  [[ -n "${TAIL_PID}" ]] && kill "${TAIL_PID}" 2>/dev/null
  rm -rf "${WORK}"
}
trap cleanup EXIT
# Stop on signals instead of continuing; the EXIT trap then cleans up once.
trap 'exit 129' HUP
trap 'exit 130' INT
trap 'exit 143' TERM

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

# The version is the second -separated part of a wheel's file name. The build
# scripts already checked it is the release version; both must agree.
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
# From here on, run from ${WORK}, not the checkout: vLLM starts some helpers
# with `python -m`, which puts the working directory first on sys.path, so the
# checkout's tpu_inference/ would shadow the installed wheel.
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

# --- Serve ------------------------------------------------------------------
echo "--- :test_tube: Serving ${TEST_MODEL}"
# A fresh model download and compilation cache, as on a user's first run, and
# none of the agent's or a CI fleet's caches. Only the hub cache moves, so a
# token saved by `huggingface-cli login` still applies.
export HF_HUB_CACHE="${WORK}/hf-cache"
export JAX_COMPILATION_CACHE_DIR="${WORK}/xla-cache"
export VLLM_XLA_CACHE_PATH="${JAX_COMPILATION_CACHE_DIR}"
PORT="$("${VENV_BIN}/python" "${WHEEL_UTILS}" free_port)"
BASE_URL="http://127.0.0.1:${PORT}"
SERVER_LOG="${WORK}/server.log"
SERVE_CMD=("${VENV_BIN}/vllm" serve "${TEST_MODEL}"
  --host 127.0.0.1 --port "${PORT}"
  --tensor-parallel-size 1
  --max-model-len "${MAX_MODEL_LEN}")
echo "${SERVE_CMD[*]}"
# Created up front: the server's redirect happens in the forked child, which
# could lose the race with tail below and leave it nothing to open.
: >"${SERVER_LOG}"
SERVE_START=${SECONDS}
if command -v setsid >/dev/null 2>&1; then
  setsid "${SERVE_CMD[@]}" >"${SERVER_LOG}" 2>&1 &
else
  "${SERVE_CMD[@]}" >"${SERVER_LOG}" 2>&1 &
fi
SERVER_PID=$!
# Stream the server's log into the job's until the end.
tail -n +1 -f "${SERVER_LOG}" &
TAIL_PID=$!

DEADLINE=$((SECONDS + SERVER_TIMEOUT_S))
until curl -sf -o /dev/null --max-time 5 "${BASE_URL}/health"; do
  kill -0 "${SERVER_PID}" 2>/dev/null \
    || die "vllm serve exited before it became healthy; its log is above."
  (( SECONDS < DEADLINE )) \
    || die "vllm serve was not healthy within ${SERVER_TIMEOUT_S}s."
  sleep 5
done
echo "--- :white_check_mark: Healthy after $((SECONDS - SERVE_START))s"

# --- Ask it -----------------------------------------------------------------
echo "--- :speech_balloon: Asking ${TEST_MODEL}"
"${VENV_BIN}/python" "${WHEEL_UTILS}" verify_server_answers \
  "${BASE_URL}" "${TEST_MODEL}" "${WORK}/answers.txt"

# --- Bench it ---------------------------------------------------------------
echo "--- :stopwatch: vllm bench serve"
BENCH_ARGS=(--model "${TEST_MODEL}" --host 127.0.0.1 --port "${PORT}"
  --backend vllm --request-rate inf
  --dataset-name random --num-prompts 30
  --random-input-len 128 --random-output-len 128
  --percentile-metrics "ttft,tpot,itl,e2el"
  --ignore-eos --skip-chat-template
  --save-result --result-dir "${WORK}" --result-filename bench.json)
echo "vllm bench serve ${BENCH_ARGS[*]}"
"${VENV_BIN}/vllm" bench serve "${BENCH_ARGS[@]}"
# vllm bench serve exits 0 even when requests fail, so check its result.
"${VENV_BIN}/python" "${WHEEL_UTILS}" verify_bench_result \
  "${WORK}/bench.json" "${WORK}/bench.txt"

echo "+++ :white_check_mark: vllm-tpu ${VERSION} serves ${TEST_MODEL} on a TPU"
if command -v buildkite-agent >/dev/null 2>&1 && [[ -n "${BUILDKITE_JOB_ID:-}" ]]; then
  buildkite-agent annotate --style success --context release-serve <<EOF || echo "WARNING: could not annotate the build." >&2
Served and benched \`${TEST_MODEL}\` on a TPU from the built wheels in a fresh venv (\`vllm-tpu==${VERSION}\`, \`tpu-inference==${VERSION}\`, the rest from pypi.org):

\`\`\`
$(cat "${WORK}/answers.txt" "${WORK}/bench.txt")
\`\`\`
EOF
fi
