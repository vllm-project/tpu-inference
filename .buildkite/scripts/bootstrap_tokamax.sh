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

# Bootstrap for the scheduled "tpu-tokamax-integration" pipeline.
#
# Resolves the newest tokamax nightly on PyPI, runs the JAX test suites against
# it, and - only if every test passes - pushes the requirements.txt bump to main
# (see integration_tokamax_promote.yml). No human in the loop.
#
# The pipeline deliberately pins vLLM to vllm_lkg.version rather than tracking
# HEAD: tokamax must be the only thing that changed, so that a red build means
# "this tokamax nightly broke us" and nothing else.
#
# Configure the Buildkite pipeline's "Steps" to run this script.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/configs/pipeline_config.sh"

REQUIREMENTS_FILE="requirements.txt"
JOB_PRIORITY="${PRIORITY_INTEGRATION}"
export JOB_PRIORITY
buildkite-agent meta-data set "JOB_PRIORITY" "${JOB_PRIORITY}"

# Handles the environment state for different TPU generations.
set_jax_envs() {
    case $1 in
        v6)
            export TESTS_GROUP_LABEL="[jax] TPU6e Tests Group"
            export TPU_VERSION="tpu6e"
            export TPU_QUEUE_SINGLE="tpu_v6e_queue"
            export TPU_QUEUE_MULTI="tpu_v6e_8_queue"
            export TENSOR_PARALLEL_SIZE_SINGLE=1
            export TENSOR_PARALLEL_SIZE_MULTI=8
            ;;
        v7)
            export TESTS_GROUP_LABEL="[jax] TPU7x Tests Group"
            export TPU_VERSION="tpu7x"
            export TPU_QUEUE_SINGLE="tpu_v7x_2_queue"
            export TPU_QUEUE_MULTI="tpu_v7x_8_queue"
            export TENSOR_PARALLEL_SIZE_SINGLE=2
            export TENSOR_PARALLEL_SIZE_MULTI=8
            export COV_FAIL_UNDER="67"
            ;;
        unset)
            unset TESTS_GROUP_LABEL TPU_VERSION TPU_QUEUE_SINGLE TPU_QUEUE_MULTI \
                  TENSOR_PARALLEL_SIZE_SINGLE TENSOR_PARALLEL_SIZE_MULTI COV_FAIL_UNDER
            ;;
    esac
}

# -----------------------------------------------------------------------------
# 1. Resolve the candidate version and short-circuit if there is nothing to do.
# -----------------------------------------------------------------------------
CURRENT_VERSION="$(sed -n 's/^tokamax==\(.*\)$/\1/p' "${REQUIREMENTS_FILE}")"
if [[ -z "${CURRENT_VERSION}" ]]; then
    echo "ERROR: no 'tokamax==' pin found in ${REQUIREMENTS_FILE}." >&2
    exit 1
fi

# TOKAMAX_VERSION can be set in the build env to re-test / force a specific
# version from the Buildkite UI.
NEW_VERSION="${TOKAMAX_VERSION:-}"
if [[ -z "${NEW_VERSION}" ]]; then
    echo "--- :package: Resolving the newest tokamax nightly from PyPI"
    NEW_VERSION="$(python3 - <<'PYEOF'
import json
import re
import sys
import urllib.request

with urllib.request.urlopen("https://pypi.org/pypi/tokamax/json", timeout=60) as resp:
    data = json.load(resp)

# Nightlies are published as <base>.devYYYYMMDD, e.g. 0.0.14.dev20260903.
# info.version only tracks the newest *stable* release, so the nightly has to be
# picked out of the full release list. Releases whose files were all removed or
# yanked are skipped - pip cannot install those.
pattern = re.compile(r"^(\d+(?:\.\d+)*)\.dev(\d{8})$")
candidates = []
for version, files in data["releases"].items():
    match = pattern.match(version)
    if not match:
        continue
    if not any(not f.get("yanked", False) for f in files):
        continue
    base = tuple(int(part) for part in match.group(1).split("."))
    candidates.append(((base, int(match.group(2))), version))

if not candidates:
    print("No installable tokamax nightly (*.devYYYYMMDD) found on PyPI.", file=sys.stderr)
    sys.exit(1)

print(max(candidates)[1])
PYEOF
)"
fi

echo "Pinned tokamax version   : ${CURRENT_VERSION}"
echo "Candidate tokamax version: ${NEW_VERSION}"

if [[ "${CURRENT_VERSION}" == "${NEW_VERSION}" ]]; then
    # tokamax does not publish a nightly every single day. Uploading no test
    # steps here is what keeps this pipeline from burning a full v6e+v7x TPU run
    # to re-validate a version main is already on.
    echo "Already on ${NEW_VERSION}. Nothing to bump; skipping the test run."
    buildkite-agent annotate \
        ":white_check_mark: tokamax already pinned to \`${NEW_VERSION}\` - no bump needed." \
        --style "success"
    exit 0
fi

buildkite-agent meta-data set "TOKAMAX_VERSION" "${NEW_VERSION}"
buildkite-agent meta-data set "TOKAMAX_PREVIOUS_VERSION" "${CURRENT_VERSION}"
buildkite-agent annotate \
    ":arrow_up: Validating tokamax bump \`${CURRENT_VERSION}\` :arrow_right: \`${NEW_VERSION}\`. main is bumped only if every step below passes." \
    --style "info"

# -----------------------------------------------------------------------------
# 2. Pin vLLM to the LKG so tokamax is the only variable in this build.
# -----------------------------------------------------------------------------
VLLM_COMMIT_HASH="$(get_vllm_commit_hash)"
buildkite-agent meta-data set "VLLM_COMMIT_HASH" "${VLLM_COMMIT_HASH}"
echo "Using vllm LKG commit hash: ${VLLM_COMMIT_HASH}"

# -----------------------------------------------------------------------------
# 3. Upload the pipelines.
# -----------------------------------------------------------------------------
# Deliberately NOT setting RUN_KERNEL_TESTS / RUN_KERNEL_COLLECTIVES_TESTS.
# tests/kernels/gmm_test.py imports the vendored copy at
# tpu_inference/kernels/megablox/gmm_v2.py, not tokamax, so those suites pass
# green regardless of which tokamax is installed. The coverage that matters here
# is _test_7_1 / _test_7_2 (ungated) plus the tpu7x-gated end-to-end GMM steps
# (_test_18, _test_30, _test_32, _test_24), none of which need extra flags.

# Buildkite inserts uploaded steps in reverse order, so the promote has to be
# uploaded first for it to end up last.
upload_with_priority .buildkite/integration_tokamax_promote.yml "${JOB_PRIORITY}"

set_jax_envs v7
upload_with_priority .buildkite/pipeline_jax.yml "${JOB_PRIORITY}"
set_jax_envs unset

set_jax_envs v6
upload_with_priority .buildkite/pipeline_jax.yml "${JOB_PRIORITY}"
set_jax_envs unset

# Uploaded last so the Docker build steps appear at the top of the Buildkite UI.
upload_with_priority .buildkite/pipeline_build.yml "${JOB_PRIORITY}"

echo "--- Tokamax Integration Bootstrap Finished"
