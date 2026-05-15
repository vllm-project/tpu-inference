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

# Verify whether the torchax model-impl path (MODEL_IMPL_TYPE=vllm) supports
# a given speculative-decoding correctness test.
#
# Usage: sd_torchax_verify.sh <pytest-target>
# Example: sd_torchax_verify.sh tests/e2e/test_speculative_decoding.py::test_ngram_correctness_greedy
set -uo pipefail

if [ "$#" -lt 1 ]; then
  echo "[sd-verify] ERROR: usage: $0 <pytest-target>"
  exit 2
fi
PYTEST_TARGET="$1"

cd /workspace/tpu_inference || exit 1

export MODEL_IMPL_TYPE=vllm
export TPU_VERSION="${TPU_VERSION:-tpu7x}"
export VLLM_USE_V1=1

echo "[sd-verify] MODEL_IMPL_TYPE=${MODEL_IMPL_TYPE}"
echo "[sd-verify] TPU_VERSION=${TPU_VERSION}"
echo "[sd-verify] target: ${PYTEST_TARGET}"

python3 -m pytest -s -v "${PYTEST_TARGET}"
RC=$?

if [ $RC -eq 0 ]; then
  echo "[sd-verify] RESULT: PASS — torchax path supports: ${PYTEST_TARGET}"
else
  echo "[sd-verify] RESULT: FAIL (rc=$RC) — torchax path does NOT support or regressed: ${PYTEST_TARGET}"
fi

exit $RC
