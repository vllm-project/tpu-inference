#!/bin/sh
# Copyright 2025 Google LLC
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

set -e

V6_ANY_FAILED=$(buildkite-agent meta-data get "v6_CI_TESTS_FAILED" --default "false")
V7_ANY_FAILED=$(buildkite-agent meta-data get "v7_CI_TESTS_FAILED" --default "false")

# If previously marked as failed, re-run matrix generation to pick up any retried test results
if [ "${V6_ANY_FAILED}" = "true" ] && [ "$(buildkite-agent meta-data get run_v6_matrix --default 'false')" = "true" ]; then
  echo "--- Re-evaluating v6 test outcomes (picking up possible retries)"
  TPU_VERSION="v6e" bash .buildkite/scripts/generate_support_matrices.sh
  V6_ANY_FAILED=$(buildkite-agent meta-data get "v6_CI_TESTS_FAILED" --default "false")
fi

if [ "${V7_ANY_FAILED}" = "true" ] && [ "$(buildkite-agent meta-data get run_v7_matrix --default 'false')" = "true" ]; then
  echo "--- Re-evaluating v7 test outcomes (picking up possible retries)"
  TPU_VERSION="v7x" bash .buildkite/scripts/generate_support_matrices.sh
  V7_ANY_FAILED=$(buildkite-agent meta-data get "v7_CI_TESTS_FAILED" --default "false")
fi

FAILURE_LABEL="Not all models and/or features passed"

echo "--- Checking test outcomes"

if [ "${V6_ANY_FAILED}" = "true" ] || [ "${V7_ANY_FAILED}" = "true" ] ; then
  echo "${FAILURE_LABEL}"
  exit 1
else
  echo "All models & features passed."
fi
