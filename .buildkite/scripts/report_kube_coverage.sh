#!/usr/bin/env bash

set -euo pipefail

artifact_args=(".coverage.part*" .)
if [[ -n "${KUBE_COVERAGE_SOURCE_BUILD:-}" ]]; then
  artifact_args+=(
    --build "${KUBE_COVERAGE_SOURCE_BUILD}"
    --pipeline "${KUBE_COVERAGE_SOURCE_PIPELINE:-kube-dev}"
  )
fi

buildkite-agent artifact download "${artifact_args[@]}"

# The CPU builder currently runs Python 3.14, while the old pinned Pex
# bootstrap only supports Python <3.13. Reuse the commit-specific test image so
# this consumer has the exact Python and coverage.py versions that produced the
# files, without pulling from Docker Hub or PyPI.
docker run --rm \
  --user "$(id -u):$(id -g)" \
  -e HOME=/tmp \
  -v "$PWD:/workspace/tpu_inference" \
  -w /workspace/tpu_inference \
  "us-central1-docker.pkg.dev/cloud-ullm-inference-ci-cd/tpu-inference-ci/vllm-tpu:kube-${BUILDKITE_COMMIT}" \
  sh -euc '
    export COVERAGE_RCFILE=.buildkite/coverage_kube.rc
    python3 -m coverage combine .coverage.part1.tpu6e .coverage.part2.tpu6e
    python3 -m coverage report --fail-under=68
  '
