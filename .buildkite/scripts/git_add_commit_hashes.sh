#!/bin/bash
set -e

VLLM_COMMIT_HASH=$(buildkite-agent meta-data get "VLLM_COMMIT_HASH" --default "")

if [ -z "${VLLM_COMMIT_HASH}" ]; then
    echo "VLLM_COMMIT_HASH not found in buildkite meta-data"
    exit 1
fi

if [ -z "${BUILDKITE_COMMIT:-}" ]; then
    echo "BUILDKITE_COMMIT not found"
    exit 1
fi

if [ ! -f commit_hashes.csv ]; then
    echo "timestamp,vllm_commit_hash,tpu_inference_commit_hash" > commit_hashes.csv
fi
echo "$(date '+%Y-%m-%d %H:%M:%S'),${VLLM_COMMIT_HASH},${BUILDKITE_COMMIT}" >> commit_hashes.csv

git add commit_hashes.csv
