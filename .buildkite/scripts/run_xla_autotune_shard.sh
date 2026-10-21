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
# Run one shard of the XLA autotune sweep.
#
# Args:   <slice_index>        1-based; the shard this job runs.
# Shard count comes from AUTOTUNE_TOTAL_SHARDS (single source of truth, kept
# next to `matrix` in pipeline.yml); an optional 2nd positional arg overrides it.
# Env (all optional):  AUTOTUNE_{MODEL,TARGET_METRIC,BASELINE_RUNS,FLAGS,
#                                SKIP_CANDIDATES,CONFIG,DRY_RUN}    see README

set -euo pipefail

SLICE_INDEX="${1:?usage: $0 <slice_index> [slice_count]}"
SLICE_COUNT="${2:-${AUTOTUNE_TOTAL_SHARDS:-1}}"

MODEL="${AUTOTUNE_MODEL:-Qwen/Qwen3.5-397B-A17B-FP8}"
METRIC="${AUTOTUNE_TARGET_METRIC:-total_token_throughput}"
BASELINES="${AUTOTUNE_BASELINE_RUNS:-2}"
SKIP="${AUTOTUNE_SKIP_CANDIDATES:-0}"
FLAGS="${AUTOTUNE_FLAGS:-.buildkite/xla_autotune/flags.txt}"
CONFIG="${AUTOTUNE_CONFIG:-}"
DRY_RUN="${AUTOTUNE_DRY_RUN:-}"

# /tmp/kernel_tuning is bind-mounted host↔container by run_in_docker.sh.
# Per-build subdir keeps shards isolated from prior builds on the same agent
# (those files are owned by root and the host buildkite-agent can't unlink).
BUILD_ROOT="/tmp/kernel_tuning/xla_autotune/build_${BUILDKITE_BUILD_NUMBER:-local}"
SHARD_DIR="shard_${SLICE_INDEX}_of_${SLICE_COUNT}"
ARTIFACT_DIR="${BUILD_ROOT}/${SHARD_DIR}"
mkdir -p "${ARTIFACT_DIR}"

echo "[xla-autotune] shard ${SLICE_INDEX}/${SLICE_COUNT} → ${ARTIFACT_DIR}"

# Background watcher: ship per-trial JSON & summary.jsonl on mtime change;
# ship each log bundle once its sibling `<bundle>.done` marker appears.
(
  cd "${BUILD_ROOT}"
  declare -A LAST=() SENT=()
  while true; do
    shopt -s nullglob
    for f in "${SHARD_DIR}"/*.json "${SHARD_DIR}/summary.jsonl"; do
      [[ -f "$f" ]] || continue
      m=$(stat -c %Y "$f" 2>/dev/null || echo 0)
      if [[ "${LAST[$f]:-0}" != "$m" ]]; then
        buildkite-agent artifact upload "$f" && LAST["$f"]=$m
      fi
    done
    for marker in "${SHARD_DIR}"/logs/*.done; do
      [[ -f "$marker" ]] || continue
      d="${marker%.done}"
      [[ -d "$d" && -z "${SENT[$d]:-}" ]] || continue
      buildkite-agent artifact upload "${d}/**/*" && SENT["$d"]=1
    done
    sleep 20
  done
) &
WATCH_PID=$!
trap 'kill ${WATCH_PID} 2>/dev/null || true' EXIT INT TERM

EXTRA=()
[[ -n "${CONFIG}"  ]] && EXTRA+=(--benchmark-args-json "${CONFIG}")
[[ -n "${DRY_RUN}" ]] && EXTRA+=(--dry-run)

set +e
.buildkite/scripts/run_in_docker.sh bash -c "
  set -euo pipefail
  cd /workspace/tpu_inference
  # Shared benchmark_serving harness — same source as tests/e2e/benchmarking.
  # Pinned so the benchmark semantics can't silently change between builds and
  # invalidate the OFAT comparison; bump deliberately when the harness changes.
  BENCH_SERVING_SHA=ee867231de0b268e2810a6e31751b23cf5903fc5
  if [ ! -e bench_serving ]; then
    git clone https://github.com/kimbochen/bench_serving.git
  fi
  git -C bench_serving checkout --quiet \"\${BENCH_SERVING_SHA}\"
  echo \"bench_serving commit: \$(git -C bench_serving rev-parse HEAD)\"
  python3 .buildkite/xla_autotune/autotuner.py \
    --flag-list-file '${FLAGS}' \
    --model '${MODEL}' \
    --target-metric '${METRIC}' \
    --slice-index ${SLICE_INDEX} --slice-count ${SLICE_COUNT} \
    --baseline-runs ${BASELINES} \
    --skip-candidates ${SKIP} \
    --artifact-dir '${ARTIFACT_DIR}' \
    ${EXTRA[*]}
"
RC=$?
set -e

# Stop the watcher and sweep up anything written after its last tick.
kill "${WATCH_PID}" 2>/dev/null || true
wait "${WATCH_PID}" 2>/dev/null || true
if cd "${BUILD_ROOT}"; then
    buildkite-agent artifact upload "${SHARD_DIR}/**/*" || true
fi

exit "${RC}"
