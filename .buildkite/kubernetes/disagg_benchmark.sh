#!/usr/bin/env bash
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

# Client of the 1P1D disaggregated benchmark: brings up the proxy in front of the
# two engines, sweeps concurrency, and publishes each cell. Runs as the
# `benchmark` role of manifests/workloads/qwen3-coder-480b-1p1d.yaml.

set -e

# JobSet per-pod DNS: <jobset>-<role>-<job>-<pod>.<jobset>
prefill_host=${WORKLOAD_NAME}-prefill-0-0.${WORKLOAD_NAME}
decode_host=${WORKLOAD_NAME}-decode-0-0.${WORKLOAD_NAME}

# Every result leaves this pod through this binary, so prove a real upload works
# before spending hours on results it could not deliver.
echo "${WORKLOAD_NAME} $(date -u +%FT%TZ)" > /tmp/disagg-preflight.txt
if ! (cd /tmp && buildkite-agent artifact upload disagg-preflight.txt); then
  echo "ERROR: this pod cannot publish artifacts; the benchmark's results would be lost."
  exit 1
fi

echo "Starting proxy server..."
python3 /opt/kube/toy_proxy_server.py \
  --host 0.0.0.0 \
  --port 10000 \
  --prefiller-host "$prefill_host" \
  --prefiller-port 8000 \
  --decoder-host "$decode_host" \
  --decoder-port 8000 &
proxy_pid=$!

# The proxy is backgrounded, so its death cannot fail the script on its own.
require_proxy() {
  kill -0 "$proxy_pid" 2>/dev/null && return 0
  echo "ERROR: the proxy (pid $proxy_pid) exited; nothing can reach the engines."
  exit 1
}

echo "Waiting for prefill ($prefill_host) and decode ($decode_host) servers..."
# -f so an error status is not read as healthy, -m so a hung connection cannot
# stretch the 10s iteration this count assumes.
health_tries=0
until curl -fsS -m 10 "http://$prefill_host:8000/health" && curl -fsS -m 10 "http://$decode_host:8000/health"; do
  require_proxy
  health_tries=$((health_tries+1))
  if [ "$health_tries" -gt 1080 ]; then
    echo "ERROR: servers not healthy after 3h; failing the benchmark."
    exit 1
  fi
  echo "Waiting for prefill and decode health checks..."
  sleep 10
done

echo "Both prefill and decode servers are HEALTHY!"

mkdir -p /tmp/benchmark_results
model="Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8"
publish_failed=0

# The pod is deleted the moment the JobSet finishes, so results leave from here.
publish_result() {
  local path=$1
  [ -f "$path" ] || return 0
  local name
  name=$(basename "$path")

  # Uploaded once - Buildkite adds a second artifact rather than replacing one
  # of the same name, so a retry pass shows up as a duplicate, not a correction.
  (cd "$(dirname "$path")" && buildkite-agent artifact upload "$name") \
    || { echo "WARNING: could not upload $name"; publish_failed=1; }

  # A cell with failed requests describes a broken serving path, not a
  # measurement; stop rather than spend the remaining hours on it.
  if grep -q -E '"failed": *[1-9]' "$path"; then
    echo "ERROR: $name recorded failed requests: $(grep -oE '"(completed|failed)": *[0-9]+' "$path" | tr '\n' ' ')"
    exit 1
  fi
}

# Each cell writes its own file and is then appended to the file for the whole
# input shape, which is what the result parser reads. Publishing per cell means a
# sweep cut short by the deadline still yields the cells that finished, under the
# c<C>_i<I>_o<O> names the parser expects.
run_sweep() {
  local input_len=$1 output_len=$2 num_prompts=$3
  local combined="/tmp/benchmark_results/${input_len}_${output_len}.json"

  # Keep this list short: each extra value is another cell against the deadline.
  for concurrency in ${DISAGG_CONCURRENCIES:-16 32 64 256}; do
    # Nearly serial at low concurrency, where the full count would dominate.
    local effective=$num_prompts
    if [ "$concurrency" -eq 1 ]; then
      effective=32
    elif [ "$concurrency" -eq 4 ]; then
      effective=64
    fi

    local cell="/tmp/benchmark_results/pd_disagg_c${concurrency}_i${input_len}_o${output_len}.json"

    echo "Starting benchmark cell: c=$concurrency, i=$input_len, o=$output_len, n=$effective"
    vllm bench serve \
      --model="$model" \
      --dataset-name=random \
      --num-warmups 10 \
      --random-input-len="$input_len" \
      --random-output-len="$output_len" \
      --num-prompts="$effective" \
      --ignore-eos \
      --host=localhost \
      --port=10000 \
      --max-concurrency="$concurrency" \
      --request-rate=inf \
      --metric-percentiles 90,99 \
      --append-result \
      --result-file="$cell"

    publish_result "$cell"

    # Appended here so each cell stays a file of its own to publish.
    cat "$cell" >> "$combined"
    sleep 30
  done
}

run_sweep 1024 8192 256
run_sweep 8192 1024 256

echo "Benchmark complete! Results saved in /tmp/benchmark_results."

if [ "$publish_failed" -ne 0 ]; then
  echo "ERROR: the benchmark completed but at least one result was not published."
  exit 1
fi
