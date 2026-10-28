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

# Serves a model across a Ray slice and asks it for one completion.
#
# Run by multihost_entry.sh on the head, once every host has joined the
# cluster. On bare metal this is two arguments to run_multihost.sh - a serve
# command and a client command - which that script starts in sequence. The kube
# entrypoint runs a single command on the head instead, so the sequence lives
# here.
#
# MODEL, TENSOR_PARALLEL_SIZE and ASYNC_SCHEDULING come from the step. The jax
# suite's bare-metal step serves without async scheduling and the features
# suite's with it, so each lane keeps its own.
set -uo pipefail

MODEL="${MODEL:?MODEL must name the checkpoint to serve}"
TP="${TENSOR_PARALLEL_SIZE:-16}"
PORT="${PORT:-8000}"
# The whole slice has to be resident before the server answers, and these
# weights stream from GCS.
READY_TIMEOUT_S="${READY_TIMEOUT_S:-3600}"
async_flag="--no-async-scheduling"
if [[ "${ASYNC_SCHEDULING:-0}" == "1" ]]; then
  async_flag="--async-scheduling"
fi

echo "--- Serving ${MODEL} at tensor-parallel-size ${TP}"
vllm serve "${MODEL}" \
  --port "${PORT}" \
  --tensor-parallel-size "${TP}" \
  --trust-remote-code \
  --max-model-len 1024 \
  "${async_flag}" \
  --load-format=runai_streamer \
  --no-enable-prefix-caching &
serve_pid=$!

echo "--- Waiting for /health"
deadline=$((SECONDS + READY_TIMEOUT_S))
ready=0
while [ "$SECONDS" -lt "$deadline" ]; do
  if curl -sf -o /dev/null --connect-timeout 2 "http://localhost:${PORT}/health"; then
    ready=1
    break
  fi
  # Nothing is going to arrive if the server has already gone.
  if ! kill -0 "$serve_pid" 2>/dev/null; then
    echo "ERROR: the server exited before it became healthy"
    break
  fi
  sleep 10
done
if [ "$ready" != "1" ]; then
  echo "ERROR: server did not become healthy within ${READY_TIMEOUT_S}s"
  kill "$serve_pid" 2>/dev/null || true
  exit 1
fi

echo "--- Asking for a completion"
response=$(curl -sf "http://localhost:${PORT}/v1/completions" \
  -X POST -H 'Content-Type: application/json' \
  -d "{\"model\": \"${MODEL}\", \"prompt\": \"San Francisco is a\", \"max_tokens\": 50}")
rc=$?
echo "$response"

# A 200 with no choices is a pass by curl's reckoning and a failure by ours.
if [ "$rc" -eq 0 ] && ! printf '%s' "$response" | python3 -c '
import json, sys
sys.exit(0 if (json.load(sys.stdin).get("choices") or [{}])[0].get("text") else 1)
'; then
  echo "ERROR: the completions response carried no text"
  rc=1
fi

kill "$serve_pid" 2>/dev/null || true
exit "$rc"
