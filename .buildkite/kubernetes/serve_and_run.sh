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

# Serves a model across a Ray slice and runs a client against it.
#
#   serve_and_run.sh "<serve command>" "<client command>"
#
# Run by multihost_entry.sh on the head, once every host has joined the
# cluster. On bare metal these are run_multihost.sh's two arguments, which it
# starts in sequence in the head's container; the kube entrypoint runs a single
# command on the head instead, so the sequence lives here. Both are bash
# command strings, as run_multihost.sh takes them, and the client runs from the
# repo root.
#
# The exit status is the client's, or 1 if the server never became healthy.
#
# SERVER_TIMEOUT_S bounds the wait for /health; 7200 by default, as in
# run_multihost.sh.
set -uo pipefail

if [ "$#" -ne 2 ]; then
  echo "usage: $0 \"<serve command>\" \"<client command>\"" >&2
  exit 2
fi
serve_cmd="$1"
client_cmd="$2"

port=8000
if [[ "$serve_cmd" =~ --port[=\ ]+([0-9]+) ]]; then
  port="${BASH_REMATCH[1]}"
fi
timeout_s="${SERVER_TIMEOUT_S:-7200}"

echo "--- Serving: ${serve_cmd}"
bash -c "$serve_cmd" &
serve_pid=$!

stop_server() {
  kill -TERM "$serve_pid" 2>/dev/null || return 0
  local deadline=$((SECONDS + 60))
  while [ "$SECONDS" -lt "$deadline" ]; do
    kill -0 "$serve_pid" 2>/dev/null || return 0
    sleep 1
  done
  kill -KILL "$serve_pid" 2>/dev/null || true
}

echo "--- Waiting up to ${timeout_s}s for /health on port ${port}"
deadline=$((SECONDS + timeout_s))
ready=0
while [ "$SECONDS" -lt "$deadline" ]; do
  if curl -sf -o /dev/null --connect-timeout 2 "http://localhost:${port}/health"; then
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
  if kill -0 "$serve_pid" 2>/dev/null; then
    echo "ERROR: the server was not healthy within ${timeout_s}s"
  fi
  stop_server
  exit 1
fi
echo "Server is healthy after ${SECONDS}s."

echo "--- Running: ${client_cmd}"
bash -c "$client_cmd"
rc=$?

stop_server
exit "$rc"
