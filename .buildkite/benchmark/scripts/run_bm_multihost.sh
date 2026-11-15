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

# The part of run_multihost.sh that runs on the Ray head, for the kube fleet.
#
#   run_bm_multihost.sh <case.json> <TARGET_CASE_NAME>
#
# Run by .buildkite/kubernetes/multihost_entry.sh on the head once every host
# of the slice has joined the Ray cluster (run_job_kube.sh chooses it for a
# multi-host shape). Starts the case's server, then hands the benchmark to
# run_bm.sh with SERVER_ALREADY_RUNNING, which is how run_multihost.sh drives a
# multi-host case on bare metal.
set -uo pipefail

CASE_FILE="${1:-}"
TARGET_CASE_NAME="${2:-}"
if [[ -z "$CASE_FILE" || -z "$TARGET_CASE_NAME" ]]; then
    echo "Usage: $0 <case.json> <TARGET_CASE_NAME>" >&2
    exit 2
fi
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)

SERVER_CMD=()
SERVER_CMD_ENVS=()
eval "$(python3 "$SCRIPT_DIR/parser_case.py" "$CASE_FILE" "$TARGET_CASE_NAME")" || {
    echo "[ERROR] parser_case.py could not read ${TARGET_CASE_NAME} from ${CASE_FILE}" >&2
    exit 1
}
if (( ${#SERVER_CMD[@]} == 0 )); then
    echo "[ERROR] ${TARGET_CASE_NAME} has no server command; a multi-host case is a server and a client" >&2
    exit 1
fi

port=8000
ready_timeout_s=3600
for ((i = 0; i < ${#SERVER_CMD[@]}; i++)); do
    case "${SERVER_CMD[i]}" in
        --port) port="${SERVER_CMD[i + 1]}" ;;
        --port=*) port="${SERVER_CMD[i]#--port=}" ;;
    esac
done
for env_item in "${SERVER_CMD_ENVS[@]}"; do
    if [[ "$env_item" == VLLM_ENGINE_READY_TIMEOUT_S=* ]]; then
        ready_timeout_s="${env_item#*=}"
    fi
done

# vLLM reads a generation config from a local directory. parser_case.py leaves
# a gs:// one in place for a multi-host case, because on bare metal
# run_multihost.sh fetches it; here that falls to this script.
for ((i = 0; i < ${#SERVER_CMD[@]}; i++)); do
    if [[ "${SERVER_CMD[i]}" == "--generation-config" && "${SERVER_CMD[i + 1]:-}" == gs://* ]]; then
        gcs_config="${SERVER_CMD[i + 1]}"
        local_config="${ARTIFACT_FOLDER:-/workspace/tpu_inference/artifacts}/generation_configs"
        mkdir -p "$local_config"
        echo "--- Fetching the generation config ${gcs_config}"
        gsutil -m cp "${gcs_config}/config.json" "${gcs_config}/generation_config.json" "$local_config/" || {
            echo "[ERROR] could not fetch ${gcs_config}" >&2
            exit 1
        }
        SERVER_CMD[i + 1]="$local_config"
    fi
done

# run_bm.sh copies the server log from here into its artifacts when the server
# runs outside it.
server_log=/root/vllm_serve.log

echo "--- Starting the server on the head, across the Ray cluster"
printf '[DEBUG] %s %s\n' "${SERVER_CMD_ENVS[*]}" "${SERVER_CMD[*]}"
env "${SERVER_CMD_ENVS[@]}" "${SERVER_CMD[@]}" > "$server_log" 2>&1 &
server_pid=$!

stop_server() {
    kill -TERM "$server_pid" 2>/dev/null || return 0
    for _ in {1..30}; do
        kill -0 "$server_pid" 2>/dev/null || return 0
        sleep 1
    done
    kill -KILL "$server_pid" 2>/dev/null || true
}

echo "--- Waiting up to ${ready_timeout_s}s for /health on port ${port}"
deadline=$((SECONDS + ready_timeout_s))
ready=0
while (( SECONDS < deadline )); do
    if curl -sf -o /dev/null --connect-timeout 2 "http://localhost:${port}/health"; then
        ready=1
        break
    fi
    # Nothing is going to answer once the server has gone.
    if ! kill -0 "$server_pid" 2>/dev/null; then
        echo "[ERROR] the server exited before it became healthy"
        break
    fi
    sleep 10
done
if (( ready == 0 )); then
    (( SECONDS >= deadline )) && echo "[ERROR] the server was not healthy after ${ready_timeout_s}s"
    tail -n 200 "$server_log" || true
    stop_server
    exit 1
fi

SERVER_ALREADY_RUNNING=true VLLM_PORT="$port" \
    bash "$SCRIPT_DIR/run_bm.sh" "$CASE_FILE" "$TARGET_CASE_NAME"
rc=$?
stop_server
exit "$rc"
