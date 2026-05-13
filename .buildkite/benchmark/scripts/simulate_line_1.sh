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

set -Eeuo pipefail

shopt -s expand_aliases

on_crash() {
    local exit_code=$?
    echo "================================================================"
    echo "🚨 [FATAL ERROR] Trap Triggered!"
    echo "Line:     $1"
    echo "Command:  $BASH_COMMAND"
    echo "ExitCode: $exit_code"
    echo "================================================================"
}

trap 'on_crash ${LINENO}' ERR

alias my_exit='exit'

report_and_exit() {
    local code=$1
    echo "[DEBUG] Entering report_and_exit with code $code"

    my_exit "$code"
}

trigger_error() {
    echo "[DEBUG] Simulating failure..."
    ls /non/existent/file 2>/dev/null || report_and_exit 1
}

echo "--- Starting Final Reproduction ---"
trigger_error