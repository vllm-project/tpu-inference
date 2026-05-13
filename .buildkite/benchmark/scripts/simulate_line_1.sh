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

# ----------------------------------------------------------------
# 完全依照 run_bm.sh 層級模擬 Line 1 靈異現象
# ----------------------------------------------------------------
set -Eeuo pipefail

# 0. Panic Handler (與原腳本一致)
on_crash() {
    local exit_code=$?
    local line_no=$1
    local command="$2"
    
    if [ "$exit_code" -eq 0 ]; then return; fi

    echo ""
    echo "================================================================"
    echo "🚨 [FATAL ERROR] Bash Script Crashed Unexpectedly!"
    echo "================================================================"
    echo "File:     $(basename "$0")"
    echo "Line:     $line_no"
    echo "Command:  $command"
    echo "ExitCode: $exit_code"
    echo "================================================================"
}

trap 'on_crash ${LINENO} "$BASH_COMMAND"' ERR

# 模擬變數
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BM_LOG="./sim_bm_log.txt"
RECORD_ID="repro-test"

# 模擬 report_and_exit (關鍵點：不拔 trap，直接 exit)
report_and_exit() {
  local exit_code=${1:-0}
  echo "--- Calling report_result.sh for RECORD_ID=${RECORD_ID}"
  # 這裡模擬原腳本執行 report 並退出
  exit "$exit_code"
}

# 模擬 run_benchmark (誘發 grep 失敗)
run_benchmark(){
  echo "running benchmark..." >&2
  
  # 模擬執行 client 指令並產生 log
  echo "Some dummy vLLM output" > "$BM_LOG"
  
  # 模擬 grep 失敗 (因為 log 裡沒有這串字)
  # 這會導致 grep 回傳 1
  throughput=$(grep "Request throughput (req/s):" "$BM_LOG" | sed 's/[^0-9.]//g')
  p99_e2el=$(grep "P99 E2EL (ms):" "$BM_LOG" | awk '{print $NF}')
  
  echo "throughput: $throughput, P99 E2EL:$p99_e2el"
  echo "$throughput $p99_e2el"
}

# 模擬 execute_benchmark_safely
execute_benchmark_safely() {
    local rate_arg="${1:-}"
    local output
    local bm_exit_code

    set +e  
    # 這裡模擬管道操作
    output=$(run_benchmark "$rate_arg" | tail -n 1)
    bm_exit_code=$?
    set -e

    if [[ "$bm_exit_code" -ne 0 ]]; then
        echo "[ERROR] Benchmark client crashed with exit code $bm_exit_code!"
        report_and_exit 1
    fi

    # 模擬後續解析失敗 (因為 grep 沒抓到東西，output 會是空的)
    local temp_throughput
    local temp_p99
    read -r temp_throughput temp_p99 <<< "$output"

    if ! [[ "$temp_throughput" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
        echo "[ERROR] Failed to parse metrics! Output was: '$output'"
        report_and_exit 1
    fi
}

# --- 模擬 Main Flow ---

# 1. 關鍵點：模擬 eval 汙染環境
# 原腳本有這行，這會讓 Bash 的行號追蹤在 exit 時變得不穩定
eval "echo 'Simulating Python Parser Output'"

echo "Starting reproduction run..."
execute_benchmark_safely