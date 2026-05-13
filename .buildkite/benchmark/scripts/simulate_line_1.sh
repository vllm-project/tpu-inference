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
# 模擬 Buildkite 環境中的 Line 1 靈異現象
# ----------------------------------------------------------------
set -Eeuo pipefail

# 1. 崩潰處理器
on_crash() {
    local exit_code=$?
    echo ""
    echo "================================================================"
    echo "🚨 [FATAL ERROR] Trap Triggered!"
    echo "File:     ${BASH_SOURCE[1]:-${BASH_SOURCE[0]}}"
    echo "Line:     $1"
    echo "Command:  $BASH_COMMAND"
    echo "ExitCode: $exit_code"
    echo "================================================================"
}

# 監聽 ERR
trap 'on_crash ${LINENO}' ERR

# 2. 模擬 report_and_exit (故意不加 trap - ERR)
report_and_exit() {
    local code=$1
    echo "--- [DEBUG] Entering report_and_exit with code $code ---"
    
    # 這裡就是關鍵：在某些環境下，exit 1 會觸發 ERR trap，
    # 但因為正在退出，Stack 已空，LINENO 會變成 1。
    exit "$code"
}

# 3. 模擬 run_benchmark 失敗
# 利用管道 (|) 誘發子 Shell 不穩定狀態
run_benchmark_sim() {
    echo "Simulating benchmark..."
    # 故意讓 grep 失敗 (Exit 1)
    grep "NON_EXISTENT_STRING" <(echo "data") | awk '{print $1}'
}

# 4. 模擬 execute_benchmark_safely
execute_safely() {
    echo "--- [DEBUG] Starting execute_safely ---"
    local output
    
    # 模擬大腳本的管道賦值邏輯
    # 這種結構在失敗時會將 Exit Code 傳遞給父層，觸發 || 
    set +e
    output=$(run_benchmark_sim | tail -n 1)
    local bm_exit_code=$?
    set -e

    if [[ "$bm_exit_code" -ne 0 ]]; then
        echo "[DEBUG] Detected failure, calling report_and_exit..."
        report_and_exit "$bm_exit_code"
    fi
}

# --- 主流程 ---
echo "--- Buildkite Reproduction Start ---"
execute_safely