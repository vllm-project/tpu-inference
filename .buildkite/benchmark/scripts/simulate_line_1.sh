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

# reproduce_line_1_bk.sh
# 模擬 run_bm.sh 的完整呼叫層級以重現 Line 1 報錯

set -Eeuo pipefail

# ==============================================================================
# 0. 全局崩潰攔截器 (與大腳本一致)
# ==============================================================================
on_crash() {
    local exit_code=$?
    local line_no=$1
    local command="$2"
    
    # 正常退出不處理
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
    echo ""
}

# 監聽 ERR 信號
trap 'on_crash ${LINENO} "$BASH_COMMAND"' ERR

# ==============================================================================
# 1. 核心退出函數
# ==============================================================================
report_and_exit() {
    local exit_code=${1:-0}
    echo "--- [DEBUG] Entering report_and_exit with code $exit_code ---"
    
    # 在大腳本中，這裡直接執行 exit 指令
    # 當外部存在 eval 殘留且處於某些 Bash 版本時，
    # 這裡的 exit 會被誤認為是執行失敗，進而觸發第二次 Trap (導致 Line 1)
    exit "$exit_code"
}

# ==============================================================================
# 2. 模擬 Benchmark 邏輯
# ==============================================================================
run_benchmark() {
    echo "running benchmark..." >&2
    # 故意製造一個會讓後續 grep 失敗的內容
    echo "vLLM is running, but metrics are missing"
}

execute_benchmark_safely() {
    local output
    local bm_exit_code

    set +e  
    # 模擬大腳本的管道賦值：這會建立子 Shell (Subshell)
    # 子 Shell 的失敗會傳遞給變數賦值動作
    output=$(run_benchmark | tail -n 1)
    bm_exit_code=$?
    set -e

    # 模擬解析失敗 (因為 output 裡面沒有數字)
    if ! [[ "$output" =~ [0-9] ]]; then
        echo "[ERROR] Failed to parse valid metrics! Output was: '$output'"
        # 這裡主動呼叫退出
        report_and_exit 1
    fi
}

# ==============================================================================
# 3. 主流程 (Main Flow)
# ==============================================================================

# 🚨 關鍵步驟：模擬 eval 汙染環境
# 大腳本一開始執行了 eval "$(python3 ...)"，這會改變 Bash 的內部行號偏移
eval "export PYTHON_ENV_LOADED=true"

echo "--- Buildkite Reproduction Start ---"

# 模擬深層嵌套呼叫
execute_benchmark_safely

echo "--- Reproduction End ---"