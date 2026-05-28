#!/bin/bash
# Copyright 2025 Google LLC
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

set -euo pipefail

echo "Running Support Matrices Generator Tests..."

export TPU_VERSION="tpu6e"
export BUILDKITE_TAG="test_version"
TEST_DIR=$(mktemp -d)
export TEST_DIR
trap 'rm -rf "$TEST_DIR" v6' EXIT

cd "$(dirname "$0")/.."

# Mock buildkite-agent
buildkite-agent() {
    local cmd="$1"
    local subcmd="$2"
    if [[ "$cmd" == "meta-data" && "$subcmd" == "get" ]]; then
        local key="$3"
        if [[ "$key" == "model-list" ]]; then
            echo -e "test_model"
        elif [[ "$key" == "feature-list" ]]; then
            echo -e "kernel support matrix microbenchmarks"
        elif [[ "$key" == "v6test_model_category" ]]; then
            echo "text-only"
        elif [[ "$key" == "v6test_model:UnitTest" ]]; then
            echo "✅ Passing"
        elif [[ "$key" == "v6test_model:Accuracy/Correctness" ]]; then
            echo "❌ Failing"
        elif [[ "$key" == "v6test_model:Benchmark" ]]; then
            echo "❓ Untested"
        elif [[ "$key" == "v6kernel support matrix microbenchmarks_category" ]]; then
            echo "kernel support matrix microbenchmarks"
        else
            echo "❓ Untested"
        fi
    elif [[ "$cmd" == "meta-data" && "$subcmd" == "set" ]]; then
        echo "$3=$4" >> "$TEST_DIR/metadata.txt"
    elif [[ "$cmd" == "artifact" && "$subcmd" == "upload" ]]; then
        cp "$3" "$TEST_DIR/"
    fi
}
export -f buildkite-agent

# Mock mapfile for MacOS compatibility (Bash 3.2 doesn't have it)
mapfile() {
    local array_name=""
    for arg in "$@"; do
        if [[ "$arg" != "-t" ]]; then
            array_name="$arg"
        fi
    done
    local lines=()
    while IFS= read -r line || [ -n "$line" ]; do
        lines+=("$line")
    done
    if [[ -n "$array_name" ]]; then
        eval "$array_name=(\"\${lines[@]}\")"
    fi
}
export -f mapfile

# Set up mock microbenchmark input
mkdir -p v6
cat << 'EOF' > v6/kernel_support_matrix_microbenchmarks.csv
kernels,CorrectnessTest,PerformanceTest,TPU Versions
fused_moe-w8a8,❓ Untested,❓ Untested,v6
fused_moe-w8a16,✅ Passing,✅ Passing,v6
fused_moe-w16a16,❓ Untested,❓ Untested,v6
mla,✅ Passing,❓ Untested,v6
generic ragged paged attention v3,✅ Passing,✅ Passing,v6
EOF

# Execute the script
bash .buildkite/scripts/generate_support_matrices.sh > /dev/null

# Assertions
echo "Verifying Outputs..."
if ! grep -q "CI_TESTS_FAILED=true" "$TEST_DIR/metadata.txt"; then
    echo "❌ ERROR: CI_TESTS_FAILED was not set to true despite test_model failing."
    exit 1
fi

PIVOT_CSV="$TEST_DIR/kernel_support_matrix-microbenchmarks.csv"
if [[ ! -f "$PIVOT_CSV" ]]; then
    echo "❌ ERROR: Pivoted Microbenchmark CSV not uploaded."
    exit 1
fi

# Verify header
if ! grep -q "Kernel,W16 A16 (Corr),W16 A16 (Perf),W8 A8 (Corr),W8 A8 (Perf),W8 A16 (Corr),W8 A16 (Perf),W4 A4 (Corr),W4 A4 (Perf),W4 A8 (Corr),W4 A8 (Perf),W4 A16 (Corr),W4 A16 (Perf)" "$PIVOT_CSV"; then
    echo "❌ ERROR: Incorrect Pivot Header."
    cat "$PIVOT_CSV"
    exit 1
fi

# Verify mla outputs correctly
if ! grep -q "\"mla\*\",✅ Passing,❓ Untested,❓ Untested,❓ Untested,❓ Untested,❓ Untested,❓ Untested,❓ Untested,❓ Untested,❓ Untested,❓ Untested,❓ Untested" "$PIVOT_CSV"; then
    echo "❌ ERROR: mla parsing failed or default values are incorrect."
    cat "$PIVOT_CSV"
    exit 1
fi

# Verify fused_moe outputs correctly
if ! grep -q "\"fused_moe\",❓ Untested,❓ Untested,❓ Untested,❓ Untested,✅ Passing,✅ Passing,❓ Untested,❓ Untested,❓ Untested,❓ Untested,❓ Untested,❓ Untested" "$PIVOT_CSV"; then
    echo "❌ ERROR: fused_moe parsing failed."
    cat "$PIVOT_CSV"
    exit 1
fi

# Verify string substitution correctness
if ! grep -q "\"generic ragged paged<br>attention v3\*\",✅ Passing,✅ Passing,❓ Untested,❓ Untested,❓ Untested,❓ Untested,❓ Untested,❓ Untested,❓ Untested,❓ Untested,❓ Untested,❓ Untested" "$PIVOT_CSV"; then
    echo "❌ ERROR: generic ragged paged attention v3 substitution failed."
    cat "$PIVOT_CSV"
    exit 1
fi

echo "✅ All tests passed successfully!"
