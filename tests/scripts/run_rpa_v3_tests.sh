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


# Install dependencies
pip install -U --pre jax jaxlib libtpu requests -i https://us-python.pkg.dev/ml-oss-artifacts-published/jax/simple/ -f https://storage.googleapis.com/jax-releases/libtpu_releases.html

TPU_INFERENCE_DIR="/workspace/tpu_inference/"

# RPA v3 test files - add new tests here
RPA_V3_TESTS=(
    "tests/kernels/ragged_paged_attention_kernel_v3_test.py"
    "tests/layers/attention/test_deepseek_v3_attention.py"
)

# Convert array to space-separated string for pytest
FULL_PATHS=()
for test in "${RPA_V3_TESTS[@]}"; do
    FULL_PATHS+=("$TPU_INFERENCE_DIR/$test")
done

# Run all tests in a single pytest command
pytest "${FULL_PATHS[@]}"
