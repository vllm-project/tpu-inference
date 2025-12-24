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

set -e

test_model=""
tensor_parallel_size=1
minimum_accuracy_threshold=0

extra_serve_args=()
echo extra_serve_args: "${extra_serve_args[@]}"

root_dir=/workspace
exit_code=0

helpFunction()
{
   echo ""
   echo "Usage: $0 [-r full_path_to_root_dir -m model_id]"
   echo -e "\t-r The path your root directory containing both 'vllm' and 'tpu_inference' (default: /workspace/, which is used in the Dockerfile)"
   exit 1
}

while [[ "$#" -gt 0 ]]; do
    case "$1" in
        -r|--root-dir-path)
            root_dir="$2"
            shift
            shift
            ;;
        -h|--help)
            helpFunction
            ;;
        *) # unknown option
            echo "Unknown option: $1"
            helpFunction
            ;;
    esac
done

if [ -n "$TEST_MODEL" ]; then
  test_model="$TEST_MODEL"
fi

if [ -n "$MINIMUM_ACCURACY_THRESHOLD" ]; then
  minimum_accuracy_threshold="$MINIMUM_ACCURACY_THRESHOLD"
fi

if [ -n "$TENSOR_PARALLEL_SIZE" ]; then
  tensor_parallel_size="$TENSOR_PARALLEL_SIZE"
fi

# Check if test_model is provided and not empty
if [[ -z "$test_model" ]]; then
    echo "Error: Test model name (-m) is a required argument." >&2
    has_error=1
fi

# Check if tensor_parallel_size is an integer and greater than 0
if ! [[ "$tensor_parallel_size" =~ ^[1-9][0-9]*$ ]]; then
    echo "Error: Tensor parallel size (-t) must be an integer greater than 0. Got: '$tensor_parallel_size'" >&2
    has_error=1
fi

# Check if minimum_accuracy_threshold is a float and greater than 0
if ! awk -v num="$minimum_accuracy_threshold" 'BEGIN { exit !(num > 0) }'; then
    echo "Error: Minimum accuracy threshold (-e) must be a number greater than 0. Got: '$minimum_accuracy_threshold'" >&2
    has_error=1
fi

# If any validation failed, print help and exit
if [[ "$has_error" -ne 0 ]]; then
    helpFunction
fi


echo "Using the root directory at $root_dir"

cd "$root_dir"/vllm/tests/entrypoints/llm || exit

# Overwrite a few of the vLLM benchmarking scripts with the TPU Inference ones
cp "$root_dir"/tpu_inference/scripts/vllm/integration/*.py "$root_dir"/vllm/tests/entrypoints/llm/

echo "--------------------------------------------------"
echo "Running integration for model: $test_model"
echo "--------------------------------------------------"

# Default action
python -m pytest -rP test_accuracy.py::test_lm_eval_accuracy_v1_engine \
    --tensor-parallel-size="$tensor_parallel_size" \
    --model-name="$test_model" \
    --expected-value="$minimum_accuracy_threshold"

exit $exit_code
