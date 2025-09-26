#!/bin/bash

model_name=""
tensor_parallel_size=1
expected_value=0

extra_serve_args=()
echo extra_serve_args: "${extra_serve_args[@]}"

root_dir=/workspace
exit_code=0

helpFunction()
{
   echo ""
   echo "Usage: $0 [-r full_path_to_root_dir -m model_id]"
   echo -e "\t-r The path your root directory containing both 'vllm' and 'tpu_commons' (default: /workspace/, which is used in the Dockerfile)"
   echo -e "\t-m A space-separated list of HuggingFace model ids to use (Required)"
   echo -e "\t-t Tensor parallel size (default: 1)"
   echo -e "\t-e Excepted value (Required)"
   exit 1
}

while [[ "$#" -gt 0 ]]; do
    case "$1" in
        -r|--root-dir-path)
            root_dir="$2"
            shift
            shift
            ;;
        -m|--model)
            model_name="$2"
            shift
            shift
            ;;
        -t|--tensor-parallel-size)
            tensor_parallel_size="$2"
            shift
            shift
            ;;
        -e|--excepted-value)
            expected_value="$2"
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

# Check if model_name is provided and not empty
if [[ -z "$model_name" ]]; then
    echo "Error: Model name (-m) is a required argument." >&2
    has_error=1
fi

# Check if tensor_parallel_size is an integer and greater than 0
if ! [[ "$tensor_parallel_size" =~ ^[1-9][0-9]*$ ]]; then
    echo "Error: Tensor parallel size (-t) must be an integer greater than 0. Got: '$tensor_parallel_size'" >&2
    has_error=1
fi

# Check if expected_value is a float and greater than 0
if ! awk -v num="$expected_value" 'BEGIN { exit !(num > 0) }'; then
    echo "Error: Expected value (-e) must be a number greater than 0. Got: '$expected_value'" >&2
    has_error=1
fi

# If any validation failed, print help and exit
if [[ "$has_error" -ne 0 ]]; then
    helpFunction
fi


echo "Using the root directory at $root_dir"

cd "$root_dir"/vllm/tests/entrypoints/llm || exit

# Overwrite a few of the vLLM benchmarking scripts with the TPU Commons ones
cp "$root_dir"/tpu_commons/scripts/vllm/integration/*.py "$root_dir"/vllm/tests/entrypoints/llm/

echo "--------------------------------------------------"
echo "Running integration for model: $model_name"
echo "--------------------------------------------------"

# Default action
python -m pytest -rP test_accuracy.py::test_lm_eval_accuracy_v1_engine \
    --tensor-parallel-size="$tensor_parallel_size" \
    --model-name="$model_name" \
    --expected-value="$expected_value"

exit $exit_code