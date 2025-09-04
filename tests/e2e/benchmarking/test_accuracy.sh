#!/bin/bash

model_list="meta-llama/Llama-3.1-8B-Instruct meta-llama/Llama-3.1-70B-Instruct"
tensor_parallel_size=1
gpu_enabled=false

extra_serve_args=()
echo extra_serve_args: "${extra_serve_args[@]}"

root_dir=/workspace
exit_code=0

helpFunction()
{
   echo ""
   echo "Usage: $0 [-r full_path_to_root_dir -m model_id]"
   echo -e "\t-r The path your root directory containing both 'vllm' and 'tpu_commons' (default: /workspace/, which is used in the Dockerfile)"
   echo -e "\t-m A space-separated list of HuggingFace model ids to use (default: meta-llama/Llama-3.1-8B-Instruct and meta-llama/Llama-3.1-70B-Instruct)"
   echo -e "\t-t Tensor parallel size (default: 1)"
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
            model_list="$2"
            shift
            shift
            ;;
        -t|--tensor-parallel-size)
            tensor_parallel_size="$2"
            shift
            shift
            ;;
        -g|--gpu)
            gpu_enabled=true
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

echo "Using the root directory at $root_dir"
echo "Testing $model_list prompts"

cd "$root_dir"/vllm/tests/entrypoints/llm || exit

# Overwrite a few of the vLLM benchmarking scripts with the TPU Commons ones
cp "$root_dir"/tpu_commons/scripts/vllm/integration/*.py "$root_dir"/vllm/tests/entrypoints/llm/

comma_model_list=${model_list// /,}

echo "--------------------------------------------------"
echo "Running integration for models: $comma_model_list"
echo "--------------------------------------------------"

# Default action
if $gpu_enabled; then
    python3 -m pytest -rP test_accuracy.py::test_lm_eval_accuracy_v1_engine --tensor-parallel-size="$tensor_parallel_size" --model-names="$comma_model_list"
else
    python -m pytest -rP test_accuracy.py::test_lm_eval_accuracy_v1_engine --tensor-parallel-size="$tensor_parallel_size" --model-names="$comma_model_list"
fi

exit $exit_code