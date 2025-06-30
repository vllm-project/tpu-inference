#!/bin/bash

if [ "$#" -eq 0 ]; then
    echo "Error: need to specify the local path to the root directory of vllm repo." >&2
    exit 1
fi

VLLM_CODE_PATH=$1

# This script appends specific import and assignment lines to the vLLM engine's core.py file.
FILE_PATH="$VLLM_CODE_PATH/vllm/v1/engine/core.py"
CONTENT_TO_ADD="
# Added for TPU support
from tpu_commons.core.core_tpu import EngineCore as TPUEngineCore
from tpu_commons.core.core_tpu import EngineCoreProc as TPUEngineCoreProc

EngineCore = TPUEngineCore
EngineCoreProc = TPUEngineCoreProc
"

# Check if the target file exists before trying to modify it
if [ -f "$FILE_PATH" ]; then
    echo "$CONTENT_TO_ADD" >> "$FILE_PATH"
fi

TPU_BACKEND_TYPE=jax python "$VLLM_CODE_PATH/examples/offline_inference/basic/generate.py" --task=generate --model=meta-llama/Meta-Llama-3-8B-Instruct --max_model_len=1024 --max_num_seqs=8

head -n -8 "$FILE_PATH" > "tmp.py"
mv "tmp.py" "$FILE_PATH"
