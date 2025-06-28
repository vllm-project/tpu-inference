#!/bin/bash

# This script appends specific import and assignment lines to the vLLM engine's core.py file.
FILE_PATH="$HOME/vllm/vllm/v1/engine/core.py"
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

cd $HOME

TPU_BACKEND_TYPE=jax python vllm/examples/offline_inference/basic/generate.py --task=generate --model=meta-llama/Meta-Llama-3-8B-Instruct --max_model_len=1024 --max_num_seqs=8

head -n -8 "$FILE_PATH" > "tmp.py"
mv "tmp.py" "$FILE_PATH"
