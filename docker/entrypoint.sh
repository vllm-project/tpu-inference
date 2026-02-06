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

# Entrypoint script for vLLM TPU Docker image
# 
# Default behavior: Starts the vLLM OpenAI-compatible API server
# To start bash shell instead: Set DISABLE_VLLM_SERVER=true
#
# Examples:
#   # Default: Start vLLM server
#   docker run vllm-tpu
#   
#   # Start vLLM server with arguments
#   docker run -e VLLM_ARGS="--model=meta-llama/Llama-2-7b --port=8080" vllm-tpu
#
#   # Start bash shell instead (backward compatible)
#   docker run -e DISABLE_VLLM_SERVER=true -it vllm-tpu
#
#   # Run custom command
#   docker run vllm-tpu python3 my_script.py

set -e

# If arguments are provided, execute them directly
if [ $# -gt 0 ]; then
    exec "$@"
fi

# If DISABLE_VLLM_SERVER is set to true, start bash shell (backward compatible)
if [ "${DISABLE_VLLM_SERVER:-false}" = "true" ]; then
    exec /bin/bash
fi

# Default: Start vLLM OpenAI-compatible API server
echo "Starting vLLM OpenAI-compatible API server..."
exec python3 -m vllm.entrypoints.openai.api_server ${VLLM_ARGS:-}
