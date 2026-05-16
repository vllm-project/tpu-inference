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

# E2B integration probe — experimental batched RPA path.
set -uo pipefail

echo "[e2b-probe] USE_BATCHED_RPA_KERNEL=1 path"
export USE_BATCHED_RPA_KERNEL=1
export JITTED_MM_MODULE_KEYS="model.vision_tower.encoder"
export REGISTER_MM_MODULE_CUSTOM_PYTREE_CLASSES="transformers.modeling_outputs.BaseModelOutputWithPast"
export SKIP_JAX_PRECOMPILE=1
export VLLM_XLA_CHECK_RECOMPILATION=0

python3 /workspace/tpu_inference/examples/offline_inference.py \
  --model google/gemma-4-E2B-it \
  --tensor-parallel-size 1 \
  --max-num-batched-tokens 4096 \
  --max-model-len 4096 \
  --use-chat-template
