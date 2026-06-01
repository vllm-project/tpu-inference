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

USE_BATCHED_RPA_KERNEL=1 vllm serve Qwen/Qwen3-32B \
  --max-model-len=2048 \
  --max-num-seqs=320 \
  --tensor-parallel-size 2 \
  --max-num-batched-tokens 4096 \
  --no-enable-prefix-caching \
  --additional_config='{"quantization": { "qwix": { "rules": [{ "module_path": ".*", "weight_qtype": "float8_e4m3fn", "act_qtype": "float8_e4m3fn"}]}}}' \
  --kv-cache-dtype=fp8 \
  --gpu-memory-utilization=0.98 \
  --async-scheduling \
  --block-size=256 > server_v11.log 2>&1
