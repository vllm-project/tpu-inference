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

echo "Starting 2.5 hour countdown at $(date)"
sleep 9500
echo "Countdown finished at $(date). Starting lm_eval..."

lm_eval \
  --model vllm \
  --model_args '{"pretrained": "Qwen/Qwen3.5-397B-A17B-FP8", "attention_backend": "FLASH_ATTN", "gdn_prefill_backend": "triton", "kv_cache_dtype": "fp8", "tensor_parallel_size": 8, "max_model_len": 5120, "max_num_batched_tokens": 4096, "block_size": 256, "limit_mm_per_prompt": {"image": 0, "video": 0}, "enable_thinking": false}' \
  --tasks mmlu_pro \
  --apply_chat_template \
  --verbosity DEBUG \
  --log_samples \
  --output_path eval_logs \
  --limit 5 \
  --seed 42
