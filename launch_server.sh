#!/bin/bash
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
