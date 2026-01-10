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

"""
Performance Benchmarks for TPU Inference

This package contains benchmarking tools for comparing model implementations:

- qwen3_vl_perf_benchmark.py: Full model comparison (flax_nnx vs vllm/TorchAX)
- attention_kernel_benchmark.py: Attention kernel microbenchmarks

Usage Examples:

1. Full Qwen3VL benchmark:
   ```
   python -m benchmarks.qwen3_vl_perf_benchmark \
       --model "Qwen/Qwen3-VL-4B" \
       --seq-lengths 128,256,512 \
       --impl all
   ```

2. Attention kernel microbenchmark:
   ```
   python -m benchmarks.attention_kernel_benchmark \
       --seq-lengths 128,256,512,1024 \
       --num-heads 28 \
       --kv-heads 4 \
       --head-dim 128
   ```

3. Profile specific implementation:
   ```
   python -m benchmarks.qwen3_vl_perf_benchmark \
       --profile-dir /tmp/profile \
       --profile-only flax_nnx
   ```

Environment Variables:
- MODEL_IMPL_TYPE: "flax_nnx", "vllm", or "auto"
- NEW_MODEL_DESIGN: Enable new model design features
- XLA_PYTHON_CLIENT_PREALLOCATE: Set to "false" for better memory management
"""
