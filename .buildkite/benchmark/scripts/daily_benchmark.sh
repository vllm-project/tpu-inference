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

CODE_HASH=$1
TIMEZONE="America/Los_Angeles"
TAG="$(TZ="$TIMEZONE" date +%Y%m%d_%H%M%S)"
JOB_PRIORITY=$(buildkite-agent meta-data get "JOB_PRIORITY")

echo "Benchmark pipeline prioirty: $JOB_PRIORITY"

# Upload Only one test case for Dev
echo "./.buildkite/benchmark/scripts/schedule_run.sh ./.buildkite/benchmark/cases/benchmark_dev_test_v7x.csv $CODE_HASH $TAG DAILY $JOB_PRIORITY"
./.buildkite/benchmark/scripts/schedule_run.sh ./.buildkite/benchmark/cases/benchmark_dev_test_v7x.csv "$CODE_HASH" "$TAG" DAILY "$JOB_PRIORITY"

# # Ironwood qwen & Llama
# echo "./.buildkite/benchmark/scripts/schedule_run.sh ./.buildkite/benchmark/cases/daily_qwen_llama_tpu7x_2.csv $CODE_HASH $TAG DAILY $JOB_PRIORITY"
# ./.buildkite/benchmark/scripts/schedule_run.sh ./.buildkite/benchmark/cases/daily_qwen_llama_tpu7x_2.csv "$CODE_HASH" "$TAG" DAILY "$JOB_PRIORITY"

# # Qwen3-32B random benchmarks (Using benchmark_serving)
# echo "./.buildkite/benchmark/scripts/schedule_run.sh ./.buildkite/benchmark/cases/daily_qwen3_32B_random_tpu7x_2.csv $CODE_HASH $TAG DAILY $JOB_PRIORITY \"USE_BENCHMARK_SERVING=1;MAX_CONCURRENCY=64;\""
# ./.buildkite/benchmark/scripts/schedule_run.sh ./.buildkite/benchmark/cases/daily_qwen3_32B_random_tpu7x_2.csv "$CODE_HASH" "$TAG" DAILY "$JOB_PRIORITY" "USE_BENCHMARK_SERVING=1;MAX_CONCURRENCY=64;"
# echo "./.buildkite/benchmark/scripts/schedule_run.sh ./.buildkite/benchmark/cases/daily_qwen3_32B_random_tpu7x_2.csv $CODE_HASH $TAG DAILY $JOB_PRIORITY \"USE_BENCHMARK_SERVING=1;MAX_CONCURRENCY=320;\""
# ./.buildkite/benchmark/scripts/schedule_run.sh ./.buildkite/benchmark/cases/daily_qwen3_32B_random_tpu7x_2.csv "$CODE_HASH" "$TAG" DAILY "$JOB_PRIORITY" "USE_BENCHMARK_SERVING=1;MAX_CONCURRENCY=320;"

# # Ironwood Deepseek DP Attention
# echo "./.buildkite/benchmark/scripts/schedule_run.sh ./.buildkite/benchmark/cases/daily_deepseek_dp_attention_tpu7x_8.csv $CODE_HASH $TAG DAILY $JOB_PRIORITY \"VLLM_MLA_DISABLE=0;NEW_MODEL_DESIGN=1;MOE_REQUANTIZE_BLOCK_SIZE=512;MOE_REQUANTIZE_WEIGHT_DTYPE=fp4;TPU_BACKEND_TYPE=jax;MODEL_IMPL_TYPE=vllm;\""
# ./.buildkite/benchmark/scripts/schedule_run.sh ./.buildkite/benchmark/cases/daily_deepseek_dp_attention_tpu7x_8.csv "$CODE_HASH" "$TAG" DAILY "$JOB_PRIORITY" "VLLM_MLA_DISABLE=0;NEW_MODEL_DESIGN=1;MOE_REQUANTIZE_BLOCK_SIZE=512;MOE_REQUANTIZE_WEIGHT_DTYPE=fp4;TPU_BACKEND_TYPE=jax;MODEL_IMPL_TYPE=vllm;"

# # Ironwood Deepseek
# echo "./.buildkite/benchmark/scripts/schedule_run.sh ./.buildkite/benchmark/cases/daily_deepseek_tpu7x_8.csv $CODE_HASH $TAG DAILY $JOB_PRIORITY \"VLLM_MLA_DISABLE=1;TPU_BACKEND_TYPE=vllm\""
# ./.buildkite/benchmark/scripts/schedule_run.sh ./.buildkite/benchmark/cases/daily_deepseek_tpu7x_8.csv "$CODE_HASH" "$TAG" DAILY "$JOB_PRIORITY" "VLLM_MLA_DISABLE=1;TPU_BACKEND_TYPE=vllm"

# # Ironwood Deepseek Accuracy
# echo "./.buildkite/benchmark/scripts/schedule_run.sh ./.buildkite/benchmark/cases/accuracy_jax_v7x.csv $CODE_HASH $TAG JAX_ACCURACY $JOB_PRIORITY \"VLLM_MLA_DISABLE=1;NEW_MODEL_DESIGN=True;TPU_BACKEND_TYPE=jax;\""
# ./.buildkite/benchmark/scripts/schedule_run.sh ./.buildkite/benchmark/cases/accuracy_jax_v7x.csv "$CODE_HASH" "$TAG" JAX_ACCURACY "$JOB_PRIORITY" "VLLM_MLA_DISABLE=1;NEW_MODEL_DESIGN=True;TPU_BACKEND_TYPE=jax;"

# # GPT OSS
# echo "./.buildkite/benchmark/scripts/schedule_run.sh ./.buildkite/benchmark/cases/daily_gpt_oss_120b_tpu7x.csv $CODE_HASH $TAG DAILY $JOB_PRIORITY \"USE_MOE_EP_KERNEL=0;MODEL_IMPL_TYPE=vllm\""
# ./.buildkite/benchmark/scripts/schedule_run.sh ./.buildkite/benchmark/cases/daily_gpt_oss_120b_tpu7x.csv "$CODE_HASH" "$TAG" DAILY "$JOB_PRIORITY" "USE_MOE_EP_KERNEL=0;MODEL_IMPL_TYPE=vllm"

# # Qwen 3-480B
# echo "./.buildkite/benchmark/scripts/schedule_run.sh ./.buildkite/benchmark/cases/daily_qwen3_480B_FP8_tpu7x_8.csv $CODE_HASH $TAG DAILY $JOB_PRIORITY"
# ./.buildkite/benchmark/scripts/schedule_run.sh ./.buildkite/benchmark/cases/daily_qwen3_480B_FP8_tpu7x_8.csv "$CODE_HASH" "$TAG" DAILY "$JOB_PRIORITY"
