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

# # Ironwood qwen & Llama
echo "./.buildkite/benchmark/scripts/schedule_run.sh ./.buildkite/benchmark/cases/daily_qwen_llama_tpu7x_2.csv $CODE_HASH $TAG DAILY"
./.buildkite/benchmark/scripts/schedule_run.sh ./.buildkite/benchmark/cases/daily_qwen_llama_tpu7x_2.csv "$CODE_HASH" $TAG DAILY

# Ironwood Deepseek
echo "./.buildkite/benchmark/scripts/schedule_run.sh ./.buildkite/benchmark/cases/daily_deepseek_tpu7x_8.csv $CODE_HASH $TAG DAILY \"JAX_RANDOM_WEIGHTS=true;VLLM_MLA_DISABLE=1;NEW_MODEL_DESIGN=True;TPU_BACKEND_TYPE=jax\""
./.buildkite/benchmark/scripts/schedule_run.sh ./.buildkite/benchmark/cases/daily_deepseek_tpu7x_8.csv "$CODE_HASH" $TAG DAILY "JAX_RANDOM_WEIGHTS=true;VLLM_MLA_DISABLE=1;NEW_MODEL_DESIGN=True;TPU_BACKEND_TYPE=jax"

# # Ironwood Deepseek Accuracy
echo "./.buildkite/benchmark/scripts/schedule_run.sh ./.buildkite/benchmark/cases/accuracy_jax_v7x.csv $CODE_HASH $TAG JAX_ACCURACY \"VLLM_MLA_DISABLE=1;NEW_MODEL_DESIGN=True;TPU_BACKEND_TYPE=jax;\""
./.buildkite/benchmark/scripts/schedule_run.sh ./.buildkite/benchmark/cases/accuracy_jax_v7x.csv "$CODE_HASH" $TAG JAX_ACCURACY "VLLM_MLA_DISABLE=1;NEW_MODEL_DESIGN=True;TPU_BACKEND_TYPE=jax;"

# # GPT OSS
echo "./.buildkite/benchmark/scripts/schedule_run.sh ./.buildkite/benchmark/cases/daily_gpt_oss_120b_tpu7x.csv $CODE_HASH $TAG DAILY \"USE_MOE_EP_KERNEL=0;MODEL_IMPL_TYPE=vllm\""
./.buildkite/benchmark/scripts/schedule_run.sh ./.buildkite/benchmark/cases/daily_gpt_oss_120b_tpu7x.csv "$CODE_HASH" $TAG DAILY "USE_MOE_EP_KERNEL=0;MODEL_IMPL_TYPE=vllm"
# # Qwen 3-480B
echo "./.buildkite/benchmark/scripts/schedule_run.sh ./.buildkite/benchmark/cases/daily_qwen3_480B_FP8_tpu7x_8.csv $CODE_HASH $TAG DAILY"
./.buildkite/benchmark/scripts/schedule_run.sh ./.buildkite/benchmark/cases/daily_qwen3_480B_FP8_tpu7x_8.csv "$CODE_HASH" $TAG DAILY
