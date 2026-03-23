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

TIMEZONE="America/Los_Angeles"
JOB_REFERENCE="$(TZ="$TIMEZONE" date +%Y%m%d_%H%M%S)"
buildkite-agent meta-data set "JOB_REFERENCE" "${JOB_REFERENCE}"
JOB_PRIORITY=$(buildkite-agent meta-data get "JOB_PRIORITY")

echo "Benchmark pipeline priority: $JOB_PRIORITY"


# A small helper to upload YMLs with dynamic priority interpolation
upload_benchmark_yml() {
  local YML_FILE="$1"
  echo "--- Uploading $YML_FILE"
  # Prepend the priority to the YAML file before uploading
  (echo "priority: ${JOB_PRIORITY}"; cat "$YML_FILE") | buildkite-agent pipeline upload
}

# Upload Only one test case for Dev
upload_benchmark_yml .buildkite/benchmark/cases/benchmark_dev_test_v7x.yml

# # 1. Ironwood qwen & Llama
# upload_benchmark_yml .buildkite/benchmark/cases/daily_qwen_llama_tpu7x_2.yml

# # 2. Qwen3-32B random benchmarks
# upload_benchmark_yml .buildkite/benchmark/cases/daily_qwen3_32B_random_tpu7x_2.yml

# # 3. Ironwood Deepseek DP Attention
# upload_benchmark_yml .buildkite/benchmark/cases/daily_deepseek_dp_attention_tpu7x_8.yml

# # 4. Ironwood Deepseek
# upload_benchmark_yml .buildkite/benchmark/cases/daily_deepseek_tpu7x_8.yml

# # 5. Ironwood Deepseek Accuracy
# upload_benchmark_yml .buildkite/benchmark/cases/accuracy_jax_v7x.yml

# # 6. GPT OSS
# upload_benchmark_yml .buildkite/benchmark/cases/daily_gpt_oss_120b_tpu7x.yml

# # 7. Qwen 3-480B
# upload_benchmark_yml .buildkite/benchmark/cases/daily_qwen3_480B_FP8_tpu7x_8.yml
