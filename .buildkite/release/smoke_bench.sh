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

# Client half of the release smoke test (smoke_test.sh). Runs in the TPU pod,
# from the image's checkout, once the server answers /health: a short
# random-prompt benchmark that passes only if every request completed.
#
#   smoke_bench.sh <model> <num_prompts>
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 <model> <num_prompts>" >&2
  exit 2
fi
model=$1
num_prompts=$2
result_dir=/tmp/release-smoke
result_file=result.json

vllm bench serve \
  --model "${model}" \
  --backend vllm \
  --request-rate inf \
  --dataset-name random \
  --num-prompts "${num_prompts}" \
  --random-input-len 128 \
  --random-output-len 128 \
  --percentile-metrics ttft,tpot,itl,e2el \
  --ignore-eos \
  --skip-chat-template \
  --save-result \
  --result-dir "${result_dir}" \
  --result-filename "${result_file}"

# vllm bench serve can exit 0 with failed requests, so check the counts.
python3 - "${result_dir}/${result_file}" "${num_prompts}" <<'EOF'
import json
import sys

with open(sys.argv[1]) as f:
    result = json.load(f)
want = int(sys.argv[2])
completed = result.get("completed")
failed = result.get("failed")
print(f"Completed {completed} of {want} requests, {failed} failed.")
sys.exit(0 if completed == want and failed == 0 else 1)
EOF
