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


echo "Waiting for vLLM server to be ready..."

# Wait for server to be ready
while true; do
    response=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8000/health 2>/dev/null)
    if [ "$response" = "200" ]; then
        echo "Server is ready!"
        break
    fi
    echo "Waiting... (checking health endpoint)"
    sleep 10
done

echo ""
echo "Running correctness test..."
echo ""

curl -X POST http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d @- << 'EOF'
{
  "model": "Qwen/Qwen3.5-397B-A17B-FP8",
  "prompt": "<|im_start|>system\nYou are a psychology expert. For the following multiple choice question, think step by step and then finish your answer with \"the answer is (X)\" where X is the correct letter choice.<|im_end|>\n<|im_start|>user\nQuestion:\nImages and sounds are maintained in sensory memory for:\nOptions:\nA. less than 5 seconds.\nB. 30 to 45 seconds.\nC. only while the sensory input is present.\nD. 10 to 15 seconds.\nAnswer: Let's think step by step.<|im_end|>\n<|im_start|>assistant\n<think>\n",
  "temperature": 0.0,
  "max_tokens": 2048,
  "stop": ["Question:", "<|im_end|>"]
}
EOF

echo ""
echo ""
echo "Test completed!"
