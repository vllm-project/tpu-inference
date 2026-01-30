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

# test_env.sh

echo "--- 🔍 Environment Variable Check"

if [ -z "$MODEL_IMPL_TYPE" ]; then
  echo "❌ Error: MODEL_IMPL_TYPE is not set in the shell environment!"
  exit 1
else
  echo "✅ Success: MODEL_IMPL_TYPE is found! Value: $MODEL_IMPL_TYPE"
fi

echo "--- 🛠️ Execution Logic"
echo "MODEL_IMPL_TYPE from yml: $MODEL_IMPL_TYPE"

MODEL_IMPL_TYPE=ccc

echo "MODEL_IMPL_TYPE in script: $MODEL_IMPL_TYPE"