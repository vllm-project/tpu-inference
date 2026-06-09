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

# Usage: ./run_spacethinker.sh <mode>
# Modes: standard, blind, shuffled, blurred, option_shuffled

MODE=$1
if [ -z "$MODE" ]; then
  echo "Usage: ./run_spacethinker.sh <mode>"
  exit 1
fi

PYTHON_EXEC="/drive/micromamba/envs/SpatialScore/bin/python"
OUTPUT_DIR="/drive/SpatialScore/eval_results"
MODEL_NAME="spacethinker-qwen2_5vl-3b"
MODEL_PATH="remyxai/SpaceThinker-Qwen2.5VL-3B"
MODE_DIR="${OUTPUT_DIR}/${MODEL_NAME}/${MODE}"

mkdir -p "${MODE_DIR}"
cd /drive/SpatialScore || exit


FREE_GPUS=(0 1 3 4 5 7)

echo "=== Starting 6-GPU SpaceThinker-3B Evaluation ==="
echo "Mode: ${MODE}"
echo "Output Directory: ${MODE_DIR}"
echo "================================================="

# 1. Launch 6 concurrent shards on the 6 idle GPUs
for i in {0..5}; do
  gpu_id=${FREE_GPUS[$i]}
  CUDA_VISIBLE_DEVICES="$gpu_id" ${PYTHON_EXEC} /drive/SpatialScore/stress_test_sharded.py \
    --shard_idx "$i" \
    --num_shards 6 \
    --mode "${MODE}" \
    --model_name "${MODEL_NAME}" \
    --model_path "${MODEL_PATH}" \
    --output_dir "${OUTPUT_DIR}" > "${MODE_DIR}/shard_${i}.log" 2>&1 &
    
  echo "Launched SpaceThinker Shard $i on GPU $gpu_id (PID: $!)"
done

# 2. Wait for all shards to complete
echo "Waiting for all SpaceThinker shards to finish..."
wait
echo "All SpaceThinker shards completed!"

# 3. Merge shard outputs
echo "Merging shard outputs..."
${PYTHON_EXEC} -c "
import os, json
mode_dir = '${MODE_DIR}'
merged = []
for i in range(6):
    shard_path = os.path.join(mode_dir, f'shard_{i}.json')
    if os.path.exists(shard_path):
        with open(shard_path, 'r') as f:
            merged.extend(json.load(f))
with open(os.path.join(mode_dir, 'all_results.json'), 'w') as f:
    json.dump(merged, f, indent=2)
print(f'Merged {len(merged)} items into all_results.json!')
"

# 4. Run official rule-based metric evaluation
echo "Running rule-based metric evaluation..."
${PYTHON_EXEC} /drive/SpatialScore/evaluate_results.py \
  --input "${MODE_DIR}" \
  --no_llm
