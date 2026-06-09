#!/bin/bash
# Usage: ./run_ablation_parallel.sh <mode>
# Modes: standard, blind, shuffled, blurred, option_shuffled

MODE=$1
MODEL_NAME=${2:-"qwen2_5vl-7b"}
MODEL_PATH=${3:-"Qwen/Qwen2.5-VL-7B-Instruct"}

if [ -z "$MODE" ]; then
  echo "Usage: ./run_ablation_parallel.sh <mode> [model_name] [model_path]"
  exit 1
fi

PYTHON_EXEC="/drive/micromamba/envs/SpatialScore/bin/python"
OUTPUT_DIR="/drive/SpatialScore/eval_results"
MODE_DIR="${OUTPUT_DIR}/${MODEL_NAME}/${MODE}"

echo "=== Starting 8-GPU Parallel Evaluation ==="
echo "Mode: ${MODE}"
echo "Model: ${MODEL_NAME} (${MODEL_PATH})"
echo "Output Directory: ${MODE_DIR}"
echo "=========================================="

mkdir -p "${MODE_DIR}"
cd /drive/SpatialScore

# 1. Spawn 8 parallel processes on GPUs 0-7
for i in {0..7}; do
  CUDA_VISIBLE_DEVICES=$i ${PYTHON_EXEC} /drive/SpatialScore/stress_test_sharded.py \
    --shard_idx $i \
    --num_shards 8 \
    --mode ${MODE} \
    --model_name ${MODEL_NAME} \
    --model_path ${MODEL_PATH} \
    --output_dir ${OUTPUT_DIR} > "${MODE_DIR}/shard_${i}.log" 2>&1 &
  
  echo "Launched Shard $i on GPU $i (PID: $!)"
done

# 2. Wait for all background tasks to complete
echo "Waiting for all shards to finish..."
wait
echo "All shards completed!"

# 3. Merge the shard outputs
echo "Merging shard outputs..."
${PYTHON_EXEC} -c "import os, json; mode_dir='${MODE_DIR}'; merged=[]; [merged.extend(json.load(open(os.path.join(mode_dir, f'shard_{i}.json')))) for i in range(8) if os.path.exists(os.path.join(mode_dir, f'shard_{i}.json'))]; open(os.path.join(mode_dir, 'all_results.json'), 'w').write(json.dumps(merged, indent=2)); print(f'Merged {len(merged)} items!')"

# 4. Run evaluation metrics
echo "Running rule-based metric evaluation..."
${PYTHON_EXEC} /drive/SpatialScore/evaluate_results.py \
  --input ${MODE_DIR} \
  --no_llm
