#!/bin/bash
set -e

# Extract necessary environment variables set by the CI YAML
TEST_MODEL=${TEST_MODEL}
TENSOR_PARALLEL_SIZE=${TENSOR_PARALLEL_SIZE}

# this assumes that the working dir is the root of the repo (i.e. /tpu-inference)
SKIP_JAX_PRECOMPILE=1 \
python /workspace/tpu_inference/examples/offline_llama_guard_4_inference.py \
  --model="$TEST_MODEL" \
  --tensor-parallel-size="$TENSOR_PARALLEL_SIZE" \
  --max_model_len=2048 \
  --max-num-batched-tokens=3072 \
  --hf_overrides '{"architectures": ["LlamaForCausalLM"]}' \
  --dataset-path "gs://jiries/datasets/ailuminate/airr_official_1.0_demo_en_us_prompt_set_release.csv"
