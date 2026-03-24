#!/bin/bash

pip install -e . --no-deps
# VLLM_TARGET_DEVICE="tpu" pip install -e ./vllm --no-deps

export VLLM_ENABLE_V1_MULTIPROCESSING=0
export HF_TOKEN={}
export PHASED_PROFILING_DIR=gs://wenxindong-cloud-tpu-inference-test/profiling/vllm_qwen3_235b_a22b_instruct_2507/dp1/2
export VLLM_ENGINE_READY_TIMEOUT_S=1800 #30min
export MODEL_IMPL_TYPE=vllm 

JAX_PLATFORMS=proxy,cpu JAX_BACKEND_TARGET=grpc://127.0.0.1:29000 vllm serve gs://wenxindong-cloud-tpu-inference-test/models--Qwen--Qwen3-235B-A22B-Instruct-2507/ \
--tensor-parallel-size 8 \
--data-parallel-size 2 \
--max-model-len 5120 \
--gpu-memory-utilization 0.95 \
--no-enable-prefix-caching \
--max-num-seqs 128 \
--max-num-batched-tokens 4096 \
--enable-expert-parallel \
--load-format runai_streamer

# xpk workload delete --workload wenxindongtest --cluster wenxindong-pw-tpu7x-16  --zone=us-central1-c   --project=cloud-tpu-inference-test

# xpk workload create-pathways   --workload wenxindongtest   --base-docker-image vllm/vllm-tpu:nightly  --script-dir ../tpu-inference   --cluster wenxindong-pw-tpu7x-16   --tpu-type=tpu7x-16   --num-slices=1   --zone=us-central1-c   --project=cloud-tpu-inference-test --priority=very-high --colocated-python-sidecar-image=us-east5-docker.pkg.dev/cloud-tpu-inference-test/wenxindong/colocated_python:latest --command "bash run_pathways.sh"