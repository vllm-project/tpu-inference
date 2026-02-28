# =============================================================================
# Default Pathways launch script for vLLM TPU inference.
# Copy this file into your script_dir and customize as needed.
#
# Usage:
#   xpk workload create-pathways \
#     --workload <WORKLOAD_NAME> \
#     --base-docker-image vllm/vllm-tpu:latest \
#     --script-dir ../tpu-inference \
#     --cluster <CLUSTER_NAME> \
#     --tpu-type=<TPU_TYPE> \
#     --num-slices=<NUM_SLICES> \
#     --zone=<ZONE> \
#     --project=<PROJECT_ID> \
#     --priority=very-high \
#     --command "bash ./run_pathways.sh"
# =============================================================================

set -euo pipefail

export HF_TOKEN="${HF_TOKEN:-}"

# ---------------------------------------------------------------------------
# Pathways environment variables (required)
# ---------------------------------------------------------------------------
export VLLM_ENABLE_V1_MULTIPROCESSING=0
export JAX_PLATFORMS=proxy,cpu
export JAX_BACKEND_TARGET=grpc://127.0.0.1:29000

# ---------------------------------------------------------------------------
# Multi-slice configuration (optional)
# Uncomment and set NUM_SLICES when running across multiple TPU slices.
# ---------------------------------------------------------------------------
# export NUM_SLICES=2

# ---------------------------------------------------------------------------
# Run your workload
# Replace the command below with your own inference or serving entry point.
# ---------------------------------------------------------------------------
python ./offline_inference.py "$@"

# ---------------------------------------------------------------------------
# Alternative: Run vLLM server for online serving
#
# Uncomment the line below (and comment out the offline_inference line above)
# to start a vLLM server inside the pod instead.
#
#   vllm serve <MODEL_NAME> "$@"
#
# To access the server, forward the port from the pod to your local machine:
#
#   kubectl get pods
#   kubectl port-forward pod/<POD_NAME> 8000:8000
#
# Then send requests to http://localhost:8000, e.g.:
#
#   vllm bench serve --model <MODEL_NAME> --dataset-name random \
#     --random-input-len 512 --random-output-len 256 --num-prompts 16
# ---------------------------------------------------------------------------
