#!/bin/bash
#
# Launch a Ray cluster inside Docker for uLLM inference.
#
# This script can start either a head node or a worker node, depending on the
# --head or --worker flag provided as the third positional argument.
#
# Usage:
# 1. Designate one machine as the head node and execute:
#    sudo bash run_cluster.sh \
#         <docker_image> \
#         <head_node_ip> \
#         --head \
#         /abs/path/to/huggingface/cache \
#         -e HF_TOKEN=<your_hf_token> \
#         -e TPU_MULTIHOST_BACKEND=ray
#         -e JAX_PLATFORMS=''

# 2. On every worker machine, execute:
#    sudo bash run_cluster.sh \
#         <docker_image> \
#         <head_node_ip> \
#         --worker \
#         /abs/path/to/huggingface/cache \
#         -e HF_TOKEN=<your_hf_token> \
#         -e TPU_MULTIHOST_BACKEND=ray
#         -e JAX_PLATFORMS=''
#
# Keep each terminal session open. Closing a session stops the associated Ray
# node and thereby shuts down the entire cluster.
# Every machine must be reachable at the supplied IP address.
#
# The container is named "node". To open a shell inside
# a container after launch, use:
#       sudo docker exec -it node /bin/bash
#
# Then, you can execute uLLM commands on the Ray cluster as if it were a
# single machine, e.g. python /workspace/tpu_inference/examples/offline_inference.py  --model=meta-llama/Llama-3.1-8B  --tensor_parallel_size=16  --task=generate  --max_model_len=1024
#
# To stop the container, use:
#       docker stop node
#       docker rm node
#

# Check for minimum number of required arguments.
if [ $# -lt 4 ]; then
    echo "Usage: $0 docker_image head_node_ip --head|--worker path_to_hf_home [additional_args...]"
    exit 1
fi

# Extract the mandatory positional arguments and remove them from $@.
DOCKER_IMAGE="$1"
HEAD_NODE_ADDRESS="$2"
NODE_TYPE="$3"  # Should be --head or --worker.
PATH_TO_HF_HOME="$4"
shift 4

# Preserve any extra arguments so they can be forwarded to Docker.
ADDITIONAL_ARGS=("$@")

# Validate the NODE_TYPE argument.
if [ "${NODE_TYPE}" != "--head" ] && [ "${NODE_TYPE}" != "--worker" ]; then
    echo "Error: Node type must be --head or --worker"
    exit 1
fi

# Set up Docker authentication for Google Container Registry.
# Modify the hostname to accomodate your specific docker region.
gcloud auth configure-docker us-east5-docker.pkg.dev

CONTAINER_NAME="node"

# Define a cleanup routine that removes the container when the script exits.
# This prevents orphaned containers from accumulating if the script is interrupted.
cleanup() {
    docker stop "${CONTAINER_NAME}"
    docker rm "${CONTAINER_NAME}"
}
trap cleanup EXIT

# Build the Ray start command based on the node role.
# The head node manages the cluster and accepts connections on port 6379,
# while workers connect to the head's address.
RAY_START_CMD="ray start --block"
if [ "${NODE_TYPE}" == "--head" ]; then
    RAY_START_CMD+=" --head --port=6379"
else
    RAY_START_CMD+=" --address=${HEAD_NODE_ADDRESS}:6379"
fi

# Launch the container with the assembled parameters.
# --privileged: Grants extended privileges to the container for TPU exposure
# --network host: Allows Ray nodes to communicate directly via host networking
# --shm-size=16G: Increases shared memory
# -v HF_HOME: Mounts HuggingFace cache to avoid re-downloading models
docker run \
    --privileged \
    --entrypoint /bin/bash \
    --network host \
    --shm-size=16G \
    --name "${CONTAINER_NAME}" \
    -v "${PATH_TO_HF_HOME}:/root/.cache/huggingface" \
    "${ADDITIONAL_ARGS[@]}" \
    "${DOCKER_IMAGE}" -c "${RAY_START_CMD}"
