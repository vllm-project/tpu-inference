#!/bin/bash
# Copyright 2025 Google LLC
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


# A self-contained script to deploy and run a Ray cluster, handling separate
# external (SSH) and internal (Ray cluster) IP addresses.

# --- Script Logic ---

# Function to display usage instructions
usage() {
    echo "Usage: $0 [options]"
    echo ""
    echo "Required Arguments:"
    echo "  -s <script>          Local path to the 'run_cluster.sh' script to deploy."
    echo "  -d <image>           Docker image to use."
    echo "  -c <cache>           Absolute path to the Hugging Face cache on remote hosts."
    echo "  -t <token>           Your Hugging Face token."
    echo "  -H <ssh_head_ip>     The PUBLIC/SSH IP of the head node."
    echo "  -i <internal_head_ip> The PRIVATE/INTERNAL IP of the head node for the Ray cluster."
    echo "  -W <ssh_worker_ips>  A comma-separated list of PUBLIC/SSH IPs for worker nodes."
    exit 1
}

# Generic cleanup function to stop and remove the container on specified hosts
cleanup() {
    local hosts_to_clean=("$@")
    if [ ${#hosts_to_clean[@]} -eq 0 ]; then
        return
    fi
    echo "🧹 Cleaning up containers on specified hosts..."
    for host in "${hosts_to_clean[@]}"; do
        echo "   -> Cleaning ${host}"
        ssh -o StrictHostKeyChecking=no -o BatchMode=yes "${SSH_USER}@${host}" "sudo docker stop node" > /dev/null 2>&1 || true
        ssh -o StrictHostKeyChecking=no -o BatchMode=yes "${SSH_USER}@${host}" "sudo docker rm -f node" > /dev/null 2>&1 || true
        ssh -o StrictHostKeyChecking=no -o BatchMode=yes "${SSH_USER}@${host}" "sudo docker rmi ${DOCKER_IMAGE}" > /dev/null 2>&1 || true
    done
}

# --- Initial Checks & Argument Parsing ---

if ! command -v ssh &> /dev/null || ! command -v scp &> /dev/null; then
    echo "❌ Error: 'ssh' and 'scp' commands must be installed."
    exit 1
fi

# Set default empty values
SCRIPT_PATH=""
DOCKER_IMAGE=""
HF_CACHE_PATH=""
HF_TOKEN=""
HEAD_SSH_IP=""
HEAD_INTERNAL_IP=""
WORKER_SSH_IP_STRING=""
SSH_USER=""

while getopts ":s:d:c:t:H:i:W:" opt; do
  case ${opt} in
    s ) SCRIPT_PATH=$OPTARG;;
    d ) DOCKER_IMAGE=$OPTARG;;
    c ) HF_CACHE_PATH=$OPTARG;;
    t ) HF_TOKEN=$OPTARG;;
    H ) HEAD_SSH_IP=$OPTARG;;
    i ) HEAD_INTERNAL_IP=$OPTARG;;
    W ) WORKER_SSH_IP_STRING=$OPTARG;;
    \? ) echo "Invalid option: $OPTARG" 1>&2; usage;;
    : ) echo "Invalid option: $OPTARG requires an argument" 1>&2; usage;;
  esac
done

if [ -z "${SCRIPT_PATH}" ] || [ -z "${DOCKER_IMAGE}" ] || [ -z "${HF_CACHE_PATH}" ] || [ -z "${HF_TOKEN}" ] || [ -z "${HEAD_SSH_IP}" ] || [ -z "${HEAD_INTERNAL_IP}" ] || [ -z "${WORKER_SSH_IP_STRING}" ]; then
    echo "❌ Error: Missing one or more required arguments."
    usage
fi
if [ ! -f "${SCRIPT_PATH}" ]; then
    echo "❌ Error: Script file not found at '${SCRIPT_PATH}'"
    exit 1
fi

# Convert comma-separated worker string to a bash array for SSH access
IFS=',' read -r -a WORKER_SSH_IPS <<< "$WORKER_SSH_IP_STRING"
# Create a list of all SSH IPs for cleanup and script distribution
ALL_SSH_IPS=("${HEAD_SSH_IP}" "${WORKER_SSH_IPS[@]}")

if [ -z "${SSH_USER}" ]; then
    printf "Enter the SSH username for all hosts: "
    read -r SSH_USER
fi

# --- STEP 1: Pre-Deployment Cleanup ---
read -p "❓ Do you want to clean up any old containers on all hosts before deployment? (y/n) " -n 1 -r
echo "" # Move to a new line
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "🧼 Performing pre-deployment cleanup..."
    cleanup "${ALL_SSH_IPS[@]}"
    echo "✅ Pre-deployment cleanup complete."
    echo "---"
fi

# --- STEP 2: Distribute the script ---
echo "📦 Distributing '${SCRIPT_PATH}' to all hosts..."
REMOTE_HOME_DIR="~"
REMOTE_SCRIPT_DIR="${REMOTE_HOME_DIR}/tpu-inference/scripts/multihost"
REMOTE_SCRIPT_PATH="${REMOTE_SCRIPT_DIR}/run_cluster.sh"
SSH_OPTIONS="-o StrictHostKeyChecking=no -o BatchMode=yes"

for host_ssh_ip in "${ALL_SSH_IPS[@]}"; do
    echo "   -> Copying to ${host_ssh_ip}"
    # shellcheck disable=SC2086,SC2029
    if ! ssh $SSH_OPTIONS "${SSH_USER}@${host_ssh_ip}" "mkdir -p ${REMOTE_SCRIPT_DIR}" || \
       ! scp $SSH_OPTIONS "${SCRIPT_PATH}" "${SSH_USER}@${host_ssh_ip}:${REMOTE_SCRIPT_PATH}"; then
        echo "❌ Failed to copy script to ${host_ssh_ip}. Aborting."
        exit 1
    fi
done
echo "✅ Script distributed successfully."
echo "---"

# --- STEP 3: Execute the script on each host ---
echo "🚀 Starting cluster setup..."
echo "Head Node SSH IP: ${HEAD_SSH_IP}"
echo "Head Node Internal IP: ${HEAD_INTERNAL_IP}"
echo "Worker Node SSH IPs: ${WORKER_SSH_IPS[*]}"
echo "---------------------------------"

# Note the use of HEAD_INTERNAL_IP here. This is passed to the script for Ray.
BASE_CMD="sudo bash ${REMOTE_SCRIPT_PATH} \
    '${DOCKER_IMAGE}' \
    '${HEAD_INTERNAL_IP}' \
    %ROLE% \
    '${HF_CACHE_PATH}' \
    -e HF_TOKEN='${HF_TOKEN}' \
    -e TPU_MULTIHOST_BACKEND=ray \
    -e JAX_PLATFORMS=''"

# Head Node (connects via SSH IP, runs command with internal IP)
echo "⚙️  Configuring head node (Connecting to ${HEAD_SSH_IP})..."
# shellcheck disable=SC2086,SC2029
HEAD_CMD="${BASE_CMD//%ROLE%/--head}"
# shellcheck disable=SC2086,SC2029
ssh $SSH_OPTIONS "${SSH_USER}@${HEAD_SSH_IP}" "${HEAD_CMD}" &

# Worker Nodes (connect via SSH IPs, run command with internal head IP)
for worker_ssh_ip in "${WORKER_SSH_IPS[@]}"; do
    echo "⚙️  Configuring worker node (Connecting to ${worker_ssh_ip})..."
    # shellcheck disable=SC2086,SC2029
    WORKER_CMD="${BASE_CMD//%ROLE%/--worker}"
    # shellcheck disable=SC2086,SC2029
    ssh $SSH_OPTIONS "${SSH_USER}@${worker_ssh_ip}" "${WORKER_CMD}" &
done
