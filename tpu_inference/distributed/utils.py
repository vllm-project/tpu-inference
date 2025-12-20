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

import os

from vllm.utils.network_utils import get_ip

from tpu_inference import envs
from tpu_inference.logger import init_logger

logger = init_logger(__name__)

# For multi-host usage only, to collect IP and port for all nodes.
_NODES_KV_IP_PORT = dict()


def set_node_kv_ip_port(ip_port: tuple[int, str, int]):
    global _NODES_KV_IP_PORT
    node_id, ip, port = ip_port
    _NODES_KV_IP_PORT[node_id] = (ip, port)


def get_kv_ips() -> str:
    if envs.TPU_MULTIHOST_BACKEND == "ray":
        num_nodes = len(_NODES_KV_IP_PORT)
        ips = []
        for node_id in range(num_nodes):
            ips.append(_NODES_KV_IP_PORT[node_id][0])
        return ips
    else:
        return get_host_ip()


def get_kv_ports() -> str:
    if envs.TPU_MULTIHOST_BACKEND == "ray":
        num_nodes = len(_NODES_KV_IP_PORT)
        ports = []
        for node_id in range(num_nodes):
            ports.append(_NODES_KV_IP_PORT[node_id][1])
        return ports
    else:
        return get_kv_transfer_port()


def get_host_ip() -> str:
    """Use `VLLM_HOST_IP` if set, otherwise use default network interface IP."""
    return get_ip()


def get_kv_transfer_port() -> str:
    port = os.getenv("TPU_KV_TRANSFER_PORT", "9100")
    return port


def get_side_channel_port() -> str:
    port = os.getenv("TPU_SIDE_CHANNEL_PORT", "9600")
    return port


def get_device_topology_order_id(local_devices, global_devices) -> int:
    """
    Calculates the topology order ID for the local device set within the global topology.

    This function determines the rank of the current host/process based on the
    coordinate of its TPU devices relative to all devices in the topology.

    Args:
        local_devices: A list of TpuDevice objects available to the current process.
        global_devices: A list of all TpuDevice objects in the global topology.

    Returns:
        The topology order ID (rank) of the local devices.
    """
    if not local_devices:
        raise ValueError("local_devices cannot be empty")
    if not global_devices:
        raise ValueError("global_devices cannot be empty")

    # 1. Find the 'anchor' (minimum coordinate) for the local devices.
    #    This represents the physical top-left corner of the local machine.
    local_anchor = min(d.coords for d in local_devices)

    # 2. Group global devices by process to find the anchor for EVERY process.
    process_anchors = {}
    for d in global_devices:
        pid = d.process_index
        # Update the minimum coordinate found for this process so far
        if pid not in process_anchors or d.coords < process_anchors[pid]:
            process_anchors[pid] = d.coords

    # 3. Sort the unique anchors to establish the canonical topology order.
    #    Tuples (x, y, z) sort lexicographically (x first, then y, then z).
    sorted_anchors = sorted(process_anchors.values())

    # 4. Return the index (rank) of the local anchor in the sorted list.
    try:
        return sorted_anchors.index(local_anchor)
    except ValueError:
        raise ValueError(
            f"Local devices: {local_devices} do not exist in the global device: {global_devices} list."
        )
