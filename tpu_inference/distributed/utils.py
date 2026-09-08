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
from typing import Optional

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


def get_kv_controller_address() -> Optional[str]:
    """Address ('host:port') of the Raiden controller, or None.

    When set, the connector registers each local KV shard as a Raiden work unit
    and routes transfers through the controller's byte-span reshard planner,
    which tolerates a different TP degree and page size on the two sides. When
    unset, the connector keeps using the index-matched symmetric pull.
    """
    address = os.getenv("TPU_KV_CONTROLLER_ADDRESS", "").strip()
    return address or None


def get_kv_controller_port() -> int:
    """Port the prefill host's controller sidecar binds. 0 selects an ephemeral port."""
    return int(os.getenv("TPU_KV_CONTROLLER_PORT", "9700"))


def get_raiden_listener_port() -> int:
    """Base port for the per-shard Raiden KVCacheListener control-plane sockets.

    Shard i binds `base + i`; the controller uses these to arm a receiver
    (PoolReshardRegisterRecv) and to fire a sender (PoolReshardPush). Distinct
    from `get_kv_transfer_port`, which is the data plane.
    """
    return int(os.getenv("TPU_KV_LISTENER_PORT", "9800"))


def get_kv_decode_tp_size(kv_transfer_config=None) -> int:
    """Number of KV shards on the decode side, or 0 to assume it matches ours.

    The prefill scheduler needs the decode fan-out width to mint one request
    id per destination rank, and each prefill worker needs it to compute its
    shard's schedule key -- both before any handshake exists to carry it. It is
    a deployment constant of a P/D pair, and the launcher already knows both TP
    sizes. 0 (the default) means symmetric, which is the only case where
    guessing is safe.

    Read from ``kv_connector_extra_config["decode_tp_size"]`` so it rides in
    ``--kv-transfer-config`` alongside every other connector option, the way
    connectors on other backends take their settings. ``TPU_KV_DECODE_TP_SIZE``
    remains as a fallback for launchers that set it in the environment.
    """
    if kv_transfer_config is not None:
        declared = kv_transfer_config.get_from_extra_config(
            "decode_tp_size", None)
        if declared is not None:
            return int(declared)
    # An empty value is what a launcher produces from an unset shell variable;
    # treat it as "not declared" rather than crashing the engine at init.
    return int(os.getenv("TPU_KV_DECODE_TP_SIZE", "").strip() or 0)


# Byte geometry of this process's local KV shards, published by the worker
# connector at init and read by the scheduler connector when it builds the
# handshake. Same bridging pattern as _NODES_KV_IP_PORT above: identical
# process for single-host, collected over collective_rpc by the Ray executor
# for multi-host.
_LOCAL_KV_GEOMETRY: Optional[dict] = None


def set_local_kv_geometry(geometry: dict) -> None:
    global _LOCAL_KV_GEOMETRY
    _LOCAL_KV_GEOMETRY = geometry


def get_local_kv_geometry() -> Optional[dict]:
    return _LOCAL_KV_GEOMETRY


def get_transfer_channel_number() -> str:
    n = os.getenv("TPU_KV_TRANSFER_CHANNEL_NUMBER", "8")
    return int(n)


def get_enable_d2h_transfer() -> bool:
    """Check if device-to-host transfer is enabled via environment variable."""
    enable_str = os.getenv("TPU_ENABLE_D2H_TRANSFER", "0").lower()
    return enable_str in ("true", "1", "yes")


def get_enable_block_kv_transfer() -> bool:
    """Check if we block the KV-cache transfer until it is ready via environment variable."""
    enable_str = os.getenv("TPU_ENABLE_BLOCK_KV_TRANSFER", "true").lower()
    return enable_str in ("true", "1", "yes")


def get_p2p_wait_pull_timeout() -> int:
    """KV-cache transfer timeout in seconds."""
    timeout_str = os.getenv("TPU_P2P_WAIT_PULL_TIMEOUT", "180")
    return int(timeout_str)


def get_max_host_kv_buffer_size() -> int:
    """Maximum size of KV requests that can be handled by the host KV pool."""
    size_str = os.getenv("TPU_MAX_HOST_KV_BUFFER_SIZE", "64")
    return int(size_str)


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

    if not all(hasattr(d, "coords") for d in local_devices):
        logger.error(
            f"Expect TPU device but got {[type(d) for d in local_devices]}")

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
