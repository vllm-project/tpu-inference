import os

from sortedcontainers import SortedDict
from vllm.utils.network_utils import get_ip

from tpu_inference import envs
from tpu_inference.logger import init_logger

logger = init_logger(__name__)

# For multi-host usage only, to collect IP, port, and local devices for all nodes.
# This dictionary should always be sorted by the device coordinates as index.
_NODES_METADATA = SortedDict()


def set_node_metadata(metadata: tuple[int, str, int, str]):
    global _NODES_METADATA
    node_id, ip, port, devices = metadata
    _NODES_METADATA[devices] = (ip, port, node_id)


def get_kv_ips() -> str:
    if envs.TPU_MULTIHOST_BACKEND == "ray":
        ips = []
        # IPs are sorted by device index
        for _, metadata in _NODES_METADATA.items():
            ips.append(metadata[0])
        return ips
    else:
        return get_host_ip()


def get_kv_ports() -> str:
    if envs.TPU_MULTIHOST_BACKEND == "ray":
        ports = []
        # Ports are sorted by device index
        for _, metadata in _NODES_METADATA.items():
            ports.append(metadata[1])
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


def get_node_id() -> int:
    # TODO(xiang): Is it possible to get this from a pre-defiend env?
    id = os.getenv("TPU_NODE_ID", 0)
    return int(id)


def get_topology_node_id() -> int:
    # Return the topology-ordered index of the node (not the node id set from
    # the environment).
    device_id = 0
    current_node_id = os.getenv("TPU_NODE_ID", 0)
    for _, metadata in _NODES_METADATA.items():
        if metadata[2] == int(current_node_id):
            return device_id
        device_id += 1
    return device_id
