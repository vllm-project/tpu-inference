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


def get_node_id() -> int:
    # TODO(xiang): Is it possible to get this from a pre-defiend env?
    id = os.getenv("TPU_NODE_ID", 0)
    return int(id)
