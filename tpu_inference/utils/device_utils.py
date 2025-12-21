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

from collections import defaultdict
from collections.abc import Sequence
from typing import Any, List, Tuple

import jax
import numpy as np
from jax._src import mesh as mesh_lib
from jax._src import xla_bridge as xb
from jax._src.lib import xla_client as xc
from jax.sharding import Mesh, NamedSharding, PartitionSpec
from vllm import envs as vllm_envs

from tpu_inference import envs
from tpu_inference.logger import init_logger

GBYTES = 1024 * 1024 * 1024
TPU_HEAD_SIZE_ALIGNMENT = 128
TPU_SECOND_LAST_MINOR = 8

_megacore = False
logger = init_logger(__name__)


def enable_megacore() -> None:
    global _megacore
    _megacore = True


def get_megacore() -> bool:
    return _megacore


def hbm_usage_bytes(devices: Any) -> List[Tuple[int, int]]:
    usage = []
    if vllm_envs.VLLM_TPU_USING_PATHWAYS:
        return pathways_hbm_usage_gb(devices)

    multihost_backend = envs.TPU_MULTIHOST_BACKEND
    if multihost_backend == "ray":
        # MemoryStats is only supported for addressable PjRt devices.
        # Assume all the devices have similar memory usage for now.
        # TODO(ranlihao): find a proper way to get the memory usage of each device.
        for device in devices:
            try:
                hbm_used = device.memory_stats()["bytes_in_use"]
                hbm_limit = device.memory_stats()["bytes_limit"]
                logger.info(
                    "Get memory stats for device %s. Assuming all devices have the same usage.",
                    device)
                usage.extend([(hbm_used, hbm_limit)] * len(devices))
                break
            except Exception as e:
                logger.warning(
                    "Failed to get memory stats for device %s: %s. ", device,
                    e)
    else:
        for device in devices:
            hbm_used = device.memory_stats()["bytes_in_use"]
            hbm_limit = device.memory_stats()["bytes_limit"]
            usage.append((hbm_used, hbm_limit))

    return usage


def get_device_name(num_devices: int | None = None):
    kind = jax.devices()[0].device_kind
    if 'TPU' not in kind:
        raise RuntimeError('Expected TPU devices')
    suffix = ''
    if kind.endswith(' lite'):
        kind = kind[:-len(' lite')]
        suffix = 'e'
    elif kind.endswith('e'):
        kind = kind[:-1]
        suffix = 'e'
    elif kind.endswith('p'):
        kind = kind[:-1]
        suffix = 'p'
    elif kind == 'TPU7x':
        kind = 'TPU v7'
    assert kind[:-1] == 'TPU v', kind
    kind += suffix
    if num_devices is not None:
        kind += f'-{num_devices}'
    return kind


def get_tpu_version() -> int:
    """Returns the numeric version of the TPU, or -1 if not on TPU."""
    kind = jax.devices()[0].device_kind
    if 'TPU' not in kind:
        return -1
    if kind.endswith(' lite'):
        kind = kind[:-len(' lite')]
    if kind.endswith('p') or kind.endswith('e'):
        kind = kind[:-1]
    if kind == 'TPU7x':
        return 7
    assert kind[:-1] == 'TPU v', kind
    return int(kind[-1])


def get_device_hbm_limit() -> int:

    device_kind = get_device_name()
    if device_kind == "TPU v5p" or device_kind == "TPU v5":
        return 95 * GBYTES
    elif device_kind == "TPU v5e":
        return 16 * GBYTES
    elif device_kind == "TPU v6e" or device_kind == "TPU v4":
        return 32 * GBYTES
    elif device_kind == "TPU v7":
        # 192 * GBYTES / 2 because each JAX device (v7x core) has
        # 1/2 of the total chip HBM
        return 96 * GBYTES
    else:
        raise ValueError(f"Unknown device kind: {device_kind}")


def pathways_hbm_usage_gb(devices: Any) -> List[Tuple[float, float]]:
    live_arrays = jax.live_arrays()
    hbm_used = defaultdict(int)
    hbm_limit = get_device_hbm_limit()
    for array in live_arrays:
        for buffer in array.addressable_shards:
            hbm_used[buffer.data.device] += buffer.data.nbytes
    return [(hbm_used[device], hbm_limit) for device in devices]


def hbm_usage_gb(devices: Any) -> List[Tuple[float, float]]:
    usage = hbm_usage_bytes(devices)
    usage = [(round(used / GBYTES, 2), round(limit / GBYTES, 2))
             for used, limit in usage]
    return usage


def device_array(mesh: Mesh, *args, sharding=None, **kwargs) -> jax.Array:
    """
    Create a device array with the specified mesh and sharding.

    Args:
        mesh: The JAX mesh to use for device placement
        *args: Positional arguments to pass to jax.device_put
        sharding: Optional sharding specification. If None, uses PartitionSpec(None)
        **kwargs: Keyword arguments to pass to jax.device_put

    Returns:
        A JAX array placed on the specified devices
    """
    if sharding is None:
        sharding = NamedSharding(mesh, PartitionSpec(None))
    return jax.device_put(*args, device=sharding, **kwargs)


def make_optimized_mesh(axis_shapes: Sequence[int],
                        axis_names: Sequence[str],
                        *,
                        devices: Sequence[xc.Device] | None = None):
    if devices is None:
        devices = xb.devices()
    # Sort the devices in case it's passed in an arbitary order
    devices = sorted(devices, key=lambda x: x.coords)

    def _is_1D(axis_shapes):
        return sum(x > 1 for x in axis_shapes) == 1

    if _is_1D(axis_shapes):
        dev_kind = devices[0].device_kind
        device_num = len(devices)
        if dev_kind == "TPU v6 lite":
            ordered_devices = None
            # NOTE(chengjiyao):
            # The coords of v6e-8 are
            # (0,0,0)
            # (1,0,0)
            # (0,1,0)
            # (1,1,0)
            # (0,2,0)
            # (1,2,0)
            # (0,3,0)
            # (1,3,0)
            if device_num == 8:
                ordered_devices = np.array([
                    devices[0],
                    devices[1],
                    devices[2],
                    devices[3],
                    devices[7],
                    devices[6],
                    devices[5],
                    devices[4],
                ])
            # NOTE(chengjiyao):
            # The coords of v6e-4 are
            # (0,0,0)
            # (1,0,0)
            # (0,1,0)
            # (1,1,0)
            elif device_num == 4:
                ordered_devices = np.array([
                    devices[0],
                    devices[1],
                    devices[3],
                    devices[2],
                ])
            if ordered_devices is not None:
                ordered_devices = np.array(ordered_devices)
                ordered_devices = ordered_devices.reshape(axis_shapes)
                mesh = mesh_lib.Mesh(ordered_devices, axis_names)
                logger.info("Use customized mesh: %s", mesh)
                return mesh

    return jax.make_mesh(axis_shapes, axis_names, devices=devices)
