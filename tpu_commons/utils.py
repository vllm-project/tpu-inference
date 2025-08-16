# SPDX-License-Identifier: Apache-2.0
import os
from collections import defaultdict
from typing import Any, List, Tuple

import jax
from jax._src import dtypes
from vllm import envs

from tpu_commons.logger import init_logger

GBYTES = 1024 * 1024 * 1024
TPU_HEAD_SIZE_ALIGNMENT = 128

_megacore = False
logger = init_logger(__name__)


def enable_megacore() -> None:
    global _megacore
    _megacore = True


def get_megacore() -> bool:
    return _megacore


def get_num_kv_heads_by_tp(num_kv_heads: int, tp_size: int) -> int:
    if tp_size <= num_kv_heads:
        assert num_kv_heads % tp_size == 0
        return num_kv_heads
    else:
        assert tp_size % num_kv_heads == 0
        return tp_size


def hbm_usage_bytes(devices: Any) -> List[Tuple[int, int]]:
    usage = []
    if envs.VLLM_TPU_USING_PATHWAYS:
        return pathways_hbm_usage_gb(devices)

    multihost_backend = os.environ.get("TPU_MULTIHOST_BACKEND", "").lower()
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


def pathways_hbm_usage_gb(devices: Any) -> List[Tuple[float, float]]:
    live_arrays = jax.live_arrays()
    hbm_used = defaultdict(int)
    # TODO(wenxindong): Find a way to get the accurate hbm limit on Pathways.
    hbm_limit = 33550237184
    for array in live_arrays:
        assert hasattr(array, 'sharding') and hasattr(
            array.sharding, 'device_set'
        ), "This function must not be called within jax tracer (e.g. jit, vmap, grad)"
        for device in array.sharding.device_set:
            hbm_used[device] += array.dtype.itemsize * array.size // len(
                array.sharding.device_set)
    return [(hbm_used[device], hbm_limit) for device in devices]


def hbm_usage_gb(devices: Any) -> List[Tuple[float, float]]:
    usage = hbm_usage_bytes(devices)
    usage = [(round(used / GBYTES, 2), round(limit / GBYTES, 2))
             for used, limit in usage]
    return usage


def get_padded_head_dim(head_dim: int) -> int:
    """Pads head_dim up to the nearest multiple of 128 for kernel performance."""
    return (head_dim + 127) // 128 * 128


def get_padded_num_heads(num_heads: int, sharding_size: int) -> int:
    if num_heads >= sharding_size:
        assert num_heads % sharding_size == 0
    else:
        assert sharding_size % num_heads == 0
        num_heads = sharding_size
    return num_heads


def get_dtype_packing(dtype):
    bits = dtypes.bit_width(dtype)
    return 32 // bits
