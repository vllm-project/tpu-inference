# SPDX-License-Identifier: Apache-2.0
from typing import List, Tuple

import jax
import numpy as np

GBYTES = 1024 * 1024 * 1024


def get_num_kv_heads_by_tp(num_kv_heads: int, tp_size: int) -> int:
    if tp_size <= num_kv_heads:
        assert num_kv_heads % tp_size == 0
        return num_kv_heads
    else:
        assert tp_size % num_kv_heads == 0
        return tp_size


def hbm_usage_bytes(devices: jax.Device) -> List[Tuple[int, int]]:
    usage = []
    for device in devices:
        hbm_used = device.memory_stats()["bytes_in_use"]
        hbm_limit = device.memory_stats()["bytes_limit"]
        usage.append((hbm_used, hbm_limit))
    return usage


def hbm_usage_gb(devices: List[jax.Device]) -> List[Tuple[float, float]]:
    usage = hbm_usage_bytes(devices)
    usage = [(round(used / GBYTES, 2), round(limit / GBYTES, 2))
             for used, limit in usage]
    return usage


def init_random(seed: int) -> jax.Array:
    np.random.seed(seed)
    key = jax.random.PRNGKey(seed)
    return key


def array_info(name: str, x: jax.Array) -> str:
    rep = f"{name} | shape={x.shape} | dtype={x.dtype} | sharding={x.sharding}"
    return rep
