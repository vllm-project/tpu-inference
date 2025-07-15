# SPDX-License-Identifier: Apache-2.0
from typing import Any, List, Tuple

from tpu_commons.core import PATHWAYS_ENABLED

GBYTES = 1024 * 1024 * 1024

_megacore = False


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
    print(f"{devices=}")
    for device in devices:
        if PATHWAYS_ENABLED:
            # The Pathways backend doesn't support memory_stats().
            # TODO(fhzhang): find the proper way to support this.
            usage.append((32384, 33550237184))
        else:
            try:
                hbm_used = device.memory_stats()["bytes_in_use"]
                hbm_limit = device.memory_stats()["bytes_limit"]
                usage.append((hbm_used, hbm_limit))
                print(f"Device {device} HBM usage: {hbm_used / GBYTES:.2f} / {hbm_limit / GBYTES:.2f} GB used, ")
            except (RuntimeError, KeyError, TypeError) as ex:
                print(f"\tMemstats unavailable, error: {ex}")
                usage.append((0, 33550237184))

    return usage


def hbm_usage_gb(devices: Any) -> List[Tuple[float, float]]:
    usage = hbm_usage_bytes(devices)
    usage = [(round(used / GBYTES, 2), round(limit / GBYTES, 2))
             for used, limit in usage]
    return usage


def get_padded_head_dim(head_dim: int) -> int:
    """Pads head_dim up to the nearest multiple of 128 for kernel performance."""
    # Details can be seen at: tpu_commons/kernels/ragged_kv_cache_update.py::_kv_cache_update()
    return (head_dim + 127) // 128 * 128


def get_padded_num_heads(num_heads: int, sharding_size: int) -> int:
    if num_heads >= sharding_size:
        assert num_heads % sharding_size == 0
    else:
        assert sharding_size % num_heads == 0
        num_heads = sharding_size
    return num_heads
