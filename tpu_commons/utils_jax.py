# SPDX-License-Identifier: Apache-2.0
from typing import Any, List, Tuple

from tpu_commons.core import PATHWAYS_ENABLED
from tpu_commons.logger import init_logger

logger = init_logger(__name__)

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
    for device in devices:
        if PATHWAYS_ENABLED:
            # The Pathways backend doesn't support memory_stats().
            # TODO(fhzhang): find the proper way to support this.
            usage.append((32384, 33550237184))
        else:
            hbm_used = device.memory_stats()["bytes_in_use"]
            hbm_limit = device.memory_stats()["bytes_limit"]
            usage.append((hbm_used, hbm_limit))

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
