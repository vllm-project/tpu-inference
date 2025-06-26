# SPDX-License-Identifier: Apache-2.0
from typing import Any, List, Tuple

import jax
from ray._private.accelerators import TPUAcceleratorManager

from tpu_commons.core.jetstream_commons.engine import PATHWAYS_ENABLED

GBYTES = 1024 * 1024 * 1024

_megacore = False


def enable_megacore() -> None:
    global _megacore
    _megacore = True


def get_megacore() -> bool:
    return _megacore


def get_local_available_devices() -> int:
    return TPUAcceleratorManager.get_current_node_num_accelerators()


def set_visible_device_ids(tpu_ids: List[int]) -> None:
    validate = TPUAcceleratorManager.validate_resource_request_quantity(
        len(tpu_ids))
    if not validate[0]:
        raise ValueError(validate[1])
    tpu_ids = [str(tpu_id) for tpu_id in tpu_ids]
    TPUAcceleratorManager.set_current_process_visible_accelerator_ids(tpu_ids)


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
            print(hbm_used, hbm_limit)
            usage.append((hbm_used, hbm_limit))

    return usage


def hbm_usage_gb(devices: Any) -> List[Tuple[float, float]]:
    usage = hbm_usage_bytes(devices)
    usage = [(round(used / GBYTES, 2), round(limit / GBYTES, 2))
             for used, limit in usage]
    return usage


def array_info(name: str, x: jax.Array) -> str:
    rep = f"{name} | shape={x.shape} | dtype={x.dtype} | sharding={x.sharding}"
    return rep
