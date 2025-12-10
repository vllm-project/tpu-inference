# SPDX-License-Identifier: Apache-2.0
import time
from collections import defaultdict
from collections.abc import Sequence
from functools import wraps
from typing import Any, Callable, List, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from jax._src import dtypes
from jax._src import mesh as mesh_lib
from jax._src import xla_bridge as xb
from jax._src.lib import xla_client as xc
from jax.sharding import Mesh, NamedSharding, PartitionSpec
from vllm import envs as vllm_envs
from vllm import utils

from tpu_inference import envs
from tpu_inference.logger import init_logger

GBYTES = 1024 * 1024 * 1024
TPU_HEAD_SIZE_ALIGNMENT = 128
TPU_SECOND_LAST_MINOR = 8

# This is used to translate from a string name for a dtype
# to formal jax.numpy DType.  One use case for this is
# converting the `--kv_cache_dtype` flag to a dtype.
TPU_STR_DTYPE_TO_JAX_DTYPE = {
    "bfloat16": jnp.bfloat16,
    "fp8": jnp.float8_e4m3fn,
    "fp8_e4m3": jnp.float8_e4m3,
    "fp8_e5m2": jnp.float8_e5m2,
    "int8": jnp.int8,
}

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


def get_padded_head_dim(head_dim: int) -> int:
    """Pads head_dim up to the nearest multiple of 128 for kernel performance."""
    # When head_dim == 64, we use kernel specificly optimized for it which does
    # not require any padding.
    if head_dim == 64:
        return 64
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
    return jax.make_array_from_process_local_data(sharding, args)
    # return jax.device_put(*args, device=sharding, **kwargs)


def get_hash_fn_by_name(hash_fn_name: str) -> Callable[[Any], bytes]:
    """
    A wrapper function of vllm.utils.get_hash_fn_by_name to support builtin
    """
    if hash_fn_name == "builtin":
        return hash
    return utils.get_hash_fn_by_name(hash_fn_name)


def quantize_kv(key: jax.Array, value: jax.Array,
                kv_cache_quantized_dtype: jnp.dtype, k_scale: float,
                v_scale: float) -> Tuple[jax.Array, jax.Array]:
    """
        Quantize the key and value tensors.

        Args:
            key: The key tensor to quantize.
            value: The value tensor to quantize.
            kv_cache_quantized_dtype: The dtype to quantize the key and value tensors to.
            q_scale: The scale to quantize the key and value tensors by.
            k_scale: The scale to quantize the key tensor by.
            v_scale: The scale to quantize the value tensor by.

        Returns:
            Tuple[jax.Array, jax.Array]: The quantized key and value tensors.
        """
    dtype_info = jnp.finfo(kv_cache_quantized_dtype)
    minval, maxval = float(dtype_info.min), float(dtype_info.max)
    key = key.astype(jnp.float32) / k_scale
    key = jnp.clip(key, minval, maxval)
    key = key.astype(kv_cache_quantized_dtype)
    value = value.astype(jnp.float32) / v_scale
    value = jnp.clip(value, minval, maxval)
    value = value.astype(kv_cache_quantized_dtype)

    return key, value


def get_jax_dtype_from_str_dtype(str_dtype: str) -> jnp.dtype:
    """
    Get the JAX dtype from a string dtype.

    Args:
        str_dtype: The string dtype to get the JAX dtype from.

    Returns:
        jnp.dtype: The JAX dtype.
    """
    str_dtype = str_dtype.lower().strip()
    return TPU_STR_DTYPE_TO_JAX_DTYPE.get(str_dtype)


def time_function(func):
    """
    A decorator to measure the execution time of a function.
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        execution_time = end_time - start_time
        logger.debug(
            f"Function '{func.__name__}' executed in {execution_time:.4f} seconds."
        )
        return result

    return wrapper
