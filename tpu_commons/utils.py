# SPDX-License-Identifier: Apache-2.0
import os
from collections import OrderedDict, defaultdict
from collections.abc import Sequence
from typing import Any, List, Tuple

import jax
import numpy as np
import torch
import torchax
from jax._src import dtypes
from jax._src import mesh as mesh_lib
from jax._src import xla_bridge as xb
from jax._src.lib import xla_client as xc
from jax.sharding import Mesh, NamedSharding, PartitionSpec
from torchax.interop import torch_view
from torchax.ops.mappings import t2j
from vllm import envs

from tpu_commons.logger import init_logger

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


def make_optimized_mesh(axis_shapes: Sequence[int],
                        axis_names: Sequence[str],
                        *,
                        devices: Sequence[xc.Device] | None = None):
    if devices is None:
        devices = xb.devices()

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
                    devices[2],
                    devices[4],
                    devices[6],
                    devices[7],
                    devices[5],
                    devices[3],
                    devices[1],
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
                    devices[2],
                    devices[3],
                    devices[1],
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
    return jax.device_put(*args, device=sharding, **kwargs)


P = PartitionSpec


def shard_and_move_tensor_to_tpu(tensor, mesh):
    if isinstance(tensor, torch.Tensor) and not isinstance(
            tensor, torchax.tensor.Tensor):
        with jax.default_device(jax.devices("tpu")[0]):
            tensor = t2j(tensor, use_dlpack=False)
        return torch_view(tensor).apply_jax_(jax.device_put,
                                             NamedSharding(mesh, P()))
    else:
        assert isinstance(tensor, torchax.tensor.Tensor)
        return tensor.apply_jax_(jax.device_put, NamedSharding(mesh, P()))


def shard_and_move_lora_to_tpu(layer: torch.nn.Module, mesh: Mesh):
    sharded_lora_a_tpu = []
    sharded_lora_b_tpu = []
    sharded_lora_bias_tpu = []

    for i in range(layer.n_slices):
        sharded_lora_a_tpu.append(
            shard_and_move_tensor_to_tpu(layer.lora_a_stacked[i], mesh))
        sharded_lora_b_tpu.append(
            shard_and_move_tensor_to_tpu(layer.lora_b_stacked[i], mesh))
        if layer.lora_bias_stacked is not None:
            sharded_lora_bias_tpu.append(
                shard_and_move_tensor_to_tpu(layer.lora_bias_stacked[i], mesh))

    layer.lora_a_stacked = tuple(sharded_lora_a_tpu)
    layer.lora_b_stacked = tuple(sharded_lora_b_tpu)
    if layer.lora_bias_stacked is not None:
        layer.lora_bias_stacked = tuple(sharded_lora_bias_tpu)


def partition_column_parallel_linear_lora(layer: torch.nn.Module,
                                          mesh: Mesh) -> torch.nn.Module:
    from vllm.lora.layers import MergedColumnParallelLinearWithLoRA
    assert isinstance(layer, MergedColumnParallelLinearWithLoRA)
    shard_and_move_lora_to_tpu(layer, mesh)
    return layer


def partition_qkv_parallel_linear_lora(layer: torch.nn.Module,
                                       mesh: Mesh) -> torch.nn.Module:
    from vllm.lora.layers import MergedQKVParallelLinearWithLoRA
    assert isinstance(layer, MergedQKVParallelLinearWithLoRA)
    shard_and_move_lora_to_tpu(layer, mesh)
    return layer


def partition_row_parallel_linear_lora(layer: torch.nn.Module,
                                       mesh: Mesh) -> torch.nn.Module:
    from vllm.lora.layers import RowParallelLinearWithLoRA
    assert isinstance(layer, RowParallelLinearWithLoRA)
    shard_and_move_lora_to_tpu(layer, mesh)
    return layer


MODULE_TYPE_TO_WRAPPING_FUNC = OrderedDict([
    # We only need to deal with:
    # 'MergedColumnParallelLinearWithLoRA'
    # 'MergedQKVParallelLinearWithLoRA'
    # 'RowParallelLinearWithLoRA'
    ("MergedColumnParallelLinearWithLoRA",
     partition_column_parallel_linear_lora),
    ("MergedQKVParallelLinearWithLoRA", partition_qkv_parallel_linear_lora),
    ("RowParallelLinearWithLoRA", partition_row_parallel_linear_lora),
])


def get_fqn(module):
    # Get the fully qualified name of the module
    return module.__class__.__qualname__


def shard_lora_weights_and_move_to_tpu(model: torch.nn.Module,
                                       mesh: Mesh) -> None:
    """
    Recursively check a PyTorch model and apply appropriate sharding based on
    the MODULE_TYPE_TO_WRAPPING_FUNC mapping.

    Args:
        model: torch.nn.Module to process
        mesh: An SPMD mesh object used for sharding
    """

    def _process_module(module, name=None, parent=None):
        for module_type, wrapping_func in MODULE_TYPE_TO_WRAPPING_FUNC.items():
            if get_fqn(module) == module_type:
                wrapped_module = wrapping_func(module, mesh)

                assert parent is not None and name is not None, (
                    "Top Level module is not expected to be wrapped.")
                if wrapped_module is not module:
                    # Wrapped module and module are different py object.
                    # The original module should be replaced by the
                    # wrapped_module.
                    logger.debug("replace %s with %s", module, wrapped_module)
                    setattr(parent, name, wrapped_module)

                module = wrapped_module
                break

        for child_name, child_module in list(module.named_children()):
            _process_module(child_module, child_name, module)

    _process_module(model)
