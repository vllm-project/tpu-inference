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

import jax
import torch
import torchax
from jax.sharding import Mesh, NamedSharding, PartitionSpec
from torch.utils import _pytree as pytree
from torchax.interop import jax_view, torch_view
from torchax.ops.mappings import t2j
from vllm import envs as vllm_envs
from vllm.lora.layers import (ColumnParallelLinearWithLoRA,
                              MergedColumnParallelLinearWithLoRA,
                              MergedQKVParallelLinearWithLoRA,
                              QKVParallelLinearWithLoRA,
                              ReplicatedLinearWithLoRA,
                              RowParallelLinearWithLoRA)
from vllm.lora.layers.base_linear import BaseLinearLayerWithLoRA

from tpu_inference.layers.common.utils import general_device_put
from tpu_inference.logger import init_logger
from tpu_inference.utils import to_jax_dtype

P = PartitionSpec

logger = init_logger(__name__)


def shard_model_to_tpu(model: torch.nn.Module,
                       mesh: Mesh) -> dict[str, torchax.torch.Tensor]:
    """
    Shard the model weights and move them to TPU.
    At the same time, also turn the weight tensors into torchax tensors so that
    jax code can interop with it and the overall program can be traced and
    compiled in XLA.
    Args:
        model: A PyTorch model whose weights are on CPU main memory.
        mesh: JAX mesh object for sharding.
    Returns:
        Dictionary of parameters and buffers that will be used as arguments of
        torch.func.functional_call
    """

    with jax.default_device(jax.devices("cpu")[0]):
        _shard_module_to_tpu(model, mesh)

        params, buffers = _extract_all_params_buffers(model)

        # For other weight tensors, repliate them on all the TPU chips.
        params, buffers = pytree.tree_map_only(
            _tensor_is_in_cpu,
            lambda tensor: _shard_tensor_to_tpu_replicated(tensor, mesh),
            (params, buffers))

        return {**params, **buffers}


def update_lora(model: torch.nn.Module,
                initial_params_buffers) -> dict[str, torchax.torch.Tensor]:
    params, buffers = _extract_all_params_buffers(model)
    params_buffers = {**params, **buffers}
    for k, v in params_buffers.items():
        if 'lora_a_stacked' in k or 'lora_b_stacked' in k:
            assert k in initial_params_buffers, f"{k} not in initial_params_buffers"
            initial_params_buffers[k] = v

    return initial_params_buffers


def _extract_all_params_buffers(model: torch.nn.Module):
    return dict(model.named_parameters()), dict(model.named_buffers())


def _tensor_is_in_cpu(tensor: torch.tensor) -> bool:
    # Check if a tensor haven't been converted to torchax tensor.
    if not isinstance(tensor, torchax.tensor.Tensor):
        return True
    # Check if torchax tensor is still in CPU.
    return tensor.jax_device == jax.devices('cpu')[0]


def _convert_to_torchax_and_shard(tensor: torch.Tensor,
                                  sharding: NamedSharding) -> torch.Tensor:
    if vllm_envs.VLLM_TPU_USING_PATHWAYS and isinstance(tensor, torch.Tensor):
        np_tensor = tensor.detach().cpu().to(torch.float32).numpy()
        dtype = to_jax_dtype(tensor.dtype)
        return torch_view(jax.device_put(np_tensor, sharding).astype(dtype))
    else:
        if isinstance(tensor, torchax.tensor.Tensor):
            tensor = jax_view(tensor)
        else:
            tensor = t2j(tensor)
        return torch_view(general_device_put(tensor, sharding))


def _shard_tensor_to_tpu_replicated(tensor: torch.Tensor,
                                    mesh: Mesh) -> torchax.tensor.Tensor:
    return _convert_to_torchax_and_shard(tensor, NamedSharding(mesh, P()))


def _shard_base_linear_lora_replicated(layer: BaseLinearLayerWithLoRA,
                                       mesh: Mesh) -> None:
    # NOTE: lora_a_stacked[i] has shape [max_loras, 1, num_out, num_in]
    sharded_lora_a_tpu = torch.nn.ParameterList()
    sharded_lora_b_tpu = torch.nn.ParameterList()

    for i in range(layer.n_slices):
        sharded_lora_a_tpu.append(
            _shard_tensor_to_tpu_replicated(layer.lora_a_stacked[i], mesh))
        sharded_lora_b_tpu.append(
            _shard_tensor_to_tpu_replicated(layer.lora_b_stacked[i], mesh))

    layer.lora_a_stacked = sharded_lora_a_tpu
    layer.lora_b_stacked = sharded_lora_b_tpu


def _shard_column_linear_lora(layer: ColumnParallelLinearWithLoRA,
                              mesh: Mesh) -> None:
    assert layer.n_slices > 0, "layer.n_slices should be greater than 0"
    # lora_a_stacked[i] has shape [max_loras, 1, max_lora_rank, in_features]
    sharded_lora_a_tpu = torch.nn.ParameterList()
    sharded_lora_b_tpu = torch.nn.ParameterList()

    # lora_b_stacked[i] has shape [max_loras, 1, out_features, max_lora_rank]
    lora_b_partition_spec = P(None, None, 'model', None)
    lora_b_sharding = NamedSharding(mesh, lora_b_partition_spec)
    for i in range(layer.n_slices):
        sharded_lora_a_tpu.append(
            _shard_tensor_to_tpu_replicated(layer.lora_a_stacked[i], mesh))

        sharded_lora_b_tpu.append(
            _convert_to_torchax_and_shard(layer.lora_b_stacked[i],
                                          lora_b_sharding))

    layer.lora_a_stacked = sharded_lora_a_tpu
    layer.lora_b_stacked = sharded_lora_b_tpu


def _shard_qkv_linear_lora(layer: ColumnParallelLinearWithLoRA,
                           mesh: Mesh) -> None:
    _shard_column_linear_lora(layer, mesh)


def _shard_merged_column_parallel_linear_lora(
        layer: MergedColumnParallelLinearWithLoRA, mesh: Mesh) -> None:
    _shard_column_linear_lora(layer, mesh)


def _shard_merged_qkv_parallel_linear_lora(
        layer: MergedQKVParallelLinearWithLoRA, mesh: Mesh) -> None:
    _shard_column_linear_lora(layer, mesh)


def _shard_row_parallel_linear_lora(layer: RowParallelLinearWithLoRA,
                                    mesh: Mesh) -> None:
    _shard_base_linear_lora_replicated(layer, mesh)


# NOTE: Ordering is important as it calls first matched type of a given module
MODULE_TYPE_TO_SHARDING_FUNC = [
    # Shard LoRA layers
    (ColumnParallelLinearWithLoRA, _shard_column_linear_lora),
    (QKVParallelLinearWithLoRA, _shard_qkv_linear_lora),
    (MergedColumnParallelLinearWithLoRA,
     _shard_merged_column_parallel_linear_lora),
    (MergedQKVParallelLinearWithLoRA, _shard_merged_qkv_parallel_linear_lora),
    (RowParallelLinearWithLoRA, _shard_row_parallel_linear_lora),
    (ReplicatedLinearWithLoRA, _shard_base_linear_lora_replicated),
]


def _shard_module_to_tpu(model: torch.nn.Module, mesh: Mesh) -> None:
    for path, module in model.named_modules():
        for module_type, sharding_func in MODULE_TYPE_TO_SHARDING_FUNC:
            if type(module) is module_type:
                logger.debug("shard %s with %s", path, sharding_func)
                sharding_func(module, mesh)
                break
