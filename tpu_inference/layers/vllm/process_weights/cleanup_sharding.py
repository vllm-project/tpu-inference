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

import gc

import jax
import torch
import torchax
from jax.sharding import Mesh, NamedSharding, PartitionSpec
from torch.nn import Parameter
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
from vllm.model_executor.layers.fused_moe.layer import (
    FusedMoE, UnquantizedFusedMoEMethod)
from vllm.model_executor.layers.linear import (LinearBase, QKVParallelLinear,
                                               UnquantizedLinearMethod)
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead, VocabParallelEmbedding)

from tpu_inference import envs
from tpu_inference.layers.common.sharding import ShardingAxisName
from tpu_inference.layers.vllm.process_weights.fused_moe_weights import (
    FusedMoEWeights, process_moe_weights, shard_moe_weights)
from tpu_inference.layers.vllm.process_weights.linear_weights import (
    LinearWeights, process_linear_weights, shard_linear_weights,
    to_parameter_list)
from tpu_inference.logger import init_logger
from tpu_inference.utils import get_mesh_shape_product, to_jax_dtype

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
        print("Finished sharding modules to TPU.")

        params, buffers = _extract_all_params_buffers(model)
        print("Finished extracting parameters and buffers.")
        # print(f'{params=}')
        # For other weight tensors, repliate them on all the TPU chips.
        params, buffers = pytree.tree_map_only(
            _tensor_is_in_cpu,
            lambda tensor: _shard_tensor_to_tpu_replicated(tensor, mesh),
            (params, buffers))
        # def _process_dict(tensor_dict):
        #     new_dict = {}
        #     for name, tensor in tensor_dict.items():
        #         if _tensor_is_in_cpu(tensor, name):
        #             new_dict[name] = _shard_tensor_to_tpu_replicated(tensor, mesh)
        #         else:
        #             new_dict[name] = tensor
        #     return new_dict

        # params = _process_dict(params)
        # buffers = _process_dict(buffers)
        print("Finished sharding replcates to TPU.")

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
        print(f'Tensor {type(tensor)} is still in CPU.')
        return True
    # Check if torchax tensor is still in CPU.
    print(f'Torchax tensor device: {tensor.jax_device}')
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
        return torch_view(_sharded_device_put(tensor, sharding))


def _shard_tensor_to_tpu_replicated(tensor: torch.Tensor,
                                    mesh: Mesh) -> torchax.tensor.Tensor:
    return _convert_to_torchax_and_shard(tensor, NamedSharding(mesh, P()))


def _shard_vocab_parallel_embedding(layer: VocabParallelEmbedding,
                                    mesh: Mesh) -> None:
    weight = _convert_to_torchax_and_shard(
        layer.weight, NamedSharding(mesh, P(ShardingAxisName.MLP_TENSOR,
                                            None)))
    layer.weight = Parameter(weight, requires_grad=False)


def _shard_lm_head(layer: ParallelLMHead, mesh: Mesh):
    # TODO(qihqi): currently this is not handling case of tie_word_weights=True.
    # if that config is set, then we should not create new weights but reuse the
    # weight from VocabParallelEmbedding
    weight = _convert_to_torchax_and_shard(
        layer.weight, NamedSharding(mesh, P(ShardingAxisName.MLP_TENSOR,
                                            None)))
    layer.weight = Parameter(weight, requires_grad=False)
    if layer.bias is not None:
        bias = _convert_to_torchax_and_shard(
            layer.bias, NamedSharding(mesh, P(ShardingAxisName.MLP_TENSOR)))
        layer.bias = Parameter(bias, requires_grad=False)


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
    # Shard embedding layers
    (ParallelLMHead, _shard_lm_head),
    (VocabParallelEmbedding, _shard_vocab_parallel_embedding),
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
                logger.info("shard %s with %s", path, sharding_func)
                sharding_func(module, mesh)
                break


def _sharded_device_put(tensor: jax.Array, sharding) -> jax.Array:
    if isinstance(tensor, tuple):
        return tuple(_sharded_device_put(t, sharding) for t in tensor)
    multihost_backend = envs.TPU_MULTIHOST_BACKEND
    if multihost_backend != "ray":
        return jax.device_put(tensor, sharding)

    # NOTE: at here, num_global_devices != num_local_devices
    # meaning we are in multi-host setup. Each host will run the same process
    # and each process only need to handle the devices accessible to this host.
    shape = tensor.shape
    x_split = [
        jax.device_put(tensor[i], device) for device, i in
        sharding.addressable_devices_indices_map(shape).items()
    ]
    return jax.make_array_from_single_device_arrays(shape,
                                                    sharding,
                                                    x_split,
                                                    dtype=tensor.dtype)


# def has_all_weights_loaded(layer: torch.nn.Module) -> bool:
#     for param_name, param in layer.named_parameters(recurse=False):
#         print(f'{param_name} _is_loaded: {getattr(param, "_is_loaded", False)}')
#         if not getattr(param, "_is_loaded", False):
#             return False
#     return True

import os

import psutil


def print_ram_usage(prefix=""):
    process = psutil.Process(os.getpid())
    # RSS (Resident Set Size) is the non-swapped physical memory a process has used.
    rss_gb = process.memory_info().rss / (1024**3)
    print(f"{prefix}Current RAM Usage: {rss_gb:.2f} GB")


def print_tensor_size(tensor, name="Tensor"):
    if tensor is None:
        print(f"{name} is None")
        return
    if hasattr(tensor, "nbytes"):
        size_gb = tensor.nbytes / (1024**3)
    elif hasattr(tensor, "element_size") and hasattr(tensor, "nelement"):
        size_gb = tensor.element_size() * tensor.nelement() / (1024**3)
    else:
        size_gb = 0.0
    print(f"{name} Size: {size_gb:.4f} GB")


def maybe_process_unquantized_linear_weight(layer: torch.nn.Module,
                                            layer_name: str):
    print('Processing linear weight for layer:', layer_name)
    print(type(layer))
    print(type(layer.weight))
    weight = t2j(layer.weight, use_dlpack=False)
    print_tensor_size(layer.weight, "weight")
    delattr(layer, 'weight')
    if layer.bias is not None and not layer.skip_bias_add:
        if layer.return_bias:
            logger.warning_once("Bias might return incorrect value.")
        bias = t2j(layer.bias, use_dlpack=False)
        delattr(layer, 'bias')
    else:
        bias = None

    linear_config = layer.quant_method.linear_config

    @jax.jit
    def process_unquantized_linear_weights(
        weight: jax.Array,
        bias: jax.Array | None,
    ) -> LinearWeights:
        return process_linear_weights(
            LinearWeights(
                weight=weight,
                weight_scale=None,
                zero_point=None,
                bias=bias,
            ),
            fused=linear_config.fuse_matmuls,
            output_sizes=linear_config.output_sizes,
            reorder_size=linear_config.n_shards,
        )

    print(f'{weight.device=}')
    weights = process_unquantized_linear_weights(weight, bias)
    print(f'{weights.weight.device=}')
    weights_sharded = shard_linear_weights(
        weights,
        mesh=linear_config.mesh,
        weight_p_spec=linear_config.weight_sharding,
        bias_p_spec=linear_config.bias_sharding,
    )
    print(f'{weights_sharded.weight.device=}')
    weights = torch_view(weights_sharded)
    print(f'{weights.weight.device=}')

    if linear_config.fuse_matmuls:
        layer.weight = Parameter(weights.weight, requires_grad=False)
        if bias is not None:
            layer.bias = Parameter(weights.bias, requires_grad=False)
    else:
        layer.weight = to_parameter_list(weights.weight)
        if bias is not None:
            layer.bias = to_parameter_list(weights.bias)

    del weights_sharded
    del weights
    gc.collect()

def print_live_tensors():
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                print(f"Type: {type(obj)}, Size: {obj.size()}, Device: {obj.device}, Dtype: {obj.dtype}")
                print_tensor_size(obj, "Live Tensor")
        except:
            pass

import sys
def inspect_tensor_refs(tensor, name="Tensor"):
    print(f"--- Inspecting References for: {name} ---")
    
    # 1. Print Reference Count
    # Remember: subtract 1 for the getrefcount argument itself
    ref_count = sys.getrefcount(tensor) - 1
    print(f"Total References: {ref_count}")
    
    # 2. Print Who is Referring
    referrers = gc.get_referrers(tensor)
    
    print("Referred to by:")
    for i, referrer in enumerate(referrers):
        # We try to print the type to avoid dumping massive objects to stdout
        ref_type = type(referrer).__name__
        
        # If it's a list or dict, we might want to know roughly what's inside
        if isinstance(referrer, list):
            info = f"List of length {len(referrer)}"
        elif isinstance(referrer, dict):
            info = f"Dict with keys {list(referrer.keys())[:5]}..." # concise preview
        else:
            info = str(referrer)[:5] + "..." # Truncate long representations
            
        print(f"  {i+1}. Type: {ref_type} | Info: {info}")
    print("-" * 40)

def maybe_process_unquantized_moe_weight(layer: torch.nn.Module,
                                         layer_name: str, param_name: str):
    # pass
    print('Processing moe weight for layer:', layer_name)
    # time.sleep(5)
    # print_ram_usage("sleep done: ")
    # time.sleep(5)
    # print_ram_usage("sleep done: ")
    print_tensor_size(layer.w13_weight, "w13_weight")
    print_tensor_size(layer.w2_weight, "w2_weight")
    w13_weight = t2j(layer.w13_weight, use_dlpack=False)
    w2_weight = t2j(layer.w2_weight, use_dlpack=False)
    print_ram_usage("After moving weights to jax: ")
    # inspect_tensor_refs(weight, "w13_weight")
    # 2. NUCLEAR OPTION: Free CPU memory immediately
    # This works even if 'weight' or other variables refer to these tensors.
    layer.w13_weight.untyped_storage().resize_(0)
    layer.w2_weight.untyped_storage().resize_(0)
    
    delattr(layer, 'w13_weight')
    delattr(layer, 'w2_weight')
    # inspect_tensor_refs(weight, "w13_weight")
    print_ram_usage("After deleting original weights: ")
    # print_live_tensors()
    # time.sleep(5)
    # print_ram_usage("sleep done: ")
    # time.sleep(5)
    # print_ram_usage("sleep done: ")

    quant_method = layer.quant_method

    if quant_method.moe.has_bias:
        w13_bias = t2j(layer.w13_bias, use_dlpack=False)
        w2_bias = t2j(layer.w2_bias, use_dlpack=False)
    else:
        w13_bias = w2_bias = None

    @jax.jit
    def process_unquantized_moe_weights(
        w13_weight: jax.Array,
        w13_bias: jax.Array | None,
        w2_weight: jax.Array,
        w2_bias: jax.Array | None,
    ) -> FusedMoEWeights:

        w13_interleave = layer.activation == "swigluoai"
        w13_reorder_size = get_mesh_shape_product(quant_method.mesh,
                                                  ShardingAxisName.MLP_TENSOR)

        return process_moe_weights(
            FusedMoEWeights(
                w13_weight=w13_weight,
                w13_weight_scale=None,
                w13_bias=w13_bias,
                w2_weight=w2_weight,
                w2_weight_scale=None,
                w2_bias=w2_bias,
            ),
            moe_backend=quant_method.moe_backend,
            w13_reorder_size=w13_reorder_size,
            w13_interleave=w13_interleave,
        )

    print(f'{w13_weight.device=}')
    weights = process_unquantized_moe_weights(
        w13_weight,
        w13_bias,
        w2_weight,
        w2_bias,
    )
    print_ram_usage("After processing moe weights: ")
    # time.sleep(5)
    # print_ram_usage("sleep done: ")
    # time.sleep(5)
    # print_ram_usage("sleep done: ")
    print(f'{weights.w13_weight.device=}')
    print(f'{quant_method.mesh=}')
    weights_sharded = shard_moe_weights(weights, quant_method.moe_backend,
                                        quant_method.mesh)
    print(f'{weights_sharded.w13_weight.device.addressable_devices=}')
    weights = torch_view(weights_sharded)
    print_ram_usage("After sharding moe weights: ")
    # time.sleep(5)
    # print_ram_usage("sleep done: ")
    print(f'{weights.w13_weight.device=}')
    # del layer.w13_weight
    # del layer.w2_weight
    layer.w13_weight = Parameter(weights.w13_weight, requires_grad=False)
    layer.w2_weight = Parameter(weights.w2_weight, requires_grad=False)
    del w13_weight
    del w2_weight
    del w13_bias
    del w2_bias
    del weights_sharded
    del weights
    gc.collect()

    if quant_method.moe.has_bias:
        layer.w13_bias = Parameter(weights.w13_bias, requires_grad=False)
        layer.w2_bias = Parameter(weights.w2_bias, requires_grad=False)


def prepare_incremental_loading(model: torch.nn.Module, mesh: Mesh) -> None:
    """
    Traverses the model and attaches a generic weight_loader to any parameter
    that doesn't already have one (i.e. wasn't handled by unquantized.py).
    """

    def create_weight_loader(layer, original_loader, layer_name, param_name):

        def weight_loader_wrapper(param: torch.nn.Parameter,
                                  loaded_weight: torch.Tensor, *args,
                                  **kwargs):
            # Weight loading
            res = original_loader(param, loaded_weight, *args, **kwargs)
            print(
                f'Loaded weight for param {layer_name}:{type(layer)} {param_name} with {args} {kwargs}'
            )
            # Sharding
            # For now, only handle unquantized linear and moe layers.
            if isinstance(layer, LinearBase) and isinstance(
                    layer.quant_method, UnquantizedLinearMethod):
                if isinstance(layer, QKVParallelLinear):
                    assert len(
                        args) == 1, "Expecting shard_id as the only argument"
                    shard_id = args[0]
                    layer._loaded_shards.add((param_name, shard_id))
                    if len(layer._loaded_shards) == 3 * len(
                            dict(layer.named_parameters(recurse=False))):
                        print_ram_usage("Before processing linear weight: ")
                        maybe_process_unquantized_linear_weight(
                            layer, layer_name)
                        print_ram_usage("After processing linear weight: ")
                else:
                    layer._loaded_params.add(param_name)
                    if len(layer._loaded_params) == len(
                            dict(layer.named_parameters(recurse=False))):
                        maybe_process_unquantized_linear_weight(layer, layer_name)
            if isinstance(layer, FusedMoE) and isinstance(
                    layer.quant_method, UnquantizedFusedMoEMethod):
                expert_id = kwargs.get('expert_id')
                shard_id = kwargs.get('shard_id')
                assert expert_id is not None, "Expecting expert_id argument"
                assert shard_id is not None, "Expecting shard_id argument"
                layer._loaded_expert_shards.add((expert_id, shard_id))

                if len(layer._loaded_expert_shards
                       ) == layer.global_num_experts * 3:
                    print_ram_usage("Before processing moe weight: ")
                    # print_live_tensors()
                    maybe_process_unquantized_moe_weight(
                        layer, layer_name, param_name)
                    print_ram_usage("After processing moe weight: ")
                    # print_live_tensors()

            return res
        return weight_loader_wrapper

    for name, module in model.named_modules():
        print(f'Preparing incremental loading for module: {name}: {type(module)}')
        if isinstance(module, FusedMoE):
            module._loaded_expert_shards = set()
        if isinstance(module, QKVParallelLinear):
            module._loaded_shards = set()
        if isinstance(module, LinearBase):
            module._loaded_params = set()
        for param_name, param in module.named_parameters(recurse=False):
            original_loader = getattr(param, "weight_loader", None)
            if original_loader is None:
                print(
                    f'Param {name}:{type(module)} {param_name} already has no weight_loader, skipping.'
                )
                continue
            setattr(
                param, "weight_loader",
                create_weight_loader(module, original_loader, name,
                                     param_name))
            print(
                f'Param {name}:{type(module)} {param_name} weight_loader is set.'
            )
