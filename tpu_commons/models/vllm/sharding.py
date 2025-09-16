import functools
from collections import OrderedDict

import humanize
import jax
import torch
import torchax
from jax.sharding import Mesh, NamedSharding, PartitionSpec
from torch.nn import Parameter
from torch.utils import _pytree as pytree
from torchax.interop import torch_view
from torchax.ops.mappings import t2j
from vllm.config import VllmConfig
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead, VocabParallelEmbedding)

from tpu_commons.logger import init_logger

P = PartitionSpec

logger = init_logger(__name__)


def shard_vocab_parallel_embedding(layer: torch.nn.Module, mesh: Mesh,
                                   vllm_config: VllmConfig):
    assert isinstance(layer, VocabParallelEmbedding)
    w = Parameter(torch_view(t2j(layer.weight)), requires_grad=False)
    layer.weight = w.apply_jax_(jax.device_put,
                                NamedSharding(mesh, P('model', None)))
    return layer


def shard_lm_head(layer: torch.nn.Module, mesh: Mesh, vllm_config: VllmConfig):
    # TODO(qihqi): currently this is not handling case of tie_word_weights=True.
    # if that config is set, then we should not create new weights but reuse the
    # weight from VocabParallelEmbedding
    assert isinstance(layer, ParallelLMHead)
    w = Parameter(torch_view(t2j(layer.weight)), requires_grad=False)
    layer.weight = w.apply_jax_(jax.device_put,
                                NamedSharding(mesh, P('model', None)))
    if hasattr(layer, 'bias'):
        bias = Parameter(torch_view(t2j(layer.bias)), requires_grad=False)
        layer.bias = bias.apply_jax_(jax.device_put,
                                     NamedSharding(mesh, P('model')))
    return layer


MODULE_TYPE_TO_WRAPPING_FUNC = {
    VocabParallelEmbedding: shard_vocab_parallel_embedding,
    ParallelLMHead: shard_lm_head,
}


def shard_parallel_layers_to_tpu(model: torch.nn.Module, mesh: Mesh,
                                 vllm_config: VllmConfig) -> None:

    def _shard_layer(module, name=None, parent=None):
        for module_type, wrapping_func in MODULE_TYPE_TO_WRAPPING_FUNC.items():
            if isinstance(module, module_type):
                wrapped_module = wrapping_func(module, mesh, vllm_config)

                assert parent is not None and name is not None, (
                    "Top Level module is not expected to be wrapped.")
                logger.debug("replace %s with %s", module, wrapped_module)
                setattr(parent, name, wrapped_module)

                module = wrapped_module
                break

        for child_name, child_module in list(module.named_children()):
            _shard_layer(child_module, child_name, module)

    _shard_layer(model)


def shard_model_to_tpu(model: torch.nn.Module, mesh: Mesh,
                       vllm_config: VllmConfig):
    """
    Shard the model weights and move them to TPU.

    At the same time, also turn the weight tensors into torchax tensors so that
    jax code can interop with it and the overall program can be traced and
    compiled in XLA.

    Args:
        model: A PyTorch model whose weights are on CPU main memory.
        mesh: JAX mesh object for sharding.
    """

    def _is_unmoved_tensor(x):
        # tensors haven't been turned into torchax tensor are the ones not moved
        # to TPU yet.
        return isinstance(
            x, torch.Tensor) and not isinstance(x, torchax.tensor.Tensor)

    def _move_to_tpu_replicated(x):
        x = t2j(x, use_dlpack=False)
        return torch_view(x).apply_jax_(jax.device_put,
                                        NamedSharding(mesh, P()))

    with jax.default_device(jax.devices("cpu")[0]), torchax.default_env():
        shard_parallel_layers_to_tpu(model, mesh, vllm_config)

        # For other weight tensors, repliate them on all the TPU chips.
        params, buffers = extract_all_params_buffers(model)

        fmt_size = functools.partial(humanize.naturalsize, binary=True)
        for qual_name, x in {**params, **buffers}.items():
            if _is_unmoved_tensor(x):
                tensor_size = fmt_size(x.nbytes)
                logger.debug(
                    f"{qual_name=} is not sharded, {tensor_size=}, {x.shape=}, {x.dtype=}"
                )

        params, buffers = pytree.tree_map_only(_is_unmoved_tensor,
                                               _move_to_tpu_replicated,
                                               (params, buffers))
        params_and_buffers = {**params, **buffers}

        return params_and_buffers


def extract_all_params_buffers(m: torch.nn.Module):
    return dict(m.named_parameters()), dict(m.named_buffers())


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
    # Note, lora_a_stacked[i] has shape [max_loras, 1, num_out_features, num_in_features]
    sharded_lora_a_tpu = torch.nn.ParameterList()
    sharded_lora_b_tpu = torch.nn.ParameterList()
    sharded_lora_bias_tpu = torch.nn.ParameterList()

    for i in range(layer.n_slices):
        sharded_lora_a_tpu.append(
            shard_and_move_tensor_to_tpu(layer.lora_a_stacked[i], mesh))
        sharded_lora_b_tpu.append(
            shard_and_move_tensor_to_tpu(layer.lora_b_stacked[i], mesh))
        if layer.lora_bias_stacked is not None:
            sharded_lora_bias_tpu.append(
                shard_and_move_tensor_to_tpu(layer.lora_bias_stacked[i], mesh))

    layer.lora_a_stacked = sharded_lora_a_tpu
    layer.lora_b_stacked = sharded_lora_b_tpu
    if layer.lora_bias_stacked is not None:
        layer.lora_bias_stacked = sharded_lora_bias_tpu


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


LORA_MODULE_TYPE_TO_WRAPPING_FUNC = OrderedDict([
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
    the LORA_MODULE_TYPE_TO_WRAPPING_FUNC mapping.

    Args:
        model: torch.nn.Module to process
        mesh: An SPMD mesh object used for sharding
    """

    def _process_module(module, name=None, parent=None):
        for module_type, wrapping_func in LORA_MODULE_TYPE_TO_WRAPPING_FUNC.items(
        ):
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
