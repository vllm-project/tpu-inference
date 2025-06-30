import functools

import humanize
import jax
import torch
import torchax
from jax.sharding import Mesh, NamedSharding, PartitionSpec
from torch.nn.parameter import Parameter
from torch.utils import _pytree as pytree
from torchax.interop import extract_all_buffers, torch_view
from torchax.tensor import t2j
from vllm.attention import Attention as VllmAttention
from vllm.config import ParallelConfig
from vllm.model_executor.layers.fused_moe import FusedMoE
from vllm.model_executor.layers.linear import (ColumnParallelLinear,
                                               QKVParallelLinear,
                                               RowParallelLinear)

from tpu_commons.logger import init_logger
from tpu_commons.models.vllm.jax_attention import JaxAttention
from tpu_commons.models.vllm.jax_fused_moe import JaxFusedMoE
from tpu_commons.models.vllm.jax_qkv_parallel_linear import \
    JaxQKVParallelLinear

P = PartitionSpec

logger = init_logger(__name__)


def shard_attention(layer: torch.nn.Module, mesh: Mesh,
                    vllm_parallel_config: ParallelConfig):
    return JaxAttention(layer, mesh)


def shard_qkv_parallel_linear(layer: torch.nn.Module, mesh: Mesh,
                              vllm_parallel_config: ParallelConfig):
    assert isinstance(layer, QKVParallelLinear)
    jax_layer = JaxQKVParallelLinear(layer, mesh)
    return jax_layer


def shard_column_parallel_linear(layer: torch.nn.Module, mesh: Mesh,
                                 vllm_parallel_config: ParallelConfig):
    assert isinstance(layer, ColumnParallelLinear)
    w = Parameter(torch_view(t2j(layer.weight)))
    layer.weight = w.apply_jax_(jax.device_put,
                                NamedSharding(mesh, P('model', None)))
    return layer


def shard_row_parallel_linear(layer: torch.nn.Module, mesh: Mesh,
                              vllm_parallel_config: ParallelConfig):
    assert isinstance(layer, RowParallelLinear)
    w = Parameter(torch_view(t2j(layer.weight)))
    layer.weight = w.apply_jax_(jax.device_put,
                                NamedSharding(mesh, P(None, 'model')))
    return layer


def shard_fused_moe(layer: torch.nn.Module, mesh: Mesh,
                    vllm_parallel_config: ParallelConfig):
    assert isinstance(layer, FusedMoE)
    jax_layer = JaxFusedMoE(layer, mesh, vllm_parallel_config)
    return jax_layer


MODULE_TYPE_TO_WRAPPING_FUNC = {
    VllmAttention: shard_attention,
    QKVParallelLinear: shard_qkv_parallel_linear,
    ColumnParallelLinear: shard_column_parallel_linear,
    RowParallelLinear: shard_row_parallel_linear,
    FusedMoE: shard_fused_moe,
}


def shard_parallel_layers_to_tpu(model: torch.nn.Module, mesh: Mesh,
                                 vllm_parallel_config: ParallelConfig) -> None:

    def _shard_layer(module, name=None, parent=None):
        for module_type, wrapping_func in MODULE_TYPE_TO_WRAPPING_FUNC.items():
            if isinstance(module, module_type):
                wrapped_module = wrapping_func(module, mesh,
                                               vllm_parallel_config)

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
                       vllm_parallel_config: ParallelConfig):
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
        # tensors haven't been turned into torchax tensor are the ones not moved to TPU yet.
        return isinstance(
            x, torch.Tensor) and not isinstance(x, torchax.tensor.Tensor)

    def _move_to_tpu_replicated(x):
        return torch_view(t2j(x)).apply_jax_(jax.device_put,
                                             NamedSharding(mesh, P()))

    with jax.default_device(jax.devices("cpu")[0]), torchax.default_env():
        shard_parallel_layers_to_tpu(model, mesh, vllm_parallel_config)

        # For other weight tensors, repliate them on all the TPU chips.
        params, buffers = extract_all_buffers(model)

        fmt_size = functools.partial(humanize.naturalsize, binary=True)
        for qual_name, x in {**params, **buffers}.items():
            if _is_unmoved_tensor(x):
                tensor_size = fmt_size(x.nbytes)
                logger.debug(f"{qual_name=} is not sharded, {tensor_size=}")

        params, buffers = pytree.tree_map_only(_is_unmoved_tensor,
                                               _move_to_tpu_replicated,
                                               (params, buffers))
        params_and_buffers = {**params, **buffers}

        return params_and_buffers
