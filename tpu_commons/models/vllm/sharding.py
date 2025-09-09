import functools

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
from vllm.model_executor.layers.fused_moe import FusedMoE
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead, VocabParallelEmbedding)

from tpu_commons.logger import init_logger
from tpu_commons.models.vllm.jax_fused_moe import JaxFusedMoE

P = PartitionSpec

logger = init_logger(__name__)

VOCAB_AXIS_NAME = ('data', 'expert', 'model')
ATTN_HEAD_AXIS_NAME = 'model'
MLP_TP_AXIS_NAME = ('additional_model', 'model')


def shard_fused_moe(layer: torch.nn.Module, mesh: Mesh,
                    vllm_config: VllmConfig):
    assert isinstance(layer, FusedMoE)
    jax_layer = JaxFusedMoE(layer, mesh, vllm_config.parallel_config)
    return jax_layer


def shard_vocab_parallel_embedding(layer: torch.nn.Module, mesh: Mesh,
                                   vllm_config: VllmConfig):
    assert isinstance(layer, VocabParallelEmbedding)
    w = Parameter(torch_view(t2j(layer.weight)), requires_grad=False)
    layer.weight = w.apply_jax_(jax.device_put,
                                NamedSharding(mesh, P(MLP_TP_AXIS_NAME, None)))
    return layer


def shard_lm_head(layer: torch.nn.Module, mesh: Mesh, vllm_config: VllmConfig):
    # TODO(qihqi): currently this is not handling case of tie_word_weights=True.
    # if that config is set, then we should not create new weights but reuse the weight
    # from VocabParallelEmbedding
    assert isinstance(layer, ParallelLMHead)
    w = Parameter(torch_view(t2j(layer.weight)), requires_grad=False)
    layer.weight = w.apply_jax_(jax.device_put,
                                NamedSharding(mesh, P(MLP_TP_AXIS_NAME, None)))
    if hasattr(layer, 'bias'):
        bias = Parameter(torch_view(t2j(layer.bias)), requires_grad=False)
        layer.bias = bias.apply_jax_(jax.device_put,
                                     NamedSharding(mesh, P(MLP_TP_AXIS_NAME)))
    return layer


MODULE_TYPE_TO_WRAPPING_FUNC = {
    # TODO(kyuyeunk): Refactor this layer to use vLLM APIs.
    FusedMoE: shard_fused_moe,
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
        # tensors haven't been turned into torchax tensor are the ones not moved to TPU yet.
        return isinstance(
            x, torch.Tensor) and not isinstance(x, torchax.tensor.Tensor)

    def _move_to_tpu_replicated(x):
        # In certain cases, if t2j puts the tensor on CPU first and then device_put it to TPU, the tensor layout get messed up. To avoid that, we set jax default_device to TPU.
        with jax.default_device(jax.devices("tpu")[0]):
            x = t2j(x, use_dlpack=False)
        return torch_view(x).apply_jax_(jax.device_put,
                                        NamedSharding(mesh, P()))

    with jax.default_device(jax.devices("cpu")[0]), torchax.default_env():
        shard_parallel_layers_to_tpu(model, mesh, vllm_config)

        # For other weight tensors, repliate them on all the TPU chips.
        params, buffers, variables = extract_all_buffers(model)

        fmt_size = functools.partial(humanize.naturalsize, binary=True)
        for qual_name, x in {**params, **buffers, **variables}.items():
            if _is_unmoved_tensor(x):
                tensor_size = fmt_size(x.nbytes)
                logger.debug(
                    f"{qual_name=} is not sharded, {tensor_size=}, {x.shape=}, {x.dtype=}"
                )

        params, buffers, variables = pytree.tree_map_only(
            _is_unmoved_tensor, _move_to_tpu_replicated,
            (params, buffers, variables))
        set_all_buffers(model, {}, {}, variables)
        params_and_buffers = {**params, **buffers}

        return params_and_buffers


def extract_all_buffers(m: torch.nn.Module):
    params = {}
    buffers = {}
    variables = {}

    def extract_one(module, prefix):
        for k in dir(module):
            v = getattr(module, k, None)
            if v is None:
                continue

            qual_name = prefix + k
            if isinstance(v, torch.nn.Parameter):
                params[qual_name] = v
            elif isinstance(v, torch.nn.ParameterList):
                for i, param in enumerate(v):
                    params[qual_name + f'.{i}'] = param
            elif k in module._buffers:
                buffers[qual_name] = v
            elif isinstance(v, torch.Tensor):
                variables[qual_name] = v

        for name, child in module.named_children():
            extract_one(child, prefix + name + '.')

    extract_one(m, '')
    return params, buffers, variables


def set_all_buffers(m, params, buffers, variables):

    def set_one(module, prefix):
        for k in dir(module):
            qual_name = prefix + k
            if (potential_v := buffers.get(qual_name)) is not None or (
                    potential_v := variables.get(qual_name)) is not None:
                setattr(module, k, potential_v)
            elif (potential_v := params.get(qual_name)) is not None:
                # print(k, potential_v)
                # setattr(module, k, torch.nn.Parameter(potential_v))
                module.register_parameter(k, potential_v)
        for name, child in module.named_children():
            set_one(child, prefix + name + '.')

    set_one(m, '')
