import functools
from dataclasses import dataclass
from typing import Optional

import humanize
import jax
import torch
import torchax
from jax.sharding import Mesh, NamedSharding, PartitionSpec
from torch.nn.parameter import Parameter
from torch.utils import _pytree as pytree
from torchax.interop import extract_all_buffers, torch_view
from torchax.ops.mappings import t2j
from vllm.attention import Attention as VllmAttention
from vllm.config import VllmConfig
from vllm.model_executor.layers.fused_moe import FusedMoE
from vllm.model_executor.layers.linear import \
    UnquantizedLinearMethod  # yapf: disable
from vllm.model_executor.layers.linear import (ColumnParallelLinear,
                                               MergedColumnParallelLinear,
                                               QKVParallelLinear,
                                               RowParallelLinear)

from tpu_commons.logger import init_logger
from tpu_commons.models.vllm.jax_attention import JaxAttention
from tpu_commons.models.vllm.jax_fused_moe import JaxFusedMoE
from tpu_commons.models.vllm.jax_merged_column_parallel_linear import \
    JaxMergedColumnParallelLinear
from tpu_commons.models.vllm.jax_merged_column_parallel_linear_fusion_assignments import \
    get_model_matmul_fusion_assignment
from tpu_commons.models.vllm.jax_qkv_parallel_linear import \
    JaxQKVParallelLinear
from tpu_commons.models.vllm.jax_row_parallel_linear import \
    JaxRowParallelLinear

P = PartitionSpec

logger = init_logger(__name__)


@dataclass
class AttentionInfo:
    layer_name: str
    num_kv_heads: int
    head_size: int
    attn_type: str
    sliding_window: Optional[int] = None
    kv_sharing_target_layer_name: Optional[str] = None


def shard_attention(layer: torch.nn.Module, mesh: Mesh,
                    vllm_config: VllmConfig):
    vllm_config.compilation_config.static_forward_context[
        layer.layer_name] = AttentionInfo(
            layer_name=layer.layer_name,
            num_kv_heads=layer.num_kv_heads,
            head_size=layer.head_size,
            attn_type=layer.attn_type,
            sliding_window=layer.sliding_window,
            kv_sharing_target_layer_name=layer.kv_sharing_target_layer_name)
    return JaxAttention(layer, mesh)


def shard_qkv_parallel_linear(layer: torch.nn.Module, mesh: Mesh,
                              vllm_config: VllmConfig):
    assert isinstance(layer, QKVParallelLinear)
    jax_layer = JaxQKVParallelLinear(
        layer,
        mesh,
        shard_qkv_parallel_linear.fuse_matmuls,
        enable_sequence_parallelism=vllm_config.compilation_config.pass_config.
        enable_sequence_parallelism)
    return jax_layer


def shard_merged_column_parallel_linear(layer: torch.nn.Module, mesh: Mesh,
                                        vllm_config: VllmConfig):
    assert isinstance(layer, MergedColumnParallelLinear)
    jax_layer = JaxMergedColumnParallelLinear(
        layer,
        mesh,
        shard_merged_column_parallel_linear.fuse_matmuls,
        enable_sequence_parallelism=vllm_config.compilation_config.pass_config.
        enable_sequence_parallelism)
    return jax_layer


def shard_column_parallel_linear(layer: torch.nn.Module, mesh: Mesh,
                                 vllm_config: VllmConfig):
    assert isinstance(layer, ColumnParallelLinear)
    if not isinstance(layer.quant_method, UnquantizedLinearMethod):
        raise ValueError(
            "tpu_commons torchax ColumnParallelLinear doesn't support quantization"
        )
    w = Parameter(torch_view(t2j(layer.weight)), requires_grad=False)
    layer.weight = w.apply_jax_(jax.device_put,
                                NamedSharding(mesh, P('model', None)))
    return layer


def shard_row_parallel_linear(layer: torch.nn.Module, mesh: Mesh,
                              vllm_config: VllmConfig):
    assert isinstance(layer, RowParallelLinear)
    jax_layer = JaxRowParallelLinear(
        layer,
        mesh,
        enable_sequence_parallelism=vllm_config.compilation_config.pass_config.
        enable_sequence_parallelism)
    return jax_layer


def shard_fused_moe(layer: torch.nn.Module, mesh: Mesh,
                    vllm_config: VllmConfig):
    assert isinstance(layer, FusedMoE)
    jax_layer = JaxFusedMoE(layer, mesh, vllm_config.parallel_config)
    return jax_layer


MODULE_TYPE_TO_WRAPPING_FUNC = {
    VllmAttention: shard_attention,
    QKVParallelLinear: shard_qkv_parallel_linear,
    MergedColumnParallelLinear: shard_merged_column_parallel_linear,
    ColumnParallelLinear: shard_column_parallel_linear,
    RowParallelLinear: shard_row_parallel_linear,
    FusedMoE: shard_fused_moe,
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

    tp_size = vllm_config.parallel_config.tensor_parallel_size
    shard_qkv_parallel_linear.fuse_matmuls = get_model_matmul_fusion_assignment(
        vllm_config.model_config.model,
        vllm_config.scheduler_config.max_num_batched_tokens, tp_size,
        "QKVParallelLinear")
    shard_merged_column_parallel_linear.fuse_matmuls = get_model_matmul_fusion_assignment(
        vllm_config.model_config.model,
        vllm_config.scheduler_config.max_num_batched_tokens, tp_size,
        "MergedColumnParallelLinear")

    with jax.default_device(jax.devices("cpu")[0]), torchax.default_env():
        shard_parallel_layers_to_tpu(model, mesh, vllm_config)

        # For other weight tensors, repliate them on all the TPU chips.
        params, buffers = extract_all_buffers(model)

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
