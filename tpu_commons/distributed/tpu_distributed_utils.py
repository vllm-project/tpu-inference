# SPDX-License-Identifier: Apache-2.0
import os
from collections import OrderedDict
from typing import Optional, Union

import jax
import jax.numpy as jnp
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_xla.distributed.spmd as xs
import torchax
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from torch.nn import Parameter
from torchax.ops.mappings import t2j_dtype
from vllm.model_executor.layers.linear import (ColumnParallelLinear,
                                               MergedColumnParallelLinear,
                                               QKVParallelLinear,
                                               RowParallelLinear)

from tpu_commons.logger import init_logger

logger = init_logger(__name__)

VLLM_TORCHAX_ENABLED = os.environ.get('VLLM_TORCHAX_ENABLED', '0') == '1'


def create_torchax_kv_cache(shape, dtype, mesh,
                            partition_spec) -> torch.Tensor:
    # This works better than device_put for large tensor allocation.
    assert VLLM_TORCHAX_ENABLED, \
        "It's expected that VLLM_TORCHAX_ENABLED is True."

    jax_dtype = t2j_dtype(dtype)
    if mesh is None:
        device = jax.devices()[0]
    else:
        sharding = NamedSharding(mesh, P(*partition_spec))
        device = sharding

    if mesh is None:
        # We have to call device_put again for single chip to generate
        # replicated sharding for kv cache
        jax_t = jax.device_put(jnp.zeros(shape, dtype=jax_dtype), device)
    else:
        jax_t = jnp.zeros(shape, dtype=jax_dtype,
                          device=device).block_until_ready()
    torchax_t = torchax.tensor.Tensor(jax_t, torchax.default_env())
    return torchax_t


def create_torchax_tensor_with_partition_spec(
        weight_t: torch.Tensor,
        mesh: Optional[Union["xs.Mesh", Mesh]] = None,
        partition_spec: Optional[tuple] = None) -> torch.Tensor:
    assert VLLM_TORCHAX_ENABLED, \
        "It's expected that VLLM_TORCHAX_ENABLED is True."
    # Validate that if mesh is None, sharding must also be None
    if mesh is None and (partition_spec is not None and partition_spec != ()):
        raise ValueError(
            "If mesh is None, partition_spec must also be None or empty")

    # Create CPU tensor first then move to jax device.
    cpu_device = jax.devices("cpu")[0]
    with jax.default_device(cpu_device):
        if weight_t.dtype == torch.bfloat16:
            jax_t = jnp.array(weight_t.to(torch.float32).numpy()).astype(
                jnp.bfloat16)
        else:
            jax_t = jnp.array(weight_t.numpy())

    if mesh is None:
        device = jax.devices()[0]
    else:
        partition_spec = partition_spec or ()
        device = NamedSharding(mesh, P(*partition_spec))

    jax_t = jax.device_put(jax_t, device)

    torchax_t = torchax.tensor.Tensor(jax_t, torchax.default_env())
    return torchax_t


class XlaQKVParallelLinear(nn.Module):

    def __init__(self,
                 qkv_linear: nn.Module,
                 mesh: Optional["xs.Mesh"] = None):
        super().__init__()
        assert isinstance(qkv_linear, QKVParallelLinear)
        self.skip_bias_add = qkv_linear.skip_bias_add
        self.return_bias = qkv_linear.return_bias
        assert qkv_linear.tp_size == 1, (
            "Tensor parallelism (TP) > 1 is only supported under SPMD mode")

        self.q_weight: Parameter
        self.k_weight: Parameter
        self.v_weight: Parameter
        self.q_bias: Optional[Parameter]
        self.k_bias: Optional[Parameter]
        self.v_bias: Optional[Parameter]
        self._load_weights_from_qkv_linear(qkv_linear)
        if mesh is not None:
            self._shard_weight(mesh)

    def _shard_weight(self, mesh: "xs.Mesh"):
        self.q_weight = Parameter(create_torchax_tensor_with_partition_spec(
            self.q_weight, mesh, ('x', None)),
                                  requires_grad=False)
        self.k_weight = Parameter(create_torchax_tensor_with_partition_spec(
            self.k_weight, mesh, ('x', None)),
                                  requires_grad=False)
        self.v_weight = Parameter(create_torchax_tensor_with_partition_spec(
            self.v_weight, mesh, ('x', None)),
                                  requires_grad=False)

        if self.q_bias is not None:
            assert self.k_bias is not None and self.v_bias is not None, \
                "QKVParallelLinear should have q, k, and v biases together."
            self.q_bias = Parameter(create_torchax_tensor_with_partition_spec(
                self.q_bias, mesh, ('x', )),
                                    requires_grad=False)
            self.k_bias = Parameter(create_torchax_tensor_with_partition_spec(
                self.k_bias, mesh, ('x', )),
                                    requires_grad=False)
            self.v_bias = Parameter(create_torchax_tensor_with_partition_spec(
                self.v_bias, mesh, ('x', )),
                                    requires_grad=False)

    def _load_weights_from_qkv_linear(self, qkv_linear: nn.Module):
        q_proj_size, k_proj_size, _ = qkv_linear.output_sizes
        # The weight of qkv linear is a concatenation of q, k, and v weights
        # along the output dimension.
        qkv_weight = qkv_linear.weight.data.cpu()
        q_weight = Parameter(qkv_weight[:q_proj_size], requires_grad=False)
        k_weight = Parameter(qkv_weight[q_proj_size:q_proj_size + k_proj_size],
                             requires_grad=False)
        v_weight = Parameter(qkv_weight[q_proj_size + k_proj_size:],
                             requires_grad=False)
        self.register_parameter("q_weight", q_weight)
        self.register_parameter("k_weight", k_weight)
        self.register_parameter("v_weight", v_weight)

        if qkv_linear.bias is not None:
            q_bias = Parameter(qkv_linear.bias[:q_proj_size],
                               requires_grad=False)
            k_bias = Parameter(qkv_linear.bias[q_proj_size:q_proj_size +
                                               k_proj_size],
                               requires_grad=False)
            v_bias = Parameter(qkv_linear.bias[q_proj_size + k_proj_size:],
                               requires_grad=False)
            self.register_parameter("q_bias", q_bias)
            self.register_parameter("k_bias", k_bias)
            self.register_parameter("v_bias", v_bias)
        else:
            self.register_parameter("q_bias", None)
            self.register_parameter("k_bias", None)
            self.register_parameter("v_bias", None)

    def forward(self, input):
        """Forward pass performing QKV projection separately then concatenating.

        This maintains compatibility with QKVParallelLinear's output format
        while enabling individual weight sharding for Q, K, and V projections.
        """
        q_bias = self.q_bias if not self.skip_bias_add else None
        k_bias = self.k_bias if not self.skip_bias_add else None
        v_bias = self.v_bias if not self.skip_bias_add else None
        q_proj = F.linear(input, self.q_weight, q_bias)
        k_proj = F.linear(input, self.k_weight, k_bias)
        v_proj = F.linear(input, self.v_weight, v_bias)
        # The Q/K/V projections will be split outside of the QKVParallelLinear.
        # Because we are replacing QKVParallelLinear with XlaQKVParallelLinear,
        # we need to concatenate q, k, and v projections to match the output
        # shape of the QKVParallelLinear implementation even if it seems
        # redundant.
        # The concat and the following split will be no-op, and should be
        # optimized away by the compiler.
        qkv_proj = torch.cat([q_proj, k_proj, v_proj], dim=-1)
        output_bias = torch.cat([q_bias, k_bias, v_bias], dim=-1) if \
                            self.skip_bias_add else None
        if not self.return_bias:
            return qkv_proj
        return qkv_proj, output_bias


class XlaMergedColumnParallelLinear(nn.Module):

    def __init__(self,
                 merged_col_parallel_linear: nn.Module,
                 mesh: Optional["xs.Mesh"] = None):
        super().__init__()
        assert isinstance(merged_col_parallel_linear,
                          MergedColumnParallelLinear)
        self.output_sizes = merged_col_parallel_linear.output_sizes
        self.n_linear_layers = len(self.output_sizes)
        self.skip_bias_add = merged_col_parallel_linear.skip_bias_add
        self.return_bias = merged_col_parallel_linear.return_bias
        assert merged_col_parallel_linear.tp_size == 1, (
            "TP > 1 is only supported under SPMD.")

        self._load_weights_from_merged_linear(merged_col_parallel_linear)
        if mesh is not None:
            self._shard_weight(mesh)

    def _shard_weight(self, mesh: "xs.Mesh"):
        # Shard all weights in the weight_list
        for i in range(self.n_linear_layers):
            weight = getattr(self, f"weight_{i}")
            sharded_weight = Parameter(
                create_torchax_tensor_with_partition_spec(
                    weight, mesh, ('x', None)),
                requires_grad=weight.requires_grad)
            setattr(self, f"weight_{i}", sharded_weight)

        if self.bias_0 is not None:
            for i, _ in enumerate(self.output_sizes):
                bias = getattr(self, f"bias_{i}")
                sharded_bias = Parameter(
                    create_torchax_tensor_with_partition_spec(
                        bias, mesh, ('x', )),
                    requires_grad=bias.requires_grad)
                setattr(self, f"bias_{i}", sharded_bias)

    def _load_weights_from_merged_linear(
            self, merged_col_parallel_linear: nn.Module):
        output_sizes = merged_col_parallel_linear.output_sizes
        concat_weight = merged_col_parallel_linear.weight.data.cpu()
        concat_bias = None
        if merged_col_parallel_linear.bias is not None:
            concat_bias = merged_col_parallel_linear.bias.data.cpu()
        start_offset = 0
        for i, size in enumerate(output_sizes):
            weight = Parameter(concat_weight[start_offset:start_offset + size],
                               requires_grad=False)
            setattr(self, f"weight_{i}", weight)

            if concat_bias is not None:
                bias = Parameter(concat_bias[start_offset:start_offset + size],
                                 requires_grad=False)
                setattr(self, f"bias_{i}", bias)
            else:
                setattr(self, f"bias_{i}", None)

            start_offset += size

    def forward(self, input):
        # Apply linear transformations for each weight/bias pair
        outputs = []
        for i, _ in enumerate(self.output_sizes):
            output = F.linear(input, getattr(self, f"weight_{i}"),
                              getattr(self, f"bias_{i}"))
            outputs.append(output)

        # Concatenate all outputs
        merged_output = torch.cat(outputs, dim=-1)

        # Handle bias return if needed
        if self.return_bias:
            output_bias = None
            if self.bias_0 is not None:
                output_bias = torch.cat([
                    getattr(self, f"bias_{i}")
                    for i, _ in enumerate(self.output_sizes)
                ],
                                        dim=-1)
            return merged_output, output_bias

        return merged_output


def partition_column_parallel_linear(layer: torch.nn.Module,
                                     mesh: xs.Mesh) -> torch.nn.Module:
    assert isinstance(layer, ColumnParallelLinear)
    if VLLM_TORCHAX_ENABLED:
        torchax_t = create_torchax_tensor_with_partition_spec(
            layer.weight.data, mesh, ('x', None))
        layer.weight = Parameter(torchax_t,
                                 requires_grad=layer.weight.requires_grad)
        logger.info("Applied column-parallel sharding to %s", layer)
    else:
        xs.mark_sharding(layer.weight, mesh, ('x', None))
        logger.info("Applied column-parallel sharding to %s", layer)
    return layer


def partition_row_parallel_linear(layer: torch.nn.Module,
                                  mesh: xs.Mesh) -> torch.nn.Module:
    assert isinstance(layer, RowParallelLinear)
    if VLLM_TORCHAX_ENABLED:

        def shard_output_hook(module, input, output):
            sharding = NamedSharding(mesh, P('x', None))
            new_output = output[0].apply_jax(jax.lax.with_sharding_constraint,
                                             sharding)
            return (new_output, output[1])

        torchax_t = create_torchax_tensor_with_partition_spec(
            layer.weight.data, mesh, (None, 'x'))
        layer.weight = Parameter(torchax_t,
                                 requires_grad=layer.weight.requires_grad)
        # layer.register_forward_hook(shard_output_hook)
        logger.info("Applied row-parallel sharding to %s", layer)
    else:
        xs.mark_sharding(layer.weight, mesh, (None, 'x'))
        logger.info("Applied row-parallel sharding to %s", layer)
    return layer


def partition_qkv_parallel_linear(layer: torch.nn.Module,
                                  mesh: xs.Mesh) -> torch.nn.Module:
    assert isinstance(layer, QKVParallelLinear)
    xla_layer = XlaQKVParallelLinear(layer, mesh)
    logger.info("Applied qkv parallel sharding to %s", layer)
    return xla_layer


def partition_merged_col_parallel_linear(layer: torch.nn.Module,
                                         mesh: xs.Mesh) -> torch.nn.Module:
    assert isinstance(layer, MergedColumnParallelLinear)
    xla_layer = XlaMergedColumnParallelLinear(layer, mesh)
    logger.info("Applied merged column parallel sharding to %s", layer)
    return xla_layer


def replicate_weights_buffers(module: torch.nn.Module,
                              mesh: "xs.Mesh") -> None:
    logger.info("Replicating weights and buffers for module %s", module)
    for name, param in module.named_parameters(recurse=False):
        torchax_t = create_torchax_tensor_with_partition_spec(
            param.data, mesh, ())
        torchax_param = Parameter(torchax_t, requires_grad=param.requires_grad)
        setattr(module, name, torchax_param)

    for name, buffer in module.named_buffers(recurse=False):
        if isinstance(buffer, torchax.tensor.Tensor):
            # If the parameter is already a torchax tensor, we can skip
            # replication.
            logger.info("parameter %s is already a torchax tensor, skipping",
                        name)
            continue
        logger.info("replicating buffer %s, buffer is on device %s", name,
                    buffer.device)
        torchax_t = create_torchax_tensor_with_partition_spec(buffer, mesh, ())
        # TODO: handle persistent buffer
        setattr(module, name, torchax_t)


MODULE_TYPE_TO_WRAPPING_FUNC = OrderedDict([
    ("QKVParallelLinear", partition_qkv_parallel_linear),
    ("ColumnParallelLinear", partition_column_parallel_linear),
    ("MergedColumnParallelLinear", partition_merged_col_parallel_linear),
    ("RowParallelLinear", partition_row_parallel_linear),
])


def get_fqn(module) -> str:
    # Get the fully qualified name of the module
    return module.__class__.__qualname__


def shard_model(model: torch.nn.Module, mesh: "xs.Mesh") -> None:
    """
    Recursively check a PyTorch model and apply appropriate sharding based on
    the MODULE_TYPE_TO_WRAPPING_FUNC mapping.

    Args:
        model: torch.nn.Module to process
        mesh: An XLA SPMD mesh object used for sharding
    """

    def _process_module(module, name=None, parent=None):
        model_processed = False
        for module_type, wrapping_func in MODULE_TYPE_TO_WRAPPING_FUNC.items():
            if get_fqn(module) == module_type:
                logger.info("processing module %s with type %s", module,
                            module_type)
                wrapped_module = wrapping_func(module, mesh)

                assert parent is not None and name is not None, (
                    "Top Level module is not expected to be wrapped.")
                if wrapped_module is not module:
                    # Wrapped module and module are different py object.
                    # The original module should be replaced by the
                    # wrapped_module.
                    logger.info("replace %s with %s", module, wrapped_module)
                    setattr(parent, name, wrapped_module)

                module = wrapped_module
                model_processed = True
                break

        # Replicate the weights and buffers if the module is not processed
        if not model_processed and VLLM_TORCHAX_ENABLED and mesh is not None:
            replicate_weights_buffers(module, mesh)

        for child_name, child_module in list(module.named_children()):
            _process_module(child_module, child_name, module)

    # for name, tensor in model.named_parameters():
    #     logger.info("weight %s: %s %s", name, tensor.shape, tensor.dtype)

    # for name, tensor in model.named_buffers():
    #     logger.info("buffer %s: %s %s", name, tensor.shape, tensor.dtype)

    assert mesh is not None, "Mesh must be provided for sharding."
    _process_module(model)
