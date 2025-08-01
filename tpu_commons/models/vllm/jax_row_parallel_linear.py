from typing import Optional

import jax
import jax.numpy as jnp
import torch
from jax.sharding import Mesh, NamedSharding, PartitionSpec
from torch.nn.parameter import Parameter
from torchax.interop import torch_view
from torchax.ops.mappings import t2j
from vllm.model_executor.layers.linear import RowParallelLinear
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import \
    CompressedTensorsLinearMethod
from vllm.model_executor.layers.quantization.compressed_tensors.schemes import \
    CompressedTensorsW8A8Int8

from tpu_commons.models.vllm.jax_linear_common import (ParallelType,
                                                       forward_unqunatized,
                                                       forward_w8a8_int8)

P = PartitionSpec


class JaxRowParallelLinear(torch.nn.Module):

    def __init__(self, row_linear: torch.nn.Module, mesh: Mesh):
        super().__init__()
        assert isinstance(row_linear, RowParallelLinear)

        self.mesh = mesh
        self.reduce_results = row_linear.reduce_results
        self.skip_bias_add = row_linear.skip_bias_add
        self.return_bias = row_linear.return_bias

        self.w8q8_int8_quant = False
        if isinstance(row_linear.quant_method,
                      CompressedTensorsLinearMethod) and isinstance(
                          row_linear.scheme, CompressedTensorsW8A8Int8):
            self.w8q8_int8_quant = True

        self.weight: Parameter
        self.bias: Optional[Parameter]
        self.weight_scale: Optional[Parameter]

        self._load_weights_from_vllm_layer(row_linear)
        self._shard_weight(mesh)

    def _shard_weight(self, mesh: Mesh):
        self.weight.apply_jax_(jax.device_put,
                               NamedSharding(mesh, P(None, 'model')))

        if self.bias is not None:
            self.bias.apply_jax_(jax.device_put, NamedSharding(
                mesh, P()))  # column parallel can't shard the bias

        if self.w8q8_int8_quant:
            self.weight_scale.apply_jax_(jax.device_put,
                                         NamedSharding(mesh, P()))

    def _load_weights_from_vllm_layer(self, row_linear: torch.nn.Module):
        weight = Parameter(torch_view(
            t2j(row_linear.weight.data, use_dlpack=False)),
                           requires_grad=False)
        self.register_parameter("weight", weight)

        if row_linear.bias is not None:
            bias = Parameter(torch_view(
                t2j(row_linear.bias.data, use_dlpack=False)),
                             requires_grad=False)
            self.register_parameter("bias", bias)
        else:
            self.register_parameter("bias", None)

        if self.w8q8_int8_quant:
            assert weight.jax().dtype == jnp.int8
            weight_scale = Parameter(torch_view(
                t2j(row_linear.weight_scale.data, use_dlpack=False)),
                                     requires_grad=False)
            self.register_parameter("weight_scale", weight_scale)
        else:
            self.register_parameter("weight_scale", None)

    def forward(self, input: torch.Tensor):
        x = input.jax()
        weight = self.weight.jax()
        bias = None if (self.skip_bias_add
                        or self.bias is None) else self.bias.jax()
        if self.w8q8_int8_quant:
            weight_scale = self.weight_scale.jax()
            output = torch_view(
                forward_w8a8_int8(x, weight, bias, weight_scale, self.mesh,
                                  self.reduce_results,
                                  ParallelType.ROW_PARALLEL))
        else:
            output = torch_view(forward_unqunatized(x, weight, bias))

        if not self.return_bias:
            return output
        output_bias = self.bias if self.skip_bias_add else None
        return output, output_bias
