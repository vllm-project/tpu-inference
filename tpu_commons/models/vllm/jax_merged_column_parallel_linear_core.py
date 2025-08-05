from typing import Optional

import jax
import jax.numpy as jnp
import torch
from jax.sharding import Mesh, NamedSharding, PartitionSpec
from torch.nn.parameter import Parameter
from torchax.interop import torch_view
from torchax.ops.mappings import t2j
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import \
    CompressedTensorsLinearMethod
from vllm.model_executor.layers.quantization.compressed_tensors.schemes import \
    CompressedTensorsW8A8Int8

from tpu_commons.models.vllm.jax_linear_common import (
    forward_unqunatized, forward_w8a8_int8_col_parallel,
    reorder_concatenated_tensor_for_sharding,
    slice_sharded_tensor_for_concatenation)

P = PartitionSpec


class JaxMergedColumnParallelLinearCore(torch.nn.Module):
    """ A common class to implement Column Parallel Linear layer whose weight are merged from a list of smaller weight tensors, e.g. vLLM's MergedColumnParallelLinear and QKVParallelLinear layer. """

    def __init__(self, vllm_col_par_linear: torch.nn.Module, mesh: Mesh,
                 name: str):
        super().__init__()

        self.gather_output = vllm_col_par_linear.gather_output
        self.skip_bias_add = vllm_col_par_linear.skip_bias_add
        self.return_bias = vllm_col_par_linear.return_bias
        self.output_sizes = vllm_col_par_linear.output_sizes
        self.mesh = mesh
        self.name = name
        assert vllm_col_par_linear.tp_size == 1, (
            "The model has to be loaded with TP== 1 in order to run in Jax SPMD."
        )

        self.w8q8_int8_quant = False
        if isinstance(vllm_col_par_linear.quant_method,
                      CompressedTensorsLinearMethod) and isinstance(
                          vllm_col_par_linear.scheme,
                          CompressedTensorsW8A8Int8):
            self.w8q8_int8_quant = True

        self.weight: Parameter
        self.bias: Optional[Parameter]
        self.weight_scale: Optional[Parameter]

        self._load_weights_from_merged_linear(vllm_col_par_linear)
        self._shard_weight(mesh)

    def _shard_weight(self, mesh: Mesh):
        self.weight.apply_jax_(jax.device_put,
                               NamedSharding(mesh, P('model', None)))

        if self.bias is not None:
            self.bias.apply_jax_(jax.device_put,
                                 NamedSharding(mesh, P('model')))

        if self.w8q8_int8_quant:
            self.weight_scale.apply_jax_(jax.device_put,
                                         NamedSharding(mesh, P('model')))

    def _load_weights_from_merged_linear(self,
                                         vllm_col_par_linear: torch.nn.Module):
        n_shards = self.mesh.shape['model']
        for _, output_size in enumerate(self.output_sizes):
            assert output_size % n_shards == 0, "Each output size in MergedColumnParallelLinear must be a  multiple of num chips in the 'model' axis."

        concat_weight = t2j(vllm_col_par_linear.weight.data, use_dlpack=False)
        weight = reorder_concatenated_tensor_for_sharding(
            concat_weight, self.output_sizes, n_shards)
        weight = Parameter(torch_view(weight), requires_grad=False)
        self.register_parameter("weight", weight)

        if vllm_col_par_linear.bias is not None:
            concat_bias = t2j(vllm_col_par_linear.bias.data, use_dlpack=False)
            bias = reorder_concatenated_tensor_for_sharding(
                concat_bias, self.output_sizes, n_shards)
            bias = Parameter(torch_view(bias), requires_grad=False)
            self.register_parameter("bias", bias)
        else:
            self.register_parameter("bias", None)

        if self.w8q8_int8_quant:
            assert self.weight.jax().dtype == jnp.int8
            concat_weight_scale = t2j(vllm_col_par_linear.weight_scale.data,
                                      use_dlpack=False)
            weight_scale = reorder_concatenated_tensor_for_sharding(
                concat_weight_scale, self.output_sizes, n_shards)
            weight_scale = Parameter(torch_view(weight_scale),
                                     requires_grad=False)
            self.register_parameter("weight_scale", weight_scale)
        else:
            self.register_parameter("weight_scale", None)

    def forward(self, input: torch.Tensor):
        with jax.named_scope(self.name):
            x = input.jax()
            weight = self.weight.jax()
            bias = None if (self.skip_bias_add
                            or self.bias is None) else self.bias.jax()
            if self.w8q8_int8_quant:
                weight_scale = self.weight_scale.jax(
                ) if self.w8q8_int8_quant else None
                output = forward_w8a8_int8_col_parallel(
                    x, weight, bias, weight_scale, self.mesh)
            else:
                output = forward_unqunatized(x, weight, bias)

            n_shards = self.mesh.shape['model']
            split_outputs = slice_sharded_tensor_for_concatenation(
                output, self.output_sizes, n_shards, self.mesh)
            if self.gather_output:
                split_outputs = [
                    jax.lax.with_sharding_constraint(
                        t, NamedSharding(self.mesh, P()))
                    for t in split_outputs
                ]
            output = torch_view(jnp.concatenate(split_outputs, axis=-1))

            if not self.return_bias:
                return output

            if self.skip_bias_add or self.bias is None:
                output_bias = None
            else:
                split_biases = slice_sharded_tensor_for_concatenation(
                    self.bias, self.output_sizes, n_shards, self.mesh)
                output_bias = torch_view(jnp.concatenate(split_biases,
                                                         axis=-1))
            return output, output_bias
