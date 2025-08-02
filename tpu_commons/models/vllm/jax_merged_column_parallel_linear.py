import jax
import jax.numpy as jnp
import torch
from jax.sharding import Mesh, NamedSharding, PartitionSpec
from torch.nn.parameter import Parameter
from torchax.interop import torch_view
from torchax.ops.mappings import t2j
from vllm.model_executor.layers.linear import MergedColumnParallelLinear
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import \
    CompressedTensorsLinearMethod
from vllm.model_executor.layers.quantization.compressed_tensors.schemes import \
    CompressedTensorsW8A8Int8

from tpu_commons.models.vllm.jax_linear_common import (
    ParallelType, forward_unqunatized, forward_w8a8_int8,
    reorder_concatenated_tensor_for_sharding,
    slice_sharded_tensor_for_concatenation)

P = PartitionSpec


class JaxMergedColumnParallelLinear(torch.nn.Module):

    def __init__(self, merged_col_parallel_linear: torch.nn.Module,
                 mesh: Mesh):
        super().__init__()
        assert isinstance(merged_col_parallel_linear,
                          MergedColumnParallelLinear)

        self.mesh = mesh
        self.gather_output = merged_col_parallel_linear.gather_output
        self.output_sizes = merged_col_parallel_linear.output_sizes
        self.n_linear_layers = len(self.output_sizes)
        self.has_bias = merged_col_parallel_linear.bias is not None
        self.skip_bias_add = merged_col_parallel_linear.skip_bias_add
        self.return_bias = merged_col_parallel_linear.return_bias
        self.output_sizes = merged_col_parallel_linear.output_sizes
        assert merged_col_parallel_linear.tp_size == 1, (
            "TP > 1 is only supported under SPMD.")

        self.w8q8_int8_quant = False
        if isinstance(merged_col_parallel_linear.quant_method,
                      CompressedTensorsLinearMethod) and isinstance(
                          merged_col_parallel_linear.scheme,
                          CompressedTensorsW8A8Int8):
            self.w8q8_int8_quant = True

        self._load_weights_from_merged_linear(merged_col_parallel_linear)
        if mesh is not None:
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

    def _load_weights_from_merged_linear(
            self, merged_col_parallel_linear: torch.nn.Module):
        n_shards = self.mesh.shape['model']
        for _, output_size in enumerate(self.output_sizes):
            assert output_size % n_shards == 0, "Each output size in MergedColumnParallelLinear must be a  multiple of num chips in the 'model' axis."

        concat_weight = t2j(merged_col_parallel_linear.weight.data,
                            use_dlpack=False)
        weight = reorder_concatenated_tensor_for_sharding(
            concat_weight, self.output_sizes, n_shards)
        weight = Parameter(torch_view(weight), requires_grad=False)
        self.register_parameter("weight", weight)

        if self.has_bias:
            concat_bias = t2j(merged_col_parallel_linear.bias.data,
                              use_dlpack=False)
            bias = reorder_concatenated_tensor_for_sharding(
                concat_bias, self.output_sizes, n_shards)
            bias = Parameter(torch_view(bias), requires_grad=False)
            self.register_parameter("bias", bias)
        else:
            self.register_parameter("bias", None)

        if self.w8q8_int8_quant:
            assert self.weight.jax().dtype == jnp.int8
            concat_weight_scale = t2j(
                merged_col_parallel_linear.weight_scale.data, use_dlpack=False)
            weight_scale = reorder_concatenated_tensor_for_sharding(
                concat_weight_scale, self.output_sizes, n_shards)
            weight_scale = Parameter(torch_view(weight_scale),
                                     requires_grad=False)
            self.register_parameter("weight_scale", weight_scale)
        else:
            self.register_parameter("weight_scale", None)

    def forward(self, input):
        x = input.jax()
        weight = self.weight.jax()
        bias = None if (self.skip_bias_add
                        or self.bias is None) else self.bias.jax()
        if self.w8q8_int8_quant:
            weight_scale = self.weight_scale.jax(
            ) if self.w8q8_int8_quant else None
            output = forward_w8a8_int8(x, weight, bias, weight_scale,
                                       self.mesh, False,
                                       ParallelType.COL_PARALLEL)
        else:
            output = forward_unqunatized(x, weight, bias)

        n_shards = self.mesh.shape['model']
        split_outputs = slice_sharded_tensor_for_concatenation(
            output, self.output_sizes, n_shards, self.mesh)
        if self.gather_output:
            split_outputs = [
                jax.lax.with_sharding_constraint(t,
                                                 NamedSharding(self.mesh, P()))
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
            output_bias = torch_view(jnp.concatenate(split_biases, axis=-1))
        return output, output_bias
