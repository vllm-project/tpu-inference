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

from tpu_commons.models.vllm.jax_linear_common import (ParallelType,
                                                       forward_unqunatized,
                                                       forward_w8a8_int8)

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
        # Shard all weights in the weight_list
        for i in range(self.n_linear_layers):
            weight = getattr(self, f"weight_{i}")
            weight.apply_jax_(jax.device_put,
                              NamedSharding(mesh, P('model', None)))
            setattr(self, f"weight_{i}", weight)

        if self.has_bias:
            for i, _ in enumerate(self.output_sizes):
                bias = getattr(self, f"bias_{i}")
                bias.apply_jax_(jax.device_put,
                                NamedSharding(mesh, P('model', )))
                setattr(self, f"bias_{i}", bias)

        if self.w8q8_int8_quant:
            for i, _ in enumerate(self.output_sizes):
                weight_scale = getattr(self, f"weight_scale_{i}")
                weight_scale.apply_jax_(jax.device_put,
                                        NamedSharding(mesh, P('model')))
                setattr(self, f"weight_scale_{i}", weight_scale)

    def _load_weights_from_merged_linear(
            self, merged_col_parallel_linear: torch.nn.Module):
        output_sizes = merged_col_parallel_linear.output_sizes
        concat_weight = torch_view(
            t2j(merged_col_parallel_linear.weight.data, use_dlpack=False))
        concat_bias = None
        if self.has_bias:
            concat_bias = torch_view(
                t2j(merged_col_parallel_linear.bias.data, use_dlpack=False))
        if self.w8q8_int8_quant:
            concat_weight_scale = torch_view(
                t2j(merged_col_parallel_linear.weight_scale.data,
                    use_dlpack=False))
        start_offset = 0
        for i, size in enumerate(output_sizes):
            weight = Parameter(concat_weight[start_offset:start_offset +
                                             size].detach(),
                               requires_grad=False)
            setattr(self, f"weight_{i}", weight)

            if concat_bias is not None:
                bias = Parameter(concat_bias[start_offset:start_offset +
                                             size].detach(),
                                 requires_grad=False)
                setattr(self, f"bias_{i}", bias)
            else:
                setattr(self, f"bias_{i}", None)

            if self.w8q8_int8_quant:
                assert weight.jax().dtype == jnp.int8
                weight_scale = Parameter(
                    concat_weight_scale[start_offset:start_offset +
                                        size].detach(),
                    requires_grad=False)
                setattr(self, f"weight_scale_{i}", weight_scale)
            else:
                setattr(self, f"weight_scale_{i}", None)

            start_offset += size

    def forward(self, input):
        x = input.jax()
        # Apply linear transformations for each weight/bias pair
        outputs = []
        for i, _ in enumerate(self.output_sizes):
            weight = getattr(self, f"weight_{i}").jax()
            bias = getattr(self, f"bias_{i}")
            bias = None if (self.skip_bias_add or bias is None) else bias.jax()
            if self.w8q8_int8_quant:
                weight_scale = getattr(self, f"weight_scale_{i}").jax()
                output = torch_view(
                    forward_w8a8_int8(x, weight, bias, weight_scale, self.mesh,
                                      self.gather_output,
                                      ParallelType.COL_PARALLEL))
            else:
                output = torch_view(forward_unqunatized(x, weight, bias))
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
