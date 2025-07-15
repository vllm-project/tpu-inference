from typing import Optional

import jax
import torch
import torch.nn.functional as F
from jax.sharding import Mesh, NamedSharding, PartitionSpec
from torch.nn.parameter import Parameter
from torchax.interop import torch_view
from torchax.ops.mappings import t2j
from vllm.model_executor.layers.linear import MergedColumnParallelLinear

P = PartitionSpec

class JaxMergedColumnParallelLinear(torch.nn.Module):

    def __init__(self,
                 merged_col_parallel_linear: torch.nn.Module,
                 mesh: Mesh):
        super().__init__()
        assert isinstance(merged_col_parallel_linear,
                          MergedColumnParallelLinear)
        self.output_sizes = merged_col_parallel_linear.output_sizes
        self.n_linear_layers = len(self.output_sizes)
        self.has_bias = merged_col_parallel_linear.bias is not None
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
            weight.apply_jax_(jax.device_put,
                NamedSharding(mesh, P('model', None)))
            setattr(self, f"weight_{i}", weight)

        if self.has_bias:
            for i, _ in enumerate(self.output_sizes):
                bias = getattr(self, f"bias_{i}")
                bias.apply_jax_(jax.device_put,
                                NamedSharding(mesh, P('model',)))
                setattr(self, f"bias_{i}", bias)

    def _load_weights_from_merged_linear(
            self, merged_col_parallel_linear: torch.nn.Module):
        output_sizes = merged_col_parallel_linear.output_sizes
        concat_weight = torch_view(t2j(merged_col_parallel_linear.weight.data))
        concat_bias = None
        if self.has_bias:
            concat_bias = torch_view(t2j(merged_col_parallel_linear.bias.data))
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
