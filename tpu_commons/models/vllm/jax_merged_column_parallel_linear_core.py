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
from tpu_commons.utils import TPU_SECOND_LAST_MINOR

P = PartitionSpec


class JaxMergedColumnParallelLinearCore(torch.nn.Module):
    """ A common class to implement Column Parallel Linear layer whose weight are merged from a list of smaller weight tensors, e.g. vLLM's MergedColumnParallelLinear and QKVParallelLinear layer. """

    def __init__(self, vllm_col_par_linear: torch.nn.Module, mesh: Mesh,
                 name: str, fuse_matmuls: bool, enable_sequence_parallelism):
        super().__init__()

        self.gather_output = vllm_col_par_linear.gather_output
        self.skip_bias_add = vllm_col_par_linear.skip_bias_add
        self.return_bias = vllm_col_par_linear.return_bias
        self.output_sizes = vllm_col_par_linear.output_sizes
        self.mesh = mesh
        self.name = name
        self.fuse_matmuls = fuse_matmuls
        self.has_bias = vllm_col_par_linear.bias is not None
        self.enable_sequence_parallelism = enable_sequence_parallelism
        self.n_matmuls = len(self.output_sizes)
        assert vllm_col_par_linear.tp_size == 1, (
            "The model has to be loaded with TP== 1 in order to run in Jax SPMD."
        )

        self.w8q8_int8_quant = False
        if isinstance(vllm_col_par_linear.quant_method,
                      CompressedTensorsLinearMethod) and isinstance(
                          vllm_col_par_linear.scheme,
                          CompressedTensorsW8A8Int8):
            self.w8q8_int8_quant = True

        if self.fuse_matmuls:
            self._load_weights_from_merged_linear_fused(vllm_col_par_linear)
            self._shard_weight_fused(mesh)
        else:
            self._load_weights_from_merged_linear_split(vllm_col_par_linear)
            self._shard_weight_split(mesh)

    def _shard_weight_fused(self, mesh: Mesh):
        self.weight.apply_jax_(jax.device_put,
                               NamedSharding(mesh, P('model', None)))

        if self.bias is not None:
            self.bias.apply_jax_(jax.device_put,
                                 NamedSharding(mesh, P('model')))

        if self.w8q8_int8_quant:
            self.weight_scale.apply_jax_(jax.device_put,
                                         NamedSharding(mesh, P('model')))

    def _shard_weight_split(self, mesh: Mesh):
        # Shard all weights in the weight_list
        for i in range(self.n_matmuls):
            weight = getattr(self, f"weight_{i}")
            weight.apply_jax_(jax.device_put,
                              NamedSharding(mesh, P('model', None)))
            setattr(self, f"weight_{i}", weight)

        if self.has_bias:
            for i in range(self.n_matmuls):
                bias = getattr(self, f"bias_{i}")
                bias.apply_jax_(jax.device_put,
                                NamedSharding(mesh, P('model', )))
                setattr(self, f"bias_{i}", bias)

        if self.w8q8_int8_quant:
            for i in range(self.n_matmuls):
                weight_scale = getattr(self, f"weight_scale_{i}")
                weight_scale.apply_jax_(jax.device_put,
                                        NamedSharding(mesh, P('model')))
                setattr(self, f"weight_scale_{i}", weight_scale)

    def _load_weights_from_merged_linear_fused(
            self, vllm_col_par_linear: torch.nn.Module):
        n_shards = self.mesh.shape['model']
        for _, output_size in enumerate(self.output_sizes):
            assert output_size % n_shards == 0, "Each output size in MergedColumnParallelLinear must be a  multiple of num chips in the 'model' axis."

        concat_weight = t2j(vllm_col_par_linear.weight.data, use_dlpack=False)
        weight = reorder_concatenated_tensor_for_sharding(concat_weight,
                                                          self.output_sizes,
                                                          n_shards,
                                                          dim=0)
        weight = Parameter(torch_view(weight), requires_grad=False)
        self.register_parameter("weight", weight)

        if vllm_col_par_linear.bias is not None:
            concat_bias = t2j(vllm_col_par_linear.bias.data, use_dlpack=False)
            bias = reorder_concatenated_tensor_for_sharding(concat_bias,
                                                            self.output_sizes,
                                                            n_shards,
                                                            dim=0)
            bias = Parameter(torch_view(bias), requires_grad=False)
            self.register_parameter("bias", bias)
        else:
            self.register_parameter("bias", None)

        if self.w8q8_int8_quant:
            assert self.weight.jax().dtype == jnp.int8
            concat_weight_scale = t2j(vllm_col_par_linear.weight_scale.data,
                                      use_dlpack=False)
            weight_scale = reorder_concatenated_tensor_for_sharding(
                concat_weight_scale, self.output_sizes, n_shards, dim=0)
            weight_scale = Parameter(torch_view(weight_scale),
                                     requires_grad=False)
            self.register_parameter("weight_scale", weight_scale)
        else:
            self.register_parameter("weight_scale", None)

    def _load_weights_from_merged_linear_split(
            self, vllm_col_par_linear: torch.nn.Module):
        output_sizes = vllm_col_par_linear.output_sizes
        concat_weight = torch_view(
            t2j(vllm_col_par_linear.weight.data, use_dlpack=False))
        concat_bias = None
        if self.has_bias:
            concat_bias = torch_view(
                t2j(vllm_col_par_linear.bias.data, use_dlpack=False))
        if self.w8q8_int8_quant:
            concat_weight_scale = torch_view(
                t2j(vllm_col_par_linear.weight_scale.data, use_dlpack=False))
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

    def forward_fused(self, input: torch.Tensor):
        x = input.jax()
        weight = self.weight.jax()
        bias = None if (self.skip_bias_add
                        or self.bias is None) else self.bias.jax()
        if self.w8q8_int8_quant:
            weight_scale = self.weight_scale.jax(
            ) if self.w8q8_int8_quant else None
            output = forward_w8a8_int8_col_parallel(x, weight, bias,
                                                    weight_scale, self.mesh)
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

    def forward_split(self, input):
        x = input.jax()
        split_outputs = []
        for i in range(self.n_matmuls):
            weight = getattr(self, f"weight_{i}").jax()
            bias = getattr(self, f"bias_{i}")
            bias = None if (self.skip_bias_add or bias is None) else bias.jax()
            if self.w8q8_int8_quant:
                weight_scale = getattr(self, f"weight_scale_{i}").jax()
                output = forward_w8a8_int8_col_parallel(
                    x, weight, bias, weight_scale, self.mesh)
            else:
                output = forward_unqunatized(x, weight, bias)
            split_outputs.append(output)
        if self.gather_output:
            split_outputs = [
                jax.lax.with_sharding_constraint(t,
                                                 NamedSharding(self.mesh, P()))
                for t in split_outputs
            ]
        output = torch_view(jnp.concatenate(split_outputs, axis=-1))

        if not self.return_bias:
            return output

        if self.skip_bias_add or not self.has_bias:
            output_bias = None
        else:
            split_biases = [
                getattr(self, f"bias_{i}").jax()
                for i, _ in enumerate(self.output_sizes)
            ]
            output_bias = torch_view(jnp.concatenate(split_biases, axis=-1))
        return output, output_bias

    def forward(self, input: torch.Tensor):
        with jax.named_scope(self.name):
            if self.enable_sequence_parallelism:
                token_num = input.shape[0]
                # NOTE(chengjiyao): make sure the sharded token_num is larger than TPU_SECOND_LAST_MINOR
                if token_num // self.mesh.shape[
                        'model'] >= TPU_SECOND_LAST_MINOR:
                    input.shard_(NamedSharding(self.mesh, P('model', None)))
            if self.fuse_matmuls:
                output, output_bias = self.forward_fused(input)
            else:
                output, output_bias = self.forward_split(input)
            return output, output_bias
