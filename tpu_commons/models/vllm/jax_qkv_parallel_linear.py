from typing import Optional

import jax
import jax.numpy as jnp
import torch
from jax.sharding import Mesh, NamedSharding, PartitionSpec
from torch.nn.parameter import Parameter
from torchax.interop import torch_view
from torchax.ops.mappings import t2j
from vllm.model_executor.layers.linear import QKVParallelLinear
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import \
    CompressedTensorsLinearMethod
from vllm.model_executor.layers.quantization.compressed_tensors.schemes import \
    CompressedTensorsW8A8Int8

from tpu_commons.models.vllm.jax_linear_common import (ParallelType,
                                                       forward_unqunatized,
                                                       forward_w8a8_int8)

P = PartitionSpec


class JaxQKVParallelLinear(torch.nn.Module):

    def __init__(self, qkv_linear: torch.nn.Module, mesh: Mesh):
        super().__init__()
        assert isinstance(qkv_linear, QKVParallelLinear)

        self.mesh = mesh
        self.gather_output = qkv_linear.gather_output
        self.skip_bias_add = qkv_linear.skip_bias_add
        self.return_bias = qkv_linear.return_bias

        self.w8q8_int8_quant = False
        if isinstance(qkv_linear.quant_method,
                      CompressedTensorsLinearMethod) and isinstance(
                          qkv_linear.scheme, CompressedTensorsW8A8Int8):
            self.w8q8_int8_quant = True

        self.q_weight: Parameter
        self.k_weight: Parameter
        self.v_weight: Parameter
        self.q_bias: Optional[Parameter]
        self.k_bias: Optional[Parameter]
        self.v_bias: Optional[Parameter]
        self.q_weight_scale: Optional[Parameter]
        self.k_weight_scale: Optional[Parameter]
        self.v_weight_scale: Optional[Parameter]

        self._load_weights_from_vllm_layer(qkv_linear)
        self._shard_weight(mesh)

    def _shard_weight(self, mesh: Mesh):
        self.q_weight.apply_jax_(jax.device_put,
                                 NamedSharding(mesh, P('model', None)))
        self.k_weight.apply_jax_(jax.device_put,
                                 NamedSharding(mesh, P('model', None)))
        self.v_weight.apply_jax_(jax.device_put,
                                 NamedSharding(mesh, P('model', None)))

        if self.q_bias is not None:
            assert self.k_bias is not None and self.v_bias is not None, \
                "QKVParallelLinear should have q, k, and v biases together."
            self.q_bias.apply_jax_(jax.device_put,
                                   NamedSharding(mesh, P('model')))
            self.k_bias.apply_jax_(jax.device_put,
                                   NamedSharding(mesh, P('model')))
            self.v_bias.apply_jax_(jax.device_put,
                                   NamedSharding(mesh, P('model')))

        if self.w8q8_int8_quant:
            self.q_weight_scale.apply_jax_(jax.device_put,
                                           NamedSharding(mesh, P('model')))
            self.k_weight_scale.apply_jax_(jax.device_put,
                                           NamedSharding(mesh, P('model')))
            self.v_weight_scale.apply_jax_(jax.device_put,
                                           NamedSharding(mesh, P('model')))

    def _load_weights_from_vllm_layer(self, qkv_linear: torch.nn.Module):
        q_proj_size, k_proj_size, _ = qkv_linear.output_sizes
        # The weight of qkv linear is a concatenation of q, k, and v weights
        # along the output dimension.
        qkv_weight = torch_view(t2j(qkv_linear.weight.data, use_dlpack=False))
        q_weight = Parameter(qkv_weight[:q_proj_size].detach(),
                             requires_grad=False)
        k_weight = Parameter(qkv_weight[q_proj_size:q_proj_size +
                                        k_proj_size].detach(),
                             requires_grad=False)
        v_weight = Parameter(qkv_weight[q_proj_size + k_proj_size:].detach(),
                             requires_grad=False)
        self.register_parameter("q_weight", q_weight)
        self.register_parameter("k_weight", k_weight)
        self.register_parameter("v_weight", v_weight)

        if qkv_linear.bias is not None:
            bias = torch_view(t2j(qkv_linear.bias.data, use_dlpack=False))
            q_bias = Parameter(bias[:q_proj_size].detach(),
                               requires_grad=False)
            k_bias = Parameter(bias[q_proj_size:q_proj_size +
                                    k_proj_size].detach(),
                               requires_grad=False)
            v_bias = Parameter(bias[q_proj_size + k_proj_size:].detach(),
                               requires_grad=False)
            self.register_parameter("q_bias", q_bias)
            self.register_parameter("k_bias", k_bias)
            self.register_parameter("v_bias", v_bias)
        else:
            self.register_parameter("q_bias", None)
            self.register_parameter("k_bias", None)
            self.register_parameter("v_bias", None)

        if self.w8q8_int8_quant:
            assert self.q_weight.jax().dtype == jnp.int8
            assert self.k_weight.jax().dtype == jnp.int8
            assert self.v_weight.jax().dtype == jnp.int8
            qkv_weight_scale = torch_view(
                t2j(qkv_linear.weight_scale.data, use_dlpack=False))
            q_weight_scale = Parameter(qkv_weight_scale[:q_proj_size].detach(),
                                       requires_grad=False)
            k_weight_scale = Parameter(
                qkv_weight_scale[q_proj_size:q_proj_size +
                                 k_proj_size].detach(),
                requires_grad=False)
            v_weight_scale = Parameter(qkv_weight_scale[q_proj_size +
                                                        k_proj_size:].detach(),
                                       requires_grad=False)
            self.register_parameter("q_weight_scale", q_weight_scale)
            self.register_parameter("k_weight_scale", k_weight_scale)
            self.register_parameter("v_weight_scale", v_weight_scale)
        else:
            self.register_parameter("q_weight_scale", None)
            self.register_parameter("k_weight_scale", None)
            self.register_parameter("v_weight_scale", None)

    def forward(self, input):
        x = input.jax()

        # Same forward functionality as QKVParallelLinear, but doing qkv porj
        # separately.
        q_weight = self.q_weight.jax()
        k_weight = self.k_weight.jax()
        v_weight = self.v_weight.jax()
        q_bias = None if (self.skip_bias_add
                          or self.q_bias is None) else self.q_bias.jax()
        k_bias = None if (self.skip_bias_add
                          or self.k_bias is None) else self.k_bias.jax()
        v_bias = None if (self.skip_bias_add
                          or self.v_bias is None) else self.v_bias.jax()
        if self.w8q8_int8_quant:
            q_weight_scale = self.q_weight_scale.jax(
            ) if self.w8q8_int8_quant else None
            k_weight_scale = self.k_weight_scale.jax(
            ) if self.w8q8_int8_quant else None
            v_weight_scale = self.v_weight_scale.jax(
            ) if self.w8q8_int8_quant else None
            q_proj = torch_view(
                forward_w8a8_int8(x, q_weight, q_bias, q_weight_scale,
                                  self.mesh, self.gather_output,
                                  ParallelType.COL_PARALLEL))
            k_proj = torch_view(
                forward_w8a8_int8(x, k_weight, k_bias, k_weight_scale,
                                  self.mesh, self.gather_output,
                                  ParallelType.COL_PARALLEL))
            v_proj = torch_view(
                forward_w8a8_int8(x, v_weight, v_bias, v_weight_scale,
                                  self.mesh, self.gather_output,
                                  ParallelType.COL_PARALLEL))
        else:
            q_proj = torch_view(forward_unqunatized(x, q_weight, q_bias))
            k_proj = torch_view(forward_unqunatized(x, k_weight, k_bias))
            v_proj = torch_view(forward_unqunatized(x, v_weight, v_bias))

        # The q/k/v projections will be split outside of the QKVParallelLinear.
        # Because we are replacing XlaQKVParallelLinear with the
        # QKVParallelLinear, we need to concatenate q, k, and v projections to
        # match the output shape of the QKVParallelLinear implementation even if
        # it seems to be redundant.
        # The concat and the following split will be noop, and should be
        # optimized away by the compiler.
        qkv_proj = torch.cat([q_proj, k_proj, v_proj], dim=-1)

        if not self.return_bias:
            return qkv_proj
        output_bias = torch.cat([q_bias, k_bias, v_bias], dim=-1) if \
                            self.skip_bias_add else None
        return qkv_proj, output_bias
