from typing import Optional

import jax
import torch
import torch.nn.functional as F
from jax.sharding import Mesh, NamedSharding, PartitionSpec
from torch.nn.parameter import Parameter
from torchax.interop import torch_view
from torchax.ops.mappings import t2j
from vllm.model_executor.layers.linear import QKVParallelLinear

P = PartitionSpec


class JaxQKVParallelLinear(torch.nn.Module):

    def __init__(self, qkv_linear: torch.nn.Module, mesh: Mesh):
        super().__init__()
        assert isinstance(qkv_linear, QKVParallelLinear)
        self.skip_bias_add = qkv_linear.skip_bias_add
        self.return_bias = qkv_linear.return_bias

        self.q_weight: Parameter
        self.k_weight: Parameter
        self.v_weight: Parameter
        self.q_bias: Optional[Parameter]
        self.k_bias: Optional[Parameter]
        self.v_bias: Optional[Parameter]
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

    def _load_weights_from_vllm_layer(self, qkv_linear: torch.nn.Module):
        q_proj_size, k_proj_size, _ = qkv_linear.output_sizes
        # The weight of qkv linear is a concatenation of q, k, and v weights
        # along the output dimension.
        qkv_weight = torch_view(t2j(qkv_linear.weight.data))
        q_weight = Parameter(qkv_weight[:q_proj_size], requires_grad=False)
        k_weight = Parameter(qkv_weight[q_proj_size:q_proj_size + k_proj_size],
                             requires_grad=False)
        v_weight = Parameter(qkv_weight[q_proj_size + k_proj_size:],
                             requires_grad=False)
        self.register_parameter("q_weight", q_weight)
        self.register_parameter("k_weight", k_weight)
        self.register_parameter("v_weight", v_weight)

        if qkv_linear.bias is not None:
            bias = torch_view(t2j(qkv_linear.bias))
            q_bias = Parameter(bias[:q_proj_size], requires_grad=False)
            k_bias = Parameter(bias[q_proj_size:q_proj_size + k_proj_size],
                               requires_grad=False)
            v_bias = Parameter(bias[q_proj_size + k_proj_size:],
                               requires_grad=False)
            self.register_parameter("q_bias", q_bias)
            self.register_parameter("k_bias", k_bias)
            self.register_parameter("v_bias", v_bias)
        else:
            self.register_parameter("q_bias", None)
            self.register_parameter("k_bias", None)
            self.register_parameter("v_bias", None)

    def forward(self, input):
        # Same forward functionality as QKVParallelLinear, but doing qkv porj
        # separately.
        q_bias = self.q_bias if not self.skip_bias_add else None
        k_bias = self.k_bias if not self.skip_bias_add else None
        v_bias = self.v_bias if not self.skip_bias_add else None
        q_proj = F.linear(input, self.q_weight, q_bias)
        k_proj = F.linear(input, self.k_weight, k_bias)
        v_proj = F.linear(input, self.v_weight, v_bias)
        # The q/k/v projections will be split outside of the QKVParallelLinear.
        # Because we are replacing XlaQKVParallelLinear with the
        # QKVParallelLinear, we need to concatenate q, k, and v projections to
        # match the output shape of the QKVParallelLinear implementation even if
        # it seems to be redundant.
        # The concat and the following split will be noop, and should be
        # optimized away by the compiler.
        qkv_proj = torch.cat([q_proj, k_proj, v_proj], dim=-1)
        output_bias = torch.cat([q_bias, k_bias, v_bias], dim=-1) if \
                            self.skip_bias_add else None
        if not self.return_bias:
            return qkv_proj
        return qkv_proj, output_bias
