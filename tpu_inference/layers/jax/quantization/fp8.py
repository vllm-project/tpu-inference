# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
from functools import partial
from typing import Optional

import jax
import jax.numpy as jnp
from flax import nnx

from tpu_inference.layers.common.quantization import fp8 as jax_common
from tpu_inference.layers.common.quantization.configs import QuantLinearConfig
from tpu_inference.layers.jax import JaxModule
from tpu_inference.layers.jax.linear import JaxEinsum
from tpu_inference.layers.jax.quantization import QuantizeMethodBase
from tpu_inference.layers.jax.quantization.configs import QuantizationConfig
from tpu_inference.models.jax.utils.weight_utils import \
    load_nnx_param_from_reshaped_torch


class Fp8BlockwiseLinearMethod(QuantizeMethodBase, common_fp8.Fp8LinearMethod):
    """Block-wise Fp8 method for JAX Linear layer."""

    def __init__(self, quant_config: "Fp8Config", layer: JaxEinsum,
                 linear_config: QuantLinearConfig):
        common_fp8.Fp8LinearMethod.__init__(self, linear_config)
        self.quant_config = quant_config

        kernel_shape = layer.kernel_shape
        if len(kernel_shape) > 2:
            adapt_info = linear_config.get_adapt_info(
                einsum_str=layer.einsum_str, weight=layer.weight)
            self.weight_sharding = adapt_info.weight_sharding
            self.output_shape = adapt_info.output_shape
            out_features = math.prod(self.output_shape)
            in_features = math.prod(adapt_info.in_features)
        else:
            in_features, out_features = kernel_shape
            self.weight_sharding = layer.weight.sharding
            self.output_shape = (out_features, )

        # Storing list of output sizes (instead of self.out_features) for compatibility.
        self.linear_config.output_sizes = [out_features]
        self.in_features = in_features

    def create_weights_jax(self, layer: JaxModule, *weight_args, rngs,
                           **extra_weight_attrs):
        assert isinstance(layer, JaxEinsum)

        out_features = sum(self.linear_config.output_sizes)
        kernel_init = layer.kernel_init
        # Follow upstream limitation that only float8_e4m3 is supported.
        # https://github.com/vllm-project/vllm/blob/2a99c5a6c86daef8c766ba2dbf05c385b192c64b/vllm/model_executor/layers/quantization/fp8.py#L283-L284
        param_dtype = jnp.float8_e4m3
        layer.weight = nnx.Param(kernel_init(rngs.params(),
                                             (out_features, self.in_features),
                                             param_dtype),
                                 weight_loader=partial(
                                     load_nnx_param_from_reshaped_torch,
                                     permute_dims=(0, 1),
                                 ))
        layer.weight.sharding = self.weight_sharding

        # Block-wise quantization scale
        block_n, block_k = self.quant_config.weight_block_size[
            0], self.quant_config.weight_block_size[1]
        layer.weight_scale_inv = nnx.Param(
            kernel_init(
                rngs.params(),
                [(out_features + block_n - 1) // block_n,
                 (self.in_features + block_k - 1) // block_k],
                layer.dtype,
            ),
            weight_loader=partial(
                load_nnx_param_from_reshaped_torch,
                permute_dims=(0, 1),
            ))

    def process_weights_after_loading(self, layer):
        assert isinstance(layer, JaxEinsum)
        assert self.quant_config.weight_block_size is not None

        weight = layer.weight.value
        weight_scale_inv = layer.weight_scale_inv.value
        bias = layer.bias.value if getattr(layer, 'bias',
                                           None) is not None else None
        if bias is not None:
            bias = bias.reshape(-1)
        weights = common_fp8.process_blockwise_fp8_linear_weights(
            weight,
            weight_scale_inv,
            bias=bias,
            weight_block_size=tuple(self.quant_config.weight_block_size),
            linear_config=self.linear_config)
        delattr(layer, 'weight')
        delattr(layer, 'weight_scale_inv')
        delattr(layer, 'bias')

        if self.linear_config.enable_quantized_matmul_kernel:
            # The quantized_matmul_kernel expects weight scales shaped (n_out_features, 1, n_blocks) for blockwisze quantization.
            weights.weight_scale = jnp.expand_dims(
                jnp.transpose(weights.weight_scale),
                axis=1,
            )
        weights = shard_linear_weights(
            weights,
            mesh=self.linear_config.mesh,
            weight_p_spec=self.linear_config.weight_sharding,
            bias_p_spec=self.linear_config.bias_sharding,
        )

        if self.linear_config.fuse_matmuls:
            layer.weight = nnx.Param(weights.weight)
            layer.weight_scale_inv = nnx.Param(weights.weight_scale)
            layer.bias = nnx.Param(weights.bias) if bias is not None else None
        else:
            raise NotImplementedError(
                "Fp8 block-wise linear method only supports fuse_matmuls.")

    def apply_jax(self, layer: JaxModule, x: jax.Array) -> jax.Array:
        if not self.linear_config.fuse_matmuls:
            raise NotImplementedError(
                "Fp8 block-wise linear method only supports fuse_matmuls.")
        weight, scale = layer.weight.value, layer.weight_scale_inv.value
        bias = layer.bias.value if layer.bias is not None else None
        out = self._apply_fused(x, weight, scale, bias=bias)
        out = out.reshape(out.shape[:-1] + self.output_shape)
        return out


class Fp8Config(QuantizationConfig):

    ACTIVATION_SCHEMES = ["dynamic", "static"]

    def __init__(self, hf_quant_config: dict):
        # Replicating upstream https://github.com/vllm-project/vllm/blob/77c09e1130661197ccac2d968a28cd4a557922d5/vllm/model_executor/layers/quantization/fp8.py#L167-L175

        quant_method = self.get_from_keys(hf_quant_config, ["quant_method"])
        self.is_checkpoint_fp8_serialized = "fp8" in quant_method
        activation_scheme = self.get_from_keys(hf_quant_config,
                                               ["activation_scheme"])
        ignored_layers = self.get_from_keys(hf_quant_config,
                                            ["ignored_layers"], None)
        weight_block_size = self.get_from_keys(hf_quant_config,
                                               ["weight_block_size"], None)
        if not ignored_layers:
            ignored_layers = self.get_from_keys(hf_quant_config,
                                                ["modules_to_not_convert"],
                                                None)

        if activation_scheme not in self.ACTIVATION_SCHEMES:
            raise ValueError(
                f"Unsupported activation scheme {activation_scheme}")
        self.activation_scheme = activation_scheme
        self.ignored_layers = ignored_layers or []
        if weight_block_size is not None:
            if not self.is_checkpoint_fp8_serialized:
                raise ValueError(
                    "The block-wise quantization only supports fp8-serialized "
                    "checkpoint for now.")
            if len(weight_block_size) != 2:
                raise ValueError(
                    "The quantization block size of weight must have 2 "
                    f"dimensions, but got {len(weight_block_size)} dimensions")
            if activation_scheme != "dynamic":
                raise ValueError("The block-wise quantization only supports "
                                 "dynamic activation scheme for now, but got "
                                 f"{activation_scheme} activation scheme.")
        self.weight_block_size = weight_block_size

    def get_quant_method(self, layer: JaxModule,
                         prefix: str) -> Optional[QuantizeMethodBase]:
        if isinstance(layer, JaxEinsum):
            linear_config = QuantLinearConfig(
                output_sizes=[layer.weight.shape[-1]], enable_sp=False)
            if self.weight_block_size is not None:
                return Fp8BlockwiseLinearMethod(self, layer, linear_config)
        return None
