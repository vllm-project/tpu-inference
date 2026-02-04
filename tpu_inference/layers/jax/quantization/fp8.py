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

from functools import partial
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from tpu_inference.layers.common.quantization import fp8 as jax_common
from tpu_inference.layers.common.quantization.configs import QuantLinearConfig
from tpu_inference.layers.jax import JaxModule
from tpu_inference.layers.jax.linear import JaxEinsum
from tpu_inference.layers.jax.quantization import QuantizeMethodBase
from tpu_inference.layers.jax.quantization.configs import QuantizationConfig
from tpu_inference.models.jax.utils.weight_utils import \
    load_nnx_param_from_reshaped_torch


class Fp8BlockwiseLinearMethod(QuantizeMethodBase, jax_common.Fp8LinearMethod):
    """Block-wise Fp8 method for JAX Linear layer."""

    def __init__(self, quant_config: "Fp8Config",
                 linear_config: QuantLinearConfig):
        jax_common.Fp8LinearMethod.__init__(self, linear_config)
        self.quant_config = quant_config

    def create_weights_jax(self, layer: JaxModule, *weight_args, rngs,
                           **extra_weight_attrs):
        assert isinstance(layer, JaxEinsum)

        kernel_init = layer.kernel_init
        # Follow upstream limitation that only float8_e4m3 is supported.
        # https://github.com/vllm-project/vllm/blob/2a99c5a6c86daef8c766ba2dbf05c385b192c64b/vllm/model_executor/layers/quantization/fp8.py#L283-L284
        param_dtype = jnp.float8_e4m3
        kernel_shape = layer.kernel_shape
        if len(kernel_shape) > 2:
            # HF model stores weight in 2-D shape. E.g. for "TD,DNH->TNH", weight shape in HF is (NH, D)
            x_axis, w_axis = layer.einsum_str.split("->")[0].split(",")
            contracting_axis = set(x_axis) & set(w_axis)
            in_features = np.prod(
                np.array([
                    layer.kernel_shape[i] for i, c in enumerate(w_axis)
                    if c in contracting_axis
                ]))
            out_features = np.prod(
                np.array([
                    layer.kernel_shape[i] for i, c in enumerate(w_axis)
                    if c not in contracting_axis
                ]))
            kernel_shape = (in_features, out_features)
            layer.weight = nnx.Param(
                kernel_init(rngs.params(), kernel_shape, param_dtype),
                weight_loader=load_nnx_param_from_reshaped_torch)

            # E.g. if weight shape is (NH, D), sharding is ('x', None, 'y'), we need to fuse sharding to ('x', 'y')
            sharding = layer.weight.sharding + (None, ) * (
                len(layer.kernel_shape) - len(layer.weight.sharding))
            in_sharding = set(
                s for i, s in enumerate(sharding)
                if w_axis[i] in contracting_axis and s is not None)
            out_sharding = set(
                s for i, s in enumerate(sharding)
                if w_axis[i] not in contracting_axis and s is not None)
            assert len(in_sharding) <= 1 and len(out_sharding) <= 1, \
                f"Cannot fuse sharding {layer.weight.sharding} into 2D weight sharding for {layer.einsum_str}"
            layer.weight.sharding = (next(iter(in_sharding),
                                          None), next(iter(out_sharding),
                                                      None))
        else:
            in_features, out_features = kernel_shape
            layer.weight = nnx.Param(
                kernel_init(rngs.params(), kernel_shape, param_dtype),
                weight_loader=load_nnx_param_from_reshaped_torch)

        # Block-wise quantization scale
        block_n, block_k = self.quant_config.weight_block_size[
            0], self.quant_config.weight_block_size[1]
        layer.weight_scale_inv = nnx.Param(
            kernel_init(
                rngs.params(),
                [(out_features + block_n - 1) // block_n,
                 (in_features + block_k - 1) // block_k],
                layer.dtype,
            ),
            weight_loader=partial(
                load_nnx_param_from_reshaped_torch,
                # Don't transpose, as the scale shape is already [out, in].
                permute_dims=(0, 1),
            ))
        layer.weight_scale_inv.sharding = layer.weight.sharding[::-1]

    def process_weights_after_loading(self, layer: JaxEinsum):
        pass

    def apply_jax(self, layer: JaxModule, x: jax.Array) -> jax.Array:
        # TODO(#1623): Implement FP8 linear method for JAX
        raise NotImplementedError()


class Fp8Config(QuantizationConfig):

    ACTIVATION_SCHEMES = ["dynamic", "static"]

    def __init__(self,
                 is_checkpoint_fp8_serialized: bool,
                 activation_scheme: str = "dynamic",
                 ignored_layers: Optional[list] = None,
                 weight_block_size: Optional[list] = None):
        self.is_checkpoint_fp8_serialized = is_checkpoint_fp8_serialized

        if activation_scheme not in self.ACTIVATION_SCHEMES:
            raise ValueError(
                f"Unsupported activation scheme {activation_scheme}")
        self.activation_scheme = activation_scheme
        self.ignored_layers = ignored_layers or []
        if weight_block_size is not None:
            if not is_checkpoint_fp8_serialized:
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

    @classmethod
    def from_config(cls, config: dict):
        """Create instance from huggingface config dict."""
        quant_method = cls.get_from_keys(config, ["quant_method"])
        is_checkpoint_fp8_serialized = "fp8" in quant_method
        activation_scheme = cls.get_from_keys(config, ["activation_scheme"])
        ignored_layers = cls.get_from_keys(config, ["ignored_layers"], None)
        weight_block_size = cls.get_from_keys(config, ["weight_block_size"],
                                              None)
        if not ignored_layers:
            ignored_layers = cls.get_from_keys(config,
                                               ["modules_to_not_convert"],
                                               None)
        return cls(
            is_checkpoint_fp8_serialized=is_checkpoint_fp8_serialized,
            activation_scheme=activation_scheme,
            ignored_layers=ignored_layers,
            weight_block_size=weight_block_size,
        )

    def get_quant_method(self, layer: JaxModule,
                         prefix: str) -> Optional[QuantizeMethodBase]:
        if isinstance(layer, JaxEinsum):
            linear_config = QuantLinearConfig(
                output_sizes=[layer.weight.shape[-1]], enable_sp=False)
            if self.weight_block_size is not None:
                return Fp8BlockwiseLinearMethod(self, linear_config)
        return None
