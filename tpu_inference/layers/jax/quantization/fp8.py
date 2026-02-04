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
from jax.sharding import Mesh

from tpu_inference import envs
from tpu_inference.layers.common.quantization import dequantize_tensor
from tpu_inference.layers.common.quantization import fp8 as jax_common
from tpu_inference.layers.common.quantization import quantize_tensor
from tpu_inference.layers.jax import JaxModule
from tpu_inference.layers.jax.linear import JaxEinsum
from tpu_inference.layers.jax.quantization import QuantizeMethodBase
from tpu_inference.layers.jax.quantization.configs import (
    JaxQuantLinearConfig, QuantizationConfig)
from tpu_inference.models.jax.utils.weight_utils import (
    load_blockwise_fp8_scale, load_nnx_param_from_reshaped_torch, shard_put)


class Fp8LinearMethod(QuantizeMethodBase, jax_common.Fp8LinearMethod):
    """Fp8 method for JAX Linear layer."""

    def create_weights_jax(self, layer: JaxModule, *weight_args,
                           **extra_weight_attrs):
        assert isinstance(layer, JaxEinsum)

        input_dim = self.linear_config.input_size
        output_dim = self.linear_config.output_size

        if self.linear_config.enable_quantized_matmul_kernel:
            # Blockwise quantization: 3D scales (n_blocks, 1, n_out)
            block_size = self.linear_config.block_size
            if input_dim % block_size != 0:
                raise ValueError(
                    f"Input dimension {input_dim} must be divisible by block size {block_size} "
                    f"for blockwise quantization. Got {input_dim} % {block_size} = {input_dim % block_size}."
                )
            n_blocks = input_dim // block_size
            layer.weight_scale = nnx.Param(
                jnp.ones((n_blocks, 1, output_dim), dtype=jnp.float32))
            layer.weight_scale.weight_loader = partial(
                load_blockwise_fp8_scale,
                output_dim=output_dim,
                n_blocks=n_blocks,
                param_name="weight_scale")
        else:
            # Per-channel quantization: 1D scales (n_out,)
            layer.weight_scale = nnx.Param(
                jnp.ones((output_dim, ), dtype=jnp.float32))
            layer.weight_scale.weight_loader = partial(
                load_nnx_param_from_reshaped_torch, param_name="weight_scale")

    def process_weights_after_loading(self,
                                      layer: JaxModule,
                                      mesh: Optional[Mesh] = None) -> None:
        assert isinstance(layer, JaxEinsum)

        weight = layer.weight.value
        weight_scale = layer.weight_scale.value
        input_size = self.linear_config.input_size
        output_size = self.linear_config.output_size
        c_dims = self.linear_config.contracting_dims
        o_dims = self.linear_config.output_dims

        # Determine if we need to (re)quantize
        block_size = envs.REQUANTIZE_BLOCK_SIZE
        is_fp8 = weight.dtype == jnp.float8_e4m3fn
        needs_blockwise = block_size is not None

        if is_fp8 and not needs_blockwise:
            # Pre-quantized FP8 checkpoint, no block size specified -> keep as-is
            return

        if not is_fp8 and not needs_blockwise:
            # Float32/BF16 weights but no block size -> error (per-channel requant not supported)
            raise ValueError(
                "FP8 requantization from float32/bfloat16 requires REQUANTIZE_BLOCK_SIZE "
                "to be set. Per-channel FP8 is only supported when loading from a "
                "pre-quantized checkpoint.")

        # Permute to (Contracting..., Output...)
        perm = tuple(c_dims + o_dims)
        if perm != tuple(range(len(perm))):
            weight_perm = weight.transpose(perm)
        else:
            weight_perm = weight

        # Flatten to (In, Out)
        weight_flat = weight_perm.reshape(input_size, output_size)

        if is_fp8:
            # Dequantize
            if weight_scale.ndim == 3:
                # Blockwise: scale shape (n_blocks, 1, output_size)
                scale_for_dequant = jnp.squeeze(weight_scale, axis=1)
                weight_flat = dequantize_tensor(weight_flat,
                                                scale_for_dequant,
                                                axis=0)
            else:
                # Per-channel: scale shape (output_size,)
                weight_flat = dequantize_tensor(weight_flat,
                                                weight_scale,
                                                axis=0)

        # Quantize along input axis (0) for blockwise
        w_q, scale = quantize_tensor(jnp.float8_e4m3fn,
                                     weight_flat,
                                     axis=0,
                                     block_size=block_size)
        # Scale shape (n_blocks, Out) -> (n_blocks, 1, Out)
        scale = jnp.expand_dims(scale, axis=1)

        # Reshape back: (In, Out) -> (Contracting..., Output...) -> Permute back
        w_q_reshaped = w_q.reshape([weight.shape[i] for i in perm])

        # Inverse permutation
        inv_perm = np.argsort(perm)
        w_q_orig = w_q_reshaped.transpose(inv_perm)

        # Update parameters
        spec = getattr(layer.weight.sharding, 'spec', layer.weight.sharding)
        mesh = getattr(layer.weight, 'mesh', None) or mesh
        layer.weight.value = shard_put(w_q_orig, spec, mesh)

        scale_spec = getattr(layer.weight_scale.sharding, 'spec',
                             layer.weight_scale.sharding)
        layer.weight_scale.value = shard_put(scale, scale_spec, mesh)

    def apply_jax(self, layer: JaxModule, x: jax.Array) -> jax.Array:
        assert isinstance(layer, JaxEinsum)

        input_size = self.linear_config.input_size
        output_size = self.linear_config.output_size
        c_dims = self.linear_config.contracting_dims
        o_dims = self.linear_config.output_dims

        # Permute and reshape weight to (In, Out)
        perm = tuple(c_dims + o_dims)
        if perm != tuple(range(len(perm))):
            w_val = layer.weight.value.transpose(perm)
        else:
            w_val = layer.weight.value

        w_val = w_val.reshape(input_size, output_size)

        bias = layer.bias.value if layer.bias else None
        if bias is not None:
            bias = bias.reshape(-1)

        # Flatten x to (Batch, In) for the kernel
        x_reshaped = x.reshape(-1, input_size)

        with jax.named_scope(layer._get_name()):
            if self.linear_config.fuse_matmuls:
                # _apply_fused expects transposed data -> (Out, In).
                out = self._apply_fused(x_reshaped, w_val.T,
                                        layer.weight_scale.value, bias)
            else:
                raise NotImplementedError(
                    "Non-fused matmuls not implemented yet.")

        # Reshape output back to (Batch, OutputDims...)
        out_dims = [layer.weight.value.shape[i] for i in o_dims]
        out_shape = (x.shape[0], ) + tuple(out_dims)
        return out.reshape(out_shape)


class Fp8Config(QuantizationConfig):
    """FP8 quantization config for JAX models.

    Uses REQUANTIZE_BLOCK_SIZE environment variable for blockwise quantization,
    consistent with vLLM's approach.
    """

    def get_quant_method(self, layer: JaxModule,
                         prefix: str) -> Optional[QuantizeMethodBase]:
        if isinstance(layer, JaxEinsum):
            linear_config = JaxQuantLinearConfig(layer)
            return Fp8LinearMethod(linear_config)
        return None
