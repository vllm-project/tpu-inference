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

from typing import Optional

import jax
import jax.numpy as jnp
import torch
from jax.sharding import PartitionSpec
from torch.nn.parameter import Parameter
from torchax.interop import jax_view, torch_view
from vllm.model_executor.layers.quantization.compressed_tensors.schemes.compressed_tensors_w4a4_nvfp4 import \
    CompressedTensorsW4A4Fp4

from tpu_inference.layers.common.linear import sharded_quantized_matmul
from tpu_inference.layers.common.process_weights.linear_weights import (
    LinearWeights, process_linear_weights, shard_linear_weights,
    to_parameter_list)
from tpu_inference.layers.common.quantization import u8_unpack_e2m1
from tpu_inference.layers.common.utils import \
    slice_sharded_tensor_for_concatenation
from tpu_inference.layers.vllm.quantization.configs import \
    VllmQuantLinearConfig
from tpu_inference.logger import init_logger
from tpu_inference.utils import t2j

P = PartitionSpec
logger = init_logger(__name__)


class VllmCompressedTensorsW4A4Fp4(CompressedTensorsW4A4Fp4):

    def __init__(
        self,
        use_a16: bool,
        is_static_input_scheme: bool,
        linear_config: VllmQuantLinearConfig,
    ):
        if is_static_input_scheme:
            raise NotImplementedError(
                "Static input scheme is not yet supported for W4A4 NVFP4.")

        if not use_a16:
            logger.warning(
                "fp4 x fp4 mmu is not natively supported by TPU hardware, so activations are always kept at bf16 for now."
            )

        # We need to monkeypatch expose_input_quant_key to handle None kernel
        # because init_nvfp4_linear_kernel is being called in super.__init__()
        from vllm.model_executor.layers.quantization.compressed_tensors.schemes import \
            compressed_tensors_w4a4_nvfp4 as vllm_ct_w4a4

        # Disable kernel initialization
        vllm_ct_w4a4.init_nvfp4_linear_kernel = lambda *args, **kwargs: None

        original_expose = vllm_ct_w4a4.expose_input_quant_key

        def safe_expose_input_quant_key(layer, kernel):
            if kernel is None:
                return
            original_expose(layer, kernel)

        vllm_ct_w4a4.expose_input_quant_key = safe_expose_input_quant_key

        super().__init__(use_a16=use_a16)

        self.linear_config = linear_config

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        # Standardize weight names
        layer.weight = layer.weight_packed
        del layer.weight_packed

        # Process weight global scale
        weight_global_scale = layer.weight_global_scale.max().to(torch.float32)
        weight_global_scale = 1.0 / weight_global_scale

        if not self.use_a16:
            input_global_scale_inv = layer.input_global_scale.max().to(
                torch.float32)
            input_global_scale = 1.0 / input_global_scale_inv

        # TPU Specific loading
        weight_packed = t2j(layer.weight, use_dlpack=False)
        weight_scale = t2j(layer.weight_scale, use_dlpack=False)

        # Convert to single elements as it's per tensor scale
        weight_global_scale_jax = t2j(weight_global_scale, use_dlpack=False)

        for attr in ('weight', 'weight_scale', 'weight_global_scale',
                     'input_global_scale', 'input_global_scale_inv', 'alpha'):
            if hasattr(layer, attr):
                delattr(layer, attr)

        if getattr(layer, "bias",
                   None) is not None and not layer.skip_bias_add:
            bias = t2j(layer.bias, use_dlpack=False)
            delattr(layer, "bias")
        else:
            bias = None

        @jax.jit
        def process_nvfp4_linear_weights(
            weight_packed: jax.Array,
            weight_scale: jax.Array,
            weight_global_scale: jax.Array,
            bias: jax.Array | None,
        ) -> LinearWeights:
            # Unpack uint8 to FP4
            fp4 = u8_unpack_e2m1(weight_packed)  # [out, in]
            fp4 = jnp.transpose(fp4)  # [in, out]

            # Combine FP8 block scale & FP32 global scale
            # weight_scale is [out, in // group_size]
            block_scale = weight_scale.astype(
                jnp.float32) * weight_global_scale
            block_scale = jnp.transpose(block_scale)  # [in // group_size, out]

            return process_linear_weights(
                LinearWeights(
                    weight=fp4,
                    weight_scale=block_scale,
                    zero_point=None,
                    bias=bias,
                ),
                fused=self.linear_config.fuse_matmuls,
                output_sizes=self.linear_config.output_sizes,
                reorder_size=self.linear_config.n_shards,
                enable_kernel=self.linear_config.
                enable_quantized_matmul_kernel,
            )

        weights = process_nvfp4_linear_weights(weight_packed, weight_scale,
                                               weight_global_scale_jax, bias)

        weights = torch_view(
            shard_linear_weights(
                weights,
                mesh=self.linear_config.mesh,
                weight_p_spec=self.linear_config.weight_sharding,
                bias_p_spec=self.linear_config.bias_sharding,
            ))

        if self.linear_config.fuse_matmuls:
            layer.weight = Parameter(weights.weight, requires_grad=False)
            layer.weight_scale = Parameter(weights.weight_scale,
                                           requires_grad=False)
            if bias is not None:
                layer.bias = Parameter(weights.bias, requires_grad=False)
        else:
            layer.weight = to_parameter_list(weights.weight)
            layer.weight_scale = to_parameter_list(weights.weight_scale)
            if bias is not None:
                layer.bias = to_parameter_list(weights.bias)

        if not self.use_a16:
            # Note: input_scale should be jax sharded tensor for split
            input_global_scale_j = jax.device_put(
                t2j(input_global_scale, use_dlpack=False),
                jax.sharding.NamedSharding(self.linear_config.mesh,
                                           PartitionSpec()))
            layer.input_scale = Parameter(torch_view(input_global_scale_j),
                                          requires_grad=False)

    def apply_weights(self, layer: torch.nn.Module, x: torch.Tensor,
                      bias: Optional[torch.Tensor]) -> torch.Tensor:
        with jax.named_scope(layer._get_name()):
            if self.linear_config.fuse_matmuls:
                return self._apply_fused(layer, x, bias)
            else:
                return self._apply_split(layer, x, bias)

    def _apply_fused(self, layer: torch.nn.Module, x: torch.Tensor,
                     bias: Optional[torch.Tensor]) -> torch.Tensor:
        x_jax = jax_view(x)
        weight_jax = jax_view(layer.weight)
        weight_scale_jax = jax_view(layer.weight_scale)

        outs = sharded_quantized_matmul(
            x_jax,
            weight_jax,
            weight_scale_jax,
            self.linear_config.weight_sharding,
            mesh=self.linear_config.mesh,
        )

        if bias is not None and not layer.skip_bias_add:
            outs += jax_view(bias)
        outs = slice_sharded_tensor_for_concatenation(
            outs, self.linear_config.output_sizes, self.linear_config.n_shards)
        return torch_view(jnp.concatenate(outs, axis=-1))

    def _apply_split(self, layer: torch.nn.Module, x: torch.Tensor,
                     bias: Optional[torch.Tensor]) -> torch.Tensor:
        assert isinstance(layer.weight, torch.nn.ParameterList)

        x_jax = jax_view(x)
        outs = []
        for i, (weight, weight_scale) in enumerate(
                zip(layer.weight, layer.weight_scale)):
            weight_jax = jax_view(weight)
            weight_scale_jax = jax_view(weight_scale)

            out = sharded_quantized_matmul(
                x_jax,
                weight_jax,
                weight_scale_jax,
                self.linear_config.weight_sharding,
                mesh=self.linear_config.mesh,
            )

            if bias is not None and not layer.skip_bias_add:
                out += jax_view(bias[i])
            outs.append(out)
        return torch_view(jnp.concatenate(outs, axis=-1))
