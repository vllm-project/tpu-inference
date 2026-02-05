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
import torch
from flax import nnx
from jax.sharding import Mesh, NamedSharding, SingleDeviceSharding
from torchax.ops.mappings import t2j

from tpu_inference import envs
from tpu_inference.layers.common.moe import MoEBackend, moe_apply
from tpu_inference.layers.common.process_weights.moe_weights import (
    FusedMoEWeights, process_fp8_moe_weights)
from tpu_inference.layers.common.quantization import dequantize_tensor
from tpu_inference.layers.common.quantization import fp8 as jax_common
from tpu_inference.layers.common.quantization import quantize_tensor
from tpu_inference.layers.jax import JaxModule
from tpu_inference.layers.jax.linear import JaxEinsum
from tpu_inference.layers.jax.moe.moe import JaxMoE
from tpu_inference.layers.jax.quantization import QuantizeMethodBase
from tpu_inference.layers.jax.quantization.configs import (
    JaxQuantLinearConfig, QuantizationConfig)
from tpu_inference.models.jax.utils.weight_utils import (
    load_nnx_param_from_reshaped_torch, shard_put)


def load_blockwise_fp8_scale(
    jax_param: nnx.Param,
    torch_weight: torch.Tensor,
    output_dim: int,
    n_blocks: int,
    param_name: str,
) -> None:
    """Load blockwise FP8 weight scales from checkpoint.

    Transforms scale shape from checkpoint format to kernel format:
    - Checkpoint: (n_blocks, n_out) or (n_blocks * n_out,) flattened
    - Kernel: (n_blocks, 1, n_out)

    Args:
        jax_param: Target nnx.Param to load into
        torch_weight: Weight tensor from checkpoint
        output_dim: Output dimension of the layer
        n_blocks: Number of quantization blocks
        param_name: Name of the parameter

    Raises:
        ValueError: If checkpoint scale shape doesn't match expected block size.
    """
    # Convert torch tensor to JAX array
    scale_jax = t2j(torch_weight, use_dlpack=False)

    # Reshape if flattened: (n_blocks * n_out,) → (n_blocks, n_out)
    if scale_jax.ndim == 1:
        expected_size = n_blocks * output_dim
        if scale_jax.size != expected_size:
            raise ValueError(
                f"Checkpoint scale size mismatch for '{param_name}': "
                f"expected {expected_size} ({n_blocks} blocks × {output_dim} outputs), "
                f"got {scale_jax.size}.")
        scale_jax = scale_jax.reshape(n_blocks, output_dim)
    elif scale_jax.ndim == 2:
        if scale_jax.shape != (n_blocks, output_dim):
            raise ValueError(
                f"Checkpoint scale shape mismatch for '{param_name}': "
                f"expected ({n_blocks}, {output_dim}), got {scale_jax.shape}. "
                f"This suggests the checkpoint uses a different block size. ")
    else:
        raise ValueError(
            f"Checkpoint scale has unexpected number of dimensions for '{param_name}': "
            f"expected 1D or 2D, got {scale_jax.ndim}D with shape {scale_jax.shape}. "
            f"Expected shapes: ({n_blocks * output_dim},) or ({n_blocks}, {output_dim})."
        )

    # Expand dims: (n_blocks, n_out) → (n_blocks, 1, n_out)
    scale_jax = jnp.expand_dims(scale_jax, axis=1)

    # Get sharding info
    spec = jax_param.sharding
    if isinstance(jax_param.sharding, NamedSharding):
        spec = jax_param.sharding.spec
    elif isinstance(jax_param.sharding, SingleDeviceSharding):
        spec = ()
    mesh = getattr(jax_param, 'mesh', None)

    # Load into parameter
    jax_param.value = shard_put(scale_jax.astype(jax_param.value.dtype),
                                spec,
                                mesh=mesh)


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
            # TODO(patemotter): Support per-channel quantization and quantized matmul kernel
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
            # TODO(patemotter): Allow per-channel requantization
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


class Fp8FusedMoEMethod(QuantizeMethodBase):
    """
    Fp8 method for JAXMoE layer.

    TODO (jacobplatin): support weight loading -- currently, model-dependent.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.extra_backend_kwargs = {}
        # TODO (jacobplatin): implement
        self.weight_block_size = [128, 128]
        self.block_quant: bool = self.weight_block_size is not None
        self.weight_scale_name = ("weight_scale_inv"
                                  if self.block_quant else "weight_scale")

    def create_weights_jax(self, layer: JaxMoE, *weight_args,
                           **extra_weight_attrs):
        num_experts = layer.num_local_experts
        intermediate_size = layer.intermediate_size_moe
        hidden_size = layer.hidden_size

        quant_config = layer.quant_config
        assert isinstance(
            quant_config,
            Fp8Config), "Expected fp8 config for Fp8FusedMoEMethod!"

        if layer.moe_backend in [MoEBackend.GMM_EP, MoEBackend.GMM_TP]:
            if not self.block_quant:
                # For per-tensor quant, the scales are per expert and weight.
                # w13_scale_data = jnp.ones(num_experts, 2, dtype=jnp.float32)
                # w2_scale_data = jnp.ones(num_experts, dtype=jnp.float32)
                raise NotImplementedError
            else:
                block_n, block_k = (
                    self.weight_block_size[0],
                    self.weight_block_size[1],
                )
                # For block quant, the scales are per block (typically 128x128).
                w13_scale_data = jnp.ones(
                    (
                        num_experts,
                        2 * ((intermediate_size + block_k - 1) // block_k),
                        (hidden_size + block_n - 1) // block_n,
                    ),
                    dtype=jnp.float32,
                )
                w2_scale_data = jnp.ones(
                    (num_experts, (hidden_size + block_k - 1) // block_k,
                     (intermediate_size + block_n - 1) // block_n),
                    dtype=jnp.float32,
                )
                setattr(layer,
                        f"kernel_gating_upproj_EDF_{self.weight_scale_name}",
                        nnx.Param(w13_scale_data, dtype=jnp.float32))
                setattr(layer,
                        f"kernel_down_proj_EFD_{self.weight_scale_name}",
                        nnx.Param(w2_scale_data, dtype=jnp.float32))
        else:
            raise NotImplementedError(
                "TODO (jacobplatin): implement create_weights_jax for FUSED_MOE backend"
            )

    def process_weights_after_loading(self, layer: JaxMoE) -> None:
        """
        Process weights after loading.

        Args:
            layer: The layer to process.
        """
        if layer.moe_backend == MoEBackend.FUSED_MOE:
            raise NotImplementedError(
                "TODO (jacobplatin): implement process_weights_after_loading for FUSED_MOE backend"
            )

        elif layer.moe_backend in [MoEBackend.GMM_EP, MoEBackend.GMM_TP]:
            w_gate = layer.kernel_gating_EDF.value
            w_up = layer.kernel_up_proj_EDF.value

            # Fuse the weights into w13: [Gate, Up]
            w13_weight = jnp.concatenate([w_gate, w_up], axis=-1)
            # NOTE: this is needed because the GMM kernels expect the RHS
            # to be transposed for w13
            w13_weight = jnp.transpose(w13_weight, (0, 2, 1))
            w13_weight_scale = getattr(
                layer,
                f"kernel_gating_upproj_EDF_{self.weight_scale_name}").value

            w2_weight = layer.kernel_down_proj_EFD.value
            w2_weight_scale = getattr(
                layer, f"kernel_down_proj_EFD_{self.weight_scale_name}").value

            weights = process_fp8_moe_weights(
                w13_weight=w13_weight,
                w13_weight_scale=w13_weight_scale,
                w2_weight=w2_weight,
                w2_weight_scale=w2_weight_scale,
                moe_backend=layer.moe_backend,
                mesh=layer.mesh,
                activation=layer.activation,
                # Convert to tuple so jax jit can hash it
                weight_block_size=tuple(self.weight_block_size)
                if self.weight_block_size is not None else None,
            )

            # TODO (jacobplatin): we probably want to make the sharding configurable
            layer.kernel_gating_upproj_EDF = nnx.Param(
                weights.w13_weight, sharding=layer.efd_sharding)
            layer.kernel_down_proj_EFD = nnx.Param(weights.w2_weight,
                                                   sharding=layer.edf_sharding)
            setattr(
                layer, f"kernel_gating_upproj_EDF_{self.weight_scale_name}",
                nnx.Param(weights.w13_weight_scale,
                          sharding=layer.efd_sharding))
            setattr(
                layer, f"kernel_down_proj_EFD_{self.weight_scale_name}",
                nnx.Param(weights.w2_weight_scale,
                          sharding=layer.edf_sharding))

            del layer.kernel_gating_EDF
            del layer.kernel_up_proj_EDF

    def apply_jax(self, layer: JaxModule, x: jax.Array) -> jax.Array:
        assert isinstance(layer, JaxMoE)

        x_TD = jnp.asarray(x, layer.dtype)
        x_TD = nnx.with_sharding_constraint(x_TD, layer.activation_ffw_td)

        router_logits = None
        # Fused weight backends
        if layer.moe_backend in MoEBackend.fused_moe_backends():
            # of shape TE, only 1D in this case
            router_logits = layer.router(x_TD)

            w13_weight = layer.kernel_gating_upproj_E2DF.value if layer.moe_backend == MoEBackend.FUSED_MOE else layer.kernel_gating_upproj_EDF.value
            w2_weight = layer.kernel_down_proj_EFD.value

            w13_weight_scale = getattr(
                layer,
                f"kernel_gating_upproj_EDF_{self.weight_scale_name}").value

            w2_weight_scale = getattr(
                layer, f"kernel_down_proj_EFD_{self.weight_scale_name}").value

            # TODO (jacobplatin/bzgoogle): we should support bias
            weights = FusedMoEWeights(
                w13_weight=w13_weight,
                w13_weight_scale=w13_weight_scale,
                w13_bias=None,
                w2_weight=w2_weight,
                w2_weight_scale=w2_weight_scale,
                w2_bias=None,
            )
        elif layer.moe_backend in [
                MoEBackend.DENSE_MAT, MoEBackend.MEGABLX_GMM
        ]:
            raise NotImplementedError("TODO (jacobplatin)")
        else:
            raise ValueError(f"Unsupported moe backend {layer.moe_backend}")

        return moe_apply(layer, x_TD, router_logits, weights,
                         layer.moe_backend, layer.mesh,
                         self.extra_backend_kwargs)


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
        elif isinstance(layer, JaxMoE):
            return Fp8FusedMoEMethod()
        return None
