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

import functools
import math
from functools import partial
from typing import Optional, Tuple

import jax
import jax.numpy as jnp
import torch
from flax import nnx
from jax._src.dtypes import TypePromotionError
from jax.sharding import PartitionSpec as P
from torchax.ops.mappings import t2j

from tpu_inference.layers.common.moe import MoEBackend, moe_apply
from tpu_inference.layers.common.process_weights.linear_weights import \
    shard_linear_weights
from tpu_inference.layers.common.process_weights.moe_weights import (
    FusedMoEWeights, process_fp8_moe_weights)
from tpu_inference.layers.common.quantization import fp8 as common_fp8
from tpu_inference.layers.common.quantization.configs import QuantLinearConfig
from tpu_inference.layers.jax import JaxModule
from tpu_inference.layers.jax.base import create_param
from tpu_inference.layers.jax.linear import JaxEinsum
from tpu_inference.layers.jax.moe.moe import JaxMoE
from tpu_inference.layers.jax.quantization import QuantizeMethodBase
from tpu_inference.layers.jax.quantization.configs import QuantizationConfig
from tpu_inference.layers.jax.quantization.unquantized import (
    UnquantizedFusedMoEMethod, UnquantizedLinearMethod)
from tpu_inference.logger import init_logger
from tpu_inference.models.jax.utils.weight_utils import (
    jax_array_from_reshaped_torch, load_nnx_param_from_reshaped_torch,
    shard_put)

logger = init_logger(__name__)

# TODO (jacobplatin): remove once we support all backends
FP8_QUANT_METHOD_SUPPORTED_MOE_BACKENDS = [
    MoEBackend.GMM_EP, MoEBackend.GMM_TP
]


def load_fp8_weight(jax_param: nnx.Param, torch_weight: torch.Tensor,
                    param_name: str):
    """Loads FP8 weights from a torch tensor into a JAX parameter.

    Args:
        jax_param: The nnx parameter to hold the FP8 weight.
        torch_weight: The source PyTorch tensor.
        param_name: Name of the parameter.
    """
    spec = jax_param.sharding
    if isinstance(jax_param.sharding, jax.sharding.NamedSharding):
        spec = jax_param.sharding.spec
    mesh = getattr(jax_param, 'mesh', None)

    jax_weight = t2j(torch_weight, use_dlpack=False)

    if jax_weight.dtype != jnp.float8_e4m3fn:
        logger.warning(
            f"Loading {param_name}: casting from {jax_weight.dtype} to {jax_param.value.dtype}"
        )
        jax_weight = jax_weight.astype(jax_param.value.dtype)

    jax_param.value = shard_put(jax_weight, spec, mesh=mesh)


def _to_partition_spec(sharding) -> P:
    """Convert a sharding value to a PartitionSpec.

    Handles NamedSharding (extracts .spec), raw tuples/lists from
    nnx.with_partitioning, and passthrough for existing PartitionSpec.
    """
    if isinstance(sharding, jax.sharding.NamedSharding):
        return sharding.spec
    if isinstance(sharding, P):
        return sharding
    if isinstance(sharding, (tuple, list)):
        return P(*sharding)
    return P()


class Fp8TensorwiseLinearMethod(QuantizeMethodBase,
                                common_fp8.Fp8LinearMethod):
    """Tensor-wise Fp8 method for JAX Linear layer."""

    def __init__(self, layer: JaxEinsum, linear_config: QuantLinearConfig):
        common_fp8.Fp8LinearMethod.__init__(self, linear_config)

        self.einsum_str = layer.einsum_str
        kernel_shape = layer.kernel_shape
        if len(kernel_shape) > 2:
            adapt_info = linear_config.get_adapt_info(
                einsum_str=layer.einsum_str, weight=layer.weight)
            self.output_shape = adapt_info.out_features
            self.batch_features = adapt_info.batch_features
            self.batch_sharding = adapt_info.batch_sharding
            out_features = math.prod(self.output_shape)
            in_features = math.prod(adapt_info.in_features)
            if self.batch_features:
                # Batched case: keep original weight sharding for the full
                # 3D weight (matches kernel_shape).
                self.weight_sharding = _to_partition_spec(
                    layer.weight.sharding)
            else:
                self.weight_sharding = adapt_info.weight_sharding
        else:
            in_features, out_features = kernel_shape
            # Reverse sharding to match transposed weight layout (out, in).
            sharding = layer.weight.sharding
            if isinstance(sharding, jax.sharding.NamedSharding):
                sharding = sharding.spec
            if isinstance(sharding, P) and len(sharding) == 2:
                sharding = P(sharding[1], sharding[0])
            elif isinstance(sharding, (tuple, list)) and len(sharding) == 2:
                sharding = (sharding[1], sharding[0])
            self.weight_sharding = sharding
            self.output_shape = (out_features, )
            self.batch_features = ()
            self.batch_sharding = ()

        self.linear_config.output_sizes = [out_features]
        self.in_features = in_features

    def create_weights_jax(self, layer: JaxModule, *weight_args, rngs,
                           **extra_weight_attrs):
        assert isinstance(layer, JaxEinsum)

        out_features = sum(self.linear_config.output_sizes)

        if self.batch_features:
            # Batched case: create weight with the original 3D kernel shape
            # so the weight loader can populate it directly after transpose.
            layer.weight = create_param(rngs,
                                        shape=layer.kernel_shape,
                                        dtype=jnp.float8_e4m3fn,
                                        sharding=self.weight_sharding)
        else:
            layer.weight = create_param(rngs,
                                        shape=(out_features, self.in_features),
                                        dtype=jnp.float8_e4m3fn,
                                        sharding=self.weight_sharding)

        # Attach custom loader to avoid default upcasting behavior
        layer.weight.set_metadata(
            'weight_loader',
            functools.partial(load_fp8_weight, param_name="weight"))

        # Scale is always per-output-channel (1D).
        scale_sharding = None
        if self.batch_features:
            # For batched weights, the output dim sharding comes from
            # the weight's non-contracting, non-batch axis.
            if self.batch_sharding:
                scale_sharding = None  # replicated scale for simplicity
        elif isinstance(self.weight_sharding, P) and len(
                self.weight_sharding) > 0:
            scale_sharding = P(self.weight_sharding[0])
        elif isinstance(self.weight_sharding,
                        (tuple, list)) and len(self.weight_sharding) > 0:
            scale_sharding = (self.weight_sharding[0], )

        layer.weight_scale = create_param(rngs,
                                          shape=(out_features, ),
                                          dtype=jnp.float32,
                                          sharding=scale_sharding)

    def apply_jax(self, layer: JaxModule, x: jax.Array) -> jax.Array:
        bias = layer.bias.value if layer.bias is not None else None

        if self.batch_features:
            # Batched case: use dot_general with batch dims.
            out = sharded_quantized_batched_matmul(
                x,
                layer.weight.value,
                layer.weight_scale.value,
                einsum_str=self.einsum_str,
                weight_sharding=self.weight_sharding,
                mesh=self.linear_config.mesh)
            if bias is not None:
                out += bias
            return out

        out = self._apply_fused(x,
                                layer.weight.value,
                                layer.weight_scale.value,
                                bias=bias)
        out = out.reshape(out.shape[:-1] + self.output_shape)
        return out


class Fp8BlockwiseLinearMethod(QuantizeMethodBase, common_fp8.Fp8LinearMethod):
    """Block-wise Fp8 method for JAX Linear layer."""

    def __init__(self, quant_config: "Fp8Config", layer: JaxEinsum,
                 linear_config: QuantLinearConfig):
        common_fp8.Fp8LinearMethod.__init__(self, linear_config)
        self.quant_config = quant_config
        self.einsum_str = layer.einsum_str

        kernel_shape = layer.kernel_shape
        if len(kernel_shape) > 2:
            adapt_info = linear_config.get_adapt_info(
                einsum_str=layer.einsum_str, weight=layer.weight)
            self.out_features = adapt_info.out_features
            self.in_features = math.prod(adapt_info.in_features)
            self.batch_features = adapt_info.batch_features
            self.batch_sharding = adapt_info.batch_sharding
            if self.batch_features:
                # Batched case: keep original weight sharding for the full
                # 3D weight (matches kernel_shape).
                self.weight_sharding = _to_partition_spec(
                    layer.weight.sharding)
            else:
                self.weight_sharding = adapt_info.weight_sharding
        else:
            in_features, out_features = kernel_shape
            self.weight_sharding = getattr(layer.weight, "sharding", (None, ))
            self.in_features = in_features
            self.out_features = (out_features, )
            self.batch_features = ()
            self.batch_sharding = ()

        # Storing list of output sizes (instead of self.out_features) for compatibility.
        self.linear_config.output_sizes = [math.prod(self.out_features)]

    def create_weights_jax(self, layer: JaxModule, *weight_args, rngs,
                           **extra_weight_attrs):
        assert isinstance(layer, JaxEinsum)

        out_features = sum(self.linear_config.output_sizes)
        kernel_init = layer.kernel_init

        if self.batch_features:
            # Batched case: create weight with the original 3D kernel shape
            # so the weight loader can populate it directly after transpose.
            # Weight stays in FP8 and is used with sharded_quantized_batched_matmul.
            param_dtype = jnp.float8_e4m3
            layer.weight = nnx.Param(
                kernel_init(rngs.params(), layer.kernel_shape, param_dtype),
                weight_loader=partial(load_nnx_param_from_reshaped_torch,
                                      permute_dims=None,
                                      param_name="linear_fp8_weight"),
                eager_sharding=False)
            layer.weight.set_metadata('sharding', self.weight_sharding)

            # Per-output-channel scale (1D, covers the free weight dim).
            layer.weight_scale_inv = nnx.Param(
                jnp.ones((out_features, ), dtype=layer.dtype),
                weight_loader=partial(
                    load_nnx_param_from_reshaped_torch,
                    permute_dims=None,
                    param_name="linear_fp8_weight_scale_inv",
                ),
                eager_sharding=False)
            layer.weight_scale_inv.set_metadata('sharding', ())
            return

        # Follow upstream limitation that only float8_e4m3 is supported.
        # https://github.com/vllm-project/vllm/blob/2a99c5a6c86daef8c766ba2dbf05c385b192c64b/vllm/model_executor/layers/quantization/fp8.py#L283-L284
        param_dtype = jnp.float8_e4m3
        layer.weight = nnx.Param(
            kernel_init(rngs.params(), (out_features, self.in_features),
                        param_dtype),
            weight_loader=partial(load_nnx_param_from_reshaped_torch,
                                  permute_dims=(0, 1),
                                  param_name="linear_fp8_weight"),
            eager_sharding=False)
        layer.weight.set_metadata('sharding', self.weight_sharding)

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
                param_name="linear_fp8_weight_scale_inv",
            ),
            eager_sharding=False)
        layer.weight_scale_inv.set_metadata('sharding', self.weight_sharding)

    def process_weights_after_loading(self, layer):
        assert isinstance(layer, JaxEinsum)
        assert self.quant_config.weight_block_size is not None

        if self.batch_features:
            # Batched case: weight stays in FP8. No blockwise processing
            # needed — the batched matmul uses dot_general with FP8 natively.
            return

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
        if self.batch_features:
            # Batched case: use dot_general with FP8 and batch dims.
            out = sharded_quantized_batched_matmul(
                x,
                layer.weight.value,
                layer.weight_scale_inv.value,
                einsum_str=self.einsum_str,
                weight_sharding=self.weight_sharding,
                mesh=self.linear_config.mesh)
            return out

        if not self.linear_config.fuse_matmuls:
            raise NotImplementedError(
                "Fp8 block-wise linear method only supports fuse_matmuls.")
        weight, scale = layer.weight.value, layer.weight_scale_inv.value
        bias = layer.bias.value if layer.bias is not None else None
        if len(x.shape) > 2:
            x = x.reshape(-1, self.in_features)
        out = self._apply_fused(x, weight, scale, bias=bias)
        out = out.reshape(out.shape[:-1] + self.out_features)
        return out


class Fp8FusedMoEMethod(QuantizeMethodBase):
    """
    Fp8 method for JAXMoE layer.

    TODO (jacobplatin): support weight loading -- currently, model-dependent.
    """

    def __init__(self, weight_block_size: Tuple[int, int], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.extra_backend_kwargs = {}
        self.weight_block_size = weight_block_size
        self.block_quant: bool = self.weight_block_size is not None
        self.weight_scale_name = ("weight_scale_inv"
                                  if self.block_quant else "weight_scale")

    def create_weights_jax(self, layer: JaxMoE, *weight_args,
                           **extra_weight_attrs) -> None:
        """
        Create the quant method-specific weights.

        Args:
            layer: The layer to create weights for.
        """

        num_experts = layer.num_local_experts
        intermediate_size = layer.intermediate_size_moe
        hidden_size = layer.hidden_size

        quant_config = layer.quant_config
        assert isinstance(
            quant_config,
            Fp8Config), "Expected fp8 config for Fp8FusedMoEMethod!"

        # TODO (#1681): support other backends
        if layer.moe_backend in FP8_QUANT_METHOD_SUPPORTED_MOE_BACKENDS:
            # vLLM reference here:
            # https://github.com/vllm-project/vllm/blob/9bdb06b/vllm/model_executor/layers/quantization/fp8.py#L763
            if not self.block_quant:
                raise NotImplementedError(
                    "Expected blockwise quantization when using Fp8FusedMoEMethod!"
                )
            else:
                assert len(
                    self.weight_block_size) == 2, "Expected 2D block size!"
                block_n, block_k = self.weight_block_size
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
                f"Unsupported moe backend: {layer.moe_backend}! Currently supported: {FP8_QUANT_METHOD_SUPPORTED_MOE_BACKENDS}"
            )

    def process_weights_after_loading(self, layer: JaxMoE) -> None:
        """
        Process weights after loading.

        Args:
            layer: The layer to process.
        """
        # TODO (#1681): support other backends
        if layer.moe_backend in FP8_QUANT_METHOD_SUPPORTED_MOE_BACKENDS:
            w_gate = layer.kernel_gating_EDF.value
            w_up = layer.kernel_up_proj_EDF.value

            # Fuse the weights into w13: [Gate, Up]
            w13_weight = jnp.concatenate([w_gate, w_up], axis=-1)
            # NOTE: this is needed because the GMM kernels expect the RHS
            # to be transposed for w13
            w13_weight = jnp.transpose(w13_weight, (0, 2, 1))
            # TODO (jacobplatin): make the string retrieval less fragile
            w13_weight_scale = getattr(
                layer,
                f"kernel_gating_upproj_EDF_{self.weight_scale_name}").value

            w2_weight = layer.kernel_down_proj_EFD.value
            w2_weight_scale = getattr(
                layer, f"kernel_down_proj_EFD_{self.weight_scale_name}").value

            weight_block_size = None
            if self.weight_block_size is not None:
                weight_block_size = tuple(self.weight_block_size)

            # TODO (jacobplatin): we should support bias
            input_weights = FusedMoEWeights(w13_weight=w13_weight,
                                            w13_weight_scale=w13_weight_scale,
                                            w13_bias=None,
                                            w2_weight=w2_weight,
                                            w2_weight_scale=w2_weight_scale,
                                            w2_bias=None)

            weights = process_fp8_moe_weights(
                input_weights,
                moe_backend=layer.moe_backend,
                mesh=layer.mesh,
                activation=layer.activation,
                # Convert to tuple so jax jit can hash it
                weight_block_size=weight_block_size,
            )

            # TODO (jacobplatin): we probably want to make the sharding configurable
            layer.kernel_gating_upproj_EDF = nnx.Param(
                weights.w13_weight, sharding=layer.edf_sharding)
            layer.kernel_down_proj_EFD = nnx.Param(weights.w2_weight,
                                                   sharding=layer.efd_sharding)
            # NOTE: we aren't sharding the weight scales
            setattr(layer,
                    f"kernel_gating_upproj_EDF_{self.weight_scale_name}",
                    nnx.Param(weights.w13_weight_scale))
            setattr(layer, f"kernel_down_proj_EFD_{self.weight_scale_name}",
                    nnx.Param(weights.w2_weight_scale))

            del layer.kernel_gating_EDF
            del layer.kernel_up_proj_EDF
        else:
            raise NotImplementedError(
                f"Unsupported moe backend: {layer.moe_backend}! Currently supported: {FP8_QUANT_METHOD_SUPPORTED_MOE_BACKENDS}"
            )

    def apply_jax(self, layer: JaxModule, x: jax.Array) -> jax.Array:
        """
        Run the forward pass of the MoE layer.

        Args:
            layer: The layer to apply the quantization method to.
            x: The input to the layer.

        Returns:
            The MoE output.
        """
        assert isinstance(layer, JaxMoE)

        x_TD = jnp.asarray(x, layer.dtype)
        x_TD = nnx.with_sharding_constraint(x_TD, layer.activation_ffw_td)

        router_logits = None
        # Fused weight backends
        if layer.moe_backend in FP8_QUANT_METHOD_SUPPORTED_MOE_BACKENDS:
            # of shape TE -- we don't return the indices
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
        else:
            raise NotImplementedError(
                f"Unsupported moe backend: {layer.moe_backend}! Currently supported: {FP8_QUANT_METHOD_SUPPORTED_MOE_BACKENDS}"
            )

        return moe_apply(layer, x_TD, router_logits, weights,
                         layer.moe_backend, layer.mesh,
                         self.extra_backend_kwargs)


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
            if self.is_layer_skipped(prefix,
                                     ignored_layers=self.ignored_layers):
                return UnquantizedLinearMethod(linear_config)
            if self.weight_block_size is not None:
                return Fp8BlockwiseLinearMethod(self, layer, linear_config)
            else:
                return Fp8TensorwiseLinearMethod(layer, linear_config)
        elif isinstance(layer, JaxMoE):
            return Fp8FusedMoEMethod(self.weight_block_size)
        return None
