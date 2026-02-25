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
import os
from functools import partial
from typing import Iterable, Optional, Tuple

import jax
import jax.numpy as jnp
import torch
from flax import nnx
from jax.sharding import PartitionSpec as P
from torchax.ops.mappings import t2j

from tpu_inference.layers.common.linear import sharded_quantized_batched_matmul
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

        adapt_info = linear_config.get_adapt_info(einsum_str=layer.einsum_str,
                                                  weight=layer.weight)
        self.output_shape = adapt_info.out_features
        self.batch_features = adapt_info.batch_features
        self.batch_sharding = adapt_info.batch_sharding
        out_features = math.prod(self.output_shape)
        in_features = math.prod(adapt_info.in_features)
        if self.batch_features:
            # Batched case: keep original weight sharding for the full
            # 3D weight (matches kernel_shape).
            self.weight_sharding = _to_partition_spec(layer.weight.sharding)
            self.kernel_shape = layer.kernel_shape
        else:
            self.weight_sharding = adapt_info.out_features_sharding + adapt_info.in_features_sharding
            self.kernel_shape = (out_features, in_features)

        self.linear_config.output_sizes = [out_features]
        self.in_features = in_features

    def create_weights_jax(self, layer: JaxEinsum, *weight_args, rngs,
                           **extra_weight_attrs):
        assert isinstance(layer, JaxEinsum)

        out_features = sum(self.linear_config.output_sizes)

        layer.weight = create_param(rngs,
                                    shape=self.kernel_shape,
                                    dtype=jnp.float8_e4m3fn,
                                    sharding=self.weight_sharding)

        # Attach custom loader to avoid default upcasting behavior
        setattr(
            layer.weight, "weight_loader",
            functools.partial(load_fp8_weight,
                              param_name=layer.prefix + ".weight"))

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

        adapt_info = linear_config.get_adapt_info(einsum_str=layer.einsum_str,
                                                  weight=layer.weight)
        self.out_features = adapt_info.out_features
        self.in_features = math.prod(adapt_info.in_features)
        self.batch_features = adapt_info.batch_features
        self.batch_sharding = adapt_info.batch_sharding
        if len(layer.weight.value.shape) > 2:
            # 3D weight (matches kernel_shape).
            self.weight_sharding = _to_partition_spec(layer.weight.sharding)
            self.kernel_shape = layer.kernel_shape
            self.input_side_indices = adapt_info.input_side_indices
            self.output_side_indices = adapt_info.output_side_indices
        else:
            self.weight_sharding = (adapt_info.out_features_sharding +
                                    adapt_info.in_features_sharding)
            self.kernel_shape = (math.prod(self.out_features),
                                 self.in_features)
            # Optimized 2D layout is always (Out, In)
            self.input_side_indices = (1,)
            self.output_side_indices = (0,)
        self.bias_sharding = adapt_info.out_features_sharding

        # Multi dimensional kernels are flattened to 2D when loading the fp8 weights.
        self.in_features_total = math.prod(self.kernel_shape[i] for i in self.input_side_indices)
        self.out_features_total = math.prod(self.kernel_shape[i] for i in self.output_side_indices)
        # Output sizes also need to be flattened (which is what out_features_total does).
        self.linear_config.output_sizes = [self.out_features_total]
        
        # Preserve original sharding for requantization layout.
        self.original_weight_sharding = _to_partition_spec(layer.weight.sharding)

    def create_weights_jax(self, layer: JaxModule, *weight_args, rngs,
                           **extra_weight_attrs):
        assert isinstance(layer, JaxEinsum)

        kernel_init = layer.kernel_init
        mesh = jax.make_mesh((1, ), ('x', ), devices=jax.devices('cpu'))
        # Follow upstream limitation that only float8_e4m3 is supported.
        # https://github.com/vllm-project/vllm/blob/2a99c5a6c86daef8c766ba2dbf05c385b192c64b/vllm/model_executor/layers/quantization/fp8.py#L283-L284
        param_dtype = jnp.float8_e4m3

        # Initialize the layer.weight as a 2D CPU placeholder to load the fp8 checkpoint weights.
        # Checkpoints typically store linear weights as (OutTotal, InTotal).
        layer.weight = nnx.Param(
            kernel_init(rngs.params(), (self.out_features_total, self.in_features_total), param_dtype),
            weight_loader=partial(load_nnx_param_from_reshaped_torch,
                                  reshape_dims=None,
                                  permute_dims=(0, 1),
                                  param_name=layer.prefix + ".weight"), 
            _is_loaded=False)
        layer.weight.get_metadata()['mesh'] = mesh
        layer.weight.sharding = () # Weight loading on CPU is unsharded.

        # Block-wise quantization scales to be loaded.
        block_n, block_k = self.quant_config.weight_block_size[0], self.quant_config.weight_block_size[1]
        
        out_blocks = (self.out_features_total + block_n - 1) // block_n
        in_blocks = (self.in_features_total + block_k - 1) // block_k
        scale_shape = (out_blocks, in_blocks)

        layer.weight_scale_inv = nnx.Param(
            kernel_init(rngs.params(), scale_shape, layer.dtype),
            weight_loader=partial(
                load_nnx_param_from_reshaped_torch,
                reshape_dims=None,
                permute_dims=(0, 1), 
                param_name=layer.prefix + ".weight_scale_inv",
            ),
            _is_loaded=False)
        layer.weight_scale_inv.get_metadata()['mesh'] = mesh
        layer.weight_scale_inv.sharding = ()

    def process_weights_after_loading(self, layer):
        logger.warning(f"PROCESS_START: {layer.prefix} | weight_loaded={getattr(layer.weight, '_is_loaded', False)} | scale_loaded={getattr(layer.weight_scale_inv, '_is_loaded', False)}")
        assert isinstance(layer, JaxEinsum)
        assert self.quant_config.weight_block_size is not None

        if not hasattr(layer, 'weight') or not hasattr(layer, 'weight_scale_inv'):
            return

        if not getattr(layer.weight, "_is_loaded", False) or not getattr(
                layer.weight_scale_inv, "_is_loaded", False):
            # Weight and scale could spread across multiple files,
            # so we only process once both of them are loaded.
            return

        with jax.set_mesh(
                jax.make_mesh((1, ), ('x', ), devices=jax.devices('cpu'))):
            # Weights & scales are 2D based on create_weights_jax
            weight_2d = layer.weight.value
            weight_scale_inv = layer.weight_scale_inv.value

            weight_block_size = tuple(self.quant_config.weight_block_size)
            old_output_sizes = self.linear_config.output_sizes
            self.linear_config.output_sizes = [self.out_features_total]

            bias = layer.bias.value if getattr(layer, 'bias', None) is not None else None
            if bias is not None:
                bias = bias.reshape(-1)
            
            weights = common_fp8.process_blockwise_fp8_linear_weights(
                weight_2d,
                weight_scale_inv,
                bias=bias,
                weight_block_size=weight_block_size,
                linear_config=self.linear_config)

            # Convert the requantized 2D results back to the 3D layout if necessary.
            logical_output_shape = tuple(self.kernel_shape[i] for i in self.output_side_indices)
            weights.weight_scale = weights.weight_scale.reshape(logical_output_shape)

            if len(self.kernel_shape) > 2:
                if self.output_side_indices[0] == 2:
                    weights.weight = weights.weight.T.reshape(self.kernel_shape)
                else:
                    # Some 3D weights like v_up_proj in deepseek-ai/DeepSeek-R1 are transposed
                    weights.weight = weights.weight.reshape(self.kernel_shape)
            else:
                weights.weight = weights.weight.reshape(self.kernel_shape)
            
            self.linear_config.output_sizes = old_output_sizes

            # Determine appropriate scale sharding for broadcasting.
            scale_sharding = P(*(self.original_weight_sharding[i] for i in self.output_side_indices))

            delattr(layer, 'weight')
            delattr(layer, 'weight_scale_inv')
            if bias is not None:
                delattr(layer, 'bias')

            if self.linear_config.enable_quantized_matmul_kernel and not self.batch_features:
                weights.weight_scale = jnp.expand_dims(
                    jnp.transpose(weights.weight_scale), axis=1)
            
            final_w = weights.weight
            final_s = weights.weight_scale
            final_b = weights.bias

        # Perform sharded device placement outside of the CPU block.
        mesh = self.linear_config.mesh
        
        if self.linear_config.fuse_matmuls:
            layer.weight = nnx.Param(shard_put(final_w, self.original_weight_sharding, mesh=mesh))
            layer.weight_scale_inv = nnx.Param(shard_put(final_s, scale_sharding, mesh=mesh))
            if final_b is not None:
                layer.bias = nnx.Param(shard_put(final_b, self.bias_sharding, mesh=mesh))
        else:
            raise NotImplementedError("Fp8 block-wise linear method only supports fuse_matmuls.")

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
        self._called_process_weights_after_loading = False

        # Parse requantization settings from environment variables
        self.requant_weight_dtype = self._parse_moe_requant_dtype(
            os.environ.get("MOE_REQUANT_WEIGHT_DTYPE"))

        requant_block_size = os.environ.get("MOE_REQUANT_BLOCK_SIZE")
        self.requant_block_size = (int(requant_block_size)
                                   if requant_block_size else None)

    def _parse_moe_requant_dtype(self, dtype_str: str | None) -> jnp.dtype:
        if dtype_str is None:
            return jnp.float8_e4m3fn
        dtype_str = dtype_str.lower()
        if dtype_str in ["fp4", "float4_e2m1fn"]:
            return jnp.float4_e2m1fn
        if dtype_str in ["fp8", "float8_e4m3fn"]:
            return jnp.float8_e4m3fn
        raise ValueError(f"Unsupported requant dtype: {dtype_str}")

    def load_weights(self, *, layer: JaxMoE, original_load_weights_fn,
                     weights: Iterable) -> set:
        """Load scale paramters and delegate the weight paramters to `original_load_weights_fn`"""

        # Remaining non-scale parameters will be loaded using original load_weights function.
        remaining_weights = dict()
        cnt = 0
        for torch_name, torch_weight in weights:
            torch_name: str = torch_name.split(
                layer.prefix)[-1]  # ".0.down_proj.weight" for example
            names = torch_name.split(".")[-3:]
            assert len(
                names
            ) == 3, f"Expected param name to be .<expert_id>.<param_name>.weight, got {torch_name=} {layer.prefix=} {type(layer)=}"
            expert_id, _, _ = names
            expert_id = int(expert_id)
            jax_param_name = ""
            if torch_name.endswith("up_proj." + self.weight_scale_name):
                jax_param_name = "kernel_up_proj_EDF_" + self.weight_scale_name
            elif torch_name.endswith("down_proj." + self.weight_scale_name):
                jax_param_name = "kernel_down_proj_EFD_" + self.weight_scale_name
            elif torch_name.endswith("gate_proj." + self.weight_scale_name):
                jax_param_name = "kernel_gating_EDF_" + self.weight_scale_name
            else:
                remaining_weights[torch_name] = torch_weight
                continue
            cnt += 1
            jax_param = getattr(layer, jax_param_name, None)

            assert isinstance(jax_param, nnx.Param)

            jax_weight = jax_array_from_reshaped_torch(
                torch_weight, reshape_dims=(1, ) +
                torch_weight.shape)  # add expert dim for concatenation later
            jax_param._weights_to_load[expert_id] = jax_weight

        logger.debug(
            f"Loaded {cnt} weight scales for {layer.prefix} MoE layer.")

        loaded_names = original_load_weights_fn(remaining_weights.items())
        for param_name in {
                "kernel_gating_EDF_" + self.weight_scale_name,
                "kernel_up_proj_EDF_" + self.weight_scale_name,
                "kernel_down_proj_EFD_" + self.weight_scale_name,
        }:
            param = getattr(layer, param_name)
            if all(w is not None for w in param._weights_to_load):
                loaded_names.add(param_name)

        return loaded_names

    def create_weights_jax(self, layer: JaxMoE, *weight_args, rngs,
                           **extra_weight_attrs) -> None:
        """
        Create the quant method-specific weights.

        Args:
            layer: The layer to create weights for.
        """

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
                    self.weight_block_size
                ) == 2, f"Expected 2D block size, got {self.weight_block_size}"
                block_n, block_k = self.weight_block_size

                # re-create the weights to be in fp8 type
                for param_name in [
                        "kernel_gating_EDF", "kernel_up_proj_EDF",
                        "kernel_down_proj_EFD"
                ]:
                    param = getattr(layer, param_name, None)
                    assert isinstance(
                        param, nnx.Param
                    ), f"Expected nnx.Param for {param_name}, got {type(param)}"
                    init_fn = param.init_fn
                    E, K, N = param.value.shape
                    value = init_fn(rngs.params(), (E, K, N),
                                    jnp.float8_e4m3fn)
                    param.value = value

                    scale_value = jnp.zeros((E, (K + block_k - 1) // block_k,
                                             (N + block_n - 1) // block_n),
                                            device=jax.devices('cpu')[0])
                    setattr(
                        layer, f"{param_name}_{self.weight_scale_name}",
                        nnx.Param(scale_value,
                                  _weights_to_load=[None for _ in range(E)]))
        else:
            raise NotImplementedError(
                f"Unsupported moe backend: {layer.moe_backend}! Currently supported: {FP8_QUANT_METHOD_SUPPORTED_MOE_BACKENDS}"
            )

    def process_weights_after_loading(self, layer: JaxMoE) -> None:
        # only to make mem profiler show clear trace paths
        # TODO: remove this
        return self.process_moe_after_loading(layer)

    def process_moe_after_loading(self, layer: JaxMoE) -> None:
        """
        Process weights after loading.

        Args:
            layer: The layer to process.
        """
        # TODO (#1681): support other backends
        if self._called_process_weights_after_loading:
            return

        if layer.moe_backend in FP8_QUANT_METHOD_SUPPORTED_MOE_BACKENDS:
            gating_scale_name = f"kernel_gating_EDF_{self.weight_scale_name}"
            up_scale_name = f"kernel_up_proj_EDF_{self.weight_scale_name}"
            down_scale_name = f"kernel_down_proj_EFD_{self.weight_scale_name}"

            if any(
                    any(w is None for w in param._weights_to_load) for param in
                [
                    getattr(layer, gating_scale_name),
                    getattr(layer, up_scale_name),
                    getattr(layer, down_scale_name), layer.kernel_gating_EDF,
                    layer.kernel_up_proj_EDF, layer.kernel_down_proj_EFD
                ]):
                # If weights for a module is spread across multiple files, this function may be called
                # more than once. We only want to process the weights once all of them are loaded.
                return

            self._called_process_weights_after_loading = True
            with jax.set_mesh(
                    jax.make_mesh((1, ), ('x', ), devices=jax.devices('cpu'))):
                w_gate = jnp.concatenate(
                    layer.kernel_gating_EDF._weights_to_load, axis=0)
                w_up = jnp.concatenate(
                    layer.kernel_up_proj_EDF._weights_to_load, axis=0)
                s_gate = jnp.concatenate(getattr(
                    layer, gating_scale_name)._weights_to_load,
                                         axis=0)
                s_up = jnp.concatenate(getattr(layer,
                                               up_scale_name)._weights_to_load,
                                       axis=0)
                w2_weight = jnp.concatenate(
                    layer.kernel_down_proj_EFD._weights_to_load, axis=0)
                w2_weight_scale = jnp.concatenate(getattr(
                    layer, down_scale_name)._weights_to_load,
                                                  axis=0)

                # Fuse the weights into w13: [Gate, Up]
                w13_weight = jnp.concatenate([w_gate, w_up], axis=1)
                w13_weight_scale = jnp.concatenate([s_gate, s_up], axis=1)

                weight_block_size = None
                if self.weight_block_size is not None:
                    weight_block_size = tuple(self.weight_block_size)

                # TODO (jacobplatin): we should support bias
                input_weights = FusedMoEWeights(
                    w13_weight=w13_weight,
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
                    requant_dtype=self.requant_weight_dtype,
                    requant_block_size=self.requant_block_size,
                )

            del layer.kernel_gating_EDF
            del layer.kernel_up_proj_EDF
            delattr(layer, gating_scale_name)
            delattr(layer, up_scale_name)

            # TODO (jacobplatin): we probably want to make the sharding configurable
            layer.kernel_gating_upproj_EDF = nnx.Param(
                shard_put(weights.w13_weight, shardings=layer.edf_sharding))
            layer.kernel_down_proj_EFD = nnx.Param(
                shard_put(weights.w2_weight, shardings=layer.efd_sharding))
            # NOTE: we aren't sharding the weight scales
            setattr(
                layer, f"kernel_gating_upproj_EDF_{self.weight_scale_name}",
                nnx.Param(
                    shard_put(weights.w13_weight_scale, shardings=(None, ))))
            setattr(
                layer, f"kernel_down_proj_EFD_{self.weight_scale_name}",
                nnx.Param(
                    shard_put(weights.w2_weight_scale, shardings=(None, ))))
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
            # Calculate logical output size across all dimensions not shared with activations.
            adapt_info = QuantLinearConfig.get_adapt_info(
                einsum_str=layer.einsum_str, weight=layer.weight)
            out_features_total = math.prod(adapt_info.out_features)

            linear_config = QuantLinearConfig(
                output_sizes=[out_features_total], enable_sp=False)
            if self.is_layer_skipped(prefix,
                                     ignored_layers=self.ignored_layers):
                return UnquantizedLinearMethod(linear_config)
            if self.weight_block_size is not None:
                return Fp8BlockwiseLinearMethod(self, layer, linear_config)
            else:
                return Fp8TensorwiseLinearMethod(layer, linear_config)
        elif isinstance(layer, JaxMoE):
            if self.is_layer_skipped(prefix,
                                     ignored_layers=self.ignored_layers):
                return UnquantizedFusedMoEMethod()
            return Fp8FusedMoEMethod(self.weight_block_size)
        return None
