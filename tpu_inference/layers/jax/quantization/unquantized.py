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

import gc
from contextlib import nullcontext
from typing import Optional

import jax
import jax.numpy as jnp
from flax import nnx
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

from tpu_inference.layers.common.moe import MoEBackend, moe_apply
from tpu_inference.layers.common.process_weights.moe_weights import (
    FusedMoEWeights, UnfusedMoEWeights, process_unquantized_moe_weights,
    shard_moe_weights)
from tpu_inference.layers.common.quantization import unquantized as jax_common
from tpu_inference.layers.common.quantization.configs import QuantLinearConfig
from tpu_inference.layers.common.utils import cpu_mesh_context
from tpu_inference.layers.jax import JaxModule
from tpu_inference.layers.jax.linear import JaxEinsum
from tpu_inference.layers.jax.moe.moe import JaxMoE
from tpu_inference.layers.jax.quantization import QuantizeMethodBase
from tpu_inference.layers.jax.quantization.configs import QuantizationConfig
from tpu_inference.logger import init_logger
from tpu_inference.models.jax.utils.weight_utils import shard_put

logger = init_logger(__name__)

MOE_WEIGHTS_STAGED_ON_HOST = "_moe_weights_staged_on_host"


def _moe_staged_weights_ready(param: nnx.Param) -> bool:
    weights_to_load = getattr(param, "_weights_to_load", None)
    return bool(weights_to_load) and all(w is not None
                                         for w in weights_to_load)


def _concat_staged_moe_weights(param: nnx.Param) -> jax.Array:
    weights_to_load = param._weights_to_load
    if len(weights_to_load) == 1:
        return weights_to_load[0]
    return jnp.concatenate(weights_to_load, axis=0)


def _param_is_staged_on_host(param: nnx.Param) -> bool:
    return bool(param.get_metadata().get(MOE_WEIGHTS_STAGED_ON_HOST, False))


class UnquantizedLinearMethod(QuantizeMethodBase,
                              jax_common.UnquantizedLinearMethod):
    """Unquantized method for JAX Linear layer.
    """

    def apply_jax(self, layer: JaxModule, x: jax.Array) -> jax.Array:
        assert isinstance(layer, JaxEinsum)

        with jax.named_scope(layer._get_name()):
            if self.linear_config.fuse_matmuls:
                out = self._apply_fused(
                    x,
                    layer.weight.value,
                    layer.bias.value if layer.bias else None,
                    einsum_str=layer.einsum_str)
            else:
                raise NotImplementedError(
                    "Non-fused matmuls not implemented yet.")

        return out


class UnquantizedFusedMoEMethod(QuantizeMethodBase):
    """
    Unquantized method for JAXMoE layer.

    TODO (jacobplatin): support weight loading -- currently, model-dependent.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.extra_backend_kwargs = {}

    def process_weights_after_loading(self, layer: JaxMoE, *args,
                                      **kwargs) -> bool:
        """
        Process weights after loading.

        Please see https://github.com/vllm-project/tpu-inference/blob/bb1a88/tpu_inference/layers/common/moe.py#L39
        for more information on the expected weights per MoE backend.

        Args:
            layer: The layer to process.
        """
        if layer.moe_backend == MoEBackend.FUSED_MOE:
            if layer.edf_sharding:
                e2df_sharding = (layer.edf_sharding[0], None,
                                 layer.edf_sharding[1], layer.edf_sharding[2])
            # fuse the weights into w13: [Gate, Up]
            w_gate = layer.kernel_gating_EDF.value
            w_up = layer.kernel_up_proj_EDF.value

            # stack to create a 4d array
            w13_val = jnp.stack([w_gate, w_up], axis=1)

            layer.kernel_gating_upproj_E2DF = nnx.Param(
                shard_put(w13_val, shardings=e2df_sharding))

            del layer.kernel_gating_EDF
            del layer.kernel_up_proj_EDF

            ep_axis_name = layer.efd_sharding[0]

            self.extra_backend_kwargs = {
                "ep_axis_name": ep_axis_name,
                "bt": 32,
                "bf": 512,
                "bd1": 512,
                "bd2": 512,
                "btc": 64,
                "bfc": 256,
                "bd1c": 256,
                "bd2c": 256,
            }

        elif layer.moe_backend in [MoEBackend.GMM_EP, MoEBackend.GMM_TP]:

            moe_params = [
                layer.kernel_gating_EDF, layer.kernel_up_proj_EDF,
                layer.kernel_down_proj_EFD
            ]
            if any(not _moe_staged_weights_ready(param)
                   for param in moe_params):
                return False

            use_staged_host_weights = all(
                _param_is_staged_on_host(param) for param in moe_params)
            if use_staged_host_weights:
                with cpu_mesh_context():
                    w_gate = _concat_staged_moe_weights(
                        layer.kernel_gating_EDF)
                    w_up = _concat_staged_moe_weights(layer.kernel_up_proj_EDF)
                    w2_val = _concat_staged_moe_weights(
                        layer.kernel_down_proj_EFD)
            else:
                w_gate = layer.kernel_gating_EDF.get_value()
                w_up = layer.kernel_up_proj_EDF.get_value()
                w2_val = layer.kernel_down_proj_EFD.get_value()

            # Free old params before processing to reduce peak memory.
            del layer.kernel_gating_EDF
            del layer.kernel_up_proj_EDF
            del layer.kernel_down_proj_EFD

            # Fuse the weights into w13: [Gate, Up]
            # Capture the target TPU mesh outside cpu_mesh_context(); the CPU
            # mesh is only for temporary host-side processing.
            mesh = jax.sharding.get_mesh()
            if use_staged_host_weights:
                mesh_context = cpu_mesh_context()
            else:
                mesh_context = nullcontext()
            with mesh_context:
                w13_val = jnp.concatenate([w_gate, w_up], axis=1)
                del w_gate, w_up
                weights = process_unquantized_moe_weights(
                    mesh=mesh,
                    moe_backend=layer.moe_backend,
                    activation=layer.activation,
                    w13_weight=w13_val,
                    w13_bias=None,
                    w2_weight=w2_val,
                    w2_bias=None,
                )
                if use_staged_host_weights:
                    # Drop CPU sharding metadata before the final layout-aware
                    # transfer to the TPU mesh.
                    weights = jax.device_get(weights)

            sharded_weights = shard_moe_weights(weights,
                                                moe_backend=layer.moe_backend,
                                                mesh=mesh)

            layer.kernel_gating_upproj_EDF = nnx.Param(
                sharded_weights.w13_weight)
            layer.kernel_down_proj_EFD = nnx.Param(sharded_weights.w2_weight)

            # When MOE_REQUANTIZE_WEIGHT_DTYPE quantizes the bf16 weights at
            # load time, scales are produced by process_unquantized_moe_weights
            # and need to be stored alongside the weights so apply_jax can pass
            # them to the MoE kernel.
            if sharded_weights.w13_weight_scale is not None:
                layer.kernel_gating_upproj_EDF_weight_scale = nnx.Param(
                    sharded_weights.w13_weight_scale)
            if sharded_weights.w2_weight_scale is not None:
                layer.kernel_down_proj_EFD_weight_scale = nnx.Param(
                    sharded_weights.w2_weight_scale)

            # Wait for final transfers so the next MoE layer does not overlap
            # with transient buffers from this layer during weight loading.
            jax.block_until_ready(sharded_weights)

            del weights
            del w13_val
            del w2_val

            # Break reference cycles between JAX arrays and flax nnx.Param
            # objects created during weight processing. Without this, stale
            # arrays accumulate across MoE layers and inflate peak memory.
            gc.collect()

        return True

    def apply_jax(self, layer: JaxMoE, x: jax.Array, *,
                  router_logits: jax.Array) -> jax.Array:
        """Forward pass for MoE layer.
        Args:
            layer: The MoE layer to apply.
            x: The input activations to the MoE layer, of shape [seq_len, hidden_size].
            router_logits: The routing logits for the MoE layer, of shape [seq_len, num_experts].
        """
        x_TD = jnp.asarray(x, layer.dtype)
        x_TD = jax.lax.with_sharding_constraint(
            x_TD, NamedSharding(layer.mesh, P(*layer.activation_ffw_td)))

        # Fused weight backends
        if layer.moe_backend in MoEBackend.fused_moe_backends():
            # router_logits is of shape TE, only 1D in this case

            w13_weight = layer.kernel_gating_upproj_E2DF.value if layer.moe_backend == MoEBackend.FUSED_MOE else layer.kernel_gating_upproj_EDF.value
            w2_weight = layer.kernel_down_proj_EFD.value
            # Although this is UnquantizedMethod, when MOE_REQUANTIZE_WEIGHT_DTYPE
            # is set, the weights are quantized on the fly and scales are produced
            w13_scale = getattr(layer, "kernel_gating_upproj_EDF_weight_scale",
                                None)
            w2_scale = getattr(layer, "kernel_down_proj_EFD_weight_scale",
                               None)
            # TODO (jacobplatin/bzgoogle): we should support bias
            weights = FusedMoEWeights(
                w13_weight=w13_weight,
                w13_weight_scale=getattr(w13_scale, "value", None),
                w13_bias=None,
                w2_weight=w2_weight,
                w2_weight_scale=getattr(w2_scale, "value", None),
                w2_bias=None,
            )
        elif layer.moe_backend in [
                MoEBackend.DENSE_MAT, MoEBackend.MEGABLX_GMM
        ]:
            # router_logits is composed of weights_TX and indices_TX, so 2D in this case
            # TODO (jacobplatin/bzgoogle): we should support bias
            weights = UnfusedMoEWeights(
                w1_weight=layer.kernel_gating_EDF.value,
                w1_weight_scale=None,
                w1_bias=None,
                w2_weight=layer.kernel_up_proj_EDF.value,
                w2_weight_scale=None,
                w2_bias=None,
                w3_weight=layer.kernel_down_proj_EFD.value,
                w3_weight_scale=None,
                w3_bias=None,
            )

        else:
            raise ValueError(f"Unsupported moe backend {layer.moe_backend}")
        return moe_apply(layer, x_TD, router_logits, weights,
                         layer.moe_backend, layer.mesh,
                         self.extra_backend_kwargs)


class UnquantizedConfig(QuantizationConfig):

    def get_quant_method(self, layer: JaxModule,
                         prefix: str) -> Optional[QuantizeMethodBase]:
        if isinstance(layer, JaxEinsum):
            # Derive output's last dim from the einsum string.
            einsum_str = layer.einsum_str.replace(" ", "")
            _, w_axis = einsum_str.split("->")[0].split(",")
            last_out_char = einsum_str.split("->")[1][-1]
            out_size = layer.kernel_shape[w_axis.index(last_out_char)]

            linear_config = QuantLinearConfig(enable_sp=False,
                                              output_sizes=[out_size])
            return UnquantizedLinearMethod(linear_config)
        if isinstance(layer, JaxMoE):
            return UnquantizedFusedMoEMethod()
        return None
