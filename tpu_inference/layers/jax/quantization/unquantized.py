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
import gc
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
from tpu_inference.layers.jax.linear import (JaxEinsum,
                                             JaxMergedColumnParallelLinear)
from tpu_inference.layers.jax.moe.moe import JaxMoE
from tpu_inference.layers.jax.quantization import QuantizeMethodBase
from tpu_inference.layers.jax.quantization.configs import QuantizationConfig
from tpu_inference.logger import init_logger
from tpu_inference.models.jax.utils.weight_utils import (
    assign_and_shard_param, jax_array_from_reshaped_torch, shard_put)

logger = init_logger(__name__)


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


class UnquantizedMergedLinearMethod(UnquantizedLinearMethod):
    """Unquantized method for JaxMerged*Linear layers.

    A merged column-parallel linear (e.g. fused gate_up_proj) holds the weights
    of several logical projections concatenated along the output dim of a single
    kernel. The checkpoint, however, ships each projection as a separate tensor
    (``gate_proj.weight``, ``up_proj.weight``).

    ``UnquantizedLinearMethod._apply_fused`` already un-interleaves the fused
    output back into the per-projection pieces via
    ``slice_sharded_tensor_for_concatenation(out, output_sizes, n_shards)``,
    which reshapes the output dim as ``(n_shards, -1)`` and reads each
    projection's per-shard slice. For that to be correct the kernel must be
    stored *interleaved* by shard: shard ``i`` holds
    ``[proj0_slice_i, proj1_slice_i, ...]``.

    ``create_weights_jax`` attaches a ``weight_loader`` that accumulates each
    projection's checkpoint tensor (by ``shard_id``) and, once all are present,
    builds that interleaved layout — the inverse of the slicing done at apply
    time, so loading and forward stay consistent (and TP-co-located whenever
    ``n_shards`` matches the model-parallel degree).
    """

    def create_weights_jax(self, layer: JaxEinsum, *weight_args, rngs,
                           **extra_weight_attrs):
        # The fused kernel (shape `(in, sum(output_sizes))` with the output
        # dim sharded) is already created by `JaxEinsum.__init__`. Keep that
        # param (so its partition metadata survives) and attach:
        #   * a per-projection accumulation buffer, and
        #   * a weight_loader that stashes each projection's tensor and fuses
        #     them once all shards have arrived.
        assert isinstance(layer, JaxEinsum)
        n_proj = len(self.linear_config.output_sizes)
        layer.weight.set_metadata("_merged_shards", [None] * n_proj)
        layer.weight.set_metadata(
            "weight_loader",
            functools.partial(self._load_merged_weight,
                              n_shards=self.linear_config.n_shards,
                              output_sizes=self.linear_config.output_sizes,
                              param_name=layer.prefix + ".weight"))

    @staticmethod
    def _load_merged_weight(param: nnx.Param, torch_weight, shard_id: int, *,
                            n_shards: int, output_sizes: list,
                            param_name: str):
        """Accumulate one projection's checkpoint tensor, fuse when complete.

        Called once per fused projection with ``shard_id`` selecting the slot
        (e.g. gate=0, up=1). ``torch_weight`` has the HF layout ``(out_i, in)``.
        The projections may arrive across multiple files, so the fuse only runs
        once every slot is filled.
        """
        shards = param.get_metadata("_merged_shards")
        shards[shard_id] = torch_weight
        if any(s is None for s in shards):
            return

        # The per-projection tensors are converted on the host (CPU); the
        # interleaving reshape/concat must run in the same CPU mesh context,
        # otherwise it conflicts with the ambient TPU mesh active during weight
        # loading. assign_and_shard_param then places the result on device.
        with cpu_mesh_context():
            interleaved_pieces = []
            for out_size, torch_w in zip(output_sizes, shards):
                assert out_size % n_shards == 0, (
                    f"Output size {out_size} not divisible by n_shards "
                    f"{n_shards}")
                # HF (out_i, in) -> (in, out_i) to match the kernel's
                # contracting-last layout, then split the output dim across
                # shards.
                w = jax_array_from_reshaped_torch(torch_w, permute_dims=(1, 0))
                in_size = w.shape[0]
                interleaved_pieces.append(
                    w.reshape(in_size, n_shards, out_size // n_shards))

            # Concatenate within each shard block so shard i ends up holding
            # [proj0_slice_i, proj1_slice_i, ...], then flatten to (in, total).
            # E.g. with gate and up weights (2,6)x2,
            #
            #     GGGGGG UUUUUU
            #     GGGGGG UUUUUU
            #
            # reshaped to (2,2,3)x2
            #     GGG GGG UUU UUU
            #     GGG GGG UUU UUU
            #
            # concatenated to (2,2,6)
            #     GGGUUU GGGUUU
            #     GGGUUU GGGUUU
            #
            # reshaped to (2,12)
            #     GGGUUUGGGUUU
            #     GGGUUUGGGUUU
            fused = jnp.concatenate(interleaved_pieces, axis=2)
            fused = fused.reshape(fused.shape[0], -1)

        assign_and_shard_param(param, fused, param_name=param_name)


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
            if any(
                    any(w is None for w in param._weights_to_load)
                    for param in [
                        layer.kernel_gating_EDF, layer.kernel_up_proj_EDF,
                        layer.kernel_down_proj_EFD
                    ]):
                return False
            w_gate = layer.kernel_gating_EDF.get_value()
            w_up = layer.kernel_up_proj_EDF.get_value()
            w2_val = layer.kernel_down_proj_EFD.get_value()

            # Free old params before processing to reduce peak memory.
            del layer.kernel_gating_EDF
            del layer.kernel_up_proj_EDF

            # Fuse the weights into w13: [Gate, Up]
            w13_val = jnp.concatenate([w_gate, w_up], axis=1)
            del w_gate, w_up

            mesh = jax.sharding.get_mesh()
            weights = process_unquantized_moe_weights(
                mesh=mesh,
                moe_backend=layer.moe_backend,
                activation=layer.activation,
                w13_weight=w13_val,
                w13_bias=None,
                w2_weight=w2_val,
                w2_bias=None,
            )

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
        # JaxMergedColumnParallelLinear is a JaxEinsum subclass whose single
        # kernel fuses several projections; it must report each projection's
        # size via output_sizes so the forward (and weight loading) can
        # interleave/de-interleave per shard. Checked before the generic
        # JaxEinsum branch. Imported locally to avoid an import cycle
        # (linear.py imports this quantization package).
        if isinstance(layer, JaxMergedColumnParallelLinear):
            linear_config = QuantLinearConfig(enable_sp=False,
                                              output_sizes=list(
                                                  layer.output_sizes))
            return UnquantizedMergedLinearMethod(linear_config)
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
