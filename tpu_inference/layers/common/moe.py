# Copyright 2025 Google LLC
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

from enum import Enum
from typing import TYPE_CHECKING, Union

import jax
import jax.numpy as jnp
from jax.sharding import Mesh
from vllm.model_executor.layers.fused_moe import FusedMoE

from tpu_inference import envs
from tpu_inference.kernels.fused_moe.v1.kernel import fused_ep_moe
from tpu_inference.layers.common.fused_moe_gmm import fused_moe_func
from tpu_inference.logger import init_logger

if TYPE_CHECKING:
    from tpu_inference.layers.common.process_weights.moe_weights import (
        FusedMoEWeights, UnfusedMoEWeights)
    from tpu_inference.layers.jax.moe.moe import JaxMoE
else:
    FusedMoEWeights = None
    UnfusedMoEWeights = None
    JaxMoE = None

logger = init_logger(__name__)


class MoEBackend(Enum):
    FUSED_MOE = "fused_moe"
    GMM_EP = "gmm_ep"
    GMM_TP = "gmm_tp"
    DENSE_MAT = "dense_mat"  # only used in the JAX path for now
    MEGABLX_GMM = "megablox_gmm"  # only used in the JAX path for now
    RAGGED_DOT = "ragged_dot_gmm"  # only used in the JAX path for now


def select_moe_backend(use_ep: bool):
    if envs.USE_MOE_EP_KERNEL:
        logger.info("[MoE]: Using fused MoE EP kernel")
        if use_ep:
            return MoEBackend.FUSED_MOE

    if envs.USE_MEGABLOCKS:
        logger.info("[MoE]: Mega Blocks is enabled for GMM in Sparse Matmul")
        return MoEBackend.MEGABLX_GMM

    if envs.USE_RAGGED_DOT:
        logger.info("[MoE]: Ragged Dot is enabled for GMM in Sparse Matmul")
        return MoEBackend.RAGGED_DOT

    if use_ep:
        logger.warning_once(
            "USE_MOE_EP_KERNEL=1 but expert parallelism is not "
            "enabled. Falling back to gmm implementation.")
        return MoEBackend.GMM_EP

    # Use default implementation.
    return MoEBackend.GMM_TP


def moe_apply(
    layer: FusedMoE | JaxMoE,
    x: jax.Array,
    gating_output: jax.Array,
    weights: Union[FusedMoEWeights, UnfusedMoEWeights],
    moe_backend: MoEBackend,
    mesh: Mesh,
    extra_backend_kwargs: dict,
) -> jax.Array:
    # TODO: JaxMoE
    assert isinstance(layer, (FusedMoE, JaxMoE))
    if layer.scoring_func != "softmax":
        raise NotImplementedError("Only softmax is supported for scoring_func")

    with jax.named_scope(layer._get_name()):
        match moe_backend:
            case MoEBackend.FUSED_MOE:
                subc_quant_w1_sz = None
                subc_quant_w2_sz = None
                if weights.w13_weight_scale is not None and weights.w2_weight_scale is not None:
                    padded_hidden_size = weights.w13_weight.shape[-2]
                    # NB: w13_weight_scale: (num_experts, 2, hidden_size // subc_quant_w1_sz, 1, intermediate_size)
                    assert padded_hidden_size % weights.w13_weight_scale.shape[
                        2] == 0
                    subc_quant_w1_sz = padded_hidden_size // weights.w13_weight_scale.shape[
                        2]
                    intermediate_size = weights.w13_weight.shape[-1]
                    # NB: w2_weight_scale: (num_experts, intermediate_size // subc_quant_w2_sz, 1, hidden_size)
                    assert intermediate_size % weights.w2_weight_scale.shape[
                        1] == 0
                    subc_quant_w2_sz = intermediate_size // weights.w2_weight_scale.shape[
                        1]

                actual_hidden_size = x.shape[-1]
                padding_size = weights.w13_weight.shape[-2] - actual_hidden_size
                x = jnp.pad(x, ((0, 0), (0, padding_size)))
                output = fused_ep_moe(
                    mesh=mesh,
                    tokens=x,
                    w1=weights.w13_weight,
                    w2=weights.w2_weight,
                    gating_output=gating_output,
                    top_k=layer.top_k,
                    renormalize_topk_logits=layer.renormalize,
                    act_fn=layer.activation,
                    subc_quant_w1_sz=subc_quant_w1_sz,
                    subc_quant_w2_sz=subc_quant_w2_sz,
                    w1_scale=weights.w13_weight_scale,
                    w2_scale=weights.w2_weight_scale,
                    b1=weights.w13_bias,
                    b2=weights.w2_bias,
                    **extra_backend_kwargs,
                )[:, :actual_hidden_size]
            case MoEBackend.GMM_EP | MoEBackend.GMM_TP:
                output = fused_moe_func(
                    hidden_states=x,
                    w1=weights.w13_weight,
                    w2=weights.w2_weight,
                    w1_scale=weights.w13_weight_scale,
                    w2_scale=weights.w2_weight_scale,
                    w1_bias=weights.w13_bias,
                    w2_bias=weights.w2_bias,
                    gating_output=gating_output,
                    topk=layer.top_k,
                    renormalize=layer.renormalize,
                    mesh=mesh,
                    use_ep=layer.use_ep,
                    activation=layer.activation,
                )
            case MoEBackend.DENSE_MAT:
                # TODO(jacobplatin): will implement in forthcoming PR
                raise ValueError

            case MoEBackend.RAGGED_DOT | MoEBackend.MEGABLX_GMM:
                # TODO(jacobplatin): will implement in forthcoming PR
                raise NotImplementedError

        return output
