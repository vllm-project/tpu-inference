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
from typing import TYPE_CHECKING, Optional, Tuple, Union

import jax
import jax.numpy as jnp
from jax.sharding import Mesh
from vllm.model_executor.layers.fused_moe import FusedMoE

from tpu_inference.kernels.fused_moe.v1.kernel import fused_ep_moe
from tpu_inference.layers.common.expert_selection import (
    LayerExpertSelection, is_expert_selection_enabled)
from tpu_inference.layers.common.fused_moe_gmm import (apply_scoring_fn,
                                                        fused_moe_func)
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
    # FUSED_MOE is using the Fused MoE kernel found in tpu-inference/tpu_inference/kernels/fused_moe/v1/kernel.py
    # and is production ready.
    # NOTE: for FusedMOE on the JAX/Flax path, we expect the MoE quant method (e.g. UnquantizedFusedMoEMethod) to
    # create a new `kernel_gating_upproj_E2DF` param that replaces `kernel_gating_EDF` and `kernel_up_proj_EDF`
    # and thus runs the forward pass on the fused weight.  `kernel_down_proj_EFD` is unchanged.
    FUSED_MOE = "fused_moe"

    # GMM_EP is using the GMM kernel found in tpu-inference/tpu_inference/layers/common/fused_moe_gmm.py
    # and is `expert_sharded_gmm` the GMM calls.  Production ready.
    # NOTE: for GMM_EP JAX/Flax path, we expect the MoE quant method (e.g. UnquantizedFusedMoEMethod) to
    # create a new `kernel_gating_upproj_EDF` param that replaces `kernel_gating_EDF` and `kernel_up_proj_EDF`
    # and thus runs the forward pass on the fused weight.  `kernel_down_proj_EFD` is unchanged.
    GMM_EP = "gmm_ep"

    # Same as GMM_EP but uses `tensor_sharded_gmm_*` for the GMM calls and is production ready
    GMM_TP = "gmm_tp"

    # DENSE_MAT uses a simple dense matmul for the MoE backend,  is intended for testing, and is
    # only used in the JAX path for now
    # NOTE: for DENSE_MAT in the JAX/Flax path, there are no changes for weights for the unfused backends (DENSE_MAT and MEGABLOX_GMM).
    # That is, `kernel_gating_EDF`, `kernel_up_proj_EDF`, and `kernel_down_proj_EFD` are unchanged.
    DENSE_MAT = "dense_mat"
    # Also only used in the JAX path for now
    MEGABLX_GMM = "megablox_gmm"

    @classmethod
    def fused_moe_backends(cls):
        """Returns those backends that use fused weights"""
        return {cls.FUSED_MOE, cls.GMM_EP, cls.GMM_TP}


def _extract_expert_selection(
    gating_output: Union[jax.Array, Tuple[jax.Array, jax.Array]],
    top_k: int,
    scoring_func: str,
    renormalize: bool,
) -> LayerExpertSelection:
    """Extract expert selection (topk weights and ids) from router output.

    For fused backends, gating_output is raw logits (T, E) - we apply scoring
    and topk. For unfused backends, it's already (weights_TX, indices_TX).

    Args:
        gating_output: Raw router logits (T, E) or tuple of (weights, indices).
        top_k: Number of experts per token.
        scoring_func: Scoring function name (e.g., "softmax", "sigmoid").
        renormalize: Whether to renormalize topk weights.

    Returns:
        LayerExpertSelection with topk_weights and topk_ids.
    """
    if isinstance(gating_output, tuple):
        # Unfused backends: already have (weights, indices)
        weights_TX, indices_TX = gating_output
        return LayerExpertSelection(topk_weights=weights_TX,
                                    topk_ids=indices_TX)
    else:
        # Fused backends: need to compute topk from raw logits
        scores = apply_scoring_fn(scoring_func, gating_output)
        topk_weights, topk_ids = jax.lax.top_k(scores, k=top_k)
        if renormalize:
            topk_weights = topk_weights / jnp.sum(
                topk_weights, axis=-1, keepdims=True)
        return LayerExpertSelection(topk_weights=topk_weights,
                                    topk_ids=topk_ids)


def moe_apply(
    layer: Union[FusedMoE, JaxMoE],
    x: jax.Array,
    gating_output: Union[jax.Array, Tuple[jax.Array, jax.Array]],
    weights: Union[FusedMoEWeights, UnfusedMoEWeights],
    moe_backend: MoEBackend,
    mesh: Mesh,
    extra_backend_kwargs: dict,
    return_expert_selection: bool = False,
) -> Union[jax.Array, Tuple[jax.Array, LayerExpertSelection]]:

    with jax.named_scope(layer._get_name()):
        activation = layer.activation if isinstance(
            layer.activation, str) else layer.activation.value

        # Extract expert selection before the MoE computation if requested
        expert_selection = None
        if return_expert_selection:
            expert_selection = _extract_expert_selection(
                gating_output, layer.top_k, layer.scoring_func,
                layer.renormalize)

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
                    act_fn=activation,
                    scoring_fn=layer.scoring_func,
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
                    activation=activation,
                    scoring_fn=layer.scoring_func,
                )
            case MoEBackend.DENSE_MAT:
                # NOTE: circular import avoidance
                from tpu_inference.layers.jax.moe.dense_moe import \
                    dense_moe_func
                assert isinstance(
                    gating_output,
                    tuple), "Expected the gating output to be a tuple"
                assert len(
                    gating_output
                ) == 2, "Expected the gating output to be have 2 entries: weights and indices"
                output = dense_moe_func(
                    weights=weights,
                    x_TD=x,
                    gating_output=gating_output,
                    cast_dtype=layer.dtype,
                    num_local_experts=layer.num_local_experts,
                    apply_expert_weight_before_computation=layer.
                    apply_expert_weight_before_computation,
                    activation_ffw_ted=layer.activation_ffw_ted,
                    activation_ffw_td=layer.activation_ffw_td,
                    hidden_act=layer.hidden_act,
                    mesh=mesh)
                if return_expert_selection:
                    return output, expert_selection
                return output

            case MoEBackend.MEGABLX_GMM:
                # NOTE: circular import avoidance
                from tpu_inference.layers.jax.moe.sparse_moe import \
                    sparse_moe_func

                output = sparse_moe_func(weights=weights,
                                       x_TD=x,
                                       gating_output=gating_output,
                                       layer=layer,
                                       mesh=mesh)
                if return_expert_selection:
                    return output, expert_selection
                return output

        if return_expert_selection:
            return output, expert_selection
        return output
