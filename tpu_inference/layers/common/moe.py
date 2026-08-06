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
from typing import TYPE_CHECKING, Tuple, Union

import jax
import jax.numpy as jnp
from jax.sharding import Mesh
from vllm.model_executor.layers.fused_moe import RoutedExperts

from tpu_inference import envs
from tpu_inference.kernels.fused_moe.v1.kernel import fused_ep_moe
from tpu_inference.layers.common.fused_moe_gmm import fused_moe_func
from tpu_inference.logger import init_logger
from tpu_inference.utils import to_jax_dtype

if TYPE_CHECKING:
    from tpu_inference.layers.common.process_weights.moe_weights import (
        FusedMoEWeights, UnfusedMoEWeights)
    from tpu_inference.layers.jax.moe.moe import JaxMoE, JaxRoutedExperts
else:
    FusedMoEWeights = None
    UnfusedMoEWeights = None
    JaxMoE = None
    JaxRoutedExperts = None

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


# Which setting selects which backend, as the two selectors implement it:
# USE_MOE_EP_KERNEL returns FUSED_MOE, USE_UNFUSED_MEGABLOCKS returns
# MEGABLX_GMM, USE_DENSE_MOE returns DENSE_MAT. GMM_TP is on this list under
# no name at all, because nothing selects it -- it is what a deployment gets
# when expert parallelism is off, which is a different conversation and the
# one the warning below has.
_BACKEND_SELECTED_BY = (
    ("USE_MOE_EP_KERNEL", MoEBackend.FUSED_MOE),
    ("USE_UNFUSED_MEGABLOCKS", MoEBackend.MEGABLX_GMM),
    ("USE_DENSE_MOE", MoEBackend.DENSE_MAT),
)


def announce_moe_backend(moe_backend: MoEBackend) -> None:
    """Say which MoE program this deployment runs, and refuse a contradiction.

    Called from the backend selectors, which run once per layer while the
    model is being built rather than inside a traced function, so the line
    reaches the log at the level a server is actually run at and before any
    batch shape exists. It is emitted whether USE_MOE_FUSED_EP_KERNEL is on
    or off, so a log that carries no such line means this build predates the
    switch rather than that the switch is off, and it names both settings by
    the spelling that was read, so a configuration written against a retired
    spelling shows up as the default rather than as the value it set.
    """
    # This is the build-time surface that says which program ran, and it is
    # called before anything else in the feature does, so a caller handing it
    # something that is not a backend should be told that rather than get an
    # attribute error out of a logging line further down.
    if not isinstance(moe_backend, MoEBackend):
        raise ValueError(
            f"announce_moe_backend was handed {moe_backend!r}, which is not "
            f"a MoEBackend; the backend selectors return one of "
            f"{tuple(b.value for b in MoEBackend)}")
    switch_on = envs.USE_MOE_FUSED_EP_KERNEL
    if switch_on and moe_backend is not MoEBackend.GMM_EP:
        # A setting conflicts when it is set AND the backend that came back
        # is the one that setting selects. Testing the names for presence
        # alone blamed whichever happened to be exported: with expert
        # parallelism off the selector logs that USE_MOE_EP_KERNEL was
        # IGNORED and returns GMM_TP, and the next line then told the
        # operator that same flag had "selected the gmm_tp backend". Unsetting
        # it, as instructed, still yields GMM_TP -- and the real remedy is in
        # the warning below, which the raise preempted.
        conflicting = [
            name for name, selected in _BACKEND_SELECTED_BY
            if selected is moe_backend and getattr(envs, name)
        ]
        if conflicting:
            raise ValueError(
                f"USE_MOE_FUSED_EP_KERNEL asks for the fused expert-parallel "
                f"MoE kernel, but {' and '.join(conflicting)} selected the "
                f"{moe_backend.value} backend instead, which the switch "
                f"cannot reach. Unset one of them.")
        # The most likely first attempt: the switch on, expert parallelism
        # off. Nothing else in the path would say a word about it.
        logger.warning_once(
            "USE_MOE_FUSED_EP_KERNEL is on, but this deployment selected the "
            "%s backend. The fused expert-parallel MoE kernel only serves "
            "expert-parallel MoE calls, so the switch does nothing here: "
            "enable expert parallelism to reach it. Every MoE call runs on "
            "fused_moe_func.", moe_backend.value)
    if switch_on:
        logger.info_once(
            "[MoE]: %s backend; USE_MOE_FUSED_EP_KERNEL=1, "
            "MOE_FUSED_EP_KERNEL_MIN_TOKENS=%d.", moe_backend.value,
            envs.MOE_FUSED_EP_KERNEL_MIN_TOKENS)
    else:
        logger.info_once(
            "[MoE]: %s backend; USE_MOE_FUSED_EP_KERNEL is off, so "
            "MOE_FUSED_EP_KERNEL_MIN_TOKENS is not read and every MoE call "
            "runs on the general MoE path.", moe_backend.value)


def moe_apply(
    layer: Union[RoutedExperts, JaxRoutedExperts, JaxMoE],
    x: jax.Array,
    gating_output: Union[jax.Array, Tuple[jax.Array, jax.Array]],
    weights: Union[FusedMoEWeights, UnfusedMoEWeights],
    moe_backend: MoEBackend,
    mesh: Mesh,
    extra_backend_kwargs: dict,
) -> jax.Array:
    extra_backend_kwargs = dict(
        extra_backend_kwargs) if extra_backend_kwargs else {}
    scatter_results = extra_backend_kwargs.pop("scatter_results", False)
    moe_chunk_size = extra_backend_kwargs.pop("moe_chunk_size", 0)
    # When set, the GMM backends skip their tensor-/expert-parallel all-reduce
    # and return per-shard partial sums, so the reduction can be deferred and
    # fused with a later collective (e.g. a shared-expert all-reduce).
    defer_all_reduce = extra_backend_kwargs.pop("defer_all_reduce", False)

    if defer_all_reduce and moe_backend not in {
            MoEBackend.GMM_EP, MoEBackend.GMM_TP
    }:
        raise ValueError(
            "defer_all_reduce can only be True for GMM_EP and GMM_TP backends")

    with jax.named_scope(layer._get_name()):
        activation = layer.activation if isinstance(
            layer.activation, str) else layer.activation.value
        if activation == "silu":
            swiglu_limit = getattr(layer, "swiglu_limit", None)
            if swiglu_limit is not None and swiglu_limit > 0:
                activation = "silu_and_mul_with_clamp"
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
                # Check if activation_dtype was passed via kwargs or as an environment variable
                activation_dtype = extra_backend_kwargs.get(
                    "activation_dtype", envs.MOE_ALL_GATHER_ACTIVATION_DTYPE)
                all_gather_fp8 = (bool(activation_dtype)
                                  and to_jax_dtype(activation_dtype)
                                  == jnp.float8_e4m3fn)

                # Opt-in: with USE_MOE_FUSED_EP_KERNEL unset this branch is
                # not taken, nothing below is imported, and the call runs the
                # general path unchanged. The adapter returns None where the
                # kernel does not serve this batch, which falls through to
                # that same path.
                if (moe_backend == MoEBackend.GMM_EP
                        and envs.USE_MOE_FUSED_EP_KERNEL):
                    from tpu_inference.layers.common.moe_fused_ep import \
                        moe_fused_ep_route
                    output = moe_fused_ep_route(
                        layer, x, gating_output, weights, mesh, activation,
                        scatter_results, extra_backend_kwargs,
                        defer_all_reduce, moe_chunk_size, activation_dtype)
                    if output is not None:
                        return output

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
                    all_gather_fp8=all_gather_fp8,
                    enable_rs_kernel=envs.ENABLE_RS_KERNEL,
                    onehot_moe_permute_threshold=envs.
                    ONEHOT_MOE_PERMUTE_THRESHOLD,
                    scatter_results=scatter_results,
                    defer_all_reduce=defer_all_reduce,
                    hash_based_topk_indices=extra_backend_kwargs.get(
                        "hash_based_topk_indices", None),
                    expert_score_correction_bias=extra_backend_kwargs.get(
                        "e_score_correction_bias", None),
                    moe_chunk_size=moe_chunk_size,
                    num_valid_tokens=extra_backend_kwargs.get(
                        "num_valid_tokens", None),
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
                return dense_moe_func(
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

            case MoEBackend.MEGABLX_GMM:
                # NOTE: circular import avoidance
                from tpu_inference.layers.jax.moe.sparse_moe import \
                    sparse_moe_func

                return sparse_moe_func(weights=weights,
                                       x_TD=x,
                                       gating_output=gating_output,
                                       layer=layer,
                                       mesh=mesh)

        return output


# TODO(#3041): Inherit from vLLM's FusedMoEMethodBase, so it can take FusedMoeConfig
# as init arg, and unify more logic.
class FusedMoEMethodBase:
    """Base class that prepare TPU specific configs"""

    def __init__(self, moe_backend: MoEBackend, ep_axis_name: str):
        self.extra_backend_kwargs: dict = {
            "moe_chunk_size": envs.VLLM_MOE_CHUNK_SIZE
        }
        if moe_backend == MoEBackend.FUSED_MOE:
            self.extra_backend_kwargs["ep_axis_name"] = ep_axis_name
