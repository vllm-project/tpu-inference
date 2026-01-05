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

import jax
import jax.numpy as jnp
import torch
from jax.sharding import Mesh
from torchax.interop import jax_view, torch_view
from vllm.model_executor.layers.fused_moe.config import FusedMoEConfig
from vllm.model_executor.layers.fused_moe.layer import FusedMoE

from tpu_inference import envs
from tpu_inference.kernels.fused_moe.v1.kernel import fused_ep_moe
from tpu_inference.layers.common.fused_moe_gmm import fused_moe_func
from tpu_inference.logger import init_logger

logger = init_logger(__name__)


class FusedMoEBackend(Enum):
    FUSED_MOE = "fused_moe"
    GMM_EP = "gmm_ep"
    GMM_TP = "gmm_tp"


def select_moe_backend(moe: FusedMoEConfig):
    if envs.USE_MOE_EP_KERNEL:
        if moe.use_ep:
            return FusedMoEBackend.FUSED_MOE
        logger.warning_once(
            "USE_MOE_EP_KERNEL=1 but expert parallelism is not "
            "enabled. Falling back to gmm implementation.")

    if moe.use_ep:
        return FusedMoEBackend.GMM_EP

    # Use default implementation.
    return FusedMoEBackend.GMM_TP


def fused_moe_apply(
    layer: torch.nn.Module,
    x: torch.Tensor,
    router_logits: torch.Tensor,
    moe_backend: FusedMoEBackend,
    mesh: Mesh,
    extra_backend_kwargs: dict,
) -> torch.Tensor:
    assert isinstance(layer, FusedMoE)
    if layer.scoring_func != "softmax":
        raise NotImplementedError("Only softmax is supported for scoring_func")

    x = jax_view(x)
    gating_output = jax_view(router_logits)

    w13_weight = jax_view(layer.w13_weight)
    w13_weight_scale = jax_view(getattr(layer, "w13_weight_scale", None))
    w13_bias = jax_view(getattr(layer, "w13_bias", None))
    w2_weight = jax_view(layer.w2_weight)
    w2_weight_scale = jax_view(getattr(layer, "w2_weight_scale", None))
    w2_bias = jax_view(getattr(layer, "w2_bias", None))

    with jax.named_scope(layer._get_name()):
        match moe_backend:
            case FusedMoEBackend.FUSED_MOE:
                actual_hidden_size = x.shape[-1]
                padding_size = w13_weight.shape[-2] - actual_hidden_size
                x = jnp.pad(x, ((0, 0), (0, padding_size)))
                output = fused_ep_moe(
                    mesh=mesh,
                    tokens=x,
                    w1=w13_weight,
                    w2=w2_weight,
                    w1_scale=w13_weight_scale,
                    w2_scale=w2_weight_scale,
                    b1=w13_bias,
                    b2=w2_bias,
                    gating_output=gating_output,
                    top_k=layer.top_k,
                    renormalize_topk_logits=layer.renormalize,
                    act_fn=layer.activation,
                    **extra_backend_kwargs,
                )[:, :actual_hidden_size]
            case FusedMoEBackend.GMM_EP | FusedMoEBackend.GMM_TP:
                output = fused_moe_func(
                    hidden_states=x,
                    w1=w13_weight,
                    w2=w2_weight,
                    w1_scale=w13_weight_scale,
                    w2_scale=w2_weight_scale,
                    w1_bias=w13_bias,
                    w2_bias=w2_bias,
                    gating_output=gating_output,
                    topk=layer.top_k,
                    renormalize=layer.renormalize,
                    mesh=mesh,
                    use_ep=layer.use_ep,
                    activation=layer.activation,
                )

        return torch_view(output)
