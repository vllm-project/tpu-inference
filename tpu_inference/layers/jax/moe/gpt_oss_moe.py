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

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from flax import nnx
from flax.typing import Sharding
from jax import lax
from jaxtyping import Float

from tpu_inference.layers.common.moe import MoEBackend
from tpu_inference.layers.jax.base import create_param
from tpu_inference.layers.jax.moe.moe import JaxRoutedExperts, Router


@dataclass(kw_only=True)
class GptOssRouter(Router):
    """Router for GPT-OSS MoE that adds a per-expert bias to the logits."""
    e_sharding: Sharding = ()

    def __post_init__(self, rngs: nnx.Rngs):
        """
        Initializes the parent's kernel and adds the new bias parameter.
        """
        super().__post_init__(rngs)

        self.bias_E = create_param(rngs,
                                   shape=(self.num_experts, ),
                                   dtype=self.dtype,
                                   sharding=self.e_sharding,
                                   random_init=self.random_init)

    def __call__(self, x_TD: Float):
        """
        Overrides the parent's forward pass to include the bias.
        """
        x_TD = jnp.asarray(x_TD, self.dtype)
        x_TD = lax.with_sharding_constraint(x_TD, self.activation_ffw_td)

        router_logits_TE = jnp.einsum('TD,DE -> TE', x_TD,
                                      self.kernel_DE.value)

        router_logits_TE += self.bias_E.value

        if self.moe_backend in MoEBackend.fused_moe_backends():
            return router_logits_TE

        weights_TX, selected_experts_TX = jax.lax.top_k(
            router_logits_TE, self.num_experts_per_tok)

        normalized_weights_TX = jax.nn.softmax(weights_TX.astype(self.dtype),
                                               axis=-1)

        return normalized_weights_TX, selected_experts_TX


class GptOssMoE(JaxRoutedExperts):
    """Mixture of Experts for GPT-OSS using FusedMoE.

    The router projection is external (GptOssRouter); this class only
    handles expert dispatch. use_ep and moe_backend are derived from
    the vLLM parallel config by JaxRoutedExperts, so the EP/TP backend
    matches the torchax path automatically.
    """

    def __init__(
        self,
        *,
        dtype: jnp.dtype,
        mesh,
        rngs: nnx.Rngs,
        quant_config=None,
        num_local_experts: int,
        hidden_size: int,
        intermediate_size_moe: int,
        num_experts_per_tok: int,
        random_init: bool = False,
        enable_return_routed_experts: bool = False,
        prefix: str = "",
    ) -> None:
        JaxRoutedExperts.__init__(
            self,
            dtype=dtype,
            num_local_experts=num_local_experts,
            hidden_size=hidden_size,
            intermediate_size_moe=intermediate_size_moe,
            hidden_act="swigluoai",
            rngs=rngs,
            mesh=mesh,
            top_k=num_experts_per_tok,
            scoring_func="softmax",
            renormalize=True,
            random_init=random_init,
            enable_return_routed_experts=enable_return_routed_experts,
            quant_config=quant_config,
            prefix=prefix,
        )
