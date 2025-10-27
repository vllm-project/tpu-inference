from dataclasses import InitVar, dataclass

import jax
import jax.numpy as jnp
from flax import nnx
from flax.typing import Sharding
from jaxtyping import Float

from tpu_inference.layers.jax.base import create_param
from tpu_inference.layers.jax.layers import FlaxUtils
from tpu_inference.layers.jax.moe.moe import Router

modeling_flax_utils = FlaxUtils()


@dataclass(kw_only=True)
class GptOssRouter(Router):
    """Router module for Mixture-of-Experts (MoE) layers.

    This module determines which experts each token should be routed to based on the input.

    """
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
        x_TD = nnx.with_sharding_constraint(x_TD, self.activation_ffw_td)

        router_logits_TE = jnp.einsum('TD,DE -> TE', x_TD,
                                      self.kernel_DE.value)

        router_logits_TE += self.bias_E.value

        weights_TX, selected_experts_TX = jax.lax.top_k(
            router_logits_TE, self.num_experts_per_tok)

        normalized_weights_TX = jax.nn.softmax(weights_TX.astype(self.dtype),
                                               axis=-1)

        return normalized_weights_TX, selected_experts_TX


def _swiglu(x: Float, alpha: Float, limit: Float) -> Float:
    """Implements the specific SwiGLU from the golden implementation."""
    x_glu, x_linear = x[..., ::2], x[..., 1::2]

    x_glu = jnp.clip(x_glu, a_max=limit)
    x_linear = jnp.clip(x_linear, a_min=-limit, a_max=limit)

    gated_activation = x_glu * jax.nn.sigmoid(alpha * x_glu)

    return gated_activation * (x_linear + 1)


@dataclass(kw_only=True)
class GptOssMoE(nnx.Module):
    """
    JAX implementation of the GPT-OSS Mixture-of-Experts MLP block.
    """
    dtype: jnp.dtype
    hidden_size: int
    intermediate_size_moe: int
    num_local_experts: int
    router: GptOssRouter
    rngs: InitVar[nnx.Rngs]

    swiglu_limit: float = 7.0
    swiglu_alpha: float = 1.702

    # Sharding specifications
    activation_ffw_td: Sharding
    edf_sharding: Sharding
    efd_sharding: Sharding
    ed_sharding: Sharding

    random_init: bool = False

    def __call__(self, x_TD: Float) -> Float:
        """Performs the forward pass for the GPT-OSS MoE layer."""
        x_TD = jnp.asarray(x_TD, self.dtype)
        x_TD = nnx.with_sharding_constraint(x_TD, self.activation_ffw_td)

        weights_TX, indices_TX = self.router(x_TD)

        one_hot_mask_TXE = jax.nn.one_hot(indices_TX,
                                          num_classes=self.num_local_experts,
                                          dtype=self.dtype)
        combined_weights_TE = jnp.sum(one_hot_mask_TXE * weights_TX[..., None],
                                      axis=1)

        # First MLP layer (up-projection)
        with jax.named_scope("MLP #1"):
            up_proj_TEF2 = jnp.einsum('TD,EDF -> TEF', x_TD,
                                      self.mlp1_weight_EDF2.value)
            up_proj_TEF2 += self.mlp1_bias_EF2.value

            fuse_TEF = _swiglu(up_proj_TEF2,
                               alpha=self.swiglu_alpha,
                               limit=self.swiglu_limit)

        # Second MLP layer (down-projection)
        with jax.named_scope("MLP #2"):
            down_proj_TED = jnp.einsum('TEF,EFD -> TED', fuse_TEF,
                                       self.mlp2_weight_EFD.value)
            down_proj_TED += self.mlp2_bias_ED.value

        # Weighted sum of expert outputs
        with jax.named_scope("sum"):
            output_TD = jnp.einsum('TED,TE -> TD', down_proj_TED,
                                   combined_weights_TE)

        return output_TD.astype(self.dtype)

    def __post_init__(self, rngs: nnx.Rngs):
        """Initializes all weights and biases for the MoE block."""
        D, F, E = self.hidden_size, self.intermediate_size_moe, self.num_local_experts

        # MLP #1 Weights (Combined Gate and Up-projection) and Bias
        self.mlp1_weight_EDF2 = create_param(
            rngs,
            shape=(E, D, F * 2),
            dtype=self.dtype,
            sharding=self.edf_sharding,
            random_init=self.random_init,
        )
        self.mlp1_bias_EF2 = create_param(
            rngs,
            shape=(E, F * 2),
            dtype=self.dtype,
            sharding=self.ed_sharding,
            random_init=self.random_init,
        )

        # MLP #2 Weights (Down-projection) and Bias
        self.mlp2_weight_EFD = create_param(
            rngs,
            shape=(E, F, D),
            dtype=self.dtype,
            sharding=self.efd_sharding,
            random_init=self.random_init,
        )
        self.mlp2_bias_ED = create_param(
            rngs,
            shape=(E, D),
            dtype=self.dtype,
            sharding=self.ed_sharding,
            random_init=self.random_init,
        )
