from dataclasses import InitVar, dataclass

import jax
import jax.numpy as jnp
from flax import nnx
from flax.typing import Sharding
from jax.sharding import Mesh
from jaxtyping import Float

from tpu_inference.kernels.fused_moe.v1.kernel import fused_ep_moe
from tpu_inference.layers.jax.base import create_param
from tpu_inference.layers.jax.layers import FlaxUtils
from tpu_inference.layers.jax.moe.moe import Router

modeling_flax_utils = FlaxUtils()


@dataclass(kw_only=True)
class GptOssRouter(Router):
    """Router module for Mixture-of-Experts (MoE) layers.

    This module determines which experts each token should be routed.

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

        return router_logits_TE


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

    mesh: Mesh

    def __call__(self, x_TD: Float) -> Float:
        """Performs the forward pass for the GPT-OSS MoE layer."""
        x_TD = jnp.asarray(x_TD, self.dtype)
        x_TD = nnx.with_sharding_constraint(x_TD, self.activation_ffw_td)

        router_logits_TE = self.router(x_TD)

        block_size = {
            "bt": 32,
            "bf": 512,
            "bd1": 512,
            "bd2": 512,
            "btc": 32,
            "bfc": 256,
            "bd1c": 256,
            "bd2c": 256,
        }
        ep_axis_name = self.efd_sharding[0]
        # TODO: Currently, we must reshape the tensors to fit the MoE kernel's
        # required shape. We will eliminate this step and load the tensors in
        # their desired final shape once the weight loading process(with fp4 
        # support) is finalized.
        mlp1_weight_E2DF = jnp.swapaxes(
            jnp.reshape(self.mlp1_weight_EDF2.value,
                        (self.num_local_experts, self.hidden_size, 2,
                         self.intermediate_size_moe)), 1, 2)
        output_TD = fused_ep_moe(
            mesh=self.mesh,
            tokens=x_TD,
            w1=mlp1_weight_E2DF,
            w2=self.mlp2_weight_EFD.value,
            gating_output=router_logits_TE,
            top_k=self.router.num_experts_per_tok,
            ep_axis_name=ep_axis_name,
            **block_size,
        )

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
