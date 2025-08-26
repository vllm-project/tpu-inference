from typing import Any

import jax
from flax import nnx


class JaxPluginModel(nnx.Module):
    """The interface required for all JAX plugin models in vLLM."""

    def forward(
        self,
        input_ids: Any,
        positions: Any,
    ) -> Any:
        raise NotImplementedError(
            "Subclasses of JaxPluginModel should implement `__call__` for the "
            "forward pass, not `forward`.")

    def compute_logits(self, hidden_states: jax.Array, **kwargs) -> jax.Array:
        raise NotImplementedError
