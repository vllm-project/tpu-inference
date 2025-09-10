from dataclasses import dataclass
from typing import Any, Tuple

# Flax and JAX sharding imports
import jax
from flax import nnx

from tpu_commons.models.jax.common.attention.attention import (
    AttentionMetadata, KVCache)
from tpu_commons.models.jax.common.layers import DenseFFW
from tpu_commons.models.jax.common.moe.moe import MoE


@dataclass(kw_only=True)
class TransformerBlock(nnx.Module):
    """
    A heavy weight module which serves as the stateful live blocks in serving

    custom_module can be either a dense module (i.e., DenseFFW) or MoE.
    """
    pre_attention_norm: nnx.Module
    pre_mlp_norm: nnx.Module
    custom_module: nnx.Module
    attn: nnx.Module
    use_attention_rope: bool = True
    quant: Any | None = None

    def __call__(
            self, x_TD: jax.Array, is_prefill: bool, kv_cache: KVCache,
            attention_metadata: AttentionMetadata
    ) -> Tuple[KVCache, jax.Array]:
        # # Attn Block
        # jax.debug.print("Layer input hidden states slice:\n{}", x_TD)

        attn_residual_TD = x_TD
        x_TD = self.pre_attention_norm(x_TD)
        # jax.debug.print("Pre-attention norm output slice:\n{}", x_TD)
        new_cache, attn_output_TD = self.attn(x_TD, is_prefill, kv_cache,
                                              attention_metadata,
                                              self.use_attention_rope)
        # jax.debug.print("Attention output (before residual): {}",
        #                 attn_output_TD)
        attn_output_TD += attn_residual_TD
        # jax.debug.print("Attention output (after residual): {}",
        #                 attn_output_TD)

        # FFW Block
        ffw_residual_TD = attn_output_TD
        normed_ffw_input_TD = self.pre_mlp_norm(attn_output_TD)
        # jax.debug.print("FFW input (after pre_mlp_norm): {}",
        #                 normed_ffw_input_TD)
        logits_TD = self.custom_module(normed_ffw_input_TD)
        # jax.debug.print("FFW output (before residual): {}", logits_TD)
        logits_TD += ffw_residual_TD
        # jax.debug.print("FFW output (after residual): {}", logits_TD)
        return new_cache, logits_TD


@dataclass(kw_only=True)
class SharedExpertsTransformerBlock(TransformerBlock):
    """Create a modified TransformerBlock that sums MoE layer output with shared expert output."""
    shared_experts: nnx.Module

    def __call__(self, x_TD, is_prefill, kv_cache, attention_metadata):
        # Attn Block
        attn_residual_TD = x_TD
        x_TD = self.pre_attention_norm(x_TD)
        new_cache, attn_output_TD = self.attn(x_TD, is_prefill, kv_cache,
                                              attention_metadata,
                                              self.use_attention_rope)
        attn_output_TD += attn_residual_TD
        # FFW Block
        ffw_residual_TD = attn_output_TD
        normed_ffw_input_TD = self.pre_mlp_norm(attn_output_TD)
        if isinstance(self.custom_module, MoE):
            logits_TD = self.custom_module(normed_ffw_input_TD)
            # Add the shared expert outputs to the MoE outputs.
            shared_expert_output_TD = self.shared_experts(normed_ffw_input_TD)
            logits_TD += shared_expert_output_TD
        elif isinstance(self.custom_module, DenseFFW):
            logits_TD = self.custom_module(normed_ffw_input_TD)
        else:
            raise ValueError(
                f"Invalid custom moduel type: {type(self.custom_module)}")
        logits_TD += ffw_residual_TD
        return new_cache, logits_TD
