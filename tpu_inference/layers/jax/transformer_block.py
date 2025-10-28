from dataclasses import dataclass
from typing import Any, Optional, Tuple, Union

# Flax and JAX sharding imports
import jax
from flax import nnx

from tpu_inference.layers.jax.attention.attention import (AttentionMetadata,
                                                          KVCache)
from tpu_inference.layers.jax.layers import DenseFFW
from tpu_inference.layers.jax.moe.moe import MoE


@dataclass(kw_only=True)
class TransformerBlock(nnx.Module):
    """
    A heavy weight module which serves as the stateful live blocks in serving

    custom_module can be either a dense module (i.e., DenseFFW) or MoE.
    """
    pre_attention_norm: nnx.Module
    pre_mlp_norm: nnx.Module
    custom_module: Optional[Union[DenseFFW, MoE]] = None
    attn: nnx.Module
    use_attention_rope: bool = True
    quant: Any | None = None

    def __call__(
            self, x_TD: jax.Array, is_prefill: bool, kv_cache: KVCache,
            attention_metadata: AttentionMetadata
    ) -> Tuple[KVCache, jax.Array]:
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
        logits_TD = self.custom_module(normed_ffw_input_TD)
        logits_TD += ffw_residual_TD
        return new_cache, logits_TD


@dataclass(kw_only=True)
class SharedExpertsTransformerBlock(TransformerBlock):
    """Create a modified TransformerBlock that sums MoE layer output with shared expert output."""
    moe_ffw: Optional[MoE] = None
    dense_ffw: Optional[DenseFFW] = None
    shared_experts: Optional[DenseFFW] = None

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

        if self.custom_module:
            if isinstance(self.custom_module, MoE):
                logits_TD = self.custom_module(normed_ffw_input_TD)
                # Add the shared expert outputs to the MoE outputs.
                shared_expert_output_TD = self.shared_experts(
                    normed_ffw_input_TD)
                logits_TD += shared_expert_output_TD
            elif isinstance(self.custom_module, DenseFFW):
                logits_TD = self.custom_module(normed_ffw_input_TD)
            else:
                raise ValueError(
                    f"Invalid custom moduel type: {type(self.custom_module)}")
        else:
            if self.moe_ffw is not None:
                if self.shared_experts is None:
                    raise ValueError(
                        "MoE layer (moe_ffw is set) but shared_experts is None!"
                    )
                moe_output_TD = self.moe_ffw(normed_ffw_input_TD)
                shared_expert_output_TD = self.shared_experts(
                    normed_ffw_input_TD)
                logits_TD = moe_output_TD + shared_expert_output_TD

            elif self.dense_ffw is not None:
                if self.shared_experts is not None:
                    raise ValueError(
                        "Dense FFW layer (dense_ffw is set) but shared_experts is not None!"
                    )
                logits_TD = self.dense_ffw(normed_ffw_input_TD)

            else:
                raise ValueError(
                    "Neither moe_ffw nor dense_ffw attribute is set for this SharedExpertsTransformerBlock!"
                )

        logits_TD += ffw_residual_TD
        return new_cache, logits_TD
