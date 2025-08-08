from dataclasses import dataclass, field, make_dataclass
from typing import Any, Tuple

# Flax and JAX sharding imports
import jax
from flax import nnx
from vllm.config import VllmConfig

from tpu_commons.models.jax.common.attention.attention import (
    AttentionConfig, AttentionMetadata, KVCache)
from tpu_commons.models.jax.common.base import Config
from tpu_commons.models.jax.common.constants import HuggingFaceArgNames
from tpu_commons.models.jax.common.layers import DenseFFW, DenseFFWConfig
from tpu_commons.models.jax.common.moe.moe import MoE, MoEConfig

TransformerBlockConfig = make_dataclass(
    "TransformerBlockConfig",
    [("attention", AttentionConfig), ("dense_ffw", DenseFFWConfig),
     (HuggingFaceArgNames.RMS_NORM_EPS.value, float),
     ("moe", MoEConfig, field(default=None)),
     ("vllm_config", VllmConfig, field(repr=False, default=None))],
    bases=(Config, ))
TransformerBlockConfig.__doc__ = f"""light weighted transformer config, which includes config for all sub-modules
    it uses make() to create the live module from this config
    Args:
        attention: AttentionConfig config used to specify attention layer parameters.
        dense_ffw: DenseFFWConfig config used to specify feed-forward layer parameters.
        {HuggingFaceArgNames.RMS_NORM_EPS.value}: float The epsilon value for RMSNorm.
        vllm_config: VllmConfig The VLLM config containing any overrides to apply.
        """


@dataclass
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

    def generate_kernel(self, rngs: nnx.Rngs):
        self.attn.generate_kernel(rngs)
        self.custom_module.generate_kernel(rngs)
        self.pre_attention_norm.generate_kernel(rngs)
        self.pre_mlp_norm.generate_kernel(rngs)


# Provide a variant that allows mixing and matching Dense & MoE layers.
SharedExpertsTransformerBlockConfig = make_dataclass(
    "SharedExpertsTransformerBlockConfig",
    [(HuggingFaceArgNames.SHARED_EXPERTS.value, int)],
    bases=(TransformerBlockConfig, ),
    kw_only=True)

SharedExpertsTransformerBlockConfig.__doc__ = f"""Transformer block with MoE block and shared experts block (i.e. Dense Block).
Additional Args:
  {HuggingFaceArgNames.SHARED_EXPERTS.value}: Number of experts to route all of the inputs to (essentially a dense layer).

Inherits TransformerBlockConfig docstring:
{TransformerBlockConfig.__doc__}
"""


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

    def generate_kernel(self, rngs: nnx.Rngs):
        super().generate_kernel(rngs)
        self.shared_experts.generate_kernel(rngs)
