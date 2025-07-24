from dataclasses import dataclass, make_dataclass

import jax
import jax.numpy as jnp
from flax import nnx

from tpu_commons.logger import init_logger
from tpu_commons.models.jax.attention_metadata import AttentionMetadata
from tpu_commons.models.jax.common.attention.attention import (Attention,
                                                               AttentionConfig,
                                                               KVCache)
from tpu_commons.models.jax.common.constants import HuggingFaceArgNames
from tpu_commons.models.jax.layers.rope import apply_rope

logger = init_logger(__name__)


class L2Norm(nnx.Module):
    """
  Implementation of L2Norm in JAX (taken from MaxText repo - maxtext/MaxText/layers/attentions.py).

  Attributes:
    eps: float, epsilon used for numerical stability (default value should be ok for most cases).
  """

    def __init__(self, eps: float = 1e-6):
        self.eps = eps

    def __call__(self, x):
        return x * jax.lax.rsqrt(
            jnp.mean(x**2, axis=-1, keepdims=True) + self.eps)


Llama4AttentionConfig = make_dataclass(
    "Llama4AttentionConfig",
    [(HuggingFaceArgNames.USE_QK_NORM.value, bool),
     (HuggingFaceArgNames.TEMPERATURE_TUNING.value, bool),
     (HuggingFaceArgNames.TEMPERATURE_TUNING_SCALE.value, float),
     (HuggingFaceArgNames.TEMPERATURE_TUNING_FLOOR_SCALE.value, float)],
    bases=(AttentionConfig, ),
    kw_only=True)
Llama4AttentionConfig.__doc__ = f"""Llama4-specific attention layer which performs layer norm to the Query and Keys after RoPE.
Additional Args:
  {HuggingFaceArgNames.USE_QK_NORM.value}: bool whether to use Llama4 normalization of query & keys
  {HuggingFaceArgNames.TEMPERATURE_TUNING.value}: bool whether to use temperature tuning
  {HuggingFaceArgNames.TEMPERATURE_TUNING_SCALE.value}: float temperature tuning scale
  {HuggingFaceArgNames.TEMPERATURE_TUNING_FLOOR_SCALE.value}: float temperature tuning floor scale

Inherits AttentionConfig docstring:
{AttentionConfig.__doc__}
"""


@dataclass
class Llama4Attention(Attention):

    def __call__(self,
                 x,
                 is_prefill,
                 kv_cache: KVCache,
                 attention_metadata: AttentionMetadata,
                 use_attention_rope: bool = True):
        """Performs the forward pass of the attention module.

        This method computes the attention output by projecting the input `x`
        to queries, keys, and values, applying RoPE, performing scaled
        dot-product attention, and projecting the result back to the model
        dimension. It updates and utilizes a KV cache.

        Args:
            x: The input tensor of shape `(seq_len, d_model)`.
            op_mode: The operational mode, either 'prefill' or 'generate'.
            kv_cache: The key-value cache for storing past attention states.
            attention_metadata: Metadata for attention, such as input positions.
            use_attention_rope: Whether to use RoPE.

        Returns:
            A tuple containing:
                - The updated KV cache.
                - The attention output tensor of shape
                  `(batch_size, seq_len, d_model)`.
        """
        op_mode = "prefill" if is_prefill else "generate"
        md = attention_metadata
        x = jnp.asarray(x, self.cfg.dtype)
        x_SD = nnx.with_sharding_constraint(
            x, self.activation_attention_td[op_mode])
        x_q_TD = nnx.with_sharding_constraint(x, self.activation_q_td[op_mode])
        rope_scaling = getattr(self.cfg,
                               HuggingFaceArgNames.ROPE_SCALING.value)
        rope_theta = getattr(self.cfg, HuggingFaceArgNames.ROPE_THETA.value)
        H = getattr(self.cfg, HuggingFaceArgNames.HEAD_DIM.value)
        l2_norm = L2Norm()
        # logger.warning(f"Using RoPE?? {use_attention_rope}")
        with jax.named_scope("q_proj"):
            q_TNH = jnp.einsum('TD,DNH -> TNH', x_q_TD,
                               self.kernel_q_proj_DNH.value)
            jax.debug.print("q_TNH:\n{val}", val=q_TNH[:3, :5, :5])
            if use_attention_rope:
                q_TNH = apply_rope(q_TNH, md.input_positions, H, rope_theta,
                                   rope_scaling)
                jax.debug.print("q_TNH after rope:\n{val}",
                                val=q_TNH[:3, :5, :5])
                # Apply normaliation after RoPE
                if self.cfg.use_qk_norm:
                    q_TNH = l2_norm(q_TNH)
                    jax.debug.print("q_TNH after L@ Norm:\n{val}",
                                    val=q_TNH[:3, :5, :5])
            else:
                if self.cfg.temperature_tuning:
                    attn_scales = (jnp.log(
                        jnp.floor(
                            (md.input_positions.astype(self.cfg.dtype) + 1.0) /
                            self.cfg.temperature_tuning_floor_scale) + 1.0) *
                                   self.cfg.temperature_tuning_scale + 1.0)
                    q_TNH = q_TNH * attn_scales[:, None, None]
                    jax.debug.print("q_TNH after temperature tuning:\n{val}",
                                    val=q_TNH[:3, :5, :5])

            q_TNH = nnx.with_sharding_constraint(q_TNH,
                                                 self.query_tnh[op_mode])
        with jax.named_scope("k_proj"):
            k_SKH = jnp.einsum('SD,DKH -> SKH', x_SD,
                               self.kernel_k_proj_DKH.value)
            if use_attention_rope:
                k_SKH = apply_rope(k_SKH, md.input_positions, H, rope_theta,
                                   rope_scaling)
                # Apply normaliation after RoPE
                if self.cfg.use_qk_norm:
                    k_SKH = l2_norm(k_SKH)
            k_SKH = nnx.with_sharding_constraint(k_SKH,
                                                 self.keyvalue_skh[op_mode])

        with jax.named_scope("v_proj"):
            v_SKH = jnp.einsum('SD,DKH -> SKH', x_SD,
                               self.kernel_v_proj_DKH.value)
            v_SKH = nnx.with_sharding_constraint(v_SKH,
                                                 self.keyvalue_skh[op_mode])

        with jax.named_scope("attn_op"):
            new_kv_cache, outputs_TNH = self.attention(
                is_prefill,
                kv_cache,
                q_TNH,
                k_SKH,
                v_SKH,
                attention_metadata,
                self.mesh,
            )

        with jax.named_scope("o_proj"):
            o_TD = jnp.einsum('TNH,NHD -> TD', outputs_TNH,
                              self.kernel_o_proj_NHD.value)
            o_TD = nnx.with_sharding_constraint(
                o_TD, self.activation_attention_out_td[op_mode])
            jax.debug.print(
                "o_TD:\n{val}",
                val=o_TD[jnp.array([0, 3])[:, None],
                         jnp.concat([jnp.arange(5),
                                     jnp.arange(-1, -6, -1)])])
        return new_kv_cache, o_TD
