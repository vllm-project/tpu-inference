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
  Implementation of L2 Norm in JAX (taken from MaxText repo - maxtext/MaxText/layers/attentions.py).

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
Llama4AttentionConfig.__doc__ = f"""Llama4-specific attention layer which performs layer norm on the Query and Keys after RoPE.
Additional Args:
  {HuggingFaceArgNames.USE_QK_NORM.value}: bool whether to use Llama4 normalization of query & keys
  {HuggingFaceArgNames.TEMPERATURE_TUNING.value}: bool whether to use temperature tuning
  {HuggingFaceArgNames.TEMPERATURE_TUNING_SCALE.value}: float temperature tuning scale
  {HuggingFaceArgNames.TEMPERATURE_TUNING_FLOOR_SCALE.value}: float temperature tuning floor scale

Inherits AttentionConfig docstring:
{AttentionConfig.__doc__}
"""


@dataclass(kw_only=True)
class Llama4Attention(Attention):
    use_qk_norm: bool
    temperature_tuning: bool
    temperature_tuning_floor_scale: float
    temperature_tuning_scale: float

    def __call__(self,
                 x,
                 is_prefill,
                 kv_cache: KVCache,
                 attention_metadata: AttentionMetadata,
                 use_attention_rope: bool = True):
        """Performs the forward pass of the attention module.

        This method computes the attention output by projecting the input `x`
        to queries, keys, and values, applying RoPE and L2Norm if specified,
        performing scaled dot-product attention, and projecting the results
        back to the model dimension.
        If no RoPE (NoPE) is specified, one can also perform temperature tuning
        which is useful to combat dilution of attention scores in long-context attention.

        Args:
            x: The input tensor of shape `(seq_len, d_model)`.
            is_prefill: Whether the operation mode is prefill (otherwise it is generate).
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
        x = jnp.asarray(x, self.dtype)
        x_SD = nnx.with_sharding_constraint(
            x, self.activation_attention_td[op_mode])
        x_q_TD = nnx.with_sharding_constraint(x, self.activation_q_td[op_mode])
        rope_scaling = self.rope_scaling
        rope_theta = self.rope_theta
        H = self.head_dim
        l2_norm = L2Norm()

        with jax.named_scope("q_proj"):
            q_TNH = jnp.einsum('TD,DNH -> TNH', x_q_TD,
                               self.kernel_q_proj_DNH.value)
            if use_attention_rope:
                q_TNH = apply_rope(q_TNH, md.input_positions, H, rope_theta,
                                   rope_scaling, self.rope_input_ordering)

                # Apply normaliation after RoPE
                if self.use_qk_norm:
                    q_TNH = l2_norm(q_TNH)
            else:
                if self.temperature_tuning:
                    q_TNH = self.apply_temperature_tuning(md, q_TNH)

            q_TNH = nnx.with_sharding_constraint(q_TNH,
                                                 self.query_tnh[op_mode])
        with jax.named_scope("k_proj"):
            k_SKH = jnp.einsum('SD,DKH -> SKH', x_SD,
                               self.kernel_k_proj_DKH.value)
            if use_attention_rope:
                k_SKH = apply_rope(k_SKH, md.input_positions, H, rope_theta,
                                   rope_scaling, self.rope_input_ordering)

                # Apply normaliation after RoPE
                if self.use_qk_norm:
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
        return new_kv_cache, o_TD

    def apply_temperature_tuning(self, md: AttentionMetadata,
                                 input_arr_TNH: jax.Array) -> jax.Array:
        """Applies temperature tuning to the input array of shape (T, N, H).
        Args:
            md: AttentionMetadata object containing the input positions.
            input_arr_TNH: Input array of shape (T, N, H) which will have scaled temperatures applied.
        """
        attn_scales = (jnp.log(
            jnp.floor((md.input_positions.astype(self.dtype) + 1.0) /
                      self.temperature_tuning_floor_scale) + 1.0) *
                       self.temperature_tuning_scale + 1.0)
        return input_arr_TNH * attn_scales[:, None, None]
