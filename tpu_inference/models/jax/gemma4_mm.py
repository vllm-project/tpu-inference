# Copyright 2026 Google LLC
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

from functools import partial
from itertools import islice
from typing import Any, Callable, Iterable, List, NamedTuple, Optional, Tuple

import jax
import jax.numpy as jnp
from flax import nnx
from jax.sharding import Mesh
from transformers import PretrainedConfig
from vllm.config import VllmConfig
from vllm.model_executor.models.gemma4_mm import \
    Gemma4ForConditionalGeneration as PtGemma4MM

from tpu_inference.layers.common.attention_interface import \
    sharded_flash_attention
from tpu_inference.layers.jax import JaxModule
from tpu_inference.layers.jax.linear import JaxEinsum
from tpu_inference.layers.jax.norm import JaxRmsNorm
from tpu_inference.layers.jax.pp_utils import make_layers
from tpu_inference.layers.jax.rope_interface import apply_rope
from tpu_inference.layers.vllm.quantization.configs import VllmQuantConfig
from tpu_inference.logger import init_logger
from tpu_inference.models.jax.utils.multi_modal_utils import \
    merge_multimodal_embeddings
from tpu_inference.models.jax.utils.weight_utils import (LoadableWithIterator,
                                                         StandardWeightLoader)

logger = init_logger(__name__)

POSITIONS_PAD_VALUE = -1
DEFAULT_ROPE_BASE_FREQUENCY = 10000
DEFAULT_ROPE_SCALE_FACTOR = 1.0
init_fn = nnx.initializers.normal(stddev=0.02)

# --- From gemma4_vision_attention.py ---


def apply_multidimensional_rope(
    inputs: jax.Array,
    positions: jax.Array,
    base_frequency: int,
    rotary_fraction: Optional[float] = None,
    rope_scaling: Optional[dict] = None,
) -> jax.Array:
    """Applies multidimensional RoPE."""

    b, seq_len, num_heads, head_dim = inputs.shape

    # tpu-inference apply_rope expects inputs as (seq_len, num_heads, head_dim)
    # and positions as (seq_len,).
    # We must flatten the batch (B) and sequence (L) dimensions into a single sequence.
    inputs_flat = inputs.reshape((b * seq_len, num_heads, head_dim))
    positions_flat = positions.reshape((b * seq_len, positions.shape[-1]))

    ndim = positions_flat.shape[-1]
    num_rotated_channels = head_dim
    if rotary_fraction is not None:
        num_rotated_channels = int(
            round(num_rotated_channels * rotary_fraction))
    num_rotated_channels_per_dim = 2 * (num_rotated_channels // (2 * ndim))

    split_points = [(k + 1) * num_rotated_channels_per_dim
                    for k in range(ndim)]
    if rotary_fraction is None:
        split_points = split_points[:-1]

    x_parts = jnp.split(inputs_flat, split_points, axis=-1)

    y_parts = [
        apply_rope(
            inputs=x_parts[k],
            positions=positions_flat[
                ..., k],  # Shape becomes (B * L,) matching expected (seq_len,)
            head_dim=x_parts[k].shape[-1],
            rope_theta=base_frequency,  # Explicitly mapping to rope_theta
            rope_scaling=rope_scaling,
        ) for k in range(ndim)
    ]

    if rotary_fraction is not None:
        y_parts.append(x_parts[-1])

    out_flat = jnp.concatenate(y_parts, axis=-1)

    # Reshape back to the original 4D Vision shape
    return out_flat.reshape((b, seq_len, num_heads, head_dim))


class SegmentIds(NamedTuple):
    """SegmentIds required by TPU sharded_flash_attention backend."""
    q: jax.Array
    kv: jax.Array


class Gemma4VisionFlashAttention(JaxModule):
    """
    Gemma 4 Vision Attention using TPU sharded_flash_attention.
    Fixes the output mean shift by introducing the required Value RMSNorm.
    """

    def __init__(self,
                 config: PretrainedConfig,
                 dtype: jnp.dtype,
                 rng: nnx.Rngs,
                 mesh: Mesh,
                 quant_config: Optional[VllmQuantConfig] = None):
        self.features = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = getattr(config, "num_key_value_heads",
                                    self.num_heads)
        self.head_dim = getattr(config, "head_dim",
                                self.features // self.num_heads)
        self.mesh = mesh

        # Fetch Gemma Vision specific RoPE config (theta=100.0)
        rope_params = getattr(config, "rope_parameters",
                              {}).get("full_attention", {})
        self.rope_base_frequency = rope_params.get("rope_theta", 100.0)
        self.rope_scaling = getattr(config, "rope_scaling", None)

        self.q_proj = JaxEinsum("BTD,DNH->BTNH",
                                (self.features, self.num_heads, self.head_dim),
                                param_dtype=dtype,
                                rngs=rng,
                                quant_config=quant_config)
        self.k_proj = JaxEinsum(
            "BTD,DKH->BTKH", (self.features, self.num_kv_heads, self.head_dim),
            param_dtype=dtype,
            rngs=rng,
            quant_config=quant_config)
        self.v_proj = JaxEinsum(
            "BTD,DKH->BTKH", (self.features, self.num_kv_heads, self.head_dim),
            param_dtype=dtype,
            rngs=rng,
            quant_config=quant_config)
        self.o_proj = JaxEinsum("BTNH,NHD->BTD",
                                (self.num_heads, self.head_dim, self.features),
                                param_dtype=dtype,
                                rngs=rng,
                                quant_config=quant_config)

        # Gemma 4 uses RMSNorm for Q, K, and V
        self.q_norm = JaxRmsNorm(self.head_dim,
                                 param_dtype=dtype,
                                 rngs=rng,
                                 quant_config=quant_config)
        self.k_norm = JaxRmsNorm(self.head_dim,
                                 param_dtype=dtype,
                                 rngs=rng,
                                 quant_config=quant_config)

        # MEAN SHIFT FIX: Added v_norm. Note `use_scale=False` to match gemma4.py text tower logic.
        self.v_norm = JaxRmsNorm(self.head_dim,
                                 param_dtype=dtype,
                                 use_scale=False,
                                 scale_init=None,
                                 rngs=rng,
                                 quant_config=quant_config)

    def __call__(self,
                 x: jax.Array,
                 segment_pos: jax.Array,
                 input_mask: Optional[jax.Array] = None) -> jax.Array:
        B, T, _ = x.shape
        orig_T = T

        # Pad sequence length to multiple of 128
        pad_len = (128 - (T % 128)) % 128
        if pad_len > 0:
            x = jnp.pad(x, ((0, 0), (0, pad_len), (0, 0)))
            segment_pos = jnp.pad(segment_pos, ((0, 0), (0, pad_len), (0, 0)))

            # Pad the input mask as well if it exists
            if input_mask is not None:
                # We pad with False (0) since this is invalid space
                input_mask = jnp.pad(input_mask, ((0, 0), (0, pad_len)))
            T = T + pad_len

        # 1. Project Q, K, V
        query_proj = self.q_proj(x)
        key_proj = self.k_proj(x)
        value_proj = self.v_proj(x)

        # 2. Apply RMSNorms
        query_proj = self.q_norm(query_proj)
        key_proj = self.k_norm(key_proj)
        value_proj = self.v_norm(value_proj)

        # 3. Apply Gemma Multidimensional RoPE
        query_proj = apply_multidimensional_rope(
            query_proj,
            segment_pos,
            base_frequency=self.rope_base_frequency,
            rope_scaling=self.rope_scaling)
        key_proj = apply_multidimensional_rope(
            key_proj,
            segment_pos,
            base_frequency=self.rope_base_frequency,
            rope_scaling=self.rope_scaling)

        # 4. Transpose for Flash Attention: (B, T, N, H) -> (B, N, T, H)
        q_BNTH = jnp.transpose(query_proj, (0, 2, 1, 3))
        k_BKTH = jnp.transpose(key_proj, (0, 2, 1, 3))
        v_BKTH = jnp.transpose(value_proj, (0, 2, 1, 3))

        # 5. Create valid Segment IDs (Vision is full attention, no causal masking)
        # Use segment 1 for valid tokens and segment 2 for padding tokens so they don't attend to each other.

        if input_mask is not None:
            # input_mask is True for valid pixels, False for padding
            # Map valid to 1, padding to 2
            segment_ids_val = jnp.where(input_mask, 1, 2).astype(jnp.int32)
        else:
            valid_ids = jnp.ones((B, orig_T), dtype=jnp.int32)
            if pad_len > 0:
                pad_ids = jnp.full((B, pad_len), 2, dtype=jnp.int32)
                segment_ids_val = jnp.concatenate([valid_ids, pad_ids], axis=1)
            else:
                segment_ids_val = valid_ids

        jax.debug.print(
            "[VISION TRACE] Gemma4VisionFlashAttention x shape: {s}, padded T: {t}, valid tokens (1): {v}, padding tokens (2): {p}",
            s=x.shape,
            t=T,
            v=jnp.sum(segment_ids_val == 1),
            p=jnp.sum(segment_ids_val == 2))
        segment_ids = SegmentIds(q=segment_ids_val, kv=segment_ids_val)

        import math

        # 6. Execute TPU Flash Attention Kernel
        outputs_BNTH = sharded_flash_attention(
            mesh=self.mesh,
            causal=False,  # Vision is non-causal
            sm_scale=1.0 / math.sqrt(self.head_dim))(q_BNTH, k_BKTH, v_BKTH,
                                                     segment_ids)

        # 7. Transpose back: (B, N, T, H) -> (B, T, N, H)
        outputs_BTNH = jnp.transpose(outputs_BNTH, (0, 2, 1, 3))

        # 8. Final Output Projection
        final_output = self.o_proj(outputs_BTNH)

        # Remove padding if it was added
        if pad_len > 0:
            final_output = final_output[:, :orig_T, :]

        return final_output


# --- From gemma4_vision.py ---


class VisionEntry(JaxModule):
    """
    Handles converting input [B, H, W, C] to patches [B, L, D],
    adding factorized positional embeddings.
    """

    def __init__(self, config, dtype, rngs: nnx.Rngs, quant_config=None):
        self.config = config
        self.dtype = dtype

        self.patch_size = config.patch_size
        print(
            f"[JAX DEBUG] config.patch_size={getattr(config, 'patch_size', 14)}, config={config}"
        )

        # Linear projection
        self.input_proj = JaxEinsum(
            "...d,dh->...h",
            (3 * self.patch_size**2, config.hidden_size),
            param_dtype=dtype,
            kernel_init=nnx.with_partitioning(init_fn, (None, "model")),
            bias_init=None,
            rngs=rngs,
            quant_config=quant_config,
            prefix="input_proj",
        )

        self.position_embedding_table = nnx.Param(
            jax.random.normal(rngs.params(), (10240, 2, config.hidden_size),
                              dtype=dtype))

    def _factorized_posemb(self, positions_xy: jax.Array) -> jax.Array:
        posemb = self.position_embedding_table.value
        one_hot = jax.nn.one_hot(positions_xy,
                                 posemb.shape[0],
                                 dtype=posemb.dtype)

        nan = jnp.logical_not(one_hot.any(axis=-1, keepdims=True))
        nan = jnp.logical_and(nan, positions_xy[..., None]
                              != POSITIONS_PAD_VALUE)
        pos_oh = jnp.where(nan, jnp.nan, one_hot)

        pe_seq = jnp.einsum('blis,sid->ibld', pos_oh,
                            posemb).astype(posemb.dtype)
        return jnp.sum(pe_seq, axis=0)

    def __call__(
        self,
        patches: jax.Array,
        positions_xy: Optional[jax.Array] = None,
    ) -> jax.Array:
        if patches.ndim != 3:
            raise ValueError(
                f"Expected patches to be 3D or images to be 4D, but got shape {patches.shape} with ndim {patches.ndim}"
            )
        assert positions_xy is not None

        jax.debug.print(
            "[VISION TRACE] VisionEntry patches shape: {s}, mean: {m}",
            s=patches.shape,
            m=patches.mean())
        patches = 2.0 * (patches - 0.5)
        x = self.input_proj(patches)
        pos_embed = self._factorized_posemb(positions_xy).astype(x.dtype)

        return x + pos_embed


class Gemma4VisionMLP(JaxModule):
    """Feed forward module."""

    def __init__(self,
                 config: PretrainedConfig,
                 dtype: jnp.dtype,
                 rng: nnx.Rngs,
                 quant_config: Optional[VllmQuantConfig] = None):
        self.features = config.hidden_size
        self.hidden_dim = config.intermediate_size

        self.gate_proj = JaxEinsum(
            "...d,df->...f",
            (self.features, self.hidden_dim),
            param_dtype=dtype,
            kernel_init=nnx.with_partitioning(init_fn, (None, "model")),
            bias_init=None,
            rngs=rng,
            quant_config=quant_config,
        )

        self.up_proj = JaxEinsum(
            "...d,df->...f",
            (self.features, self.hidden_dim),
            param_dtype=dtype,
            kernel_init=nnx.with_partitioning(init_fn, (None, "model")),
            bias_init=None,
            rngs=rng,
            quant_config=quant_config,
        )

        self.down_proj = JaxEinsum(
            "...f,fd->...d",
            (self.hidden_dim, self.features),
            param_dtype=dtype,
            kernel_init=nnx.with_partitioning(init_fn, ("model", None)),
            bias_init=None,
            rngs=rng,
            quant_config=quant_config,
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        gate = jax.nn.gelu(self.gate_proj(x), approximate=True)
        return self.down_proj(gate * self.up_proj(x))


class Gemma4VisionEncoderLayer(JaxModule):
    # Added `mesh` to init so we can pass it to Flash Attention
    def __init__(self,
                 config: PretrainedConfig,
                 dtype: jnp.dtype,
                 rng: nnx.Rngs,
                 mesh: Mesh,
                 quant_config: Optional[VllmQuantConfig] = None):
        self.input_layernorm = JaxRmsNorm(config.hidden_size,
                                          param_dtype=dtype,
                                          rngs=rng,
                                          quant_config=quant_config)

        self.self_attn = Gemma4VisionFlashAttention(config, dtype, rng, mesh,
                                                    quant_config)

        self.post_attention_layernorm = JaxRmsNorm(config.hidden_size,
                                                   param_dtype=dtype,
                                                   rngs=rng,
                                                   quant_config=quant_config)

        self.pre_feedforward_layernorm = JaxRmsNorm(config.hidden_size,
                                                    param_dtype=dtype,
                                                    rngs=rng,
                                                    quant_config=quant_config)
        self.mlp = Gemma4VisionMLP(config, dtype, rng, quant_config)
        self.post_feedforward_layernorm = JaxRmsNorm(config.hidden_size,
                                                     param_dtype=dtype,
                                                     rngs=rng,
                                                     quant_config=quant_config)

        # self.layer_scalar = nnx.Param(jnp.ones((), dtype=dtype))

    def __call__(self,
                 inputs: jax.Array,
                 positions: jax.Array,
                 input_mask: Optional[jax.Array] = None) -> jax.Array:
        normed_inputs = self.input_layernorm(inputs)

        # Pass the 1D mask down to the Flash Attention kernel
        attn_output = self.self_attn(normed_inputs,
                                     positions,
                                     input_mask=input_mask)

        attn_output = self.post_attention_layernorm(attn_output)
        attn_output += inputs

        outputs = self.pre_feedforward_layernorm(attn_output)
        outputs = self.mlp(outputs)
        outputs = self.post_feedforward_layernorm(outputs)
        outputs += attn_output

        # outputs = outputs * self.layer_scalar.value
        return outputs


class VisionExit(JaxModule):
    """
    Vision exit layer with dynamic spatial pooling.
    Gemma 4 strictly uses a 3x3 pooling kernel rather than a hardcoded output length.
    """

    def __init__(self, config: PretrainedConfig, dtype: jnp.dtype):
        self.config = config
        self.d_model = config.hidden_size
        self.param_dtype = dtype

    def _avg_pool_by_positions(
        self,
        x: jax.Array,
        positions_xy: jax.Array,
    ) -> Tuple[jax.Array, jax.Array]:
        # Gemma 4 uses a strict pooling kernel (default 3x3)
        k = getattr(self.config, 'pooling_kernel_size', 3)

        # Dynamically calculate the true target length based on input patches
        length = x.shape[1] // (k**2)

        # Positions are [X, Y], so index 0 is X (Width) and index 1 is Y (Height)
        max_x = positions_xy[..., 0].max(axis=-1, keepdims=True) + 1
        kernel_idxs = jnp.floor_divide(positions_xy, k)

        # Row-major flat index calculation: (Y_pool * Width_pool) + X_pool
        pooled_width = max_x // k
        flat_kernel_idx = kernel_idxs[..., 1] * pooled_width + kernel_idxs[...,
                                                                           0]

        weights = jax.nn.one_hot(flat_kernel_idx, length, dtype=x.dtype) / (k**
                                                                            2)
        output = jnp.einsum('bLl,bLd->bld', weights, x)

        mask = jnp.logical_not((weights == 0).all(axis=1))
        return output, mask

    def _maybe_downsample(
        self,
        x: jax.Array,
        positions_xy: Optional[jax.Array],
    ) -> Tuple[jax.Array, jax.Array]:
        if positions_xy is not None:
            return self._avg_pool_by_positions(x, positions_xy)

        # Fallback if no positions are provided (e.g., dummy testing)
        k = getattr(self.config, 'pooling_kernel_size', 3)
        length = x.shape[1] // (k**2)
        cur_width = int(x.shape[1]**0.5)
        output_width = cur_width // k

        x_2d = x.reshape((x.shape[0], cur_width, cur_width, x.shape[-1]))
        x_2d = x_2d.reshape(x.shape[0], output_width, k, output_width, k,
                            x.shape[-1])
        x_pooled = x_2d.mean(axis=(2, 4))

        x_pooled = x_pooled.reshape(x.shape[0], length, x.shape[-1])
        mask = jnp.ones(x_pooled.shape[:-1], dtype=jnp.bool_)
        return x_pooled, mask

    def __call__(
        self,
        x: jax.Array,
        positions_xy: Optional[jax.Array] = None,
        output_length_overrides: Optional[Tuple[int, ...]] = None,
    ) -> Tuple[Tuple[jax.Array, jax.Array], ...]:

        x = x.astype(self.param_dtype)

        pooled_x, mask = self._maybe_downsample(x, positions_xy)
        jax.debug.print(
            "[VISION TRACE] VisionExit pooled_x shape: {s}, valid tokens: {v}",
            s=pooled_x.shape,
            v=jnp.sum(mask))
        pooled_x = pooled_x * jnp.sqrt(self.d_model)

        # Return as a tuple of tuples to match the expected API
        return ((pooled_x, mask), )


class Gemma4VisionModel(JaxModule):
    """
    Top-level wrapper for the Gemma 4 Vision Encoder.
    Translates VisionTransformer from _transformer.py.
    """

    def __init__(self,
                 config: PretrainedConfig,
                 dtype: jnp.dtype,
                 rng: nnx.Rngs,
                 mesh: Mesh,
                 quant_config: Optional[VllmQuantConfig] = None):
        self.config = config

        self.dtype = dtype
        self.mesh = mesh

        # 1. Vision Entry (Positional Embeddings)
        self.patch_embedder = VisionEntry(config, dtype, rng, quant_config)

        # 2. Transformer Blocks
        # We use make_layers instead of nn.scan to natively support tpu-inference Pipeline Parallelism
        num_layers = getattr(config, "num_hidden_layers", 32)
        self.start_layer, self.end_layer, self.layers = make_layers(
            num_layers, lambda *_: Gemma4VisionEncoderLayer(
                config, dtype, rng, self.mesh, quant_config))

        # self.final_norm = JaxRmsNorm(self.config.hidden_size, param_dtype=self.dtype, rngs=rng)

        # 3. Vision Exit (Spatial Pooling)
        self.vision_exit = VisionExit(config, dtype)

        # Gemma 4 standardization parameters for Vision Model outputs
        self.std_bias = nnx.Param(
            jnp.zeros((config.hidden_size, ), dtype=dtype))
        self.std_scale = nnx.Param(
            jnp.ones((config.hidden_size, ), dtype=dtype))

    def __call__(
        self,
        pixel_values: jax.Array,
        input_mask: Optional[jax.Array] = None,
        positions_xy: Optional[jax.Array] = None,
    ):
        """
        Forward pass for the complete Vision Encoder.
        """
        # This now receives the newly generated positions_xy instead of None
        hidden_states = self.patch_embedder(pixel_values, positions_xy)

        # 3. Forward through Transformer Layers
        for layer in islice(self.layers, self.start_layer, self.end_layer):
            hidden_states = layer(hidden_states, positions_xy, input_mask)
        # Apply standardization
        hidden_states = hidden_states * self.std_scale.value + self.std_bias.value

        # 4. Forward through Exit (Pooling)
        outputs = self.vision_exit(hidden_states, positions_xy)

        return outputs


# --- From gemma4.py ---


class Gemma4MultimodalEmbedder(JaxModule):

    def __init__(self,
                 vision_hidden_size: int,
                 text_hidden_size: int,
                 dtype: jnp.dtype,
                 rng: nnx.Rngs,
                 quant_config: Optional[VllmQuantConfig] = None,
                 prefix: str = "",
                 rms_norm_eps: float = 1e-6):
        self.embedding_projection = JaxEinsum(
            "bld,dh->blh",
            (vision_hidden_size, text_hidden_size),
            bias_shape=None,
            param_dtype=dtype,
            kernel_init=nnx.with_partitioning(init_fn, (None, "model")),
            bias_init=None,
            rngs=rng,
            quant_config=quant_config,
            prefix=prefix + ".embedding_projection",
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        x = self.embedding_projection(x)
        return x


class Gemma4ForConditionalGeneration(JaxModule, LoadableWithIterator):
    packed_modules_mapping = {"__no_packing__": []}
    WeightLoader = StandardWeightLoader
    supports_multimodal = True
    _processor_factory = getattr(PtGemma4MM, "_processor_factory", None)

    def __init__(self, vllm_config: VllmConfig, rng_key: jax.Array,
                 mesh: Mesh) -> None:
        self.vllm_config = vllm_config
        rng = nnx.Rngs(rng_key)
        self.mesh = mesh

        from tpu_inference.models.jax.gemma4 import Gemma4Model
        self.model = Gemma4Model(
            vllm_config=vllm_config,
            rng=rng,
            mesh=mesh,
            prefix="model",
        )
        model_config = vllm_config.model_config

        # Initialize Vision Tower (Make sure to pass mesh if Flash Attention needs it)
        vision_config = model_config.hf_config.vision_config
        self.image_token_id = getattr(model_config.hf_config, "image_token_id",
                                      258880)

        self.vision_tower = Gemma4VisionModel(
            config=vision_config,
            dtype=model_config.dtype,
            rng=rng,
            quant_config=vllm_config.quant_config,
            mesh=mesh)

        self.embed_vision = Gemma4MultimodalEmbedder(
            vision_hidden_size=vision_config.hidden_size,
            text_hidden_size=model_config.hf_config.text_config.hidden_size,
            dtype=model_config.dtype,
            rng=rng,
            quant_config=vllm_config.quant_config,
            prefix="embed_vision")

        # Gemma 4: soft-capping in the final logits.
        self.final_logit_softcapping = getattr(
            model_config.hf_config.text_config, "final_logit_softcapping",
            None)

        if not model_config.hf_config.tie_word_embeddings:
            if self.model.is_last_rank:
                vocab_size = model_config.get_vocab_size()
                hidden_size = model_config.hf_config.text_config.hidden_size
                from tpu_inference.layers.jax.linear import JaxEinsum
                self.lm_head = JaxEinsum(
                    "TD,DV->TV",
                    (hidden_size, vocab_size),
                    param_dtype=model_config.dtype,
                    kernel_init=nnx.with_partitioning(init_fn,
                                                      ("model", None)),
                    rngs=rng,
                    quant_config=vllm_config.quant_config,
                    prefix="lm_head",
                )
            else:
                from tpu_inference.layers.jax.pp_utils import PPMissingLayer
                self.lm_head = PPMissingLayer()

    def load_weights(self, weights: Iterable[Tuple[str, Any]]):

        def map_name(name: str) -> str:
            # Gemma 4 multimodal remappings
            name = name.replace("model.embed_vision.", "embed_vision.")
            name = name.replace("model.vision_tower.encoder.", "vision_tower.")
            name = name.replace("model.vision_tower.", "vision_tower.")
            name = name.replace("model.multi_modal_projector.linear.",
                                "embed_vision.embedding_projection.")
            name = name.replace("model.multi_modal_projector.",
                                "embed_vision.")

            if "vision_tower.layers." in name:
                name = name.replace(".linear.weight", ".weight")

            # Text model remapping
            name = name.replace("model.language_model.", "model.")
            if "model.lm_head" in name:
                name = name.replace("model.lm_head", "lm_head")

            return name

        def process_tensor(mapped_name, tensor):
            # 1. Shape and Math Fixes
            if "position_embedding_table" in mapped_name:
                # PyTorch (2, 10240, hidden) -> JAX (10240, 2, hidden)
                return tensor.transpose(0, 1)

            return tensor

        def filter_weights(weights_iterator):
            import re
            for name, weight in weights_iterator:
                mapped_name = map_name(name)

                # Handle packed QKV weights for the text tower
                if "qkv_proj" in mapped_name:
                    m = re.search(r"layers\.(\d+)\.", mapped_name)
                    if m:
                        layer_idx = int(m.group(1))
                        if self.model.start_layer <= layer_idx < self.model.end_layer:
                            jax_attn = self.model.layers[
                                layer_idx - self.model.start_layer].self_attn
                            q_size = jax_attn.num_heads * jax_attn.head_dim_original
                            kv_size = jax_attn.num_kv_heads * jax_attn.head_dim_original

                            q_weight = weight[:q_size]
                            k_weight = weight[q_size:q_size + kv_size]
                            v_weight = weight[q_size + kv_size:q_size +
                                              2 * kv_size]

                            yield mapped_name.replace(
                                "qkv_proj", "q_proj"), process_tensor(
                                    mapped_name.replace("qkv_proj", "q_proj"),
                                    q_weight)
                            yield mapped_name.replace(
                                "qkv_proj", "k_proj"), process_tensor(
                                    mapped_name.replace("qkv_proj", "k_proj"),
                                    k_weight)
                            yield mapped_name.replace(
                                "qkv_proj", "v_proj"), process_tensor(
                                    mapped_name.replace("qkv_proj", "v_proj"),
                                    v_weight)
                            continue

                yield mapped_name, process_tensor(mapped_name, weight)

        return super().load_weights(filter_weights(weights))

    def embed_input_ids(self,
                        input_ids: jax.Array,
                        multimodal_embeddings: Optional[jax.Array] = None,
                        **kwargs) -> jax.Array:
        # 1. Standard Token Embedding
        inputs_embeds = self.model.embed_tokens(input_ids)
        target_dtype = inputs_embeds.dtype

        # 2. Gemma 4 specific scaling
        inputs_embeds = (inputs_embeds *
                         self.model.embedding_scale).astype(target_dtype)

        # 3. Merge Vision Tokens
        if multimodal_embeddings is not None and multimodal_embeddings.shape[
                0] > 0:
            # We need to ensure the merge utility knows these are VISION tokens
            # and potentially apply the projector if it hasn't been applied yet.
            inputs_embeds = merge_multimodal_embeddings(
                input_ids, inputs_embeds, multimodal_embeddings,
                [self.image_token_id])

        return inputs_embeds.astype(target_dtype)

    @partial(jax.jit, static_argnames=["has_positions"])
    def get_single_image_embedding(self, pixel_values: jax.Array,
                                   positions_xy: jax.Array,
                                   has_positions: bool) -> jax.Array:
        input_mask = None
        if has_positions:
            input_mask = positions_xy[..., 0] != -1
            pos_xy = positions_xy
        else:
            pos_xy = None

        vision_outputs = self.vision_tower(pixel_values,
                                           input_mask=input_mask,
                                           positions_xy=pos_xy)

        projected_vision_features = vision_outputs[0][0]
        pooler_mask = vision_outputs[0][1]

        projected_vision_features = self.embed_vision(
            projected_vision_features)

        # Pack valid tokens to the front
        seq_len = pooler_mask.shape[1]
        indices = jnp.arange(seq_len)
        sort_key = jnp.where(pooler_mask, indices, seq_len + indices)
        sort_idx = jnp.argsort(sort_key, axis=1)
        projected_vision_features = jnp.take_along_axis(
            projected_vision_features,
            jnp.expand_dims(sort_idx, axis=-1),
            axis=1)

        return projected_vision_features

    def _parse_and_validate_image_input(self,
                                        **kwargs: object) -> Optional[dict]:
        pixel_values = kwargs.pop("pixel_values", None)
        positions_xy = kwargs.pop("pixel_position_ids", None)
        patches_per_image = kwargs.pop("patches_per_image", None)

        if pixel_values is None:
            return None

        # Ensure correct layout for JAX Vision Model
        from tpu_inference import utils
        dtype_str = str(self.vllm_config.model_config.dtype).split('.')[-1]
        jax_dtype = utils.get_jax_dtype_from_str_dtype(dtype_str)
        pixel_values = jnp.asarray(pixel_values, dtype=jax_dtype)

        if positions_xy is not None:
            positions_xy = jnp.asarray(positions_xy, dtype=jnp.int32)

        return {
            "type": "pixel_values",
            "pixel_values": pixel_values,
            "positions_xy": positions_xy,
            "patches_per_image": patches_per_image
        }

    def _process_image_input(self, image_input: dict) -> list[jax.Array]:
        pixel_values = image_input["pixel_values"]
        positions_xy = image_input["positions_xy"]
        patches_per_image = image_input["patches_per_image"]

        num_images = pixel_values.shape[0]

        image_embeds = []
        has_positions = positions_xy is not None

        for i in range(num_images):
            pv = pixel_values[i:i + 1]  # Keep batch dim
            if has_positions:
                pos = positions_xy[i:i + 1]
            else:
                # Dummy array to keep JAX happy, since it must be an array for JIT
                pos = jnp.zeros((1, 1, 2), dtype=jnp.int32)

            emb = self.get_single_image_embedding(pv, pos, has_positions)
            image_embeds.append(emb)

        if not image_embeds:
            return []

        projected_vision_features = jnp.concatenate(image_embeds, axis=0)

        # Reshape and Split logic
        tokens_per_tile = projected_vision_features.shape[1]
        hidden_dim = projected_vision_features.shape[2]
        all_tokens_flat = projected_vision_features.reshape(-1, hidden_dim)

        if hasattr(patches_per_image, 'tolist'):
            tile_counts = patches_per_image.tolist()
        else:
            tile_counts = list(
                patches_per_image) if patches_per_image is not None else [1]

        split_sizes = [c * tokens_per_tile for c in tile_counts]
        split_indices = jnp.cumsum(jnp.array(split_sizes[:-1]))
        output_list = jnp.split(all_tokens_flat, split_indices)

        return list(output_list)

    def embed_multimodal(self,
                         image_grid_thw=None,
                         **kwargs) -> List[jax.Array]:
        jax.debug.print(
            "\n[BACKEND DEBUG] embed_multimodal called! pixel_values present: {p}",
            p=1 if "pixel_values" in kwargs
            and kwargs["pixel_values"] is not None else 0)

        image_input = self._parse_and_validate_image_input(**kwargs)
        if image_input is None:
            return []

        return self._process_image_input(image_input)

    def precompile_vision_encoder(
        self,
        run_compilation_fn: Callable,
    ) -> None:

        image_shapes = []
        # Attempt to extract warm-up configurations specifically tailored for vision
        if hasattr(self.vllm_config,
                   'additional_config') and self.vllm_config.additional_config:
            warmup_config = self.vllm_config.additional_config.get(
                "vision_warmup_config", {})
            if warmup_config:
                image_shapes = warmup_config.get("image_shapes", [])

        # Run compilation for all requested visual resolutions
        for input_hw in image_shapes:
            if not isinstance(input_hw, list) or len(input_hw) != 2:
                logger.warning(f"Skipping invalid shape {input_hw}.")
                continue
            h_input, w_input = input_hw

            from tpu_inference import utils
            dtype_str = str(self.vllm_config.model_config.dtype).split('.')[-1]
            jax_dtype = utils.get_jax_dtype_from_str_dtype(dtype_str)

            dummy_pixel_values = jnp.ones(
                (1, h_input, w_input, 3),
                dtype=jax_dtype,
            )

            # Positions shape logic mirrors _patchify from VisionEntry
            p = self.vision_tower.patch_embedder.patch_size
            h_p, w_p = h_input // p, w_input // p
            dummy_positions_xy = jnp.ones((1, h_p * w_p, 2), dtype=jnp.int32)
            has_positions = True

            # Trigger JIT
            run_compilation_fn("vision_encoder",
                               self.get_single_image_embedding,
                               dummy_pixel_values,
                               dummy_positions_xy,
                               has_positions,
                               image_shape=input_hw)

    def __call__(
        self,
        kv_caches: List[jax.Array],
        input_ids: jax.Array,
        attention_metadata: Any,
        inputs_embeds: Optional[jax.Array] = None,
        _input_positions=None,
        _layer_name_to_kv_cache=None,
        _lora_metadata=None,
        intermediate_tensors: Any | None = None,
        is_first_rank: bool = True,
        is_last_rank: bool = True,
        *args,
    ) -> Tuple[List[jax.Array], jax.Array | Any, List[jax.Array]]:

        multimodal_embeddings = getattr(attention_metadata,
                                        "multimodal_embeddings", None)

        if inputs_embeds is None:
            inputs_embeds = self.embed_input_ids(input_ids,
                                                 multimodal_embeddings)

        if not is_first_rank and intermediate_tensors is not None:
            inputs_embeds = intermediate_tensors["hidden_states"]

        layer_name_to_kv_cache = dict(
            _layer_name_to_kv_cache) if _layer_name_to_kv_cache else None

        kv_caches, x = self.model(
            kv_caches,
            input_ids,
            attention_metadata,
            inputs_embeds,
            layer_name_to_kv_cache=layer_name_to_kv_cache,
        )

        if not is_last_rank:
            from tpu_inference.models.jax.jax_intermediate_tensor import \
                JaxIntermediateTensors
            x = JaxIntermediateTensors(tensors={"hidden_states": x})

        return kv_caches, x, []

    def compute_logits(self, hidden_states: jax.Array) -> jax.Array:
        if hasattr(self, 'lm_head'):
            logits = self.lm_head(hidden_states)
        else:
            logits = self.model.embed_tokens.decode(hidden_states)

        # Gemma4: Use Logit Soft-capping
        if self.final_logit_softcapping is not None:
            logits = jnp.tanh(
                logits /
                self.final_logit_softcapping) * self.final_logit_softcapping
        return logits
