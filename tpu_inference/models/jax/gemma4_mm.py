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

from itertools import islice
from typing import (Any, Callable, Iterable, List, Literal, NamedTuple,
                    Optional, Tuple, TypedDict)

import jax
import jax.numpy as jnp
import torch
from flax import nnx
from jax.sharding import Mesh
from transformers import PretrainedConfig
from vllm.config import VllmConfig

from tpu_inference.models.jax.jax_intermediate_tensor import \
    JaxIntermediateTensors

try:
    from vllm.model_executor.models.gemma4_mm import \
        Gemma4ForConditionalGeneration as PtGemma4MM
except ImportError:
    # TODO(#2308): Remove try-except once we have transformers>=5.5.0
    PtGemma4MM = None

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
init_fn = nnx.initializers.normal(stddev=0.02)


class Gemma4ImagePixelInputs(TypedDict):
    """
    Pre-patchified image inputs from the Gemma4 image processor.

    Dimensions:
        - bn: Batch size * number of images
        - np: Number of patches (max_patches = max_soft_tokens * pooling_kernel_size²)
        - pp: Patch pixels (patch_size² * 3)

    The HF Gemma4ImageProcessor outputs pixel_values as
    (batch, max_patches, patch_pixels) — already patchified with
    zero-padding for patches beyond the real image content.
    pixel_position_ids provides (x, y) coordinates per patch,
    with (-1, -1) for padding patches.
    """
    type: Literal["pixel_values"]
    pixel_values: jax.Array
    """
    Shape: `(bn, np, pp)`
    """
    pixel_position_ids: jax.Array
    """
    Shape: `(bn, np, 2)`
    """


def apply_multidimensional_rope(
    inputs: jax.Array,
    positions: jax.Array,
    base_frequency: int,
    rotary_fraction: Optional[float] = None,
    rope_scaling: Optional[dict] = None,
) -> jax.Array:
    """Applies multidimensional RoPE."""

    b, seq_len, num_heads, head_dim = inputs.shape

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
            positions=positions_flat[..., k],
            head_dim=x_parts[k].shape[-1],
            rope_theta=base_frequency,
            rope_scaling=rope_scaling,
        ) for k in range(ndim)
    ]

    if rotary_fraction is not None:
        y_parts.append(x_parts[-1])

    out_flat = jnp.concatenate(y_parts, axis=-1)

    return out_flat.reshape((b, seq_len, num_heads, head_dim))


class SegmentIds(NamedTuple):
    """SegmentIds required by TPU sharded_flash_attention backend."""
    q: jax.Array
    kv: jax.Array


class Gemma4VisionFlashAttention(JaxModule):
    """
    Gemma 4 Vision Attention using TPU sharded_flash_attention.
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

        self.q_norm = JaxRmsNorm(self.head_dim,
                                 param_dtype=dtype,
                                 rngs=rng,
                                 quant_config=quant_config)
        self.k_norm = JaxRmsNorm(self.head_dim,
                                 param_dtype=dtype,
                                 rngs=rng,
                                 quant_config=quant_config)
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

        pad_len = (128 - (T % 128)) % 128
        if pad_len > 0:
            x = jnp.pad(x, ((0, 0), (0, pad_len), (0, 0)))
            segment_pos = jnp.pad(segment_pos, ((0, 0), (0, pad_len), (0, 0)))

            if input_mask is not None:
                input_mask = jnp.pad(input_mask, ((0, 0), (0, pad_len)))
            T = T + pad_len

        query_proj = self.q_proj(x)
        key_proj = self.k_proj(x)
        value_proj = self.v_proj(x)

        query_proj = self.q_norm(query_proj)
        key_proj = self.k_norm(key_proj)
        value_proj = self.v_norm(value_proj)

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

        # Transpose for Flash Attention: (B, T, N, H) -> (B, N, T, H)
        q_BNTH = jnp.transpose(query_proj, (0, 2, 1, 3))
        k_BKTH = jnp.transpose(key_proj, (0, 2, 1, 3))
        v_BKTH = jnp.transpose(value_proj, (0, 2, 1, 3))

        if input_mask is not None:
            segment_ids_val = jnp.where(input_mask, 1, 2).astype(jnp.int32)
        else:
            valid_ids = jnp.ones((B, orig_T), dtype=jnp.int32)
            if pad_len > 0:
                pad_ids = jnp.full((B, pad_len), 2, dtype=jnp.int32)
                segment_ids_val = jnp.concatenate([valid_ids, pad_ids], axis=1)
            else:
                segment_ids_val = valid_ids

        segment_ids = SegmentIds(q=segment_ids_val, kv=segment_ids_val)

        outputs_BNTH = sharded_flash_attention(mesh=self.mesh,
                                               causal=False,
                                               sm_scale=1.0)(q_BNTH, k_BKTH,
                                                             v_BKTH,
                                                             segment_ids)

        # Transpose back: (B, N, T, H) -> (B, T, N, H)
        outputs_BTNH = jnp.transpose(outputs_BNTH, (0, 2, 1, 3))

        final_output = self.o_proj(outputs_BTNH)

        if pad_len > 0:
            final_output = final_output[:, :orig_T, :]

        return final_output


class Gemma4VisionPatchEmbedder(JaxModule):
    """
    Handles converting input [B, H, W, C] to patches [B, L, D],
    adding factorized positional embeddings.
    """

    def __init__(self, config, dtype, rngs: nnx.Rngs, quant_config=None):
        self.config = config
        self.dtype = dtype

        self.patch_size = config.patch_size

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

    def _factorized_posemb(self, pixel_position_ids: jax.Array) -> jax.Array:
        posemb = self.position_embedding_table.value
        one_hot = jax.nn.one_hot(pixel_position_ids,
                                 posemb.shape[0],
                                 dtype=posemb.dtype)

        nan = jnp.logical_not(one_hot.any(axis=-1, keepdims=True))
        nan = jnp.logical_and(
            nan, pixel_position_ids[..., None] != POSITIONS_PAD_VALUE)
        pos_oh = jnp.where(nan, jnp.nan, one_hot)

        pe_seq = jnp.einsum('blis,sid->ibld', pos_oh,
                            posemb).astype(posemb.dtype)
        return jnp.sum(pe_seq, axis=0)

    def __call__(
        self,
        patches: jax.Array,
        pixel_position_ids: jax.Array,
    ) -> jax.Array:
        if patches.ndim != 3:
            raise ValueError(
                f"Expected patches to be 3D or images to be 4D, but got shape {patches.shape} with ndim {patches.ndim}"
            )
        assert pixel_position_ids is not None

        patches = 2.0 * (patches - 0.5)
        x = self.input_proj(patches)
        pos_embed = self._factorized_posemb(pixel_position_ids).astype(x.dtype)

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

    def __call__(self,
                 inputs: jax.Array,
                 positions: jax.Array,
                 input_mask: Optional[jax.Array] = None) -> jax.Array:
        normed_inputs = self.input_layernorm(inputs)

        attn_output = self.self_attn(normed_inputs,
                                     positions,
                                     input_mask=input_mask)

        attn_output = self.post_attention_layernorm(attn_output)
        attn_output += inputs

        outputs = self.pre_feedforward_layernorm(attn_output)
        outputs = self.mlp(outputs)
        outputs = self.post_feedforward_layernorm(outputs)
        outputs += attn_output

        return outputs


class Gemma4VisionPooler(JaxModule):
    """
    Vision exit layer with dynamic spatial pooling.
    """

    def __init__(self, config: PretrainedConfig, dtype: jnp.dtype):
        self.config = config
        self.d_model = config.hidden_size
        self.param_dtype = dtype

    def __call__(
        self,
        x: jax.Array,
        pixel_position_ids: jax.Array,
    ) -> Tuple[Tuple[jax.Array, jax.Array], ...]:

        x = x.astype(self.param_dtype)
        k = getattr(self.config, 'pooling_kernel_size', 3)
        length = x.shape[1] // (k**2)

        max_x = pixel_position_ids[..., 0].max(axis=-1, keepdims=True) + 1
        kernel_idxs = jnp.floor_divide(pixel_position_ids, k)

        pooled_width = max_x // k
        flat_kernel_idx = kernel_idxs[..., 1] * pooled_width + kernel_idxs[...,
                                                                           0]

        weights = jax.nn.one_hot(flat_kernel_idx, length, dtype=x.dtype) / (k**
                                                                            2)
        pooled_x = jnp.einsum('bLl,bLd->bld', weights, x)
        mask = jnp.logical_not((weights == 0).all(axis=1))

        pooled_x = pooled_x * jnp.sqrt(self.d_model)

        return ((pooled_x, mask), )


class Gemma4VisionModel(JaxModule):
    """
    Top-level wrapper for the Gemma 4 Vision Encoder.
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

        self.patch_embedder = Gemma4VisionPatchEmbedder(
            config, dtype, rng, quant_config)

        num_layers = getattr(config, "num_hidden_layers", 32)
        self.start_layer, self.end_layer, self.layers = make_layers(
            num_layers, lambda *_: Gemma4VisionEncoderLayer(
                config, dtype, rng, self.mesh, quant_config))

        self.pooler = Gemma4VisionPooler(config, dtype)

        self.standardize = getattr(config, "standardize", False)
        if self.standardize:
            self.std_bias = nnx.Param(
                jnp.zeros((config.hidden_size, ), dtype=dtype))
            self.std_scale = nnx.Param(
                jnp.ones((config.hidden_size, ), dtype=dtype))

    def __call__(
        self,
        pixel_values: jax.Array,
        pixel_position_ids: jax.Array,
        input_mask: Optional[jax.Array] = None,
    ):
        hidden_states = self.patch_embedder(pixel_values, pixel_position_ids)

        for layer in islice(self.layers, self.start_layer, self.end_layer):
            hidden_states = layer(hidden_states, pixel_position_ids,
                                  input_mask)

        outputs = self.pooler(hidden_states, pixel_position_ids)

        if self.standardize:
            pooled_x, mask = outputs[0]
            pooled_x = (pooled_x - self.std_bias.value) * self.std_scale.value
            outputs = ((pooled_x, mask), )

        return outputs


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
        self.embedding_pre_projection_norm = JaxRmsNorm(
            vision_hidden_size,
            epsilon=rms_norm_eps,
            param_dtype=dtype,
            use_scale=False,
            scale_init=None,
            rngs=rng,
            quant_config=quant_config,
            prefix=prefix + ".embedding_pre_projection_norm",
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        x = self.embedding_pre_projection_norm(x)
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

            if "vision_tower.layers." in name:
                name = name.replace(".linear.weight", ".weight")

            # Text model remapping
            name = name.replace("model.language_model.", "model.")
            if "model.lm_head" in name:
                name = name.replace("model.lm_head", "lm_head")

            return name

        def process_tensor(mapped_name, tensor):
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
        inputs_embeds = self.model.embed_tokens(input_ids)
        target_dtype = inputs_embeds.dtype

        inputs_embeds = (inputs_embeds *
                         self.model.embedding_scale).astype(target_dtype)

        if multimodal_embeddings is not None and multimodal_embeddings.shape[
                0] > 0:
            inputs_embeds = merge_multimodal_embeddings(
                input_ids, inputs_embeds, multimodal_embeddings,
                [self.image_token_id])

        return inputs_embeds.astype(target_dtype)

    @jax.jit
    def get_single_image_embedding(self, pixel_values: jax.Array,
                                   pixel_position_ids: jax.Array) -> jax.Array:
        input_mask = pixel_position_ids[..., 0] != -1

        vision_outputs = self.vision_tower(
            pixel_values,
            input_mask=input_mask,
            pixel_position_ids=pixel_position_ids)

        projected_vision_features = vision_outputs[0][0]
        pooler_mask = vision_outputs[0][1]

        projected_vision_features = self.embed_vision(
            projected_vision_features)

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
        pixel_position_ids = kwargs.pop("pixel_position_ids", None)
        image_embeds = kwargs.pop("image_embeds", None)
        assert image_embeds is None, "Gemma4 does not support image_embeds."
        if pixel_values is None:
            return None
        if isinstance(pixel_values, torch.Tensor):
            pixel_values = pixel_values.contiguous().view(
                torch.int16).numpy().view(jnp.bfloat16)
            pixel_values = jnp.asarray(pixel_values)
        if isinstance(pixel_position_ids, torch.Tensor):
            pixel_position_ids = pixel_position_ids.to(
                torch.int32).contiguous().numpy()
            pixel_position_ids = jnp.asarray(pixel_position_ids)

        return Gemma4ImagePixelInputs(type="pixel_values",
                                      pixel_values=pixel_values,
                                      pixel_position_ids=pixel_position_ids)

    def _process_image_input(self, image_input: dict) -> list[jax.Array]:
        pixel_values = image_input["pixel_values"]
        pixel_position_ids = image_input["pixel_position_ids"]

        if pixel_values.ndim == 2:
            pixel_values = jnp.expand_dims(pixel_values, axis=0)
        if pixel_position_ids.ndim == 2:
            pixel_position_ids = jnp.expand_dims(pixel_position_ids, axis=0)

        per_image_features = []
        batch_size = pixel_values.shape[0]
        for i in range(batch_size):
            pv = pixel_values[i:i + 1, ...]
            pp = pixel_position_ids[i:i + 1, ...]
            vt_output = self.get_single_image_embedding(pv, pp)
            per_image_features.append(vt_output[0])
        return per_image_features

    def embed_multimodal(self, **kwargs) -> List[jax.Array]:
        image_input = self._parse_and_validate_image_input(**kwargs)
        if image_input is None:
            return []

        return self._process_image_input(image_input)

    def __call__(
        self,
        kv_caches: List[jax.Array],
        input_ids: jax.Array,
        attention_metadata: Any,
        inputs_embeds: Optional[jax.Array] = None,
        _input_positions=None,
        _layer_name_to_kv_cache=None,
        _lora_metadata=None,
        intermediate_tensors: JaxIntermediateTensors | None = None,
        is_first_rank: bool = True,
        is_last_rank: bool = True,
        *args,
    ) -> Tuple[List[jax.Array], jax.Array | JaxIntermediateTensors,
               List[jax.Array], Optional[jax.Array]]:

        multimodal_embeddings = getattr(attention_metadata,
                                        "multimodal_embeddings", None)

        if inputs_embeds is None:
            inputs_embeds = self.embed_input_ids(input_ids,
                                                 multimodal_embeddings)

        if not is_first_rank and intermediate_tensors is not None:
            inputs_embeds = intermediate_tensors["hidden_states"]

        layer_name_to_kv_cache = dict(
            _layer_name_to_kv_cache) if _layer_name_to_kv_cache else None

        kv_caches, x, expert_indices = self.model(
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

        return kv_caches, x, [], expert_indices

    def compute_logits(self, hidden_states: jax.Array) -> jax.Array:
        if hasattr(self, 'lm_head'):
            logits = self.lm_head(hidden_states)
        else:
            logits = self.model.embed_tokens.decode(hidden_states)

        if self.final_logit_softcapping is not None:
            logits = jnp.tanh(
                logits /
                self.final_logit_softcapping) * self.final_logit_softcapping
        return logits

    def precompile_vision_encoder(
        self,
        run_compilation_fn: Callable,
    ) -> None:

        image_shapes = []
        if hasattr(self.vllm_config,
                   'additional_config') and self.vllm_config.additional_config:
            warmup_config = self.vllm_config.additional_config.get(
                "vision_warmup_config", {})
            if warmup_config:
                image_shapes = warmup_config.get("image_shapes", [])

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

            p = self.vision_tower.patch_embedder.patch_size
            h_p, w_p = h_input // p, w_input // p
            dummy_pixel_position_ids = jnp.ones((1, h_p * w_p, 2),
                                                dtype=jnp.int32)

            run_compilation_fn("vision_encoder",
                               self.get_single_image_embedding,
                               dummy_pixel_values,
                               dummy_pixel_position_ids,
                               image_shape=input_hw)
