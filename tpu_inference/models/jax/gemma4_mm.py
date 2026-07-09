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

import functools
from itertools import islice
from typing import (Any, Callable, Iterable, List, Literal, NamedTuple,
                    Optional, Tuple, TypedDict)

import jax
import jax.numpy as jnp
import torch
from flax import nnx
from jax.sharding import Mesh, NamedSharding, PartitionSpec
from transformers import PretrainedConfig
from vllm.config import VllmConfig
from vllm.model_executor.models.gemma4_mm import \
    Gemma4ForConditionalGeneration as PtGemma4MM
from vllm.model_executor.models.utils import WeightsMapper

from tpu_inference.layers.common.attention_interface import \
    sharded_flash_attention
from tpu_inference.layers.common.sharding import ShardingAxisName
from tpu_inference.layers.jax import JaxModule
from tpu_inference.layers.jax.linear import JaxEinsum
from tpu_inference.layers.jax.norm import JaxRmsNorm
from tpu_inference.layers.jax.pp_utils import make_layers
from tpu_inference.layers.vllm.quantization.configs import VllmQuantConfig
from tpu_inference.logger import init_logger
from tpu_inference.models.jax.gemma4 import Gemma4ForCausalLM, Gemma4Model
from tpu_inference.models.jax.jax_intermediate_tensor import \
    JaxIntermediateTensors
from tpu_inference.models.jax.utils.multi_modal_utils import \
    merge_multimodal_embeddings
from tpu_inference.models.jax.utils.weight_utils import (
    JaxAutoWeightsLoader, LoadableWithIterator, StandardWeightLoader,
    load_nnx_param_from_reshaped_torch)
from tpu_inference.utils import get_mesh_shape_product

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
) -> jax.Array:
    """Applies multidimensional RoPE."""

    b, seq_len, num_heads, head_dim = inputs.shape
    ndim = positions.shape[-1]

    num_rotated_channels = head_dim
    if rotary_fraction is not None:
        num_rotated_channels = int(
            round(num_rotated_channels * rotary_fraction))

    c_per_dim = 2 * (num_rotated_channels // (2 * ndim))
    half_c = c_per_dim // 2
    rot_dim = ndim * c_per_dim

    x_rot = inputs[..., :rot_dim]
    x_unrot = inputs[..., rot_dim:]

    x_reshaped = x_rot.reshape(b, seq_len, num_heads, ndim, 2, half_c)
    x1, x2 = x_reshaped[..., 0, :], x_reshaped[..., 1, :]

    inv_freq = 1.0 / (base_frequency**(
        jnp.arange(0, c_per_dim, 2, dtype=jnp.float32) / c_per_dim))
    freqs = jnp.expand_dims(positions[..., None] * inv_freq, axis=2)

    cos = jnp.cos(freqs).astype(inputs.dtype)
    sin = jnp.sin(freqs).astype(inputs.dtype)

    out1 = (x1 * cos) - (x2 * sin)
    out2 = (x2 * cos) + (x1 * sin)

    out_rotated = jnp.stack([out1, out2],
                            axis=-2).reshape(b, seq_len, num_heads, rot_dim)

    if x_unrot.shape[-1] > 0:
        return jnp.concatenate([out_rotated, x_unrot], axis=-1)

    return out_rotated


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
                 quant_config: Optional[VllmQuantConfig] = None,
                 prefix: str = ""):
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

        self.q_proj = JaxEinsum("BTD,DNH->BTNH",
                                (self.features, self.num_heads, self.head_dim),
                                param_dtype=dtype,
                                kernel_init=nnx.with_partitioning(
                                    init_fn,
                                    (None, ShardingAxisName.VIT_MODEL, None)),
                                rngs=rng,
                                quant_config=quant_config,
                                prefix=f"{prefix}.q_proj.linear")
        self.k_proj = JaxEinsum(
            "BTD,DKH->BTKH", (self.features, self.num_kv_heads, self.head_dim),
            param_dtype=dtype,
            kernel_init=nnx.with_partitioning(
                init_fn, (None, ShardingAxisName.VIT_MODEL, None)),
            rngs=rng,
            quant_config=quant_config,
            prefix=f"{prefix}.k_proj.linear")
        self.v_proj = JaxEinsum(
            "BTD,DKH->BTKH", (self.features, self.num_kv_heads, self.head_dim),
            param_dtype=dtype,
            kernel_init=nnx.with_partitioning(
                init_fn, (None, ShardingAxisName.VIT_MODEL, None)),
            rngs=rng,
            quant_config=quant_config,
            prefix=f"{prefix}.v_proj.linear")
        self.o_proj = JaxEinsum("BTNH,NHD->BTD",
                                (self.num_heads, self.head_dim, self.features),
                                param_dtype=dtype,
                                kernel_init=nnx.with_partitioning(
                                    init_fn,
                                    (ShardingAxisName.VIT_MODEL, None, None)),
                                rngs=rng,
                                quant_config=quant_config,
                                prefix=f"{prefix}.o_proj.linear")

        self.q_norm = JaxRmsNorm(self.head_dim,
                                 param_dtype=dtype,
                                 scale_init=nnx.with_partitioning(
                                     init_fn, (None, )),
                                 rngs=rng,
                                 quant_config=quant_config)
        self.k_norm = JaxRmsNorm(self.head_dim,
                                 param_dtype=dtype,
                                 scale_init=nnx.with_partitioning(
                                     init_fn, (None, )),
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
            query_proj, segment_pos, base_frequency=self.rope_base_frequency)
        key_proj = apply_multidimensional_rope(
            key_proj, segment_pos, base_frequency=self.rope_base_frequency)

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

        outputs_BNTH = sharded_flash_attention(
            mesh=self.mesh,
            causal=False,
            sm_scale=1.0,
            batch_axis=ShardingAxisName.VIT_BATCH,
            head_axis=ShardingAxisName.VIT_MODEL,
        )(q_BNTH, k_BKTH, v_BKTH, segment_ids)

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

    def __init__(self,
                 config,
                 dtype,
                 rngs: nnx.Rngs,
                 quant_config=None,
                 prefix: str = ""):
        self.config = config
        self.dtype = dtype

        self.patch_size = config.patch_size

        self.input_proj = JaxEinsum(
            "...d,dh->...h",
            (3 * self.patch_size**2, config.hidden_size),
            param_dtype=dtype,
            kernel_init=nnx.with_partitioning(
                init_fn, (None, ShardingAxisName.VIT_MODEL)),
            bias_init=None,
            rngs=rngs,
            quant_config=quant_config,
            prefix=f"{prefix}.input_proj",
        )

        self.position_embedding_table = nnx.Param(
            jax.random.normal(rngs.params(), (10240, 2, config.hidden_size),
                              dtype=dtype),
            # PyTorch (2, 10240, hidden) -> JAX (10240, 2, hidden)
            weight_loader=functools.partial(load_nnx_param_from_reshaped_torch,
                                            permute_dims=(1, 0, 2)))

    def _factorized_posemb(self, pixel_position_ids: jax.Array) -> jax.Array:
        posemb = self.position_embedding_table.get_value()
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
                 quant_config: Optional[VllmQuantConfig] = None,
                 prefix: str = ""):
        self.features = config.hidden_size
        self.hidden_dim = config.intermediate_size

        self.gate_proj = JaxEinsum(
            "...d,df->...f",
            (self.features, self.hidden_dim),
            param_dtype=dtype,
            kernel_init=nnx.with_partitioning(
                init_fn, (None, ShardingAxisName.VIT_MODEL)),
            bias_init=None,
            rngs=rng,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_proj.linear",
        )

        self.up_proj = JaxEinsum(
            "...d,df->...f",
            (self.features, self.hidden_dim),
            param_dtype=dtype,
            kernel_init=nnx.with_partitioning(
                init_fn, (None, ShardingAxisName.VIT_MODEL)),
            bias_init=None,
            rngs=rng,
            quant_config=quant_config,
            prefix=f"{prefix}.up_proj.linear",
        )

        self.down_proj = JaxEinsum(
            "...f,fd->...d",
            (self.hidden_dim, self.features),
            param_dtype=dtype,
            kernel_init=nnx.with_partitioning(
                init_fn, (ShardingAxisName.VIT_MODEL, None)),
            bias_init=None,
            prefix=f"{prefix}.down_proj.linear",
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
                 quant_config: Optional[VllmQuantConfig] = None,
                 prefix: str = ""):
        self.input_layernorm = JaxRmsNorm(config.hidden_size,
                                          param_dtype=dtype,
                                          scale_init=nnx.with_partitioning(
                                              init_fn, (None, )),
                                          rngs=rng,
                                          quant_config=quant_config)

        self.self_attn = Gemma4VisionFlashAttention(
            config,
            dtype,
            rng,
            mesh,
            quant_config,
            prefix=f"{prefix}.self_attn")

        self.post_attention_layernorm = JaxRmsNorm(
            config.hidden_size,
            param_dtype=dtype,
            scale_init=nnx.with_partitioning(init_fn, (None, )),
            rngs=rng,
            quant_config=quant_config)

        self.pre_feedforward_layernorm = JaxRmsNorm(
            config.hidden_size,
            param_dtype=dtype,
            scale_init=nnx.with_partitioning(init_fn, (None, )),
            rngs=rng,
            quant_config=quant_config)
        self.mlp = Gemma4VisionMLP(config,
                                   dtype,
                                   rng,
                                   quant_config,
                                   prefix=f"{prefix}.mlp")
        self.post_feedforward_layernorm = JaxRmsNorm(
            config.hidden_size,
            param_dtype=dtype,
            scale_init=nnx.with_partitioning(init_fn, (None, )),
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
                 quant_config: Optional[VllmQuantConfig] = None,
                 prefix: str = "vision_tower"):
        self.config = config

        self.dtype = dtype
        self.mesh = mesh

        self.patch_embedder = Gemma4VisionPatchEmbedder(
            config,
            dtype,
            rng,
            quant_config,
            prefix=f"{prefix}.patch_embedder")

        num_layers = getattr(config, "num_hidden_layers", 32)
        self.start_layer, self.end_layer, self.layers = make_layers(
            num_layers, lambda i: Gemma4VisionEncoderLayer(
                config,
                dtype,
                rng,
                self.mesh,
                quant_config,
                prefix=f"{prefix}.encoder.layers.{i}"))

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
            pooled_x = (pooled_x - self.std_bias.get_value()
                        ) * self.std_scale.get_value()
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
            kernel_init=nnx.with_partitioning(
                init_fn, (None, ShardingAxisName.VIT_MODEL)),
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


class Gemma4MmModel(JaxModule):
    """Gemma4 multimodal model combining text backbone and vision encoder.

    Mirrors the HF checkpoint layout:
      model.language_model.layers.*  — transformer text layers
      model.vision_tower.*           — SigLIP encoder
      model.embed_vision.*           — vision-text projection
    """

    def __init__(self,
                 vllm_config: VllmConfig,
                 rng: nnx.Rngs,
                 mesh: Mesh,
                 prefix: str = "model"):
        model_config = vllm_config.model_config
        vision_config = model_config.hf_config.vision_config

        self.language_model = Gemma4Model(
            vllm_config=vllm_config,
            rng=rng,
            mesh=mesh,
            prefix=prefix + ".language_model",
        )

        self.vision_tower = Gemma4VisionModel(
            config=vision_config,
            dtype=model_config.dtype,
            rng=rng,
            quant_config=vllm_config.quant_config,
            mesh=mesh,
            prefix=f"{prefix}.vision_tower",
        )

        self.embed_vision = Gemma4MultimodalEmbedder(
            vision_hidden_size=vision_config.hidden_size,
            text_hidden_size=model_config.hf_config.text_config.hidden_size,
            dtype=model_config.dtype,
            rng=rng,
            quant_config=vllm_config.quant_config,
            prefix=f"{prefix}.embed_vision",
        )


class Gemma4ForConditionalGeneration(JaxModule, LoadableWithIterator):
    packed_modules_mapping = Gemma4ForCausalLM.packed_modules_mapping
    WeightLoader = StandardWeightLoader
    supports_multimodal = True
    supports_encoder_tp_data = True
    supports_encoder_cudagraph = True
    _processor_factory = getattr(PtGemma4MM, "_processor_factory", None)

    def __init__(self, vllm_config: VllmConfig, rng_key: jax.Array,
                 mesh: Mesh) -> None:
        self.vllm_config = vllm_config
        rng = nnx.Rngs(rng_key)
        self.mesh = mesh

        vllm_config.sharding_config.apply_vision_sharding()

        self.model = Gemma4MmModel(
            vllm_config=vllm_config,
            rng=rng,
            mesh=mesh,
            prefix="model",
        )
        model_config = vllm_config.model_config
        vision_config = model_config.hf_config.vision_config
        self.image_token_id = getattr(model_config.hf_config, "image_token_id",
                                      258880)
        self.pooling_kernel_size = vision_config.pooling_kernel_size
        self.max_soft_tokens = vision_config.default_output_length
        self.patch_pixels = vision_config.patch_size**2 * 3

        self.final_logit_softcapping = getattr(
            model_config.hf_config.text_config, "final_logit_softcapping",
            None)

        if not model_config.hf_config.tie_word_embeddings:
            if self.model.language_model.is_last_rank:
                vocab_size = model_config.get_vocab_size()
                hidden_size = model_config.hf_config.text_config.hidden_size
                from tpu_inference.layers.jax.linear import JaxLmHead
                self.lm_head = JaxLmHead(
                    hidden_size=hidden_size,
                    vocab_size=vocab_size,
                    param_dtype=model_config.dtype,
                    kernel_init=nnx.with_partitioning(init_fn,
                                                      ("model", None)),
                    rngs=rng,
                    prefix="lm_head",
                )
            else:
                from tpu_inference.layers.jax.pp_utils import PPMissingLayer
                self.lm_head = PPMissingLayer()

    def load_weights(self, weights: Iterable[Tuple[str, Any]]):
        # Remap checkpoint names to Python attr paths.  self.model makes
        # "model.*" resolve naturally (has_model_child=True), so no prefix
        # stripping is needed for the text/vision/embed weights.
        # vision_tower.encoder.* is a checkpoint-only sub-level; .linear is
        # a checkpoint-only wrapper on vision attention projections.
        # model.lm_head lives at the top level in the JAX model.
        mapper = WeightsMapper(
            orig_to_new_prefix={
                "model.vision_tower.encoder.": "model.vision_tower.",
                "model.lm_head.": "lm_head.",
            },
            orig_to_new_suffix={".linear.weight": ".weight"},
        )
        loader = JaxAutoWeightsLoader(
            self,
            skip_prefixes=(["lm_head"]
                           if not hasattr(self, 'lm_head') else []),
            skip_substrs=[
                "audio_tower",
                "embed_audio",
                ".input_max",
                ".input_min",
                ".output_max",
                ".output_min",
            ],
        )
        return loader.load_weights(mapper.apply(weights))

    def embed_input_ids(self,
                        input_ids: jax.Array,
                        multimodal_embeddings: Optional[jax.Array] = None,
                        **kwargs) -> jax.Array:
        inputs_embeds = self.model.language_model.embed_tokens(input_ids)
        target_dtype = inputs_embeds.dtype

        inputs_embeds = (
            inputs_embeds *
            self.model.language_model.embedding_scale).astype(target_dtype)

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

        vision_outputs = self.model.vision_tower(
            pixel_values,
            input_mask=input_mask,
            pixel_position_ids=pixel_position_ids)

        projected_vision_features = vision_outputs[0][0]
        pooler_mask = vision_outputs[0][1]

        projected_vision_features = self.model.embed_vision(
            projected_vision_features)

        seq_len = pooler_mask.shape[1]
        indices = jnp.arange(seq_len)
        sort_key = jnp.where(pooler_mask, indices, seq_len + indices)
        sort_idx = jnp.argsort(sort_key, axis=1)
        projected_vision_features = jnp.take_along_axis(
            projected_vision_features,
            jnp.expand_dims(sort_idx, axis=-1),
            axis=1)

        projected_vision_features = jax.lax.with_sharding_constraint(
            projected_vision_features,
            NamedSharding(self.mesh, PartitionSpec(None, None, None)))

        return projected_vision_features

    def _parse_and_validate_image_input(
            self, **kwargs: object) -> Optional[Gemma4ImagePixelInputs]:
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

    def _process_image_input(
            self, image_input: Gemma4ImagePixelInputs) -> list[jax.Array]:
        pixel_values = image_input["pixel_values"]
        pixel_position_ids = image_input["pixel_position_ids"]

        if pixel_values.ndim == 2:
            pixel_values = jnp.expand_dims(pixel_values, axis=0)
        if pixel_position_ids.ndim == 2:
            pixel_position_ids = jnp.expand_dims(pixel_position_ids, axis=0)

        # Process images in fixed-size micro-batches of exactly `dp_size` so
        # every vision-tower call shares the same (B=dp_size, ...) JIT cache
        # entry regardless of how many images this step has. The last
        # micro-batch is right-padded with pixel_position_ids=POSITIONS_PAD_VALUE
        # entries so input_mask masks the padded entries out of the encoder.
        n_images = pixel_values.shape[0]
        dp_size = get_mesh_shape_product(self.mesh, ShardingAxisName.VIT_BATCH)
        input_sharding = NamedSharding(
            self.mesh, PartitionSpec(ShardingAxisName.VIT_BATCH, None, None))

        per_image_features = []
        for chunk_start in range(0, n_images, dp_size):
            chunk_end = min(chunk_start + dp_size, n_images)
            chunk_size = chunk_end - chunk_start
            pv_chunk = pixel_values[chunk_start:chunk_end]
            pp_chunk = pixel_position_ids[chunk_start:chunk_end]
            if chunk_size < dp_size:
                pad_count = dp_size - chunk_size
                pv_chunk = jnp.pad(pv_chunk, ((0, pad_count), (0, 0), (0, 0)))
                pp_chunk = jnp.pad(pp_chunk, ((0, pad_count), (0, 0), (0, 0)),
                                   constant_values=POSITIONS_PAD_VALUE)
            pv_chunk = jax.device_put(pv_chunk, input_sharding)
            pp_chunk = jax.device_put(pp_chunk, input_sharding)
            vt_output = self.get_single_image_embedding(pv_chunk, pp_chunk)
            for i in range(chunk_size):
                per_image_features.append(vt_output[i])
        return per_image_features

    def embed_multimodal(self, **kwargs) -> List[jax.Array]:
        image_input = self._parse_and_validate_image_input(**kwargs)
        if image_input is None:
            return []

        return self._process_image_input(image_input)

    # -- SupportsEncoderCudaGraph protocol methods --

    def get_encoder_cudagraph_config(self):
        from vllm.v1.worker.encoder_cudagraph_defs import \
            EncoderCudaGraphConfig

        def pad_pixel_position_ids(dst: torch.Tensor,
                                   src: torch.Tensor) -> None:
            # pixel_position_ids uses -1 as the pad sentinel (POSITIONS_PAD_VALUE).
            dst.fill_(POSITIONS_PAD_VALUE)
            dst[:src.shape[0]].copy_(src)

        text_hidden = self.vllm_config.model_config.hf_config.text_config.hidden_size
        config = EncoderCudaGraphConfig(
            modalities=["image"],
            buffer_keys=["pixel_values", "pixel_position_ids"],
            out_hidden_size=text_hidden,
            max_frames_per_video=1,
            padding_logics={"pixel_position_ids": pad_pixel_position_ids},
        )
        return config

    def get_max_frames_per_video(self) -> int:
        raise NotImplementedError("Video not yet supported.")

    def get_input_modality(self, mm_kwargs: dict[str, Any]) -> str:
        if "pixel_values_videos" in mm_kwargs:
            raise NotImplementedError("Video not yet supported.")
        else:
            return "image"

    def get_encoder_cudagraph_budget_range(self, vllm_config):
        min_budget = self.max_soft_tokens
        max_num_seqs = vllm_config.scheduler_config.max_num_seqs
        text_dp_size = vllm_config.sharding_config.total_dp_size
        max_budget = min(
            self.max_soft_tokens * max_num_seqs * text_dp_size,
            vllm_config.scheduler_config.max_num_batched_tokens * text_dp_size,
        )
        return min_budget, max_budget

    def _get_pixel_values_by_modality(
            self, mm_kwargs: dict[str, Any]) -> torch.Tensor:
        if self.get_input_modality(mm_kwargs) == "image":
            return mm_kwargs["pixel_values"]
        raise NotImplementedError("Video not yet supported.")

    def _get_pixel_position_ids_by_modality(
            self, mm_kwargs: dict[str, Any]) -> torch.Tensor:
        if self.get_input_modality(mm_kwargs) == "image":
            return mm_kwargs["pixel_position_ids"]
        raise NotImplementedError("Video not yet supported.")

    def get_encoder_cudagraph_item_specs(self, mm_kwargs):
        from vllm.v1.worker.encoder_cudagraph_defs import EncoderItemSpec
        pixel_position_ids = self._get_pixel_position_ids_by_modality(
            mm_kwargs)
        if not isinstance(pixel_position_ids, torch.Tensor):
            return []
        k = self.pooling_kernel_size
        specs = []
        for i in range(pixel_position_ids.shape[0]):
            pid = pixel_position_ids[i]  # (num_patches, 2)
            valid_patches = int((pid[:, 0]
                                 != POSITIONS_PAD_VALUE).sum().item())
            output_tokens = min(max(valid_patches // k**2, 1),
                                self.max_soft_tokens)
            specs.append(
                EncoderItemSpec(
                    input_size=pixel_position_ids.shape[1],
                    output_tokens=output_tokens,
                ))
        return specs

    def select_encoder_cudagraph_items(
        self,
        mm_kwargs: dict[str, Any],
        indices: list[int],
    ) -> dict[str, Any]:
        modality = self.get_input_modality(mm_kwargs)
        pixel_values = self._get_pixel_values_by_modality(mm_kwargs)
        pixel_position_ids = self._get_pixel_position_ids_by_modality(
            mm_kwargs)

        if len(indices) == 0:
            if modality == "image":
                return {
                    "pixel_values": pixel_values[:0],
                    "pixel_position_ids": pixel_position_ids[:0],
                }
            raise NotImplementedError("Video not yet supported.")

        if modality == "image":
            return {
                "pixel_values": pixel_values[list(indices)],
                "pixel_position_ids": pixel_position_ids[list(indices)],
            }
        raise NotImplementedError("Video not yet supported.")

    def prepare_encoder_cudagraph_capture_inputs(
        self,
        token_budget: int,
        max_batch_size: int,
        max_frames_per_batch: int,
        device: torch.device,
        dtype: torch.dtype,
    ):
        from vllm.v1.worker.encoder_cudagraph_defs import \
            EncoderCudaGraphCaptureInputs
        num_patches = self.max_soft_tokens * self.pooling_kernel_size**2
        dp_size = get_mesh_shape_product(self.mesh, ShardingAxisName.VIT_BATCH)
        images_for_budget = max(
            min(token_budget // self.max_soft_tokens, max_batch_size), 1)
        batch_size = ((images_for_budget + dp_size - 1) // dp_size) * dp_size

        pixel_values = torch.randn(batch_size,
                                   num_patches,
                                   self.patch_pixels,
                                   dtype=dtype,
                                   device=device)
        pixel_position_ids = torch.full(
            (batch_size, num_patches, 2),
            POSITIONS_PAD_VALUE,
            dtype=torch.int32,
            device=device,
        )
        return EncoderCudaGraphCaptureInputs(
            values={
                "pixel_values": pixel_values,
                "pixel_position_ids": pixel_position_ids,
            })

    def prepare_encoder_cudagraph_replay_buffers(
        self,
        mm_kwargs,
        max_batch_size: int,
        max_frames_per_batch: int,
        path: str = "default",
    ):
        from vllm.v1.worker.encoder_cudagraph_defs import \
            EncoderCudaGraphReplayBuffers
        pv = self._get_pixel_values_by_modality(mm_kwargs)
        ppid = self._get_pixel_position_ids_by_modality(mm_kwargs)
        return EncoderCudaGraphReplayBuffers(values={
            "pixel_values": pv,
            "pixel_position_ids": ppid,
        })

    @jax.jit
    def encoder_cudagraph_forward(self, inputs: dict) -> jax.Array:
        """Run the vision encoder on fixed-shape inputs.

        Args:
            inputs: dict with "pixel_values" (B, NP, PP) and
                "pixel_position_ids" (B, NP, 2) as jax.Arrays.

        Returns:
            jax.Array of shape (B, max_soft_tokens, hidden).
        """
        pixel_values = inputs["pixel_values"]
        pixel_position_ids = inputs["pixel_position_ids"]

        input_sharding = NamedSharding(
            self.mesh, PartitionSpec(ShardingAxisName.VIT_BATCH, None, None))
        pixel_values = jax.device_put(pixel_values, input_sharding)
        pixel_position_ids = jax.device_put(pixel_position_ids, input_sharding)

        input_mask = pixel_position_ids[..., 0] != POSITIONS_PAD_VALUE

        vision_outputs = self.model.vision_tower(
            pixel_values,
            input_mask=input_mask,
            pixel_position_ids=pixel_position_ids,
        )
        projected = vision_outputs[0][0]
        pooler_mask = vision_outputs[0][1]
        projected = self.model.embed_vision(projected)

        seq_len = pooler_mask.shape[1]
        sort_indices = jnp.arange(seq_len)
        sort_key = jnp.where(pooler_mask, sort_indices, seq_len + sort_indices)
        sort_idx = jnp.argsort(sort_key, axis=1)
        projected = jnp.take_along_axis(projected,
                                        jnp.expand_dims(sort_idx, axis=-1),
                                        axis=1)
        projected = jax.lax.with_sharding_constraint(
            projected, NamedSharding(self.mesh,
                                     PartitionSpec(None, None, None)))
        return projected

    def encoder_eager_forward(self, mm_kwargs: dict[str,
                                                    Any]) -> list[jax.Array]:
        """Fallback for inputs that exceed all budget sizes."""
        return self.embed_multimodal(**mm_kwargs)

    def postprocess_encoder_output(
        self,
        output,
        indices,
        per_item_out_tokens,
        dest,
        clone: bool = False,
        batch_mm_kwargs=None,
    ) -> None:
        """Split batch encoder output into per-image entries in dest."""
        for local_idx, global_idx in enumerate(indices):
            n = per_item_out_tokens[global_idx]
            feat = output[local_idx, :n]
            if clone and hasattr(feat, 'clone'):
                feat = feat.clone()
            dest[global_idx] = feat

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

        # PLE multimodal mask: mark image-token positions
        # so embed_tokens_per_layer redirects them to slot 0. None when not
        # PLE-active or first rank where input_ids isn't reliable. Cheap to
        # always compute; the PLE compute path is the only consumer.
        is_multimodal = (input_ids == self.image_token_id
                         ) if input_ids is not None else None

        kv_caches, x, expert_indices = self.model.language_model(
            kv_caches,
            input_ids,
            attention_metadata,
            inputs_embeds,
            layer_name_to_kv_cache=layer_name_to_kv_cache,
            is_multimodal=is_multimodal,
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
            logits = self.model.language_model.embed_tokens.decode(
                hidden_states)

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

            p = self.model.vision_tower.patch_embedder.patch_size
            h_p, w_p = h_input // p, w_input // p
            dummy_pixel_position_ids = jnp.ones((1, h_p * w_p, 2),
                                                dtype=jnp.int32)

            run_compilation_fn("vision_encoder",
                               self.get_single_image_embedding,
                               dummy_pixel_values,
                               dummy_pixel_position_ids,
                               image_shape=input_hw)
