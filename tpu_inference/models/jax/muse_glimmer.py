# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""JAX text decoder for Muse Glimmer.

Muse Glimmer is a Gemma-family model, but its text stack is not numerically
interchangeable with Gemma4.  This module keeps the TPU implementation small
while spelling out the Muse-specific normalization, attention, and logits
semantics that affect model correctness.
"""

from itertools import islice
from typing import Any, Iterable, List, Optional, Tuple

import jax
import jax.numpy as jnp
from flax import nnx
from jax.sharding import Mesh
from vllm.config import VllmConfig

from tpu_inference import utils
from tpu_inference.distributed.jax_parallel_state import get_pp_group
from tpu_inference.layers.common.attention_interface import attention
from tpu_inference.layers.common.attention_metadata import AttentionMetadata
from tpu_inference.layers.common.quantization import quantize_kv
from tpu_inference.layers.jax import JaxModule
from tpu_inference.layers.jax.embed import JaxEmbed
from tpu_inference.layers.jax.linear import (JaxEinsum, JaxLinear, JaxLmHead,
                                             JaxMergedColumnParallelLinear,
                                             JaxQKVParallelLinear)
from tpu_inference.layers.jax.norm import JaxRmsNorm
from tpu_inference.layers.jax.pp_utils import PPMissingLayer, make_layers
from tpu_inference.layers.jax.rope_interface import (apply_rope,
                                                     get_rope_scaling,
                                                     get_rope_theta)
from tpu_inference.layers.vllm.quantization.configs import VllmQuantConfig
from tpu_inference.models.jax.jax_intermediate_tensor import \
    JaxIntermediateTensors
from tpu_inference.models.jax.utils.multi_modal_utils import \
    merge_multimodal_embeddings
from tpu_inference.models.jax.utils.weight_utils import (LoadableWithIterator,
                                                         StandardWeightLoader)

init_fn = nnx.initializers.uniform()


def _text_config(config: Any) -> Any:
    return getattr(config, "text_config", config)


def muse_glimmer_query_prescale(config: Any) -> float:
    """Normalize native and modular Muse query-scale config conventions."""
    explicit = getattr(config, "scale_query_by", None)
    if explicit is not None:
        return float(explicit)

    qk_scale = getattr(config, "qk_scale_factor", None)
    if qk_scale is None:
        return 1.0

    qk_scale = float(qk_scale)
    sqrt_head_dim = float(config.head_dim)**0.5
    if qk_scale >= sqrt_head_dim:
        return qk_scale / sqrt_head_dim
    return qk_scale


def muse_glimmer_layer_uses_rope(config: Any, layer_idx: int) -> bool:
    no_rope_layers = getattr(config, "no_rope_layers", None)
    if no_rope_layers is not None:
        return bool(no_rope_layers[layer_idx])
    layer_types = getattr(config, "layer_types", None)
    if layer_types is not None:
        return layer_types[layer_idx] == "sliding_attention"
    return True


def muse_glimmer_dflash_target_layer_ids(
        vllm_config: VllmConfig) -> Tuple[int, ...]:
    speculative_config = vllm_config.speculative_config
    if (speculative_config is None or speculative_config.method != "dflash"):
        return ()
    draft_config = speculative_config.draft_model_config.hf_config
    dflash_config = getattr(draft_config, "dflash_config", None) or {}
    target_layer_ids = dflash_config.get(
        "target_layer_ids", getattr(draft_config, "target_layer_ids", ()))
    return tuple(int(layer_id) for layer_id in target_layer_ids)


class MuseGlimmerRMSNorm(JaxRmsNorm):
    """Muse RMSNorm: fp32 math and an optional baked weight offset."""

    def __init__(self,
                 dim: Optional[int],
                 epsilon: float,
                 dtype: jnp.dtype,
                 rngs: nnx.Rngs,
                 *,
                 use_scale: bool = True,
                 weight_offset: float = 0.0,
                 prefix: str = "") -> None:
        super().__init__(
            dim,
            epsilon=epsilon,
            use_scale=use_scale,
            param_dtype=dtype,
            scale_init=nnx.with_partitioning(nnx.initializers.zeros,
                                             (None, )) if use_scale else None,
            rngs=rngs,
            quant_config=None,
            prefix=prefix,
        )
        self.use_scale = use_scale
        self.weight_offset = weight_offset

    def __call__(self, x: jax.Array) -> jax.Array:
        input_dtype = x.dtype
        x_f32 = x.astype(jnp.float32)
        output = x_f32 * jax.lax.rsqrt(
            jnp.mean(jnp.square(x_f32), axis=-1, keepdims=True) + self.epsilon)
        if self.use_scale:
            output *= self.weight.get_value().astype(
                jnp.float32) + self.weight_offset
        return output.astype(input_dtype)


class MuseGlimmerMLP(JaxModule):

    def __init__(self,
                 config: Any,
                 dtype: jnp.dtype,
                 rngs: nnx.Rngs,
                 quant_config: VllmQuantConfig,
                 prefix: str = "") -> None:
        if getattr(config, "hidden_activation",
                   getattr(config, "hidden_act", "silu")) != "silu":
            raise ValueError("Muse Glimmer requires the `silu` activation")
        self.gate_up_proj = JaxMergedColumnParallelLinear(
            config.hidden_size,
            [config.intermediate_size] * 2,
            use_bias=False,
            param_dtype=dtype,
            kernel_init=nnx.with_partitioning(init_fn, (None, "model")),
            rngs=rngs,
            quant_config=quant_config,
            prefix=prefix + ".gate_up_proj",
        )
        self.down_proj = JaxLinear(
            config.intermediate_size,
            config.hidden_size,
            use_bias=False,
            param_dtype=dtype,
            kernel_init=nnx.with_partitioning(init_fn, ("model", None)),
            rngs=rngs,
            quant_config=quant_config,
            prefix=prefix + ".down_proj",
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        gate, up = jnp.split(self.gate_up_proj(x), 2, axis=-1)
        return self.down_proj(jax.nn.silu(gate) * up)


class MuseGlimmerAttention(JaxModule):

    def __init__(self,
                 config: Any,
                 layer_idx: int,
                 dtype: jnp.dtype,
                 rngs: nnx.Rngs,
                 mesh: Mesh,
                 kv_cache_dtype: str,
                 quant_config: VllmQuantConfig,
                 prefix: str = "") -> None:
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.mesh = mesh
        self.scaling = self.head_dim**-0.5
        self.use_rope = muse_glimmer_layer_uses_rope(config, layer_idx)
        self.sliding_window = (getattr(config, "sliding_window", 2048)
                               if self.use_rope else None)
        self.rope_theta = get_rope_theta(config, 500000.0)
        self.rope_scaling = get_rope_scaling(config)
        self.scale_query_by = muse_glimmer_query_prescale(config)

        self.qkv_proj = JaxQKVParallelLinear(
            hidden_size=self.hidden_size,
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            head_dim=self.head_dim,
            use_bias=False,
            dtype=dtype,
            rngs=rngs,
            quant_config=quant_config,
            prefix=prefix + ".qkv_proj",
        )
        self.qk_norm = MuseGlimmerRMSNorm(
            self.head_dim,
            epsilon=config.rms_norm_eps,
            dtype=dtype,
            rngs=rngs,
            use_scale=False,
            prefix=prefix + ".qk_norm",
        )
        self.output_gate_proj = JaxLinear(
            self.hidden_size,
            self.num_heads * self.head_dim,
            use_bias=False,
            param_dtype=dtype,
            kernel_init=nnx.with_partitioning(init_fn, (None, "model")),
            rngs=rngs,
            quant_config=quant_config,
            prefix=prefix + ".output_gate_proj",
        )
        self.o_proj = JaxEinsum(
            "TNH,NHD->TD",
            (self.num_heads, self.head_dim, self.hidden_size),
            param_dtype=dtype,
            kernel_init=nnx.with_partitioning(init_fn, ("model", None, None)),
            rngs=rngs,
            quant_config=quant_config,
            prefix=prefix + ".o_proj",
        )

        self._q_scale = 1.0
        self._k_scale = 1.0
        self._v_scale = 1.0
        self.kv_cache_quantized_dtype = None
        if kv_cache_dtype != "auto":
            self.kv_cache_quantized_dtype = utils.get_jax_dtype_from_str_dtype(
                kv_cache_dtype)

    def __call__(
        self,
        kv_cache: Optional[jax.Array],
        x: jax.Array,
        attention_metadata: AttentionMetadata,
    ) -> Tuple[jax.Array, jax.Array]:
        q, k, v = self.qkv_proj(x)
        q = self.qk_norm(q) * self.scale_query_by
        k = self.qk_norm(k)

        if self.use_rope:
            q = apply_rope(q, attention_metadata.input_positions,
                           self.head_dim, self.rope_theta, self.rope_scaling)
            k = apply_rope(k, attention_metadata.input_positions,
                           self.head_dim, self.rope_theta, self.rope_scaling)

        q_scale = k_scale = v_scale = None
        if self.kv_cache_quantized_dtype:
            k_scale = self._k_scale
            v_scale = self._v_scale
            k, v = quantize_kv(self.kv_cache_quantized_dtype, k, v, k_scale,
                               v_scale)

        new_kv_cache, output = attention(
            kv_cache,
            q,
            k,
            v,
            attention_metadata,
            self.mesh,
            self.head_dim,
            sm_scale=self.scaling,
            attention_chunk_size=self.sliding_window,
            q_scale=q_scale,
            k_scale=k_scale,
            v_scale=v_scale,
        )
        gate = self.output_gate_proj(x).reshape(output.shape)
        output = jax.nn.sigmoid(gate) * output
        return new_kv_cache, self.o_proj(output)


class MuseGlimmerDecoderLayer(JaxModule):

    def __init__(self,
                 config: Any,
                 layer_idx: int,
                 dtype: jnp.dtype,
                 rngs: nnx.Rngs,
                 mesh: Mesh,
                 kv_cache_dtype: str,
                 quant_config: VllmQuantConfig,
                 prefix: str = "") -> None:
        hidden_size = config.hidden_size
        self.input_layernorm = MuseGlimmerRMSNorm(
            hidden_size,
            config.rms_norm_eps,
            dtype,
            rngs,
            weight_offset=1.0,
            prefix=prefix + ".input_layernorm",
        )
        self.self_attn = MuseGlimmerAttention(
            config,
            layer_idx,
            dtype,
            rngs,
            mesh,
            kv_cache_dtype,
            quant_config,
            prefix=prefix + ".self_attn",
        )
        self.post_attention_layernorm = MuseGlimmerRMSNorm(
            hidden_size,
            config.post_norm_eps,
            dtype,
            rngs,
            weight_offset=1.0,
            prefix=prefix + ".post_attention_layernorm",
        )
        self.pre_feedforward_layernorm = MuseGlimmerRMSNorm(
            hidden_size,
            config.rms_norm_eps,
            dtype,
            rngs,
            weight_offset=1.0,
            prefix=prefix + ".pre_feedforward_layernorm",
        )
        self.mlp = MuseGlimmerMLP(config, dtype, rngs, quant_config,
                                  prefix + ".mlp")
        self.post_feedforward_layernorm = MuseGlimmerRMSNorm(
            hidden_size,
            config.post_norm_eps,
            dtype,
            rngs,
            weight_offset=1.0,
            prefix=prefix + ".post_feedforward_layernorm",
        )

    def __call__(
        self,
        kv_cache: jax.Array,
        x: jax.Array,
        attention_metadata: AttentionMetadata,
    ) -> Tuple[jax.Array, jax.Array, None]:
        residual = x
        x = self.input_layernorm(x)
        kv_cache, x = self.self_attn(kv_cache, x, attention_metadata)
        x = residual + self.post_attention_layernorm(x)

        residual = x
        x = self.pre_feedforward_layernorm(x)
        x = self.mlp(x)
        x = residual + self.post_feedforward_layernorm(x)
        return kv_cache, x, None


class MuseGlimmerModel(JaxModule):

    def __init__(self,
                 vllm_config: VllmConfig,
                 rngs: nnx.Rngs,
                 mesh: Mesh,
                 prefix: str = "model") -> None:
        model_config = vllm_config.model_config
        hf_config = model_config.hf_config
        config = _text_config(hf_config)
        self.multimodal_token_ids = [
            getattr(hf_config, "image_token_id",
                    getattr(hf_config, "patch_token_id", 200092)),
            getattr(hf_config, "video_token_id", 200091),
        ]
        dtype = model_config.dtype
        self.is_first_rank = get_pp_group().is_first_rank
        self.is_last_rank = get_pp_group().is_last_rank

        if self.is_first_rank:
            self.embed_tokens = JaxEmbed(
                num_embeddings=config.vocab_size,
                features=config.hidden_size,
                param_dtype=dtype,
                embedding_init=nnx.with_partitioning(init_fn, ("model", None)),
                rngs=rngs,
                quant_config=vllm_config.quant_config,
                prefix=prefix + ".embed_tokens",
            )
            self.embed_norm = MuseGlimmerRMSNorm(
                config.hidden_size,
                config.rms_norm_eps,
                dtype,
                rngs,
                use_scale=False,
                prefix=prefix + ".embed_norm",
            )
        else:
            self.embed_tokens = PPMissingLayer()
            self.embed_norm = PPMissingLayer()

        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda layer_idx: MuseGlimmerDecoderLayer(
                config,
                layer_idx,
                dtype,
                rngs,
                mesh,
                vllm_config.cache_config.cache_dtype,
                vllm_config.quant_config,
                prefix=f"{prefix}.layers.{layer_idx}",
            ),
        )
        self.aux_hidden_state_layers = muse_glimmer_dflash_target_layer_ids(
            vllm_config)
        if self.is_last_rank:
            self.norm = MuseGlimmerRMSNorm(
                config.hidden_size,
                config.rms_norm_eps,
                dtype,
                rngs,
                prefix=prefix + ".norm",
            )
        else:
            self.norm = PPMissingLayer()

    def __call__(
        self,
        kv_caches: List[jax.Array],
        input_ids: Optional[jax.Array],
        attention_metadata: AttentionMetadata,
        inputs_embeds: Optional[jax.Array] = None,
    ) -> Tuple[List[jax.Array], jax.Array, List[jax.Array]]:
        if inputs_embeds is None:
            x = self.embed_norm(self.embed_tokens(input_ids))
        else:
            x = inputs_embeds

        aux_hidden_states = []
        for layer_idx, layer in enumerate(islice(self.layers, self.start_layer,
                                                 self.end_layer),
                                          start=self.start_layer):
            layer_metadata = (attention_metadata[f"layer.{layer_idx}"]
                              if isinstance(attention_metadata, dict) else
                              attention_metadata)
            kv_caches[layer_idx], x, _ = layer(kv_caches[layer_idx], x,
                                               layer_metadata)
            if layer_idx in self.aux_hidden_state_layers:
                aux_hidden_states.append(x)
        return kv_caches, self.norm(x), aux_hidden_states


def map_muse_glimmer_weight_name(name: str) -> Optional[str]:
    """Map canonical and legacy Muse checkpoint names to the JAX tree."""
    if any(part in name for part in ("vision_tower", "vision_encoder",
                                     "vision_adapter", "vision_projection",
                                     "perception_emb_norm", "rotary_emb")):
        return None

    is_legacy = name.startswith("model.layers.")
    if is_legacy:
        name = name.replace(".post_attention_layernorm.",
                            ".pre_feedforward_layernorm.")
        name = name.replace(".post_attn_norm.", ".post_attention_layernorm.")
        name = name.replace(".post_ffn_norm.", ".post_feedforward_layernorm.")

    for old_prefix in ("model.language_model.", "language_model."):
        if name.startswith(old_prefix):
            name = "model." + name[len(old_prefix):]
            break
    return name.replace(".self_attn.gate_proj.",
                        ".self_attn.output_gate_proj.")


class MuseGlimmerForConditionalGeneration(JaxModule, LoadableWithIterator):
    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }
    WeightLoader = StandardWeightLoader

    def __init__(self, vllm_config: VllmConfig, rng_key: jax.Array,
                 mesh: Mesh) -> None:
        self.vllm_config = vllm_config
        rngs = nnx.Rngs(rng_key)
        self.mesh = mesh
        self.model = MuseGlimmerModel(vllm_config, rngs, mesh)

        model_config = vllm_config.model_config
        config = _text_config(model_config.hf_config)
        self.output_multiplier = float(
            getattr(config, "output_multiplier", 1.0))
        self.final_logit_softcapping = getattr(config,
                                               "final_logit_softcapping", None)
        if self.model.is_last_rank:
            self.lm_head = JaxLmHead(
                hidden_size=config.hidden_size,
                vocab_size=config.vocab_size,
                dtype=model_config.dtype,
                rngs=rngs,
                prefix="lm_head",
            )
        else:
            self.lm_head = PPMissingLayer()

    def load_weights(self, weights: Iterable[Tuple[str, Any]]) -> set[str]:

        def mapped_weights():
            for name, tensor in weights:
                mapped_name = map_muse_glimmer_weight_name(name)
                if mapped_name is not None:
                    yield mapped_name, tensor

        return super().load_weights(mapped_weights())

    def __call__(
        self,
        kv_caches: List[jax.Array],
        input_ids: jax.Array,
        attention_metadata: AttentionMetadata,
        inputs_embeds: Optional[jax.Array] = None,
        _input_positions=None,
        _layer_name_to_kv_cache=None,
        _lora_metadata=None,
        intermediate_tensors: Optional[JaxIntermediateTensors] = None,
        is_first_rank: bool = True,
        is_last_rank: bool = True,
        *args,
    ) -> Tuple[List[jax.Array], jax.Array | JaxIntermediateTensors,
               List[jax.Array], None]:
        if not is_first_rank:
            assert intermediate_tensors is not None
            inputs_embeds = intermediate_tensors["hidden_states"]

        kv_caches, x, aux_hidden_states = self.model(kv_caches, input_ids,
                                                     attention_metadata,
                                                     inputs_embeds)
        if not is_last_rank:
            x = JaxIntermediateTensors(tensors={"hidden_states": x})
        return kv_caches, x, aux_hidden_states, None

    def embed_input_ids(
        self,
        input_ids: jax.Array,
        multimodal_embeddings: Optional[jax.Array] = None,
        *,
        is_multimodal: Optional[jax.Array] = None,
    ) -> jax.Array:
        del is_multimodal
        if not self.model.is_first_rank:
            return None
        inputs_embeds = self.model.embed_norm(
            self.model.embed_tokens(input_ids))
        if (multimodal_embeddings is not None
                and multimodal_embeddings.shape[0] != 0):
            inputs_embeds = merge_multimodal_embeddings(
                input_ids, inputs_embeds, multimodal_embeddings,
                self.model.multimodal_token_ids)
        return inputs_embeds

    def compute_logits(self, hidden_states: jax.Array) -> jax.Array:
        logits = self.lm_head(hidden_states) * self.output_multiplier
        if self.final_logit_softcapping is not None:
            cap = self.final_logit_softcapping
            logits = jnp.tanh(logits / cap) * cap
        return logits


MuseGlimmerForCausalLM = MuseGlimmerForConditionalGeneration
