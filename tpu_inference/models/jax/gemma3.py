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
from typing import List, Optional, Tuple

import jax
import jax.numpy as jnp
from flax import nnx
from jax.sharding import Mesh
from transformers.models.gemma3.configuration_gemma3 import Gemma3Config
from vllm.config import VllmConfig

from tpu_inference import utils
from tpu_inference.distributed.jax_parallel_state import get_pp_group
from tpu_inference.layers.common.attention_interface import attention
from tpu_inference.layers.common.attention_metadata import AttentionMetadata
from tpu_inference.layers.common.quantization import quantize_kv
from tpu_inference.layers.jax import JaxModule
from tpu_inference.layers.jax.embed import JaxEmbed
from tpu_inference.layers.jax.linear import JaxEinsum, JaxLinear
from tpu_inference.layers.jax.pp_utils import PPMissingLayer, make_layers
from tpu_inference.layers.jax.rope_interface import apply_rope
from tpu_inference.layers.vllm.quantization.configs import VllmQuantConfig
from tpu_inference.logger import init_logger
from tpu_inference.layers.jax.norm import JaxRmsNorm
from tpu_inference.models.jax.jax_intermediate_tensor import \
    JaxIntermediateTensors
from tpu_inference.models.jax.utils.weight_utils import StandardWeightLoader

logger = init_logger(__name__)

init_fn = nnx.initializers.uniform()

class Gemma3ForCausalLM(JaxModule):
    """JAX implementation of the Gemma 3 model for causal language modeling."""
    WeightLoader = StandardWeightLoader

    def __init__(self, vllm_config: VllmConfig, rng_key: jax.Array,
                 mesh: Mesh) -> None:
        self.vllm_config = vllm_config
        
        self.rng = nnx.Rngs(rng_key)
        self.mesh = mesh

        self.model = Gemma3Model(
            vllm_config=vllm_config,
            rng=self.rng,
            mesh=mesh,
            prefix="model",
        )
        model_config = vllm_config.model_config
        
        hf_config = model_config.hf_config
        # only load text_config as currently this code only support Text context
        if hasattr(hf_config, "text_config"):
            hf_config = hf_config.text_config
        
        # Gemma3 natively enables tie_word_embedding for memory effieiency. 
        # consider supporting tie_word_embedding=false where user is required to pass
        # in their own lm_head.weight as the output matrix. 
        if not model_config.hf_config.tie_word_embeddings:
            raise ValueError("Untied word embeddings (tie_word_embeddings=False) are not supported for Gemma 3 in this implementation.")

    def __call__(
        self,
        kv_caches: List[jax.Array],
        input_ids: Optional[jax.Array],
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
               List[jax.Array]]:

        kv_caches, hidden_states, residual = self.model(
            kv_caches,
            input_ids,
            attention_metadata,
            inputs_embeds,
            intermediate_tensors=intermediate_tensors,
        )
        
        # Prepare intermediate tensors for the next pipeline stage
        if not is_last_rank:
            res = JaxIntermediateTensors(
                tensors={
                    "hidden_states": hidden_states,
                    "residual": residual
                })
            return kv_caches, res, []
            
        return kv_caches, hidden_states, []

    def embed_multimodal(self, *args, **kwargs) -> tuple:
        # Placeholder for future Gemma 3 vision tower support
        return ()

    def embed_input_ids(
            self, input_ids: jax.Array,
            multimodal_embeddings: Optional[jax.Array] = None) -> Optional[jax.Array]:
        if not self.model.is_first_rank:
            return None
                        
        inputs_embeds = self.model.embed_tokens(input_ids) * self.model.normalizer

        return inputs_embeds

    def compute_logits(self, hidden_states: jax.Array) -> jax.Array:
        if hasattr(self, 'lm_head'):
            logits = self.lm_head(hidden_states)
        else:
            logits = self.model.embed_tokens.decode(hidden_states)
        return logits

    def load_weights(self, rng_key: jax.Array) -> None:
        self.rng = nnx.Rngs(rng_key)
        
        self.pp_missing_layers = []
        for path, module in nnx.iter_graph(self):
            if isinstance(module, PPMissingLayer):
                layer_name = ".".join([str(s) for s in path])
                self.pp_missing_layers.append(layer_name)

        # Skip vision and projector weights since this is a text-only implementation
        self.pp_missing_layers.extend(["vision_model", "multi_modal_projector"])
        # mapping from https://huggingface.co/google/gemma-3-4b-it/blob/main/model.safetensors.index.json 
        mappings = {
            "language_model.model.embed_tokens.weight": "model.embed_tokens.weight",
            "language_model.model.norm.weight": "model.norm.weight",
        }
        
        if not self.vllm_config.model_config.hf_config.tie_word_embeddings:
            mappings["language_model.lm_head.weight"] = "lm_head.weight"
            
        for layer_attr in [
            "self_attn.q_proj",
            "self_attn.k_proj",
            "self_attn.v_proj",
            "self_attn.o_proj",
            "self_attn.q_norm",
            "self_attn.k_norm",
            "mlp.gate_proj",
            "mlp.up_proj",
            "mlp.down_proj",
            "input_layernorm",
            "post_attention_layernorm",
            "pre_feedforward_layernorm",
            "post_feedforward_layernorm",
        ]:
            for suffix in [".weight", ".bias"]:
                mappings[f"language_model.model.layers.*.{layer_attr}{suffix}"] = f"model.layers.*.{layer_attr}{suffix}"
                
        # Temporarily expose the language model head counts to the top-level config
        # so the default weight loader can resolve reshaping and padding layouts.
        # Need to change after this code start to support video context. 
        hf_config = self.vllm_config.model_config.hf_config
        if hasattr(hf_config, "text_config"):
            if not hasattr(hf_config, "num_attention_heads"):
                hf_config.num_attention_heads = hf_config.text_config.num_attention_heads
            if not hasattr(hf_config, "num_key_value_heads"):
                hf_config.num_key_value_heads = hf_config.text_config.num_key_value_heads

        loader = self.WeightLoader(self.vllm_config, self.mesh)
            
        loader.load_weights(
            self,
            mappings,
            keep_hf_weight_suffix_when_match=['model'])
            
        # Restore PyTorch dtype in model_config so vLLM multimodal processor 
        # doesn't crash during dummy input generation when calling x.to(dtype=...)
        from tpu_inference.utils import to_torch_dtype
        if hasattr(self.vllm_config.model_config.dtype, "type"):
            self.vllm_config.model_config.dtype = to_torch_dtype(self.vllm_config.model_config.dtype)


class GemmaRMSNorm(JaxRmsNorm):
    """RMS normalization layer specific to Gemma models."""

    def __call__(self,
                 x: jax.Array,
                 mask: Optional[jax.Array] = None) -> jax.Array:
        # Gemma style RMSNorm: x * (1 + w)
        # Save original weight
        w = self.weight.value
        # Temporarily set weight to ones to get pure norm and re-use JaxRmsNorm for efficient normalization. 
        self.weight.value = jnp.ones_like(w)
        normed_x = super().__call__(x, mask=mask)
        # Restore weight
        self.weight.value = w

        return normed_x * (1.0 + w)


class Gemma3MLP(JaxModule):
    """Gated Feed-Forward Network (MLP) for Gemma 3."""

    def __init__(self,
                 config: Gemma3Config,
                 dtype: jnp.dtype,
                 rng: nnx.Rngs,
                 quant_config: VllmQuantConfig,
                 prefix: str = ""):
        hidden_size = config.hidden_size
        intermediate_size = config.intermediate_size

        self.gate_proj = JaxLinear(
            hidden_size,
            intermediate_size,
            use_bias=False,
            dtype=dtype,
            kernel_init=nnx.with_partitioning(init_fn, (None, "model")),
            rngs=rng,
            quant_config=quant_config,
            prefix=prefix + ".gate_proj",
        )
        self.up_proj = JaxLinear(
            hidden_size,
            intermediate_size,
            use_bias=False,
            dtype=dtype,
            kernel_init=nnx.with_partitioning(init_fn, (None, "model")),
            rngs=rng,
            quant_config=quant_config,
            prefix=prefix + ".up_proj",
        )
        self.down_proj = JaxLinear(
            intermediate_size,
            hidden_size,
            use_bias=False,
            dtype=dtype,
            kernel_init=nnx.with_partitioning(init_fn, ("model", None)),
            rngs=rng,
            quant_config=quant_config,
            prefix=prefix + ".down_proj",
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        gate = self.gate_proj(x)
        # Gemma 3 uses the approximate GELU activation function.
        gate = jax.nn.gelu(gate, approximate=True)
        up = self.up_proj(x)
        fuse = gate * up
        result = self.down_proj(fuse)
        return result

class Gemma3Attention(JaxModule):
    """Multi-head attention mechanism for Gemma 3, supporting both global and sliding window attention."""

    def __init__(self,
                 config: Gemma3Config,
                 dtype: jnp.dtype,
                 rng: nnx.Rngs,
                 mesh: Mesh,
                 kv_cache_dtype: str,
                 quant_config: VllmQuantConfig,
                 prefix: str = ""):
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.attention_chunk_size = None

        self.head_dim_original = getattr(
            config, "head_dim", self.hidden_size // self.num_heads)
        self.head_dim = utils.get_padded_head_dim(self.head_dim_original)

        sharding_size = mesh.shape["model"]
        self.num_heads = utils.get_padded_num_heads(self.num_heads,
                                                    sharding_size)
        self.num_kv_heads = utils.get_padded_num_heads(self.num_kv_heads,
                                                       sharding_size)

        self.mesh = mesh

        # Determine if this specific layer uses sliding window attention 
        # based on the model configuration's layer_types list.
        try:
            layer_idx = int(prefix.split('.')[-2])
        except (ValueError, IndexError):
            layer_idx = 0

        layer_type = config.layer_types[
            layer_idx] if hasattr(config, "layer_types") else "global"
        self.is_sliding = layer_type == "sliding_attention"
        self.rope_theta = getattr(config, "rope_theta", 10000.0)
        
        # Gemma3 using Local layer and Global layer with a ratial of 5:1, in the local layer, we apply a
        # sliding window mask and use a relatively smaller rope_theta.
        if self.is_sliding:
            self.rope_theta = getattr(config, "rope_local_base_freq", 10000.0)
            self.attention_chunk_size = getattr(config, "sliding_window", 1024)
    
        self.rope_scaling = getattr(config, "rope_scaling", None)
        attention_bias = getattr(config, "attention_bias", False)

        self.q_proj = JaxEinsum(
            "TD,DNH->TNH",
            (self.hidden_size, self.num_heads, self.head_dim),
            bias_shape=(self.num_heads,
                        self.head_dim) if attention_bias else None,
            dtype=dtype,
            kernel_init=nnx.with_partitioning(init_fn, (None, "model", None)),
            bias_init=nnx.with_partitioning(init_fn, ("model", None))
            if attention_bias else None,
            rngs=rng,
            quant_config=quant_config,
            prefix=prefix + ".q_proj",
        )
        self.k_proj = JaxEinsum(
            "TD,DKH->TKH",
            (self.hidden_size, self.num_kv_heads, self.head_dim),
            bias_shape=(self.num_kv_heads,
                        self.head_dim) if attention_bias else None,
            dtype=dtype,
            kernel_init=nnx.with_partitioning(init_fn, (None, "model", None)),
            bias_init=nnx.with_partitioning(init_fn, ("model", None))
            if attention_bias else None,
            rngs=rng,
            quant_config=quant_config,
            prefix=prefix + ".k_proj",
        )
        self.v_proj = JaxEinsum(
            "TD,DKH->TKH",
            (self.hidden_size, self.num_kv_heads, self.head_dim),
            bias_shape=(self.num_kv_heads,
                        self.head_dim) if attention_bias else None,
            dtype=dtype,
            kernel_init=nnx.with_partitioning(init_fn, (None, "model", None)),
            bias_init=nnx.with_partitioning(init_fn, ("model", None))
            if attention_bias else None,
            rngs=rng,
            quant_config=quant_config,
            prefix=prefix + ".v_proj",
        )
        self.o_proj = JaxEinsum(
            "TNH,NHD->TD",
            (self.num_heads, self.head_dim, self.hidden_size),
            bias_shape=(self.hidden_size, ) if attention_bias else None,
            dtype=dtype,
            kernel_init=nnx.with_partitioning(init_fn, ("model", None, None)),
            bias_init=nnx.with_partitioning(
                init_fn, (None, )) if attention_bias else None,
            rngs=rng,
            quant_config=quant_config,
            prefix=prefix + ".o_proj",
        )

        self.q_norm = GemmaRMSNorm(self.head_dim,
                                   epsilon=config.rms_norm_eps,
                                   dtype=dtype,
                                   rngs=rng,
                                   quant_config=quant_config,
                                   prefix=prefix + ".q_norm")
        self.k_norm = GemmaRMSNorm(self.head_dim,
                                   epsilon=config.rms_norm_eps,
                                   dtype=dtype,
                                   rngs=rng,
                                   quant_config=quant_config,
                                   prefix=prefix + ".k_norm")

        # Gemma 3 queries scale pre-attention.
        self._q_scale = getattr(config, "query_pre_attn_scalar", 1.0)**-0.5
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
        md = attention_metadata
        
        # q: (T, N, H)
        q = self.q_proj(x)
        q = self.q_norm(q)
        q = apply_rope(q, md.input_positions, self.head_dim_original,
                       self.rope_theta, self.rope_scaling)

        # k: (T, K, H)
        k = self.k_proj(x)
        k = self.k_norm(k)
        k = apply_rope(k, md.input_positions, self.head_dim_original,
                       self.rope_theta, self.rope_scaling)

        # v: (T, K, H)
        v = self.v_proj(x)

        # Apply KV cache quantization if enabled
        q_scale = k_scale = v_scale = None
        if self.kv_cache_quantized_dtype:
            k_scale = self._k_scale
            v_scale = self._v_scale
            k, v = quantize_kv(self.kv_cache_quantized_dtype, k, v, k_scale,
                               v_scale)

        # Attention operation
        new_kv_cache, outputs = attention(
            kv_cache,
            q,
            k,
            v,
            attention_metadata,
            self.mesh,
            self.head_dim_original,
            attention_chunk_size=self.attention_chunk_size,
            q_scale=self._q_scale,
            k_scale=k_scale,
            v_scale=v_scale,
        )

        # o: (T, D)
        o = self.o_proj(outputs)
        return new_kv_cache, o


class Gemma3DecoderLayer(JaxModule):
    """A single Transformer block for Gemma 3."""

    def __init__(self,
                 config: Gemma3Config,
                 dtype: jnp.dtype,
                 rng: nnx.Rngs,
                 mesh: Mesh,
                 kv_cache_dtype: str,
                 quant_config: VllmQuantConfig,
                 prefix: str = ""):
        rms_norm_eps = config.rms_norm_eps
        hidden_size = config.hidden_size

        self.input_layernorm = GemmaRMSNorm(
            hidden_size,
            epsilon=rms_norm_eps,
            dtype=dtype,
            rngs=rng,
            quant_config=quant_config,
            prefix=prefix + ".input_layernorm")
        self.self_attn = Gemma3Attention(config=config,
                                         dtype=dtype,
                                         rng=rng,
                                         mesh=mesh,
                                         kv_cache_dtype=kv_cache_dtype,
                                         quant_config=quant_config,
                                         prefix=prefix + ".self_attn")
        self.post_attention_layernorm = GemmaRMSNorm(
            hidden_size,
            epsilon=rms_norm_eps,
            dtype=dtype,
            rngs=rng,
            quant_config=quant_config,
            prefix=prefix + ".post_attention_layernorm")
        self.pre_feedforward_layernorm = GemmaRMSNorm(
            hidden_size,
            epsilon=rms_norm_eps,
            dtype=dtype,
            rngs=rng,
            quant_config=quant_config,
            prefix=prefix + ".pre_feedforward_layernorm")
        self.mlp = Gemma3MLP(config=config,
                             dtype=dtype,
                             rng=rng,
                             quant_config=quant_config,
                             prefix=prefix + ".mlp")
        self.post_feedforward_layernorm = GemmaRMSNorm(
            hidden_size,
            epsilon=rms_norm_eps,
            dtype=dtype,
            rngs=rng,
            quant_config=quant_config,
            prefix=prefix + ".post_feedforward_layernorm")

    def __call__(
        self,
        kv_cache: jax.Array,
        hidden_states: jax.Array,
        residual: Optional[jax.Array],
        attention_metadata: AttentionMetadata,
    ) -> Tuple[jax.Array, jax.Array, jax.Array]:

        # Uses a specific residual stream structure where `residual` and 
        # `hidden_states` are tracked separately across the transformer block.
        # For the first layer where residual is not initialized. 
        if residual is None:
            residual = hidden_states
            normed_hidden_states = self.input_layernorm(hidden_states)
        else:
        # For the other layers
            hidden_states = hidden_states + residual
            residual = hidden_states
            normed_hidden_states = self.input_layernorm(hidden_states)

        kv_cache, attn_output = self.self_attn(
            kv_cache,
            normed_hidden_states,
            attention_metadata,
        )
        attn_output = self.post_attention_layernorm(attn_output)

        hidden_states = attn_output + residual
        residual = hidden_states
        normed_hidden_states = self.pre_feedforward_layernorm(hidden_states)

        mlp_output = self.mlp(normed_hidden_states)
        hidden_states = self.post_feedforward_layernorm(mlp_output)

        return kv_cache, hidden_states, residual

class Gemma3Model(JaxModule):
    """The core Gemma 3 model without the language modeling head."""

    def __init__(self,
                 vllm_config: VllmConfig,
                 rng: nnx.Rngs,
                 mesh: Mesh,
                 prefix: str = "model") -> None:
        model_config = vllm_config.model_config
        hf_config = model_config.hf_config
        
        # Unwrap the text config if this is a multimodal model
        if hasattr(hf_config, "text_config"):
            hf_config = hf_config.text_config
            
        vocab_size = model_config.get_vocab_size()
        dtype = model_config.dtype
        rms_norm_eps = hf_config.rms_norm_eps
        hidden_size = hf_config.hidden_size

        self.is_first_rank = get_pp_group().is_first_rank
        self.is_last_rank = get_pp_group().is_last_rank

        if self.is_first_rank or (hf_config.tie_word_embeddings
                                  and self.is_last_rank):
            self.embed_tokens = JaxEmbed(
                num_embeddings=vocab_size,
                features=hidden_size,
                dtype=dtype,
                embedding_init=nnx.with_partitioning(init_fn, ("model", None)),
                rngs=rng,
                quant_config=vllm_config.quant_config,
                prefix=prefix + ".embed_tokens",
            )
        else:
            self.embed_tokens = PPMissingLayer()

        # Normalize the embedding by sqrt(hidden_size)
        # The normalizer's data type should be downcasted to the model's
        # data type such as bfloat16, not float32.
        # See https://github.com/huggingface/transformers/pull/29402 
        self.normalizer = hidden_size**0.5

        self.start_layer, self.end_layer, self.layers = make_layers(
            hf_config.num_hidden_layers,
   
            lambda layer_index: Gemma3DecoderLayer(
                config=hf_config,
                dtype=dtype,
                rng=rng,
                mesh=mesh,
                kv_cache_dtype=vllm_config.cache_config.cache_dtype,
                quant_config=vllm_config.quant_config,
                prefix=f"{prefix}.layers.{layer_index}",
            ),
        )
        if self.is_last_rank:
    
            self.norm = GemmaRMSNorm(
                hidden_size,
                epsilon=rms_norm_eps,
                dtype=dtype,
                rngs=rng,
                quant_config=vllm_config.quant_config,
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
        intermediate_tensors: Optional[JaxIntermediateTensors] = None,
    ) -> Tuple[List[jax.Array], jax.Array, jax.Array]:

        if self.is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.embed_tokens(
                    input_ids) * self.normalizer
            residual = None
        else:
            # Handle pipeline parallelism intermediate states
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors.get("residual", None)

        for i, layer in enumerate(
                islice(self.layers, self.start_layer, self.end_layer)):
            kv_cache = kv_caches[i]
            kv_cache, hidden_states, residual = layer(
                kv_cache,
                hidden_states,
                residual,
                attention_metadata,
            )
            kv_caches[i] = kv_cache

        if self.is_last_rank:
            # normalize again at the very last layer. 
            hidden_states = hidden_states + residual
            hidden_states = self.norm(hidden_states)

        return kv_caches, hidden_states, residual
