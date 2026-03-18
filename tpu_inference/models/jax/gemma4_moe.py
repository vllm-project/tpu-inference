# Copyright 2025 Google LLC
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
from typing import Any, Iterable, List, Optional, Tuple

import jax
import jax.numpy as jnp
from flax import nnx
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P
from transformers import Gemma4TextConfig
from vllm.config import ModelConfig, VllmConfig
from vllm.model_executor.models.utils import WeightsMapper

from tpu_inference.distributed.jax_parallel_state import get_pp_group
from tpu_inference.layers.common.attention_metadata import AttentionMetadata
from tpu_inference.layers.common.sharding import ShardingAxisName
from tpu_inference.layers.jax import JaxModule
from tpu_inference.layers.jax.embed import JaxEmbed
from tpu_inference.layers.jax.linear import JaxEinsum, JaxLinear
from tpu_inference.layers.jax.moe.moe import JaxMoE
from tpu_inference.layers.jax.moe.utils import (get_expert_parallelism,
                                                select_moe_backend)
from tpu_inference.layers.jax.norm import JaxRmsNorm
from tpu_inference.layers.jax.pp_utils import PPMissingLayer, make_layers
from tpu_inference.layers.jax.quantization.configs import QuantizationConfig
from tpu_inference.logger import init_logger
from tpu_inference.models.jax.gemma4 import Gemma4Attention
from tpu_inference.models.jax.jax_intermediate_tensor import \
    JaxIntermediateTensors
from tpu_inference.models.jax.utils.weight_utils import (JaxAutoWeightsLoader,
                                                         LoadableWithIterator)

logger = init_logger(__name__)

init_fn = nnx.initializers.uniform()


class Gemma4MoeSparseMoeBlock(JaxModule):

    def __init__(self,
                 model_config: ModelConfig,
                 rng: nnx.Rngs,
                 mesh: Mesh,
                 quant_config,
                 prefix: str = ""):
        config: Gemma4TextConfig = model_config.hf_config.text_config
        dtype = model_config.dtype

        # --- Sharding Config ---
        edf_sharding = (None, None, None)
        expert_axis_name = edf_sharding[0]
        num_expert_parallelism = get_expert_parallelism(expert_axis_name, mesh)
        use_ep = num_expert_parallelism > 1
        moe_backend = select_moe_backend(use_ep)

        # Router
        self.gate = JaxLinear(
            config.hidden_size,
            config.num_experts,
            rngs=rng,
            use_bias=False,
            quant_config=quant_config,
            prefix=prefix + ".gate",
        )
        self.gate.num_experts_per_tok = config.num_experts

        # Gemma4: (flax moe module does not have shared expert, keep for modification if e2e code has)
        shared_expert_intermediate_size = getattr(
            config, "shared_expert_intermediate_size", 0)
        if shared_expert_intermediate_size > 0:
            raise NotImplementedError(
                f"Shared expert is not implemented yet. Found {shared_expert_intermediate_size=} in config."
            )
        else:
            self.shared_expert = None

        # Experts (Routed)
        self.experts = JaxMoE(
            dtype=dtype,
            num_local_experts=config.num_experts,
            hidden_size=config.hidden_size,
            intermediate_size_moe=getattr(config, "moe_intermediate_size",
                                          config.intermediate_size),
            hidden_act=config.
            hidden_activation,  # gelu, need to check if this is supported
            rngs=rng,
            router=self.gate,
            mesh=mesh,
            activation_ffw_td=P(ShardingAxisName.MLP_DATA, None),
            activation_ffw_ted=P(ShardingAxisName.MLP_DATA, None, None),
            edf_sharding=P(None, ),
            efd_sharding=P(None, ),
            apply_expert_weight_before_computation=False,
            expert_axis_name=expert_axis_name,
            num_expert_parallelism=num_expert_parallelism,
            moe_backend=moe_backend,
            quant_config=quant_config,
            prefix=prefix + ".experts")

    def __call__(self, x: jax.Array) -> jax.Array:
        out = self.experts(x)
        if self.shared_expert is not None:
            out += self.shared_expert(x)
        return out


class Gemma4MoeDecoderLayer(JaxModule):

    def __init__(self,
                 config: Gemma4TextConfig,
                 dtype: jnp.dtype,
                 rng: nnx.Rngs,
                 mesh: Mesh,
                 kv_cache_dtype: str,
                 quant_config: QuantizationConfig,
                 layer_idx: int,
                 vllm_config: VllmConfig,
                 prefix: str = ""):
        rms_norm_eps = config.rms_norm_eps
        hidden_size = config.hidden_size

        self.layer_type = "full_attention"
        if hasattr(config, "layer_types") and layer_idx < len(
                config.layer_types):
            self.layer_type = config.layer_types[layer_idx]

        self.is_sliding = self.layer_type == "sliding_attention"

        # Gemma 4: skip scale logic
        if not self.is_sliding:
            self.skip_scale = nnx.Param(jnp.ones((1, ), dtype=dtype))
        else:
            self.skip_scale = None

        self.input_layernorm = JaxRmsNorm(
            hidden_size,
            epsilon=rms_norm_eps,
            param_dtype=dtype,
            scale_init=nnx.with_partitioning(init_fn, (None, )),
            rngs=rng,
            quant_config=quant_config,
            prefix=prefix + ".input_layernorm",
        )
        self.self_attn = Gemma4Attention(
            config=config,
            layer_idx=layer_idx,
            dtype=dtype,
            rng=rng,
            mesh=mesh,
            kv_cache_dtype=kv_cache_dtype,
            quant_config=quant_config,
            prefix=prefix + ".self_attn",
        )
        self.post_attention_layernorm = JaxRmsNorm(
            hidden_size,
            epsilon=rms_norm_eps,
            param_dtype=dtype,
            scale_init=nnx.with_partitioning(init_fn, (None, )),
            rngs=rng,
            quant_config=quant_config,
            prefix=prefix + ".post_attention_layernorm",
        )

        self.pre_feedforward_layernorm = JaxRmsNorm(
            hidden_size,
            epsilon=rms_norm_eps,
            param_dtype=dtype,
            scale_init=nnx.with_partitioning(init_fn, (None, )),
            rngs=rng,
            quant_config=quant_config,
            prefix=prefix + ".pre_feedforward_layernorm",
        )

        self.mlp = Gemma4MoeSparseMoeBlock(vllm_config=vllm_config,
                                           rng=rng,
                                           mesh=mesh,
                                           prefix=prefix + ".mlp")

        self.post_feedforward_layernorm = JaxRmsNorm(
            hidden_size,
            epsilon=rms_norm_eps,
            param_dtype=dtype,
            scale_init=nnx.with_partitioning(init_fn, (None, )),
            rngs=rng,
            quant_config=quant_config,
            prefix=prefix + ".post_feedforward_layernorm",
        )

    def __call__(
        self,
        kv_cache: jax.Array,
        x: jax.Array,
        attention_metadata: AttentionMetadata,
    ) -> Tuple[jax.Array, jax.Array]:
        residual = x
        hidden_states = self.input_layernorm(x)
        kv_cache, attn_output = self.self_attn(
            kv_cache,
            hidden_states,
            attention_metadata,
        )
        attn_output = self.post_attention_layernorm(attn_output)
        residual = residual + attn_output

        hidden_states = self.pre_feedforward_layernorm(residual)
        mlp_output = self.mlp(hidden_states)
        mlp_output = self.post_feedforward_layernorm(mlp_output)
        outputs = residual + mlp_output

        if self.skip_scale is not None:
            outputs = outputs * self.skip_scale.value

        return kv_cache, outputs


class Gemma4MoeModel(JaxModule):

    def __init__(self,
                 vllm_config: VllmConfig,
                 rng: nnx.Rngs,
                 mesh: Mesh,
                 prefix: str = "") -> None:
        model_config = vllm_config.model_config
        hf_config = model_config.hf_config
        text_config = hf_config.text_config
        vocab_size = model_config.get_vocab_size()
        dtype = model_config.dtype
        rms_norm_eps = text_config.rms_norm_eps
        hidden_size = text_config.hidden_size

        self.is_first_rank = get_pp_group().is_first_rank
        self.is_last_rank = get_pp_group().is_last_rank

        self.embedding_scale = hidden_size**0.5

        if self.is_first_rank or (hf_config.tie_word_embeddings
                                  and self.is_last_rank):
            self.embed_tokens = JaxEmbed(
                num_embeddings=vocab_size,
                features=hidden_size,
                param_dtype=dtype,
                embedding_init=nnx.with_partitioning(init_fn, ("model", None)),
                rngs=rng,
                quant_config=vllm_config.quant_config,
                prefix=prefix + ".embed_tokens",
            )
        else:
            self.embed_tokens = PPMissingLayer()

        self.start_layer, self.end_layer, self.layers = make_layers(
            text_config.num_hidden_layers,
            lambda layer_index: Gemma4MoeDecoderLayer(
                config=text_config,
                dtype=dtype,
                rng=rng,
                mesh=mesh,
                kv_cache_dtype=vllm_config.cache_config.cache_dtype,
                quant_config=vllm_config.quant_config,
                layer_idx=layer_index,
                vllm_config=vllm_config,
                prefix=f"{prefix}.layers.{layer_index}",
            ))

        if self.is_last_rank:
            self.norm = JaxRmsNorm(
                hidden_size,
                epsilon=rms_norm_eps,
                param_dtype=dtype,
                scale_init=nnx.with_partitioning(init_fn, (None, )),
                rngs=rng,
                quant_config=vllm_config.quant_config,
                prefix=prefix + ".final_layernorm",
            )
        else:
            self.norm = PPMissingLayer()

    def __call__(
        self,
        kv_caches: List[jax.Array],
        input_ids: jax.Array,
        attention_metadata: AttentionMetadata,
        inputs_embeds: Optional[jax.Array] = None,
        layer_name_to_kv_cache: Optional[dict] = None,
    ) -> Tuple[List[jax.Array], jax.Array]:
        if inputs_embeds is not None:
            x = inputs_embeds
        else:
            x = self.embed_tokens(input_ids)
            # Gemma4: Apply embedding scaling
            x = x * self.embedding_scale

        for i, layer in enumerate(
                islice(self.layers, self.start_layer, self.end_layer)):
            layer_name = f"layer.{i + self.start_layer}"
            if isinstance(attention_metadata, dict):
                layer_attn_metadata = attention_metadata[layer_name]
            else:
                layer_attn_metadata = attention_metadata

            if layer_name_to_kv_cache and layer_name in layer_name_to_kv_cache:
                cache_idx = layer_name_to_kv_cache[layer_name]
            else:
                cache_idx = i + self.start_layer

            kv_cache = kv_caches[cache_idx]
            kv_cache, x = layer(
                kv_cache,
                x,
                layer_attn_metadata,
            )
            kv_caches[cache_idx] = kv_cache
        x = self.norm(x)
        return kv_caches, x


class Gemma4MoeForCausalLM(JaxModule, LoadableWithIterator):

    def __init__(self, vllm_config: VllmConfig, rng_key: jax.Array,
                 mesh: Mesh) -> None:
        if getattr(vllm_config.model_config, "quantization", None) == "fp8":
            # `get_tpu_quantization_config` returns None for "fp8" because
            # the work in #1623 is not fully merged. So this block overrides
            # the logic to return Fp8Config when model_config indicates fp8.
            # TODO(#1623): Remove this block when `get_tpu_quantization_config`
            # is updated.
            from tpu_inference.layers.jax.quantization.fp8 import Fp8Config
            hg_quant_config = getattr(vllm_config.model_config.hf_config,
                                      "quantization_config", {})
            vllm_config.quant_config = Fp8Config(hg_quant_config)

        self.vllm_config = vllm_config
        rng = nnx.Rngs(rng_key)
        self.mesh = mesh

        self.model = Gemma4MoeModel(
            vllm_config=vllm_config,
            rng=rng,
            mesh=mesh,
            prefix="model",
        )
        model_config = vllm_config.model_config

        self.final_logit_softcapping = getattr(
            model_config.hf_config.text_config, "final_logit_softcapping",
            None)

        if not model_config.hf_config.tie_word_embeddings:
            if self.model.is_last_rank:
                vocab_size = model_config.get_vocab_size()
                hidden_size = model_config.hf_config.text_config.hidden_size
                self.lm_head = JaxEinsum(
                    einsum_str="TD,DV->TV",
                    kernel_shape=(hidden_size, vocab_size),
                    dtype=model_config.dtype,
                    rngs=rng,
                    quant_config=vllm_config.quant_config,
                    prefix="lm_head",
                )
            else:
                self.lm_head = PPMissingLayer()

    def load_weights(self, weights: Iterable[Tuple[str, Any]]):
        stripped_weights = ((name, 1.0 + tensor if "norm" in name else tensor)
                            for name, tensor in weights)

        hf_to_vllm_mapper = WeightsMapper(
            orig_to_new_substr={".language_model.": "."})
        loader = JaxAutoWeightsLoader(
            self,
            skip_prefixes=(["lm_head"]
                           if not hasattr(self, 'lm_head') else None),
            skip_substrs=["vision"])
        return loader.load_weights(stripped_weights, mapper=hf_to_vllm_mapper)

    def __call__(
        self,
        kv_caches: List[jax.Array],
        input_ids: jax.Array,
        attention_metadata: AttentionMetadata,
        inputs_embeds: Optional[jax.Array] = None,
        _input_positions=None,
        _layer_name_to_kv_cache=None,
        _lora_metadata=None,
        intermediate_tensors: JaxIntermediateTensors | None = None,
        is_first_rank: bool = True,
        is_last_rank: bool = True,
        *args,
    ) -> Tuple[List[jax.Array], jax.Array | JaxIntermediateTensors,
               List[jax.Array]]:

        if not is_first_rank:
            assert intermediate_tensors is not None
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
            x = JaxIntermediateTensors(tensors={"hidden_states": x}, )
        return kv_caches, x, []

    def compute_logits(self, hidden_states: jax.Array) -> jax.Array:
        if hasattr(self, 'lm_head'):
            logits = self.lm_head(hidden_states)
        else:
            assert isinstance(self.model.embed_tokens, JaxEmbed)
            logits = self.model.embed_tokens.decode(hidden_states)

        # Gemma4: Use Logit Soft-capping
        if self.final_logit_softcapping is not None:
            logits = jnp.tanh(
                logits /
                self.final_logit_softcapping) * self.final_logit_softcapping
        return logits
