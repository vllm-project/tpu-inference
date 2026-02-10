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

from typing import List, Optional, Tuple

import jax
import jax.numpy as jnp
from flax import nnx
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P
from transformers import Qwen3Config
from vllm.config import VllmConfig

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
from tpu_inference.models.jax.jax_intermediate_tensor import \
    JaxIntermediateTensors
from tpu_inference.models.jax.qwen3 import Qwen3Attention
from tpu_inference.models.jax.utils.weight_utils import LoadableWithIterator

logger = init_logger(__name__)

init_fn = nnx.initializers.uniform()


class Qwen3MoeSparseMoeBlock(JaxModule):

    def __init__(self,
                 vllm_config: VllmConfig,
                 rng: nnx.Rngs,
                 mesh: Mesh,
                 prefix: str = ""):
        config = vllm_config.model_config.hf_text_config
        dtype = vllm_config.model_config.dtype
        quant_config = vllm_config.quant_config

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
        self.gate.num_experts_per_tok = config.num_experts_per_tok

        # Shared Expert
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
            intermediate_size_moe=config.moe_intermediate_size,
            hidden_act=config.hidden_act,
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


class Qwen3MoeDecoderLayer(JaxModule):

    def __init__(self,
                 config: Qwen3Config,
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

        self.input_layernorm = JaxRmsNorm(
            hidden_size,
            epsilon=rms_norm_eps,
            param_dtype=dtype,
            scale_init=nnx.with_partitioning(init_fn, (None, )),
            rngs=rng,
            quant_config=quant_config,
            prefix=prefix + ".input_layernorm",
        )
        self.self_attn = Qwen3Attention(
            config=config,
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

        mlp_only_layers = getattr(config, "mlp_only_layers", [])
        if (layer_idx not in mlp_only_layers) and (
                config.num_experts > 0 and
            (layer_idx + 1) % config.decoder_sparse_step == 0):
            self.mlp = Qwen3MoeSparseMoeBlock(vllm_config=vllm_config,
                                              rng=rng,
                                              mesh=mesh,
                                              prefix=prefix + ".mlp")
        else:
            raise NotImplementedError(
                f"Non-sparse MLP is not implemented yet. Found {mlp_only_layers=}, {config.num_experts=}, and {config.decoder_sparse_step=} in config."
            )

    def __call__(
        self,
        kv_cache: jax.Array,
        x: jax.Array,
        attention_metadata: AttentionMetadata,
    ) -> Tuple[jax.Array, jax.Array]:
        hidden_states = self.input_layernorm(x)
        kv_cache, attn_output = self.self_attn(
            kv_cache,
            hidden_states,
            attention_metadata,
        )
        attn_output += x

        residual = attn_output
        attn_output = self.post_attention_layernorm(attn_output)
        outputs = self.mlp(attn_output)
        outputs = residual + outputs
        return kv_cache, outputs


class Qwen3MoeModel(JaxModule):

    def __init__(self,
                 vllm_config: VllmConfig,
                 rng: nnx.Rngs,
                 mesh: Mesh,
                 prefix: str = "") -> None:
        model_config = vllm_config.model_config
        hf_config = model_config.hf_config
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
                param_dtype=dtype,
                embedding_init=nnx.with_partitioning(init_fn, ("model", None)),
                rngs=rng,
                quant_config=vllm_config.quant_config,
                prefix=prefix + ".embed_tokens",
            )
        else:
            self.embed_tokens = PPMissingLayer()

        self.start_layer, self.end_layer, self.layers = make_layers(
            hf_config.num_hidden_layers,
            lambda layer_index: Qwen3MoeDecoderLayer(
                config=hf_config,
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
    ) -> Tuple[List[jax.Array], jax.Array]:
        if self.is_first_rank:
            assert inputs_embeds is None
            inputs_embeds = self.embed_tokens(input_ids)
        else:
            assert inputs_embeds is not None

        x = inputs_embeds
        new_kv_caches = []
        for i, layer in enumerate(self.layers):
            if isinstance(layer, PPMissingLayer):
                new_kv_caches.append(kv_caches[i])
                continue
            kv_cache = kv_caches[i]
            kv_cache, x = layer(kv_cache, x, attention_metadata)
            new_kv_caches.append(kv_cache)

        if self.is_last_rank:
            x = self.norm(x)

        return new_kv_caches, x


class Qwen3MoeForCausalLM(JaxModule, LoadableWithIterator):

    def __init__(self, vllm_config: VllmConfig, rng_key: jax.Array,
                 mesh: Mesh) -> None:
        self.vllm_config = vllm_config
        rng = nnx.Rngs(rng_key)
        self.mesh = mesh

        self.model = Qwen3MoeModel(
            vllm_config=vllm_config,
            rng=rng,
            mesh=mesh,
            prefix="model",
        )
        model_config = vllm_config.model_config
        if not model_config.hf_config.tie_word_embeddings:
            if self.model.is_last_rank:
                vocab_size = model_config.get_vocab_size()
                hidden_size = model_config.hf_config.hidden_size
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
        kv_caches, x = self.model(
            kv_caches,
            input_ids,
            attention_metadata,
            inputs_embeds,
        )
        if not is_last_rank:
            x = JaxIntermediateTensors(tensors={"hidden_states": x}, )
        return kv_caches, x, []

    def compute_logits(self, hidden_states: jax.Array) -> jax.Array:
        if hasattr(self, 'lm_head'):
            return self.lm_head(hidden_states)

        assert isinstance(self.model.embed_tokens, JaxEmbed)
        return self.model.embed_tokens.decode(hidden_states)
