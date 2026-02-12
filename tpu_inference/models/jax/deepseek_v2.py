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
from typing import Any, List, Optional, Tuple

import jax
import jax.numpy as jnp
from flax import nnx
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P
from transformers import DeepseekV2Config
from vllm.config import VllmConfig

from tpu_inference import utils
from tpu_inference.distributed.jax_parallel_state import get_pp_group
from tpu_inference.kernels.mla.v1.kernel import mla_ragged_paged_attention
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
from tpu_inference.layers.jax.rope import DeepseekScalingRotaryEmbedding
from tpu_inference.layers.vllm.quantization.configs import VllmQuantConfig
from tpu_inference.logger import init_logger
from tpu_inference.models.jax.jax_intermediate_tensor import \
    JaxIntermediateTensors
from tpu_inference.models.jax.utils.weight_utils import (
    LoadableWithIterator, load_nnx_param_from_reshaped_torch)

logger = init_logger(__name__)

init_fn = nnx.initializers.uniform()


class DeepseekV2MLP(JaxModule):

    def __init__(self,
                 config: DeepseekV2Config,
                 dtype: jnp.dtype,
                 rng: nnx.Rngs,
                 quant_config: VllmQuantConfig,
                 intermediate_size: Optional[int] = None,
                 prefix: str = ""):
        self.hidden_size = config.hidden_size
        self.intermediate_size = intermediate_size or config.intermediate_size

        self.gate_up_proj = JaxEinsum(
            "TD,DF->TF",
            (self.hidden_size, 2 * self.intermediate_size),
            dtype=dtype,
            rngs=rng,
            quant_config=quant_config,
            prefix=prefix + ".gate_up_proj",
        )
        self.down_proj = JaxEinsum(
            "TF,FD->TD",
            (self.intermediate_size, self.hidden_size),
            dtype=dtype,
            rngs=rng,
            quant_config=quant_config,
            prefix=prefix + ".down_proj",
        )
        self.act_fn = nnx.silu

    def __call__(self, x: jax.Array) -> jax.Array:
        gate_up = self.gate_up_proj(x)
        gate, up = jnp.split(gate_up, 2, axis=-1)
        return self.down_proj(self.act_fn(gate) * up)


class DeepseekV2MoE(JaxModule):

    def __init__(self,
                 vllm_config: VllmConfig,
                 rng: nnx.Rngs,
                 mesh: Mesh,
                 prefix: str = ""):
        config: DeepseekV2Config = vllm_config.model_config.hf_config
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
            config.n_routed_experts,
            rngs=rng,
            use_bias=False,
            quant_config=quant_config,
            prefix=prefix + ".gate",
        )
        self.gate.num_experts_per_tok = config.num_experts_per_tok

        # Shared Experts
        if config.n_shared_experts:
            intermediate_size = config.moe_intermediate_size * config.n_shared_experts
            self.shared_experts = DeepseekV2MLP(
                config=config,
                dtype=dtype,
                rng=rng,
                quant_config=quant_config,
                intermediate_size=intermediate_size,
                prefix=prefix + ".shared_experts",
            )
        else:
            self.shared_experts = None

        # Experts (Routed)
        self.experts = JaxMoE(
            dtype=dtype,
            num_local_experts=config.n_routed_experts,
            hidden_size=config.hidden_size,
            intermediate_size_moe=config.moe_intermediate_size,
            hidden_act=config.hidden_act,
            rngs=rng,
            router=self.gate,
            mesh=mesh,
            activation_ffw_td=(ShardingAxisName.MLP_DATA, None),
            activation_ffw_ted=(ShardingAxisName.MLP_DATA, None, None),
            edf_sharding=(None, ),
            efd_sharding=(None, ),
            apply_expert_weight_before_computation=False,
            expert_axis_name=expert_axis_name,
            num_expert_parallelism=num_expert_parallelism,
            moe_backend=moe_backend,
            quant_config=quant_config,
            prefix=prefix + ".experts")

    def __call__(self, x: jax.Array) -> jax.Array:
        out = self.experts(x)
        if self.shared_experts is not None:
            out += self.shared_experts(x)
        return out


class DeepseekV2Attention(JaxModule):

    def __init__(self,
                 config: DeepseekV2Config,
                 dtype: jnp.dtype,
                 rng: nnx.Rngs,
                 mesh: Mesh,
                 kv_cache_dtype: str,
                 quant_config: VllmQuantConfig,
                 prefix: str = ""):
        self.config = config
        self.dtype = dtype
        self.mesh = mesh

        self.num_heads = config.num_attention_heads
        self.q_lora_rank = getattr(config, "q_lora_rank", None)
        self.kv_lora_rank = config.kv_lora_rank
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.qk_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        self.v_head_dim = config.v_head_dim

        self.q_head_dim = self.qk_head_dim

        if self.q_lora_rank is not None:
            self.q_a_proj = JaxEinsum(
                "TD,DA->TA",
                (config.hidden_size, self.q_lora_rank),
                dtype=dtype,
                rngs=rng,
                quant_config=quant_config,
                prefix=prefix + ".q_a_proj",
            )
            self.q_a_layernorm = JaxRmsNorm(
                self.q_lora_rank,
                epsilon=config.rms_norm_eps,
                dtype=dtype,
                rngs=rng,
                quant_config=quant_config,
                prefix=prefix + ".q_a_layernorm",
            )
            self.q_b_proj = JaxEinsum(
                "TA,AP->TP",
                (self.q_lora_rank, self.num_heads * self.qk_head_dim),
                dtype=dtype,
                rngs=rng,
                quant_config=quant_config,
                prefix=prefix + ".q_b_proj",
            )
        else:
            self.q_proj = JaxEinsum(
                "TD,DNH->TNH",
                (config.hidden_size, self.num_heads, self.qk_head_dim),
                dtype=dtype,
                rngs=rng,
                quant_config=quant_config,
                prefix=prefix + ".q_proj",
            )

        self.kv_a_proj_with_mqa = JaxEinsum(
            "TD,DA->TA",
            (config.hidden_size, self.kv_lora_rank + self.qk_rope_head_dim),
            dtype=dtype,
            rngs=rng,
            quant_config=quant_config,
            prefix=prefix + ".kv_a_proj_with_mqa",
        )
        self.kv_layernorm = JaxRmsNorm(
            self.kv_lora_rank,
            epsilon=config.rms_norm_eps,
            dtype=dtype,
            rngs=rng,
            quant_config=quant_config,
            prefix=prefix + ".kv_a_layernorm",
        )

        # Projections for the MLA kernel
        self.kernel_k_up_proj_ANH = nnx.Param(
            init_fn(
                rng.params(),
                (self.kv_lora_rank, self.num_heads, self.qk_nope_head_dim)))
        self.kernel_v_up_proj_ANH = nnx.Param(
            init_fn(rng.params(),
                    (self.kv_lora_rank, self.num_heads, self.v_head_dim)))

        # Manual weight loaders for MLA params that need splitting from kv_b_proj
        setattr(
            self.kernel_k_up_proj_ANH, "weight_loader",
            functools.partial(self._load_kv_b_proj,
                              part="k",
                              name=prefix + ".k_b_proj"))
        setattr(
            self.kernel_v_up_proj_ANH, "weight_loader",
            functools.partial(self._load_kv_b_proj,
                              part="v",
                              name=prefix + ".v_b_proj"))

        self.o_proj = JaxEinsum(
            "TNH,NHD->TD",
            (self.num_heads, self.v_head_dim, config.hidden_size),
            dtype=dtype,
            rngs=rng,
            quant_config=quant_config,
            prefix=prefix + ".o_proj",
        )

        rope_scaling = getattr(config, "rope_scaling", None)
        self.rope = DeepseekScalingRotaryEmbedding(
            rotary_dim=self.qk_rope_head_dim,
            rope_theta=config.rope_theta,
            original_max_position_embeddings=config.max_position_embeddings,
            scaling_factor=rope_scaling["factor"] if rope_scaling else 1.0,
            dtype=dtype,
        )

        if rope_scaling is not None and rope_scaling["factor"] > 1.0:
            mscale_all_dim = rope_scaling.get("mscale_all_dim", 0.0)
            scaling_factor = rope_scaling["factor"]
            yarn_mscale = 0.1 * mscale_all_dim * jnp.log(scaling_factor) + 1.0
        else:
            yarn_mscale = 1.0

        self.scale = (self.qk_head_dim**-0.5) * (yarn_mscale**2)
        self.kv_cache_quantized_dtype = None
        if kv_cache_dtype != "auto":
            self.kv_cache_quantized_dtype = utils.get_jax_dtype_from_str_dtype(
                kv_cache_dtype)

    def _load_kv_b_proj(self,
                        param: nnx.Param,
                        torch_weight: Any,
                        part: str = "k",
                        name: str = "Unknown"):
        # HF kv_b_proj is (num_heads * (qk_nope + v), kv_lora_rank)
        # We want (kv_lora_rank, num_heads, qk_nope) for k
        # or (kv_lora_rank, num_heads, v_head_dim) for v
        weight_reshaped = torch_weight.view(
            self.num_heads, self.qk_nope_head_dim + self.v_head_dim,
            self.kv_lora_rank)
        if part == "k":
            weight_part = weight_reshaped[:, :self.qk_nope_head_dim, :]
            # (N, H_nope, A) -> (A, N, H_nope)
            permute_dims = (2, 0, 1)
        else:
            weight_part = weight_reshaped[:, self.qk_nope_head_dim:, :]
            # (N, H_v, A) -> (A, N, H_v)
            permute_dims = (2, 0, 1)

        load_nnx_param_from_reshaped_torch(param,
                                           weight_part,
                                           permute_dims=permute_dims,
                                           param_name=name)

    def __call__(
        self,
        kv_cache: jax.Array,
        x: jax.Array,
        attention_metadata: AttentionMetadata,
    ) -> Tuple[jax.Array, jax.Array]:
        md = attention_metadata

        # Query projection
        if self.q_lora_rank is not None:
            q_TA = self.q_a_proj(x)
            q_TA = self.q_a_layernorm(q_TA)
            q_TP = self.q_b_proj(q_TA)
            q_TNH = q_TP.reshape(x.shape[0], self.num_heads, self.qk_head_dim)
        else:
            q_TNH = self.q_proj(x)

        q_nope_TNH = q_TNH[..., :self.qk_nope_head_dim]
        q_rope_TNH = q_TNH[..., self.qk_nope_head_dim:]
        q_rope_TNH = self.rope.apply_rope(md.input_positions, q_rope_TNH)

        # MLA Specific: Project q_nope to latent space for kernel
        q_TNA = jnp.einsum("TNH,ANH -> TNA", q_nope_TNH,
                           self.kernel_k_up_proj_ANH.value)

        # KV projection
        kv_SA = self.kv_a_proj_with_mqa(x)
        k_rope_SH = kv_SA[..., self.kv_lora_rank:]
        k_rope_SNH = k_rope_SH[..., None, :]
        k_rope_SNH = self.rope.apply_rope(md.input_positions, k_rope_SNH)
        k_rope_SH = k_rope_SNH[:, 0, :]

        kv_SA = kv_SA[..., :self.kv_lora_rank]
        kv_SA = self.kv_layernorm(kv_SA)

        in_specs = (
            P(None, ShardingAxisName.MLP_TENSOR, None),  # q
            P(None, ShardingAxisName.MLP_TENSOR, None),  # q_rope
            P(None, ShardingAxisName.MLP_TENSOR),  # k
            P(None, ShardingAxisName.MLP_TENSOR),  # k_rope
            P(ShardingAxisName.MLP_TENSOR),  # kv_cache
            P(),  # md.seq_lens
            P(),  # md.block_tables
            P(),  # md.query_start_loc
            P(),  # md.request_distribution
        )
        out_specs = (P(None, ShardingAxisName.MLP_TENSOR,
                       None), P(ShardingAxisName.MLP_TENSOR))

        def _mla_ragged_paged_attention(q, q_rope, k, k_rope, cache, *args):
            num_kv_pages_per_block = 4
            num_queries_per_block = 4

            out, new_cache = mla_ragged_paged_attention(
                q,
                q_rope,
                k,
                k_rope,
                cache,
                *args,
                sm_scale=self.scale,
                num_kv_pages_per_block=num_kv_pages_per_block,
                num_queries_per_block=num_queries_per_block)
            return out, new_cache

        output_TNA, new_kv_cache = jax.shard_map(
            _mla_ragged_paged_attention,
            mesh=self.mesh,
            in_specs=in_specs,
            out_specs=out_specs,
            check_vma=False)(q_TNA, q_rope_TNH, kv_SA, k_rope_SH, kv_cache,
                             md.seq_lens, md.block_tables, md.query_start_loc,
                             md.request_distribution)

        # MLA Specific: Project latent output back to TNH
        outputs_TNH = jnp.einsum("TNA,ANH -> TNH", output_TNA,
                                 self.kernel_v_up_proj_ANH.value)

        # Output projection
        outputs_TR = outputs_TNH.reshape(x.shape[0],
                                         self.num_heads * self.v_head_dim)
        o_TD = self.o_proj(outputs_TR)

        return new_kv_cache, o_TD


class DeepseekV2DecoderLayer(JaxModule):

    def __init__(self,
                 config: DeepseekV2Config,
                 dtype: jnp.dtype,
                 rng: nnx.Rngs,
                 mesh: Mesh,
                 kv_cache_dtype: str,
                 quant_config: VllmQuantConfig,
                 layer_idx: int,
                 vllm_config: VllmConfig,
                 prefix: str = ""):
        self.input_layernorm = JaxRmsNorm(
            config.hidden_size,
            epsilon=config.rms_norm_eps,
            dtype=dtype,
            rngs=rng,
            quant_config=quant_config,
            prefix=prefix + ".input_layernorm",
        )
        self.self_attn = DeepseekV2Attention(config=config,
                                             dtype=dtype,
                                             rng=rng,
                                             mesh=mesh,
                                             kv_cache_dtype=kv_cache_dtype,
                                             quant_config=quant_config,
                                             prefix=prefix + ".self_attn")
        self.post_attention_layernorm = JaxRmsNorm(
            config.hidden_size,
            epsilon=config.rms_norm_eps,
            dtype=dtype,
            rngs=rng,
            quant_config=quant_config,
            prefix=prefix + ".post_attention_layernorm",
        )

        if (config.n_routed_experts is not None
                and layer_idx >= config.first_k_dense_replace
                and layer_idx % config.moe_layer_freq == 0):
            self.mlp = DeepseekV2MoE(vllm_config=vllm_config,
                                     rng=rng,
                                     mesh=mesh,
                                     prefix=prefix + ".mlp")
        else:
            self.mlp = DeepseekV2MLP(config=config,
                                     dtype=dtype,
                                     rng=rng,
                                     quant_config=quant_config,
                                     prefix=prefix + ".mlp")

    def __call__(
        self,
        kv_cache: jax.Array,
        x: jax.Array,
        attention_metadata: AttentionMetadata,
    ) -> Tuple[jax.Array, jax.Array]:
        residual = x
        hidden_states = self.input_layernorm(x)
        kv_cache, hidden_states = self.self_attn(kv_cache, hidden_states,
                                                 attention_metadata)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return kv_cache, hidden_states


class DeepseekV2Model(JaxModule):

    def __init__(self,
                 vllm_config: VllmConfig,
                 rng: nnx.Rngs,
                 mesh: Mesh,
                 prefix: str = "model") -> None:
        hf_config: DeepseekV2Config = vllm_config.model_config.hf_config
        vocab_size = vllm_config.model_config.get_vocab_size()
        dtype = vllm_config.model_config.dtype

        self.is_first_rank = get_pp_group().is_first_rank
        self.is_last_rank = get_pp_group().is_last_rank

        if self.is_first_rank:
            self.embed_tokens = JaxEmbed(
                num_embeddings=vocab_size,
                features=hf_config.hidden_size,
                dtype=dtype,
                rngs=rng,
                quant_config=vllm_config.quant_config,
                prefix=prefix + ".embed_tokens",
            )
        else:
            self.embed_tokens = PPMissingLayer()

        self.start_layer, self.end_layer, self.layers = make_layers(
            hf_config.num_hidden_layers,
            lambda layer_index: DeepseekV2DecoderLayer(
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
                hf_config.hidden_size,
                epsilon=hf_config.rms_norm_eps,
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
        input_ids: jax.Array,
        attention_metadata: AttentionMetadata,
        inputs_embeds: Optional[jax.Array] = None,
    ) -> Tuple[List[jax.Array], jax.Array]:
        if self.is_first_rank:
            x = self.embed_tokens(input_ids)
        else:
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


class DeepseekV2ForCausalLM(JaxModule, LoadableWithIterator):

    def __init__(self, vllm_config: VllmConfig, rng_key: jax.Array,
                 mesh: Mesh) -> None:
        self.vllm_config = vllm_config
        rng = nnx.Rngs(rng_key)
        self.mesh = mesh

        self.model = DeepseekV2Model(
            vllm_config=vllm_config,
            rng=rng,
            mesh=mesh,
            prefix="model",
        )

        model_config = vllm_config.model_config
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

    def load_weights(self, weights: Any) -> set[str]:
        loaded_keys = super().load_weights(weights)
        self.initialize_cache()
        return loaded_keys

    def initialize_cache(self):
        # Initialize RoPE caches after weights are loaded.
        for layer in self.model.layers:
            if hasattr(layer, 'self_attn') and hasattr(layer.self_attn,
                                                       'rope'):
                if hasattr(layer.self_attn.rope, 'initialize_cache'):
                    layer.self_attn.rope.initialize_cache(self.mesh)

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
        return self.lm_head(hidden_states)


class DeepseekV3ForCausalLM(DeepseekV2ForCausalLM):
    pass
