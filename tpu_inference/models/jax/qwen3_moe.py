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
from tpu_inference.layers.jax.linear import JaxLinear, JaxLmHead
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


class SharedExpertMLP(JaxModule):

    def __init__(self, config, dtype, rng, quant_config, intermediate_size, prefix=""):
        hidden_size = config.hidden_size
        act = getattr(config, "hidden_act", "silu")

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
        from tpu_inference.layers.jax.layers import FlaxUtils
        self.act_fn = FlaxUtils().ACT2FN[act]

    def __call__(self, x: jax.Array) -> jax.Array:
        gate = self.act_fn(self.gate_proj(x))
        up = self.up_proj(x)
        fuse = gate * up
        return self.down_proj(fuse)


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
        use_hybrid = getattr(vllm_config.sharding_config, "enable_hybrid_moe",
                             False)
        moe_backend = select_moe_backend(use_ep)

        # Router
        self.gate = JaxLinear(
            config.hidden_size,
            config.num_experts,
            dtype=dtype,
            param_dtype=dtype,
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
            self.shared_expert_gate = JaxLinear(
                config.hidden_size,
                1,
                use_bias=False,
                dtype=dtype,
                kernel_init=nnx.with_partitioning(init_fn, (None, "model")),
                rngs=rng,
                quant_config=quant_config,
                prefix=prefix + ".shared_expert_gate",
            )
            self.shared_expert = SharedExpertMLP(
                config=config,
                dtype=dtype,
                rng=rng,
                quant_config=quant_config,
                intermediate_size=shared_expert_intermediate_size,
                prefix=prefix + ".shared_expert",
            )
        else:
            self.shared_expert_gate = None
            self.shared_expert = None

        # Experts (Routed)
        self.enable_return_routed_experts = True
        self.experts = JaxMoE(
            dtype=dtype,
            num_local_experts=config.num_experts,
            hidden_size=config.hidden_size,
            intermediate_size_moe=getattr(config, "moe_intermediate_size", getattr(config, "intermediate_size", 1024)),
            hidden_act=config.hidden_act,
            rngs=rng,
            router=self.gate,
            num_experts_per_tok=config.num_experts_per_tok,
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
            enable_return_routed_experts=self.enable_return_routed_experts,
            prefix=prefix + ".experts")

    def __call__(self, x: jax.Array) -> Tuple[jax.Array, Optional[jax.Array]]:

        out, expert_ids = self.experts(x)

        if self.shared_expert is not None:
            shared_out = self.shared_expert(x)
            if self.shared_expert_gate is not None:
                gate = jax.nn.sigmoid(self.shared_expert_gate(x))
                shared_out = shared_out * gate
            out += shared_out

        return out, expert_ids


class JaxGatedDeltaNetAttention(JaxModule):

    def __init__(self,
                 config: Qwen3Config,
                 dtype: jnp.dtype,
                 rng: nnx.Rngs,
                 mesh: Mesh,
                 quant_config: QuantizationConfig,
                 prefix: str = ""):
        self.config = config
        self.dtype = dtype
        self.mesh = mesh

        text_config = getattr(config, "text_config", config)
        self.hidden_size = config.hidden_size
        self.n_kq = getattr(text_config, "linear_num_key_heads", 16)
        self.n_v = getattr(text_config, "linear_num_value_heads", 32)
        self.d_k = getattr(text_config, "linear_key_head_dim", 128)
        self.d_v = getattr(text_config, "linear_value_head_dim", 128)
        self.kernel_size = getattr(text_config, "linear_conv_kernel_dim", 4)

        key_dim = self.n_kq * self.d_k
        value_dim = self.n_v * self.d_v
        conv_dim = key_dim + key_dim + value_dim

        self.in_proj_qkv = JaxLinear(
            self.hidden_size,
            conv_dim,
            use_bias=False,
            dtype=dtype,
            kernel_init=nnx.with_partitioning(init_fn, (None, "model")),
            rngs=rng,
            quant_config=quant_config,
            prefix=prefix + ".in_proj_qkv",
        )
        self.in_proj_z = JaxLinear(
            self.hidden_size,
            value_dim,
            use_bias=False,
            dtype=dtype,
            kernel_init=nnx.with_partitioning(init_fn, (None, "model")),
            rngs=rng,
            quant_config=quant_config,
            prefix=prefix + ".in_proj_z",
        )
        self.b_proj = JaxLinear(
            self.hidden_size,
            self.n_v,
            use_bias=False,
            dtype=dtype,
            kernel_init=nnx.with_partitioning(init_fn, (None, "model")),
            rngs=rng,
            quant_config=quant_config,
            prefix=prefix + ".b_proj",
        )
        self.a_proj = JaxLinear(
            self.hidden_size,
            self.n_v,
            use_bias=False,
            dtype=dtype,
            kernel_init=nnx.with_partitioning(init_fn, (None, "model")),
            rngs=rng,
            quant_config=quant_config,
            prefix=prefix + ".a_proj",
        )
        self.out_proj = JaxLinear(
            value_dim,
            self.hidden_size,
            use_bias=False,
            dtype=dtype,
            kernel_init=nnx.with_partitioning(init_fn, ("model", None)),
            rngs=rng,
            quant_config=quant_config,
            prefix=prefix + ".out_proj",
        )
        self.norm = JaxRmsNorm(
            self.d_v,
            epsilon=getattr(config, "rms_norm_eps", 1e-6),
            dtype=dtype,
            rngs=rng,
            prefix=prefix + ".norm",
        )

        from tpu_inference.layers.jax.base import create_param
        self.conv1d_weight = create_param(
            rngs=rng,
            shape=(conv_dim, 1, self.kernel_size),
            sharding=(None, None, None),
            dtype=dtype,
        )
        self.conv1d_bias = create_param(
            rngs=rng,
            shape=(conv_dim,),
            sharding=(None,),
            dtype=dtype,
        )
        self.A_log = create_param(
            rngs=rng,
            shape=(self.n_v,),
            sharding=(None,),
            dtype=jnp.float32,
        )
        self.dt_bias = create_param(
            rngs=rng,
            shape=(self.n_v,),
            sharding=(None,),
            dtype=jnp.float32,
        )

    def __call__(
        self,
        gdn_state: Tuple[jax.Array, jax.Array] | jax.Array,
        x: jax.Array,
        attention_metadata: AttentionMetadata,
    ) -> Tuple[Tuple[jax.Array, jax.Array], jax.Array]:
        if isinstance(gdn_state, tuple):
            conv_state, recurrent_state = gdn_state
        else:
            conv_dim = self.n_kq * self.d_k * 2 + self.n_v * self.d_v
            conv_state = jnp.zeros((x.shape[0], self.kernel_size - 1, conv_dim), dtype=x.dtype)
            recurrent_state = jnp.zeros((x.shape[0], self.n_v, self.d_k, self.d_v), dtype=jnp.float32)

        mixed_qkv = self.in_proj_qkv(x)
        b = self.b_proj(x)
        a = self.a_proj(x)

        query_start_loc = getattr(attention_metadata, "query_start_loc", jnp.array([0, x.shape[0]], dtype=jnp.int32))
        num_seqs = query_start_loc.shape[0] - 1
        state_indices = getattr(attention_metadata, "mamba_state_indices", None)
        if state_indices is None:
            state_indices = jnp.arange(num_seqs, dtype=jnp.int32)
        distribution = getattr(attention_metadata, "request_distribution", None)
        if distribution is None:
            distribution = jnp.array([0, x.shape[0], 0], dtype=jnp.int32)
        seq_lens = getattr(attention_metadata, "seq_lens", jnp.ones((x.shape[0],), dtype=jnp.int32))

        from tpu_inference.layers.common.gdn_attention import run_jax_gdn_attention
        (new_conv_state, new_recurrent_state), gdn_out = run_jax_gdn_attention(
            j_mixed_qkv=mixed_qkv,
            j_b=b,
            j_a=a,
            conv_state=conv_state,
            recurrent_state=recurrent_state,
            j_conv_weight=self.conv1d_weight.value if hasattr(self.conv1d_weight, 'value') else self.conv1d_weight,
            j_conv_bias=self.conv1d_bias.value if hasattr(self.conv1d_bias, 'value') else self.conv1d_bias,
            j_A_log=self.A_log.value if hasattr(self.A_log, 'value') else self.A_log,
            j_dt_bias=self.dt_bias.value if hasattr(self.dt_bias, 'value') else self.dt_bias,
            state_indices=state_indices,
            query_start_loc=query_start_loc,
            distribution=distribution,
            seq_lens=seq_lens,
            n_kq=self.n_kq,
            n_v=self.n_v,
            d_k=self.d_k,
            d_v=self.d_v,
            kernel_size=self.kernel_size,
            mesh=self.mesh,
        )

        if gdn_out.ndim == 2:
            orig_shape = gdn_out.shape
            gdn_out = gdn_out.reshape(-1, self.n_v, self.d_v)
            gdn_out = self.norm(gdn_out)
            gdn_out = gdn_out.reshape(orig_shape)
        else:
            gdn_out = self.norm(gdn_out)
        z = self.in_proj_z(x)
        gdn_out = gdn_out * jax.nn.silu(z)

        out = self.out_proj(gdn_out)
        return (new_conv_state, new_recurrent_state), out


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
            dtype=dtype,
            param_dtype=dtype,
            scale_init=nnx.with_partitioning(init_fn, (None, )),
            rngs=rng,
            quant_config=quant_config,
            prefix=prefix + ".input_layernorm",
        )

        layer_types = getattr(config, "layer_types", None)
        if layer_types is not None and layer_idx < len(layer_types):
            self.layer_type = layer_types[layer_idx]
        else:
            interval = getattr(config, "full_attention_interval", 4)
            if (layer_idx + 1) % interval == 0:
                self.layer_type = "full_attention"
            else:
                self.layer_type = "linear_attention"

        if self.layer_type == "linear_attention":
            self.linear_attn = JaxGatedDeltaNetAttention(
                config=config,
                dtype=dtype,
                rng=rng,
                mesh=mesh,
                quant_config=quant_config,
                prefix=prefix + ".linear_attn",
            )
        else:
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
            dtype=dtype,
            scale_init=nnx.with_partitioning(init_fn, (None, )),
            rngs=rng,
            quant_config=quant_config,
            prefix=prefix + ".post_attention_layernorm",
        )

        mlp_only_layers = getattr(config, "mlp_only_layers", [])
        if (layer_idx not in mlp_only_layers) and (
                config.num_experts > 0 and
            (layer_idx + 1) % getattr(config, "decoder_sparse_step", 1) == 0):
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
        kv_cache: jax.Array | Tuple[jax.Array, jax.Array],
        x: jax.Array,
        attention_metadata: AttentionMetadata,
    ) -> Tuple[jax.Array | Tuple[jax.Array, jax.Array], jax.Array, Optional[jax.Array]]:
        hidden_states = self.input_layernorm(x)
        if self.layer_type == "linear_attention":
            kv_cache, attn_output = self.linear_attn(
                kv_cache,
                hidden_states,
                attention_metadata,
            )
        else:
            kv_cache, attn_output = self.self_attn(
                kv_cache,
                hidden_states,
                attention_metadata,
            )
        attn_output += x

        residual = attn_output
        attn_output = self.post_attention_layernorm(attn_output)

        expert_ids = None
        mlp_output = self.mlp(attn_output)
        if isinstance(mlp_output, tuple):
            mlp_output, expert_ids = mlp_output

        outputs = residual + mlp_output

        return kv_cache, outputs, expert_ids


class Qwen3MoeModel(JaxModule):

    def __init__(self,
                 vllm_config: VllmConfig,
                 rng: nnx.Rngs,
                 mesh: Mesh,
                 prefix: str = "") -> None:
        model_config = vllm_config.model_config
        hf_config = getattr(model_config.hf_config, "text_config", model_config.hf_config)
        vocab_size = model_config.get_vocab_size()
        dtype = model_config.dtype
        rms_norm_eps = getattr(hf_config, "rms_norm_eps", getattr(hf_config, "rms_norm_epsilon", getattr(hf_config, "layer_norm_epsilon", 1e-6)))
        hidden_size = hf_config.hidden_size

        self.is_first_rank = get_pp_group().is_first_rank
        self.is_last_rank = get_pp_group().is_last_rank

        if self.is_first_rank or (hf_config.tie_word_embeddings
                                  and self.is_last_rank):
            self.embed_tokens = JaxEmbed(
                num_embeddings=vocab_size,
                features=hidden_size,
                dtype=dtype,
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
                dtype=dtype,
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
    ) -> Tuple[List[jax.Array], jax.Array] | Tuple[List[jax.Array], jax.Array,
                                                   jax.Array]:
        if self.is_first_rank:
            assert inputs_embeds is None
            inputs_embeds = self.embed_tokens(input_ids)
        else:
            assert inputs_embeds is not None

        x = inputs_embeds
        new_kv_caches = []
        all_expert_ids = []
        for i, layer in enumerate(self.layers):
            if isinstance(layer, PPMissingLayer):
                new_kv_caches.append(kv_caches[i])
                continue
            kv_cache = kv_caches[i]
            kv_cache, x, expert_ids = layer(kv_cache, x, attention_metadata)
            if expert_ids is not None:
                all_expert_ids.append(expert_ids)
            new_kv_caches.append(kv_cache)

        if self.is_last_rank:
            x = self.norm(x)

        stacked_expert_ids = jnp.stack(all_expert_ids,
                                       axis=0) if all_expert_ids else None
        return new_kv_caches, x, stacked_expert_ids


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
        hf_config = getattr(model_config.hf_config, "text_config", model_config.hf_config)
        if not hf_config.tie_word_embeddings:
            if self.model.is_last_rank:
                vocab_size = model_config.get_vocab_size()
                hidden_size = hf_config.hidden_size
                self.lm_head = JaxLmHead(
                    hidden_size=hidden_size,
                    vocab_size=vocab_size,
                    dtype=model_config.dtype,
                    param_dtype=model_config.dtype,
                    rngs=rng,
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
               List[jax.Array], Optional[jax.Array]]:
        if not is_first_rank:
            assert intermediate_tensors is not None
            inputs_embeds = intermediate_tensors["hidden_states"]
        kv_caches, x, expert_indices = self.model(
            kv_caches,
            input_ids,
            attention_metadata,
            inputs_embeds,
        )

        if not is_last_rank:
            x = JaxIntermediateTensors(tensors={"hidden_states": x}, )

        return kv_caches, x, [], expert_indices

    def compute_logits(self, hidden_states: jax.Array) -> jax.Array:
        if hasattr(self, 'lm_head'):
            return self.lm_head(hidden_states)

        assert isinstance(self.model.embed_tokens, JaxEmbed)
        return self.model.embed_tokens.decode(hidden_states)
