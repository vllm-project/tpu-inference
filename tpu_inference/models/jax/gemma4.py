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
from typing import Any, Iterable, List, Optional, Tuple

import jax
import jax.numpy as jnp
from flax import nnx
from jax.sharding import Mesh
from transformers import Gemma4TextConfig
from vllm.config import VllmConfig

from tpu_inference import utils
from tpu_inference.distributed.jax_parallel_state import get_pp_group
from tpu_inference.layers.common.attention_interface import attention
from tpu_inference.layers.common.attention_metadata import AttentionMetadata
from tpu_inference.layers.common.moe import MoEBackend
from tpu_inference.layers.common.quantization import quantize_kv
from tpu_inference.layers.common.sharding import ShardingAxisName
from tpu_inference.layers.jax import JaxModule
from tpu_inference.layers.jax.embed import JaxEmbed
from tpu_inference.layers.jax.linear import JaxEinsum, JaxLinear
from tpu_inference.layers.jax.moe.moe import JaxMoE
from tpu_inference.layers.jax.norm import JaxRmsNorm
from tpu_inference.layers.jax.pp_utils import PPMissingLayer, make_layers
from tpu_inference.layers.jax.rope_interface import apply_rope
from tpu_inference.layers.vllm.quantization.configs import VllmQuantConfig
from tpu_inference.logger import init_logger
from tpu_inference.models.jax.jax_intermediate_tensor import \
    JaxIntermediateTensors
from tpu_inference.models.jax.utils.weight_utils import (
    LoadableWithIterator, StandardWeightLoader,
    load_nnx_param_from_reshaped_torch)

logger = init_logger(__name__)

init_fn = nnx.initializers.uniform()


# MLP arch is the same as Gemma3
class Gemma4MLP(JaxModule):

    def __init__(self,
                 config: Gemma4TextConfig,
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
            param_dtype=dtype,
            kernel_init=nnx.with_partitioning(init_fn, (None, "model")),
            rngs=rng,
            quant_config=quant_config,
            prefix=prefix + ".gate_proj",
        )
        self.up_proj = JaxLinear(
            hidden_size,
            intermediate_size,
            use_bias=False,
            param_dtype=dtype,
            kernel_init=nnx.with_partitioning(init_fn, (None, "model")),
            rngs=rng,
            quant_config=quant_config,
            prefix=prefix + ".up_proj",
        )
        self.down_proj = JaxLinear(
            intermediate_size,
            hidden_size,
            use_bias=False,
            param_dtype=dtype,
            kernel_init=nnx.with_partitioning(init_fn, ("model", None)),
            rngs=rng,
            quant_config=quant_config,
            prefix=prefix + ".down_proj",
        )
        self.act_fn = partial(nnx.gelu, approximate=True)

    def __call__(self, x: jax.Array) -> jax.Array:
        gate = self.act_fn(self.gate_proj(x))
        up = self.up_proj(x)
        fuse = gate * up
        result = self.down_proj(fuse)
        return result


class Gemma4Router(JaxModule):
    """Router for Gemma4 MoE that preprocesses input before projection.

    Applies RMSNorm (no learned weight), root_size scaling
    (hidden_size^{-0.5}), then a learned per-dimension scale before
    projecting to expert logits.

    This preprocessing is applied ONLY to the router's input, not to
    the expert MLPs' input.
    """

    def __init__(
        self,
        config: Gemma4TextConfig,
        dtype,
        rngs: nnx.Rngs,
        quant_config,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size: int = config.hidden_size

        # RMSNorm without learned weight — pure normalization only
        self.norm = JaxRmsNorm(self.hidden_size,
                               epsilon=config.rms_norm_eps,
                               use_scale=False,
                               rngs=rngs,
                               quant_config=quant_config,
                               prefix=prefix + ".norm")
        # Per-dimension learned scale, applied after norm + root_size
        self.scale = nnx.Param(init_fn(rngs.params(), (self.hidden_size, ),
                                       dtype),
                               eager_sharding=False)
        # Constant 1/sqrt(hidden_size) scaling factor
        self.root_size = self.hidden_size**-0.5
        # Project to expert logits; replicated across TP for consistent routing
        self.proj = JaxLinear(
            self.hidden_size,
            config.num_experts,
            rngs=rngs,
            use_bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.proj",
        )
        self.per_expert_scale = nnx.Param(init_fn(rngs.params(),
                                                  (config.num_experts, ),
                                                  dtype),
                                          eager_sharding=False)

    def __call__(self, x: jax.Array) -> jax.Array:
        """Returns raw router logits [T, E]."""
        x = self.norm(x)
        x = x * self.root_size
        x = x * self.scale.value.astype(x.dtype)
        router_logits = self.proj(x)
        return router_logits


class Gemma4MoE(JaxMoE):
    """Mixture of Experts for Gemma4 using FusedMoE.

    Wraps FusedMoE with custom routing. The router projection is
    external (Gemma4Router) — this class only handles expert dispatch.

    Gemma4 routing: softmax over ALL experts → top-k → renormalize.
    per_expert_scale is folded into routing weights for mathematical
    correctness with FusedMoE's fused kernel.
    """

    def __init__(
        self,
        config: Gemma4TextConfig,
        dtype,
        mesh,
        rngs: nnx.Rngs,
        quant_config,
        prefix: str = "",
    ) -> None:
        noop_router = JaxModule()
        noop_router.num_experts_per_tok = config.top_k_experts

        # FusedMoE experts with custom Gemma4 routing
        JaxMoE.__init__(
            self,
            dtype=dtype,
            num_local_experts=config.num_experts,
            hidden_size=config.hidden_size,
            intermediate_size_moe=config.moe_intermediate_size,
            hidden_act="gelu",
            rngs=rngs,
            router=noop_router,
            mesh=mesh,
            activation_ffw_td=(ShardingAxisName.MLP_DATA, None),
            activation_ffw_ted=(ShardingAxisName.MLP_DATA, None, None),
            edf_sharding=(None, None, None),
            efd_sharding=(None, None, None),
            apply_expert_weight_before_computation=False,
            expert_axis_name=None,
            # Disable EP for MVP, can enable later if needed
            # TODO: Enable EP
            num_expert_parallelism=1,
            moe_backend=MoEBackend.GMM_TP,
            scoring_func=
            "softmax",  # vLLM implementation has a custom routing function, here we just use "softmax" for MVP
            renormalize=True,
            enable_return_routed_experts=True,
            num_experts_per_tok=config.top_k_experts,
            quant_config=quant_config,
            prefix=prefix)

    def load_weights(self, weights: Iterable):
        """Load weights for Gemma4 MoE layer.

        Unlike other MoE, Gemma4 didn't provide per-expert weights, but already fuse projection weight in the checkpoint.
        """
        loaded = set()
        for name, tensor in weights:
            if name.endswith("down_proj"):
                load_nnx_param_from_reshaped_torch(self.kernel_down_proj_EFD,
                                                   tensor,
                                                   permute_dims=(0, 2, 1),
                                                   param_name=name)
                loaded.add("kernel_down_proj_EFD")
                self.kernel_down_proj_EFD._weights_to_load.clear()
                # Other MoE models store expert weights in shape (D, F) and permute in *FusedMoEMethod.process_weights_after_loading.
                # For compatibility, we permute here then expect another permute in process_weights_after_loading.
                self.kernel_down_proj_EFD.value = jnp.swapaxes(
                    self.kernel_down_proj_EFD.value, 1, 2)
            elif name.endswith("gate_up_proj"):
                F = tensor.shape[1] // 2
                load_nnx_param_from_reshaped_torch(self.kernel_gating_EDF,
                                                   tensor[:, :F, :],
                                                   permute_dims=(0, 2, 1),
                                                   param_name=name)
                load_nnx_param_from_reshaped_torch(self.kernel_up_proj_EDF,
                                                   tensor[:, F:, :],
                                                   permute_dims=(0, 2, 1),
                                                   param_name=name)
                loaded.add("kernel_up_proj_EDF")
                self.kernel_up_proj_EDF._weights_to_load.clear()
                loaded.add("kernel_gating_EDF")
                self.kernel_gating_EDF._weights_to_load.clear()
                # Other MoE models store expert weights in shape (F, D) and permute in *FusedMoEMethod.process_weights_after_loading.
                # For compatibility, we permute here then expect another permute in process_weights_after_loading.
                self.kernel_up_proj_EDF.value = jnp.swapaxes(
                    self.kernel_up_proj_EDF.value, 1, 2)
                self.kernel_gating_EDF.value = jnp.swapaxes(
                    self.kernel_gating_EDF.value, 1, 2)
        return loaded


class Gemma4Attention(JaxModule):

    def __init__(self,
                 config: Gemma4TextConfig,
                 layer_idx: int,
                 dtype: jnp.dtype,
                 rng: nnx.Rngs,
                 mesh: Mesh,
                 kv_cache_dtype: str,
                 quant_config: VllmQuantConfig,
                 prefix: str = ""):
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.rms_norm_eps = config.rms_norm_eps

        # Assuming Gemma 4 also uses a custom scalar, not 1/sqrt(head_dim)
        self.scaling = 1.0

        # Same as Gemma3: use layer_idx to handle GLOBAL/LOCAL layer
        self.layer_type = "full_attention"
        if hasattr(config, "layer_types") and layer_idx < len(
                config.layer_types):
            self.layer_type = config.layer_types[layer_idx]

        self.is_sliding = self.layer_type == "sliding_attention"
        self.sliding_window = config.sliding_window if self.is_sliding else None

        # Gemma4 use partial rope (0.25) in GLOBAL layer
        rope_parameters = getattr(config, "rope_parameters", {})
        if self.layer_type in rope_parameters:
            # Transformers v5 rope config.
            rope_parameters = rope_parameters[self.layer_type]
            self.rope_theta = rope_parameters.get(
                "rope_theta", getattr(config, "rope_theta", 10000.0))
            self.rope_scaling = rope_parameters.get(
                "rope_scaling", getattr(config, "rope_scaling", None))
            self.rope_proportion = rope_parameters.get("partial_rotary_factor",
                                                       1.0)
        else:
            # Transformers v4 rope config.
            # Fallback for config backward compatibility
            self.rope_theta = config.rope_local_base_freq if self.is_sliding else config.rope_theta
            self.rope_scaling = getattr(config, "rope_scaling", None)
            self.rope_proportion = 0.25 if not self.is_sliding else 1.0

        # Gemma4: use different num_kv_heads and head_dim in GLOBAL/LOCAL layers
        if not self.is_sliding:
            # GLOBAL layers
            self.head_dim_original = config.global_head_dim
        else:
            # LOCAL layers
            self.head_dim_original = config.head_dim

        # Determine if this full-attention layer uses k_eq_v
        use_k_eq_v = ((not self.is_sliding)
                      and getattr(config, "attention_k_eq_v", False))
        if use_k_eq_v:
            self.num_kv_heads = config.num_global_key_value_heads or config.num_key_value_heads
        else:
            self.num_kv_heads = config.num_key_value_heads

        self.head_dim = utils.get_padded_head_dim(self.head_dim_original)

        self.mesh = mesh

        self.q_proj = JaxEinsum(
            "TD,DNH->TNH",
            (self.hidden_size, self.num_heads, self.head_dim),
            bias_shape=(self.num_heads,
                        self.head_dim) if config.attention_bias else None,
            param_dtype=dtype,
            kernel_init=nnx.with_partitioning(init_fn, (None, "model", None)),
            bias_init=nnx.with_partitioning(init_fn, ("model", None))
            if config.attention_bias else None,
            rngs=rng,
            quant_config=quant_config,
            prefix=prefix + ".q_proj",
        )
        self.q_norm = JaxRmsNorm(
            self.head_dim,
            epsilon=self.rms_norm_eps,
            param_dtype=dtype,
            scale_init=nnx.with_partitioning(init_fn, (None, )),
            rngs=rng,
            quant_config=quant_config,
            prefix=prefix + ".q_norm",
        )

        self.k_proj = JaxEinsum(
            "TD,DKH->TKH",
            (self.hidden_size, self.num_kv_heads, self.head_dim),
            bias_shape=(self.num_kv_heads,
                        self.head_dim) if config.attention_bias else None,
            param_dtype=dtype,
            kernel_init=nnx.with_partitioning(init_fn, (None, None, "model")),
            bias_init=nnx.with_partitioning(init_fn, (None, "model"))
            if config.attention_bias else None,
            rngs=rng,
            quant_config=quant_config,
            prefix=prefix + ".k_proj",
        )
        # --- Shared KV Projection Logic ---
        if use_k_eq_v:
            self.v_proj = None
        else:
            self.v_proj = JaxEinsum(
                "TD,DKH->TKH",
                (self.hidden_size, self.num_kv_heads, self.head_dim),
                bias_shape=(self.num_kv_heads,
                            self.head_dim) if config.attention_bias else None,
                param_dtype=dtype,
                kernel_init=nnx.with_partitioning(init_fn,
                                                  (None, None, "model")),
                bias_init=nnx.with_partitioning(init_fn, (None, "model"))
                if config.attention_bias else None,
                rngs=rng,
                quant_config=quant_config,
                prefix=prefix + ".v_proj",
            )

        self.k_norm = JaxRmsNorm(
            self.head_dim,
            epsilon=self.rms_norm_eps,
            param_dtype=dtype,
            scale_init=nnx.with_partitioning(init_fn, (None, )),
            rngs=rng,
            quant_config=quant_config,
            prefix=prefix + ".k_norm",
        )
        # V norm: no learnable scale (pure normalization only)
        self.v_norm = JaxRmsNorm(
            self.head_dim,
            epsilon=self.rms_norm_eps,
            param_dtype=dtype,
            use_scale=False,
            scale_init=None,
            rngs=rng,
            quant_config=quant_config,
            prefix=prefix + ".v_norm",
        )
        self.o_proj = JaxEinsum(
            "TNH,NHD->TD",
            (self.num_heads, self.head_dim, self.hidden_size),
            bias_shape=(self.hidden_size, ) if config.attention_bias else None,
            param_dtype=dtype,
            kernel_init=nnx.with_partitioning(init_fn, (None, "model", None)),
            bias_init=nnx.with_partitioning(init_fn, (None, ))
            if config.attention_bias else None,
            rngs=rng,
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

        num_kv_shared_layers = getattr(config, "num_kv_shared_layers", 0)
        assert num_kv_shared_layers == 0, "Expect no shared layers"
        self.is_kv_shared_layer = False

    def __call__(
        self,
        kv_cache: Optional[jax.Array],
        x: jax.Array,
        attention_metadata: AttentionMetadata,
    ) -> Tuple[jax.Array, jax.Array]:
        md = attention_metadata
        k = self.k_proj(x)
        if self.v_proj is None:
            v = k
        else:
            v = self.v_proj(x)
        # q: (T, N, H)
        q = self.q_proj(x)
        # Q norm (always applied)
        q = self.q_norm(q)

        if not self.is_kv_shared_layer:
            # Non-shared: apply K norm + RoPE, V norm
            k = self.k_norm(k)
            q = apply_rope(q,
                           md.input_positions,
                           self.head_dim_original,
                           self.rope_theta,
                           self.rope_scaling,
                           rope_proportion=self.rope_proportion)
            k = apply_rope(k,
                           md.input_positions,
                           self.head_dim_original,
                           self.rope_theta,
                           self.rope_scaling,
                           rope_proportion=self.rope_proportion)

            v = self.v_norm(v)
        else:
            raise NotImplementedError("Expect no shared layers")

        q_scale = k_scale = v_scale = None
        if self.kv_cache_quantized_dtype:
            # q_scale = self._q_scale
            k_scale = self._k_scale
            v_scale = self._v_scale
            k, v = quantize_kv(self.kv_cache_quantized_dtype, k, v, k_scale,
                               v_scale)
        new_kv_cache, outputs = attention(
            kv_cache,
            q,
            k,
            v,
            attention_metadata,
            self.mesh,
            self.head_dim_original,
            sm_scale=self.scaling,
            attention_chunk_size=self.sliding_window,
            q_scale=q_scale,
            k_scale=k_scale,
            v_scale=v_scale,
        )
        # (T, D)
        o = self.o_proj(outputs)
        return new_kv_cache, o


class Gemma4DecoderLayer(JaxModule):

    def __init__(self,
                 config,
                 layer_idx: int,
                 dtype: jnp.dtype,
                 rng: nnx.Rngs,
                 mesh: Mesh,
                 kv_cache_dtype: str,
                 quant_config: VllmQuantConfig,
                 prefix: str = ""):
        text_config: Gemma4TextConfig = config.hf_config.text_config
        rms_norm_eps = text_config.rms_norm_eps
        hidden_size = text_config.hidden_size

        # Same as Gemma3: use layer_idx to handle GLOBAL/LOCAL layer
        self.layer_type = "full_attention"
        if hasattr(text_config, "layer_types") and layer_idx < len(
                text_config.layer_types):
            self.layer_type = text_config.layer_types[layer_idx]

        self.is_sliding = self.layer_type == "sliding_attention"

        self.layer_scalar = nnx.Param(jnp.ones((1, ), dtype=dtype))

        self.input_layernorm = JaxRmsNorm(
            hidden_size,
            epsilon=rms_norm_eps,
            param_dtype=dtype,
            scale_init=nnx.with_partitioning(init_fn, (None, )),
            rngs=rng,
            quant_config=quant_config,
            prefix=prefix + ".input_layernorm",
        )
        self.self_attn = Gemma4Attention(config=text_config,
                                         layer_idx=layer_idx,
                                         dtype=dtype,
                                         rng=rng,
                                         mesh=mesh,
                                         kv_cache_dtype=kv_cache_dtype,
                                         quant_config=quant_config,
                                         prefix=prefix + ".self_attn")
        self.post_attention_layernorm = JaxRmsNorm(
            hidden_size,
            epsilon=rms_norm_eps,
            dtype=dtype,
            param_dtype=dtype,
            scale_init=nnx.with_partitioning(init_fn, (None, )),
            rngs=rng,
            quant_config=quant_config,
            prefix=prefix + ".post_attention_layernorm",
        )
        self.pre_feedforward_layernorm = JaxRmsNorm(
            hidden_size,
            epsilon=rms_norm_eps,
            dtype=dtype,
            param_dtype=dtype,
            scale_init=nnx.with_partitioning(init_fn, (None, )),
            rngs=rng,
            quant_config=quant_config,
            prefix=prefix + ".pre_feedforward_layernorm",
        )
        self.mlp = Gemma4MLP(
            config=text_config,
            dtype=dtype,
            rng=rng,
            quant_config=quant_config,
            prefix=prefix + ".mlp",
        )
        self.post_feedforward_layernorm = JaxRmsNorm(
            hidden_size,
            epsilon=rms_norm_eps,
            dtype=dtype,
            param_dtype=dtype,
            scale_init=nnx.with_partitioning(init_fn, (None, )),
            rngs=rng,
            quant_config=quant_config,
            prefix=prefix + ".post_feedforward_layernorm",
        )

        # MoE (Mixture of Experts) — router + expert block parallel to MLP
        self.enable_moe_block = getattr(text_config, "enable_moe_block", False)
        if self.enable_moe_block:
            self.router = Gemma4Router(
                config=text_config,
                dtype=dtype,
                rngs=rng,
                quant_config=quant_config,
                prefix=prefix + ".router",
            )
            self.experts = Gemma4MoE(config=text_config,
                                     dtype=dtype,
                                     mesh=mesh,
                                     rngs=rng,
                                     quant_config=quant_config,
                                     prefix=prefix + ".experts")
            self.post_feedforward_layernorm_1 = JaxRmsNorm(
                text_config.hidden_size,
                epsilon=text_config.rms_norm_eps,
                dtype=dtype,
                param_dtype=dtype,
                rngs=rng,
                quant_config=quant_config,
                prefix=prefix + ".post_feedforward_layernorm_1")
            self.post_feedforward_layernorm_2 = JaxRmsNorm(
                text_config.hidden_size,
                epsilon=text_config.rms_norm_eps,
                dtype=dtype,
                param_dtype=dtype,
                rngs=rng,
                quant_config=quant_config,
                prefix=prefix + ".post_feedforward_layernorm_2")
            self.pre_feedforward_layernorm_2 = JaxRmsNorm(
                text_config.hidden_size,
                epsilon=text_config.rms_norm_eps,
                dtype=dtype,
                param_dtype=dtype,
                rngs=rng,
                quant_config=quant_config,
                prefix=prefix + ".pre_feedforward_layernorm_2")
        else:
            self.router = None
            self.moe = None
            self.post_feedforward_layernorm_1 = None
            self.post_feedforward_layernorm_2 = None
            self.pre_feedforward_layernorm_2 = None

    def __call__(
        self,
        kv_cache: jax.Array,
        x: jax.Array,
        attention_metadata: AttentionMetadata,
    ) -> Tuple[jax.Array, jax.Array, Optional[jax.Array]]:
        residual = x
        hidden_states = self.input_layernorm(x)
        kv_cache, attn_output = self.self_attn(
            kv_cache,
            hidden_states,
            attention_metadata,
        )
        attn_output = self.post_attention_layernorm(attn_output)
        hidden_states = residual + attn_output
        residual = hidden_states

        expert_ids = None
        if self.enable_moe_block:
            # Dense MLP branch
            hidden_states_1 = self.pre_feedforward_layernorm(hidden_states)
            hidden_states_1 = self.mlp(hidden_states_1)
            hidden_states_1 = self.post_feedforward_layernorm_1(
                hidden_states_1)

            # MoE branch: router sees raw hidden_states (applies its own
            # norm + scale internally); experts see separately normed input
            router_logits = self.router(hidden_states)
            hidden_states_2 = self.pre_feedforward_layernorm_2(hidden_states)
            hidden_states_2, expert_ids = self.experts(hidden_states_2,
                                                       router_logits)
            hidden_states_2 = self.post_feedforward_layernorm_2(
                hidden_states_2)

            # Combine branches
            hidden_states = hidden_states_1 + hidden_states_2
        else:
            # Dense MLP
            hidden_states = self.pre_feedforward_layernorm(residual)
            hidden_states = self.mlp(hidden_states)

        mlp_output = self.post_feedforward_layernorm(hidden_states)
        outputs = residual + mlp_output

        outputs = outputs * self.layer_scalar.value

        return kv_cache, outputs, expert_ids


class Gemma4Model(JaxModule):

    def __init__(self,
                 vllm_config: VllmConfig,
                 rng: nnx.Rngs,
                 mesh: Mesh,
                 prefix: str = "model") -> None:
        model_config = vllm_config.model_config
        hf_config = model_config.hf_config
        text_config = hf_config.text_config
        vocab_size = model_config.get_vocab_size()
        dtype = model_config.dtype
        rms_norm_eps = text_config.rms_norm_eps
        hidden_size = text_config.hidden_size

        self.is_first_rank = get_pp_group().is_first_rank
        self.is_last_rank = get_pp_group().is_last_rank

        # Gemma 4: Embeddings are scaled by sqrt(hidden_size)
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
            lambda layer_index: Gemma4DecoderLayer(
                config=model_config,
                layer_idx=layer_index,
                dtype=dtype,
                rng=rng,
                mesh=mesh,
                kv_cache_dtype=vllm_config.cache_config.cache_dtype,
                quant_config=vllm_config.quant_config,
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
        layer_name_to_kv_cache: Optional[dict] = None,
    ) -> Tuple[List[jax.Array], jax.Array, Optional[jax.Array]]:

        if inputs_embeds is not None:
            x = inputs_embeds
        else:
            x = self.embed_tokens(input_ids)
            # Gemma4: Apply embedding scaling
            x = x * self.embedding_scale

        all_expert_ids = []
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
            kv_cache, x, expert_ids = layer(
                kv_cache,
                x,
                layer_attn_metadata,
            )
            if expert_ids is not None:
                all_expert_ids.append(expert_ids)
            kv_caches[cache_idx] = kv_cache
        x = self.norm(x)
        stacked_expert_ids = jnp.stack(all_expert_ids,
                                       axis=0) if all_expert_ids else None
        return kv_caches, x, stacked_expert_ids


class Gemma4ForCausalLM(JaxModule, LoadableWithIterator):
    WeightLoader = StandardWeightLoader

    def __init__(self, vllm_config: VllmConfig, rng_key: jax.Array,
                 mesh: Mesh) -> None:
        self.vllm_config = vllm_config
        rng = nnx.Rngs(rng_key)
        self.mesh = mesh

        self.model = Gemma4Model(
            vllm_config=vllm_config,
            rng=rng,
            mesh=mesh,
            prefix="model",
        )
        model_config = vllm_config.model_config

        # Gemma 4: soft-capping in the final logits.
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
        allowed_layers = set(f"layers.{i}."
                             for i in range(len(self.model.layers)))
        stripped_weights = (
            (clean_name, tensor) for name, tensor in weights
            if (clean_name := name.replace("language_model.", "")).startswith((
                "model.", "lm_head")) and
            "vision" not in clean_name  # Exclude vision tower weights for now
        )
        return super().load_weights(
            (name, tensor) for name, tensor in stripped_weights
            if not ("layers." in name and not any(
                layer_prefix in name for layer_prefix in allowed_layers)))

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
            x = JaxIntermediateTensors(tensors={"hidden_states": x}, )

        return kv_caches, x, [], expert_indices

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
