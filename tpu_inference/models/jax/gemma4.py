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
from tpu_inference.layers.common.quantization import quantize_kv
from tpu_inference.layers.common.sharding import ShardingAxisName
from tpu_inference.layers.jax import JaxModule
from tpu_inference.layers.jax.embed import JaxEmbed
from tpu_inference.layers.jax.linear import (JaxEinsum, JaxLinear, JaxLmHead,
                                             JaxMergedColumnParallelLinear,
                                             JaxQKVParallelLinear)
from tpu_inference.layers.jax.moe.moe import JaxRoutedExperts
from tpu_inference.layers.jax.norm import JaxRmsNorm
from tpu_inference.layers.jax.pp_utils import PPMissingLayer, make_layers
from tpu_inference.layers.jax.rope_interface import (apply_rope,
                                                     normalize_rope_scaling)
from tpu_inference.layers.vllm.quantization.configs import VllmQuantConfig
from tpu_inference.logger import init_logger
from tpu_inference.models.common.kv_share import compute_kv_share_map
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
                 intermediate_size: int,
                 prefix: str = ""):
        hidden_size = config.hidden_size
        # `intermediate_size` is the per-layer MLP width. KV-shared layers
        # use 2x config.intermediate_size when text_config.use_double_wide_mlp
        # is set (matches vllm-pytorch `Gemma4DecoderLayer.__init__`'s
        # `layer_intermediate_size` computation); otherwise it's just
        # config.intermediate_size. The caller computes this in
        # Gemma4DecoderLayer and passes it in explicitly.

        self.gate_up_proj = JaxMergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            use_bias=False,
            param_dtype=dtype,
            kernel_init=nnx.with_partitioning(init_fn, (None, "model")),
            rngs=rng,
            quant_config=quant_config,
            prefix=prefix + ".gate_proj",
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
        gate_up = self.gate_up_proj(x)
        gate, up = jnp.split(gate_up, 2, axis=-1)
        gate = self.act_fn(gate)
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
        x = x * self.scale.get_value().astype(x.dtype)
        router_logits = self.proj(x)
        return router_logits


class Gemma4MoE(JaxRoutedExperts):
    """Mixture of Experts for Gemma4 using FusedMoE.

    The router projection is external (Gemma4Router); this class only
    handles expert dispatch.  use_ep and moe_backend are derived from
    the vLLM parallel config by JaxRoutedExperts so the EP/TP backend
    matches the torchax path automatically.
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
        JaxRoutedExperts.__init__(
            self,
            dtype=dtype,
            num_local_experts=config.num_experts,
            hidden_size=config.hidden_size,
            intermediate_size_moe=config.moe_intermediate_size,
            hidden_act="gelu",
            rngs=rngs,
            mesh=mesh,
            top_k=config.top_k_experts,
            scoring_func="softmax",
            renormalize=True,
            enable_return_routed_experts=True,
            quant_config=quant_config,
            prefix=prefix)

    def load_weights(self, weights: Iterable):
        """Load weights for Gemma4 MoE layer.

        Unlike other MoE, Gemma4 didn't provide per-expert weights, but
        already consolidates each projection weights into a single tensor
        stacked along the expert axis
        — e.g. `down_proj` is `(E, D, F)` rather than separate
        per-expert `(D, F)` tensors in. The generic per-expert loader
        (`JaxMoE._load_weights` / `Fp8FusedMoEMethod.load_weights`) expects
        the latter, keyed as `"<expert_id>.<param_name>"`. Slice each stacked
        tensor into per-expert pieces and synthesize that naming, then
        delegate to super().

        Per-expert slices are handed over as-is (no transpose): the generic
        loader does no permute itself (just adds the expert dim back via
        reshape), and `*FusedMoEMethod.process_weights_after_loading`
        concatenates gate/up scale halves along the same axis it
        concatenates the gate/up weight halves — so the checkpoint's native
        per-expert orientation already lines up for both weights and their
        scales, for the same reason the un-permuted raw weights do.
        """

        def per_expert_slice(stacked_tensor, param_name: str):
            return ((f"{i}.{param_name}", expert_tensor)
                    for i, expert_tensor in enumerate(stacked_tensor))

        synthesized = []
        for name, tensor in weights:
            if name.endswith("down_proj"):
                synthesized.extend(per_expert_slice(tensor,
                                                    "down_proj.weight"))
            elif name.endswith("gate_up_proj"):
                F = tensor.shape[1] // 2
                synthesized.extend(
                    per_expert_slice(tensor[:, :F, :], "gate_proj.weight"))
                synthesized.extend(
                    per_expert_slice(tensor[:, F:, :], "up_proj.weight"))

        return super().load_weights(synthesized)


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
            rope_scaling = rope_parameters.get(
                "rope_scaling", None) or getattr(config, "rope_scaling", None)
            if rope_scaling is None and "rope_type" in rope_parameters:
                rope_scaling = rope_parameters
            self.rope_scaling = normalize_rope_scaling(rope_scaling)
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

        # Shard k/v projections along the kv-heads dimension when num_kv_heads
        # is divisible by tp_size — this matches the ragged-paged-attention
        # kernel's expected sharding (P(ATTN_DATA, ATTN_HEAD, None)) and avoids
        # XLA inserting all-to-all reshuffles every layer. When num_kv_heads <
        # tp_size (e.g. global layers with k_eq_v + num_global_key_value_heads
        # = 4 at TP=8), fall back to sharding the head_dim axis; the kernel
        # replicates kv-heads internally for that case.
        _tp_size = utils.get_mesh_shape_product(mesh, ShardingAxisName.MODEL)
        _shard_kv_on_k = (_tp_size <= 1) or (self.num_kv_heads % _tp_size == 0)
        if not _shard_kv_on_k:
            logger.warning_once(
                f"num_kv_heads={self.num_kv_heads} is not divisible by TP size {_tp_size}, "
                "sharding k/v projections on head_dim instead of kv-heads. This may cause "
                "all-to-all communication overhead.")
        _kv_kernel_spec = (None, "model",
                           None) if _shard_kv_on_k else (None, None, "model")
        _kv_bias_spec = ("model", None) if _shard_kv_on_k else (None, "model")

        if use_k_eq_v:  # TODO: Add QKV fusion logic for k == v case.
            self.qkv_proj = None
            self.q_proj = JaxEinsum(
                "TD,DNH->TNH",
                (self.hidden_size, self.num_heads, self.head_dim),
                bias_shape=(self.num_heads,
                            self.head_dim) if config.attention_bias else None,
                param_dtype=dtype,
                kernel_init=nnx.with_partitioning(init_fn,
                                                  (None, "model", None)),
                bias_init=nnx.with_partitioning(init_fn, ("model", None))
                if config.attention_bias else None,
                rngs=rng,
                quant_config=quant_config,
                prefix=prefix + ".q_proj",
            )
            self.k_proj = JaxEinsum(
                "TD,DKH->TKH",
                (self.hidden_size, self.num_kv_heads, self.head_dim),
                bias_shape=(self.num_kv_heads,
                            self.head_dim) if config.attention_bias else None,
                param_dtype=dtype,
                kernel_init=nnx.with_partitioning(init_fn, _kv_kernel_spec),
                bias_init=nnx.with_partitioning(init_fn, _kv_bias_spec)
                if config.attention_bias else None,
                rngs=rng,
                quant_config=quant_config,
                prefix=prefix + ".k_proj",
            )
            self.v_proj = None
        else:
            self.qkv_proj = JaxQKVParallelLinear(
                hidden_size=self.hidden_size,
                num_heads=self.num_heads,
                num_kv_heads=self.num_kv_heads,
                head_dim=self.head_dim,
                use_bias=config.attention_bias,
                dtype=dtype,
                rngs=rng,
                quant_config=quant_config,
                prefix=prefix,
            )
            self.q_proj = None
            self.k_proj = None
            self.v_proj = None

        self.q_norm = JaxRmsNorm(
            self.head_dim,
            epsilon=self.rms_norm_eps,
            param_dtype=dtype,
            scale_init=nnx.with_partitioning(init_fn, (None, )),
            rngs=rng,
            quant_config=quant_config,
            prefix=prefix + ".q_norm",
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
            kernel_init=nnx.with_partitioning(init_fn, ("model", None, None)),
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

        # KV-cache sharing (mirrors vllm-pytorch `Gemma4Attention.__init__`
        # KV-share derivation). Layers in the last `num_kv_shared_layers`
        # reuse K/V from earlier layers of matching attention type. The
        # runner side populates the redirect mapping; this layer just needs
        # to set is_kv_shared_layer + kv_sharing_target_layer_name.
        kv_share_map = compute_kv_share_map(config)
        self.is_kv_shared_layer = layer_idx in kv_share_map
        self.kv_sharing_target_layer_name: Optional[str] = None
        if self.is_kv_shared_layer:
            # The runner uses unprefixed "layer.{i}" keys; this string must
            # match the keys produced by KVCacheManager's spec-creation loop
            # and Gemma4Model's layer-name iteration.
            self.kv_sharing_target_layer_name = (
                f"layer.{kv_share_map[layer_idx]}")

    def __call__(
        self,
        kv_cache: Optional[jax.Array],
        x: jax.Array,
        attention_metadata: AttentionMetadata,
    ) -> Tuple[jax.Array, jax.Array]:
        md = attention_metadata
        if self.qkv_proj is not None:
            q, k, v = self.qkv_proj(x)
        else:
            k = self.k_proj(x)
            v = k
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
            # KV-shared branch (mirrors the `else: # Shared: only apply RoPE
            # to Q` branch of vllm-pytorch `Gemma4Attention.forward`):
            # Only Q gets RoPE. K and V are NOT normalized and NOT RoPE-rotated.
            # The cache slot (redirected by the runner)
            # holds the source layer's already-normed-and-roped K/V for ALL
            # positions (source's call ran first in the same step and wrote
            # them). We pass update_kv_cache=False so the kernel both (a) does
            # not overwrite the source slot with our raw k,v and (b) reads
            # everything from cache rather than mixing in the layer's own
            # input k,v. Shared's input k,v is therefore unused for attention
            # math; we still allocate q_proj/k_proj/v_proj because gemma-4's
            # checkpoint stores full Q/K/V weights for shared layers.
            q = apply_rope(q,
                           md.input_positions,
                           self.head_dim_original,
                           self.rope_theta,
                           self.rope_scaling,
                           rope_proportion=self.rope_proportion)

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
            update_kv_cache=not self.is_kv_shared_layer,
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

        # PLE (Per-Layer Embedding) — per-layer modules in the decoder.
        # Active when hidden_size_per_layer_input > 0 (E2B/E4B: 256; 26B/31B: 0).
        self.hidden_size_per_layer_input = getattr(
            text_config, "hidden_size_per_layer_input", 0)

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
        # Double-wide MLP: KV-shared layers get 2x intermediate_size when
        # text_config.use_double_wide_mlp is set (mirrors vllm-pytorch's
        # `Gemma4DecoderLayer.__init__` gating).
        is_kv_shared = layer_idx in compute_kv_share_map(text_config)
        use_double_wide_mlp = (getattr(text_config, "use_double_wide_mlp",
                                       False) and is_kv_shared)
        layer_intermediate_size = (text_config.intermediate_size *
                                   (2 if use_double_wide_mlp else 1))

        self.mlp = Gemma4MLP(
            config=text_config,
            dtype=dtype,
            rng=rng,
            quant_config=quant_config,
            intermediate_size=layer_intermediate_size,
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

        # PLE per-layer modules. Constructed only when
        # hidden_size_per_layer_input > 0; otherwise these stay None and
        # the PLE block in __call__ is gated off.
        if self.hidden_size_per_layer_input > 0:
            P = self.hidden_size_per_layer_input
            self.per_layer_input_gate = JaxEinsum(
                "TD,DP->TP",
                (hidden_size, P),
                bias_shape=None,
                param_dtype=dtype,
                kernel_init=nnx.with_partitioning(init_fn, (None, None)),
                rngs=rng,
                quant_config=quant_config,
                prefix=prefix + ".per_layer_input_gate",
            )
            self.per_layer_projection = JaxEinsum(
                "TP,PD->TD",
                (P, hidden_size),
                bias_shape=None,
                param_dtype=dtype,
                kernel_init=nnx.with_partitioning(init_fn, (None, None)),
                rngs=rng,
                quant_config=quant_config,
                prefix=prefix + ".per_layer_projection",
            )
            self.post_per_layer_input_norm = JaxRmsNorm(
                hidden_size,
                epsilon=rms_norm_eps,
                param_dtype=dtype,
                scale_init=nnx.with_partitioning(init_fn, (None, )),
                rngs=rng,
                quant_config=quant_config,
                prefix=prefix + ".post_per_layer_input_norm",
            )
        else:
            self.per_layer_input_gate = None
            self.per_layer_projection = None
            self.post_per_layer_input_norm = None

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
        per_layer_input: Optional[jax.Array] = None,
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

        # PLE per-layer block. Gated on having both a
        # per_layer_input AND the modules being constructed (the latter
        # is governed by hidden_size_per_layer_input > 0 in __init__).
        if per_layer_input is not None and self.per_layer_input_gate is not None:
            gate = self.per_layer_input_gate(outputs)
            gate = nnx.gelu(gate, approximate=True)
            gated_per_layer = gate * per_layer_input
            per_layer_contribution = self.per_layer_projection(gated_per_layer)
            per_layer_contribution = self.post_per_layer_input_norm(
                per_layer_contribution)
            outputs = outputs + per_layer_contribution

        outputs = outputs * self.layer_scalar.get_value()

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

        # PLE (Per-Layer Embedding) — model-level modules.
        # Active when hidden_size_per_layer_input > 0 (E2B/E4B).
        self.hidden_size_per_layer_input = getattr(
            text_config, "hidden_size_per_layer_input", 0)
        self.vocab_size_per_layer_input = getattr(
            text_config, "vocab_size_per_layer_input", vocab_size)
        self.num_hidden_layers = text_config.num_hidden_layers

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

        # Model-level PLE modules. Constructed only on first rank when
        # PLE is active, since they're consumed in __call__ before the
        # decoder loop.
        if self.hidden_size_per_layer_input > 0 and self.is_first_rank:
            P = self.hidden_size_per_layer_input
            L = self.num_hidden_layers
            # embed_tokens_per_layer: vocab_size_per_layer_input -> L*P.
            # Replicated across model axis (small enough; gather_output
            # at use site).
            self.embed_tokens_per_layer = JaxEmbed(
                num_embeddings=self.vocab_size_per_layer_input,
                features=L * P,
                param_dtype=dtype,
                embedding_init=nnx.with_partitioning(init_fn, ("model", None)),
                rngs=rng,
                quant_config=vllm_config.quant_config,
                prefix=prefix + ".embed_tokens_per_layer",
            )
            # PLE is non-standard so we set its weight_loader here (per the
            # guidance in weight_utils.py:883) rather than adding a new
            # special case to the central pattern list. HF stores
            # embed_tokens_per_layer.weight as (V_ple, L*P); JaxEmbed expects
            # the same shape — explicit permute_dims=(0,1) suppresses the
            # default 2D-transpose that load_nnx_param_from_reshaped_torch
            # applies when permute_dims is None.
            self.embed_tokens_per_layer.weight.set_metadata(
                "weight_loader",
                partial(load_nnx_param_from_reshaped_torch,
                        permute_dims=(0, 1),
                        param_name=prefix + ".embed_tokens_per_layer.weight"))
            # per_layer_model_projection: H -> L*P. ColumnParallelLinear
            # with gather_output=True in vllm; we replicate output.
            self.per_layer_model_projection = JaxEinsum(
                "TD,DM->TM",
                (hidden_size, L * P),
                bias_shape=None,
                param_dtype=dtype,
                kernel_init=nnx.with_partitioning(init_fn, (None, None)),
                rngs=rng,
                quant_config=vllm_config.quant_config,
                prefix=prefix + ".per_layer_model_projection",
            )
            # RMSNorm over P (last dim of [T, L, P]).
            self.per_layer_projection_norm = JaxRmsNorm(
                P,
                epsilon=rms_norm_eps,
                param_dtype=dtype,
                scale_init=nnx.with_partitioning(init_fn, (None, )),
                rngs=rng,
                quant_config=vllm_config.quant_config,
                prefix=prefix + ".per_layer_projection_norm",
            )
            # Constants as Python floats (single-ownership).
            # NOTE: gemma-4 uses H^-0.5 (gemma-3n flipped to H^0.5).
            self.embed_scale_per_layer = float(P)**0.5
            self.per_layer_input_scale = 1.0 / (2.0**0.5)
            self.per_layer_projection_scale = float(hidden_size)**-0.5
        else:
            self.embed_tokens_per_layer = None
            self.per_layer_model_projection = None
            self.per_layer_projection_norm = None
            self.embed_scale_per_layer = 0.0
            self.per_layer_input_scale = 0.0
            self.per_layer_projection_scale = 0.0

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

    def compute_per_layer_inputs(
        self,
        input_ids: Optional[jax.Array],
        inputs_embeds: jax.Array,
        is_multimodal: Optional[jax.Array] = None,
    ) -> Optional[jax.Array]:
        """Compute per_layer_inputs of shape [T, L, P].

        See vllm/model_executor/models/gemma4.py:Gemma4SelfDecoderLayers.
        get_per_layer_inputs / project_per_layer_inputs for the reference
        algorithm: Track A is an embed_tokens_per_layer lookup scaled by
        sqrt(P) and reshaped to (T, L, P); Track B is a linear projection
        of inputs_embeds (post embedding-scale) scaled by 1/sqrt(H), reshaped
        to (T, L, P) and RMSNorm'd over P. The result is
        `(track_a + track_b) * 1/sqrt(2)`.

        Returns None when PLE is disabled (hidden_size_per_layer_input=0)
        or when the model-level PLE modules aren't constructed (non-first
        pp rank).

        Args:
          input_ids: [T] token ids. Real inference always provides this;
            the precompile path
            (compilation_manager._precompile_backbone_with_inputs_embeds)
            passes None. In that case we synthesize zeros so the JIT trace
            shape matches real inference — the precompile output is
            discarded.
          inputs_embeds: [T, H] post-scaling residual stream (already
            multiplied by embedding_scale + multimodal-merged).
          is_multimodal: [T] bool. Multimodal positions are masked to
            slot 0 in the embed_tokens_per_layer lookup.
        """
        if (self.hidden_size_per_layer_input == 0
                or self.embed_tokens_per_layer is None):
            return None
        if input_ids is None:
            # Precompile path with inputs_embeds-only entry. Zeros produce
            # a shape-identical compute (all PLE lookups hit slot 0) so the
            # JIT cache key matches the real-inference trace.
            input_ids = jnp.zeros((inputs_embeds.shape[0], ), dtype=jnp.int32)
        T = input_ids.shape[0]
        L = self.num_hidden_layers
        P = self.hidden_size_per_layer_input

        # Multimodal masking: MM positions look up slot 0.
        if is_multimodal is not None:
            ple_input_ids = jnp.where(is_multimodal, 0, input_ids)
        else:
            ple_input_ids = input_ids

        # Out-of-vocab masking: when vocab_size_per_layer_input
        # < vocab_size, mask high token ids to 0.
        ple_input_ids = jnp.where(
            ple_input_ids < self.vocab_size_per_layer_input, ple_input_ids, 0)

        # Track A — embedding lookup.
        per_layer_embeds = self.embed_tokens_per_layer(ple_input_ids)
        per_layer_embeds = per_layer_embeds * self.embed_scale_per_layer
        per_layer_embeds = per_layer_embeds.reshape(T, L, P)

        # Track B — projection of inputs_embeds.
        per_layer_projection = self.per_layer_model_projection(inputs_embeds)
        per_layer_projection = (per_layer_projection *
                                self.per_layer_projection_scale)
        per_layer_projection = per_layer_projection.reshape(T, L, P)
        per_layer_projection = self.per_layer_projection_norm(
            per_layer_projection)

        # Combine.
        return ((per_layer_projection + per_layer_embeds) *
                self.per_layer_input_scale)

    def __call__(
        self,
        kv_caches: List[jax.Array],
        input_ids: Optional[jax.Array],
        attention_metadata: AttentionMetadata,
        inputs_embeds: Optional[jax.Array] = None,
        layer_name_to_kv_cache: Optional[dict] = None,
        is_multimodal: Optional[jax.Array] = None,
    ) -> Tuple[List[jax.Array], jax.Array, Optional[jax.Array]]:

        if inputs_embeds is not None:
            x = inputs_embeds
        else:
            x = self.embed_tokens(input_ids)
            # Gemma4: Apply embedding scaling
            x = x * self.embedding_scale

        # PLE: compute per_layer_inputs once; slice [T, layer_idx, :] per layer.
        # Returns None for non-PLE configs (e.g. 26B/31B), in which case each
        # decoder layer gates the PLE block off via its own per_layer_input=None.
        per_layer_inputs = self.compute_per_layer_inputs(
            input_ids, x, is_multimodal=is_multimodal)

        all_expert_ids = []
        for i, layer in enumerate(
                islice(self.layers, self.start_layer, self.end_layer)):
            layer_idx = i + self.start_layer
            layer_name = f"layer.{layer_idx}"
            if isinstance(attention_metadata, dict):
                layer_attn_metadata = attention_metadata[layer_name]
            else:
                layer_attn_metadata = attention_metadata

            if layer_name_to_kv_cache and layer_name in layer_name_to_kv_cache:
                cache_idx = layer_name_to_kv_cache[layer_name]
            else:
                cache_idx = layer_idx

            kv_cache = kv_caches[cache_idx]
            layer_per_input = (per_layer_inputs[:, layer_idx, :]
                               if per_layer_inputs is not None else None)
            kv_cache, x, expert_ids = layer(
                kv_cache,
                x,
                layer_attn_metadata,
                per_layer_input=layer_per_input,
            )
            if expert_ids is not None:
                all_expert_ids.append(expert_ids)
            kv_caches[cache_idx] = kv_cache
        x = self.norm(x)
        stacked_expert_ids = jnp.stack(all_expert_ids,
                                       axis=0) if all_expert_ids else None
        return kv_caches, x, stacked_expert_ids


class Gemma4ForCausalLM(JaxModule, LoadableWithIterator):
    packed_modules_mapping = {
        "qkv_proj": [
            "q_proj",
            "k_proj",
            "v_proj",
        ],
        "gate_up_proj": [
            "gate_proj",
            "up_proj",
        ],
    }
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
                self.lm_head = JaxLmHead(
                    hidden_size=hidden_size,
                    vocab_size=vocab_size,
                    dtype=model_config.dtype,
                    rngs=rng,
                    prefix="lm_head",
                )
            else:
                self.lm_head = PPMissingLayer()

    def load_weights(self, weights: Iterable[Tuple[str, Any]]):
        allowed_layers = set(f"layers.{i}."
                             for i in range(len(self.model.layers)))
        ignored_prefixes = (
            "model.audio_tower.",
            "model.vision_tower.",
            "model.multi_modal_projector.",
            "model.embed_audio.",
        )
        stripped_weights = (
            (clean_name, tensor) for name, tensor in weights
            if (clean_name := name.replace("language_model.", "")).startswith((
                "model.", "lm_head")) and "vision" not in clean_name
            and  # Exclude vision tower weights for now
            not any(clean_name.startswith(p) for p in ignored_prefixes))
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
        # Text-only causal LM has no multimodal tokens; pass None.
        kv_caches, x, expert_indices = self.model(
            kv_caches,
            input_ids,
            attention_metadata,
            inputs_embeds,
            layer_name_to_kv_cache=layer_name_to_kv_cache,
            is_multimodal=None,
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
