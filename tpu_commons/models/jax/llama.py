import functools
import itertools
from collections import deque
from typing import Any, Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.traverse_util import unflatten_dict
from flax.typing import PRNGKey
from jax.sharding import Mesh, NamedSharding, PartitionSpec
from transformers import LlamaConfig

from tpu_commons.logger import init_logger
from tpu_commons.models.jax import layers
from tpu_commons.models.jax.config import (CacheConfig, LoRAConfig,
                                           ModelConfig, ParallelConfig)
from tpu_commons.models.jax.kv_cache_eviction import (
    KVCacheUpdater, get_kv_cache_updater_class)
from tpu_commons.models.jax.param_init import sharding_init
from tpu_commons.models.jax.quantization.awq import reverse_awq_order, unpack
from tpu_commons.models.jax.quantization_config import QuantizationConfig
from tpu_commons.models.jax.rope import generic_rope
from tpu_commons.models.jax.sampling import sample
from tpu_commons.models.jax.weight_utils import (get_num_kv_heads_by_tp,
                                                 get_num_q_heads_by_tp,
                                                 hf_model_weights_iterator)

logger = init_logger(__name__)

KVCache = Tuple[jax.Array, jax.Array]


class Embedder(nn.Module):
    vocab_size: int
    hidden_size: int
    dtype: jnp.dtype
    mesh: Mesh
    quantization_config: QuantizationConfig = None

    def setup(self) -> None:
        self.input_embedding_table = self.param(
            "weight",
            sharding_init(("model", None),
                          self.mesh,
                          quantization_config=self.quantization_config),
            (self.vocab_size, self.hidden_size),
            self.dtype,
        )

    def encode(self, x: jax.Array) -> jax.Array:
        x = self.input_embedding_table[(x, )]
        return x

    def decode(self, x: jax.Array) -> jax.Array:
        return jnp.dot(x, self.input_embedding_table.T)


class LlamaAttention(nn.Module):
    config: LlamaConfig
    kv_cache_updater: KVCacheUpdater
    dtype: jnp.dtype
    mesh: Mesh
    num_kv_heads: int
    layer_id: int
    quantization_config: Optional[QuantizationConfig] = None
    lora_config: Optional[LoRAConfig] = None
    cache_config: Optional[CacheConfig] = None

    def setup(self) -> None:
        self.hidden_size = self.config.hidden_size
        self.num_heads = get_num_q_heads_by_tp(
            self.config.num_attention_heads,
            self.config.num_key_value_heads,
            self.mesh.shape["model"],
        )
        self.rope_theta = self.config.rope_theta
        self.rope_scaling = getattr(self.config, "rope_scaling", None)
        self.head_dim = self.config.head_dim
        self.head_dim_original = getattr(self.config, "head_dim_original",
                                         self.config.head_dim)

        self.is_nope_layer = False
        if (getattr(self.config, "nope_layer_interval", None) and
            (self.layer_id + 1) % self.config.nope_layer_interval == 0):
            self.is_nope_layer = True

        self.use_rope = not self.is_nope_layer
        self.use_qk_norm = (getattr(self.config, "use_qk_norm", False)
                            and not self.is_nope_layer)

        # ruff: noqa: E731
        if not self.use_rope:
            prepare_qk_fn_orig = lambda x, positions: x  # No-op
        else:
            prepare_qk_fn_orig = functools.partial(
                generic_rope.apply_rope,
                head_dim=getattr(self.config, "head_dim_original",
                                 self.config.head_dim),
                rope_theta=self.config.rope_theta,
                rope_scaling=self.config.rope_scaling,
            )

        # Llama 4 does qk norm immediately after rope.
        if self.use_qk_norm:
            self.prepare_qk_fn = lambda x, positions: layers.rms_norm(
                prepare_qk_fn_orig(x, positions), self.config.rms_norm_eps)
        else:
            self.prepare_qk_fn = prepare_qk_fn_orig

        _einsum_class = layers.get_einsum(
            quantization_config=self.quantization_config)
        self.q_proj = _einsum_class(
            shape=(self.num_heads, self.hidden_size, self.head_dim),
            dtype=self.dtype,
            named_axes=("model", None, None),
            mesh=self.mesh,
            hidden_dim=1,
        )
        self.k_proj = _einsum_class(
            shape=(self.num_kv_heads, self.hidden_size, self.head_dim),
            dtype=self.dtype,
            named_axes=("model", None, None),
            mesh=self.mesh,
            hidden_dim=1,
        )
        self.v_proj = _einsum_class(
            shape=(self.num_kv_heads, self.hidden_size, self.head_dim),
            dtype=self.dtype,
            named_axes=("model", None, None),
            mesh=self.mesh,
            hidden_dim=1,
        )
        self.o_proj = _einsum_class(
            shape=(self.num_heads, self.head_dim, self.hidden_size),
            dtype=self.dtype,
            named_axes=("model", None, None),
            mesh=self.mesh,
            hidden_dim=2,
        )
        if self.lora_config is not None and self.lora_config.enable_lora:
            self.q_proj_lora = layers.LoRALayer(
                num_lora=self.lora_config.max_num_lora,
                shape_a=(
                    self.hidden_size,
                    self.lora_config.max_lora_rank,
                ),
                shape_b=(
                    self.lora_config.max_lora_rank,
                    self.num_heads,
                    self.head_dim,
                ),
                dtype=self.dtype,
                named_axes_a=(None, None),
                named_axes_b=(None, "model", None),
                mesh=self.mesh,
            )
            self.k_proj_lora = layers.LoRALayer(
                num_lora=self.lora_config.max_num_lora,
                shape_a=(
                    self.hidden_size,
                    self.lora_config.max_lora_rank,
                ),
                shape_b=(
                    self.lora_config.max_lora_rank,
                    self.num_kv_heads,
                    self.head_dim,
                ),
                dtype=self.dtype,
                named_axes_a=(None, None),
                named_axes_b=(None, "model", None),
                mesh=self.mesh,
            )
            self.v_proj_lora = layers.LoRALayer(
                num_lora=self.lora_config.max_num_lora,
                shape_a=(
                    self.hidden_size,
                    self.lora_config.max_lora_rank,
                ),
                shape_b=(
                    self.lora_config.max_lora_rank,
                    self.num_kv_heads,
                    self.head_dim,
                ),
                dtype=self.dtype,
                named_axes_a=(None, None),
                named_axes_b=(None, "model", None),
                mesh=self.mesh,
            )
            self.o_proj_lora = layers.LoRALayer(
                num_lora=self.lora_config.max_num_lora,
                shape_a=(
                    self.num_heads,
                    self.head_dim,
                    self.lora_config.max_lora_rank,
                ),
                shape_b=(
                    self.lora_config.max_lora_rank,
                    self.hidden_size,
                ),
                dtype=self.dtype,
                named_axes_a=("model", None, None),
                named_axes_b=(None, None),
                mesh=self.mesh,
            )

    def __call__(
        self,
        is_prefill: bool,
        kv_cache: Optional[KVCache],
        x: jax.Array,
        attention_metadata: layers.AttentionMetadata,
        kv_cache_position_indices: Optional[jax.Array] = None,
        evict_write_indices: Optional[jax.Array] = None,
        replacement_write_indices: Optional[jax.Array] = None,
    ) -> Tuple[KVCache, jax.Array, Optional[jax.Array]]:
        # B: batch_size
        # T: seq_len
        # N: num_heads
        # K: num_kv_heads
        # D: hidden_size
        # H: head_dim
        # L: num_blocks
        # S: block_size
        # R: max_lora_rank
        # M: max_num_lora

        # (B, N, T, H)
        q = self.q_proj("BTD,NDH->BNTH", x)
        if self.lora_config is not None and self.lora_config.enable_lora:
            # Pad input B dimension to M and truncate back to B.
            q_lora_outputs = self.q_proj_lora("MTD,MDR->MTR", "MTR,MRNH->MNTH",
                                              x)
            q = q + q_lora_outputs

        # RoPE and qk_norm (if applicable) at the same time.
        q = self.prepare_qk_fn(q, attention_metadata.input_positions)

        q = q * (self.head_dim_original**-0.5)

        # (B, K, T, H)
        k = self.k_proj("BTD,KDH->BKTH", x)
        if self.lora_config is not None and self.lora_config.enable_lora:
            # Pad input B dimension to M and truncate back to B.
            k_lora_outputs = self.k_proj_lora("MTD,MDR->MTR", "MTR,MRKH->MKTH",
                                              x)
            k = k + k_lora_outputs

        v = self.v_proj("BTD,KDH->BKTH", x)
        if self.lora_config is not None and self.lora_config.enable_lora:
            # Pad input B dimension to M and truncate back to B.
            v_lora_outputs = self.v_proj_lora("MTD,MDR->MTR", "MTR,MRKH->MKTH",
                                              x)
            v = v + v_lora_outputs

        (k_cache, v_cache), outputs, attn_scores = (
            self.kv_cache_updater.update_cache_with_attn(
                is_prefill,
                kv_cache,
                q,
                k,
                v,
                attention_metadata,
                self.prepare_qk_fn,
                kv_cache_position_indices,
                evict_write_indices,
                replacement_write_indices,
            ))
        # (B, T, D)
        o = self.o_proj("BNTH,NHD->BTD", outputs)
        if self.lora_config is not None and self.lora_config.enable_lora:
            # Pad input B dimension to M and truncate back to B.
            o_lora_outputs = self.o_proj_lora("MNTH,MNHR->MTR", "MTR,MRD->MTD",
                                              outputs)
            o = o + o_lora_outputs
        return ((k_cache, v_cache), o, attn_scores)


class LlamaDecoderLayer(nn.Module):
    config: LlamaConfig
    kv_cache_updater: KVCacheUpdater
    dtype: jnp.dtype
    mesh: Mesh
    num_kv_heads: int
    layer_id: int
    quantization_config: Optional[QuantizationConfig] = None
    lora_config: Optional[LoRAConfig] = None
    cache_config: Optional[CacheConfig] = None

    def setup(self) -> None:
        self.rms_norm_eps = self.config.rms_norm_eps
        self.hidden_size = self.config.hidden_size
        self.intermediate_size = self.config.intermediate_size
        self.num_heads = self.config.num_attention_heads
        self.act = self.config.hidden_act
        self.rope_theta = self.config.rope_theta
        self.head_dim = self.config.head_dim

        self.input_layernorm = layers.RMSNorm(
            rms_norm_eps=self.rms_norm_eps,
            dtype=self.dtype,
            mesh=self.mesh,
        )

        self.self_attn = LlamaAttention(
            config=self.config,
            kv_cache_updater=self.kv_cache_updater,
            dtype=self.dtype,
            mesh=self.mesh,
            num_kv_heads=self.num_kv_heads,
            layer_id=self.layer_id,
            quantization_config=self.quantization_config,
            lora_config=self.lora_config,
            cache_config=self.cache_config,
        )

        self.post_attention_layernorm = layers.RMSNorm(
            rms_norm_eps=self.rms_norm_eps,
            dtype=self.dtype,
            mesh=self.mesh,
        )

        self.is_moe = getattr(self.config, "num_local_experts",
                              None) is not None
        if not self.is_moe:
            _mlp_class = layers.get_mlp(
                quantization_config=self.quantization_config,
                lora_config=self.lora_config,
            )
            if self.lora_config is not None and self.lora_config.enable_lora:
                self.mlp = _mlp_class(
                    hidden_size=self.hidden_size,
                    intermediate_size=self.intermediate_size,
                    dtype=self.dtype,
                    mesh=self.mesh,
                    act=self.act,
                    lora_config=self.lora_config,
                )
            else:
                self.mlp = _mlp_class(
                    hidden_size=self.hidden_size,
                    intermediate_size=self.intermediate_size,
                    dtype=self.dtype,
                    mesh=self.mesh,
                    act=self.act,
                    name="mlp",
                )
        else:
            # TODO:
            raise NotImplementedError("Moe is not supported yet.")
            # moe_class = moe.get_moe(quantization_config=self.quantization_config)
            # self.mlp = moe_class(
            #     hidden_size=self.config.hidden_size,
            #     intermediate_size=self.config.intermediate_size,
            #     dtype=self.dtype,
            #     mesh=self.mesh,
            #     act=self.config.hidden_act,
            #     num_experts=self.config.num_local_experts,
            #     num_experts_per_tok=self.config.num_experts_per_tok,
            #     score_normalizer=lambda x: jax.nn.sigmoid(x.astype(jnp.float32)).astype(
            #         x.dtype
            #     ),
            #     apply_expert_weight_before_computation=True,
            #     name="experts",
            # )

            # _mlp_class = layers.get_mlp(
            #     quantization_config=self.quantization_config,
            #     lora_config=self.lora_config,
            # )
            # self.shared_expert = _mlp_class(
            #     hidden_size=self.hidden_size,
            #     intermediate_size=self.intermediate_size,
            #     dtype=self.dtype,
            #     mesh=self.mesh,
            #     act=self.act,
            # )

    def __call__(
        self,
        is_prefill: bool,
        kv_cache: KVCache,
        x: jax.Array,
        attention_metadata: layers.AttentionMetadata,
        kv_cache_position_indices: Optional[jax.Array] = None,
        evict_write_indices: Optional[jax.Array] = None,
        replacement_write_indices: Optional[jax.Array] = None,
    ) -> Tuple[KVCache, jax.Array, jax.Array]:
        # Self attention.
        hidden_states = self.input_layernorm(x)
        kv_cache, attn_output, attn_scores = self.self_attn(
            is_prefill,
            kv_cache,
            hidden_states,
            attention_metadata,
            kv_cache_position_indices,
            evict_write_indices,
            replacement_write_indices,
        )
        attn_output += x

        # MLP.
        residual = attn_output
        attn_output = self.post_attention_layernorm(attn_output)
        if not self.is_moe:
            outputs = self.mlp(attn_output)
        else:
            raise NotImplementedError("Moe is not supported yet.")
            # backend = jax.default_backend()
            # if backend not in ["cpu", "tpu"]:
            #     raise ValueError(f"Unsupported backend: {backend}")
            # outputs = self.mlp(
            #     attn_output,
            #     impl=(moe.MoEImpl.NON_FUSED_MEGABLOX
            #           if backend == "tpu" else moe.MoEImpl.LOOP),
            # )
            # shared_expert_output = self.shared_expert(attn_output)
            # outputs += shared_expert_output
        outputs = residual + outputs
        return kv_cache, outputs, attn_scores


class LlamaModel(nn.Module):
    config: LlamaConfig
    dtype: jnp.dtype
    mesh: Mesh
    quantization_config: Optional[QuantizationConfig] = None
    lora_config: Optional[LoRAConfig] = None
    cache_config: Optional[CacheConfig] = None

    def setup(self) -> None:
        num_kv_heads_by_tp = get_num_kv_heads_by_tp(
            self.config.num_key_value_heads, self.mesh.shape["model"])
        # Apply MQA if kv-heads is 1 after sharding.
        is_mqa = num_kv_heads_by_tp == self.mesh.shape["model"]

        self.kv_cache_updater = get_kv_cache_updater_class(
            self.cache_config
            and self.cache_config.kv_cache_eviction_algorithm)(
                sliding_window=getattr(self.config, "sliding_window", None),
                sink_size=self.cache_config and self.cache_config.sink_size,
                mesh=self.mesh,
                cache_attention_scores=self.cache_config
                and self.cache_config.cache_attention_scores,
                prefill_repeat_kv=self.config.num_attention_heads //
                num_kv_heads_by_tp,
                is_mqa=is_mqa,
            )
        self.layers = [
            LlamaDecoderLayer(
                config=self.config,
                kv_cache_updater=self.kv_cache_updater,
                dtype=self.dtype,
                mesh=self.mesh,
                num_kv_heads=num_kv_heads_by_tp,
                layer_id=i,
                quantization_config=self.quantization_config,
                lora_config=self.lora_config,
                cache_config=self.cache_config,
            ) for i in range(self.config.num_hidden_layers)
        ]
        self.norm = layers.RMSNorm(
            rms_norm_eps=self.config.rms_norm_eps,
            dtype=self.dtype,
            mesh=self.mesh,
        )

    def __call__(
        self,
        is_prefill: bool,
        kv_caches: List[KVCache],
        x: jax.Array,
        attention_metadata: layers.AttentionMetadata,
        kv_cache_position_indices: Optional[jax.Array] = None,
        evict_write_indices: Optional[jax.Array] = None,
        replacement_write_indices: Optional[jax.Array] = None,
        eviction_score_mask: Optional[jax.Array] = None,
    ) -> Tuple[List[KVCache], jax.Array, jax.Array]:
        accumulated_attn_scores = None
        cache_attention_scores = self.kv_cache_updater.cache_attention_scores

        for i, layer in enumerate(self.layers):
            kv_cache = kv_caches[i]
            kv_cache, x, attn_scores = layer(
                is_prefill,
                kv_cache,
                x,
                attention_metadata,
                kv_cache_position_indices,
                evict_write_indices,
                replacement_write_indices,
            )
            kv_caches[i] = kv_cache

            if cache_attention_scores:
                if accumulated_attn_scores is None:
                    accumulated_attn_scores = attn_scores
                else:
                    accumulated_attn_scores += attn_scores

        kv_caches, accumulated_attn_scores = self.kv_cache_updater.update_kv_caches(
            is_prefill,
            x,
            kv_caches,
            attention_metadata.seq_lens,
            accumulated_attn_scores,
            eviction_score_mask,
            attention_metadata.kv_cache_write_indices,
        )
        x = self.norm(x)
        return kv_caches, x, accumulated_attn_scores


class LlamaForCausalLM(nn.Module):
    config: LlamaConfig
    rng: PRNGKey
    dtype: jnp.dtype
    mesh: Mesh
    lora_config: LoRAConfig
    cache_config: CacheConfig
    quantization_config: Optional[QuantizationConfig] = None

    def setup(self) -> None:
        # TODO(pooyam): Remove the following block after b/411232437.
        if self.config.model_type == "llama4":
            if jax.__version__ != "0.4.33":
                raise ValueError(
                    f"Llama4 is not supposed for jax {jax.__version__}")

        config_to_use = self.config
        if getattr(self.config, "text_config", None):
            config_to_use = self.config.text_config

        self.embed_tokens = Embedder(
            vocab_size=config_to_use.vocab_size,
            hidden_size=config_to_use.hidden_size,
            dtype=self.dtype,
            mesh=self.mesh,
            quantization_config=self.quantization_config,
        )
        self.model = LlamaModel(
            config=config_to_use,
            dtype=self.dtype,
            mesh=self.mesh,
            quantization_config=self.quantization_config,
            lora_config=self.lora_config,
            cache_config=self.cache_config,
        )
        if getattr(self.config, "use_embedding_as_last_layer", False):
            self.lm_head = None
        else:
            self.lm_head = self.param(
                "lm_head",
                sharding_init(
                    (None, "model"),
                    self.mesh,
                    quantization_config=self.quantization_config,
                ),
                (config_to_use.hidden_size, config_to_use.vocab_size),
                self.dtype,
            )

        if getattr(self.config, "vision_config", None):
            raise NotImplementedError("Vision model not supported yet.")
            # self.vision_model = VisionEmbeddings(
            #     config=self.config.vision_config,
            #     mesh=self.mesh,
            #     dtype=self.dtype,
            #     quantization_config=self.quantization_config,
            # )
            # _einsum_class = layers.get_einsum(
            #     quantization_config=self.quantization_config)
            # self.vision_projection = _einsum_class(
            #     shape=(
            #         self.config.vision_config.projector_output_dim,
            #         config_to_use.hidden_size,
            #     ),
            #     dtype=self.dtype,
            #     named_axes=(None, "model"),
            #     mesh=self.mesh,
            #     hidden_dim=1,
            # )

    def __call__(
        self,
        is_prefill: bool,
        do_sampling: bool,
        kv_caches: List[KVCache],
        input_ids: jax.Array,
        attention_metadata: layers.AttentionMetadata,
        temperatures: jax.Array = None,
        top_ps: jax.Array = None,
        top_ks: jax.Array = None,
        kv_cache_position_indices: Optional[jax.Array] = None,
        evict_write_indices: Optional[jax.Array] = None,
        replacement_write_indices: Optional[jax.Array] = None,
        eviction_score_mask: Optional[jax.Array] = None,
        images_flattened: Optional[jax.Array] = None,
        image_lens: Optional[jax.Array] = None,
        *args,
    ) -> Tuple[List[KVCache], jax.Array, jax.Array, jax.Array]:
        x = self.embed_tokens.encode(input_ids)

        if is_prefill and images_flattened is not None:
            image_mask = input_ids == self.config.image_token_index
            image_embedding = self.vision_model(images_flattened, image_mask,
                                                x)
            image_embedding = self.vision_projection("BTH,HD->BTD",
                                                     image_embedding)
            image_mask = jnp.expand_dims(image_mask, axis=-1)
            x = x * ~image_mask + image_embedding * image_mask

        kv_caches, x, accumulated_attn_scores = self.model(
            is_prefill,
            kv_caches,
            x,
            attention_metadata,
            kv_cache_position_indices,
            evict_write_indices,
            replacement_write_indices,
            eviction_score_mask,
        )

        if self.lm_head is not None:
            logits = jnp.dot(x, self.lm_head)
        else:
            logits = self.embed_tokens.decode(x)
        next_tokens = sample(
            is_prefill,
            do_sampling,
            self.rng,
            self.mesh,
            logits,
            attention_metadata.seq_lens,
            temperatures,
            top_ps,
            top_ks,
            attention_metadata.chunked_prefill_enabled,
        )
        if not self.cache_config or not self.cache_config.output_logits:
            logits = None
        return kv_caches, next_tokens, logits, accumulated_attn_scores

    def load_lora_adapter(self, seq_index: int,
                          lora_local_path: str) -> Dict[str, Any]:

        def _device_weight(weight: jax.Array,
                           sharding_names: Tuple[str, ...]) -> jax.Array:
            return jax.device_put(
                weight,
                NamedSharding(self.mesh, PartitionSpec(*sharding_names)),
            )

        lora_params = {}
        head_dim = self.config.head_dim
        num_kv_heads = self.config.num_key_value_heads
        num_kv_heads_by_tp = get_num_kv_heads_by_tp(num_kv_heads,
                                                    self.mesh.shape["model"])
        for name, checkpoint_weight in hf_model_weights_iterator(
                lora_local_path, framework="flax"):
            if "lora" not in name:
                continue

            name = name.replace("base_model.model", "params")
            name = name.replace("_orig_module.", "")
            name = name.replace("layers.", "layers_")
            name = name.replace(".lora_A.weight", f"_lora.a_{seq_index}")
            name = name.replace(".lora_B.weight", f"_lora.b_{seq_index}")

            weight = checkpoint_weight.astype(self.lora_config.dtype)

            weight = jnp.transpose(weight, (1, 0))

            # Zero-pad to maximum LoRA rank.
            if "lora.a" in name:
                lora_dim = 1
            else:
                lora_dim = 0
            if weight.shape[lora_dim] < self.lora_config.max_lora_rank:
                pad_width = [
                    (0,
                     self.lora_config.max_lora_rank - weight.shape[lora_dim]),
                    (0, 0),
                ]
                weight = jnp.pad(
                    weight,
                    pad_width=pad_width if lora_dim == 0 else pad_width[::-1],
                    mode="constant",
                    constant_values=0,
                )

            if "q_proj_lora" in name:
                if "q_proj_lora.a" in name:
                    weight = _device_weight(weight, (None, None))
                else:
                    weight = jnp.reshape(
                        weight,
                        (
                            self.lora_config.max_lora_rank,
                            self.config.num_attention_heads,
                            head_dim,
                        ),
                    )
                    weight = _device_weight(weight, (None, "model", None))
            elif "k_proj_lora" in name or "v_proj_lora" in name:
                if "k_proj_lora.a" in name or "v_proj_lora.a" in name:
                    weight = _device_weight(weight, (None, None))
                else:
                    weight = jnp.reshape(
                        weight,
                        (
                            self.lora_config.max_lora_rank,
                            num_kv_heads,
                            head_dim,
                        ),
                    )
                    weight = jnp.repeat(weight,
                                        num_kv_heads_by_tp // num_kv_heads,
                                        axis=0)
                    weight = _device_weight(weight, (None, "model", None))
            elif "o_proj_lora" in name:
                if "o_proj_lora.a" in name:
                    weight = jnp.reshape(
                        weight,
                        (
                            self.config.num_attention_heads,
                            head_dim,
                            self.lora_config.max_lora_rank,
                        ),
                    )
                    weight = _device_weight(weight, ("model", None, None))
                else:
                    weight = _device_weight(weight, (None, None))
            elif "gate_proj_lora" in name or "up_proj_lora" in name:
                if "o_proj_lora.a" in name or "o_proj_lora.a" in name:
                    weight = _device_weight(weight, (None, None))
                else:
                    weight = _device_weight(weight, (None, "model"))
            elif "down_proj_lora" in name:
                if "down_proj_lora.a" in name:
                    weight = _device_weight(weight, ("model", None))
                else:
                    weight = _device_weight(weight, (None, None))

            lora_params[name] = weight
        return lora_params

    def load_zero_lora_adapter(self, seq_index: int) -> Dict[str, Any]:

        def _device_weight(weight: jax.Array,
                           sharding_names: Tuple[str, ...]) -> jax.Array:
            return jax.device_put(
                weight, NamedSharding(self.mesh,
                                      PartitionSpec(*sharding_names)))

        lora_params = {}
        head_dim = self.config.head_dim
        num_kv_heads = self.config.num_key_value_heads
        num_kv_heads_by_tp = get_num_kv_heads_by_tp(num_kv_heads,
                                                    self.mesh.shape["model"])
        for i in range(self.config.num_hidden_layers):
            lora_params.update({
                # Zero-initialize attention LoRA weights.
                f"params.model.layers_{i}.self_attn.q_proj_lora.a_{seq_index}":
                _device_weight(
                    jnp.zeros(
                        (
                            self.config.hidden_size,
                            self.lora_config.max_lora_rank,
                        ),
                        dtype=self.dtype,
                    ),
                    (None, None),
                ),
                f"params.model.layers_{i}.self_attn.q_proj_lora.b_{seq_index}":
                _device_weight(
                    jnp.zeros(
                        (
                            self.lora_config.max_lora_rank,
                            self.config.num_attention_heads,
                            head_dim,
                        ),
                        dtype=self.dtype,
                    ),
                    (None, "model", None),
                ),
                f"params.model.layers_{i}.self_attn.k_proj_lora.a_{seq_index}":
                _device_weight(
                    jnp.zeros(
                        (
                            self.config.hidden_size,
                            self.lora_config.max_lora_rank,
                        ),
                        dtype=self.dtype,
                    ),
                    (None, None),
                ),
                f"params.model.layers_{i}.self_attn.k_proj_lora.b_{seq_index}":
                _device_weight(
                    jnp.repeat(
                        jnp.zeros(
                            (
                                self.lora_config.max_lora_rank,
                                num_kv_heads,
                                head_dim,
                            ),
                            dtype=self.dtype,
                        ),
                        num_kv_heads_by_tp // num_kv_heads,
                        axis=0,
                    ),
                    (None, "model", None),
                ),
                f"params.model.layers_{i}.self_attn.v_proj_lora.a_{seq_index}":
                _device_weight(
                    jnp.zeros(
                        (
                            self.config.hidden_size,
                            self.lora_config.max_lora_rank,
                        ),
                        dtype=self.dtype,
                    ),
                    (None, None),
                ),
                f"params.model.layers_{i}.self_attn.v_proj_lora.b_{seq_index}":
                _device_weight(
                    jnp.repeat(
                        jnp.zeros(
                            (
                                self.lora_config.max_lora_rank,
                                num_kv_heads,
                                head_dim,
                            ),
                            dtype=self.dtype,
                        ),
                        num_kv_heads_by_tp // num_kv_heads,
                        axis=0,
                    ),
                    (None, "model", None),
                ),
                f"params.model.layers_{i}.self_attn.o_proj_lora.a_{seq_index}":
                _device_weight(
                    jnp.zeros(
                        (
                            self.config.num_attention_heads,
                            head_dim,
                            self.lora_config.max_lora_rank,
                        ),
                        dtype=self.dtype,
                    ),
                    ("model", None, None),
                ),
                f"params.model.layers_{i}.self_attn.o_proj_lora.b_{seq_index}":
                _device_weight(
                    jnp.zeros(
                        (
                            self.lora_config.max_lora_rank,
                            self.config.hidden_size,
                        ),
                        dtype=self.dtype,
                    ),
                    (None, None),
                ),
                # Zero-initialize MLP LoRA weights.
                f"params.model.layers_{i}.mlp.gate_proj_lora.a_{seq_index}":
                _device_weight(
                    jnp.zeros(
                        (
                            self.config.hidden_size,
                            self.lora_config.max_lora_rank,
                        ),
                        dtype=self.dtype,
                    ),
                    (None, None),
                ),
                f"params.model.layers_{i}.mlp.gate_proj_lora.b_{seq_index}":
                _device_weight(
                    jnp.zeros(
                        (
                            self.lora_config.max_lora_rank,
                            self.config.intermediate_size,
                        ),
                        dtype=self.dtype,
                    ),
                    (None, "model"),
                ),
                f"params.model.layers_{i}.mlp.up_proj_lora.a_{seq_index}":
                _device_weight(
                    jnp.zeros(
                        (
                            self.config.hidden_size,
                            self.lora_config.max_lora_rank,
                        ),
                        dtype=self.dtype,
                    ),
                    (None, None),
                ),
                f"params.model.layers_{i}.mlp.up_proj_lora.b_{seq_index}":
                _device_weight(
                    jnp.zeros(
                        (
                            self.lora_config.max_lora_rank,
                            self.config.intermediate_size,
                        ),
                        dtype=self.dtype,
                    ),
                    (None, "model"),
                ),
                f"params.model.layers_{i}.mlp.down_proj_lora.a_{seq_index}":
                _device_weight(
                    jnp.zeros(
                        (
                            self.config.intermediate_size,
                            self.lora_config.max_lora_rank,
                        ),
                        dtype=self.dtype,
                    ),
                    ("model", None),
                ),
                f"params.model.layers_{i}.mlp.down_proj_lora.b_{seq_index}":
                _device_weight(
                    jnp.zeros(
                        (
                            self.lora_config.max_lora_rank,
                            self.config.hidden_size,
                        ),
                        dtype=self.dtype,
                    ),
                    (None, None),
                ),
            })
        return lora_params

    @staticmethod
    def get_lora_adapter_bytes(
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
        lora_config: LoRAConfig,
    ) -> int:
        tp_size = parallel_config.tensor_parallel_size
        num_lora_params = (
            model_config.get_num_layers() * lora_config.max_lora_rank * (
                7  # LoRA weights for q_proj A, k_proj A, v_proj A, o_proj B, gate_proj A, up_proj A, down_proj B
                * model_config.get_hidden_size() +
                2  # LoRA weights for q_proj B, o_proj A
                * model_config.get_num_attention_heads() // tp_size *
                model_config.get_head_dim() +
                2  # LoRA weights for k_proj B, v_proj B
                * model_config.get_num_kv_heads() // tp_size *
                model_config.get_head_dim() +
                3  # LoRA weights for gate_proj B, up_proj B, down_proj A
                * model_config.get_intermediate_size() // tp_size))
        dtype_size = model_config.dtype.dtype.itemsize
        lora_bytes = num_lora_params * dtype_size * lora_config.max_num_mem_cached_lora
        return lora_bytes

    def load_weights(
        self,
        model_name_or_path: str,
        cache_dir: Optional[str] = None,
    ) -> Dict[str, Any]:
        # Permute for sliced rotary
        def permute_qk(w, n_heads, dim1, dim2):
            return jnp.transpose(
                w.reshape((n_heads, dim1 // n_heads // 2, 2, dim2)),
                (0, 2, 1, 3)).reshape(dim1, dim2)

        def _device_weight(weight: jax.Array,
                           sharding_names: Tuple[str, ...]) -> jax.Array:
            return jax.device_put(
                weight, NamedSharding(self.mesh,
                                      PartitionSpec(*sharding_names)))

        params_dict = {}
        weights_iterator = hf_model_weights_iterator(model_name_or_path,
                                                     framework="flax",
                                                     cache_dir=cache_dir)
        reprocessing_items = deque()
        combined_iterator = itertools.chain(weights_iterator,
                                            reprocessing_items)

        for name, checkpoint_weight in combined_iterator:
            # llama 4 vision
            if "vision_model" in name:
                config_to_use = self.config.vision_config
            elif getattr(self.config, "text_config", None):
                config_to_use = self.config.text_config
            else:
                config_to_use = self.config

            should_permute_qk = (self.config.model_type == "llama4"
                                 and "vision" not in name)

            # Pad the head_dim to multiple of 128 as the PagedAttention kernel requires.
            if hasattr(config_to_use, "head_dim_original"):
                head_dim = config_to_use.head_dim_original
                head_dim_pad = config_to_use.head_dim - config_to_use.head_dim_original
            else:
                head_dim = config_to_use.head_dim
                head_dim_pad = 0
            num_kv_heads = getattr(
                config_to_use, "num_key_value_heads", None) or getattr(
                    config_to_use, "num_attention_heads", None)
            num_kv_heads_by_tp = get_num_kv_heads_by_tp(
                num_kv_heads, self.mesh.shape["model"])
            hidden_size = config_to_use.hidden_size
            num_attention_heads = config_to_use.num_attention_heads

            num_q_heads_by_tp = get_num_q_heads_by_tp(num_attention_heads,
                                                      num_kv_heads,
                                                      self.mesh.shape["model"])
            q_repeats = num_q_heads_by_tp // num_attention_heads

            if "inv_freq" in name or "rope.freqs" in name:
                continue

            key = name

            # I encourage everyone to start using direct mapping replacement instead of bunch of if-else conditions.
            # The if-else below has become extremely buggy and difficult to maintain for compatability purposes.
            MAPPING = {
                # Llama < 4: `model.embed_tokens` -> `params.embed_tokens`
                # Llama 4: `language_model.model.embed_tokens` -> `params.embed_tokens`
                r"(.*?)model.embed_tokens":
                r"params.embed_tokens",
                r"^model":
                r"params.model",
                # Llama 4:
                # MoE keys
                r"feed_forward.experts":
                r"experts",
                r"feed_forward.shared_expert":
                r"shared_expert",
                r"feed_forward.router.weight":
                r"experts.gate",
                # Embedding
                r"language_model.model.embed_tokens.weight":
                r"params.embed_tokens.weight",
                # Prefix
                r"layers.(\d+)":
                r"layers_\1",
                r"language_model.model":
                r"params.model",
                # Vision
                r"multi_modal_projector.linear_1.weight":
                r"vision_projection.weight",
                r"vision_model.layernorm_post":
                r"vision_model.vision_encoder.layernorm_post",
                r"vision_model.layernorm_pre":
                r"vision_model.vision_encoder.layernorm_pre",
                r"vision_model.class_embedding":
                r"vision_model.vision_encoder.class_embedding",
                r"vision_model.positional_embedding_vlm":
                r"vision_model.vision_encoder.positional_embedding_vlm",
                r"patch_embedding.linear":
                r"vision_encoder.conv1",
                r"vision_model.model.layers":
                r"vision_model.vision_encoder.model.layers",
                # LayerNorm (used in vision encoder) `weight` -> `scale`
                r"vision_model(.+?)norm(_post|_pre)?.weight":
                r"vision_model\1norm\2.scale",
                ###########################
                ###########################
            }
            import re

            for pattern, replacement in MAPPING.items():
                if replacement is None:
                    key = re.sub(pattern, "", key)  # an empty line
                    continue
                key = re.sub(pattern, replacement, key)

            key = key.replace("layers.", "model.layers_")
            key = key.replace("params.norm", "params.model.norm")

            if "gate_proj" in key or "up_proj" in key or "down_proj" in key:
                key = key.strip(".weight")
            if "lm_head" in key:
                key = "params.lm_head"
            if key == "norm.weight":
                key = "params.model.norm.weight"
            if not key.startswith("params"):
                key = f"params.{key}"
            if "gate_up_proj" in key:
                last_dim = checkpoint_weight.shape[-1] // 2
                gate_weight = checkpoint_weight[..., :last_dim]
                up_weight = checkpoint_weight[..., last_dim:]
                reprocessing_items.append(
                    (key.replace("gate_up_proj", "gate_proj"), gate_weight))
                reprocessing_items.append((key.replace("gate_up_proj",
                                                       "up_proj"), up_weight))
                continue

            weight = checkpoint_weight.astype(self.dtype)
            key = key.replace("attention.wo.weight", "self_attn.o_proj.weight")
            key = key.replace("feed_forward.norm", "post_attention_layernorm")
            key = key.replace("params.params", "params.model")

            replicated_params_endswith = [
                "class_embedding",
                "positional_embedding_vlm",
                "bias",
                "scale",
                "norm.weight",
                "layernorm_post.weight",
                "layernorm_pre.weight",
            ]

            if "embed_tokens" in key:
                weight = _device_weight(weight, ("model", None))
            elif "lm_head" in key:
                weight = jnp.transpose(weight)
                weight = _device_weight(weight, (None, "model"))
            elif "experts.gate_proj" in key or "experts.up_proj" in key:
                weight = weight.reshape(
                    config_to_use.num_local_experts,
                    config_to_use.hidden_size,
                    config_to_use.intermediate_size,
                )
                weight = _device_weight(weight, (None, None, "model"))
            elif "experts.down_proj" in key:
                weight = weight.reshape(
                    config_to_use.num_local_experts,
                    config_to_use.intermediate_size,
                    config_to_use.hidden_size,
                )
                weight = _device_weight(weight, (None, "model", None))
            elif "gate_proj" in key or "up_proj" in key or "fc1.weight" in key:
                weight = jnp.transpose(weight)
                weight = _device_weight(weight, (None, "model"))
            elif "down_proj" in key or "fc2.weight" in key:
                weight = jnp.transpose(weight)
                weight = _device_weight(weight, ("model", None))
            elif "experts.gate" in key:
                weight = jnp.transpose(weight)
                weight = _device_weight(weight, (None, None))
            elif "q_proj" in key:
                if key.endswith("bias"):
                    weight = jnp.reshape(
                        weight,
                        (
                            config_to_use.num_attention_heads,
                            head_dim,
                        ),
                    )
                    if head_dim_pad:
                        weight = jnp.pad(weight, ((0, 0), (0, head_dim_pad)))

                    weight = jnp.repeat(weight, q_repeats, axis=0)
                    weight = _device_weight(weight, ("model", None))
                elif key.endswith("weight"):
                    if should_permute_qk:
                        logger.warning(f">>> Permuting qk for {key}")
                        weight = permute_qk(
                            weight,
                            config_to_use.num_attention_heads,
                            config_to_use.hidden_size,
                            config_to_use.hidden_size,
                        )
                    weight = jnp.reshape(
                        weight,
                        (
                            config_to_use.num_attention_heads,
                            head_dim,
                            config_to_use.hidden_size,
                        ),
                    )
                    if head_dim_pad:
                        weight = jnp.pad(
                            weight,
                            ((0, 0), (0, head_dim_pad), (0, 0)),
                        )
                    weight = jnp.repeat(weight, q_repeats, axis=0)
                    weight = jnp.transpose(weight, (0, 2, 1))
                    weight = _device_weight(weight, ("model", None, None))
            elif "k_proj" in key or "v_proj" in key:
                if key.endswith("bias"):
                    weight = jnp.reshape(
                        weight,
                        (
                            num_kv_heads,
                            head_dim,
                        ),
                    )
                    if head_dim_pad:
                        weight = jnp.pad(weight, ((0, 0), (0, head_dim_pad)))

                    weight = _device_weight(weight, ("model", None))
                elif key.endswith("weight"):
                    if "k_proj" in key:
                        if should_permute_qk:
                            logger.warning(f">>> Permuting qk for {key}")
                            weight = permute_qk(
                                weight,
                                num_kv_heads,
                                num_kv_heads * head_dim,
                                config_to_use.hidden_size,
                            )
                    weight = jnp.reshape(
                        weight,
                        (
                            num_kv_heads,
                            head_dim,
                            hidden_size,
                        ),
                    )
                    if head_dim_pad:
                        weight = jnp.pad(
                            weight,
                            ((0, 0), (0, head_dim_pad), (0, 0)),
                        )
                    weight = jnp.repeat(weight,
                                        num_kv_heads_by_tp // num_kv_heads,
                                        axis=0)
                    weight = jnp.transpose(weight, (0, 2, 1))
                    weight = _device_weight(weight, ("model", None, None))
            elif "o_proj" in key:
                if key.endswith("bias"):
                    weight = _device_weight(weight, (None, ))
                elif key.endswith("weight"):
                    weight = jnp.reshape(
                        weight,
                        (
                            hidden_size,
                            num_attention_heads,
                            head_dim,
                        ),
                    )
                    if head_dim_pad:
                        weight = jnp.pad(
                            weight,
                            ((0, 0), (0, 0), (0, head_dim_pad)),
                        )
                    weight = jnp.transpose(weight, (1, 2, 0))
                    target_weight = jnp.zeros(
                        (weight.shape[0] * q_repeats, weight.shape[1],
                         weight.shape[2]),
                        dtype=weight.dtype,
                    )
                    target_weight = target_weight.at[::q_repeats].set(weight)
                    weight = _device_weight(target_weight,
                                            ("model", None, None))
            elif any([key.endswith(el) for el in replicated_params_endswith]):
                weight = _device_weight(weight, (None, ))
            elif key.endswith("vision_projection.weight"):
                weight = jnp.transpose(weight)
                weight = _device_weight(weight, (None, "model"))
            elif key.endswith("conv1.weight"):
                weight = jnp.transpose(weight)
                weight = _device_weight(weight, (None, "model"))
            else:
                logger.warning(f"Unhandled key: {key}")

            params_dict[key] = weight

        if self.lora_config is not None and self.lora_config.enable_lora:
            for seq_index in range(self.lora_config.max_num_lora):
                lora_params = self.load_zero_lora_adapter(seq_index)
                params_dict.update(lora_params)
        return unflatten_dict(params_dict, ".")

    def load_quant_weights(
        self,
        model_name_or_path: str,
        quantization_config: QuantizationConfig,
        cache_dir: Optional[str] = None,
    ) -> Dict[str, Any]:
        # TODO: Refactor code to hide different weight loading functions.
        # TODO: Update code for additional quantization weight formats.

        def _device_weight(weight: jax.Array,
                           sharding_names: Tuple[str, ...]) -> jax.Array:
            return jax.device_put(
                weight, NamedSharding(self.mesh,
                                      PartitionSpec(*sharding_names)))

        num_kv_heads = self.config.num_key_value_heads
        num_kv_heads_by_tp = get_num_kv_heads_by_tp(num_kv_heads,
                                                    self.mesh.shape["model"])
        params_dict = {}
        for name, checkpoint_weight in hf_model_weights_iterator(
                model_name_or_path, framework="flax", cache_dir=cache_dir):
            if "inv_freq" in name:
                pass
            if "weight_format" in name:
                # Skip tensors that specify the quantization format, such as 0
                # signifying row quantization.
                continue

            key = name.replace("model", "params")
            key = key.replace("layers.", "model.layers_")
            key = key.replace("params.norm", "params.model.norm")
            if "gate_proj" in key or "up_proj" in key or "down_proj" in key:
                key = key.replace(".weight", "_weight")
                key = key.replace(".SCB", "_SCB")

                # AWQ Quantization
                key = key.replace(".qweight", "_qweight")
                key = key.replace(".qzeros", "_qzeros")
                key = key.replace(".scales", "_scales")
            if "lm_head" in key:
                key = "params.lm_head"

            is_awq_key = "qweight" in key or "qzeros" in key or "scales" in key
            weight = checkpoint_weight
            if "proj_weight" in key or "proj.weight" in key:
                if quantization_config.load_in_8bit:
                    weight = checkpoint_weight.astype(jnp.int8)
                else:
                    raise ValueError(
                        "Only 8-bit quantization is supported for bitsandbytes."
                    )
            elif is_awq_key:  # Don't naively castly dtype
                pass
            else:
                # This is needed as self.dtype is not necessarily equal to checkpoint_weight.dtype. For instance, we autocast float16 to bfloat16.
                weight = checkpoint_weight.astype(self.dtype)

            if is_awq_key:
                # Unpack AWQ weights
                if "qweight" in key or "qzeros" in key:
                    # Move from CPU to TPU first to allow uint4 HLOs.
                    weight = jax.device_put(weight,
                                            device=jax.devices("tpu")[0])
                    weight = unpack(weight, bits=4)
                    weight = reverse_awq_order(weight, bits=4)
                    assert weight.dtype == jnp.uint4

                # AWQ weights are already transposed.
                weight = jnp.transpose(weight)

            if "embed_tokens" in key:
                weight = _device_weight(weight, ("model", None))
            elif "lm_head" in key:
                weight = jnp.transpose(weight)
                weight = _device_weight(weight, (None, "model"))
            elif "gate_proj" in key or "up_proj" in key:
                if "SCB" in key:
                    weight = _device_weight(weight, (None, ))
                else:
                    weight = jnp.transpose(weight)
                    weight = _device_weight(weight, (None, "model"))
            elif "down_proj" in key:
                if "SCB" in key:
                    weight = _device_weight(weight, ("model", ))
                else:
                    weight = jnp.transpose(weight)
                    weight = _device_weight(weight, ("model", None))
            elif "q_proj" in key:
                if "SCB" in key:
                    weight = _device_weight(weight, (None, ))
                else:
                    if "qzero" in key or "scale" in key:
                        last_dim = (self.config.hidden_size //
                                    self.quantization_config.group_size)
                    else:
                        last_dim = self.config.hidden_size

                    # Workaround for b/349670488. TODO: Remove this once the bug is resolved.
                    if weight.dtype == jnp.uint4:
                        weight = weight.astype(jnp.int32)
                        weight = jnp.reshape(
                            weight,
                            (self.config.num_attention_heads, -1, last_dim))
                        weight = weight.astype(jnp.uint4)
                    else:
                        weight = jnp.reshape(
                            weight,
                            (self.config.num_attention_heads, -1, last_dim))

                    weight = jnp.transpose(weight, (0, 2, 1))
                    weight = _device_weight(weight, ("model", None, None))
            elif "k_proj" in key or "v_proj" in key:
                if "SCB" in key:
                    weight = _device_weight(weight, (None, ))
                else:
                    if "qzero" in key or "scale" in key:
                        last_dim = (self.config.hidden_size //
                                    self.quantization_config.group_size)
                    else:
                        last_dim = self.config.hidden_size

                    # Workaround for b/349670488. TODO: Remove this once the bug is resolved.
                    if weight.dtype == jnp.uint4:
                        weight = weight.astype(jnp.int32)
                        weight = jnp.reshape(weight,
                                             (num_kv_heads, -1, last_dim))
                        weight = weight.astype(jnp.uint4)
                    else:
                        weight = jnp.reshape(weight,
                                             (num_kv_heads, -1, last_dim))

                    weight = jnp.repeat(weight,
                                        num_kv_heads_by_tp // num_kv_heads,
                                        axis=0)
                    weight = jnp.transpose(weight, (0, 2, 1))
                    weight = _device_weight(weight, ("model", None, None))
            elif "o_proj" in key:
                if "SCB" in key:
                    weight = _device_weight(weight, (None, ))
                else:
                    weight = jnp.reshape(
                        weight,
                        (self.config.hidden_size,
                         self.config.num_attention_heads, -1),
                    )
                    weight = jnp.transpose(weight, (1, 2, 0))
                    weight = _device_weight(weight, ("model", None, None))
            elif "norm" in key:
                weight = _device_weight(weight, (None, ))

            params_dict[key] = weight

        return unflatten_dict(params_dict, ".")
