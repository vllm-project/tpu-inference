import itertools
from collections import deque
from typing import Any, Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.traverse_util import unflatten_dict
from flax.typing import PRNGKey
from jax.sharding import Mesh, NamedSharding, PartitionSpec
from transformers import LlamaConfig, modeling_flax_utils
from vllm.config import VllmConfig

from tpu_commons.logger import init_logger
from tpu_commons.models.jax import layers
from tpu_commons.models.jax.param_init import sharding_init
from tpu_commons.models.jax.rope.generic_rope import apply_rope
from tpu_commons.models.jax.sampling import sample
from tpu_commons.models.jax.utils.weight_utils import (
    get_num_kv_heads_by_tp, get_num_q_heads_by_tp, hf_model_weights_iterator)

logger = init_logger(__name__)

KVCache = Tuple[jax.Array, jax.Array]


class LlamaMLP(nn.Module):
    hidden_size: int
    intermediate_size: int
    act: str
    dtype: jnp.dtype
    mesh: Mesh

    @nn.compact
    def __call__(self, x) -> jax.Array:
        gate_proj = self.param(
            "gate_proj",
            sharding_init((None, "model"), self.mesh),
            (self.hidden_size, self.intermediate_size),
            self.dtype,
        )

        up_proj = self.param(
            "up_proj",
            sharding_init((None, "model"), self.mesh),
            (self.hidden_size, self.intermediate_size),
            self.dtype,
        )
        down_proj = self.param(
            "down_proj",
            sharding_init(("model", None), self.mesh),
            (self.intermediate_size, self.hidden_size),
            self.dtype,
        )

        up = jnp.dot(x, up_proj)
        fuse = up
        gate = jnp.dot(x, gate_proj)
        gate = modeling_flax_utils.ACT2FN[self.act](gate)
        fuse = gate * up
        return jnp.dot(fuse, down_proj)


class LlamaAttention(nn.Module):
    config: LlamaConfig
    dtype: jnp.dtype
    mesh: Mesh

    def setup(self) -> None:
        self.hidden_size = self.config.hidden_size
        # TODO(xiang): shard by TP
        self.num_heads = self.config.num_attention_heads
        self.num_kv_heads = self.config.num_key_value_heads
        self.rope_theta = self.config.rope_theta
        self.rope_scaling = getattr(self.config, "rope_scaling", None)
        self.head_dim = self.config.head_dim

        self.q_proj = layers.Einsum(
            shape=(self.num_heads, self.hidden_size, self.head_dim),
            dtype=self.dtype,
            named_axes=("model", None, None),
            mesh=self.mesh,
        )
        self.k_proj = layers.Einsum(
            shape=(self.num_kv_heads, self.hidden_size, self.head_dim),
            dtype=self.dtype,
            named_axes=("model", None, None),
            mesh=self.mesh,
        )
        self.v_proj = layers.Einsum(
            shape=(self.num_kv_heads, self.hidden_size, self.head_dim),
            dtype=self.dtype,
            named_axes=("model", None, None),
            mesh=self.mesh,
        )
        self.o_proj = layers.Einsum(
            shape=(self.num_heads, self.head_dim, self.hidden_size),
            dtype=self.dtype,
            named_axes=("model", None, None),
            mesh=self.mesh,
        )
        self.flash_attention = layers.sharded_flash_attention(self.mesh)
        self.paged_attention = layers.sharded_paged_attention(self.mesh)

    def __call__(
        self,
        is_prefill: bool,
        kv_cache: Optional[KVCache],
        x: jax.Array,
        attention_metadata: layers.AttentionMetadata,
    ) -> Tuple[KVCache, jax.Array, Optional[jax.Array]]:
        # B: batch_size
        # T: seq_len
        # N: num_heads
        # K: num_kv_heads
        # D: hidden_size
        # H: head_dim
        # L: num_blocks
        # S: block_size

        md = attention_metadata

        # q: (B, N, T, H)
        q = self.q_proj("BTD,NDH->BNTH", x)
        q = apply_rope(q, md.input_positions, self.head_dim, self.rope_theta,
                       self.rope_scaling)
        q = q * self.head_dim**-0.5

        # k: (B, K, T, H)
        k = self.k_proj("BTD,KDH->BKTH", x)
        k = apply_rope(k, md.input_positions, self.head_dim, self.rope_theta,
                       self.rope_scaling)

        # v: (B, K, T, H)
        v = self.v_proj("BTD,KDH->BKTH", x)

        # (K, L, S, H)
        k_cache, v_cache = kv_cache
        k_cache = layers.update_cache(is_prefill, k_cache,
                                      md.kv_cache_write_indices, k)
        v_cache = layers.update_cache(is_prefill, v_cache,
                                      md.kv_cache_write_indices, v)

        if is_prefill:
            # (B, N, T, H)
            # TODO(xiang): support MQA and GQA
            if self.num_kv_heads != self.num_heads:
                k = jnp.repeat(k, self.num_heads // self.num_kv_heads, axis=1)
                v = jnp.repeat(v, self.num_heads // self.num_kv_heads, axis=1)
            outputs = self.flash_attention(q, k, v)
        else:
            # (B, N, H)
            q = jnp.squeeze(q, 2)
            outputs = self.paged_attention(q, k_cache, v_cache, md.seq_lens,
                                           md.block_indices)
            # (B, N, 1, H)
            outputs = jnp.expand_dims(outputs, 2)

        # (B, T, D)
        o = self.o_proj("BNTH,NHD->BTD", outputs)
        return (k_cache, v_cache), o


class LlamaDecoderLayer(nn.Module):
    config: LlamaConfig
    dtype: jnp.dtype
    mesh: Mesh

    def setup(self) -> None:
        rms_norm_eps = self.config.rms_norm_eps
        hidden_size = self.config.hidden_size
        intermediate_size = self.config.intermediate_size
        act = self.config.hidden_act

        self.input_layernorm = layers.RMSNorm(
            rms_norm_eps=rms_norm_eps,
            dtype=self.dtype,
            mesh=self.mesh,
        )

        self.self_attn = LlamaAttention(
            config=self.config,
            dtype=self.dtype,
            mesh=self.mesh,
        )

        self.post_attention_layernorm = layers.RMSNorm(
            rms_norm_eps=rms_norm_eps,
            dtype=self.dtype,
            mesh=self.mesh,
        )

        self.mlp = LlamaMLP(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            act=act,
            dtype=self.dtype,
            mesh=self.mesh,
        )

    def __call__(
        self,
        is_prefill: bool,
        kv_cache: KVCache,
        x: jax.Array,
        attention_metadata: layers.AttentionMetadata,
    ) -> Tuple[KVCache, jax.Array]:
        # Self attention.
        hidden_states = self.input_layernorm(x)
        kv_cache, attn_output = self.self_attn(
            is_prefill,
            kv_cache,
            hidden_states,
            attention_metadata,
        )
        attn_output += x

        # MLP.
        residual = attn_output
        attn_output = self.post_attention_layernorm(attn_output)
        outputs = self.mlp(attn_output)
        outputs = residual + outputs
        return kv_cache, outputs


class LlamaModel(nn.Module):
    vllm_config: VllmConfig
    mesh: Mesh

    def setup(self) -> None:
        model_config = self.vllm_config.model_config
        hf_config = model_config.hf_config

        self.layers = [
            LlamaDecoderLayer(
                config=hf_config,
                dtype=model_config.dtype,
                mesh=self.mesh,
            ) for i in range(hf_config.num_hidden_layers)
        ]
        self.norm = layers.RMSNorm(
            rms_norm_eps=hf_config.rms_norm_eps,
            dtype=model_config.dtype,
            mesh=self.mesh,
        )

    def __call__(
        self,
        is_prefill: bool,
        kv_caches: List[KVCache],
        x: jax.Array,
        attention_metadata: layers.AttentionMetadata,
    ) -> Tuple[List[KVCache], jax.Array]:
        for i, layer in enumerate(self.layers):
            kv_cache = kv_caches[i]
            kv_cache, x = layer(
                is_prefill,
                kv_cache,
                x,
                attention_metadata,
            )
            kv_caches[i] = kv_cache
        x = self.norm(x)
        return kv_caches, x


class LlamaForCausalLM(nn.Module):
    vllm_config: VllmConfig
    rng: PRNGKey
    mesh: Mesh

    def setup(self) -> None:
        model_config = self.vllm_config.model_config
        hf_config = model_config.hf_config
        self.embed_tokens = layers.Embedder(
            vocab_size=model_config.get_vocab_size(),
            hidden_size=model_config.get_hidden_size(),
            dtype=model_config.dtype,
            mesh=self.mesh,
        )
        self.model = LlamaModel(
            vllm_config=self.vllm_config,
            mesh=self.mesh,
        )
        # TODO(xiang): check this for llama3.2
        if getattr(hf_config, "use_embedding_as_last_layer", False):
            self.lm_head = None
        else:
            self.lm_head = self.param(
                "lm_head",
                sharding_init(
                    (None, "model"),
                    self.mesh,
                ),
                (model_config.get_hidden_size(),
                 model_config.get_vocab_size()),
                model_config.dtype,
            )

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
        *args,
    ) -> Tuple[List[KVCache], jax.Array, jax.Array]:
        x = self.embed_tokens.encode(input_ids)

        kv_caches, x = self.model(
            is_prefill,
            kv_caches,
            x,
            attention_metadata,
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
        )
        return kv_caches, next_tokens, logits

    # TODO(xiang): fix this
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

        model_config = self.vllm_config.model_config
        hf_config = model_config.hf_config

        for name, checkpoint_weight in combined_iterator:
            # llama 4 vision
            if "vision_model" in name:
                config_to_use = hf_config.vision_config
            elif getattr(hf_config, "text_config", None):
                config_to_use = hf_config.text_config
            else:
                config_to_use = hf_config

            should_permute_qk = (hf_config.model_type == "llama4"
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

            weight = checkpoint_weight.astype(model_config.dtype)
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

        # TODO (jacobplatin)
        # if self.lora_config is not None and self.lora_config.enable_lora:
        #     for seq_index in range(self.lora_config.max_num_lora):
        #         lora_params = self.load_zero_lora_adapter(seq_index)
        #         params_dict.update(lora_params)
        return unflatten_dict(params_dict, ".")
