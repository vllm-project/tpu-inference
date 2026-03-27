# Qwen3 VL MoE - Text model inherits from Qwen3 MoE with MRoPE attention
# and DeepStack support. Vision components are identical to Qwen3 VL dense.

import re
from itertools import islice
from typing import List, Optional, Tuple

import jax
import jax.numpy as jnp
import torchax
from flax import nnx
from jax.sharding import Mesh
from vllm.config import VllmConfig

from tpu_inference.distributed.jax_parallel_state import get_pp_group
from tpu_inference.layers.jax.linear import JaxEinsum
from tpu_inference.layers.jax.norm import JaxRmsNorm
from tpu_inference.layers.jax.pp_utils import PPMissingLayer, make_layers
from tpu_inference.layers.jax.embed import JaxEmbed
from tpu_inference.logger import init_logger
from tpu_inference.models.jax.qwen3_moe import (
    Qwen3MoeDecoderLayer,
    Qwen3MoeModel,
    Qwen3MoeSparseMoeBlock,
)
from tpu_inference.models.jax.qwen2 import Qwen2MLP as Qwen3MoeDenseMLP
from tpu_inference.models.jax.qwen3_vl import (
    build_mrope_input_positions,
    _ModelConfigAdapter,
    _VllmConfigAdapter,
    Qwen3VLForConditionalGeneration,
    Qwen3VLTextAttention,
    Qwen3VLVisionTransformer,
)
from tpu_inference.models.jax.utils.multi_modal_utils import (
    merge_multimodal_embeddings,
)
from tpu_inference.models.jax.utils.weight_utils import (
    _load_and_shard_weight,
    assign_and_shard_param,
    check_all_loaded,
    get_default_maps,
    model_weights_generator,
)

init_fn = nnx.initializers.uniform()

logger = init_logger(__name__)


class Qwen3VLMoeDecoderLayer(Qwen3MoeDecoderLayer):
    """MoE decoder layer with MRoPE-aware attention and dense MLP fallback.

    Overrides __init__ to:
    - Use Qwen3VLTextAttention (MRoPE-aware) instead of Qwen3Attention
    - Support dense MLP fallback for non-MoE layers
    Inherits __call__ from Qwen3MoeDecoderLayer unchanged.
    """

    def __init__(self,
                 config,
                 dtype: jnp.dtype,
                 rng: nnx.Rngs,
                 mesh: Mesh,
                 kv_cache_dtype: str,
                 quant_config,
                 layer_idx: int,
                 vllm_config: VllmConfig,
                 prefix: str = ""):
        # Skip Qwen3MoeDecoderLayer.__init__ — we set up all attributes here
        # but inherit __call__ which uses self.input_layernorm, self.self_attn,
        # self.post_attention_layernorm, self.mlp
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
        # MRoPE-aware attention for VL (handles 3D position IDs)
        self.self_attn = Qwen3VLTextAttention(
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

        # MoE or dense MLP based on layer index
        mlp_only_layers = getattr(config, "mlp_only_layers", [])
        num_experts = getattr(config, "num_experts", 0)
        decoder_sparse_step = getattr(config, "decoder_sparse_step", 1)
        use_moe = (
            layer_idx not in mlp_only_layers
            and num_experts > 0
            and (layer_idx + 1) % decoder_sparse_step == 0
        )
        if use_moe:
            self.mlp = Qwen3MoeSparseMoeBlock(
                vllm_config=vllm_config,
                rng=rng,
                mesh=mesh,
                prefix=prefix + ".mlp",
            )
        else:
            self.mlp = Qwen3MoeDenseMLP(
                config=config,
                dtype=dtype,
                rng=rng,
                quant_config=quant_config,
                prefix=prefix + ".mlp",
            )


class Qwen3VLMoeTextModel(Qwen3MoeModel):
    """Text model for Qwen3VL MoE with MRoPE and DeepStack support.

    Overrides __init__ to use Qwen3VLMoeDecoderLayer (with MRoPE attention).
    Overrides __call__ to add DeepStack visual feature injection.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        rng: nnx.Rngs,
        mesh: Mesh,
    ):
        # Adapt the VL config so text_config attributes are accessible
        # directly on hf_config (same pattern as Qwen3VLModel).
        adapted = _VllmConfigAdapter(vllm_config)
        model_config = adapted.model_config
        hf_config = model_config.hf_config
        vocab_size = model_config.get_vocab_size()
        dtype = model_config.dtype
        rms_norm_eps = hf_config.rms_norm_eps
        hidden_size = hf_config.hidden_size
        prefix = "model.language_model"

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
                quant_config=adapted.quant_config,
                prefix=prefix + ".embed_tokens",
            )
        else:
            self.embed_tokens = PPMissingLayer()

        self.start_layer, self.end_layer, self.layers = make_layers(
            hf_config.num_hidden_layers,
            lambda layer_index: Qwen3VLMoeDecoderLayer(
                config=hf_config,
                dtype=dtype,
                rng=rng,
                mesh=mesh,
                kv_cache_dtype=adapted.cache_config.cache_dtype,
                quant_config=adapted.quant_config,
                layer_idx=layer_index,
                vllm_config=adapted,
                prefix=f"{prefix}.layers.{layer_index}",
            ))

        if self.is_last_rank:
            self.norm = JaxRmsNorm(
                hidden_size,
                epsilon=rms_norm_eps,
                param_dtype=dtype,
                scale_init=nnx.with_partitioning(init_fn, (None, )),
                rngs=rng,
                quant_config=adapted.quant_config,
                prefix=prefix + ".final_layernorm",
            )
        else:
            self.norm = PPMissingLayer()

    def _inject_visual_features(
        self,
        hidden_states: jax.Array,
        visual_pos_mask: jax.Array,
        visual_embeds: jax.Array,
    ) -> jax.Array:
        """Add DeepStack visual features at masked positions.

        Args:
            hidden_states: (seq_len, hidden_size) or (batch, seq_len, hidden_size)
            visual_pos_mask: Boolean mask matching hidden_states without the last dim
            visual_embeds: Visual features (num_visual_tokens, hidden_size)

        Returns:
            Updated hidden_states with visual features added
        """
        flat_hidden = hidden_states.reshape(-1, hidden_states.shape[-1])
        mask = jnp.broadcast_to(visual_pos_mask, hidden_states.shape[:-1])
        flat_mask = mask.reshape(-1).astype(jnp.bool_)

        visual_embeds = visual_embeds.astype(flat_hidden.dtype)
        dummy_row = jnp.zeros((1, flat_hidden.shape[-1]), dtype=flat_hidden.dtype)
        padded_embeds = jnp.concatenate([dummy_row, visual_embeds, dummy_row], axis=0)
        gather_indices = jnp.cumsum(flat_mask, dtype=jnp.int32)
        max_index = visual_embeds.shape[0] + 1
        gather_indices = jnp.minimum(gather_indices, max_index)

        updates = padded_embeds[gather_indices] * flat_mask[:, None]
        updated = flat_hidden + updates

        return updated.reshape(hidden_states.shape)

    def __call__(
        self,
        kv_caches: List[jax.Array],
        input_ids: Optional[jax.Array],
        attention_metadata,
        inputs_embeds: Optional[jax.Array] = None,
        visual_pos_mask: Optional[jax.Array] = None,
        deepstack_visual_embeds: Optional[List[jax.Array]] = None,
    ) -> Tuple[List[jax.Array], jax.Array]:
        """Forward pass with KV cache, MRoPE, and DeepStack support."""
        if inputs_embeds is not None:
            x = inputs_embeds
        else:
            x = self.embed_tokens(input_ids)

        new_kv_caches = []
        for i, layer in enumerate(
            islice(self.layers, self.start_layer, self.end_layer)
        ):
            if isinstance(layer, PPMissingLayer):
                new_kv_caches.append(kv_caches[i])
                continue

            global_i = self.start_layer + i
            kv_cache = kv_caches[i]
            kv_cache, x = layer(kv_cache, x, attention_metadata)
            new_kv_caches.append(kv_cache)

            if (
                deepstack_visual_embeds is not None
                and global_i < len(deepstack_visual_embeds)
                and visual_pos_mask is not None
            ):
                x = self._inject_visual_features(
                    x, visual_pos_mask, deepstack_visual_embeds[global_i]
                )

        if self.is_last_rank:
            x = self.norm(x)

        return new_kv_caches, x


class Qwen3VLMoeForConditionalGeneration(Qwen3VLForConditionalGeneration):
    """Qwen3-VL MoE wrapper that reuses the dense VL multimodal surface."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        rng_key: jax.Array,
        mesh: Mesh,
    ):
        self.vllm_config = vllm_config
        self.rng = nnx.Rngs(rng_key)
        self.mesh = mesh

        config = vllm_config.model_config.hf_config
        self.config = config
        text_config = getattr(config, "text_config", config)

        self.visual = Qwen3VLVisionTransformer(
            vllm_config=vllm_config,
            rngs=self.rng,
            mesh=mesh,
            norm_eps=getattr(text_config, "rms_norm_eps", 1e-6),
        )
        self.language_model = Qwen3VLMoeTextModel(
            vllm_config=vllm_config,
            rng=self.rng,
            mesh=mesh,
        )

        model_config = vllm_config.model_config
        if not config.tie_word_embeddings:
            vocab_size = model_config.get_vocab_size()
            hidden_size = text_config.hidden_size
            self.lm_head = JaxEinsum(
                einsum_str="TD,DV->TV",
                kernel_shape=(hidden_size, vocab_size),
                dtype=model_config.dtype,
                rngs=self.rng,
                quant_config=vllm_config.quant_config,
                kernel_init=nnx.with_partitioning(init_fn, (None, "model")),
            )

        self.image_token_id = config.image_token_id
        self.video_token_id = config.video_token_id
        self.vision_start_token_id = getattr(config, "vision_start_token_id",
                                             151652)
        self.spatial_merge_size = config.vision_config.spatial_merge_size

    def get_input_embeddings(
        self,
        input_ids: jax.Array,
        multimodal_embeddings: Optional[jax.Array],
    ) -> jax.Array:
        inputs_embeds = self.language_model.embed_tokens(input_ids)
        if multimodal_embeddings is not None and multimodal_embeddings.shape[0] != 0:
            inputs_embeds = merge_multimodal_embeddings(
                input_ids,
                inputs_embeds,
                multimodal_embeddings,
                [self.image_token_id, self.video_token_id],
            )
        return inputs_embeds

    def embed_input_ids(
        self,
        input_ids: jax.Array,
        multimodal_embeddings: Optional[jax.Array] = None,
        *,
        is_multimodal: jax.Array | None = None,
    ) -> jax.Array:
        del is_multimodal
        return self.get_input_embeddings(input_ids, multimodal_embeddings)

    def get_mrope_input_positions(
        self,
        input_tokens: List[int],
        hf_config=None,
        image_grid_thw=None,
        video_grid_thw=None,
        second_per_grid_ts=None,
        context_len: int = 0,
        seq_len: Optional[int] = None,
        audio_feature_lengths=None,
        use_audio_in_video: bool = False,
    ) -> Tuple[jax.Array, int]:
        del second_per_grid_ts, audio_feature_lengths, use_audio_in_video

        if hf_config is None:
            hf_config = self.config

        if video_grid_thw is not None:
            expanded_video = []
            for t, h, w in video_grid_thw:
                expanded_video.extend([(1, int(h), int(w))] * int(t))
            video_grid_thw = expanded_video

        llm_positions, mrope_position_delta = build_mrope_input_positions(
            input_tokens=input_tokens,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            image_token_id=hf_config.image_token_id,
            video_token_id=hf_config.video_token_id,
            vision_start_token_id=getattr(hf_config, "vision_start_token_id",
                                          self.vision_start_token_id),
            spatial_merge_size=hf_config.vision_config.spatial_merge_size,
        )
        llm_positions = llm_positions[:, context_len:seq_len]
        return llm_positions, mrope_position_delta

    @staticmethod
    def _clear_loaded_expert_buffers(experts,
                                     loaded_names: Optional[set[str]]) -> None:
        if not loaded_names:
            return

        for param_name in loaded_names:
            param = getattr(experts, param_name, None)
            weights_to_load = getattr(param, "_weights_to_load", None)
            if weights_to_load is None:
                continue
            param._weights_to_load = [None] * len(weights_to_load)

    def _skip_non_expert_weight(self, hf_key: str) -> bool:
        layer_match = re.match(r".*layers\.(\d+)\.(.*)", hf_key)
        if layer_match is None:
            if (hf_key == "language_model.embed_tokens.weight"
                    and isinstance(self.language_model.embed_tokens,
                                   PPMissingLayer)):
                return True
            if (hf_key == "language_model.norm.weight"
                    and isinstance(self.language_model.norm, PPMissingLayer)):
                return True
            return False

        layer_idx = int(layer_match.group(1))
        suffix = layer_match.group(2)
        layer = self.language_model.layers[layer_idx]
        if isinstance(layer, PPMissingLayer):
            return True

        is_moe_layer = hasattr(layer.mlp, "experts")
        if suffix == "mlp.gate.weight":
            return not is_moe_layer
        if suffix.startswith("mlp.experts."):
            return not is_moe_layer
        if suffix in {
                "mlp.gate_proj.weight",
                "mlp.up_proj.weight",
                "mlp.down_proj.weight",
        }:
            return is_moe_layer
        return False

    def _load_moe_expert_weight(self, hf_key, hf_weight,
                                loaded_expert_modules) -> None:
        indexed_match = re.match(
            r".*layers\.(\d+)\.mlp\.experts\.(\d+\.(?:gate_proj|up_proj|down_proj)(?:\.weight)?)$",
            hf_key,
        )
        fused_match = re.match(
            r".*layers\.(\d+)\.mlp\.experts\.(gate_up_proj|down_proj)(?:\.weight)?$",
            hf_key,
        )
        if not indexed_match and not fused_match:
            return

        layer_idx = int((indexed_match or fused_match).group(1))
        layer = self.language_model.layers[layer_idx]
        if isinstance(layer, PPMissingLayer) or not hasattr(layer.mlp,
                                                            "experts"):
            return

        experts = layer.mlp.experts
        loaded_expert_modules[layer_idx] = experts
        env = torchax.default_env()

        if indexed_match:
            expert_suffix = indexed_match.group(2)
            if not expert_suffix.endswith(".weight"):
                expert_suffix += ".weight"
            loaded_names = experts.load_weights([(expert_suffix, hf_weight)])
            if getattr(experts, "quant_method", None) is None:
                self._clear_loaded_expert_buffers(experts, loaded_names)
            return

        fused_name = fused_match.group(2)
        if fused_name == "down_proj":
            target_shape = tuple(experts.kernel_down_proj_EFD.value.shape)
            source_shape = tuple(hf_weight.shape)
            permute_dims = None
            if source_shape != target_shape:
                swapped_shape = source_shape[:-2] + (
                    source_shape[-1], source_shape[-2])
                if swapped_shape == target_shape:
                    permute_dims = (0, 2, 1)
                else:
                    raise ValueError(
                        "Unsupported fused down_proj layout for "
                        f"language_model.layers.{layer_idx}.mlp.experts: "
                        f"source {source_shape} vs target {target_shape}")
            assign_and_shard_param(
                experts.kernel_down_proj_EFD,
                jnp.transpose(env.t2j_copy(hf_weight), permute_dims)
                if permute_dims is not None else env.t2j_copy(hf_weight),
                param_name=(
                    f"language_model.layers.{layer_idx}.mlp.experts.down_proj"),
            )
            return

        E, D, F = experts.kernel_gating_EDF.value.shape
        fused_shape = tuple(hf_weight.shape)
        if fused_shape == (E, 2 * F, D):
            chunk_dim = 1
            permute_dims = (0, 2, 1)
        elif fused_shape == (E, D, 2 * F):
            chunk_dim = 2
            permute_dims = None
        else:
            raise ValueError(
                "Unsupported fused gate_up_proj layout for "
                f"language_model.layers.{layer_idx}.mlp.experts: "
                f"source {fused_shape} vs expected EDF=({E}, {D}, {F})")
        gate_proj, up_proj = hf_weight.chunk(2, dim=chunk_dim)
        assign_and_shard_param(
            experts.kernel_gating_EDF,
            jnp.transpose(env.t2j_copy(gate_proj), permute_dims)
            if permute_dims is not None else env.t2j_copy(gate_proj),
            param_name=(
                f"language_model.layers.{layer_idx}.mlp.experts.gate_proj"),
        )
        assign_and_shard_param(
            experts.kernel_up_proj_EDF,
            jnp.transpose(env.t2j_copy(up_proj), permute_dims)
            if permute_dims is not None else env.t2j_copy(up_proj),
            param_name=(
                f"language_model.layers.{layer_idx}.mlp.experts.up_proj"),
        )

    def _finalize_loaded_expert_modules(self, loaded_expert_modules) -> None:
        for experts in loaded_expert_modules.values():
            quant_method = getattr(experts, "quant_method", None)
            if quant_method is not None:
                processed = quant_method.process_weights_after_loading(experts)
                if processed is not False:
                    self._clear_loaded_expert_buffers(
                        experts, {
                            "kernel_gating_EDF",
                            "kernel_up_proj_EDF",
                            "kernel_down_proj_EFD",
                        })

    def _load_moe_expert_weights(self) -> None:
        """Load fused or per-expert HF MoE tensors using real param metadata."""
        weights_iterator = model_weights_generator(
            model_name_or_path=self.vllm_config.model_config.model,
            download_dir=self.vllm_config.load_config.download_dir,
            framework="pt",
            filter_regex=r".*\.mlp\.experts(?:\.\d+\.)?.*",
        )
        loaded_expert_modules = {}
        for hf_key, hf_weight in weights_iterator:
            self._load_moe_expert_weight(hf_key, hf_weight,
                                         loaded_expert_modules)
        self._finalize_loaded_expert_modules(loaded_expert_modules)

    def load_weights(self, rng_key: jax.Array) -> None:
        self.rng = nnx.Rngs(rng_key)
        pp_missing_layers = []
        for path, module in nnx.iter_graph(self):
            if isinstance(module, PPMissingLayer):
                pp_missing_layers.append(".".join(str(segment)
                                                  for segment in path))
        mappings = {
            "model.language_model.embed_tokens": "language_model.embed_tokens.weight",
            "model.language_model.layers.*.input_layernorm": "language_model.layers.*.input_layernorm.weight",
            "model.language_model.layers.*.post_attention_layernorm": "language_model.layers.*.post_attention_layernorm.weight",
            "model.language_model.layers.*.self_attn.q_proj": "language_model.layers.*.self_attn.q_proj.weight",
            "model.language_model.layers.*.self_attn.k_proj": "language_model.layers.*.self_attn.k_proj.weight",
            "model.language_model.layers.*.self_attn.v_proj": "language_model.layers.*.self_attn.v_proj.weight",
            "model.language_model.layers.*.self_attn.o_proj": "language_model.layers.*.self_attn.o_proj.weight",
            "model.language_model.layers.*.self_attn.q_norm": "language_model.layers.*.self_attn.q_norm.weight",
            "model.language_model.layers.*.self_attn.k_norm": "language_model.layers.*.self_attn.k_norm.weight",
            "model.language_model.norm": "language_model.norm.weight",
            "model.language_model.layers.*.mlp.gate": "language_model.layers.*.mlp.gate.weight",
            "model.language_model.layers.*.mlp.gate_proj": "language_model.layers.*.mlp.gate_proj.weight",
            "model.language_model.layers.*.mlp.up_proj": "language_model.layers.*.mlp.up_proj.weight",
            "model.language_model.layers.*.mlp.down_proj": "language_model.layers.*.mlp.down_proj.weight",
            "model.visual.patch_embed.proj": "visual.patch_embed.proj.kernel",
            "model.visual.patch_embed.proj.bias": "visual.patch_embed.proj.bias",
            "model.visual.pos_embed": "visual.pos_embed.embedding",
            "model.visual.blocks.*.attn.qkv": "visual.blocks.*.attn.qkv_proj.kernel",
            "model.visual.blocks.*.attn.qkv.bias": "visual.blocks.*.attn.qkv_proj.bias",
            "model.visual.blocks.*.attn.proj": "visual.blocks.*.attn.proj.kernel",
            "model.visual.blocks.*.attn.proj.bias": "visual.blocks.*.attn.proj.bias",
            "model.visual.blocks.*.mlp.linear_fc1": "visual.blocks.*.mlp.fc1.kernel",
            "model.visual.blocks.*.mlp.linear_fc1.bias": "visual.blocks.*.mlp.fc1.bias",
            "model.visual.blocks.*.mlp.linear_fc2": "visual.blocks.*.mlp.fc2.kernel",
            "model.visual.blocks.*.mlp.linear_fc2.bias": "visual.blocks.*.mlp.fc2.bias",
            "model.visual.blocks.*.norm1": "visual.blocks.*.norm1.scale",
            "model.visual.blocks.*.norm1.bias": "visual.blocks.*.norm1.bias",
            "model.visual.blocks.*.norm2": "visual.blocks.*.norm2.scale",
            "model.visual.blocks.*.norm2.bias": "visual.blocks.*.norm2.bias",
            "model.visual.merger.norm": "visual.merger.norm.scale",
            "model.visual.merger.norm.bias": "visual.merger.norm.bias",
            "model.visual.merger.linear_fc1": "visual.merger.linear_fc1.kernel",
            "model.visual.merger.linear_fc1.bias": "visual.merger.linear_fc1.bias",
            "model.visual.merger.linear_fc2": "visual.merger.linear_fc2.kernel",
            "model.visual.merger.linear_fc2.bias": "visual.merger.linear_fc2.bias",
        }

        hf_config = self.vllm_config.model_config.hf_config
        if not hf_config.tie_word_embeddings:
            mappings["lm_head"] = "lm_head.weight"

        vision_config = hf_config.vision_config
        deepstack_indexes = getattr(vision_config, "deepstack_visual_indexes",
                                    [8, 16, 24])
        for i in range(len(deepstack_indexes)):
            mappings[f"model.visual.deepstack_merger_list.{i}.norm"] = \
                f"visual.deepstack_merger_list.{i}.norm.scale"
            mappings[f"model.visual.deepstack_merger_list.{i}.norm.bias"] = \
                f"visual.deepstack_merger_list.{i}.norm.bias"
            mappings[f"model.visual.deepstack_merger_list.{i}.linear_fc1"] = \
                f"visual.deepstack_merger_list.{i}.linear_fc1.kernel"
            mappings[f"model.visual.deepstack_merger_list.{i}.linear_fc1.bias"] = \
                f"visual.deepstack_merger_list.{i}.linear_fc1.bias"
            mappings[f"model.visual.deepstack_merger_list.{i}.linear_fc2"] = \
                f"visual.deepstack_merger_list.{i}.linear_fc2.kernel"
            mappings[f"model.visual.deepstack_merger_list.{i}.linear_fc2.bias"] = \
                f"visual.deepstack_merger_list.{i}.linear_fc2.bias"

        adapted_model_config = _ModelConfigAdapter(self.vllm_config.model_config)
        metadata_map = get_default_maps(adapted_model_config, self.mesh,
                                        mappings)
        metadata_map.transpose_map["mlp.gate"] = (1, 0)
        loaded_expert_modules = {}
        params = nnx.state(self)
        try:
            shardings = nnx.get_named_sharding(params, self.mesh)
        except TypeError:
            shardings = params
        env = torchax.default_env()
        weights_iterator = getattr(self.vllm_config.model_config,
                                   "runai_model_weights_iterator", None)
        if weights_iterator is None:
            weights_iterator = model_weights_generator(
                model_name_or_path=self.vllm_config.model_config.model,
                download_dir=self.vllm_config.load_config.download_dir,
                framework="pt",
            )

        for hf_key, hf_weight in weights_iterator:
            if self._skip_non_expert_weight(hf_key):
                continue
            if re.match(r".*\.mlp\.experts(?:\.\d+\.)?.*", hf_key):
                self._load_moe_expert_weight(hf_key, hf_weight,
                                             loaded_expert_modules)
                continue

            _load_and_shard_weight(
                self.vllm_config,
                params,
                shardings,
                metadata_map,
                self.mesh,
                hf_key,
                env.t2j_copy(hf_weight),
                keep_hf_weight_suffix_when_match=[],
                pp_missing_layers=pp_missing_layers,
            )

        self._finalize_loaded_expert_modules(loaded_expert_modules)
        check_all_loaded(params)
        nnx.update(self, params)
