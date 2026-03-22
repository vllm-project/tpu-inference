# Qwen3 VL MoE - Text model inherits from Qwen3 MoE with MRoPE attention
# and DeepStack support. Vision components are identical to Qwen3 VL dense.

import re
from itertools import islice
from typing import List, Optional, Tuple

import jax
import jax.numpy as jnp
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
    Qwen3VLImagePixelInputs,
    Qwen3VLImageInputs,
    Qwen3VLTextAttention,
    Qwen3VLVisionTransformer,
)
from tpu_inference.models.jax.utils.multi_modal_utils import (
    merge_multimodal_embeddings,
    normalize_mm_grid_thw,
    reshape_mm_tensor,
    split_mm_embeddings_by_grid,
)
from tpu_inference.models.jax.utils.weight_utils import (
    assign_and_shard_param,
    get_default_maps,
    jax_array_from_reshaped_torch,
    load_hf_weights,
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


class Qwen3VLMoeForConditionalGeneration(nnx.Module):
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
        self.vision_start_token_id = getattr(config, "vision_start_token_id", 151652)
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
    ) -> jax.Array:
        return self.get_input_embeddings(input_ids, multimodal_embeddings)

    def _parse_and_validate_image_input(
            self, image_grid_thw: Tuple[Tuple[int, int, int], ...],
            **kwargs: object) -> Optional[Qwen3VLImageInputs]:
        pixel_values = kwargs.pop("pixel_values", None)
        if pixel_values is None:
            pixel_values = kwargs.pop("pixel_values_videos", None)
        image_embeds = kwargs.pop("image_embeds", None)

        if pixel_values is None and image_embeds is None:
            return None

        if pixel_values is not None:
            pixel_values = reshape_mm_tensor(pixel_values, "pixel values")

            if not isinstance(pixel_values, jax.Array):
                raise ValueError("Incorrect type of pixel values. "
                                 f"Got type: {type(pixel_values)}")

            return Qwen3VLImagePixelInputs(
                type="pixel_values",
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw)

    def _parse_and_validate_multimodal_inputs(self,
                                              image_grid_thw: Tuple[Tuple[int,
                                                                          int,
                                                                          int],
                                                                    ...],
                                              **kwargs: object) -> dict:
        mm_input_by_modality = {}
        for input_key in kwargs:
            if input_key in ("pixel_values", "pixel_values_videos",
                             "image_embeds"
                             ) and "image" not in mm_input_by_modality:
                mm_input_by_modality[
                    "image"] = self._parse_and_validate_image_input(
                        image_grid_thw, **kwargs)
        return mm_input_by_modality

    def _process_image_input(
            self, image_input: Qwen3VLImageInputs
    ) -> tuple[tuple[jax.Array, ...],
               Optional[list[list[jax.Array]]]]:
        grid_thw = image_input["image_grid_thw"]
        if not grid_thw:
            return (), None

        if image_input["type"] == "image_embeds":
            image_embeds = image_input["image_embeds"].astype(self.visual.dtype)
            deepstack_embeds = None
        else:
            pixel_values = image_input["pixel_values"]
            image_embeds, deepstack_embeds = self.visual(pixel_values, grid_thw)
        return split_mm_embeddings_by_grid(image_embeds, grid_thw,
                                           self.spatial_merge_size,
                                           deepstack_embeds)

    def embed_multimodal(
        self,
        image_grid_thw: Tuple[Tuple[int, int, int], ...],
        **kwargs,
    ) -> dict:
        image_grid_thw = normalize_mm_grid_thw(image_grid_thw)
        if not image_grid_thw:
            image_grid_thw = normalize_mm_grid_thw(
                kwargs.get("video_grid_thw", None))

        mm_input_by_modality = self._parse_and_validate_multimodal_inputs(
            image_grid_thw, **kwargs)
        if not mm_input_by_modality:
            return {}
        if not image_grid_thw:
            return {}

        multimodal_embeddings: tuple[jax.Array, ...] = ()
        deepstack_outputs = None
        for modality in mm_input_by_modality:
            multimodal_input = mm_input_by_modality[modality]
            if modality == "image":
                image_splits, deepstack_by_item = self._process_image_input(
                    multimodal_input)
                multimodal_embeddings += image_splits
                if deepstack_by_item is not None:
                    if deepstack_outputs is None:
                        deepstack_outputs = []
                    deepstack_outputs.extend(deepstack_by_item)

        return {"embeds": multimodal_embeddings, "deepstack": deepstack_outputs}

    def __call__(
        self,
        kv_caches: List[jax.Array],
        input_ids: Optional[jax.Array],
        attention_metadata,
        inputs_embeds: Optional[jax.Array] = None,
        _input_positions=None,
        _layer_name_to_kv_cache=None,
        _lora_metadata=None,
        _intermediate_tensors=None,
        _is_first_rank: bool = True,
        _is_last_rank: bool = True,
        deepstack_embeds: Optional[List[jax.Array]] = None,
    ) -> Tuple[List[jax.Array], jax.Array, List[jax.Array]]:
        visual_pos_mask = None

        if deepstack_embeds is not None and input_ids is not None:
            visual_pos_mask = (input_ids == self.image_token_id) | (
                input_ids == self.video_token_id
            )

        kv_caches, hidden_states = self.language_model(
            kv_caches=kv_caches,
            input_ids=input_ids,
            attention_metadata=attention_metadata,
            inputs_embeds=inputs_embeds,
            visual_pos_mask=visual_pos_mask,
            deepstack_visual_embeds=deepstack_embeds,
        )

        return kv_caches, hidden_states, []

    def compute_logits(self, hidden_states: jax.Array) -> jax.Array:
        if hasattr(self, 'lm_head'):
            return self.lm_head(hidden_states)
        return self.language_model.embed_tokens.decode(hidden_states)

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
                t_val = int(t)
                expanded_video.extend([(1, int(h), int(w))] * t_val)
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

    def precompile_vision_encoder(
        self,
        run_compilation_fn,
    ) -> None:
        vc = self.config.vision_config
        patch_input_dim = (
            vc.in_channels * vc.temporal_patch_size * vc.patch_size * vc.patch_size
        )

        image_shapes = []
        if warmup_config := self.vllm_config.additional_config.get(
            "vision_warmup_config"
        ):
            image_shapes = warmup_config.get("image_shapes", [])

        factor = vc.patch_size * vc.spatial_merge_size
        for input_hw in image_shapes:
            if not isinstance(input_hw, list) or len(input_hw) != 2:
                logger.warning(f"Skipping invalid shape {input_hw}.")
                continue
            h_input, w_input = input_hw
            h_processed = round(h_input / factor) * factor
            w_processed = round(w_input / factor) * factor
            t, h, w = 1, h_processed // vc.patch_size, w_processed // vc.patch_size
            grid_thw = (t, h, w)
            num_patches = t * h * w

            dummy_pixel_values = jnp.ones(
                (num_patches, patch_input_dim),
                self.vllm_config.model_config.dtype,
            )
            dummy_grid_thw = (grid_thw,)

            run_compilation_fn(
                "vision_encoder",
                self.visual.encode_jit,
                dummy_pixel_values,
                dummy_grid_thw,
                image_shape=input_hw,
            )

    def _load_moe_expert_weights(self):
        """Load MoE expert weights from either fused or per-expert HF tensors."""
        from tpu_inference.models.jax.utils.weight_utils import (
            get_model_weights_files,
            model_weights_single_file_generator,
        )

        model_path = self.vllm_config.model_config.model
        download_dir = self.vllm_config.load_config.download_dir
        weights_files = get_model_weights_files(model_path, download_dir)

        for weights_file in weights_files:
            for hf_key, hf_weight in model_weights_single_file_generator(
                weights_file, framework="pt",
                filter_regex=r".*\.mlp\.experts(?:\.\d+\.)?.*",
            ):
                indexed_match = re.match(
                    r".*layers\.(\d+)\.mlp\.experts\.(\d+\.(?:gate_proj|up_proj|down_proj)(?:\.weight)?)$",
                    hf_key,
                )
                fused_match = re.match(
                    r".*layers\.(\d+)\.mlp\.experts\.(gate_up_proj|down_proj)(?:\.weight)?$",
                    hf_key,
                )

                if not indexed_match and not fused_match:
                    continue

                layer_idx = int((indexed_match or fused_match).group(1))
                layer = self.language_model.layers[layer_idx]
                if isinstance(layer, PPMissingLayer):
                    continue
                if not hasattr(layer.mlp, "experts"):
                    continue

                experts = layer.mlp.experts
                if indexed_match:
                    expert_suffix = indexed_match.group(2)
                    if not expert_suffix.endswith(".weight"):
                        expert_suffix += ".weight"
                    experts.load_weights([(expert_suffix, hf_weight)])
                    continue

                fused_name = fused_match.group(2)
                if fused_name == "down_proj":
                    down_proj_permute = None
                    target_shape = tuple(experts.kernel_down_proj_EFD.value.shape)
                    source_shape = tuple(hf_weight.shape)
                    if source_shape != target_shape:
                        swapped_shape = source_shape[:-2] + (
                            source_shape[-1], source_shape[-2])
                        if swapped_shape == target_shape:
                            down_proj_permute = (0, 2, 1)
                        else:
                            raise ValueError(
                                "Unsupported fused down_proj layout for "
                                f"language_model.layers.{layer_idx}.mlp.experts: "
                                f"source {source_shape} vs target {target_shape}"
                            )
                    jax_w = jax_array_from_reshaped_torch(
                        hf_weight, permute_dims=down_proj_permute)
                    assert jax_w.shape == target_shape, \
                        f"down_proj shape mismatch: {jax_w.shape} vs {target_shape}"
                    assign_and_shard_param(
                        experts.kernel_down_proj_EFD,
                        jax_w,
                        param_name=(
                            f"language_model.layers.{layer_idx}.mlp.experts.down_proj"),
                    )
                elif fused_name == "gate_up_proj":
                    E, D, F = experts.kernel_gating_EDF.value.shape
                    fused_shape = tuple(hf_weight.shape)
                    if fused_shape == (E, 2 * F, D):
                        # Standard layout: (E, 2*moe_intermediate, hidden)
                        chunk_dim = 1
                        gate_permute = (0, 2, 1)
                    elif fused_shape == (E, D, 2 * F):
                        # Transposed layout: (E, hidden, 2*moe_intermediate)
                        chunk_dim = 2
                        gate_permute = None
                    else:
                        raise ValueError(
                            "Unsupported fused gate_up_proj layout for "
                            f"language_model.layers.{layer_idx}.mlp.experts: "
                            f"source {fused_shape} vs expected EDF=({E}, {D}, {F})"
                        )
                    gate_proj, up_proj = hf_weight.chunk(2, dim=chunk_dim)
                    jax_gate = jax_array_from_reshaped_torch(
                        gate_proj, permute_dims=gate_permute)
                    assert tuple(jax_gate.shape) == (E, D, F), \
                        f"gate shape mismatch: {jax_gate.shape} vs ({E}, {D}, {F})"
                    assign_and_shard_param(
                        experts.kernel_gating_EDF,
                        jax_gate,
                        param_name=(
                            f"language_model.layers.{layer_idx}.mlp.experts.gate_proj"),
                    )
                    jax_up = jax_array_from_reshaped_torch(
                        up_proj, permute_dims=gate_permute)
                    assert tuple(jax_up.shape) == (E, D, F), \
                        f"up_proj shape mismatch: {jax_up.shape} vs ({E}, {D}, {F})"
                    assign_and_shard_param(
                        experts.kernel_up_proj_EDF,
                        jax_up,
                        param_name=(
                            f"language_model.layers.{layer_idx}.mlp.experts.up_proj"),
                    )

    def load_weights(self, rng_key: jax.Array) -> None:
        self.rng = nnx.Rngs(rng_key)

        # Step 1: Load MoE expert weights first (handles fused + per-expert HF layouts)
        self._load_moe_expert_weights()

        # Step 2: Load all other weights via load_hf_weights
        # (filter out expert tensors which are already loaded)
        mappings = {
            # Language model weights (framework layer names use .weight)
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
            # MoE router gate
            "model.language_model.layers.*.mlp.gate": "language_model.layers.*.mlp.gate.weight",
            # Dense MLP layers (for non-MoE layers if any)
            "model.language_model.layers.*.mlp.gate_proj": "language_model.layers.*.mlp.gate_proj.weight",
            "model.language_model.layers.*.mlp.up_proj": "language_model.layers.*.mlp.up_proj.weight",
            "model.language_model.layers.*.mlp.down_proj": "language_model.layers.*.mlp.down_proj.weight",
            # Vision encoder weights
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
        deepstack_indexes = getattr(vision_config, "deepstack_visual_indexes", [8, 16, 24])
        for i in range(len(deepstack_indexes)):
            mappings[f"model.visual.deepstack_merger_list.{i}.norm"] = f"visual.deepstack_merger_list.{i}.norm.scale"
            mappings[f"model.visual.deepstack_merger_list.{i}.norm.bias"] = f"visual.deepstack_merger_list.{i}.norm.bias"
            mappings[f"model.visual.deepstack_merger_list.{i}.linear_fc1"] = f"visual.deepstack_merger_list.{i}.linear_fc1.kernel"
            mappings[f"model.visual.deepstack_merger_list.{i}.linear_fc1.bias"] = f"visual.deepstack_merger_list.{i}.linear_fc1.bias"
            mappings[f"model.visual.deepstack_merger_list.{i}.linear_fc2"] = f"visual.deepstack_merger_list.{i}.linear_fc2.kernel"
            mappings[f"model.visual.deepstack_merger_list.{i}.linear_fc2.bias"] = f"visual.deepstack_merger_list.{i}.linear_fc2.bias"

        adapted_model_config = _ModelConfigAdapter(self.vllm_config.model_config)
        metadata_map = get_default_maps(
            adapted_model_config, self.mesh, mappings
        )

        # Add transpose for MoE router gate: HF stores (E, D), JaxLinear expects (D, E)
        metadata_map.transpose_map["mlp.gate"] = (1, 0)

        load_hf_weights(
            vllm_config=self.vllm_config,
            model=self,
            metadata_map=metadata_map,
            mesh=self.mesh,
            filter_regex=r"^(?!.*\.mlp\.experts(?:\.|$)).*$",
        )
