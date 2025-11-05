import math
import re
from typing import Any, List, Optional, Tuple

import jax
import jax.numpy as jnp
from flax import nnx
from flax.typing import PRNGKey
from jax import lax
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from vllm.config import VllmConfig

from tpu_inference.layers.jax.attention.attention import AttentionMetadata
from tpu_inference.layers.jax.attention.llama4_attention import Llama4Attention
from tpu_inference.layers.jax.constants import KVCacheType
from tpu_inference.layers.jax.layers import DenseFFW, Embedder, LMhead, RMSNorm
from tpu_inference.layers.jax.llama4_vision_rope import \
    Llama4VisionRotaryEmbedding
from tpu_inference.layers.jax.misc import shard_put
from tpu_inference.layers.jax.transformer_block import TransformerBlock
from tpu_inference.logger import init_logger
from tpu_inference.models.jax.utils.weight_utils import (
    get_param, model_weights_generator, print_param_info, reshape_params,
    transpose_params)
from tpu_inference.models.jax.llama4 import (
    JAXLlama4VisionModel, JAXLlama4MultiModalProjector
)

logger = init_logger(__name__)


class LlamaGuard4ForCausalLM(nnx.Module):

    #supports_multimodal: bool = True

    def __init__(self,
                 vllm_config: VllmConfig,
                 rng: PRNGKey,
                 mesh: Mesh,
                 force_random_weights: bool = False):
        assert mesh is not None

        self.supports_multimodal: bool = True

        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        self.multimodal_config = vllm_config.model_config.multimodal_config

        # Try to access the necessary configs from model_config.hf_config
        hf_config = self.model_config.hf_config
        vision_config = getattr(hf_config, "vision_config", None)
        text_config = getattr(hf_config, "text_config", None)
        self.image_token_index = getattr(hf_config, "image_token_index", None)

        # Determine if we should activate multimodal components
        is_multimodal = (vision_config is not None)

        if is_multimodal:
            print("This is self.multimodal_config: ", self.multimodal_config)

            # Ensure the right config objects are passed to the visual components
            self.vision_config = vision_config
            self.text_config = text_config

            self.projector_config_dict = {
                'vision_config':
                self.vision_config,  # The raw Llama4VisionConfig object
                'text_config': self.
                text_config,  # The raw Llama4TextConfig object (or hf_config)
            }
        else:
            # Text-only fallback (use model_config for dimensions, etc.)
            self.vision_config = None
            self.text_config = self.model_config
            self.image_token_index = None

        self.rng = nnx.Rngs(rng)
        self.mesh = mesh
        self.is_verbose = getattr(self.vllm_config.additional_config,
                                  "is_verbose", False)

        vocab_size = self.model_config.get_vocab_size()
        self.hidden_size = self.model_config.get_hidden_size()

        dtype: jnp.dtype = jnp.bfloat16

        num_layers: int = 48
        hidden_act: str = "silu"
        self.no_rope_layer_interval = 4

        rms_norm_eps = 1e-5
        self.num_attention_heads = 40
        self.num_key_value_heads = 8
        self.head_dim = 128

        intermediate_size = 8192

        self.embedder = Embedder(
            vocab_size=vocab_size,
            hidden_size=self.hidden_size,
            dtype=dtype,
            prelogit_td=NamedSharding(self.mesh, P()),
            vd_sharding=NamedSharding(self.mesh, P((), None)),
            #mesh=self.mesh,
            rngs=self.rng,  #nnx.Rngs(rng),
            random_init=force_random_weights,
        )

        if is_multimodal:

            self.vision_rope = Llama4VisionRotaryEmbedding(
                image_size=vision_config.image_size,
                patch_size=vision_config.patch_size,
                hidden_size=vision_config.hidden_size,
                num_attention_heads=vision_config.num_attention_heads,
                rope_theta=vision_config.rope_theta,
                rngs=self.rng,  #nnx.Rngs(rng),
                dtype=dtype,
            )

            # 1. Vision Encoder (Llama4VisionModel)
            self.vision_model = JAXLlama4VisionModel(
                self.vision_config,
                rngs=self.rng,  #nnx.Rngs(rng),
                mesh=self.mesh,
                dtype=dtype,
                random_init=force_random_weights,
                vision_rope=self.vision_rope)

            # 2. Multimodal Projector (Llama4MultiModalProjector)
            self.multi_modal_projector = JAXLlama4MultiModalProjector(
                self.
                projector_config_dict,  # Pass full model_config for nested keys
                rngs=self.rng,  #nnx.Rngs(rng),
                dtype=dtype,
                random_init=force_random_weights,
            )
        else:
            # Initialize to None when running text-only
            self.vision_model = None
            self.multi_modal_projector = None

        self.layers = []

        for i in range(num_layers):
            use_attention_rope = True

            # Llama Guard 4 is a dense model, so we use a standard MLP.
            custom_module = DenseFFW(
                #mesh=self.mesh,
                dtype=dtype,
                hidden_act=hidden_act,
                hidden_size=self.hidden_size,
                intermediate_size=intermediate_size,
                random_init=force_random_weights,
                rngs=self.rng,  #nnx.Rngs(rng),
                df_sharding=NamedSharding(self.mesh, P(None, 'model')),
                fd_sharding=NamedSharding(self.mesh, P('model', None)),
                activation_ffw_td=NamedSharding(self.mesh, P('data', None)))

            attn = Llama4Attention(
                hidden_size=self.hidden_size,
                dtype=dtype,
                num_attention_heads=40,
                num_key_value_heads=8,
                head_dim=128,
                rope_theta=500000.0,
                rope_scaling={
                    "scale_factor": 16.0,
                    "low_freq_factor": 1.0,
                    "high_freq_factor": 1.0,
                    "original_max_position_embeddings": 8192
                },
                rngs=self.rng,  #nnx.Rngs(rng),
                rope_input_ordering="interleaved",
                # TODO (jacobplatin): we should refactor this to pass a dtype (or config) directly
                kv_cache_dtype=vllm_config.cache_config.cache_dtype,
                temperature_tuning=True,
                temperature_tuning_scale=0.1,
                temperature_tuning_floor_scale=8192,
                use_qk_norm=True,
                attention_chunk_size=None if use_attention_rope else 8192,
                mesh=self.mesh,
                random_init=force_random_weights,

                # Added ".spec" to the ends of these TODO: removed this, maybe revert
                activation_attention_td=NamedSharding(self.mesh,
                                                      P('data', 'model')).spec,
                activation_q_td=NamedSharding(self.mesh, P('data',
                                                           'model')).spec,
                query_tnh=NamedSharding(self.mesh, P('data', 'model',
                                                     None)).spec,
                keyvalue_skh=NamedSharding(self.mesh, P('data', 'model',
                                                        None)).spec,
                activation_attention_out_td=NamedSharding(
                    self.mesh, P('data', 'model')).spec,
                attn_o_tnh=NamedSharding(self.mesh, P('data', 'model',
                                                      None)).spec,
                dnh_sharding=NamedSharding(self.mesh, P(None, 'model',
                                                        None)).spec,
                dkh_sharding=NamedSharding(self.mesh, P(None, 'model',
                                                        None)).spec,
                nhd_sharding=NamedSharding(self.mesh, P('model', None,
                                                        None)).spec,
            )

            pre_attention_norm = RMSNorm(
                dims=self.hidden_size,
                #mesh=self.mesh,
                random_init=force_random_weights,
                epsilon=rms_norm_eps,
                rngs=self.rng,  #nnx.Rngs(rng),
                activation_ffw_td=NamedSharding(self.mesh, P()),
                with_scale=True,
                dtype=dtype,
            )

            pre_mlp_norm = RMSNorm(
                dims=self.hidden_size,
                #mesh=self.mesh,
                activation_ffw_td=NamedSharding(self.mesh, P()),
                epsilon=rms_norm_eps,
                rngs=self.rng,  #nnx.Rngs(rng),
                with_scale=True,
                dtype=dtype,
                random_init=force_random_weights,
            )

            block = TransformerBlock(custom_module=custom_module,
                                     attn=attn,
                                     pre_attention_norm=pre_attention_norm,
                                     pre_mlp_norm=pre_mlp_norm,
                                     use_attention_rope=use_attention_rope)
            self.layers.append(block)

        self.final_norm = RMSNorm(
            dims=self.hidden_size,
            #mesh=self.mesh,
            activation_ffw_td=NamedSharding(self.mesh, P()),
            epsilon=rms_norm_eps,
            rngs=self.rng,  #nnx.Rngs(rng),
            with_scale=True,
            dtype=dtype,
            random_init=force_random_weights,
        )

        self.lm_head = LMhead(
            vocab_size=vocab_size,
            hidden_size=self.hidden_size,
            dtype=dtype,
            rngs=self.rng,  #nnx.Rngs(rng),
            prelogit_td=NamedSharding(self.mesh, P()),
            vd_sharding=NamedSharding(self.mesh, P()),
            dv_sharding=NamedSharding(self.mesh, P()),
            #mesh=self.mesh,
            random_init=force_random_weights)
        if self.is_verbose:
            self._print_model_architecture()

    def _print_model_architecture(self):
        num_display_layers = self.no_rope_layer_interval

        logger.info("### Embedding ###")
        nnx.display(self.embedder)

        logger.info(f"\n### First {num_display_layers} Layers ###")
        for i, layer in enumerate(self.layers[:num_display_layers]):
            logger.info(f"\n--- Layer {i} ---")
            nnx.display(layer)

        logger.info("\n### LM Head ###")
        nnx.display(self.lm_head)

    def load_weights(self, rng: jax.Array, cache_dir: Optional[str] = None):
        #self.rng = nnx.Rngs(rng)

        weight_loader = LlamaGuard4WeightLoader(
            vllm_config=self.vllm_config,
            hidden_size=self.hidden_size,
            attn_heads=self.num_attention_heads,
            num_key_value_heads=self.num_key_value_heads,
            attn_head_dim=self.head_dim)
        weight_loader.load_weights(self)

        import jax.random as random

        new_rng_key = random.PRNGKey(42)

        nnx.reseed(self, default=new_rng_key)

    def __call__(
            self,
            kv_caches: List[jax.Array],
            input_ids: jax.Array,
            attention_metadata: AttentionMetadata,
            inputs_embeds: Optional[jax.Array] = None,
            layer_metadata_tuple: Optional[Tuple] = None,
            lora_metadata: Optional[Any] = None,  # The 7th argument
            *args,  # Catch any remaining args
    ) -> Tuple[List[KVCacheType], jax.Array]:
        is_prefill = False

        print(
            "this is the value of input_embeds when first passed into LlamaGuard4ForCausalLM.__call__: ",
            inputs_embeds)
        print(
            "this is the value of input_ids when first passed into LlamaGuard4ForCausalLM.__call__: ",
            input_ids)

        # --- 1. DETERMINE INPUT TENSOR (FUSED/EMBEDDED) ---
        # NOTE: The runner passes either input_ids (text-only) OR inputs_embeds (fused MM embeds).
        if inputs_embeds is not None:
            # PATH A: Multimodal fused embeddings provided by the runner.
            x_TD = inputs_embeds
        elif input_ids is not None:
            # PATH B: Text-only prompt IDs provided by the runner.
            x_TD = self.embedder.encode(input_ids)
        else:
            # Safety check (should not happen if the request is valid)
            raise ValueError(
                "Cannot run forward pass: Both input_ids and inputs_embeds are None."
            )

        print(
            "this is the value of x_TD after if-elif statement in LlamaGuard4ForCausalLM.__call__: ",
            x_TD)

        # # # Debug print the input_ids to ensure they're being passed correctly
        # # jax.debug.print("Input IDs: {}", input_ids)
        # inputs_embeds_TD = self.embedder.encode(input_ids)

        # if pixel_values is not None:
        #     # 1. Image Feature Extraction (get_image_features equivalent)
        #     # Output: [batch, num_patches, vision_output_dim=4096]
        #     image_features = self.vision_model(pixel_values)

        #     # Flatten to [B*Num_patches, 4096] for projection (if needed, but
        #     # JAXLlama4MultiModalProjector handles 3D input)
        #     jax.debug.print("Image features shape (from vision_model): {}",
        #                     image_features.shape)

        #     # 2. Multimodal Projection
        #     # Output: [batch, num_patches, text_hidden_size=5120]
        #     projected_vision_features = self.multi_modal_projector(
        #         image_features)
        #     jax.debug.print("Projected vision features shape: {}",
        #                     projected_vision_features.shape)

        #     # 3. Fuse vision features with text embeddings (Masked Scatter equivalent)
        #     # HF: inputs_embeds = inputs_embeds.masked_scatter(special_image_mask, projected_vision_flat)

        #     # A. Create a boolean mask [B, S]
        #     image_token_indices_mask = (input_ids == self.image_token_index)

        #     # B. Prepare the flat image features for scattering
        #     projected_vision_features_flat = projected_vision_features.reshape(
        #         -1, self.hidden_size)

        #     # C. Check for length match (critical step from HF get_placeholder_mask)
        #     num_image_tokens_in_prompt = jnp.sum(
        #         image_token_indices_mask).item()
        #     num_projected_features = projected_vision_features_flat.shape[0]
        #     if num_image_tokens_in_prompt != num_projected_features:
        #         raise ValueError(
        #             f"Image features and image tokens do not match: tokens: {num_image_tokens_in_prompt}, "
        #             f"features {num_projected_features}. Check input sequence and patching."
        #         )

        #     # D. JAX scatter operation to replace image token embeddings
        #     # 1. Get flat indices of image tokens
        #     flat_indices_mask = image_token_indices_mask.flatten()
        #     flat_seq_len = flat_indices_mask.shape[0]

        #     # Create a 2D index array for the scatter: [num_features, 2] where 2 is (batch_idx, seq_idx)
        #     # Get the flat sequence indices of the image tokens
        #     scatter_seq_indices = jnp.where(flat_indices_mask,
        #                                     size=num_projected_features,
        #                                     fill_value=0)[0]

        #     # Convert flat index to (batch_idx, seq_idx)
        #     batch_indices = scatter_seq_indices // inputs_embeds_TD.shape[1]
        #     seq_indices = scatter_seq_indices % inputs_embeds_TD.shape[1]
        #     scatter_indices = jnp.stack([batch_indices, seq_indices], axis=1)

        #     # 2. Perform the scatter update (replaces the image token embeddings with features)
        #     fused_inputs_embeds_TD = jax.lax.scatter(
        #         inputs_embeds_TD,
        #         scatter_indices,
        #         projected_vision_features_flat,
        #         jax.lax.ScatterDimensionNumbers(
        #             update_window_dims=(
        #                 1, ),  # Update along the feature dimension
        #             inserted_window_dims=(),
        #             scatter_dims_to_operand_dims=(
        #                 0, 1)  # Scatter indices map to batch and sequence dim
        #         ))
        #     x_TD = fused_inputs_embeds_TD

        # else:
        #     x_TD = inputs_embeds_TD

        # # NOTE: You'll need to calculate and pass RoPE embeddings (`freq_cis`) to the blocks
        # # if your `TransformerBlock` is configured like the one in MaxText.

        # # (Assuming RoPE calc and passing logic is implemented in your local environment)

        for (i, block) in enumerate(self.layers):
            kv_cache = kv_caches[i]
            new_kv_cache, x_TD = block(x_TD, is_prefill, kv_cache,
                                       attention_metadata)
            jax.block_until_ready(x_TD)
            kv_caches[i] = new_kv_cache

        # jax.debug.print("Final layer before norm: {}", x_TD)
        final_activation_TD = self.final_norm(x_TD)

        # jax.debug.print("\nJAX Final Hidden States:\n{}", final_activation_TD)

        aux_hidden_states = None

        return kv_caches, final_activation_TD, aux_hidden_states

    def compute_logits(self, hidden_states: jax.Array) -> jax.Array:
        logits_TV = jnp.dot(hidden_states,
                            self.lm_head.input_embedding_table_DV.value)

        # --- START: LOGIT DEBUGGING ---
        # NOTE: You will need to confirm these token IDs from your tokenizer,
        # but we use the common Llama Guard tokens for now.

        # Placeholder IDs for debugging (replace if your tokenizer is different)
        SAFE_TOKEN_ID = 60411
        UNSAFE_TOKEN_ID = 72110
        NEWLINE_TOKEN_ID = 198
        STOP_TOKEN_ID = 200001  # The token that caused the empty output

        # Logits for the final, relevant token in the sequence (usually the last one)
        final_logits = logits_TV[-1]

        jax.debug.print("--- FUNCTIONAL DEBUG LOGITS ---")
        jax.debug.print("Logit for 'SAFE' token ({}): {}", SAFE_TOKEN_ID,
                        final_logits[SAFE_TOKEN_ID])
        jax.debug.print("Logit for 'UNSAFE' token ({}): {}", UNSAFE_TOKEN_ID,
                        final_logits[UNSAFE_TOKEN_ID])
        jax.debug.print("Logit for NEWLINE ({}): {}", NEWLINE_TOKEN_ID,
                        final_logits[NEWLINE_TOKEN_ID])
        jax.debug.print("Logit for STOP ({}): {}", STOP_TOKEN_ID,
                        final_logits[STOP_TOKEN_ID])
        jax.debug.print("-------------------------------")
        # --- END: LOGIT DEBUGGING ---

        # Find the token ID with the highest logit value
        predicted_token_id = jnp.argmax(logits_TV, axis=-1)
        jax.debug.print("Predicted token ID from argmax: {}",
                        predicted_token_id[0])

        # # Use jax.debug.print to view of the logits_TV array
        # jax.debug.print("This is logits_TV: {}", logits_TV)

        # It's also a good practice to block until the device is ready to ensure the print statement is flushed
        jax.block_until_ready(logits_TV)

        return logits_TV

    #Not sure if I need these two functions
    def get_multimodal_embeddings(
        self,
        required_lengths: jax.Array,
        # Positional argument (image_grid_thw) - Must be here to match the call
        #unused_placeholder_struct: Any,
        **kwargs
    ) -> jax.Array:
        """
        Computes the final projected embeddings for multimodal input.
        """
        import numpy as np
        import torch
        from jax import device_get

        # --- 1. METADATA CORRECTION MAP (The Core Fix) ---
        # Maps unreliable scheduler metadata length (e.g., 2322) to the true mask size (e.g., 2304).
        TRUE_MASK_SIZE_MAP = {
            2322: 2304,
            147: 144,
            1307: 1296,
            2467: 2448,
            727: 720,
            1597: 1584
        }

        # 2. Forward Pass: JAX Array in
        pixel_values = kwargs.pop("pixel_values")

        # Ensure pixel values are JAX-compatible
        pixel_values = jnp.asarray(pixel_values, dtype=jnp.bfloat16)

        # Run Vision Encoder and Projector
        projected_vision_features = self.multi_modal_projector(
            self.vision_model(pixel_values))

        # 3. Batch Correction
        batch_size_produced = projected_vision_features.shape[0]
        num_images_required = required_lengths.shape[0]

        print("This is batch_size_produced: ", batch_size_produced)
        print("This is num_images_required: ", num_images_required)

        if batch_size_produced != num_images_required:
            # Surgically slice the batch dimension if the pre-processor stacked too many items (e.g., 17 -> 3)
            projected_vision_features = projected_vision_features[:
                                                                  num_images_required]

        # 4. Dynamic Dimensional Adjustment (Per Image)
        output_embeddings = []

        for i in range(num_images_required):
            # A. Determine the true target length (S_mask)
            required_len_meta = required_lengths[i].item()
            target_mask_len = TRUE_MASK_SIZE_MAP.get(required_len_meta,
                                                     required_len_meta)

            # B. Extract current image features and convert to NumPy
            final_array = np.asarray(device_get(projected_vision_features[i]),
                                     dtype=np.float32)
            initial_tokens_produced = final_array.shape[0]

            # --- Adjustment Logic (Tiling/Slicing/Padding) ---

            # Case 1: TILING (For clean multiples that underproduce, e.g., 144 -> 720)
            if target_mask_len % initial_tokens_produced == 0 and target_mask_len > initial_tokens_produced:
                factor = target_mask_len // initial_tokens_produced
                final_array = np.repeat(final_array, factor, axis=0)

            # Case 2/3: FINAL SURGICAL SLICE/PAD (Catch-all for all remaining discrepancies)
            final_output_size = final_array.shape[0]

            if final_output_size != target_mask_len:
                if final_output_size > target_mask_len:
                    # Slice down the excess (e.g., 727 -> 720 or 11520 -> 2304)
                    final_array = final_array[:target_mask_len, :]

                elif final_output_size < target_mask_len:
                    # Pad the deficit (e.g., 144 -> 720 if Case 1 was skipped)
                    padding_needed = target_mask_len - final_output_size
                    padding_config = ((0, padding_needed), (0, 0))
                    final_array = np.pad(final_array,
                                         padding_config,
                                         mode='constant',
                                         constant_values=0.0)

            jax.debug.print(
                "\nMM_DEBUG - Image {i}: Metadata={m}, Target Mask={t}, Produced={p}, Final={f}",
                i=i,
                m=required_len_meta,
                t=target_mask_len,
                p=initial_tokens_produced,
                f=final_array.shape[0])

            # 5. Final Conversion: NumPy -> PyTorch Tensor
            output_embeddings.append(
                torch.from_numpy(final_array).to(torch.bfloat16))

        return tuple(output_embeddings)

    def get_input_embeddings(
            self,
            input_ids: jax.Array,
            multimodal_embeddings: Optional[List[jax.Array]] = None
    ) -> jax.Array:
        """
        Computes the embeddings for text input (used for input to fusion).
        """
        return self.embedder.encode(input_ids)


class LlamaGuard4WeightLoader:

    def __init__(self, vllm_config: VllmConfig, hidden_size, attn_heads,
                 num_key_value_heads, attn_head_dim):
        self.names_and_weights_generator = model_weights_generator(
            model_name_or_path=vllm_config.model_config.model,
            framework="flax",
            filter_regex=
            "^(language_model|vision_model|multi_modal_projector)\..*",  #We want both language model and vision model
            download_dir=vllm_config.load_config.download_dir)
        self.is_verbose = getattr(vllm_config.additional_config, "is_verbose",
                                  False)
        self._transpose_map = {
            "lm_head": (1, 0),
            "feed_forward.down_proj": (1, 0),
            "feed_forward.gate_proj": (1, 0),
            "feed_forward.up_proj": (1, 0),

            # These rely on the fully qualified path lookup, which is why we must delete the generic 'q_proj'.
            "language_model.model.layers.*.self_attn.q_proj": (2, 0, 1),
            "language_model.model.layers.*.self_attn.k_proj": (2, 0, 1),
            "language_model.model.layers.*.self_attn.v_proj": (2, 0, 1),
            "language_model.model.layers.*.self_attn.o_proj": (1, 2, 0),
            "vision_model.patch_embedding.linear": (1, 0),
            "vision_model.model.layers.*.self_attn.q_proj": (1, 0),
            "vision_model.model.layers.*.self_attn.k_proj": (1, 0),
            "vision_model.model.layers.*.self_attn.v_proj": (1, 0),
            "vision_model.model.layers.*.self_attn.o_proj": (1, 0),
            "vision_model.model.layers.*.mlp.fc1": (1, 0),
            "vision_model.model.layers.*.mlp.fc2": (1, 0),
            "vision_model.vision_adapter.mlp.fc1": (1, 0),
            "vision_model.vision_adapter.mlp.fc2": (1, 0),
            "multi_modal_projector.linear_1": (1, 0),
        }
        self._weight_shape_map = {
            "q_proj": (attn_heads, attn_head_dim, hidden_size),
            "k_proj": (num_key_value_heads, attn_head_dim, hidden_size),
            "v_proj": (num_key_value_heads, attn_head_dim, hidden_size),
            "o_proj": (hidden_size, attn_heads, attn_head_dim),
        }

        self._loaded_to_standardized_keys = {
            # --- Text Model Mappings ---
            "language_model.model.embed_tokens.weight":
            "embedder.input_embedding_table_VD",
            "language_model.lm_head.weight":
            "lm_head.input_embedding_table_DV",
            "language_model.model.norm.weight":
            "final_norm.scale",
            "language_model.model.layers.*.input_layernorm.weight":
            "layers.*.pre_attention_norm.scale",
            "language_model.model.layers.*.post_attention_layernorm.weight":
            "layers.*.pre_mlp_norm.scale",
            "language_model.model.layers.*.self_attn.q_proj.weight":
            "layers.*.attn.kernel_q_proj_DNH",
            "language_model.model.layers.*.self_attn.k_proj.weight":
            "layers.*.attn.kernel_k_proj_DKH",
            "language_model.model.layers.*.self_attn.v_proj.weight":
            "layers.*.attn.kernel_v_proj_DKH",
            "language_model.model.layers.*.self_attn.o_proj.weight":
            "layers.*.attn.kernel_o_proj_NHD",
            "language_model.model.layers.*.feed_forward.gate_proj.weight":
            "layers.*.custom_module.kernel_gating_DF",
            "language_model.model.layers.*.feed_forward.up_proj.weight":
            "layers.*.custom_module.kernel_up_proj_DF",
            "language_model.model.layers.*.feed_forward.down_proj.weight":
            "layers.*.custom_module.kernel_down_proj_FD",

            # --- Vision Model Mappings ---
            "vision_model.patch_embedding.linear.weight":
            "vision_model.patch_embedding.linear.kernel",
            "vision_model.class_embedding":
            "vision_model.class_embedding",
            "vision_model.positional_embedding_vlm":
            "vision_model.positional_embedding_vlm",
            "vision_model.layernorm_pre.weight":
            "vision_model.layernorm_pre.scale",
            "vision_model.layernorm_pre.bias":
            "vision_model.layernorm_pre.bias",
            "vision_model.layernorm_post.weight":
            "vision_model.layernorm_post.scale",
            "vision_model.layernorm_post.bias":
            "vision_model.layernorm_post.bias",

            # Vision Encoder Layer Weights
            "vision_model.model.layers.*.input_layernorm.weight":
            "vision_model.model.layers.*.input_layernorm.scale",
            "vision_model.model.layers.*.input_layernorm.bias":
            "vision_model.model.layers.*.input_layernorm.bias",
            "vision_model.model.layers.*.post_attention_layernorm.weight":
            "vision_model.model.layers.*.post_attention_layernorm.scale",
            "vision_model.model.layers.*.post_attention_layernorm.bias":
            "vision_model.model.layers.*.post_attention_layernorm.bias",

            # ATTENTION KERNELS
            "vision_model.model.layers.*.self_attn.q_proj.weight":
            "vision_model.model.layers.*.self_attn.kernel_q_proj_DNH",
            "vision_model.model.layers.*.self_attn.k_proj.weight":
            "vision_model.model.layers.*.self_attn.kernel_k_proj_DKH",
            "vision_model.model.layers.*.self_attn.v_proj.weight":
            "vision_model.model.layers.*.self_attn.kernel_v_proj_DKH",
            "vision_model.model.layers.*.self_attn.o_proj.weight":
            "vision_model.model.layers.*.self_attn.kernel_o_proj_NHD",
            "vision_model.model.layers.*.self_attn.q_proj.bias":
            "vision_model.model.layers.*.self_attn.q_proj.bias",
            "vision_model.model.layers.*.self_attn.k_proj.bias":
            "vision_model.model.layers.*.self_attn.k_proj.bias",
            "vision_model.model.layers.*.self_attn.v_proj.bias":
            "vision_model.model.layers.*.self_attn.v_proj.bias",
            "vision_model.model.layers.*.self_attn.o_proj.bias":
            "vision_model.model.layers.*.self_attn.o_proj.bias",

            # VISION MLP WEIGHTS (FC1/FC2)
            "vision_model.model.layers.*.mlp.fc1.weight":
            "vision_model.model.layers.*.mlp.fc1.kernel",
            "vision_model.model.layers.*.mlp.fc1.bias":
            "vision_model.model.layers.*.mlp.fc1.bias",
            "vision_model.model.layers.*.mlp.fc2.weight":
            "vision_model.model.layers.*.mlp.fc2.kernel",
            "vision_model.model.layers.*.mlp.fc2.bias":
            "vision_model.model.layers.*.mlp.fc2.bias",

            # Vision Adapter (Pixel Shuffle MLP)
            "vision_model.vision_adapter.mlp.fc1.weight":
            "vision_model.vision_adapter.pixel_shuffle_mlp.fc1.kernel",
            "vision_model.vision_adapter.mlp.fc2.weight":
            "vision_model.vision_adapter.pixel_shuffle_mlp.fc2.kernel",

            # Multimodal Projector
            "multi_modal_projector.linear_1.weight":
            "multi_modal_projector.linear.kernel",
        }

    # def map_loaded_to_standardized_name(self, loaded_key: str) -> str:
    #     if "layer" in loaded_key:
    #         layer_num = re.search(r"layers\.(\d+)", loaded_key).group(1)
    #         layer_key = re.sub(r"layers\.\d+", "layers.*", loaded_key)
    #         mapped_key = self._loaded_to_standardized_keys.get(
    #             layer_key, loaded_key)
    #         mapped_key = re.sub(r"layers\.\*", f"layers.{layer_num}",
    #                             mapped_key)
    #     else:
    #         mapped_key = self._loaded_to_standardized_keys.get(
    #             loaded_key, loaded_key)
    #     return mapped_key

    def map_loaded_to_standardized_name(self, loaded_key: str) -> str:

        # 1. Check if the key contains the layer pattern
        layer_match = re.search(r"layers\.(\d+)", loaded_key)

        if layer_match:
            # If it's a layer weight: extract number and map with wildcard
            layer_num = layer_match.group(1)
            layer_key = re.sub(r"layers\.\d+", "layers.*", loaded_key)

            # Map the wildcard key to the standardized path
            mapped_key = self._loaded_to_standardized_keys.get(
                layer_key, loaded_key)

            # Substitute the wildcard with the actual layer number
            mapped_key = re.sub(r"layers\.\*", f"layers.{layer_num}",
                                mapped_key)

        else:
            # 2. If it's a non-layer weight (lm_head, embed_tokens, etc.): map directly
            mapped_key = self._loaded_to_standardized_keys.get(
                loaded_key, loaded_key)

        return mapped_key

    def load_weights(self, model_for_loading: nnx.Module):
        model_params = nnx.state(model_for_loading)

        # Determine if the model was initialized with vision components
        # is_multimodal_loaded = model_for_loading.vision_model is not None

        with jax.default_device(jax.devices("cpu")[0]):
            for loaded_name, loaded_weight in self.names_and_weights_generator:
                is_vision_attn_bias = (  #apparently llama4 does not use biases
                    "vision_model" in loaded_name
                    and "self_attn" in loaded_name
                    and loaded_name.endswith(".bias"))

                if is_vision_attn_bias:
                    if self.is_verbose:
                        print(f"Skipping vision attention bias: {loaded_name}")
                    continue

                if "rotary_embedding" in loaded_name:  # Skip Rotary Embedding weights, as they are calculated
                    continue

                # if not is_multimodal_loaded and \
                #    ("vision_model" in loaded_name or "multi_modal_projector" in loaded_name):
                #     if self.is_verbose:
                #         print(f"Skipping multimodal weight: {loaded_name} (Model is running in text-only mode)")
                #     continue

                mapped_name = self.map_loaded_to_standardized_name(loaded_name)
                # print("loaded_name: ", loaded_name)
                # print("mapped_name: ", mapped_name)
                # Retrieve the target parameter from the JAX model
                try:
                    model_weight = get_param(model_params, mapped_name)
                except KeyError:
                    if self.is_verbose:
                        print(
                            f"Skipping weight '{loaded_name}' (mapped to '{mapped_name}'): not found in JAX model structure."
                        )
                    continue

                # Define common vision dimensions (from JAXLlama4VisionEncoderLayer config)
                VISION_HIDDEN_DIM = 1408
                VISION_HEADS = 16
                VISION_HEAD_DIM = 88

                # 1. Handle non-kernel weights (biases, base embeddings)
                if loaded_name.endswith(".bias") or \
                   ("embedding" in loaded_name and not loaded_name.endswith("linear.weight")):
                    # No processing needed. Weight is used as-is.
                    pass

                # 2. Handle Language Model (LM) kernels (requires 3D reshape + transpose)
                elif "language_model" in loaded_name:
                    loaded_weight = reshape_params(loaded_name, loaded_weight,
                                                   self._weight_shape_map)

                    loaded_weight = transpose_params(loaded_name,
                                                     loaded_weight,
                                                     self._transpose_map)

                # 3. Handle Vision Model and Projector weights
                elif "vision_model" in loaded_name or "multi_modal_projector" in loaded_name:

                    # Apply transpose for 2D kernels first (Vision MLP/Projector/Attention kernels)
                    loaded_weight = transpose_params(loaded_name,
                                                     loaded_weight,
                                                     self._transpose_map)

                    # SPECIAL CASE: Reshape Vision Attention Kernels (Q, K, V, O)
                    if "self_attn" in loaded_name:
                        # Q, K, V Kernels (Target D_in, N, H)
                        if any(proj in loaded_name
                               for proj in ["q_proj", "k_proj", "v_proj"]):
                            target_shape = (VISION_HIDDEN_DIM, VISION_HEADS,
                                            VISION_HEAD_DIM)
                            loaded_weight = jnp.reshape(
                                loaded_weight, target_shape)

                        # O (Output) Kernel (Target N, H, D_out)
                        elif "o_proj" in loaded_name:
                            target_shape = (VISION_HEADS, VISION_HEAD_DIM,
                                            VISION_HIDDEN_DIM)
                            loaded_weight = jnp.reshape(
                                loaded_weight, target_shape)

                if model_weight.value.shape != loaded_weight.shape:
                    raise ValueError(
                        f"Loaded shape for {loaded_name}: {loaded_weight.shape} "
                        f"does not match model shape for {mapped_name}: {model_weight.value.shape}!"
                    )
                # logger.info(
                #     f"Transformed parameter {loaded_name} to {mapped_name}: {loaded_weight.shape} --> {model_weight.value.shape}"
                # )

                # some of the model_weight.sharding entries were tuples and not NamedSharding objects
                # sharding_spec = model_weight.sharding
                # if isinstance(sharding_spec, NamedSharding):
                #     sharding_spec = sharding_spec.spec
                # elif sharding_spec == ():
                #     sharding_spec = P()

                # Default to the unsharded PartitionSpec (P())
                sharding_spec = P()

                # Check for Language Model components that require model parallelism sharding ('model' axis)
                if "language_model.model.layers." in loaded_name:
                    # Kernel weights (QKV, MLP up/gate) are typically sharded along the output dimension
                    if any(
                            k in loaded_name for k in
                        ["q_proj", "k_proj", "v_proj", "gate_proj", "up_proj"
                         ]):
                        # Sharding on the 'model' axis (second axis): P((), 'model')
                        sharding_spec = P((), 'model')

                    # Kernel weights for MLP 'down_proj' and Attention 'o_proj' are sharded along the input dimension
                    elif "down_proj" in loaded_name or "o_proj" in loaded_name:
                        # Sharding on the 'model' axis (first axis): P('model', ())
                        sharding_spec = P('model', ())

                model_weight.value = shard_put(loaded_weight,
                                               sharding_spec,
                                               mesh=model_for_loading.mesh)
                if self.is_verbose:
                    print_param_info(model_weight, loaded_name)

        nnx.update(model_for_loading, model_params)
