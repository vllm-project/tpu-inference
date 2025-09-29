import re
from typing import List, Optional, Tuple, Any

import jax
import jax.numpy as jnp
from jax import lax
from flax import nnx
from flax.typing import PRNGKey
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
import math
from vllm.config import VllmConfig

from tpu_commons.logger import init_logger
from tpu_commons.models.jax.common.attention.attention import AttentionMetadata
from tpu_commons.models.jax.common.attention.llama4_attention import \
    Llama4Attention
from tpu_commons.models.jax.common.constants import KVCacheType
from tpu_commons.models.jax.common.layers import (DenseFFW, Embedder, LMhead,
                                                  RMSNorm)
from tpu_commons.models.jax.common.transformer_block import TransformerBlock
from tpu_commons.models.jax.layers.misc import shard_put
from tpu_commons.models.jax.utils.weight_utils import (get_param,
                                                       model_weights_generator,
                                                       print_param_info,
                                                       reshape_params,
                                                       transpose_params)
from tpu_commons.models.jax.layers.llama4_vision_rope import Llama4VisionRotaryEmbedding 

logger = init_logger(__name__)

# --- Jax Vision Classes ---

# Define JAX GELU activation
def gelu_jax(x):
    return jax.nn.gelu(x)

# JAX equivalent of torch.nn.Unfold (Llama4UnfoldConvolution in MaxText)
# Conceptual: Step 1 of vision input processing: take pixels and project them to embedding dimension so model can interpret the image. 
class JAXUnfoldConvolution(nnx.Module):
    def __init__(self, config: dict, rngs: nnx.Rngs, dtype: jnp.dtype = jnp.bfloat16, random_init: bool = False):
        cfg = config
        self.kernel_size = cfg.patch_size
        self.num_channels = cfg.num_channels
        patch_flat_dim = cfg.num_channels * cfg.patch_size * cfg.patch_size
        
        # Corresponds to HF: patch_embedding.linear, MaxText: vit_unfold_linear
        # Projects flattened patch into vision_hidden_size (1408)
        self.linear = nnx.Linear(
            patch_flat_dim,         #input dimension
            cfg.hidden_size,     #output dimension
            use_bias=False, 
            dtype=dtype,
            kernel_init=nnx.initializers.lecun_normal(), #TODO: could probably just generate weight matrix of all zeros tbh
            rngs=rngs,
            #random_init=random_init,
        )

    def __call__(self, inputs: jax.Array) -> jax.Array:
        # inputs: [batch_size, channels, img, img]
        batch_size, num_channels, img, _ = inputs.shape

        # Use JAX's lax.conv_general_dilated_patches, equivalent to MaxText's approach
        # This extracts patches and flattens them to [B, Patch_Dim, Num_Patches]
        patches = lax.conv_general_dilated_patches(
            inputs,
            filter_shape=[self.kernel_size, self.kernel_size],
            window_strides=[self.kernel_size, self.kernel_size],
            padding="VALID",
            dimension_numbers=("NCHW", "HWIO", "NCHW"),
        )
        
        # Reshape and transpose to [batch_size, num_patches, num_channels * patch_size * patch_size]
        patches = patches.reshape(batch_size, -1, num_channels * self.kernel_size * self.kernel_size)
        
        # Project patches to hidden dimension
        hidden_states = self.linear(patches) # [B, num_patches, hidden_size_for_vit]

        return hidden_states


# --- Modules for Vision Encoder Layers (Simplified for nnx) ---

class JAXLlama4VisionMLP(nnx.Module):
    def __init__(self, config: dict, rngs: nnx.Rngs, dtype: jnp.dtype = jnp.bfloat16, random_init: bool = False):
        cfg = config
        # Corresponds to HF: vision_model.model.layers.*.mlp.fc1/fc2 (has bias=True)
        self.fc1 = nnx.Linear(
            cfg.hidden_size, cfg.intermediate_size, use_bias=True, dtype=dtype, rngs=rngs, #random_init=random_init
        )
        self.fc2 = nnx.Linear(
            cfg.intermediate_size, cfg.hidden_size, use_bias=True, dtype=dtype, rngs=rngs, #random_init=random_init
        )

    def __call__(self, hidden_states: jax.Array) -> jax.Array:
        hidden_states = self.fc1(hidden_states)
        hidden_states = gelu_jax(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states


class JAXLlama4VisionEncoderLayer(nnx.Module):
    def __init__(self, config: dict, rngs: nnx.Rngs, dtype: jnp.dtype = jnp.bfloat16, random_init: bool = False):
        cfg = config
        self.hidden_size = cfg.hidden_size

        self.self_attn = Llama4Attention( # Included is_causal flag with the approval of Jevin.
                hidden_size=cfg.hidden_size, 
                dtype=dtype, 
                num_attention_heads=cfg.num_attention_heads, 
                num_key_value_heads=cfg.num_attention_heads,
                head_dim=cfg.hidden_size // cfg.num_attention_heads,
                rope_theta=10000.0, # From vision_config
                rope_scaling=None,
                rngs=rngs,
                rope_input_ordering="interleaved",
                temperature_tuning=False,
                use_qk_norm=False,
                mesh=None, # Assuming no sharding. Not sure if this is correct
                random_init=random_init,
                
                is_causal=False, # Forces bidirectional mask for ViT Encoder

                temperature_tuning_floor_scale=0,        # placed these values to get past kwarg error (they are required)
                temperature_tuning_scale=0.0,            
                activation_attention_td=None,            
                activation_attention_out_td=None,        
            )
        
        #flash attention: original attention (maybe just use this to ensure that your code works)
        #splash attention: investigating in Qwen-VL model. (might be faster implementation)

        self.mlp = JAXLlama4VisionMLP(cfg, rngs=rngs, dtype=dtype, random_init=random_init)
        
        # HF/MaxText use nn.LayerNorm for vision
        self.input_layernorm = nnx.LayerNorm(cfg.hidden_size, epsilon=cfg.norm_eps, dtype=dtype, rngs=rngs)#, random_init=random_init)
        self.post_attention_layernorm = nnx.LayerNorm(cfg.hidden_size, epsilon=cfg.norm_eps, dtype=dtype, rngs=rngs)#, random_init=random_init)

    def __call__(self, hidden_state: jax.Array, freqs_ci_stacked: jax.Array) -> jax.Array:
        # Self Attention
        residual = hidden_state
        hidden_state = self.input_layernorm(hidden_state)

        vision_metadata = AttentionMetadata(
            input_positions=freqs_ci_stacked, 
        )
        #TODO: I don't believe I took care of the causal mask issue yet
        # there is a "mask_value" variable in ref_ragged_paged_attention
        attention_output, new_kv_cache = self.self_attn( 
            x=hidden_state, 
            is_prefill=True,        # Encoder layer is always a prefill/non-autoregressive run
            kv_cache=None,          # Vision Encoder does not use KV cache
            attention_metadata=vision_metadata, # Pass the object containing the frequencies
            use_attention_rope=True # Explicitly ensure RoPE is executed
        )
        hidden_state = residual + attention_output
        residual = hidden_state

        # MLP
        hidden_state = self.post_attention_layernorm(hidden_state)
        hidden_state = self.mlp(hidden_state)
        hidden_state = residual + hidden_state
        return hidden_state


class JAXLlama4VisionEncoder(nnx.Module):
    def __init__(self, config: dict, rngs: nnx.Rngs, dtype: jnp.dtype = jnp.bfloat16, random_init: bool = False):
        cfg = config
        # Use cfg.num_hidden_layers for text, cfg.num_hidden_layers_for_vit for vision
        num_layers = cfg.num_hidden_layers if 'num_hidden_layers' in cfg else 34 
        self.layers = [JAXLlama4VisionEncoderLayer(cfg, rngs=rngs, dtype=dtype, random_init=random_init) for _ in range(num_layers)]

    def __call__(self, hidden_states: jax.Array, freqs_ci_stacked: jax.Array) -> jax.Array: # <--- MODIFIED
        # Loop over the layers, passing the frequencies to each one.
        for encoder_layer in self.layers:
            # MODIFIED CALL: Pass freqs to the layer
            hidden_states = encoder_layer(hidden_states, freqs_ci_stacked)
        return hidden_states


# --- Modules for Vision Adapter (PixelShuffleMLP) ---

# JAX implementation of pixel_shuffle (from MaxText reference)
def jax_pixel_shuffle(input_tensor: jax.Array, shuffle_ratio: float) -> jax.Array:
    # input_tensor: [batch_size, num_patches, channels]
    batch_size, num_patches, channels = input_tensor.shape
    patch_size = int(math.sqrt(num_patches))

    # Reshape to [batch_size, patch_size, patch_size, channels]
    input_tensor = input_tensor.reshape(batch_size, patch_size, patch_size, -1)
    batch_size, height, width, channels = input_tensor.shape

    # Reshape 1: [batch_size, height, width * shuffle_ratio, channels / shuffle_ratio]
    reshaped_tensor = input_tensor.reshape(batch_size, height, int(width * shuffle_ratio), int(channels / shuffle_ratio))
    reshaped_tensor = reshaped_tensor.transpose(0, 2, 1, 3) # permute(0, 2, 1, 3) in HF/MaxText

    # Reshape 2: [batch_size, height * shuffle_ratio, width * shuffle_ratio, channels / (shuffle_ratio^2)]
    reshaped_tensor = reshaped_tensor.reshape(
        batch_size, int(height * shuffle_ratio), int(width * shuffle_ratio), int(channels / (shuffle_ratio**2))
    )
    reshaped_tensor = reshaped_tensor.transpose(0, 2, 1, 3) # permute(0, 2, 1, 3) in HF/MaxText

    # Reshape back to [batch_size, num_new_patches, channels_out]
    output_tensor = reshaped_tensor.reshape(batch_size, -1, reshaped_tensor.shape[-1])
    return output_tensor


class JAXLlama4VisionMLP2(nnx.Module):
    def __init__(self, config: dict, rngs: nnx.Rngs, dtype: jnp.dtype = jnp.bfloat16, random_init: bool = False):
        cfg = config
        
        # Dimensions based on MaxText/HF:
        # Input to fc1 is intermediate_size (5632) (output of pixel_shuffle)
        # fc1: 5632 -> projector_input_dim (4096)
        # fc2: projector_output_dim (4096) -> projector_output_dim (4096)
        
        self.fc1 = nnx.Linear(
            cfg.intermediate_size, cfg.projector_output_dim, use_bias=False, dtype=dtype, rngs=rngs, #random_init=random_init
        )
        self.fc2 = nnx.Linear(
            cfg.projector_output_dim, cfg.projector_output_dim, use_bias=False, dtype=dtype, rngs=rngs, #random_init=random_init
        )
        # Dropout is not strictly nnx module, but its rate is needed
        self.dropout_rate = cfg.projector_dropout
        self.dropout_rng = rngs.dropout

    def __call__(self, hidden_states: jax.Array, deterministic: bool = False) -> jax.Array:
        # First linear layer with GELU activation
        hidden_states = self.fc1(hidden_states)
        hidden_states = gelu_jax(hidden_states)

        # Apply dropout
        if self.dropout_rate > 0 and not deterministic:
             hidden_states = nnx.Dropout(self.dropout_rate, rngs=self.dropout_rng)(hidden_states)

        # Second linear layer with GELU activation
        hidden_states = self.fc2(hidden_states)
        hidden_states = gelu_jax(hidden_states)
        return hidden_states


class JAXLlama4VisionPixelShuffleMLP(nnx.Module):
    def __init__(self, config: dict, rngs: nnx.Rngs, dtype: jnp.dtype = jnp.bfloat16, random_init: bool = False):
        cfg = config
        self.pixel_shuffle_ratio = cfg.pixel_shuffle_ratio
        self.pixel_shuffle_mlp = JAXLlama4VisionMLP2(cfg, rngs=rngs, dtype=dtype, random_init=random_init)

    def __call__(self, encoded_patches: jax.Array, deterministic: bool = False) -> jax.Array:
        # Apply pixel shuffle operation
        encoded_patches = jax_pixel_shuffle(encoded_patches, self.pixel_shuffle_ratio)
        
        # Apply MLP transformation
        result = self.pixel_shuffle_mlp(encoded_patches, deterministic=deterministic)
        return result


class JAXLlama4VisionModel(nnx.Module):
    def __init__(self, config: dict, rngs: nnx.Rngs, mesh: Mesh, dtype: jnp.dtype = jnp.bfloat16, random_init: bool = False, vision_rope: Optional[Llama4VisionRotaryEmbedding] = None):
        cfg = config
        self.scale = cfg.hidden_size**-0.5
        self.image_size = cfg.image_size
        self.patch_size = cfg.patch_size
        self.hidden_size = cfg.hidden_size
        self.norm_eps = cfg.norm_eps

        self.num_patches = (self.image_size // self.patch_size) ** 2 + 1 # +1 for CLS token

        # 1. Patch Embedding (Llama4UnfoldConvolution)
        self.patch_embedding = JAXUnfoldConvolution(cfg, rngs=rngs, dtype=dtype, random_init=random_init)

        # 2. Embeddings (HF: nn.Parameter)
        # 1. Get the raw JAX PRNGKey array from the RngStream's internal RngKey
        raw_key = rngs.params.key.value

        # Force raw_key to a safe integer type immediately before splitting
        if raw_key.dtype == jnp.uint32: # <--- New Safety Check
             raw_key = raw_key.astype(jnp.int32)
        
        # 2. Split the raw key three times: for class_embedding, positional_embedding, and the remaining stream
        key_cls, key_pos, unused_key = jax.random.split(raw_key, 3) 

        # 3. Initialize nnx.Param using the raw jax.random function and the split keys
        self.class_embedding = nnx.Param(
            self.scale * jax.random.normal(key_cls, (self.hidden_size,), dtype=dtype)
        )
        self.positional_embedding_vlm = nnx.Param(
            self.scale * jax.random.normal(key_pos, (self.num_patches, self.hidden_size), dtype=dtype)
        )
        # Note: Rotary embedding initialization is complex, assumed to be constructed in the attention layer

        # 3. Layer Norms (HF: nn.LayerNorm)
        self.layernorm_pre = nnx.LayerNorm(self.hidden_size, epsilon=self.norm_eps, dtype=dtype, rngs=rngs) #, random_init=random_init)
        self.layernorm_post = nnx.LayerNorm(self.hidden_size, epsilon=self.norm_eps, dtype=dtype, rngs=rngs) #, random_init=random_init)

        # 4. Encoder (Llama4VisionEncoder)
        self.model = JAXLlama4VisionEncoder(cfg, rngs=rngs, dtype=dtype, random_init=random_init)

        # Store the RoPE module passed from the main constructor
        self.vision_rope = vision_rope 
        
        # 5. Adapter (Llama4VisionPixelShuffleMLP)
        self.vision_adapter = JAXLlama4VisionPixelShuffleMLP(cfg, rngs=rngs, dtype=dtype, random_init=random_init)

    def __call__(self, pixel_values: jax.Array) -> jax.Array:
        # pixel_values: [batch_size, num_tiles, channels, tile_size, tile_size] in MaxText, 
        # but HF and your model use [batch, channels, height, width] for simplicity
        
        # For simplicity, assume pixel_values is [B, C, H, W] for now
        # If your input is [B, T, C, H, W], reshape to [B*T, C, H, W] first.
        # MaxText example handles the reshape:
        b, t, c, h, w = pixel_values.shape # Assuming [B, 1, C, H, W] for single image per prompt
        pixel_values = jnp.reshape(pixel_values, [b * t, c, h, w])

        # 1. Unfold convolution to extract patches
        hidden_states = self.patch_embedding(pixel_values)

        # 2. Add class embedding
        class_embedding_expanded = self.class_embedding.value[None, None, :].repeat(hidden_states.shape[0], axis=0)
        hidden_states = jnp.concatenate([class_embedding_expanded, hidden_states], axis=1)

        # 3. Add positional embedding
        hidden_states += self.positional_embedding_vlm.value

        # 4. Transformation layers
        hidden_states = self.layernorm_pre(hidden_states)
        freqs_ci_stacked = self.vision_rope() 
        hidden_states = self.model(hidden_states, freqs_ci_stacked)
        #hidden_states = self.model(hidden_states)
        hidden_states = self.layernorm_post(hidden_states)
        
        # 5. Remove CLS token (MaxText/HF: hidden_states[:, :-1, :])
        hidden_states = hidden_states[:, 1:, :] # HF: hidden_states[:, :-1, :] removes the last one.
                                               # If CLS is prepended, we remove the first one (index 0).
                                               # MaxText/HF prepend the CLS token, so we remove the first element [:, 1:, :]

        # 6. Vision Adapter (Pixel Shuffle MLP)
        hidden_states = self.vision_adapter(hidden_states)

        # 7. Reshape back to [B, T, N_patches, H_out]
        _, patch_num, patch_dim = hidden_states.shape
        hidden_states = jnp.reshape(hidden_states, [b, t, patch_num, patch_dim])
        
        # Final output for a single image will be [B, Num_patches, H_out=4096] 
        # where Num_patches is the number of final projected patches
        return hidden_states.reshape(b, -1, patch_dim) # [B, Total_patches, 4096]


class JAXLlama4MultiModalProjector(nnx.Module):
    def __init__(self, config: dict, rngs: nnx.Rngs, dtype: jnp.dtype = jnp.bfloat16, random_init: bool = False):
        cfg = config
        # HF/MaxText Projector: Linear(vision_output_dim, text_hidden_size, bias=False)
        self.linear = nnx.Linear(
            cfg["vision_config"].vision_output_dim,
            cfg["text_config"].hidden_size,
            use_bias=False,
            dtype=dtype,
            kernel_init=nnx.initializers.lecun_normal(),
            rngs=rngs,
            #random_init=random_init,
        )

    def __call__(self, image_features: jax.Array) -> jax.Array:
        # image_features: [batch, num_patches, vision_output_dim=4096]
        hidden_states = self.linear(image_features)
        return hidden_states # Output shape: [batch, num_patches, text_hidden_size=5120]

# --- END: Jax Vision Classes ---


class LlamaGuard4ForCausalLM(nnx.Module):

    #supports_multimodal: bool = True 

    def __init__(self,
                 vllm_config: VllmConfig,
                 rng: PRNGKey,
                 mesh: Mesh,
                 force_random_weights: bool = False):
        assert mesh is not None

        if rng.dtype == jnp.uint32:
            rng = rng.astype(jnp.int32)

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
                'vision_config': self.vision_config, # The raw Llama4VisionConfig object
                'text_config': self.text_config,   # The raw Llama4TextConfig object (or hf_config)
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
            rngs=self.rng, #nnx.Rngs(rng),
            random_init=force_random_weights,
        )


        if is_multimodal:
            # 1. Vision Encoder (Llama4VisionModel)
            self.vision_model = JAXLlama4VisionModel(
                self.vision_config,
                rngs=self.rng, #nnx.Rngs(rng),
                mesh=self.mesh,
                dtype=dtype,
                random_init=force_random_weights,
            )

            self.vision_rope = Llama4VisionRotaryEmbedding(
                image_size=vision_config.image_size,
                patch_size=vision_config.patch_size,
                hidden_size=vision_config.hidden_size,
                num_attention_heads=vision_config.num_attention_heads,
                rope_theta=vision_config.rope_theta,
                rngs=self.rng, #nnx.Rngs(rng),
                dtype=dtype,
            )

            # 2. Multimodal Projector (Llama4MultiModalProjector)
            self.multi_modal_projector = JAXLlama4MultiModalProjector(
                self.projector_config_dict, # Pass full model_config for nested keys
                rngs=self.rng, #nnx.Rngs(rng),
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
                rngs=self.rng, #nnx.Rngs(rng),
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
                rngs=self.rng, #nnx.Rngs(rng),
                rope_input_ordering="interleaved",
                temperature_tuning=True,
                temperature_tuning_scale=0.1,
                temperature_tuning_floor_scale=8192,
                use_qk_norm=True,
                attention_chunk_size=None if use_attention_rope else 8192,
                mesh=self.mesh,
                random_init=force_random_weights,

                # Added ".spec" to the ends of these TODO: removed this, maybe revert
                activation_attention_td=NamedSharding(self.mesh,
                                                      P('data', 'model')),#.spec,
                activation_q_td=NamedSharding(self.mesh, P('data',
                                                           'model')),#.spec,
                query_tnh=NamedSharding(self.mesh, P('data', 'model',
                                                     None)),#.spec,
                keyvalue_skh=NamedSharding(self.mesh, P('data', 'model',
                                                        None)),#.spec,
                activation_attention_out_td=NamedSharding(
                    self.mesh, P('data', 'model')),#.spec,
                attn_o_tnh=NamedSharding(self.mesh, P('data', 'model',
                                                      None)),#.spec,
                dnh_sharding=NamedSharding(self.mesh, P(None, 'model',
                                                        None)),#.spec,
                dkh_sharding=NamedSharding(self.mesh, P(None, 'model',
                                                        None)),#.spec,
                nhd_sharding=NamedSharding(self.mesh, P('model', None,
                                                        None)),#.spec,
            )

            pre_attention_norm = RMSNorm(
                dims=self.hidden_size,
                #mesh=self.mesh,
                random_init=force_random_weights,
                epsilon=rms_norm_eps,
                rngs=self.rng, #nnx.Rngs(rng),
                activation_ffw_td=NamedSharding(self.mesh, P()),
                with_scale=True,
                dtype=dtype,
            )

            pre_mlp_norm = RMSNorm(
                dims=self.hidden_size,
                #mesh=self.mesh,
                activation_ffw_td=NamedSharding(self.mesh, P()),
                epsilon=rms_norm_eps,
                rngs=self.rng, #nnx.Rngs(rng),
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
            rngs=self.rng, #nnx.Rngs(rng),
            with_scale=True,
            dtype=dtype,
            random_init=force_random_weights,
        )

        self.lm_head = LMhead(
            vocab_size=vocab_size,
            hidden_size=self.hidden_size,
            dtype=dtype,
            rngs=self.rng, #nnx.Rngs(rng),
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
        weight_loader = LlamaGuard4WeightLoader(
            vllm_config=self.vllm_config,
            hidden_size=self.hidden_size,
            attn_heads=self.num_attention_heads,
            num_key_value_heads=self.num_key_value_heads,
            attn_head_dim=self.head_dim)
        weight_loader.load_weights(self)

    def __call__(  
        self,
        kv_caches: List[jax.Array],
        input_ids: jax.Array,
        attention_metadata: AttentionMetadata,
        pixel_values: Optional[jax.Array] = None, #TODO: integrate visual input
        *args,
    ) -> Tuple[List[KVCacheType], jax.Array]:
        is_prefill = False

        # # Debug print the input_ids to ensure they're being passed correctly
        # jax.debug.print("Input IDs: {}", input_ids)
        inputs_embeds_TD = self.embedder.encode(input_ids)

        if pixel_values is not None:
            # 1. Image Feature Extraction (get_image_features equivalent)
            # Output: [batch, num_patches, vision_output_dim=4096]
            image_features = self.vision_model(pixel_values) 
            
            # Flatten to [B*Num_patches, 4096] for projection (if needed, but 
            # JAXLlama4MultiModalProjector handles 3D input)
            jax.debug.print("Image features shape (from vision_model): {}", image_features.shape)

            # 2. Multimodal Projection
            # Output: [batch, num_patches, text_hidden_size=5120]
            projected_vision_features = self.multi_modal_projector(image_features)
            jax.debug.print("Projected vision features shape: {}", projected_vision_features.shape)

            # 3. Fuse vision features with text embeddings (Masked Scatter equivalent)
            # HF: inputs_embeds = inputs_embeds.masked_scatter(special_image_mask, projected_vision_flat)
            
            # A. Create a boolean mask [B, S]
            image_token_indices_mask = (input_ids == self.image_token_index)
            
            # B. Prepare the flat image features for scattering
            projected_vision_features_flat = projected_vision_features.reshape(-1, self.hidden_size)
            
            # C. Check for length match (critical step from HF get_placeholder_mask)
            num_image_tokens_in_prompt = jnp.sum(image_token_indices_mask).item()
            num_projected_features = projected_vision_features_flat.shape[0]
            if num_image_tokens_in_prompt != num_projected_features:
                 raise ValueError(
                    f"Image features and image tokens do not match: tokens: {num_image_tokens_in_prompt}, "
                    f"features {num_projected_features}. Check input sequence and patching."
                )

            # D. JAX scatter operation to replace image token embeddings
            # 1. Get flat indices of image tokens
            flat_indices_mask = image_token_indices_mask.flatten()
            flat_seq_len = flat_indices_mask.shape[0]
            
            # Create a 2D index array for the scatter: [num_features, 2] where 2 is (batch_idx, seq_idx)
            # Get the flat sequence indices of the image tokens
            scatter_seq_indices = jnp.where(flat_indices_mask, size=num_projected_features, fill_value=0)[0] 
            
            # Convert flat index to (batch_idx, seq_idx)
            batch_indices = scatter_seq_indices // inputs_embeds_TD.shape[1]
            seq_indices = scatter_seq_indices % inputs_embeds_TD.shape[1]
            scatter_indices = jnp.stack([batch_indices, seq_indices], axis=1)

            # 2. Perform the scatter update (replaces the image token embeddings with features)
            fused_inputs_embeds_TD = jax.lax.scatter(
                inputs_embeds_TD,
                scatter_indices,
                projected_vision_features_flat,
                jax.lax.ScatterDimensionNumbers(
                    update_window_dims=(1,), # Update along the feature dimension
                    inserted_window_dims=(),
                    scatter_dims_to_operand_dims=(0, 1) # Scatter indices map to batch and sequence dim
                )
            )
            x_TD = fused_inputs_embeds_TD
            
        else:
            x_TD = inputs_embeds_TD
        
        # NOTE: You'll need to calculate and pass RoPE embeddings (`freq_cis`) to the blocks
        # if your `TransformerBlock` is configured like the one in MaxText.
        
        # (Assuming RoPE calc and passing logic is implemented in your local environment)

        for (i, block) in enumerate(self.layers):
            kv_cache = kv_caches[i]
            new_kv_cache, x_TD = block(x_TD, is_prefill, kv_cache,
                                       attention_metadata)
            jax.block_until_ready(x_TD)
            kv_caches[i] = new_kv_cache

        # jax.debug.print("Final layer before norm: {}", x_TD)
        final_activation_TD = self.final_norm(x_TD)

        # jax.debug.print("\nJAX Final Hidden States:\n{}", final_activation_TD)

        return kv_caches, final_activation_TD

    def compute_logits(self, hidden_states: jax.Array) -> jax.Array:
        logits_TV = jnp.dot(hidden_states,
                            self.lm_head.input_embedding_table_DV.value)

        # Check the max and min values of the logits to see if they're reasonable
        # jax.debug.print("Logits min/max: {}/{}", jnp.min(logits_TV),
        #                 jnp.max(logits_TV))

        # # Also check the logits for the `safe` and `unsafe` tokens
        # # You'll need to find the token IDs for these from your tokenizer
        # safe_token_id = 60411  # From your debug output
        # unsafe_token_id = 72110  # From your debug output
        # jax.debug.print("Logits for 'safe' token: {}",
        #                 logits_TV[0, safe_token_id])
        # jax.debug.print("Logits for 'unsafe' token: {}",
        #                 logits_TV[0, unsafe_token_id])

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
    def get_multimodal_embeddings(self, pixel_values: jax.Array, *args, **kwargs) -> jax.Array:
        """
        Computes the final projected embeddings for multimodal input. 
        """
        image_features = self.vision_model(pixel_values)

        projected_vision_features = self.multi_modal_projector(image_features)

        batch_size, num_patches, hidden_size = projected_vision_features.shape
        return projected_vision_features.reshape(batch_size * num_patches, hidden_size)

    def get_input_embeddings(self, input_ids: jax.Array) -> jax.Array:
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
            filter_regex="^(language_model|vision_model|multi_modal_projector)\..*", #We want both language model and vision model
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
            "vision_model.patch_embedding.linear.weight": "vision_model.patch_embedding.linear.kernel",
            "vision_model.class_embedding": "vision_model.class_embedding",
            "vision_model.positional_embedding_vlm": "vision_model.positional_embedding_vlm",
            
            "vision_model.layernorm_pre.weight": "vision_model.layernorm_pre.scale",
            "vision_model.layernorm_pre.bias": "vision_model.layernorm_pre.bias",
            "vision_model.layernorm_post.weight": "vision_model.layernorm_post.scale",
            "vision_model.layernorm_post.bias": "vision_model.layernorm_post.bias",
            
            # Vision Encoder Layer Weights 
            "vision_model.model.layers.*.input_layernorm.weight": "vision_model.model.layers.*.input_layernorm.scale",
            "vision_model.model.layers.*.input_layernorm.bias": "vision_model.model.layers.*.input_layernorm.bias",
            "vision_model.model.layers.*.post_attention_layernorm.weight": "vision_model.model.layers.*.post_attention_layernorm.scale",
            "vision_model.model.layers.*.post_attention_layernorm.bias": "vision_model.model.layers.*.post_attention_layernorm.bias",
            
            # ATTENTION KERNELS 
            "vision_model.model.layers.*.self_attn.q_proj.weight": "vision_model.model.layers.*.self_attn.kernel_q_proj_DNH",
            "vision_model.model.layers.*.self_attn.k_proj.weight": "vision_model.model.layers.*.self_attn.kernel_k_proj_DKH",
            "vision_model.model.layers.*.self_attn.v_proj.weight": "vision_model.model.layers.*.self_attn.kernel_v_proj_DKH",
            "vision_model.model.layers.*.self_attn.o_proj.weight": "vision_model.model.layers.*.self_attn.kernel_o_proj_NHD",
            "vision_model.model.layers.*.self_attn.q_proj.bias": "vision_model.model.layers.*.self_attn.q_proj.bias",
            "vision_model.model.layers.*.self_attn.k_proj.bias": "vision_model.model.layers.*.self_attn.k_proj.bias",
            "vision_model.model.layers.*.self_attn.v_proj.bias": "vision_model.model.layers.*.self_attn.v_proj.bias",
            "vision_model.model.layers.*.self_attn.o_proj.bias": "vision_model.model.layers.*.self_attn.o_proj.bias", 

            
            # VISION MLP WEIGHTS (FC1/FC2)
            "vision_model.model.layers.*.mlp.fc1.weight": "vision_model.model.layers.*.mlp.fc1.kernel",
            "vision_model.model.layers.*.mlp.fc1.bias": "vision_model.model.layers.*.mlp.fc1.bias",
            "vision_model.model.layers.*.mlp.fc2.weight": "vision_model.model.layers.*.mlp.fc2.kernel",
            "vision_model.model.layers.*.mlp.fc2.bias": "vision_model.model.layers.*.mlp.fc2.bias",

            # Vision Adapter (Pixel Shuffle MLP)
            "vision_model.vision_adapter.mlp.fc1.weight": "vision_model.vision_adapter.pixel_shuffle_mlp.fc1.kernel",
            "vision_model.vision_adapter.mlp.fc2.weight": "vision_model.vision_adapter.pixel_shuffle_mlp.fc2.kernel",
            
            # Multimodal Projector
            "multi_modal_projector.linear_1.weight": "multi_modal_projector.linear.kernel",
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
            mapped_key = self._loaded_to_standardized_keys.get(layer_key, loaded_key)
            
            # Substitute the wildcard with the actual layer number
            mapped_key = re.sub(r"layers\.\*", f"layers.{layer_num}", mapped_key)
        
        else:
            # 2. If it's a non-layer weight (lm_head, embed_tokens, etc.): map directly
            mapped_key = self._loaded_to_standardized_keys.get(loaded_key, loaded_key)
            
        return mapped_key

    def load_weights(self, model_for_loading: nnx.Module):        
        model_params = nnx.state(model_for_loading)

        # Determine if the model was initialized with vision components
        is_multimodal_loaded = model_for_loading.vision_model is not None

        with jax.default_device(jax.devices("cpu")[0]):
            for loaded_name, loaded_weight in self.names_and_weights_generator:
                is_vision_attn_bias = ( #apparently llama4 does not use biases
                    "vision_model" in loaded_name and 
                    "self_attn" in loaded_name and 
                    loaded_name.endswith(".bias")
                )
                
                if is_vision_attn_bias:
                    if self.is_verbose:
                        print(f"Skipping vision attention bias: {loaded_name}")
                    continue

                if "rotary_embedding" in loaded_name: # Skip Rotary Embedding weights, as they are calculated
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
                         print(f"Skipping weight '{loaded_name}' (mapped to '{mapped_name}'): not found in JAX model structure.")
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
                        if any(proj in loaded_name for proj in ["q_proj", "k_proj", "v_proj"]):
                            target_shape = (VISION_HIDDEN_DIM, VISION_HEADS, VISION_HEAD_DIM)
                            loaded_weight = jnp.reshape(loaded_weight, target_shape)
                        
                        # O (Output) Kernel (Target N, H, D_out)
                        elif "o_proj" in loaded_name:
                            target_shape = (VISION_HEADS, VISION_HEAD_DIM, VISION_HIDDEN_DIM)
                            loaded_weight = jnp.reshape(loaded_weight, target_shape)
                            
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
                    if any(k in loaded_name for k in ["q_proj", "k_proj", "v_proj", "gate_proj", "up_proj"]):
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
