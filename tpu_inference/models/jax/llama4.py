import re
import math
from typing import List, Optional, Tuple

import jax
import jax.numpy as jnp
from jax import lax
from flax import nnx
from flax.typing import PRNGKey
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P
from vllm.config import VllmConfig

from tpu_inference.layers.jax.attention.attention import AttentionMetadata
from tpu_inference.layers.jax.attention.llama4_attention import Llama4Attention, Llama4VisionAttention
from tpu_inference.layers.jax.constants import KVCacheType
from tpu_inference.layers.jax.layers import DenseFFW, Embedder, LMhead, RMSNorm
from tpu_inference.layers.jax.misc import shard_put
from tpu_inference.layers.jax.moe.moe import MoE, Router
from tpu_inference.layers.jax.transformer_block import \
    SharedExpertsTransformerBlock
from ...layers.jax.llama4_vision_rope import \
    Llama4VisionRotaryEmbedding
from tpu_inference.logger import init_logger
from tpu_inference.models.jax.utils.weight_utils import (get_param,
                                                       model_weights_generator,
                                                       print_param_info,
                                                       reshape_params,
                                                       transpose_params)



logger = init_logger(__name__)


class Llama4ForCausalLM(nnx.Module):

    def __init__(self,
                 vllm_config: VllmConfig,
                 rng: PRNGKey,
                 mesh: Mesh,
                 force_random_weights: bool = False):
        assert mesh is not None

        self.vllm_config = vllm_config
        model_config = vllm_config.model_config

        self.rng = nnx.Rngs(rng)
        self.mesh = mesh
        self.is_verbose = getattr(self.vllm_config.additional_config,
                                  "is_verbose", False)

        # Currently the runner will always set a mesh, so the custom default sharding (when
        #  no sharding is set in vllm config) doesn't take effect.
        # TODO(fhzhang): figure out whether we need to actually enable this.
        #    strategy_dict = {"tensor_parallelism": 4, "expert_parallelism": 2}

        # TODO(fhzhang): remove these once we confirm that the values we get from config are good.
        # self.hidden_size: int = 5120
        # vocab_size = 202048
        vocab_size = model_config.get_vocab_size()
        self.hidden_size = model_config.get_hidden_size()

        dtype: jnp.dtype = jnp.bfloat16

        num_layers: int = 48
        self.interleave_moe_layer_step = 1  # All layers are MoE for Scout
        intermediate_size_moe: int = 8192
        num_local_experts: int = 16
        hidden_act: str = "silu"
        self.no_rope_layer_interval = 4

        num_shared_experts = 1
        rms_norm_eps = 1e-5
        self.num_attention_heads = 40
        self.num_key_value_heads = 8
        self.head_dim = 128

        intermediate_size = 16384

        self.embedder = Embedder(vocab_size=vocab_size,
                                 hidden_size=self.hidden_size,
                                 dtype=dtype,
                                 vd_sharding=(('data', 'expert', 'model'),
                                              None),
                                 rngs=self.rng,
                                 random_init=force_random_weights)

        self.layers = []

        for i in range(num_layers):
            # For Llama4-Scout, all layers are MoE layers.
            # This can be adjusted for other variants.
            is_moe_layer = (i + 1) % \
                            self.interleave_moe_layer_step == 0
            use_attention_rope = (i + 1) % self.no_rope_layer_interval != 0

            router = Router(dtype=dtype,
                            hidden_size=self.hidden_size,
                            num_experts=num_local_experts,
                            num_experts_per_tok=1,
                            router_act="sigmoid",
                            rngs=self.rng,
                            activation_ffw_td=('data', None),
                            ed_sharding=(None, 'expert'),
                            random_init=force_random_weights)

            custom_module = MoE(dtype=dtype,
                                num_local_experts=num_local_experts,
                                apply_expert_weight_before_computation=True,
                                hidden_size=self.hidden_size,
                                intermediate_size_moe=intermediate_size_moe,
                                hidden_act=hidden_act,
                                router=router,
                                rngs=self.rng,
                                activation_ffw_td=('data', None),
                                activation_ffw_ted=('data', 'expert', None),
                                edf_sharding=('expert', None, 'model'),
                                efd_sharding=('expert', 'model', None),
                                random_init=force_random_weights
                                ) if is_moe_layer else DenseFFW(
                                    dtype=dtype,
                                    hidden_act=hidden_act,
                                    hidden_size=self.hidden_size,
                                    intermediate_size=intermediate_size,
                                    random_init=force_random_weights,
                                    rngs=self.rng,
                                    df_sharding=(None, 'model'),
                                    fd_sharding=('model', None),
                                    activation_ffw_td=('data', None))

            attn = Llama4Attention(
                hidden_size=self.hidden_size,
                dtype=dtype,
                # TODO (jacobplatin): we should refactor this to pass a dtype (or config) directly
                kv_cache_dtype=vllm_config.cache_config.cache_dtype,
                num_attention_heads=40,
                num_key_value_heads=8,
                head_dim=128,
                rope_theta=500000.0,
                # https://huggingface.co/meta-llama/Llama-4-Scout-17B-16E-Instruct/blob/main/config.json
                rope_scaling={
                    "scale_factor": 16.0,
                    "low_freq_factor": 1.0,
                    "high_freq_factor": 1.0,
                    "original_max_position_embeddings": 8192
                },
                rngs=self.rng,
                rope_input_ordering="interleaved",
                temperature_tuning=True,
                temperature_tuning_scale=0.1,
                temperature_tuning_floor_scale=8192,
                use_qk_norm=False,
                attention_chunk_size=None if use_attention_rope else 8192,
                mesh=self.mesh,
                random_init=force_random_weights,
                activation_attention_td=('data', 'model'),
                activation_q_td=('data', 'model'),
                query_tnh=P('data', 'model', None),
                keyvalue_skh=P('data', 'model', None),
                activation_attention_out_td=('data', 'model'),
                attn_o_tnh=P('data', 'model', None),
                dnh_sharding=(None, 'model', None),
                dkh_sharding=(None, 'model', None),
                nhd_sharding=('model', None, None),
            )

            shared_experts = DenseFFW(dtype=dtype,
                                      hidden_act=hidden_act,
                                      hidden_size=self.hidden_size,
                                      intermediate_size=num_shared_experts *
                                      intermediate_size_moe,
                                      rngs=self.rng,
                                      random_init=force_random_weights,
                                      df_sharding=(None, 'model'),
                                      fd_sharding=('model', None),
                                      activation_ffw_td=('data', None))

            pre_attention_norm = RMSNorm(
                dims=self.hidden_size,
                random_init=force_random_weights,
                epsilon=rms_norm_eps,
                rngs=self.rng,
                with_scale=True,
                dtype=dtype,
            )

            pre_mlp_norm = RMSNorm(
                dims=self.hidden_size,
                epsilon=rms_norm_eps,
                rngs=self.rng,
                with_scale=True,
                dtype=dtype,
                random_init=force_random_weights,
            )

            block = SharedExpertsTransformerBlock(
                custom_module=custom_module,
                attn=attn,
                pre_attention_norm=pre_attention_norm,
                pre_mlp_norm=pre_mlp_norm,
                shared_experts=shared_experts,
                use_attention_rope=use_attention_rope)
            self.layers.append(block)

        self.final_norm = RMSNorm(
            dims=self.hidden_size,
            epsilon=rms_norm_eps,
            rngs=self.rng,
            with_scale=True,
            dtype=dtype,
            random_init=force_random_weights,
        )

        self.lm_head = LMhead(vocab_size=vocab_size,
                              hidden_size=self.hidden_size,
                              dtype=dtype,
                              rngs=self.rng,
                              vd_sharding=(('data', 'expert', 'model'), None),
                              dv_sharding=(None, ('data', 'expert', 'model')),
                              random_init=force_random_weights)
        if self.is_verbose:
            self._print_model_architecture()

    def _print_model_architecture(self):
        num_display_layers = max(self.interleave_moe_layer_step,
                                 self.no_rope_layer_interval)

        logger.info("### Embedding ###")
        nnx.display(self.embedder)

        logger.info(f"\n### First {num_display_layers} Layers ###")
        # Loop through the slice and display each layer
        for i, layer in enumerate(self.layers[:num_display_layers]):
            logger.info(f"\n--- Layer {i} ---")
            nnx.display(layer)

        logger.info("\n### LM Head ###")
        nnx.display(self.lm_head)

    def load_weights(self, rng: jax.Array, cache_dir: Optional[str] = None):
        # NOTE: Since we are using nnx.eval_shape to init the model,
        # we have to pass dynamic arrays here for __call__'s usage.
        self.rng = nnx.Rngs(rng)

        weight_loader = Llama4WeightLoader(
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
        *args,
    ) -> Tuple[List[KVCacheType], jax.Array, List[jax.Array]]:
        is_prefill = False
        x_TD = self.embedder.encode(input_ids)
        for (i, block) in enumerate(self.layers):
            kv_cache = kv_caches[i]
            new_kv_cache, x_TD = block(x_TD, is_prefill, kv_cache,
                                       attention_metadata)
            jax.block_until_ready(x_TD)
            kv_caches[i] = new_kv_cache

        final_activation_TD = self.final_norm(x_TD)

        return kv_caches, final_activation_TD, []

    def compute_logits(self, hidden_states: jax.Array) -> jax.Array:
        logits_TV = jnp.dot(hidden_states,
                            self.lm_head.input_embedding_table_DV.value)
        return logits_TV


class Llama4WeightLoader:

    def __init__(self, vllm_config: VllmConfig, hidden_size, attn_heads,
                 num_key_value_heads, attn_head_dim):
        self.names_and_weights_generator = model_weights_generator(
            model_name_or_path=vllm_config.model_config.model,
            framework="flax",
            filter_regex="language_model",
            download_dir=vllm_config.load_config.download_dir)
        self.is_verbose = getattr(vllm_config.additional_config, "is_verbose",
                                  False)
        self._transpose_map = {
            "q_proj": (2, 0, 1),
            "k_proj": (2, 0, 1),
            "v_proj": (2, 0, 1),
            "router": (1, 0),
            "shared_expert.down_proj": (1, 0),
            "shared_expert.gate_proj": (1, 0),
            "shared_expert.up_proj": (1, 0),
            "o_proj": (1, 2, 0),
            "lm_head": (1, 0),
        }
        self._weight_shape_map = {
            "q_proj": (attn_heads, attn_head_dim, hidden_size),
            "k_proj": (num_key_value_heads, attn_head_dim, hidden_size),
            "v_proj": (num_key_value_heads, attn_head_dim, hidden_size),
            # o_proj is inverted: https://github.com/huggingface/transformers/blob/v4.53.2/src/transformers/models/llama4/modeling_llama4.py#L298
            "o_proj": (hidden_size, attn_heads, attn_head_dim),
        }

        # Set the mappings from loaded parameter keys to standardized names.
        self._loaded_to_standardized_keys = {
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
            "language_model.model.layers.*.feed_forward.router.weight":
            "layers.*.custom_module.router.kernel_DE",
            "language_model.model.layers.*.feed_forward.experts.down_proj":
            "layers.*.custom_module.kernel_down_proj_EFD",
            "language_model.model.layers.*.feed_forward.experts.gate_up_proj":
            "layers.*.custom_module.kernel_up_proj_EDF",
            "language_model.model.layers.*.feed_forward.shared_expert.down_proj.weight":
            "layers.*.shared_experts.kernel_down_proj_FD",
            "language_model.model.layers.*.feed_forward.shared_expert.gate_proj.weight":
            "layers.*.shared_experts.kernel_gating_DF",
            "language_model.model.layers.*.feed_forward.shared_expert.up_proj.weight":
            "layers.*.shared_experts.kernel_up_proj_DF",
        }

    def map_loaded_to_standardized_name(self, loaded_key: str) -> str:
        # Find the corresponding model key using the HF key
        if "layer" in loaded_key:
            layer_num = re.search(r"layers\.(\d+)", loaded_key).group(1)
            layer_key = re.sub(r"layers\.\d+", "layers.*", loaded_key)
            mapped_key = self._loaded_to_standardized_keys.get(
                layer_key, loaded_key)
            mapped_key = re.sub(r"layers\.\*", f"layers.{layer_num}",
                                mapped_key)
        else:
            mapped_key = self._loaded_to_standardized_keys.get(
                loaded_key, loaded_key)
        return mapped_key

    def _map_llama4_gate_up_proj(self, model_for_loading: nnx.Module,
                                 model_params: nnx.State, loaded_name: str,
                                 loaded_weight: jax.Array):
        """HF's gate_up_proj is a fused tensor of gate and up projections. It needs to be split."""
        # gate_proj is first & up_proj is second
        split_weights = jnp.split(loaded_weight, 2, axis=-1)

        for split_type in ["gate", "up"]:
            split_loaded_name = loaded_name.replace("gate_up_proj",
                                                    f"{split_type}_proj")
            if split_type == "gate":
                mapped_name = "layers.*.custom_module.kernel_gating_EDF"
                loaded_weight = split_weights[0]
            else:
                mapped_name = "layers.*.custom_module.kernel_up_proj_EDF"
                loaded_weight = split_weights[1]

            layer_num = re.search(r"layers\.(\d+)", split_loaded_name).group(1)
            mapped_name = re.sub(r"layers\.\*", f"layers.{layer_num}",
                                 mapped_name)
            mapped_model_weight = get_param(model_params, mapped_name)
            if not loaded_name.endswith(".bias"):
                loaded_weight = reshape_params(loaded_name, loaded_weight,
                                               self._weight_shape_map)
                loaded_weight = transpose_params(loaded_name, loaded_weight,
                                                 self._transpose_map)
            if mapped_model_weight.value.shape != loaded_weight.shape:
                raise ValueError(
                    f"Loaded shape for {split_loaded_name}: {loaded_weight.shape} "
                    f"does not match model shape for {mapped_name}: {mapped_model_weight.value.shape}!"
                )
            mapped_model_weight.value = shard_put(loaded_weight,
                                                  mapped_model_weight.sharding,
                                                  mesh=model_for_loading.mesh)
            logger.debug(
                f"{split_loaded_name}: {loaded_weight.shape}  -->  {mapped_name}: {mapped_model_weight.value.shape}"
            )
            if self.is_verbose:
                print_param_info(mapped_model_weight, mapped_name)

    def load_weights(self, model_for_loading: nnx.Module):
        model_params = nnx.state(model_for_loading)
        with jax.default_device(jax.devices("cpu")[0]):
            for loaded_name, loaded_weight in self.names_and_weights_generator:
                if "gate_up_proj" in loaded_name:
                    self._map_llama4_gate_up_proj(model_for_loading,
                                                  model_params, loaded_name,
                                                  loaded_weight)
                    continue
                mapped_name = self.map_loaded_to_standardized_name(loaded_name)
                model_weight = get_param(model_params, mapped_name)

                if not loaded_name.endswith(".bias"):
                    loaded_weight = reshape_params(loaded_name, loaded_weight,
                                                   self._weight_shape_map)
                    loaded_weight = transpose_params(loaded_name,
                                                     loaded_weight,
                                                     self._transpose_map)
                if model_weight.value.shape != loaded_weight.shape:
                    raise ValueError(
                        f"Loaded shape for {loaded_name}: {loaded_weight.shape} "
                        f"does not match model shape for {mapped_name}: {model_weight.value.shape}!"
                    )
                logger.debug(
                    f"Transformed parameter {loaded_name} to {mapped_name}: {loaded_weight.shape} --> {model_weight.value.shape}"
                )
                model_weight.value = shard_put(loaded_weight,
                                               model_weight.sharding,
                                               mesh=model_for_loading.mesh)
                if self.is_verbose:
                    print_param_info(model_weight, loaded_name)

        nnx.update(model_for_loading, model_params)


# --- Jax Vision Classes ---


# Define JAX GELU activation
def gelu_jax(x):
    return jax.nn.gelu(x)


# JAX equivalent of torch.nn.Unfold (Llama4UnfoldConvolution in MaxText)
# Conceptual: Step 1 of vision input processing: take pixels and project them to embedding dimension so model can interpret the image.
class JAXUnfoldConvolution(nnx.Module):

    def __init__(self,
                 config: dict,
                 rngs: nnx.Rngs,
                 dtype: jnp.dtype = jnp.bfloat16,
                 random_init: bool = False):
        cfg = config
        self.kernel_size = cfg.patch_size
        self.num_channels = cfg.num_channels
        patch_flat_dim = cfg.num_channels * cfg.patch_size * cfg.patch_size

        # Corresponds to HF: patch_embedding.linear, MaxText: vit_unfold_linear
        # Projects flattened patch into vision_hidden_size (1408)
        self.linear = nnx.Linear(
            patch_flat_dim,  #input dimension
            cfg.hidden_size,  #output dimension
            use_bias=False,
            dtype=dtype,
            kernel_init=nnx.initializers.lecun_normal(
            ),  #TODO: could probably just generate weight matrix of all zeros tbh
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
        patches = patches.reshape(
            batch_size, -1, num_channels * self.kernel_size * self.kernel_size)

        # Project patches to hidden dimension
        hidden_states = self.linear(
            patches)  # [B, num_patches, hidden_size_for_vit]

        return hidden_states


# --- Modules for Vision Encoder Layers (Simplified for nnx) ---


class JAXLlama4VisionMLP(nnx.Module):

    def __init__(self,
                 config: dict,
                 rngs: nnx.Rngs,
                 dtype: jnp.dtype = jnp.bfloat16,
                 random_init: bool = False):
        cfg = config
        # Corresponds to HF: vision_model.model.layers.*.mlp.fc1/fc2 (has bias=True)
        self.fc1 = nnx.Linear(
            cfg.hidden_size,
            cfg.intermediate_size,
            use_bias=True,
            dtype=dtype,
            rngs=rngs,  #random_init=random_init
        )
        self.fc2 = nnx.Linear(
            cfg.intermediate_size,
            cfg.hidden_size,
            use_bias=True,
            dtype=dtype,
            rngs=rngs,  #random_init=random_init
        )

    def __call__(self, hidden_states: jax.Array) -> jax.Array:
        hidden_states = self.fc1(hidden_states)
        hidden_states = gelu_jax(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states


class JAXLlama4VisionEncoderLayer(nnx.Module):

    def __init__(self,
                 config: dict,
                 rngs: nnx.Rngs,
                 mesh: Mesh,
                 dtype: jnp.dtype = jnp.bfloat16,
                 random_init: bool = False):
        cfg = config
        self.hidden_size = cfg.hidden_size

        self.self_attn = Llama4VisionAttention( 
            hidden_size=cfg.hidden_size,
            dtype=dtype,
            num_attention_heads=cfg.num_attention_heads,
            num_key_value_heads=cfg.num_attention_heads,
            # TODO (jacobplatin): we should refactor this to pass a dtype (or config) directly
            #kv_cache_dtype="auto",
            head_dim=cfg.hidden_size // cfg.num_attention_heads,
            rope_theta=10000.0,  # From vision_config
            rope_scaling=None,
            rngs=rngs,
            rope_input_ordering="interleaved",
            temperature_tuning=False,
            use_qk_norm=False, #TODO: Was False before
            mesh=mesh,  # Assuming no sharding. Not sure if this is correct
            #random_init=random_init,
            is_causal=False,  # Forces bidirectional mask for ViT Encoder
            temperature_tuning_floor_scale=
            0,  # placed these values to get past kwarg error (they are required)
            temperature_tuning_scale=0.0,
            activation_attention_td=None,
            activation_attention_out_td=None,
        )

        #flash attention: original attention (maybe just use this to ensure that your code works)
        #splash attention: investigating in Qwen-VL model. (might be faster implementation)

        self.mlp = JAXLlama4VisionMLP(cfg,
                                      rngs=rngs,
                                      dtype=dtype,
                                      random_init=random_init)
        
        # Note: We take a slice [0, :5] from the large kernel matrix
        # jax.debug.print("DEBUG: FC1 Kernel Slice (Layer 0): {}", 
        #                 self.mlp.fc1.kernel.value[0, :5])
        # jax.debug.print("DEBUG: FC1 Bias Slice (Layer 0): {}",
        #                 self.mlp.fc1.bias.value[:5])

        # HF/MaxText use nn.LayerNorm for vision
        self.input_layernorm = nnx.LayerNorm(
            cfg.hidden_size, epsilon=cfg.norm_eps, dtype=dtype,
            rngs=rngs)  #, random_init=random_init)
        self.post_attention_layernorm = nnx.LayerNorm(
            cfg.hidden_size, epsilon=cfg.norm_eps, dtype=dtype,
            rngs=rngs)  #, random_init=random_init)

    def __call__(self, hidden_state: jax.Array,
                 freqs_ci_stacked: jax.Array, **kwargs) -> jax.Array:

        jax.debug.print("TRACE 0 (Input): Hidden State Slice: {}", hidden_state[0, 0, :5])

        # Self Attention
        residual = hidden_state
        hidden_state = self.input_layernorm(hidden_state)

        jax.debug.print("TRACE 1: After Input LayerNorm Slice: {}", hidden_state[0, 0, :5])

        original_shape = hidden_state.shape
        B, S, D = original_shape # Capture B, S, D for later use

        hidden_state_2D = hidden_state.reshape(-1, original_shape[-1])

        vision_metadata = AttentionMetadata(input_positions=freqs_ci_stacked, )
        #TODO: I don't believe I took care of the causal mask issue yet
        # there is a "mask_value" variable in ref_ragged_paged_attention
        new_kv_cache, attention_output_2D = self.self_attn(
            x=hidden_state_2D,
            is_prefill=
            True,  # Encoder layer is always a prefill/non-autoregressive run
            kv_cache=None,  # Vision Encoder does not use KV cache
            attention_metadata=
            vision_metadata,  # Pass the object containing the frequencies
            use_attention_rope=True,  # Explicitly ensure RoPE is executed
            **kwargs
        )
        attention_output = attention_output_2D.reshape(original_shape)

        jax.debug.print("TRACE 2: Attention Output Slice: {}", attention_output[0, 0, :5])

        hidden_state = residual + attention_output

        jax.debug.print("TRACE 3: After Attn Residual Add Slice: {}", hidden_state[0, 0, :5])
        
        residual = hidden_state

        # MLP
        hidden_state = self.post_attention_layernorm(hidden_state)
        # 2. Print state AFTER MLP LayerNorm
        jax.debug.print("TRACE 4: After MLP LayerNorm Slice: {}", hidden_state[0, 0, :5])

        # ** FIX: Flatten hidden_state to 2D before passing to MLP **
        hidden_state_2D = hidden_state.reshape(B * S, D)

        hidden_state_2D = self.mlp(hidden_state_2D)
        
        # ** FIX: Restore original 3D shape **
        hidden_state = hidden_state_2D.reshape(B, S, D)

        jax.debug.print("TRACE 5: After MLP FFW Slice: {}", hidden_state[0, 0, :5])


        hidden_state = residual + hidden_state
        # 4. Print final output slice
        jax.debug.print("TRACE 6: FINAL Layer Output Slice: {}", hidden_state[0, 0, :5])
        # --- DEBUG MLP END ---
        
        return hidden_state


class JAXLlama4VisionEncoder(nnx.Module):

    def __init__(self,
                 config: dict,
                 rngs: nnx.Rngs,
                 mesh: Mesh,
                 dtype: jnp.dtype = jnp.bfloat16,
                 random_init: bool = False):
        cfg = config
        # Use cfg.num_hidden_layers for text, cfg.num_hidden_layers_for_vit for vision
        num_layers = cfg.num_hidden_layers if 'num_hidden_layers' in cfg else 34
        self.layers = [
            JAXLlama4VisionEncoderLayer(cfg,
                                        rngs=rngs,
                                        dtype=dtype,
                                        random_init=random_init,
                                        mesh=mesh) for _ in range(num_layers)
        ]

    def __call__(self, hidden_states: jax.Array,
                 freqs_ci_stacked: jax.Array, **kwargs) -> jax.Array:  # <--- MODIFIED
        # Loop over the layers, passing the frequencies to each one.
        for encoder_layer in self.layers:
            # MODIFIED CALL: Pass freqs to the layer
            hidden_states = encoder_layer(hidden_states, freqs_ci_stacked, **kwargs)
        return hidden_states


# --- Modules for Vision Adapter (PixelShuffleMLP) ---


# JAX implementation of pixel_shuffle (from MaxText reference)
def jax_pixel_shuffle(input_tensor: jax.Array,
                      shuffle_ratio: float) -> jax.Array:
    # input_tensor: [batch_size, num_patches, channels]
    batch_size, num_patches, channels = input_tensor.shape
    patch_size = int(math.sqrt(num_patches))

    # Reshape to [batch_size, patch_size, patch_size, channels]
    input_tensor = input_tensor.reshape(batch_size, patch_size, patch_size, -1)
    batch_size, height, width, channels = input_tensor.shape

    # Reshape 1: [batch_size, height, width * shuffle_ratio, channels / shuffle_ratio]
    reshaped_tensor = input_tensor.reshape(batch_size, height,
                                           int(width * shuffle_ratio),
                                           int(channels / shuffle_ratio))
    reshaped_tensor = reshaped_tensor.transpose(
        0, 2, 1, 3)  # permute(0, 2, 1, 3) in HF/MaxText

    # Reshape 2: [batch_size, height * shuffle_ratio, width * shuffle_ratio, channels / (shuffle_ratio^2)]
    reshaped_tensor = reshaped_tensor.reshape(
        batch_size, int(height * shuffle_ratio), int(width * shuffle_ratio),
        int(channels / (shuffle_ratio**2)))
    reshaped_tensor = reshaped_tensor.transpose(
        0, 2, 1, 3)  # permute(0, 2, 1, 3) in HF/MaxText

    # Reshape back to [batch_size, num_new_patches, channels_out]
    output_tensor = reshaped_tensor.reshape(batch_size, -1,
                                            reshaped_tensor.shape[-1])
    return output_tensor


class JAXLlama4VisionMLP2(nnx.Module):

    def __init__(self,
                 config: dict,
                 rngs: nnx.Rngs,
                 dtype: jnp.dtype = jnp.bfloat16,
                 random_init: bool = False):
        cfg = config

        # Dimensions based on MaxText/HF:
        # Input to fc1 is intermediate_size (5632) (output of pixel_shuffle)
        # fc1: 5632 -> projector_input_dim (4096)
        # fc2: projector_output_dim (4096) -> projector_output_dim (4096)

        self.fc1 = nnx.Linear(
            cfg.intermediate_size,
            cfg.projector_output_dim,
            use_bias=False,
            dtype=dtype,
            rngs=rngs,  #random_init=random_init
        )
        self.fc2 = nnx.Linear(
            cfg.projector_output_dim,
            cfg.projector_output_dim,
            use_bias=False,
            dtype=dtype,
            rngs=rngs,  #random_init=random_init
        )
        # Dropout is not strictly nnx module, but its rate is needed
        self.dropout_rate = cfg.projector_dropout
        self.dropout_rng = rngs.dropout

    def __call__(self,
                 hidden_states: jax.Array,
                 deterministic: bool = False) -> jax.Array:
        # First linear layer with GELU activation
        hidden_states = self.fc1(hidden_states)
        hidden_states = gelu_jax(hidden_states)

        # Apply dropout
        if self.dropout_rate > 0 and not deterministic:
            hidden_states = nnx.Dropout(self.dropout_rate,
                                        rngs=self.dropout_rng)(hidden_states)

        # Second linear layer with GELU activation
        hidden_states = self.fc2(hidden_states)
        hidden_states = gelu_jax(hidden_states)
        return hidden_states


class JAXLlama4VisionPixelShuffleMLP(nnx.Module):

    def __init__(self,
                 config: dict,
                 rngs: nnx.Rngs,
                 dtype: jnp.dtype = jnp.bfloat16,
                 random_init: bool = False):
        cfg = config
        self.pixel_shuffle_ratio = cfg.pixel_shuffle_ratio
        self.pixel_shuffle_mlp = JAXLlama4VisionMLP2(cfg,
                                                     rngs=rngs,
                                                     dtype=dtype,
                                                     random_init=random_init)

    def __call__(self,
                 encoded_patches: jax.Array,
                 deterministic: bool = False) -> jax.Array:
        # Apply pixel shuffle operation
        encoded_patches = jax_pixel_shuffle(encoded_patches,
                                            self.pixel_shuffle_ratio)

        # Apply MLP transformation
        result = self.pixel_shuffle_mlp(encoded_patches,
                                        deterministic=deterministic)
        return result


class JAXLlama4VisionModel(nnx.Module):

    def __init__(self,
                 config: dict,
                 rngs: nnx.Rngs,
                 mesh: Mesh,
                 dtype: jnp.dtype = jnp.bfloat16,
                 random_init: bool = False,
                 vision_rope: Optional[Llama4VisionRotaryEmbedding] = None):
        cfg = config
        self.scale = cfg.hidden_size**-0.5
        self.image_size = cfg.image_size
        self.patch_size = cfg.patch_size
        self.hidden_size = cfg.hidden_size
        self.norm_eps = cfg.norm_eps

        self.num_patches = (self.image_size //
                            self.patch_size)**2 + 1  # +1 for CLS token

        # 1. Patch Embedding (Llama4UnfoldConvolution)
        self.patch_embedding = JAXUnfoldConvolution(cfg,
                                                    rngs=rngs,
                                                    dtype=dtype,
                                                    random_init=random_init)

        # 2. Embeddings (HF: nn.Parameter)
        # 1. Get the raw JAX PRNGKey array from the RngStream's internal RngKey
        raw_key = rngs.params.key.value

        # Force raw_key to a safe integer type immediately before splitting
        if raw_key.dtype == jnp.uint32:  # <--- New Safety Check
            raw_key = raw_key.astype(jnp.int32)

        # 2. Split the raw key three times: for class_embedding, positional_embedding, and the remaining stream
        key_cls, key_pos, unused_key = jax.random.split(raw_key, 3)

        # 3. Initialize nnx.Param using the raw jax.random function and the split keys
        self.class_embedding = nnx.Param(
            self.scale *
            jax.random.normal(key_cls, (self.hidden_size, ), dtype=dtype))
        self.positional_embedding_vlm = nnx.Param(
            self.scale * jax.random.normal(
                key_pos, (self.num_patches, self.hidden_size), dtype=dtype))
        # Note: Rotary embedding initialization is complex, assumed to be constructed in the attention layer

        # 3. Layer Norms (HF: nn.LayerNorm)
        self.layernorm_pre = nnx.LayerNorm(
            self.hidden_size, epsilon=self.norm_eps, dtype=dtype,
            rngs=rngs)  #, random_init=random_init)
        self.layernorm_post = nnx.LayerNorm(
            self.hidden_size, epsilon=self.norm_eps, dtype=dtype,
            rngs=rngs)  #, random_init=random_init)

        # 4. Encoder (Llama4VisionEncoder)
        self.model = JAXLlama4VisionEncoder(cfg,
                                            rngs=rngs,
                                            mesh=mesh,
                                            dtype=dtype,
                                            random_init=random_init)

        # Store the RoPE module passed from the main constructor
        self.vision_rope = vision_rope

        # 5. Adapter (Llama4VisionPixelShuffleMLP)
        self.vision_adapter = JAXLlama4VisionPixelShuffleMLP(
            cfg, rngs=rngs, dtype=dtype, random_init=random_init)

    def __call__(self, pixel_values: jax.Array) -> jax.Array:
        # pixel_values: [batch_size, num_tiles, channels, tile_size, tile_size] in MaxText,
        # but HF and your model use [batch, channels, height, width] for simplicity

        # For simplicity, assume pixel_values is [B, C, H, W] for now
        # If your input is [B, T, C, H, W], reshape to [B*T, C, H, W] first.
        # MaxText example handles the reshape:
        input_shape = pixel_values.shape
        if len(input_shape) == 5:
            # Expected VLM format: [Batch, Time/Tile, Channel, Height, Width]
            b, t, c, h, w = input_shape
        elif len(input_shape) == 4:
            # Standard single image format: [Batch, Channel, Height, Width]
            # We insert the missing Time/Tile dimension (t=1)
            b, c, h, w = input_shape
            t = 1
            pixel_values = jnp.expand_dims(
                pixel_values, axis=1)  # Reshapes to [B, 1, C, H, W]
        else:
            raise ValueError(f"Unexpected pixel_values shape: {input_shape}")
        pixel_values = jnp.reshape(pixel_values, [b * t, c, h, w])

        # 1. Unfold convolution to extract patches
        hidden_states = self.patch_embedding(pixel_values)

        # 2. Add class embedding
        class_embedding_expanded = self.class_embedding.value[
            None, None, :].repeat(hidden_states.shape[0], axis=0)
        hidden_states = jnp.concatenate(
            [class_embedding_expanded, hidden_states], axis=1)

        # 3. Add positional embedding
        hidden_states += self.positional_embedding_vlm.value

        # 4. Transformation layers
        hidden_states = self.layernorm_pre(hidden_states)
        freqs_ci_stacked = self.vision_rope()
        hidden_states = self.model(hidden_states, freqs_ci_stacked)
        #hidden_states = self.model(hidden_states)
        hidden_states = self.layernorm_post(hidden_states)

        # 5. Remove CLS token (MaxText/HF: hidden_states[:, :-1, :])
        hidden_states = hidden_states[:,
                                      1:, :]  # HF: hidden_states[:, :-1, :] removes the last one.
        # If CLS is prepended, we remove the first one (index 0).
        # MaxText/HF prepend the CLS token, so we remove the first element [:, 1:, :]

        # 6. Vision Adapter (Pixel Shuffle MLP)
        hidden_states = self.vision_adapter(hidden_states)

        # 7. Reshape back to [B, T, N_patches, H_out]
        _, patch_num, patch_dim = hidden_states.shape
        hidden_states = jnp.reshape(hidden_states,
                                    [b, t, patch_num, patch_dim])

        # Final output for a single image will be [B, Num_patches, H_out=4096]
        # where Num_patches is the number of final projected patches
        return hidden_states.reshape(b, -1,
                                     patch_dim)  # [B, Total_patches, 4096]


class JAXLlama4MultiModalProjector(nnx.Module):

    def __init__(self,
                 config: dict,
                 rngs: nnx.Rngs,
                 dtype: jnp.dtype = jnp.bfloat16,
                 random_init: bool = False):
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
        return hidden_states  # Output shape: [batch, num_patches, text_hidden_size=5120]


# --- END: Jax Vision Classes ---
