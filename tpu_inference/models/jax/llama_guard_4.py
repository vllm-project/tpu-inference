# Copyright 2025 Google LLC
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

# yapf: disable
# isort: skip_file

import re
from itertools import islice
from typing import List, Callable, Optional, Tuple

import jax
import jax.numpy as jnp
import torch
from flax import nnx
from flax.typing import PRNGKey
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P
from vllm.config import VllmConfig

from tpu_inference.distributed.jax_parallel_state import get_pp_group
from tpu_inference.layers.jax.attention.attention import AttentionMetadata
from tpu_inference.layers.jax.attention.llama4_attention import Llama4Attention
from tpu_inference.layers.jax.constants import KVCacheType
from tpu_inference.layers.jax.layers import DenseFFW, Embedder, LMhead, RMSNorm
from tpu_inference.layers.jax.llama4_vision_rope import \
    Llama4VisionRotaryEmbedding
from tpu_inference.layers.jax.misc import shard_put
from tpu_inference.layers.jax.pp_utils import (PPMissingLayer,
                                               get_start_end_layer)
from tpu_inference.layers.jax.transformer_block import TransformerBlock
from tpu_inference.logger import init_logger
from tpu_inference.models.jax.llama4 import (JAXLlama4MultiModalProjector,
                                             JAXLlama4VisionModel)
from tpu_inference.models.jax.utils.multi_modal_utils import \
    merge_multimodal_embeddings
from tpu_inference.models.jax.jax_intermediate_tensor import \
    JaxIntermediateTensors
from tpu_inference.models.jax.utils.weight_utils import (
    BaseWeightLoader, _is_pp_missing_layer, get_param, print_param_info,
    reshape_params, transpose_params)

import torchax

logger = init_logger(__name__)


class LlamaGuard4WeightLoader(BaseWeightLoader):

    def __init__(self, vllm_config: VllmConfig, hidden_size, attn_heads,
                 num_key_value_heads, attn_head_dim):
        super().__init__(
            vllm_config,
            framework="flax",
            filter_regex=
            "^(language_model|vision_model|multi_modal_projector)\..*",
        )
        # Set is_runai_streamer so we can use this in load_weights
        self.is_runai_streamer = getattr(
            getattr(vllm_config, 'load_config', None), 'load_format',
            None) == 'runai_streamer'
        self.is_verbose = getattr(vllm_config.additional_config, "is_verbose",
                                  False)

        # --- Language model transpose map ---
        self._language_transpose_map = {
            "self_attn.q_proj": (2, 0, 1),
            "self_attn.k_proj": (2, 0, 1),
            "self_attn.v_proj": (2, 0, 1),
            "self_attn.o_proj": (1, 2, 0),
            "lm_head": (1, 0),
            "feed_forward.down_proj": (1, 0),
            "feed_forward.gate_proj": (1, 0),
            "feed_forward.up_proj": (1, 0),
        }

        # Vision model transpose map
        self._vision_transpose_map = {
            "patch_embedding.linear":
            (2, 3, 1, 0),
            "self_attn.q_proj": (2, 0, 1),
            "self_attn.k_proj": (2, 0, 1),
            "self_attn.v_proj": (2, 0, 1),
            "self_attn.o_proj": (1, 2, 0),
            "mlp.fc1": (1, 0),
            "mlp.fc2": (1, 0),
            "vision_adapter.mlp.fc1": (1, 0),
            "vision_adapter.mlp.fc2": (1, 0),
        }

        # --- Projector transpose map ---
        self._projector_transpose_map = {
            "linear_1": (1, 0),
        }

        # --- Shape Maps ---
        self._weight_shape_map = {
            "q_proj": (attn_heads, attn_head_dim, hidden_size),
            "k_proj": (num_key_value_heads, attn_head_dim, hidden_size),
            "v_proj": (num_key_value_heads, attn_head_dim, hidden_size),
            "o_proj": (hidden_size, attn_heads, attn_head_dim),
        }

        vision_config = vllm_config.model_config.hf_config.vision_config
        vision_hidden_size = vision_config.hidden_size
        vision_attn_heads = vision_config.num_attention_heads
        vision_head_dim = vision_hidden_size // vision_attn_heads

        self._vision_weight_shape_map = {
            "patch_embedding.linear":
            (vision_hidden_size, 3, vision_config.patch_size,
             vision_config.patch_size),
            "q_proj": (vision_attn_heads, vision_head_dim, vision_hidden_size),
            "k_proj": (vision_attn_heads, vision_head_dim, vision_hidden_size),
            "v_proj": (vision_attn_heads, vision_head_dim, vision_hidden_size),
            "o_proj": (vision_hidden_size, vision_attn_heads, vision_head_dim),
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
            "vision_model.model.layers.*.self_attn.bias_q_proj_NH",
            "vision_model.model.layers.*.self_attn.k_proj.bias":
            "vision_model.model.layers.*.self_attn.bias_k_proj_KH",
            "vision_model.model.layers.*.self_attn.v_proj.bias":
            "vision_model.model.layers.*.self_attn.bias_v_proj_KH",
            "vision_model.model.layers.*.self_attn.o_proj.bias":
            "vision_model.model.layers.*.self_attn.bias_o_proj_D",

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

        self.pp_missing_layers = []
        for path, module in nnx.iter_graph(model_for_loading):
            if isinstance(module, PPMissingLayer):
                layer_name = ".".join([str(s) for s in path])
                self.pp_missing_layers.append(layer_name)

        with jax.default_device(jax.devices("cpu")[0]):
            for loaded_name, loaded_weight in self.get_weights_iterator():
                if self.is_runai_streamer:
                    if torchax is not None:
                        env = torchax.default_env()
                        loaded_weight = env.t2j_copy(loaded_weight)
                    else:
                        loaded_weight = jnp.array(loaded_weight.cpu().numpy())

                mapped_name = self.map_loaded_to_standardized_name(loaded_name)

                if _is_pp_missing_layer(mapped_name, self.pp_missing_layers):
                    continue

                try:
                    model_weight = get_param(model_params, mapped_name)
                except KeyError:
                    # Optional debug logging for skipped weights
                    if self.is_verbose:
                        print(f"Skipping {loaded_name}")
                    continue

                # --- Vision Model Bias Reshaping ---
                if "vision_model" in loaded_name and ".self_attn." in loaded_name and loaded_name.endswith(
                        ".bias"):
                    vision_config = model_for_loading.vision_config
                    vision_heads = vision_config.num_attention_heads
                    vision_head_dim = vision_config.hidden_size // vision_heads
                    if "q_proj.bias" in loaded_name or "k_proj.bias" in loaded_name or "v_proj.bias" in loaded_name:
                        loaded_weight = jnp.reshape(
                            loaded_weight, (vision_heads, vision_head_dim))

                # --- General Weight Reshaping and Transposition ---
                if not loaded_name.endswith(".bias"):
                    shape_map_to_use = None
                    transpose_map_to_use = None

                    if "language_model" in loaded_name:
                        shape_map_to_use = self._weight_shape_map
                        transpose_map_to_use = self._language_transpose_map
                    elif "vision_model" in loaded_name:
                        shape_map_to_use = self._vision_weight_shape_map
                        transpose_map_to_use = self._vision_transpose_map
                    elif "multi_modal_projector" in loaded_name:
                        transpose_map_to_use = self._projector_transpose_map

                    if shape_map_to_use:
                        loaded_weight = reshape_params(loaded_name,
                                                       loaded_weight,
                                                       shape_map_to_use)

                    if transpose_map_to_use:
                        loaded_weight = transpose_params(loaded_name,
                                                         loaded_weight,
                                                         transpose_map_to_use)

                    if "patch_embedding.linear" in mapped_name:
                        loaded_weight = loaded_weight.reshape(
                            -1, loaded_weight.shape[-1])

                if model_weight.value.shape != loaded_weight.shape:
                    raise ValueError(
                        f"Loaded shape for {loaded_name}: {loaded_weight.shape} "
                        f"does not match model shape for {mapped_name}: {model_weight.value.shape}!"
                    )

                if self.is_verbose:
                    print_param_info(model_weight, loaded_name)

                model_weight.value = shard_put(loaded_weight,
                                               model_weight.out_sharding,
                                               mesh=model_for_loading.mesh)

        nnx.update(model_for_loading, model_params)


class LlamaGuard4ForCausalLM(nnx.Module):

    def __init__(self,
                 vllm_config: VllmConfig,
                 rng: PRNGKey,
                 mesh: Mesh,
                 force_random_weights: bool = False):
        assert mesh is not None

        self.vllm_config = vllm_config
        self.vllm_config.model_config.dtype = torch.bfloat16
        self.model_config = vllm_config.model_config
        self.text_config = self.model_config.hf_config.text_config
        self.vision_config = self.model_config.hf_config.vision_config

        self.projector_config_dict = {
            'vision_config': self.vision_config,
            'text_config': self.text_config,
        }

        self.mesh = mesh
        self.is_verbose = getattr(self.vllm_config.additional_config,
                                   "is_verbose", False)

        self.use_qk_norm = getattr(self.text_config, "use_qk_norm", True)

        vocab_size = self.model_config.get_vocab_size()
        self.hidden_size = self.model_config.get_hidden_size()

        self.dtype: jnp.dtype = jnp.bfloat16

        self.num_layers: int = getattr(self.text_config, "num_layers", 48)
        hidden_act: str = getattr(self.text_config, "hidden_act", "silu")

        rms_norm_eps = getattr(self.text_config, "rms_norm_eps", 1e-5)
        self.num_attention_heads = getattr(self.text_config,
                                           "num_attention_heads", 40)
        self.num_key_value_heads = getattr(self.text_config,
                                           "num_key_value_heads", 8)
        self.head_dim = getattr(self.text_config, "head_dim", 128)

        intermediate_size = getattr(self.text_config, "intermediate_size",
                                    8192)

        self.rope_theta_text = getattr(self.text_config, "rope_theta",
                                       500000.0)
        self.rope_scaling = getattr(self.text_config, "rope_scaling")

        self.image_token_id = getattr(self.model_config.hf_config,
                                      "image_token_index")

        self.rng = nnx.Rngs(rng)

        self.is_first_rank = get_pp_group().is_first_rank
        self.is_last_rank = get_pp_group().is_last_rank

        if self.is_first_rank or (self.model_config.hf_config.tie_word_embeddings
                                  and self.is_last_rank):
            self.embedder = Embedder(
                vocab_size=vocab_size,
                hidden_size=self.hidden_size,
                dtype=self.dtype,
                vd_sharding=P(('data', 'model'), None),
                rngs=self.rng,
                random_init=force_random_weights,
            )
        else:
            self.embedder = PPMissingLayer()

        self.vision_rope = Llama4VisionRotaryEmbedding(
            image_size=self.vision_config.image_size,
            patch_size=self.vision_config.patch_size,
            hidden_size=self.vision_config.hidden_size,
            num_attention_heads=self.vision_config.num_attention_heads,
            rope_theta=self.vision_config.rope_theta,
            dtype=self.dtype,
        )

        self.vision_model = JAXLlama4VisionModel(
            self.vision_config,
            rngs=self.rng,
            mesh=self.mesh,
            dtype=self.dtype,
            random_init=force_random_weights,
            vision_rope=self.vision_rope)


        self.multi_modal_projector = JAXLlama4MultiModalProjector(
            self.projector_config_dict,
            rngs=self.rng,
            dtype=self.dtype,
            random_init=force_random_weights,
        )

        self.layers = nnx.List([])
        self.start_layer, self.end_layer = get_start_end_layer(
            self.num_layers,
            get_pp_group().rank_in_group,
            get_pp_group().world_size)

        for i in range(self.start_layer):
            self.layers.append(PPMissingLayer())

        for i in range(self.start_layer, self.end_layer):
            use_attention_rope = True

            custom_module = DenseFFW(dtype=self.dtype,
                                     hidden_act=hidden_act,
                                     hidden_size=self.hidden_size,
                                     intermediate_size=intermediate_size,
                                     random_init=force_random_weights,
                                     rngs=self.rng,
                                     df_sharding=P(None, 'model'),
                                     fd_sharding=P('model', None),
                                     activation_ffw_td=P('data', None),
                                     mesh=self.mesh)

            attn = Llama4Attention(
                hidden_size=self.hidden_size,
                dtype=self.dtype,
                num_attention_heads=self.num_attention_heads,
                num_key_value_heads=self.num_key_value_heads,
                head_dim=self.head_dim,
                rope_theta=self.rope_theta_text,
                rope_scaling={
                    "scale_factor":
                    self.rope_scaling["factor"],
                    "low_freq_factor":
                    self.rope_scaling["low_freq_factor"],
                    "high_freq_factor":
                    self.rope_scaling["high_freq_factor"],
                    "original_max_position_embeddings":
                    self.rope_scaling["original_max_position_embeddings"]
                },
                rngs=self.rng,
                rope_input_ordering="interleaved",
                # TODO (jacobplatin): we should refactor this to pass a dtype (or config) directly
                kv_cache_dtype=vllm_config.cache_config.cache_dtype,
                temperature_tuning=True,
                temperature_tuning_scale=0.1,
                temperature_tuning_floor_scale=8192,
                use_qk_norm=self.use_qk_norm,
                attention_chunk_size=None if use_attention_rope else 8192,
                mesh=self.mesh,
                random_init=force_random_weights,
                activation_attention_td=P('data', 'model'),
                activation_q_td=P('data', 'model'),
                query_tnh=P('data', 'model', None),
                keyvalue_skh=P('data', 'model', None),
                activation_attention_out_td=P('data', 'model'),
                attn_o_tnh=P('data', 'model', None),
                dnh_sharding=(None, 'model', None),
                dkh_sharding=(None, 'model', None),
                nhd_sharding=('model', None, None),
            )

            pre_attention_norm = RMSNorm(
                dims=self.hidden_size,
                random_init=force_random_weights,
                epsilon=rms_norm_eps,
                rngs=self.rng,
                activation_ffw_td=P('data', None),
                with_scale=True,
                dtype=self.dtype,
            )

            pre_mlp_norm = RMSNorm(
                dims=self.hidden_size,
                activation_ffw_td=P('data', None),
                epsilon=rms_norm_eps,
                rngs=self.rng,
                with_scale=True,
                dtype=self.dtype,
                random_init=force_random_weights,
            )

            block = TransformerBlock(custom_module=custom_module,
                                     attn=attn,
                                     pre_attention_norm=pre_attention_norm,
                                     pre_mlp_norm=pre_mlp_norm,
                                     use_attention_rope=use_attention_rope)
            self.layers.append(block)

        for i in range(self.end_layer, self.num_layers):
            self.layers.append(PPMissingLayer())

        if self.is_last_rank:
            self.final_norm = RMSNorm(
                dims=self.hidden_size,
                activation_ffw_td=P(),
                epsilon=rms_norm_eps,
                rngs=self.rng,
                with_scale=True,
                dtype=self.dtype,
                random_init=force_random_weights,
            )
            self.lm_head = LMhead(vocab_size=vocab_size,
                                  hidden_size=self.hidden_size,
                                  dtype=self.dtype,
                                  rngs=self.rng,
                                  dv_sharding=P(None, ('data', 'model')),
                                  random_init=force_random_weights)
        else:
            self.final_norm = PPMissingLayer()
            self.lm_head = PPMissingLayer()

        if self.is_verbose:
            self._print_model_architecture()

    def _print_model_architecture(self):
        logger.info("### Embedding ###")
        nnx.display(self.embedder)
        logger.info("\n### Layers ###")
        for i, layer in enumerate(self.layers):
            logger.info(f"\n--- Layer {i} ---")
            nnx.display(layer)
        logger.info("\n### LM Head ###")
        nnx.display(self.lm_head)

    # In LlamaGuard4ForCausalLM
    @nnx.jit
    def _compute_vision_features_jit(self, pixel_values: jax.Array) -> jax.Array:
        vision_outputs = self.vision_model(pixel_values)
        projected = self.multi_modal_projector(vision_outputs)
        return projected

    def precompile_vision_encoder(self, run_compilation_fn: Callable) -> None:
        warmup_config = self.vllm_config.additional_config.get(
            "vision_warmup_config")
        if not warmup_config:
            return

        image_shapes = warmup_config.get("image_shapes", [])

        for (h, w) in image_shapes:
            # Create dummy input: Batch=1, H, W, Channels=3
            # We must use the same layout (NHWC) as used in execution
            dummy_pixel_values = jnp.ones((1, h, w, 3), dtype=jnp.bfloat16)

            run_compilation_fn(
                "vision_encoder_projector",
                self._compute_vision_features_jit,
                dummy_pixel_values
            )

    def load_weights(self, rng: jax.Array, cache_dir: Optional[str] = None):
        self.rng = nnx.Rngs(rng)

        weight_loader = LlamaGuard4WeightLoader(
            vllm_config=self.vllm_config,
            hidden_size=self.hidden_size,
            attn_heads=self.num_attention_heads,
            num_key_value_heads=self.num_key_value_heads,
            attn_head_dim=self.head_dim)
        weight_loader.load_weights(self)

        logger.info("Materializing Llama 4 Vision RoPE frequencies...")
        self.vision_rope.__post_init__()

    def __call__(
        self,
        kv_caches: List[jax.Array],
        input_ids: jax.Array,
        attention_metadata: AttentionMetadata,
        inputs_embeds: Optional[jax.Array] = None,
        *args,
    ) -> Tuple[List[KVCacheType], jax.Array]:
        is_prefill = False

        if self.is_first_rank:
            if inputs_embeds is not None:
                x_TD = inputs_embeds
            elif input_ids is not None:
                x_TD = self.embedder.encode(input_ids)
            else:
                raise ValueError(
                    "Cannot run forward pass: Both input_ids and inputs_embeds are None."
                )
        else:
            # For pipeline parallelism (not active here, but good practice)
            # x_TD would come from intermediate_tensors
            pass

        for (i, block) in enumerate(
                islice(self.layers, self.start_layer, self.end_layer)):
            kv_cache = kv_caches[i]
            new_kv_cache, x_TD = block(x_TD, is_prefill, kv_cache,
                                       attention_metadata)
            jax.block_until_ready(x_TD)
            kv_caches[i] = new_kv_cache

        if not self.is_last_rank:
            return kv_caches, JaxIntermediateTensors({"hidden_states": x_TD}), []

        final_activation_TD = self.final_norm(x_TD)
        return kv_caches, final_activation_TD, []

    def compute_logits(self, hidden_states: jax.Array) -> jax.Array:
        logits_TV = jnp.dot(hidden_states,
                            self.lm_head.input_embedding_table_DV.value)
        return logits_TV

    def embed_input_ids(
            self,
            input_ids: jax.Array,
            multimodal_embeddings: Optional[jax.Array] = None) -> jax.Array:


        inputs_embeds = self.embedder.encode(input_ids)


        if multimodal_embeddings is not None and multimodal_embeddings.shape[
                0] > 0:
            inputs_embeds = merge_multimodal_embeddings(
                input_ids, inputs_embeds, multimodal_embeddings,
                [self.image_token_id])

        return inputs_embeds

    def embed_multimodal(self, image_grid_thw, **kwargs) -> List[jax.Array]:
        pixel_values = kwargs.pop("pixel_values")
        patches_per_image = kwargs.pop("patches_per_image", None)

        if pixel_values is None:
            return []

        # Ensure JAX handling
        pixel_values = jnp.asarray(pixel_values, dtype=jnp.bfloat16)

        # 1. Transpose Input from NCHW to NHWC
        # vLLM/HF loaders typically give NCHW (Channels First)
        # We ensure NHWC (Channels Last) for the JAX Vision Model
        if pixel_values.ndim == 4 and pixel_values.shape[1] == 3:
            pixel_values = jnp.transpose(pixel_values, (0, 2, 3, 1))

        # 2. Run Vision Encoder (on NHWC data)
        projected_vision_features = self.multi_modal_projector(
            self.vision_model(pixel_values))

        tokens_per_tile = projected_vision_features.shape[1]
        hidden_dim = projected_vision_features.shape[2]

        # 3. Flatten tokens
        all_tokens_flat = projected_vision_features.reshape(-1, hidden_dim)

        # 4. Split batch logic
        if hasattr(patches_per_image, 'tolist'):
            tile_counts = patches_per_image.tolist()
        else:
            tile_counts = list(patches_per_image)
        if isinstance(tile_counts, list) and isinstance(tile_counts[0], list):
            import itertools
            tile_counts = list(itertools.chain.from_iterable(tile_counts))
        split_sizes = [c * tokens_per_tile for c in tile_counts]

        if not split_sizes or sum(split_sizes) == 0:
            return []

        if sum(split_sizes) != all_tokens_flat.shape[0]:
            raise ValueError(
                "Sum of split sizes must match the total number of tokens.")

        split_indices = jnp.cumsum(jnp.array(split_sizes[:-1]))
        output_list = jnp.split(all_tokens_flat, split_indices)

        return list(output_list)