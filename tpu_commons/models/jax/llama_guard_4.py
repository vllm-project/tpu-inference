import re
from typing import List, Optional, Tuple

import jax
import jax.numpy as jnp
from flax import nnx
from flax.typing import PRNGKey
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
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

logger = init_logger(__name__)


class LlamaGuard4ForCausalLM(nnx.Module):

    def __init__(self,
                 vllm_config: VllmConfig,
                 rng: PRNGKey,
                 mesh: Mesh,
                 force_random_weights: bool = False):
        assert mesh is not None

        self.vllm_config = vllm_config
        model_config = vllm_config.model_config

        # self.rng = nnx.Rngs(rng)
        self.mesh = mesh
        self.is_verbose = getattr(self.vllm_config.additional_config,
                                  "is_verbose", False)

        vocab_size = model_config.get_vocab_size()
        self.hidden_size = model_config.get_hidden_size()

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
            vd_sharding=NamedSharding(self.mesh,
                                      P(('data', 'expert', 'model'), None)),
            #mesh=self.mesh,
            rngs=nnx.Rngs(rng),
            random_init=force_random_weights)

        self.layers = []

        for i in range(num_layers):
            use_attention_rope = False  #(i + 1) % self.no_rope_layer_interval != 0 #Llama 4 Guard does not use RoPe.

            # Llama Guard 4 is a dense model, so we use a standard MLP.
            custom_module = DenseFFW(
                #mesh=self.mesh,
                dtype=dtype,
                hidden_act=hidden_act,
                hidden_size=self.hidden_size,
                intermediate_size=intermediate_size,
                random_init=force_random_weights,
                rngs=nnx.Rngs(rng),
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
                rngs=nnx.Rngs(rng),
                rope_input_ordering="interleaved",
                temperature_tuning=True,
                temperature_tuning_scale=0.1,
                temperature_tuning_floor_scale=8192,
                use_qk_norm=True,
                attention_chunk_size=None if use_attention_rope else 8192,
                mesh=self.mesh,
                random_init=force_random_weights,

                # Added ".spec" to the ends of these
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
                rngs=nnx.Rngs(rng),
                activation_ffw_td=NamedSharding(self.mesh, P()),
                with_scale=True,
                dtype=dtype,
            )

            pre_mlp_norm = RMSNorm(
                dims=self.hidden_size,
                #mesh=self.mesh,
                activation_ffw_td=NamedSharding(self.mesh, P()),
                epsilon=rms_norm_eps,
                rngs=nnx.Rngs(rng),
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
            rngs=nnx.Rngs(rng),
            with_scale=True,
            dtype=dtype,
            random_init=force_random_weights,
        )

        self.lm_head = LMhead(
            vocab_size=vocab_size,
            hidden_size=self.hidden_size,
            dtype=dtype,
            rngs=nnx.Rngs(rng),
            prelogit_td=NamedSharding(self.mesh, P()),
            vd_sharding=NamedSharding(self.mesh,
                                      P(('data', 'expert', 'model'), None)),
            dv_sharding=NamedSharding(self.mesh,
                                      P(None, ('data', 'expert', 'model'))),
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
        *args,
    ) -> Tuple[List[KVCacheType], jax.Array]:
        is_prefill = False

        # Debug print the input_ids to ensure they're being passed correctly
        jax.debug.print("Input IDs: {}", input_ids)

        x_TD = self.embedder.encode(input_ids)

        # Add debug print to check the embeddings
        jax.debug.print("Input embedding slice: {}", x_TD[:1, :5])

        for (i, block) in enumerate(self.layers):
            kv_cache = kv_caches[i]
            new_kv_cache, x_TD = block(x_TD, is_prefill, kv_cache,
                                       attention_metadata)
            jax.block_until_ready(x_TD)
            kv_caches[i] = new_kv_cache

        final_activation_TD = self.final_norm(x_TD)

        return kv_caches, final_activation_TD

    def compute_logits(self, hidden_states: jax.Array) -> jax.Array:
        logits_TV = jnp.dot(hidden_states,
                            self.lm_head.input_embedding_table_DV.value)

        # Check the max and min values of the logits to see if they're reasonable
        jax.debug.print("Logits min/max: {}/{}", jnp.min(logits_TV),
                        jnp.max(logits_TV))

        # Also check the logits for the `safe` and `unsafe` tokens
        # You'll need to find the token IDs for these from your tokenizer
        safe_token_id = 60411  # From your debug output
        unsafe_token_id = 72110  # From your debug output
        jax.debug.print("Logits for 'safe' token: {}",
                        logits_TV[0, safe_token_id])
        jax.debug.print("Logits for 'unsafe' token: {}",
                        logits_TV[0, unsafe_token_id])

        # Find the token ID with the highest logit value
        predicted_token_id = jnp.argmax(logits_TV, axis=-1)
        jax.debug.print("Predicted token ID from argmax: {}",
                        predicted_token_id[0])

        # Use jax.debug.print to view a slice of the logits_TV array
        jax.debug.print("This is logits_TV: {}", logits_TV[0, :20])

        # It's also a good practice to block until the device is ready to ensure the print statement is flushed
        jax.block_until_ready(logits_TV)

        return logits_TV


class LlamaGuard4WeightLoader:

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
            "o_proj": (1, 2, 0),
            "lm_head": (1, 0),
            "feed_forward.down_proj": (1, 0),
            "feed_forward.gate_proj": (1, 0),
            "feed_forward.up_proj": (1, 0),
            "mlp.down_proj": (1, 0),
            "mlp.gate_proj": (1, 0),
            "mlp.up_proj": (1, 0),
        }
        self._weight_shape_map = {
            "q_proj": (attn_heads, attn_head_dim, hidden_size),
            "k_proj": (num_key_value_heads, attn_head_dim, hidden_size),
            "v_proj": (num_key_value_heads, attn_head_dim, hidden_size),
            "o_proj": (hidden_size, attn_heads, attn_head_dim),
        }

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
            "language_model.model.layers.*.feed_forward.gate_proj.weight":
            "layers.*.custom_module.kernel_gating_DF",
            "language_model.model.layers.*.feed_forward.up_proj.weight":
            "layers.*.custom_module.kernel_up_proj_DF",
            "language_model.model.layers.*.feed_forward.down_proj.weight":
            "layers.*.custom_module.kernel_down_proj_FD",
        }

    def map_loaded_to_standardized_name(self, loaded_key: str) -> str:
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

    def load_weights(self, model_for_loading: nnx.Module):
        model_params = nnx.state(model_for_loading)
        with jax.default_device(jax.devices("cpu")[0]):
            for loaded_name, loaded_weight in self.names_and_weights_generator:
                if loaded_name.endswith(".bias"):
                    continue
                if "vision_model" in loaded_name or "multi_modal_projector" in loaded_name:
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
                logger.info(
                    f"Transformed parameter {loaded_name} to {mapped_name}: {loaded_weight.shape} --> {model_weight.value.shape}"
                )

                # some of the model_weight.sharding entries were tuples and not NamedSharding objects
                sharding_spec = model_weight.sharding
                if isinstance(sharding_spec, NamedSharding):
                    sharding_spec = sharding_spec.spec
                elif sharding_spec == ():
                    sharding_spec = P()

                model_weight.value = shard_put(loaded_weight,
                                               sharding_spec,
                                               mesh=model_for_loading.mesh)
                if self.is_verbose:
                    print_param_info(model_weight, loaded_name)

        nnx.update(model_for_loading, model_params)
