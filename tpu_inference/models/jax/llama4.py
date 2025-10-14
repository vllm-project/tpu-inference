import re
from typing import List, Optional, Tuple

import jax
import jax.numpy as jnp
from flax import nnx
from flax.typing import PRNGKey
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P
from vllm.config import VllmConfig

from tpu_inference.layers.jax.attention.attention import AttentionMetadata
from tpu_inference.layers.jax.attention.llama4_attention import Llama4Attention
from tpu_inference.layers.jax.constants import KVCacheType
from tpu_inference.layers.jax.layers import DenseFFW, Embedder, LMhead, RMSNorm
from tpu_inference.layers.jax.misc import shard_put
from tpu_inference.layers.jax.moe.moe import MoE, Router
from tpu_inference.layers.jax.transformer_block import \
    SharedExpertsTransformerBlock
from tpu_inference.logger import init_logger
from tpu_inference.models.jax.utils.weight_utils import (
    get_param, model_weights_generator, print_param_info, reshape_params,
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
        text_config = model_config.hf_config.text_config

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
        self.vocab_size = model_config.get_vocab_size()
        self.hidden_size = model_config.get_hidden_size()

        dtype: jnp.dtype = jnp.bfloat16

        self.num_layers: int = getattr(text_config, "num_hidden_layers", 48)

        self.intermediate_size_moe: int = getattr(text_config,
                                                  "intermediate_size", 8192)
        self.intermediate_size_mlp = getattr(text_config,
                                             "intermediate_size_mlp", 16384)

        # num_local_experts: uses 16 experts for Llama-4-Scout-17B-16E-Instruct and uses 128 experts Llama-4-Maverick-17B-128E-Instruct.
        # The default value is set to 16 for compatibility with Llama-4-Scout.
        self.num_local_experts: int = getattr(text_config, "num_local_experts",
                                              16)
        self.hidden_act: str = getattr(text_config, "hidden_act", "silu")
        self.no_rope_layer_interval = getattr(text_config, "no_rope_layers",
                                              [])

        # interleave_moe_layer_step has a layer step of 2 to interleave MoE and dense layers for Llama-4-Maverick-17B-128E-Instruct.
        # The default value is set to 1 for compatibility with Llama-4-Scout.
        self.interleave_moe_layer_step = getattr(text_config,
                                                 "interleave_moe_layer_step",
                                                 1)

        self.num_attention_heads = getattr(text_config, "num_attention_heads",
                                           40)
        self.num_key_value_heads = getattr(text_config, "num_key_value_heads",
                                           8)
        self.head_dim = getattr(text_config, "head_dim", 128)

        self.num_shared_experts = getattr(text_config, "num_experts_per_tok",
                                          1)
        self.rms_norm_eps = getattr(text_config, "rms_norm_eps", 1e-5)

        self.embedder = Embedder(vocab_size=self.vocab_size,
                                 hidden_size=self.hidden_size,
                                 dtype=dtype,
                                 vd_sharding=(('data', 'expert', 'model'),
                                              None),
                                 rngs=self.rng,
                                 random_init=force_random_weights)

        self.layers = []

        for i in range(self.num_layers):
            # For Llama4-Scout, all layers are MoE layers.
            # This can be adjusted for other variants.
            is_moe_layer = (i + 1) % \
                            self.interleave_moe_layer_step == 0

            # Llama-4-Scout config: It has "no_rope_layers": []
            use_attention_rope = (i + 1) not in self.no_rope_layer_interval

            router = Router(dtype=dtype,
                            hidden_size=self.hidden_size,
                            num_experts=self.num_local_experts,
                            num_experts_per_tok=1,
                            router_act="sigmoid",
                            rngs=self.rng,
                            activation_ffw_td=('data', None),
                            ed_sharding=(None, 'expert'),
                            random_init=force_random_weights)

            custom_module = MoE(
                dtype=dtype,
                num_local_experts=self.num_local_experts,
                apply_expert_weight_before_computation=True,
                hidden_size=self.hidden_size,
                intermediate_size_moe=self.intermediate_size_moe,
                hidden_act=self.hidden_act,
                router=router,
                rngs=self.rng,
                activation_ffw_td=('data', None),
                activation_ffw_ted=('data', 'expert', None),
                edf_sharding=('expert', None, 'model'),
                efd_sharding=('expert', 'model', None),
                random_init=force_random_weights
            ) if is_moe_layer else DenseFFW(
                dtype=dtype,
                hidden_act=self.hidden_act,
                hidden_size=self.hidden_size,
                intermediate_size=self.intermediate_size_mlp,
                random_init=force_random_weights,
                rngs=self.rng,
                df_sharding=(None, 'model'),
                fd_sharding=('model', None),
                activation_ffw_td=('data', None))

            attn = Llama4Attention(
                hidden_size=self.hidden_size,
                dtype=dtype,
                kv_cache_dtype=vllm_config.cache_config.cache_dtype,
                num_attention_heads=self.num_attention_heads,
                num_key_value_heads=self.num_key_value_heads,
                head_dim=self.head_dim,
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
                use_qk_norm=True,
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

            shared_experts = DenseFFW(
                dtype=dtype,
                hidden_act=self.hidden_act,
                hidden_size=self.hidden_size,
                intermediate_size=self.num_shared_experts *
                self.intermediate_size_moe,
                rngs=self.rng,
                random_init=force_random_weights,
                df_sharding=(None, 'model'),
                fd_sharding=('model', None),
                activation_ffw_td=('data', None)) if is_moe_layer else None

            pre_attention_norm = RMSNorm(
                dims=self.hidden_size,
                random_init=force_random_weights,
                epsilon=self.rms_norm_eps,
                rngs=self.rng,
                with_scale=True,
                dtype=dtype,
            )

            pre_mlp_norm = RMSNorm(
                dims=self.hidden_size,
                epsilon=self.rms_norm_eps,
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
            epsilon=self.rms_norm_eps,
            rngs=self.rng,
            with_scale=True,
            dtype=dtype,
            random_init=force_random_weights,
        )

        self.lm_head = LMhead(vocab_size=self.vocab_size,
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
        self.interleave_moe_layer_step = getattr(
            vllm_config.model_config.hf_config.text_config,
            "interleave_moe_layer_step", 1)

        self.expert_prefix = "shared_expert."
        self._transpose_map = {
            "q_proj": (2, 0, 1),
            "k_proj": (2, 0, 1),
            "v_proj": (2, 0, 1),
            "router": (1, 0),
            f"{self.expert_prefix}down_proj": (1, 0),
            f"{self.expert_prefix}gate_proj": (1, 0),
            f"{self.expert_prefix}up_proj": (1, 0),
            "feed_forward.down_proj": (1, 0),
            "feed_forward.gate_proj": (1, 0),
            "feed_forward.up_proj": (1, 0),
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
            "language_model.model.layers.*.feed_forward.down_proj.weight":
            "layers.*.custom_module.kernel_down_proj_FD",
            "language_model.model.layers.*.feed_forward.up_proj.weight":
            "layers.*.custom_module.kernel_up_proj_DF",
            "language_model.model.layers.*.feed_forward.gate_proj.weight":
            "layers.*.custom_module.kernel_gating_DF",
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

    def _get_layer_num(self, loaded_key: str) -> Optional[int]:
        """
        Extracts the layer number from a HuggingFace weight key string.
        Returns the layer number (int) or None if no layer number is found.
        """
        match = re.search(r"layers\.(\d+)", loaded_key)
        if match:
            return int(match.group(1))
        return None

    def load_weights(self, model_for_loading: nnx.Module):
        model_params = nnx.state(model_for_loading)

        with jax.default_device(jax.devices("cpu")[0]):
            for loaded_name, loaded_weight in self.names_and_weights_generator:
                is_moe_layer = False
                layer_num = self._get_layer_num(loaded_name)

                if layer_num is not None:
                    is_moe_layer = (layer_num + 1) % \
                            self.interleave_moe_layer_step == 0
                    self.expert_prefix = "shared_expert." if is_moe_layer else ""

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
