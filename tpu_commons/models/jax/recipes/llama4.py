import re
from dataclasses import dataclass, replace
from typing import List, Optional, Tuple

import jax
import jax.numpy as jnp
from flax import nnx
from flax.typing import PRNGKey
from jax.sharding import Mesh
from vllm.config import VllmConfig

import tpu_commons.models.jax.common.sharding as sharding
from tpu_commons.logger import init_logger
from tpu_commons.models.jax.common.attention.attention import AttentionMetadata
from tpu_commons.models.jax.common.attention.llama4_attention import (
    Llama4Attention, Llama4AttentionConfig)
from tpu_commons.models.jax.common.base import ParamFactory
from tpu_commons.models.jax.common.constants import KVCacheType, RouterType
from tpu_commons.models.jax.common.layers import (DenseFFW, DenseFFWConfig,
                                                  Embedder, LMhead, RMSNorm)
from tpu_commons.models.jax.common.model import Model
from tpu_commons.models.jax.common.moe.moe import MoE, MoEConfig, RouterConfig
from tpu_commons.models.jax.common.sharding import (Sharding,
                                                    ShardingRulesConfig)
from tpu_commons.models.jax.common.transformer_block import (
    SharedExpertsTransformerBlock, SharedExpertsTransformerBlockConfig)
from tpu_commons.models.jax.layers.misc import shard_put
from tpu_commons.models.jax.utils.weight_utils import (
    get_param, hf_model_weights_iterator, print_param_info, reshape_params,
    transpose_params)

logger = init_logger(__name__)


@dataclass
class Llama4ShardingRulesConfig(ShardingRulesConfig):
    lm_head_dv: tuple = (None, sharding.MLP_TENSOR_AXIS_NAME)


class Llama4ForCausalLM(Model):

    def __init__(self,
                 vllm_config: VllmConfig,
                 rng: PRNGKey,
                 mesh: Mesh,
                 param_factory: ParamFactory | None = None):
        assert mesh is not None

        self.vllm_config = vllm_config
        self.rng = nnx.Rngs(rng)
        self.mesh = mesh
        self.param_factory = param_factory
        self.is_verbose = getattr(self.vllm_config.additional_config,
                                  "is_verbose", False)

        # Currently the runner will always set a mesh, so the custom default sharding (when
        #  no sharding is set in vllm config) doesn't take effect.
        # TODO(fhzhang): figure out whether we need to actually enable this.
        #    strategy_dict = {"tensor_parallelism": 4, "expert_parallelism": 2}

        self._sharding_config = Sharding(
            default_rules_cls=Llama4ShardingRulesConfig,
            vllm_config=self.vllm_config).sharding_cfg

        self.hidden_size: int = 5120
        dtype: jnp.dtype = jnp.bfloat16
        num_layers: int = 48
        self.interleave_moe_layer_step = 1  # All layers are MoE for Scout
        intermediate_size_moe: int = 8192
        num_local_experts: int = 16
        hidden_act: str = "silu"
        self.no_rope_layer_interval = 4

        layer_config = SharedExpertsTransformerBlockConfig(
            shared_experts=1,
            attention=Llama4AttentionConfig(
                hidden_size=self.hidden_size,
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
                rope_input_ordering="interleaved",
                temperature_tuning=True,
                temperature_tuning_scale=0.1,
                temperature_tuning_floor_scale=8192,
                use_qk_norm=True,
                attention_chunk_size=8192,
                dtype=dtype,
                vllm_config=self.vllm_config),
            dense_ffw=DenseFFWConfig(hidden_size=self.hidden_size,
                                     intermediate_size=16384,
                                     hidden_act=hidden_act,
                                     dtype=dtype,
                                     vllm_config=self.vllm_config),
            moe=MoEConfig(hidden_size=self.hidden_size,
                          intermediate_size_moe=intermediate_size_moe,
                          dtype=dtype,
                          num_local_experts=num_local_experts,
                          hidden_act=hidden_act,
                          apply_expert_weight_before_computation=True,
                          router=RouterConfig(
                              hidden_size=self.hidden_size,
                              num_local_experts=num_local_experts,
                              num_experts_per_token=1,
                              router_type=RouterType.TOP_K,
                              router_act="sigmoid",
                              expert_capacity=-1,
                              dtype=dtype,
                              vllm_config=self.vllm_config),
                          vllm_config=self.vllm_config),
            rms_norm_eps=1e-5,
            vllm_config=self.vllm_config)

        self.num_attention_heads = layer_config.attention.num_attention_heads
        self.num_key_value_heads = layer_config.attention.num_key_value_heads
        self.head_dim = layer_config.attention.head_dim

        logger.info(f"Using the following config:\n {self._sharding_config}")

        if not self.param_factory:
            self.param_factory = ParamFactory(
                kernel_initializer=nnx.initializers.xavier_normal(),
                scale_initializer=nnx.initializers.ones,
                random_init=False)

        vocab_size = 202048
        self.embedder = Embedder(vocab_size=vocab_size,
                                 hidden_size=self.hidden_size,
                                 dtype=dtype,
                                 generate_rules_prelogit_td=self.
                                 _sharding_config.generate_rules.prelogit_td,
                                 generate_rules_vocab_vd=self._sharding_config.
                                 generate_rules.vocab_vd,
                                 mesh=self.mesh,
                                 param_factory=self.param_factory)
        self.embedder.generate_kernel(self.rng)

        self.layers = []

        for i in range(num_layers):
            # For Llama4-Scout, all layers are MoE layers.
            # This can be adjusted for other variants.
            is_moe_layer = (i + 1) % \
                            self.interleave_moe_layer_step == 0
            use_attention_rope = (i + 1) % self.no_rope_layer_interval != 0
            block_cfg_nope = layer_config
            # RoPE layers do not use chunked attention
            block_cfg_rope = replace(
                layer_config,
                attention=replace(layer_config.attention,
                                  attention_chunk_size=None),
            )
            block_cfg = block_cfg_rope if use_attention_rope else block_cfg_nope
            custom_module = MoE(cfg=layer_config.moe,
                                mesh=self.mesh,
                                param_factory=self.param_factory,
                                sharding_cfg=self._sharding_config
                                ) if is_moe_layer else DenseFFW(
                                    cfg=layer_config.dense_ffw,
                                    mesh=self.mesh,
                                    param_factory=self.param_factory,
                                    sharding_cfg=self._sharding_config)
            block = SharedExpertsTransformerBlock(
                cfg=block_cfg,
                custom_module=custom_module,
                attention_cls=Llama4Attention,
                use_attention_rope=use_attention_rope,
                param_factory=self.param_factory,
                mesh=self.mesh,
                sharding_cfg=self._sharding_config)
            self.layers.append(block)

        for i in range(len(self.layers)):
            self.layers[i].generate_kernel(self.rng)

        self.final_norm = RMSNorm(
            dims=self.hidden_size,
            mesh=self.mesh,
            param_factory=self.param_factory,
            prefill_rules=self._sharding_config.prefill_rules,
            generate_rules=self._sharding_config.generate_rules,
            epsilon=layer_config.rms_norm_eps,
            with_scale=True,
            dtype=dtype,
        )
        self.final_norm.generate_kernel(self.rng)

        self.lm_head = LMhead(vocab_size=vocab_size,
                              hidden_size=self.hidden_size,
                              dtype=dtype,
                              generate_rules_prelogit_td=self._sharding_config.
                              generate_rules.prelogit_td,
                              generate_rules_vocab_vd=self._sharding_config.
                              generate_rules.vocab_vd,
                              generate_rules_vocab_dv=self._sharding_config.
                              generate_rules.vocab_dv,
                              mesh=self.mesh,
                              param_factory=self.param_factory)
        self.lm_head.generate_kernel(self.rng)
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
    ) -> Tuple[List[KVCacheType], jax.Array, jax.Array]:
        is_prefill = False
        x_TD = self.embedder.encode(input_ids)
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
        return logits_TV


class Llama4WeightLoader:

    def __init__(self, vllm_config: VllmConfig, hidden_size, attn_heads,
                 num_key_value_heads, attn_head_dim):
        self.names_and_weights_generator = hf_model_weights_iterator(
            model_name_or_path=vllm_config.model_config.model,
            framework="flax",
            filter_regex="language_model")
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
                mapped_name = "layers.*.moe.kernel_gating_EDF"
                loaded_weight = split_weights[0]
            else:
                mapped_name = "layers.*.moe.kernel_up_proj_EDF"
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
            mapped_model_weight.value = shard_put(
                loaded_weight,
                mapped_model_weight.sharding.spec,
                mesh=model_for_loading.mesh)
            logger.info(
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
                logger.info(
                    f"Transformed parameter {loaded_name} to {mapped_name}: {loaded_weight.shape} --> {model_weight.value.shape}"
                )
                model_weight.value = shard_put(loaded_weight,
                                               model_weight.sharding.spec,
                                               mesh=model_for_loading.mesh)
                if self.is_verbose:
                    print_param_info(model_weight, loaded_name)

        nnx.update(model_for_loading, model_params)
