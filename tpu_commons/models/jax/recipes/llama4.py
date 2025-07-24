import io
import pprint
import re
from dataclasses import dataclass, field, replace
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
from tpu_commons.models.jax.common.attention.llama4_attention import \
    Llama4AttentionConfig
from tpu_commons.models.jax.common.base import Config, ParamFactory
from tpu_commons.models.jax.common.constants import RouterType
from tpu_commons.models.jax.common.kv_cache import KVCacheType
from tpu_commons.models.jax.common.layers import (DenseFFWConfig, Embedder,
                                                  EmbedderConfig, RMSNorm)
from tpu_commons.models.jax.common.model import Model, ModelConfig
from tpu_commons.models.jax.common.moe.moe import MoEConfig, RouterConfig
from tpu_commons.models.jax.common.sharding import (Sharding, ShardingConfig,
                                                    ShardingRulesConfig)
from tpu_commons.models.jax.common.transformer_block import (
    SharedExpertsTransformerBlock, SharedExpertsTransformerBlockConfig)
from tpu_commons.models.jax.layers.misc import shard_put
from tpu_commons.models.jax.recipes.recipe import RecipeConfig
from tpu_commons.models.jax.utils.weight_utils import WeightLoader, get_param

logger = init_logger(__name__)
pp = pprint.PrettyPrinter(depth=6)
string_buffer = io.StringIO()


def print_param_info(param: nnx.Param, name: str):
    logger.warning(f"Global shape for {name}: {param.value.shape}"
                   )  # Note: sharding is a PartitionSpec
    logger.warning(f"Sharding for {name}: {param.sharding}"
                   )  # Note: sharding is a PartitionSpec

    # Print tensor shape for a single device
    # buffers = [shard.data for shard in my_array.addressable_shards]
    logger.warning(
        f"Shape of {name} on a single device: {param.value.addressable_shards[0].data.shape}"
    )


@dataclass
class Llama4ModelConfig(ModelConfig):
    # Add parameters shared across multiple layers here.
    hidden_size: int = 5120
    dtype: jnp.dtype = jnp.bfloat16
    num_layers: int = 48
    emb: EmbedderConfig = None
    layers: SharedExpertsTransformerBlockConfig = None
    vllm_config: VllmConfig = field(repr=False, default=None)
    interleave_moe_layer_step: int = 1  # All layers are MoE for Scout
    intermediate_size_moe: int = 8192
    num_local_experts: int = 16
    hidden_act = "silu"
    no_rope_layer_interval = 4

    def __post_init__(self):
        # Initialize defaults:
        if not self.emb:
            self.emb = EmbedderConfig(vocab_size=202048,
                                      hidden_size=self.hidden_size,
                                      dtype=self.dtype,
                                      normalize_embeddings=False,
                                      vllm_config=self.vllm_config)
        if not self.layers:
            self.layers = SharedExpertsTransformerBlockConfig(
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
                    temperature_tuning=True,
                    temperature_tuning_scale=0.1,
                    temperature_tuning_floor_scale=8192,
                    use_qk_norm=True,
                    attention_chunk_size=8192,
                    dtype=self.dtype,
                    vllm_config=self.vllm_config),
                dense_ffw=DenseFFWConfig(hidden_size=self.hidden_size,
                                         intermediate_size=16384,
                                         hidden_act=self.hidden_act,
                                         dtype=self.dtype,
                                         vllm_config=self.vllm_config),
                moe=MoEConfig(hidden_size=self.hidden_size,
                              intermediate_size_moe=self.intermediate_size_moe,
                              dtype=self.dtype,
                              num_local_experts=self.num_local_experts,
                              hidden_act=self.hidden_act,
                              apply_expert_weight_before_computation=True,
                              router=RouterConfig(
                                  hidden_size=self.hidden_size,
                                  num_local_experts=self.num_local_experts,
                                  num_experts_per_token=1,
                                  router_type=RouterType.TOP_K,
                                  router_act="sigmoid",
                                  expert_capacity=-1,
                                  dtype=self.dtype,
                                  vllm_config=self.vllm_config),
                              vllm_config=self.vllm_config),
                rms_norm_eps=1e-5,
                vllm_config=self.vllm_config)


@dataclass
class Llama4ShardingRulesConfig(ShardingRulesConfig):
    lm_head_dv: tuple = (None, sharding.MLP_TENSOR_AXIS_NAME)


@dataclass
class Llama4ServingConfig(Config):
    vllm_config: VllmConfig = field(repr=False, default=None)


@dataclass(frozen=True)
class Llama4Config(RecipeConfig):
    model: Llama4ModelConfig = field(default_factory=Llama4ModelConfig)
    sharding: ShardingConfig = field(
        default_factory=ShardingConfig,
        repr=False,
    )
    serving: Llama4ServingConfig = field(default_factory=Llama4ServingConfig)


class Llama4Scout(Model):

    def __init__(self, vllm_config: VllmConfig, rng: PRNGKey, mesh: Mesh):
        self.vllm_config = vllm_config
        self.rng = nnx.Rngs(rng)
        self.mesh = mesh
        try:
            strategy_dict = self.vllm_config.additional_config["sharding"][
                "sharding_strategy"]
        except (KeyError, TypeError):
            strategy_dict = {"tensor_parallelism": 4, "expert_parallelism": 2}
        self.sharding = Sharding(strategy_dict=strategy_dict,
                                 mesh=self.mesh,
                                 default_rules_cls=Llama4ShardingRulesConfig,
                                 vllm_config=self.vllm_config)
        self.use_random_init = self.vllm_config.additional_config.get(
            "random_weights", False)
        self.cfg = Llama4Config(
            model=Llama4ModelConfig(vllm_config=self.vllm_config),
            sharding=self.sharding.sharding_cfg,
            serving=Llama4ServingConfig(vllm_config=self.vllm_config))
        logger.info(f"Using the following config:\n{self.cfg}")
        # TODO: write the shardings to a JSON in the respective experiment dir.
        logger.info(
            f"Using the following sharding overrides:\n{self.sharding}")
        self.mesh = self.sharding.mesh
        self.weight_loader = Llama4WeightLoader(vllm_config=self.vllm_config,
                                                model_config=self.cfg.model)
        self._init_layers()

    def _init_layers(self):
        param_factory = ParamFactory(
            kernel_initializer=nnx.initializers.xavier_normal(),
            scale_initializer=nnx.initializers.ones,
            random_init=self.use_random_init)
        self.embedder = Embedder(cfg=self.cfg.model.emb,
                                 mesh=self.mesh,
                                 param_factory=param_factory,
                                 sharding_cfg=self.cfg.sharding)
        self.embedder.generate_kernel(self.rng)

        self.layers = []

        for i in range(self.cfg.model.num_layers):
            # For Llama4-Scout, all layers are MoE layers.
            # This can be adjusted for other variants.
            is_moe_layer = (i + 1) % \
                            self.cfg.model.interleave_moe_layer_step == 0
            use_attention_rope = (
                i + 1) % self.cfg.model.no_rope_layer_interval != 0
            block_type = "moe" if is_moe_layer else "dense"
            block_cfg_nope = self.cfg.model.layers
            # RoPE layers do not use chunked attention
            block_cfg_rope = replace(
                self.cfg.model.layers,
                attention=replace(self.cfg.model.layers.attention,
                                  attention_chunk_size=None),
            )
            block_cfg = block_cfg_rope if use_attention_rope else block_cfg_nope
            block = SharedExpertsTransformerBlock(
                cfg=block_cfg,
                block_type=block_type,
                attention_type="llama4",
                use_attention_rope=use_attention_rope,
                param_factory=param_factory,
                mesh=self.mesh,
                sharding_cfg=self.cfg.sharding)
            self.layers.append(block)

        for i in range(len(self.layers)):
            self.layers[i].generate_kernel(self.rng)

        self.final_norm = RMSNorm(
            dims=self.cfg.model.hidden_size,
            mesh=self.mesh,
            param_factory=param_factory,
            sharding_cfg=self.cfg.sharding,
            epsilon=self.cfg.model.layers.rms_norm_eps,
            with_scale=True,
            dtype=self.cfg.model.dtype,
        )
        self.final_norm.generate_kernel(self.rng)

        self.lm_head = Embedder(cfg=self.cfg.model.emb,
                                mesh=self.mesh,
                                param_factory=param_factory,
                                sharding_cfg=self.cfg.sharding)
        self.lm_head.generate_kernel(self.rng)
        self._print_model_architecture()

    def _print_model_architecture(self):
        num_display_layers = max(self.cfg.model.interleave_moe_layer_step,
                                 self.cfg.model.no_rope_layer_interval)

        logger.info("### Embedding ###")
        nnx.display(self.embedder)

        logger.info(f"\n### First {num_display_layers} Layers ###")
        # Loop through the slice and display each layer
        for i, layer in enumerate(self.layers[:num_display_layers]):
            logger.info(f"\n--- Layer {i} ---")
            nnx.display(layer)

        logger.info("\n### LM Head ###")
        nnx.display(self.lm_head)

    def load_weights(self, rng: PRNGKey, cache_dir: Optional[str] = None):
        try:
            use_random_weights = self.vllm_config.additional_config[
                "random_weights"]
            logger.warning(
                "Using randomly initialized weights instead of loading parameter weights."
            )
            return
        except KeyError:
            use_random_weights = False
        self.weight_loader.load_weights(self)

    def __call__(
        self,
        kv_caches: List[jax.Array],
        input_ids: jax.Array,
        attention_metadata: AttentionMetadata,
        *args,
    ) -> Tuple[List[KVCacheType], jax.Array, jax.Array]:
        is_prefill = False
        x = self.embedder.encode(input_ids)
        jax.debug.print(
            "Embedder output = {val}",
            val=x[jnp.array([0, 3])[:, None],
                  jnp.concat([jnp.
                              arange(5), jnp.arange(-1, -6, -1)])])
        # def loop_body(i, val):
        #     jax.debug.print("**********Transformer layer: {val}**********", val=i)
        #     # kv_cache = kv_caches[i]
        #     x, kv_caches = val
        #     kv_cache = jax.tree_util.tree_map(lambda x: x[i], kv_caches)
        #     new_kv_cache, x = self.layers[i](x, is_prefill, kv_cache,
        #                             attention_metadata)
        #     # kv_caches[i] = new_kv_cache
        #     updated_kv_caches = jax.tree_util.tree_map(
        #         lambda cache, update: cache.at[i].set(update),
        #         kv_caches,
        #         new_kv_cache
        #     )
        #     return(updated_kv_caches, x)
        # kv_caches, x = jax.lax.fori_loop(0, len(self.layers), loop_body, (x, kv_caches))
        for (i, block) in enumerate(self.layers):
            jax.debug.print("**********Transformer layer: {val}**********",
                            val=i)
            kv_cache = kv_caches[i]
            new_kv_cache, x = block(x, is_prefill, kv_cache,
                                    attention_metadata)
            jax.block_until_ready(x)
            kv_caches[i] = new_kv_cache

        final_activation = self.final_norm(x)
        jax.debug.print(
            "Final activation = {val}",
            val=final_activation[
                jnp.array([0, 3])[:, None],
                jnp.concat([jnp.
                            arange(5), jnp.arange(-1, -6, -1)])])

        return kv_caches, final_activation

    def compute_logits(self, hidden_states: jax.Array) -> jax.Array:
        return self.lm_head.decode(hidden_states)


class Llama4WeightLoader(WeightLoader):

    def __init__(self, vllm_config: VllmConfig, model_config: ModelConfig):
        super().__init__(vllm_config=vllm_config,
                         model_config=model_config,
                         framework="flax",
                         filter_regex="language_model")
        self.setup()

    def setup(self):
        super().setup()
        self.set_transpose_param_map({
            "q_proj": (2, 0, 1),
            "k_proj": (2, 0, 1),
            "v_proj": (2, 0, 1),
            "router": (1, 0),
            "shared_expert.down_proj": (1, 0),
            "shared_expert.gate_proj": (1, 0),
            "shared_expert.up_proj": (1, 0),
            "o_proj": (1, 2, 0),
            "embed_tokens": (1, 0),
            "lm_head": (1, 0),
        })
        hidden_size = self.model_config.hidden_size
        attn_heads = self.model_config.layers.attention.num_attention_heads
        num_key_value_heads = self.model_config.layers.attention.num_key_value_heads
        attn_head_dim = self.model_config.layers.attention.head_dim
        self.set_reshape_param_map(
            {
                "q_proj": (attn_heads, attn_head_dim, hidden_size),
                "k_proj": (num_key_value_heads, attn_head_dim, hidden_size),
                "v_proj": (num_key_value_heads, attn_head_dim, hidden_size),
                # o_proj is inverted: https://github.com/huggingface/transformers/blob/v4.53.2/src/transformers/models/llama4/modeling_llama4.py#L298
                "o_proj": (hidden_size, attn_heads, attn_head_dim),
            },
            param_type="weight",
        )
        # Set the mappings from loaded parameter keys to standardized names.
        self.set_loaded_to_standardized_keys({
            "language_model.model.embed_tokens.weight":
            "embedder.input_embedding_table_DV",
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
            "layers.*.moe.router.kernel_DE",
            "language_model.model.layers.*.feed_forward.experts.down_proj":
            "layers.*.moe.kernel_down_proj_EFD",
            "language_model.model.layers.*.feed_forward.experts.gate_up_proj":
            "layers.*.moe.kernel_up_proj_EDF",
            "language_model.model.layers.*.feed_forward.shared_expert.down_proj.weight":
            "layers.*.shared_experts.kernel_down_proj_FD",
            "language_model.model.layers.*.feed_forward.shared_expert.gate_proj.weight":
            "layers.*.shared_experts.kernel_gating_DF",
            "language_model.model.layers.*.feed_forward.shared_expert.up_proj.weight":
            "layers.*.shared_experts.kernel_up_proj_DF",
        })

    def map_loaded_to_standardized_name(self, loaded_key: str) -> str:
        # Find the corresponding model key using the HF key
        if "layer" in loaded_key:
            layer_num = re.search(r"layers\.(\d+)", loaded_key).group(1)
            layer_key = re.sub(r"layers\.\d+", "layers.*", loaded_key)
            mapped_key = self.loaded_to_standardized_keys.get(
                layer_key, loaded_key)
            mapped_key = re.sub(r"layers\.\*", f"layers.{layer_num}",
                                mapped_key)
        else:
            mapped_key = self.loaded_to_standardized_keys.get(
                loaded_key, loaded_key)
        return mapped_key

    def load_weights(self, model_for_loading: nnx.Module):
        model_params = nnx.state(model_for_loading)
        # names_and_weights = [f"name: {name}; shape = {weight.shape}" for (name, weight) in  self.names_and_weights_generator]
        # logger.info(f"Number of loaded weights: {len(names_and_weights)}")
        # logger.info(f"Loaded weights are the following:\n{pp.pformat(names_and_weights)}")
        logger.warning(
            f"loaded_to_stand/ardized_keys: {self.loaded_to_standardized_keys}"
        )
        cumulative_global_memory = 0
        cumulative_local_memory = 0
        with jax.default_device(jax.devices("cpu")[0]):
            for loaded_name, loaded_weight in self.names_and_weights_generator:
                old_param_name = loaded_name
                # if loaded_name.endswith(".weight"):
                #     loaded_name = loaded_name.removesuffix(".weight")
                logger.info(
                    f"Loaded parameter {loaded_name}: {loaded_weight.shape}")
                if "gate_up_proj" in loaded_name:
                    # HF's gate_up_proj is a fused tensor of gate and up projections.
                    # It needs to be split.
                    gate_w, up_w = jnp.split(loaded_weight, 2, axis=-1)

                    # Handle gate weights
                    gate_loaded_name = loaded_name.replace(
                        "gate_up_proj", "gate_proj")
                    gate_mapped_name = "layers.*.moe.kernel_gating_EDF"
                    layer_num = re.search(r"layers\.(\d+)",
                                          gate_loaded_name).group(1)
                    gate_mapped_name = re.sub(r"layers\.\*",
                                              f"layers.{layer_num}",
                                              gate_mapped_name)
                    gate_model_weight = get_param(model_params,
                                                  gate_mapped_name)
                    # gate_w = gate_w.transpose((0, 2, 1))
                    if gate_model_weight.value.shape != gate_w.shape:
                        raise ValueError(
                            f"Loaded shape for {gate_loaded_name}: {gate_w.shape} "
                            f"does not match model shape for {gate_mapped_name}: {gate_model_weight.value.shape}!"
                        )
                    gate_model_weight.value = shard_put(
                        gate_w,
                        gate_model_weight.sharding.spec,
                        mesh=model_for_loading.mesh)
                    gate_model_weight.value.block_until_ready()
                    logger.info(
                        f"{gate_loaded_name}: {loaded_weight.shape}  -->  {gate_mapped_name}: {gate_model_weight.value.shape}"
                    )
                    del gate_w
                    print_param_info(gate_model_weight, "gate_proj")
                    cumulative_global_memory += gate_model_weight.value.nbytes / 1e9
                    cumulative_local_memory += gate_model_weight.value.addressable_shards[
                        0].data.nbytes / 1e9
                    logger.info(
                        f"Cumulative global memory: {cumulative_global_memory} GB"
                    )
                    logger.info(
                        f"Cumulative local memory: {cumulative_local_memory} GB"
                    )

                    # Handle up weights
                    up_loaded_name = loaded_name.replace(
                        "gate_up_proj", "up_proj")
                    up_mapped_name = "layers.*.moe.kernel_up_proj_EDF"
                    up_mapped_name = re.sub(r"layers\.\*",
                                            f"layers.{layer_num}",
                                            up_mapped_name)
                    up_model_weight = get_param(model_params, up_mapped_name)
                    # up_w = up_w.transpose((0, 2, 1))
                    if up_model_weight.value.shape != up_w.shape:
                        raise ValueError(
                            f"Loaded shape for {up_loaded_name}: {up_w.shape} "
                            f"does not match model shape for {up_mapped_name}: {up_model_weight.value.shape}!"
                        )
                    up_model_weight.value = shard_put(
                        up_w,
                        up_model_weight.sharding.spec,
                        mesh=model_for_loading.mesh)
                    up_model_weight.value.block_until_ready()
                    del up_w
                    print_param_info(up_model_weight, "up_proj")
                    cumulative_global_memory += up_model_weight.value.nbytes / 1e9
                    cumulative_local_memory += up_model_weight.value.addressable_shards[
                        0].data.nbytes / 1e9
                    logger.info(
                        f"{up_loaded_name}: {loaded_weight.shape}  -->  {up_mapped_name}: {up_model_weight.value.shape}"
                    )
                    logger.info(
                        f"Cumulative global memory: {cumulative_global_memory} GB"
                    )
                    logger.info(
                        f"Cumulative local memory: {cumulative_local_memory} GB"
                    )
                    continue

                mapped_name = self.map_loaded_to_standardized_name(loaded_name)
                model_weight = get_param(model_params, mapped_name)
                logger.info(
                    f"{old_param_name}: {loaded_weight.shape}  -->  {mapped_name}: {model_weight.value.shape}"
                )
                if loaded_name.endswith(".bias"):
                    loaded_weight = self.reshape_params(
                        loaded_name, loaded_weight, "bias")
                else:
                    loaded_weight = self.reshape_params(
                        loaded_name, loaded_weight, "weight")
                    loaded_weight = self.transpose_params(
                        loaded_name, loaded_weight)
                if model_weight.value.shape != loaded_weight.shape:
                    raise ValueError(
                        f"Loaded shape for {loaded_name}: {loaded_weight.shape} "
                        f"does not match model shape for {mapped_name}: {model_weight.value.shape}!"
                    )
                # logger.info(f"Transformed parameter {loaded_name} to {mapped_name}: {loaded_weight.shape} --> {model_weight.value.shape}")
                model_weight.value = shard_put(loaded_weight,
                                               model_weight.sharding.spec,
                                               mesh=model_for_loading.mesh)
                model_weight.value.block_until_ready()
                del loaded_weight
                print_param_info(model_weight, loaded_name)
                cumulative_global_memory += model_weight.value.nbytes / 1e9
                cumulative_local_memory += model_weight.value.addressable_shards[
                    0].data.nbytes / 1e9
                logger.info(
                    f"Cumulative global memory: {cumulative_global_memory} GB")
                logger.info(
                    f"Cumulative local memory: {cumulative_local_memory} GB")
        # TODO: validate that all of the model_params were accounted for as well.
        nnx.update(model_for_loading, model_params)
