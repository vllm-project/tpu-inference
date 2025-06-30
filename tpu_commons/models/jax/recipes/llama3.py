# TODO: Update documentation
# Input flags are stored in below configs
# model_flag_config: Config
#   d_model: 2048
#   n_layers: 61
#   n_moe_layer: 58
# parallelism_flag_config: Config
#   tp: 2
#   ep: 4
# quant_flag_config: Config
#
# Each Block(attn, mlp etc) has Config Class and Live Class
# There're 2 ways to initialize a Config:
# 1. manual specify as Config(d_model=, num_layer=, ..)
# 2. auto assignment from a config as Config.from_cfg(cfg)

# The foundation Model/ModelConfig class is to-be-implemented

import re
from dataclasses import dataclass, field
from typing import List, Mapping, Optional, Tuple

import jax
import jax.numpy as jnp
from flax import nnx
from flax.typing import PRNGKey
from jax.sharding import Mesh
from vllm.config import VllmConfig

import tpu_commons.models.jax.common.sharding as sharding
from tpu_commons.logger import init_logger
from tpu_commons.models.jax.common.attention.attention import (
    AttentionConfig, AttentionMetadata)
from tpu_commons.models.jax.common.base import Config, ParamFactory
from tpu_commons.models.jax.common.kv_cache import KVCacheType
from tpu_commons.models.jax.common.layers import (Embedder, EmbedderConfig,
                                                  FFWConfig, RMSNorm)
from tpu_commons.models.jax.common.model import Model, ModelConfig
from tpu_commons.models.jax.common.sharding import (OpShardingConfig, Sharding,
                                                    ShardingConfig)
from tpu_commons.models.jax.common.transformer_block import (
    TransformerBlock, TransformerBlockConfig)
from tpu_commons.models.jax.layers.misc import shard_put
from tpu_commons.models.jax.layers.sampling import sample
from tpu_commons.models.jax.utils.weight_utils import (ParameterType,
                                                       WeightLoader, get_param)

logger = init_logger(__name__)


@dataclass
class Llama8BModelConfig(ModelConfig):
    hidden_size: int = 4096
    dtype: jnp.dtype = jnp.bfloat16
    num_layers: int = 32
    emb: EmbedderConfig = field(default_factory=EmbedderConfig)
    layers: TransformerBlockConfig = field(default_factory=TransformerBlockConfig)

    def __post_init__(self):

        # Initialize defaults:
        if not self.emb:
            self.emb = EmbedderConfig(
                vocab_size=128256,
                hidden_size=self.hidden_size,
                dtype=self.dtype,
                normalize_embeddings=False  # TODO: Confirm
            )
        if not self.layers:
            self.layers = TransformerBlockConfig(
                attention=AttentionConfig(
                hidden_size=self.hidden_size, 
                num_attention_heads=32,
                num_key_value_heads=8,
                head_dim=128,
                rope_theta=500000.0,
                rope_scaling={},
                dtype=jnp.bfloat16),
            ffw=FFWConfig(d_model=4096,
                          hidden_size=14336,
                          act="silu",
                          dtype=jnp.bfloat16),
            rmsnorm_epsilon=1e-5,
            block_type="dense")


class Llama8BOpShardingConfig(OpShardingConfig):
    lm_head_dv: tuple = (None, None)


class Llama8BSharding(Sharding):

    def make_sharding_config(self,
                             prefill_overrides=None,
                             generate_overrides=None) -> ShardingConfig:
        sharding_config = super().make_sharding_config(prefill_overrides,
                                                       generate_overrides)
        sharding_config.prefill_sharding_cfg.lm_head_dv = (
            None, sharding.MLP_TENSOR_AXIS_NAME)
        sharding_config.generate_sharding_cfg.lm_head_dv = (
            None, sharding.MLP_TENSOR_AXIS_NAME)
        return sharding_config


class Llama8BServingConfig(Config):
    pass


@dataclass
class Llama8BConfig():
    model: Llama8BModelConfig = field(default_factory=Llama8BModelConfig)
    sharding: ShardingConfig = field(default_factory=ShardingConfig)
    serving: Llama8BServingConfig = None
    overrides: Mapping[str, any] = None

    def __post_init__(self):
        # TODO: Allow for command-line overrides. Maybe inherit from recipe.py.
        if self.overrides:
            if self.overrides.get("model", None):
                pass
            if self.overrides.get("sharding", None):
                pass
            if self.overrides.get("quant", None):
                pass
            if self.overrides.get("serving", None):
                pass


class Llama3_8B(Model):

    def __init__(self, vllm_config: VllmConfig, rng: PRNGKey, mesh: Mesh):
        self.vllm_config = vllm_config
        self.rng = nnx.Rngs(rng)
        self.mesh = mesh
        self.cfg = Llama8BConfig(
            model=Llama8BModelConfig(),
            sharding=ShardingConfig(default_ops_cls=Llama8BOpShardingConfig),
            serving=Llama8BServingConfig(),
            overrides=self.vllm_config.additional_config.get(
                "overrides", None))
        self.runtime_params = self.vllm_config.additional_config.get(
            "overrides", None)

        param_factory = ParamFactory(
            kernel_initializer=nnx.initializers.xavier_normal(),
            scale_initializer=nnx.initializers.ones)
        try:
            strategy_dict = self.runtime_params["sharding"][
                "sharding_strategy"]
        except (KeyError, TypeError):
            strategy_dict = {"tensor_parallelism": 4, "expert_parallelism": 2}
        self.sharding = Llama8BSharding(
            strategy_dict=strategy_dict,
            mesh=self.mesh,
        )
        self.cfg.sharding = self.sharding.sharding_cfg
        self.mesh = self.sharding.mesh
        self.embedder = Embedder(cfg=self.cfg.model.emb,
                                 mesh=self.mesh,
                                 param_factory=param_factory,
                                 sharding_cfg=self.cfg.sharding)
        self.layers = [
            TransformerBlock(cfg=self.cfg.model.layers,
                             block_type="dense",
                             param_factory=param_factory,
                             mesh=self.mesh,
                             sharding_cfg=self.cfg.sharding)
            for i in range(self.cfg.model.num_layers)
        ]
        self.final_norm = RMSNorm(
            dims=self.cfg.model.layers.ffw.d_model,
            mesh=self.mesh,
            param_factory=param_factory,
            sharding_cfg=self.cfg.sharding,
            epsilon=self.cfg.model.layers.rmsnorm_epsilon,
            with_scale=True,
            dtype=self.cfg.model.layers.ffw.dtype,
        )

        self.lm_head = Embedder(cfg=self.cfg.model.emb,
                                mesh=self.mesh,
                                param_factory=param_factory,
                                sharding_cfg=self.cfg.sharding)

        self.setup()

    def setup(self) -> None:
        self.embedder.generate_kernel(self.rng)
        for i in range(len(self.layers)):
            self.layers[i].generate_kernel(self.rng)
        self.final_norm.generate_kernel(self.rng)
        self.lm_head.generate_kernel(self.rng)

    # For compatibility with flax.
    def apply(self, variables, *args, **kwargs):
        return self.__call__(*args, **kwargs)

    def load_weights(self, rng: PRNGKey, cache_dir: Optional[str] = None):
        # TODO: support gcs paths as well.
        model_name_or_path = self.vllm_config.model_config.model
        if not model_name_or_path:  # TODO
            logger.warning(
                "Model name or path not provided - randomly randomly initializing the weights."
            )
        else:
            weight_loader = Llama3WeightLoader(vllm_config=self.vllm_config,
                                               model_config=self.cfg.model,
                                               cache_dir=None,
                                               sharding_cfg=self.cfg.sharding)
            weight_loader.load_weights(self)

    def __call__(
            self,
            is_prefill: bool,
            do_sampling: bool,
            kv_caches:
        List[
            KVCacheType],  # TODO: Make sure to use this instead of creating in model.
            input_ids: jax.Array,
            attention_metadata: AttentionMetadata,
            temperatures: jax.Array = None,
            top_ps: jax.Array = None,
            top_ks: jax.Array = None,
            *args,
            **kwargs) -> Tuple[List[KVCacheType], jax.Array, jax.Array]:
        x = self.embedder.encode(input_ids)
        for (i, block) in enumerate(self.layers):
            kv_cache = kv_caches[i]
            new_kv_cache, x = block(x, is_prefill, kv_cache,
                                    attention_metadata)
            kv_caches[i] = new_kv_cache

        final_activation = self.final_norm(x)
        decoder_output = self.embedder.decode(final_activation)

        next_tokens = sample(
            is_prefill,
            do_sampling,
            self.rng.params(),
            self.mesh,
            decoder_output,
            attention_metadata.seq_lens,
            temperatures,
            top_ps,
            top_ks,
            attention_metadata.chunked_prefill_enabled,
        )

        return kv_caches, next_tokens, decoder_output


class Llama3WeightLoader(WeightLoader):

    def __init__(self,
                 vllm_config: VllmConfig,
                 model_config: ModelConfig,
                 cache_dir: Optional[str] = None,
                 sharding_cfg: Optional[ShardingConfig] = None):
        super().__init__(vllm_config=vllm_config,
                         model_config=model_config,
                         framework="flax",
                         cache_dir=cache_dir,
                         sharding_cfg=sharding_cfg)
        self.setup()

    def setup(self):
        super().setup()
        self.set_transpose_param_map({
            "gate_proj": (1, 0),
            "up_proj": (1, 0),
            "down_proj": (1, 0),
            "q_proj": (0, 2, 1),
            "k_proj": (0, 2, 1),
            "v_proj": (0, 2, 1),
            "o_proj": (1, 2, 0),
        })
        # Set weights reshape map
        hidden_size = self.model_config.layers.attention.d_model
        attn_heads = self.model_config.layers.attention.num_attention_heads
        num_key_value_heads = self.model_config.layers.attention.num_key_value_heads
        attn_head_dim = self.model_config.layers.attention.head_dim
        self.set_reshape_param_map(
            {
                "q_proj": (attn_heads, -1, hidden_size),
                "k_proj": (num_key_value_heads, -1, hidden_size),
                "v_proj": (num_key_value_heads, -1, hidden_size),
                "o_proj": (hidden_size, attn_heads, -1),
            },
            param_type=ParameterType.weight)
        # Set bias reshape map
        self.set_reshape_param_map(param_reshape_dict={
            "q_proj.bias": (attn_heads, attn_head_dim),
            "k_proj.bias": (num_key_value_heads, attn_head_dim),
            "v_proj.bias": (num_key_value_heads, attn_head_dim)
        },
                                   param_type=ParameterType.bias)
        # Set the mappings from loaded parameter keys to standardized names.
        # TODO: Update with hf names where possible.
        self.set_loaded_to_standardized_keys({
            "model.embed_tokens":
            "embedder.input_embedding_table_VD",
            "model.layers.*.input_layernorm":
            "layers.*.post_mlp_norm.scale",
            "model.layers.*.mlp.down_proj":
            "layers.*.mlp.kernel_down_proj_FD",
            "model.layers.*.mlp.gate_proj":
            "layers.*.mlp.kernel_gating_DF",
            "model.layers.*.mlp.up_proj":
            "layers.*.mlp.kernel_up_proj_DF",
            "model.layers.*.post_attention_layernorm":
            "layers.*.post_attention_norm.scale",
            "model.layers.*.self_attn.k_proj":
            "layers.*.attn.kernel_k_proj_KDH",
            "model.layers.*.self_attn.o_proj":
            "layers.*.attn.kernel_o_proj_NHD",
            "model.layers.*.self_attn.q_proj":
            "layers.*.attn.kernel_q_proj_NDH",
            "model.layers.*.self_attn.v_proj":
            "layers.*.attn.kernel_v_proj_KDH",
            "model.norm":
            "final_norm.scale",  # TODO: is this correct??
            "lm_head":
            "lm_head.input_embedding_table_VD"
        })

    def map_loaded_to_standardized_name(self, loaded_key: str) -> str:
        # Find the corresponding model key using the HF key
        if "layer" in loaded_key:
            layer_num = re.search(r"layers\.(\d+)", loaded_key).group(1)
            layer_key = re.sub(r"layers\.\d+", "layers.*", loaded_key)
            mapped_key = self.loaded_to_standardized_keys[layer_key]
            mapped_key = re.sub(r"layers\.\*", f"layers.{layer_num}",
                                mapped_key)
        else:
            mapped_key = self.loaded_to_standardized_keys.get(
                loaded_key, loaded_key)
        return mapped_key

    def load_weights(self, model_for_loading: nnx.Module):
        model_params = nnx.state(model_for_loading)
        for loaded_name, loaded_weight in self.names_and_weights_generator:
            old_param_name = loaded_name
            if loaded_name.endswith(".weight"):
                loaded_name = loaded_name.removesuffix(".weight")
            mapped_name = self.map_loaded_to_standardized_name(loaded_name)
            model_weight = get_param(model_params, mapped_name)
            logger.debug(
                f"{old_param_name}: {loaded_weight.shape}  -->  {mapped_name}: {model_weight.value.shape}"
            )
            if loaded_name.endswith(".bias"):
                loaded_weight = self.reshape_params(loaded_name, loaded_weight,
                                                    "bias")
            else:
                loaded_weight = self.reshape_params(loaded_name, loaded_weight,
                                                    "weight")
                loaded_weight = self.transpose_params(loaded_name,
                                                      loaded_weight)
            if model_weight.value.shape != loaded_weight.shape:
                raise ValueError(
                    f"Loaded shape for {loaded_name}: {loaded_weight.shape} "
                    f"does not match model shape for {mapped_name}: {model_weight.value.shape}!"
                )
            model_weight.value = shard_put(loaded_weight,
                                           model_weight.sharding.spec,
                                           mesh=model_for_loading.mesh)
        # TODO: validate that all of the model_params were accounted for as well.
        nnx.update(model_for_loading, model_params)
