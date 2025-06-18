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

from dataclasses import dataclass, field
from typing import List, Mapping, Optional, Tuple

import jax
import jax.numpy as jnp
from flax import nnx
from flax.typing import PRNGKey
from jax.sharding import Mesh
from vllm.config import VllmConfig

from tpu_commons.models.jax.common.attention.attention import (
    AttentionConfig, AttentionMetadata)
from tpu_commons.models.jax.common.base import Config, ParamFactory
from tpu_commons.models.jax.common.constants import RouterType
from tpu_commons.models.jax.common.kv_cache import KVCacheConfig, KVCacheType
from tpu_commons.models.jax.common.layers import (Embedder, EmbedderConfig,
                                                  RMSNorm)
from tpu_commons.models.jax.common.model import Model, ModelConfig
from tpu_commons.models.jax.common.moe.moe import MoEConfig, RoutingConfig
from tpu_commons.models.jax.common.sharding import (Sharding,
                                                    ShardingConfig)
from tpu_commons.models.jax.common.transformer_block import (
    TransformerBlock, TransformerBlockConfig)

# from tpu_commons.models.jax.recipes.recipe import Recipe


@dataclass
class Llama4ScoutModelConfig(ModelConfig):
    emb: EmbedderConfig = field(default_factory=lambda: EmbedderConfig(
        vocab_size=202048,
        d_model=5120,  ## TODO: Is this correct?
        dtype=jnp.bfloat16,
        normalize_embeddings=False  # TODO: Confirm
    ))
    layers: TransformerBlockConfig = field(
        default_factory=lambda: TransformerBlockConfig(
            attention=AttentionConfig(
                d_model=5120,  ## TODO: Is this correct?
                num_q_heads=40,
                num_kv_heads=8,
                head_dim=128,
                rope_theta=500000.0,
                rope_scaling={
                    "long_factor": 1.0,
                    "short_factor": 1.0,
                    "scale_factor": 16.0,
                    "original_max_position_embeddings": 8192,
                    # rope_type: llama3 ??? TODO: Confirm if needed.
                },
                dtype=jnp.bfloat16),
            ffw=MoEConfig(
                d_model=5120,  ## TODO: Is this correct?
                hidden_size=16384,
                act="silu",
                dtype=jnp.bfloat16,
                expert_hidden_size=8192,  ## TODO: Is this correct?
                num_experts=16,  ## TODO: Is this correct?
                expert_act="silu",
                apply_expert_weight_before_computation=False,
                router_config=RoutingConfig(
                    d_model=5120,
                    hidden_size=8192,  ## TODO: Is this correct?
                    num_experts=16,  ## TODO: Is this correct?
                    num_experts_per_tok=1,
                    router_type=RouterType.TOP_K,
                    act="silu",
                    expert_capacity=-1,
                    routed_bias=False,
                    routed_scaling_factor=1.0,
                    dtype=jnp.bfloat16)),
            kv_cache=KVCacheConfig(batch_size=1,
                                   cache_len=1024,
                                   num_kv_heads=8,
                                   head_dim=128,
                                   dtype=jnp.float16),
            rmsnorm_epsilon=1e-5,
            block_type="MoE"))
    interleave_moe_layer_step: int = 1
    # num_layers: int = 48
    # num_moe_layers: int = 48
    num_layers: int = 16
    num_moe_layers: int = 16


class Llama4ScoutShardingConfig(ShardingConfig):
    pass


# class Llama4ScoutQuantizationConfig(QuantizationConfig):
#     pass


class Llama4ScoutServingConfig(Config):
    pass


@dataclass
class Llama4ScoutConfig():
    model: Llama4ScoutModelConfig = field(
        default_factory=Llama4ScoutModelConfig)
    sharding: Llama4ScoutShardingConfig = field(
        default_factory=Llama4ScoutShardingConfig)
    # quant: Llama4ScoutQuantizationConfig = None
    serving: Llama4ScoutServingConfig = None
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


# Class Model(nnx.module) will be added
class Llama4Scout(Model):

    def __init__(self, vllm_config: VllmConfig, rng: PRNGKey, mesh: Mesh):
        self.vllm_config = vllm_config
        self.rng = nnx.Rngs(rng)
        self.mesh = mesh
        # jax.debug.breakpoint()
        self.cfg = Llama4ScoutConfig(
            model=Llama4ScoutModelConfig(),
            sharding=Llama4ScoutShardingConfig(),
            # quant=Llama4ScoutQuantizationConfig(),
            serving=Llama4ScoutServingConfig(),
            overrides=self.vllm_config.additional_config.get(
                "overrides", None))
        self.runtime_params = self.vllm_config.additional_config.get(
            "overrides", None)
        self.setup()

    def setup(self) -> None:
        param_factory = ParamFactory(
            rngs=self.rng,
            initializer=nnx.initializers.ones)  # Default for RMSNorm scale
        self.sharding = Sharding(
            # cfg=self.cfg.sharding,
            strategy_dict={"tensor_parallelism": 8})
        self.embedder = Embedder(cfg=self.cfg.model.emb,
                                 mesh=self.mesh,
                                 param_factory=param_factory,
                                 sharding_cfg=self.cfg.sharding)
        # a better way to guarantee the order of initialization
        # i.e. sharding_cfg, quantization should be ready before global_KV_cache etc
        # self.global_KV_cache = self.create_KV_cache()

        # TODO: Confirm against MaxText
        # dense_blocks = [
        #     self.cfg.transformer_moe_blocks_config.make(
        #         name=f'dense_layer_{i}',
        #         runtime_param=self.global_runtime_params[i])
        #     for i in self.cfg.dense_layers
        # ]
        # TODO: What is ParamFactory?
        self.moe_blocks = [
            TransformerBlock(cfg=self.cfg.model.layers,
                             block_type="moe",
                             param_factory=param_factory,
                             mesh=self.mesh,
                             sharding_cfg=self.cfg.sharding)
            for i in range(self.cfg.model.num_moe_layers)
        ]
        self.final_norm = RMSNorm(
            dims=self.cfg.model.layers.ffw.d_model,
            mesh=self.mesh,
            param_factory=param_factory,  # TODO: what to set this to?
            sharding_cfg=self.cfg.sharding,  # Kept for API consistency
            epsilon=self.cfg.model.layers.rmsnorm_epsilon,
            with_scale=True,  # TODO: What is this?
            dtype=self.cfg.model.layers.ffw.dtype,
            # quant=self.cfg.quant,
            # num_groups=self.cfg, # TODO: What is this?
        )

    def apply(self, variables, *args, **kwargs):
        return self.__call__(*args, **kwargs)

    def load_weights(self,
                     model_name_or_path: str,
                     cache_dir: Optional[str] = None):
        # jax.debug.breakpoint()
        # return self.init(self.param_factory.rngs)['params']
        return nnx.state(self).filter(nnx.Param)

    def __call__(self,
                 is_prefill: bool,
                 do_sampling: bool,
                 kv_caches: List[KVCacheType],
                 input_ids: jax.Array,
                 attention_metadata: AttentionMetadata,
                 temperatures: jax.Array = None,
                 top_ps: jax.Array = None,
                 top_ks: jax.Array = None,
                 *args,
                 **kwargs) -> Tuple[List[KVCacheType], jax.Array, jax.Array]:

        x = self.embedder.encode(input_ids)
        # for i, block in self.dense_blocks + self.moe_blocks:
        for block in self.moe_blocks:
            kv_cache = block.kv_cache
            x, kv_cache = block(x)
            # TODO: need to confirm functionality.
            # self.global_KV_cache.append(new_cache)
            block.kv_cache = kv_cache

        final_activation = self.final_norm(x)
        decoder_output = self.embedder.decode(final_activation)

        return decoder_output
