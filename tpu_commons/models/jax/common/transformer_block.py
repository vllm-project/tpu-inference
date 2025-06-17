from dataclasses import dataclass
from typing import Any, Type

# Flax and JAX sharding imports
from flax import nnx
from jax.sharding import Mesh

from tpu_commons.models.jax.common.attention.attention import *
from tpu_commons.models.jax.common.constants import *
from tpu_commons.models.jax.common.kv_cache import *
from tpu_commons.models.jax.common.layers import *
from tpu_commons.models.jax.common.moe.moe import *
from tpu_commons.models.jax.common.sharding import *


@dataclass
class TransformerConfig(Config):
    """
    light weighted transformer config, which includes config for all sub-modules
    it uses make() to create the live module from this config
    """
    routing: bool = False

    def from_cfg(self, flags_cfg):
        self.attention_cfg = AttentionConfig.from_cfg(flags_cfg)
        self.kv_cache_cfg = KVCacheConfig.from_cfg(flags_cfg)
        if self.routing:
            self.moe_cfg = MoEConfig.from_cfg(flags_cfg)
            self.router_cfg = RoutingConfig.from_cfg(flags_cfg)
        self.ffw_cfg = FFWConfig.from_cfg(flags_cfg)
        self.cfg = flags_cfg


@dataclass
class TransformerBlock(nnx.Module):
    """
    A heavy weight module which serves as the stateful live blocks in serving
    """
    cfg: TransformerConfig
    param_factory: ParamFactory
    mesh: Mesh
    sharding_cfg: ShardingConfig
    quant: Any | None = None

    def _create_module(self, module_cls: Type[nnx.Module], cfg: Any,
                       **overrides) -> nnx.Module:
        args = {
            "mesh": self.mesh,
            "param_factory": self.param_factory,
            "sharding_cfg": self.sharding_cfg,
            "quant": self.quant
        }
        args.update(overrides)
        return module_cls(cfg=cfg, **args)

    def __post_init__(self):

        self.d_model = self.cfg.attention_cfg.d_model
        self.attn = self._create_module(Attention, cfg=self.cfg.attention_cfg)
        self.kv_cache = KVCache(
            cfg=self.cfg.kv_cache_cfg,
            mesh=self.mesh,
            sharding_cfg=self.sharding_cfg,
            updater=StandardUpdater(),
        )

        self.mlp = self._create_module(FFW, cfg=self.cfg.ffw_cfg)

        if self.cfg.routing:
            self.router = self._create_module(Router, cfg=self.cfg.router_cfg)
            self.moe = self._create_module(MoE,
                                           cfg=self.cfg.moe_cfg,
                                           router=self.router)

        self.post_attention_norm = RMSNorm(dims=self.d_model,
                                           mesh=self.mesh,
                                           param_factory=self.param_factory,
                                           sharding_cfg=self.sharding_cfg,
                                           quant=self.quant)
        self.post_mlp_norm = RMSNorm(dims=self.d_model,
                                     mesh=self.mesh,
                                     param_factory=self.param_factory,
                                     sharding_cfg=self.sharding_cfg,
                                     quant=self.quant)

    def __call__(
        self,
        x,
        attention_metadata,
        op_mode,
    ):
        new_cache, score = self.attn(x, op_mode, self.kv_cache,
                                     attention_metadata)

        x = self.post_attention_norm(x + score)
        if self.cfg.routing:
            y = self.moe(x, op_mode)
        else:
            y = self.mlp(x, op_mode)
        logits = self.post_mlp_norm(x + y)

        return new_cache, logits
