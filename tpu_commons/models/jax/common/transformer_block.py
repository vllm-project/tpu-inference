from dataclasses import dataclass
from typing import Any, List, Mapping, Tuple, Type

# Flax and JAX sharding imports
from flax import nnx
from jax.sharding import Mesh

from tpu_commons.models.jax.common.attention.attention import (
    Attention, AttentionConfig, AttentionMetadata, KVCache)
from tpu_commons.models.jax.common.base import Config, ParamFactory
from tpu_commons.models.jax.common.layers import FFW, FFWConfig, RMSNorm
from tpu_commons.models.jax.common.moe.moe import MoE, Router, RoutingConfig
from tpu_commons.models.jax.common.sharding import *


@dataclass
class TransformerBlockConfig(Config):
    """
    light weighted transformer config, which includes config for all sub-modules
    it uses make() to create the live module from this config
    """
    attention: AttentionConfig
    ffw: FFWConfig = None
    router: RoutingConfig = None
    block_type: str = None
    rmsnorm_epsilon: float = None
    overrides: Mapping[str, Any] = None

    def _block_type_cls(self) -> FFW:
        if self.block_type.lower() == "moe":
            return MoE
        elif self.block_type.lower() == "dense":
            return FFW
        else:
            raise ValueError(f"Invalid block type: {self.block_type}")

    def from_cfg(self, flags_cfg):
        self.attention = AttentionConfig.from_cfg(flags_cfg)
        block_class = self._block_type_cls()
        self.ffw = block_class.from_cfg(flags_cfg)
        self.cfg = flags_cfg


@dataclass
class TransformerBlock(nnx.Module):
    """
    A heavy weight module which serves as the stateful live blocks in serving
    """
    cfg: TransformerBlockConfig
    block_type: str
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

        self.d_model = self.cfg.attention.d_model
        self.attn = self._create_module(Attention, cfg=self.cfg.attention)

        if self.block_type == "moe":
            self.router = self._create_module(Router, cfg=self.cfg.router)
            self.moe = self._create_module(MoE,
                                           cfg=self.cfg.ffw,
                                           router=self.router)
        else:
            self.mlp = self._create_module(FFW, cfg=self.cfg.ffw)

        self.post_attention_norm = RMSNorm(
            dims=self.cfg.ffw.d_model,
            mesh=self.mesh,
            param_factory=self.param_factory,
            sharding_cfg=self.sharding_cfg,
            epsilon=self.cfg.rmsnorm_epsilon,
            with_scale=True,
            dtype=self.cfg.ffw.dtype,
        )
        self.post_mlp_norm = RMSNorm(
            dims=self.cfg.ffw.d_model,
            mesh=self.mesh,
            param_factory=self.param_factory,
            sharding_cfg=self.sharding_cfg,
            epsilon=self.cfg.rmsnorm_epsilon,
            with_scale=True,
            dtype=self.cfg.ffw.dtype,
        )

    def __call__(self,
                 x: jax.Array,
                 is_prefill: bool,
                 kv_cache: KVCache,
                 attention_metadata: AttentionMetadata) -> Tuple[KVCache, jax.Array]:
        op_mode = "prefill" if is_prefill else "generate"
        new_cache, score = self.attn(x, is_prefill, kv_cache,
                                          attention_metadata)
        x = self.post_attention_norm(x + score)
        if self.block_type == "moe":
            y = self.moe(x, op_mode)
        elif self.block_type == "dense":
            y = self.mlp(x, op_mode)
        else:
            raise ValueError(f"Invalid block type: {self.block_type}")

        logits = self.post_mlp_norm(x + y)

        return new_cache, logits

    def generate_kernel(self, rngs: nnx.Rngs):
        self.attn.generate_kernel(rngs)
        if self.block_type == "moe":
            self.router.generate_kernel(rngs)
            self.moe.generate_kernel(rngs)
        else:
            self.mlp.generate_kernel(rngs)
        self.post_attention_norm.generate_kernel(rngs)
        self.post_mlp_norm.generate_kernel(rngs)