from dataclasses import dataclass
from typing import Any, List, Mapping, Tuple, Type

# Flax and JAX sharding imports
from flax import nnx
from jax.sharding import Mesh

from tpu_commons.models.jax.common.attention.attention import (Attention,
                                                               AttentionConfig,
                                                               AttentionMetadata)
from tpu_commons.models.jax.common.base import (Config,
                                                ParamFactory)
from tpu_commons.models.jax.common.kv_cache import (KVCache,
                                                    KVCacheConfig,
                                                    KVCacheType,
                                                    StandardUpdater)
from tpu_commons.models.jax.common.layers import (FFW,
                                                  FFWConfig,
                                                  RMSNorm)
from tpu_commons.models.jax.common.moe.moe import MoE
from tpu_commons.models.jax.common.sharding import *


@dataclass
class TransformerBlockConfig(Config):
    """
    light weighted transformer config, which includes config for all sub-modules
    it uses make() to create the live module from this config
    """
    attention: AttentionConfig
    kv_cache: KVCacheConfig
    ffw: FFWConfig = None
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
        self.kv_cache = KVCacheConfig.from_cfg(flags_cfg)
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
        # TODO: If this is now tied to each Transformer block, will it make
        # insert/update (especially for Disag) a bit more cumbersome?
        self.kv_cache = KVCache(
            cfg=self.cfg.kv_cache,
            mesh=self.mesh,
            sharding_cfg=self.sharding_cfg,
            updater=StandardUpdater(),
        )

        self.mlp = self._create_module(FFW, cfg=self.cfg.ffw_cfg)

        if self.block_type == "moe":
            self.moe = self._create_module(MoE, cfg=self.cfg.ffw_cfg)

        self.post_attention_norm = self._create_module(
            RMSNorm,
            cfg={"dims": self.cfg.d_model},
        )
        self.post_mlp_norm = self._create_module(
            RMSNorm, cfg={"dims": self.cfg.d_model})

    # TODO:
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
        op_mode = "prefill" if is_prefill else "generate"
        new_cache, score = self.self_attn(input_ids, op_mode, self.kv_cache,
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
