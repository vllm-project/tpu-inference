from dataclasses import dataclass, field, make_dataclass
from typing import Any, Tuple, Type

# Flax and JAX sharding imports
import jax
from flax import nnx
from jax.sharding import Mesh
from vllm.config import VllmConfig

from tpu_commons.models.jax.common.attention.attention import (
    Attention, AttentionConfig, AttentionMetadata, KVCache)
from tpu_commons.models.jax.common.base import Config, ParamFactory
from tpu_commons.models.jax.common.constants import HuggingFaceArgNames
from tpu_commons.models.jax.common.layers import (DenseFFW, DenseFFWConfig,
                                                  RMSNorm)
from tpu_commons.models.jax.common.moe.moe import MoE, MoEConfig, Router
from tpu_commons.models.jax.common.sharding import ShardingConfig

TransformerBlockConfig = make_dataclass(
    "TransformerBlockConfig",
    [("attention", AttentionConfig), ("dense_ffw", DenseFFWConfig),
     ("block_type", str), (HuggingFaceArgNames.RMS_NORM_EPS.value, float),
     ("moe", MoEConfig, field(default=None)),
     ("vllm_config", VllmConfig, field(repr=False, default=None))],
    bases=(Config, ))
TransformerBlockConfig.__doc__ = f"""light weighted transformer config, which includes config for all sub-modules
    it uses make() to create the live module from this config
    Args:
        attention: AttentionConfig config used to specify attention layer parameters.
        dense_ffw: DenseFFWConfig config used to specify feed-forward layer parameters.
        block_type: str The type of transformer block (currently support ["moe", "dense"]).
        {HuggingFaceArgNames.RMS_NORM_EPS.value}: float The epsilon value for RMSNorm.
        vllm_config: VllmConfig The VLLM config containing any overrides to apply.
        """


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
        hidden_size = getattr(self.cfg.attention,
                              HuggingFaceArgNames.HIDDEN_SIZE.value)
        rmsnorm_epsilon = getattr(self.cfg,
                                  HuggingFaceArgNames.RMS_NORM_EPS.value)
        self.attn = self._create_module(Attention, cfg=self.cfg.attention)

        if self.block_type == "moe":
            router = self._create_module(Router, cfg=self.cfg.moe.router)
            self.moe = self._create_module(MoE,
                                           cfg=self.cfg.moe,
                                           router=router)
        else:
            self.mlp = self._create_module(DenseFFW, cfg=self.cfg.dense_ffw)

        self.pre_attention_norm = RMSNorm(
            dims=hidden_size,
            mesh=self.mesh,
            param_factory=self.param_factory,
            sharding_cfg=self.sharding_cfg,
            epsilon=rmsnorm_epsilon,
            with_scale=True,
            dtype=self.cfg.attention.dtype,
        )
        self.pre_mlp_norm = RMSNorm(
            dims=hidden_size,
            mesh=self.mesh,
            param_factory=self.param_factory,
            sharding_cfg=self.sharding_cfg,
            epsilon=rmsnorm_epsilon,
            with_scale=True,
            dtype=self.cfg.dense_ffw.dtype,
        )

    def __call__(
            self, x: jax.Array, is_prefill: bool, kv_cache: KVCache,
            attention_metadata: AttentionMetadata
    ) -> Tuple[KVCache, jax.Array]:
        op_mode = "prefill" if is_prefill else "generate"
        # Attn Block
        attn_residual = x
        x = self.pre_attention_norm(x)
        new_cache, attn_output = self.attn(x, is_prefill, kv_cache,
                                           attention_metadata)
        attn_output += attn_residual

        # FFW Block
        ffw_residual = attn_output
        normed_ffw_input = self.pre_mlp_norm(attn_output)
        if self.block_type == "moe":
            logits = self.moe(normed_ffw_input, op_mode)
        elif self.block_type == "dense":
            logits = self.mlp(normed_ffw_input, op_mode)
        else:
            raise ValueError(f"Invalid block type: {self.block_type}")
        logits += ffw_residual
        return new_cache, logits

    def generate_kernel(self, rngs: nnx.Rngs):
        self.attn.generate_kernel(rngs)
        if self.block_type == "moe":
            self.router.generate_kernel(rngs)
            self.moe.generate_kernel(rngs)
        else:
            self.mlp.generate_kernel(rngs)
        self.pre_attention_norm.generate_kernel(rngs)
        self.pre_mlp_norm.generate_kernel(rngs)
