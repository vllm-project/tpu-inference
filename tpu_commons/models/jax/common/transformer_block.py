from copy import deepcopy
from dataclasses import dataclass, field, make_dataclass
from typing import Any, Tuple, Type

# Flax and JAX sharding imports
import jax
from flax import nnx
from jax.sharding import Mesh
from vllm.config import VllmConfig

from tpu_commons.models.jax.common.attention.attention import (
    Attention, AttentionConfig, AttentionMetadata, KVCache)
from tpu_commons.models.jax.common.attention.llama4_attention import \
    Llama4Attention
from tpu_commons.models.jax.common.base import Config, ParamFactory
from tpu_commons.models.jax.common.constants import HuggingFaceArgNames
from tpu_commons.models.jax.common.layers import (DenseFFW, DenseFFWConfig,
                                                  RMSNorm)
from tpu_commons.models.jax.common.moe.moe import MoE, MoEConfig
from tpu_commons.models.jax.common.sharding import ShardingConfig

ATTENTION_BLOCK_REGISTR = {"default": Attention, "llama4": Llama4Attention}

TransformerBlockConfig = make_dataclass(
    "TransformerBlockConfig",
    [("attention", AttentionConfig), ("dense_ffw", DenseFFWConfig),
     (HuggingFaceArgNames.RMS_NORM_EPS.value, float),
     ("moe", MoEConfig, field(default=None)), ("block_type", str, "default"),
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
    attention_type: str = "default"
    use_attention_rope: bool = True
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
        try:
            attn_block = ATTENTION_BLOCK_REGISTR[self.attention_type]
        except KeyError:
            raise ValueError(f"Invalid attention type: {self.attention_type}")

        self.attn = self._create_module(attn_block, cfg=self.cfg.attention)

        if self.block_type == "moe":
            self.moe = self._create_module(MoE, cfg=self.cfg.moe)
        elif self.block_type == "dense":
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

    def __call__(self,
                 x: jax.Array,
                 is_prefill: bool,
                 kv_cache: KVCache,
                 attention_metadata: AttentionMetadata,
                 use_attention_rope: bool = True) -> Tuple[KVCache, jax.Array]:
        op_mode = "prefill" if is_prefill else "generate"
        # Attn Block
        attn_residual = x
        x = self.pre_attention_norm(x)
        new_cache, attn_output = self.attn(x, is_prefill, kv_cache,
                                           attention_metadata,
                                           use_attention_rope)
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
            self.moe.generate_kernel(rngs)
        else:
            self.mlp.generate_kernel(rngs)
        self.pre_attention_norm.generate_kernel(rngs)
        self.pre_mlp_norm.generate_kernel(rngs)


# Provide a variante that allows mixing and matching Dense & MoE layers.
SharedExpertsTransformerBlockConfig = make_dataclass(
    "SharedExpertsTransformerBlockConfig",
    [(HuggingFaceArgNames.SHARED_EXPERTS.value, int)],
    bases=(TransformerBlockConfig, ),
    kw_only=True)

SharedExpertsTransformerBlockConfig.__doc__ = f"""Transformer block with MoE block and shared experts block (i.e. Dense Block).
Additional Args:
  {HuggingFaceArgNames.SHARED_EXPERTS.value}: Number of experts to route all of the inputs to (essentially a dense layer).

Inherits TransformerBlockConfig docstring:
{TransformerBlockConfig.__doc__}
"""


@dataclass(kw_only=True)
class SharedExpertsTransformerBlock(TransformerBlock):
    """Create a modified TransformerBlock that sums MoE layer output with shared expert output."""

    def __post_init__(self):
        super().__post_init__()
        # Create a modified config for the shared expert (which is a dense FFW layer)
        shared_experts = getattr(self.cfg,
                                 HuggingFaceArgNames.SHARED_EXPERTS.value)
        moe_intermediate_size = getattr(
            self.cfg.moe, HuggingFaceArgNames.INTERMEDIATE_SIZE_MOE.value)
        shared_experts_cfg = deepcopy(self.cfg.dense_ffw)
        setattr(shared_experts_cfg,
                HuggingFaceArgNames.INTERMEDIATE_SIZE.value,
                shared_experts * moe_intermediate_size)
        self.shared_experts = self._create_module(DenseFFW,
                                                  cfg=shared_experts_cfg)

    def __call__(self,
                 x,
                 is_prefill,
                 kv_cache,
                 attention_metadata,
                 use_attention_rope=True):
        op_mode = "prefill" if is_prefill else "generate"
        # Attn Block
        attn_residual = x
        x = self.pre_attention_norm(x)
        new_cache, attn_output = self.attn(x, is_prefill, kv_cache,
                                           attention_metadata,
                                           use_attention_rope)
        attn_output += attn_residual

        # FFW Block
        ffw_residual = attn_output
        normed_ffw_input = self.pre_mlp_norm(attn_output)
        if self.block_type == "moe":
            logits = self.moe(normed_ffw_input, op_mode)
            # Add the shared expert outputs to the MoE outputs.
            shared_expert_output = self.shared_experts(normed_ffw_input,
                                                       op_mode)
            logits += shared_expert_output
        elif self.block_type == "dense":
            logits = self.mlp(normed_ffw_input, op_mode)
        else:
            raise ValueError(f"Invalid block type: {self.block_type}")
        logits += ffw_residual
        return new_cache, logits

    def generate_kernel(self, rngs: nnx.Rngs):
        super().generate_kernel(rngs)
        self.shared_experts.generate_kernel(rngs)
