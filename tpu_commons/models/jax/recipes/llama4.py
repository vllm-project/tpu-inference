from dataclasses import dataclass, field
from typing import List, Mapping, Optional, Tuple

import jax.debug
import jax
import jax.numpy as jnp
from flax import nnx
from flax.core import pretty_repr
from flax.typing import PRNGKey
from jax.sharding import Mesh
from vllm.config import VllmConfig

from tpu_commons.models.jax.common.attention.attention import (
    AttentionConfig, AttentionMetadata, KVCache)
from tpu_commons.models.jax.common.base import Config, ParamFactory
from tpu_commons.models.jax.common.constants import RouterType
from tpu_commons.models.jax.common.layers import (Embedder, EmbedderConfig,
                                                  RMSNorm)
from tpu_commons.models.jax.common.model import Model, ModelConfig
from tpu_commons.models.jax.common.moe.moe import MoEConfig, RoutingConfig
from tpu_commons.models.jax.common.sharding import Sharding, ShardingConfig
from tpu_commons.models.jax.common.transformer_block import (
    TransformerBlock, TransformerBlockConfig)
from tpu_commons.models.jax.layers.sampling import sample


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
                # num_experts=1,  ## TODO REVERT
                expert_act="silu",
                apply_expert_weight_before_computation=False),
            router=RoutingConfig(
                    d_model=5120,
                    hidden_size=8192,  ## TODO: Is this correct?
                    num_experts=16,  ## TODO: Is this correct?
                    num_experts_per_tok=1,
                    router_type=RouterType.TOP_K,
                    act="silu",
                    expert_capacity=-1,
                    routed_bias=False,
                    routed_scaling_factor=1.0,
                    dtype=jnp.bfloat16),
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


class Llama4Scout(Model):

    def __init__(self, vllm_config: VllmConfig, rng: PRNGKey, mesh: Mesh):
        self.vllm_config = vllm_config
        self.rng = nnx.Rngs(rng)
        self.mesh = mesh
        self.cfg = Llama4ScoutConfig(
            model=Llama4ScoutModelConfig(),
            # TODO: we don't need to initialize the sharding_cfg
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
            kernel_initializer=nnx.initializers.xavier_normal(),
            scale_initializer=nnx.initializers.ones
        )
        # TODO: for test purpose, we applied the same/duplicated sharding here 
        # as those in tpu_jax_runner_v2. Need a better way to pass sharding in/out
        try:
            sharding_strategy = \
                self.vllm_config.additional_config["overrides"]["sharding"]["sharding_strategy"]
        except KeyError:
            print(
                f"No sharding strategy passed! Using default of full model parallelism={len(jax.devices())}"
            )
            sharding_strategy = {"tensor_parallelism": len(jax.devices())}
        self.sharding = Sharding(strategy_dict=sharding_strategy)
        self.cfg.sharding = self.sharding.sharding_cfg
        self.embedder = Embedder(cfg=self.cfg.model.emb,
                                 mesh=self.mesh,
                                 param_factory=param_factory,
                                 sharding_cfg=self.cfg.sharding)

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
            param_factory=param_factory,
            sharding_cfg=self.cfg.sharding,  # Kept for API consistency
            epsilon=self.cfg.model.layers.rmsnorm_epsilon,
            with_scale=True,  # TODO: What is this?
            dtype=self.cfg.model.layers.ffw.dtype,
            # quant=self.cfg.quant,
        )

    def apply(self, variables, *args, **kwargs):
        return self.__call__(*args, **kwargs)

    def load_weights(self,
                     rng: PRNGKey,
                     cache_dir: Optional[str] = None):
        self.rng = nnx.Rngs(rng)

        self.embedder.generate_kernel(self.rng)
        for i in range(self.cfg.model.num_moe_layers):
            self.moe_blocks[i].generate_kernel(self.rng)
        self.final_norm.generate_kernel(self.rng)
        return 

    def __call__(self,
                 is_prefill: bool,
                 do_sampling: bool,
                 kv_caches: List[KVCache],
                 input_ids: jax.Array,
                 attention_metadata: AttentionMetadata,
                 temperatures: jax.Array = None,
                 top_ps: jax.Array = None,
                 top_ks: jax.Array = None,
                 *args,
                ) -> Tuple[List[KVCache], jax.Array, jax.Array]:

        print(f"DEBUG: is_prefill: {is_prefill}, do_sampling: {do_sampling}")
        jax.debug.print("DEBUG: input token ID: {token}", token=input_ids)
        x = self.embedder.encode(input_ids)
        for i, block in enumerate(self.moe_blocks):
            kv_cache = kv_caches[i]
            new_kv_cache, x = block(x, is_prefill, kv_cache, attention_metadata)
            kv_caches[i] = new_kv_cache

        final_activation = self.final_norm(x)

        decoder_output = self.embedder.decode(final_activation)

        jax.debug.print("DEBUG: Logits for last token (first 10 values): {logits}",
                        logits=decoder_output[:, -1, :10])
        next_tokens = sample(
            is_prefill,
            do_sampling,
            self.rng,
            self.mesh,
            decoder_output,
            attention_metadata.seq_lens,
            temperatures,
            top_ps,
            top_ks,
            attention_metadata.chunked_prefill_enabled,
        )
        jax.debug.print("DEBUG: Sampled next_token ID: {token}", token=next_tokens)

        return kv_caches, next_tokens, decoder_output
