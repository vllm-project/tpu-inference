# TODO: Update documentation

import pprint
import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

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
from tpu_commons.models.jax.common.layers import (DenseFFWConfig, Embedder,
                                                  EmbedderConfig, RMSNorm)
from tpu_commons.models.jax.common.model import Model
from tpu_commons.models.jax.common.sharding import (Sharding, ShardingConfig,
                                                    ShardingRulesConfig)
from tpu_commons.models.jax.common.transformer_block import (
    TransformerBlock, TransformerBlockConfig)
from tpu_commons.models.jax.layers.misc import shard_put
from tpu_commons.models.jax.recipes.recipe import RecipeConfig
from tpu_commons.models.jax.utils.weight_utils import (ParameterType,
                                                       WeightLoader, get_param)

logger = init_logger(__name__)
pp = pprint.PrettyPrinter(depth=6)


@dataclass
class Llama3ModelConfig():
    """Configuration for the Llama3 model family."""
    hidden_size: int
    num_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    intermediate_size: int
    dtype: jnp.dtype = jnp.bfloat16
    head_dim: int = 128
    rope_theta: float = 500000.0
    vocab_size: int = 128256
    rms_norm_eps: float = 1e-5
    emb: EmbedderConfig = None
    layers: TransformerBlockConfig = None
    vllm_config: VllmConfig = field(repr=False, default=None)

    def __post_init__(self):

        # Initialize defaults:
        if not self.emb:
            self.emb = EmbedderConfig(vocab_size=self.vocab_size,
                                      hidden_size=self.hidden_size,
                                      dtype=self.dtype,
                                      normalize_embeddings=False,
                                      vllm_config=self.vllm_config)
        if not self.layers:
            self.layers = TransformerBlockConfig(
                attention=AttentionConfig(
                    hidden_size=self.hidden_size,
                    num_attention_heads=self.num_attention_heads,
                    num_key_value_heads=self.num_key_value_heads,
                    head_dim=self.head_dim,
                    rope_theta=self.rope_theta,
                    rope_scaling={},
                    dtype=self.dtype,
                    vllm_config=self.vllm_config),
                dense_ffw=DenseFFWConfig(
                    hidden_size=self.hidden_size,
                    intermediate_size=self.intermediate_size,
                    hidden_act="silu",
                    dtype=self.dtype,
                    vllm_config=self.vllm_config),
                rms_norm_eps=self.rms_norm_eps,
                block_type="dense",
                vllm_config=self.vllm_config)


@dataclass
class Llama3ShardingRulesConfig(ShardingRulesConfig):
    lm_head_dv: tuple = (None, sharding.MLP_TENSOR_AXIS_NAME)


@dataclass
class Llama3ServingConfig(Config):
    vllm_config: VllmConfig = field(repr=False, default=None)


@dataclass(frozen=True)
class Llama3RecipeConfig(RecipeConfig):
    model: Llama3ModelConfig
    sharding: ShardingConfig = field(
        default_factory=ShardingConfig,
        repr=False,
    )
    serving: Llama3ServingConfig = field(default_factory=Llama3ServingConfig)


class LlamaForCausalLM(Model):

    def __init__(self, vllm_config: VllmConfig, rng: jax.Array, mesh: Mesh):
        self.vllm_config = vllm_config
        self.rng = nnx.Rngs(rng)
        self.mesh = mesh
        try:
            strategy_dict = self.vllm_config.additional_config["sharding"][
                "sharding_strategy"]
        except (KeyError, TypeError):
            strategy_dict = {"tensor_parallelism": 1}
        #TODO: after all models are migrated to the new sharding, 
        # we need to only create sharding obj in TPU runner
        self.sharding = Sharding(strategy_dict=strategy_dict,
                                 mesh=self.mesh,
                                 default_rules_cls=Llama3ShardingRulesConfig,
                                 vllm_config=self.vllm_config)
        self.use_random_init = self.vllm_config.additional_config.get(
            "random_weights", False)
        # TODO: support to loading from HF checkpoints.
        model_name = self.vllm_config.model_config.model.lower()
        if "70b" in model_name:
            logger.info("Initializing Llama3 70B model variant.")
            model_params = {
                "hidden_size": 8192,
                "num_layers": 80,
                "num_attention_heads": 64,
                "num_key_value_heads": 8,
                "intermediate_size": 28672,
            }
        elif "8b" in model_name:
            logger.info("Initializing Llama3 8B model variant.")
            model_params = {
                "hidden_size": 4096,
                "num_layers": 32,
                "num_attention_heads": 32,
                "num_key_value_heads": 8,
                "intermediate_size": 14336,
            }
        else:
            raise ValueError(
                f"Could not determine Llama3 variant (8B or 70B) from model name: '{model_name}'. "
                "Please ensure '8b' or '70b' is in the model path.")

        self.cfg = Llama3RecipeConfig(
            model=Llama3ModelConfig(vllm_config=self.vllm_config,
                                    **model_params),
            sharding=self.sharding.sharding_cfg,
            serving=Llama3ServingConfig(vllm_config=self.vllm_config))

        logger.info(f"Using the following config:\n{self.cfg}")
        logger.info(
            f"Using the following sharding overrides:\n{self.sharding}")
        self.mesh = self.sharding.mesh
        self.weight_loader = Llama3WeightLoader(vllm_config=self.vllm_config,
                                                model_config=self.cfg.model,
                                                cache_dir=None,
                                                sharding_cfg=self.cfg.sharding)
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

        self.layers = [
            TransformerBlock(cfg=self.cfg.model.layers,
                             block_type="dense",
                             param_factory=param_factory,
                             mesh=self.mesh,
                             sharding_cfg=self.cfg.sharding)
            for i in range(self.cfg.model.num_layers)
        ]
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

    # For compatibility with flax.
    def apply(self, variables, *args, **kwargs):
        return self.__call__(*args, **kwargs)

    def load_weights(self, rng: PRNGKey, cache_dir: Optional[str] = None):
        self.rng = nnx.Rngs(rng)
        if self.use_random_init:
            #TODO: Support loading random weights, either here or in tpu_runner
            logger.warning(
                "Model name or path not provided - randomly initializing the weights."
            )
        else:
            weight_loader = Llama3WeightLoader(vllm_config=self.vllm_config,
                                               model_config=self.cfg.model,
                                               cache_dir=cache_dir,
                                               sharding_cfg=self.cfg.sharding)
            weight_loader.load_weights(self)

    def __call__(
        self,
        kv_caches: List[jax.Array],
        input_ids: jax.Array,
        attention_metadata: AttentionMetadata,
        *args,
    ) -> Tuple[List[KVCacheType], jax.Array]:
        is_prefill = False
        x = self.embedder.encode(input_ids)
        for (i, layer) in enumerate(self.layers):
            kv_cache = kv_caches[i]
            with jax.named_scope(f'layer_{i}'):
                new_kv_cache, x = layer(x, is_prefill, kv_cache,
                                        attention_metadata)
            kv_caches[i] = new_kv_cache

        final_activation = self.final_norm(x)

        return kv_caches, final_activation

    def compute_logits(self, hidden_states: jax.Array) -> jax.Array:
        logits = jnp.dot(hidden_states, self.lm_head.input_embedding_table_DV.value)

        return logits


class Llama3WeightLoader(WeightLoader):

    def __init__(self,
                 vllm_config: VllmConfig,
                 model_config: Llama3ModelConfig,
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
            "embed_tokens": (1, 0),
            "lm_head": (1, 0),
            "gate_proj": (1, 0),
            "up_proj": (1, 0),
            "down_proj": (1, 0),
            "q_proj": (2, 0, 1),
            "k_proj": (2, 0, 1),
            "v_proj": (2, 0, 1),
            "o_proj": (1, 2, 0),
        })
        # Set weights reshape map
        hidden_size = self.model_config.hidden_size
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
        self.set_loaded_to_standardized_keys({
            "model.embed_tokens":
            "embedder.input_embedding_table_DV",
            "model.layers.*.input_layernorm":
            "layers.*.pre_attention_norm.scale",
            "model.layers.*.mlp.down_proj":
            "layers.*.mlp.kernel_down_proj_FD",
            "model.layers.*.mlp.gate_proj":
            "layers.*.mlp.kernel_gating_DF",
            "model.layers.*.mlp.up_proj":
            "layers.*.mlp.kernel_up_proj_DF",
            "model.layers.*.post_attention_layernorm":
            "layers.*.pre_mlp_norm.scale",
            "model.layers.*.self_attn.k_proj":
            "layers.*.attn.kernel_k_proj_DKH",
            "model.layers.*.self_attn.o_proj":
            "layers.*.attn.kernel_o_proj_NHD",
            "model.layers.*.self_attn.q_proj":
            "layers.*.attn.kernel_q_proj_DNH",
            "model.layers.*.self_attn.v_proj":
            "layers.*.attn.kernel_v_proj_DKH",
            "model.norm":
            "final_norm.scale",
            "lm_head":
            "lm_head.input_embedding_table_DV"
        })

    def map_loaded_to_standardized_name(self, loaded_key: str) -> str:
        # Find the corresponding model key using the HF key
        if "layer" in loaded_key:
            layer_num_match = re.search(r"layers\.(\d+)", loaded_key)
            if layer_num_match:
                layer_num = layer_num_match.group(1)
                layer_key = re.sub(r"layers\.\d+", "layers.*", loaded_key)
                mapped_key = self.loaded_to_standardized_keys.get(
                    layer_key, layer_key)
                mapped_key = re.sub(r"layers\.\*", f"layers.{layer_num}",
                                    mapped_key)
                return mapped_key

        return self.loaded_to_standardized_keys.get(loaded_key, loaded_key)

    def load_weights(self, model_for_loading: nnx.Module):
        model_params = nnx.state(model_for_loading)
        for loaded_name, loaded_weight in self.names_and_weights_generator:
            old_param_name = loaded_name
            if loaded_name.endswith(".weight"):
                loaded_name = loaded_name.removesuffix(".weight")
            mapped_name = self.map_loaded_to_standardized_name(loaded_name)
            model_weight = get_param(model_params, mapped_name)

            if model_weight is None:
                logger.warning(
                    f"Could not find a matching model parameter for loaded weight: '{old_param_name}' (mapped to: '{mapped_name}')"
                )
                continue

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
