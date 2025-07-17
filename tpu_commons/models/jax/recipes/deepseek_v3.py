from dataclasses import dataclass, field, asdict
from typing import List, Tuple, Optional
import re

import jax
import jax.numpy as jnp
from flax import nnx
from flax.typing import PRNGKey
from jax.sharding import Mesh
from vllm.config import VllmConfig

from tpu_commons.models.jax.common.attention.attention import (
    AttentionMetadata)
from tpu_commons.models.jax.common.attention.deepseek_v3_attention import \
    MLAConfig
from tpu_commons.models.jax.common.base import Config, ParamFactory
from tpu_commons.models.jax.common.kv_cache import KVCacheType
from tpu_commons.models.jax.common.layers import (EmbedderConfig)
from tpu_commons.models.jax.common.model import Model, ModelConfig
# import tpu_commons.models.jax.common.sharding as sharding
from tpu_commons.models.jax.common.sharding import (ATTN_HEAD_AXIS_NAME,
                                                    ATTN_TENSOR_AXIS_NAME,
                                                    ShardingConfig,
                                                    ShardingRulesConfig)
from tpu_commons.models.jax.recipes.recipe import RecipeConfig
from tpu_commons.models.jax.utils.weight_utils import WeightLoader, get_param
from tpu_commons.models.jax.common.moe.deepseek_moe import DeepSeekV3RoutingConfig
from tpu_commons.models.jax.common.moe.moe import MoEConfig
from tpu_commons.models.jax.common.sharding import (Sharding, ShardingConfig,
                                                    ShardingRulesConfig)
from tpu_commons.models.jax.common.layers import (DenseFFWConfig, Embedder,
                                                  EmbedderConfig, RMSNorm)
from tpu_commons.models.jax.common.transformer_block import (
    SharedExpertsTransformerBlock, SharedExpertsTransformerBlockConfig, TransformerBlock, TransformerBlockConfig)
from tpu_commons.logger import init_logger

logger = init_logger(__name__)

@dataclass
class DeepseekV3ModelConfig(ModelConfig):
    vocab_size: int = 129280
    hidden_size: int = 7168
    dtype: jnp.dtype = jnp.bfloat16
    num_layers: int = 61
    emb: EmbedderConfig = None
    layers: TransformerBlockConfig = None
    vllm_config: VllmConfig = field(repr=False, default=None)
    interleave_moe_layer_step: int = 1  # Deepseek V3 has moe_layer_freq=1 in hf config.
    hidden_act: str = "silu"
    rms_norm_eps: float = 1e-06
    first_k_dense_replace: int = 3  # replace the first few MOE layers to dense layer. 

    def __post_init__(self):
        if not self.emb:
            self.emb = EmbedderConfig(vocab_size=self.vocab_size,
                                      hidden_size=self.hidden_size,
                                      dtype=jnp.bfloat16,
                                      normalize_embeddings=False)
        if not self.layers:
            self.layers = SharedExpertsTransformerBlockConfig(
                shared_experts=1,
                attention=MLAConfig(hidden_size=self.hidden_size,
                                    num_attention_heads=128,
                                    num_key_value_heads=128,
                                    rope_theta=10000,
                                    rope_scaling={
                                        "beta_fast": 32,
                                        "beta_slow": 1,
                                        "factor": 40,
                                        "mscale": 1.0,
                                        "mscale_all_dim": 1.0,
                                        "original_max_position_embeddings":
                                        4096,
                                        "type": "yarn"
                                    },
                                    q_lora_rank=1536,
                                    kv_lora_rank=512,
                                    qk_nope_head_dim=128,
                                    qk_rope_head_dim=64,
                                    v_head_dim=128,
                                    rms_norm_eps=self.rms_norm_eps,
                                    dtype=self.dtype,
                                    vllm_config=self.vllm_config),
                dense_ffw=DenseFFWConfig(
                    hidden_size=self.hidden_size,
                    intermediate_size=18432,
                    hidden_act=self.hidden_act,
                    dtype=self.dtype,
                    vllm_config=self.vllm_config
                ),
                dense_ffw_for_shared_moe=DenseFFWConfig(
                    hidden_size=self.hidden_size,
                    intermediate_size=2048,
                    hidden_act=self.hidden_act,
                    dtype=self.dtype,
                    vllm_config=self.vllm_config
                ),
                moe=MoEConfig(
                    hidden_size=self.hidden_size,
                    intermediate_size_moe=2048,
                    dtype=self.dtype,
                    num_local_experts=256,
                    hidden_act=self.hidden_act,
                    apply_expert_weight_before_computation=False,
                    router=DeepSeekV3RoutingConfig(
                        hidden_size=self.hidden_size,
                        n_routed_experts=256,
                        num_experts_per_token=8,
                        n_group=8,
                        routed_scaling_factor=2.5,
                        topk_group=4,
                        norm_topk_prob=True,
                        dtype=self.dtype,
                        vllm_config=self.vllm_config
                    ),
                    vllm_config=self.vllm_config),
                rms_norm_eps=self.rms_norm_eps,
                vllm_config=self.vllm_config)


@dataclass
class DeepSeekV3ShardingRulesConfig(ShardingRulesConfig):
    # MLA Query down projection weight: (Dim, QueryLoraRank)
    attn_mla_qa_weight_da: tuple = (None, None)
    # MLA Query up projection weight: (QueryLoraRank, NumHeads, HeadDim)
    attn_mla_qb_weight_anh: tuple = (None, None, None)
    # MLA KV down projection weight: (Dim, KVLoRA + QKRopeHeadDim)
    attn_mla_kva_weight_da: tuple = (None, None)
    # MLA KV up projection weight: (KVLoRA, NumHeads, QKNopeHeadDim + VHeadDim)
    attn_mla_kvb_weight_anh: tuple = (None, None, None)


@dataclass
class DeepSeekV3PrefillShardingRulesConfig(DeepSeekV3ShardingRulesConfig):
    # MLA Query up projection weight: (QueryLoraRank, NumHeads, HeadDim)
    attn_mla_qb_weight_anh: tuple = (None, ATTN_HEAD_AXIS_NAME,
                                     ATTN_TENSOR_AXIS_NAME)
    # MLA Query up projection weight: (QueryLoraRank, NumHeads, HeadDim)
    attn_mla_kvb_weight_anh: tuple = (None, ATTN_HEAD_AXIS_NAME,
                                      ATTN_TENSOR_AXIS_NAME)


@dataclass
class DeepSeekV3GenerateShardingRulesConfig(DeepSeekV3ShardingRulesConfig):
    # MLA Query up projection weight: (QueryLoraRank, NumHeads, HeadDim)
    attn_mla_qb_weight_anh: tuple = (None, ATTN_HEAD_AXIS_NAME,
                                     ATTN_TENSOR_AXIS_NAME)
    # MLA KV up projection weight: (KVLoRA, NumHeads, QKNopeHeadDim + VHeadDim)
    attn_mla_kvb_weight_anh: tuple = (None, ATTN_HEAD_AXIS_NAME,
                                      ATTN_TENSOR_AXIS_NAME)


@dataclass
class DeepSeekV3ServingConfig(Config):
    vllm_config: VllmConfig = field(repr=False, default=None)


@dataclass(frozen=True)
class DeepSeekV3Config(RecipeConfig):
    model: DeepseekV3ModelConfig = field(default_factory=DeepseekV3ModelConfig)
    sharding: ShardingConfig = field(
        default_factory=ShardingConfig,
        repr=False,
    )
    serving: DeepSeekV3ServingConfig = field(
        default_factory=DeepSeekV3ServingConfig)


@dataclass
class DeepSeekV3(Model):

    def __init__(self, vllm_config: VllmConfig, rng: PRNGKey, mesh: Mesh):
        self.vllm_config = vllm_config
        self.rng = nnx.Rngs(rng)
        try:
            strategy_dict = self.vllm_config.additional_config["sharding"][
                "sharding_strategy"]
        except (KeyError, TypeError):
            strategy_dict = {"tensor_parallelism": 4, "expert_parallelism": 2}  # todo: update this.
        self.sharding = Sharding(strategy_dict=strategy_dict,
                                 prefill_rules=asdict(DeepSeekV3PrefillShardingRulesConfig()),
                                 generate_rules=asdict(DeepSeekV3GenerateShardingRulesConfig()),
                                 default_rules_cls=DeepSeekV3ShardingRulesConfig,
                                 mesh=mesh,
                                 vllm_config=self.vllm_config)
        self.cfg = DeepSeekV3Config(
            model=DeepseekV3ModelConfig(vllm_config=self.vllm_config),
            sharding=self.sharding.sharding_cfg,
            serving=DeepSeekV3ServingConfig(vllm_config=self.vllm_config)
        )
        logger.info(f"Using the following config:\n{self.cfg}")
        self.use_random_init = self.vllm_config.additional_config.get(
            "random_weights", False)
        self.mesh = self.sharding.mesh

        self.weight_loader = DeepSeekV3WeightLoader(vllm_config=vllm_config, model_config=self.cfg.model)
        
        self._init_layers()
    
    def _init_layers(self):
        param_factory = ParamFactory(
            kernel_initializer=nnx.initializers.xavier_normal(),
            scale_initializer=nnx.initializers.ones,
            random_init = self.use_random_init)
        self.embedder = Embedder(cfg=self.cfg.model.emb,
                                 mesh=self.mesh,
                                 param_factory=param_factory,
                                 sharding_cfg=self.cfg.sharding)
        self.embedder.generate_kernel(self.rng)

        self.layers = []

        for i in range(self.cfg.model.first_k_dense_replace):
            block = TransformerBlock(
                cfg=self.cfg.model.layers,
                block_type="dense",
                attention_type="mla",  #?
                param_factory=param_factory,
                mesh=self.mesh,
                sharding_cfg=self.cfg.sharding)
        for i in range(self.cfg.model.first_k_dense_replace, self.cfg.model.num_layers):
            is_moe_layer = ((i + 1) %
                            self.cfg.model.interleave_moe_layer_step == 0)
            block_type = "moe" if is_moe_layer else "dense"
            block = SharedExpertsTransformerBlock(
                cfg=self.cfg.model.layers,
                block_type=block_type,
                attention_type="mla",
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

        # TODO: Add MTP.
    
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
    ) -> Tuple[List[KVCacheType], jax.Array]:
        is_prefill = False
        x = self.embedder.encode(input_ids)
        for (i, block) in enumerate(self.layers):
            kv_cache = kv_caches[i]
            new_kv_cache, x = block(x, is_prefill, kv_cache,
                                    attention_metadata)
            kv_caches[i] = new_kv_cache

        final_activation = self.final_norm(x)

        return kv_caches, final_activation

    def compute_logits(self, hidden_states: jax.Array) -> jax.Array:
        return self.lm_head.decode(hidden_states)


@dataclass
class DeepSeekV3WeightLoader(WeightLoader):
    def __init__(self, vllm_config: VllmConfig, model_config: ModelConfig):
        super().__init__(vllm_config=vllm_config,
                         model_config=model_config,
                         framework="flax",
                         filter_regex="language_model")
        self.setup()

    def setup(self):
        super().setup()
        self.set_transpose_param_map({
            "q_proj": (0, 2, 1),
            "k_proj": (0, 2, 1),
            "v_proj": (0, 2, 1),
            "router": (1, 0),
            "shared_expert.down_proj": (1, 0),
            "shared_expert.gate_proj": (1, 0),
            "shared_expert.up_proj": (1, 0),
            # "o_proj": (1, 2, 0),
        })
        # hidden_size = self.model_config.hidden_size
        # attn_heads = self.model_config.layers.attention.num_attention_heads
        # num_key_value_heads = self.model_config.layers.attention.num_key_value_heads
        # attn_head_dim = self.model_config.layers.attention.head_dim
        # self.set_reshape_param_map(
        #     {
        #         "q_proj": (attn_heads, attn_head_dim, hidden_size),
        #         "k_proj": (num_key_value_heads, attn_head_dim, hidden_size),
        #         "v_proj": (num_key_value_heads, attn_head_dim, hidden_size),
        #         "o_proj": (attn_heads, attn_head_dim, hidden_size),
        #     },
        #     param_type="weight",
        # )
        # Set the mappings from loaded parameter keys to standardized names.
        # todo: the weights are quantized.
        self.set_loaded_to_standardized_keys({
            # encode & decode
            "model.embed_tokens.weight":
            "embedder.input_embedding_table_VD",
            "lm_head.weight":
            "lm_head.input_embedding_table_VD",
            # final norm
            "model.norm.weight":
            "final_norm.scale",
            # norm in transformer blocks
            "model.layers.*.input_layernorm.weight":
            "layers.*.pre_attention_norm.scale",
            "model.layers.*.post_attention_layernorm.weight":
            "layers.*.pre_mlp_norm.scale",
            # attention (MLA)
            "model.layers.*.self_attn.q_a_proj.weight":
            "layers.*.attn.kernel_q_down_proj_DA",
            "model.layers.*.self_attn.q_b_proj.weight":
            "layers.*.attn.kernel_q_up_proj_ANH",
            "model.layers.*.self_attn.kv_a_proj_with_mqa.weight":
            "layers.*.attn.kernel_kv_down_proj_DA",
            "model.layers.*.self_attn.kv_b_proj.weight":
            "layers.*.attn.kernel_kv_up_proj_ANH",
            "model.layers.*.self_attn.o_proj.weight":
            "layers.*.attn.kernel_o_proj_NHD",
            # Dense ffw
            "model.layers.*.mlp.gate_proj.weight":
            "layers.*.mlp.kernel_gating_DF",
            "model.layers.*.mlp.up_proj.weight":
            "layers.*.mlp.kernel_up_proj_DF",
            "model.layers.*.mlp.down_proj.weight":
            "layers.*.mlp.kernel_down_proj_FD",
            # MOE(routed experts)
            "model.layers.*.mlp.gate.weight":
            "layers.*.moe.router.kernel_DE",
            "model.layers.*.mlp.experts.*.gate_proj.weight":
            "layers.*.moe.*.kernel_gating_EDF",
            "model.layers.*.mlp.experts.*.down_proj.weight":
            "layers.*.moe.*.kernel_down_proj_EFD",
            "model.layers.*.mlp.experts.*.up_proj.weight":
            "layers.*.moe.*.kernel_up_proj_EDF",
            # MOE(shared experts)
            "model.layers.*.mlp.shared_experts.down_proj.weight":
            "layers.*.shared_experts.kernel_down_proj_FD",
            "model.layers.*.mlp.shared_experts.gate_proj.weight":
            "layers.*.shared_experts.kernel_gating_DF",
            "model.layers.*.mlp.shared_experts.up_proj.weight":
            "layers.*.shared_experts.kernel_up_proj_DF",
        })
    
    def map_loaded_to_standardized_name(self, loaded_key: str) -> str:
        # Find the corresponding model key using the HF key
        layer_num, expert_num = None, None
        if "layer" in loaded_key:
            # extract layer number and replace it with *
            layer_num = re.search(r"layers\.(\d+)", loaded_key).group(1)
            layer_key = re.sub(r"layers\.\d+", "layers.*", loaded_key)
            # extract expert number if exists and replace it with *
            if "experts" in loaded_key and "shared_experts" not in loaded_key:
                expert_num = re.search(r"experts\.(\d+)", loaded_key).group(1)
            layer_key = re.sub(r"experts\.\d+", "experts.*", layer_key)
            # get standardized key and replace * with layer number.
            mapped_key = self.loaded_to_standardized_keys.get(
                layer_key, loaded_key)
            mapped_key = re.sub(r"layers\.\*", f"layers.{layer_num}",
                                mapped_key)
            # also replace moe.* into the expert number.
            if expert_num is not None:
                mapped_key = re.sub(r"moe\.\*", f"moe.{expert_num}",
                                    mapped_key)
        else:
            mapped_key = self.loaded_to_standardized_keys.get(
                loaded_key, loaded_key)
        return mapped_key

    def load_weights(self, model_for_loading: nnx.Module):
        model_params = nnx.state(model_for_loading)
        logger.warning(
            f"loaded_to_standardized_keys: {self.loaded_to_standardized_keys}"
        )
        cumulative_global_memory = 0
        cumulative_local_memory = 0
        with jax.default_device(jax.devices("cpu")[0]):
            for loaded_name, loaded_weight in self.names_and_weights_generator:
                old_param_name = loaded_name
                mapped_name = self.map_loaded_to_standardized_name(loaded_name)
                model_weight = get_param(model_params, mapped_name)
                logger.info(
                    f"{old_param_name}: {loaded_weight.shape}  -->  {mapped_name}: {model_weight.value.shape}"
                )
                # reshape/traspose weights
                if loaded_name.endswith(".weight"):
                    loaded_weight = self.reshape_params(
                        loaded_name, loaded_weight, "weight")
                    loaded_weight = self.transpose_params(
                        loaded_name, loaded_weight)
                if model_weight.value.shape != loaded_weight.shape:
                    raise ValueError(
                        f"Loaded shape for {loaded_name}: {loaded_weight.shape} "
                        f"does not match model shape for {mapped_name}: {model_weight.value.shape}!"
                    )