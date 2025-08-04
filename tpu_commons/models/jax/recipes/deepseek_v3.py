import re
from dataclasses import asdict, dataclass, field
from typing import List, Optional, Tuple

import jax
import jax.numpy as jnp
import ml_dtypes
import torch
from flax import nnx
from flax.typing import PRNGKey
from jax.sharding import Mesh
from vllm.config import VllmConfig

from tpu_commons.logger import init_logger
from tpu_commons.models.jax.common.attention.attention import AttentionMetadata
from tpu_commons.models.jax.common.attention.deepseek_v3_attention import (
    MLA, MLAConfig)
from tpu_commons.models.jax.common.base import Config, ParamFactory
from tpu_commons.models.jax.common.constants import KVCacheType
from tpu_commons.models.jax.common.layers import (DenseFFWConfig, Embedder,
                                                  EmbedderConfig, RMSNorm)
from tpu_commons.models.jax.common.model import Model, ModelConfig
from tpu_commons.models.jax.common.moe.deepseek_moe import \
    DeepSeekV3RoutingConfig
from tpu_commons.models.jax.common.moe.moe import MoEConfig
from tpu_commons.models.jax.common.sharding import (ATTN_HEAD_AXIS_NAME,
                                                    ATTN_TENSOR_AXIS_NAME,
                                                    Sharding, ShardingConfig,
                                                    ShardingRulesConfig)
from tpu_commons.models.jax.common.transformer_block import (
    SharedExpertsTransformerBlock, SharedExpertsTransformerBlockConfig,
    TransformerBlock, TransformerBlockConfig)
from tpu_commons.models.jax.layers.misc import shard_put
from tpu_commons.models.jax.recipes.recipe import RecipeConfig
from tpu_commons.models.jax.utils.weight_utils import WeightLoader, get_param

logger = init_logger(__name__)


def print_param_info(param: nnx.Param, name: str):
    logger.warning(f"Global shape for {name}: {param.value.shape}"
                   )  # Note: sharding is a PartitionSpec
    logger.warning(f"Sharding for {name}: {param.sharding}"
                   )  # Note: sharding is a PartitionSpec

    # Print tensor shape for a single device
    # buffers = [shard.data for shard in my_array.addressable_shards]
    logger.warning(
        f"Shape of {name} on a single device: {param.value.addressable_shards[0].data.shape}"
    )


@dataclass
class DeepseekV3ModelConfig(ModelConfig):
    vocab_size: int = 129280
    hidden_size: int = 7168
    dtype: jnp.dtype = jnp.bfloat16
    num_layers: int = 61
    num_attention_heads: int = 128
    num_key_value_heads: int = 128
    ffw_intermediate_size: int = 18432
    moe_intermediate_size: int = 2048
    num_local_experts: int = 256
    num_experts_per_token: int = 8
    n_group: int = 8
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
                attention=MLAConfig(
                    hidden_size=self.hidden_size,
                    num_attention_heads=self.num_attention_heads,
                    num_key_value_heads=self.num_key_value_heads,
                    rope_theta=10000,
                    rope_scaling={
                        "beta_fast": 32,
                        "beta_slow": 1,
                        "factor": 40,
                        "mscale": 1.0,
                        "mscale_all_dim": 1.0,
                        "original_max_position_embeddings": 4096,
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
                    intermediate_size=self.ffw_intermediate_size,
                    hidden_act=self.hidden_act,
                    dtype=self.dtype,
                    vllm_config=self.vllm_config),
                moe=MoEConfig(
                    hidden_size=self.hidden_size,
                    intermediate_size_moe=self.moe_intermediate_size,
                    dtype=self.dtype,
                    num_local_experts=self.num_local_experts,
                    hidden_act=self.hidden_act,
                    apply_expert_weight_before_computation=False,
                    router=DeepSeekV3RoutingConfig(
                        hidden_size=self.hidden_size,
                        n_routed_experts=self.num_local_experts,
                        num_experts_per_token=self.num_experts_per_token,
                        n_group=self.n_group,
                        routed_scaling_factor=2.5,
                        topk_group=4,
                        norm_topk_prob=True,
                        dtype=self.dtype,
                        vllm_config=self.vllm_config),
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
    # Attention V3 output: (actual_num_kv_heads, max_num_tokens, num_q_heads_per_kv_head // q_packing, q_packing, head_dim)
    attn_o_ktnph: tuple = (ATTN_HEAD_AXIS_NAME, None, None, None,
                           ATTN_TENSOR_AXIS_NAME)
    # Attention V3 query: (actual_num_kv_heads, max_num_tokens, num_q_heads_per_kv_head // q_packing, q_packing, head_dim)
    query_ktnph: tuple = (ATTN_HEAD_AXIS_NAME, None, None, None,
                          ATTN_TENSOR_AXIS_NAME)
    # Attention V3 kv_cache: (total_num_pages, page_size, num_kv_heads_x2 // kv_packing, kv_packing, head_dim)
    keyvalue_cache_nbkph: tuple = (None, None, ATTN_HEAD_AXIS_NAME, None,
                                   ATTN_TENSOR_AXIS_NAME)


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

    def __init__(self,
                 vllm_config: VllmConfig,
                 rng: jax.Array,
                 mesh: Mesh,
                 param_factory: ParamFactory | None = None):
        self.vllm_config = vllm_config
        self.rng = nnx.Rngs(rng)
        self.param_factory = param_factory
        try:
            strategy_dict = self.vllm_config.additional_config["sharding"][
                "sharding_strategy"]
        except (KeyError, TypeError):
            strategy_dict = {
                "tensor_parallelism": 4,
                "expert_parallelism": 2
            }  # todo: update this.
        self.sharding = Sharding(
            strategy_dict=strategy_dict,
            prefill_rules=asdict(DeepSeekV3PrefillShardingRulesConfig()),
            generate_rules=asdict(DeepSeekV3GenerateShardingRulesConfig()),
            default_rules_cls=DeepSeekV3ShardingRulesConfig,
            mesh=mesh,
            vllm_config=self.vllm_config)
        self.cfg = DeepSeekV3Config(
            model=DeepseekV3ModelConfig(vllm_config=self.vllm_config),
            sharding=self.sharding.sharding_cfg,
            serving=DeepSeekV3ServingConfig(vllm_config=self.vllm_config))
        logger.info(f"Using the following config:\n{self.cfg}")
        self.use_random_init = self.vllm_config.additional_config.get(
            "random_weights", False)
        self.mesh = self.sharding.mesh

        self.weight_loader = DeepSeekV3WeightLoader(
            vllm_config=vllm_config, model_config=self.cfg.model)

        self._init_layers()

    def _init_layers(self):
        if not self.param_factory:
            self.param_factory = ParamFactory(
                kernel_initializer=nnx.initializers.xavier_normal(),
                scale_initializer=nnx.initializers.ones,
                random_init=self.use_random_init)
        self.embedder = Embedder(cfg=self.cfg.model.emb,
                                 mesh=self.mesh,
                                 param_factory=self.param_factory,
                                 sharding_cfg=self.cfg.sharding)
        self.embedder.generate_kernel(self.rng)

        self.layers = []

        for i in range(self.cfg.model.first_k_dense_replace):
            block = TransformerBlock(cfg=self.cfg.model.layers,
                                     block_type="dense",
                                     attention_cls=MLA,
                                     param_factory=self.param_factory,
                                     mesh=self.mesh,
                                     sharding_cfg=self.cfg.sharding)
            self.layers.append(block)

        for i in range(self.cfg.model.first_k_dense_replace,
                       self.cfg.model.num_layers):
            is_moe_layer = ((i + 1) %
                            self.cfg.model.interleave_moe_layer_step == 0)
            block_type = "moe" if is_moe_layer else "dense"
            block = SharedExpertsTransformerBlock(
                cfg=self.cfg.model.layers,
                block_type=block_type,
                attention_cls=MLA,
                param_factory=self.param_factory,
                mesh=self.mesh,
                sharding_cfg=self.cfg.sharding)
            self.layers.append(block)

        for i in range(len(self.layers)):
            self.layers[i].generate_kernel(self.rng)

        self.final_norm = RMSNorm(
            dims=self.cfg.model.hidden_size,
            mesh=self.mesh,
            param_factory=self.param_factory,
            sharding_cfg=self.cfg.sharding,
            epsilon=self.cfg.model.layers.rms_norm_eps,
            with_scale=True,
            dtype=self.cfg.model.dtype,
        )
        self.final_norm.generate_kernel(self.rng)

        self.lm_head = Embedder(cfg=self.cfg.model.emb,
                                mesh=self.mesh,
                                param_factory=self.param_factory,
                                sharding_cfg=self.cfg.sharding)
        self.lm_head.generate_kernel(self.rng)

        # TODO: Add MTP.
    @classmethod
    def create_model_with_random_weights(cls, vllm_config: VllmConfig,
                                         rng: jax.Array, mesh: Mesh):
        """to create a model with random weights."""
        logger.info("Initializing model with random weights.")
        param_factory = ParamFactory(
            kernel_initializer=nnx.initializers.xavier_normal(),
            scale_initializer=nnx.initializers.ones,
            random_init=True)
        return cls(vllm_config, rng, mesh, param_factory)

    @classmethod
    def create_model_for_checkpoint_loading(cls, vllm_config: VllmConfig,
                                            rng: jax.Array, mesh: Mesh):
        """to create a model with abstract shapes for checkpoint loading."""
        logger.info("Initializing abstract model for checkpoint loading.")
        param_factory = ParamFactory(
            kernel_initializer=nnx.initializers.xavier_normal(),
            scale_initializer=nnx.initializers.ones,
            random_init=False)
        return cls(vllm_config, rng, mesh, param_factory)

    # For compatibility with flax.
    def apply(self, variables, *args, **kwargs):
        return self.__call__(*args, **kwargs)

    def load_weights(self, rng: PRNGKey, cache_dir: Optional[str] = None):
        self.rng = nnx.Rngs(rng)
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
                         framework="pt",
                         filter_regex="")
        self.setup()
        self.num_routed_experts = model_config.layers.moe.num_local_experts

    def setup(self):
        super().setup()
        self.set_transpose_param_map({
            # dense mlp
            r"mlp\.down_proj": (1, 0),
            r"mlp\.gate_proj": (1, 0),
            r"mlp\.up_proj": (1, 0),
            # mla
            r"q_a_proj": (1, 0),
            r"q_b_proj": (2, 0, 1),
            r"kv_a_proj_with_mqa": (1, 0),
            r"kv_b_proj": (2, 0, 1),
            r"o_proj": (1, 2, 0),
            # moe
            r"mlp\.gate\.weight": (1, 0),
            r"mlp\.experts\.\d+\.gate_proj": (0, 2, 1),
            r"mlp\.experts\.\d+\.down_proj": (0, 2, 1),
            r"mlp\.experts\.\d+\.up_proj": (0, 2, 1),
            r"mlp\.shared_experts\.down_proj": (1, 0),
            r"mlp\.shared_experts\.gate_proj": (1, 0),
            r"mlp\.shared_experts\.up_proj": (1, 0)
        })
        hidden_size = self.model_config.hidden_size
        q_lora_rank = self.model_config.layers.attention.q_lora_rank
        kv_lora_rank = self.model_config.layers.attention.kv_lora_rank
        attn_heads = self.model_config.layers.attention.num_attention_heads
        qk_nope_head_dim = self.model_config.layers.attention.qk_nope_head_dim
        qk_rope_head_dim = self.model_config.layers.attention.qk_rope_head_dim
        v_head_dim = self.model_config.layers.attention.v_head_dim
        self.set_reshape_param_map(
            {
                "q_b_proj":
                (attn_heads, qk_nope_head_dim + qk_rope_head_dim, q_lora_rank),
                "kv_b_proj":
                (attn_heads, qk_nope_head_dim + v_head_dim, kv_lora_rank),
                "o_proj": (hidden_size, attn_heads, v_head_dim)
            },
            param_type="weight",
        )
        # Set the mappings from loaded parameter keys to standardized names.
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
            "model.layers.*.self_attn.q_a_layernorm.weight":
            "layers.*.attn.q_rms_norm.scale",
            "model.layers.*.self_attn.kv_a_layernorm.weight":
            "layers.*.attn.kv_rms_norm.scale",
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
            "model.layers.*.mlp.gate.e_score_correction_bias":
            "layers.*.moe.router.bias_E",
            "model.layers.*.mlp.experts.*.gate_proj.weight":
            "layers.*.moe.kernel_gating_EDF",
            "model.layers.*.mlp.experts.*.down_proj.weight":
            "layers.*.moe.kernel_down_proj_EFD",
            "model.layers.*.mlp.experts.*.up_proj.weight":
            "layers.*.moe.kernel_up_proj_EDF",
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
        if "layer" in loaded_key:
            # extract layer number and replace it with *
            layer_num = re.search(r"layers\.(\d+)", loaded_key).group(1)
            layer_key = re.sub(r"layers\.\d+", "layers.*", loaded_key)
            # extract expert number if exists and replace it with *
            if "experts" in loaded_key and "shared_experts" not in loaded_key:
                layer_key = re.sub(r"experts\.\d+", "experts.*", layer_key)
            # get standardized key and replace * with layer number.
            mapped_key = self.loaded_to_standardized_keys.get(
                layer_key, loaded_key)
            mapped_key = re.sub(r"layers\.\*", f"layers.{layer_num}",
                                mapped_key)
        else:
            mapped_key = self.loaded_to_standardized_keys.get(
                loaded_key, loaded_key)
        return mapped_key

    def transpose_params(self, param_key: str, param_tensor: jax.Array):
        for key in self.transformation_cfg.transpose:
            if re.search(key, param_key):
                return jnp.transpose(param_tensor,
                                     self.transformation_cfg.transpose[key])
        return param_tensor  # Base case / no-op

    def _process_moe_weights(self, loaded_name, loaded_weight, weights_dict):
        layer_num = re.search(r"layers\.(\d+)", loaded_name).group(1)
        expert_num = re.search(r"experts\.(\d+)", loaded_name).group(1)
        if layer_num not in weights_dict:
            weights_dict[layer_num] = {}
        weights_dict[layer_num][expert_num] = loaded_weight
        # Stack all the weights from the expert in this layer
        if len(weights_dict[layer_num]) == self.num_routed_experts:
            weight_list = []
            for expert_index in range(self.num_routed_experts):
                weight_list.append(weights_dict[layer_num][str(expert_index)])
            stacked_weights = torch.stack(weight_list, axis=0)
            del weights_dict[layer_num]
            return stacked_weights
        return None

    def _load_individual_weight(self, name, weight, model_params, model_mesh):
        mapped_name = self.map_loaded_to_standardized_name(name)
        model_weight = get_param(model_params, mapped_name)
        logger.debug(
            f"{name}: {weight.shape}  -->  {mapped_name}: {model_weight.value.shape}"
        )

        # Convert weights from torch into numpy
        # TODO: set cast_type based on model weight's type.
        cast_type = ml_dtypes.bfloat16
        weight = weight.to(torch.float32).numpy().astype(cast_type)

        # Reshape and transpose weights if necessary.
        weight = self.reshape_params(name, weight, "weight")
        weight = self.transpose_params(name, weight)
        if model_weight.value.shape != weight.shape:
            raise ValueError(
                f"Loaded shape for {name}: {weight.shape} "
                f"does not match model shape for {mapped_name}: {model_weight.value.shape}!"
            )
        model_weight.value = shard_put(weight,
                                       model_weight.sharding.spec,
                                       mesh=model_mesh)
        model_weight.value.block_until_ready()
        del weight
        print_param_info(model_weight, name)
        return model_weight.value.nbytes / 1e9, model_weight.value.addressable_shards[
            0].data.nbytes / 1e9

    def load_weights(self, model_for_loading: nnx.Module):
        model_params = nnx.state(model_for_loading)
        logger.warning(
            f"loaded_to_standardized_keys: {self.loaded_to_standardized_keys}")
        cumulative_global_memory = 0
        cumulative_local_memory = 0
        mlp_experts_gate_proj_weights = {}
        mlp_experts_up_proj_weights = {}
        mlp_experts_down_proj_weights = {}
        fp8_weights = {}
        with jax.default_device(jax.devices("cpu")[0]):
            for loaded_name, loaded_weight in self.names_and_weights_generator:
                # Skip if the model has fewer layers than original.
                if re.search(r"layers\.(\d+)", loaded_name):
                    layer_num = re.search(r"layers\.(\d+)",
                                          loaded_name).group(1)
                    if int(layer_num) >= self.model_config.num_layers:
                        del loaded_weight
                        continue
                if 'layers.61' in loaded_name:
                    # skip loading MTP module.
                    del loaded_weight
                    continue
                if re.search(r"experts\.(\d+)", loaded_name):
                    expert_num = re.search(r"experts\.(\d+)",
                                           loaded_name).group(1)
                    if int(expert_num) >= self.model_config.num_local_experts:
                        del loaded_weight
                        continue
                if loaded_weight.dtype == torch.float8_e4m3fn:
                    fp8_weights[loaded_name] = loaded_weight
                    continue
                if loaded_name.endswith(".weight_scale_inv"):
                    # assuming weights are loaded before scales.
                    weight_name = loaded_name.replace(".weight_scale_inv",
                                                      ".weight")

                    loaded_weight = weights_dequant_cpu(
                        fp8_weights[weight_name], loaded_weight)
                    loaded_name = weight_name
                    del fp8_weights[weight_name]
                # concat mlp.experts weights
                if "mlp.experts" in loaded_name:
                    print(f'[debug] {loaded_name=}')
                    if "down_proj" in loaded_name:
                        stacked_weights = self._process_moe_weights(
                            loaded_name, loaded_weight,
                            mlp_experts_down_proj_weights)
                    if "gate_proj" in loaded_name:
                        stacked_weights = self._process_moe_weights(
                            loaded_name, loaded_weight,
                            mlp_experts_gate_proj_weights)
                    if "up_proj" in loaded_name:
                        stacked_weights = self._process_moe_weights(
                            loaded_name, loaded_weight,
                            mlp_experts_up_proj_weights)
                    if stacked_weights is not None:
                        weight_bytes, weight_shards = self._load_individual_weight(
                            loaded_name, stacked_weights, model_params,
                            model_for_loading.mesh)
                        cumulative_global_memory += weight_bytes
                        cumulative_local_memory += weight_shards
                        logger.info(
                            f"Cumulative global memory: {cumulative_global_memory} GB"
                        )
                        logger.info(
                            f"Cumulative local memory: {cumulative_local_memory} GB"
                        )
                else:
                    weight_bytes, weight_shards = self._load_individual_weight(
                        loaded_name, loaded_weight, model_params,
                        model_for_loading.mesh)
                    cumulative_global_memory += weight_bytes
                    cumulative_local_memory += weight_shards
                    logger.info(
                        f"Cumulative global memory: {cumulative_global_memory} GB"
                    )
                    logger.info(
                        f"Cumulative local memory: {cumulative_local_memory} GB"
                    )

        del mlp_experts_gate_proj_weights
        del mlp_experts_up_proj_weights
        del mlp_experts_down_proj_weights
        del fp8_weights
        # TODO: validate that all of the model_params were accounted for as well.
        nnx.update(model_for_loading, model_params)


def weights_dequant_cpu(x: torch.Tensor,
                        s: torch.Tensor,
                        block_size: int = 128) -> torch.Tensor:
    assert x.dim() == 2 and s.dim() == 2, "Both x and s must be 2D tensors"
    M, N = x.shape

    x = x.to(torch.float32)
    y = torch.empty_like(x, dtype=torch.get_default_dtype())

    for i in range(0, M, block_size):
        for j in range(0, N, block_size):
            row_start = i
            row_end = min(i + block_size, M)
            col_start = j
            col_end = min(j + block_size, N)
            block = x[row_start:row_end, col_start:col_end]
            scale = s[i // block_size, j // block_size]
            y[row_start:row_end, col_start:col_end] = (block * scale).to(
                torch.get_default_dtype())

    return y
